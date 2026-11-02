"""Reuse the existing parent ML overlay; change only the child evaluation venv."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

PACKAGES={'torch':'torch','transformers':'transformers','tokenizers':'tokenizers',
          'huggingface-hub':'huggingface_hub','safetensors':'safetensors','Pillow':'PIL'}


def overlay_line(paths):
    # This .pth is processed in the child site before its base system sites.
    # addsitedir also processes the existing parent's own ordinary .pth files.
    return ''.join('import site; site.addsitedir('+repr(str(Path(p).resolve()))+')\n' for p in paths)


def parent_reference(parent_python):
    script='''import importlib.metadata as m,importlib.util,json,sys,sysconfig
packages=%r
r={'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python':list(sys.version_info[:3]),
   'sites':list(dict.fromkeys([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')])),
   'versions':{},'origins':{},'errors':[]}
for dist,module in packages.items():
 try:
  r['versions'][dist]=m.version(dist)
  spec=importlib.util.find_spec(module)
  if spec is None:raise ImportError(module)
  r['origins'][module]=spec.origin
 except Exception as e:r['errors'].append(dist+': '+type(e).__name__+': '+str(e))
print(json.dumps(r))
''' % PACKAGES
    return json.loads(subprocess.check_output([str(parent_python),'-c',script],text=True))


def verify(root, reference, output):
    r={'passed':False,'errors':[],'versions':{},'origins':{},'parent_reference':reference,
       'training_updates':0,'model_instances':0,'model_forwards':0,'physics_steps':0,
       'scope':'Full CPU model-entry imports and dependency identity; no native EGL context or task episode'}
    def check(label, action):
        try:action()
        except Exception as e:r['errors'].append(label+': '+type(e).__name__+': '+str(e))
    def package(dist,module):
        r['versions'][dist]=metadata.version(dist)
        spec=importlib.util.find_spec(module)
        if spec is None:raise ImportError(module)
        r['origins'][module]=spec.origin
        if r['versions'][dist]!=reference['versions'][dist]:
            raise ValueError('Effective version differs from working training parent')
        if module in ('torch','transformers','tokenizers','huggingface_hub','safetensors'):
            if Path(spec.origin).resolve()!=Path(reference['origins'][module]).resolve():
                raise ValueError('Effective package does not come from the working parent/base stack')
    for dist,module in PACKAGES.items():check(dist,lambda d=dist,m=module:package(d,m))
    deps=('numpy','opencv-python-headless','termcolor','mujoco','robosuite','gym','bddl',
          'easydict','h5py','scipy','numba','PyOpenGL','glfw','PyYAML','matplotlib')
    for dist in deps:
        check(dist,lambda d=dist:r['versions'].update({d:metadata.version(d)}))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path[:0]=[str(root/'libero_original_evaluation_v1/code'),str(root/'libero_original_evaluation_v1')]
    # Exercise actual class/module import dependencies, never from_pretrained or a model constructor.
    for module in ('cv2','torch','transformers','frozen_visual_backbone','models','train','libero_project_policy'):
        check('import '+module,lambda n=module:__import__(n))
    def classes():
        from transformers import CLIPTokenizer,CLIPTextModel,Dinov2Model
        import cv2,numpy,torch,re
        if (torch.__version__!='2.5.1+cu118' or numpy.__version__!='1.26.4'
                or cv2.__version__!='4.6.0' or not re.search(r'GUI:\s+NONE',cv2.getBuildInformation())):
            raise ValueError('Previously pinned torch/numpy/headless OpenCV changed')
        if torch.cuda.is_initialized():raise ValueError('CPU import audit unexpectedly initialized CUDA')
        r['cuda_initialized']=False
    check('actual class imports and fixed versions',classes)
    def source():
        from run_evaluation import package_check
        p,_=package_check();r['evaluation_protocol_identity']=p['protocol_identity']
    check('original immutable evaluation sources',source)
    r['passed']=not r['errors']
    output.write_text(json.dumps(r,indent=2)+'\n')
    print('LIBERO_PARENT_DEPENDENCIES',json.dumps(r),flush=True)
    return r['passed']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    p.add_argument('--verify',action='store_true');p.add_argument('--reference',type=Path)
    args=p.parse_args();root=args.root.resolve()
    env=(root/'envs/libero-eval-v1').resolve()
    if Path(sys.prefix).resolve()!=env:raise ValueError('Run with the isolated eval-venv Python only')
    output=root/'runs/libero-parent-overlay-repair';output.mkdir(parents=True,exist_ok=True)
    if args.verify:
        if not verify(root,json.loads(args.reference.read_text()),output/'dependency_report.json'):
            raise SystemExit('Dependency audit failed; see all collected errors; no job submitted')
        return
    episodes=root/'runs/libero-original-dino-noaug-s42/episodes'
    if any(episodes.glob('*.json')):raise ValueError('Completed evaluation results exist; do not change their runtime')
    marker=root/'libero_original_evaluation_v1/job_id.txt'
    if marker.exists():
        job=marker.read_text().strip()
        if not job.isdigit():raise ValueError('Invalid previous evaluation job marker')
        active=subprocess.run(['squeue','-h','-j',job,'-o','%T'],text=True,capture_output=True)
        if active.stdout.strip():raise ValueError('Previous evaluation job still active: '+active.stdout.strip())
    parent=root/'envs/multitask-preflight-v1'
    reference=parent_reference(parent/'bin/python')
    if reference['errors']:raise ValueError('Working parent metadata errors: '+json.dumps(reference['errors']))
    if Path(reference['prefix']).resolve()!=parent.resolve():raise ValueError('Unexpected working parent prefix')
    sites=[Path(x).resolve() for x in reference['sites']]
    if any(not x.is_dir() or not x.is_relative_to(parent.resolve()) for x in sites):
        raise ValueError('Parent overlay must be within the known working parent venv')
    binding=json.loads((root/'oxe_dino_noaug_training_v1/cluster_preflight_binding.json').read_text())
    if (reference['versions']['torch']!=binding['torch_version']
            or reference['versions']['transformers']!=binding['packages']['transformers']
            or reference['versions']['Pillow']!=binding['packages']['Pillow']):
        raise ValueError('Working parent differs from passed training preflight')
    target=Path(sysconfig.get_path('purelib')).resolve()/'zz_oxe_parent_overlay.pth'
    if not target.parent.is_relative_to(env):raise ValueError('Child site path escapes eval environment')
    text=overlay_line(sites)
    if target.exists() and target.read_text()!=text:raise ValueError('A different parent overlay is already configured')
    target.write_text(text)
    ref=output/'working_parent_reference.json';ref.write_text(json.dumps(reference,indent=2)+'\n')
    print('EVAL_PARENT_OVERLAY_BOUND',str(target),[str(s) for s in sites],flush=True)
    # New process is essential: writing a .pth cannot change the current startup state.
    subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),
                    '--verify','--reference',str(ref)],check=True,
                   env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'})
    # This only resubmits the unfinished original episode. Seed43 job227920 is untouched.
    subprocess.run(['bash','submit_once.sh','1'],cwd=root/'libero_original_evaluation_v1',check=True)


if __name__=='__main__':main()
