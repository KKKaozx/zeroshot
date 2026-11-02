"""CPU import boundary only; never creates an environment/model or EGL context."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/projects/Zeroshot');args=p.parse_args()
    root=Path(args.root).resolve()
    out=root/'runs/libero-dependency-repair-227817'
    out.mkdir(parents=True,exist_ok=True)
    config=out/'config';config.mkdir(exist_ok=True)
    base=root/'libero_runtime_v1/libero/libero'
    import yaml
    (config/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(base),
        'bddl_files':str(base/'bddl_files'),'init_states':str(base/'init_files'),
        'assets':str(base/'assets'),'datasets':str(out)}))
    os.environ.update(LIBERO_CONFIG_PATH=str(config),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    sys.path.insert(0,str(root/'libero_runtime_v1'))
    from termcolor import colored
    from libero.libero.envs import OffScreenRenderEnv
    import torch
    if torch.__version__!='2.5.1+cu118':raise ValueError('Inherited torch must remain unchanged')
    report={'passed':True,'termcolor':metadata.version('termcolor'),'torch':torch.__version__,
        'native_environment_class_imported':OffScreenRenderEnv.__name__,
        'cuda_initialized':torch.cuda.is_initialized(),'environment_instances':0,
        'model_forwards':0,'physics_steps':0,'training_updates':0,
        'formal_completed_episodes':0,'scope':'Missing dependency/import fix only; actual EGL rollout still untested'}
    if report['cuda_initialized']:raise ValueError('Import check unexpectedly initialized CUDA')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('LIBERO IMPORT DEPENDENCY VERIFIED',json.dumps(report),flush=True)


if __name__=='__main__':main()
