"""Prepare an offline data adapter for the pinned author workspace, with no model downloads."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_author_preflight_v1'
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
AUTH=ROOT/'baselines/diffusion_policy_data_reference'
LOCAL=ROOT/'reports/bridge_author_contract_20261014'
CLIP=ROOT.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def prepare():
    assert not (PACK/'protocol.json').exists()
    result=read(ROOT/'reports/bridge_state_rep_analysis_202463_20261014/analysis.json')
    assert result['verified'] and not result['development_static_protocol_passed']
    manifest=read(BASE/'manifest.json');rows=manifest['train']+manifest['development']
    from transformers import CLIPModel,CLIPTokenizer
    import torch
    tasks=sorted({x['task'] for x in rows[:128]});assert len(tasks)==5
    assert all(x['task'] in tasks for x in rows)
    tokenizer=CLIPTokenizer.from_pretrained(str(CLIP),local_files_only=True)
    model=CLIPModel.from_pretrained(str(CLIP),local_files_only=True).eval()
    tokens=tokenizer(tasks,padding=True,truncation=True,max_length=77,return_tensors='pt')
    with torch.no_grad():language=model.text_model(**tokens).pooler_output.cpu().numpy().astype('<f4')
    assert language.shape==(5,768) and np.isfinite(language).all()
    np.savez_compressed(PACK/'language_cache.npz',texts=np.array(tasks),embeddings=language)
    provenance=read(AUTH/'provenance.json')['files']+read(AUTH/'bridge_workspace_provenance.json')['files']
    source_hash={x['path']:x['sha256'] for x in provenance if x['path'].endswith('.py')}
    assert all(sha(AUTH/n)==h for n,h in source_hash.items())
    state=ROOT/'cluster_packages/bridge_state_conditioning_v1'
    protocol=dict(date='2026-10-14',stage='same_bridge_author_workspace_resource_preflight',
        author_repository='https://github.com/real-stanford/diffusion_policy',
        author_commit='5ba07ac6661db573af695b419a7947ecb704690f',author_source_sha256=source_hash,
        base_manifest_sha256=sha(BASE/'manifest.json'),rows=rows,
        state_protocol_sha256=sha(state/'protocol.json'),current_pose_sha256=sha(state/'current_pose.npz'),
        language_cache_sha256=sha(PACK/'language_cache.npz'),language_input=dict(source='Frozen CLIP text_model.pooler_output',
            texts=tasks,dimension=768,canonical_instruction=True,fit_from_development=False,
            local_clip_snapshot=CLIP.name,clip_config_sha256=sha(CLIP/'config.json'),no_language_encoder_training=True),
        shapes=dict(image=[1,3,224,224],ee_state=[1,9],language=[1,768],action=[16,7]),
        normalization=dict(action='Author LinearNormalizer limits fit on train128 only; inverse before physical error',
            ee_state='Same current-state9 features, then train-only author limits',image='Invert existing CLIP mean/std without re-resizing or uint8 quantization; author image2*x-1',language='identity'),
        preflight=dict(actual_optimizer_updates=4,initialization=43,batch=8,checkpoint=False,rollout=False,
            holdout_policy_evaluation=False,history=1),
        planned_bounded_reference=dict(status='not prepared; requires successful resource/interface preflight',
            initialization=43,batch=8,epochs=512,updates=8192,draw_slots_per_window=512,
            max_planning_wall_hours=3,max_reserved_gib=40,final_checkpoints_budget_gib=3,
            evaluate_fixed_final_not_development_best=True,training_fit_required_before_development_verdict=True),
        changes_from_project=['Trainable author ResNet/SpatialSoftmax observation encoder instead of frozen visual CLIP+Adapter',
            'Frozen EOS-pooled768 language low_dim condition instead of language token cross attention',
            'Author widths512/1024/2048 and timestep128 instead of project256/512/1024 and timestep256',
            'Epsilon target, author train-only limits normalizer, AdamW/warmup-cosine/EMA instead of projectx0/Adam/lateLR',
            'One current observation; no added history, random crop, gripper supervision or rollout'],
        source_inputs_unchanged=['same128train/40development window identities','same image content and letterbox',
            'same observed current pose, instruction and first16 future-pose targets; no future observation'],
        limitations=['Author architecture adapted to Bridge pose-only multitask observations; not a full paper reproduction.',
            'Several modules differ simultaneously; any result is a reference, not a causal module verdict.',
            'Bounded8192-update plan is not original3050-epoch paper training budget.',
            'Reused development set; no unseen task, untouched test or robot grasp success claim.'])
    (PACK/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8',newline='\n')
    import yaml
    cfg=dict(name='bridge_author_preflight',policy=dict(_target_='diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy',
        shape_meta=dict(obs=dict(image=dict(shape=[3,224,224],type='rgb'),ee_state=dict(shape=[9],type='low_dim'),language=dict(shape=[768],type='low_dim')),action=dict(shape=[7])),
        noise_scheduler=dict(_target_='diffusers.schedulers.scheduling_ddpm.DDPMScheduler',num_train_timesteps=100,
            beta_start=.0001,beta_end=.02,beta_schedule='squaredcos_cap_v2',variance_type='fixed_small',clip_sample=True,prediction_type='epsilon'),
        horizon=16,n_action_steps=16,n_obs_steps=1,num_inference_steps=100,obs_as_global_cond=True,crop_shape=None,
        diffusion_step_embed_dim=128,down_dims=[512,1024,2048],kernel_size=5,n_groups=8,cond_predict_scale=True,
        obs_encoder_group_norm=True,eval_fixed_crop=True),
        optimizer=dict(_target_='torch.optim.AdamW',lr=1e-4,betas=[.95,.999],eps=1e-8,weight_decay=1e-6),
        ema=dict(_target_='diffusion_policy.model.diffusion.ema_model.EMAModel',update_after_step=0,inv_gamma=1.,power=.75,min_value=0.,max_value=.9999),
        task=dict(dataset=dict(_target_='bridge_author_dataset.BridgeAuthorDataset',cache='OVERRIDE',protocol='OVERRIDE'),env_runner=dict(_target_='bridge_author_dataset.NoRolloutRunner')),
        dataloader=dict(batch_size=8,num_workers=0,shuffle=True,pin_memory=True,persistent_workers=False),
        val_dataloader=dict(batch_size=4,num_workers=0,shuffle=False,pin_memory=True,persistent_workers=False),
        training=dict(device='cuda:0',seed=43,debug=False,resume=False,num_epochs=1,gradient_accumulate_every=1,
            lr_scheduler='cosine',lr_warmup_steps=500,use_ema=True,rollout_every=1,checkpoint_every=1,
            val_every=1,sample_every=1,max_train_steps=4,max_val_steps=1,tqdm_interval_sec=1.),
        logging=dict(project='bridge_author_preflight',mode='disabled',resume=False,name='contract_resource_only'),
        checkpoint=dict(save_last_ckpt=False,save_last_snapshot=False,topk=dict(monitor_key='train_loss',mode='min',k=0,format_str='unused.ckpt')))
    (PACK/'preflight_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8',newline='\n')
    shell='''#!/bin/bash
#SBATCH --job-name=bridge-author-preflight
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-author-preflight-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/author-bridge-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MPLCONFIGDIR="/projects/Zeroshot/.tmp/author-bridge-mpl"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" /projects/Zeroshot/runs
PROJECT_BYTES=$(du -sb /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_BYTES + 1024*1024*1024 + 15*1024*1024*1024 > 200000000000 )); then exit 1; fi
SOURCE=/projects/Zeroshot/paper_reproduction/official_pusht_v1/source
AUTHOR_ENV=/projects/Zeroshot/envs/official-pusht-v1
test "$(git -C "$SOURCE" rev-parse HEAD)" = 5ba07ac6661db573af695b419a7947ecb704690f
test -z "$(git -C "$SOURCE" status --porcelain)"
cd /projects/Zeroshot/bridge_author_preflight_v1
sha256sum -c SHA256SUMS
nvidia-smi
OUT="/projects/Zeroshot/runs/bridge-author-preflight-${SLURM_JOB_ID}"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="" /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u export_bridge_author_inputs.py \\
  --project /projects/Zeroshot --data-root /projects/Zeroshot/bridge_expansion_training_v1/data --output "$OUT/inputs"
LD_LIBRARY_PATH="$AUTHOR_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$AUTHOR_ENV/bin/python" -u preflight_bridge_author_workspace.py \\
  --source "$SOURCE" --cache "$OUT/inputs" --output "$OUT/author"
echo "AUTHOR BRIDGE RESOURCE PREFLIGHT COMPLETE: $OUT/author/report.json"
'''
    (PACK/'preflight.sh').write_text(shell,encoding='utf-8',newline='\n')
    print('Prepared author workspace preflight: same168 input identities, no full training launcher.')

def package():
    r=read(LOCAL/'verification.json');export=read(LOCAL/'export_report.json')
    assert r['passed'] and export['passed'] and export['audit_only']
    assert r['protocol_sha256']==sha(PACK/'protocol.json')==export['protocol_sha256']
    for f in PACK.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    (PACK/'local_contract_summary.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    files=sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files),encoding='ascii',newline='\n');files.append(PACK/'SHA256SUMS')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(z.read(p.relative_to(PACK.parent).as_posix())==p.read_bytes() for p in files)
    assert archive.stat().st_size<100*2**20
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))))
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','package')
    globals()[sys.argv[1]]()
