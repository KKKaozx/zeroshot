"""Minimal audited transformation of the qualified original8 training entry."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'scripts/cluster/pusht_project_training_v1'
CACHE = ROOT/'scripts/cluster/pusht_dino_cache_v1'
B = ROOT/'scripts/cluster/pusht_dino_training_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,text):
    (B/name).write_text(text,encoding='utf-8',newline='\n')


def transform_entry(source):
    changes = []
    def replace(label,old,new,count=1):
        nonlocal source
        actual = source.count(old)
        if actual != count:raise ValueError((label,actual,count))
        source = source.replace(old,new)
        changes.append(dict(label=label,sites=count))
    replace('Description', 'Matched frozen-CLIP frontend training, exactly two continuous budget segments.',
            'Matched frozen-DINO visual control, original eight-block frontend and two budget segments.')
    replace('Identity helpers','import training_core as core\n',
            'import training_core as core\nimport dino_state\nfrom formal_binding import verify_binding, storage_guard\n')
    replace('Resume sidecar binding argument',
            'def validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, segment):',
            'def validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, segment, binding_sha):\n'
            '    if sidecar.get("dino_backbone_binding_sha256") != binding_sha:\n'
            '        raise ValueError("Resume visual backbone/cache identity differs")')
    replace('Complete DINO checkpoint format',
            'def snapshot(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha, contract_sha):\n'
            '    payload = core.capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha)\n'
            '    payload["project_contract_sha256"] = contract_sha\n'
            '    return payload',
            'def snapshot(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha, contract_sha, binding_sha):\n'
            '    return dino_state.capture(policy, averaged, optimizer, scheduler, ema, completed,\n'
            '                              protocol_sha, plan_sha, contract_sha, binding_sha)')
    replace('Quota and atomic-checkpoint headroom',
            '        if shutil.disk_usage(output).free < 12 * 1024 ** 3:\n'
            '            raise ValueError("Need12GiB free for atomic complete checkpoints; preserve cache and source checkpoint")',
            '        result["storage_at_start"] = storage_guard(root, contract)')
    replace('Qualified cache binding before loading data',
            '        data = TrainingData(root, protocol, output)',
            '        binding = verify_binding(root, bundle)\n'
            '        binding_sha = contract["dino_backbone_binding_sha256"]\n'
            '        shutil.copyfile(root / contract["cache_directory"] / "backbone_binding.json", output / "backbone_binding.json")\n'
            '        data = TrainingData(root, protocol, output)')
    replace('Bind complete snapshot',
            '                            protocol_sha, plan_sha, contract_sha)',
            '                            protocol_sha, plan_sha, contract_sha, binding_sha)')
    replace('Bind checkpoint sidecar','"startup_checks": startup}',
            '"startup_checks": startup, "dino_backbone_binding_sha256": binding_sha}')
    replace('Validate resume sidecar binding',
            'validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, args.segment)',
            'validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, args.segment, binding_sha)')
    replace('Validate DINO complete restore before modifying models',
            'core.restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)',
            'dino_state.restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha, contract_sha, binding_sha)',2)
    replace('Correct cache label','frozen_clip_cache_verified=True','frozen_dino_cache_verified=True')
    replace('Record independent visual control scope','                      fixed_text_digest=fixed_text_digest)',
            '                      fixed_text_digest=fixed_text_digest)\n'
            '        result.update(dino_backbone_binding_sha256=binding_sha, dino_training_executed=False,\n'
            '                      active_cross_attention_blocks=8, actual_gpu_name=torch.cuda.get_device_name(0),\n'
            '                      actual_node=os.environ.get("SLURMD_NODENAME", os.environ.get("HOSTNAME")),\n'
            '                      comparison_scope=contract["comparison_scope"])')
    replace('Use qualified DINO live worker','            from live_client import CLIPClient, LiveProjectPolicy',
            '            from dino_client import DINOClient, LiveProjectPolicy')
    replace('Instantiate DINO worker','            client = CLIPClient(root, output)',
            '            client = DINOClient(root, output)')
    replace('Read-only fixed endpoint EMA','            development = evaluate(LiveProjectPolicy(averaged, client), output, protocol)',
            '            ema_before = state_hash(averaged)\n'
            '            development = evaluate(LiveProjectPolicy(averaged, client), output, protocol)')
    replace('Check endpoint backbone and paired comparison',
            '            result.update(development_complete=True, development=development, evaluation_worker=close)',
            '            if (state_hash(averaged) != ema_before\n'
            '                    or any(p.requires_grad or p.grad is not None for p in averaged.parameters())):\n'
            '                raise ValueError("EMA changed during fixed endpoint evaluation")\n'
            '            if any(close[k] != binding[k] for k in ("model_state_sha256", "backbone_weights_sha256", "preprocessing")):\n'
            '                raise ValueError("Live evaluation visual backbone differs from bound training cache")\n'
            '            result.update(development_complete=True, development=development, evaluation_worker=close,\n'
            '                          ema_unchanged_during_evaluation=True)\n'
            '            from compare_backbones import compare\n'
            '            outcome = compare(development, json.loads((bundle / "clip_development_reference.json").read_text()),\n'
            '                              json.loads((bundle / "native_development_reference.json").read_text()), protocol)\n'
            '            save_json(output / "comparison.json", outcome)\n'
            '            result["backbone_comparison"] = outcome\n'
            '            print("DINO_BACKBONE_COMPARISON", json.dumps(outcome), flush=True)')
    # Log labels do not change model execution.
    source = source.replace('PUSHT_PROJECT_','PUSHT_DINO_').replace('PUSHT PROJECT SEGMENT COMPLETE','PUSHT DINO SEGMENT COMPLETE')
    changes.append(dict(label='DINO log labels only',sites='all original project labels'))
    return source,changes


def main():
    B.mkdir(exist_ok=True)
    reused = ['base_protocol.json','base_verifier.py','cache_tools.py','checkpoint_config.yaml',
        'native_data.py','project_frontend.py','frozen_adapter.py','resource_probe.py',
        'token_binding.py','training_core.py','live_client.py','wire.py','evaluate_project.py']
    for name in reused:(B/name).write_bytes((OLD/name).read_bytes())
    (B/'frozen_visual.py').write_bytes((CACHE/'frozen_visual.py').read_bytes())
    (B/'visual_protocol.json').write_bytes((CACHE/'protocol.json').read_bytes())
    for name in ('dino_state.py','formal_binding.py','compare_backbones.py','verify_dino.py'):
        (B/name).write_bytes((ROOT/'experiments/pusht_dinov2_v1'/name).read_bytes())
    for name in ('dino_worker.py','dino_client.py'):
        text = (CACHE/name).read_text(encoding='utf-8')
        assert text.count("'protocol.json'") == 1
        write(name,text.replace("'protocol.json'","'visual_protocol.json'"))
    original = (OLD/'train_project.py').read_text(encoding='utf-8')
    source,changes = transform_entry(original)
    write('train_dino.py',source)
    write('entry_changes.json',json.dumps(dict(original_entry_sha256=sha(OLD/'train_project.py'),
        new_entry_sha256=sha(B/'train_dino.py'),changes=changes,
        unchanged='Original loss/update/microbatch/RNG/optimizer/LR/EMA/sample-plan and first disk replay execution'),indent=2)+'\n')
    received = ROOT/'reports/pusht_dino_cache_221055'
    (B/'cache_report_reference.json').write_bytes((received/'report.json').read_bytes())
    (B/'cache_binding_reference.json').write_bytes((received/'backbone_binding.json').read_bytes())
    for group,name in (('pusht_project_215999','clip'),('pusht_native_212158','native')):
        (B/(name+'_development_reference.json')).write_bytes((ROOT/'reports'/group/'development.json').read_bytes())
    for segment in (1,2):
        resume = '''if [[ -n "${PUSHT_DINO_RESUME:-}" ]]; then ARGS+=(--resume "$PUSHT_DINO_RESUME"); fi'''
        if segment==2:resume = ''': "${PUSHT_DINO_RESUME:?Set exact verified segment1 latest.pt}"
ARGS+=(--resume "$PUSHT_DINO_RESUME")'''
        write('part%d.sh'%segment,'''#!/bin/bash
#SBATCH --job-name=pusht-dino%d
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-dino-part%d-%%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
ARGS=(--root "$ROOT" --output "$ROOT/runs/pusht-dino-part%d-${SLURM_JOB_ID}" --segment %d)
%s
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B train_dino.py "${ARGS[@]}"
'''%(segment,segment,segment,segment,resume))
    write('submit_part1_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
if [[ -n "${PUSHT_DINO_RESUME:-}" ]]; then echo 'Fresh submit only; inspect interrupted runs before an explicit resume.' >&2; exit 1; fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B formal_binding.py --root "$ROOT"
if ! mkdir .part1_submission; then echo 'Already submitted; inspect saved job before any retry.' >&2; exit 1; fi
if JOB=$(sbatch --parsable part1.sh); then
  printf '%s\\n' "$JOB" > .part1_submission/job_id
  echo "Push-T DINO项目组第一段：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write('submit_part2_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_dino_training_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
: "${1:?Pass the verified DINO segment1 latest.pt}"
PUSHT_DINO_RESUME=$(realpath -e -- "$1")
export PUSHT_DINO_RESUME
case "$PUSHT_DINO_RESUME" in
  /projects/_ssd/Zeroshot/runs/pusht-dino-part1-*/latest.pt) ;;
  *) echo 'Expected exact DINO part1 checkpoint under canonical project/runs.' >&2; exit 1 ;;
esac
cd "$BUNDLE"
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B verify_dino.py "$(dirname "$PUSHT_DINO_RESUME")"
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B formal_binding.py --root "$ROOT"
if ! mkdir .part2_submission; then echo 'Already submitted; inspect saved job before retrying.' >&2; exit 1; fi
if JOB=$(sbatch --parsable --export=ALL part2.sh); then
  printf '%s\\n' "$JOB" > .part2_submission/job_id
  echo "Push-T DINO项目组第二段：$JOB"
else
  echo 'Submission failed; guard retained for inspection.' >&2
  exit 1
fi
''')
    write('README.md','''# 冻结DINO-L/14同预算视觉骨干对照

依据老师第六封邮件，在项目CLIP前端明显落后原生后，只替换视觉骨干。原CLIP8组215999和原生212158直接复用，不重新训练；冻结CLIP主项目不修改，不扩大/重抽数据，不追加零块/旁路/R3M分支。新DINO-L/14仍完全冻结，原8层Adapter、池化、1024→64投影、固定CLIP文本、agent_pos、作者U-Net各层FiLM保持。

数据90训练演示/11356帧/10726窗口、帧ID/原始像素及normalizer精确沿用。只用221055已完成的11.13GiB DINO缓存，引用220882实际资源测量，生成的checkpoint使用独立DINO格式并嵌入完整cache/backbone绑定SHA；旧CLIP、原生、零块和更换缓存的状态拒绝恢复。无需重复资源更新或缓存。Worker/client与221055只有protocol文件改名为visual_protocol.json，推理/通信操作不变。视觉协议原字节保留，sources条目属于历史缓存包；正式完整文件散列由project_contract.json独立检查。

正式新建共同seed42作者后端与seed43原8层前端，禁止任何已有学习端点初始化。AdamW/lr1e-4/betas.95,.999/weight_decay1e-6、原cosine horizon512400+warmup500、原epsilon-DDPM100、batch64/micro8含尾批38、独立shuffle/扩散/前端RNG、EMA power.75均原样保留。不是把25小时训练压缩到短schedule。

总有效84168更新，分42084/84168两段连续保存完整raw/EMA/Adam/LR/EMA计数/Python/NumPy/CPU/CUDA RNG。第一段内原首两步磁盘恢复audit必须逐状态/损失精确，额外重算一次不计入预算；每2000步覆盖本段latest.pt并用原子writing保存。第一段不评价、不挑checkpoint；核验后手工提交第二段。恢复带原合同与完整DINO绑定，不允许重置优化器、改变余弦进度或自动重投。

两段各A40/4CPU/40GiB/20小时硬上限/no-requeue。220882纯训练估算共25.46小时、每段约12.73小时，另有I/O/保存/第二段固定闭环，不是完成时间承诺。原CLIP使用A6000，新DINO使用A40，保留硬件和单训练seed的归因限制。

提交前与作业内检查du项目占用/文件系统空间，保守200十进制GB项目预算内至少18GiB余量；不是权威账户quota查询，不自动删任何文件。221055缓存、原数据、两个基准端点都保留。新正式完整checkpoint磁盘实际大小尚未实测。

最终仅84168 EMA在原dev20种子5000000..19、原DDPM采样计划和8步执行、max300步上评价一次。固定前两场视频，原max overlap reward含义不改。报告DINO/CLIP/原生均值和满奖励比例、DINO减CLIP配对增益与场景bootstrap区间；复用开发场景/一个训练seed，区间不含训练seed不确定性，不是final50确认或机器人抓取成功。final50不访问，不按dev中途调参、加预算或换候选，也不自动替换主CLIP模型。

DINO原生ImageNet归一化与最终LayerNorm随骨干变化；为保留画面，沿用旧224resize/224crop而非DINO默认256resize/224crop。结果属于该明确表示/预处理边界下的冻结视觉替换对照，不把任何差距归为Adapter单独因素。

第一段提交：bash /projects/Zeroshot/pusht_dino_training_v1/submit_part1_once.sh
第一段完成核验：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_dino_training_v1/verify_dino.py /projects/Zeroshot/runs/pusht-dino-part1-实际编号
第二段需核验后手工：bash /projects/Zeroshot/pusht_dino_training_v1/submit_part2_once.sh /projects/Zeroshot/runs/pusht-dino-part1-实际编号/latest.pt

若第一/二段中断，保留latest及日志，先检查错误和完整侧车，再另行审阅显式恢复。正常提交防重复；不会自行重新开始或更改预算。
''')
    old_contract = json.loads((OLD/'project_contract.json').read_text())
    required = {
        'runs/pusht-dino-cache-221055/report.json':sha(received/'report.json'),
        'runs/pusht-dino-cache-221055/backbone_binding.json':sha(received/'backbone_binding.json'),
        'runs/pusht-native84k-212158/report.json':sha(ROOT/'reports/pusht_native_212158/report.json'),
        'runs/pusht-project-part2-215999/report.json':sha(ROOT/'reports/pusht_project_215999/report.json')}
    contract = dict(stage='matched_frozen_dino_visual_training',
        base_protocol_sha256=sha(B/'base_protocol.json'),visual_protocol_sha256=sha(B/'visual_protocol.json'),
        cache_directory='runs/pusht-dino-cache-221055',cache_report_sha256=sha(received/'report.json'),
        dino_backbone_binding_sha256=sha(received/'backbone_binding.json'),required_reports_sha256=required,
        sample_order_sha256=old_contract['sample_order_sha256'],sample_plan_sha256=old_contract['sample_plan_sha256'],
        total_updates=84168,segment_endpoints=[42084,84168],lr_horizon_updates=512400,
        fresh_backend_seed=42,adapter_projection_seed=43,minimum_free_bytes=18*1024**3,
        conservative_project_budget_bytes=200_000_000_000,
        visual_backbone_frozen=True,clip_text_frozen=True,trained_native_or_clip_weights_allowed=False,
        native_reference_job=212158,clip_reference_job=215999,cache_job=221055,resource_job=220882,
        comparison_scope='Frozen visual representation control; original eight-block Adapter/backend/data/budget retained. '
                         'Native DINO normalization/final LayerNorm differ; preserve original CLIP224 camera geometry. '
                         'OldCLIP trainedA6000/newDINO trainedA40; one seed and reused dev20.',
        final50_access=False,automatic_following_segment_submission=False,
        unchanged_old_files_byte_identical=reused,
        modified_worker_client_scope='Only protocol.json filename changed to visual_protocol.json; operations unchanged')
    contract['bundle_sources_sha256'] = {f.name:sha(f) for f in sorted(B.iterdir())
        if f.is_file() and f.name not in ('project_contract.json','SHA256SUMS')}
    write('project_contract.json',json.dumps(contract,indent=2)+'\n')
    for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    files = sorted(f for f in B.iterdir() if f.is_file() and f.name!='SHA256SUMS')
    write('SHA256SUMS',''.join(sha(f)+'  '+f.name+'\n' for f in files))
    archive = ROOT/'cluster_packages/pusht_dino_training_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(B.iterdir()):
            if f.is_file():z.write(f,B.name+'/'+f.name)
    result = dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(files)+1,local_validation_pending=True,new_jobs_submitted=False,
        formal_training_started=False,resource_probe_repeated=False,cache_regenerated=False)
    (ROOT/'reports/pusht_dino_training_release_20261023.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
