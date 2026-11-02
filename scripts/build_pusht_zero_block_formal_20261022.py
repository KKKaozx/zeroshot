"""Derive one zero-block training control from the pinned original training entry."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'scripts/cluster/pusht_project_training_v1'
PRE=ROOT/'scripts/cluster/pusht_zero_block_preflight_v1'
B=ROOT/'scripts/cluster/pusht_zero_block_formal_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,text):(B/name).write_text(text,encoding='utf-8',newline='\n')


def derive():
    source=(OLD/'train_project.py').read_text(encoding='utf-8');changes=[]
    def replace(before,after,count=1):
        nonlocal source
        if source.count(before)!=count:raise ValueError('Unexpected source layout: '+before[:90])
        source=source.replace(before,after);changes.append(dict(before=before,after=after,count=count))
    replace('import training_core as core\n','import training_core as core\nimport zero_state\nfrom zero_blocks import remove_blocks\nfrom bind_preflight import verify_binding\nfrom pair_metrics import compare\n')
    replace('payload = core.capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha)',
        'payload = zero_state.capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha, contract_sha)')
    replace('choices=(1, 2), required=True','choices=(1,), required=True')
    replace('    args = parser.parse_args()','    parser.add_argument("--binding", type=Path, required=True)\n    args = parser.parse_args()')
    replace('        data = TrainingData(root, protocol, output)',
        '''        binding = verify_binding(args.binding.resolve(), root, bundle)
        binding_sha = sha(args.binding)
        shutil.copyfile(args.binding, output / "preflight_binding.json")
        data = TrainingData(root, protocol, output)''')
    replace('        normalizer = LinearNormalizer()',
        '''        initialization = remove_blocks(policy.obs_encoder)
        if initialization != binding["initialization"]:
            raise ValueError("Common norm/projection initialization differs from admitted zero-block preflight")
        normalizer = LinearNormalizer()''')
    replace('''            return snapshot(policy, averaged, optimizer, scheduler, ema, completed,
                            protocol_sha, plan_sha, contract_sha)''',
        '''            payload = snapshot(policy, averaged, optimizer, scheduler, ema, completed,
                               protocol_sha, plan_sha, contract_sha)
            payload["preflight_binding_sha256"] = binding_sha
            return payload''')
    replace('                       "startup_checks": startup}',
        '                       "startup_checks": startup, "preflight_binding_sha256": binding_sha}')
    replace('            validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, args.segment)',
        '''            validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, args.segment)
            if sidecar.get("preflight_binding_sha256") != binding_sha:
                raise ValueError("Resume uses a different admitted preflight")''')
    replace('core.restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)',
        'zero_state.restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha, contract_sha)',2)
    replace('                      fixed_text_digest=fixed_text_digest)',
        '''                      fixed_text_digest=fixed_text_digest)
        result.update(preflight_binding_sha256=binding_sha, zero_block_initialization=initialization,
                      active_cross_attention_blocks=0, actual_gpu_name=torch.cuda.get_device_name(0),
                      actual_node=os.environ.get("SLURMD_NODENAME", os.environ.get("HOSTNAME")),
                      comparison_scope=contract["comparison_scope"])''')
    replace('''        if args.segment == 2:
            del optimizer, scheduler, policy''',
        '''        if completed == 84168:
            del optimizer, scheduler, policy''')
    replace('            development = evaluate(LiveProjectPolicy(averaged, client), output, protocol)',
        '''            ema_before = state_hash(averaged)
            development = evaluate(LiveProjectPolicy(averaged, client), output, protocol)''')
    replace('            result.update(development_complete=True, development=development, evaluation_worker=close)',
        '''            if (state_hash(averaged) != ema_before
                    or any(p.requires_grad or p.grad is not None for p in averaged.parameters())):
                raise ValueError("EMA changed during evaluation")
            result.update(development_complete=True, development=development, evaluation_worker=close,
                          ema_unchanged_during_evaluation=True)
            reference = json.loads((bundle / "eight_block_reference.json").read_text())
            outcome = compare(development, reference["development"], protocol, contract)
            save_json(output / "comparison.json", outcome)
            result.update(formal_complete=True, zero_vs_eight=outcome)
            print("ZERO_VS_EIGHT", json.dumps(outcome), flush=True)''')
    source=source.replace('PUSHT_PROJECT','ZERO_BLOCK').replace('PUSHT PROJECT SEGMENT COMPLETE','ZERO BLOCK FORMAL COMPLETE')
    source=source.replace('"""Matched frozen-CLIP frontend training, exactly two continuous budget segments."""',
        '"""Zero-block control: original training loop, one84168-update endpoint, no automatic retry."""')
    return source,changes


def main():
    reused=('base_protocol.json','checkpoint_config.yaml','resource_probe.py','native_data.py','cache_tools.py',
        'token_binding.py','training_core.py','project_frontend.py','frozen_adapter.py','live_client.py','clip_worker.py',
        'live_contract.json','wire.py','base_verifier.py','evaluate_project.py')
    for name in reused:shutil.copyfile(OLD/name,B/name)
    for name in ('zero_blocks.py','zero_state.py','eight_block_reference.json'):
        shutil.copyfile(PRE/name,B/name)
    shutil.copyfile(PRE/'protocol.json',B/'preflight_protocol.json')
    shutil.copyfile(OLD/'project_contract.json',B/'original_project_contract.json')
    source,changes=derive();write('train_zero.py',source)
    write('entry_changes.json',json.dumps(dict(source_sha256=sha(OLD/'train_project.py'),changes=changes,
        extra_changes='Doc/log labels only; core loop/evaluator math unchanged'),indent=2)+'\n')
    write('formal.sh','''#!/bin/bash
#SBATCH --job-name=pusht-zero
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/pusht-zero-formal-%j.out
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_zero_block_formal_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BUNDLE"
sha256sum -c SHA256SUMS
ARGS=(--root "$ROOT" --output "$ROOT/runs/pusht-zero-formal-${SLURM_JOB_ID}" --segment 1 --binding "$BUNDLE/.submission/preflight_binding.json")
if [[ -n "${PUSHT_ZERO_RESUME:-}" ]]; then ARGS+=(--resume "$PUSHT_ZERO_RESUME"); fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B train_zero.py "${ARGS[@]}"
''')
    write('submit_once.sh','''#!/bin/bash
set -euo pipefail
ROOT=/projects/Zeroshot
BUNDLE="$ROOT/pusht_zero_block_formal_v1"
OFFICIAL="$ROOT/envs/official-pusht-v1"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
if ! mkdir .submission; then echo 'Already submitted; inspect saved job/checkpoint before retrying.' >&2; exit 1; fi
LD_LIBRARY_PATH="$OFFICIAL/lib" "$OFFICIAL/bin/python" -B bind_preflight.py --root "$ROOT" --binding "$BUNDLE/.submission/preflight_binding.json"
if JOB=$(sbatch --parsable formal.sh); then
  printf '%s\n' "$JOB" > .submission/job_id
  echo "Push-T无注意力块正式对照：$JOB"
else
  echo 'Submission failed; binding and guard retained for inspection.' >&2
  exit 1
fi
''')
    write('README.md','''# Push-T零块正式控制：只新增一组

承接219260成功预检。只训练一个零块控制，现有8块215999固定结果直接复用，不重训8层。CLIP冻结、保留原CLS/patch汇总/输出LayerNorm/1024→64投影与agent_pos；零块不利用语言，只作Push-T固定指令下的机制消融，不替代正式多指令模型。

从头seed42相同后端、seed43原8层完整构造后移除8个注意力/FFN块，逐项核对219260的共同初始化，绝不截断已训练215999，不加载预检权重。数据90演示/10726窗/11356帧、完整缓存、batch64/micro8/末批38、样本与噪声计划、epsilon/DDPM100/AdamW/EMA及原512400学习率时域不变。新控制完整84168更新，与旧8层同预算；没有更少步数的成绩替代比较。

原训练入口通过列明的有限替换派生，真实训练循环与评估数学不变。初始2步完整恢复检查和每2000/终点原子检查点保留，raw/EMA/Adam/LR/EMA计数/全部RNG独立零块协议绑定。允许显式--resume自己的完整检查点，禁止8层/原生/预检权重；没有自动重投。若中断，先检查已有计数与输出，不从头重跑。

预检每步0.547230543秒，84168纯训练估算12.794小时，另有载入、保存/哈希、闭环及排队；申请A40/4CPU/40GiB/20小时硬上限，不保证结束时间。8层原训练A6000、此控制A40，硬件/数值和单训练seed限制必须报告；不能宣称唯一注意力算子因果。

只在84168固定EMA评价一次原同20开发场景/同噪声流/两帧16→8/300步，固定前2视频，无中间选模，不访问final50。旧8层闭环不重跑。比较零减八奖励：均差>=0.05、场景配对bootstrap95%下界>0且满奖励比例不低，才称此设置下证据支持移除8块；对称负差/上界<0且满比例不高才支持保留8块；其余为未定。开发集已复用，三个分支都不自动追加预算/seed/变体，也不替换主语言模型或开放最终集。

提交时先调用原219260输出验证器一次，绑定原报告/初始化与新协议SHA，不重跑预检。请保留原pusht_zero_block_preflight_v1及runs/pusht-zero-preflight-219260目录。

提交：bash /projects/Zeroshot/pusht_zero_block_formal_v1/submit_once.sh

完成验证：/projects/Zeroshot/envs/official-pusht-v1/bin/python /projects/Zeroshot/pusht_zero_block_formal_v1/verify_formal.py /projects/Zeroshot/runs/pusht-zero-formal-实际编号

输出report/comparison/development/latest.json、latest.pt、updates.jsonl、绑定、startup_checks和前2固定视频。大权重留服务器。只做这一组控制，不重开空间旁路、不补已存在FiLM，不改原模型或依赖。
''')
    c=json.loads((OLD/'project_contract.json').read_text())
    c.update(stage='zero_block_single_fresh_control',segment_endpoints=[84168],total_updates=84168,
        preflight_job=219260,preflight_directory='runs/pusht-zero-preflight-219260',
        preflight_protocol_sha256=sha(B/'preflight_protocol.json'),reference_eight_block_job=215999,
        active_cross_attention_blocks=0,diagnostic_reward_margin=.05,paired_bootstrap_seed=20261022,
        source_train_entry_sha256=sha(OLD/'train_project.py'),
        comparison_scope='SameCLIP0vs8 whole attention/FFN stack incl capacity/dropout; reused dev20,1 seed,oldA6000/newA40; not final or replacement language policy',
        measured_preflight=dict(warm_update_seconds=.5472305431962013,peak_reserved_gib=6.63671875,
            estimated_formal_training_hours_excluding_io_and_rollout=12.794250099927188),
        resource_request=dict(gpu='a40:1',cpus=4,memory_gib=40,walltime_hours=20,no_requeue=True),
        trained_preflight_or_reference_weights_allowed=False,automatic_following_action=False)
    c['reused_files_byte_identical']={name:'pusht_project_training_v1/'+name for name in reused}
    files=sorted(p for p in B.iterdir() if p.is_file() and p.name not in ('project_contract.json','SHA256SUMS'))
    c['bundle_sources_sha256']={p.name:sha(p) for p in files}
    write('project_contract.json',json.dumps(c,indent=2)+'\n')
    for f in B.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
    files=sorted(p for p in B.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    write('SHA256SUMS',''.join(sha(p)+'  '+p.name+'\n' for p in files))
    archive=ROOT/'cluster_packages/pusht_zero_block_formal_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(B.iterdir()):
            if f.is_file():z.write(f,B.name+'/'+f.name)
    result=dict(archive=str(archive),archive_sha256=sha(archive),files=len(files)+1,bytes=archive.stat().st_size,
        contract_sha256=sha(B/'project_contract.json'),preflight_job=219260,reference8_job=215999,
        formal_training_submitted=False,local_validation_pending=True,
        total_new_formal_updates=84168,estimated_pure_training_hours=12.794250099927188)
    (ROOT/'reports/pusht_zero_block_formal_release_20261022.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
