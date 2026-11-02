"""Finish the saved-artifact audit and record207746; never execute a model."""
import csv
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bridge_rotation_weight_analysis_207746'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    assert not (OUT/'completion.json').exists(), 'Do not repeat completed analysis'
    a=read(OUT/'analysis.json');assert a['verified'] and a['job_id']=='207746'
    pack=ROOT/'cluster_packages/bridge_rotation_weight_v1'
    baseline=ROOT/'cluster_packages/bridge_coverage_replication_v1'
    pre=ROOT/'reports/bridge-rotation-weight-preflight-207722'
    fit=ROOT/'reports/bridge-rotation-weight-fit-207746'
    p=read(pack/'protocol.json')
    # Before optimization, the real cluster predictions/contexts/labels match the old seed42 arm.
    with np.load(pre/'common_u0000_train.npz',allow_pickle=False) as x, np.load(ROOT/'reports/bridge-coverage-rep-preflight-207521/common_u0000_train.npz',allow_pickle=False) as y:
        assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
    origin=ROOT/'cluster_packages/bridge_training_coverage_v1'
    assert sha(origin/'run_training_coverage.py')==p['cache_origin']['original_entry_sha256']
    assert sha(origin/'coverage_inputs.py')==p['cache_origin']['original_helper_sha256']
    assert sha(ROOT/'reports/bridge-coverage-preflight-207334/report.json')==p['cache_origin']['prior_preflight_report_sha256']
    assert sha(ROOT/'reports/bridge-coverage-fit-207348/report.json')==p['cache_origin']['prior_formal_report_sha256']
    diff=''.join(difflib.unified_diff((baseline/'run_training_coverage.py').read_text().splitlines(True),(pack/'run_training_coverage.py').read_text().splitlines(True),fromfile='coverage207558',tofile='rotation_weight207746'))
    (OUT/'runner_difference.patch').write_text(diff,encoding='utf-8')
    d=a['metrics']['development'];n=d['coverage'];old=d['original'];static=d['static']
    changes={k:dict(percent_change_vs_control=100*(n[k]/old[k]-1),percent_improvement_vs_static=100*(1-n[k]/static[k])) for k in n}
    tasks=[v for v in csv.DictReader((OUT/'tasks.csv').open(encoding='utf-8-sig')) if v['split']=='development']
    assert len(tasks)==5 and all(float(t['coverage_rotation_deg'])<float(t['original_rotation_deg']) for t in tasks)
    completion=dict(verified=True,analysis_sha256=sha(OUT/'analysis.json'),initial_saved_arrays_exact_vs_control=True,
        cache_origin_source_and_report_hashes_verified=True,changes=changes,all5_tasks_rotation_better_control=True,
        task_position_better_control=sum(float(t['coverage_position_cm'])<float(t['original_position_cm']) for t in tasks),
        task_position_better_static=sum(float(t['coverage_position_cm'])<float(t['static_position_cm']) for t in tasks),
        task_rotation_better_static=sum(float(t['coverage_rotation_deg'])<float(t['static_rotation_deg']) for t in tasks),
        candidate_passed=a['candidate_passed'],new_neural_execution=False,new_job_submitted=False,
        cache_file_independently_downloaded=False,checkpoint_downloaded=False)
    assert not a['candidate_passed'] and a['gates']['rotation_threshold'] and a['gates']['episode_rotation_majority'] and not a['gates']['position_threshold']
    document='''# 207746旋转权重对照：独立验收完成

21个小文件、7份保存预测齐全，约12.08MB压缩包。未下载权重或完整特征缓存，未重新训练、编码或采样。以下均为保存数组的float64独立复算，替代此前终端待验收状态。

| 原40开发窗口，来自10条独立演示 | 静止基线 | 原大池207558 | 旋转加权207746 |
|---|---:|---:|---:|
| 路径位置/cm | 8.212326 | 6.808208 | 7.301274 |
| 路径旋转/° | 15.098500 | 15.138891 | **12.275860** |
| 第16目标位置/cm | 10.565334 | 8.221684 | 8.653703 |
| 第16目标旋转/° | 19.986393 | 19.032170 | **16.242081** |

相对同初始化207558，路径旋转误差降低18.91%，位置误差增加7.24%；第16目标旋转降低14.66%，位置增加5.25%。相对静止，路径位置仍改善11.09%、旋转改善18.69%，第16目标两项也改善。

10条演示中，8条旋转优于原模型、6条优于静止；位置4条优于原模型、8条优于静止。按演示等权平均也呈同样方向：新模型6.896561cm/11.965259°，原模型6.449932cm/15.213293°，静止7.960118cm/15.066972°。旋转收益并非只靠一个大误差演示。

## 原判据与结论

- 旋转≤14.381946°：通过。
- 至少6/10演示旋转优于207558：8/10，通过。
- 位置≤6.808208cm：7.301274cm，未通过。

**完整候选未通过原定位置保护。** 不放宽门槛，不把所有结果说成退步，也不把它当成全面解决。损失权重确实能改变这套模型的位置/旋转取舍，但不能据此断定原权重是唯一根因或35.445是最优值。共享参数和全局梯度裁剪会共同影响优化。新增总损失和旧总损失不具备直接大小可比性。

## 五个任务均保留

| 任务 | 原位置/cm | 新位置/cm | 原旋转/° | 新旋转/° | 静止位置/cm | 静止旋转/° |
|---|---:|---:|---:|---:|---:|---:|
'''
    labels={'close the drawer':'关抽屉','close microwave':'关微波炉','take broccoli out of pan':'取出西兰花','turn lever vertical to front':'转动拨杆','open the drawer':'开抽屉'}
    for t in tasks:
        cols=['original_position_cm','coverage_position_cm','original_rotation_deg','coverage_rotation_deg','static_position_cm','static_rotation_deg']
        document+='| '+labels[t['task']]+' | '+' | '.join(f'{float(t[k]):.3f}' for k in cols)+' |\n'
    document+='''
五任务旋转均优于原模型；只有拨杆任务的位置优于原模型。对静止基线，四任务位置更好，开抽屉位置仍稍差；三任务旋转更好，微波炉和开抽屉仍稍差。总体改善不等于每个任务都改善。

## 完整性核验

实际源码/损失源码/协议快照与冻结包一致；415条演示payload、1876个窗口身份、405训练演示与10开发演示隔离、完整缓存key及原保留输入一致。与207558同初始化42、同批索引、同噪声与采样设置、同状态统计。训练前四窗口预测、目标、上下文与207521保存数组逐元素一致。

8192条更新日志的损失分项和权重公式全部核对，4096更新后学习率由1e-4变为1e-5；完整预检的恢复同学习率组日志与保存预测精确一致，正式首4步与预检一致。全部1836训练目标重新计算的统计与事前固定系数35.444970688934426完全一致，未用开发标签选择系数。CLIP与冻结参数完整性由运行时哈希/断言确认，未下载模型再次执行。

7份数组的指标复算与报告最大差异：位置0.000001674cm、旋转0.000636°，属数值精度差异。全1836训练窗6.631538cm/10.544924°；训练拟合只作诊断，不追加毫米级拟合。峰值保留显存5.623GiB。

本项按固定系数、固定初始化、固定8192预算结束，不扫权重、不追加种子或训练。当前没有新作业。仅一个训练初始化、三个扩散采样种子，开发集已反复使用，不是未触碰最终测试；离线末端位姿误差不能证明物理任务或抓取成功，夹爪冻结。

证据：[独立验收JSON](../reports/bridge_rotation_weight_analysis_207746/analysis.json)、[演示表](../reports/bridge_rotation_weight_analysis_207746/episodes.csv)、[任务表](../reports/bridge_rotation_weight_analysis_207746/tasks.csv)、[各预测目标表](../reports/bridge_rotation_weight_analysis_207746/horizons.csv)、[补充核验](../reports/bridge_rotation_weight_analysis_207746/completion.json)。
'''
    (ROOT/'docs/BRIDGE_ROTATION_WEIGHT_RESULT_207746_20261016.md').write_text(document,encoding='utf-8')
    note='**207746/207722已独立验收完成。** 21文件/7预测、源码/缓存指纹/415演示隔离/初始化42/8192损失日志与预检恢复核验。原40开发7.301274cm/12.275860°，第16目标8.653703cm/16.242081°；相对207558旋转改善18.91%、位置退步7.24%。8/10演示旋转优于原模型，五任务旋转均改善；位置仅4/10演示、1/5任务改善。两项总体均优于静止，但原位置保护未达，完整候选未通过。固定系数分支收口，不扫权重/加种子/拟合；无新作业或NN执行。见[独立结果](BRIDGE_ROTATION_WEIGHT_RESULT_207746_20261016.md)。下文待下载/待验收为历史。'
    for name in ['STATUS_20261015.md','PROGRESS.md','BRIDGE_ROTATION_WEIGHT_CONTROL_20261016.md']:
        path=ROOT/'docs'/name;parts=path.read_text(encoding='utf-8').split('\n\n',2)
        assert len(parts)==3
        path.write_text(parts[0]+'\n\n'+note+'\n\n'+parts[2],encoding='utf-8')
    for job in ['207722','207746']:
        path=ROOT/f'reports/bridge_rotation_weight_submission_{job}.json';record=read(path)
        record.update(report_verified=True,artifact_audit_sha256=sha(OUT/'analysis.json'),new_job_submitted=False,
            next_step='Fixed coefficient trial closed after saved-array audit; no weight sweep, extra seed, extension or repeated fit.')
        if job=='207746':
            record.update(preflight_report_independently_downloaded_and_verified=True,generalization_evidence_level='independent_float64_saved_array_audit',
                verified_metrics=d,gates=a['gates'],candidate_passed=False)
        else: record.update(validator_evidence_source='complete_saved_report_source_logs_arrays_independently_verified_except_cluster_cache_file_not_downloaded')
        write(path,record)
    write(OUT/'completion.json',completion)
    print(json.dumps(completion,indent=2))

if __name__=='__main__':main()
