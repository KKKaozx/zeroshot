if /projects/Zeroshot/envs/multitask-preflight-v1/bin/python - <<'PY'
import json
r = json.load(open('/projects/Zeroshot/runs/bridge-history-preflight-204150/report.json'))
assert r['passed'] and r['stage'] == 'preflight' and r['error'] is None
assert r['initial_conditions_and_predictions_exact'] and r['real_history_projection_gradient_exists']
print('执行预检通过：', r['passed'])
print('两组初始条件和预测完全一致：', r['initial_conditions_and_predictions_exact'])
print('原控制重放差异：', r['original_initial_replay'])
print('前16次控制损失最大差：', r['first16_control_loss_max_difference'])
print('真实历史分支有梯度：', r['real_history_projection_gradient_exists'])
PY
then
    HISTORY_FIT=$(sbatch --parsable --export=ALL,HISTORY_PREFLIGHT_JOB=204150 /projects/Zeroshot/bridge_history_generalization_v1/fit.sh)
    echo "历史正式对照：$HISTORY_FIT"
    squeue --start -j "$HISTORY_FIT"
fi
