# 直接正式入口失败留证补丁

226603直接启动再次触发原合并身份守卫；226586使用runpy/trace的诊断没有复现，不能当作直接启动已修复。

本包仅新增正常启动时的两项校验结果打印，并在原错误分支记录真实冻结摘要、分词器差异、逐名冻结张量差异、调度缓冲区、实际模块路径、CPU能力和少量执行环境字段。原条件与ValueError完全保留，错误仍在任何前向/更新前停止；不重绑、不调容差、不改变权重/训练协议/数据/预算。校验正常通过才继续原训练。此包不自动提交。

apply_patch.py验证已知原始或本补丁manifest及每个文件；所有seed输出必须为空，且无对应活动作业。保存被修改文件与manifest的小备份，再更新代码/manifest。不会覆盖训练输出或作业marker；原部署zip本地保留。仅适用于尚未产生训练checkpoint的当前包。

上传解压后用python3 apply_patch.py --root /projects/Zeroshot，再回原训练包运行bash submit_stage_once.sh 42 1 retry。保持A40直接脚本启动，不使用runpy/trace。正式作业若失败，读取runs/oxe-id-failure-JOBID/report.json；正常通过则日志出现FORMAL_IDENTITY_CHECK并继续训练。首次失败记录仍是证据，不宣称此补丁修复了身份差异原因。
