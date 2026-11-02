# 8维位姿与7维命令的本地执行接口修复

2026-10-31。本机根code已应用，原部署快照与全部模型权重未改；这是可审阅的源码包，不含训练或仿真启动入口，无集群作业/提交/tail命令。

现有转换本来就在libero_action_interface.py：按固定参考还原目标位置/旋转，相对于当前观测计算平移误差与世界轴角误差，除OSC各轴尺度，夹爪开正转闭正。它是反馈跟踪规则，不能保证实测未来到达位姿就是控制器期望目标；没有重新实现这段几何或声称已有物理根因修复。

新增libero_execution_binding.py，并接入libero_waypoint_execution.py：在调用策略前保存完整仿真state，策略返回后及每次env.step前要求state未变化；执行后重新绑定。过期命令、8维位姿直接进入7维命令入口、7维输出直接进入8维轨迹入口均拒绝。无截断维度、无新增插值/控制增益/专家动作。

这是同步调用期间的状态一致性检查。它不能证明调用者最初传入的图像/位姿一定最新，也不提供异步线程原子性；没有证据认定此前失败由过期状态导致。CPU假环境仅验证首条非零位置/轴角/夹爪转换、固定参考位置反馈与拒绝路径，不验证接触动力学。

本机复查（无需上传/提交）：

```powershell
& 'D:\ntu_related\dissertation\Zero_shot\training_cache\libero_native_env_v1\Scripts\python.exe' -B 'D:\ntu_related\dissertation\Zero_shot\tests\test_libero_execution_binding.py'
```

4项新CPU接口用例通过；原test_libero_pose_policy.py的4项CPU回归通过。新训练/模型推理/仿真均0。检查结束，不追加守卫、回合或控制扫描。原236840域内监督5/5成功参照保留；OXE源能力与到达位姿→原生命令的物理执行仍未解决，13模型未放行。
