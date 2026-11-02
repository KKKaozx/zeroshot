# 已落实：18D位姿输入、pose-port与双向空间映射接口

2026-10-30。本轮完成具体代码修复，范围只针对234100的Bridge当前位姿策略格式；不是新训练或新控制器。**B2中的“最新位姿支路不能被LIBERO推理回调消费”软件缺口已修复并通过针对性CPU检查。实际标定、真实权重加载和5Hz接触执行仍未通过，完整B2没有解除。**

## 可执行行为

新增 `code/libero_pose_policy.py::load_bridge_pose_policy` 和 `PoseConditionedWaypointPolicy`：

1. 恢复checkpoint的`state.base`和`state.pose_port`两组权重，以既有源结果报告的`final_identity`绑定；缺少port拒绝，不能以零port或旧基础回调替代。
2. 从当前请求的实测EEF位姿求对应源参考位姿。复用冻结部署包的原`pose_features.encode`，使用checkpoint原训练均值/标准差，生成9个值+9个mask的18D输入；不在LIBERO重拟合统计。
3. 使用源训练同样的77-token固定padding，计算`图文context + pose_port(current_pose_features)`，再调用原生完整sampler。没有改扩散步数、目标、损失或训练权重。
4. 把源位移、左相对旋转及保持的开正夹爪命令转换成执行器使用的目标世界相对轨迹。输入逆映射与输出正映射使用同一组参数，不只改其中一端。
5. 现有无pose-port回调保持原默认路径；对新checkpoint明确报错提示使用完整加载器，避免漏权重。现有执行频率、轨迹调度、夹爪取反位置和OSC控制不改。

`code/libero_pose_transfer.py`已经实现单位缩放、轴旋转、原点、右侧工具轴对应与两端TCP偏移。定义为`p_G^T=b+s A p_G^S`及`R_E^T=A R_E^S C`，G为声明的对应物理点。逆映射先考虑目标EEF到G的偏移，正映射先恢复源未来EEF及G，再还原目标EEF；旋转引起的TCP杠杆位移没有省略。每个未来点都相对同一个当前参考，不累加。

## 缺少的具体事实和文件

可填写参数文件：`configs/libero_bridge_pose_mapping_REQUIRED.json`。目前保留null，加载器会在加载模型或推理前列出缺项：

| 参数 | 需要什么事实，不能怎样猜 |
|---|---|
| `target_meters_per_source_unit` | 精确Bridge缓存中记录位置到米的比例；归一化除0.1不能证明这个比例 |
| `source_axes_to_target` | 源记录轴到目标世界轴的正旋转A；不能直接假定单位阵 |
| `source_origin_in_target_m` | 源/目标工作区原点对应b；18D绝对当前位置归一化需要它，不能只给位移旋转 |
| `right_tool_alignment` | 源记录的姿态轴到目标EEF姿态轴的右对应C；需考虑源采集的default rotation约定，左相对输出中抵消的常量仍影响当前姿态输入 |
| 两个`eef_to_point`偏移 | 源/目标所记录EEF到对应物理点G的局部向量，分别按源单位和目标米；零偏移也必须有同一物理点依据 |
| `evidence` | 上述字段的配置/标定出处及适用范围；代码只能验证数值合法，不能证明标定真实 |

另需确定这套源记录约定是否适用于整个Bridge训练子池；若采集配置/TCP/默认姿态约定跨演示不同，不能用一组全局参数冒充全部对齐。这些事实未在现有精确缓存谱系中建立，代码不提供假数据。调用方显式给定的工程约定也不会自动获得正式评分资格。

**完整权重也不在本地。** 已有六文件小结果包不含base与pose-port；实际文件仍是集群`/projects/Zeroshot/runs/bridge-full-capability-234100/final_policy.pt`。新加载器需要该文件，以及现有本地CLIP/DINO资产路径和冻结profile。源报告已在`reports/bridge_full_capability_234100/cluster/report.json`，profile在`deployment/bridge_full_capability_v1/profile_sample.json`。本轮没有SSH、下载或重新训练来补文件。

## 代码入口与检查范围

调用入口：

```python
from libero_pose_policy import load_bridge_pose_policy

# spatial_mapping必须来自已明确的参数，不能直接使用当前null模板。
policy = load_bridge_pose_policy(
    checkpoint_path, local_clip_path, execution_contract,
    spatial_mapping=spatial_mapping,
    source_report_path=source_report_path,
    local_dino_path=local_dino_path,
    frozen_profile_path=frozen_profile_path,
    device="cpu", sampling_seed=1101,
)
# policy可作为原execute_policy_chunk的callback，依然只读当前观测。
```

`tests/test_libero_pose_policy.py`的4项检查通过：非单位缩放+非平凡A/C+非零TCP偏移与独立构造的世界位姿一致；非零port确实进入采样上下文、目标输入使用冻结源特征编码、源采样结果确实反映射到目标；base/port完整恢复和缺失拒绝；缺标定/缺本地完整权重在真实模型加载前拒绝。也检查了旧回调默认上下文未被改变。

使用合成小模型与合成标定，仅调用实际接口代码和冻结18D编码器；没有加载真实DINO或234100权重、没有运行仿真或重复旧性能结论。完整真实加载器的权重恢复/冻结身份路径**尚未实际运行**；真实模型相关6个实现文件与冻结训练参考的文件身份已相符，不等于检查点已加载。

## 现在解除与没有解除的项

- **已解除的软件项：** 18D当前位姿/独立port被遗漏，以及只映射输出而不映射状态输入的接口缺失。
- **仍缺：** 实际空间参数及出处、完整本地权重；因此不能进行真实部署验收。
- **仍未解决：** 234100源旋转能力不足、5Hz专家接触执行失败、四来源共同定义和主矩阵。此接口仅为Bridge诊断格式，不将其自动升级为13模型主入口。

本轮到此停止：不发明标定、不跑旧oracle、不调整频率/增益、不启动训练或LIBERO评分。后续取得上述参数和既有权重后，才可做一次新接口的真实恢复/调用；它也只能证明接口闭合，不能替代任务成功验收。机器检查记录见`reports/libero_pose_port_interface_fix_20261030.json`。
