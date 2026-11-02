# 来源依据补充：四元数顺序与采样时间

2026-10-20。接续转换链核对。只查看公开实现与本地元数据；没有重新读取样本目标、拟合、训练、推理或仿真，也没有修改生产标签、归一化、模型和冻结演示归属。

## Fractal四元数：获得同名版本的明确实现支持

[SIMPLER系统辨识数据准备代码](https://github.com/simpler-env/SimplerEnv/blob/main/tools/sysid/prepare_sysid_dataset.py)明确读取 `gs://gresearch/robotics/fractal20220817_data/0.1.0`，与本地数据名称、版本一致。它把原始位姿的最后一个分量移到四元数首位，并注明输出为 `wxyz`。因此其输入解释为：

```text
原始数据：[x,y,z,qx,qy,qz,qw]
SIMPLER内部：[x,y,z,qw,qx,qy,qz]
```

**这支持当前解析器的xyzw顺序，无须为此修改或重训。** 它比直接将四元数作为本体特征传递的下游模型代码更明确，但仍是独立系统辨识实现，不是已恢复原始导出定义；也没有证明远端同名目录与本地文件逐字相同。不按模型误差挑选四元数顺序。

## Bridge单位：补上底层接口说明

[Interbotix接口](https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py)的 `set_ee_pose_components` 明确标注XYZ为米、角度为弧度，并将XYZ直接放入变换矩阵。[Bridge控制器](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py)使用该机器人的运动学描述得到位姿，后续状态与已查转换家族直接复制位置。

这是Bridge采集链使用米/弧度的代码依据，保留历史导出提交未完全绑定的限制。没有改变现有0.10尺度，也没有用“数值看起来合理”替代单位依据。

SIMPLER的[辨识程序](https://github.com/simpler-env/SimplerEnv/blob/main/tools/sysid/sysid.py)也将Fractal XYZ不经缩放地用于仿真位姿与实机轨迹对照。这支持该实现中的SI用法，但尚不作为原始导出单位规范的完整替代。

## 时间：确认不能直接移用仿真频率

[SIMPLER说明的附录](https://github.com/simpler-env/SimplerEnv/blob/main/ADDING_NEW_ENVS_ROBOTS.md)明确区分实机Google Robot非阻塞控制与为简化设置的仿真3Hz。辨识程序中的Google Robot 3Hz、WidowX 5Hz，是其仿真配置，不能当成本地导出逐帧时间戳。因此，没有据此插值、重采样或宣称16个行号已具有统一物理时长。

当前本地OpenX目录的文件名搜索未找到 `obs_dict.pkl`、`out.npy` 或时间戳命名文件。E盘顶层可见Bridge与Language Table的TFDS导出目录；没有声称对E盘做了完整原始内容扫描。

## 剩余事项

Fractal的四元数顺序疑点已获得独立明确支持，保留现有解析方式。下一项仍限定为Language Table真实导出和BC-Z 1.0.0的位置单位、各来源实际观测时间规则，以及Fractal明确的位置单位说明。公开SIMPLER资产不包含BC-Z或Language Table的实机导出实现，不能用它替代这两个来源的依据。

未发现应立即修改生产代码的证据，也未获得新的泛化结果。四来源统一物理时间窗口仍未就绪，`training_ready=false`。

[结构化补充报告](../reports/oxe_pose_source_support_20261020/report.json)绑定本地元数据、既有报告与生产源码SHA。先前的[转换链核对](OXE_CONVERSION_LINEAGE_20261020.md)保留为历史记录。
