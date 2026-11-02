# 官方存储桶版本核对结果

2026-10-20。用户确认BC-Z与Language Table均来自 `gs://gresearch/robotics`，并提供作者Language Table仓库及其0.0.1桶目录链接。本次只读取8个小型元数据JSON、官方总表的四来源条目及少量源码；没有下载训练分片、读取新的样本目标、训练或改动环境。

## 已解决：本地版本确实对应桶内发布元数据

下列四个本地文件与对应GCS对象**逐字节完全相同**，不只是字段名称近似：

| 本地来源 | 桶内版本 | 核对文件 |
| --- | --- | --- |
| Language Table real | `language_table/0.1.0` | `dataset_info.json`、`features.json` |
| BC-Z | `bc_z/1.0.0` | `dataset_info.json`、`features.json` |

证据：[读取记录及原始元数据快照](../reports/oxe_bucket_metadata_20261020/report.json)、[逐字节验证](../reports/oxe_bucket_release_evidence_20261020/verification.json)。这证明上述元数据的发布对应关系；没有将其扩大为全部本地数据分片与远端逐字节一致。

## 两组版本差异

| 发布目录 | train演示数 | train分片 | 主要字段差异 |
| --- | ---: | ---: | --- |
| Language Table 0.0.1 | 442,226 | 500 | 有原始`episode_id` |
| Language Table 0.1.0（当前本地） | 442,226 | 1,024 | 省略`episode_id`；其他叶字段定义与0.0.1一致 |
| BC-Z 1.0.0（当前本地） | 39,350 | 512 | 有`episode_id`，图像shape为`[640,512,3]`，保留分区名`eval` |
| BC-Z 0.1.0 | 39,350 | 1,024 | 省略`episode_id`，图像shape为`[171,213,3]`，说明为缩小3倍，保留分区名`val` |

BC-Z版本不只是目录重命名，不能为匹配当前TFDS网页而换成本地未使用的缩小图像版本。也不能因相同演示总数就声称两版本按相同顺序、数值与时刻逐条对应。

当前TFDS通用importer确实省略`episode_id`等顶层身份字段，其处理与上述代理版本特征相符。不过，固定提交中的`rtx.py`已没有Language Table类，不能假称找回了该历史构建的完整路由。**本地0.1.0元数据与桶内0.1.0完全相同；0.0.1→0.1.0的具体历史构建提交仍未恢复。**

源码核对固定到TensorFlow Datasets提交 `7b86414cb6408640c84620ea8c401dbac79138eb`，保存[源码URL与SHA](../reports/oxe_bucket_release_evidence_20261020/source_snapshots.json)。这只是本轮查看的源码提交，不冒充历史导出提交。

## 新发现：Language Table频率资料存在不同口径

[OXE官方仓库](https://github.com/google-deepmind/open_x_embodiment)引用的[公开数据总表](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0)在`Control Frequency`列给出：Fractal 3、Bridge 5、Language Table **10**、BC-Z 10。Language Table同一行标明442,226条演示和推杆末端，不能直接解释成另一份仿真来源。

[Language Table论文附录A](https://arxiv.org/pdf/2210.06407)则写原始状态100Hz、RGB 30Hz、训练整理成近似同步**5Hz**观测—动作对。总表控制频率与论文训练对频率可能描述不同环节；现有资料不足以确认当前缓存行号间隔到底应绑定哪一个，不能擅自选5或10后重采样。

四份桶内`features.json`均没有逐帧时间字段，也没有明确的位置米制说明。作者仓库所查源码中，数据示例指向0.0.1；所查环境代码的频率只明确说明仿真时间推进，未提供实机导出单位/抽帧规范。

## 当前结论与边界

下载桶及本地发布元数据已对应，无须继续要求用户寻找下载脚本。剩余是**准确字段的物理单位与导出时间规则**，不是下载来源不明；元数据相同不能回答元数据从未记录的信息。若没有新增发布说明或维护者答复，不再反复搜索同样目录、也不通过拟合误差猜单位或帧率。

可供维护者确认的最小问题是：BC-Z 1.0.0的`present/xyz`及Language Table 0.1.0的`effector_translation`是否以米计量；两者导出观测行采用什么配对/抽帧规则；Language Table总表10与论文5对应哪个环节。Fractal明确位置单位和实际时间规则同样未完全落实。没有向任何人发送消息。

CLIP冻结、207558、Push-T214273和已有演示归属不变；当前仍没有共同物理时间窗口或新泛化结果，`training_ready=false`。本轮完整证据：[发布核对报告](../reports/oxe_bucket_release_evidence_20261020/report.json)。
