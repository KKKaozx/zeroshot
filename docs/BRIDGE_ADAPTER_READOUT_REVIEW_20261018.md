# 冻结CLIP之后的Adapter读出：结构审查

后续状态：隔离原型的合成CPU接口验证已完成，见[结果](BRIDGE_SPATIAL_READOUT_CPU_20261018.md)。没有实际207558检查点重放或训练；下文“尚未实现”为审查时状态。

日期2026-10-18。仅阅读固定源码、配置和已有报告；无原始数据/权重读取、模型运行、训练、GPU或新评分。CLIP视觉与文本编码器必须冻结，扩散保留。207558仍为主参照。

## 当前实际路径

以207558协议中的`adapter_pooling=cls_patch_mean`为准，不以普通CLI默认值猜测权重配置：

1. 冻结CLIP输出视觉CLS和patch token、文本token。
2. 八层Adapter中，视觉token为query，语言token为key/value；残差、LayerNorm和FFN均逐视觉token执行。Adapter本身没有新增视觉token互相读取的self-attention。
3. 输出为`LayerNorm(0.5 * (fused_CLS + mean(fused_patches)))`，1024维。patch直接参与当前读出，不能沿用早期CLS-only诊断说它们不影响207558。
4. 207558 runner再加入当前9维位姿的投影与重复当前输入的历史差端口，将相同维数条件送入扩散头。

代码依据：`cluster_packages/bridge_prefix_pair_v1/frozen/adapter.py`、`cluster_packages/bridge_coverage_replication_v1/protocol.json`、`reports/bridge-coverage-rep-fit-207558/runner_snapshot.py`。

## 能推出与不能推出的结论

固定平均给予每个融合后patch相同的显式汇总系数，没有一个单独训练的区域选择读出。新增Adapter层也没有让融合后的局部token相互传递信息。这是结构选择，不是已经证实的代码错误。

不能说空间信息被全部抹掉：CLIP本身加入位置嵌入并执行视觉self-attention，输出patch已包含全局上下文；Adapter逐token变换也可将相关特征编码进通道。对“已经带着位置信息的CLIP输出token”进行排列，并不等于把输入图片的物体搬到另一位置。池化的排列不变性本身不能证明真实图像位置不可区分。

本机所用transformers的`models/clip/modeling_clip.py`中，`CLIPVisionEmbeddings`显式将position embedding加入图像embedding。既有`results/bridge_patch_features_v1/结果说明.md`也保留了CLIP全局上下文的限定。没有重新编码图片或重复旧patch扰动。

## 查重结果

| 已有工作 | 已回答的范围 | 不重复的内容 |
|---|---|---|
| `results/bridge_patch_features_v1` | 早期CLS-only模型的patch扰动、固定网格近邻 | 不重做近邻、热图或置零/交换patch |
| `results/bridge_pooling_readout_v1` | 冻结CLIP+Adapter，回归头的CLS对CLS+patch均值 | 不再提出这两个读出模式的同类比较 |
| 206274固定视觉、206915图像增强 | 改视觉输入变化方式的作用 | 不再重做均值图像或平移增强 |
| Bridge/Panda辅助读出与固定条件头 | 表示允许其他映射利用；部分是已见拟合 | 不重跑ridge、记忆阶梯，不能用Panda训练结果替代Bridge泛化 |
| 208964固定上下文 | 指定时点冻结条件后继续训练头未改善该分支 | 不换冻结时点再追加预算 |

在本次检查的代码、脚本、报告中，未发现207558同数据/同扩散接口下“固定patch平均对可学习区域权重”的正式配对结果。未找到记录不是绝对证明历史上从未尝试；任何训练前仍需以具体产物身份查重。

## 唯一候选草案：可学习的区域加权读出

目标是检验固定等权汇总是否构成瓶颈，而不是证明CLIP坏了或保证修复泛化。保持原八层Adapter、CLS支路、1024维条件、当前状态和扩散网络不变，只替换末端patch求平均这一步。

一种明确且可回到原平均的实现草案：对每个融合后的patch向量`p_i`，计算

`score_i = w · LayerNorm_without_affine(p_i) + u_x*x_i + u_y*y_i`

`a = softmax(score)`，`patch_readout = sum_i a_i*p_i`。

继续输出`原output_norm(0.5*(fused_CLS+patch_readout))`。图像网格坐标固定为[-1,1]，顺序沿用原CLIP patch网格；这是图像索引，不是物体标注或机器人世界坐标。文本仍仅通过原图文融合进入，没有加入新未来信息、任务ID或目标位置。

1024维时新增`w`的1024个参数和两个坐标系数，共1026个参数；新增参数初始化全零，则数学上权重均匀，退化为原读出。实际浮点求和次序可能不同，未执行前不承诺逐元素差0。后续必须核对旧条件和旧扩散重放，而不能通过更换旧权重默认配置静默启用。

这是一个带图像网格偏置的内容加权池化，不是完整视觉self-attention、物体检测器或抓取位置预测器；它不能直接回答“是否缺少patch间交互”。空间偏置也可能记住背景布局、损害泛化，因此不能按训练误差降低判定成功。

## 训练前边界

本轮仅完成结构审查和候选定义，没有实现该模块或写作业入口。若继续，先用隔离CPU接口验证零初始化回退、有限值和梯度流；该验证只证明接线，不作为泛化证据。CLIP所有参数和缓存保持冻结；原权重/固定源码不改。

正式训练提案还必须明确从头或接续、等预算控制、数据与采样计划、开发评价及采用门槛。不能把接续训练与原207558直接比较就声称唯一池化因素，也不能因候选新增参数而称两组参数量完全相同。若采用接续，控制组需相同接续预算；若采用从头，原初始化及随机计划需有可核对依据。

评价仍用固定开发输入和相同权重，完整16目标及近期目标均报告；不删起点、换任务、改动作定义或开放最终25。最终采用依据必须是新演示误差，不能是注意力图“看起来对”、训练毫米级拟合或接口测试通过。当前没有新泛化提升，唯一根因仍未确定。
