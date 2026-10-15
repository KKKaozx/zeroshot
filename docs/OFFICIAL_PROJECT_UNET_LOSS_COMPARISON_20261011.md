# 官方与项目 U-Net/条件/损失对照（2026-10-11）

本次仅源码读取和本地CPU数值对照；没有集群作业、训练、模型或数据下载。新增官方源码三个文件，固定到成功复现使用的提交5ba07ac6661db573af695b419a7947ecb704690f，来源/哈希保存于baselines/diffusion_policy_data_reference/provenance.json。没有修改项目生产核心模块。

## 实际跑了什么

1. 将官方ConditionalUnet1D按项目实际条件维1024、动作维7、时间嵌入128、宽度[256,512,1024]、kernel5、GroupNorm8、cond_predict_scale=True构造于meta设备；映射148个state项，所有参数/缓冲项形状对应，参数总数均为77670023。meta只检验结构，不是实际宽度的完整前向。
2. 官方与项目原类在相同的小型宽度[32,64,128]、相同参数、相同动作/时刻/条件上执行前向和反向。输出、动作梯度、条件梯度及映射后的全部参数梯度最大差值都是0。仅为降低本地RAM用了小型宽度；实际类与操作没有重写。
3. 执行官方真实compute_loss函数体和项目真实policy_loss函数体，用合成全局条件、同一预测与监督目标；分别比较sample(x0)和epsilon。位姿MSE及预测梯度均一致。官方函数所在类未完整实例化，normalizer/观测编码器/scheduler使用合成对象；没有比较整个视觉策略。夹爪BCE不属于官方二维动作损失，未包含于此数值一致性检查。

脚本：code/compare_official_unet_and_loss.py。数值结果及各文件哈希：reports/official_project_unet_loss_comparison_20261011.json。

## 对照结果

|项目|官方成功检查点|当前Bridge多尺度x0|判读|
|---|---|---|---|
|卷积块|Conv1d→GroupNorm→Mish|相同；配置下均8组|已测算子一致|
|FiLM|预测scale/bias，scale*hidden+bias|相同，未额外添加1+scale|已测条件梯度一致|
|各层|每尺度两个残差块、两个中间块、拼接skip|相同|148个state项可映射；均留第一尺度skip不用|
|上下采样|Conv1d(3,2,1)、ConvTranspose1d(4,2,1)|相同|已测对应|
|时间嵌入|128维正弦+MLP|相同|实际偶数维配置一致|
|网络宽度|[512,1024,2048]|[256,512,1024]|实际模型容量不同；不能把同宽度数值对照称为已复现全部官方网络|
|生成监督|epsilon|sample(x0)|两者均由官方源码支持，正确实现不等于相同优化|
|条件内容|两帧图像特征与agent_pos拼接后全局条件|单帧CLIP图像/语言经Adapter生成上下文；末端位姿不进入pose context|内容不同，但FiLM传入方式一致；本次未证明输入信息充分或已被正确利用|
|损失|动作目标MSE|位姿MSE+0.25夹爪BCE（冻结配方）|完全有效的位姿项已测一致；整体任务目标不同|
|mask归约|mask置零后仍除全部元素|除有效pose元素数|存在部分监督时不同；这批固定训练pose全部有效，因此不作为当前差异根因|

官方源：[U-Net](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/conditional_unet1d.py)、[卷积组件](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/conv1d_components.py)、[策略损失](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)。

## x0与epsilon为何不能直接当作相同训练

相同x_t下，如果把预测x0转换为对应epsilon，其平方误差满足L_epsilon = alpha_bar/(1-alpha_bar) * L_x0。沿本项目cosine表，系数t0约1583.1、t24约5.54、t49约0.976、t74约0.169、t99约2.43e-7。直接均匀抽时刻训练x0与epsilon，相当于不同噪声等级下的优化权重不同；不能通过比较两者训练loss大小判断优劣，也不能因官方成功epsilon就断言epsilon必然修复Bridge。

## 已见优化差异（留待独立模块）

实际官方检查点配置：batch64、AdamW lr1e-4、betas[0.95,0.999]、weight_decay1e-6、500步warmup、cosine按训练步计划、EMA。当前配方batch2、AdamW默认betas[0.9,0.999]、weight_decay1e-4、无warmup、cosine每epoch推进。不能同时改这些再归因。配置num_epochs=3050是计划值，不证明下载检查点实际训练了3050轮。旧EMA对照未解决整体生成问题。

## 下一项建议及边界

无需因为“自己写的”就整体替换U-Net：本次对齐权重后已验证其公共路径的前向与梯度一致。下一项较明确的可隔离差异是同一当前多尺度网络上的sample(x0)与epsilon训练目标，固定原尺度数据/条件/优化预算/初始化/随机采样规则，并分别使用对应正确的生成公式。

193195旧epsilon结果使用的是紧凑解码器，而非当前多尺度网络；它说明简单改epsilon不保证改善，不能替代当前网络的目标对照，也不能被忽略。若进行新对照，应标记为“当前多尺度预测目标”实验，只改变该因素，不同时扩大宽度、增加状态或改变优化。小样本条件如需采用，要两组共同使用且预先固定；本轮未准备或启动任何训练包。

没有找到已证实的U-Net基础算子/位姿MSE错误。降低这些因素的嫌疑不等于所有代码正确，更不等于已确定CLIP、Adapter或目标参数化是根因。
