# 冻结CLIP后的区域读出：唯一集群预检已准备

2026-10-18。只准备预检，尚未提交；没有正式训练入口。主参照207558保持，CLIP视觉/文本全程冻结，扩散保留。

## 固定比较与采用条件

两组均从同一207558权重接续，原1836窗/405演示、相同4096×8抽样表（复用原计划后4096批），新Adam、学习率1e-5、裁剪1。两组都更新相同原Adapter/扩散/状态/历史端口；候选额外学习1026个区域评分参数，控制评分固定零。新评分零初始化时回到原平均。仍为原完整16目标和DDPM100，没有K1请求端口、新增窗口、删首帧、增强或新动作定义。

正式比较计划已写入包内protocol.json，但本包不能执行正式训练。正式预算为每组4096更新，各只取最后权重；两份最终权重冻结后才评分固定已用10开发演示/50窗，三个既定采样种子。原207558的新50窗预测直接复用209481已存结果，经输入一致性检查后作不变参照，不重评分旧1836训练池、旧40开发或13条已完成验收。开发50窗保留全部起点，不作为全新最终测试。

采用条件是：近期第一目标位置演示等权严格优于控制/原207558/不动，旋转均不退步；至少同6/10演示第一目标位置同时胜控制和原模型；完整16路径及终点位置/旋转不劣于控制和原模型，窗口计权近期/全段也不能反向退步。所有条件须通过，不能用训练拟合、热图或一个均值代替。失败则保留207558，结束该读出分支，不加种子/预算/变体；最终25不自动开放。成功也仅为开发候选，不是抓取成功或正式最终确认。

这不是唯一因果根因已确定，而是冻结CLIP要求内、一项具体可回退的结构干预。新增参数和全局梯度裁剪对优化的影响属于该干预；不能声称两组参数量相同或已单独定位视觉信息损失。

## 本地已检查与集群待检查

已完成1024维末端CPU测试及小尺寸32维八层实际Adapter接线：零初始化eval和含dropout训练模式上下文/原参数梯度均差0，新权重梯度非零，旧参数键不改。全部是合成输入，未加载CLIP/207558权重或执行真实尺寸模型。

新包Python语法及实际`--help`导入通过；Bash `-n`通过；源文件身份、全部SHA、原训练NPZ指纹、ZIP逐字节和路径检查通过。Windows沙箱内Git Bash启动受MSYS权限限制，之后仅在允许的沙箱外运行`bash -n`完成检查，没有执行脚本或提交作业。

唯一集群预检会：

1. 校验真实207558权重、原保存训练NPZ及既有1876行冻结token缓存身份；仅重新解析405条已固定训练记录的数值，图片不解码、不重新编码，不读新开发或最终25载荷。
2. 原生模型在旧4窗重放原context/完整8维采样；再接上零初始化读出，用相同context和随机流检查原生完整DDPM采样回退。容差保持context/预测1e-4，不临时放宽。
3. 各做8次更新，首损失一致；候选新增评分有非零梯度并改变，控制保持零；原可训练部分都改变。CLIP和其他冻结参数无梯度且前后指纹一致。
4. 各在第4次保存/重载参数及Adam，下一计算一致；临时文件成功核验后按明确文件名移除。不保存最终新权重。
5. 测显存和暖更新时间，峰值保留≤7.5GiB、预计正式总时长≤7200秒才通过。此预检不评分开发、不给泛化判决，也不自动提交正式作业。

原生完整loss/sample还会调用冻结夹爪层，预检明确将这些层放到正确设备但不解冻、不计夹爪损失，不把其输出当新增学习成果。

## 发布包

- `cluster_packages/bridge_spatial_readout_v1.zip`，837757字节，23文件。
- SHA256：`70dc9f543782a0ab685f59cebaadd6c06374666f38aade089ad8c206ee498b02`。
- 无权重、图片、TFRecord载荷、环境或CLIP缓存；复用集群已有资产，缺失/身份变化即停止，不下载或回退。
- 首个未发布草稿ZIP保留为`bridge_spatial_readout_v1.unpublished-draft.zip`，不要上传该草稿。

## 上传和提交

本地PowerShell：

```powershell
scp "D:/ntu_related/dissertation/Zero_shot/cluster_packages/bridge_spatial_readout_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群Bash：

```bash
set -euo pipefail
cd /projects/Zeroshot
printf '%s  %s\n' '70dc9f543782a0ab685f59cebaadd6c06374666f38aade089ad8c206ee498b02' 'bridge_spatial_readout_v1.zip' | sha256sum -c -
if [[ ! -d bridge_spatial_readout_v1 ]]; then
  /projects/Zeroshot/envs/multitask-preflight-v1/bin/python -m zipfile -e bridge_spatial_readout_v1.zip /projects/Zeroshot
fi
bash -n /projects/Zeroshot/bridge_spatial_readout_v1/preflight.sh
bash -n /projects/Zeroshot/bridge_spatial_readout_v1/submit_preflight_once.sh
bash /projects/Zeroshot/bridge_spatial_readout_v1/submit_preflight_once.sh
SPATIAL_PREFLIGHT=$(cat /projects/Zeroshot/bridge_spatial_readout_v1/submitted_preflight.job)
echo "区域读出预检：$SPATIAL_PREFLIGHT"
squeue --start -j "$SPATIAL_PREFLIGHT"
```

完成后：

```bash
sacct -j "$SPATIAL_PREFLIGHT" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 30 "/projects/Zeroshot/logs/bridge-spatial-preflight-$SPATIAL_PREFLIGHT.out"
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_readout_v1/verify_preflight.py "/projects/Zeroshot/runs/bridge-spatial-preflight-$SPATIAL_PREFLIGHT"
```

提交入口带锁和已提交编号保护。重新连线时可从该编号文件恢复变量；失败后先看错误，不自动重复提交。只有真实预检报告通过，才能继续准备一个正式入口。

证据：[协议](../cluster_packages/bridge_spatial_readout_v1/protocol.json)、[发布身份](../reports/bridge_spatial_preflight_release_20261018.json)、[八层小尺寸接线](../reports/bridge_spatial_interface_cpu_20261018.json)、[1024维末端CPU](BRIDGE_SPATIAL_READOUT_CPU_20261018.md)。
