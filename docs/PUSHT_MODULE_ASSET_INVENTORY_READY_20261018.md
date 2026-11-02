# Push-T模块对照：现有资产盘点包已准备

2026-10-18。承接[前端对照提案](PUSHT_FROZEN_CLIP_MODULE_COMPARISON_PROPOSAL_20261018.md)，当前仅准备CPU只读盘点包，尚未访问集群、提交作业、安装、下载或执行模型。207558、冻结CLIP要求和最终25均保持。

## 本次要回答的问题

1. 集群已工作的作者源码commit及关键文件指纹、两个现有环境的软件版本是什么？版本元信息不能证明运行兼容。
2. 作者检查点保存的epoch/global_step、优化器步计数及FiLM条件权重形状是什么？从实际权重与配置推导每观察特征宽度及RGB宽度，不预设64、不实例化模型。保存计数不是配置计划3050轮全部跑完的证明。
3. 作者配置路径下是否已有训练Zarr？读取数组元信息和小型episode_ends，复算作者原配方的演示/窗口数量；不读图像、动作、状态内容。只在列出的三个根路径及对应zip中查找，找不到不能称整个集群没有；元信息一致也不能证明所有数据块完整。
4. 冻结CLIP资产是否仍在、配置和顶层文件是否齐备？不加载其权重，不改原环境。

原4GB检查点在CPU作业内以map_location=cpu读取；复制旧SHA记录并明确不作新鲜全文件验证。不重跑195527评估，不直接在登录节点加载检查点。没有GPU请求或后续训练自动入口。

## 文件与已完成验证

- `cluster_packages/pusht_module_assets_v1.zip`：7525字节、4文件，SHA256 `c8c29d6feaaf762210c06ef0bf77c9c90a22ab301cdff35a68ad8676f5fcb4d5`。
- 源：`scripts/cluster/pusht_module_assets_v1/`。
- 资源：CPU 4核、16GB、上限15分钟。时限不是承诺运行时间。
- 本地Python3.9语法解析、CLI帮助、18组与固定作者采样函数比较、缺失路径处理、Bash语法及ZIP/SHA检查通过。未做真实集群数据或检查点加载；未验证Zarr依赖及模型兼容。
- 发布记录：`reports/pusht_module_assets_release_20261018.json`。

## 提交

Windows PowerShell上传：

```powershell
scp "D:\ntu_related\dissertation\Zero_shot\cluster_packages\pusht_module_assets_v1.zip" zixiao005@10.97.216.128:/projects/Zeroshot/
```

集群首次解压及CPU提交（已有目录则拒绝覆盖）：

```bash
cd /projects/Zeroshot
if [ ! -e pusht_module_assets_v1 ]; then
  /projects/Zeroshot/envs/official-pusht-v1/bin/python -m zipfile -e pusht_module_assets_v1.zip .
  PUSHT_ASSETS=$(sbatch --parsable /projects/Zeroshot/pusht_module_assets_v1/inventory.sh)
  echo "Push-T资产盘点：$PUSHT_ASSETS"
else
  echo '目录已存在；先核对已有盘点结果，不重复覆盖或提交。'
fi
```

查看：

```bash
sacct -j "$PUSHT_ASSETS" --format=JobID,State,ExitCode,Elapsed,NodeList
tail -n 15 "/projects/Zeroshot/logs/pusht-assets-$PUSHT_ASSETS.out"
```

完成后下载`/projects/Zeroshot/runs/pusht-module-assets-作业号/report.json`。盘点执行成功不代表数据就绪、训练可行或泛化改善。

## 盘点之后的决策边界

先根据实际资产确定是否需要数据、如何隔离兼容接口、两组训练/开发/最终回合与训练统计协议，再决定是否做一次有界的资源预检；实际显存、暖更新时间和合理匹配训练预算仍未知。不能套用Bridge的4096次耗时，也不能把随机Adapter直接接作者预训练后端的失败当成模块失败。数据或预算不适合时停止本提案，不通过反复安装或不断缩短预算维持实验。
