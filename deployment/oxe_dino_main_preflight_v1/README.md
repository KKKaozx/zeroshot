# OXE DINO主入口预检（不提交正式训练）

固定145,812个训练窗口与17,949个验证窗口。完整八层Adapter、multiscale后端、batch8，仅2次更新，验证只读一个批次以检查有限loss。不会查看源test、LIBERO评分或自动接续正式训练；验证loss不是泛化结论。约1.58GB工程检查点不能算成正式seed42。

## 传输前

先仅上传并解压本代码/数值标签zip到`/projects/Zeroshot`。利用现有环境检查空间：

```bash
cd /projects/Zeroshot/oxe_dino_main_preflight_v1
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B check_storage_before_upload.py /projects/Zeroshot
```

只有passed=true再上传独立的`oxe_nominal_pool_v1_20261025.tar`（17.07GB，15.90GiB）。空间检查预留tar与解包各一份加8GiB预检余量；失败就先报告输出，不清空旧记录或权重。本代码zip含数值标签/指令，不含原始图像分片或DINO/CLIP资产。

## 原始分片

把tar放在`/projects/Zeroshot`，解包到新的专用data目录：

```bash
mkdir -p /projects/Zeroshot/data
tar -xf /projects/Zeroshot/oxe_nominal_pool_v1_20261025.tar -C /projects/Zeroshot/data
cd /projects/Zeroshot/data/oxe_nominal_pool_v1
sha256sum -c RAW_SHA256SUMS
```

128个原始文件全部OK后，可删除**仅本次上传的tar文件**以归还17GB；不要同时保留tar和解包副本进入长训练。原文件和旧Bridge/Push-T权重都不需要删除。tar只按字节复制容器，没有解释其中test或隔离记录的数值。

## 提交唯一一个短预检

```bash
cd /projects/Zeroshot/oxe_dino_main_preflight_v1
bash submit_preflight_once.sh
```

默认A40、4CPU、40GB内存、最长30分钟；只用已存在的`envs/multitask-preflight-v1`、`baseline_setup/clip-vit-large-patch14-v1`及`dinov2_large_frozen_assets_20261023`，没有安装/下载。提交前检查包内容、原分片存在/大小、冻结资产文件存在与当前保守空间；每次实际取样再核对payload SHA256。代码中的数值单位/TCP未验收标志继续保留，执行约定见execution_candidate.json，不进行物理动作或评分。

输出`runs/oxe-dino-pref-JOBID`；完成后：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -B /projects/Zeroshot/oxe_dino_main_preflight_v1/verify_preflight.py /projects/Zeroshot/runs/oxe-dino-pref-JOBID
```

将JOBID替换为实际编号。此包不暴露正式长训练入口。共同长训练预算、源单位/坐标/TCP假设、正式LIBERO评估安排仍须记录清楚，不能把两次预检当成完成正式训练。
