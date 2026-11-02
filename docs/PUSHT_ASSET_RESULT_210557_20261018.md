# Push-T资产盘点210557：结果与下一步

用户已下载完整report，本地于2026-10-18读取。作业CPU-1、COMPLETED/0:0、12秒；报告内检查耗时8.81秒，无GPU/模型前向/训练/安装/下载。不是泛化结果，也未找到Bridge唯一根因。原207558与封存最终25保持。

## 已确认

- 作者源码为固定commit `5ba07ac6661db573af695b419a7947ecb704690f`，跟踪文件无更改；报告的五个关键文件SHA与本地固定源逐个一致。
- 现有官方检查点4,293,295,813字节可读取；保存epoch=500、global_step=84167。优化器各参数计数84168–84183，不将全局计数当所有参数严格相同的更新次数，也不把配置3050轮说成实际完成3050轮。旧SHA记录存在，本次未重新全文件计算。
- 第一FiLM权重为[1024,260]；扣除时间嵌入128，两个观察合计条件132，即每观察图像64+agent_pos 2。项目前端须对接此宽度，不能随意改变后端条件维度来规避匹配。
- CLIP本地配置及约1.71GB safetensors文件存在，配置为L/14、视觉宽1024、文本宽768。只是文件/配置盘点，不是权重完整性或运行兼容验证。
- 已检查的三个根路径及对应zip六个位置均无训练Zarr；不能扩展成整个集群没有。旧安装确实只下载检查点，未建立从零训练数据。

## 盘点对象的遗漏

脚本列了official-pusht-v1与早期bridge-diffusion，但近期项目实际启动器为`/projects/Zeroshot/envs/multitask-preflight-v1/bin/python`。依据：`scripts/prepare_bridge_spatial_preflight_20261018.py`的预检命令。故报告中bridge-diffusion的包null不能用来判断当前项目依赖丢失，不以此重装环境。

官方环境Python3.9/Torch1.12.1，确无transformers元信息；这只说明两套前端的隔离/加载方式还需设计，不能直接推出不可兼容。补实际项目环境只需要一个轻量元信息命令，**不重读4GB检查点、不再交同样的CPU作业**：

```bash
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -I - <<'PY'
import importlib.metadata as m
import json
import sys
result = {'python': sys.version, 'executable': sys.executable, 'packages': {}}
for name in ('torch', 'torchvision', 'numpy', 'transformers', 'huggingface-hub', 'safetensors', 'Pillow', 'zarr', 'numcodecs', 'diffusers', 'dill', 'omegaconf'):
    try:
        result['packages'][name] = m.version(name)
    except m.PackageNotFoundError:
        result['packages'][name] = None
print(json.dumps(result, indent=2))
PY
```

## 数据成本和可行性边界

2026-10-18查询[作者训练数据目录](https://diffusion-policy.cs.columbia.edu/data/training/)，`pusht.zip`显示**29 MB**（目录四舍五入值，不是已核验精确字节数），不同于原4GB权重和2GB pusht_real数据。未下载；解压体积、演示数、chunk完整性尚未核验，不据此承诺下载/训练时间。

下一步先补齐该训练数据并读取实际项目环境元信息，选择不更改成功官方环境的冻结CLIP接口；再明确同训练数据、训练统计、初始化、更新预算和独立开发/最终回合。原生参考必须在匹配配方中建立闭环能力，项目组才能归因。现有0.9309只属旧官方预训练评估，不可拿来与短训练项目组宣称公平比较。

两组正式训练仍**未就绪**：真实batch64显存、暖更新时间和匹配预算未知。数据较小与训练成本较低不是同一件事；不照搬Bridge每4096步时间，不开始新训练、不解冻CLIP。

完整报告：`reports/pusht_module_assets_210557.json`；本地一致性及遗漏记录：`reports/pusht_module_assets_analysis_210557.json`。
