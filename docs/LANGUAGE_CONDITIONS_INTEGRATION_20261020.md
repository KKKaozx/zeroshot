# 四种语言训练条件的主入口接线

2026-10-20。已将可选增强接入`code/train.py`的真实训练循环、tokenizer之前。默认`disabled`；不改变既有Push-T214273提交包、Bridge207558参照或CLIP冻结要求。此次未加载模型、未更新模型参数、未提交集群作业；不是泛化改善结果。

## 四条件的实际处理

| `--language-condition` | 训练文本 | 目标动作 |
| --- | --- | --- |
| `no_augmentation` | 原指令 | 原标签 |
| `paraphrase_only` | 从原指令和五条审阅同义改写中均匀选择 | 原标签 |
| `counterfactual_only` | 按明确声明的比例换成训练集另一目标指令 | 零相对位移、单位相对四元数、保持已知夹爪命令 |
| `combined` | 相同负样本选择；其余样本做同义改写 | 按上述正/负规则 |

两种含反事实的条件必须提供`--counterfactual-fraction`，且严格介于0与1。没有替研究协议擅自确定比例。固定同一seed、epoch、step、batch行及原数据索引时，两组使用相同负样本抽样；正样本改写抽样也配对。增强不消耗Python/NumPy/Torch全局随机流。原图像、实测当前夹爪输入、监督掩码不改动；未知夹爪且已屏蔽监督时继续屏蔽，不能凭空制造监督。

四条件均要求同一份`--split-manifest`和`--language-bank`。库只能覆盖该划分的全部训练索引，不接受留出索引；同时核对数据身份、演示group key和实际解码指令。主四条件入口只允许OXE的TFRecord来源，不能将LIBERO/CLIPort测试演示混入主训练。该限制不代替四来源各自动作/掩码合同的完整验收。

## 增强库格式

下面仅是字段示意，不是已经过审阅的真实训练内容。库的`entries`需包含全部训练索引，反事实donor必须指向同一库中的另一个训练索引。

```json
{
  "format": "reviewed-training-language-bank-v1",
  "scope": "training_only",
  "dataset_identity": "与split manifest一致的数据身份",
  "entries": {
    "12": {
      "group_key": "原演示身份",
      "instruction": "原始指令",
      "suite_id": "明确的训练任务组",
      "goal_id": "目标身份",
      "paraphrases": ["改写1", "改写2", "改写3", "改写4", "改写5"],
      "paraphrase_review": {"status": "verified", "evidence": "语义审阅依据"},
      "hold_gripper_command": {"open_positive": 1, "source_evidence": "当前命令的真实来源"},
      "counterfactuals": [{
        "donor_index": 13,
        "instruction": "donor的原始指令",
        "scene_review": {
          "status": "verified_mismatch",
          "another_valid_goal": false,
          "evidence": "为什么该指令在当前场景不适用"
        }
      }]
    }
  }
}
```

同义改写必须五条、互不相同且不重复原句；反事实要求同一任务组、不同目标，并保留场景不匹配的审阅依据。仅“指令不同”不足以证明机器人应该不动。当前实现检查审阅记录存在，不能自动证明自然语言语义正确。真实库尚未生成；老师要求的本地语言模型生成与审阅仍待实施。

夹爪保持值只接受明确的开正/关负二值命令。不能将传感器开度直接猜成保持命令；没有命令依据而又需要夹爪监督时拒绝该负样本。全零四元数不是合法不动标签，因此相对四元数采用xyzw=[0,0,0,1]。

## 检查及边界

CPU检查命令：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& 'D:/miniconda/envs/zeroshot-gpu/python.exe' scripts/check_language_training_conditions_20261020.py
```

[检查报告](../reports/language_condition_integration_20261020/report.json)确认：实际训练循环在tokenizer前应用干预、默认循环前缀与原路径一致、评价及其他原函数AST未改、正样本目标保持、空动作/夹爪/监督掩码正确、留出成员及不合法审阅被拒绝、语言抽样重放一致且不扰动全局RNG。使用合成fixture，没有验证真实语义库，也没有运行完整模型更新。原主文件保存在同报告目录`train_before.py`，包含修改前已有工作。

启用时保存`language_condition.json`及checkpoint内的库SHA/模式/seed/比例/干预计数；恢复时拒绝变更或关闭条件。这只保证语言抽样合同，不代表现有主训练器的完整优化器/调度器/RNG恢复已达到Push-T专用runner的验收程度。固定正式预算和完整断点恢复仍需落实。

## LIBERO执行仍需明确的部分

现有几何接口只将一个固定观测参考系的相对目标转换成原生OSC命令，不是完整执行调度器。LIBERO原生20Hz，不等于所有OXE来源目标间隔都是0.05秒。不能将16个预测到达位姿直接按20Hz逐行发送，或逐行累计成增量；还需明确训练来源时间含义、部署目标时间、每次重规划执行长度和TCP参考点。

官方数据生成器在`env.step(action)`之后记录观测。若将原生演示用于独立工程对照，行内观测与动作不是天然的“当前观测→下一个动作”配对，需显式核对时间相位。主零样本训练不使用这些LIBERO演示；这一发现不能直接解释Bridge离线泛化误差。

下一项是完成真实训练专用语言库与来源合同，并把当前完整模型接入已可运行的LIBERO工程环境。正式套件、初始状态、训练预算固定后才具备四条件×三seed的可执行实验协议；不能将当前接口检查当作12组正式实验已经开始。
