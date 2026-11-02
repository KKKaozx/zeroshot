# 当前进度与模块证据

**2026-11-01：237579原包已独立验收，候选与局部修补训练链正式结束，无新训练/仿真。** ZIP949399B/SHA 1cc0b6be09b0b94a62a67bc1dfbdc61ceb2e1a552bc5276be24f5a9b93c9fe2d，6成员原字节保留，无独立远端ZIP摘要。全部192组×3draw目标/mask/索引匹配冻结源材料，逐窗/均值/配对bootstrap由独立SciPy/NumPy复算，最大指标差4.72e-8；全部8192条history计划/LR/有限标量/损失算术核对，不等于重执行训练或独立看过噪声/全权重。train/dev各六项运动均值均差于零命令、训练均值及已关闭237424，按固定组bootstrap的差值CI下界全>0；开发64/64窗全段位置、63/64窗全段旋转差于零命令。开发全段模型.264809/4.016745度，零命令.176162/2.714177，误差高50.32%/47.99%。这是本联合候选固定协议结果，不证明哪个单因子为根因或零样本不可行。原verification支持集群全policy/Adam读回，本机未取新权重Adam；fresh/无旧策略/encoder0为原回执证据。不再自动准备/提交初始化、尺度、LR等修补训练包或追加诊断/物理回合；下一阶段只基于现有证据整体技术审阅，零样本与老师13模型目标不改。B1源能力/B2物理迁移仍未解除，229116/234100及独立236840监督5/5保留，关闭支线不重开。见reports/bridge_command8_fresh_scaled_237579/independent_array_audit.json；下方待下载/验收为历史，不重做此审计。

**2026-11-01收到237579完成结果：fresh+尺度候选8192步，两角色均未准入，按预注册停止局部修补训练链。** 训练首命令位置/旋转=.215644/3.446082度，开发=.262096/3.545427度，四个均值均差于原零命令/训练均值基线及已关闭237424。新初始化/标准化/LR联合候选未解决B1，不能据此声称某项单因素无效或零样本不可行。可见34条update的计划/LR/有限标量/损失算术及6个生成计数已核对；8192完成、fresh/不加载旧策略/encoder0为集群报告，非独立全history/权重检查。训练44.839分钟/程序46.340分钟/reserved4.533GiB，程序不含CPU准入和最终验证打包，非完整Slurm计时；COMPLETE在全验证/ZIP后打印，仅按代码顺序推断，原JSON/预测/CI/新权重Adam尚未下载。下一项只取原small_results.zip完成验收，随后整体技术审阅；不自动准备/提交新的初始化、尺度、LR等修补训练包或仿真。B1/B2及13模型未解除，旧229116/234100与236840独立监督5/5保留，零样本目标不改；TF/CLIP提示未中断训练，Ctrl+C在完成之后，不重开环境排查。见reports/bridge_command8_fresh_scaled_237579/terminal_audit.json；下方待结果为历史。

**2026-10-31用户报告fresh+尺度源端候选已提交237579。** 仅收到BRIDGE COMMAND8 FRESH SCALED: 237579，尚无实际更新/指标/完成证据，无独立Slurm状态；CPU环境/材料/normalizer/空间准入仅按submit_once先检查再sbatch的顺序推断通过，原输出未收到。计划fresh学习模块/零18D端口，不加载旧策略，源训练运动标准化、外部仍8维；单模型8192×8/LR1e-4，单A40预计50–65分钟/硬90分钟；无LIBERO监督或236840权重。只看/projects/Zeroshot/logs/bridge-command8-fresh-scaled-237579.out，完成BRIDGE_COMMAND8_FRESH_SCALED_COMPLETE后先取原small_results.zip。不重提/续训/自动仿真；若失败停止局部修补训练链，转整体技术审阅，不自动换初始化/尺度/LR或改零样本目标。B1/B2与13模型均未解除，237424保持关闭。回执reports/bridge_command8_fresh_scaled_submission_237579.json；下方未提交为历史。

**2026-10-31用户同意改变推进方式：fresh+运动尺度候选已完整成包，未提交/训练/仿真。** deployment/bridge_command8_fresh_scaled_v1保留原外部8维控制增量/10352训练窗/1151演示/64开发组/训练计划/18D状态和执行规则；冻结DINO/CLIP文本，Adapter/运动/夹爪全部fresh seed42、状态端口零初始化，不加载任何234100/237424学习权重。39572条唯一训练命令拟合7运动分量标准化，std约.10412/.14396/.12646/.01504/.01591/.02584/.001179；DDPM和grip轨迹条件用同一标准化坐标，clip在逆变换的原命令坐标做，最终8维评估口径及旧门槛不变。固定新LR1e-4/空AdamW/8192×8，初始化/尺度/LR联合候选，不声称单一根因已找到。11项CPU替身/合成检查含原100步常数的oracle、loss/Adam/RNG恢复和加载执行接线通过；46个Python AST/依赖导入通过，无CUDA/真实策略构造；远端准入未执行。单A40预计50–65分/硬90分/8GiB，须用户审阅后手动提交。若仍失败则停止这条局部修补训练链，转整体技术审阅，不自动再换初始化/尺度/LR或改零样本目标。ZIP3527512B/68成员/SHA be5fcb2588bf147382d9f4d21046ef6f342601444f8b2d221749d08655870959。只解除新初始化/尺度训练坐标到在线逆变换的实现子项，B1/B2能力和13模型均未解除；237424保持关闭，语言/夹爪/频率/坐标旧实验不重开。见deployment/bridge_command8_fresh_scaled_v1/README.md及configs/bridge_command8_fresh_scaled_candidate_20261031.json。

**2026-10-31控制增量候选237424原包已下载并独立验收，本轮完整收口、未准入、不替换原模型。** 原包949212B/SHA 5a9b6404f2178a73fde168b8c67f793b4a82359736b6c71e8bc6d45b95f5d828，6成员按原字节保留；无独立远端ZIP摘要，不声称远端字节SHA已对过。192组×3draw的标签/mask/索引与冻结材料一致，首/全段/末条命令逐窗指标和bootstrap用独立SciPy/NumPy复算，最大指标差4.72e-8；全部8192条history的计划/LR/有限标量/损失算术核对，非重执行训练或噪声验证。train/dev各6项运动均值均差于零命令和训练均值；开发首命令差值CI跨0，全段与末条运动四项相对两基线CI均为正。开发全段模型位置.222056/旋转3.051717度，零命令.176162/2.714177；位置为归一化源记录单位，不能横比旧位姿指标。原verification支持集群全policy/Adam读回，本机未取新权重Adam；原admission环境/源恢复回执已收到，非本机重执行。新源生成能力B1和物理B2均未解除，不续步/扫变体/自动仿真或放行13模型；旧229116/234100和独立236840监督5/5保留，零样本主线不改。见reports/bridge_command8_source_237424/independent_array_audit.json；下方待下载/验收为历史，不重做此审计。

**2026-10-31控制增量候选237424已完成8192步，两角色源端门槛均false，停止训练候选，不仿真/续训。** 首命令训练位置/旋转=.177339/2.837948度，开发=.235171/3.027097度；按冻结待评标签独立SciPy复算零命令基线分别.156754/2.659707、.224004/2.875341，训练均值基线分别.157670/2.673205、.223699/2.869783。终端模型四个均值均差于两简单基线，不只是CI不显著；新源命令误差不可与旧到达位姿误差横比，位置不是新cm标定。可见34条update的计划/LR/有限标量/总loss算术和6个生成计数已核对；8192总步数/原源保护/encoder0更新为集群报告，未独立查完完整history/实际权重。训练45.072分钟、程序46.648分钟/峰值4.535GiB，程序计时不含CPU准入与最终验证ZIP，非sacct全job时长。COMPLETE在全验证/ZIP后打印，仅按顺序推断完成，原JSON/NPZ/CI/全权重Adam尚未收到；下一项仅取原small_results.zip验收收尾，不新增训练/变体/物理回合或13模型。TF/CLIP提示未中断训练，末尾Ctrl+C在COMPLETE之后，不因此重开环境排查。B1源能力/B2物理迁移仍未解除，零样本主线不改。见reports/bridge_command8_source_237424/terminal_audit.json及docs/BRIDGE_COMMAND8_SOURCE_RESULT_237424_20261031.md；下方待结果是历史。

**2026-10-31用户报告控制增量8维源端候选已提交237424。** 仅收到BRIDGE COMMAND8 SOURCE: 237424，尚无实际更新/指标/完成证据，无独立Slurm状态。submit_once先完整CPU准入再sbatch，因此仅按脚本顺序推断准入通过，原输出未独立收到；不提前宣称原生权重恢复/冻结保护已过。计划单模型8192×8、234100权重warm start/新空AdamW、Bridge源端新8维控制增量，单A40预计50–65分钟/硬90分钟；不用LIBERO监督或236840权重。只看同一日志`/projects/Zeroshot/logs/bridge-command8-source-237424.out`，完成标记BRIDGE_COMMAND8_SOURCE_COMPLETE后先取原small_results.zip验收。不重提/续步/扫变体/自动仿真；B1源能力/B2物理迁移和13模型均未解除。回执reports/bridge_command8_source_submission_237424.json；下方候选未提交为历史。

**2026-10-31控制增量8维候选已完整成包，未提交/真实训练/仿真。** 新deployment/bridge_command8_source_v1已完成原始Bridge控制增量标签、独立当前图文reader、训练损失、完整Adam/计划/RNG保存恢复、新端点加载及持续期望目标执行接线；沿用10352训练窗/1151演示和64开发组，不用LIBERO监督或236840权重。7项CPU替身/合成检查和本地完整依赖导入通过，45个Python文件AST通过、原vendor字节不变；不是源策略/物理成功证据，远端Torch2.5.1环境与材料准入未执行。234100仅warm start，新空AdamW；单模型8192×8/LR1e-5，单A40预计50–65分钟/硬90分钟/8GiB预留。公开采集族裁剪/相位/5Hz/米解释及跨机器人控制点映射仍为明示工程假设，直接增量监督不伪造缺失desired_state。只看最终端点：固定128训练/64开发，第一命令位置旋转均优零命令和训练均值，并且相对零命令两CI上界<0；三draw不是三训练seed。预算结束即止，不续训/扫变体/自动仿真或放行13模型。此轮解除的是新动作合同缺完整可训练/可加载/可执行实现的子阻塞，B1源能力/B2物理迁移未解除；237013等关闭实验和语言工具不重开，旧229116及236840独立监督5/5保留。ZIP 3522753B/66成员，SHA 855636a5002ba3ce3b3048d719e93cd983a85ef17c0044af375ef6ce65cb71f4。优先读deployment/bridge_command8_source_v1/README.md、configs/bridge_command8_source_candidate_20261031.json、reports/bridge_command8_source_preparation_20261031/package_receipt.json。用户手动审阅/提交；下方仅原型/下一步准备为历史。

**2026-10-31新8维控制增量候选已实现，仅本机数值原型，未训练。** 新code/bridge_controller_command_contract.py将原Bridge动作平移3+Euler增量3编码为位置增量3+四元数4+二值夹爪1，独立版本不混用旧未来到达位姿合同；实现“围绕当前EEF支点作用于前一期望目标”的更新及7维原生反馈，目标每条源命令只更新一次。固定一条45行训练记录新增解码state/action、0图像，前44行数值编码含2条中间夹爪值masked，不当Panda速度。5项CPU检查过；44×2种显式历史假设对照公开纯数学函数，误差≤6.67e-16。相同当前状态/action但不同前一期望可产生1cm不同目标，不能用state+action伪造真实期望标签；缓存缺desired_state，公开导出1.0.0与本地0.0.1精确对应/对齐/裁剪仍未证明。直接学习原始增量不必恢复绝对期望轨迹；若按公开采集族工程假设准备源端训练，须明示假设和代码预算，不将追历史commit变成无限前置，也不自动训练。原数据/模型/部署不变，0新模型/训练/仿真，未解决B1/B2物理能力。看docs/BRIDGE_COMMAND8_DEFINITION_20261031.md、configs/bridge_command8_candidate_definition_20261031.json及reports/bridge_command_target_definition_20261031/。本原型结束，不追加参数/频率/坐标扫描或旧实验。零样本目标及老师13模型要求保留。

**2026-10-31用户再次明确核心约束：零样本泛化是项目核心创新，非万不得已不能改路线。** 默认目标继续为OXE训练→LIBERO零样本评估及老师语言增强五条件/13模型；不得因近期失败自行转成LIBERO监督研究。236840监督5/5仅独立已关闭正对照，其权重及LIBERO示教拟合统计不迁入主线，也不以LIBERO示教训练主策略或可学习执行器后仍称严格零样本；部署当前观测与明确的非学习机器人几何/控制器参数仍可用。可在原目标内提出源数据有依据的动作标签、状态/坐标和执行接口修正，默认保留8维外部接口；改变表示须清楚单列，不能悄悄替换或借此改变研究问题。只有用户另行明确决定并与老师对齐才能改研究范围；不能自动判定“已经万不得已”。本消息不授权新训练/仿真，237013等关闭实验和语言工具不重开。规则见configs/zero_shot_core_constraints_20261031.json；最近结果仍是237013两组未准入、236840独立监督5/5，不混写。

**2026-10-31最新收口：237013原ZIP已下载，原损失/Min-SNR两组均未过联合能力，不替换原模型，本轮结束。** 包3982335B/SHA bd7dc60d8323d889794000c5110cddc40eff2eb669c4bbb459e515a02c558dd3，11成员按原字节归档；没有独立远端ZIP摘要，不声称远端字节SHA已对过。原验证文件支持集群全policy/Adam检查，本机未取新权重Adam。192窗目标mask等于冻结源数组，两组全部逐窗/均值/原静止与配对CI用独立SciPy SO3/NumPy复算，最大差2.01e-12；全部16384条保存history的计划/LR/有限标量及8192配对t/noise/RNG摘要核对通过，不等于重执行训练或独立看过实际噪声/权重。开发轨迹位置/旋转：原234100 .943930/15.66306度，uniform .963550/15.34258，Min-SNR .945504/15.91639，不动1.024137/12.06304；位置为归一化源记录单位非cm。控制旋转均值虽略低，但相对原234100四项旋转CI均跨0，位置均变差；候选四项位姿在两角色均值全差于原234100。两组train/dev基本门槛均false，selected=null，不续步/扫权重或仿真。两臂计时之和92.576分钟/每臂reserved4.533GiB，不含CPU准入、构造守卫及最终pair验证ZIP，不是完整job/sacct计时。原229116/234100及236840域内监督5/5分别保留；B1/B2与13模型均未解除。看reports/bridge_timestep_repair_237013/independent_array_audit.json及docs/BRIDGE_TIMESTEP_REPAIR_RESULT_237013_20261031.md最新补充；下方待下载/控制未知均为历史，不重做审计。

**2026-10-31最新：237013两组各8192/总16384更新完成，Min-SNR候选未准入，本候选收口。** 收到BRIDGE_TIMESTEP_REPAIR_COMPLETE，集群报告全8192步配对indices/LR/t/noisy-input/RNG一致。可见34条均为候选组，计划/LR/有限loss梯度/权重元数据/总loss算术本地核对通过；控制逐步日志未收到，不声称独立查完16384步。训练候选减控制的旋转轨迹95%CI[.3241,1.6839]度、终点[.3529,2.3085]度均为正；开发分别[-.4791,1.6603]/[-1.1676,1.9116]跨0，两角色旋转均值全优控制条件均false，基本能力均false；候选两位置均值相对234100退化≤5%报告均true。候选结束、不续训/扫gamma或仿真。原损失控制的绝对指标及自身准入未在终端打印，不能认定两组都失败或丢掉潜在控制改善；下一项只取原small_results.zip审计两组。原ZIP/预测/验证文件及新权重Adam尚未下载，完整验证仅按COMPLETE打印顺序推断；计时/峰值/sacct未知。TF GPU库和CLIP视觉多余键提示未中断实际训练，不因此重开环境排查；末尾Ctrl+C在COMPLETE后。原8维标签不变、无LIBERO新回合；236840监督5/5与229116保留，B1/B2能力及13模型均未解除。回执reports/bridge_timestep_repair_237013/terminal_audit.json，说明docs/BRIDGE_TIMESTEP_REPAIR_RESULT_237013_20261031.md。下方待结果为历史。

**2026-10-31用户报告源端时间步损失配对v1r1已提交237013。** 仅收到编号，尚无实际更新/源恢复守卫/冻结保护/指标或完成证据，无独立Slurm状态。完整CPU依赖/材料准入仅按submit_once先检查再sbatch顺序推断通过，未独立收到输出。计划同234100端点两组各新增8192、共16384，原损失vs固定Min-SNR5，8维未来位姿/原标签/18D状态不变、无LIBERO监督；A40预计95–110分钟/硬2小时。不重提、不续步/扫变体或自动仿真。只看同一日志`/projects/Zeroshot/logs/bridge-timestep-repair-v1r1-237013.out`；完成标记为BRIDGE_TIMESTEP_REPAIR_COMPLETE，之后取原small_results.zip审计。B1源能力/B2物理合同仍未解除，236840监督5/5与229116保留，13模型未放行。回执reports/bridge_timestep_repair_submission_237013.json；当前状态configs/source_timestep_repair_review_20261031.json。下方候选未提交为历史。

**2026-10-31下一候选已准备、未提交：Bridge源端时间步损失配对v1r1。** 恢复此前因夹爪收尾而搁置、从未执行的Min-SNR候选；不重跑已关闭实验。仅用现有Bridge10352窗/1151演示，从234100同端点分原损失/固定Min-SNR5网格均值1两组，各8192×8/新空AdamW/LR1e-5，总16384；输出仍8维未来位姿、标签/18D状态/100步采样不变，不使用LIBERO监督。A40预计95–110分钟/硬2小时/8GiB预留，失败即收口、不自动提交/续步/扫变体或仿真。v1r1只增加提交前完整依赖与绑定数值/原图存在准入、独立名称；原v1/vendor/损失/计划/源数组字节不变，训练臂AST仅输出格式名不同。3项CPU替身检查通过（根/隔离同3项不累计），本地完整依赖导入通过且0CUDA/真实模型；远端Torch2.5.1+cu118/源材料/空间完整准入尚未执行，不能提前写已通过。候选只是可审阅准备，B1性能/B2物理阻塞均未解除；236840监督5/5与229116旧成果保留，语言工具/5Hz插值/相对状态等关闭实验不重开。优先读deployment/bridge_timestep_repair_v1r1/README.md、configs/source_timestep_repair_review_20261031.json及reports/bridge_timestep_review_20261031/package_receipt.json。下方其他下一步为历史。

**2026-10-31接口收口：OXE输出仍8维，LIBERO控制输入仍7维；原几何转换本来已有，本轮不重复实现。** 新增执行状态绑定并接入根code轨迹执行器：策略预测期间及下发命令前若仿真state变化即拒绝，防止已过期反馈被执行；维数不符明确拒绝。4项新CPU合成接口检查、4项原策略回归通过，0训练/真实模型推理/仿真；没有证据认定旧失败由过期状态造成，不声称物理接口能力解除。该检查也不认证调用者传入初始观测的新鲜度或异步线程原子性。冻结历史部署/原229116及236840权重未改，236840五回合成功参照保留。仅软件一致性缺项已处理，源端能力和未来实测位姿→控制期望的接触语义仍待解决，13模型未放行；本轮停止，不追加工程检查或旧5Hz/插值实验。见docs/LIBERO_POSE8_NATIVE7_INTERFACE_20261031.md及reports/libero_pose8_native7_interface_20261031/receipt.json。

**2026-10-31最新：236840原端点/原预测已本地验收，固定五回合关抽屉5/5成功，本轮完整收尾。** policy字节SHA及状态SHA匹配；真实131261063参数CPU严格恢复、train/dev全部保存目标/mask/历史命令/索引与绑定输入一致，三项运动6D MSE各角色独立float64复算一致（正常舍入差）。原admission确认Torch2.5.1+cu118；原verification支持集群完整Adam读回，本机未下载Adam。demo25–29分别84/72/72/72/64个20Hz原生tick成功，共364tick/91策略调用，整个child113.141秒；无专家动作、中途状态修正、额外训练或错误，5份GIF/初末帧/动作trace齐全。首/末帧已查看，轨迹成功标志与全部预算/采样种子及源快照SHA核对通过。此为单任务域内监督学习策略的闭环正对照，target_supervised=true、zero_shot=false、formal_matrix_admitted=false；恒定张开夹爪不算夹爪切换改进。解除“缺少一个实际可完成LIBERO任务的DINO+Adapter+扩散原生命令参照”子项，未解除OXE源生成及共同动作/状态合同，不能归因单一根因或放行13模型。固定预算结束，不追加训练/回合/任务/变体；保留原229116成果和新成功参照，回论文主线。原结果reports/libero_native_supervised_results_20261031/rollout1/report.json；完整审计reports/libero_native_supervised_236840/rollout_closeout.json。下方待下载/待仿真均为历史。

**2026-10-31最新：236840域内原生动作对照已完成4096步，离线运动学习改善，待模型取回后完成固定五回合。** 训练/开发运动6D原生命令MSE=.002821958/.011661449，开发低于训练均值基线.051401504及上一命令保持.037054243，分别低77.313%/68.529%；夹爪常量未计入此MSE，不是成功率或零样本结果。1888冻结编码/encoder0更新/OXE权重未加载为报告；主体665.917秒、reserved5.666GiB，不含CPU准入及最终验证ZIP，无sacct。COMPLETE在全状态/Adam验证与ZIP后打印，当前原ZIP/完整模型尚未下载，新预测未本地独立复算，模型未本地加载/仿真。policy SHA cb79d8333de212aa95810050908dba90e543ccf0f76560e70fc29deb4d824bd8。注意力反向非确定性警告符合warn_only设置，不重训、不宣称GPU逐位可复现，不凭包路径重开环境排查。仅取现有small_results.zip和policy.pt，核对后完成demo25–29各200tick/50调用、总600秒的既定五回合并交付视频；不续训/变体/新任务。看docs/LIBERO_NATIVE_SUPERVISED_RESULT_236840_20261031.md和reports/libero_native_supervised_236840/audit.json；target_supervised=true/zero_shot=false，13模型主线未因此解除。下方236840待结果为历史。

**2026-10-31用户报告域内监督/原生动作对照已提交236840。** 仅收到编号，未收到冻结编码、实际更新、指标或完成证据，无独立Slurm状态；提交前完整CPU准入仅按脚本顺序推断通过，未独立取得输出。单关顶层抽屉，计划4096×16、A40硬1小时，20训练/5开发示教；这是target_supervised=true、zero_shot=false的独立对照，不计13模型。只看同一日志`/projects/Zeroshot/logs/libero-native-supervised-236840.out`，不重提/续训/自动仿真。完成后取回端点及原结果，再按预定5初态/每200tick验收，尚未执行。回执reports/libero_native_supervised_submission_236840.json；下方未提交为历史。

**2026-10-31最新：用户同意的LIBERO域内监督/原生动作独立对照已成包，未提交。** 单独关顶层抽屉任务KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet已核对不属正式40；demo0–19训练、20–24开发、25–29仅固定初态回合。1509train/379dev观测、约39MB材料；原数据写入先step后保存obs，因此标签明确从下一动作i+1开始，H16为20Hz原生命令，不是旧5Hz未来位姿。冻结DINO-L/14和CLIP文本，8Adapter+多尺度x0DDPM及9D关节/两指状态端口全新seed42训练；4096×16/Adam1e-4，A40预计20–40分钟/硬1小时/8GiB输出余量。新job内算冻结特征、RAM保存，不依赖旧token缓存；CPU提交前检查完整依赖、材料/编码器身份与空间。6项CPU检查及一张训练图的真实冻结DINO/CLIP文本/完整原生后端CPU前向通过，真实文本完整恢复兼容4.x/5.x命名空间；本机完整check-only已过，真实更新/仿真/新GPU0，集群环境仍待用户手动入口。训练后固定5初态各200tick、共600秒真实原生命令入口、GIF/动作/成功回执均已准备，当前未执行，不按MSE无限阻挡验收，不自动续训/仿真。该实验更换监督域/原生动作/初始化，明确target_supervised=true、zero_shot=false，不是单变量OXE修复，不计13模型结果；原229116及旧成果保留，236744训练停止、原ZIP仍待回传审计。看deployment/libero_native_supervised_v1/README.md和reports/libero_native_supervised_preparation_20261031/package_receipt.json。

**2026-10-31最新结果：236744已完成，固定相对位姿候选结束，不续训或仿真。** 两组各2048/总4096，原文件未变、Adapter/history/夹爪/编码器0更新为集群回执。开发40窗按演示等权：原229116同口径5.499297cm/12.991050度，absolute5.422218/12.830988，relative7.096348/14.161350；relative比同期控制差30.875%/10.368%，两组总体5%保护均false，工程候选false/正式选择null。控制开发虽略好，但总体失败意味着至少一项训练锚点指标超界；具体训练值尚未打印，不编造。程序901.411秒/1.791GiB，计时不含CPU准入及最终验证打包，未收到sacct。新预测ZIP尚未下载，当前只核对终端差值并用旧保存数组按冻结口径复算原基线；不能称新原预测或Adam已本地独立验收。下一项仅取现有small_results.zip审计，不自动新增GPU/诊断/训练/仿真。原229116成果保留，B1/B2能力均未解除，夹爪支线不重开，13模型未放行。读docs/EPISODE_RELATIVE_PAIR_RESULT_236744_20261031.md及reports/episode_relative_pair_236744/terminal_audit.json。下方236744待结果是历史。

**2026-10-31用户报告运行修复版v2r1已提交236744。** 只收到编号，未收到实际更新/原生恢复守卫/冻结保护/指标或完成证据，无独立Slurm状态。CPU完整导入及文件准入仅按submit_once先检查再sbatch的顺序推断通过，未独立收到输出。沿用v2两组各2048、Adapter/history冻结、decoder/state训练、总4096计划及1小时硬上限；不能把计划标成已训练。看同一日志`/projects/Zeroshot/logs/episode-relative-pair-v2r1-236744.out`，不重提/续步或自动仿真。236711导入失败记录保留。回执`reports/episode_relative_pair_submission_236744.json`；下方v2r1未提交为历史。

**2026-10-31：236711在train_pair.py导入episode_relative_pose_state时缺SciPy退出，尚未创建优化器/执行更新；已收到运行错误，Slurm终态未独立查询。已交付约20KB v2r1运行修复包，未提交：NumPy替代四元数转矩阵的SciPy依赖，原生models仅延迟未使用的CLIP构造导入；native loss/sample AST不变，协议/输入/计划原字节不变。提交前check-only现在导入完整训练模块，不建模型/用CUDA；阻断SciPy和Transformers的独立进程检查通过，6项原CPU检查仍过，矩阵误差<=5.56e-16，1876特征误差<=1.67e-16；0真实更新/采样/仿真。安装器校验旧v2完整SHA和job_id236711，在全新v2r1目录复用原材料，不复制旧job_id/锁，不覆盖236711历史、不安装环境、不自动提交。安装器已测试源/标记保护与拒绝重复。用户手动上传cluster_packages/episode_relative_pose_pair_v2r1_runtime_fix_20261031.zip，运行install.py后在新目录submit_once；仍是v2冻结Adapter/history、只训练decoder/state，各2048、1小时上限，无新条件/预算。看deployment/episode_relative_pose_pair_v2r1/README.md和reports/episode_relative_runtime_fix_236711/。下方236711等待运行状态为历史，不重提未修复的v2。**

**2026-10-31用户报告相对位姿配对v2已提交236711。** 只收到编号，未收到实际更新/原生恢复守卫/冻结保护/指标或完成证据，无独立Slurm状态。计划两组同229116权重各2048，仅更新扩散后端和状态端口；Adapter/history/编码器/夹爪冻结、原文件保护尚属计划，不能提前称已验收。CPU准入仅按提交脚本顺序推断通过，未独立取得输出。看同一日志`/projects/Zeroshot/logs/episode-relative-pair-v2-236711.out`，不重提、不续步或自动仿真。作业回执`reports/episode_relative_pair_submission_236711.json`；下方v2未提交为历史。

**2026-10-31缓存缺失修复v2已交付、未提交：用户v1校验通过，但CPU准入缺runs/bridge-dino-only-cache-v1/full_tokens.pt，在sbatch前停止，本次没有训练编号或更新。旧v1不改不重提。新episode_relative_pose_pair_v2随包带约7.16MB原1876评估上下文，不需要旧token缓存或新编码；两组均冻结Adapter/history，只训练扩散后端+状态端口，c=c_old+(W*f_candidate-W0*f_abs)，控制初始化差分严格0。此为明确的训练范围修订，不能叫v1 Adapter联合微调。仍各2048×8/LR1e-5，总4096，单A40预计15–30分/硬1小时/8GiB输出预留，不自动训练或仿真。6项CPU检查含无旧缓存准入/冻结保护通过，原上下文与基线保存重叠窗逐值相同；无新真实采样/训练/编码/SSH。包内训练、完整状态验证、新格式在线加载和上传提交tail均已相应修订。看deployment/episode_relative_pose_pair_v2/README.md及reports/episode_relative_pair_cache_repair_20261031/package_receipt.json。B1/B2能力仍未解除、夹爪专项不重开。下方v1待提交为历史，当前只审阅v2。**

**相对位姿输入主线候选已交付，未提交（2026-10-31）。** 用户同意后实现统一初态相对9D输入、训练副本及新格式在线加载/reset。两组同229116第8192步权重/空Adam/同批次与RNG，absolute原绝对状态、relative初态局部位置/姿态；仅此输入/因果参考改变，动作标签/原模型/夹爪成果保留。各2048×8、LR1e-5，总4096位姿副本更新；A40预计20–35分钟、硬1小时，输出预留8GiB。原1836train/40dev的数据身份绑定，原168控制状态字节SHA完全一致，0新编码/真实模型前向/训练/仿真；5项CPU检查根及包内均通过（同5项）、真实四模块132309639参数CPU完整恢复，旧加载器3项回归过。23成员约1.94MB包/原生loss与sample复用/完整Adam保存与回读/在线新格式首帧reset/上传提交tail齐全。实际远端cache存在性与A40恢复由固定入口守卫，无自动重建/续步/仿真。源位姿前瞻5%保护只用于审阅工程候选，不称能力已改善或证明唯一根因；B1/B2整体未解除，13模型未放行，夹爪专项仍关闭。见[代码、预算与命令](../deployment/episode_relative_pose_pair_v1/README.md)及`reports/episode_relative_pair_preparation_20261031_package/package_receipt.json`。

**夹爪专项完整收尾并回主线（2026-10-31）：** 236140原ZIP2,676,136B与完整229116源文件已取回；两组train/dev全部原预测独立复算一致，开发控制BA/.openF1/.closeF1=.878543/.052632/.178862，事件=.864363/.082192/.202703。相对旧235928不全面改善，原绝对门槛失败、selected=null，旧头保留。两种真实新头完整CPU恢复、完整真实DINO/9D输入/实测夹爪/工具右相对转换已接LIBERO，显式未准入事件候选一次50调用/200tick/74.921秒，0训练/无错误/无饱和，关抽屉与放碗均false，执行首目标全张开。源位姿及旧头运行后本地SHA匹配。软件加载/执行阻塞子项解除，B1源联合能力及B2跨机器人任务能力仍未解除；不把回合当正式评分。结果保留、接口、有限整链验收结论和复现入口已完成，性能未通过，本轮不继续调夹爪或重跑。恢复源策略能力/共同动作状态合同主线，13模型/大评分未放行，无新训练提交。见[最终收尾](GRIPPER_FUNCTION_CLOSEOUT_20261031.md)及`reports/dino_gripper_function_closeout_20261031/closeout.json`；下方待下载/收尾进行中为历史。

**236140两组已完成、均未准入；按用户要求继续完整功能收尾而非加第三轮训练（2026-10-31）。** 完整终端含GRIPPER_EVENT_COMPLETE：4096新增头更新/原位姿0，源位姿与235928头文件未变；状态保护train/dev两组均true，事件两方向train/dev均false，事件组开发两事件F1优于控制，selected=null。原各组数值未打印，不能编造；原小ZIP待取回，无sacct。主体22.590秒/最大reserved.08984GiB，不含排队等。CPU核对20个可见更新组成/有限性及两组首前向分量相等，未称逐步核验4096。新`preserved_dino_composite.py`完成两格式完整头恢复/原DINO源恢复与接线，3项合成CPU恢复/拒绝/准入隔离通过，旧235928真实头CPU恢复SHA一致，无前向；完整源DINO尚缺、新236140原头尚缺，真实策略加载/仿真未做。新固定整链入口最多200tick/50call/600秒，默认只就绪检查；未准入头必须显式工程模式且仍保留offline=false/formal=false，不改旧gate。没有新训练/源位姿采样/仿真，功能收尾未完成。已给两个scp取回小ZIP和原229116源权重，继续已有产物审计和真实加载/有限接线，不以缺参数停工，不提前弃项。见[收尾状态与命令](GRIPPER_FUNCTION_CLOSEOUT_20261031.md)、`reports/gripper_event_repair_236140/audit.json`及`loader_validation.json`。

**用户最新优先级约束（2026-10-31）：夹爪是必要配套功能，不是论文主要矛盾；当前工作要完整收尾后回主线，不能因提醒而直接停做，也不能继续长期聚焦单项。** 236140仍等待结果，未取消或结束。收尾必须覆盖两组原结果/保护条件核对、旧位姿与头权重保留、候选选择或明确未准入、所选新格式权重加载与执行接线、按结果作有固定预算的整链工程验收判断及可复现入口；不能只留F1结论或未接完的接口。没有结果时不得声称收尾已完成。原准入门槛不追溯改变；若未过，保留未过事实，不能冒充主实验成功，也不把夹爪单项分数设成回主线的无限前置门槛。不自动再开第三轮头/阈值/系数/结构训练。完整收尾后优先恢复有效DINO源策略→LIBERO整体验证和共同训练/执行协议，随后推进老师五条件/13模型；原语言工具、骨干对照及已关闭实验不重开。下一步仍先接收236140同一作业结果并把已授权的本地收尾落实，不直接转走遗留当前功能。

**夹爪事件配对作业236140由用户报告提交（2026-10-31）。** 回执`GRIPPER EVENT REPAIR: 236140`。计划从235928头同权重分出state_control/event_aware，各新增2048×64，原DINO位姿0更新，单A40/10分钟硬上限。当前只收到编号，没有Slurm状态、实际更新/终点指标、旧文件保护结果或选择结论；提交前身份检查仅按冻结脚本流程推断，未独立收到输出。日志`/projects/Zeroshot/logs/gripper-event-repair-236140.out`，输出`/projects/Zeroshot/runs/gripper-event-repair-236140`。等待同一作业，不重提、不续训、不自动仿真或Min-SNR。见`reports/gripper_event_repair_submission_236140.json`。

**2026-10-31下一步收敛为一个夹爪事件损失候选，原DINO/旧头均保留，未提交。** 仅用235928现有logits一次分解：3draw开发24个张开事件中13个所在窗无同方向预测，闭合60个均有同方向预测但11精确/34晚/15早；±1步仅6张开/22闭合，48个真值恒定draw-window中6个误切。容差仅描述，原精确门槛不变，约1.4秒/0模型/采样/训练/仿真，此分析收口不再加诊断。新`gripper_event_objective.py`保留状态BCE、额外监督内部相邻开闭事件，训练频率固定平衡，系数1/阈值0/结构不变；3项独立概率/梯度/饱和/频率CPU检查通过。独立`gripper_event_repair_v1`从同235928头权重出发，空Adam、相同旧计划，控制原BCE/候选BCE+事件各新增2048×64，总4096头更新；不加载更新位姿。保护train/dev BA均≥max(.8,对应235928BA−.02)，两角色两事件精确F1均≥.5；候选须开发两事件F1优于控制，否则控制若过绝对门槛则保留控制。固定单A40预计1–4分钟/10分钟硬上限、480秒进程、每512完整头断点，无自动训练/扫参数/仿真。约16KB小ZIP依赖已上传旧v1，不复制大数据/模型；上传提交tail齐全。仅完成单一修复准备，不称能力已改善。见[改动、保护与预算](GRIPPER_EVENT_REPAIR_20261031.md)、`reports/gripper_timing_235928_20261031/analysis.json`、`reports/gripper_event_repair_preparation_20261031/`。

**235928原小结果已下载并独立CPU复算一致（2026-10-30）。** 5成员ZIP1,338,384字节/SHA9e536e54…f0b2234，原report与终端一致、gripper.pt字节SHA与终端61bfbe19…c9117d一致。实际train/dev全部logits的状态召回/准确率/整窗比例/两方向事件计数F1独立复算一致，准入true/true/false/false保持：开发平均开闭召回87.62%有改善，但事件F1.256/.176未过.50。集群verification.json已收到支持完整头/Adam/2048预算回读；未本地下载latest Adam/完整源位姿模型，保留相应证据边界。保留原229116和头的状态识别改善，本固定候选收口、不准入LIBERO、不追加训练/变体/阈值/Min-SNR；本地0新模型前向/采样/训练/仿真/远端。见`reports/dino_gripper_sidecar_235928/audit.json`及[下载后结果](DINO_GRIPPER_SIDECAR_RESULT_235928_20261030.md)。

**235928独立夹爪2048步完成：原DINO位姿保护由集群SHA回执支持，状态识别改善但切换时机未准入（2026-10-30）。** 用户完整日志含COMPLETE，无报错。train/dev平均开闭召回90.67%/87.62%，均优当前测量保持56.53%/54.13%；开发准确率89.01%、整窗命令全对47.5%不是抓取成功率。张开/闭合精确步事件F1训练.135/.210、开发.256/.176，两个开发.50门槛失败；事件训练内也弱，不能只称泛化失败或锁定原因。开发TP/FP/FN为张开5/10/19、闭合11/54/49；真实24/60是3draw重复原窗口事件。主体13.291秒/最大reserved.08984GiB，不含排队/部分启动与后续打包。原位姿网络未加载/更新，源文件前后SHA一致为集群报告支持；本地尚无完整源文件复核。CPU独立核对日志包清单、召回/F1算术及固定标签事件数、复算3简单基线，准入true/true/false/false。原logits/head/小ZIP待下载，未称候选原预测已独立复算或本机加载新头；无sacct。固定候选收口、不接入LIBERO、不追加训练/阈值/结构/seed或Min-SNR。下一项只取已有small_results并CPU复核。见[结果与下载命令](DINO_GRIPPER_SIDECAR_RESULT_235928_20261030.md)、`reports/dino_gripper_sidecar_235928/audit.json`。

**用户报告独立夹爪作业235928已提交（2026-10-30）。** 回执`DINO GRIPPER SIDECAR: 235928`。计划保留229116 DINO位姿权重，只训练独立夹爪2048步，batch64，单A40/10分钟上限。现仅取得提交编号，未收到Slurm状态、实际GPU准入、任何更新/端点指标或源文件保护结果；提交前CPU身份检查只按冻结脚本流程推断，没有独立输出。日志`/projects/Zeroshot/logs/dino-gripper-sidecar-235928.out`，输出`/projects/Zeroshot/runs/dino-gripper-sidecar-235928`。等待同一作业结果，不重复提交/续训/自动仿真，Min-SNR继续停用。见`reports/dino_gripper_sidecar_submission_235928.json`。

**当前路线已按用户确认固定为“保留229116有效DINO位姿、补独立夹爪”，代码与材料已准备，未提交（2026-10-30）。** 新`dino_gripper_sidecar.py`独立约27万参数，原位姿模型在训练中不加载/不更新，只替换最终第8列。复用原1836train+40dev的上下文/3个生成draw，1876实测连续开度逐条SHA一致，状态抽取约65.6秒/211既有分片、9.54MB材料，0新图像编码/位姿采样/真实训练。原9D端口和工具右相对动作保留，新增执行回调明确转换为目标世界左相对，测量绑定同一观测，不能混用234100的18D/历史命令输入。3项CPU检查在原目录和隔离包内通过：实际原100步采样+头synthetic更新，7D位姿/RNG/冻结权重完全不变；测量/9D/几何/事件与因果绑定通过。保护只限相同输入，不承诺闭环后续观测不变。单A40预计1–4分钟/10分钟硬上限，2048×64/Adam1e-4；开发平均开闭召回≥.8且优最佳基线，开闭内部切换F1均≥.5，终点验收，失败不自动重试/仿真。9.94MB/28成员ZIP逐成员核对，训练/验证/完整模型加载/可选200tick工程入口和上传提交tail齐全；实际完整DINO在线加载/新仿真尚未执行。Min-SNR包停用留档，不作为当前下一步。229116本地完整权重仍缺，集群训练直接核对复用原路径，不再以查参数或重跑旧实验为前置。见[代码、预算与命令](DINO_GRIPPER_SIDECAR_CODE_20261030.md)、`configs/active_capability_extension_20261030.json`、`reports/dino_gripper_sidecar_preparation_20261030/`。仅解除保护旧位姿成果的夹爪实现准备缺项，尚无新增夹爪性能/闭环主结果。

**用户指出长期以“缺参数”停工的问题，已停止参数核查支线并交付单一源端修复训练代码/预算（2026-10-30，尚未提交）。** 新`code/bridge_timestep_repair.py`+独立`deployment/bridge_timestep_repair_v1`实现234100权重起点的两组受控微调：原损失控制vs固定Min-SNR5/固定100步网格均值1的x0时间步权重；同完整Bridge子池/批次/随机流、两组空Adam/常LR1e-5，各8192、总16384更新。原动作/单位/时钟/架构/状态统计/夹爪/采样不变，不重新扫描旋转通道权重或跑旧诊断。预计95–110分钟、一张A40/2小时硬上限，8GiB余量、每2048完整断点，无自动续训或提交。原234100预测复用，终点128train+64dev×3采样，原联合门槛另加同期控制旋转CI和原位置5%保护，失败收口。3项CPU检查覆盖原生loss及旧train守卫/两组各3toy更新/完整Adam与两组权重落盘/配对RNG等，0真实DINO/训练/仿真/远端。2.42MB/55成员ZIP逐成员核对，上传/提交/tail齐全。未知物理参数仍披露，**不作为源端候选准备/启动的新增门槛**。解除可执行候选准备缺项，B1性能未解除，不冒充13模型。见[代码、预算与命令](BRIDGE_TIMESTEP_REPAIR_CODE_20261030.md)和`reports/bridge_timestep_repair_preparation_20261030/package_receipt.json`。

**动作定义事实核查收口，目标EEF观测定义已落实代码/配置，未找到据此修改执行器的错误（2026-10-30）。** 固定一个实际Bridge训练记录metadata和既有排除任务XML，0模型/仿真/训练/远端。新`libero_eef_frame_definition.py`静态读取固定body→site几何，2项CPU检查通过；生成`libero_panda_eef_observation_definition_20261030.json`绑定源码/XML。实际位置为grip site、姿态为hand body；hand到site为[0,0,.097]模型单位与Z−90°，位置已在site不得重复加偏移，固定右旋转在世界左相对旋转抵消，不改现有反馈执行。源记录绑定当前第一Bridge训练episode，只有原numpy路径/episode_id，没有标定字段；旧谱系缺精确导出绑定结论复用，不重做。仍缺源采集EEF/TCP/绝对姿态配置、匹配物理点、明确共同迁移坐标定义；最后一项是设计选择，不能无限等待“通用世界标定”。raw XYZ米/名义5Hz工程假设保留；源左相对旋转监督无需已知固定默认工具旋转，不以此阻碍源能力方案。仅解除本目标配置原点/轴定义子项，不解除B1或完整B2。此轮关闭，下一项单一源端联合能力修复方案/代码/预算，不自动提交。见[事实与明确缺项](BRIDGE_LIBERO_FRAME_FACTS_20261030.md)、`reports/bridge_libero_frame_facts_20261030/facts.json`。

**完整234100新接口真实LIBERO工程回合已完成并收口（2026-10-30）。** 用户手动下载完整权重后，真实恢复base+pose-port、冻结编码器/Tokenizer及训练身份通过；排除LIBERO90任务固定既有初态，50次调用/200tick，100.922秒，CUDA最大reserved2.152GiB。全调用实际应用18D端口和输入/输出映射，初始未知夹爪与采样seed按约定，0异常/原生饱和/训练更新；末态关抽屉和放碗两谓词均false，未完成任务。解除B2“完整位姿策略未接入真实LIBERO”的软件子项；空间参数仍是s1/A=C=I/b与TCP偏置0的显式未标定假设，不解除B1联合能力或B2物理/5Hz接触问题，不算正式评分。预算结束，未加回合/重试/调参/训练。见[结果与固定入口](BRIDGE_POSE_LIBERO_ENGINEERING_20261030.md)、`reports/bridge_pose_libero_engineering_20261030/run1/report.json`与动画；完整权重SHA03682dc…1c554。下方较早缺权重/尚未运行段落为历史状态。

**具体接口已修复：Bridge当前位姿策略的18D输入、独立pose-port与双向空间映射已接入推理，4项针对性CPU检查通过（2026-10-30）。** 新`code/libero_pose_policy.py`恢复base+port并绑定既有结果identity，复用冻结pose_features和源训练统计、77-token固定padding，将port加到图文context后再原生采样；源样本再转换为目标相对轨迹。`code/libero_pose_transfer.py`实现显式单位/轴/原点/右工具旋转和两端TCP偏移，当前观测逆映射与输出正映射同参数，不默认identity。旧回调通过默认hook保留原路径，新格式拒绝base-only加载。4项CPU合成检查覆盖独立世界位姿、非零port进入采样及输出实际转换、两权重组恢复/缺失拒绝、缺事实与缺文件先拒绝；6个真实模型/几何实现与训练参考匹配。**解除B2的软件接口遗漏项；未解除完整B2，也没有改变B1。** 实际缺：源单位比例、源→目标轴/原点、记录姿态工具轴对应、两端TCP物理点与采集适用范围；null参数文件明确保留；234100完整final_policy.pt仍不在本地，不能拿小结果包做真实恢复。真实检查点/DINO加载、物理标定及5Hz接触任务均未验收，四源/13模型不放行。0真实策略/仿真/训练/集群/生成调用；未新增决策文档替代代码。见[代码与明确缺项](LIBERO_POSE_PORT_INTERFACE_FIX_20261030.md)、`configs/libero_bridge_pose_mapping_REQUIRED.json`及`reports/libero_pose_port_interface_fix_20261030.json`。

**用户要求立即收口语言生成并返回论文主线，已执行；本轮主实验技术阻塞解除数为0（2026-10-30，优先于下方历史“下一步”）。** 停止换模型、改提示、追加生成、扩展frame/语法工具；现有最后40条沿用既有39接受/1拒绝、7/8完整审阅直接归档，不重复生成或重验规则。9成员16928B ZIP `archives/language_generation_closed_20261030.zip` SHA bc59043b…247a527b，51条旧候选库存仅作分开标注附录；受控模板≠完整增强库≠机器人实验成果。最小阻塞收敛为B1源端联合能力、B2输入/输出动作与执行合同、B3真实增强数据、B4共同正式协议/规模；DINO选择/恢复/既有oracle等不重验。234100保留位置正证据、旋转失败，235095不是唯一因果定位；5Hz专家/插值失败及20Hz专家成功直接复用。新增静态接口缺项：234100的18D源归一化当前位姿+独立pose-port不在现有LIBERO回调中；不能仅看8D输出或直接喂目标世界位姿。选定下一步为B2共同输入/输出动作合同定版，已完成可审阅决策包和未发老师短稿，本地45分钟/0模型/0仿真/0训练/0远端预算；缺映射事实停在明确缺项，不写假校准、不再做控制/损失扫描或开训练。实际B1/B2未解除，不把决策材料算物理修复，不放行13模型/大评估。见[主线最小清单与停止条件](LIBERO_MAINLINE_RESET_20261030.md)、[动作合同决策包](OXE_LIBERO_ACTION_CONTRACT_DECISION_20261030.md)、`TEACHER_MAINLINE_UPDATE_DRAFT_20261030.md`。后续先报告B1/B2解除情况，不以检查数或模板通过率汇报主实验进展。

**原生GBNF受控生成已完成：8组机械通过、39/40文本接受、7/8完整；历史合格候选去重池51条，8指令各≥5条，未准入训练（2026-10-30）。** 既有7B Q4/b11960/Vulkan1，角色与关系受保护的人工句式约束实际解码，5个位置语言互斥且排除原句；允许弃答，无自由回退/重试/全池。执行前280种有限形式CPU核对，修订了未执行草稿的右下方介词歧义；实际协议v2请求SHA2386cde…，8次completion，21.735秒含核对、4356MiB离散显存最大、服务已停。原始SHAaa043b6c…ea48a5。逐条审阅拒绝`shift apple nearby the can`介词用法，未修句补数；同时承认旧7B一条同类接受错误，追加勘误使旧25接受→24、完整组仍1，原文件/引用SHA保留。规则修正版删除nearby+NP，checker加版本2/源码绑定；实际执行40机械通过与修后39匹配分别记录，执行helper按冻结SHA保存，未重跑。当前8+3项CPU检查通过，未再推理。把既有24+本轮39文本接受候选按原文去重，得51条/8原指令各≥5，披露两机制来源及勘误，未选择最终5条或产生训练库，不能写成本轮8/8语义通过。台账追加8条当前版本，合计24尝试记录/8唯一指令，本批收口拦重复。人工受控形式限制语言多样性，8条开发样例不证明全池泛化；另3678条未覆盖。下一项盘点训练语言结构并扩展源frame，不围绕8条再生成。0训练/集群/新增下载。见[本轮结果与候选池](PROTECTED_LANGUAGE_GENERATION_RESULT_20261030.md)、`reports/protected_generation_20261030/manual_text_review.json`、`prior_review_errata.json`及`reviewed_candidate_pool.json`。

**动作/对象/关系保护检查已实现并可选接入候选导入，5项保护器及3项候选流程CPU检查通过，0新模型调用（2026-10-30）。** 8条手工文本frame绑定当前训练身份，完整句式中保护移动对象/参照/工具角色、push与双参照and、in/into、top和near，不只比词表；实体别名明确声明，保留原top-of歧义。现有40条开发例上25先前文本接受匹配、15拒绝未匹配，但例子参与规则设计，不作独立准确率/泛化证据。另检查8角色交换、19否定/额外动作/关系漂移、8词序/别名组合及身份错误；未识别表达和未建profile指令继续待审阅，匹配也不可训练。`ingest --protected-frames`已用保存的7B原输出实际接线，不重写旧审阅，默认路径不变。当前只覆盖8/3686，3678未覆盖，没有接入原生受控解码或新生成质量结果，已有25文本接受/1完整指令保留。下一项先固定新受控生成机制及有界输入/预算，不自动重跑已关闭生成或训练。见[实现与边界](LANGUAGE_SEMANTIC_PROTECTION_20261030.md)、`reports/language_semantic_protection_20261030/audit.json`及`ingested_candidate_review/`。0新增标签/模型/训练/集群。

**7B量化本地生成验收完成：结构8/8通过，40条中25条文本接受/15条拒绝，1条原指令完整5条；仍不扩展全池或训练（2026-10-30）。** 官方Qwen2.5-7B-Instruct-GGUF Q4_K_M revisionbb5d59e…、llama.cpp b11960 Vulkan公开资产SHA核对，下载4.720GB/含解压占4.825GB低于6GiB；未改Python环境。指定Vulkan1 RTX4060避免核显，使用已保存8条失败训练指令/v3消息验收不同模型+量化+运行时+允许弃答的结构约束，不重启旧1.5B实验、不作容量单因素因果解释。首次模型成功加载但沙箱Win10013回环不可达，0completion输出、就绪超时后关闭；修复访问后同8条仅完成一次生成，127.0.0.1/offline/服务结束关闭。成功含核对25.594秒、最大离散显存4347MiB，结束回到7MiB。唯一完整指令为银锅右下方0025f40…；push被泛化、in丢失、upper/top、滑动/取工具等约束仍导致拒绝，文字接受不等于场景审核或完整训练库准入。原始SHA f5c53b15…c85c9fa，40条审阅与原始响应保留；ledger保留原8条并追加7B8条版本记录。下一项应补动作/对象/关系保护机制，先固定新机制输入与预算，不自动重试本批或新增候选。0机器人训练/集群/全池生成。见[验收结果](LOCAL_7B_LANGUAGE_QUALIFICATION_20261030.md)、`reports/local_7b_qualification_20261030_loopback_repair/manual_text_review.json`及`configs/local_7b_language_qualification_20261030.json`。

**当前训练池首批8条本地候选已生成并逐位置核对：严格合格0/8，40位置全部不可训练，本批已关闭并加入重复拦截（2026-10-30）。** 既有冻结Qwen2.5-1.5B revision989aa798…，本地RTX4060/离线/无远程代码，8次贪心生成，含检查加载32.97秒、reserved3.088GiB，无训练更新。7组无效JSON花括号、1组元素为结构对象；同时出现右下方→炉内/炉上、near→on、RXBAR→苹果等语义错误，不能只靠修JSON解决。逐40位置Codex文本核对：3条文字参考、7照抄、25含义变化/等价不足保守拒绝、5非字符串；3条不作训练批准，完整5条合格指令仍0。真实增强库未就绪，不扩展当前模型/提示到3686条全池；输入与旧预检不同不作提示优劣因果比较。原始SHA d4044fca…7e1c26及运行快照、请求、审阅均保留。8条登记closed_rejected_group；后续prepare排除已尝试ID、runner加载前拒绝重放，新增排除逻辑的2项CPU工作流检查通过，无重复模型调用。下一生成方案需解决忠实性机制或能力，不自动重试/补数/加训练。见[本批结果](LANGUAGE_CANDIDATE_BATCH_RESULT_20261030.md)、`reports/language_candidate_generation_20261030/`、`reports/language_candidate_review_20261030/manual_text_review.json`及`reports/language_candidate_attempts_20261030.json`。0新增集群/训练/机器人模型调用。

**反事实资格回退已在新版库实现，两个主训练入口接入曝光计数，8项CPU检查通过；当前训练池首批8条未生成指令请求已准备（2026-10-30）。** 新`reviewed-training-language-bank-v2`要求显式场景审阅可用/不可用记录；有合法donor且有监督夹爪的保持命令已知才生成不动标签，否则保留正样本和对应改写规则，不删窗、不猜命令。只替换mask有效元素，保留LT缺失维度和尾窗；两反事实条件的请求/资格/donor配对，记录实际负样本及资格/回退原因。v1行为和合同保留，v2不能静默用于v1恢复。合成检查通过配对/掩码/RNG/合同拒绝、原指令控制及两个真实训练干预代码块接线，不是模型训练或真实场景审核。候选准备/导入工具已为当前3686指令四来源各选2条新指令，绑定数据/提示/代表窗；针对既有语义丢失添加保守提示及弃答，未调用模型，不能声称质量改善。失败原文保留、格式通过仍待审阅，不产生训练库。首次系统temp权限失败后改项目内fixture，8项通过。主实验还受真实增强库、执行合同及尺度约束；0生成/更新/远端作业。见[实现与边界](LANGUAGE_ELIGIBILITY_AND_REVIEW_20261030.md)、`configs/augmentation_implementation_status_20261030.json`和`reports/language_candidate_requests_20261030/bundle.json`。

**老师第一至第七封原文自检完成：核心研究问题仍对齐，连续诊断存在挤占主实验的优先级风险，已收口并恢复语言增强准备（2026-10-30）。** 第七封DINO四增强×3seed+CLIP完整增强×1seed、LIBERO成功率仍为关键路径；CLIPort、基线、深度/跨机器人/鲁棒性和预期数据规模未取消、未完成。第一封31明确允许训练Adapter及扩散解码器，不能误判联合训练本身违例，也不能宣称实际只更新小Adapter。当前Bridge单源和低噪声结果仅是诊断，不替代主矩阵或唯一根因。CPU核对当前3686原指令/145812训练窗：旧8条预检仅3条仍匹配，15旧候选中8旧文本接受/7拒绝，无一匹配指令凑齐5条；旧已完整的2条均不在当前池。已生成3686条训练专用审阅任务，绑定当前catalog/来源/代表窗，全部不可训练，不冒充增强库。发现反事实曝光设计的“不合格回退正样本”尚未在当前增强器实现；既有目标报告三维未知夹爪窗15155，LT仅XY另12214，无新场景核验。源到目标执行合同及正式数据规模仍阻塞，不能因本次盘点放行。下一项先处理候选审阅与反事实资格回退，不追加已关闭GPU追踪；任何训练另需具体方案和预算。见[自检全文](PROJECT_GOAL_SELF_AUDIT_20261030.md)、`reports/project_goal_audit_20261030/language_readiness_inventory.json`及`training_language_review_queue.jsonl`。0模型/生成/更新/远端作业/邮件/Git提交，输入SHA不变。

**235095两文件下载、独立CPU复算一致：低噪声局部重建有稳定偏差，开发输入被拉偏（2026-10-30）。** 包82023B/SHA3607b528…fe6bc87匹配，原生t0/目标/mask/初始高斯逐值相等，前向公式及独立SciPy逐窗/均值/CI/MSE一致。旋转带噪示教输入→去噪输出：训练4.576°→4.734°，仅3/8改善，差+.158°CI[-1.640,1.767]；开发4.564°→9.943°，7/8变差，差+5.380°CI[2.484,8.152]。输出都优于原生raw t0的13.911°/14.537°（8/8各组），但专家输入含未来真值，不能当新策略/zero-shot结果。三噪声间角距输入6.267°/6.563°→输出.606°/.539°，有限3抽样原始q分量MSE波动仅.636%/.130%，残差主要为均值偏离；不是总体角误差/梯度分解。位置局部误差也.03981→.06091、.04110→.06650（归一化），不只四元数后处理；原完整采样位置正结果保留。低噪声局部纠偏不稳健，不能只归因生成输入漂移；未知唯一原因，不自动换损失/权重/采样器或训练。报告主体48.78秒/2.291GiB，完整权重未下载、无sacct终态。诊断收口，见[结果](BRIDGE_LOW_NOISE_PAIR_RESULT_235095_20261030.md)、`reports/bridge_low_noise_pair_235095/independent_array_audit_20261030.json`及图。

**235095单t0响应诊断完成，六组高斯匹配/六前向可见，误差原文件待下载（2026-10-30用户日志）。** EXACT6、六条T0_FORWARD及RESULT optimizer_updates:0/direct_denoiser_forwards:6/new_full_sampling_runs:0/source_unchanged:true、COMPLETE均可见；末尾^C在完成后。TF/CLIP提示未中止流程。日志没有具体误差，不能提前判断训练式输入纠偏好坏或唯一根因。小包`/projects/Zeroshot/runs/bridge-low-pair-235095/small_results.zip`，SHA3607b528c45e6af2f68ee9ca14859902ffbd9a1cdf0099434c1c109b3fe6bc87；字节数未知、未本地下载、无sacct最终态。下一步仅下载report.json/paired_inputs_outputs.npz并CPU复算，与既有235002 raw t0比较；不重跑GPU或加训练。见`reports/bridge_low_noise_pair_235095/terminal_audit_20261030.json`。

**单t0低噪声输入响应作业235095由用户报告提交（2026-10-30）。** 用户回执`BRIDGE LOW NOISE PAIR: 235095`；仅收到编号，未观察Slurm状态、原高斯核验、六次前向或完成结果。计划复用234100权重和235002原生raw t0输出，固定16窗、6次训练式t0去噪前向、0完整重采样/0更新、5分钟作业上限。CPU准入仅按固定提交流程推断，未独立收到完整输出。日志`/projects/Zeroshot/logs/bridge-low-pair-235095.out`，输出`/projects/Zeroshot/runs/bridge-low-pair-235095`。等待同一作业，不重复提交、新训练或自动变体，原模型/gate不改。见`reports/bridge_low_noise_pair_submission_235095.json`。

**235002训练式噪声兼容性CPU核对完成；单t0六前向补证包就绪，未提交（2026-10-30）。** 实际99步反向随机量还原汇总std1.00078/1.00185，末步后验公式与pre-unit记录差≤2.98e-8，未见明显噪声尺度异常。窗口等权的对应示教标准化残差能量在t0为15.418/9.613（训练式前向噪声期望1），但范数及全训练单四元数最近覆盖未见整体脱离；不能把与一个专家后续的偏差直接称全部训练支持外或唯一根因。见[输入差异分析](BRIDGE_ROTATION_INPUT_GAP_235002_20261030.md)、`reports/bridge_rotation_input_gap_235002_20261030/analysis.json`。尚缺当前端点在训练式输入上的网络响应，准备独立小包复用235002已有raw t0，只增加16固定窗×3原始高斯、共6次t0直接去噪前向；无t扫描/条件置换/完整重采样/更新。3项CPU原生公式等价、原高斯随机流/错误拒绝、保存复核及篡改拒绝通过，非真实DINO/CUDA。8成员10387B ZIP，SHA c594db0725647077ca310d46df012b9a78872b05de8ba69e63a326e0c4910a86。预计含加载1–2分钟/5分钟A40硬上限/256MiB预留，无自动提交/训练/后续任务。见[代码和上传提交tail](BRIDGE_LOW_NOISE_PAIR_CODE_20261030.md)、`reports/bridge_low_noise_pair_validation_20261030.json`。

**235002三文件已下载独立复算：原始x0方向在反向去噪后段变差，范数/末端单位化未见对应异常（2026-10-30）。** 包4614522B/SHA485315e8…b5e88aa匹配，16窗×3seed的原目标/mask/最终预测与234100逐值相同，100步摘要及独立SciPy几何/MSE分解一致。固定训练8窗raw t99→t0误差8.615°→13.911°（8/8变差），开发8窗11.524°→14.537°（6/8变差）；不动11.093°/12.447°。全100步raw范数训练.950–1.031、开发.942–1.034，末端identityfallback0，单位化平均角改变量约2e-7°；末端分量MSE方向项99.896%/99.703%。最大单步平均重建角变化.0084°、裁剪.0040°，clip只在w；仅即时作用，未证明累计因果效应为零。旋转幅度随生成增大而轴向余弦下降，定位到生成后段方向质量，尚不能锁定loss/训练生成分布/随机过程唯一原因。早期raw估计不是新策略，不凭16窗按真值选早停或改原gate。报告71.02秒/2.291GiB，完整权重未下载、无sacct终态。诊断收口，不自动GPU/训练/变体/LIBERO。见[结果](BRIDGE_ROTATION_TRACE_RESULT_235002_20261030.md)、`reports/bridge_rotation_trace_235002/independent_array_audit_20261030.json`及曲线。

**旋转只读追踪235002完成，16窗×3seed全部精确重放，0更新，trace小包待下载（2026-10-30用户日志）。** 六条REPLAY_EXACT、RESULT all16windows3seeds_exact:true/optimizer_updates:0/source_unchanged:true及COMPLETE均可见。TF库/CLIP多余视觉键提示没有中止流程，^C在完成后。源234100及其gate不改；尚未取得原report/trace，不能宣称范数/方向/裁剪中间量已独立验收或已找到根因。小包路径`/projects/Zeroshot/runs/bridge-rotation-trace-235002/small_results.zip`，打印SHA485315e848dbebf2b6bffd26615ec5a078d9f88e8576f3ace359c3d68b5e88aa；字节数未知，无sacct最终态。下一步仅下载已有三文件包CPU复算，不重跑GPU/新训练/变体。见`reports/bridge_rotation_trace_235002/terminal_audit_20261030.json`。

**原生旋转只读追踪235002由用户报告提交（2026-10-30）。** 用户回执`BRIDGE ROTATION TRACE: 235002`；仅收到作业编号，未观察Slurm状态、真实GPU精确重放或中间量追踪。计划读取234100final_policy，固定16窗×原3采样seed，0优化更新、10分钟作业上限。CPU准入仅按不可变提交脚本推断，未独立收到完整输出。日志`/projects/Zeroshot/logs/bridge-rotation-trace-235002.out`，输出`/projects/Zeroshot/runs/bridge-rotation-trace-235002`。等待本作业，不重复提交或追加训练，原234100模型与gate保持不变。见`reports/bridge_rotation_trace_submission_235002.json`。

**234100原生采样旋转只读追踪代码准备完成，3项CPU测试通过，未提交（2026-10-30）。** 现有最终轨迹缺少raw x0/范数/裁剪中间量，独立小overlay仅加载234100final_policy，固定原训练前8+开发前8窗、原3seed，原生100步中只加入CPU张量拷贝。开发批跳过15批随机调用并以实际首批RNG校验，16窗×3seed原输出必须逐值重放；不构造专家带噪输入、不重做旧epsilon/条件置换五t试验。记录raw/latent/reconstructed/clipped及末端单位化前轨迹，分解四元数径向/方向分量MSE；单位化不改变非零q代表的旋转，不能预设它是根因。CPU实际原生采样替身/真实标签、精确重放/随机推进、错误拒绝/零范数、参数/梯度/空Adam不变、保存数组重算及篡改拒绝通过，非真实DINO/CUDA。8成员14481B ZIP，SHA06a5e80a94b60f1e7252f989fb6e7c755a3581d6816d328c0d02ab2293085f84。预计2–4分钟A40、硬上限10分钟/记录器300秒、预留256MiB；0更新，不改主模型/原gate，不自动任务。见[ZIP与上传/提交/tail](BRIDGE_ROTATION_TRACE_CODE_20261030.md)、`reports/bridge_rotation_trace_validation_20261030.json`。目前0真实GPU/远端调用，无作业编号或追踪结果。

**234100旋转CPU定位完成：小角训练窗受损、采样与偏差并存，未发现标签符号冲突（2026-10-30）。** 全Bridge10352窗/130668有效四元数单位范数偏差≤3.92e-8，负w及120316相邻对负点积均0，无近180°目标；192窗评分与独立SciPy SO(3)最大差2.44e-11°。训练专家<10°的71窗贡献总体超不动误差+2.273°（全体+2.247°），开发多个角度组均弱。既有样本两两角距约10.66°/10.81°，三采样各自均差于不动；纯描述性三样本Markley共识仍11.49°/14.06°，不动10.80°/12.06°，未替换原评分/gate。首0.2秒以及固定完整16步子集同样弱，不仅远期或尾窗；全局旋转向量均值平方约占均方1.13%/1.77%，非简单统一偏移已足够解释。冻结源码确认x0为7维掩码MSE，regression_rotation_weight不作用于当前分支；数值通道尺度统计不是梯度因果证据，不重开旧旋转权重扫描。见[旋转分析](BRIDGE_ROTATION_ANALYSIS_234100_20261030.md)、`reports/bridge_rotation_analysis_234100_20261030/analysis.json`及图/逐窗CSV。原数组/模型/gate不变，0新模型/采样/更新/GPU，无新增训练或LIBERO。

**234100六文件已下载、独立CPU复算一致：位置有明确学习，旋转及开发稳定性阻塞联合门槛（2026-10-30）。** 小包938429B/SHA31aa1cfa…24839一致，192目标/mask与本地冻结仓逐值相等，全部逐窗/均值/组bootstrap及门槛复算1e-10内一致。训练轨迹位置1.13334→0.57841，不动0.77420，优25.29%、CI[-.26464,-.12757]；终点优36.52%。开发轨迹位置1.25408→.94393，不动1.02414，均值优7.83%但CI[-.19414,.03192]跨0，终点优13.23%且CI低于0。训练轨迹/终点旋转13.051°/18.817°仍差于不动10.804°/16.714°；开发15.663°/20.701°差于12.063°/16.297°，四旋转CI均正。门槛训练因两旋转失败，开发另因轨迹位置CI失败，不能写成四指标全差或完全没学。位置幅度/专家96.6%/88.9%、方向余弦.471/.334；旋转幅度9.34°/10.29°小于专家10.80°/12.06°但旋转向量方向余弦仅.182/.093，仅描述不锁定根因。报告训练5.985h、进程6.036h、reserved4.535GiB；完整模型/Adam/全轨迹未下载、无sacct最终态。预算结束，不自动续训/变体/seed/LIBERO；单源与历史混训配方/曝光不同，不证明混训唯一原因。见[更新结果](BRIDGE_FULL_CAPABILITY_RESULT_234100_20261030.md)、`reports/bridge_full_capability_234100/independent_array_audit_20261030.json`。

**234100完整65536步与端点采样完成，训练/开发联合能力门槛均false，原小结果待下载（2026-10-30用户日志）。** 可见最终更新、192窗×采样seed1101/1102/1103、RESULT verified/arrays_metrics_recomputed/full_policy_adam_plan_checked均true及COMPLETE。54条可见更新的indices与冻结计划全部匹配，loss/梯度有限，pose+0.25gripper组成正确；本机同LR公式最大差3.3881e-20，仅数值核对，不称跨平台逐位一致或复核全部65536条。gate false不能推出四指标全差或完全没学到，具体均值/区间、幅度/方向和资源尚待六文件小包。包SHA31aa1cfa990b5f0062778b803ca503ac446f036002b1f47834d6ccfb20124839已打印，尚未本地验证；完整模型/Adam未下载，无sacct最终态。Invalid job id不独自证明失败，^C在COMPLETE之后。此轮不自动加预算/seed/变体或LIBERO。见[结果与下载命令](BRIDGE_FULL_CAPABILITY_RESULT_234100_20261030.md)、`reports/bridge_full_capability_234100/terminal_audit_20261030.json`。

**完整Bridge单源能力作业234100由用户报告提交（2026-10-29）。** 用户回执`BRIDGE FULL CAPABILITY: 234100`。计划seed42从头训练当前10352窗Bridge子池，x0/当前位姿、65536步，8小时作业上限。现仅收到编号，未观察Slurm状态、真实GPU准入、零更新守卫或任何实际更新；固定提交流程中的CPU准入仅按脚本推断，未独立收到完整输出。日志`/projects/Zeroshot/logs/bridge-full-capability-234100.out`，输出`/projects/Zeroshot/runs/bridge-full-capability-234100`。等待同一作业，不重复提交、追加预算或LIBERO。见`reports/bridge_full_capability_submission_234100.json`。

**用户选择完整Bridge单源能力实验，独立代码包与5项CPU验证完成，尚未集群提交（2026-10-29）。** seed42新初始化全部可训练参数/零位姿端口/空三组AdamW，只使用当前有限池全部10352Bridge窗（1151演示）；x0/源坐标5Hz H16/原联合loss与mask，B8×65536，warmup1024与完整cosine、base2e-4。固定计划524288呈现，每窗26–78次，无未见窗；Bridge训练统计单独拟合。初始/最终仅评价既定128训练组+64开发组，各3采样seed，非3独立训练seed；不是混训因果对照或LIBERO放行。CPU真实标签/身份/来源隔离、原生x0损失及100步采样替身、首梯度/RNG复现、三Adam/计划/调度/RNG恢复、预算停止及小结果篡改拒绝5项通过，vendored train/models实际导入通过；未跑真实DINO/CUDA、SSH或远端作业。43成员1613715字节小ZIP，SHA d7382354af27203be40659b2467eca98b4b2f8815233643a35309655dd66fa38。预计6.5–7.5小时占卡/8小时Slurm硬上限，训练累计7小时cap，每4096完整原子断点，至少8GiB空间预留，防重复提交，无自动续作业/清理。见[代码与上传/提交/tail](BRIDGE_FULL_CAPABILITY_CODE_20261029.md)、`reports/bridge_full_capability_validation_20261029.json`。用户手动执行，未取得作业编号或性能结果；历史分支不重开，主模型不替换。

**接手后的能力候选与老师沟通稿已整理，仅供审阅，未新增训练（2026-10-29）。** 见[证据收敛](EVIDENCE_CONVERGENCE_20261029.md)、[完整当前Bridge子池能力候选](BRIDGE_CAPABILITY_DECISION_DRAFT_20261029.md)及[老师进度/范围确认稿，未发送](TEACHER_PROGRESS_SCOPE_DRAFT_20261029.md)。新候选区分230933监督屏蔽、已混训端点局部续训和从头单源能力；拟用10352窗Bridge子池、x0/当前位姿从第1步、65536更新，约6.05小时纯训练外推，预计6.5–7.5小时占卡/建议8小时硬上限。它不是来源干扰因果对照，未获新增GPU预算，未生成/验证训练包；执行合同仍独立未通过。旧分支保持关闭，本轮0模型/优化/仿真/远端操作，未发邮件、不自动提交或LIBERO。

**233818四个原文件下载齐全，全部逐窗指标/区间/幅度方向独立CPU复算一致（2026-10-29）。** 压缩包538553字节；120目标/掩码与本地冻结仓逐值匹配，source/control/static与233531原预测逐值相同，新报告完整指标树1e-10内复算通过，合同/起点参数AdamRNG身份和首控制探针loss/noise/RNG元数据一致。报告约24.99分钟、reserved4.57031GiB，非排队总耗时。训练32锚点与开发88窗各14来源汇总指标全部差于控制与不动；Bridge64开发位置55窗/旋转48窗比旧控制差，四差值中位数均正，不是只由个别坏例子主导。固定计划中32锚点各新增46–76次，仍未建立准确生成，不能继续单归因曝光少，也不证明混训唯一原因。完整权重/Adam和全部4096内部轨迹未下载，不冒称独立字节验收；尚无Slurm最终态。0新增模型/优化/GPU，有限变体结束，不自动加预算或LIBERO。见[补充结果](OXE_COMPETENCE_POOL_RESULT_233818_20261029.md)、`reports/oxe_competence_pool_233818/independent_array_audit.json`。

**233818单组512窗4096步与端点评价完成，训练锚点14项均值全差于控制和不动（2026-10-29，用户终端）。** output_verified/COMPLETE可见，无异常；源120窗重放与全实际noise/t/RNG匹配由集群验证器报告通过，对照不重训。18条更新索引/实际LinuxLR/掩码/P77本地逐值核对，32训练锚点来源均值由逐窗复算；旧source/control/static逐窗值及幅度方向均与233531已下载原预测独立复算一致。Bridge轨迹位置1.5830/旋转17.3757度，对照1.1486/10.1744，不动0.8854/9.9153；四源全部14可用轨迹/终点均值比二者差，训练能力门槛false。新幅度/目标123%、177%、181%、185%，平均方向余弦-.123/-.097/.137/.389，不能用动作偏小概括新组。缩池重复在本起点/预算/高LR配方下未解决锚点训练能力，不证明唯一混训原因、DINO固有失败或完整x0时域失败；只测32非全512。新NPZ/完整report/权重未下载，未独立重算新原预测/区间或看到88开发指标及Slurm最终态。先取4小文件做CPU复核，关闭自动延长/变体/seed/LIBERO。见[结果](OXE_COMPETENCE_POOL_RESULT_233818_20261029.md)、`reports/oxe_competence_pool_233818/analysis.json`。

**512-window moderate-pool job233818 submitted by user (2026-10-29).** User receipt: OXE MODERATE POOL COMPETENCE:233818. Only submission identity received; no Slurm state, actual source/control replay, zero-update probe, CUDA updates or endpoint verdict observed. Same232587source10240, one4096update arm, reuse233531replacement without retraining. Log `/projects/Zeroshot/logs/oxe-competence-233818.out`; output `/projects/Zeroshot/runs/oxe-competence-233818`. Await this job; no duplicate submission, extra budget or automatic LIBERO. See `reports/oxe_competence_pool_submission_233818.json`.

**用户选定512窗单组诊断，代码及CPU检查完成，未提交（2026-10-29）。** 从232587real_pose10240原core/port/三Adam/RNG恢复，只新增4096至14336；四源各128不同训练组，有放回，复用233531replacement完整端点/轨迹，不重训对照。原来源槽位、实际LinuxLR、77token及网络监督保持；原120窗×3seed精确重放和旧控制首批次loss/梯度/噪声0优化探针必须通过，再逐步核对实际原始noise/t/RNG。每2048完整独立断点，约26分钟估计/35分钟作业上限；4GiB预算余量检查。5项新增CPU主入口/元数据/真实标签与原生采样/中断恢复/错误拒绝检查及8项旧引擎与实际加载回归通过；新集成为4步toy，4096行控制元数据夹具不冒称真实训练。13成员286502字节ZIP逐成员核验，无权重图片；0真实DINO/CUDA/远端调用。只评价固定32训练锚点和88开发窗，报告幅度方向，不称测完512或证明曝光唯一根因，不改主模型/放行LIBERO。见[代码与上传提交](OXE_COMPETENCE_POOL_CODE_20261029.md)、`reports/oxe_competence_pool_validation_20261029.json`。

**中等规模四源训练能力候选完成CPU设计，需另选预算，未准备/提交训练包（2026-10-29）。** 不重复骨干/x0/状态/8窗正对照，不重开轮转。候选四源各128不同训练组、每组1窗，共512：保留每源原8训练评价锚点，再固定seed抽120组，不看目标/误差，验证组排除。仅一组从232587real_pose10240继续4096至14336，复用233531有放回等预算控制；来源槽位/原LinuxLR/P77/网络监督优化设置不变，同为有放回。每窗平均63–65次（最少42—47、最多83—95），没有替换/删除原完整池；新候选缩小数据变化且提高重复，不是曝光唯一因果控制。按实测约26分钟候选含评价估计，上一轮预算结束，尚无新预算/包/任务；实际RNG等价须之后以保存控制轨迹守卫，不以CPU计划冒称。正结果仅支持有限多源训练能力，负结果不自动变池/来源/模块/预算，均不放行LIBERO。见[候选审阅](OXE_COMPETENCE_POOL_DESIGN_20261029.md)、`reports/oxe_competence_pool_design_20261029/design.json`。

**233531原文件下载完整，指标/区间全部独立复算通过，确认误差下降伴随动作幅度收缩（2026-10-29）。** 546323B/5文件，SHA0b08bfaa…60566b；120目标/掩码与本地不可变仓逐值相同，全部四源逐窗指标、Bridge64bootstrap和门槛复算一致。报告计时47.449分钟，两组reserved4.570/4.574GiB，未含排队/部分启动核对。Bridge开发预测/专家位移幅度从129.3%变39.9%/44.9%，方向余弦仅0.063/0.082；平方位置误差下降按恒等式93.5%/87.2%对应预测能量降低项（描述性非因果）。各11/64轨迹位姿双优不动；抽查8Bridge训练窗位置仍1.1486/1.1513>不动0.8854，方向余弦负，问题也存在于这些训练窗，不只留出误差。所有来源开发位置均值仍高于不动，但有放回Fractal训练位姿均值略优，不泛称全数据不能学。完整模型/4096内部轨迹未下载，不冒称独立权重/RNG全轨迹核验；不作DINO固有失败或完成65536后的结论。零新训练/推理/GPU，轮转分支关闭，不追加预算；后续先建立当前大池训练端正确运动能力。见[更新结果](OXE_SAMPLING_PAIR_RESULT_233531_20261029.md)、`reports/oxe_sampling_pair_233531/independent_array_audit.json`。

**抽样对照233531两组各4096步及端点评价完成，轮转未建立优势，结束此有界变体（2026-10-29用户日志）。** 集群最终verified:true，实际噪声/t/RNG配对检查通过；10条有放回/18条轮转可见日志的索引、原LinuxLR、掩码、token77本地核对，10共同步来源槽位及前后RNG相同。Bridge64轨迹位置源1.702680→有放回1.096477/轮转1.079425，旋转15.178584→12.522370/12.675547；不动1.024137/12.063044，两组四均值仍全高于不动。轮转相对有放回位置低1.56%/终点低3.59%，旋转高1.22%/终点高1.65%，四组配对区间全含0，抽样及基本能力门槛均false。共同改善不能单独归因预算，源参照更新数不同且两新组共同77token补齐。位置仍归一化记录单位，单训练seed/重复开发组限制保留；原数组/全部内部轨迹尚未下载，无独立bootstrap复算或Slurm最终状态。保留输出，不追加轮转预算/种子/LIBERO；下一步只取小型报告与预测核对四源及动作幅度。见[结果](OXE_SAMPLING_PAIR_RESULT_233531_20261029.md)、`reports/oxe_sampling_pair_233531/analysis.json`。

**集群约179GiB，用户授权尽量清理；约50GiB白名单脚本就绪，尚未远端执行（2026-10-29）。** 盘点233531为PENDING；计划逐文件清pip/Conda下载与索引、两份Push-Tvisual共22.266GiB、关闭Bridge特征缓存、失败/被完整后继替代的半程及一次性预检状态、既有Bridge Adam白名单和所有完整解压且逐字节一致的根ZIP。当前OXE两源/233531、seed42/43、有效Bridge、Push-T最终模型和全部数据/环境/源码/结果保留；226133首次绑定仍需读取其合同，特意保留主预检原检查点。Conda解压包及HF/TFHub资产不盲删。10项微型文件路径/摘要/作业/精确删除/保留/重复运行/归档核对回归通过，inode别名使用stat夹具；0真实集群删除/GPU/取消作业。实际释放量待CLEANUP_COMPLETE回执；删除后的旧特征缓存须重建才能重跑旧缓存训练。见[范围与执行](CLUSTER_STORAGE_RECLAIM_20261029.md)、`reports/cluster_storage_review_20261029/plan_and_validation.json`。

**来源内抽样对照233531由用户报告提交（2026-10-29）。** 原文`OXE SAMPLING SHORT PAIR: 233531`；计划从232587real_pose10240恢复，两组各4096步至14336，约45分钟估计/60分钟作业上限。只有提交标记，尚无Slurm状态、真实GPU源重放、实际新增更新或端点指标证据；提交前CPU检查仅按固定脚本流程推断，未独立收到原检查输出。日志`/projects/Zeroshot/logs/oxe-sampling-pair-233531.out`，输出`/projects/Zeroshot/runs/oxe-sampling-pair-233531/`。等待同一作业，不重复提交或自动追加预算/LIBERO，不作抽样改善或泛化判断。见`reports/oxe_sampling_pair_submission_233531.json`。

**用户选定两组各4,096步，来源内抽样对照代码已准备，未提交（2026-10-29）。** 从232587real_pose10240恢复至14336，保留DINO/x0/当前位姿/三组Adam/源RNG/完整145812窗池，比较原有放回与来源内无放回轮转；每步来源槽位、批大小、总更新及真实LinuxLR相同，平均曝光相同。两组共同77token训练补齐，显式披露相对旧动态补齐的调用变化；实际每步噪声/时间步/前后RNG仍须配对相同。Bridge后缀不同窗覆盖55.5%→80.1%，BC-Z8.10%→8.44%，不声称完成全池。约45分钟估计/60分钟作业上限，每2048步完整断点，提交前实际源CPU权重/合同/order/全部4096LR检查；优化前原120窗×3seed bit-exact重放。5项真实数值目标/原采样CPU替身回归通过，含实际主入口与保存核验、配对随机流、半程中断Adam/RNG精确恢复和错误拒绝；13成员265952字节包核验，无模型/图片缓存。尚未在集群执行真实准入/推理/训练，原233426不追加预算、不当新起点，主模型/13条件协议不改，不自动LIBERO。见[方案与上传提交](OXE_SAMPLING_PAIR_20261029.md)、`reports/oxe_sampling_pair_validation_20261029.json`。

**训练安排CPU审计完成：窗口曝光与收尾LR均未与旧有效方案匹配；全覆盖轮转仅形成CPU候选（2026-10-29）。** 原DINO229116每Bridge窗平均35.695次（23–72），旧后4096步LR1e-5；当前x08192前缀Bridge1.583/LT1.338/BC-Z0.170/Fractal0.602，到10240约1.976/1.679/0.213/0.750，新位姿端口只见其中2048步（Bridge0.393次/BC-Z0.043次）。短前缀在原65536曲线约1.9e-4高LR段结束，未完成退火；完整65536是epsilon版本，不混称完整x0失败。原full等来源有放回抽样BC-Z仍24389窗未见（25.46%）；source次数相近不等于每窗次数相近。CPU保持原来源槽位/批尾/524276呈现/65536步/全部实际LR，来源内部改无放回打乱轮转，四源未见数均可为0、Bridge12–13/LT10–11/BC-Z1–2/Fractal4–5；仅覆盖计算，不是性能改善或CUDA随机流一致性证明，BC-Z平均次数仍1.37。4项CPU统计/真日程/暖身分类/轮转不重复及来源保持检查通过，0真实模型/训练/新任务，旧合同/模型不改。曝光不是充分性阈值，233426高LR仍能学也反对单因归罪LR；不得直接宣称唯一欠训练或重复混训因果实验。见[审计](OXE_EXPOSURE_SCHEDULE_REVIEW_20261029.md)、`reports/oxe_exposure_schedule_review_20261029/report.json`；候选design_only_not_submittable，无新训练包或自动加预算/LIBERO。

**233426原文件齐全，独立CPU复算通过：8/8轨迹位姿改善，旋转主要为稳定残差（2026-10-29）。** report423591B+NPZ24339B共447930B；所有目标/掩码与冻结8窗参考逐值相等，512批次/真实LinuxLR/噪声记录及合同修复SHA一致，原数组全部均值/逐窗/严格门槛复算通过，3项CPU方法/真实入口检查通过，无真实模型前向或新增采样/训练。8窗轨迹位置和旋转均优于训练前及各自不动；旋转范围2.36–8.04°，三采样间旋转差均值0.197°，对目标4.617°，描述为稳定偏差而非抽样抖动主导，不追加种子。夹爪有效103标签窗口等权准确率100%，不作抓取成功。严格四均值/7窗门槛仍false（1/8）；不调阈值或加预算。核心训练/评价/保存73.05秒（不含之前加载/缓存），峰值4.451GiB。完整权重未下载，不冒称从字节独立验收。诊断收口，后续先审训练曝光/优化安排，不再把问题简单归因DINO无法拟合，也不推出唯一曝光根因或LIBERO可成功。见`reports/oxe_fixed8_result_233426/download_verification.json`、`cluster/independent_audit.json`及[更新结果](OXE_FIXED8_RESULT_233426_20261029.md)。

**233426固定8窗512步及完整生成完成：显著学会，严格精拟合门槛仍未过（2026-10-29，用户终端）。** 两个完成标记可见，无新异常；10个更新打印含最后512、原8固定indices及原集群LR精确匹配，依据不可变完成流原预测重放/缓存全梯度噪声RNG/冻结及落盘模型与512日程验收已通过，但新完整报告/NPZ/权重未本地下载。轨迹位置1.67264→0.09503、旋转16.12038→4.61720°，终点2.38945→0.16442/21.15207→6.82274°，四均值均超过不动；位置仍归一化记录单位。严格门槛四项≤不动10%且7/8逐窗满足仅1/8，gate仍false，旋转主要差距，不调低门槛或宣称精拟合/泛化成功。固定反复曝光证明当前DINO+新OXE链路能明显改善这些样本的完整生成，同时减少来源/任务竞争，不能锁定唯一曝光根因或直接扩大训练。0新增预算，诊断模型不替换主模型。下一步仅收取现成小报告/NPZ做CPU逐窗旋转与门槛复算。见[结果](OXE_FIXED8_RESULT_233426_20261029.md)、`reports/oxe_fixed8_result_233426/analysis.json`。

**固定8窗LR修复后作业233426由用户报告提交（2026-10-29）。** 原文`OXE FIXED8 LR REPAIRED: 233426`。依不可变提交脚本，打印此回执前须旧233339已终止、真实232587检查点CPU元数据512步原集群LR逐值核对通过；未收到完整CPU输出，不冒称独立逐文件验收。当前只有编号，未观察Slurm状态、GPU入口实际加载/原预测重放/缓存梯度守卫、任何新增更新或拟合结果。仍同8个Bridge窗/512步，无新增预算；旧失败目录和原模型保留。日志`/projects/Zeroshot/logs/oxe-fixed8-lr-233426.out`，输出`/projects/Zeroshot/runs/oxe-fixed8-lr-repair-233426`。等待同一作业，不重复提交、追加seed/预算或LIBERO。见`reports/oxe_fixed8_lr_repair_submission_233426.json`。

**233339已启动并加载权重，在首次优化前因新LR参考逐值校验失败；独立修复就绪、未提交（2026-10-29）。** 退出是qualification.load_source的512步LR exact断言，非调度拒绝、TF警告或CLIP多余视觉键；依固定控制流0优化、缓存/采样尚未开始，无拟合结论。直接对已下载Linux完整65536历史复算，Windows新计算NPZ的5步10258/10301/10405/10492/10552各3标量差最多2.7105054312e-20；此前CPU512入口mock了load_source且两端都在Windows计算，漏测实际加载守卫，已向用户承认。新13KB独立包不改原14文件/失败marker/模型/Adam/RNG/8窗/512预算，仅用原集群历史实际512LR替换资格核对及落盘复算的新增参考，exact守卫和零容差保留，报告/诊断policy另附显式修复绑定及有效源码。提交前登录节点现有环境CPU mmap真实232587检查点元数据，要求全部512步精确匹配才sbatch；不另排GPU诊断/预检，失败日志须无更新且旧作业已终止。3项CPU测试复现原实际load_source失败、同加载函数修复后含全部3Adam恢复/512更新/原100步采样/落盘核验、真正CPU元数据入口无CUDA及拒绝已训练失败，生产Linux数值在Windows夹具由存档表模拟，实际集群CPU检查仍待用户执行。原成功CLIP/DINO证据保留，无新提交/真实模型执行或泛化结论。见`reports/oxe_fixed8_failure_233339/analysis.json`、`reports/oxe_fixed8_lr_repair_validation_20261029.json`。

**当前OXE固定8窗可学习性检查233339由用户报告提交（2026-10-29）。** 原文`OXE FIXED8 LEARNABILITY: 233339`。计划从232587 real_pose10240原权重/三组Adam矩/RNG恢复，先原3seed前8预测精确重放与原生/缓存梯度、噪声/RNG一致守卫，再同8个Bridge训练窗口反复512更新并仅完整采样评价固定终点。当前仅收到编号，未观察Slurm状态、GPU运行、守卫通过、任何新增更新或拟合结果。日志`/projects/Zeroshot/logs/oxe-fixed8-233339.out`，输出`/projects/Zeroshot/runs/oxe-fixed8-233339`。等待同一作业，不重复提交、重做旧CLIP/DINO正对照、加预算/seed或LIBERO。见`reports/oxe_fixed8_submission_233339.json`。

**当前OXE固定8窗可学习性检查代码完成，未提交（2026-10-29）。** 用户确认继续定位新OXE链路；旧CLIP固定场景3/3抓取与229116只换DINO仍改善均保留，不重做骨干/旧小样本正对照。复用232587 real_pose10240全部core/port/三组Adam矩/RNG，仅把曝光改为原固定120评价中首8个Bridge训练窗同序batch8反复512步；原DINO/CLIP文本冻结、CLS/x0/源轴标签/状态/掩码/联合夹爪损失/dropout不改，LR仍65536时域10240→10752。仅缓存冻结特征，训练B8与评价B4/全120指令padding分别保留，先原3seed前8预测bit-exact，再原生/缓存路径全部梯度、loss、噪声及RNG逐值守卫，0优化探针；首实际更新也须重现。只评价512固定终点，四项均值≤不动10%且7/8逐窗≤20%（含声明的数值下限）才算此8窗位姿拟合；不是厘米标定/泛化/抓取，失败不自动加预算。预计缓存路径几分钟，更新段600秒/作业20分钟上限；仅保存约0.5GB独立诊断策略和小报告，2GiB空间预留，不复制原Adam/编码器缓存，不写源checkpoint。4项CPU实际入口（真实数字标签、toy编码器及解码器、完整512步/原生100步采样/落盘复算、缓存拒绝与固定门槛、metadata拒绝）通过；编译/LF及14成员25,859字节ZIP核验，未加载真实冻结模型或跑CUDA/集群/Bash。见`deployment/oxe_fixed8_learnability_v1/README.md`、`reports/oxe_fixed8_validation_20261029.json`。

**232587分来源汇总补齐：训练窗也未超过不动，但曝光稀疏，尚非记忆测试（2026-10-29）。** 用户报告8组来源/角色汇总；真实位姿对空值在验证全部14项可用指标更优，训练12项更优、BC-Z终点位置与旋转2项更差；不能概括所有来源/角色全改善。真实位姿相对不动训练/验证各14项均差；相对旧8192训练14项全差，验证仅LT轨迹位置略好。旧参考及不动与232091完整原报告全部均值在1e-12内匹配，新原预测仍未下载。进一步CPU数固定日程：32已见训练窗前8192只见1–5次，续2048里28窗未再次抽到（Bridge6/8、BC-Z8/8、Fractal8/8）；所以已见窗差不等于反复练习后仍无法记住，不能认定唯一欠拟合或普通过拟合。已有条件置换/五噪声步与公式审计不重跑，下一项候选为少量固定训练窗的完整生成可学习性检查，先补基本链路证据；尚未设定/提交新训练或LIBERO。见`reports/oxe_current_pose_formal_232587/source_summary_analysis.json`。

**当前位姿短对照232587终点评价完成：状态信息有配对收益，仍未达到不动能力（2026-10-29，用户日志）。** 同容量空值对照与真实位姿均从8192续2048到10240，固定120窗/三seed采样及METRICS verified、PAIR COMPLETE可见，源冻结91175b…不变。Bridge64真实轨迹位置1.70268/旋转15.17858°，空值1.99396/15.93659°；真实四项配对区间全<0，支持本次状态信息作用。但旧8192为1.55689/13.69836°，不动1.02414/12.06304°，真实四项均值仍差于二者，未升级主模型或建立基本能力；不能把两组终点退化纯归因训练时长，新增分支/优化是共同变化。日志5个批次/LR/掩码、12个均值差和门槛已本地复算，旧参考8均值与232091原报告逐值一致；新NPZ/完整报告未下载，未重算逐组bootstrap或独立验收完整checkpoint。单训练seed/复用验证组局限保留。下一步只读现成报告训练及各来源指标，不追加训练/推理/seed或LIBERO。见[结果](OXE_CURRENT_POSE_RESULT_232587_20261029.md)、`reports/oxe_current_pose_formal_232587/analysis.json`。

**当前位姿正式短对照232587由用户报告提交（2026-10-29）。** 原文`OXE CURRENT POSE SHORT PAIR: 232587`。计划从231297 sample8192原权重/Adam/plan/RNG，两组真实当前位姿与同容量mask空值控制各续2048到10240；原LR时域65536，随后仅固定120窗/三采样seed终点评价。当前只有提交编号，未观察Slurm状态、首两步逐值重放、实际新增更新或生成结果，不据此认定已运行/完成或泛化改善。日志`/projects/Zeroshot/logs/oxe-pose-pair-232587.out`，输出`/projects/Zeroshot/runs/oxe-pose-pair-232587`。等待同一作业，不重复提交/预检、追加预算或放行LIBERO。见`reports/oxe_current_pose_formal_submission_232587.json`。

**当前位姿预检232371资源回执已验收，正式短对照代码就绪，未提交（2026-10-29）。** 用户原verify输出确认6次可丢弃优化、原core首步loss/梯度/噪声一致、完整core/port/Adam/RNG精确恢复；峰值对照4.69531、真实4.73438GiB，正式两臂更新合计预计21.71分钟（不含重放/初始化/保存/生成/IO）。新约20KB源代码包固定从231297 sample8192原权重/Adam/plan/RNG各续2048到10240，原LR时域65536、真实当前位姿对同容量mask空值控制；不使用工程预检学习权重。首两步须与232371逐值重现，三组优化器/实际噪声流与固定批次/掩码逐步守卫，另存完整断点，保留旧文件；同一A40一小时上限，不自动重试或加预算。终点只用原120窗/三seed/DDPM100，含旧8192回归参照与不动；位置仍归一化记录单位，LT仅XY。4项新CPU集成（实际check-only、正式驱动缩短toy夹具及中断恢复、真实标签120窗评价入口替身、指标门槛）及原8项回归通过，编译/LF/ZIP逐成员核验；无真实编码器/CUDA正式训练，未执行本地Bash，未提交集群或放行LIBERO。原232371完整report仍在集群，正式入口将绑定其真实SHA并核对资源回执。见`deployment/oxe_current_pose_formal_v1/README.md`、`reports/oxe_current_pose_formal_validation_20261029.json`。

**当前位姿预检232371实际通过，资源回执待只读验收（2026-10-28）。** 用户输出120窗1101/1102/1103逐值重放exact、两组各8193/8194及恢复后重复8194，共6优化调用，末尾PREFLIGHT PASSED。原首批loss0.0931148976/pose0.0480357073/gripper0.1803167462及core梯度SHA c3fb670a…c43bd相同；真实值梯度0.00734154、对照严格0；两组重复第二步loss/梯度逐值相同，本地按原suffix复核indices通过。依据原固定源码末尾守卫推断core/port/Adam/plan/schedule/RNG完整摘要恢复与源文件未写通过，原报告尚未下载，不冒称独立完整state文件验收。warm update对照0.36434秒/真实0.27183秒，2048/臂合计更新约21.71分钟，不含冷启动/重放/IO/评价；日志未打印显存，下一步仅执行已有verify_preflight.py读取report，不重训或重跑预检。此为接线/恢复证据，不根据两步loss判断泛化收益，正式包未发布/提交，LIBERO仍不放行。见`reports/oxe_current_pose_preflight_232371/analysis.json`。

**当前位姿条件工程预检232371由用户报告提交（2026-10-28）。** 原文`OXE CURRENT POSE PREFLIGHT: 232371`。计划复用231297 x0的8192权重/Adam/plan/RNG，在原120窗重放后做真实位姿与同容量mask空值控制，共6次可丢弃优化并验证精确恢复。当前只有编号，未观察Slurm状态、真实重放/梯度/恢复、更新或资源，不认定预检已通过或泛化已改善。日志`/projects/Zeroshot/logs/oxe-pose-preflight-232371.out`，输出`/projects/Zeroshot/runs/oxe-pose-preflight-232371/`。等待同一预检，不重复提交；正式2048/臂仍暂拟，未发布/提交，不追加预算或LIBERO。见`reports/oxe_current_pose_preflight_submission_232371.json`。

**当前位姿条件预检代码完成，CPU验收通过，未提交（2026-10-28）。** 复用231297 sample8192原学习/合同SHA9be1b13d…121fd/9cd86ab7…844ae及原Adam/plan/RNG，不重训旧8192。新增外置18→1024无bias零线性分支18432参数：真实当前XYZ+旋转6D值与9维可用性mask，对照同容量同mask但值置零；LT仅XY，训练145812当前行统计标准化XYZ，旋转不缩放，原CLS/x0/DINO/标签/来源配比不变。先120窗三seed原预测bit-exact重放与两分支context零端点一致，再每组2逻辑步、恢复第2步一次，共6可丢弃工程优化，原core/port/Adam/plan/LR/RNG须逐值重现；在RAM序列化，不新增两份1.6GB磁盘checkpoint或迁移工程权重。8项CPU含实际check-only/重放入口/两臂工程驱动替身、真实数据特征、缺失/因果/梯度/Adam迁移与恢复通过；真实CUDA未跑。约94KB/13文件包哈希核验，旧48文件不改。正式暂拟相同8192起点各续2048到10240、原LR时域65536，需预检资源后才发正式代码，无自动任务/加预算或LIBERO。见`deployment/oxe_current_pose_preflight_v1/README.md`、`reports/oxe_current_pose_preflight_validation_20261028.json`。

**232091原结果已下载并完成CPU审计：总体幅度接近，位移方向对应弱（2026-10-28）。** 三文件857055字节，全部原NPZ指标复算/120标签掩码与角色group/payload/input_row逐值核对、修复合同/有效源码SHA通过，未加载真实模型。x0 Bridge训练/验证位移RMS比0.961/1.049，能量方向对齐−0.002/−0.059；三采样波动占位置MSE16.4%/12.7%，三次位置均值仍是不动MSE的1.611/1.943倍。验证第1步误差0.29831对不动0.18745，问题已在近期输出出现，不能只怪长未来或采样随机性。统计不直接证明坐标bug/条件失效根因，不概括所有窗口幅度正确。结合已核源码，当前位姿去噪器无当前XYZ/姿态、旧正对照有9D状态输入，已整理仅补当前位姿的同容量/mask空值控制候选；仍设计态，无预算/训练包/提交，不同时改x0/DINO/标签/读出或放行LIBERO。见[CPU审计与下一项假设](OXE_EXISTING_POSITION_AUDIT_232091_20261028.md)、`reports/oxe_target_eval_repair_232091/position_audit_summary.json`。

**232091现成分来源摘要补齐：已见训练窗也未超过不动，不能只按过拟合解释（2026-10-28）。** 每源8个实际见过训练窗，x0轨迹位置相对各自不动高Bridge48.45%、LT63.99%、BC-Z90.02%、Fractal54.19%；验证高52.02%/86.48%/93.13%/136.09%。所有可用轨迹/终点位姿均值仍差于不动，LT旋转null符合XY能力；Fractal训练位置sample比epsilon高0.44%，不能概括所有指标全改善。Bridge验证与已复算日志逐值一致，其他均值来自用户报告读取，原完整NPZ/report未下载。只确定这些训练窗生成也差，不能凭小样本判定纯欠拟合、数据错误或条件信息根因，8192不足以外推65536x0。已审旧诊断避免重做混训/公式/骨干试验，下一步只下载已有3小文件、本地CPU做位移幅度/方向/三采样波动分解并从原数组复算。新只读工具5项CPU方法/完整NPZ入口夹具通过，真实诊断待文件，不产生缩放拟合、新策略/采样/训练或LIBERO。见`reports/oxe_target_eval_repair_232091/source_summary_analysis.json`、`tools/audit_existing_oxe_position.py`、[结果与下载步骤](OXE_TARGET_PAIR_SHORT_RESULT_232091_20261028.md)。

**232091修复评价完成：主入口只改x0有明确改善，8192短终点仍未超过不动（2026-10-28）。** 用户日志含六组120/120采样与REPAIR COMPLETE、Bridge64全部逐窗指标，无新异常；未收到sacct或原NPZ/report，尾部^C不当作取消作业。epsilon轨迹位置1.908989/旋转22.45654°，x0为1.556888/13.69836°，不动1.024137/12.06304°；x0相对epsilon位置降18.44%、旋转降39.00%，终点位置降24.88%、旋转降37.07%。四项x0−epsilon预定组bootstrap区间均<0，但x0−不动四项均值/区间全>0；未建立基本超过不动能力，不外推65536步x0无效或LIBERO成功。位置为归一化记录单位不能称厘米。已按日志192逐窗复算12均值/比例/组区间，不动还与本地不可变同标签逐窗复算一致；候选原预测未下载，单训练seed/重复验证组局限保留。下一步只读现成报告32训练窗/各来源汇总，不重训/重推理/加seed或恢复大评估。见[结果与下一步](OXE_TARGET_PAIR_SHORT_RESULT_232091_20261028.md)、`reports/oxe_target_eval_repair_232091/analysis.json`。

**231297评价修复232091由用户报告提交（2026-10-28）。** 原文`OXE TARGET EVALUATION ONLY REPAIR: 232091`。仅复用231297的epsilon/sample两个8192终点，原120窗/三采样seed和全部训练/冻结/噪声流守卫保持，新增训练预算为0，原失败目录保留。当前只有编号，未观察Slurm状态、真实修复入口验收、推理完成或生成误差，不认定x0已改善。日志`/projects/Zeroshot/logs/oxe-target-eval-fix-232091.out`，新结果`/projects/Zeroshot/runs/oxe-target-short-eval-fixed-232091/`。等待同一评价，不重复提交/重训、追加窗口/seed或放行LIBERO。见`reports/oxe_target_eval_repair_submission_232091.json`。

**231297双短训练终点保留，评价入口字段错误已修成独立推理包，未重训/提交修复（2026-10-28）。** 用户sample终点SHA9be1b13d…121fd、合同9cd86ab7…844ae、2346.64秒/4.60352GiB，已知两表常量恢复91175b…、编码器/学习参数未写。原run_pair及evaluate在错误行前的verify_pair均已返回，依原控制流推断两臂8192、原首批/合同/输入噪声流检查通过；原报告/权重未下载，不冒作独立远端文件验收。真正退出是我把选择协议generation_steps误写成sampling_steps的KeyError，发生在模型/特征与生成开始前，TF CPU警告不是该异常原因；先前本地六项检查漏测实际评价main。新约11KB独立包保持原48文件/训练协议/权重/120窗/三seed/指标和所有守卫，只在内存修字段并把评价及复算目录指向新路径，保留旧空evaluation及失败证据。三项CPU回归先复现原异常，再以编码器/策略替身和真实选定数字目标执行修复main全路径、90批采样调用/保存复算/旧文件未写，未用真实编码器/CUDA/优化器；ZIP与源码/绑定核验通过。不重提原双训练入口，仅用户手动提交评价修复，不加窗口/seed/训练或LIBERO。见`reports/oxe_target_pair_short_231297_failure.json`、`reports/oxe_target_eval_repair_regression_20261028.json`、[评价修复命令](../deployment/oxe_target_pair_eval_repair_v1/README.md)。

**主入口目标短配对231297由用户报告提交（2026-10-28）。** 原文`OXE MAIN TARGET SHORT PAIR: 231297`。同一作业内fresh epsilon/sample各8192步，原数据池/seed42/B8/65536学习率时域固定，仅原生预测目标不同，随后按既定120窗/三采样生成评价。当前仅收到编号，未观察Slurm状态、A40冻结身份/首批epsilon损失梯度重放、实际训练更新或生成误差，不据提交编号认定运行/通过。日志`/projects/Zeroshot/logs/oxe-target-short-231297.out`，结果`/projects/Zeroshot/runs/oxe-target-short-231297/`。等待同一作业，不重复提交、不追加预算/seed或恢复LIBERO。见`reports/oxe_target_pair_short_submission_231297.json`。

**当前主入口目标短配对代码完成，未提交（2026-10-28）。** 用户同意只改目标。原主模型没有8192权重，故fresh epsilon/sample各8192步，原145812训练池/抽样前缀、seed42/B8和65536学习率时域不变；当前CLS读出/无9D位姿/数据标签/夹爪/AdamW均保留。同一次A40配对，预计80–100分钟、3小时上限、8GiB余量，保留两份完整前缀状态，禁止不完整导出删Adam。原core/完整协议不改，新显式诊断协议及sample冻结profile元数据派生如实记录；原冻结SHA/首批epsilon损失梯度逐值守卫、全批次掩码/LR、两臂噪声与timestep/noisy-action流摘要检查。120预定生成窗口：每来源8个实际见过训练窗，64Bridge验证组+其他来源各8旧锚点，DDPM100三采样；按来源有效维度评价，LT仅XY、位置无新增物理校准。六项CPU集成检查、33 Python编译、shell LF/嵌入Python和48 ZIP成员哈希通过，Bash/真实编码器/CUDA未执行；约0.90MB包不含模型数据。不同目标训练MSE不直接比较，只看固定生成误差及不动，短结果不外推长训练结论、不自动追加或恢复LIBERO。见[设计、边界及提交](OXE_TARGET_PAIR_SHORT_20261028.md)、`reports/oxe_target_pair_short_validation_20261028.json`。

**231050现成训练指标补齐：epsilon的退步并非只在留出集（2026-10-28）。** x0训练轨迹3.65484cm/10.08074°、终点5.10615cm/14.14550°；epsilon训练7.99423cm/15.16188°、终点10.76013cm/20.74004°，位置放大2.187倍、增118.73%。x0数值与本地原预测复算后的报告逐值相同；候选训练数据来自用户现成report读取，候选原报告/NPZ未下载，不冒作独立预测复算。训练与开发生成都受损，不能按纯过拟合/只泛化失败解释，也不能据完整生成差直接断言噪声MSE未收敛或epsilon算法有错。该分支继续关闭，不重读/重训/扫seed。下一步为当前主入口显式x0训练的有界匹配方案，数据/状态/读出/优化同时保持不动，以真实生成留出误差判读；未实现/提交新控制，不改已训练epsilon推理flag，不放行主LIBERO。见`reports/bridge_dino_epsilon_231050/train_and_development_review.json`。

**231050条件目标对照完成：仅换epsilon使有效DINO+状态配方失去不动优势（2026-10-28用户比较尾部）。** 8192步、旧预测重放/缓存/初值一致；x0轨迹5.64326cm/12.84923°，epsilon由基线加报告差值恢复9.00136cm/17.02612°，不动8.21233cm/15.09840°，终点epsilon11.01783cm/22.58841°。窗口加权位置/旋转较x0高59.51%/32.51%，x0四指标都好于不动、epsilon四指标都差。演示等权位置退步2.95813cm、区间[1.75469,4.14761]；终点位置3.01717cm、区间[1.12937,4.95455]。路径旋转区间跨0、终点旋转区间>0，均不含训练seed不确定性；不混用窗口加权差与等权区间。当前只收到终端比较，候选原report/预测未下载，训练误差未粘贴，不冒作独立预测或拟合诊断。关闭该epsilon分支、保留x0正对照，不追加预算/seed；已确认这项改动在固定有效配方下有害，不宣称当前四源只改x0已修好或DDPM公式bug。下一步先读现成训练指标再限定主入口匹配小控制，不直接改已训练epsilon推理flag，不自动提交或放行LIBERO。见[结果与边界](BRIDGE_DINO_EPSILON_RESULT_231050_20261028.md)。

**DINO状态配方epsilon候选231050由用户报告提交（2026-10-28）。** 原文`BRIDGE DINO EPSILON CONDITIONAL CONTROL: 231050`，仅一个seed42/8192步epsilon，复用229116 x0，不追加x0训练或工程训练。当前仅收到编号，未观察Slurm状态、旧4窗×3seed重放、首批x0损失/梯度检查或任何更新，不推断已运行或通过。日志`/projects/Zeroshot/logs/bridge-dino-epsilon-231050.out`，结果`/projects/Zeroshot/runs/bridge-dino-epsilon-231050/`。等待同一作业，不重复提交，不改主OXE或放行LIBERO。见`reports/bridge_dino_epsilon_submission_231050.json`。

**229116现成正对照报告已收齐，DINO状态配方单臂epsilon代码准备完毕，未提交（2026-10-28）。** 22.52MB归档SHA0b11d4c6…，19指定文件、两臂完整8192日志/学习率及全部训练/开发指标独立复算通过；原模型未下载，远端DINO权重SHA730f06ec…6f5绑定，提交前重核。该DINO臂实际3019.90秒，预计新臂50–65分钟，非主模型5–6小时。只新增epsilon8192，复用原x0；不重做197845/公式或重训x0。新21,634字节/11文件包保留旧core/数据/cache/9D状态/读出/Adam日程，四显式插入点增加常量准入、旧训练权重首4窗×3seed bit-exact重放、fresh参数恢复、原首批x0损失/模块及全局梯度逐值检查和checkpoint干预身份。0额外工程训练，任一不一致在首候选优化前停止；旧常量变体只允许两root表、要求原完整初始化digest匹配，防止再漏接既有常量处理。新CPU合成集成/拒绝夹具/同图清梯度复用及源哈希编译/LF/ZIP核验通过，0真实模型/CUDA/候选更新或作业。完整1836train/40dev原3seed生成评价保留，历史同型号A40非同allocation、10重复开发演示和单训练seed限制明确；主OXE/LIBERO仍暂停。见[准备与提交步骤](BRIDGE_DINO_EPSILON_CONTROL_20261028.md)。

**x0/epsilon重复项审阅完成，先复用旧结果而不立即重训（2026-10-28）。** 找到186009小窗配对、193195/193390及197783训练/197845完整DDPM100评价。原197845报告哈希与分析一致，重新按3采样数据复算全部训练/开发轨迹/终点均值相符；开发x0位置8.97654cm/旋转13.79191°，epsilon9.90849cm/18.36505°，不动8.21233cm/15.09840°，不能说x0旋转也差于不动。旧与当前核心diffusion_loss/sample完整AST相同，不重验通用公式/调度或旧CLIP实验。197845与229116训练/开发窗口集合相同，但后者DINO+9D当前位姿支路、batch8/Adam阶梯LR/冻结夹爪不同；旧实验不直接识别该有效配方的目标条件效应。先收取现成229116原report/预测/历史，本地/默认Downloads未找到原归档，复用已有19小文件打包/下载验收工具，不含权重/新GPU/训练；拟最多一个epsilon候选复用旧x0，仍须完整正对照和初始常量准入，当前仅候选合同，未生成训练包或提交。首次审阅脚本误用x0作为原字典键而报KeyError，按实际sample键修正后只读核对通过，源报告未改。见[重复项与未回答条件问题](PREDICTION_TARGET_REUSE_REVIEW_20261028.md)、`reports/prediction_target_reuse_review_20261028/report.json`。

**固定曝光监督评价230933指标收到并收束该分支（2026-10-28）。** 64预定验证组/3采样，原混训轨迹位置1.452749、候选1.410752、不动1.024137（归一化记录单位）；旋转17.3499/17.3738/12.0630度。去其他来源监督位置仅低2.89%、终点位置低4.64%，两位置组bootstrap区间<0，旋转区间含0；候选位置/旋转仍比不动差37.75%/44.03%，四指标全差，仅弥补约9.8%轨迹位置缺口。支持此固定曝光单seed下小幅其他监督位置拖累，不支持它足以解释主失败或所有混训有害；Bridge曝光/覆盖及状态/x0/读出/语义交互未排除。当前只有用户report摘要，差值/百分比已复算，原report/预测未下载，不冒作独立预测验收。关闭该有界分支，不追加seed/预算或主LIBERO评分；下一项拟沿已有效Bridge+DINO配方只隔离sample/x0→epsilon，先审既有实现/重复项，未实现/提交新实验。见[结果及边界](OXE_BRIDGE_SUPERVISION_RESULT_20261028.md)。

**评价230933两模型六组64窗采样均已打印完成，报告已写出，误差尚未读取（2026-10-28用户日志）。** mixed和bridge_supervision各1101/1102/1103三seed，共384条策略采样；当前构造冻结摘要前后均91175b…，恢复buffer列表为空、编码器/学习参数未写入，保留环境分派差异记录，不为TF重复工厂/不可用GPU警告改环境。日志只给`/projects/_ssd/Zeroshot/runs/oxe-bridge-supervision-eval-230933/report.json`路径，没有execution_complete字段/误差/区间，不能据报告路径推断最终成功或混训效应。下一步只用标准库读取既有report的完成标志、64组均值和配对区间，不重跑推理/训练，不作LIBERO或因果结论。原日志与解析保存于`reports/oxe_bridge_supervision_evaluation_230933/`。

**固定曝光监督配对评价230933由用户报告提交（2026-10-28）。** 提交前已有失败文件CPU比较明确编码器张量/顺序一致，仅posterior_mean_x0/xt各15/14元素浮点末位差（max7.45e-9/1.19e-7），当前命名比较终端证据已收，原始profile未下载；两次RECEIPT VERIFIED是读取原预检/完整训练结果，不是重训。评价计划同A40/同64验证组/1101,1102,1103三采样，比较mixed、bridge_supervision和static，0优化；只有提交回执，未观察实际推理/完成或误差。日志`/projects/Zeroshot/logs/oxe-bridge-supervision-repaired-230933.out`，结果`/projects/Zeroshot/runs/oxe-bridge-supervision-eval-230933/`。等待同一评价，不重复提交，不作混训因果或LIBERO泛化结论。见`reports/oxe_bridge_supervision_evaluation_submission_230933.json`。

**监督对照第二段230276完成全部65,536步，终端验收和导出通过（2026-10-28用户输出）。** Bridge131,244/全部524,276输入，保留6,617零Bridge步骤、末批keep4，原首批三损失/梯度范数与初始学习摘要保持；本地按原完整日程独立复算五项计数匹配。A40 reserved4.62695GiB、第二段9756.28秒。最终策略526,913,072字节，学习SHA`30bcedfe…48ef7fdb`、合同SHA`3abce3a3…a38a98`；原导出检查报告参数完整保留后退役本候选Adam状态。尚未下载最终策略/完整集群报告，终端通过不冒作独立文件验收；未提供独立Slurm状态。固定预算训练结束，下一步在同修复目录手动`bash submit_once.sh evaluate`，既定64不同Bridge验证组/3相同采样种子，比较混训/仅Bridge监督/不动，不再优化。不重复预检、训练、追加预算或主LIBERO评分；尚未提交评价，暂无泛化或混训负迁移结论。见`reports/oxe_bridge_supervision_stage2_230276_result.json`。

**修复版监督对照第二段230276由用户报告提交。** 用户原文`BRIDGE SUPERVISION REPAIRED stage=2: 230276`；计划恢复229800的32,768更新继续至65,536，仍原seed42/固定曝光/相同修复合同，不是fresh或追加预算。当前仅收到提交回执，未观察实际恢复、Slurm状态或新增更新，最新已确认计数仍32,768。日志`/projects/Zeroshot/logs/oxe-bridge-supervision-repaired-230276.out`，输出目录不变。等待同一作业，不重复第一段/预检，不自动提交评价；暂无新的混训因果或泛化结论。见`reports/oxe_bridge_supervision_stage2_submission_230276.json`。

**修复后监督对照第一段229800完成32,768步，终端验收通过（用户输出）。** 首批三损失/梯度范数逐值重放、参数版本/RNG不变，初始学习摘要9d332704…39e1b与预检相同；Bridge65,531/全部262,140呈现，保留3,300零Bridge步骤、末批keep0。本地从不可变原日程独立复算四项计数逐值相同；尚未下载完整集群历史或常量恢复事件报告，未冒作逐文件验收。A40 reserved4.60352GiB、9515.88秒约2小时39分。完成受守卫训练证明此前初始化故障已不再阻断该段，但不是泛化/负迁移结论。下一步仅在修复目录`bash submit_once.sh 2`恢复32,768至65,536，原预算不增、不从头训；第二段尚未提交，不重复第一段/预检或恢复主LIBERO。见`reports/oxe_bridge_supervision_stage1_229800_result.json`。

**冻结常量修复后的监督对照第一段229800由用户报告提交（2026-10-27）。** 对应零更新失败229597和已通过预检229577，仍是原seed42/32,768第一段、总65,536预算，不是新seed或重复预检训练。仅收到编号，尚未收到登录节点逐张量比较输出、实际常量恢复/完整冻结守卫/首批回放或训练更新，不据编号推断修复完成或作业运行。新日志`/projects/Zeroshot/logs/oxe-bridge-supervision-repaired-229800.out`，恢复记录`/projects/Zeroshot/runs/oxe-bridge-supervision-repair-229800/`，候选输出目录不变。等待同一作业，不重复提交；后续续训必须使用修复目录以保持合同一致，不自动接第二段/评价。见`reports/oxe_bridge_supervision_repaired_stage1_submission_229800.json`。

**229597在首次前向/更新前冻结身份守卫失败，既有常量修复已独立封装（2026-10-27）。** 用户完整日志确认A40 CUDA健康、tokenizer一致、0前向/更新/训练checkpoint，实际摘要`fc5316b4…0daa4`与此前227815相同；旧逐张量证据只差posterior_mean_x0/xt，编码器一致。当前229597逐张量actual_profile已在集群自动保存但尚未本地读取，不能冒称当前命名比较已完成。新监督封装漏接既有修复，已向用户承认；不重跑长模型诊断/两更新预检，不盲重提或松守卫。独立52,556字节修复包复用原成功seed43恢复helper/profile字节；登录节点先比较已有229597报告与profile，要求仅原两常量变体、编码器/顺序/值字节完全一致才可手动重提。构造时恢复原始常量后必须完整摘要91175b…和原首批损失/梯度及初始学习digest9d3327…通过，原包/失败marker保留。修复代码/协议摘要另写入候选checkpoint合同，第二段及评价须用同一新修复目录保持恢复一致；原训练预算/来源监督/曝光不变。新CPU比较拒绝夹具、toy仅两buffer恢复/参数/RNG及稳定合同绑定、编译/LF和ZIP逐成员核验通过；0真实模型/训练/新集群作业，复用旧断点测试记录，不冒称实际修复完成或泛化改善。见`reports/oxe_bridge_supervision_failure_229597/`、`reports/oxe_bridge_constant_repair_package_20261027.json`。

**固定Bridge曝光监督正式对照第一段229597由用户报告提交（2026-10-27）。** 绑定已验收短预检229577，从相同初始学习摘要fresh训练至32,768步，总65,536步不变；仅删除非Bridge监督分子，保留批次/原分母/Bridge曝光/零Bridge优化器步骤。当前只有编号，未观察Slurm状态、正式首批回放或任何更新/完成，不推断已运行。日志`/projects/Zeroshot/logs/oxe-bridge-supervision-229597.out`，输出`/projects/Zeroshot/runs/oxe-bridge-supervision-s42/`。等待同一作业，不重复提交或自动接第二段；主LIBERO仍暂停，无新泛化结论。见`reports/oxe_bridge_supervision_stage1_submission_229597.json`。

**固定Bridge曝光监督预检229577终端验收通过（2026-10-27，用户粘贴输出）。** 原首批总/位姿/夹爪损失及未裁剪梯度范数四标量与基线逐值相同，重放前0优化步、参数版本与RNG未变；原验证器通过两更新及中间保存/重构恢复检查，共16输入呈现、Bridge7次。实际A40 reserved4.37305GiB、61.30秒，新完整初始学习摘要`9d332704...39e1b`绑定正式fresh初始化；不将其冒充旧初始权重已存档的证明。当前依据用户终端，尚未独立下载原报告/检查点。新包无需修改或重复预检；下一步用户手动`bash submit_once.sh fit 1 229577`，提交脚本仍核实Slurm完成/原文件/空间，不转移预检学习权重，第一段32,768步、总65,536固定。尚未提交正式候选或主LIBERO，未产生新泛化/混训因果结论。见`reports/oxe_bridge_supervision_preflight_229577_result.json`。

**固定Bridge曝光监督对照短预检229577由用户报告提交（2026-10-27）。** 计划只做更新1保存、重构恢复后更新2，先核对原首批完整损失与梯度范数；当前仅收到作业编号，未观察Slurm状态、首批重放、恢复或更新结果，不推断已运行/完成/通过。日志`/projects/Zeroshot/logs/oxe-bridge-supervision-229577.out`，结果`/projects/Zeroshot/runs/oxe-bridge-supervision-preflight-229577/`。等同一作业结果，不重复提交；正式65,536步候选和主LIBERO评估均未启动，无新泛化结论。见`reports/oxe_bridge_supervision_preflight_submission_229577.json`。

**固定Bridge曝光的其他来源监督对照已实现并打包（2026-10-27），未提交。** 保持当前主DINO/CLS/epsilon/源轴标签与全部65,536批次，只去掉非Bridge监督分子；原分母、131,244次Bridge曝光、524,276全部输入及6,617个无Bridge优化器步骤保留，验证损失未改。复用seed42已完成混训基线，不重做旧Bridge骨干实验。新短预检实际更新1保存、重构恢复AdamW/plan/RNG、更新2，共2次；先精确重放旧首批四标量/梯度范数并记录新完整初始摘要，未伪造旧未存初始digest。CPU损失/梯度、零梯度AdamW、原解析参数、全日程/状态恢复及拒绝夹具、评价尾窗/四元数符号和26Python编译通过；无真实模型/编码器/CUDA前向或主策略更新，未实际执行bash。开发中修正新脚本NPZ每步重复解压，测试夹具遗漏Path/__file__也已修正，未改旧训练源文件。ZIP38成员2,595,549字节，完整成员/哈希核验；预检后才可手动分两段正式训练，估计5–6小时，之后固定64验证组/三采样做同A40配对，旧8组为辅助锚点。尚无新泛化结果，不认定混训根因，不恢复主LIBERO评估。见[对照设计和上传命令](OXE_BRIDGE_SUPERVISION_COUNTERFACTUAL_20261027.md)、`reports/oxe_bridge_supervision_package_validation_20261027.json`。

**按用户混训假设完成全量目标/监督权重审阅和只读后端梯度探针（2026-10-27）。** 原65,536历史的mask计数逐值一致，四源呈现各约25%，非BC-Z数量支配抽样；pose有效维度分母份额Bridge28.52%、LT7.42%、BC-Z32.03%、Fractal32.03%，不是实测梯度份额，不能凭此改权重。所有有效位置目标在逐轴±8内。固定原32开发缓存context，仅重建77.67M参数位姿后端，3预定噪声/8分层t共12反向13前向，0优化/新采样/编码器重建；平均Bridge对LT/BCZ/Fractal梯度余弦−0.459/−0.052/+0.645，合并仍+0.750，局部反向不证明整体负迁移。首次脚本遗漏旧TF32关闭设置，回放守卫阻止求导；按原设置修正后标量回放差0、阈值不变，失败保留。成功7.44秒、reserved0.70GiB、权重SHA前后一致；CPU矩阵复核通过。候选对照要固定Bridge曝光/全局日程，仅去掉其他来源监督分子并保留原分母；普通全Bridge同更新会多约4倍曝光，不能作纯负迁移判别。候选仅design_only，初始化绑定/预算待限定，无新训练/删除数据或主评估。见[混训诊断与边界](OXE_MIXTURE_DIAGNOSIS_20261027.md)。

**A40同卡CLIP/DINO配对229116两组8192步完成，终端验收通过（2026-10-27）。** 原40开发窗CLIP6.8082cm/15.1388°，DINO5.6433cm/12.8492°，不动8.2123cm/15.0984°；DINO较CLIP位置/旋转低17.11%/15.12%，较不动低31.28%/14.90%。终点和演示等权均改善；10演示6位置改善、7旋转改善、5共同改善，保留全部失败/退步。新A40 CLIP六最终指标及34个打印loss点与旧A6000逐值相同，未推广为所有权重或跨硬件都逐位一致。DINO在旧配方有效，不能把当前混合主模型差简单归因换骨干；仍未区分状态输入/x0目标/读出/混合来源因素，不宣称LIBERO成功，不自动放行大评估。当前本地仅用户终端与解析分析，原报告/NPZ/完整历史未下载；报告收取和纯CPU验收脚本已准备，夹具通过但不冒充真实验收，不下载大模型或重跑。见[配对结果及收取步骤](BRIDGE_A40_PAIR_RESULT_20261027.md)、`reports/bridge_backbone_pair_a40_229116/analysis.json`。

**A40同卡CLIP/DINO配对作业229116由用户报告提交（2026-10-27）。** 一个allocation先CLIP后DINO，各fresh seed42/8192步，复用原两份缓存与40开发窗；仅收到编号，尚未观察GPU健康、双缓存检查、初始参数核验或训练更新，不推断已排队/运行/完成。等待同一作业，不追加seed/预算或主LIBERO评分。日志`/projects/Zeroshot/logs/bridge-backbone-pair-a40-229116.out`，预期对比`/projects/Zeroshot/runs/bridge-backbone-pair-a40-229116/pair_comparison.json`。见`reports/bridge_backbone_pair_a40_submission_229116.json`。

**用户明确选择A40同卡双臂CLIP/DINO对照，新约1MB包准备完成（2026-10-27），未提交。** 一个A40 allocation依次fresh训练CLIP、DINO各8192步，保持旧Bridge完整日程/状态输入/x0/patch读出/动作/优化器和评价。复用两份A6000编码的视觉缓存和同一文本缓存，不新编码、不重复16更新预检；长训练前CUDA检查与完整双缓存核验，源228978实际预检通过按显式新helper合同绑定，不冒称A40已工程预检。两组实际初始化/真实输入/梯度/冻结检查保留，最终同allocation并独立复算NPZ。23文件编译/导入、原core/loader/context/check/advance代码一致性、6源准入与配对拒绝夹具、同预测零差和ZIP验证通过；0真实模型/更新/集群操作。原A6000包、两次零更新失败、缓存与主模型均未改。GPU影响数值复现的边界已向用户说明，不能据此解释巨大泛化差或CUDA未启动。见`docs/BRIDGE_A40_MATCHED_BACKBONE_PAIR_20261027.md`、`reports/bridge_backbone_pair_a40_package_receipt_20261027.json`。

**229042在CUDA健康门控再次失败，原训练入口未执行、零更新（2026-10-27用户日志）。** Slurm GPU标号3/本地可见0，UUID GPU-1331d351-1136-494f-8045-05df4cb4bc18、PCI00:1C.0，torch.cuda.init报No CUDA GPUs are available；与229016同PCI路径，不据此确定物理卡/驱动唯一原因或其余9卡状态。停止同样重提，原预检/缓存/失败证据保留。已向用户询问等A6000修复还是转A40双臂匹配重训；后者需额外CLIP对照，不能把硬件也变了的历史比较称严格单变量。未启动新作业/改代码/发送管理员消息。故障和未发送草稿见`docs/CLUSTER_A6000_CUDA_FAILURE_20261027.md`。

**Bridge仅换DINO同型号CUDA门控重提229042由用户报告提交（2026-10-27）。** 对应228978预检与零更新失败229016，同一8192步候选的运维重提，不是额外seed/预算。应先观察CUDA_HEALTH_PASSED，再看原runner输入/初始权重核验和训练更新；当前仅收到编号，未观察健康检查通过、排队/运行状态或任何更新。日志`/projects/Zeroshot/logs/bridge-dino-only-229042.out`，预期比较`/projects/Zeroshot/runs/bridge-dino-only-fit-229042/backbone_comparison.json`，不自动重复提交。见`reports/bridge_dino_only_cuda_retry_submission_229042.json`。

**用户Slurm记录确认只有gpu-a6000-1一个A6000节点、共10张卡（2026-10-27）。** 228978 COMPLETED和229016 FAILED均在该节点；预检日志PCI02:12.0与失败00:1C.0不同，不能据此排除整节点或确定具体物理故障原因。v1不同节点守卫不适用于此布局，未实际重提；准备v2同节点单次重提，复制原SBATCH全部资源/日志/时间头，先对Slurm分配的可见GPU做限时60秒CUDA初始化/单元素检查，再exec原job.sh。绝不手选/覆盖CUDA_VISIBLE_DEVICES，不变GPU型号、训练包/预算/缓存，不自动循环失败重试；旧FAILED marker与记录保留，锁/状态/无训练进展仍核实。内嵌Python与生成资源头本地核验通过，未实际运行bash/健康检查/提交。见`scripts/cluster/retry_bridge_dino_only_cuda_229016_v2.sh`、`reports/bridge_dino_only_cuda_retry_v2_20261027/receipt.json`。

**229016运维重提被distinct-node守卫停止，未提交新作业（2026-10-27用户输出）。** 仅观察“No distinct previously successful node confirmed”，尚无228978/229016实际NodeList，不推断所有A6000损坏。先读取两份sacct与cluster02逐节点GRES，检查同节点/名字解析及有无其他A6000，再决定节点约束；不改训练模型、GPU型号或重复预检。

**Bridge仅换DINO正式229016在CUDA启动守卫失败，未构造训练模型/未更新（2026-10-27用户日志）。** nvidia-smi风扇/温度/功耗ERR，torch.cuda不可用；run.py在原runner执行前退出。日志前面的PREFLIGHT VERIFIED是复核已存228978，不是新训练，不能把本次故障归为DINO拟合差。保留旧作业/日志和完整缓存，预检不重跑。独立运维重提脚本会查询两作业真实Slurm状态/节点，要求229016失败且228978完成、不同单节点；原锁/marker和无输出/无更新日志核实后归档失败marker，固定成功预检节点并排除失败节点，仍用原A6000/同包/同8192步/同cache。仅准备脚本未联系或提交集群；节点未知/相同则停止，不猜或换A40。见`scripts/cluster/retry_bridge_dino_only_cuda_229016.sh`和`reports/bridge_dino_only_cuda_retry_229016.json`。

**Bridge仅换DINO正式对照229016已由用户报告提交（2026-10-27）。** 对应已通过预检228978，固定fresh seed42/8192更新与旧207558配方，最终比较原40开发窗；不是当前主OXE模型的续训。仅收到作业编号，未观察Slurm状态、实际fresh初始化/首批更新或完成，不推断已运行。等待同一作业输出，不重复提交或追加预算，主LIBERO批量评估仍暂停。日志`/projects/Zeroshot/logs/bridge-dino-only-229016.out`，预期比较`/projects/Zeroshot/runs/bridge-dino-only-fit-229016/backbone_comparison.json`。见`reports/bridge_dino_only_formal_submission_229016.json`。

**Bridge仅换DINO预检228978终端验收通过（2026-10-27，用户完整日志）。** 415记录/1876输入全部恢复并与旧207558数值身份一致，DINO1876帧编码结束；16次工程更新/恢复分支通过，初始参数匹配、文本token逐值一致。reserved5.63086GiB、warm0.13965秒、正式估计4245.65秒约70.76分钟，不含排队/实际I/O偏差。CuBLAS/attention及TF重复工厂警告未阻断运行；不为警告修改已固定训练设置。只检查训练探针、未开发评分，不作DINO效果判断。正式一次8192步尚未提交，已有包支持`bash submit_once.sh fit 228978`且提交前重验本地集群报告/空间；不重复预检、CLIP训练或上传新包。原始终端与receipt保存在`reports/bridge_dino_only_preflight_228978/`；未独立下载report.json或查询Slurm完成状态。

**Bridge仅换DINO短预检228978已由用户提交（2026-10-27）。** 仅收到编号，尚未观察Slurm状态、CUDA初始化、逐行输入核对、DINO编码或预检通过；不推断已排队/运行/完成。正式8192步尚未提交，不重复提交或追加训练。日志`/projects/Zeroshot/logs/bridge-dino-only-228978.out`，预期结果`/projects/Zeroshot/runs/bridge-dino-only-preflight-228978/report.json`。见`reports/bridge_dino_only_preflight_submission_228978.json`。

**用户明确授权Bridge仅换冻结视觉骨干对照，约1MB提交包准备完成（2026-10-27），未提交。** 以207558原1836/40窗、seed42、8192×8批次/Adam阶梯LR、9D当前位姿、CLS+patch读出、x0/sample、工具相对标签/裁剪3和冻结夹爪为固定配方，仅把视觉换为冻结DINOv2-L/14及必需通道归一化。复用旧CLIP结果/精确文本缓存；真实1876行输入/payload、初始下游参数、A6000/Torch版本必须与旧证据匹配。保留旧runner/core原字节，新帮助函数核验真实数值后单独缓存DINO特征；新预检16工程更新通过后，正式fresh一次8192步，手动分阶段提交、无自动作业/额外预算。比较原40开发窗CLIP6.8082cm而非另一批留出的4.7661cm，保留全部预测与逐演示差，重复开发/单训练seed/历史对照限制明示。21文件编译/导入路径/哈希/旧NPZ度量复算、5验收正负夹具及ZIP逐成员核验通过；0真实模型/前向/训练/集群操作，集群预检仍待用户运行。不打包原始数据或权重。见[控制变量设计及提交说明](BRIDGE_DINO_ONLY_CONTROL_20261027.md)、`reports/bridge_dino_only_package_validation_20261027.json`与`reports/bridge_dino_only_package_receipt_20261027.json`。

**旧Bridge到当前主模型的变更清单核对完成（2026-10-27）。** 新旧并非只换CLIP/DINO：旧207558把当前XYZ+6D旋转经9→1024投影加入条件，实际训练梯度非零；当前神经条件只有图像/文本，上一夹爪命令及可用性仅给独立夹爪头。旧sample/x0目标改为epsilon，旧CLS+patch均值读出改CLS，动作改源轴左相对，裁剪3→8，优化与人群也变。两者均多尺度FiLM/DDPM100cosine，图像prepare_image函数AST一致；Bridge名义5Hz到5Hz的有效完整目标仍是后16行，不能把Bridge差异笼统归因插值。新训练/验证选中的8+8窗均不与旧训练/开发/留出精确窗口键重合；不同键是否别名不重做原内容审计。旧Bridge收益保留，DINO单独因果影响未识别。仅读源码/元数据/原梯度，0模型/推理/更新/仿真/集群作业，未直接回滚设计。见[变更审阅](BRIDGE_TO_MAIN_TRANSITION_AUDIT_20261027.md)、`reports/bridge_to_main_transition_20261027/report.json`。

**当前四来源模型实际见过的训练窗完整生成检查完成（2026-10-27）。** 复用同一seed42最终权重/原DDPM100与已有验证结果，新选每来源8个不同训练组中实际抽中过的窗口，共32窗、每窗一次预定采样。训练整段位置模型/不动比值Bridge1.403、LT1.388、BC-Z1.716、Fractal1.640；对应验证1.934/1.938/2.042/1.785。训练优于不动窗口分别1/8、3/8、2/8、1/8；LT首目标比值0.965例外保留，不泛称所有指标差。所有选窗实际呈现1至18次，绑定目标/mask逐值一致；独立CPU位置/旋转复算及权重不变核验通过，输入拼图已查看。总53.20秒、推理25.25秒、reserved2.15GiB，0更新/仿真/新集群提交。旧Bridge小窗拟合不代表当前混合来源模型已拟合；不能把当前问题仅归为纯过拟合。不同角色内容分布、单次随机续接与专家逐点误差限制均保留，不冒作完整训练集或任务成功率。后续先审阅已有去噪/完整生成证据以限定单一假设，不重复公式/冻结身份/旧拟合，不追加长训练或放行大评估。见[训练与验证动作生成诊断](OXE_TRAIN_VALIDATION_GENERATION_DIAGNOSIS_20261027.md)。

**额度恢复后接续已完成条件对照，未重跑（2026-10-27）。** 中断前32窗错配条件、1窗身份回放及独立复算均已保存，只有总结/台账未写入。本次补存审阅：条件替换改变动作但Bridge/LT/BC-Z正确条件平均误差未比错配更好，Fractal正确条件低约18%，不能说条件支路完全断开。已核对多尺度FiLM、编码器eval、100步余弦epsilon及前后权重摘要。既有训练曝光每窗平均12.68/10.73/1.37/4.76；BC-Z24,389起点未直接抽中但邻窗可共享帧，不推断同量任务从未见过或加预算必然有效。下一项只对实际抽中过训练窗做一次只读解码，与既有验证结果比较，不新增拟合/训练/集群任务。见`docs/OXE_CONDITIONING_SAMPLING_AUDIT_20261027.md`及原条件诊断目录。

**当前seed42四来源模型的固定源验证解码均未超过不动，学习/采样问题需要独立定位（2026-10-27）。** 每来源固定SHA元数据排名8个不同origin/content组、各一个中间窗口，32窗、1次预定DDPM100抽样，无动作/误差选窗。有效轨迹位置误差相对不动Bridge1.934、LT1.938、BC-Z2.042、Fractal1.785，分别1/8、1/8、1/8、0/8窗口更好；首目标比值3.697/1.864/1.648/2.082。三维来源旋转亦高于不动，LT仅XY有效，夹爪只按真实mask并对已知上一命令子集报告保持参照。这是当前65,536更新四来源模型的小样本开发验证，不是旧207558或整个验证集/正式任务结论。模型/编码器前后摘要精确不变，输入图已查看；独立CPU核对绑定标签/mask、原观测位置重建与float64位置/SciPy旋转复算通过。总65.16秒、推理26.61秒、reserved2.15GiB，0更新/仿真/正式LIBERO。仅改变LIBERO执行层不足以解释此源内误差；不能用验证噪声loss0.125当位姿优势。源码复核也记录训练夹爪用真实干净姿态、采样夹爪用生成姿态的条件差异，但未确立为位置误差根因。下一项限定训练去噪目标/条件注入/完整采样对应及必要的单一条件有效性对照，不新拟合、不改标签/预算、不挑seed或新样本覆盖结果。见[源验证解码结果](OXE_SOURCE_VALIDATION_DECODE_20261027.md)、`reports/oxe_source_validation_decode_20261027/attempt1/`。

**四来源期望控制目标可恢复性核对完成，不能直接整批换标签（2026-10-27）。** 四个精确缓存features SHA与既有证据一致。BC-Z已有同一行10槽位置/轴角残差和同槽夹爪目标的数值候选，复用旧33,320目标检查而不重复拟合；槽位不是固定未来行/秒数。Language Table有2D期望位置，但新增冻结训练顺序首/中/末3记录（12/12/42步、60内部动作行、3,756,778字节、约7.7秒）的两种相邻目标差与动作等式均不成立：最大分量残差pre0.0111至0.0334、post0.0208至0.1057，原记录单位、不冒报cm；作者实机异步配对及相机延迟说明提示需要精确导出phase，未据更小残差挑延迟修标签。新核对公开Bridge采集器存在上一desired目标累积、当前EEF旋转枢轴及限位/重置分支，缓存未暴露desired_state且精确历史配置/导出绑定未恢复；不能仅用当前实测pose+action冒充原目标。Fractal元数据明确平移为基座相对命令位移、旋转为RPY命令位移，不可当轴角，组合/尺度/时距仍待限定。本次0图像/模型/仿真/训练/新标签，未删来源或修改原协议；目标恢复新训练版本仍未准入。控制目标工程成功不等于四源标签问题已解决，完整源内固定留出窗解码诊断仍未执行。见[来源差异与证据](OXE_DESIRED_GOAL_FEASIBILITY_20261027.md)、`reports/oxe_desired_goal_feasibility_20261026/`。

**原生20Hz控制器期望位姿经8维转换的单工程案例验收通过（2026-10-26）。** 复用已排除正式评分的LIBERO90 demo_0，197条控制器目标从记录状态/原动作提取（0物理推进），再只恢复一次初态自由执行；最终attempt7在186tick原生双目标成功，夹爪已执行前缀与原动作一致、0饱和。实际控制器目标最大位置差1.83e-9米/朝向差3.09e-9弧度，原定1e-6米/1e-5弧度门槛不变且全部通过；EEF相对记录轨迹平均0.283mm/最大0.539mm，不能混称数学目标误差为物理精度。首次配置导入错误、新诊断绝对朝向框架混用、反馈/控制器缓存与sim.forward读取顺序、qpos位级不变断言被2.22e-16归一化触发的中间记录均保留；attempt3/4虽任务成功但严格目标未通过，未抹掉或改判。初态缓存偏差17.01mm；手部body与控制grip-site相差固定90度，后者在正确左乘相对旋转里可抵消，不将新诊断的框架混用归因于原主模型。最终以同状态新鲜控制反馈运行，3/4/7提取目标文件SHA一致；完整尝试和部分失败记录分别注明，所有尝试墙钟106.5秒，0模型/训练/正式40任务，未重做旧原生成功基线。先前20Hz到达位姿oracle也成功过，所以本次不能证明新目标普遍更优、5Hz足够或学习泛化改善。主训练标签、权重、原几何和执行源码未改，不恢复大实验；下一项仅核对各OXE来源能否构造来源有据的期望控制目标。见[工程验收及中间记录](LIBERO_CONTROLLER_GOAL_ROUNDTRIP_20261026.md)、`reports/libero_controller_goal_roundtrip_20261026/summary.json`。

**原邮件再次核对：OXE到LIBERO零样本有原文依据，控制适配边界未明确（2026-10-26）。** 只读第一/三/四/六/七封英文段落并保存文件SHA与段落定位：第一封主问题明确OXE训练后在LIBERO/CLIPort新物体/指令上评估，第三封另要求单一机器人到另一机器人及同机器人基线，第四封指定主四来源/版本和Language Table棍状末端差异，第七封DINO主骨干与13运行清单未撤销原目标。统一3+4+1格式没有同时定义实际到达轨迹、控制器期望设定点和跨机器人标定关系；也没有明示任何目标控制适配一律禁止。已把保留轨迹监督加独立跟踪层、重定义来源有据的控制目标、目标域策略训练三条路线的影响与边界写入现有审阅文档，形成给老师的三项具体澄清问题；未发送邮件、未擅自改成LIBERO监督训练。0模型/仿真/新作业，继续保留大实验暂停。见`reports/oxe_action_execution_semantics_20261026/teacher_scope_check.json`和`docs/OXE_ACTION_EXECUTION_SEMANTICS_REVIEW_20261026.md`末节。

**seed43模型及小报告已收齐并完成本地核验（2026-10-26）。** 报告压缩包6,649,519字节/SHA`54e1dab5...d5f7`，五个预期普通文件齐全，无额外路径。模型文件SHA与包内校验值相同，学习张量/合同与此前集群终点摘要匹配；完整65,536条更新计数、有限loss/梯度、训练索引、学习率与两次全验证记录核对通过，524,276窗次、145,812训练/17,949验证/0测试索引。跨Windows/Linux学习率重算仅已有定义舍入界内的末位差，未改训练或恢复守卫。复用原下载核验程序增加显式seed43端点参数，seed42默认路径保持，未重跑seed42或加载编码器。现在无增强seed42/43两份最终权重和记录均已本地验收；0前向/训练/物理步，不构成LIBERO泛化成绩，不恢复大实验。证据`reports/oxe_noaug_seed43_227981/download_verification.json`。

**seed43最终模型已下载且学习张量/合同核对通过，小报告包尚缺（2026-10-26）。** 本地`final_policy.pt`526,910,896字节，文件SHA`a190a3ed...fca6c`；321个学习张量摘要与用户集群导出`f032cf51...fdecf`逐字节一致，训练合同`243800bd...29f`、seed43/无增强/65,536更新/原协议核对通过。仅CPU读取权重和来源，无模型前向/训练/仿真。预期目录及本地默认Downloads未找到`oxe-noaug-s43-227981-reports.tar.gz`，完整更新历史和小报告独立核验仍待补，不能冒称整套下载已收齐。无需重新下载模型。证据`reports/oxe_noaug_seed43_227981/policy_download_verification.json`。

**seed43第二段达到65,536更新并通过导出检查（227981，2026-10-26用户日志）。** 17,949窗全验证loss0.1254391955，seed42同口径0.1254948328；两者接近只说明该扩散验证指标接近，不是LIBERO泛化或执行能力证明。第二段9831.37秒、reserved4.607GiB，最终策略526,910,896字节，学习张量SHA`f032cf51...fdecf`、训练合同SHA`243800bd...29f`，报告称全部学习张量保留、完成运行的完整优化器状态已退役。无增强条件已有seed42/43两个独立训练完成报告；seed43最终文件和完整日志尚未下载核验，未收到单独stage2 verifier或sacct完成输出，不混同本地逐字节验收。现有seed43运行已到终点，不再称其仍在训练；不重提，不自动启动seed44/其他条件或大评估。后续主线仍是动作目标与跨机器人执行语义验收，保留两份最终权重及既有首回合有效失败。证据`reports/oxe_noaug_seed43_stage2_227981_20261026.json`；台账和主计划已更新。

**四来源动作标签—LIBERO执行语义复核完成，主行为验收仍未成立（2026-10-26）。** 只读当前生成器/实际NominalOXEDataset消费者与已有版本溯源，确认Bridge/BC-Z/Fractal运动监督均为未来观测到达位姿，Language Table只监督实际XY；不是直接学习原生控制命令，也不是16个累加增量。目标除数0.1保留记录单位，正式训练明确采用原XYZ当米、源坐标直接迁移的未标定研究假设；名义5/5/10/3Hz不等于逐帧时间证明。LIBERO端将这些结果位姿当作期望世界目标通过OSC反馈追踪，单位/轴/TCP对应和接触行为保真仍未通过独立验收。该量义差异不证明未来位姿预测不可行，也不证明所有来源实际单位错误。复用旧Bridge推导确认常量右侧工具朝向偏置在当前左乘相对旋转中抵消，不再误报为新根因。五个实际核心源码与冻结评估快照逐SHA一致；0原始payload重读、0模型/仿真/训练/新作业，无标签/权重/执行参数修改。语义复核不冒充源验证窗解码结果；后者仍未运行。暂停新条件/种子及大评估状态保持，下一项先建立可执行动作合同及明确允许的跨机器人标定/控制适配边界，不追加低层参数试探或盲目把原生动作换作监督。见[四来源审阅](OXE_ACTION_EXECUTION_SEMANTICS_REVIEW_20261026.md)、`reports/oxe_action_execution_semantics_20261026/report.json`。

**固定5Hz信息的单次插值对照未修复失败，停止插值分支（2026-10-26）。** 同初态、相同5Hz目标行与精确原夹爪时间，线性平移/Slerp从实际区间起点生成四个20Hz子目标；实际控制200tick/18.218秒，碗在柜顶true、关抽屉false，位置轨迹平均误差由原对照11.13mm增到30.54mm。六项新增合成边界检查通过，目标行/时刻和实际夹爪序列逐项匹配；0学习模型/训练/正式40任务，原执行与编码源码不改，仅独立诊断原型。未试第二插值/增益/速度变体。只读复核OSC与标签：控制器期望设定点=当前实测位置+缩放原生动作，训练标签为未来实测到达位姿；该工程196行的推算设定点与下一帧到达位置平均差21.51mm（正常动力学下两者本来可不同），不能默认互为无损转换。现在优先审阅动作目标与部署控制语义及各来源接口，不因一次插值失败就改20Hz训练或扩13组。不声称该差异已解释所有泛化失败。见[结果与目标语义](LIBERO_INTERPOLATION_RESULT_20261026.md)、`reports/libero_oracle_interpolation_20261026/`。

**首项执行链oracle已得到实质问题，停止大实验（2026-10-26）。** 只在排除正式评分的原LIBERO90工程demo_0运行新8维目标路径，既有197原生动作成功参照直接复用。专家未来位姿经当前5Hz目标/执行器：200步失败；固定5Hz位姿、只还原原20Hz夹爪切换118/190步：仍失败（碗在柜顶true、抽屉关闭false）；夹爪时刻相同，仅用20Hz更密专家目标：第72步关抽屉、第190步双目标成功。原生控制器三条件均20Hz，不把目标刷新频率混称控制器频率。三个新条件共590tick/57.67秒、0学习模型/训练/正式40任务，几何与原评估执行源码未改，前116步位姿控制及共同精确夹爪时序逐元素对照通过。20Hz成功条件位置轨迹均误差21.18mm高于5Hz11.13mm，不能声称它总体跟踪更准；接触过程及目标时间信息重要。该结果限一工程案例，不证明正式首回合根因或泛化已修复；20Hzoracle有额外未来信息，不能直接把当前5Hz模型改频率上线。已按预声明两个追因对照停止，不继续增益/频率搜索，也先不做更多神经模型采样。下一项只区分同一5Hz信息下执行方式与缺失高频轨迹信息，先定单一对照。见[结果和边界](LIBERO_ORACLE_EXECUTION_FINDING_20261026.md)、`reports/libero_oracle_execution_20261026/summary.json`。大评估撤回状态不变。

**用户质疑后撤回大评估安排，转向主流程可行性诊断（2026-10-26）。** 用户明确100回合脚本尚未执行，没有待取消的新评估作业；老师DINO骨干选择与13条件清单保留，但Push-T提升、原生回放、合成几何检查和8决策连通不构成OXE→LIBERO完整行为验收。只读复核确认源单位/基座/TCP对应仍未验证，当前多源主checkpoint缺解码后的源验证行为评价；已有工程8目标的控制到期位置误差2.74–12.03mm，不等于预测误差或根因证据。首正式600步失败缺完整视频/逐步轨迹，无法定位失败环节。撤回新增100及新种子/条件长作业，本地续跑脚本改为明确拒绝提交并保留旧版SHA/源码历史；原评估程序、有效失败记录及227981不动。下一轮仅正确轨迹→当前8D/5Hz执行器、当前权重源验证解码、排除工程任务的可解释闭环三项定位，上限30GPU分钟，无重复Push-T/Bridge/原生回放或新拟合；未实际运行新模型/仿真。计划见[主链路诊断重排](LIBERO_MAIN_CHAIN_DIAGNOSTIC_RESET_20261026.md)，证据`reports/libero_main_chain_review_20261026/audit.json`。此前“下一段新增100”现已撤回。

**首回合账本验收通过且228153 COMPLETED于gpu-a40-1；准备同节点新增100（2026-10-26用户stdout）。** 原verify_output核对绑定45404d...fed、1/4000、overall null、未完成不能报最终率；Slurm明确COMPLETED与A40-1节点。原首槽有效失败保留。独立操作提交helper已准备：原账本验收、旧作业终止态保护、互斥与更新原job_id.txt，sbatch --nodelist=gpu-a40-1执行未修改evaluate.sh，每段默认100新回合，累计若完成为101/4000；只保持既有精确运行身份，不调整学习方法/执行/任务顺序/样本数/原绑定，也不再装包/跑预检。固定节点可能增加排队，当前148.38秒全超时样本粗算下一段rollout4.12h，其他任务与reset开销未知。尚未代提交或收到新作业ID，227981训练状态未新增证据。见`scripts/cluster/continue_libero_seed42_same_node_20261026.sh`与`reports/libero_evaluation_protocol_20261026/next_segment_preparation.json`。

**首个真实LIBERO评估回合228153完成：有效超时失败，1/4000已记录（2026-10-26用户终端）。** DINO无增强seed42在Spatial首任务“取盘子与ramekin之间黑碗放盘上”slot0运行600原生控制步未成功，148.38秒；segment185.337秒，绑定45404d...fed、completed1/planned4000/overall null。模型加载、原精确冻结守卫（沿未改入口推得）、实际EGL图像及策略闭环已执行，不再重复环境安装/预检；Gym维护提示和CLIPTextModel加载全CLIP包的视觉多余键未阻断运行，固定NumPy1.26.4不据通用提示升级。不能由0/1宣称40任务整体成功率为0，也不以接线成功当作泛化改善。尚待原ledger verifier及Slurm最终账目，已完成slot保留不重跑。单次全600步超时的粗算：100新增仅rollout4.12h；4000每模型164.87GPU小时（约6.87天），另有reset/load/排队且成功提前结束可缩短，其他任务耗时未测；13模型单GPU串行量不应误称几小时。下一项沿原协议先验收已写回合、再手动分段新增100（按固定任务顺序，非每任务抽样），不改变4000/600/DDPM/控制/权重，不自动提交。见`reports/libero_evaluation_protocol_20261026/first_episode_228153.json`和`first_runtime_cost_228153.json`。

**当前评估228153 Priority排队，seed43续训227981在gpu-a40-1运行54分22秒（2026-10-26用户squeue）。** 228153沿原submit_once.sh1在future安装后重提未完成首回合，尚未分配节点/EGL/回合结果；227981 Slurm R已观测，但未收到restored=32768及更新日志，不用运行时长替代完整恢复/实际步数证据。当前不再安装软件/重试login EGL探针或重复提交。日志评估`logs/libero-s42-228153.out`，训练预期`logs/oxe-dino-noaug-s43-p2-227981.out`。

**future0.18.2已安装；登录节点完整EGL导入探针位置不当，停止追加依赖修补（2026-10-26用户终端）。** 登录login-2在PyOpenGL EGL平台库为None时访问eglQueryString失败，没有提交新作业、没有模型/物理/回合。前述227964原GPU评估日志已经过robosuite/mujoco导入到bddl_base_domain第15行future错误处，说明与此次登录节点栈顺序不同；不能把登录节点缺EGL判作GPU渲染已失败，更不能据GPU导入越过就声称实际EGL上下文通过。承认CPU完整原生探针放错节点；不再在login安装驱动/更改渲染器/运行该helper。下一步直接沿现有submit_once.sh1在A40重试未完成首槽，把实际图形资格与这个计数回合一起完成，不另加GPU预检。227981训练不动，future修复不回退，4000/600tick/DDPM/权重和代码不改。见`reports/libero_native_dependency_repair_20261026/preparation_report.json`。

**评估227964原生仿真导入停于BDDL缺future，未生成回合（2026-10-26用户日志）。** 原评估源码检查通过，macros_private缺失只是警告；致命错误为bddl.backend_abc导入future.utils失败，0模型/物理/评分。先前父层审核覆盖实际主模型导入及仿真发行版元数据，没有覆盖完整OffScreenRenderEnv导入链，记录漏检边界；官方LIBERO原requirements及既有本地可用环境均为future0.18.2，BDDL安装元数据未声明该依赖。准备只在eval子环境补future0.18.2，再执行已有实际原生环境导入helper，成功才重提原首回合，不继续只检查metadata后排队。本地同版本BDDLBackend/parser及cloudpickle导入通过，未重做模型/GPU/工程物理检查。227981训练不改，原4000计数/权重/源码协议不改，集群修复尚未执行。见`scripts/cluster/repair_libero_future_and_verify_native_20261026.sh`和`reports/libero_native_dependency_repair_20261026/preparation_report.json`。

**seed43第二段线程对齐续训227981已由用户提交（2026-10-26）。** 原check-only明确seed43/stage2/restored_update32768/requested_end65536/fresh=false；用户脚本含同包提交锁、旧作业终止态保护及OMP/BLAS单线程等，但WRAP中段粘贴缺失，未收到实际interop设定/完整原合同恢复或下一更新日志。227920由此后续重提取代跟进，具体失败错误/Slurm终态未提供，不擅自认定新训练已正常恢复。总训练预算和独立seed数不变，提交尝试累计8次。当前跟进训练227981与评估227964，不再重复提交或改环境。见`reports/oxe_training_jobs_20261025.json`。

**父软件层接通及完整CPU主模型导入审核通过，首回合评估227964已重提（2026-10-26用户完整stdout）。** 依赖检查passed=true/errors=[]，Torch2.5.1+cu118、Transformers5.17.0及tokenizers/hub/safetensors均与工作父层版本/有效来源一致，子环境Pillow12.3版本一致、headless OpenCV4.6.0.66及全部仿真版本已记录；CUDA未初始化，0模型实例/前向/物理步。原评估源码/协议SHA校验通过，submission明确最多1个新增完整回合、总4000。尚无squeue或GPU/EGL/权重加载/目标成功日志，不能记成评估已通过或取得成绩。227920训练不重提，其实际恢复仍待日志。当前只看两个原作业输出，不再安装/重建环境或运行额外GPU预检。见`reports/libero_evaluation_protocol_20261026/job_227964.json`与`reports/libero_parent_overlay_repair_20261026/cluster_dependency_report_user_received.json`。

**headless OpenCV已安装并通过前置断言，评估仍因嵌套venv软件层缺失未重提（2026-10-26用户输出）。** opencv-python4.6.0.66卸载及同版本headless安装成功，cv2/torch/numpy导入、GUI NONE和版本等在依赖表前均已执行；随后metadata.version(transformers)报缺失，set-e未进入submit，不能归为OpenCV仍失败或新评估运行。确认Python venv使用sys._base_executable，本地两个无pip临时venv复现父overlay不自动继承，显式child .pth+site.addsitedir后父包可见且child优先，4项机制检查通过。新脚本只在eval venv绑定实际multitask父软件层，核对预检版本和torch/transformers/tokenizers等有效来源，fresh-process汇总实际CLIP/DINO类与主模型入口导入错误后才提交原首回合；不安装/复制新Transformer、无GPU/模型实例/物理步/重传资源。227920继续保留，未观察其实际恢复日志，不重复提交。原评估代码/4000协议及训练环境未修改。见`scripts/cluster/repair_libero_parent_overlay_20261026.py`和`reports/libero_parent_overlay_repair_20261026/preparation_report.json`。

**seed43修复续训227920已提交，评估重提因OpenCV图形库依赖停止（2026-10-26用户完整输出）。** 原训练与修复包校验通过，原check-only读到restored_update32768/fresh=false/终点65536，尚未看到GPU内原守卫及完整optimizer/RNG恢复，不冒称已续训成功。termcolor3.3.0安装成功后CPU导入继续在cv2缺libGL.so.1失败；set-e使后续评估submit未执行，不能记成新的评估作业。官方OpenCV说明服务器应采用无GUI轮子且cv2四种轮子不可同目录混装，已准备仅移除eval venv自有GUI轮子、安装同4.6.0.66 headless的脚本；保留基环境、训练源码与现有227920。登录节点只检查cv2、包版本与共享库解析元数据，原生EGL在实际GPU首回合内核验，仍计4000；无额外GPU预检或已完成回合丢弃。脚本内嵌Python语法通过，实际Linux修复未执行，不重跑上一整段训练提交命令。见`scripts/cluster/repair_libero_headless_opencv_20261026.sh`与`reports/libero_headless_dependency_repair_20261026/preparation_report.json`。

**227815与227817两项启动失败已定位并准备独立修复（2026-10-26用户日志）。** seed43第二段原守卫直接留证：无编码器/分词器/键序差异，仅两个posterior固定buffer分别15/14值不同，最大7.45e-9/1.19e-7；0前向/更新/断点写入，旧32768断点保留。新独立构造器入口只从已验证训练profile恢复原两buffer，仍要求原冻结总SHA逐位一致并由原restore_checkpoint验证完整合同/AdamW/采样/LR/RNG，不改已训练源码/manifest或预算；7项新合成续训边界检查通过，后续学习张量/优化器/RNG与不中断路径完全一致。LIBERO在robosuite导入logger时缺termcolor，原安装清单遗漏该隐式依赖；0模型/环境/物理/评分。只给独立eval venv补termcolor3.3.0，旧评估协议和239MB资源不改。52KB联合修复包就绪、无新作业代提交、实际Linux修复尚待执行。Python语法通过；本地MSYS bash因沙箱IPC启动失败，未取得bash -n结果，不误报脚本语法失败或通过。见`reports/oxe_resume_and_libero_repair_20261026/package_report.json`与`cluster_packages/oxe_resume_and_libero_dependency_repair_v1.zip`。

**seed43第二段227815与seed42首个LIBERO回合227817均已提交，Priority排队（2026-10-26用户squeue）。** 按前述提交顺序关联：前者期望从32768完整恢复至65536；后者期望首段最多1个完整回合、计入固定4000。尚无恢复/实际评分入口/首回合/EGL成功证据，不把队列出现当作环境验收或泛化表现。无新独立训练seed或额外训练预算，正式训练提交尝试累计6次、独立训练运行仍2个。等待作业运行，不重复提交或重做本地工程测试。证据`reports/oxe_training_jobs_20261025.json`及`reports/libero_evaluation_protocol_20261026/job_227817.json`。

**无增强seed43首段32,768更新完成（2026-10-26用户终端）。** 关联此前作业227512，stage1 passed=true，17,949窗全验证loss0.1318207913，协议SHA仍297e03...1be，reserved4.604GiB、10061.21秒（约2小时48分）。这是第二个独立训练seed的前半段，不是新增条件或LIBERO泛化成绩；不能与末训练单批loss直接比较判断过拟合。尚未收到原stage verifier/stdout、Slurm完成账目或断点字节；下一步先验收seed43首段，再沿既有入口43/2从32,768接到65,536，恢复优化器/调度/RNG，不重跑首段或改总预算。末尾^C出现在完成报告后，未提供被中断命令，不能据此认定训练失败。无新作业代提交，seed42评估包计划不变。证据`reports/oxe_noaug_seed43_stage1_227512_20261026.json`。

**用户明确选择每任务100次，LIBERO原指令评估协议与断点入口已冻结（2026-10-26）。** 四套件40任务，每模型4000次，13个计划模型共52000次；原生零动作另4000次只做一份共享，五种评估改写的预算未包含。保留先前候选及老师原文歧义记录，新协议SHA为`5156ea88...3381`。逐回合原子保存、内核互斥锁、检查点/源码/初态/运行时精确绑定；完成不重跑，技术错误留原槽，600原生控制步无成功记失败，未完成不报最终分母。12项新增合成评分/续跑边界检查通过，未重复旧32tick工程验证或训练。集群独立仿真环境与EGL还未实跑：准备约239.4MB的官方仿真源码/资源压缩包和0.694MB评估包，无演示、预训练资产或checkpoint重复打包，不重建17GB数据包。首次只提交一个完整固定回合计入4000，用实际耗时决定后续分段；不自动提交/链式作业，当前正式回合0。现有源单位/基座/TCP未校准事实保持不变，评分是公开近似约定下的零样本实验。见[实际命令与剩余项](LIBERO_FORMAL_EVALUATION_20261026.md)。

**LIBERO共同评估规则/40任务元数据候选完成，计数口径问题待用户选择（2026-10-26）。** 只读提取老师原邮件，明确第一/二封原文为100 episodes per task split，而非已明确的每个单任务100；不能沿中文摘要擅自消除歧义。官方固定evaluate.py支持Spatial/Object/Goal/10四套件，共40任务，作为本批主评估提案，不声称覆盖全部130；原90工程任务仍排除。40任务canonical语言、80个BDDL/初态原始文件SHA及4,000个候选episode槽/配对种子已固定成候选，未解码初态数值。600控制步/20Hz/5稳定步取自本地官方代码，沿既定执行和源到目标近似，分离模型输入与控制器EEF；原生目标成功、固定最终权重、不动参照复用、独立训练seed均值/样本SD、初态聚类统计及无重采样错误处理写入。100/任务为4,000/模型，100/套件为400/模型，已用异步文本问题提示10倍计算量差；selection=null、评分准入false，未启动模型/物理/评分，不把前者已批准或后者已满足论文规模。原指令与五种改写预算分开，后者未定。见`docs/LIBERO_EVALUATION_PROTOCOL_CANDIDATE_20261026.md`及`reports/libero_evaluation_protocol_20261026/preparation_report.json`。

**跨环境加载差异逐项确定并恢复训练常量；真实学习策略完成首个LIBERO工程执行（2026-10-26）。** 下载既有226567 preflight.json后比对同根摘要的缓存实际本地profile：键序/全部编码器及639个冻结张量一致，仅posterior_mean_x0的15值和posterior_mean_xt的14值末位不同，最大差7.45e-9/1.19e-7，其余4个扩散常量完全一致。新显式frozen_profile_path仅允许从同训练合同/配置/源码身份/分词器/逐张量SHA的清单恢复已知root固定buffer，拒绝编码器/学习参数/形状/次序/其他配置变化，失败回滚；最终仍用原frozen_digest要求精确等于训练`91175b...1999`，没有容差或新期望值。微型正/负/回滚夹具通过，0优化；源模型/Adapter/扩散实现及训练/权重文件未改，仅推理加载与新常量辅助入口变更。真实seed42完整权重在native环境恢复两buffer后，冻结和分词器严格通过，沿既定工程约定在已排除正式评分的LIBERO90场景运行8次策略决策/32个20Hz物理步（1.6秒），两帧已查看、0饱和tick、无错误，命令特征先未知再显式已知。没有训练、读取专家动作/最终init、挑checkpoint或任务评分，不证明成功率或跨机器人坐标/TCP校准。原attempt1加载失败保留，新attempt2约30.49秒通过；不重训seed42或改集群seed43运行。证据`reports/oxe_noaug_seed42_227223/frozen_runtime_comparison.json`、`frozen_portability_checks.json`与`reports/oxe_seed42_libero_engineering_20261026/attempt2/report.json`。下一项是事先固定正式LIBERO任务/初始状态/统计协议后评价，继续增强语料与其余条件；正式评分准入false保持。

**seed42最终模型与小报告下载核验通过，实际本地策略加载停在跨运行时冻结身份检查（2026-10-26）。** 文件526,910,896字节/SHA`33df7e...16d3`与集群校验一致；321个学习张量共131,521,928值、学习SHA`2295d8...66d`和合同SHA`b3b59a...fc`逐字节匹配，65,536条更新计数/有限loss/训练索引/LR和两次全验证检查通过，实际524,276窗次。Windows重算cosine有2,650个LR值末位差，最大2.71e-20，按4机器eps乘baseLR的绝对舍入界记录，训练/恢复/冻结守卫未改；首次新审核脚本过严逐位LR比较失败亦保留，不当作训练错误。核心模型/几何四文件SHA与训练一致，test未评价，2更新预检与正式权重角色区分通过。随后首次用完整真实学习策略进入既定LIBERO工程入口，原冻结检查报本地`7cd855...82ea`、训练`91175b...1999`，分词器一致；本机Torch2.11/cu128与集群2.5.1/cu118不同，但具体差异尚须逐名清单证明，不能据版本就认定原因。0策略前向/0物理步/0训练/0任务评分；不绕过身份检查、不修改模型/预算、不重复训练。沿用已经保存且根摘要同为`7cd855...82ea`的本地逐张量清单，下一项只取回既有226567的preflight.json，用新CPU比较脚本定位，不申请新作业或再加载编码器。下载证据`reports/oxe_noaug_seed42_227223/download_verification.json`，首次加载失败`reports/oxe_seed42_libero_engineering_20261026/attempt1/report.json`。

**无增强独立seed43首段227512已提交，Priority排队（2026-10-26用户终端）。** squeue PD/START_TIME N/A，尚未观察fresh/restored=0、初始化或更新。第一段32,768、总65,536，同冻结数据/结构/预算，独立seed43不是seed42重训；两条件/seed运行已提交（均属无增强条件），含失败/分段共5次提交尝试，seed42已有完整训练终点报告，另11运行未提交。用户未贴最终seed42验收stdout或本地下载完成，仍不冒称其文件已收齐或LIBERO已评估；下一项继续收seed42模型/小报告并接实际策略执行。日志`logs/oxe-dino-noaug-s43-p1-227512.out`，台账`reports/oxe_training_jobs_20261025.json`。

**首个完整OXE训练seed42已达65,536并导出最终策略（227223，2026-10-26用户终端）。** 第二段stage2 passed=true，17,949窗全验证loss0.1254948328（第一段0.1300912872），末训练批loss0.1220554；reserved4.627GiB、第二段10218.38秒。两段合计20434.80秒，约5小时40分，不含排队。导出final_policy.pt为526,910,896字节，学习张量SHA`2295d8...66d`、训练合同SHA`b3b59a...fc`，导出阶段核验通过后移除该完成运行1,596,160,066字节优化器断点，所有学习张量与来源合同保留。seed42不得再以训练提交；这是13个独立条件/seed中的首个训练完成报告，不是首个LIBERO成绩。尚未收到原stage2 verifier输出、下载最终文件或实际LIBERO策略执行/评分；不把扩散验证loss当cm或成功率，不宣称源到目标单位/TCP校准已成立。下一项原终点验收/小报告和模型收取，实际学习策略的LIBERO工程执行检查，并行推进计划内独立seed43及尚未就绪的增强语料；不重复seed42拟合/预算或旧Push-T诊断。证据`reports/oxe_noaug_seed42_stage2_227223_20261026.json`。

**用户报告第二段作业227223，实际恢复/执行状态待日志（2026-10-26）。** 沿前述首段验收及第二段提交命令收到编号，仍为DINO无增强seed42同一独立运行，期望从32,768完整恢复到65,536，不增加预算。尚未收到首段验收stdout、Slurm排队/运行状态或第二段restored=32768实证，不推断已经恢复或更新。第一段已观察32,768保留；总计4个正式提交尝试属于同一条件/seed（含两次更新前失败），其余12运行未提交。日志`logs/oxe-dino-noaug-s42-p2-227223.out`，台账`reports/oxe_training_jobs_20261025.json`。

**226619第一段32,768更新及17,949窗全验证完成，待原验收后接续第二段（2026-10-26）。** 用户终端给出stage1 passed=true、完整预算65,536、协议SHA仍`297e03...1be`、reserved4.617GiB、总10216.42秒（约2小时50分）；全验证loss0.1300912872，末训练批loss0.0465243，不能用一个训练批与全验证均值比较来认定过拟合或泛化进步。尚未收到原verify_stage.py输出、Slurm完成账目或本地检查点字节，未提交第二段；下一项原验收通过后`submit_stage_once.sh 42 2`恢复同一优化器/RNG/LR进度到65,536，不重训首段、不加预算、不重做初始化诊断，尚无LIBERO成绩。证据`reports/oxe_noaug_seed42_stage1_226619_20261026.json`。

**第二轮本地旧排查资产清理完成，额外净释放8.488GB（2026-10-26）。** 用户明确授权后续不用的旧排查/CLIP调整包退役；删除14个已结束且experiment_kind=training_fit_diagnostic的早期BC-Z小样本拟合best.pt（8.073GB），先保存非权重配置/历史/切分及参数形状元数据，原JSON指标/样本清单不动。这14份学习权重已退役，不能当作仍可直接恢复的断点。另退役61个旧递送ZIP和46个独立checksum、7个旧临时包装目录，全部成员内容通过去重blob快照及19个字节SHA一致的保留原始副本留存；保留13个含唯一过滤数据的806数据包。清理344.79MB pip/Numba可再生缓存，安装环境保留。130项共删除8.640GB，扣151.224MB留存快照后净释放8.488GB，D盘现约70.231GB可用。全部目标路径/大小/mtime/父级与无reparse核验后用PowerShell原生精确删除；当前OXE/DINO包、CLIP文本资产、DINO/Qwen、原始数据、基线/泛化/融合模型、所有指标预测和发布Git快照保留，未联系集群。执行和保留项复核通过，记录`reports/local_storage_cleanup_extended_20261026/receipt.json`、`verification.json`，旧包内容在同目录`retired_package_contents.zip`及索引；原始ZIP容器已退役，留存的是逐成员内容而非原ZIP二进制。

**按用户要求清理本地D盘18.055GB，保留源数据与实验记录（2026-10-26）。** 删除已上传且集群已解包使用的`cluster_packages/oxe_nominal_pool_v1_20261025.tar`17,068,789,760字节；确认128原分片仍存在/大小匹配、tar条目及内嵌清单一致，保留原始SHA记录/候选清单/重建脚本，未再读取17GB重哈希。两份工程2更新checkpoint全嵌套字段/张量比较完全一致，仅删resumed副本986,070,735字节，保留后续脚本引用的full/latest_step.pt及全部JSON报告。另删两个新合成诊断测试夹具目录约0.48MB；未清空其他.tmp、旧唯一权重、原始数据、环境或骨干。PowerShell逐路径/大小/目录内容/无reparse验证后精确清理，D盘空闲由43.695GB升至61.751GB；没有联系集群，不影响226619训练。清单与执行回执`reports/local_storage_cleanup_20261026/plan.json`、`receipt.json`。上传tar现在本地不存在，后续仅在确需重新上传时从原分片重建，不自动重复打包。

**226619已在直接正式入口通过原身份检查，实际训练至少2,048更新（2026-10-26）。** 用户日志中冻结摘要与绑定均为`91175b...1999`，frozen/tokenizer均true；从restored=0进入U65,536、第一段stop32,768循环，256至2048八个训练loss有限，2048时段内632.2秒/loss0.224545。当前只确认训练进展，不推断实时Slurm状态、节点、checkpoint写完、完整首段或验证结果。留证代码已实际执行，但完整安装回执未收到；旧两次身份失败根因仍未复现，不将日志补丁称作根因修复。第三次提交尝试仍属同一条件/seed运行，其余12运行未提交。继续当前作业到原首段及全验证，不再重提或追加同类诊断；尚无本组泛化或LIBERO结论。台账`reports/oxe_training_jobs_20261025.json`，日志`logs/oxe-dino-noaug-s42-p1-226619.out`。

**226603直接启动再次在原身份守卫失败，撤回原包直接重试就绪判断（2026-10-25）。** 用户收到相同未捕获ValueError，循环前0更新；具体sacct状态/节点/耗时未提供，不推断。226586使用runpy与trace的包装诊断通过，未证明直接启动也能通过，根因仍未知；不再原样重试或运行包装诊断。约16KB独立补丁只在直接正式runner打印原比较结果、在原拒绝分支逐名留证，校验条件/digest/异常保持，权重/数据/预算/预检绑定不改，异常留证失败也不能绕过原拒绝。已知base SHA限定安装、无活动目标作业/非空训练输出保护、原文件小备份与新manifest；原部署zip保留，尚未上传/安装/提交。新增日志分类/四种匹配拒错等价/失败保护夹具及包完整性和Bash语法通过，0真实预训练加载/前向/优化。下次直接正式启动若通过则按原计划训练，若失败即输出`runs/oxe-id-failure-JOBID/report.json`并停止，取得真实失败条件。见`reports/frozen_identity_failure_20261025/direct_logging_patch_checks.json`。

**用户已原包重提正式seed42第一段226603，Priority排队（2026-10-25）。** 用户终端确认所有原包SHA通过、复用不可变预检绑定226133，check-only显示restored_update=0/fresh=true、第一段32,768、总65,536；sbatch成功，squeue PD/START_TIME N/A。尚未开始运行或观察新更新；保留旧226378更新前失败和226586现场身份匹配证据，未修改守卫/权重/预算。两个提交尝试仍属同一个独立条件/seed运行，另12运行不变；没有新泛化/LIBERO结果。日志`logs/oxe-dino-noaug-s42-p1-226603.out`，输出仍为`runs/oxe-dino-noaug-s42`，后续先核实真实训练进展，再按原协议验收首段。

**226586原正式CUDA身份守卫实际匹配，可按原包重提首段（2026-10-25）。** 用户现场报告guard_captured=true、would_fail_original_guard=false；A40/torch2.5.1+cu118/CUDA11.8下真实冻结摘要与绑定均为`91175b...1999`，逐名张量、键序、调度常量、配置及分词器均无差异；CPU能力AVX2，与CPU诊断AVX512不同仍完全相同。0前向/优化/训练checkpoint；旧226378故障没有复现，根因未确认，不声称修复了某项模型逻辑。停止继续同类初始化诊断，保留原包与全部守卫，以现有`submit_stage_once.sh 42 1 retry`重提原seed42第1段32,768更新，完整65,536预算不变。尚未收到重提编号，旧失败历史保留，其余12运行状态不变，未产生泛化或LIBERO结论。证据`reports/frozen_identity_failure_20261025/cluster_runtime_result_226586.json`。

**用户已提交实际正式守卫诊断226586，当前Priority排队（2026-10-25）。** 用户squeue显示PD、START_TIME N/A；尚未初始化或取得比较结果。短A40诊断沿原正式入口在身份守卫前无条件停止，计划0前向/优化；不是正式训练重提。等待`runs/oxe-id-runtime-226586/report.json`，不重复CPU重建或训练。台账`reports/frozen_identity_failure_20261025/cluster_runtime_submission_226586.json`。

**226567实际CPU报告全部匹配，原正式失败尚未复现（2026-10-25）。** 用户报告两套配置重建摘要均为`91175b...1999`，逐名冻结张量、键序、6个调度缓冲区、配置与分词器全部相同，CPU为AVX512。不能据此宣布226378根因已修复，也没有理由重绑摘要/降低检查。新约8.6KB运行时诊断沿未修改的正式run_stage.py seed42/stage1，在原合并身份守卫求值前记录真实局部变量和逐名摘要并无条件停止；申请短A40，不做forward/优化，不写正式输出。匹配/不匹配都停止及意外forward阻断夹具、包完整性与脚本语法通过。未提交新作业，原训练包/预算不改；下一项获取实际CUDA正式入口的失败条件，而非再做CPU重建或重复训练。证据`reports/frozen_identity_failure_20261025/cluster_cpu_result_226567.json`及`runtime_diagnostic_bundle.json`。

**用户已提交CPU冻结状态诊断226567（2026-10-25）。** 去除零GPU GRES声明后的诊断收到真实编号；尚未观察排队/执行/完成，不视为通过。预定只加载并逐名比较冻结张量、调度缓冲区和分词器，0前向/训练，不申请GPU。等待`runs/oxe-id-cpu-226567/report.json`，正式226378仍为更新前失败，未重提、未修改身份守卫。台账`reports/frozen_identity_failure_20261025/cluster_cpu_submission_226567.json`。

**CPU诊断首次提交被Slurm拒绝，未创建作业（2026-10-25）。** `--gres=gpu:0`被集群新规则判为未指定型号的GPU请求，故没有job_id或报告，不是模型诊断再次失败。仅移除此GRES声明，保持4CPU/12GB/10分钟，原模型/守卫与诊断逻辑不变；同步脚本SHA和ZIP，提供只接受已知旧脚本身份的集群原地修复，不重传数据、不申请GPU。尚未收到修复后作业编号。

**226378分词器元数据排除，冻结状态差异待逐项定位（2026-10-25）。** 用户实际集群诊断确认预检226133的绑定与fresh tokenizer完全相同；显式encode之后rules摘要随临时padding/truncation改变，不能据此判定本次正式初始化错误。现有合并守卫未给出实际失败摘要，根因仍未确认。已准备约5KB独立CPU诊断包，对预检/正式两套配置在独立进程重建并按名字核对所有冻结张量，单列6个扩散调度缓冲区和键顺序；若CPU不能复现原绑定会明确报告，不自动重绑/修复。包完整性、Python语法与比较器分类夹具通过；没有GPU申请、前向、优化或新集群提交，原训练包/守卫未改。见`reports/frozen_identity_failure_20261025/cpu_diagnostic_bundle.json`。

**226378首段启动失败，训练尚未更新（2026-10-25）。** 用户日志确认A40-1上95秒、FAILED/1:0，停止于循环前的冻结状态/分词器合并检查。没有新优化更新或泛化结果；保留原身份守卫，不盲目重绑或绕过。CPU两进程加载实际资产的冻结张量和键序一致；分词器原始backend SHA在编码后会因运行时padding/truncation状态变化，但尚无集群逐项差异，不能确定其为本次根因。下一项仅取只读分词器/配置元数据，不重跑训练或使用GPU。

**首个正式离线训练已由用户提交：226378（2026-10-25）。** DINO无增强seed42第1段，终点32,768、完整预算65,536；这是13个独立训练运行中的第1个，不与2更新短预检混计。仅收到真实作业编号，排队/运行/更新数均未观察，不标记已运行或完成；LIBERO结果与泛化结论仍为空。固定日志`logs/oxe-dino-noaug-s42-p1-226378.out`，权重目录`runs/oxe-dino-noaug-s42`，作业台账`reports/oxe_training_jobs_20261025.json`。

**用户确认A40集群预检通过，首组正式离线训练包准备完成（2026-10-25）。** 用户校验输出为torch2.5.1+cu118/CUDA11.8、峰值reserved4.572GiB、总56.35秒，只有2工程更新+1验证批；不外推单步速度或泛化。共同预算预先固定65,536更新、batch8，分32,768/65,536两段，共用cosine时域/warmup1024，三个种子独立fresh；原预检学习权重不迁移。新独立训练协议披露有界池/名义时间、源XYZ米制解释与未完全核验的单位/基座/TCP，不篡改原数据物理false标记，训练与LIBERO评分角色分开。约138KB新包复用已上传数据，提交时绑定真实预检元数据和冻结身份，先只提交无增强seed42第1段。新协议拒错/微型合成导出及包完整性/语法检查通过，0新优化/模型/拟合/资源重复。活动训练保留完整latest，完成后核验导出全部权重和记录，再仅清理该已完成运行的优化器断点以控制200GB；旧记录不动。尚未收到正式编号，13次正式任务仍未提交。见[首组预算、假设及执行安排](OXE_FIRST_FORMAL_TRAINING_20261025.md)。

**冻结validation、推理时间/检查点绑定与首组短集群包完成（2026-10-25）。** 相同资格规则筛得1,136条独立源验证演示、17,949窗，省去47个BC-Z零有效尾窗，原train145,812窗/角色/范围候选8不变；未解释test/隔离数值。新增分角色store与组合入口，四源8个真实验证图片/目标、collator、角色交换拒绝及训练目录重绑定通过，原训练文本/组条目不变。推理现在拒绝同16目标但0.1秒执行的错配，识别主固定更新检查点并校验学习模块、冻结编码器/tokenizer；此前工程权重仅作一次回调，0优化/物理步/评分。执行候选明确单位/基座/TCP尚未标定。已生成92MB代码/数值zip与17.07GB/128分片流式tar（不解析记录）；本地包hash/CRC/目录/Python编译/Bash语法及真实128文件check-only通过，无上传/集群接触/作业。先上传小包查实时空间，再传raw包并提交一次A40两更新+一个验证批的短预检，不把工程权重或验证接线loss算正式实验/泛化结果。见[验证、执行与可提交包](NOMINAL_VALIDATION_AND_CLUSTER_PREFLIGHT_20261025.md)。

**名义时间标签主入口、真实恢复与完整结构本地资源检查完成（2026-10-25）。** 新显式读取器直接定位原分片记录、每次实际读取验证payload SHA后解码图片/原指令，使用新目标/因果输入/mask；原raw-row路径保留。145,812窗prepare清单明确仅train，不随机制造validation。真实四源8窗通过；实际预训练DINO/CLIP文本、缩小Adapter和完整multiscale后端，连续2与1+1恢复的全部检查点字段一致，4次有界更新。首次compact夹具严格确定性遇到CUDA linear上采样反传不支持而止于0更新，保留原因，转用既定multiscale，未改旧结构/守卫。另以完整8层、attention512/8头、batch8、100步epsilon模型测2个真实批次，本地峰值reserved4.381GiB、131,521,928可训练参数，冻结编码器未变；资源权重不保存、不计正式种子，不把本地耗时当集群估计。合计6次接线/资源更新，0物理步/源留出数值/新集群作业/泛化判定。下一项验证标签、执行映射与共同预算，再准备首组无增强集群预检。见[实际入口与证据边界](NOMINAL_MAIN_ENTRY_20261025.md)。

**用户同意名义频率近似，真实训练侧统一5Hz标签完成（2026-10-25）。** Bridge/LT5Hz、BC-Z10Hz、Fractal3Hz按明确假设映射到0.2至3.2秒16目标，未伪称实测时间戳。8,918条已合格演示、582,766源观测行生成145,812窗；仅省去475个不足0.2秒的BC-Z尾部窗口。位置插值、四元数最短路径插值、二值夹爪保持、当前输入因果性和LT缺失维度/尾部屏蔽检查通过；角色和旧标签未变。数值缓存可复用，标签/缓存/原文本约197MB，新增原指令目录39MB，145,812窗与3,686种规范化指令通过现有无增强类绑定。独立SciPy对照的近重合线性回退差约0.000057度，容差和首次过严阈值失败明确留档。新训练统计最大归一化位置7.660、算术候选界限8，旧raw-row最大9.596/候选10保留历史；默认3不变，物理尺度尚未固定。0模型/优化/物理步/新作业，未读validation/test数值；尚未切换正式主Dataset，后续接入和实际完整结构短预检不重复旧拟合。见[时间决定、真实标签及执行接线](LIBERO_TIMING_DECISION_AND_FEEDBACK_20261025.md)。

**有界反馈执行选项已完成（2026-10-25）。** 几何转换/执行合同新增显式bounded_feedback模式，目标位姿保持，原生每tick命令限幅并用测量反馈重算；原reject默认保留。实际安装的OSC goal代码与合成反馈检查通过，10合成tick、0物理步/模型/评分，不重复旧回放；该接线不证明真实任务表现。

**128分片身份与冻结train数值审核完成（2026-10-25），没有正式训练。** 15.896GiB候选的12931记录CRC/计数/长度通过，建立12814源内容组、198旧角色锚点；两条LT候选连接旧train/test，隔离而不移动旧角色。共同候选角色train10340/val1291/test1298/隔离2。冻结train侧8918记录、146287窗口通过数值规则，实际生产解析器首/尾17836探针，无像素解码；空指令、最少10行规则和9条Bridge未解析夹爪命令的排除明确，不按误差或大动作删样本。新增训练指令JSONL供复用，验证/源留出/隔离数值未读。发现旧±3位置输出界限不覆盖全部合格train标签（最大9.596），已改为显式参数、默认保持3，候选10待统一物理尺度后定；小型采样夹具通过。3D未知当前夹爪15155窗的反事实规则仍需确定。身份扫描131秒、train数值126秒，0模型训练/物理步/集群作业；单位、时距、真实执行与正式配置尚未验收。见[计数、准入范围和下一项](OXE_BOUNDED_POOL_QUALIFICATION_20261025.md)。

**无增强入口独立、完整数据可用性修复与200GB候选方案完成（2026-10-25），未正式训练。** 原始训练指令目录可单独绑定no_augmentation，不再等待已审阅增强库；35,933工程窗口身份、原文本/目标/mask与RNG检查通过，新合成3步/1+2恢复和disabled对照完全一致，仅9次微型合成优化，未重跑52路径旧套件。完整四源train文件约665.84GiB已在D/E盘：发现LT六个等长但MD5不符的gstmp，固定官方对象generation补下约2.36GiB并逐个校验，保留原临时文件，现在3584个train分片文件名齐。用户没有额外存储，故准备每源4GiB、共15.896GiB/128分片候选清单，元数据预计12931演示；不看模型误差选取，不称全量训练，尚未复制/记录资格/分区。按旧项目用量提出24GB检查点日志+6GB余量与至多两作业并行，非实时quota验收。动作单位/时间/执行缺口仍未关闭，无集群作业、真实模型训练或物理步骤。见[证据、容量和后续顺序](OXE_STORAGE_AND_ORIGINAL_CONTROL_20261025.md)。

**DINO已接入实际主模型和LIBERO策略回调（2026-10-25），未正式训练。** 新视觉分支固定DINOv2 ViT-L/14本地资产身份，文本保留冻结CLIP；主Dataset与回调共用原图像几何，DINO内部统一转换通道归一化。CLI/配置、固定更新代码绑定和检查点加载入口均已接通，旧CLIP默认行为保留。真实预训练编码器加缩小下游随机工程模型的12项新增路径检查通过，13.55秒/3.350GiB；冻结无梯度、主loss反传有效、主检查点重载与回调/直接采样逐元素一致。未优化、无物理步/目标数据/集群作业，不声称正式八层资源或LIBERO性能通过；OXE索引prepare-only限制保留。下一项是正式训练池与动作/时间/执行合同，增强语料并行。见[接入及剩余前置项](MAIN_DINO_INTEGRATION_20261025.md)。

**第七封邮件确认DINO主方案，计划改为13次LIBERO训练（2026-10-25）。** 冻结DINOv2 ViT-L/14四种增强条件各三种子，另加冻结CLIP完整增强一个种子；Push-T三方对照作为论文独立诊断。两周内目标为启动全部五种条件，非必须全部跑完。骨干意见不再是等待项；先固定共同数据/动作/预算/评估协议，打通DINO实际主入口与LIBERO执行，尽早启动无增强，并行完成同义与反事实语料，随后提交其余条件。当前未提交正式LIBERO训练，不将工程缓存或短检查当正式结果。见[顺序、13次清单与两周目标](LIBERO_NEXT_STEPS_TEACHER7_20261025.md)。

**本地语言生成短预检完成，未扩展为正式标签（2026-10-25）。** 固定官方Qwen2.5-1.5B-Instruct版本及文件身份；四来源各两条冻结训练指令，全程本地离线。首次8组格式失败保留；补示例并改贪心后7/8格式通过，但40个输出的Codex文本审阅仅22个可保留、8个拒绝、5个原文含糊待来源核对、5个重复组拒绝，仅2组凑齐5条。第二次29.28秒/3.107GiB，未全量生成、未创建可训练语言库、未修改机器人模型或新增训练/集群作业。12项解析与隔离检查通过。该结果只说明生成与审阅链路能运行，不是泛化改善；避免错误方向/弱化目标/含糊改写进入后续标签。见[证据、失败样例与下一项](LOCAL_LANGUAGE_PILOT_20261025.md)。

**等待老师回信期间，完成LIBERO主线实际OXE窗口接入和训练指令目录（2026-10-25）。** 新`oxe_window_index`与Dataset/主prepare-only入口显式接通，默认旧行为保留，来自原32分片工程缓存的45,049窗口按旧整演示角色固定：训练35,933/验证5,021/源留出4,095；排除940条过短/空指令记录，组与角色不重分配。20项接线/拒绝检查通过，seed44与42分区相同，四来源8个真实训练首尾窗目标/mask/payload读取一致，LT仅XY、补齐零监督。源单位/时间/TCP未知保持null，当前真实训练在模型初始化前拒绝，不冒称checkpoint/执行合同已放行。另从3,363条训练记录提取覆盖全部训练窗口的2,447条原始指令，payload SHA匹配；12.64秒、无验证/留出/LIBERO指令读取、无图片解码或动作构造。生成本地LLM五条改写的12,235候选请求，未实际生成/审阅，目录不可当已审阅语言库。至少1,275个三维起始窗口无先前夹爪命令，反事实保持标签不可伪造，列为协议缺口不改增强算法。本轮本地CPU、0模型训练/物理步/新集群作业，DINO结果与冻结CLIP主模型不变。见[新成果、计数与正式缺口](LIBERO_PREPARATION_WINDOWS_LANGUAGE_20261025.md)。

**DINO222758小输出与固定视频收齐，本地核对通过（2026-10-25）。** 九文件992KB包，第二段42084更新逐条counter/LR/有限loss、原初始化/数据/样本/归一化/完整恢复证据、骨干绑定、侧车元数据、只读EMA/worker与三向奖励/配对区间一致；大权重SHA仍依据集群原验收，不冒称本地加载。DINO0.829858/CLIP0.445488/原生0.845878，满7/5/10场。实际A40-2，第二段训练14.34h、含闭环14.54h、策略峰值7.572GiB。固定视频5000000 DINO最大0.408/末步0.127未完成，5000001最大/末步1；六段旧新固定视频可解码并已查看首/中/末采样帧，保留成功与失败。奖励图/逐场景增益及视频采样图生成并目视检查。当前足以汇报骨干受控消融的明显开发收益，仍是单seed/复用dev20，主CLIP与final50保留，不自动扩预算或继续局部结构试验。见[最终验收与后续边界](PUSHT_DINO_222758_RESULT_20261025.md)。

**DINO222758原完整输出验证通过（2026-10-25，用户终端）。** verify_dino返回segment_verified=true/segment2/84168，完整checkpoint/侧车/更新和LR记录、DINO缓存骨干绑定、初始恢复与第二段完整恢复、冻结评估和三向配对比较均通过原验收。均值DINO0.829858/CLIP0.445488/原生0.845878、满奖励7/5/10场，增益+0.384369、场景区间[0.200636,0.575308]保持。仅一个训练seed及复用dev20，不自动扩预算、开变体或访问final50；原CLIP主模型保留。下一项接收小结果包/两段固定视频并核对归档，大权重继续留集群。见[验收记录](../reports/pusht_dino_training_submission_222758.json)。

**DINO222758达到84168并在dev20明显改善（2026-10-25，用户终端，完整验收待收）。** 平均最大重叠奖励DINO0.829858、原CLIP0.445488、原生0.845878；满奖励7/20、5/20、10/20。粘贴的20分数与本地固定参考独立复算，DINO减CLIP+0.384369，配对场景95%区间[0.200636,0.575308]，12好/2差/6平；平均原生差距约缩小96%，不是等效证明。强支持冻结视觉表示/后续前端匹配为重要瓶颈候选，原8层Adapter仍可承载较高闭环表现；不能认定CLIP唯一根因或Adapter单独收益。既定能力门槛仍false因满奖励35%<50%，均值已过0.8。单训练seed/复用dev20/旧A6000新A40/归一化及最终LN差异公开。原日志完成全部训练与20闭环，但完整verify_dino、报告原字节/视频/Slurm账目尚未接收；没有新训练或复跑，final50与CLIP主模型保留。下一项仅原输出核验及九个小文件归档。见[数字、解释边界和命令](PUSHT_DINO_222758_TERMINAL_RESULT_20261025.md)。

**DINO第一段完整核验通过，第二段222758已提交（2026-10-24，用户终端）。** 原verify_dino确认221186第一段42084更新、完整checkpoint/侧车/更新与LR记录/首步磁盘恢复及DINO身份通过；新包SHA、缓存绑定与18GiB空间余量检查通过，项目占用152091529216字节。第二段从已核验latest接续至84168，申请A40/40GiB/20h；尚未核实调度、实际完整状态恢复、最终完成或dev20评价，暂无DINO泛化结论。不重训第一段、不改冻结骨干/预算，原CLIP主模型和final50保留。日志`logs/pusht-dino-part2-222758.out`，输出`runs/pusht-dino-part2-222758`。见[第二段记录](../reports/pusht_dino_training_submission_222758.json)。

**DINO第一段221186终端完成42084更新（2026-10-24，用户日志）。** 日志显示42000步50610.4秒、随后保存42084完整latest并输出segment1 complete，约14.1小时，无开发评分或新泛化成绩；Slurm账目和verify_dino完整核验输出尚未收到。接续使用既有submit_part2_once.sh，先检查第一段报告/权重SHA/全部更新与LR/恢复证据及缓存绑定/空间，再提交第二段至84168；不重训第一段、不重置优化器/EMA/RNG或缩短LR。原CLIP模型保留，final50封存。见[221186记录](../reports/pusht_dino_training_submission_221186.json)。

**DINO正式第一段221186已由用户提交（2026-10-23）。** 新包逐文件SHA通过、221055缓存身份绑定通过；提交时项目占用147203981824字节，保守200GB预算内18GiB余量检查通过。申请A40/4CPU/40GiB/20小时，目标42084有效更新，完整总预算84168，原90演示/11356帧/10726窗、八层Adapter及原作者训练核心保留。尚未验证调度运行、实际首步loss/梯度/完整磁盘恢复或完成；第一段无开发评分，第二段未提交，原CLIP主模型与final50不动，无新泛化结论。日志`logs/pusht-dino-part1-221186.out`，输出`runs/pusht-dino-part1-221186`。见[提交记录](../reports/pusht_dino_training_submission_221186.json)。

**DINO同预算正式训练包就绪（2026-10-23），未提交。** 221055六份小输出收齐，本地核对cache/backbone/文本/数据协议及保存动作逐值一致，完整缓存留集群；完整390.26秒。新34文件包复用13个原模块，原8层Adapter/作者U-Net/归一化/样本/RNG/loss/Adam/EMA/LR及84168有效更新不变；新冻结DINO缓存替换视觉表示。42/43种子新建，分42084/84168两段，第一段不评分、核验后手工接续。独立DINO完整checkpoint格式绑定缓存骨干SHA，旧CLIP/原生/其他缓存拒绝；首步原loss/梯度及完整恢复audit随正式启动执行，不重复资源作业。35项本地新格式小模型磁盘重放/元数据拒绝/脚本包装检查及4个Bash语法通过，实际大模型首步仍待集群。两段各A40/40GiB/20h，纯训练估算每段12.73h，空间余量18GiB、无自动删文件/重投/续段。最终固定dev20一次三向奖励对照，单seed/复用场景/旧A6000新A40限制公开，final50封存、CLIP主模型不替换。见[范围与提交命令](PUSHT_DINO_TRAINING_READY_20261023.md)。

**DINO缓存221055的原验证器与实际在线路径通过（2026-10-23，用户终端）。** 日志末尾给出live_identity true，1/2/4批量token/context/action最大差均0，原seed42后端与train-only normalizer身份吻合，固定文本精确相同、策略状态未变，0更新/0环境执行。该summary由原run_cache完成完整缓存与绑定验证后输出；不再重复生成缓存或跑资源更新。不属于已训练DINO效果/泛化结论；Slurm完成账目及小报告原字节尚未接收，11.13GiB缓存继续留集群。下一项收取既有小报告并绑定84168同预算正式保存/恢复入口，未提交长训练。见[221055记录](../reports/pusht_dino_cache_submission_221055.json)。

**DINO训练缓存与在线接线221055已由用户提交（2026-10-23）。** 20文件包SHA通过，提交前项目占用135249221632字节，完整缓存加24GiB余量共需37723947008字节，保守200GB项目预算检查通过；这不是权威账户quota回执，也不等于已收到旧清理脚本执行账本。A40/4CPU/40GiB/30分钟，只做原90训练演示11356帧缓存、原DINO128帧对照和新worker实际在线一致性，无参数更新/环境步/奖励；运行、完成、缓存与实际跨环境接线尚未验证，不自动提交正式训练。日志`logs/pusht-dino-cache-221055.out`，输出`runs/pusht-dino-cache-221055`。见[提交记录](../reports/pusht_dino_cache_submission_221055.json)。

**DINO训练缓存与在线接线包就绪（2026-10-23），未提交。** 220882三份小报告已接收，本地核对原始像素/骨干/文本/后端/归一化身份及资源算术，资源预检不重复。新20文件包复用10原核心模块，仅编码既有90训练演示11356帧（11.13GiB），对照原DINO128帧，再启动新worker经过原策略回调比较1/2/4批量token/context及2窗DDPM100动作；无参数更新、环境步或奖励。提交前与作业内按保守200GB项目上限保留缓存外24GiB，不自动删除旧资产。29项包装/空间/损坏拒绝/数值socket替身检查及Bash语法通过；沙箱限制回环后在获准本机测试中通过，实际DINO跨运行时一致性仍待集群。A40/4CPU/40GiB/30分钟硬上限，无正式训练入口、自动提交或final50访问。生成身份文件供未来正式checkpoint绑定，未冒称正式恢复已验收。见[范围、空间与提交命令](PUSHT_DINO_CACHE_READY_20261023.md)。

**Bridge旧状态也纳入空间清理，第二批约8.87GiB脚本就绪（2026-10-23），未执行。** 明确6份：200150及200208两seed的common_u1024_with_adam、208586两个training_state、208964终点training_state。已核对各保存函数保留完整学习前缀（包括固定上下文），每组小型推理checkpoint另存；脚本会验证5报告SHA、8份保留推理权重和6待删状态SHA，重用原Slurm/路径/账本机制，Bridge层13项极小文件检查通过。207558/207746/207348/203699、数据、缓存和全部小结果不在删除范围。删后不可原样恢复旧Adam分叉状态；最终推理与已归档结果仍可用。加上Push-T首批预计约34G，不能把脚本准备记为实际已释放。见[Bridge清单与独立命令](CLUSTER_STORAGE_REVIEW_20261023.md)。

**集群空间清单已核对，首批6个旧检查点清理脚本就绪（2026-10-23），未执行。** 项目157G/runs98G/.tmp22G/envs19G/logs2.7M。只拟退役217778两份replay、211279旧失败latest、211309原生20k父状态、214273项目前半父状态、219260两步replay，约25.6G；212158/215999/219466及217820两正式终点、全部小报告/视频/代码、数据、缓存与环境保留。脚本默认预览，--apply核对5终点及候选完整SHA、恢复关联与Slurm无作业、逐文件写账本，不递归或通配删除；缺211279侧车则跳过。20项临时小文件正反例通过，未访问/删除集群文件。中间权重删后不能从对应时点恢复，历史完整预检重放需重新取得该状态，限制已说明。.tmp尚未分类，不能整删。见[清单、保留依据和命令](CLUSTER_STORAGE_REVIEW_20261023.md)。

**DINOv2资源预检220882完整通过（2026-10-23，用户终端）。** sacct为A40-2/COMPLETED0:0/70秒；原run_probe summary验证器通过，128旧训练帧编码、原固定文本/CLIP画面几何、冻结前后权重、Adapter/投影/后端梯度和5次丢弃更新均在本次原验证范围。warm每步1.088761秒、峰值7.537GiB、DINO每8帧0.131099秒；同84168纯训练估算25.455小时，不含I/O和闭环，不是效果结论。PIL回退后端正常完成，不安装依赖。不重复预检；下一项仅取得report/dino/project三个小报告，再准备11356训练帧DINO缓存、真实新worker IPC与正式骨干/cache身份绑定，仍未提交长训练、未访问最终集。见[作业与资源记录](../reports/pusht_dino_resource_submission_220882.json)。

**DINOv2资源预检220882已由用户提交（2026-10-23）。** 终端代码包SHA全部通过，1.22GB官方固定DINOv2权重SHA通过，返回sbatch编号220882；调度/运行/完成尚未验证。只编码旧128训练帧、复用原固定CLIP文本并做5次丢弃资源更新，A40/40GiB/20min，无闭环评分、全缓存或自动正式训练。日志`logs/pusht-dino-resource-220882.out`，输出`runs/pusht-dino-resource-220882`；不重复提交，不据此声称泛化改善。见[作业记录](../reports/pusht_dino_resource_submission_220882.json)。

**冻结DINOv2替代骨干对照已进入可执行资源预检阶段（2026-10-23），未提交。** 官方facebook/dinov2-large固定47b73e…ea2d6c的1.22GB safetensors取得并与公开LFS SHA一致；本机实际冻结模型对两帧旧训练图像输出257×1024，前后state SHA不变，原8层Adapter梯度与状态列有效，无新增宽度映射，在线RGB往返/pad8一致。文本仅合成接口检查，不冒称原固定语言或策略效果。原画面resize224/crop224几何保留，DINO ImageNet归一化及最终LN差异公开记录。12文件短包复用5旧核心文件，20入口/验证器/Bash/散列检查通过；服务器只重用旧128帧/64训练窗，编码新DINO并做5次丢弃更新，A40/40GiB/20min，不复跑旧原生/CLIP组、不生成全缓存或自动长训。见[资源包与上传命令](PUSHT_DINO_RESOURCE_READY_20261023.md)。同时核对LIBERO既有矩阵，真实窗口/审阅增强库/物理执行合同仍阻塞正式12组，具体下一项独立列于[并行工作流](LIBERO_PARALLEL_EXECUTION_BLOCKERS_20261023.md)，不以Push-T收尾为等待条件。

**重读第六封邮件并纠正后续顺序（2026-10-23）。** 直接读取原Word（SHA85bc7b…f62），老师明确“项目前端明显较差→DINOv2/R3M骨干对照；与原生相近→扩大OXE”，LIBERO四条件×三seed同步推进，并尽快汇报Push-T。当前0.445/零块0.527对原生0.846，不能归入相近分支。此前直接把扩大OXE/数据合同作为下一主要方向的建议排早了，现纠正为独立骨干对照设计与LIBERO并行；冻结CLIP主模型保持，不静默替换，也未在本轮修改模型/提交新训练。见[原文重读和研究顺序](TEACHER_EMAIL6_PRIORITIES_20261020.md)。

**219466小输出收齐并本地复核通过（2026-10-23），本分支关闭。** 10文件/1.88MB压缩包，84168更新及每条LR/数据/初始化/冻结状态/恢复摘要/奖励配对一致，视频可解码；大权重字节核验仍依据服务器原验证器，不冒称本地重跑。A40训练10.983h、进程11.214h。零块0.526787与8块0.445488的配对区间跨0，均满5场，原生0.845878/满10场，保持证据不足，不追加该分支。固定5000000零块最大0.614但终点0，5000001最大0.983终点0.174，不能把最大重叠当稳定终态。图已生成并查看。没有根因修复/新泛化提升结论；后续回到主线真实OXE窗口manifest、时间/单位与LIBERO评价合同缺口，不再重复局部结构长试验。见[完整结论、边界与后续顺序](PUSHT_ZERO_BLOCK_219466_RESULT_20261023.md)。

**219466原完整输出验证通过（2026-10-23，用户终端）。** 原verify_formal.py返回output_verified=true、84168更新，配对比较与终端一致；本地另已用固定旧8块参考复算20场景均值、满奖励计数和配对区间一致。零块0.526787/满5场，旧8块0.445488/满5场，配对区间跨0，保持证据不足结论，结束此消融，不自动续训或替换语言模型。小报告/更新日志/两固定视频尚未下载，大权重留服务器；下一步只收取并阅读已有输出，不增加GPU作业。

**零块控制219466终点完成，预定结论为证据不足（2026-10-23，用户终端）。** 零块平均奖励0.526787，旧8块0.445488，原生0.845878；零减八+0.081299，但配对场景bootstrap95%区间[-0.093343,0.254864]跨0，9好/7差/4平，满奖励均5/20。平均增益不能认定稳定改善，不能归因为8层唯一根因，也不能证明CLIP必然无解。只一个训练seed、复用dev20、旧A6000/新A40限制保留；不替换主模型、不追加此分支预算/seed/变体、不访问final50。完整输出原验证器尚待，终端完成标记不替代检查点/更新记录验证。见[终端结果记录](../reports/pusht_zero_block_formal_219466_terminal.json)。

**零块正式控制219466已由用户提交（2026-10-22）。** 终端逐文件SHA和219260原预检绑定通过，返回sbatch编号219466；调度/启动/完成均尚未验证。只从头训练一个零块控制至84168更新，冻结CLIP，旧8块215999直接复用；A40/4CPU/40GiB/20小时，纯训练预估12.794小时，非完成时间承诺。日志`logs/pusht-zero-formal-219466.out`，输出`runs/pusht-zero-formal-219466`；等待启动完整恢复检查及固定终点dev20，不提交重复作业、不访问final50、暂无新效果结论。见[提交记录](../reports/pusht_zero_block_formal_submission_219466.json)。

**219260预检验收通过，零块正式单组控制包就绪（2026-10-22），未提交。** 用户终端原验证器通过；实测每步0.547秒、峰值6.637GiB，84168纯训练估算12.794小时。只从头训练新零块控制，旧8块215999不重训，CLIP冻结、共同后端/池化/归一化/投影与数据/更新预算保留，零块固定语言不参与。原15个核心文件字节不变，新正式入口41项合成/计数/恢复/验证器/Bash/ZIP检查通过，不重复47项前端检查。申请A40/4CPU/40GiB/20小时，完整状态每2000步保存，支持检查后显式恢复；没有自动提交/重投。固定终点同dev20一次评价，final50封存，三向预定判据。单训练seed、复用开发场景、旧A6000/新A40限制明确。见[包与提交命令](PUSHT_ZERO_BLOCK_FORMAL_READY_20261022.md)。

**同CLIP有无八层堆叠的零块控制预检包就绪（2026-10-22），未提交。** 用户同意先隔离Adapter整体贡献。新增控制在原seed42后端和seed43旧前端构造完成后移除八个跨注意力/FFN块，保留旧池化、LayerNorm、投影、状态列与RNG；67648前端参数，CLIP冻结，Push-T固定语言不使用。重用原数据/统计/样本/噪声/Adam/EMA/LR，原八层215999不重训/不截断旧权重。仅8个独立短更新+1重算+1隔离末批38探针及2窗采样形状；无开发评分、正式入口、自动提交或final50。A40/4CPU/40GiB/15分钟硬上限，实际耗时未测；正式仍需同84168更新且只训新控制组，不能以短预检声称泛化改善。生产前端随机初始化及合成小后端/原loss/EMA等47项检查通过，20文件/50.8KB包。见[范围、可比性、资源与提交命令](PUSHT_ZERO_BLOCK_PREFLIGHT_READY_20261022.md)。

**冻结Push-T端点动作诊断219129完成并收口（2026-10-22），无新训练。** 原A6000在219059/裸CUDA健康219094失败，原包换A40成功，实际Python用时131秒；源权重/模型不改，具体A6000驱动/访问原因未定位。5个文件收齐，本地原输出验证器通过。执行8槽原生/项目：抽查训练6.202/5.392像素、未见27.928/39.508，静止未见47.758；项目在未见14/16演示更差，两采样seed方向一致。项目减原生未见均差+11.580、配对演示区间[5.236,17.895]，删任一演示仍为正。t49监督去噪项目未见MSE约原生2倍、13/16更差，训练侧反而较好；差距不只出现在闭环累积，首要方向是可迁移动作条件，而非重复拟合或补已有FiLM。单训练seed/固定32+32窗/复用诊断角色，不是最终成功率或唯一模块根因，保留215999与final50封存；217820旁路仍关闭，不自动长训。本地生成并查看逐演示、噪声与固定观测/动作图。见[完整结论与边界](PUSHT_ACTION_AUDIT_219129_RESULT_20261022.md)。

**冻结Push-T端点动作短检查包已就绪（2026-10-22），未提交。** 18文件/33.9KB，只读212158与215999原84168更新EMA，32训练/32未见演示窗、两个固定采样seed、DDPM100与1:9执行切片；辅助t=0/49/99去噪误差。0次优化/环境步，不动CLIP与旧权重，不复跑闭环分数，final50封存。独立选窗对照、噪声/RNG/切片/padding、轻量假策略只读路径、输出验证正反例、Python3.9/Bash/ZIP共43项本地检查通过；不声称真实大模型已本地运行。A6000/4CPU/40GiB/30分钟硬上限，内部1700秒截止、不requeue，实际服务器耗时未测。见[提交包与命令](PUSHT_ENDPOINT_ACTION_AUDIT_READY_20261022.md)。

**Push-T旁路217820已关闭，改为证据收敛后再决定训练（2026-10-22）。** 原服务器完整验收通过；17个小文件已收齐，本地核对16384条更新/LR记录、两组奖励门槛及固定视频，与终端一致。空间旁路0.390775、全局控制0.281801，都未超过原项目215999的0.445488；保持原模型及final50封存，不追加本分支。已将数据/扩散/FiLM/缓存恢复等既有检查与仍未隔离的前端疑点汇总。唯一下一项设计为冻结原生212158及项目215999端点，在相同32训练/32未见演示窗口上比较动作生成，不更新权重；复用既有90/40演示角色并固定32条演示选择。仅方案与演示ID已写入，真实帧窗清单、可执行集群包及实测耗时尚未完成，未提交作业，计划墙钟上限30分钟。不能将已存在的多层FiLM重做为改进，也不拿线性读出改善自动触发长训练。见[排查表、单项短检查及解释边界](PUSHT_FRONTEND_EVIDENCE_AND_SHORT_CHECK_20261022.md)。

**按计划完成四来源时间依据盘点与工程演示角色固定（2026-10-20），无模型/训练代码变更。** 文献名义频率Bridge/LT/BC-Z/RT-1=5/5/10/3Hz，本地schema均无逐帧时间戳，不能当导出间隔已验收；单位到米的缓存版本转换链仍待核对。复用LT审核，其余候选补查身份/CRC/共享JPEG，新增BC-Z两组4记录（ID不同、400种共享帧）；5158候选合5090组，无新增受保护Bridge身份别名，未解释最终标签。调用现有split_by_trajectory默认80/10/余下及seed42，固定工程角色4127训练/516验证/515源留出；全部已知重叠组不跨角色，三训练seed/四条件共用。不是主CLI窗口manifest，也未证明所有物理session独立；以后仅按固定角色过滤，扩池冲突需隔离。没改归一化/插值/模型/动作接口，未训练或提交。见[证据、固定归属与下一项](OXE_SOURCE_TIME_AND_ROLES_20261020.md)。

**四来源源笛卡尔动作候选接口已接入（2026-10-20），默认关闭且未训练。** 新`--action-frame source_cartesian`统一四源位移为记录轴下的当前→未来差，旋转改为源轴左乘相对q；LT只监督XY，不再需要伪造工具朝向。须配命令/可用性输入，LT须绑定real共享帧清单并逐次校验payload SHA，group_key使用连通组；新模式全部来源的补齐目标mask为0。数据身份/配置/恢复与旧表示隔离，LIBERO转换/时间合同/回调同版匹配。9个完整位姿合成窗、2个平面窗、一条旧工程记录的新编码、主损失222有效位姿分量/30夹爪步梯度检查及64组实际OSC目标几何检查通过；旧默认数据/配置/身份/原生动作不变。无神经模型训练/真实物理步/新泛化成绩；只统一语义，不宣称各机器人轴/单位/时序/TCP已经物理校准。214273/207558不改，主训练预算与分区仍未冻结。见[表示、接入与验收边界](OXE_CARTESIAN_INTERFACE_20261020.md)。

**Language Table真实缓存的重叠分组与缺失夹爪检查完成（2026-10-20），坐标接入仍未放行。** 对照第三/第四封老师邮件，只用real、推杆不伪装夹爪。全部3436缓存记录经TFRecord CRC/尺寸/有限性/语言与边界检查；55155帧出现中413种编码帧跨记录，130演示形成64多记录组（均有不同语言），共3370连通组；这只是exact-JPEG别名发现，不能证明物理原始演示独立，尚未划分。完整16帧/步长4候选3760窗来自1414记录，短记录未删除。LT解析器仅新模式改为[0,0]缺失输入，默认/图像/目标/mask不变，其他函数AST不改；实际主loss/metrics合成梯度检查确认无夹爪标签不产生BCE或梯度，sample/epsilon输出无效维度直接梯度0，评价不虚构旋转/夹爪成绩。不是降噪模型训练、无新泛化分数；实机坐标映射仍缺，实际构造器拒绝保留，未提交新作业。见[老师要求、数据证据与剩余项](LANGUAGE_TABLE_ADMISSION_20261020.md)。

**两维命令/可用性输入已接入数据—模型—训练—评价—LIBERO回调（2026-10-20）。** 新`--oxe-command-input`默认关，三3D解析器只改当前夹爪特征、不改目标/mask，state目标/1倍权重及明确各源策略强制检查；模型仅夹爪输入投影从1列变2列，CLIP冻结。数据身份/配置/恢复隔离旧新模式。未知先前命令不进入保持/change基线，覆盖率显式报告，全未知为NaN；预测准确率仍含有效目标。9合成窗口、实际主模型/主损失一次合成AdamW更新、两列/Adapter/扩散梯度、保存重载已知/未知推理、旧模式预测/指标一致与旧权重拒绝均通过；其他解析器/主损失AST未改。无真实数据训练/模型闭环/新泛化成绩，LT仍被实际入口拒绝、投影损失未接；214273/207558保持。见[接入与验收范围](OXE_COMMAND_INPUT_CHAIN_20261020.md)。

**新混源命令输入规则已实现，Language Table平面接入条件明确（2026-10-20）。** 独立模块使用`[上一开正命令,可用性]`，不拿测量替命令、缺失不默认开、不借未来推初态、不用终止action。固定候选记录新命令字段检查通过（Bridge9/49、BC-Z200/201、Fractal80/115可识别，LT全缺失），未重做旧位姿/RGB检查；Bridge原值含约-.009，保持原值、输入标缺失，不等于目标验收。LT本地schema无朝向，未取得实机标定；新增经标定旋转投影到源xy的损失核，合成45度旋转/未观测方向梯度检查通过，不能用单位旋转或二值xyz mask冒充正确接入。旧loader/model/loss未改，新2维输入与投影核尚未接主训练，无训练/新性能/模型加载；214273/207558保持。见[规则、实读范围与剩余条件](OXE_SOURCE_SEMANTICS_RULES_20261020.md)。

**OXE现有缓存候选清单已落实（2026-10-20），尚非可提交混源训练。** 复用已有源记录验收，只读四来源TFDS元数据与32缓存分片头：5162记录与元数据逐片计数一致，排除1旧开发/3已用留出Bridge记录，留5158候选（399/3436/643/680），共5.635GiB无新副本。候选未分区/未验收窗口，CRC/完整payload/原始演示别名及共享帧去重未做，无新标签/图片/最终任务读取。确认真实Language Table来源名与作者计数一致；帧/时间及夹爪测量、命令、缺失输入仍是混源前置缺口。修正上一步新推理入口只凭binary判断命令语义的不足，新增明确语义要求，缺失则拒绝；仅补查该接口条件，旧模型不能事后伪造语义声明。正式预算未臆填，214273/207558不改，无新训练。见[清单、缺口与限定下一项](OXE_BASELINE_READINESS_20261020.md)。

**主模型→LIBERO推理回调已实现并完成CPU一致性检查（2026-10-20）。** 复用`prepare_image/tokenise/RobotAdapterModel.sample`，严格恢复完整学习模块，检查尺度/长度/扩散设置与二值夹爪编码，CLIP冻结并eval，每次明确seed且保留调用者RNG。实际主Adapter/降噪器/夹爪代码配本地随机小CLIP，直接/回调最大差0，保存重载连续两次推理一致、state不变及不兼容输入拒绝通过；0训练/物理步/预训练下载，CUDA与真实训练模型未验证，不能称模型闭环已跑通。本地CLIP身份仍须绑定训练指纹。214273和207558不改；不重复工程接线，下一项转向真实OXE无增强主基线清单、来源合同和预算。见[入口与范围](LIBERO_PROJECT_POLICY_ENTRY_20261020.md)。

**LIBERO专用读取错位已修正，显式时间执行逻辑检查通过（2026-10-20），无训练/新物理回放。** 官方生成器保存post-step观测，旧读取器却用前一行夹爪命令且未翻OpenGL图像；仅修此函数，同一工程demo的5个窗口含两次117/189行开合核对通过，位姿前7维标签及OXE解析器不变。新目标执行器要求显式目标时间/尺度/执行前缀，逐20Hz测量反馈重算差值、整个块固定参考、夹爪从区间起点发送，不裁剪/重置/读成功标签；实际OSC目标代码配合合成滞后动力学检查通过，物理步数0、无模型。该LIBERO路径未用于Bridge训练，不能当Bridge泛化根因；8维完整策略、源时序/TCP/夹爪及正式任务预算仍待验收。Push-T214273保持。见[实际修复与边界](LIBERO_TIMING_AND_READER_REPAIR_20261020.md)。

**四种语言条件已接入主训练入口（2026-10-20），无新训练。** `train.py`新增默认关闭的无增强/同义改写/反事实/组合入口，绑定统一训练划分和审阅库，在原tokenizer前干预；同义改写保留目标，反事实使用零位移/单位四元数/明确夹爪保持命令，监督掩码保留。CPU合成检查通过实际循环前缀、默认路径、四组抽样配对、留出隔离、非法目标/命令拒绝及语言抽样恢复，评价函数未改；不等于完整模型训练或真实语义验收。真实训练专用库、固定预算/完整恢复及LIBERO时序/TCP/模型执行仍待完成，未启动12组实验。Push-T214273及Bridge207558不改。见[实现、检查与缺口](LANGUAGE_CONDITIONS_INTEGRATION_20261020.md)。

**Push-T运行期间并行打通LIBERO原生入口（2026-10-20）。** 官方8f1084e源码与本地130任务/6500演示名称、场景/初始状态资源和控制元数据全部匹配，无93GB重复下载；新工作区独立仿真venv、Windows渲染适配完成，不改214273环境。既有LIBERO90工程任务demo_0实际重置/双相机8步控制通过，197原生动作无中途状态恢复连续回放，第186步及终点原生成功；这是示范动作正对照，不是学习策略或正式成功率，工程任务排除最终成绩。新增相对位姿→原生OSC几何边界，经实际set_goal与128合成姿态核对通过，时序/TCP/模型夹爪仍未验收。四条件×3seed配置草案及真实缺口已明确，未提交新训练，不把LIBERO适配当OXE零样本。原生源码未改；继承的无关包pip冲突保留说明。详见[本次实际成果与下一项](LIBERO_ENGINEERING_PROGRESS_20261020.md)。

**第六封邮件已对照实现并明确研究顺序（2026-10-20），无新增训练。** 继续214273项目组既定两段Push-T对照，同时将LIBERO四条件×3独立训练seed的正式流程准备提升为并行主线，不再以全部Bridge诊断结束作为前置条件。源码核对证实207558固定multiscale与当前作者Push-T后端均在down/mid/up块注入FiLM；不能把“首次补上多层注入”当新改进重做。CLIP空间精度/向量读出压缩仍是待证候选；DINOv2/R3M由老师建议作为后续骨干对照，保留冻结CLIP主参照且不改变当前作业，不解冻CLIP。Push-T相似成绩也不证明Bridge只缺数据。记录见[优先级及实现核对](TEACHER_EMAIL6_PRIORITIES_20261020.md)。

**Push-T项目组第一段214273已由用户提交（2026-10-20）。** 包内逐文件SHA检查通过；真实执行状态、启动损失/梯度及完整恢复重放尚待。目标累计42084更新，冻结CLIP训练缓存213439，项目Adapter+映射+新初始化作者扩散后端参与训练；同原生总预算84168分两段连续执行，第一段无开发评分、第二段尚未提交、不自动重提。日志logs/pusht-project-part1-214273.out，报告runs/pusht-project-part1-214273/report.json。见[作业记录](../reports/pusht_project_training_submission_214273.json)。

**213439缓存核验通过，项目同预算训练包已准备（2026-10-20），未提交。** 用户运行原CPU验证器通过，11356帧/11.133GiB，原128帧视觉差0、文本差0，CLIP冻结/前后权重未变；小report本地已读，11GiB数组仅集群散列核对。复用原生已修复core、原数据类、原Adapter和实时CLIP接线，原生同seed42新初始化后端，项目seed43；84168更新分42084/84168两段，第一段不评分/不自动提交第二段，完整状态摘要恢复。CPU真实作者loss/EMA配小模型核对新缓存绑定与直接输入更新一致（8/38/64含尾批）、恢复下一步完全一致、错误索引/边界/合同拒绝；实际两段循环不重置计数，评价函数除标签外AST相同，原文件字节/3.9语法/CLI/Bash/ZIP通过。不重复资源/在线预检，不加载已训练原生后端；完整新前端正式训练和闭环效果仍待集群。见[提交说明](PUSHT_PROJECT_TRAINING_READY_20261020.md)。

**213439冻结缓存已正常完成（2026-10-20，用户终端证据）。** A6000/6:28/COMPLETED0:0，11356/11356帧及完成标记齐全；尚待原verify_cache.py核验CLIP冻结/前后权重、128参考特征及三个数值文件SHA，不能仅凭退出码宣称全部完整性通过。0训练更新、无新泛化结果，不重复缓存或自动提交训练。另原生212158缺失summary已到，本地现有分析器核对完整报告、侧车、恢复状态和逐步奖励通过；不再请求重复下载该summary。下一步下载213439小报告，11GiB缓存留集群。

**Push-T冻结训练特征缓存作业213439已由用户报告提交（2026-10-19）。** 仅跟踪此缓存，不重复提交；作业状态、11356帧输出、CLIP冻结/权重不变、原128帧参考一致性及文件SHA尚待。日志logs/pusht-clip-cache-213439.out，报告runs/pusht-project-cache-213439/report.json。0训练更新，项目Adapter匹配训练尚未提交。见[提交记录](../reports/pusht_project_cache_submission_213439.json)。

**原生212158已收到4个JSON及2视频，summary缺失；冻结CLIP全训练帧缓存包准备完成（2026-10-19），未提交。** 用户cap验证器通过；本地原奖励复算.84587825/10满奖励与84168侧车、原20k来源/恢复记录和合同一致。report.json未下载，视频未解码，4GB权重未本地散列，不冒称全summary已审。独立缓存包提取同90演示/11356原图，精确复用native TrainingData AST及211193 worker五文件；32帧请求、内部批8，冻结CLIP前后hash相同、原128帧特征容差不变，产FP32约11.133GiB供后续同84168预算Adapter训练。0训练更新/无评分，无重装下载或重复资源/接线实验；未自动正式训练。数值memmap/尾批/索引及拒错、AST/CLI/Bash/ZIP通过，全缓存真实执行仍待集群。计划项目训练约24h纯训练、分段恢复，仅终点dev20；先补cap报告并核验缓存，再放行正式入口。见[缺文件命令与缓存提交](PUSHT_PROJECT_CACHE_READY_20261019.md)。

**Push-T原生上限212158完成并通过预定参照门槛（2026-10-19，终端证据）。** A6000/9:33:39/0:0，到84168更新；同20开发初始场景平均最大奖励0.84587825、严格满奖励10/20=50%，原门槛平均>=.8且满奖励>=50%通过。20k对应.79588565、3/20，不能将平均当成功率或称用户Adapter已完成。仍有失败场景/一个0分，最终50未评。下一步仅现有cap验证器及完整小工件核对，随后准备同90演示/10726窗/相同新初始化后端及样本噪声计划/同84168预算的冻结CLIP+Adapter前端组；不继承原生已训练后端，不自动提交或重跑原生。完整cap报告尚未本地收到，CLIP硬冻结、Bridge207558/final25保持。见[结果及对照边界](PUSHT_NATIVE_CAP_RESULT_212158_20261019.md)。

**Push-T原生上限接续作业212158已由用户报告提交（2026-10-19）。** 从211309完整20k状态接续到原定84168，新增64168次；排队/运行状态、完整恢复核对及终点开发成绩均未收到。只跟踪212158，不重复提交。日志logs/pusht-native84k-212158.out，报告runs/pusht-native84k-212158/report.json；原20k参照保留，项目Adapter组和最终集未启动。见[提交记录](../reports/pusht_native_extension_submission_212158.json)。

**211309完整小工件已核验，原生84168上限接续包已准备（2026-10-19），未提交。** 用户集群验证器通过；本地逐步奖励复算0.79588565/3满奖励、完整启动恢复与设备修复记录一致，gate仍False。4GB权重未本地下载/散列，两个固定视频仅文件头/大小/SHA检查、画面未解码。新包保留20k源状态，仅从20000接续到原定84168，新增64168次；原优化器/LR/EMA计数/RNG/样本噪声计划全部恢复并完整状态摘要核对，基协议/修复core字节相同。只评估终点同20开发场景，cap失败停本分支，不自动项目组/最终。小CUDA20k计数边界续训精确匹配、实际生成循环64168新计数/终点保存、接线/合同拒错与AST/CLI/Bash/ZIP通过；真实集群入口尚待执行。实际速度投影约9.91小时新增纯训练，I/O/评估/排队另计。见[协议、验证边界与提交命令](PUSHT_NATIVE_EXTENSION_READY_20261019.md)。

**Push-T原生211309完成20k，部分新场景闭环完成，但预设参照门槛未过（2026-10-19，终端证据）。** A6000/3:21:10/0:0，20开发场景平均最大重叠奖励0.79588565，严格满奖励3/20=15%；原门槛平均>=.8且满奖励>=50%，不移动门槛、不把平均当成功率。已保存20k完整权重；原恢复错误未再导致此作业中断，但启动完整工件及权重尚未本地核验。下一步仅现有CPU验证器和小JSON/固定前2视频审阅，再考虑原协议上限内的原生完整状态接续；不重训前缀、不自动项目训练/新分支/最终评价。项目Adapter仍无本次Push-T训练成绩，CLIP硬冻结、Bridge207558/final25保持。见[结果与下一步](PUSHT_NATIVE_RESULT_211309_20261019.md)。

**Push-T设备恢复修复后正式作业211309已由用户报告提交（2026-10-18）。** 仅跟踪此作业，不重复提交；PUSHT_RESTORE_DEVICE及原PUSHT_NATIVE_STARTUP检查、训练进度与开发结果均尚待真实输出。原211279失败目录保留，原20k协议不变，项目组尚未训练、最终集封存。日志logs/pusht-native20k-211309.out，报告runs/pusht-native20k-211309/report.json。见[提交记录](../reports/pusht_native_training_submission_211309.json)。

**211279因新入口恢复设备错误失败，独立修复包已准备（2026-10-18），未提交。** 用户终端A6000/1:13/1:0，Conv2d输入CPU、权重CUDA；原CPU检查点加载导致作者normalizer重建CPU统计并将图像转回CPU。旧CPU测试使用简化normalizer漏检，这是助手新增恢复逻辑错误，非泛化结论。新overlay仅改restore：加载后raw/EMA重新搬回目标设备并全张量检查，原包/协议/损失/2万预算及精确重放自检均保留。RTX4060/Torch2.11真实GPU小Conv2d+实际作者normalizer复现逐字相同报错，修复后下一更新loss和完整状态精确一致；实际原runner绑定及旧文件变更拒绝通过，集群Torch1.12执行仍待。失败目录保留，不使用失败启动检查点、不重复资源/在线预检；CLIP硬冻结、207558和最终集不动。见[修复及提交说明](PUSHT_NATIVE_DEVICE_REPAIR_211279_20261018.md)。

**Push-T原生20k正式作业211279已由用户报告提交（2026-10-18）。** 只跟踪此作业，不重复提交；排队/运行状态、真实GPU损失与磁盘恢复启动自检、训练及固定20开发回合结果尚未收到。日志为logs/pusht-native20k-211279.out，输出runs/pusht-native20k-211279/report.json。项目Adapter组尚未提交，CLIP硬冻结，Bridge207558和最终集保持。见[提交记录](../reports/pusht_native_training_submission_211279.json)。

**211193完整在线报告已核验；Push-T原生20k正式训练包已准备（2026-10-18），未提交。** CLIP冻结/权重未变、五项缓存/在线差异0、2决策16步，无任务得分。第一阶段固定90演示/10726窗、作者完整U-Net/原epsilon损失和EMA，保留512400长程cosine/warmup500，有效batch64按尾批计权，独立前端/扩散RNG计划。新入口内做原损失/梯度及第1步保存、第2步完整恢复重放自检，通过直接训练到20000，非重复资源/在线预检。每2000保存完整状态、固定20新开发场景评分/前2视频，不自动延长/项目训练/最终集。CPU原作者损失与真实EMA小模型、随机配对、磁盘恢复、数据补帧/计权、奖励复算拒错和CLI/Bash/ZIP通过；此新入口真实GPU执行尚待集群。预计原生纯训练2.84小时，额外I/O/评价另计；CLIP硬冻结、207558/final25保持。见[协议、边界与提交命令](PUSHT_NATIVE_TRAINING_READY_20261018.md)。

**Push-T在线接线211193终端通过（2026-10-18，完整报告待本地读取）。** A6000/1:28/0:0；文本、原批8视觉特征、归一化图像往返及补批特征、Adapter上下文、同seed DDPM100动作最大差均0，真实环境完成2次决策/16步。0训练更新、未训练策略、未记录任务分数，不能称泛化或任务成功。接线入口通过，不重复资源5更新或本预检；下一步只取完整report核对冻结及执行记录，再锁定正式共同样本/噪声、LR/EMA和保存恢复协议。正式训练尚未提交，CLIP硬冻结，Bridge207558与最终25保持。见[终端证据](../reports/pusht_online_211193_terminal.json)。

**210763完整资源报告已核验；在线接线短预检包已准备（2026-10-18），未提交。** CLIP冻结/权重未变，共同后端与仅训练normalizer SHA相同，位置列64/65，项目Adapter/映射均有梯度；原生峰值5.36GiB/0.511秒更新，项目7.65GiB/1.037秒。84,168次每组纯训练合计约36.19小时，非已授权正式预算。提出原生先20k、必要时续至84,168，再给项目匹配预算的草案，尚需锁协议。新包不重复资源训练；只接现有两环境私有socket，核对缓存/在线特征、归一化观察逆回像素与同seed动作，再做未训练策略最多2决策/16步接线，不记录奖励或泛化分数。CPU256像素往返、消息与异常清缓存、CLI/Bash通过，真正跨环境/仿真待集群；CLIP冻结、207558与最终25保持。见[资源结果、预算草案与提交代码](PUSHT_RESOURCE_RESULT_AND_ONLINE_READY_20261018.md)。

**210690数据层收尾；Push-T冻结前端接线/资源预检包已准备（2026-10-18），未提交。** 本地完整report确认206演示/25650帧、图像float32值域65–255、所有块解码通过；原配方90训练/10726窗，validation补集116/13482窗。选定两环境隔离：项目环境冻结CLIP缓存，作者环境运行原CNN组与原八层Adapter+64维映射组，共同完整U-Net；不安装。新包只抽64训练窗、各5次更新+EMA/一次batch2 DDPM100形状计时，逻辑batch64=8×8，不评分开发/最终。实际识别原位置列，保留后端与仅训练normalizer相同SHA；共享样本，但资源探测噪声未配对，不作loss效果比较。CPU小实例注解兼容逐元素同值、三种列布局/梯度、CLI/Bash/ZIP通过；实际两环境模型兼容/资源待GPU。正式预算/实时CLIP通信与闭环接线尚未就绪，不拿0.9309作短训练比较；CLIP冻结、207558/最终25保持。见[事实、接口、停止规则与提交命令](PUSHT_FRONTEND_RESOURCE_READY_20261018.md)。

**210631失败已定位为准备器错误限定uint8，缓存修复包已准备（2026-10-18），未提交。** 用户ZIP元信息为25650帧96×96×3 float32、206边界，尺寸正确；作者源码除255并不限定uint8。新6.9KB包无下载/解压/重装/模型，核对原报告与缓存SHA后原地使用210631暂存数据，精确匹配已见schema、补完缺块/全解码/原始值范围与计数；若像素已在0–1或越界仍停止，不cast/clip。原数据包未改，合成float小数不损失及五种拒绝案例、CLI/Bash/ZIP通过；真实完整性/像素范围仍待集群CPU检查。后续须保留210631数据目录；207558/CLIP冻结/最终25保持。见[修复范围及提交命令](PUSHT_FLOAT_IMAGE_REPAIR_READY_20261018.md)。

**Push-T数据准备210631因图像规格检查失败（2026-10-18，只有终端证据）。** CPU2/8:32/1:0，包SHA均通过；实际multitask-preflight-v1版本已输出，修正旧环境盘点遗漏。下载走到28MiB并完成，程序已通过ZIP路径/CRC和暂存解压，到validate_zarr时硬写的uint896×96×3条件未满足；实际shape/dtype尚未输出，不能判断是尺寸、类型、数据范围或下载对象问题。检查点输入规格不能直接当原始存储规格；不盲目改dtype/尺寸或跳过断言。下一步仅读已缓存ZIP的.zarray与旧报告，不重下、不重新整套提交、不安装/训练。全数据完整性仍未通过，207558/CLIP冻结/最终25保持。见[终端记录](../reports/pusht_data_prepare_210631_terminal.json)。

**Push-T训练数据/实际环境CPU准备包已完成（2026-10-18），未提交。** 用户同意210557之后的准备步骤；新包8.6KB/6文件，只下载作者约29MB pusht.zip，复用已出现的原路径数据，检查ZIP/块存在性/全部数组逐块解码，描述作者演示/窗口计数，并补查multitask-preflight-v1版本。无模型/GPU/环境改动/检查点重读/新划分或统计拟合。Python3.9/CLI/Bash/ZIP与模拟下载、危险归档、缺块检查通过；本地无Zarr，真实数据与codec待集群执行，不声称完整性已通过。CPU4核12GB/30分钟，不自动后续训练；CLIP冻结、207558/最终25保持。见[包、范围和提交命令](PUSHT_TRAINING_DATA_READY_20261018.md)。

**Push-T资产盘点210557完整报告已本地读取（2026-10-18）。** CPU12秒/0:0；固定作者五文件SHA均与本地一致，检查点保存epoch500/global_step84167，实际图像64+位置2、两观察条件132维已确认。所查六路径无训练Zarr，不能称全盘不存在；官网训练pusht.zip目录约29MB，尚未下载，解压完整性与GPU成本未知。发现盘点遗漏：查了旧bridge-diffusion，近期实际用multitask-preflight-v1，不能据前者null判当前缺包；只补轻量元信息，不重跑盘点/评估、不重装官方环境。两组匹配训练尚未就绪，没有新泛化成果或Bridge根因结论；CLIP冻结、207558/最终25保持。见[事实、遗漏纠正和下一步](PUSHT_ASSET_RESULT_210557_20261018.md)。

**Push-T模块对照的现有资产CPU盘点包已准备（2026-10-18），未提交。** 7.5KB/4文件，只读既有作者检查点保存进度与条件权重形状、固定源码指纹、两环境版本元信息、限定路径Zarr元信息/episode_ends与CLIP配置；不读帧、不构造模型/完整Dataset、不重复原评估、不安装/下载/请求GPU。Python3.9语法、CLI、18组作者采样计数对照、Bash与ZIP/SHA通过；真实集群资产、运行兼容和GPU预算仍未知。CPU4核16GB/15分钟，新输出独立目录；207558与最终25保持。见[用途、范围与提交命令](PUSHT_MODULE_ASSET_INVENTORY_READY_20261018.md)。

**210334收尾后新增Push-T前端模块对照可行性提案（2026-10-18），未准备训练入口。** 核查195527只是作者预训练评估，尚无同Push-T数据下冻结CLIP+项目Adapter的匹配训练/闭环结果；203699迁入Bridge不能代替该对照。建议固定作者原生二维动作/执行端/扩散后端，检验项目前端闭环能力，避免继续Bridge小改动与仅凭模仿距离判断成功。当前训练Zarr存在性、实际encoder输出维度/global_step、旧环境兼容和匹配训练成本均未确认；作者3050计划不当已训练步数，0.9309不当二元成功率，也不把随机前端直接接旧后端作为公平比较。先做资产/接口/预算可行性，不下载/安装/提交GPU；CLIP硬冻结、207558和最终25不变。见[具体问题、控制与停止条件](PUSHT_FROZEN_CLIP_MODULE_COMPARISON_PROPOSAL_20261018.md)。

**210334完整输出已由用户在集群核验，区域读出分支正式归档（2026-10-18）。** 发布验证器输出SPATIAL FORMAL OUTPUT VERIFIED，权重/日志/NPZ指标及采用判定检查通过，五门槛仍全失败、共同改善3/10；不是改善通过。保留207558，不加读出预算/种子/变体、不再重复预检或完整验证，CLIP硬冻结、最终25封存。完整report/NPZ/权重未本地收到，不声称本地复算；唯一根因仍未确认，没有新增已采用模型。见[最终结论](BRIDGE_SPATIAL_RESULT_210334_20261018.md)。

**区域读出正式210334终端完成，按协议关闭该分支（2026-10-18）。** COMPLETED/0:0、A6000/25:44，各4096更新；新10已用开发演示/50窗，候选第一目标1.4325cm/2.7873度，控制1.4469/2.7837，不动1.2269/2.4401；候选全16为5.8471cm/13.7874度，原207558为5.9863/13.7815。第一目标位置较控制只改善0.144mm、仍差于不动约16.8%，五门槛全失败、共同改善3/10。保留207558，停止该读出追加预算/种子/变体，CLIP硬冻结、最终25未读。当前只有用户日志，完整输出验证器及NPZ独立复算尚未报告；下一步仅现有CPU验证器核验归档，不自动新训练。见[结果和收尾命令](BRIDGE_SPATIAL_RESULT_210334_20261018.md)。

**210291已由用户集群完整验证，正式配对追加包准备完成（2026-10-18），未提交。** 验证器报告3.77734375GiB、0.160323583秒/暖更新、预计1956.053秒；原report尚未本地下载。正式10文件/16276字节追加包复用不可变预检源，从207558各接续4096更新、原1836完整窗/同计划/新Adam1e-5，只候选学习1026区域评分；CLIP/夹爪冻结。实际calculate AST与预检相同，正式前8损失须对应；两最终权重先保存/重载，再评分原新50开发窗，CLIP编码与209481 SHA一致且原预测直接复用。近期+全段采用规则保持、不动最终25。小型八层合成loop、保存重载与正/负验收案例通过，Python CLI/Bash/依赖SHA/ZIP核验通过，无真实正式模型训练或作业提交。见[追加包和提交命令](BRIDGE_SPATIAL_FORMAL_READY_20261018.md)。

**区域读出预检210291终端完成（2026-10-18，完整报告待验证）。** 用户粘贴COMPLETED/0:0、A6000/1:56；控制/候选各8更新，首损失均0.035673804581165314，终端输出SPATIAL PREFLIGHT PASSED。当前只有日志，没有本地完整report或显存/时长数据；不冒称独立核验、泛化提升或正式训练已提交。下一步只运行包内既有verify_preflight.py读取该报告，不新交作业、不重复预检。参照207558和冻结CLIP要求不变。见[终端记录](../reports/bridge_spatial_preflight_210291_terminal.json)。

**区域读出等预算协议与唯一集群预检包已准备（2026-10-18），未提交。** CLIP硬冻结；两组计划从207558各接续4096次，同原1836窗/405演示、原计划后4096批、新Adam/1e-5，候选仅多1026区域评分参数。不启用关闭的请求长度分支。固定新50已用开发窗，近期第一目标优于控制/原模型/不动且全16/终点不退步等采用条件；失败不加预算/种子/变体。包837757字节/23文件，无权重/图片/TFRecord及正式入口。实际Python CLI导入、Bash语法、SHA与ZIP身份通过；八层32维合成接线train/eval上下文/原梯度0差，无真实策略执行。集群仅真实207558+零读出原生采样回退、各8更新、Adam恢复/冻结指纹/资源检查；不评分开发、不自动正式。新预检还数值解析原405记录，不解码图片或新编码。207558保持、最终25未读。见[协议、包和短提交命令](BRIDGE_SPATIAL_PREFLIGHT_READY_20261018.md)。

**区域加权读出隔离CPU接口通过（2026-10-18）。** 仅新增独立1026参数原型，未改生产Adapter或固定源。合成[2,257,1024]输入，与存档Adapter零层实例的真实末端读出比较，零初始化输出/输入梯度均差0，LayerNorm梯度一致，新内容/坐标评分梯度非零；独立加权参考、坐标配对、有限值、输入拒绝与保存重载通过。无CLIP/真实八层/207558权重/扩散执行、无数据读取或优化器、0更新/GPU。只证明接线，不是泛化提升；实际权重重放、同预算控制与采用协议尚未执行/确定，无作业入口。CLIP冻结、207558保持、最终25未读。见[结果与边界](BRIDGE_SPATIAL_READOUT_CPU_20261018.md)。

**冻结CLIP后Adapter读出结构审查完成（2026-10-18），无运行/训练。** 固定207558为8层视觉query/文本key-value，Adapter无新增视觉self-attention，最终CLS与等权patch均值混合；CLIP本身已有位置和全局交互，不能称空间信息全丢失。已查重早期CLS/patch均值回归对照、网格近邻、视觉消融、增强、ridge与固定条件头，不重复。定义一个尚未实现的1026参数区域加权读出草案，零分数时数学上退回原平均；不是定位器或已验证修复。训练起始、控制、预算与采用门槛未定，无GPU入口，CLIP硬冻结、207558保持、最终25未读。见[结构、旧证据与候选边界](BRIDGE_ADAPTER_READOUT_REVIEW_20261018.md)。

**新增明确硬约束（2026-10-18）：CLIP必须全程冻结。** 用户说明这是老师和项目要求；撤回助手刚提出的视觉最后一层解冻对照，未执行该建议。以后不得以诊断为由微调CLIP；后续仅考虑冻结特征之后的处理及训练，扩散保留，泛化为主、查重优先。独立记录于[项目硬性约束](PROJECT_HARD_CONSTRAINTS.md)，未改动被其他协议引用的207558指纹清单。

**部署语义源检查完成，删首步候选关闭（2026-10-18）。** 207558实际context是当前图文+当前9维位姿；`is_first`只作加载器结构核验，不输入策略，也没有记录起点/原始运动指令条件。没有该Bridge权重已验证的闭环启动映射可证明初始化状态应排除；旧Panda启动不构成证据。由于监督是实测未来位姿，原指令首步近零不足认定标签错误。故不移动窗口、不准备该路线训练/启动标志/静止阈值变体，审查候选仅归档。无新模型运行、数据读取、训练或GPU；207558保持、近端泛化仍待解决、最终25未读。见[关闭理由与源引用](BRIDGE_INITIAL_STEP_REVIEW_20261018.md)。

**首步全池核对与取窗对照草案完成（2026-10-18），无新GPU/训练。** 405训练+10已用开发均匹配载荷SHA；415条原始首步前六维运动指令均近零，但监督是实测到达位姿，不能据此认定标签错误。直接删start0使1836窗/405演示变为1431窗/389演示；重新编号会整体平移取窗。审查候选只将404个有足够未来的起点0→1，保留1836窗/405演示和1432行原输入，一条17帧演示明确保留start0，故不是完全去首步。所有模型须使用相同原50开发窗（含起点），不删评估输入。尚未选定训练起始方式、预算/门槛或实现入口；证据不足认定去首步会改善泛化，不自动启动该路线。207558保持、209481关闭、最终25未读。见[审查及比较边界](BRIDGE_INITIAL_STEP_REVIEW_20261018.md)。

**低位移原始时序核对完成（2026-10-18），无训练/改标签。** 先按位移和身份SHA固定12低/大位移配对窗口、只读7既有导出演示，复用旧几何/时移审计。6低例5个start0，原运动指令近零、无整姿态/RGB原样复制，时间戳缺失，不能确定控制时延。OpenVLA Bridge变换会去首步，而项目保留，这是候选处理差异；Bridge作者/Octo也支持到达状态重标，故不能以原指令与实测不同就判标签错。全量保存预测中低位移374/520训练、9/13开发为起点；后续窗口第一目标训练1.459cm优于不动1.712，新开发1.638仍差于1.568，说明首步不能解释全部泛化。全部原评分/窗口保留，209481关闭、207558不变、25最终未读，不自动删首步或新增GPU。见[结果及对照边界](BRIDGE_LOW_MOTION_TIMELINE_20261018.md)。

**已用原207558保存训练预测定位近端模式（2026-10-18），无新拟合/模型执行。** 历史SHA/1836索引对应，405训练与新10开发不交叉；统一五任务权重后，第1目标模型/不动训练1.306/1.282cm，新开发1.446/1.227cm。训练平方位置距离2.572cm²优于不动2.878，故不能称训练完全没学到；新开发3.132对2.486cm²仍差。真值下一步<0.5cm的520训练窗/13开发窗，模型误差0.867/0.968cm，不动0.152/0.160cm，模型输出位移约0.879/0.969cm，显示微动过度运动模式。全窗口保留，真值分组只作诊断、不作部署门控。下一项建议固定少量低位移记录审查动作/实测时序语义，先避开已有审计；不直接改loss/阈值、重跑历史或新增GPU。见[分析与边界](BRIDGE_NEAR_MOTION_TRAIN_DEV_20261018.md)。

**209481保存结果的有边界CPU失败分析完成（2026-10-18）。** JSON/NPZ已到本地，无模型/GPU/新推理/训练/原始数据读取。独立公式复算摘要最大差1.07e-14。207558全16位置五任务均优于不动，8/10演示改善；但五任务第1目标位置均较差，总体1.446cm对1.227cm，前4平均3.269对3.133，远期优势支撑全段5.986对8.111。第1预测平均位移1.039对真实1.227cm，无整体幅度放大；方向比较偏弱，采样散布仅占约15%第1/17.6%全段位置平方误差。新增K1也未胜不动，保持分支关闭。下一阶段建议聚焦新演示的近端动作可靠性；当前证据不足指定因果修复，不追加GPU。全部任务/演示/目标/残差表和两张曲线已保存，25最终仍关闭。见[完整分析](BRIDGE_FAILURE_ANALYSIS_209481_20261018.md)。

**209481完整输出验证器通过，分支已收尾（2026-10-18，用户集群验证输出）。** 同一新10演示50窗：原207558=5.9863cm/13.7815度，不动=8.1106cm/15.5170度，分别降低约26.2%/11.2%；支持另一批同任务新演示的离线优势。新增后段候选6.2400/13.6646仍未达采用条件，保留207558、关闭此分支、25最终不读。不重复验证或追加GPU实验。完整文件尚未本地下载，验证通过依据用户在集群运行现有验证器的输出。见[已复核结果](BRIDGE_PREFIX_RESULT_209481_20261018.md)。

**209481已完成，后段输入请求长度分支按协议关闭（2026-10-18）。** 用户终端报告A6000/28:18/0:0。新10演示50完整窗演示等权：原207558=5.9863cm/13.7815度，控制=6.3061/13.9220，候选=6.2400/13.6646。候选略胜控制但位置劣于原基线，仅同3/10演示位置同时胜两参照，四项采用门槛失败三项。保留207558，不加预算/种子、不读25最终内容。完整报告/数组尚未收到，本地未推理；仅待已有CPU输出验证器核验归档，不启动新的GPU分支。见[结果与证据范围](BRIDGE_PREFIX_RESULT_209481_20261018.md)。

**209420工程预检通过，固定配对正式入口已准备（2026-10-17），尚未提交正式作业。** 用户日志/验证器确认旧4训练窗上下文和预测0差、两组各8更新/恢复检查通过；3.75GiB，暖更新0.172294秒。新15KB追加包复用不可变预检源码和新增视觉缓存，两组同207558、零请求投影、新Adam，各4096更新；主要比较新10演示/50完整16步泛化，K1仅辅助，25最终不读。代码CPU检查、实际数值/token入口、语法/ZIP通过，无本地真实策略训练或新泛化结果；207558保持主参照。见[正式范围和上传提交命令](BRIDGE_PREFIX_FORMAL_READY_20261017.md)。

**新增输入资格与唯一集群预检包已准备（2026-10-17），未提交作业。** 全1558后段输入/405训练演示及新10开发/50完整窗通过资格核对，25最终内容未读；只导出指定10开发记录，旧训练原数据复用。新包CPU入口已完整核对目标/当前状态/开度/固定归一化网格均0差；官方Example解析器交叉验证10开发记录，完整RGB/编码身份和源码SHA保护，非全归一化像素比较。88原始指令变体token IDs与同任务一致，集群仍按实际token身份复用语言。ZIP约24.8MiB含防重复提交入口；语法、导入、CPU输入、Bash和逐文件SHA检查通过，无权重/环境/CLIP缓存/最终25数据。仅一个A6000预检：真实207558SHA/旧4训练窗重放、1558新视觉编码、两组各8更新及权重/Adam恢复、资源估计；不评分开发或自动正式提交。实际GPU权重兼容/更新/资源尚待集群执行。见[发布包、明确范围与短提交命令](BRIDGE_PREFIX_PREFLIGHT_READY_20261017.md)。

**完整规模请求接口与一次配对设计完成（2026-10-17）。** CPU实际加载本地冻结CLIP及新初始化8层Adapter/原尺寸U-Net，原图像/token/207558上下文算式一致，K16原损失预测差0，K1/DDPM100输出有限且四元数单位化；Adapter/头/状态/历史/新请求有梯度，CLIP/夹爪无，全部参数未更新，内存权重格式往返通过。不是207558训练权重重放，无GPU/优化器/真实数据评价或新大PT。新协议两组均从207558继续各4096次，K16样本batch完全相同，K1仅旧1836vs全部3394，任务槽/预算/新接口一致；新增条件影响由新控制组承担，不重训旧参照。元数据选新10开发（预计50窗，资格未审）、沿用25未评分最终候选；主指标仍完整16步，候选须同时优于控制/207558且不过旋转/逐演示门槛。尚缺新增1558图像/标签核验、真实权重集群重放和唯一资源预检，无Slurm入口/作业提交。见[具体预算、采用规则与待办](BRIDGE_PREFIX_PAIR_PREPARATION_20261017.md)。

**后段窗口请求长度的CPU契约通过（2026-10-17），没有新训练/泛化结果。** 新隔离原型将调用方请求K=1/16与记录剩余长度分离：K1覆盖3394候选当前画面，K16保留原1836完整起点，两者都100次去噪；未请求槽在q、网络输入/输出及每次后验更新投影为结构零，最终不输出占位位姿。新增请求条件投影（1024上下文需16384参数），不是不变模型的单变量扩窗。随机小型三尺度U-Net上K16与存档sample AST的100次输入/上下文/输出及最终位姿全差0；K1不随可用未来1/15变化，哨兵污染不影响结果、无效输入梯度0，完整100步尾部保持零。未加载CLIP/Adapter/真实权重或读真实数据/评估标签，0更新、GPU禁用。生产集成/抽样预算/新划分仍待明确，不提交作业，不以K1误差替代原16步泛化指标。见[完整规则与验证范围](BRIDGE_REQUESTED_PREFIX_CONTRACT_20261017.md)。

**后段窗口CPU契约验证完成（2026-10-17），未启动新训练。** 隔离原型保留旧1836行，新增1558个部分未来起点共3394；全索引边界及两条固定训练记录（17/48帧）真实未来、夹爪对齐/末步排除、手算短轨迹、NaN哨兵和等窗口有效损失通过。旧完整标签最大差1.20e-7；全有效MSE/加噪算术差0，无效输出梯度0。但固定卷积反例显示无效输入仍影响真实输出，原U-Net有跨时间卷积/GroupNorm且无时间mask；训练时L已知也不能作为部署未来信息。训练集成ready=False，尚无采样/填充/未知长度一致契约，不提供GPU入口、不执行3394窗训练。未加载项目模型/权重、未读评估数值或解码图像，207558不变，无泛化改善声明。见[数据规则、证据及停止决定](BRIDGE_PARTIAL_FUTURE_CONTRACT_20261017.md)。

**207558整体训练方案审查完成（2026-10-17），没有新训练或性能结论。** 405条训练演示实际含13214帧，12809个时刻有真实下一帧；步长4且完整16未来的全部1836起点均已使用，不再是每演示两窗。4898个完整未来起点因stride未作输入，另6075个后段时刻因不足16未来未作输入；12204帧已作为未来目标出现，不能说只用14.33%的标签。夹爪冻结、任务抽样比例继承旧128窗，当前证据为同五任务的新演示离线位姿。下一项建议只准备同405演示后段有效未来/掩码的数据监督契约与CPU验证；尚未实现或授权新GPU训练，不认定尾部缺失为既有误差根因。不重复806扩容、追加预算、历史或Panda拟合。见[整体审查与具体边界](ACTIVE_TRAINING_SCHEME_REVIEW_20261017.md)。

**按用户指示回到原405演示有效基线207558（2026-10-17）。** 已固定主参照清单及原检查点SHA/配置/源快照引用，直接复用原权重和已核验结果，无新加载/推理/训练或Git重置。已完成13条独立演示位置4.766cm/旋转12.045度，对应不动7.779cm/12.742度；原40开发窗位置改善、旋转略差，两集合不混比。806及后续改进分支停止继续推进，暂停新GPU/微调实验，保留旧文件；后续先与用户确定端到端目标和协议，不自动重做拟合阶梯。见[当前有效参照](ACTIVE_BASELINE_207558_20261017.md)与[机器可读清单](../configs/active_experiment_reference.json)。

以下为此前记录。

**XYZ表示CPU审计完成并收尾，无新GPU/训练。** 806/10演示固定；复用1956姿态，从旧导出中仅提取1708训练输入姿态，不读E盘或解码图像。工具轴/基座轴任务先验开发窗口7.6874/7.7083cm、演示7.4074/7.3831cm，仅5/10演示改善，预设门槛失败，不启动坐标表示神经对照或修改标签。新变换和独立SciPy四元数/均值/计权核验通过，预测差5.77e-15cm；不是重跑旧几何审计。唯一根因仍未确定、没有新的模型泛化提升。见[完整结果](BRIDGE_XYZ_FRAME_AUDIT_20261017.md)。

以下为此前记录。

**806池任务先验CPU比较完成，未增加训练。** 一次固定任务均值（仅训练标签、无画面/状态）开发演示等权7.4074cm，既有8192/16170上下文读出5.4620/5.5722cm，两版8/10演示改善；独立one-hot重建0差。支持上下文优势超出任务平均动作，不能归因纯视觉或证明唯一头部故障；不重新拟合ridge。下一项拟做一次XYZ工具轴/基座轴相对表示的CPU结构审计，尚未执行、不直接提交GPU；保持原单位、未来16目标和当前状态信息，不重做几何往返或旧尺度实验。见[结果及下一范围](BRIDGE_806_TASK_PRIOR_20261017.md)。

以下为此前记录。

**208964完整核验完成，固定上下文干预已收尾。** 原真实预检验证器、源/协议/修复/7978日志通过；3664上下文与8192全部0差，float64位姿复算位置最大差2.48e-6cm、旋转0.000859度。开发窗口7.6989cm/15.7178度、演示7.3424cm/16.8857度，四项均劣于联合延长，仅2/10演示位置改善；不再追加预算/冻结时点/种子。保留405有效参照（旧开发窗口6.8082cm；已完成13独立演示4.7661cm，对应不动7.7790cm，评估集不同不混比）。最近806系列没有稳定提高最好成绩，不能把排查进展称为性能改善；不重复小样本记忆实验，不重读预留测试。见[完整结果](BRIDGE_FIXED_CONTEXT_RESULT_208964_20261017.md)。

以下为此前记录。

**固定上下文208964运行完成，终端改善门槛失败。** 用户报告COMPLETED 0:0，A6000，1:44:36。开发窗口7.6989cm/15.7177度，差于联合208586的7.1117cm/14.9675度；训练6.1746cm/14.1388度也更差。位置仍好于不动、旋转差于不动，不能称为完整位姿改善。四项合取门槛已失败，演示等权和全部保存工件仍待本地只读核验。关闭此干预、不追加预算/种子或读取预留测试；未确定唯一根因。见[初判及小文件下载](BRIDGE_FIXED_CONTEXT_RESULT_208964_20261017.md)。

以下为此前记录。

**固定上下文修复后正式对照208964已由用户报告提交。** 仅修复验证器模块解析，复用208897及原8192参数/Adam；同7978更新和泛化验收不变。运行状态、修复入口实际检查及最终开发成绩尚未收到，只跟踪此作业，不重复提交。见[正式记录](../reports/bridge_fixed_context_formal_submission_208964.json)。

以下为此前记录。

**208926正式入口导入错误已定位并准备修复（2026-10-17）。** 原fit延迟导入validate_preflight命中了旧目录同名模块，失败早于新增更新；旧上下文/预测0差，无泛化成绩。新6,905字节外层入口显式绑定原验证器，原fit函数体/协议/门槛不变；本地复现错误，实际原fit经原验证器后到达advance，测试停止于训练前，0更新/GPU。篡改拒绝、原文件及旧ZIP不变、启动参数/Bash语法/ZIP核验通过。复用208897，不重跑GPU预检或8192前缀；修复后作业未提交。见[修复记录与命令](BRIDGE_FIXED_CONTEXT_IMPORT_REPAIR_208926_20261017.md)。

以下为此前记录。

**806固定上下文正式对照208926已由用户报告提交。** 使用预检208897，启动器再次验证完整预检。直接恢复8192参数/Adam，只新增7978次扩散头更新；联合参照复用208586。运行状态及最终开发成绩尚未收到，不重复提交、不重训前缀或读取预留测试。见[正式追踪记录](../reports/bridge_fixed_context_formal_submission_208926.json)。

以下为此前记录。

**806固定上下文预检208897运行完成。** 用户报告COMPLETED 0:0，A6000，9分24秒；旧上下文/预测0差，三支首步损失一致，日志PASSED。完整报告未本地取得，正式尚未提交；下一步在集群用冻结验证器检查完整报告，通过才提交唯一正式7978步对照，不重复前缀或联合参照。此时没有新的泛化成绩。见[记录](../reports/bridge_fixed_context_preflight_submission_208897.json)。

以下为此前记录。

**806固定上下文预检208897已由用户报告提交。** 状态及完整报告尚未收到；正式未提交，不重复作业。先核验真实8192参数/Adam恢复、旧4窗重放、原联合首4步和冻结更新恢复检查，再判断是否放行正式泛化对照。见[提交记录](../reports/bridge_fixed_context_preflight_submission_208897.json)。

以下为此前记录。

**806固定上下文泛化对照已准备，尚未提交（2026-10-17）。** 直接复用208586保存的8192参数和Adam，联合正式参照也复用，不重跑前缀。唯一干预为原全局裁剪后屏蔽Adapter/历史/状态参数更新；保留原训练Dropout与上下文梯度计算，只有扩散头更新。当前806池新增7978步到16170，四项开发均值为主要验收，训练拟合不放行。32参数CPU探针、AST、启动参数与两脚本Bash语法通过，包23,720字节，无项目模型/GPU执行。下一步仅12操作更新的集群预检，验证真实恢复和冻结状态，不评分预留25条。见[协议、检查与上传命令](BRIDGE_FIXED_CONTEXT_GENERALIZATION_20261017.md)。

以下为此前记录。

**806池上下文CPU位置读出诊断完成（2026-10-17）。** 沿用旧128窗方法但未重跑旧实验，当前3624窗/806演示的8192与16170两版保存context各一次固定全局ridge，无调参/神经更新/GPU/预留测试。开发演示等权：不动7.960cm，原扩散6.690/6.816cm，读出5.462/5.572cm；两版通过预设位置边界门槛。均值采样核验显示随机散布解释部分差距，仍不能认定唯一头部代码故障；西兰花读出本身也退化。独立重建0差、正规方程残差<3.27e-12。原扩散模型没有因此得到改善，不代表旋转/夹爪/抓取成功。下一候选为当前806池固定8192上下文仅更新扩散头，对比已完成联合延长；尚未准备或提交新训练。见[范围、完整结果与限制](BRIDGE_806_CONTEXT_READOUT_20261017.md)。

以下为此前记录。

**西兰花预算退化只读诊断完成（2026-10-17）。** 复用旧原图/状态审计，分析两版保存预测，未重读E盘、加载模型或启动训练/GPU。当前同任务训练109演示/603窗仍7.388cm；开发第1目标1.755→2.358cm、路径8.516→10.849cm，采样均值MSE80.107→134.910cm²，预测平均位移9.504→13.014cm（真值9.201cm），退化不仅是末段或随机散布。144开发XYZ目标均在原405和806同任务标签轴范围内，但不证明条件分布覆盖。全部9窗含3窗改善，完整保留，生成约1.99MB两张核验图和数值。尚未定位唯一模块故障，不续训、不删任务或评分预留25条。见[新诊断与旧证据复用范围](BRIDGE_BROCCOLI_BUDGET_DIAGNOSIS_208586_20261017.md)。

以下为此前记录。

**806池预算对照208586已核验并收尾（2026-10-17）。** 12文件/27.9MB齐全，无PT；8192日志匹配原SHA，旧4窗上下文/预测/目标0差，新旧全目标一致。CPU复算通过；16170开发窗口等权7.1117cm/14.9675°、演示等权6.8162cm/16.5143°，相对8192四门槛均失败。训练位置5.944→5.327cm，旋转未改善。4/5任务位置下降，西兰花8.516→10.849cm且两演示均退化，是具体待解释现象，不能据此认定任务干扰或数据错误。报告loss/parameter数值字段被上下文重放摘要覆盖；损失可由精确日志验证，参数仅有冻结源码门槛证据，不冒称本地逐参数重放。停止该预算分支，不续训/扫参/评分预留25条，保留405参照。见[208586结果](BRIDGE_DEMO806_BUDGET_RESULT_208586_20261017.md)。

以下为此前记录。

**806池训练预算对照已准备，未提交（2026-10-17）。** 用户接受一次16170更新的独立新实验；前8192批及4096降学习率时点保持，之后新增7978次1e-5更新。208207无Adam状态，故必须重建其8192更新，在全部损失/参数/旧4窗重放通过后才延长；复用缓存，不重做8192处全池评价。约304KB代码包，无数据/权重。采样、三函数AST、命令参数、Python/Bash语法和9参数CPU Adam恢复检查通过，未运行项目模型/GPU。集群先12次更新的优化器连续性预检，再一次正式终点评价；不自动延长到其他预算。见[方案与上传命令](BRIDGE_DEMO806_BUDGET_CONTROL_20261017.md)。

以下为此前记录。

**806演示固定预算对照208207已完成并本地复核（2026-10-17）。** 7文件齐全，源码/协议与旧目标身份一致，CPU复算通过。开发集窗口等权位置6.808→6.874cm、旋转15.139→14.504°；演示等权位置6.450→6.690cm、旋转15.213→15.588°，未通过事先四项均不变差的验收。10演示中位置改善4条、旋转7条，一条关抽屉明显退化导致均值计权趋势不同。相同旧405训练样本位置5.110→5.942cm，不能仅归因新样本较难。保留207558参照，本轮收尾，不自动加更新/扫参数或动预留25条；未运行新模型推理，完整分析见[208207核验](BRIDGE_DEMO806_RESULT_208207_20261017.md)。

以下为此前记录。

**806演示集群预检包已准备，尚未提交（2026-10-17）。** 仅401条新增训练记录导出为13份小包，含代码共约658.5MiB，无权重；原数据和旧缓存不覆盖。本地实际记录/编号/源SHA、全部任务槽位、训练循环/上下文AST及Bash语法通过，无模型或GPU执行。集群预检将验证3664输入窗、复用旧1876 token行、编码新增1788行，先做207558旧4窗严格重放，再做16次运行与资源检查。新25条预留数据未导出或评分，正式训练未启动。只提交一次预检，报告通过后才正式。见[上传和预检命令](BRIDGE_DEMO806_CLUSTER_PREFLIGHT_20261017.md)。

以下为此前记录。

**同任务新增训练演示盘点和CPU源检查完成（2026-10-17）。** 复用256分片元数据、扫描剩余768，发现1470条未纳入原清单的候选。按固定规则选405条新增训练候选，401条通过，4条原夹爪标签规则不适用且未补选；有效池为806演示/3624窗。原10开发演示不变，已用13条验收数据排除，新25条预留候选仅保留元数据、未读取目标或评分。8192×8的任务槽位完全保持、平均每窗18.08次。没有训练、GPU推理、源数据导出或新作业。下一步仅准备动态规模入口及一次资源/重放预检。见[盘点和对照方案](BRIDGE_DEMONSTRATION_EXPANSION_20261017.md)。

以下为此前记录。

**208056完整结果已本地复核。** 11文件齐全，CPU独立复算全部总体/演示/任务/目标指标，与报告最大差2.54e-13；旧4窗上下文/预测均0差异。coverage位置13/13演示、5/5任务优于同批不动，旋转6/13演示、3/5任务；旋转加权位置12/13、旋转10/13演示更好，任务分别5/5与4/5。两模型的演示等权与窗口等权平均位置/旋转均优于同批不动，但西兰花任务旋转加权整体接近不动、coverage旋转更差。范围仅同任务新演示离线泛化，无抓取成功结论。详见[完整核验](BRIDGE_HELDOUT_RESULT_208056_20261017.md)。本次评估收尾，不重跑、不用已消耗测试集调参。

以下为此前记录。

**208056同批基线已收到（用户从报告提取，演示等权）。** 静止7.779cm/12.742°；coverage4.766cm/12.045°，位置/旋转平均误差约降低38.7%/5.5%；rotation_weight6.094cm/10.264°，约降低21.7%/19.4%。两组均有新13条独立演示上的平均优势，范围为五个已见任务的离线预测。完整报告/数组尚待本地复核，尚不声称逐任务全面改善、统计显著或抓取成功。下一步只核验已有证据，不重跑或据此调参。详见[208056结果](BRIDGE_HELDOUT_RESULT_208056_20261017.md)。

以下为此前记录。

**208056已完成（用户终端证据，COMPLETED/0:0，3分15秒，A6000）。** 两模型旧4窗上下文及预测重放差均为0。新13条演示/60窗已评分：按演示等权，coverage路径4.766cm/12.045°，rotation_weight路径6.094cm/10.264°。同批静止基线尚未贴出，完整报告和数组尚未本地复核，因此暂不判断是否优于静止基线。新集合已使用，禁止重跑或据此调参；下一步仅读取现有报告并核验保存证据。下方提交和诊断状态为历史记录。

**修复后独立留出评估已提交208056（用户编号）。** 状态与成绩尚未收到；只跟踪这一作业。先CPU核验208034实际保存证据，再运行原入口和原1e-4重放门槛，之后才评分新13演示/60窗。两检查点、数据、协议和采样不变，无训练。日志`/projects/Zeroshot/logs/bridge-heldout-fixed-208056.out`，结果目录`/projects/Zeroshot/runs/bridge-heldout-fixed-208056`。

**208034完成并定位新入口的上下文执行方式差异（终端配对证据）。** 两模型在原no_grad/参数标志下，缓存或重新编码均上下文/预测差0；失败方式分别产生0.000116825与0.000137717预测差。原门槛未变，这不解释历史泛化误差。约7KB代码包装已准备：先在集群复核208034全部16数组，再恢复原上下文作用域，保留原入口/协议/两个检查点/严格重放和原子claim，之后才评分新13演示。尚未提交修复作业，也无新泛化成绩。见[定位和修复命令](BRIDGE_HELDOUT_CONTEXT_REPAIR_20261017.md)。

**旧窗口一致性诊断已提交208034（用户编号）。** 当前未收到Slurm状态或诊断报告；只跟踪该作业。仅旧4窗八预定组合，分离保存上下文/旧缓存/重新编码、计算模式与状态数组算术，精确相同输入复用采样。原门槛和两个模型不变，不训练、不评价新13演示。日志`/projects/Zeroshot/logs/bridge-replay-diag-208034.out`。

**208005在旧4窗重放门槛处失败，新13演示尚未评分（源码路径推断，实际报告待核对）。** 最大动作张量差0.000116825超过原0.0001；不放宽门槛或重提原作业。确认源码存在缓存/重新编码及no_grad/梯度标志/inference_mode差异，尚未证明谁导致失败。约9KB仅旧4窗诊断包已准备，八预定组合保留分量证据，精确重复上下文复用采样，无新测试/训练/环境/权重下载。见[失败记录与诊断命令](BRIDGE_HELDOUT_REPLAY_FAILURE_208005_20261017.md)。

**独立留出评估已提交208005（用户编号）。** 当前尚未收到Slurm状态或评分；只跟踪这一作业，不重复提交。先通过207558/207746的旧4窗恢复重放，才评价新13演示/60窗；零训练更新。日志`/projects/Zeroshot/logs/bridge-heldout-eval-208005.out`，结果目录`/projects/Zeroshot/runs/bridge-heldout-eval-208005`。

以下尚未提交的描述为准备时历史。

**13条新演示一次性泛化验收已准备，尚未提交。** 冻结207558与207746检查点、DDPM100/三采样/batch2；原预留13演示共60窗，与405训练/10开发在记录、原始来源加演示号、完整payload均无重叠。先冻结协议再CPU解码核验，全部保留；没有新模型预测、训练或GPU执行。一个集群作业内先校验两个模型的旧4窗重放，均通过后才评分新60窗及同批静止基线。约56MB包，不含权重或新缓存，不单独排预检/训练。评分后不再用此集合调参。见[固定协议与上传命令](BRIDGE_HELDOUT_EVALUATION_20261016.md)。

**207746/207722已独立验收完成。** 21文件/7预测、源码/缓存指纹/415演示隔离/初始化42/8192损失日志与预检恢复核验。原40开发7.301274cm/12.275860°，第16目标8.653703cm/16.242081°；相对207558旋转改善18.91%、位置退步7.24%。8/10演示旋转优于原模型，五任务旋转均改善；位置仅4/10演示、1/5任务改善。两项总体均优于静止，但原位置保护未达，完整候选未通过。固定系数分支收口，不扫权重/加种子/拟合；无新作业或NN执行。见[独立结果](BRIDGE_ROTATION_WEIGHT_RESULT_207746_20261016.md)。下文待下载/待验收为历史。

旋转权重正式对照207746已提交，用户由条件验证/提交命令返回编号；预检207722验证器成功依该命令分支记录，完整工件仍待独立核验。暂无正式状态或开发成绩，仅跟踪此一次8192更新，复用207558控制与完整缓存，不扫系数/加预算。见[当前状态](BRIDGE_ROTATION_WEIGHT_CONTROL_20261016.md)。

旋转损失完整集群预检207722已完成（用户日志）：0:0、4分34秒/A6000，常学习率与恢复同学习率首损失和探针一致，原入口16操作结束。无新开发评分或正式编号；先用原验证器核对报告，通过再提交一次8192更新正式，复用207558控制与现有缓存，不重复本地/集群预检。见[当前状态](BRIDGE_ROTATION_WEIGHT_CONTROL_20261016.md)。

旋转权重对照完整预检207722已提交（用户编号）。状态与完整报告尚未收到，无正式作业或新泛化指标；验证原包完整预检后再提交唯一8192更新正式组，不重复本地验证或207558控制。见[当前命令](BRIDGE_ROTATION_WEIGHT_CONTROL_20261016.md)。

旋转损失对照包已准备：训练标签静止分量能量比35.444971为唯一权重，数据/初始化42/8192更新/采样不变，207558控制复用。CPU损失/梯度/全训练统计和局部5实际更新/2预测核验通过；本机GPU争用导致完整局部16步和恢复未完成，已停止自身验证进程并保留证据，不能把它记作完整预检通过。当前仅准备197313字节包供集群完整预检，尚无新作业/开发评分。见[旋转权重协议与严格位置保护](BRIDGE_ROTATION_WEIGHT_CONTROL_20261016.md)。

207558覆盖复核独立验收完成：17文件/7预测齐全、输入/源码/缓存指纹/初始化42/8192更新和预检恢复核验。开发6.808208cm/15.138891°，第16目标8.221684cm/19.032170°，9/10演示位置优于静止、8/10优于小池，五任务位置均改善；旋转仅2/10演示优于静止。两个初始化的位置方向重复，完整原候选因旋转门槛未复现；不改门槛/追加种子/预算。仅CPU保存预测复算和标准绘图，无NN执行或权重/缓存下载，无新作业。见[验收与两初始化图](BRIDGE_COVERAGE_REPLICATION_RESULT_207558_20261016.md)。

覆盖复核207558已完成（用户终端）：原40开发6.808208cm/15.138787°，同seed42小池8.381655cm/15.993918°，位置改善方向与seed43重复；比静止位置改善17.10%，旋转高0.040287°且超原门槛0.020287°，不能称完整候选复现通过。不变更门槛、追加种子或拟合，先下载207558/207521两run小工件独立核验；当前无源PT/全缓存下载或新NN执行。见[终端结果与边界](BRIDGE_COVERAGE_REPLICATION_20261016.md)。

覆盖复核207521完整验证器通过，正式207558已提交（用户终端证据）。固定初始化42大池1836/40、8192更新；复用旧小池202463及完整缓存，资源估计1.498208小时不含排队。当前正式状态和开发成绩尚无，只跟踪这一个作业；8192训练更新后仍需完整训练/开发采样评价，不追加拟合或重提作业。见[当前复核](BRIDGE_COVERAGE_REPLICATION_20261016.md)。

2026-10-16覆盖复核完整预检实际编号207521已完成：0:0、5分22秒/A6000，常学习率和恢复组日志一致。20752为漏位记录已指向207521。没有正式训练或新开发指标；原验证器通过后才提交唯一正式组，不重复预检/207348/202463。见[复核状态与协议](BRIDGE_COVERAGE_REPLICATION_20261016.md)。

2026-10-16覆盖独立初始化42复核包已准备：核对并复用小池202463/real已有完整审计与工件；新大池仅参数初始化变更，1836/40、8192批索引/噪声/Dropout/学习率/状态统计不动，已有1.88GiB完整缓存要求复用。局部预检224输入/16优化/5快照、参数起点/恢复/冻结检查与CPU独立复算通过，96.890秒/5.658GiB。zip194670字节，未提交作业或获得新泛化结果。见[复核协议与命令](BRIDGE_COVERAGE_REPLICATION_20261016.md)，不重跑207348或小池控制。

2026-10-16训练覆盖207348独立验收完成：17文件/7预测及源码、初始化、输入/415演示payload、训练任务槽/8192日志、预检恢复核对；float64开发7.079412cm/14.559847°，第16目标8.943367cm/18.480202°，优于旧控制与静止。7/10演示位置优于两者，达到原预设候选；旋转仅3/10优于静止。单一训练初始化、复用开发、覆盖和曝光共同变化，不能唯一归因或宣称抓取成功；无新NN执行。下一项规划一份独立初始化大池复核，先核查seed42小池202463以复用，不立即追加曝光匹配长训练。见[验收与边界](BRIDGE_TRAINING_COVERAGE_RESULT_207348_20261016.md)。下文终端待核验为历史。

2026-10-16视觉对照206274独立复核完成：27文件归档逐字节一致、13预测快照/8192日志/128与40身份/初始化与状态统计/实际源码通过，预检与正式前4步日志精确一致。开发6.966cm/17.396°，8/10位置优于静止、6/10优于原图像；旋转主要退步在关抽屉29.330→44.598°，开抽屉位置与旋转都改善。第16目标8.983cm/22.897°也呈位置改善、旋转退步。训练8.437mm/3.160°、51/128窗通过；完整候选依原判据结束，不改任务或追加训练。说明视觉条件作用存在任务/输出差异，不唯一定位CLIP/Adapter等故障；无新模型执行或PT下载。见[独立结果、CSV与科学图](BRIDGE_VISUAL_VARIATION_RESULT_206274_20261016.md)。下文终端待复核记录为历史。

2026-10-16视觉对照206274收到终端完成：0:0、37分20秒/A6000/8192更新。固定视觉组原40开发位置6.966cm优于静止8.212及原完整图像8.349，分别改善15.17%/16.56%；旋转17.396°退步，完整候选未通过。训练8.437mm/3.160°、门槛False不作为继续拟合理由。按计划结束，只取回排除PT的小报告/预测核验身份、实际均值、日志和逐演示；不追加训练/种子/视觉强度。该部分位置收益还不能唯一归因模块或宣称完整泛化修复。见[终端结果与边界](BRIDGE_VISUAL_VARIATION_RESULT_206274_20261016.md)。下文准备阶段记录为历史。

2026-10-16视觉输入变化对照已准备：查重确认分布/近邻旧检查不再执行；状态岭读出6.28cm与固定权重图像交换不能替代训练时视觉输入消融。新组复用原202278控制、相同128/40与8192预算，所有查询用原128训练token均值，语言/真实当前状态/目标/DDPM/参数集合不变；只检验该输入变化的净泛化作用，不作为最终无视觉策略。实际函数与均值无留出信息、原始身份、本地16入口更新恢复和5快照复算通过（约103秒/3.785GiB），无新编码/下载/开发评分；10,022字节包已准备，尚未集群提交或正式训练。固定原40窗位置/旋转/逐演示及停止规则，不追求追加拟合。见[协议、证据和命令](BRIDGE_VISUAL_VARIATION_CONTROL_20261016.md)。

205997小结果独立验收完成：32文件齐全、归档逐字节一致，22预测快照/6144更新/16内部选模/输入标签与96窗统计/整演示隔离通过。真正选中768步，最终6144训练1.660mm/0.531°、96/96拟合；早期768训练4.670cm/10.735°。原40外部最终8.135cm/15.422°、选中8.236cm/15.645°，主要判据失败；较好内部权重未迁移成更好外部路径。浮点复算最大位置差1.52e-6cm、旋转0.00196°，不改变门槛/选中时间点。未发现这次选模实现/身份/统计/评分错误，不能排除其他过拟合或唯一归因。无新模型加载、训练、编码、采样、仿真或PT下载；此规则收口，不重复或追加实验。见[完整结果与曲线](BRIDGE_INNER_SELECTION_RESULT_205997_20261015.md)。下文终端待复核记录为历史。

205997内部选模终端结果已收到：COMPLETED/0:0、33分22秒/A6000。同次最终权重外部40窗8.135cm/15.422°，内部选中8.236cm/15.644°，未通过冻结的两项主要误差改善判据。选中模型终点有改善，但不能事后改判据；该规则按6144更新预算结束，不追加种子/时刻/训练。完整工件尚未下载，下一步仅核对16次内部选择、具体选中步数、96训练窗误差与保存数组。见[终端数值与边界](BRIDGE_INNER_SELECTION_RESULT_205997_20261015.md)。下文入口未提交阶段记录为历史。

内部选模固定对照入口已准备：96窗/48演示实际优化，32窗/16演示仅内部选模，原40窗/10演示锁定权重后才评分；原直接x0/DDPM100和状态端口保留，停止残差候选。训练统计只拟合96窗，抽样及隔离构造检查通过；本地16次操作更新、监测前后参数/Adam/损失精确重放、选中权重重载采样差0，独立数组复算通过。约171秒/3.924GiB，无外部评分，临时权重已移除；原冻结token缓存复用，没有下载。15,796字节上传包与两份Bash语法检查通过，尚未提交集群或正式6144更新。原128窗与新96窗结果不能作单因素因果比较，只比较同次选中和终点。见[操作、边界与停止规则](BRIDGE_INNER_SELECTION_RUN_20261015.md)。

205648下载后独立验收完成：27文件/13预测快照及8192更新核对通过，原128/40数据/初始化/目标、源代码、基础预测与完整还原一致，预检恢复精确，正式复用缓存（张量未下载）。开发7.1846cm/16.6972°，8/10演示位置优于旧扩散、7/10优于静止，旋转仅4/10优于旧模型；完整事前候选因旋转退步失败。训练128/128通过，基础5.460cm→加修正1.419mm，但开发基础6.280cm→加修正7.185cm，当前修正迁移不足；不能唯一定位Adapter或U-Net。最终训练/开发还原XYZ裁剪计数0，保存预测逐元素重建差0；没有新训练/编码/推理/权重下载或仿真。按预算收口，不扫修正比例/种子或加轮数。见[完整结果和逐演示证据](BRIDGE_STATE_PRIOR_RESULT_205648_20261015.md)。下文待下载/复核记录现已完成。

205648用户终端结果已收到：COMPLETED/0:0，gpu-a6000-1，37分钟，8192更新。训练1.4186mm/0.6322°、门槛True；原40开发窗口7.1846cm/16.6971°。位置优于静止8.2123cm约12.5%、旧扩散8.3491cm约13.9%，但旋转劣于旧15.2560°且超过预定候选容差；位置仍比固定状态基础6.2798cm差。完整数组/报告尚未本地取得，逐演示、终点与独立重建待复核；日志均值成立时完整候选未通过。按固定预算停止，只取回小结果，不加轮数/比例/种子，不下载PT或缓存。见[结果、限制及下载命令](BRIDGE_STATE_PRIOR_CANDIDATE_20261015.md)。

用户返回“报告校验通过，正式编号205648”：集群端预检205618关键报告和源码身份断言已通过，唯一状态基础＋扩散残差正式候选205648已提交。尚未收到该正式作业状态/结果，无新拟合或泛化结论。下一步只查看205648，保持8192更新及原128训练/40开发协议，旧202278对照不重训；不重复提交。见[固定作业与查看命令](BRIDGE_STATE_PRIOR_CANDIDATE_20261015.md)。下文未提交状态保留为历史。

用户贴回205618完成：COMPLETED/0:0，gpu-a6000-1，用时4分03秒。短预检常学习率/恢复同学习率的首损失与最终指标一致，报告已保存；gate False不是正式泛化结论。尚未取得完整JSON独立复核，也未收到正式作业编号。下一步给出集群端断言执行/恢复/冻结/零基础身份及源码指纹后提交唯一正式作业的命令，提交号落盘避免重复；不重跑预检或旧对照。见[205618完成记录与正式提交守卫](BRIDGE_STATE_PRIOR_CANDIDATE_20261015.md)。

用户已提交单个状态基础＋扩散残差预检205618，尚未收到Slurm状态/日志。当前不能判定集群执行或通过；只检查该作业，不重复提交，也不提交正式训练。见[固定作业号及查看命令](BRIDGE_STATE_PRIOR_CANDIDATE_20261015.md)。下文未提交描述为准备时记录。

2026-10-15单个状态基础＋扩散残差候选已准备并完成本地入口预检：只将XYZ监督改为减去训练集固定状态读出、生成后加回，原8层Adapter/扩散/状态输入、128/40划分、初始化43、8192预算及四元数监督保留，夹爪冻结。202278原控制直接复用，不重训。全168输入、零基础前向差0、16次操作更新含4次同学习率恢复重放、参数/损失/预测精确一致、5保存数组还原核对通过。当前状态只用观测时刻，没有未来输入；减/加回目标最大差约1.2e-6cm，训练残差幅度1.7704未超±3界。约103秒/3.934GiB，无本地权重或开发评价。172MiB完整token缓存已保留，正式可复用；上传ZIP45,164字节无图像/权重/环境，Bash只读语法检查通过。集群和正式泛化尚未运行；停止规则要求原固定40开发位置/旋转及逐演示收益，不能靠训练记忆判定成功。见[协议、验收与唯一预检提交命令](BRIDGE_STATE_PRIOR_CANDIDATE_20261015.md)。

2026-10-15按用户要求回到泛化，撤回未执行的Panda特征对齐计划。新增固定CPU辅助读出检验，使用已有Bridge128训练/40开发特征、当前状态与训练标签，按任务岭回归正则1.0，输入/标签与原目标一致，没有正则扫描或神经训练。开发路径位置：状态6.280cm、CLIP加状态6.239cm、学习后条件加状态6.823cm，均9/10独立演示优于静止8.212cm；原扩散8.349cm保持原样。状态读出五任务平均均优于静止；窗口与演示等权结论同向。主求解/SVD及独立核eigen数值验证通过，当前状态没有未来输入，所有统计仅用训练窗口。说明该划分已有可泛化的位置关系、不能再只追求已见记忆，但不同读出不唯一定位Adapter或U-Net。旋转/夹爪/物理不评价，未触及预留测试；约0.83秒/小结果约0.95MiB。此固定分析结束，不扫参数或启动新拟合。见[完整结果与边界](BRIDGE_GENERALIZATION_READOUT_20261015.md)。

2026-10-15条件处理联合训练验证完成，候选未通过、不采用：只新增一个1024更新组，原样联合控制直接复用，无旧控制重训。训练路径位置42.926→39.707mm（改善7.50%），旋转0.673→2.507°变差，两个组均0/16三采样一致达标。此前冻结Adapter条件下63.39%的收益没有在这套动态统计联合实现中重现，尚未确定唯一根因。初始参数、首batch原样损失及全部1024步实际噪声/时间步核对通过；最终统计只来自训练观察，独立CPU复算、新进程权重/统计重载通过。完整16条CLIP token已缓存，避免后续重编码。无开发/测试或物理回合，原整任务留出抓取0/12不变。约311秒/峰值3.025GiB，最终权重约502MiB及token约15.5MiB，无下载；到预算停止，不继续统计/步数/种子扫描。见[最终结果与证据](PANDA_JOINT_CONDITION_TRANSFORM_20261015.md)。

2026-10-15条件处理缺失控制已补完：仅新训原样Adapter条件组1024步，直接复用已完成处理组，无旧组重训/推理。相同初始化头和全部1024步实际噪声/时间步、样本/目标/学习率一致，独立复算与权重回读通过。原样10.465mm/1.296°、2/16窗口；处理3.832mm/0.951°、14/16，位置下降63.39%，15/16窗改善。条件向量居中/尺度处理是本固定冻结头设置的可训练性因素，两组仍未过整体门槛，不等于动作/图像归一化bug、原联合训练修复或唯一根因；未做夹爪/开发/测试/物理回合，原抓取0/12不变。一份缓存头约297MiB，约79秒/1.598GiB，无下载。到1024预算停止，不再扫描处理变体，生产代码未改。见[单组协议、完整结果及边界](PANDA_RAW_CONDITION_CONTROL_20261015.md)。

2026-10-15冻结条件U-Net配对已完成：缓存16初始训练窗，CLIP池化/初始专项Adapter两组，相同新77,670,023参数U-Net、每组1024更新/每窗512次、Adam分段学习率、原生DDPM100；只更新位姿头，条件共用训练集居中与标量RMS处理，不编码图像或前向/优化CLIP/Adapter。全部1024实际含噪输入/时间步哈希相同、初始头指纹相同，最终权重张量回读及首batch预测差0，独立复算通过。CLIP0.301mm/0.252°、16/16门槛通过；Adapter3.832mm/0.951°、14/16，因两窗6.83/17.28mm未过90%一致性门槛。该任务现有原生扩散训练成功参照，Adapter表示也非完全丢信息；与原联合训练有冻结、条件处理和头初始化等差异，未唯一归因。无夹爪训练、开发/测试或新物理回合，原抓取0/12不变。两组各约79秒/显存1.598GiB，缓存头总约595MiB。按预算结束，不重跑；如继续条件处理，只新增原样Adapter组复用已完成处理组。见[配对协议、结果和边界](PANDA_FROZEN_CONDITION_HEAD_PAIR_20261015.md)。

2026-10-15按用户避免重复要求，先核对旧Bridge余弦近邻及Panda图像交换记录，再只复用已有16初始训练观察的3种1024维表示做固定岭读出。无模型加载、重编码、神经训练、开发/测试或新物理回合。CLIP/原混合Adapter/专项Adapter全16拟合路径0.00775/2.282/1.377cm，辅助逐次留1为3.253/4.492/2.907cm，目标均值5.836cm，错配控制7.538/7.890/7.740cm。对偶/SVD相差1.77e-13m、保存预测独立复算与身份指纹通过。CLIP已见可拟合，Adapter差异被首主成分主导，但固定正则会影响小方向，未唯一证明信息丢失；辅助留1未从Adapter原监督训练中排除此样本，不是策略泛化。按单读出规则停止，不扫超参数。见[报告与限制](PANDA_FEATURE_READOUT_20261015.md)，后续先查[避免重复索引](DIAGNOSTIC_EVIDENCE_INDEX_20261015.md)。

2026-10-15初始接近专项能力检查完成：原16位置起点0、相同初始化43、每窗512抽取、1024更新（512次1e-4后512次1e-5），只移除其他阶段，未改正式全阶段模型。原初始7.152cm/0.606°→新4.293cm/0.673°，终点10.942→6.508cm，仍0/16三采样一致达标；静止5.520cm。完整CPU复算、权重重载和16窗3采样精确重放通过，CLIP不变，约221秒/峰值保留3.018GiB，一份最终权重约502MiB。全打开目标100%同一直打开，不证明开合能力；无开发/预留测试或新物理回合。按1024预算停止，不追加更新或种子。说明移除其他阶段有部分帮助，但仍未建立可靠初始几何映射，唯一故障模块未确定。见[专项协议、结果与边界](PANDA_INITIAL_ONLY_FIT_20261015.md)。

2026-10-15 Panda初始接近冻结诊断完成：16初始+16已对准训练观察；独立CPU复核通过，参数不变、0更新、无开发/测试目标或新物理回合。原生3采样32窗最终预测与正式结果逐元素差0，初始x0估计4.706→6.716→7.152cm，已对准0.260→0.182→0.128cm。正确图像纯噪声直接读出4.527cm，换同阶段图像6.557cm，说明图像并非完全忽略，但训练接近映射不足。后续阶段同位置相对目标几乎一样，其他15演示同阶段均值即可很准，不能把这些阶段拟合当成视觉定位证据。目标重建差1.20e-8m，缓存/在线上下文最大差1.43e-6，没有支持典型索引/尺度/缓存大错，唯一层归因仍未确定。见[冻结范围、真实图像、配对链及限制](PANDA_INITIAL_APPROACH_PROBE_20261015.md)。

2026-10-15原生Panda新位置固定实验完成且独立核验：8192更新、初始化43、16演示128训练窗与4演示32开发窗，最终权重重载一致、CLIP不变、Adapter/扩散头/夹爪均改变。离线训练/开发路径位置误差1.339/1.255cm，静止3.959/3.794cm；夹爪准确率与平衡准确率100%，没有释放目标。但原位姿拟合门槛未通过，4位置各3次真实学习策略闭环全部失败（0/12）。初始训练观察的接近动作误差7.152cm，多个后续专家观察阶段才为毫米级；不能只归为留出泛化差，也未唯一定位故障模块。按规则结束本轮，不追加训练或种子，预留测试未使用。一份最终非CLIP权重约502MiB，峰值保留3.135GiB，本地训练约27.93分钟，无下载/安装/集群提交。见[完整数值、阶段拆分、独立验收及真实失败动画](PANDA_SPATIAL_FIXED_TRIAL_20261015.md)。下文为此前准备阶段状态。

2026-10-15下一阶段原生Panda最小对照完成前置验收：固定相机、机器人和方块，仅变初始XY，16训练/4开发/5预留测试。第一版±2cm开发位置盲中心也4/4成功，故正式学习前停止该版；第二版±4.5cm开发、训练网格到±9cm，专家20/20，盲中心0/4、静止0/4，4条原生命令和未来实测位姿重放均通过。新数据T+1观测/T命令与夹爪索引明确，原始轨迹及成功判据独立复算；本地GPU6更新联合位姿/夹爪预检通过、CLIP不变，未做正式训练或学习模型泛化评价。CPU包装入口曾隐藏GPU、在0更新时失败，已改独立GPU入口并保留记录。两版数据约203MiB，无新大权重、依赖或集群作业。见[冻结协议、准备结果和边界](PANDA_MINIMAL_GENERALIZATION_PREPARATION_20261015.md)。

2026-10-15端到端逻辑核对完成，无新训练或采样。168窗原始字段与保存目标身份一致，独立世界位置重建最大差3.23e-8米，所有窗口均有16个真实目标；未找到解释全部误差的索引/坐标大bug。开发第1目标1.439cm已差于静止1.280cm，统一世界偏置仅占位置均方误差3.08%，±1/2帧对齐也未恢复训练精度。128窗结果仍只证明位姿拟合，夹爪被冻结；物理同步与Bridge输出到仿真控制映射没有独立闭环证明。当前五任务追加训练仍停止。见[本次证据、来源目录与范围](BRIDGE_END_TO_END_AUDIT_20261015.md)。

2026-10-15当前Bridge诊断阶段按事前规则收口。204258下载17文件齐全、约10.48MiB，独立复算通过；控制8192损失及初始/最终预测、目标、上下文精确复现202278。两组训练门槛均通过（128/128、127/128），控制/真实历史开发8.349/8.749cm、15.256/15.614°，位置增加4.79%，只有4/10演示位置改善；真实历史可用30窗也从8.582升到9.025cm。位置10%/旋转不恶化/6演示候选条件均失败，不启动初始化42复核，不追加当前设置的参数或训练。仍未唯一定位CLIP/Adapter/动作头/数据覆盖根因。已完成[本阶段收口总结](BRIDGE_DIAGNOSTIC_PHASE_CLOSE_20261015.md)，下一阶段只准备单任务同场景、完整演示留出的最小泛化协议，未选定新划分或训练。见[配对完整验收、图与范围](BRIDGE_HISTORY_GENERALIZATION_RESULT_204258.md)。

2026-10-15历史正式对照204258已收到COMPLETED/0:0，A6000，1小时30分29秒。真实历史组8192更新，训练1.965mm/0.542°、原拟合门槛通过，开发40窗8.749cm/15.614°。本次实际控制值待下载，不能用旧值冒充；但包装器正常退出要求新控制位置在旧8.349cm±0.1cm内，故真实历史已不可能达到≥10%位置改善筛选条件。这项方案按事前规则收口，不增加更新/帧间隔/融合宽度/种子，不启动初始化42复核。下一步仅下载两组报告、共同初始及最终预测、源码快照和更新日志（不含权重）做独立完整验收。见[终端证据、边界和下载命令](BRIDGE_HISTORY_GENERALIZATION_RESULT_204258.md)。

历史GPU预检204150已收到COMPLETED/0:0、12分18秒、A6000的用户日志。包装器正常结束意味着冻结入口中的原控制初始重放、共同起点、前16控制损失/真实历史梯度及两组接口保护断言通过，具体差值仍待读小JSON，尚未独立复算预检输出。短32步`gate False`不是正式拟合/泛化结果；无新效果结论。下一步命令先打印预检关键字段，只有passed时提交已冻结的一组8192更新/臂正式对照；仍遵守10%/旋转/6演示/训练门槛及阴性收口规则，未实际提交正式作业。见[状态、报告门槛和有条件提交命令](BRIDGE_HISTORY_GENERALIZATION_BOUND_20261014.md)。下文未提交预检为准备时状态。

有界历史对照准备完成：用户同意当前设置只保留这一项诊断。保持202278的128训练/40开发与真实状态输入；原74条演示、42分片、168窗的历史/当前原始身份及处理核验通过，有真实t-1的训练64/开发30窗，其余64/10在起点重复当前，94个可用历史图像均不完全相同。新加载器实际168窗检查、原训练核心函数AST一致、入口导入/参数/Bash与ZIP复读通过。50,644字节小包复用已有资产；只准备25分钟GPU预检，不保存checkpoint，尚未提交或正式训练。候选要求全40位置改善≥10%、旋转不恶化、≥6/10演示位置改善且两组拟合/原控制重放通过；否则结束当前设置追加诊断，不扫描其他帧间隔/参数。见[最终有界协议和预检命令](BRIDGE_HISTORY_GENERALIZATION_BOUND_20261014.md)。

204017两个结果齐全、共387,847字节，独立CPU复算通过：原40窗精确重放差0，固定索引/目标/融合条件及所有逐窗、原始/裁剪x0阶段指标一致，0次优化。9/9留出窗首次位置错误已达最终90%以上，首次2.638–15.424cm、末次2.596–15.796cm；单窗链前后误差净变化绝对值最大0.454cm，训练条件最大窗口×采样×阶段均误差4.49mm。两条演示最终位置8.350/13.893cm，自身静止8.604/10.395cm；第二条更严重，两条旋转均差于静止。记录阶段XYZ没有范围裁剪，首次浮点转换最大每轴差0.002374cm，不能解释10cm错误。本组首次新条件读出不足已确认，但CLIP/Adapter/动作头唯一根因未分离；这一生成链诊断收口，下一项范围为真实历史vs重复当前的共享观察信息对照，先核查输入边界，无新包或训练。见[完整复算、图和下一项范围](BRIDGE_PROJECT_CONDITION_RESULT_204017.md)。下文待下载为历史状态。

项目条件诊断204017已收到COMPLETED/0:0、3分09秒、A6000的终端结果：原40窗三个采样种子重放均差0。固定9对西兰花观察，留出条件干净轨迹估计首次10.056cm/12.347°、末次10.197cm/12.394°；训练近邻条件2.027mm/0.730°→1.819mm/0.632°。本组误差主要在首次条件读出已出现，后续链净增仅0.141cm，不支持后续链破坏准确起点是本组主因；配对条件明显改变输出，不是完全无条件响应。不能定位CLIP/Adapter/动作头唯一根因或宣称泛化修复。尚待report/NPZ下载及逐窗独立复算；下一步仅取回两份小结果，不重训。见[终端证据与下载命令](BRIDGE_PROJECT_CONDITION_RESULT_204017.md)。下文未提交为准备阶段状态。

项目202278真实状态条件→扩散头冻结诊断已准备：先按原batch=2、3采样种子重放全部40开发窗，再固定9对有效观察、相同初始/逐步随机流记录原生DDPM100的99/74/49/24/0步x0估计。CPU原生采样机制检查通过，钩子预测/RNG及未选行完全不变；Bash语法与ZIP校验通过。ZIP13,298字节，无优化器、新权重、依赖安装或CLIP/Adapter前向；复用202278检查点与保存融合条件。尚未提交或加载真实检查点，不能声称定位根因或泛化改善。见[范围、停止条件和提交命令](BRIDGE_PROJECT_CONDITION_PROBE_202278.md)。下文“下一项尚未打包”为此前状态。

203876两个小结果独立核验完成：原40窗三个采样逐元素精确重放（差0），固定9对索引/目标及所有配对变化/有效观察误差复算一致，0次优化。8/9窗换图像变化较大；窗口146图像/状态/二者变化1.416/1.481/1.003cm，两个有效观察对应专家轨迹差16.489cm而输出差1.003cm，确认作者在这对条件下未复现专家轨迹差异，但不定位唯一层或判闭环失败。换回训练观察对训练目标为0.868cm，原留出为12.726cm，换了观察及真值不是泛化修复。项目201663/202278真实组保存条件cos0.676/0.728，均未完全塌成同一向量；固定余弦训练近邻读出也未优于静止。作者对照收口；下一项只规划项目真实组融合条件→扩散头的配对噪声/原生链诊断，未打包或提交，不重训。见[完整独立结果与解释边界](BRIDGE_AUTHOR_OBSERVATION_RESULT_203876.md)。下文待下载为历史记录。

作者冻结观察对照203876已由用户提供完成状态：COMPLETED/0:0，1分32秒，A6000。日志显示原40窗三个种子重放和9窗四观察推理完成；仅图像／仅状态／二者的平均位置输出变化10.0426／1.2730／10.6544cm，旋转16.0936／1.9388／16.7816°。这些是配对输出变化而非动作误差；两路均影响作者输出，平均图像替换变化更大，不能判定状态忽略、故障层或修复。尚未下载两个小结果独立核验；下一步仅下载并复算，不重训。见[结果与下载命令](BRIDGE_AUTHOR_OBSERVATION_RESULT_203876.md)。以下待运行为历史记录。

用户已提交作者图像／状态冻结输入对照203876；尚未收到状态或结果。仅一个推理作业，0次优化更新；先完整重放203699的40开发窗，再比较9对固定输入替换，原重放不通过则停止。下一步查看203876日志，不重复提交或判断条件效果。见[固定作业号和协议](BRIDGE_AUTHOR_OBSERVATION_PROBE_203699.md)。下文尚未提交为准备时记录。

作者最终EMA图像／状态冻结对照包已准备并冻结：固定西兰花9窗／同任务视觉训练近邻，先原生重放全部40开发窗，配对CPU/CUDA噪声分别换图像、当前状态、二者。输入替换、语言／未选行保护、CPU随机数恢复、实际启动参数及Py3.9/Bash语法本地检查通过；本地未加载作者模型或验证GPU重放。ZIP8,533字节，复用203699权重／现有缓存环境，10分钟作业上限，无优化器、新checkpoint或训练。尚未提交；混合观察仅敏感性诊断，不能按其误差宣称改进。见[固定协议、停止条件和提交命令](BRIDGE_AUTHOR_OBSERVATION_PROBE_203699.md)。下文未打包是设计时状态。

203699西兰花共享观察/监督审计完成：原12训练演示/24窗与2留出演示/9窗，14条原始记录和哈希、状态/未来缓存身份及指令一致核验通过；无训练、新模型推理或CLIP编码。4/9当前XYZ及32/144未来XYZ超同任务训练范围，但均在全128训练的相应XYZ逐轴范围内，不支持直接全局limits截断解释。训练/留出都存在高CLIP相似、较大几何/轨迹差；长度25/38在训练已有，物理时间戳缺失。窗口146范围内仍错15.39cm，作者预测到固定视觉训练近邻仅1.64cm；原项目同窗未呈同样贴近，故不能把两者错误统一归因。发现覆盖和条件读取线索，未确认唯一根因。下一项仅设计作者冻结图像/状态干预，未打包或提交，不重训或重复中段换窗。见[审计范围、真实画面与解释边界](BRIDGE_BROCCOLI_OBSERVATION_AUDIT_203699.md)。

作者Bridge203699八份小结果已下载，独立CPU复算、源码/配置/输入目标/索引及8192更新/学习率日志验收通过。训练EMA8.74mm/2.08°，64训练演示均优于训练静止，但位置与三次采样一致性仍未达5mm门槛；项目201663同128窗则1.83mm/0.58°已达标。开发EMA8.378cm/15.220°，6/10演示位置改善，整体未优于静止；西兰花两条留出演示位置明显更差，后段误差增长。保存三次输出的波动项只占开发位置平方误差1.34%，不支持靠换采样种子修复。作者参照有效性未通过，不能定位唯一模块或声称算法无效。本次分析完成；下一项转向共享观察/监督适用性审计，不追加训练。见[独立结果、任务与步长拆分](BRIDGE_AUTHOR_REFERENCE_RESULT_203699_20261014.md)。以下待下载为历史状态。

作者Bridge正式参照203699已收到COMPLETED/0:0、19分12秒、A6000的完成状态。用户日志记录8192更新完成；最终raw训练7.02mm/1.58°、5/128达标，EMA训练8.74mm/2.08°、0/128达标，均未达到预定拟合门槛。主要EMA开发8.378cm/15.220°略差于静止8.212cm/15.098°。本固定预算尚未建立有效同数据作者参照，不能归因唯一模块、宣称算法无效或问题无解。下一步仅取回八份小结果独立复算，不追加训练。见[日志结果与固定下载命令](BRIDGE_AUTHOR_REFERENCE_RESULT_203699_20261014.md)。下文待运行为历史记录。

2026-10-14：用户已提交同Bridge作者实现正式参照203699，固定8192更新、最终EMA主要结果、128训练/40开发完整评价。尚未收到Slurm状态或结果，不判断拟合或泛化。下一步查看203699状态与预算门槛/更新日志，不重复提交。见[固定作业号和协议](BRIDGE_AUTHOR_REFERENCE_TRAINING_20261014.md)。下文“尚未提交”为准备时记录。

2026-10-14：作者Bridge预检203498六份小结果已下载核验。168原始输入、四更新日志、训练统计归一化和反变换通过；峰值保留显存5.91GiB，编码器/U-Net/EMA变化按报告核对，未下载权重独立检查。固定最终EMA主要结果的8192更新正式参照包已冻结，11,978字节，复用集群缓存/源码/环境；前8更新测量预算，作业上限3小时，不自动延长。评价全部128训练及40开发窗口，夹爪/仿真不参与。多项模块差异不能直接做因果归因，尚未提交正式作业或取得拟合/泛化结果。见[核验、固定协议和操作命令](BRIDGE_AUTHOR_REFERENCE_TRAINING_20261014.md)。下文待下载及未准备状态为历史记录。

用户已提交作者Bridge资源预检203498；尚未收到Slurm状态或结果。仅一个4更新预检，未提交正式作者训练，也不能认定作者参照已有效。下一步查看203498状态和日志，确认原始数据导出及作者Workspace运行情况，不重复提交。见[预检协议与固定作业号](BRIDGE_AUTHOR_REFERENCE_PREFLIGHT_20261014.md)。下文“尚未提交预检”为准备时状态。

2026-10-14同Bridge作者Workspace资源预检已准备并冻结：复用原128训练/40开发输入与既有官方Push-T环境，作者源码固定5ba07ac，不改原训练Workspace/策略/调度/EMA。薄数据封装保留同224浮点画面、真实当前状态和固定CLIP文本EOS768，作者ResNet/融合、U-Net宽度、epsilon、limits/AdamW/EMA等多项与项目不同，不能当作单模块因果对照。168原始输入转换通过，作者归一化函数真实执行和独立limits/训练统计/反变换核验通过（最大分量差2.384e-7），实际参数与Bash语法通过。ZIP29.2KiB，无图像/权重/环境；本地没有作者GPU运行或训练。集群仅提交一个4更新资源预检，缓存约100MB留集群，不自动启动正式训练。后续8192预算仅暂定，待资源/接口结果冻结。见[接口、差异、验收与提交命令](BRIDGE_AUTHOR_REFERENCE_PREFLIGHT_20261014.md)。下文“下一阶段尚未准备”为状态复核收口时记录。

2026-10-14状态复核202463独立核验完成：预检15快照、正式27必要文件/16快照、两组各8192更新及输入/源码/冻结约束通过。初始化42零/真实组训练1.861mm/0.667°和2.681mm/1.067°，达标127/128、117/128，均通过原≥90%总体门槛，真实组未全部拟合且训练比控制差。开发8.878cm/16.478°→8.382cm/15.994°；6/10位置改善、7/10旋转改善、5/10两项改善，演示等权也同向。与首轮43一起满足预定候选复核条件，但仍差于静止8.212cm/15.099°；仅5条演示在两初始化都改善位置。状态方案提供部分帮助，未确认唯一根因或修复泛化。本轮配对收口，不追加种子/状态扫描；下一阶段仅准备同Bridge作者实现参照的接口和预算，尚无新作业。本地无PT/重训。见[完整结果与下一步](BRIDGE_STATE_REPLICATION_RESULT_202463_20261014.md)。下文待下载/复算现已完成。

2026-10-14收到Slurm完成状态：状态初始化42复核预检202462、正式配对202463均COMPLETED/0:0，在gpu-a6000-1执行，分别18分55秒/1小时22分22秒。所贴gate False来自预检real的32次短更新，不是正式8192次拟合结果。正式报告和预测尚未下载，训练拟合及收益是否重复仍待核验；下一步只下载不含PT的小结果，不启动新实验。见[完成状态与下载命令](BRIDGE_STATE_REPLICATION_20261013.md)。下文“尚未收到状态”为历史记录。

用户已提交状态初始化42复核：预检202462、正式配对202463。尚未收到Slurm状态或结果，不能认定预检或正式训练已通过。按冻结启动方式正式作业依赖预检成功；下一步检查两个作业状态及预检末尾日志，不重复提交。下文“集群尚未提交”为准备阶段历史记录。见[复核协议与固定作业号](BRIDGE_STATE_REPLICATION_20261013.md)。

状态候选的一次初始化复核已准备：固定43→42，两个训练入口反向恢复种子后与首轮逐字一致；128/40窗口、批次/噪声/LR/8192更新和接口均不变。168输入、15预测独立CPU验收、启动参数和Bash语法通过。零状态组原参数、64损失和五份预测/上下文精确重放同初始化旧模型，真实状态有梯度和贡献，冻结/恢复检查通过。845.66秒、峰值3.936GiB；本地小文件4.41MiB，无PT或完整拟合。ZIP28.6KiB已冻结，无新增数据/环境/权重。集群尚未提交；下一项仅执行这一次配对，若候选不能重复则收口，不追加种子。见[复核设置、验收与提交命令](BRIDGE_STATE_REPLICATION_20261013.md)。下文“尚未准备”是首轮核验时状态。

202278完整工件核验完成：正式27份必要小文件、16份预测快照、每组8192条更新及集群预检均验收通过。本地未下载PT或重训。zero最终训练/开发预测与201663逐元素完全一致，最大差异0；两组128/128训练窗口达标。真实当前状态使开发路径位置8.728→8.349cm、旋转15.847→15.256°；10条演示7条位置改善、5条旋转改善、4条两项均改善，演示等权平均也同向。达到预设候选筛选门槛，但仍略差于静止8.212cm/15.099°；取西兰花任务位置退步，不能认定唯一根因或泛化通过。下一步仅允许一个额外初始化的同样配对复核，尚未准备或执行，不扩数据或扫超参。见[完整结果与停止条件](BRIDGE_STATE_PAIR_RESULT_202278_20261013.md)。以下“待下载/复算”为历史记录，现已完成。

202278用户终端摘要：执行True、两组训练拟合True。zero训练1.83mm/0.58°、开发8.728cm/15.847°、第16目标位置11.359cm；real训练1.69mm/0.60°、开发8.349cm/15.256°、第16目标位置10.728cm。当前状态使开发位置/旋转下降约4.3%/3.7%，第16目标位置下降约5.6%；仍略差于静止8.212cm/15.098°，未通过开发泛化协议。zero的四舍五入均值与201663吻合。以上为终端摘要，完整报告/预测尚未本地核验，10演示至少6条位置改善的筛选条件未知；先下载不含PT的小文件独立复算，不提交新训练。见[待核验摘要](../reports/bridge_state_202278_terminal_summary.json)。

用户提供Slurm最终状态：状态预检202277、正式配对202278均COMPLETED/0:0，同gpu-a6000-1，分别17分30秒/1小时18分10秒。已收到预检real短分支末尾日志，未收到正式报告/预测；短分支32更新的fit_gate False不代表正式8192更新拟合失败。下一步提取zero/real正式训练及开发指标，随后下载小报告和预测独立复算。不追加训练或重复提交，暂不判定状态输入是否改善泛化。下文“尚未收到Slurm状态”为先前记录。

用户已提交当前状态配对：预检202277，正式对照202278。尚未收到Slurm状态或报告，不能认定预检/训练已开始或通过。下一步检查两个作业状态及202277末尾日志，确认afterok依赖；不重复提交。下文“集群尚未提交”为准备阶段历史状态。

2026-10-13当前末端状态配对已准备并冻结：复用201663原128训练/40开发窗，零输入和真实观测XYZ/旋转列输入使用同一零初始化状态接口。零组短跑原参数、64次损失和五份预测/上下文精确重放；真实组状态梯度非零、参数/上下文改变；共享初始参数与首步损失一致。168原始状态和训练统计量、15份预测独立CPU复算、冻结模块/Adam恢复/随机数保护、真实启动参数和bash语法均验收通过。包28.3KiB，无重复图像/权重/未来状态；本地无PT。首轮零梯度范数求和舍入问题已在新接口修正并重新验收，不是原泛化根因。集群尚未提交、无完整训练或开发改善结果；下一项只做该配对，按预设结果分支收口。见[协议与提交命令](BRIDGE_STATE_CONDITIONING_PAIR_20261013.md)。下文“状态配对尚未准备”是审计完成时的历史状态。

计划第一步已完成（只读当前状态审计）：原128训练/40开发窗口、74演示原始身份/同索引图文state及完整16目标重建通过；图像/夹爪精确一致，独立位置/四元数最大分量差3.40e-7/1.38e-7。实际位姿头上下文没有数值末端位姿，夹爪测量仅入独立分支，历史分支重复当前图像。视觉近邻当前位置差训练9.62cm/开发8.26cm；状态近邻未来位置差较小，但开发旋转未改善。6/40开发当前XYZ超同任务训练包围盒；无同任务完全相同处理图像重复对。证据支持状态输入配对候选，未证明根因；无训练/新编码/下载，输出约0.25MiB。状态配对尚未准备或提交。见[全部检查、限制与下一项](BRIDGE_CONDITION_INPUT_AUDIT_20261013.md)。

202069完整核验完成：13份正式小文件及202068预检报告齐全，无PT；8192日志/八份预测、冻结源码/配置/共享初始化、168输入和同40开发标签复算通过。保留89窗最终2.040mm/0.765°、88/89，替换39窗2.248mm/0.761°、39/39，两个组后期四次门槛均通过；总体127/128。开发9.420cm/17.267°较201663退步，10演示仅3条位置改善、1条两项均改善；打开抽屉改善，其他四任务更差。该具体替换方案不采用并收口，不追加训练/256扩容；不能因此完全排除数据选择或归因唯一模块。见[结果、证据与停止条件](BRIDGE_WINDOW_COVERAGE_RESULT_202069_20261013.md)。下文“尚待下载/复算”为历史状态。

202069终端日志已完成8192更新并输出报告：训练2.103mm/0.762°、总体门槛True；开发9.420cm/17.267°，差于201663控制8.728cm/15.847°和静止8.212cm/15.098°。39窗口内部覆盖替换目前未显示泛化收益，暂不采用，不继续增加规模或追加训练。本地尚无正式报告/预测，未收到Slurm最终状态，待13份小文件及预检报告独立核验、拆分89/39和逐任务/演示/时段。不将具体方案失败等同于完全排除取窗问题。见[本轮日志与待核验范围](../reports/bridge_window_202069_log_summary.json)。下文“尚无结果”为提交时状态。

取窗对照已由用户提交：预检202068，正式202069（依赖预检成功）。尚未收到状态/结果；下一步查看sacct及两份末尾日志，待完成后下载小报告/预测，核验128训练窗口拟合及同40留出窗口泛化。请勿重复提交。下文“正式尚未提交”为准备阶段状态。

2026-10-13同规模取窗对照已准备并冻结：仍64演示/128窗口/8192更新，保留89窗、替换39窗，最早/内部/最晚从64/12/52变为38/51/39。40开发窗、模型/尺度/预处理、初始化43、抽样/噪声及学习率预算均固定。168输入、64条短跑日志和五份预测独立CPU核验通过，恢复日志/数组精确一致，本地新旧初始化哈希相同，CLIP及冻结模块未变。267.73秒、3.936GiB，无本地PT；ZIP19.24MiB，无新安装或原始数据/权重下载。正式集群尚未提交，不能宣称泛化改善。不扩256。见[协议、判断顺序与提交命令](BRIDGE_WINDOW_COVERAGE_CONTROL_20261013.md)。下文“新输入/包未执行”为先前计划状态。

201663泛化只读诊断完成：五个留出任务训练均见过；同任务XYZ范围仅43/640留出目标越界，不能解释所有失败，关抽屉另有21/96旋转幅度超训练最大值。确认取窗分布差异：训练最早/内部/最晚64/12/52，开发10/20/10；开发最早位置/旋转平均优于静止，内部和末尾均更差。冻结CLIP近邻显示相似整体图像可对应明显不同轨迹，最高例0.973但动作差21cm，原始姿态/坐标重建核验通过，基座轨迹差仍约20.92cm。均不构成唯一根因证明。拟议同规模取窗对照保持64演示/128窗口/8192更新，保留89窗、仅替换39窗，内部覆盖提高至51；身份计划已记录，新输入/包/训练未执行。不扩256、不混新数据。见[诊断证据与下一项](BRIDGE_GROWTH128_GENERALIZATION_201663_20261013.md)。

201663完整核验完成：13份文件齐全（8.19MiB，无PT），8192条日志、运行脚本、168输入检查及八份预测独立CPU复算通过。最终原64为1.885mm/0.569°、新增64为1.773mm/0.588°，各64/64；总体128/128，四个后期观测总体与两组门槛均通过（7680新增63/64）。开发8.728cm/15.847°仍差于静止8.212cm/15.098°，0/40达到训练门槛，10演示位置5条优于静止。128仅初始化43、夹爪冻结，当前扩容拟合检查收口，不自动扩256，下一项转向固定数据/权重的泛化诊断。见[完整结果及证据范围](BRIDGE_GROWTH128_RESULT_201663_20261013.md)。以下缺件/待复算记录已完成。

201663补件后独立核验：13份所需文件已收到11份（7,661,003字节，无PT），仍缺reduced_u8192_train.npz和reduced_u8192_development.npz。运行脚本哈希、8192条更新日志、原64标签与200600一致性、新增64参考标签及六份预测独立复算通过。6144/7168/7680原64均64/64，新64为64/64、64/64、63/64，三次总体与子组门槛均通过；训练位置总体1.473/1.705/1.752mm。最终训练和留出预测尚不能复算，完整verified=False仅表示缺件。没有新训练。见[独立核验及缺件](../reports/bridge_growth128_analysis_201663_20261013/analysis.json)。下文仅收到两份JSON为此前下载状态。

201663下载核验状态：本地仅有report.json、plan.json，缺两份JSONL、运行脚本及八份NPZ共11份，不能完成独立指标复算。两份JSON的清单/源哈希声明、原104输入与200600一致性、168输入检查和预算核验通过。报告记载原64为1.885mm/0.568°、新增64为1.773mm/0.587°，各64/64；四个后期观测两组门槛均True，7680新增组63/64，其余均64/64。泛化总体仍弱于静止，待补齐小文件；没有新训练或256扩容。见[文件清单](../reports/bridge_growth128_analysis_201663_20261013/partial_inventory.json)。

128窗口最新日志：集群预检201662（8分08秒）及正式201663（44分18秒，A6000）均COMPLETED/0:0。正式8192更新后训练1.829mm/0.577°、总体门槛True；6144/7168/7680位置1.473/1.705/1.752mm，存在轻微回升，分组和逐窗稳定性尚待核验。开发8.728cm/15.847°仍差于静止8.212cm/15.098°。尚未下载报告/日志/8份预测，不能宣称两子组或全部窗口通过。不自动扩到256；先独立复算这一轮，夹爪冻结。见[本轮状态与下载命令](BRIDGE_GROWTH128_FIT_20261013.md)。

2026-10-13下一阶段准备完成：128窗口/64训练演示包已冻结（31.33MiB），保留200600原64窗口及40开发窗口，新增64仍来自原五任务训练分区。模型、预处理、原生DDPM100及每窗512次抽样保持一致，总8192更新，4096后降低学习率。168输入、64次短跑更新、5份预测独立核验通过，恢复loss/预测精确一致，冻结参数未变；本地约259秒、3.936GiB，无PT。正式集群训练尚未提交；本项检查扩大规模的训练拟合，尚不能宣称泛化改善。见[冻结协议与提交命令](BRIDGE_GROWTH128_FIT_20261013.md)。下文“128尚未准备”为历史状态。

64窗口复核核验完成：201429的13份文件齐全（约4.28MiB，无PT），4096日志和8份预测独立复算通过。与200600输入哈希/标签/模型/训练配方一致，训练源代码只改初始化43→42。原32为1.481mm/0.674°，新增32为1.517mm/0.615°，各32/32达标，后期四次观测均64/64。留出仍9.691cm/18.713°差于静止基线，10条演示仅3条位置更好。两个初始化下64窗口拟合可重复，该项收口；建议下一阶段128窗口规模检查，尚未准备或提交。夹爪冻结，泛化根因未定位。见[完整复核结果](BRIDGE_GROWTH64_REPLICATION_RESULT_201429_20261013.md)。以下为历史记录，201429下载/核验待办已完成。

64窗口复核最新日志：集群预检201428、正式复核201429已提交并输出报告。初始化42在4096更新后训练1.4992mm/0.6434°、门槛True，开发仍9.6913cm/18.7132°差于静止基线。与200600初始化43结果接近；尚待下载13份小文件，独立核验执行、原/新增分组及逐窗预测。请勿重复提交本轮作业。见[复核状态](BRIDGE_GROWTH64_REPLICATION_20261013.md)。

下一项准备完成（2026-10-13）：64窗口配方初始化42复核包已冻结，训练入口反向替换种子后与200600完全一致，数据/配置/抽样预算均未变。104输入、32短跑更新及5份预测独立核验通过，恢复参数/loss/预测精确一致，shell语法及真实argparse参数检查通过。本地约231秒、3.871GiB，包约16.04MiB，未保存本地权重，正式集群复核尚未提交。见[复核设计与提交命令](BRIDGE_GROWTH64_REPLICATION_20261013.md)。

2026-10-13核验完成：200600所需13份文件齐全（约4.28MiB，无PT），104输入检查、4096更新、源文件/清单/共享初始化及全部8份预测已独立复算通过。原32窗口1.244mm/0.594°，新增32窗口1.306mm/0.611°，各32/32达标；3072/3584/3840/4096均64/64。开发仍9.749cm/18.894°差于静止8.212cm/15.098°，10条演示仅3条位置优于静止。本轮仅初始化43且夹爪冻结，训练拟合收获明确、泛化未解决。建议先复核同一64窗口配方的另一个初始化，尚未准备或提交。见[完整核验与限制](BRIDGE_GROWTH64_RESULT_200600_20261013.md)。以下为历史进度，200600下载/核验待办已完成。

2026-10-13最新日志：修正启动后正式作业200600完成4096次更新并输出报告。64训练窗口原生DDPM100平均位姿误差为1.2746mm/0.6009°，日志最终fit gate=True；3072/3584/3840观测位置约1.28/1.33/1.35mm。开发40窗口为9.7494cm/18.8939°，仍差于静止8.2123cm/15.0984°。这是64窗口训练拟合的初步证据，单初始化，夹爪冻结；尚未下载核验报告、原/新增分组及逐窗预测，不代表全部1836窗口拟合或泛化成功。等待小报告和NPZ下载后独立复算。

2026-10-13：64窗口预检200365产生短跑及恢复对照日志；正式200366被启动参数中的多余`+`拒绝，未开始训练。已修复生成器，保留冻结包，另生成包外启动脚本复用200365，真实argparse参数核验通过。待用户重提正式拟合；这次失败属于启动脚本错误，不提供拟合效果证据。见[修复与重提命令](BRIDGE_GROWTH64_FIT_20261012.md)。

下一项准备状态：64窗口/32训练演示的扩展拟合包已冻结，原32窗口及40开发窗口保留，新增16训练演示仍属于原五个任务；每窗保持512份抽样，共4096更新，256次完整遍历后降低学习率。104输入身份/数值检查及短预检恢复训练精确重放通过，正式训练尚未执行。包约16.04MiB，复用现有环境、数据、CLIP，集群只保存约0.5GiB最终可训练权重。见[固定设计、证据与提交命令](BRIDGE_GROWTH64_FIT_20261012.md)。

最新更新2026-10-12：后期学习率单变量对照200150及复核200208已独立核验。42/44共38份小文件齐全，26份预测及6144次逐步日志通过检查，原控制预测逐元素复现200039。三个初始化降低组在四个继续训练观测点均32/32达标，最终训练位姿误差约0.93–1.08mm/0.34–0.40°（GPU原生指标），原学习率组约3.03–21.86mm/1.62–5.24°。这是当前固定32窗口训练优化稳定性的可重复改善，泛化仍未通过，夹爪未训练。本项诊断收口；扩展拟合的新包已准备，正式训练尚未执行。见[最新结果和证据级别](BRIDGE_LATE_LR_REPLICATION_RESULT_200208_20261012.md)。以下为历史记录，旧的“下一步”不代表当前计划。

194912完整多尺度单步评价已取回：训练位置7.597cm，仅略优于紧凑7.701，仍差于回归3.984；开发7.363cm/14.189°优于紧凑7.672/14.726，但仍差于回归6.626/12.108。10条开发演示位置全部优于紧凑，只有2条优于回归；释放时刻准确率仍0。结构增加的收益普遍但较小，训练集预测不足仍在。下一项固定193761权重，在每任务8条训练演示各一窗口上比较t0/24/49/74/99带噪还原、纯噪声生成和同任务换图，只诊断不更新，详见[判定协议](MULTISCALE_NOISE_DIAGNOSIS.md)。

193761多尺度x0结构对照训练已COMPLETED/0:0：20轮18360更新、37分14秒，共享初始化/顺序/CLIP哈希与192651匹配；CLIP不变，三个可训练模块均改变。最终权重526.2MB已在评价前保存，峰值保留显存4.166GiB。轮均位姿损失0.09467→0.04508不能证明离线误差改善。已准备冻结权重的一步配对评价：与193030使用相同三组噪声，旧紧凑x0和回归直接读取保存预测，覆盖完整1836/40窗口并按任务/演示拆分，不更新参数、不读保留测试。评价尚未执行，当前不能声称新结构更好。见[完整协议](BRIDGE_OFFICIAL_UNET_PREFLIGHT.md)、[训练报告](../reports/bridge-unet-train-193761.json)和[history](../reports/bridge-unet-train-193761-history.json)。

193711多尺度U-Net真实Bridge batch预检已COMPLETED/0:0：完整模型5.58亿参数、可训练1.315亿，峰值保留显存4.145GiB、暖更新0.1116秒。CLIP首步无梯度且六步后不变，Adapter/解码器/夹爪头均有梯度并变化；真实输入与x0损失链路通过，临时权重已丢弃，没有精度评价。已准备正式结构单变量训练：复用192651共享初始化和20轮顺序，只把紧凑U-Net改为多尺度U-Net；18360更新后先保存约0.5GiB最终权重，再独立评价。本次训练不读取开发图像/动作目标、不按开发结果选权重。见[协议与两次预检](BRIDGE_OFFICIAL_UNET_PREFLIGHT.md)和[193711报告](../reports/bridge-unet-real-193711.json)。

193548官方式Conditional U-Net资源预检已COMPLETED/0:0：A40上合成batch2完成4次前后向与AdamW更新。新结构77,670,023参数，峰值保留显存1.566GiB、暖更新0.0281秒；旧紧凑头6,120,455参数、0.137GiB、0.0105秒。输出形状、权重变化和上下文梯度均通过，但本次未加载CLIP/Adapter、真实数据或checkpoint，不能说明精度。现已将新结构作为可选配置接入，默认紧凑结构和旧权重行为保持不变；下一项仅运行冻结CLIP+Adapter+真实Bridge batch的6更新预检，使用x0目标、不评价开发集、不保存临时权重。见[结构与资源记录](BRIDGE_OFFICIAL_UNET_PREFLIGHT.md)和[报告](../reports/bridge-unet-check-193548.json)。

193390冻结epsilon步数核对已COMPLETED/0:0：固定193195权重，不训练、不读取保留测试，8步读取权威保存预测，只新算16/32/100步。开发集从8步13.108cm/21.549°改善到100步11.193cm/20.964°，但仍差于静止8.212cm/15.098°、x0一步7.672cm/14.726°和回归6.626cm/12.108°；旋转在32步19.613°后反而回升。所有步数在10条开发演示的路径位置上均0条优于x0一步或回归，关抽屉最佳仍约11.301cm/43.234°。当前“小U-Net + epsilon监督”分支停止，不再扫步数或加轮数。下一项只做官方Conditional U-Net结构的参数/显存/单批反向预检，避免同时改变EMA、数据和训练预算。见[最终步数核对](BRIDGE_EPSILON_STEP_SWEEP.md)与[完整报告](../reports/bridge-epsilon-steps-193390.json)。下文193310失败及“需重跑”为历史记录。

193310冻结epsilon步数核对在完成1876窗口推理后FAILED/1:0：跨A40节点重新计算的8步数组与193195保存数组最大分量差0.02633，触发了不必要的`1e-6`逐元素复现门槛；没有形成报告。该失败属于验证门槛设计过严，不能解释为模型新失败。修正版直接读取193195权威8步数组，只计算16/32/100步，并在最终检查前保存预测，需重跑。

193182 epsilon六步预检与193195正式训练均COMPLETED/0:0；正式训练26分19秒、18360次更新，CLIP不变。8步DDIM开发13.108cm/21.549°、第16目标17.405cm/28.755°，训练也达13.689cm/23.501°，显著差于x0一步、回归与静止；10条开发演示路径位置/旋转0条优于x0一步或回归。epsilon监督单变量替换失败。为排除仅8步不适配当前小网络，已准备冻结193195权重的8/16/32/100步最终核对；若仍无恢复便停止epsilon分支。见[训练结果](BRIDGE_EPSILON_CONTROL.md)与[步数核对](BRIDGE_EPSILON_STEP_SWEEP.md)。

已实现epsilon单变量配对训练：相对192651只改变扩散监督目标`sample(x0)→epsilon`，固定共同初始化哈希、CLIP哈希、18360次更新顺序、数据与模型结构，开发评估为8步DDIM。正式训练受6步GPU资源/梯度预检门控；本地epsilon/x0理想去噪器、步数表、旧一步语义、初始化状态与顺序共11项测试通过。尚未在GPU运行，不能预先宣称epsilon有效。见[实验协议](BRIDGE_EPSILON_CONTROL.md)。

193087冻结权重DDIM步数对照完成并核对：一步精确重放193030；开发1步7.672cm/14.726°、2步7.700cm/14.701°，4步起快速退化，100步10.815cm/17.266°。一步路径位置在10/10条演示优于100步，但仅2/10优于回归；关抽屉最佳仍约7.989cm/40.602°。确定性链同样退化，排除“主要只是逐步随机噪声”的解释，支持自生成状态分布漂移/误差累积。官方Diffusion Policy使用epsilon监督、8步DDIM、EMA和更大网络；下一项收敛为只改变x0→epsilon的配对训练，其他差异暂不混入。见[步数结果](BRIDGE_DDIM_STEP_SWEEP.md)与[epsilon对照协议](BRIDGE_EPSILON_CONTROL.md)。

193030同权重一步x0评估COMPLETED（A40，1分47秒，0:0）：开发一步三种子平均7.672cm/14.726°、第16目标10.156cm/19.424°，明显优于保存的100步随机DDPM 10.235cm/16.472°、13.687cm/20.925°，并略优于静止；但仍弱于回归6.626cm/12.108°、8.324cm/14.660°。10条开发演示路径位置一步9条优于完整链、8条优于静止、仅2条优于回归；夹爪释放仍0/8。说明100步误差累积是重要瓶颈，高噪声预测本身也仍不足。已准备固定权重的1/2/4/8/16/32/50/100步确定性DDIM对照，不重训、不读保留测试目标。见[一步结果](BRIDGE_ONE_STEP_SAMPLER.md)与[步数协议](BRIDGE_DDIM_STEP_SWEEP.md)。

192971同权重一步x0评估在导入阶段失败（47秒，1:0）：小型传输包遗漏`probe_bridge_conditioning.py`，尚未加载模型或产生预测；TensorFlow未注册GPU的日志与PyTorch作业无关。已将所需的固定窗口选择逻辑内置并增加完整性测试，需用修正版小包重跑。

已准备192651同权重的一步x0/完整DDPM严格配对评估：固定1836/40窗口与三个初始噪声，一步分支仅在`t=99`调用一次已训练U-Net，并复用生产位姿约束和夹爪解码；完整链及回归直接读取192651保存预测。该实验不重训、不读保留测试目标，用于区分高噪声去噪能力与100步采样误差累积，尚未在集群执行。见[实验协议](BRIDGE_ONE_STEP_SAMPLER.md)。

192939保存预测补充评估完成（CPU，1秒，0:0）：开发第16目标回归8.324cm/14.660°，扩散三样本平均13.687cm/20.925°，静止10.565cm/19.986°；普通扩散并非路线不同但终点正确。扩散oracle best-of-3为9.362cm/15.018°，说明分布有一定覆盖，但依赖真值选择且每项指标可选不同样本，不能当部署性能。10条开发演示中回归第16目标位置10/10优于扩散平均，但仅6/10优于静止。事件位姿回归较好，抓取切换仍3/20、释放0/8。见[终点与事件结果](BRIDGE_ENDPOINT_METRICS.md)。

已准备192651保存预测的轨迹/窗口终点/操作事件补充评估，不重训或重新推理。将同时报告16目标平均、第16目标、真值夹爪事件位姿、扩散逐样本及oracle best-of-3；第16目标只是固定输入坐标系下的窗口终点，并非任务最终状态，oracle也不能当部署性能。两项合成指标测试通过。CPU作业尚未执行，见[评估协议](BRIDGE_ENDPOINT_METRICS.md)。

192651配对联合训练已完成（A40，59分10秒，0:0）：回归与扩散各18360更新，共用初始Adapter/夹爪哈希，CLIP冻结哈希不变。回归训练3.984cm/11.113°、开发6.626cm/12.108°，同时优于扩散10.777/17.833与10.235/16.472、静止以及训练任务均值；开发10条演示中位置8条、旋转8条优于扩散。说明当前数据处理与视觉条件链能在替代位姿方案下学习，现有扩散方案是主要性能瓶颈，但此次同时改变头、损失和推理，不能唯一归因采样公式。夹爪切换仍差：回归开发开→关3/20、关→开0/8。尚非仿真或零样本成功，暂不进入仿真。见[完整对照](BRIDGE_HEAD_CONTROL.md)。

已准备回归/扩散联合训练配对方案，当前仅本地准备通过：同1836/40窗口、相同Adapter/夹爪初始化、独立固定20轮样本顺序，各18360更新，固定最终权重。复用归档网络/损失/评分，不复用190189已训练参数；两项CPU顺序与反传控制测试通过。下一步先GPU每组6步预检，尚未运行真实CLIP训练。该比较改变动作头及损失，不是仅采样消融；旧冻结降噪器对照不能替代。见[协议](BRIDGE_HEAD_CONTROL.md)。


191256一次预测换图检查完整核对通过：同窗口/权重/噪声，正确组精确重放191004第一次预测。开发正确7.449cm/14.769°，换图7.877cm/15.577°；10条演示位置和旋转各7条正确图像更好（包含极小差异，非显著性结论）。训练405条分别279/276条更好。开发收益主要来自开抽屉和西兰花，微波炉近乎不敏感，拨杆无明显开发收益。证明本条件下视觉总体有用，不证明视觉模块全正确。诊断收敛为第一次预测相对均值优势有限、原100步总体误差进一步增加；下一阶段应固定协议做有针对性的回归/扩散训练对照，不继续无界堆叠诊断。见[详细结果](BRIDGE_ONE_STEP_IMAGE_PROBE.md)。


已准备一次预测同任务换图对照：全1836训练/40开发窗口真实复读匹配191004，1876个换图对象逐项验证同任务同划分不同演示；四项采样/覆盖测试通过。只复用冻结模型t=99预测，固定语言与噪声、记录噪声哈希和逐演示结果，不运行100步或重训。GPU尚未执行，见[协议](BRIDGE_ONE_STEP_IMAGE_PROBE.md)。


训练集任务均值对照已真实执行：仅1836训练窗口拟合、开发40窗口仅评分，不加载模型、不读保留测试。任务均值训练7.862cm/13.707°、开发7.722cm/14.592°；第一次预测训练7.572/13.727、开发7.449/14.769，总体位置改善约0.27–0.29cm、旋转无优势。不看图像的均值也超过静止，不能仅由首次预测超过静止判断视觉阶段学习有效；逐任务不完全相同。下一项应只隔离一次预测的同任务换图影响。见[计算协议与限制](BRIDGE_TASK_MEAN_CONTROL.md)。


191004全窗口冻结对照已完成并下载核对：1836训练/40开发窗口、同checkpoint/归档模块、配对初始状态与参数不变检查通过。第一次纯噪声x0训练7.572cm/13.727°、开发7.449cm/14.769°；原随机100步10.427/16.949与9.770/16.273；无步间噪声7.761/14.366与7.845/16.365。原采样在完整训练集也增加平均误差；去步间噪声只确认位置改善，开发旋转并未改善。不能归结为仅泛化差或直接宣称采样bug，尚需第一次预测与训练任务均值对照，确认条件信息利用。未训练/改权重/读取保留测试。见[完整核对](BRIDGE_FULL_CHAIN_PROBE.md)。下文准备阶段为历史。


已准备全窗口冻结采样对照：本地真实读取1836个训练窗口、40个开发窗口通过；四项采样/覆盖测试通过。复用190189 latest，比较纯噪声一次x0预测、原随机100步、同初始噪声但无步间噪声100步。此次尚未在GPU执行，不重训、不改模型、不读取保留测试目标。见[协议](BRIDGE_FULL_CHAIN_PROBE.md)。


更新日期：2026-10-07。当前完成的是离线动作预测与模块诊断，尚未证明新演示泛化或机器人任务成功。

已取回190952完整链路报告，初始噪声/初次x0一致、方差恢复、参数不变与190853报告哈希均核对通过。开发原随机中间x0：t99 7.429cm/14.674°，t74 8.095/15.611，t49 9.267/16.512，t0 10.175/17.261；无步间噪声t99相同、t0 7.845/16.365，仍有旋转退化。逐任务无步间噪声开发位置：微波炉10.586cm、关抽屉8.423、开抽屉6.318、西兰花8.648、拨杆2.516；微波炉位置仍不如9.132静止，关抽屉/西兰花旋转仍不如各自静止。原随机L40重放与A40位置差绝对值均小于0.00011cm，硬件更换不解释本次链路退化。证据表明该固定模型后续生成阶段增加了这批样本的目标误差，不证明所有特征提取/动作监督都正确，也不唯一归因采样公式。静止是位置增量零、相对旋转单位的基线，不是未训练网络，故“不如静止”不能等价为“学习使网络比初始化更差”。后续应停止盲目加轮数/来源，以单步纯噪声、无步间噪声、原随机输出在完整训练/开发上的固定对照验证优先；尚未执行该扩展，不使用保留测试。见[完整链路核对](../reports/bridge_chain_190952_verified_summary.json)。下文“完整JSON尚待取回”为历史状态。

配对链路检查改用L40的190952已COMPLETED/0:0，有FROZEN CHAIN PROBE: COMPLETE。用户终端三个采样种子均值：开发原随机10.175cm/17.264°，不追加步间噪声7.845cm/16.365°；训练诊断40窗口原10.508cm/16.935°、无步间噪声9.004cm/15.377°。去噪声改善这批样本的平均目标误差，开发位置低于8.212cm静止，但旋转仍高于15.098°，且开发纯噪声一次估计7.429cm/14.674°仍更好。不能据此直接修改模型或宣称采样公式bug/机器人成功。完整中间x0及逐任务报告尚待取回；已要求取消重复A40作业190939，取消状态未确认。见[此次终端摘要](../reports/bridge_chain_190952_log_summary.json)。下文“GPU链路检查尚未执行”为准备阶段历史。

已准备[同初始噪声的链路检查](BRIDGE_CHAIN_PROBE.md)，复用原probe脚本并直接调用归档生产model.sample；比较原方差与临时置零posterior_variance，结束恢复，不改models.py或权重。沿用190853同80窗口/三种子，强制初始噪声与初次x0逐元素相同，hook记录7个中间步的x0误差，核对原随机组重放差值。CPU三项输出/RNG不扰动、配对初始状态、异常hook清理测试通过；真实80窗口清单复读匹配。小脚本包已导出，默认A40、复用环境/权重/数据，GPU链路检查尚未执行。不能把准备通过当无步间噪声能改善的证据。

已取回190853完整诊断JSON，选择与替换索引匹配本地准备，全部5个归档模块哈希一致，可训练参数前后哈希相同；80对图像均不同、80对语言均改变，运行103.5秒、峰值1.86GiB。纯噪声一步→完整生成的开发位置：关微波炉9.571→12.499cm（静止9.132）、关抽屉7.793→12.675（9.757）、开抽屉6.396→8.672（7.483）、西兰花8.334→11.047（9.201）、拨杆2.334→2.781（3.376）；五任务均变差，训练诊断子集也同方向。平均条件有效性不等于每任务视觉有效，微波炉对同任务换图近乎不敏感。进一步在CPU用归档生产sampler、100步float32余弦表及理想x0去噪器核对：三种子最终最大分量误差5.96e-8；均值公式与独立后验形式最大差1.66e-5，通过。排除了该合成条件下明显的公式/最终还原错误，不能证明实际网络在生成中间状态上预测准确或GPU所有路径无问题。下一步应保持网络/权重固定，在相同噪声下比较生产随机DDPM与不追加步间噪声的均值链路，并记录中间x0预测；这是诊断方案，尚未执行，不能预先宣称去噪声能解决任务。见[完整诊断核对](../reports/bridge_conditioning_190853_verified_summary.json)与[CPU理想去噪器检查](../reports/bridge_sampler_cpu_oracle_check.json)。下文“完整JSON尚待取回”为历史状态。

固定latest检查改用A40的190853已COMPLETED/0:0，末尾有FROZEN CONDITIONING PROBE: COMPLETE。依据用户终端片段，三噪声种子均值开发纯噪声t99一步x0估计7.429cm/14.674°，低于固定开发静止8.212cm/15.098°；生产100步完整生成10.175cm/17.264°。完整生成替换同任务图像后10.964cm/19.086°、替换另一任务语言后11.583cm/21.913°，配对输出也变化，不能解释为完全忽略图文。训练子集40窗口纯噪声一步8.426cm/14.668°、完整生成10.509cm/16.935°，不能替代完整训练评价。此结果将下一项核对集中到生成过程/预测分布，但条件均值式估计比随机样本误差低也可能无需实现bug，不能认定采样公式错误或所有视觉语义正确。完整JSON（控制输入、参数摘要、逐任务、初始t0正确条件）尚待取回。已要求取消重复A6000排队190743，取消状态未确认。见[诊断日志摘要](../reports/bridge_conditioning_190853_log_summary.json)。

已准备[固定latest条件依赖检查](BRIDGE_CONDITIONING_PROBE.md)：复用190189归档模型/数据/环境，40个固定训练窗口（每任务8条演示各1窗）和全部40开发窗口；正确图文、同任务另一演示图像、另一任务语言三组配对噪声对照，比较不同噪声强度的一步x0还原、纯噪声t99和生产100步采样，仅位姿、不更新权重。真实子集/完整mask复读与替换分区隔离检查通过，约7KB纯脚本包已生成；本地未加载CLIP/checkpoint，GPU实验未执行。不能把准备通过当条件有效或采样bug证据。

已取回并核对190189完整评价与history：20轮、18360更新，终端汇总与下载JSON一致。逐任务latest：拨杆训练3.727cm优于自身4.555cm、旋转7.715°优于9.541°；开发2.628cm/7.257°优于3.376cm/10.625°，仍仅2条演示/4窗口。其余四任务训练位置均劣于自身静止。固定开发集位置：西兰花13.506→10.199cm、开抽屉10.144→8.494cm，关抽屉9.497→11.560cm，微波炉14.249→14.248cm；不是五任务全面改善。微波炉夹爪始终仅一个标签类别，100%不代表切换学会。20轮开发位置/旋转均未超越各自基线，最低加噪损失在16轮、最低采样位置14轮、最低采样旋转13轮；只改checkpoint选择不足以达到基线。详细证据见[逐任务与历史核对](../reports/bridge_expansion_190189_task_summary.json)。下一项应使用现成latest区分带噪目标一步还原与从纯噪声生成，并检查图文条件是否影响预测，保持权重固定；尚未执行该检查，不能据此断言DDPM、Adapter或数据是唯一根因。下文“完整JSON尚待取回”为此前状态。

Bridge扩充作业190188检查、190189训练及best/latest完整评价均COMPLETED/0:0。依据用户终端完整汇总日志，latest三个采样种子均值：训练1836窗口10.299cm/16.748°，自身静止8.227cm/14.030°；开发40窗口10.189cm/17.015°，自身静止8.212cm/15.098°。固定开发集相对旧pilot latest位置11.108降至10.189cm、夹爪平衡74.22%升至82.98%，旋转16.262升至17.015°。开发打开→关闭目标对23.33%降至15%，关闭→打开0升至12.5%，不能称稳定切换；目标对重叠且只有20/8对。训练集位姿本身仍劣于静止，不能仅归因于留出泛化，但未定位唯一模块根因。数据规模和更新预算同时增加，不能隔离数据量因果。见[完整日志摘要](../reports/bridge_expansion_190189_log_summary.json)。新run的逐任务/演示完整JSON和history尚待取回；下一步只读取已有报告，不加来源、不重训或仿真。下文“尚未进行此次训练”为准备阶段历史。

当前只推进[Bridge扩充](BRIDGE_EXPANSION_V1.md)，不一次混入全部来源。原45条训练扩大到405条、207窗口扩大到1836窗口；原10条开发/40窗口与13条保留测试清单完全固定，开发payload哈希不变。412条元数据候选中7条新增拨杆演示末尾夹爪无法按既定规则确定二值标签，已透明记录并剔除，不补选；这不是原始文件损坏的结论。415条训练/开发演示、13,517帧RGB及全部窗口生产加载器/独立位姿/夹爪时序验收通过。网络和20轮设置沿用，但总更新从2080增至18360，不能宣称只改变数据量的因果比较。尚未进行此次新训练。

已完成[三来源实读与夹爪合同修正](CORE_SOURCE_RECORD_CHECKS.md)：Language Table/BC-Z/Fractal各一条固定train记录的RGB、数值还原与mask通过，单位/频率尚未独立验证。发现Fractal相对夹爪命令被旧阈值错误当绝对状态，新增relative_scan_v2；固定400目标中162项变化，5项语义测试通过。BC-Z另外三个固定记录确认未来测量与前一命令标签有差异，新增preceding_command_v2，1792目标生产核对通过；旧策略不暗中替换。Bridge新增52候选与旧划分origin/id无重复，尚未数值验收。混源输入/掩码/频率合同仍未就绪，没有新增训练；这些来源问题不解释此前Bridge-only预测失败。

已实际盘点用户现有八来源文件/TFDS元数据与VIOLA字段形状，形成[多数据源接入清单](MULTISOURCE_INGESTION_PLAN.md)。四核心训练来源均已存在，总约685GiB，不能全量放入200GB SSD；当前连续Bridge模式限制单来源，不能直接混合训练。Language Table仅xy监督、BC-Z专用诊断目标与Bridge不同、Fanuc6维命令及语言字段需独立处理。原盘Bridge train元数据53,191、Fractal87,212与邮件参考数不同，来源计数口径待核对。未解码图像、未读取数值动作目标或训练；下一阶段是Bridge扩充与其余三核心来源补缺项验收，LIBERO/CLIPort评价分区继续隔离。

已取回189937完整评价JSON与20轮history，确认2080更新。逐任务latest三个采样种子均值：拨杆训练位置3.737cm优于静止4.166cm、旋转8.226°优于10.778°；开发2.925cm/8.513°也优于自身3.376cm/10.625°，仅4开发窗口，不是任务成功。其余4任务训练位置均劣于自身静止基线，不能归为单条开发演示拖累。关闭抽屉开发位置略优但旋转略差；微波炉夹爪仅一个类别，100%不代表双向开合能力。优先处理固定预算下训练集动作预测不足，不直接扩大泛化实验，根因仍未确定。见[逐任务摘要](../reports/multitask_pilot_189937_task_summary.json)。

五任务作业189937已COMPLETED/0:0，20轮2080更新及best/latest完整训练/开发评估均完成。用户日志三个采样种子均值：latest训练10.367cm/13.249°，静止8.222cm/12.572°；开发11.108cm/16.262°，静止8.212cm/15.098°。开发夹爪准确78.80%、平衡74.22%；打开→关闭目标对23.33%，关闭→打开0%，均为重叠窗口目标对而非物理事件。best开发12.105cm/16.292°，不能以最低验证加噪损失宣称动作最优。当前完整模型的训练集采样预测本身不足，不能只归因于过拟合或泛化；没有定位唯一根因。先读取现成逐任务/演示JSON及history，不加轮数、不使用测试、不启动仿真。见[日志摘要](../reports/multitask_pilot_189937_log_summary.json)，完整JSON尚未取回。此前189915首批因隔离环境缺Pillow停止且未更新，补装Pillow12.3.0后189937完成。

集群189626完整模型8更新预检通过：Adapter/位姿/夹爪均更新，CLIP不变，batch2峰值2.754GiB。已完成[五任务固定预算试运行准备](MULTITASK_FIXED_BUDGET_PILOT.md)：55演示247窗口、测试目标不导出，主入口实际prepare-only通过；预算20轮2080更新，每轮40开发窗口，结束评价best/latest的完整训练与开发集。等待集群补读记录依赖及提交；尚无此次训练或泛化结果。以下“GPU未执行”等为此前准备阶段。

[完整模型短预检包](MULTITASK_PREFLIGHT.md)已准备，约4.9MiB，仅10个已验收训练窗口，复用模型和生产损失。语法及导出核查通过；本地缓存缺CLIP权重，尚未执行模型更新。先盘点集群权重和依赖，再提交8次联合更新的GPU预检；无新训练结果、无权重下载。

最新：[五任务原始数据与生产加载器实读验收](BRIDGE_MULTITASK_SOURCE_AUDIT.md)通过。45条训练/10条开发覆盖207/40个完整16步窗口，1779帧图像及3952个目标核对完成；独立坐标还原与夹爪时间对应通过，13条保留测试目标未读。初次失败源于审计脚本将原文大写与小写归组键直接比较，已修正检查，没有改生产数据处理。下一步是多任务训练入口接入与短GPU资源/梯度预检；尚未加载模型或训练，不宣称泛化通过。下文“未读数字动作或图像”为此前元数据盘点阶段。

当前下一步已确定为[五任务离线完整模型pilot准备](FORMAL_EXPERIMENT_PROTOCOL.md)，不再重复旧单任务头部/Adapter训练状态检查。只读原盘已有32分片元数据后筛出五指令68条候选，预分45训练/10开发/13保留测试；排除旧25诊断演示及sweep任务，未读数字动作或图像、未下载/复制数据、未训练。候选清单不等于训练就绪：下一步核对训练/开发的原生状态/边界/夹爪/图像及16步目标，再做冻结CLIP、8层512 Adapter与DDPM/夹爪联合学习的资源预检。不把增加层数/任务当作已解决泛化，也不宣称完成四数据源或正式基准。

已取回188886的三条开发演示提取明细（非完整JSON），[保存记录及存档对照](../reports/octo_bridge_188886_validation_detail.json)。位置均值分别1.156/1.613/1.657cm，均优于各自静止1.704/2.008/1.787cm；改善约32.2%/19.6%/7.3%，三个采样种子的9项位置均优于各自基线。旋转均值三条均劣于保持姿态，9项种子中仅00015::31的seed1优于旋转基线。夹爪平衡均值89.8%/94.4%/100%，第一条种子波动77.8%至97.2%，第三条只有2个打开标签，100%不能推断稳定切换能力。历史单步模型best/latest存档55个验证索引、数据身份及逐演示基线与此匹配；latest为2.301cm/4.698°、平衡72.22%，Octo均值1.453cm/2.935°、平衡94.12%。这是完整策略参照且存档较早，不是最新配对动作头或模块因果实验；不再把数据完全不可预测、单个Adapter根因或抓取成功作为结论。

Octo单步子集评估188886已COMPLETED/0:0，训练分区316、开发分区55个窗口全部覆盖。日志的三采样种子平均：训练1.395cm/3.090°、夹爪96.10%/平衡91.86%；开发1.453cm/2.935°、夹爪95.15%/平衡94.12%。相比单步静止基线，位置和夹爪改善，旋转仍未优于保持姿态（训练2.702°、开发2.446°）。完整JSON/逐演示及种子差异待取回；[日志摘要](../reports/octo_bridge_188886_log_summary.json)明确区分集群日志指标与本地核验基线。这里的“训练分区”不是Octo微调训练集，本次无更新，Bridge预训练重叠及版本不同限制仍适用；不归因于单一模块，不声称抓取/泛化通过。下文“子集未运行”均为历史准备状态。

最新：Octo安装188861、官方单图GPU推理188862均COMPLETED/0:0，[完整JSON](../reports/octo_inference_188862.json)已取回并逐项核对：GPU后端cuda:0、模型已加载、[1,4,7]共28个动作值全部有限。官方推理链路正对照通过；不是抓取成功、独立泛化或本项目CLIP/Adapter验收。下一项是核对Bridge原生动作合同后做单步离线评估，尚未执行。项目当前du -sh约26GB。以下失败与未运行记录为历史阶段，由此次日志更新。

Bridge单步评估包已准备，本地[实读检查](../reports/octo_bridge_single_step_local_check.json)通过：316/55窗口、17/3演示、SHA256及顺序核对；固定Octo原始relabel函数371项一致，姿态误差跨分支及独立SciPy核对通过。单步开发静止基线1.82cm/2.45°、一直关闭83.6%、平衡50%；不得沿用16步基线。集群将复用Octo环境，冻结模型、真实语言、历史1，仅评价预测第0步，输出整体/逐演示/三个采样种子指标；未执行子集GPU推理。Octo自有Bridge版本与旧OXE子集不同且可能训练重叠，非正式泛化实验。

Octo安装188838通过（pip check、资源缓存成功，环境约5.2GB、项目约25GB）；GPU推理188839在JAX后端初始化失败，未加载模型。发现cuDNN9与固定cuDNN8版jaxlib不匹配，补锁nvidia-cudnn-cu11 8.9.6.50，待集群安装及GPU重试。下文较早的“尚未下载/安装”状态已由本次安装日志更新；尚未取得Octo动作输出。

Octo 安装作业 188827 在依赖解析阶段失败：新版 wandb 与固定 TF2.15 的 protobuf 范围冲突。已补锁 wandb0.16.6，待集群重跑安装；推理尚未执行。该日志未提供模型性能或泛化证据。

重试188832暴露 TFMetadata1.14.0 的 protobuf<4.21 约束，改锁 protobuf3.20.3，与日志中全部 protobuf 依赖范围对齐；仍待安装验收。此前安装约束核查不完整，不能将这些环境故障归因于模型或数据。

当前研究下一项改为 [Octo 最小官方复现](OCTO_MINIMAL_REPRODUCTION.md)：使用已发布完整策略作正对照，暂停 CLIPort 专家轨迹调整及旧动作头调参。用户取回预检作业 188824：x86_64、A6000 可见，项目 apparent use 23.18GiB，inventory_passed=true，但 octo_inference_ready=false。已准备独立环境安装与官方单图 GPU 推理脚本，尚未执行安装、下载权重或推理。安装使用集群项目 SSD，旧环境保留；本地仅小型上传包。推理成功只证明官方模型加载及有限值动作输出，不代表机器人任务成功或新数据泛化。

当前操作入口：[最小执行协议与阶段验收](MINIMAL_EXECUTION_PROTOCOL.md)。专家回放诊断已收敛；候选流程为预测 16 步、执行 1 步、重新观察，但尚未实现闭环。相机、夹爪输入映射与固定周期运动/接触联合执行仍缺验收，不启动失败 Bridge 模型 rollout，也不重训。

最新合同检查：固定 front/index=0 相机，重复渲染不推进物理时间，现有训练预处理实跑输出 `[3,224,224]`。Bridge 连续夹爪输入已核对为实测开度仿射值，未用吸盘布尔状态替代。第一操作前全部 6 个合法规则位姿目标固定周期执行，跟踪通过但未附着、部分奖励 0；原接触深度发生在两采样点之间，稀疏到达位姿未保留接触过程。阶段 B 未通过，不接入模型。27 项测试通过，无训练/下载，见 [合同实跑证据](../reports/cliport_observation_contact_contract.json)。

2026-10-07 核对固定 Bridge 作者源代码：作者也用相邻实测状态差重标记，不能把到达位姿监督本身定为错误。已有训练划分中的一条 98 步演示，1,432 对未来位姿编码/解码运行通过（最大位置误差 3.73e−8m）。当前项目的固定初始锚点四元数轨迹，与作者逐步 Euler 状态差及原始控制命令需要区分；没有证明执行等价。保留标签和权重，下一项核对作者动作消费端，不再调专家回放或盲目重训。见 [对应关系与限制](MINIMAL_EXECUTION_PROTOCOL.md) 和 [证据](../reports/bridge_target_execution_audit.json)。

作者部署入口已核查：反归一化、默认执行一行、0.2s 标称请求周期、夹爪开/关整理；另固定机器人仓库版本查到旧命令目标累积与世界轴旋转合同。隔离运行实际纯变换及 `_next_qpos`，确认实测 30cm/旧目标 35cm/增量 1cm 会得到 36cm，三个目标逆变换往返通过。没有硬件/ROS/模型执行，机器人版本未确认为策略历史运行版本。旧模型输出不能冒充该增量命令；先补部署语义兼容性检查，保留标签、不重训。源文件新增约 59.5KB，无数据或权重下载。见 [部署证据](../reports/bridge_deployment_contract.json)。

已完成 [与 Diffusion Policy 作者代码的只读对照](DIFFUSION_POLICY_COMPARISON.md)：固定作者提交、记录输入/动作/U-Net/条件/采样差异。未新增训练或修改网络；不能把结构差异当作已确定根因。

随后已完成 [实际运行对照](DDPM_RUNTIME_COMPARISON.md)：执行生产加噪/采样与原始库，每种目标覆盖 100 个时间步；对齐裁剪后最大反向差 <1e-5。当前及作者 U-Net 的前向/条件/梯度通过。该数值检查本身没有优化器更新。

## 配对动作头训练结果（作业 187797）

2026-10-06 完成两组各 9,875 次更新。训练输入 SHA256 一致，每个训练窗口均抽取 2,000 次；冻结上下文与夹爪参数一致且未改变，验证未用于优化，测试目标未使用。结果摘要见 [decoder_pair_187797_summary.json](../reports/decoder_pair_187797_summary.json)。

| 动作头 | 训练位置 cm | 训练旋转 ° | 开发位置 cm | 开发旋转 ° |
|---|---:|---:|---:|---:|
| 当前项目 | 0.261 | 0.979 | 10.422 | 17.011 |
| 作者 U-Net | 4.685 | 12.839 | 10.774 | 14.632 |
| 开发静止基线 | — | — | 9.608 | 12.942 |

数值为三个采样种子的均值；每组只有一次训练初始化，不是三次独立训练。当前组在三条开发演示上的位置与旋转均未优于静止基线。作者组最后阶段的记录损失从第 9,250 步的 0.000193 升至第 9,500 步的 0.119864，最终为 0.048827；最终训练拟合不足，不能据此排除动作头影响，也不能宣称此前已经拟合成功。未保存中途权重或评估，无法回溯中途动作误差。

夹爪参数冻结，但读出依赖预测位姿，因此两组夹爪输出可以不同。两组开发切换均为 0/13、0/6。该实验仅替换作者 U-Net 模块，未复现作者完整策略、观测输入、优化方案或正式零样本基准。

## 已完成的工作

### 作者低学习率补充（188044）

集群 setup 188032、预检 188033、正式训练 188044 均完成。旧作业初始化、冻结参数、完整训练输入哈希核验通过；作者分支学习率降为 3e-5，保持 9,875 次更新并增加训练分区监测。日志结果：最终训练位置 0.1151 cm、旋转 0.442°，开发位置约 10.1699 cm、旋转 15.445°，仍差于静止 9.608 cm / 12.942°。七个固定监测点显示训练拟合有波动，但在第 2,500 步以后记录的训练平均位姿均达到既定门槛；未再出现上一轮最终训练误差的明显退化。未观测所有中间更新的完整动作误差，不能称全过程无波动。

这支持较低学习率改善该作者分支最终训练拟合，不能解释或解决开发泛化。当前项目头与较低学习率作者头均能拟合而开发失败，但学习率不同，不能把两者当作严格单因素架构排名。停止继续扫描学习率或更换降噪器。三个采样种子不等于三个独立训练初始化。完整 JSON 已取回，初始化、冻结参数和最终完整训练输入哈希均与 187797 一致，逐演示及夹爪已核验。精确证据见 [结果摘要](../reports/author_lr_188044_summary.json)，另保留 [日志摘要](../reports/author_lr_188044_log_summary.json)。

| 开发演示 | 作者/静止位置 cm | 作者/保持姿态旋转 ° |
|---|---:|---:|
| 00003::14 | 10.77 / 10.15 | 15.86 / 12.68 |
| 00015::31 | 12.74 / 11.23 | 12.20 / 11.87 |
| 00019::5 | 7.18 / 7.53 | 17.84 / 14.20 |

表为三个采样种子的均值。第三条位置改善，但旋转变差；没有一条同时通过两项位姿基线。开发夹爪总准确率 90.91%（一直关闭 91.93%）、平衡准确率 67.43%、打开召回率 39.44%；重叠窗口切换对为 0/13、0/6，按事件去重也没有正确切换。夹爪参数本轮冻结且接收预测位姿，这不是新的夹爪训练实验。完成本轮诊断收尾，不进入仿真、不再提交头部参数扫描。

| 环节 | 当前证据 | 结论边界 |
|---|---|---|
| 数据读取和处理 | 固定 Bridge 子集全部 371 个训练/开发窗口经过独立图像、数值和分区核验 | 支持该子集的实现一致性；不代表全部 OXE 或真实控制接口正确 |
| CLIP 和 Adapter 上下文 | 特征能支持后续动作头拟合训练窗口 | 不证明具有足够的新演示、语言或任务泛化信息 |
| 回归动作头 | 隔离诊断能拟合完整训练窗口 | 原始完整训练结果及开发验证仍失败 |
| 扩散计算 | 当前 100 步余弦表、7 维位姿加独立夹爪路径的 20 项审计通过 | 不是所有配置通过；非分离 8 维路径有已记录的掩码问题 |
| 扩散训练目标 | 14 窗口相同初始化、相同带噪输入配对比较，x0 比 epsilon 拟合更好 | 不能据此说 epsilon 算法错误或已解决泛化 |
| 完整训练 x0 拟合 | 316 窗口平均位置 0.21 cm、旋转 0.95° | 是冻结上下文动作头诊断，不是目标 8 层结构联合训练 |
| 开发验证 | 全部 55 窗口及 3 条演示分开检查 | 泛化未通过；不是独立最终测试 |
| GPU 集群 | 数据传输核验、JAX GPU 计算、作者加载器和 MUSE 缓存检查通过 | 这些是基础设施检查，不是策略效果 |

## 关键结果

原 2 层 Adapter 回归模型完成 1,000 次更新，开发集位置误差 10.48 cm（静止 9.61 cm），旋转 19.40°（保持姿态 12.94°）。夹爪总准确率 94.2%，但平衡准确率 66.7%，两方向切换仅 1/13、0/6。训练损失下降没有证明模型有效。

后续完整 x0 诊断（集群作业 186047）固定最终第 9,875 次更新，不按开发指标选权重。batch=64，每个训练窗口恰好抽样 2,000 次。CLIP/Adapter 上下文及夹爪头冻结。

| 分区 | 演示数 | 窗口数 | 动作目标数 | 位置误差 cm | 旋转误差 ° |
|---|---:|---:|---:|---:|---:|
| 训练 | 17 | 316 | 5056 | 0.21 | 0.95 |
| 开发验证 | 3 | 55 | 880 | 10.41 | 17.07 |
| 最终留出测试 | 5 | 69 | 未用于本次指标 | 未评估 | 未评估 |

311/316 训练窗口在三个采样种子下均通过位姿检查；未通过窗口为 90、235、241、247、327。单动作最大位置误差 2.85 cm，95% 分位 0.44 cm。平均拟合通过不等于所有动作完全拟合。

| 开发演示（分片::记录） | 窗口 | 模型/静止位置 cm | 模型/保持姿态旋转 ° | 打开召回率 |
|---|---:|---:|---:|---:|
| 00003::14 | 21 | 11.04 / 10.15 | 16.15 / 12.68 | 0.0% |
| 00015::31 | 16 | 12.84 / 11.23 | 14.50 / 11.87 | 93.3% |
| 00019::5 | 18 | 7.52 / 7.53 | 20.42 / 14.20 | 0.0% |

开发夹爪平衡准确率约 67.4%，两方向切换为 0/13、0/6。给夹爪头输入真实目标位姿时仍只有约 67.1% 平衡准确率，所以不能只归因于位姿采样误差。真实目标输入仅用于诊断。

## 尚未确定的根因

已完成 [输入与动作范围复查](BRIDGE_DISTRIBUTION_REVIEW.md)：动作长度及相对旋转范围大体重叠；验证仅 4/880 个位移目标超出训练各轴范围。固定 60 帧原始图像哈希及当前 xyz 核验全部通过，粗场景大类在两侧均出现。训练有 20 个动作超过输出裁剪范围，验证没有，因此该边界不能解释整体开发失败。此复查不确定根因，也不重复已有当前位姿消融。

已有证据支持训练集学习能力，但不能确定留出失败的唯一来源。可能涉及上下文对动作阶段的可辨识性、输入信息不足、演示分布差异、融合方式或训练方案。近邻特征复制失败也不能证明 CLIP 完全没有可用信息。

## 下一步与停止条件

Bridge 头部诊断路线已结束。正式阶段见 [FORMAL_EXPERIMENT_PROTOCOL.md](FORMAL_EXPERIMENT_PROTOCOL.md)。更正此前的路径判断：已有 `_append` 会排除 CLIPort，旧绝对事件位姿函数没有经正常路径进入统一训练，不能被当作历史 Bridge 泛化失败根因。现已阻止旧函数绕过调用，新增独立原生读取：10 条训练演示、61 个 primitive 的记录检查与 5 项合同测试通过。本地 LIBERO 原生动作仍为 7 维 OSC_POSE 增量控制。完整旧演示回放与统一转换尚未通过；当前原生执行核查见下节。须完成动作/执行接口检查后再进入完整模型基线。

有限补充方案 [NEXT_EXPERIMENT.md](NEXT_EXPERIMENT.md) 已由作业 188044 完成，按停止条件结束，不再扫描参数。它不替代老师的正式模型与基准实验。

1. 对照成熟机器人扩散策略实现，记录输入、条件注入、动作归一化、预测目标及采样差异。
2. 明确正式数据清单、逐来源动作契约、任务/场景/语言留出及跨机器人协议。
3. 确定可复现基线与阶段范围后再训练；报告独立训练种子的均值和标准差。

不继续重复头部记忆实验或仅增加轮数，不把小样本训练拟合写成泛化，不把离线动作误差写成仿真成功率。已有 3 条开发演示反复查看，正式最终测试需要保持独立。

当前仅一条 `sweep into pile` 指令，不能开展有效的多指令/反事实泛化结论。正式 8 层 Adapter、联合 DDPM 训练、四种增强、基线比较、LIBERO/CLIPort rollout 均未完成。

## CLIPort 原生执行核查（2026-10-06）

固定作者提交后，仅检查训练 seed 0。作者专家在当前场景完成任务（奖励 1）；旧演示按 seed 重建时最大物体位置差 33.13 cm，未执行旧动作。恢复记录初始物体位姿后，前 6 个动作奖励一致，最后堆叠未完成（总奖励 5/6）。这是环境复现缺口，不涉及学习模型，也不能解释 Bridge 离线泛化失败。见 [CLIPORT_NATIVE_EXECUTION.md](CLIPORT_NATIVE_EXECUTION.md)。

后续受控检查：原生抓取均成功且无 primitive 超时。只在最后动作前恢复记录位姿并清零物体速度，总奖励从 5/6 变为 1。说明累积物体状态参与该次回放失败，但干预成功不能计作无辅助成功率；没有确认具体依赖或数值根因。旧版本清单未找到，避免继续盲试版本。见 [干预证据](../reports/cliport_native_state_intervention.json)。

固定当前版本生成的一条作者专家演示（训练模式 seed=0、7 个动作）已通过保存→原生读取→同 seed 无辅助连续回放：奖励 1，初始与逐步物体位姿相同，RGB 及作者 float32 保存精度的深度一致。新增数据 49.23 MiB。**此项没有学习模型，不是泛化实验，也不验证 OpenX 统一加载/动作转换。** [证据](../reports/cliport_fresh_expert_roundtrip.json)。

模型输入/动作合同已核对：主上下文接收 RGB 与语言，当前夹爪可进入夹爪支路，未输入 TCP 位姿。16 个动作共享输入末端参考系；合成坐标往返通过，但它们无法直接作为 CLIPort 两位姿 primitive。视角选择、阶段选择、吸盘/连续控制桥接仍未确定。无网络预测或训练。见 [合同核对](MODEL_BENCHMARK_ACTION_CONTRACT.md)。

自主推进接口：严格事件转换原型与 12 项合同测试完成；作者端点合成轨迹对照在单场景奖励 1，无模型预测。现有 Bridge 真值窗口即使假设初始打开，仅训练 16/316、开发 2/55 满足完整周期；窗口内完整开→关→开为 0。7 个工程对照中的 6 个超出现有 xyz 裁剪范围。因此原生 primitive 映射不作为默认部署路线，后续保留 16 步连续输出并实现明确标注的连续 TCP 控制协议，不等待老师确认。见 [接口实施记录](MODEL_BENCHMARK_ACTION_CONTRACT.md)。

连续 TCP 控制器已实现，读取/动作转换/控制合同测试 19 项通过。实际单场景工程对照中，原样作者底层绝对命令回放奖励 1；相对转换、范围约束与路径拆分后的连续控制仅奖励 1/3，第四个操作奖励不匹配，随后停止。修复吸盘事件引入额外移动后仍失败，因此控制接口尚未验收。没有学习模型、泛化测试或新训练；下一项先隔离坐标转换与路径拆分的影响。见 [工程对照](CLIPORT_NATIVE_EXECUTION.md) 和 [摘要](../reports/cliport_continuous_tcp_control.json)。

隔离检查已完成：只做坐标转换、不拆分路径，奖励 1；保留绝对坐标、只拆分路径，奖励 1/3。这确认新增中间点足以破坏这一作者命令回放，不能继续把路径拆分视为等价转换。不拆分条件需作者命令幅值约 4.237，诊断范围放宽不等于部署允许，生产默认 3 未变。后续先确定动作范围与实际控制协议，不通过裁剪、插入中间点或扩大范围宣称模型已有效。没有学习模型或新训练。见 [隔离证据](../reports/cliport_tcp_subdivision_ablation.json)。

范围与接口已明确：模型及控制器使用同一配置声明的动作表示、0.10m 尺度和界限；新增配置工厂拒绝不兼容几何或缺失界限，21 项合同测试通过。复核已有裁剪记录：训练 20/5,056 目标、10 窗口越界，开发 0/880；平均位置误差下限仅 0.004728cm，不能解释开发失败。生产界限与权重未修改，后续所选训练数据必须先通过范围一致性检查。直接执行合法目标，不补造中间点。控制时间、到达位姿/命令及吸盘迁移仍未验收；没有学习策略 rollout。见 [接口约定](MODEL_BENCHMARK_ACTION_CONTRACT.md)。

时间/夹爪核查：官方名义采集频率 5Hz，抽查一条训练记录没有逐步时间字段；不能视为已完成时序标定。修复无接触关闭请求未激活后不再重试的问题，区分请求、激活、附着；22 项合同测试通过。真实原生吸盘接触前未激活，接触后重复关闭成功附着，打开后释放。修复后单场景不拆分专家命令仍奖励 1；没有学习模型或泛化评估。固定控制周期、标签命令时间与仿真触发时间仍待验收，未改标签或模型。见 [语义与物理检查](../reports/action_timing_gripper_contract.json)。

固定周期对照已执行：一个 16 行作者抓取目标抽样/保持工程块，每目标 96 个实际物理步（0.2s 仿真时间），采用默认动作范围和正高度界限，成功吸附并释放；24 项合同测试通过。但位姿跟踪未通过，最大截止位置误差 6.074cm，旋转接近 180°。作者原生阻塞预定位也有 179.984° 旋转误差；原生 TCP/IK 姿态关系待确认。仅验证时间预算及接触路径，不是堆叠任务评分、模型成功或泛化。此前作者 movej 的计数器存在每物理步双增，旧报告计数差不能直接当物理时长；新模式计数实际调用。见 [证据](../reports/cliport_fixed_period_control.json)。

上述 180° 失败归因已修正：URDF tool_tip link 与 inertial/COM 坐标系混用。三个目标运行确认原生请求接近 COM 姿态；按 URDF/runtime 变换统一执行与测量后，固定周期对照最大旋转误差降至 0.05855°，16/16 姿态通过。位置仍仅 15/16 通过，抬升误差 6.074cm；整体未验收。吸附释放及不拆分专家命令回归对照保持通过，25 项合同测试通过。无模型、训练或新权重；Bridge 泛化结论不变。见 [坐标系修正](../reports/cliport_tcp_frame_correction.json)。

抬升预算已核查：初始关节目标差 1.154rad，96 步理想指令预算 0.96rad，实际路径 0.962rad；后续保持同目标时，第 113 个物理步才进入 1cm/5° 门槛（0.2354s）。原版阻塞抬升用了 114 步（0.2375s）。说明工程轨迹给阻塞最终目标分配 0.2s 缺少依据，不能据此判模型有错；原失败记录保留。未改速度、周期或门槛，后续按实际时钟采集到达轨迹与吸盘事件再构造对照。见 [预算证据](../reports/cliport_lift_deadline_audit.json)。

实际时钟轨迹已采集：5,371 物理步、56 个规则采样时刻和非规则末尾、14 次区间内部吸盘请求，文件约 19KB；原生整条任务奖励 1，物体位置记录一致。完整 16 点因范围被拒绝；默认范围不变，单独执行初始合法的连续 15 点，吸盘保持打开，不恢复状态，最大位置误差 0.2338cm/旋转 0.1511°，前缀跟踪通过。不能推广为 16 点、吸盘事件、任务或模型成功。实际虚拟 TCP 在失败尝试中确实低于零，最低约 −6.44cm，未用放宽工作空间绕过。25 项合同测试通过，事件时序仍待验收。见 [时钟与前缀证据](../reports/cliport_reached_clock_control.json)。

吸盘时序已实跑三条件：同一作者运动命令，原请求时刻奖励 1、逐步物体位置与记录完全相同；一次性请求提前/推迟到 0.2s 边界均第一次操作奖励 0，首次关闭未激活/附着，后续因语言目标不匹配停止。证明这一单场景下边界对齐会破坏一次性事件执行，不能推广为全部 5Hz 策略或模型失败。连续控制器保持关闭重试尚未在此对照评估，释放时刻未单独隔离；下一步先验收保持请求语义。27 项合同测试通过，无训练/新权重/下载，Bridge 泛化结论不变。见 [时序证据](../reports/cliport_suction_timing_control.json)。

保持关闭重试后实跑四条件：原时刻奖励 1；区间起点奖励 1/3、首次附着恢复至 tick=143；区间终点仍首次操作奖励 0，重试也未附着。只提前关闭、保持原释放时刻仍为 1/3，提前释放不是剩余失败唯一原因；第三次附着比原请求提前 8 个物理步，附着对象/接触变化尚未确认。失败条件按原语言目标保护停止，未强行执行余下操作。此项仅作者阻塞命令单场景工程对照，不是模型固定周期接口验收或泛化成功。27 项合同测试通过，未训练或下载。见 [保持请求证据](../reports/cliport_held_suction_control.json)。

附着对象核查完成：第三次两组都吸到正确的 8 号方块；提前条件在 tick=1536 建立约束，原条件为 1544。第三次操作后物体中心位置相差 4.509cm，第四次固定恢复抓取在提前条件未接触附着、原条件成功。只读观察器未改变上一轮奖励、物体误差或时钟结果。已确认状态偏离后旧恢复动作不能照搬，未确定碰撞/摩擦的具体贡献，也不能解释 Bridge 泛化。下一步收敛为最小控制协议与验收条件，再规划模型接入，停止盲调专家回放。27 项测试通过，无训练或下载。见 [对象与状态证据](../reports/cliport_suction_object_audit.json)。
# 2026-10-16：下一项只测试训练图像增强的泛化效果

**训练覆盖207348已完成并显示两项主要泛化改善（终端待工件复核）。** 原40开发7.079412cm/14.559741°，均低于旧128窗控制及静止，达到事前均值门槛；至少6/10演示优于旧控制等尚待核验，未判全部通过。8192更新/54分19秒结束，当前只取两run小工件，不下载缓存/权重、不再跑NN。见[结果与下载命令](BRIDGE_TRAINING_COVERAGE_RESULT_207348_20261016.md)。下文未提交/运行状态为历史。

**新的有限项是训练覆盖泛化对照，增强已结束。** 用户同意后准备405演示/1836窗，保留原40开发与原65536任务槽、模型、状态统计、初始化及8192训练更新；只新增一个全池组。局部16步本地预检/五预测核验通过（224输入/120演示，99秒/5.658GiB），不等于全池验收，没有开发评分、本地完整缓存或正式训练。下一步仅提交全415演示/1876窗集群预检，旧控制不重跑。见[固定计划与命令](BRIDGE_TRAINING_COVERAGE_CONTROL_20261016.md)。

**更新：该项完整工件核验完成，不再执行。** 29文件/13预测、8192更新及原数据/源码/初始化/增强轨迹通过，开发8.499733cm/15.764689°，候选失败收口。已补仅实际条件的固定全局位置读出（无额外状态/任务分组），7.977745cm但仅5/10演示优于静止，未达到定位门槛；没有新NN或集群作业。优先准备同当前配方、固定预算的数据覆盖对照，直接用既有405训练演示/1836窗，原40开发不变；不做逐级窗口记忆。具体边界、旧1836配方差异与停止条件见[下一步决定](BRIDGE_GENERALIZATION_NEXT_STEP_20261016.md)。下文为历史状态。

**该项已完成206915（终端证据，完整工件待复核）**：8192更新/56分43秒，开发8.499733cm/15.764585°，两项均高于原控制和静止，按事前判据结束。训练7.296mm/2.548°不影响以泛化为主的决定，不追加拟合、增强强度或种子。后续只下载两run不含PT的小预测与日志核验，不重跑NN。见[结果](BRIDGE_IMAGE_AUGMENTATION_RESULT_206915_20261016.md)；下文准备状态为历史。

用户更正并确认以独立留出演示泛化为主，拟合只作检查。对比成功PushT实际权重配置（84/96随机裁剪）与Bridge迁移参照（crop_shape:null）后，准备一个保留完整视觉信息的固定±8像素训练平移候选，评估仍使用原图。继承原128/40数据、CLIP+Adapter+DDPM与8192更新，旧202278控制直接复用，不重开206274均值消融。16步本地入口和小工件独立核验通过（109秒/5.074GiB），无开发评分、新下载或正式训练；当前尚未提交集群。按[已冻结判据和命令](BRIDGE_IMAGE_AUGMENTATION_CONTROL_20261016.md)先做一次集群预检，不能据入口通过宣称泛化改善。

# 2026-10-30：新位姿接口的单次真实仿真入口已实现，等待完整权重

B2的软件接线修复之后，已实现 `diagnostics/run_bridge_pose_libero_engineering_20261030.py`：固定排除任务/既有初态，最多200tick、50次调用、含加载600秒硬上限，使用完整234100 base+pose-port；原5Hz/20Hz执行器不变。单独显式未标定mapping，输入与输出都应用，不读取专家动作/未来观测，0训练/正式评分/自动重试。默认就绪检查及语法检查完成，唯一缺项为本机完整234100权重，其他既有资产存在；实际模型加载和仿真尚未开始。用户需按既有手动scp规则取回，助手不SSH。尚未新增解除B1/B2物理或能力阻塞；下一项仅执行该固定回合并收口，不再增加诊断系列。见[入口、预算和命令](BRIDGE_POSE_LIBERO_ENGINEERING_20261030.md)及 `reports/bridge_pose_libero_engineering_20261030/readiness.json`。

