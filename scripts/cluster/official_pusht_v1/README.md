# 官方 Diffusion Policy：图像版 Push-T

195374 的 OpenCV、pip check 和作者 PushTImageEnv.reset 均已通过。下载阶段系统 curl 因环境 LD_LIBRARY_PATH 加载了 Conda libcurl 而发生 ABI 冲突。setup.sh 改为仅在调用 /usr/bin/curl 时移除 LD_LIBRARY_PATH；Python 运行配置不变。尚未确认 checkpoint 下载完成。

最终 OpenCV 修复方案改为 headless 4.6.0.66：原 X11 1.8.4 不满足所选 libglx 依赖，直接库修复在修改前停止。repair_headless.sh 使用 force-remove 仅移除 Conda 中 opencv/py-opencv/libopencv，保留其他包，再装无 GUI 的同版本 cv2 wheel，避免重求解环境或升级 X11。此为环境构建差异，必须以实际 reset 和 rollout 日志确认；旧 repair_gl.sh 已弃用。

195183 安装作业已完成环境创建及 pip check，但在导入 JsonLogger 时发现精简环境漏了 pandas；未到权重下载阶段。补入 pandas 1.5.3（适配 Python 3.9 / NumPy 1.23.3）。既有环境可仅安装此依赖后复用 setup.sh，不必重建环境。

195216 已补 pandas，但 OpenCV 在计算节点导入时缺少 libGL.so.1。repair_gl.sh 在该独立环境添加 conda-forge libgl/libglx 1.7.0，使用 freeze-installed 保持既有包版本，设置环境内动态库路径后重试导入并继续 setup。保留原 OpenCV，不混装另一套 cv2。集群修复效果尚待实际日志确认。

只准备集群复现；本地尚未运行 Linux 环境或评估。

固定仓库：https://github.com/real-stanford/diffusion_policy

固定提交：5ba07ac6661db573af695b419a7947ecb704690f。

setup.sh 用 CPU Slurm 作业安装独立环境，验证导入和环境 reset，然后下载作者单个 4,293,295,813-byte checkpoint。evaluate.sh 用 A40 调用未修改的 eval.py。保留 checkpoint 的原模型、EMA、100 步 DDPM、6 个训练初始化、50 个测试初始化及视频输出。不训练、不下载训练集。

environment.yaml 从作者 conda_environment.yaml 中选取 Push-T 评估依赖。核心 PyTorch/CUDA/gym/Pymunk/diffusers 等版本保持作者设置；省略其他任务的 MuJoCo、PyTorch3D、R3M、Ray 等依赖。额外固定 setuptools 65.5.0、huggingface-hub 0.14.1、protobuf 3.20.3，避免旧代码依赖开放版本范围漂移。这是环境调整，不能称完全原封不动的环境复现。集群安装兼容性尚待实际作业验证。

200 GB 是用户报告的项目配额，脚本使用 du apparent bytes 做保守规划，没有查询实时配额。安装预留 20 GiB，评估预留 2 GiB。缓存和环境全部放 /projects/Zeroshot；没有清理用户已有文件。

下载按服务器文件大小校验；SHA256 是首次下载后记录的本地指纹，用于后续重放完整性检查，作者未提供可核对的 SHA256，因此不宣称获得作者 checksum 验证。

在集群项目目录解压后：

```bash
mkdir -p /projects/Zeroshot/logs
PUSHT_SETUP=$(sbatch --parsable /projects/Zeroshot/official_pusht_v1/setup.sh)
echo "安装作业：$PUSHT_SETUP"
```

安装完成后检查 State=COMPLETED、ExitCode=0:0 及日志最后的 SETUP COMPLETE，再提交评估：

```bash
PUSHT_EVAL=$(sbatch --parsable /projects/Zeroshot/official_pusht_v1/evaluate.sh)
echo "评估作业：$PUSHT_EVAL"
```

结果位于 /projects/Zeroshot/runs/official-pusht-eval-作业号/eval_log.json 和 media。test/mean_score 是官方任务覆盖得分，不是抓取成功率。作者选定权重文件名包含 0.884，复现值尚未知。Push-T 没有语言、CLIP、夹爪，不能据此验证我们 Bridge 策略的泛化。
