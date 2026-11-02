# 冻结状态逐项对比
仅CPU读取/初始化预检与正式模型配置，按参数名、形状、dtype、原始字节核对全部固定张量，同时单列6个扩散调度缓冲区与分词器。零前向、零优化、零原始图像读取，不修改现有训练包或身份守卫。
4CPU/12GB/最长10分钟，不申请GPU。实际旧预检号从cluster_preflight_binding.json取得；CPU重建若不能复现旧摘要会明确说明，不会自动重绑或绕过。
上传解压后运行bash submit_once.sh。完成后cat /projects/Zeroshot/runs/oxe-id-cpu-JOBID/report.json，报告与两个完整profile都保留在该新目录。
