# Fitted-state late learning-rate control

Same seed43/32 train windows and recipe as trace200039. Common warmup1024 then branch with Adam state restored; only continuation LR differs. Preflight verifies continuous versus restored same-LR exact replay. Common warmup must fit and match origin. Constant arm must reproduce deterioration before causal interpretation. No model, dataset, package downloads; no gripper or simulation training. PT files stay on cluster, capped4GiB.
