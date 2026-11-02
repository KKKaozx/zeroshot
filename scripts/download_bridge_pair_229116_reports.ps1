$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$TaskDestination = Join-Path $TaskRoot 'reports\bridge_backbone_pair_a40_229116\cluster'
New-Item -ItemType Directory -Path $TaskDestination -Force | Out-Null
$TaskRemote = 'zixiao005@10.97.216.128:/projects/Zeroshot/runs/bridge-backbone-pair-a40-229116'
$TaskArchive = 'bridge-backbone-pair-a40-229116-reports.tar.gz'
scp "${TaskRemote}/${TaskArchive}.sha256" $TaskDestination
if ($LASTEXITCODE -ne 0) { throw 'Checksum download failed; first package the reports on the cluster.' }
scp "${TaskRemote}/${TaskArchive}" $TaskDestination
if ($LASTEXITCODE -ne 0) { throw 'Reports download failed.' }
$TaskPython = Join-Path $TaskRoot 'training_cache\libero_native_env_v1\Scripts\python.exe'
& $TaskPython -B (Join-Path $TaskRoot 'scripts\verify_bridge_pair_229116_download.py')
if ($LASTEXITCODE -ne 0) { throw 'Local verification failed; retain downloaded files for diagnosis.' }
