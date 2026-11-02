$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath 'D:\ntu_related\dissertation\Zero_shot').ProviderPath.TrimEnd('\')
$taskOut = Join-Path $taskRoot 'reports\local_storage_cleanup_extended_20261026'
$taskPlanPath = Join-Path $taskOut 'plan.json'
$taskReceiptPath = Join-Path $taskOut 'receipt.json'
if (Test-Path -LiteralPath $taskReceiptPath) { throw 'An execution receipt already exists; no second deletion pass.' }
if ((Get-FileHash -LiteralPath $taskPlanPath -Algorithm SHA256).Hash -ne '708F3287B115E00669154E161E508C70E31D1A0AACF26E2AD699858FD5E3C835') { throw 'Reviewed plan changed.' }
$taskPlan = Get-Content -LiteralPath $taskPlanPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskPlan.root -ne $taskRoot -or @($taskPlan.targets).Count -ne 130) { throw 'Unexpected root or target count.' }
if ((Get-FileHash -LiteralPath $taskPlan.retained_package_snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskPlan.snapshot_sha256) { throw 'Retained package contents changed.' }
$taskAllowedDirs = @(
    '.tmp\bridge_official_unet_evaluation_v1', '.tmp\bridge_official_unet_preflight_v1',
    '.tmp\bridge_official_unet_real_preflight_v1', '.tmp\bridge_official_unet_training_v1',
    '.tmp\multiscale_noise_probe_v1', '.tmp\multiscale_noise_probe_v2', '.tmp\multiscale_tail_probe_v1',
    '.tmp\libero_pip_cache', '.tmp\libero_numba'
)
function Assert-TaskPath($taskPath) {
    $taskResolved = (Resolve-Path -LiteralPath $taskPath).ProviderPath
    if (-not $taskResolved.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFullPath($taskPath) -ne $taskResolved) { throw "Out-of-scope or noncanonical path: $taskPath" }
    $taskItem = Get-Item -LiteralPath $taskResolved -Force
    $taskAncestor = $taskItem
    while ($taskAncestor.FullName -ne $taskRoot) {
        if (($taskAncestor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Linked target/ancestor refused: $taskPath" }
        $taskAncestor = if ($taskAncestor -is [IO.DirectoryInfo]) { $taskAncestor.Parent } else { $taskAncestor.Directory }
        if ($null -eq $taskAncestor -or (-not $taskAncestor.FullName.StartsWith($taskRoot, [StringComparison]::OrdinalIgnoreCase))) { throw 'Ancestor escaped workspace.' }
    }
    return $taskItem
}

# Validate the entire reviewed set before deleting anything, including exact recursive roots.
foreach ($taskTarget in $taskPlan.targets) {
    $taskItem = Assert-TaskPath $taskTarget.path
    $taskRelative = $taskTarget.relative_path.Replace('/', '\')
    if ($taskTarget.kind -eq 'file') {
        if ($taskItem.PSIsContainer -or $taskItem.Length -ne [long]$taskTarget.bytes) { throw "File changed: $($taskTarget.path)" }
        $taskMtime = [long](($taskItem.LastWriteTimeUtc.Ticks - [long]621355968000000000) * [long]100)
        if ($taskMtime -ne [long]$taskTarget.mtime_ns) { throw "Timestamp changed: $($taskTarget.path)" }
        if (-not ($taskRelative.StartsWith('cluster_packages\') -or $taskRelative.StartsWith('training_cache\exports\') -or $taskRelative.StartsWith('results\bcz_'))) { throw 'Unexpected file category scope.' }
        if ($taskTarget.category -eq 'retired historical BC-Z small-sample fit checkpoint') {
            if ($taskItem.Name -ne 'best.pt') { throw 'Unexpected checkpoint name.' }
            foreach ($taskEvidence in @($taskTarget.retained_metadata, $taskTarget.retained_history, $taskTarget.retained_manifest)) {
                $taskKept = Assert-TaskPath $taskEvidence
                if ($taskKept.PSIsContainer -or $taskKept.Length -lt 2) { throw 'Retained checkpoint evidence missing.' }
            }
        }
    } elseif ($taskTarget.kind -eq 'directory') {
        if (-not $taskItem.PSIsContainer -or $taskRelative -notin $taskAllowedDirs) { throw 'Unexpected recursive deletion root.' }
        $taskChildren = @(Get-ChildItem -LiteralPath $taskItem.FullName -Force -Recurse)
        if (@($taskChildren | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count -ne 0) { throw 'Directory contains a reparse point.' }
        $taskFiles = @($taskChildren | Where-Object { -not $_.PSIsContainer })
        if ($taskFiles.Count -ne @($taskTarget.files).Count) { throw "Directory inventory changed: $($taskTarget.path)" }
        $taskExpected = @{}
        foreach ($taskFile in $taskTarget.files) { $taskExpected[$taskFile.path] = [long]$taskFile.bytes }
        foreach ($taskFile in $taskFiles) {
            if (-not $taskExpected.ContainsKey($taskFile.FullName) -or $taskExpected[$taskFile.FullName] -ne $taskFile.Length) { throw 'Directory member changed.' }
        }
    } else { throw 'Unknown target kind.' }
}

$taskReceipt = [ordered]@{
    date = '2026-10-26'; root = $taskRoot; status = 'validated';
    free_bytes_before_execution = [long](Get-PSDrive D).Free;
    deleted = @(); deleted_bytes = [long]0; retained_snapshot_bytes = [long]$taskPlan.snapshot_bytes;
    cluster_contacted = $false; error = $null
}
try {
    foreach ($taskTarget in $taskPlan.targets) {
        $taskItem = Assert-TaskPath $taskTarget.path
        if ($taskTarget.kind -eq 'directory') { Remove-Item -LiteralPath $taskItem.FullName -Recurse -Force }
        else { Remove-Item -LiteralPath $taskItem.FullName -Force }
        if (Test-Path -LiteralPath $taskTarget.path) { throw 'Deletion did not complete.' }
        $taskReceipt.deleted += [ordered]@{path=$taskTarget.relative_path;bytes=[long]$taskTarget.bytes;category=$taskTarget.category}
        $taskReceipt.deleted_bytes += [long]$taskTarget.bytes
        $taskReceipt.status = 'deleting'
        $taskReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReceiptPath -Encoding utf8
    }
    $taskReceipt.status = 'completed'
} catch {
    $taskReceipt.status = 'stopped_after_error'
    $taskReceipt.error = $_.Exception.Message
    throw
} finally {
    $taskReceipt['free_bytes_after_execution'] = [long](Get-PSDrive D).Free
    $taskReceipt['net_freed_bytes_after_snapshot'] = $taskReceipt.deleted_bytes - $taskReceipt.retained_snapshot_bytes
    $taskReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskReceiptPath -Encoding utf8
}
[ordered]@{status=$taskReceipt.status;deleted_targets=$taskReceipt.deleted.Count;deleted_bytes=$taskReceipt.deleted_bytes;net_freed_bytes_after_snapshot=$taskReceipt.net_freed_bytes_after_snapshot;free_bytes=$taskReceipt.free_bytes_after_execution} | ConvertTo-Json
