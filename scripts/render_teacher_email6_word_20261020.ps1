$ErrorActionPreference = 'Stop'
$taskDocx = 'D:\ntu_related\dissertation\老师邮件\第六封邮件.docx'
$taskPdf = 'D:\ntu_related\dissertation\Zero_shot\reports\teacher_email6_20261020\render\email6.pdf'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $taskPdf) | Out-Null
$taskWord = $null
$taskDocument = $null
try {
  $taskWord = New-Object -ComObject Word.Application
  $taskWord.Visible = $false
  $taskWord.DisplayAlerts = 0
  $taskDocument = $taskWord.Documents.Open($taskDocx, $false, $true)
  $taskDocument.ExportAsFixedFormat($taskPdf, 17)
  Write-Output ('Word PDF exported: ' + $taskPdf)
} finally {
  if ($null -ne $taskDocument) {
    $taskDocument.Close(0)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskDocument)
  }
  if ($null -ne $taskWord) {
    $taskWord.Quit(0)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskWord)
  }
}
