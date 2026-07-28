
$paths = @(
  'outputs\task1\checkpoints','outputs\task1\oof','outputs\task1\logs','outputs\task1\resume_audit','outputs\task1\paper',
  'outputs\task2\checkpoints','outputs\task2\oof','outputs\task2\logs','outputs\task2\resume_audit','outputs\task2\paper',
  'outputs\task3\checkpoints','outputs\task3\oof','outputs\task3\logs','outputs\task3\resume_audit','outputs\task3\paper',
  'outputs\task3_audit','backups'
)
foreach ($p in $paths) {
  if (Test-Path $p) { Remove-Item -Recurse -Force $p; Write-Host "removed $p" }
}
Get-ChildItem outputs -Recurse -File -Filter '*_oof*.csv' -EA 0 | Remove-Item -Force
Get-ChildItem outputs -Recurse -File -Filter '*.cbm' -EA 0 | Remove-Item -Force
Get-ChildItem . -Filter '*.log' -File -EA 0 | Remove-Item -Force
Write-Host 'done'
