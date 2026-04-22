# Appends a progress line to logs/torch_cuda_install.log every 10s
# until the torch wheel download finishes (2.5GB target).
$log = "logs\download_progress.log"
$target = 2532 * 1MB
while ($true) {
  $files = Get-ChildItem -Path "$env:TEMP\pip-unpack-*\torch-*.whl" -ErrorAction SilentlyContinue
  if ($files) {
    $active = $files | Sort-Object LastWriteTime -Desc | Select-Object -First 1
    $mb = [int]($active.Length / 1MB)
    $pct = [int](100 * $active.Length / $target)
    $ts = Get-Date -Format "HH:mm:ss"
    "[$ts] download $mb MB / 2532 MB ($pct%)  $($active.FullName)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($pct -ge 99) { "[$ts] download complete" | Out-File -FilePath $log -Append -Encoding utf8; break }
  } else {
    "[$(Get-Date -Format HH:mm:ss)] no active pip-unpack file (install finished or not started)" | Out-File -FilePath $log -Append -Encoding utf8
    break
  }
  Start-Sleep -Seconds 10
}
