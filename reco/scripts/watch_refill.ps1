# watch_refill.ps1 -- auto-regenerate paper_filled/ whenever a new final.json
# lands anywhere under runs_v2/grid/. Designed to run alongside run_grid_v2.ps1.
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\watch_refill.ps1
#
# Optional args:
#   -IntervalSec 30      poll interval (default: 30)
#   -GridRoot runs_v2/grid
#   -PaperDir paper
#   -FilledDir paper_filled
#   -Once                run a single refill and exit (useful for CI / manual)
#   -Quiet               suppress per-tick status output
#
# What it does each tick:
#   1. Enumerate all final.json files under $GridRoot.
#   2. Hash (path + mtime + size) tuples to a fingerprint.
#   3. If the fingerprint changed since last tick, re-run fill_placeholders.py
#      and print a short summary (resolved / remaining placeholders).
#
# Exit: Ctrl-C. The refill script itself is idempotent, so interrupting in
# the middle is safe -- the next tick will pick up where this one left off.

param(
    [int]    $IntervalSec = 30,
    [string] $GridRoot    = "runs_v2/grid",
    [string] $PaperDir    = "paper",
    [string] $FilledDir   = "paper_filled",
    [switch] $Once,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to repo root (script lives in scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$GridRoot  = Join-Path $RepoRoot $GridRoot
$PaperDir  = Join-Path $RepoRoot $PaperDir
$FilledDir = Join-Path $RepoRoot $FilledDir
$FillScript = Join-Path $RepoRoot "scripts\fill_placeholders.py"

if (-not (Test-Path $FillScript)) {
    Write-Error "fill_placeholders.py not found at $FillScript"
    exit 1
}

function Get-GridFingerprint {
    param([string] $Root)
    if (-not (Test-Path $Root)) { return "" }
    $files = Get-ChildItem -Path $Root -Recurse -Filter "final.json" -ErrorAction SilentlyContinue |
             Sort-Object FullName
    if ($null -eq $files -or $files.Count -eq 0) { return "" }
    $parts = foreach ($f in $files) {
        "{0}|{1}|{2}" -f $f.FullName, $f.LastWriteTimeUtc.Ticks, $f.Length
    }
    $joined = $parts -join "`n"
    $hasher = [System.Security.Cryptography.SHA1]::Create()
    $bytes  = [System.Text.Encoding]::UTF8.GetBytes($joined)
    $hash   = [BitConverter]::ToString($hasher.ComputeHash($bytes)) -replace "-", ""
    return "$($files.Count):$hash"
}

function Invoke-Refill {
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$stamp] change detected -- regenerating $FilledDir" -ForegroundColor Cyan
    $out = & py -3.12 $FillScript --paper_in $PaperDir --paper_out $FilledDir --runs $GridRoot 2>&1 | Out-String
    $ec = $LASTEXITCODE
    if ($ec -ne 0) {
        Write-Host "  fill_placeholders.py FAILED (exit $ec)" -ForegroundColor Red
        if ($out) { Write-Host $out -ForegroundColor Red }
        return $false
    }
    # Print only the summary lines from the filler -- it prints a RESOLVED/REMAINING
    # block at the end; keep the tail for at-a-glance status.
    if ($out) {
        $tail = ($out -split "`n") | Select-Object -Last 12
        foreach ($line in $tail) {
            if ($line.Trim()) { Write-Host "  $line" }
        }
    }
    return $true
}

# ---------- main loop ----------
Write-Host "watch_refill: polling $GridRoot every ${IntervalSec}s" -ForegroundColor Green
Write-Host "  paper source: $PaperDir"
Write-Host "  filled out:   $FilledDir"
Write-Host "  Ctrl-C to stop." -ForegroundColor DarkGray

$lastFingerprint = ""
$tickCount = 0

while ($true) {
    $tickCount++
    $fp = Get-GridFingerprint -Root $GridRoot
    $runCount = if ($fp) { ($fp -split ":")[0] } else { 0 }

    if ($fp -ne $lastFingerprint) {
        if (-not $Quiet) {
            Write-Host "[tick $tickCount] runs=$runCount (was fingerprint=$lastFingerprint)" -ForegroundColor DarkGray
        }
        $ok = Invoke-Refill
        if ($ok) { $lastFingerprint = $fp }
    } elseif (-not $Quiet) {
        $stamp = Get-Date -Format "HH:mm:ss"
        Write-Host "[$stamp] tick $tickCount -- no change (runs=$runCount)" -ForegroundColor DarkGray
    }

    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
}
