# Run the whole study end to end.  Usage: .\scripts\run_all.ps1 [config] [out_dir]
param(
    [string]$Config = "configs/default.yaml",
    [string]$Out = "results"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:PYTHONPATH = "$($env:PYTHONPATH);src"
python -m rsv.cli all --config $Config --out $Out

Write-Host ""
Write-Host "Done.  Summary: $Out/summary.md   Figures: $Out/figures/"
