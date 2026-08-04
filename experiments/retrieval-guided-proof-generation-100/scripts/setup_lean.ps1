$ErrorActionPreference = 'Stop'

$experiment = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtimeRoot = Join-Path $experiment '.runtime'
$elanArchive = Join-Path $runtimeRoot 'elan.zip'
$elanInstallerDirectory = Join-Path $runtimeRoot 'elan-init'
$elanInstaller = Join-Path $elanInstallerDirectory 'elan-init.exe'
$env:ELAN_HOME = Join-Path $runtimeRoot 'elan-home'
$mathlib = Join-Path $runtimeRoot 'mathlib4-pinned'
$commit = '3c307701fa7e9acbdc0680d7f3b9c9fed9081740'

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
New-Item -ItemType Directory -Force -Path $elanInstallerDirectory | Out-Null

if (-not (Test-Path $elanInstaller)) {
    Invoke-WebRequest `
        -UseBasicParsing `
        -Uri 'https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-pc-windows-msvc.zip' `
        -OutFile $elanArchive
    Expand-Archive -LiteralPath $elanArchive -DestinationPath $elanInstallerDirectory -Force
}

if (-not (Test-Path (Join-Path $env:ELAN_HOME 'bin/elan.exe'))) {
    & $elanInstaller -y --no-modify-path --default-toolchain none
}

if (-not (Test-Path (Join-Path $mathlib '.git'))) {
    git clone --filter=blob:none --no-checkout https://github.com/leanprover-community/mathlib4.git $mathlib
    git -C $mathlib fetch --depth 1 origin $commit
    git -C $mathlib checkout --detach $commit
}

if ((git -C $mathlib rev-parse HEAD).Trim() -ne $commit) {
    throw "Mathlib checkout is not pinned to $commit"
}

$toolchain = (Get-Content (Join-Path $mathlib 'lean-toolchain')).Trim()
$elan = Join-Path $env:ELAN_HOME 'bin/elan.exe'
$lake = Join-Path $env:ELAN_HOME 'bin/lake.exe'
& $elan toolchain install $toolchain

$env:PATH = (Join-Path $env:ELAN_HOME 'bin') + ';' + $env:PATH
Push-Location $mathlib
try {
    # Use the committed dependency manifest. Do not run `lake update`, which
    # would move this historical environment away from its pinned revisions.
    & $lake exe cache get
    & $lake env lean Mathlib/Algebra/Ring/Commute.lean
} finally {
    Pop-Location
}

Write-Output "Lean environment ready at $mathlib"
