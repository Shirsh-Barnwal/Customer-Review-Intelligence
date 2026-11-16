# setup_env.ps1 -- Automated environment setup for SentinelDrift
# ================================================================
# Usage:  powershell -ExecutionPolicy Bypass -File setup_env.ps1
#
# What this does:
#   1. Configures conda user-space dirs (if needed)
#   2. Creates the conda environment from environment.yml
#   3. Installs PyTorch with CUDA 12.8 (RTX PRO 5000 / Blackwell sm_120)
#   4. Downloads NLTK data
#   5. Verifies GPU and critical imports

$ErrorActionPreference = "Stop"
$EnvName = "sentineldrift"

function Invoke-CondaPythonScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Code,

        [Parameter(Mandatory = $true)]
        [string]$StepName
    )

    $tmpScript = Join-Path $env:TEMP ("sentineldrift_" + [guid]::NewGuid().ToString("N") + ".py")

    try {
        Set-Content -Path $tmpScript -Value $Code -Encoding UTF8
        conda run -n $EnvName python $tmpScript

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: $StepName failed. Check output above." -ForegroundColor Red
            exit 1
        }
    } finally {
        Remove-Item $tmpScript -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n=== SentinelDrift Environment Setup ===" -ForegroundColor Cyan

# ---------------------------------------------------------------
# Step 0: Ensure conda user-space dirs exist (avoids write errors
#         on IT-managed machines where base env is read-only)
# ---------------------------------------------------------------
$UserPkgs = "$env:USERPROFILE\.conda\pkgs"
$UserEnvs = "$env:USERPROFILE\.conda\envs"

# Some managed installs force tar.bz2-only mode, which hides many modern
# .conda builds and causes false PackagesNotFound errors during solve.
conda config --set use_only_tar_bz2 false

if (-not (Test-Path $UserPkgs)) {
    Write-Host "Configuring user-space conda dirs..." -ForegroundColor Yellow
    conda config --add pkgs_dirs $UserPkgs
    conda config --add envs_dirs $UserEnvs
}

# ---------------------------------------------------------------
# Step 1: Create the conda environment
# ---------------------------------------------------------------
Write-Host "`n[1/5] Creating conda environment from environment.yml..." -ForegroundColor Green

# Check if env already exists
$existingEnvs = conda env list 2>&1 | Select-String "^$EnvName\s"
if ($existingEnvs) {
    Write-Host "Environment '$EnvName' already exists. Updating..." -ForegroundColor Yellow
    conda env update -n $EnvName -f environment.yml --prune
} else {
    conda env create -f environment.yml
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: conda env create/update failed. Check output above." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------
# Step 2: Install PyTorch with CUDA 12.8
# ---------------------------------------------------------------
Write-Host "`n[2/5] Installing PyTorch (cu128 for Blackwell sm_120)..." -ForegroundColor Green

conda run -n $EnvName python -m pip install `
    torch torchvision torchaudio `
    --upgrade --force-reinstall `
    --index-url https://download.pytorch.org/whl/cu128 `
    --extra-index-url https://pypi.org/simple

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch install failed. Check output above." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------
# Step 3: Download NLTK data (used by pyabsa_runner.py)
# ---------------------------------------------------------------
Write-Host "`n[3/5] Downloading NLTK data..." -ForegroundColor Green

conda run -n $EnvName python -c "import nltk; nltk.download('punkt_tab', quiet=True); print('NLTK punkt_tab OK')"

# ---------------------------------------------------------------
# Step 4: Verify GPU
# ---------------------------------------------------------------
Write-Host "`n[4/5] Verifying GPU setup..." -ForegroundColor Green

Invoke-CondaPythonScript -StepName "GPU verification" -Code @"
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
arch_list = torch.cuda.get_arch_list()
print(f'sm_120 present: {"sm_120" in arch_list}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected -- CUDA ops will fail at runtime')
"@

# ---------------------------------------------------------------
# Step 5: Verify critical imports
# ---------------------------------------------------------------
Write-Host "`n[5/5] Verifying critical imports..." -ForegroundColor Green

Invoke-CondaPythonScript -StepName "Critical import verification" -Code @"
import numpy, pandas, scipy, sklearn, matplotlib, seaborn
import torch, transformers, pyabsa, ruptures, streamlit
import statsmodels, h5py, nltk, langdetect, captum
print('All critical imports OK')
"@

Write-Host "`n=== Setup complete! ===" -ForegroundColor Cyan
Write-Host "Activate with:  conda activate $EnvName"
Write-Host "Or use:         conda run -n $EnvName <command>`n"
