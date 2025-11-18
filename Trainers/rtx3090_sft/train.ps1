# ============================================================================
# SFT Training Script for Windows PowerShell
# ============================================================================
# This script will:
# 1. Check prerequisites (conda, GPU, disk space)
# 2. Activate unsloth_env
# 3. Verify dataset exists
# 4. Run SFT training with Mistral 7B
# ============================================================================

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "SFT Training - Mistral 7B v0.3" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 1: Navigate to training directory
# ============================================================================
Write-Host "[1/6] Navigating to training directory..." -ForegroundColor Yellow

$ProjectRoot = "C:\Users\Joseph\Documents\Code\Toolset-Training"
$TrainingDir = Join-Path $ProjectRoot "Trainers\rtx3090_sft"
$DatasetPath = Join-Path $ProjectRoot "Datasets\syngen_tools_sft_11.18.25.jsonl"

Set-Location $TrainingDir
Write-Host "  [OK] Current directory: $TrainingDir" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Check dataset exists
# ============================================================================
Write-Host "[2/6] Checking dataset..." -ForegroundColor Yellow

if (-not (Test-Path $DatasetPath)) {
    Write-Host "  [ERROR] Dataset not found at $DatasetPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$DatasetSizeMB = [math]::Round((Get-Item $DatasetPath).Length / 1MB, 1)
Write-Host "  [OK] Dataset found: $DatasetSizeMB MB" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 3: Check disk space
# ============================================================================
Write-Host "[3/6] Checking disk space..." -ForegroundColor Yellow

$Drive = Get-PSDrive C
$FreeSpaceGB = [math]::Round($Drive.Free / 1GB, 2)
$RequiredGB = 30

if ($FreeSpaceGB -lt $RequiredGB) {
    Write-Host "  [ERROR] Insufficient disk space" -ForegroundColor Red
    Write-Host "  [ERROR] Available: $FreeSpaceGB GB" -ForegroundColor Red
    Write-Host "  [ERROR] Required: $RequiredGB GB" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Disk space available: $FreeSpaceGB GB" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 4: Find Python from unsloth_env
# ============================================================================
Write-Host "[4/6] Finding Python from unsloth_env..." -ForegroundColor Yellow

$UnslothEnvPaths = @(
    "$env:USERPROFILE\miniconda3\envs\unsloth_env\python.exe",
    "$env:USERPROFILE\anaconda3\envs\unsloth_env\python.exe",
    "C:\ProgramData\miniconda3\envs\unsloth_env\python.exe",
    "C:\ProgramData\anaconda3\envs\unsloth_env\python.exe"
)

$PythonExe = $null
foreach ($path in $UnslothEnvPaths) {
    if (Test-Path $path) {
        $PythonExe = $path
        break
    }
}

if (-not $PythonExe) {
    Write-Host "  [ERROR] unsloth_env Python not found" -ForegroundColor Red
    Write-Host "  [INFO] Searched in:" -ForegroundColor Yellow
    foreach ($path in $UnslothEnvPaths) {
        Write-Host "    - $path" -ForegroundColor Yellow
    }
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "  [OK] Python found: $PythonExe" -ForegroundColor Green

$PythonVersion = & $PythonExe --version 2>&1
Write-Host "  [OK] Version: $PythonVersion" -ForegroundColor Green

# Test CUDA availability
Write-Host "  -> Testing CUDA availability..." -ForegroundColor Cyan
$CudaTest = & $PythonExe -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
if ($CudaTest -match "CUDA") {
    $GpuName = & $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    Write-Host "  [OK] CUDA available: $GpuName" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] CUDA not available!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# ============================================================================
# STEP 5: Display training configuration
# ============================================================================
Write-Host "[5/6] Training configuration..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Model:           Mistral-7B-v0.3" -ForegroundColor White
Write-Host "  Method:          SFT (Supervised Fine-Tuning)" -ForegroundColor White
Write-Host "  Dataset:         syngen_tools_sft_11.18.25.jsonl (2,676 examples)" -ForegroundColor White
Write-Host "  Batch size:      6 (effective: 24 with grad accum)" -ForegroundColor White
Write-Host "  Learning rate:   2e-4" -ForegroundColor White
Write-Host "  Epochs:          3" -ForegroundColor White
Write-Host "  Max seq length:  2048 tokens" -ForegroundColor White
Write-Host ""
Write-Host "  Expected time:   ~45 minutes (3 epochs × ~15 min/epoch)" -ForegroundColor Cyan
Write-Host "  Expected VRAM:   ~7-9 GB" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# STEP 6: Confirm and run training
# ============================================================================
Write-Host "[6/6] Ready to start training" -ForegroundColor Yellow
Write-Host ""
$Confirmation = Read-Host "Start training? (y/n)"

if ($Confirmation -ne "y" -and $Confirmation -ne "Y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "STARTING TRAINING" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Run training with local dataset
& $PythonExe train_sft.py `
    --model-size 7b `
    --local-file $DatasetPath

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
if ($ExitCode -eq 0) {
    Write-Host "TRAINING COMPLETED SUCCESSFULLY" -ForegroundColor Green
} else {
    Write-Host "TRAINING FAILED (Exit code: $ExitCode)" -ForegroundColor Red
}
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
exit $ExitCode
