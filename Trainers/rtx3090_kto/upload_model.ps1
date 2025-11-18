# Upload to HuggingFace - PowerShell Script
# Interactive model upload with folder selection

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "HuggingFace Model Upload" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue
Write-Host ""

# Load environment variables from root .env
$EnvFile = "..\..\.env"
$CondaEnv = "unsloth_env"  # Default

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^HF_API_KEY=(.+)$') {
            $env:HF_TOKEN = $matches[1]
        }
        elseif ($_ -match '^CONDA_ENV=(.+)$') {
            $CondaEnv = $matches[1]
        }
    }

    if ($env:HF_TOKEN) {
        Write-Host "[OK] HuggingFace token loaded from .env" -ForegroundColor Green
    } else {
        Write-Host "Error: HF_API_KEY not found in .env file" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Using conda environment: $CondaEnv" -ForegroundColor Green
} else {
    Write-Host "Error: .env file not found at $EnvFile" -ForegroundColor Red
    Write-Host ""
    Write-Host "Create .env file with your HuggingFace token:"
    Write-Host "  HF_API_KEY=hf_your_token_here"
    Write-Host "  CONDA_ENV=unsloth_env"
    exit 1
}

Write-Host ""

# List available model folders
Write-Host "Available trained models:" -ForegroundColor Cyan
Write-Host ""

$OutputDir = ".\kto_output_rtx3090"
if (-not (Test-Path $OutputDir)) {
    Write-Host "Error: Output directory not found: $OutputDir" -ForegroundColor Red
    exit 1
}

# Get all folders that look like training runs with final_model (timestamp format)
$ModelFolders = Get-ChildItem -Path $OutputDir -Directory |
    Where-Object {
        $_.Name -match '^\d{8}_\d{6}$' -and
        (Test-Path (Join-Path $_.FullName "final_model"))
    } |
    Sort-Object Name -Descending

if ($ModelFolders.Count -eq 0) {
    Write-Host "No trained models with final_model found in $OutputDir" -ForegroundColor Red
    Write-Host "Model folders should be in format: YYYYMMDD_HHMMSS and contain a final_model directory" -ForegroundColor Yellow
    exit 1
}

# Display models with numbers
for ($i = 0; $i -lt $ModelFolders.Count; $i++) {
    $folder = $ModelFolders[$i]
    $timestamp = $folder.Name
    Write-Host "  [$($i + 1)] $timestamp" -ForegroundColor Green
}

Write-Host ""

# Ask user to select a model
do {
    $selection = Read-Host "Select model number (1-$($ModelFolders.Count))"
    $selectionNum = [int]$selection
} while ($selectionNum -lt 1 -or $selectionNum -gt $ModelFolders.Count)

$SelectedFolder = $ModelFolders[$selectionNum - 1]
$ModelPath = Join-Path $SelectedFolder.FullName "final_model"

Write-Host ""
Write-Host "Selected: $($SelectedFolder.Name)" -ForegroundColor Cyan

# Verify model exists
if (-not (Test-Path $ModelPath)) {
    Write-Host "Error: final_model not found in $($SelectedFolder.Name)" -ForegroundColor Red
    Write-Host "Expected at: $ModelPath" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Model found at: $ModelPath" -ForegroundColor Green
Write-Host ""

# Ask for model name
Write-Host "Enter HuggingFace model name:" -ForegroundColor Cyan
Write-Host "  Will be uploaded to: professorsynapse/[your-model-name]" -ForegroundColor Gray
Write-Host "  Example: nexus-tools_v0.0.2" -ForegroundColor Gray
$ModelName = Read-Host "Model name"

if ([string]::IsNullOrWhiteSpace($ModelName)) {
    Write-Host "Error: Model name cannot be empty" -ForegroundColor Red
    exit 1
}

# Build full repository ID
$RepoId = "professorsynapse/$ModelName"

Write-Host ""

# Ask for save method
Write-Host "Select upload format:" -ForegroundColor Cyan
Write-Host "  [1] merged_16bit (recommended) - Full quality, ~14GB" -ForegroundColor White
Write-Host "  [2] merged_4bit - Smaller size, ~3.5GB" -ForegroundColor White
Write-Host "  [3] lora - LoRA adapters only, ~320MB" -ForegroundColor White

do {
    $methodChoice = Read-Host "Select format (1-3)"
} while ($methodChoice -notmatch '^[1-3]$')

$SaveMethod = switch ($methodChoice) {
    "1" { "merged_16bit" }
    "2" { "merged_4bit" }
    "3" { "lora" }
}

Write-Host ""

# Ask about GGUF creation
Write-Host "Create GGUF versions for llama.cpp/Ollama?" -ForegroundColor Cyan
Write-Host "  This will create Q4_K_M, Q5_K_M, and Q8_0 quantizations" -ForegroundColor Gray
Write-Host "  Note: GGUF conversion requires additional time and disk space" -ForegroundColor Gray
$createGguf = Read-Host "Create GGUF? (y/n)"
$CreateGGUF = $createGguf -eq 'y'

Write-Host ""
Write-Host "==========================================" -ForegroundColor Blue
Write-Host "Upload Summary:" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue
Write-Host "  Model: $($SelectedFolder.Name)" -ForegroundColor White
Write-Host "  Repository: $RepoId" -ForegroundColor White
Write-Host "  Format: $SaveMethod" -ForegroundColor White
Write-Host "  GGUF: $(if ($CreateGGUF) { 'Yes (Q4_K_M, Q5_K_M, Q8_0)' } else { 'No' })" -ForegroundColor White
Write-Host ""
Write-Host "  Process:" -ForegroundColor Yellow
Write-Host "    1. Load base model (Mistral 7B)" -ForegroundColor Gray
Write-Host "    2. Merge LoRA adapters" -ForegroundColor Gray
Write-Host "    3. Upload merged model to HuggingFace" -ForegroundColor Gray
if ($CreateGGUF) {
    Write-Host "    4. Create GGUF versions (f16, Q4_K_M, Q5_K_M, Q8_0)" -ForegroundColor Gray
    Write-Host "    5. Upload GGUF files to HuggingFace" -ForegroundColor Gray
}
Write-Host "==========================================" -ForegroundColor Blue
Write-Host ""

# Final confirmation
$confirmation = Read-Host "Start upload? (y/n)"
if ($confirmation -ne 'y') {
    Write-Host "Upload cancelled."
    exit 0
}

# Activate conda environment
Write-Host ""
Write-Host "Activating conda environment: $CondaEnv..." -ForegroundColor Blue
conda activate $CondaEnv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to activate conda environment '$CondaEnv'" -ForegroundColor Red
    Write-Host "Available environments:" -ForegroundColor Yellow
    conda env list
    exit 1
}

# Run upload
Write-Host ""
Write-Host "Uploading model..." -ForegroundColor Blue
Write-Host "This may take 10-30 minutes depending on your internet speed..." -ForegroundColor Yellow
Write-Host ""

# Build command with optional GGUF flag
$uploadCommand = "python src/upload_to_hf.py $ModelPath $RepoId --save-method $SaveMethod"
if ($CreateGGUF) {
    $uploadCommand += " --create-gguf"
}

Invoke-Expression $uploadCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Upload complete!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "View your model at:"
    Write-Host "  https://huggingface.co/$RepoId" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Upload failed. Check error messages above." -ForegroundColor Red
    exit 1
}
