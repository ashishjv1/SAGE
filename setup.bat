@echo off

REM SAGE Project Setup Script for Windows
REM This script creates a conda environment and sets up the SAGE project

echo Setting up SAGE project...

REM Check if conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: conda is not installed. Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

REM Create conda environment from environment.yml
echo Creating conda environment 'sage'...
conda env create -f environment.yml

if %errorlevel% neq 0 (
    echo Error: Failed to create conda environment. Please check environment.yml
    pause
    exit /b 1
)

echo Environment created successfully!

REM Activate the environment
echo Activating environment...
call conda activate sage

REM Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo Setup complete! To use the SAGE project:
echo 1. Activate the environment: conda activate sage
echo 2. Run training: python sage_train.py --dataset cifar100 --subset_fraction 0.1
echo 3. See README.md for more usage examples
echo.
echo For Jupyter notebooks:
echo 1. conda activate sage
echo 2. jupyter notebook
echo 3. Navigate to plots/ directory for analysis notebooks
pause
