Write-Host "🚀 Setting up SimuSearch Project..." -ForegroundColor Green
Write-Host ""

Write-Host "📋 Creating new virtual environment..." -ForegroundColor Yellow
python -m venv venv_clean
Write-Host "✅ Virtual environment created" -ForegroundColor Green

Write-Host ""
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
.\venv_clean\Scripts\Activate.ps1
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

Write-Host ""
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "✅ Dependencies installed" -ForegroundColor Green

Write-Host ""
Write-Host "🎯 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future:" -ForegroundColor Cyan
Write-Host "  .\venv_clean\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run tests:" -ForegroundColor Cyan
Write-Host "  python -m pytest" -ForegroundColor White
Write-Host ""
Write-Host "To run your agents:" -ForegroundColor Cyan
Write-Host "  python test_google_integration.py" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"
