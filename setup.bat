@echo off
echo 🚀 Setting up SimuSearch Project...
echo.

echo 📋 Creating new virtual environment...
python -m venv venv_clean
echo ✅ Virtual environment created

echo.
echo 🔧 Activating virtual environment...
call venv_clean\Scripts\activate.bat
echo ✅ Virtual environment activated

echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt
echo ✅ Dependencies installed

echo.
echo 🎯 Setup complete! 
echo.
echo To activate the environment in the future:
echo   venv_clean\Scripts\activate.bat
echo.
echo To run tests:
echo   python -m pytest
echo.
echo To run your agents:
echo   python test_google_integration.py
echo.
pause
