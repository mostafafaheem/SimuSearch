# 🚀 SimuSearch - Simple Setup Guide

## 🎯 **What We've Done:**
- ✅ **Removed Poetry** (complex dependency management)
- ✅ **Cleaned up** stray dependencies and cache files
- ✅ **Simplified** to basic pip + virtual environment
- ✅ **Streamlined** project structure

## 🛠️ **Quick Setup (Choose One):**

### **Option 1: Windows Batch File (Easiest)**
```bash
# Double-click this file:
setup.bat
```

### **Option 2: PowerShell Script**
```bash
# Right-click and "Run with PowerShell":
setup.ps1
```

### **Option 3: Manual Setup**
```bash
# Create virtual environment
python -m venv venv_clean

# Activate it (Windows)
venv_clean\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## 🔑 **Set Your API Key:**
```bash
# Create .env file in project root
echo GOOGLE_API_KEY=your-actual-api-key-here > .env
```

## 🧪 **Test Your Setup:**
```bash
# Activate environment
venv_clean\Scripts\activate.bat

# Run test
python test_google_integration.py

# Run your agents
python tests/test_communication_agent.py
```

## 📁 **What's in requirements.txt:**
- **langchain** - AI agent framework
- **langchain-google-genai** - Google AI integration
- **google-generativeai** - Google's AI API
- **pydantic** - Data validation
- **numpy** - Scientific computing
- **python-dotenv** - Environment variables
- **pytest** - Testing framework

## 🎉 **Benefits of This Setup:**
- **Simple** - No complex Poetry commands
- **Fast** - Minimal dependencies
- **Clean** - No stray packages
- **Standard** - Uses standard Python tools
- **Maintainable** - Easy to understand and modify

## 🆘 **Need Help?**
- Check that Python 3.10+ is installed
- Make sure you have internet connection for pip
- Verify your Google API key is set in `.env` file
