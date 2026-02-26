# setup.ps1 - One-time setup script for both capstone projects
# Run this in PowerShell from the Bridge folder:
#   .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Capstone Projects Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ─────────────────────────────────────────────
# PROJECT 1: Meeting Liaison
# ─────────────────────────────────────────────
Write-Host ""
Write-Host "[1/2] Setting up Project 1: Meeting Liaison..." -ForegroundColor Yellow

Set-Location project1_meeting_liaison

# Create virtual environment (skip if already exists)
if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "  [OK] Virtual environment created"
} else {
    Write-Host "  [OK] Virtual environment already exists"
}

# Install packages directly via venv's pip (no activate/deactivate needed)
Write-Host "  Installing dependencies (this may take a minute)..."
.\venv\Scripts\pip install -r requirements.txt --quiet
Write-Host "  [OK] Dependencies installed" -ForegroundColor Green

# Copy .env.example to .env if not already there
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  [OK] .env file created - add your OPENAI_API_KEY" -ForegroundColor Green
} else {
    Write-Host "  [OK] .env already exists"
}

Set-Location ..

# ─────────────────────────────────────────────
# PROJECT 2: Gift Scout
# ─────────────────────────────────────────────
Write-Host ""
Write-Host "[2/2] Setting up Project 2: Gift Scout..." -ForegroundColor Yellow

Set-Location project2_gift_scout

if (-Not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "  [OK] Virtual environment created"
} else {
    Write-Host "  [OK] Virtual environment already exists"
}

Write-Host "  Installing dependencies (this may take a minute)..."
.\venv\Scripts\pip install -r requirements.txt --quiet
Write-Host "  [OK] Dependencies installed" -ForegroundColor Green

if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  [OK] .env file created - add OPENAI_API_KEY + TAVILY_API_KEY" -ForegroundColor Green
} else {
    Write-Host "  [OK] .env already exists"
}

Set-Location ..

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
Write-Host "`n========================================"  -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================"  -ForegroundColor Cyan

Write-Host @"

NEXT STEPS:
  1. Edit project1_meeting_liaison\.env  → add your OPENAI_API_KEY
  2. Edit project2_gift_scout\.env       → add OPENAI_API_KEY + TAVILY_API_KEY
     (Get Tavily free key at: https://app.tavily.com)

TO RUN PROJECT 1:
  cd project1_meeting_liaison
  .\venv\Scripts\Activate.ps1
  streamlit run app.py

TO RUN PROJECT 2:
  cd project2_gift_scout
  .\venv\Scripts\Activate.ps1
  streamlit run app.py

"@ -ForegroundColor White
