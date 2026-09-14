@echo off
title Sync to GitHub
echo ===================================================
echo   Syncing JSS Code & Schedules to GitHub...
echo ===================================================
git add .
git commit -m "Auto-sync update: %date% %time%"
git push origin main
echo.
echo ===================================================
echo   Done! Streamlit Cloud is updating in real-time.
echo ===================================================
pause

