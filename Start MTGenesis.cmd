@echo off
rem One-click start: Ollama + Flask backend + ngrok tunnel, then publish the site to GitHub Pages.
rem Options: -Rebuild  -NoPublish  -NgrokDomain your-name.ngrok-free.app  -CreateShortcut
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\start-site.ps1" %*
