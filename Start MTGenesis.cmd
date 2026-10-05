@echo off
rem One-click start: Ollama + Flask backend + Cloudflare tunnel, then publish the site to GitHub Pages.
rem Options: -Rebuild  -NoPublish  -CreateShortcut
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\start-site.ps1" %*
