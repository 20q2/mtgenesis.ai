<#
.SYNOPSIS
    One-click launcher for MTGenesis.AI: starts Ollama, the Flask backend and a public
    tunnel (a free Cloudflare quick tunnel), then publishes the site to GitHub Pages pointing
    at the live tunnel.

.DESCRIPTION
    Run it by double-clicking "Start MTGenesis.cmd" in the repo root. Steps:
      1. Ollama      - reuses a running server or starts `ollama serve`; pulls the model if missing.
      2. Backend     - reuses a healthy server on :5000 or opens a "MTGenesis backend" window.
      3. Tunnel      - reuses a tunnel to :5000 or opens a "MTGenesis tunnel" window
                       (a Cloudflare quick tunnel; the URL is new each launch).
      4. Publish     - pushes the gh-pages branch: api-config.json always carries the live
                       tunnel URL; the Angular app is rebuilt only when the frontend source
                       changed since the last deploy (or with -Rebuild).
      5. Opens the public site once GitHub Pages is serving the new URL.
    Press Enter in the launcher window to stop the backend and tunnel it started.

.PARAMETER Rebuild
    Rebuild and redeploy the frontend even if the source hasn't changed.

.PARAMETER NoPublish
    Start everything locally and print the tunnel URL, but don't push to GitHub Pages.

.PARAMETER CreateShortcut
    Put an "MTGenesis" shortcut on the desktop that runs this launcher, then exit.
#>
[CmdletBinding()]
param(
    [switch]$Rebuild,
    [switch]$NoPublish,
    [switch]$CreateShortcut
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$ServerDir   = Join-Path $RepoRoot 'proxy-server'
$DeployDir   = Join-Path $RepoRoot '.deploy'
$PagesDir    = Join-Path $DeployDir 'gh-pages'
$BuildDir    = Join-Path $DeployDir 'build'
$PagesBranch = 'gh-pages'
$BackendPort = 5000
$BackendUrl  = "http://127.0.0.1:$BackendPort"
$OllamaUrl   = 'http://127.0.0.1:11434'
# The rules-text model, read from proxy-server/config.py (TEXT_MODEL) so there is one source of truth.
$OllamaModel = 'qwen3:8b'
$configMatch = Select-String -Path (Join-Path $ServerDir 'config.py') -Pattern '"MTG_TEXT_MODEL",\s*"([^"]+)"' -ErrorAction SilentlyContinue
if ($configMatch) { $OllamaModel = $configMatch.Matches[0].Groups[1].Value }
# cloudflared's metrics server on a fixed port: /quicktunnel reports the tunnel's hostname,
# which also lets a second run find and reuse a tunnel that is already up.
$CloudflaredMetrics = '127.0.0.1:20241'
$CloudflaredLog     = Join-Path $DeployDir 'cloudflared.log'
# Frontend inputs; a change to any of these since the last deploy triggers a rebuild.
$FrontendPaths = @('src', 'angular.json', 'package.json', 'package-lock.json', 'tsconfig.json', 'tsconfig.app.json')

$script:Started = New-Object System.Collections.Generic.List[object]

# ---------------------------------------------------------------- helpers

function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text)   { Write-Host "    $Text" -ForegroundColor Green }
function Write-Info([string]$Text) { Write-Host "    $Text" }
function Write-Warn([string]$Text) { Write-Host "    $Text" -ForegroundColor Yellow }

function Test-Http([string]$Url, [hashtable]$Headers = @{}) {
    try {
        $r = Invoke-WebRequest -Uri $Url -Headers $Headers -UseBasicParsing -TimeoutSec 5
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
    } catch {
        return $false
    }
}

# Polls $Condition once a second until it returns something truthy or $TimeoutSec passes.
# $Alive (optional) returning $false aborts early, e.g. when the process being waited on died.
function Wait-For([string]$What, [scriptblock]$Condition, [int]$TimeoutSec, [scriptblock]$Alive = $null) {
    Write-Host "    Waiting for $what " -NoNewline
    $start = Get-Date
    $deadline = $start.AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $result = & $Condition
        if ($result) { Write-Host ' ready' -ForegroundColor Green; return $result }
        # Grace period: the child process takes a moment to appear under its window.
        if ($Alive -and ((Get-Date) - $start).TotalSeconds -gt 5 -and -not (& $Alive)) { Write-Host ' stopped' -ForegroundColor Red; return $null }
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds 1
    }
    Write-Host ' timed out' -ForegroundColor Red
    return $null
}

# Opens a titled console window running $Exe; the window stays open if the program exits so
# its error stays readable.
function Start-Window([string]$Title, [string]$Exe, [string]$Arguments, [string]$WorkingDir) {
    $cmdLine = "/c title $Title & `"$Exe`" $Arguments & echo. & echo [$Title exited] & pause"
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdLine -WorkingDirectory $WorkingDir -PassThru
    $script:Started.Add([pscustomobject]@{ Name = $Title; Process = $proc })
    return $proc
}

# True while $ExeName is still running under the cmd window started by Start-Window.
function Test-ChildRunning($Proc, [string]$ExeName) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($Proc.Id)" -ErrorAction SilentlyContinue
    return [bool]($children | Where-Object { $_.Name -ieq $ExeName })
}

function Stop-Started {
    foreach ($s in $script:Started) {
        if (-not $s.Process.HasExited) {
            Write-Info "Stopping $($s.Name)"
            & taskkill.exe /PID $s.Process.Id /T /F 2>&1 | Out-Null
        }
    }
    $script:Started.Clear()
}

function Invoke-Git([string]$Dir, [string[]]$GitArgs) {
    & git -C $Dir @GitArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE)" }
}

function Write-Utf8File([string]$Path, [string]$Content) {
    [IO.File]::WriteAllText($Path, $Content, (New-Object Text.UTF8Encoding $false))
}

function Update-PathFromRegistry {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Find-Exe([string]$Name, [string[]]$Candidates) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    foreach ($c in $Candidates) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}

# ---------------------------------------------------------------- steps

function Start-Ollama {
    Write-Step 'Ollama'
    if (Test-Http "$OllamaUrl/api/tags") {
        Write-Ok 'Already running'
    } else {
        $ollama = Find-Exe 'ollama' @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")
        if (-not $ollama) { throw 'Ollama is not installed. Get it from https://ollama.com/download and run this again.' }
        # Ollama is shared with other apps, so it's left running on exit (not added to $Started).
        Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Minimized | Out-Null
        if (-not (Wait-For 'Ollama' { Test-Http "$OllamaUrl/api/tags" } 60)) { throw 'Ollama did not start.' }
    }

    $tags = Invoke-RestMethod "$OllamaUrl/api/tags" -TimeoutSec 5
    if (-not ($tags.models | Where-Object { $_.name -eq $OllamaModel })) {
        Write-Info "Pulling $OllamaModel (one-time download, a few GB)..."
        $ollama = Find-Exe 'ollama' @("$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")
        & $ollama pull $OllamaModel | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "ollama pull $OllamaModel failed." }
    }
    Write-Ok "Model $OllamaModel available"
}

function Start-Backend {
    Write-Step "Backend (Flask on :$BackendPort)"
    if (Test-Http "$BackendUrl/health") {
        Write-Ok 'Already running - reusing it'
        return
    }
    $python = Join-Path $ServerDir '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        $python = Find-Exe 'python' @()
        if (-not $python) { throw 'No Python found (expected proxy-server\.venv or python on PATH).' }
        Write-Warn "No proxy-server\.venv found; using $python"
    }
    $proc = Start-Window 'MTGenesis backend' $python 'app.py' $ServerDir
    $up = Wait-For 'the backend (loading torch can take a minute)' { Test-Http "$BackendUrl/health" } 300 { Test-ChildRunning $proc 'python.exe' }
    if (-not $up) { throw 'The backend did not come up. Check the "MTGenesis backend" window for the error.' }
}

function Get-CloudflaredExe {
    $candidates = @(
        "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
        "$env:ProgramFiles\cloudflared\cloudflared.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe",
        "$env:ProgramData\chocolatey\bin\cloudflared.exe",
        "$env:USERPROFILE\scoop\shims\cloudflared.exe"
    )
    $exe = Find-Exe 'cloudflared' $candidates
    if ($exe) { return $exe }
    $pkgRoot = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages"
    if (Test-Path $pkgRoot) {
        $found = Get-ChildItem $pkgRoot -Filter 'cloudflared.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    return $null
}

function Initialize-Cloudflared {
    $exe = Get-CloudflaredExe
    if ($exe) { return $exe }
    Write-Warn 'cloudflared is not installed (free, no account needed).'
    $answer = Read-Host '    Install it now with winget? [Y/n]'
    if ($answer -match '^[nN]') { throw 'cloudflared is required. Install it from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ and run this again.' }
    & winget install --id Cloudflare.cloudflared --exact --accept-source-agreements --accept-package-agreements | Out-Host
    Update-PathFromRegistry
    $exe = Get-CloudflaredExe
    if (-not $exe) { throw 'cloudflared installed but could not be found. Open a new window and run this again.' }
    return $exe
}

# The quick tunnel's https URL: from cloudflared's metrics server, else from its log file.
function Get-CloudflareTunnelUrl {
    try {
        $q = Invoke-RestMethod "http://$CloudflaredMetrics/quicktunnel" -TimeoutSec 2
        if ($q.hostname) { return "https://$($q.hostname)" }
    } catch { }
    if (Test-Path $CloudflaredLog) {
        $m = Select-String -Path $CloudflaredLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches -ErrorAction SilentlyContinue |
            Select-Object -Last 1
        if ($m) { return $m.Matches[-1].Value }
    }
    return $null
}

function Start-Tunnel {
    Write-Step 'Cloudflare tunnel'
    $url = $null
    try {
        $q = Invoke-RestMethod "http://$CloudflaredMetrics/quicktunnel" -TimeoutSec 2
        if ($q.hostname) { $url = "https://$($q.hostname)" }
    } catch { }
    if ($url) {
        Write-Ok "Already running - reusing $url"
    } else {
        $exe = Initialize-Cloudflared
        New-Item -ItemType Directory -Force $DeployDir | Out-Null
        if (Test-Path $CloudflaredLog) { Remove-Item $CloudflaredLog -Force }
        $cfArgs = "tunnel --no-autoupdate --metrics $CloudflaredMetrics --logfile `"$CloudflaredLog`" --url http://localhost:$BackendPort"
        $proc = Start-Window 'MTGenesis tunnel' $exe $cfArgs $RepoRoot
        $url = Wait-For 'the tunnel' { Get-CloudflareTunnelUrl } 90 { Test-ChildRunning $proc 'cloudflared.exe' }
        if (-not $url) { throw 'cloudflared did not open a tunnel. Check the "MTGenesis tunnel" window for the error.' }
        Write-Ok $url
    }
    # A new quick-tunnel hostname can take a few seconds to resolve, so give it a minute.
    $ok = Wait-For 'the backend through the tunnel' { Test-Http "$url/health" } 60
    if (-not $ok) { throw "The tunnel is up but $url/health does not answer. Check the ""MTGenesis tunnel"" window." }
    Write-Ok 'Backend reachable through the tunnel'
    return $url
}

function Get-GitHubRepo {
    $origin = (& git -C $RepoRoot remote get-url origin).Trim()
    if ($origin -notmatch 'github\.com[:/]([^/]+)/(.+?)(\.git)?/?$') { throw "origin ($origin) is not a GitHub repo." }
    $owner = $Matches[1]; $name = $Matches[2]
    if ($name -ieq "$owner.github.io") { $base = '/' } else { $base = "/$name/" }
    return [pscustomobject]@{
        Origin   = $origin
        Owner    = $owner
        Name     = $name
        BaseHref = $base
        SiteUrl  = "https://$($owner.ToLower()).github.io$base"
    }
}

# Clones (or refreshes) the gh-pages branch into .deploy\gh-pages, creating the branch if needed.
function Sync-PagesCheckout($Repo) {
    $remoteHasBranch = [bool](& git -C $RepoRoot ls-remote --heads $Repo.Origin $PagesBranch)
    if ($LASTEXITCODE -ne 0) { throw 'Could not reach GitHub (git ls-remote failed).' }

    if (-not (Test-Path (Join-Path $PagesDir '.git'))) {
        if (Test-Path $PagesDir) { Remove-Item $PagesDir -Recurse -Force }
        New-Item -ItemType Directory -Force $DeployDir | Out-Null
        if ($remoteHasBranch) {
            Invoke-Git $DeployDir @('clone', '--quiet', '--depth', '1', '--branch', $PagesBranch, '--single-branch', $Repo.Origin, $PagesDir)
        } else {
            New-Item -ItemType Directory -Force $PagesDir | Out-Null
            Invoke-Git $PagesDir @('init', '--quiet', '-b', $PagesBranch)
            Invoke-Git $PagesDir @('remote', 'add', 'origin', $Repo.Origin)
        }
    } elseif ($remoteHasBranch) {
        Invoke-Git $PagesDir @('fetch', '--quiet', '--depth', '1', 'origin', $PagesBranch)
        Invoke-Git $PagesDir @('reset', '--quiet', '--hard', "origin/$PagesBranch")
        Invoke-Git $PagesDir @('clean', '--quiet', '-fdx')
    }

    # Publish build output byte-for-byte, committing as the same identity the main repo uses.
    Invoke-Git $PagesDir @('config', 'core.autocrlf', 'false')
    foreach ($key in 'user.name', 'user.email') {
        $value = & git -C $RepoRoot config $key
        if ($value) { Invoke-Git $PagesDir @('config', $key, $value) }
    }
}

function Build-Frontend($Repo) {
    $ng = Join-Path $RepoRoot 'node_modules\.bin\ng.cmd'
    if (-not (Test-Path $ng)) {
        Write-Info 'Installing npm dependencies...'
        & npm.cmd ci --prefix $RepoRoot | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    }
    Write-Info "Building the Angular app (base href $($Repo.BaseHref))..."
    Push-Location $RepoRoot
    try {
        & $ng build --configuration production --base-href $Repo.BaseHref --output-path $BuildDir | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'ng build failed.' }
    } finally {
        Pop-Location
    }
}

function Publish-Site([string]$TunnelUrl) {
    Write-Step 'Publish to GitHub Pages'
    $repo = Get-GitHubRepo
    Sync-PagesCheckout $repo

    $head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    $dirty = [bool](& git -C $RepoRoot status --porcelain -- @FrontendPaths)
    $infoPath = Join-Path $PagesDir 'build-info.json'
    $info = $null
    if (Test-Path $infoPath) { $info = Get-Content $infoPath -Raw | ConvertFrom-Json }

    $needBuild = $Rebuild -or $dirty -or -not $info -or $info.dirty -or $info.commit -ne $head -or
                 -not (Test-Path (Join-Path $PagesDir 'index.html'))
    if ($needBuild) {
        if ($dirty) { Write-Warn 'Frontend has uncommitted changes; they will be deployed.' }
        Build-Frontend $repo
        Get-ChildItem $PagesDir -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force
        Copy-Item (Join-Path $BuildDir '*') $PagesDir -Recurse -Force
        # Deep links (/gallery, /vote, ...) land on 404.html; serving the app there lets the router take over.
        Copy-Item (Join-Path $PagesDir 'index.html') (Join-Path $PagesDir '404.html') -Force
        Write-Utf8File (Join-Path $PagesDir '.nojekyll') ''
        $buildInfo = [ordered]@{ commit = $head; dirty = $dirty }
        Write-Utf8File $infoPath ($buildInfo | ConvertTo-Json)
    } else {
        Write-Ok "Frontend unchanged since the last deploy ($($head.Substring(0, 7))) - skipping the build"
    }

    Write-Utf8File (Join-Path $PagesDir 'api-config.json') (([ordered]@{ apiUrl = $TunnelUrl }) | ConvertTo-Json)

    Invoke-Git $PagesDir @('add', '--all')
    & git -C $PagesDir diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Ok 'GitHub Pages already points at this tunnel - nothing to push'
    } else {
        if ($needBuild) { $msg = "Deploy $($head.Substring(0, 7)) -> $TunnelUrl" } else { $msg = "Point site at $TunnelUrl" }
        Invoke-Git $PagesDir @('commit', '--quiet', '-m', $msg)
        Write-Info "Pushing $PagesBranch..."
        Invoke-Git $PagesDir @('push', '--quiet', 'origin', $PagesBranch)
        Write-Ok 'Pushed'
    }

    try {
        $meta = Invoke-RestMethod "https://api.github.com/repos/$($repo.Owner)/$($repo.Name)" -TimeoutSec 10
        if (-not $meta.has_pages) {
            Write-Warn 'GitHub Pages is not enabled for this repo yet (one-time setup).'
            Write-Warn "In the page that just opened choose: Source = Deploy from a branch, Branch = $PagesBranch, folder = / (root), then Save."
            Start-Process "https://github.com/$($repo.Owner)/$($repo.Name)/settings/pages"
            Read-Host '    Press Enter once Pages is enabled'
        }
    } catch {
        Write-Warn "Could not check the GitHub Pages setting ($($_.Exception.Message))."
    }

    $live = Wait-For 'GitHub Pages to serve the new URL (usually under a minute)' {
        try {
            $c = Invoke-RestMethod "$($repo.SiteUrl)api-config.json?t=$([DateTime]::UtcNow.Ticks)" -TimeoutSec 5
            return ($c.apiUrl -eq $TunnelUrl)
        } catch { return $false }
    } 240
    if (-not $live) { Write-Warn 'Pages has not picked up the change yet; it may take a few more minutes.' }
    return $repo.SiteUrl
}

function New-DesktopShortcut {
    $target = Join-Path $RepoRoot 'Start MTGenesis.cmd'
    $lnkPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'MTGenesis.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $lnk = $shell.CreateShortcut($lnkPath)
    $lnk.TargetPath = $target
    $lnk.WorkingDirectory = $RepoRoot
    $icon = Join-Path $RepoRoot 'src\assets\site_logo.ico'
    if (Test-Path $icon) { $lnk.IconLocation = $icon }
    $lnk.Description = 'Start the MTGenesis.AI backend, tunnel and site'
    $lnk.Save()
    Write-Host "Created $lnkPath" -ForegroundColor Green
}

# ---------------------------------------------------------------- main

if ($CreateShortcut) { New-DesktopShortcut; return }

$Host.UI.RawUI.WindowTitle = 'MTGenesis launcher'
$exitCode = 0
Write-Host 'MTGenesis.AI launcher' -ForegroundColor Magenta

try {
    Start-Ollama
    Start-Backend
    $tunnelUrl = Start-Tunnel

    $siteUrl = $null
    if ($NoPublish) {
        Write-Step 'Publish skipped (-NoPublish)'
    } else {
        try {
            $siteUrl = Publish-Site $tunnelUrl
        } catch {
            Write-Warn "Publishing failed: $($_.Exception.Message)"
            Write-Warn 'The backend and tunnel are still running; the public site may point at an old URL.'
        }
    }

    Write-Host ''
    Write-Host '------------------------------------------------------------' -ForegroundColor Magenta
    if ($siteUrl) { Write-Host "  Site:    $siteUrl" -ForegroundColor Green }
    Write-Host "  API:     $tunnelUrl"
    Write-Host "  Local:   $BackendUrl"
    Write-Host '------------------------------------------------------------' -ForegroundColor Magenta
    if ($siteUrl) { Start-Process $siteUrl }

    Write-Host ''
    if ($script:Started.Count -gt 0) {
        Read-Host 'Press Enter to stop the backend and tunnel (closing this window leaves them running)'
    } else {
        Read-Host 'Everything was already running and has been left alone. Press Enter to close'
    }
} catch {
    Write-Host ''
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host 'Press Enter to stop anything this launcher started and exit'
    $exitCode = 1
} finally {
    Stop-Started
}
exit $exitCode
