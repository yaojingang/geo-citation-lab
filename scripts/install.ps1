[CmdletBinding()]
param(
    [string]$InstallDir,
    [string]$AssetBase = "https://github.com/yaojingang/geo-citation-lab/releases/download/v0.1.0",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$archiveName = "geo-citation-lab-viewer.zip"
$checksumName = "$archiveName.sha256"
$installMarker = ".geo-citation-lab-install"
$checksumMarker = ".installed-checksum"
$filesManifest = "geo-citation-lab-files.sha256"
$filesChecksumMarker = ".installed-files-checksum"

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "GeoCitationLab\viewer"
}

$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
$installRoot = [System.IO.Path]::GetPathRoot($InstallDir)
$userProfile = [System.IO.Path]::GetFullPath($env:USERPROFILE)
if ($InstallDir -eq $installRoot -or $InstallDir -eq $userProfile) {
    throw "Refusing to install into a broad system or user-profile directory: $InstallDir"
}
if ((Test-Path -LiteralPath $InstallDir -PathType Leaf)) {
    throw "Installation target exists and is not a directory: $InstallDir"
}
if ((Test-Path -LiteralPath $InstallDir -PathType Container) -and
    -not (Test-Path -LiteralPath (Join-Path $InstallDir $installMarker) -PathType Leaf)) {
    throw "Existing directory is not managed by this installer: $InstallDir"
}

$installParent = Split-Path -Parent $InstallDir
New-Item -ItemType Directory -Path $installParent -Force | Out-Null
$temporaryDir = Join-Path ([System.IO.Path]::GetTempPath()) "geo-citation-lab-$([guid]::NewGuid())"
$stageDir = "$InstallDir.new.$PID"
$backupDir = $null
New-Item -ItemType Directory -Path $temporaryDir | Out-Null

function Copy-ReleaseAsset {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $base = $AssetBase.TrimEnd("/", "\")
    if ($base -match "^https?://") {
        Invoke-WebRequest -Uri "$base/$Name" -OutFile $Destination
    }
    elseif ($base -match "^file://") {
        $localBase = ([uri]$base).LocalPath
        Copy-Item -LiteralPath (Join-Path $localBase $Name) -Destination $Destination
    }
    else {
        Copy-Item -LiteralPath (Join-Path $base $Name) -Destination $Destination
    }
}

function Test-ViewerFileTree {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][bool]$RequireMarker
    )

    try {
        $manifestPath = Join-Path $Root $filesManifest
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            return $false
        }
        $manifestItem = Get-Item -LiteralPath $manifestPath
        if (($manifestItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            return $false
        }
        $manifestDigest = (
            Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        if ($RequireMarker) {
            $markerPath = Join-Path $Root $filesChecksumMarker
            if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
                return $false
            }
            if ((Get-Content -LiteralPath $markerPath -Raw).Trim() -ne $manifestDigest) {
                return $false
            }
        }

        $rootPrefix = [System.IO.Path]::GetFullPath(
            $Root + [System.IO.Path]::DirectorySeparatorChar
        )
        $fileCount = 0
        foreach ($line in Get-Content -LiteralPath $manifestPath -Encoding UTF8) {
            if ($line -notmatch "^([0-9a-f]{64})  (.+)$") {
                return $false
            }
            $expectedDigest = $Matches[1]
            $relativePath = $Matches[2]
            if ([System.IO.Path]::IsPathRooted($relativePath) -or
                $relativePath.Contains("\")) {
                return $false
            }
            $destination = [System.IO.Path]::GetFullPath(
                (Join-Path $Root $relativePath.Replace(
                    "/",
                    [System.IO.Path]::DirectorySeparatorChar
                ))
            )
            if (-not $destination.StartsWith(
                $rootPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                return $false
            }
            if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
                return $false
            }
            $item = Get-Item -LiteralPath $destination
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                return $false
            }
            $actualDigest = (
                Get-FileHash -LiteralPath $destination -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            if ($actualDigest -ne $expectedDigest) {
                return $false
            }
            $fileCount += 1
        }
        return $fileCount -gt 0
    }
    catch {
        return $false
    }
}

try {
    $archivePath = Join-Path $temporaryDir $archiveName
    $checksumPath = Join-Path $temporaryDir $checksumName
    Copy-ReleaseAsset -Name $archiveName -Destination $archivePath
    Copy-ReleaseAsset -Name $checksumName -Destination $checksumPath

    $checksumLine = (Get-Content -LiteralPath $checksumPath -TotalCount 1).Trim()
    $expectedChecksum = ($checksumLine -split "\s+")[0].ToLowerInvariant()
    if ($expectedChecksum -notmatch "^[0-9a-f]{64}$") {
        throw "Release checksum is malformed"
    }

    $actualChecksum = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualChecksum -ne $expectedChecksum) {
        throw "Release checksum verification failed"
    }

    $installedChecksumPath = Join-Path $InstallDir $checksumMarker
    if ((Test-Path -LiteralPath (Join-Path $InstallDir "index.html") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $InstallDir "geo-citation-lab-manifest.json") -PathType Leaf) -and
        (Test-Path -LiteralPath $installedChecksumPath -PathType Leaf) -and
        ((Get-Content -LiteralPath $installedChecksumPath -Raw).Trim() -eq $expectedChecksum) -and
        (Test-ViewerFileTree -Root $InstallDir -RequireMarker $true)) {
        $entrypoint = Join-Path $InstallDir "index.html"
        $installedManifest = Get-Content `
            -LiteralPath (Join-Path $InstallDir "geo-citation-lab-manifest.json") `
            -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Output "status=already-installed"
        Write-Output "distribution_version=$($installedManifest.distribution_version)"
        Write-Output "install_path=$InstallDir"
        Write-Output "entrypoint=$entrypoint"
        if (-not $NoOpen) {
            try {
                Start-Process $entrypoint
            }
            catch {
                Write-Output "open_manually=$entrypoint"
            }
        }
        return
    }

    if (Test-Path -LiteralPath $stageDir) {
        throw "Temporary installation path already exists: $stageDir"
    }
    New-Item -ItemType Directory -Path $stageDir | Out-Null

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $stageRoot = [System.IO.Path]::GetFullPath(
        $stageDir + [System.IO.Path]::DirectorySeparatorChar
    )
    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        [long]$uncompressedSize = 0
        foreach ($entry in $archive.Entries) {
            $uncompressedSize += $entry.Length
            $destination = [System.IO.Path]::GetFullPath(
                (Join-Path $stageDir $entry.FullName)
            )
            if (-not $destination.StartsWith(
                $stageRoot,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
                throw "Release archive contains an unsafe path: $($entry.FullName)"
            }
            $unixFileType = ($entry.ExternalAttributes -shr 16) -band 0xF000
            if ($unixFileType -eq 0xA000) {
                throw "Release archive contains a symbolic link: $($entry.FullName)"
            }
        }
        if ($uncompressedSize -gt (15 * 1024 * 1024)) {
            throw "Release archive exceeds the 15 MiB extraction limit"
        }
    }
    finally {
        $archive.Dispose()
    }

    Expand-Archive -LiteralPath $archivePath -DestinationPath $stageDir

    $stageEntrypoint = Join-Path $stageDir "index.html"
    if (-not (Test-Path -LiteralPath $stageEntrypoint -PathType Leaf)) {
        throw "Release archive does not contain index.html"
    }
    if (-not (Test-ViewerFileTree -Root $stageDir -RequireMarker $false)) {
        throw "Release archive file integrity verification failed"
    }
    $releaseManifestPath = Join-Path $stageDir "geo-citation-lab-manifest.json"
    if (-not (Test-Path -LiteralPath $releaseManifestPath -PathType Leaf)) {
        throw "Release archive does not contain its manifest"
    }
    $releaseManifest = Get-Content -LiteralPath $releaseManifestPath -Raw -Encoding UTF8 |
        ConvertFrom-Json
    $distributionVersion = [string]$releaseManifest.distribution_version
    if ($distributionVersion -notmatch "^[0-9]+\.[0-9]+\.[0-9]+$") {
        throw "Release manifest has no valid distribution version"
    }
    Set-Content -LiteralPath (Join-Path $stageDir $installMarker) `
        -Value "managed-by=geo-citation-lab-installer"
    Set-Content -LiteralPath (Join-Path $stageDir $checksumMarker) `
        -Value $expectedChecksum
    $filesManifestDigest = (
        Get-FileHash -LiteralPath (Join-Path $stageDir $filesManifest) -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    Set-Content -LiteralPath (Join-Path $stageDir $filesChecksumMarker) `
        -Value $filesManifestDigest

    if (Test-Path -LiteralPath $InstallDir -PathType Container) {
        $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
        $backupDir = "$InstallDir.previous.$timestamp.$PID"
        Move-Item -LiteralPath $InstallDir -Destination $backupDir
    }

    try {
        Move-Item -LiteralPath $stageDir -Destination $InstallDir
    }
    catch {
        if ($backupDir -and (Test-Path -LiteralPath $backupDir -PathType Container)) {
            Move-Item -LiteralPath $backupDir -Destination $InstallDir
        }
        throw "Installation failed; the previous installation was restored. $($_.Exception.Message)"
    }

    $entrypoint = Join-Path $InstallDir "index.html"
    Write-Output "status=installed"
    Write-Output "distribution_version=$distributionVersion"
    Write-Output "install_path=$InstallDir"
    Write-Output "entrypoint=$entrypoint"
    if ($backupDir) {
        Write-Output "backup_path=$backupDir"
    }
    if (-not $NoOpen) {
        try {
            Start-Process $entrypoint
        }
        catch {
            Write-Output "open_manually=$entrypoint"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $temporaryDir) {
        Remove-Item -LiteralPath $temporaryDir -Recurse -Force
    }
    if (Test-Path -LiteralPath $stageDir) {
        Remove-Item -LiteralPath $stageDir -Recurse -Force
    }
}
