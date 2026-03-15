# Run this once in PowerShell (as Administrator) to permanently add CMake to PATH
[System.Environment]::SetEnvironmentVariable(
    "PATH",
    [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin",
    "Machine"
)
Write-Host "CMake added to system PATH. Restart PowerShell for it to take effect."
