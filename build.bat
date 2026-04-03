@echo off
setlocal EnableDelayedExpansion

:: === Move to script directory ===
cd /d "%~dp0"

:: === Config ===
set "SLANG_VERSION=2026.4.2"
set "SLANG_DIR=vendors\slang-bin"
set "BUILD=true"
set "BUILD_TYPE=Release"
set "PRESET=default"

:: === Args ===
:parse_args
if "%~1"=="" goto end_args
if /i "%~1"=="--setup-only" (set "BUILD=false" & shift & goto parse_args)
if /i "%~1"=="--debug"      (set "BUILD_TYPE=Debug" & set "PRESET=debug" & shift & goto parse_args)
if /i "%~1"=="--help" (
    echo Usage: setup.bat [options]
    echo.
    echo Options:
    echo   (no flags)    Pull submodules, download Slang, configure + build ^(default^)
    echo   --debug       Build Debug instead of Release
    echo   --setup-only  Pull submodules + download Slang, skip build
    echo   --help        Show this message
    exit /b 0
)
echo Unknown option: %~1
exit /b 1
:end_args

:: === Banner ===
echo.
echo ==================================
echo           Tsunami
echo ==================================
echo.
echo Platform:  Windows ^(x86_64^)
echo Slang:     v%SLANG_VERSION%
if "%BUILD%"=="true" (
    echo Build:     %BUILD_TYPE%
) else (
    echo Build:     skipped ^(--setup-only^)
)
echo.

:: === Step 1: Submodules ===
echo [1/4] Pulling submodules...
echo       - vk-bootstrap
echo       - VulkanMemoryAllocator
echo       - glfw
echo       - volk
echo       - glm
echo.
git submodule update --init --recursive
if errorlevel 1 (
    echo ERROR: git submodule update failed
    pause
    exit /b 1
)
echo       OK: Submodules ready

:: === Step 2: Slang ===
echo.
echo [2/4] Setting up Slang prebuilt binary...

if exist "%SLANG_DIR%\include\slang.h" (
    echo       Slang already present -- skipping download
) else (
    set "SLANG_URL=https://github.com/shader-slang/slang/releases/download/v%SLANG_VERSION%/slang-%SLANG_VERSION%-windows-x86_64.zip"
    echo       Downloading from GitHub releases...
    if not exist "%SLANG_DIR%" mkdir "%SLANG_DIR%"
    curl -L -o "%SLANG_DIR%\slang.zip" "!SLANG_URL!"
    if errorlevel 1 (
        echo ERROR: Download failed
        pause
        exit /b 1
    )
    tar -xf "%SLANG_DIR%\slang.zip" -C "%SLANG_DIR%"
    if errorlevel 1 (
        echo ERROR: Extraction failed
        pause
        exit /b 1
    )
    del "%SLANG_DIR%\slang.zip"
    echo       OK: Slang %SLANG_VERSION% installed
)
echo.

:: === Step 3: Git cppformat hook ===
echo [3/4] Installing git hooks...
python3 scripts\install_cppformat.py
if errorlevel 1 (
    python scripts\install_cppformat.py
    if errorlevel 1 (
        echo ERROR: cppformat installation failed -- is Python installed and scripts\install_cppformat.py present?
        pause
        exit /b 1
    )
)
echo       OK: cppformat ready
echo.

:: === Step 4: Find and load MSVC environment ===
echo [4/4] Locating Visual Studio C++ toolchain...
call :find_vcvars
if errorlevel 1 (
    echo ERROR: Could not find Visual Studio vcvars64.bat
    echo.
    echo Checked via vswhere and common install paths for:
    echo   - Visual Studio 2022 / 2019
    echo   - BuildTools / Community / Professional / Enterprise
    echo.
    echo Make sure the "Desktop development with C++" workload or MSVC build tools are installed.
    exit /b 1
)

echo       Using: !VCVARS_PATH!
call "!VCVARS_PATH!"
if errorlevel 1 (
    echo ERROR: Failed to initialize MSVC environment
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat ran, but cl.exe is still not available in PATH
    exit /b 1
)
echo       OK: MSVC environment ready

if "%BUILD%"=="true" goto do_build
echo.
echo Skipping build ^(--setup-only^)
echo.
echo When ready to build:
echo   cmake --preset default ^&^& cmake --build --preset default
goto done

:do_build
echo.
echo Configuring + building ^(%BUILD_TYPE%^)^...
echo.
cmake --preset %PRESET%
if errorlevel 1 (
    echo ERROR: CMake configure failed
    exit /b 1
)
cmake --build --preset %PRESET%
if errorlevel 1 (
    echo ERROR: CMake build failed
    exit /b 1
)
echo.
echo       OK: Build complete -- binary at build\tsunami.exe
goto done

:: =========================================================
:: find_vcvars
:: Sets VCVARS_PATH if found
:: Prefers vswhere, then falls back to common paths
:: =========================================================
:find_vcvars
set "VCVARS_PATH="

:: 1) Try vswhere first
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
    for /f "usebackq delims=" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        if exist "%%~I\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS_PATH=%%~I\VC\Auxiliary\Build\vcvars64.bat"
            goto :find_vcvars_done
        )
    )

    :: If -latest fails for some odd reason, try all matching installs
    for /f "usebackq delims=" %%I in (`"!VSWHERE!" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        if exist "%%~I\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS_PATH=%%~I\VC\Auxiliary\Build\vcvars64.bat"
            goto :find_vcvars_done
        )
    )
)

:: 2) Fallback: common editions / versions / roots
for %%R in ("%ProgramFiles%" "%ProgramFiles(x86)%") do (
    for %%V in (2022 2019) do (
        for %%E in (BuildTools Community Professional Enterprise) do (
            if exist "%%~R\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS_PATH=%%~R\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat"
                goto :find_vcvars_done
            )
        )
    )
)

:find_vcvars_done
if defined VCVARS_PATH (
    exit /b 0
)
exit /b 1

:done
echo.
echo ==========================================
echo         All done^! Let it rip
echo ==========================================
echo.
echo   Run: build\tsunami.exe
echo.
pause
endlocal