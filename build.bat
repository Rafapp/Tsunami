@echo off
setlocal EnableDelayedExpansion

:: === Move to script directory ===
cd /d "%~dp0"

:: === Config ===
set SLANG_VERSION=2026.4.2
set SLANG_DIR=vendors\slang-bin
set BUILD=true
set BUILD_TYPE=Release
set PRESET=default

:: === Args ===
:parse_args
if "%~1"=="" goto end_args
if /i "%~1"=="--setup-only" (set BUILD=false & shift & goto parse_args)
if /i "%~1"=="--debug"      (set BUILD_TYPE=Debug & set PRESET=debug & shift & goto parse_args)
if /i "%~1"=="--help" (
    echo Usage: setup.bat [options]
    echo.
    echo Options:
    echo   (no flags)    Pull submodules, download Slang, configure + build (default)
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
echo Platform:  Windows (x86_64)
echo Slang:     v%SLANG_VERSION%
if "%BUILD%"=="true" (
    echo Build:     %BUILD_TYPE%
) else (
    echo Build:     skipped (--setup-only)
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
if errorlevel 1 (echo ERROR: git submodule update failed & pause & exit /b 1)
echo       OK: Submodules ready

:: === Step 2: Slang ===
echo.
echo [2/4] Setting up Slang prebuilt binary...

if exist "%SLANG_DIR%\include\slang.h" (
    echo       Slang already present -- skipping download
) else (
    set SLANG_URL=https://github.com/shader-slang/slang/releases/download/v%SLANG_VERSION%/slang-%SLANG_VERSION%-windows-x86_64.zip
    echo       Downloading from GitHub releases...
    if not exist "%SLANG_DIR%" mkdir "%SLANG_DIR%"
    curl -L -o "%SLANG_DIR%\slang.zip" "!SLANG_URL!"
    if errorlevel 1 (echo ERROR: Download failed & pause & exit /b 1)
    tar -xf "%SLANG_DIR%\slang.zip" -C "%SLANG_DIR%"
    if errorlevel 1 (echo ERROR: Extraction failed & pause & exit /b 1)
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

:: === Step 4: CMake ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (echo ERROR: Could not find Visual Studio vcvars64.bat - is VS 2022 installed? & exit /b 1)
if "%BUILD%"=="true" goto do_build
echo [4/4] Skipping build (--setup-only)
echo.
echo When ready to build:
echo   cmake --preset default ^&^& cmake --build --preset default
goto done

:do_build
echo [4/4] Configuring + building (%BUILD_TYPE%)...
echo.
cmake --preset %PRESET%
if errorlevel 1 (echo ERROR: CMake configure failed & exit /b 1)
cmake --build --preset %PRESET%
if errorlevel 1 (echo ERROR: CMake build failed & exit /b 1)
echo.
echo       OK: Build complete -- binary at build\tsunami.exe

:done
:: === Done ===
echo.
echo ==========================================
echo         All done! Let it rip
echo ==========================================
echo.
echo   Run: build\tsunami.exe
echo.
pause
endlocal