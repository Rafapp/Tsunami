#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# === Colors ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# === Config ===
SLANG_VERSION="2026.4.2"
SLANG_DIR="vendors/slang-bin"
BUILD=true
BUILD_TYPE="Release"

# === Args ===
while [[ $# -gt 0 ]]; do
  case $1 in
    --setup-only) BUILD=false; shift ;;
    --debug) BUILD_TYPE=Debug; shift ;;
    --help)
      echo "Usage: ./build.sh [options]"
      echo ""
      echo "Options:"
      echo "  (no flags)    Pull submodules, download Slang, configure + build (default)"
      echo "  --setup-only  Pull submodules + download Slang, skip build"
      echo "  --debug       Build Debug instead of Release"
      echo "  --help        Show this message"
      exit 0 ;;
    *) echo -e "${RED}Unknown option: $1${NC}" && exit 1 ;;
  esac
done

# === Banner ===
echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}               Tsunami                  ${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""

# === Detect platform ===
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
  PLATFORM_LABEL="macOS ($ARCH)"
  SLANG_PLATFORM=$([ "$ARCH" = "arm64" ] && echo "macos-aarch64" || echo "macos-x86_64")
elif [ "$OS" = "Linux" ]; then
  PLATFORM_LABEL="Linux ($ARCH)"
  SLANG_PLATFORM="linux-x86_64"
else
  echo -e "${RED}Unsupported platform: $OS. Use build.bat on Windows.${NC}"
  exit 1
fi

echo -e "${BOLD}Platform:${NC} ${MAGENTA}${PLATFORM_LABEL}${NC}"
echo -e "${BOLD}Slang:${NC}    ${MAGENTA}v${SLANG_VERSION}${NC}"
echo -e "${BOLD}Build:${NC}    ${MAGENTA}$([ "$BUILD" = true ] && echo "$BUILD_TYPE" || echo "skipped (--setup-only)")${NC}"
echo ""

# === Step 1: Submodules ===
echo -e "${BOLD}${BLUE}[1/4]${NC} ${BOLD}Pulling submodules...${NC}"
git submodule update --init --recursive
echo -e "${GREEN}      OK: Submodules ready${NC}"

# === Step 2: Slang ===
echo -e "${BOLD}${BLUE}[2/4]${NC} ${BOLD}Setting up Slang prebuilt binary...${NC}"

if [ -f "${SLANG_DIR}/include/slang.h" ]; then
  echo -e "${YELLOW}      Slang already present. Skipping download.${NC}"
else
  SLANG_URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-${SLANG_PLATFORM}.tar.gz"
  echo "      Downloading from GitHub releases..."
  mkdir -p "$SLANG_DIR"
  curl -L --progress-bar "$SLANG_URL" | tar xz -C "$SLANG_DIR"
  echo -e "${GREEN}      OK: Slang ${SLANG_VERSION} installed${NC}"
fi
echo ""

# === Step 3: Git cppformat hook ===
echo -e "${BOLD}${BLUE}[3/4]${NC} ${BOLD}Installing git hooks...${NC}"

if python3 scripts/install_cppformat.py; then
  echo -e "${GREEN}      OK: cppformat ready${NC}"
else
  echo -e "${RED}      ERROR: cppformat installation failed. Ensure Python 3 and scripts/install_cppformat.py are available.${NC}"
  exit 1
fi
echo ""

# === Step 4: CMake ===
if [ "$BUILD" = true ]; then
  PRESET=$([ "$BUILD_TYPE" = "Debug" ] && echo "debug" || echo "default")
  OUTPUT_DIR=$([ "$BUILD_TYPE" = "Debug" ] && echo "build-debug" || echo "build")
  echo -e "${BOLD}${BLUE}[4/4]${NC} ${BOLD}Configuring + building (${BUILD_TYPE})...${NC}"
  echo ""
  cmake --preset "$PRESET"
  cmake --build --preset "$PRESET"
  echo ""
  echo -e "${GREEN}      OK: Build complete. Binary at ${OUTPUT_DIR}/bin/tsunami${NC}"
else
  echo -e "${BOLD}${BLUE}[4/4]${NC} ${BOLD}Skipping build${NC} ( --setup-only )"
  echo ""
  echo -e "${BOLD}When ready to build:${NC}"
  echo -e "  ${MAGENTA}cmake --preset default && cmake --build --preset default${NC}"
fi

# === Done ===
echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}            All done. Let it rip        ${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
