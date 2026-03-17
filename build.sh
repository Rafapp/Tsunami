#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

SLANG_VERSION="2026.4.2"
SLANG_DIR="vendors/slang-bin"
BUILD=false
BUILD_TYPE="Release"

# === ARGS ===
while [[ $# -gt 0 ]]; do
  case $1 in
    --build)         BUILD=true;                    shift ;;
    --build-debug)   BUILD=true; BUILD_TYPE=Debug;  shift ;;
    --help)
      echo "Usage: ./setup.sh [options]"
      echo ""
      echo "Options:"
      echo "  --build         Run cmake configure + build (Release) after setup"
      echo "  --build-debug   Run cmake configure + build (Debug) after setup"
      echo "  --help          Show this message"
      exit 0 ;;
    *) echo -e "${RED}Unknown option: $1${NC}" && exit 1 ;;
  esac
done

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║           Tsunami 🌊 — Setup             ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# === PLATFORM ===
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
  PLATFORM_LABEL="macOS ($ARCH)"
  if [ "$ARCH" = "arm64" ]; then
    SLANG_PLATFORM="macos-aarch64"
  else
    SLANG_PLATFORM="macos-x86_64"
  fi
elif [ "$OS" = "Linux" ]; then
  PLATFORM_LABEL="Linux ($ARCH)"
  SLANG_PLATFORM="linux-x86_64"
else
  echo -e "${RED}Unsupported platform: $OS — use setup.bat on Windows${NC}"
  exit 1
fi

echo -e "${BOLD}Platform:${NC}  ${MAGENTA}${PLATFORM_LABEL}${NC}"
echo -e "${BOLD}Slang:${NC}     ${MAGENTA}v${SLANG_VERSION}${NC}"
echo ""

# === I. Submodules ===
echo -e "${BOLD}${BLUE}[1/3]${NC} ${BOLD}Pulling submodules...${NC}"
echo -e "      ${CYAN}◆${NC} vk-bootstrap"
echo -e "      ${CYAN}◆${NC} VulkanMemoryAllocator"
echo -e "      ${CYAN}◆${NC} glfw"
echo -e "      ${CYAN}◆${NC} volk"
echo ""
git submodule update --init --recursive
echo -e "${GREEN}      ✓ Submodules ready${NC}"
echo ""

# === II. Slang binary ===
echo -e "${BOLD}${BLUE}[2/3]${NC} ${BOLD}Setting up Slang prebuilt binary...${NC}"

if [ -f "${SLANG_DIR}/include/slang.h" ]; then
  echo -e "${YELLOW}      ↺ Slang already present — skipping download${NC}"
else
  SLANG_URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-${SLANG_PLATFORM}.tar.gz"
  echo -e "      Downloading from GitHub releases..."
  mkdir -p "$SLANG_DIR"
  curl -L --progress-bar "$SLANG_URL" | tar xz -C "$SLANG_DIR"
  echo -e "${GREEN}      ✓ Slang ${SLANG_VERSION} installed${NC}"
fi
echo ""

# === III. CMake ===
if [ "$BUILD" = true ]; then
  PRESET=$([ "$BUILD_TYPE" = "Debug" ] && echo "debug" || echo "default")
  echo -e "${BOLD}${BLUE}[3/3]${NC} ${BOLD}Configuring + building (${BUILD_TYPE})...${NC}"
  echo ""
  cmake --preset "$PRESET"
  cmake --build --preset "$PRESET"
  echo ""
  echo -e "${GREEN}      ✓ Build complete — binary at build/bin/tsunami${NC}"
else
  echo -e "${BOLD}${BLUE}[3/3]${NC} ${BOLD}Skipping build${NC} ${CYAN}(pass --build to build now)${NC}"
fi

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║           Setup Complete! 🎉             ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo -e "  Configure:  ${MAGENTA}cmake --preset default${NC}"
echo -e "  Build:      ${MAGENTA}cmake --build --preset default${NC}"
echo -e "  Run:        ${MAGENTA}./build/bin/tsunami${NC}"
echo ""