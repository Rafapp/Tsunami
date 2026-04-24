// Purpose: Internal shader utility declarations for Slang source resolution and SPIR-V compilation.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace shader::slang {

std::string resolveShaderPathOrThrow(const std::string& relative_path);

std::vector<uint32_t> compileSlangShaderOrThrow(const std::string&              path,
                                                const std::string&              entry_point,
                                                const std::vector<std::string>& search_paths = {},
                                                const char* target_profile = "spirv_1_4");

}        // namespace shader::slang
