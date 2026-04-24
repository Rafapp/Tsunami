// Purpose: Implements shared Slang shader path lookup and compile-to-SPIR-V helpers.
#include "shader/internal/slang_shader_utils.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "slang.h"

namespace shader::slang {

std::string resolveShaderPathOrThrow(const std::string& relative_path) {
	namespace fs = std::filesystem;

	const std::array<fs::path, 5> candidates = {
	    fs::path(relative_path),
	    fs::path("tsunami") / relative_path,
	    fs::path("bin") / relative_path,
	    fs::path("build/bin") / relative_path,
	    fs::path("build-debug/bin") / relative_path,
	};

	for (const fs::path& candidate : candidates) {
		if (fs::exists(candidate)) {
			return candidate.lexically_normal().string();
		}
	}

	throw std::runtime_error("could not find shader source: " + relative_path);
}

std::vector<uint32_t> compileSlangShaderOrThrow(const std::string&              path,
                                                const std::string&              entry_point,
                                                const std::vector<std::string>& search_paths,
                                                const char*                     target_profile) {
	SlangSession*        session = spCreateSession(nullptr);
	SlangCompileRequest* request = nullptr;
	if (session == nullptr) {
		throw std::runtime_error("failed to create Slang session");
	}

	request = spCreateCompileRequest(session);
	if (request == nullptr) {
		spDestroySession(session);
		throw std::runtime_error("failed to create Slang compile request");
	}

	for (const std::string& search_path : search_paths) {
		spAddSearchPath(request, search_path.c_str());
	}

	const int target_index = spAddCodeGenTarget(request, SLANG_SPIRV);
	spSetTargetProfile(request, target_index, spFindProfile(session, target_profile));

	const int unit_index = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
	spAddTranslationUnitSourceFile(request, unit_index, path.c_str());
	spAddEntryPoint(request, unit_index, entry_point.c_str(), SLANG_STAGE_COMPUTE);

	const SlangResult result      = spCompile(request);
	const char*       diagnostics = spGetDiagnosticOutput(request);
	if (diagnostics != nullptr && diagnostics[0] != '\0') {
		std::cerr << "[SLANG] " << path << ":\n" << diagnostics << "\n";
	}
	if (result != SLANG_OK) {
		spDestroyCompileRequest(request);
		spDestroySession(session);
		throw std::runtime_error("slang compilation failed: " + path);
	}

	size_t                spirv_size = 0;
	const void*           spirv_data = spGetEntryPointCode(request, 0, &spirv_size);
	std::vector<uint32_t> spirv(spirv_size / sizeof(uint32_t));
	std::memcpy(spirv.data(), spirv_data, spirv_size);

	spDestroyCompileRequest(request);
	spDestroySession(session);
	return spirv;
}

}        // namespace shader::slang
