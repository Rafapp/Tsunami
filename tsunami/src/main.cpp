// Purpose: Program entry point that parses scene arguments, runs App, and reports fatal errors.
#include <iostream>
#include <string>

#include "app/internal/scene_catalog.h"
#include "tsunami/app/app.h"

int main(int argc, char** argv) {
	// Keep diagnostics visible even if the process terminates abruptly.
	std::cout << std::unitbuf;
	std::cerr << std::unitbuf;
	std::cerr << "[INFO] main() entered\n";

	try {
		if (argc > 2) {
			std::cerr << "Usage: tsunami " << app::scene::sceneArgumentUsage() << "\n";
			return 1;
		}

		const std::string scene_argument = (argc == 2) ? argv[1] : "";
		App               app(scene_argument);
		app.run();
	} catch (const std::exception& e) {
		std::cerr << "[Tsunami] fatal: " << e.what() << "\n";
		return 1;
	} catch (...) {
		std::cerr << "[Tsunami] fatal: unknown exception\n";
		return 1;
	}

	return 0;
}
