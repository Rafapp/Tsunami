#include <iostream>
#include <string>

#include "tsunami/app/app.h"

static void print_help() {
	std::cout << "Usage: tsunami [SCENE]\n"
	             "\n"
	             "Options:\n"
	             "  -h, --help    Show this message and exit\n"
	             "\n"
	             "Built-in scenes:\n"
	             "  pool          Water pool with floating objects (default)\n"
	             "  chess         Chess set\n"
	             "  cornell       Cornell box\n"
	             "  cornellsimple Simplified Cornell box\n"
	             "  sponza        Sponza atrium\n"
	             "\n"
	             "Custom scenes:\n"
	             "  Pass the full path to any .glb or .gltf file:\n"
	             "    tsunami C:/path/to/scene.glb\n";
}

int main(int argc, char** argv) {
	try {
		if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
			print_help();
			return 0;
		}

		if (argc > 2) {
			std::cerr << "Usage: tsunami "
			             "[pool|chess|cornell|cornellsimple|sponza|<path/to/scene.gltf|.glb>]\n";
			return 1;
		}

		const std::string scene_argument = (argc == 2) ? argv[1] : "";
		App               app(scene_argument);
		app.run();
	} catch (const std::exception& e) {
		std::cerr << "[Tsunami] fatal: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
