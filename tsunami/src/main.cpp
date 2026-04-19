#include <iostream>
#include <string>

#include "tsunami/app/app.h"

int main(int argc, char** argv) {
	try {
		if (argc > 2) {
			std::cerr << "Usage: tsunami "
			             "[pool|pool_and_water|chess|cornell|cornellsimple|sponza|<path/to/"
			             "scene.gltf|.glb>]\n";
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
