#include <iostream>

#include "tsunami/app/app.h"

int main() {
	try {
		App app;
		app.run();
	} catch (const std::exception& e) {
		std::cerr << "[Tsunami] fatal: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
