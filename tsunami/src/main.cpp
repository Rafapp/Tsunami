#include <iostream>

#include "tsunami/app.h"
 
int main() {
    App app;
 
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "[tsunami] fatal: " << e.what() << "\n";
        return 1;
    }
 
    return 0;
}