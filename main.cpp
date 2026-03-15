#include <cstdlib>
#include <iostream>

#include "src/core/Application.h"

int main() {
  Application app;
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << "[Fatal] " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
