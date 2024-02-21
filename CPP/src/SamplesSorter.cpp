#include <iostream>
#include <filesystem>
#include "SamplesBinner.h"

namespace fs = std::filesystem;

int main( int argc, const char* argv[] ) {
    if ( argc != 2 ) {
        std::cout << "Usage: " << argv[0] << " <file path>" << std::endl;
        return 1;
    }
    fs::path filepath( argv[1] );
    SamplesBinner binner;
    return binner.SortSamplesFile(filepath) ? 0 : -1;
}
