#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>

#pragma pack(push, 1)
struct Sample {
    double field[32][3];
};
#pragma pack(pop)

int main( int argc, const char* argv[] ) {
    if ( argc != 3 ) {
        std::cout << "Usage: " << argv[0] << " <file path> <number of pieces>" << std::endl;
        return 1;
    }
    std::filesystem::path filepath( argv[1] );
    std::filesystem::path dirpath = filepath.parent_path();
    std::ifstream fIn( filepath, std::ios::binary );
    if ( !fIn ) {
        std::cerr << "Couldn't open file " << filepath << std::endl;
        return 1;
    }
    const auto filesize = std::filesystem::file_size(filepath);
    const auto nsamples = filesize / sizeof(double) / 3 / 32;
    const size_t nPieces = std::atoi( argv[2] );
    if ( filesize % (sizeof(double) * 3 * 32 * nPieces) != 0 ) {
        std::cerr << "Cannot split samples evenly to " << nPieces << " pieces." << std::endl;
        return 1;
    }
    std::vector<Sample> samples(nsamples);
    fIn.read( (char*) &samples[0], filesize );
    std::sort( std::begin(samples), std::end(samples), []( const Sample& lhs, const Sample& rhs ) {return lhs.field[0][0] < rhs.field[0][0]; } );
    for ( size_t i = 0; i != nPieces; i++ ) {
        std::filesystem::path outfilepath = dirpath / ("FDStatsPiece." + std::to_string(i) + ".np");
        std::ofstream fOut( outfilepath, std::ios::binary );
        size_t beg = nsamples * i / nPieces;
        fOut.write( (char*) &samples[beg], filesize / nPieces );
    }
    return 0;
}
