#define _USE_MATH_DEFINES
#include <algorithm>
#include <cstring>
#include <iostream>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>
#include <filesystem>
#include "SortedMerger.h"

namespace fs = std::filesystem;
using namespace std::string_literals;

bool SamplesBinner::ProcessSamplesDir( std::filesystem::path dirpath, size_t levels ) {
    this->levels = levels;
    if ( !dirpath.empty() && dirpath.generic_string().back() != '/' ) {
        dirpath += '/';
    }

    SortedMerger merger(dirpath);
    size_t nSamples = merger.size();
    size_t nStats = size_t(1) << levels;
    stats.clear();
    stats.resize(nStats);
    size_t sample_idx = 0;
    for ( const Sample& sample : merger ) {
        size_t bin_idx = sample_idx * nStats / nSamples;
        UpdateStat( stats[bin_idx], sample );
        sample_idx++;
        if ( sample_idx % 1'000'000 == 0 ) {
            std::cout << "[DEBUG] Processed " << sample_idx << " samples out of " << nSamples << std::endl;
        }
    }
    fs::path outfilepath = dirpath / ("FDBins.np");
    std::ofstream fOut( outfilepath, std::ios::binary );
    for ( const auto& st : stats ) {
        st.Write(fOut);
    }

    return true;
}

#pragma pack(push, 1)
template <typename T>
struct KeyIdx {
    T key;
    size_t idx;

    bool operator< ( const KeyIdx<T>& other ) const {
        return key < other.key;
    }
};
#pragma pack(pop)

bool ProduceKeys( std::filesystem::path filepath ) {
    int jobId; // 1-based
    std::regex rxInputFileName(R"(Fdata\..+\.(\d+)\.sorted\.np)"); // "Fdata.E.524288.10.sorted.np"
    std::smatch m;
    auto filename = filepath.filename().string();
    if ( std::regex_match( filename, m, rxInputFileName ) ) {
        jobId = std::stoi( m[1] );
    } else {
        std::cerr << "[FATAL] Filename \"" << filename << "\" not in right format" << std::endl;
        return false;
    }

    std::vector<Sample> samples;
    std::vector<KeyIdx<double>> output;
    size_t nSamples = SamplesBinner::GetNumSamples(filepath);
    samples.reserve(nSamples);
    output.reserve(nSamples);

    std::ios_base::sync_with_stdio(false);
    if ( !SamplesBinner::AppendFromSamplesFile( samples, filepath ) ) {
        std::cerr << "[FATAL] Couldn't read file " << filepath << std::endl;
        return false;
    }
    size_t idx = (jobId - 1) * nSamples;
    std::transform( samples.begin(), samples.end(), std::back_inserter(output), [&idx](const Sample& sample){ return KeyIdx<double>{sample.ds, idx++}; } );

    filepath.replace_extension();
    filepath.replace_extension( "idx" );
    std::ofstream fOut( filepath, std::ios::binary );
    fOut.write( (char*) &output[0], nSamples * sizeof(output[0]) );
    return true;
}

bool Merge( std::filesystem::path filepath ) {
    int fileIdx; // 1-based
    std::regex rxInputFileName(R"((Fdata\..+\.)(\d+)(\.idx))"); // "Fdata.E.524288.1.idx"
    std::smatch m;
    auto filename = filepath.filename().string();
    if ( std::regex_match( filename, m, rxInputFileName ) ) {
        fileIdx = std::stoi( m[2] );
    } else {
        std::cerr << "[FATAL] Filename \"" << filename << "\" not in right format" << std::endl;
        return false;
    }
    if ( fileIdx % 2 != 0 ) {
        std::cerr << "[FATAL] File number (" << fileIdx << ") must be even number" << std::endl;
        return false;
    }
    std::filesystem::path filepath2 = filepath.parent_path() / (m[1].str() + std::to_string(fileIdx - 1) + m[3].str());
    std::filesystem::path outfilepath = filepath.parent_path() / (m[1].str() + std::to_string(fileIdx / 2) + m[3].str() + ".out");
    std::ios_base::sync_with_stdio(false);
    std::ifstream fIn0( filepath, std::ios::binary );
    std::ifstream fIn1( filepath2, std::ios::binary );
    std::ofstream fOut( outfilepath, std::ios::binary );
    KeyIdx<double> key[2];
    fIn0.read( (char*) &key[0], sizeof(KeyIdx<double>) );
    fIn1.read( (char*) &key[1], sizeof(KeyIdx<double>) );
    while ( true ) {
        if ( key[0] < key[1] ) {
            fOut.write( (char*) &key[0], sizeof(KeyIdx<double>) );
            fIn0.read( (char*) &key[0], sizeof(KeyIdx<double>) );
            if ( !fIn0 ) {
                do {
                    fOut.write( (char*) &key[1], sizeof(KeyIdx<double>) );
                    fIn1.read( (char*) &key[1], sizeof(KeyIdx<double>) );
                } while ( fIn1 );
                break;
            }
        } else {
            fOut.write( (char*) &key[1], sizeof(KeyIdx<double>) );
            fIn1.read( (char*) &key[1], sizeof(KeyIdx<double>) );
            if ( !fIn1 ) {
                do {
                    fOut.write( (char*) &key[0], sizeof(KeyIdx<double>) );
                    fIn0.read( (char*) &key[0], sizeof(KeyIdx<double>) );
                } while ( fIn0 );
                break;
            }
        }
    }
    return true;
}

int main( int argc, const char* argv[] ) {
    if ( strcmp( argv[1], "--keys" ) == 0 ) {
        if ( argc != 3 ) {
            std::cout << "Usage: " << argv[0] << " --keys <file_path>" << std::endl;
            return 1;
        }
        fs::path filepath( argv[2] );
        return ProduceKeys(filepath) ? 0 : 1;
    } else if ( strcmp( argv[1], "--merge" ) == 0 ) {
        if ( argc != 3 ) {
            std::cout << "Usage: " << argv[0] << " --merge <file_path>" << std::endl;
            return 1;
        }
        fs::path filepath( argv[2] );
        return Merge(filepath) ? 0 : 1;
    }
}
