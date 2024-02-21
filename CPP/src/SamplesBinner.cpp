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
    }
    fs::path outfilepath = dirpath / ("FDBins.np");
    std::ofstream fOut( outfilepath, std::ios::binary );
    for ( const auto& st : stats ) {
        st.Write(fOut);
    }

    return true;
}

int main( int argc, const char* argv[] ) {
    if ( argc != 3 ) {
        std::cout << "Usage: " << argv[0] << " <file path> <M>" << std::endl;
        return 1;
    }
    fs::path dirpath( argv[1] );
    const size_t levels = std::atoi( argv[2] );
    SamplesBinner binner;
    return binner.ProcessSamplesDir( dirpath, levels ) ? 0 : -1;
}
