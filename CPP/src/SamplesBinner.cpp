#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>
#include <filesystem>
#include "MergeSorter.h"

namespace fs = std::filesystem;
using namespace std::string_literals;

bool SamplesBinner::ProcessSamplesDir( std::filesystem::path dirpath, size_t levels ) {
    this->levels = levels;
    if ( !dirpath.empty() && dirpath.generic_string().back() != '/' ) {
        dirpath += '/';
    }

    rxInputFileName.assign(R"(Fdata\..+\.np)"); // "Fdata.E.524288.10.np"
    PartialSortSamplesFiles(dirpath);
    MergeSorter msr(dirpath);
    if ( !LoadSamplesDir(dirpath) ) { return false; }

    std::sort( std::begin(samples), std::end(samples) );
    stats.clear();
    stats.resize( size_t(1) << levels );
    ProcessSamples(samples);
    fs::path outfilepath = dirpath / ("FDBins.np");
    std::ofstream fOut( outfilepath, std::ios::binary );
    for ( const auto& st : stats ) {
        st.Write(fOut);
    }

    return true;
}

void SamplesBinner::ProcessSamplesBySort( std::vector<Sample>& samples ) {
    std::sort( std::begin(samples), std::end(samples) );
    std::vector<size_t> binSizes;
    binSizes.reserve( size_t(1) << levels );
    size_t beg = 0;
    for ( size_t bin = 0; bin != size_t(1) << levels; bin++ ) {
        size_t binlen = GetBinLen( bin, samples.size(), levels );
        stats[bin] = std::accumulate( std::begin(samples) + beg, std::begin(samples) + beg + binlen, Stats(), [this](Stats st, const Sample& sam){
                                                                                                UpdateStat(st, sam);
                                                                                                return st;
            } );
        beg += binlen;
    }
}

size_t GetBinLen( size_t bin, size_t len, size_t levels ) {
    for ( size_t l = 0; l != levels; l++ ) {
        len = len / 2 + ((bin & (size_t(1) << (levels - l - 1))) != 0) * (len % 2);
    }
    return len;
}

void TestBinLen() {
    size_t len = 8384356;
    size_t levels = 20;
    size_t beg = 0;
    for ( size_t bin = 0; bin != size_t(1) << levels; bin++ ) {
        size_t binlen = GetBinLen( bin, len, levels );
        //std::cout << binlen << '\n';
        beg += binlen;
    }
    std::cout << beg << '\n';
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
