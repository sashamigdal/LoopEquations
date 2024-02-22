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
        std::cerr << "[FATAL] Filename not in right format" << std::endl;
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

int main( int argc, const char* argv[] ) {
    if ( strcmp( argv[1], "--keys" ) == 0 ) {
        if ( argc != 3 ) {
            std::cout << "Usage: " << argv[0] << " --keys <file_path>" << std::endl;
            return 1;
        }
        fs::path filepath( argv[2] );
        return ProduceKeys(filepath) ? 0 : 1;
    }
}
