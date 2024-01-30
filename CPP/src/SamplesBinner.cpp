#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>
#include <iostream>
#include <fstream>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>
#include <future>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std::string_literals;

size_t GetBinLen( size_t bin, size_t len, size_t levels );

constexpr int NUM_FIELDS = 4;

#pragma pack(push, 1)
struct Sample {
    double ctg, ds, oo, q;
    bool operator< ( const Sample& other ) const {
        return fabs(ctg) < fabs( other.ctg );
    }
};

struct accum {
    double sum;
    double sum2;
    accum() : sum(0), sum2(0) {}
    void Add( double x ) {
        sum += x;
        sum2 += x * x;
    }
    void Add( const accum &other) {
        sum += other.sum;
        sum2 += other.sum2;
    }

    double Mean( size_t n ) const {
        return sum / n;
    }

    double Stdev( size_t n ) const {
        return n == 1 ? 0 : sqrt((sum2 - sum * sum / n) / (n - 1));
    }

    void Clear() {
        sum = 0;
        sum2 = 0;
    }
};

struct Stats {
    size_t n;
    accum acc[NUM_FIELDS];
    Stats() : n(0) {
        std::memset( acc, 0, sizeof acc );
    }

    void Add( const Stats& other ) {
        n += other.n;
        for (int i = 0; i != NUM_FIELDS; i++) {
            acc[i].Add( other.acc[i] );
        }
    }
    void Clear() {
        n = 0;
        for ( accum& a : acc ) {
            a.Clear();
        }
    }
};
#pragma pack(pop)

std::ostream& operator<< ( std::ostream& out, const Stats& st ) {
    double dblN = static_cast<double>(st.n);
    out.write( (char*)&dblN, sizeof dblN );
    for ( size_t i = 0; i != NUM_FIELDS; i++ ) {
        double val = st.acc[i].Mean(st.n);
        out.write( (char*)&val, sizeof val );
        val = st.acc[i].Stdev(st.n);
        out.write( (char*)&val, sizeof val );
    }
    return out;
}

/*std::ostream& operator<< ( std::ostream& out, const Stats& st ) {
    out << st.n;
    for ( size_t i = 0; i != NUM_FIELDS; i++ ) {
        out << '\t' << st.acc[i].Mean(st.n);
        out << '\t' << st.acc[i].Stdev(st.n);
    }
    out << '\n';
    return out;
}*/

enum SIGN { POS, NEG };

class SamplesBinner {
public:
    SamplesBinner() {
        auto maxThreads = std::thread::hardware_concurrency();
        if ( maxThreads == 0 ) {
            maxThreads = 32;
        }
        maxThreadsLevel = int( log(maxThreads) / log(2) );
    }

    bool ProcessSamplesDir( fs::path dirpath, size_t levels ) {
        this->levels = levels;
        if ( !dirpath.empty() && dirpath.generic_string().back() != '/' ) {
            dirpath += '/';
        }

        rxInputFileName.assign(R"(Fdata\..+\.np)"); // "Fdata.E.524288.10.np"
        if ( !LoadSamplesDir(dirpath) ) { return false; }

        for ( size_t i = 0; i != 2; i++ ) {
            stats.clear();
            stats.resize( size_t(1) << levels );
            ProcessSamples( samples[i] );
            fs::path outfilepath = dirpath / ("FDBins."s + (i == 0 ? "pos" : "neg") + ".np");
            std::ofstream fOut( outfilepath, std::ios::binary );
            for ( const auto& st : stats) {
                fOut << st;
            }
        }
        return true;
    }

    bool LoadSamplesDir( fs::path dirpath ) {
        std::smatch m;
        size_t nSamples = 0;
        for ( const auto& entry : fs::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                nSamples += std::filesystem::file_size( entry.path() ) / sizeof(double) / NUM_FIELDS;
            }
        }

        for ( auto& smps : samples ) {
            smps.reserve( static_cast<size_t>(nSamples * 0.55) ); // avoid dynamic alloc
        }

        for ( const auto& entry : fs::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                if ( !LoadSamplesFile( entry.path() ) ) {
                    return false;
                }
            }
        }
        return true;
    }

    bool LoadSamplesFile( fs::path filepath ) {
        size_t nSamples = std::filesystem::file_size(filepath) / sizeof(double) / NUM_FIELDS;
        Sample sample;
        std::ifstream fIn( filepath, std::ios::binary );
        if ( !fIn ) {
            std::cerr << "Couldn't open file " << filepath << std::endl;
            return false;
        }
        for ( size_t i = 0; i != nSamples; i++ ) {
            fIn.read( (char*) &sample, sizeof sample );
            int sgn = sample.oo > 0 ? POS : NEG;
            samples[sgn].push_back(sample);
        }
        return true;
    }

    bool ProcessSamples( const std::vector<Sample>& smpls ) {
        std::vector<const Sample*> V; // array of pointers to samples
        V.reserve( smpls.size() );
        std::transform( std::begin(smpls), std::end(smpls), std::back_inserter(V), [](const Sample& sample){ return &sample; } );

        //we pass V.begin() which is a pointer to the first element of pointers to Sample, i.e. pointer to a pointer to Sample
        ProcessSamplesRecur(V.begin(), V.size(), 0 ,0);
        return true;
    }

    bool UpdateStat( Stats& stat, const Sample& sample ) {
        if (!( std::isfinite(sample.ctg) && std::isfinite(sample.ds) && std::isfinite(sample.oo) && sample.ctg > 0 && sample.ds > 0 ))
        {
            std::cout << "Bad sample: " << sample.ctg << ", " << sample.ds << ", " << sample.oo << std::endl;
            return false;
        }
        if ( sample.oo == 0 ) {
            return false;
        }
        double log_fabs = log( fabs( sample.ctg ) );
        if ( !std::isfinite(log_fabs) ) {
            std::cout << "Infinite value detected among samples" << ": " << sample.ctg << ", " << sample.ds << ", " << sample.oo << std::endl;
            return false;
        }

        stat.n++;
        stat.acc[0].Add( log_fabs );
        stat.acc[1].Add( log( fabs( sample.ds ) ) );
        stat.acc[2].Add( log( fabs( sample.oo ) ) );
        stat.acc[3].Add( log( sample.q ) );
        return true;
    }

    void ProcessSamplesRecur( std::vector<const Sample*>::iterator v, size_t len, int level, size_t index ) {
        //we pass v =V.begin() which is a pointer to the first element of pointers to Sample, 
        //i.e. pointer to a pointer to Sample
        if ( level == levels ) {
            Stats &stat = stats[index];
            for(size_t i =0; i < len; i++){
                UpdateStat(stat, *(v[i]));
            }
            return;
        }

        auto mid = v + len/2;
        std::nth_element( v, mid, v + len, []( const Sample* a, const Sample* b ) { return *a < *b; } );
        std::future<void> fut;
        if ( level < maxThreadsLevel ) {
            fut = std::async( std::launch::async, &SamplesBinner::ProcessSamplesRecur, this, v, len / 2, level + 1, index * 2 );
        } else {
            ProcessSamplesRecur( v, len / 2, level + 1, index * 2 );
        }
        ProcessSamplesRecur( v + len / 2, len - len / 2, level + 1, index * 2 + 1 );
        if ( level < maxThreadsLevel ) {
            fut.wait();
        }
        return;
    }
private:
    void ProcessSamplesBySort( std::vector<Sample>& samples ) {
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

    std::vector<Stats> stats;
    std::vector<Sample> samples[2];
    std::regex rxInputFileName;
    size_t levels; // stats[treepath] bins
    unsigned int maxThreadsLevel;
};

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
