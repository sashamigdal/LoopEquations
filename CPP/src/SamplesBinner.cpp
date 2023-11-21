#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstring>
#include <regex>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace std::string_literals;

#pragma pack(push, 1)
struct Sample {
    double field[3];
};

struct accum {
    double sum;
    double sum2;
    accum() : sum(0), sum2(0) {}
    void Add( double x ) {
        sum += x;
        sum2 += x * x;
    }

    double Mean( size_t n ) const {
        return sum / n;
    }

    double Stdev( size_t n ) const {
        return n == 1 ? 0 : sqrt((sum2 - sum * sum / n) / (n - 1));
    }
};

struct Stats {
    size_t n;
    accum acc[3];
    Stats() : n(0) {
        std::memset( acc, 0, sizeof acc );
    }
};
#pragma pack(pop)

std::ostream& operator<< ( std::ostream& out, const Stats& st ) {
    double dblN = st.n;
    out.write( (char*)&dblN, sizeof dblN );
    for ( size_t i = 0; i != 3; i++ ) {
        double val = st.acc[i].Mean(st.n);
        out.write( (char*)&val, sizeof val );
        val = st.acc[i].Stdev(st.n);
        out.write( (char*)&val, sizeof val );
    }
    return out;
}

/*std::ostream& operator<< ( std::ostream& out, const Stats& st ) {
    out << st.n;
    for ( size_t i = 0; i != 3; i++ ) {
        out << '\t' << st.acc[i].Mean(st.n);
        out << '\t' << st.acc[i].Stdev(st.n);
    }
    out << '\n';
    return out;
}*/

class SamplesBinner {
public:
    bool ProcessSamplesDir( fs::path dirpath, size_t M ) {
        this->M = M;
        auto last_dirname = dirpath.parent_path().filename().string();
        std::smatch m;
        std::regex rxDataDirName(R"(VorticityCorr\.(\d+)\..+\.\d+)"); // "VorticityCorr.100000000.GPU.1"
        if ( !std::regex_match( last_dirname, m, rxDataDirName ) ) {
            std::cerr << "[ERROR] Couldn't get N from directory path." << std::endl;
            return false;
        }
        unsigned long long N = std::stoull( m[1] );
        minlog = 2 * log(M_PI) - 4 * log(N);
        double maxlog = -2 * log(M_PI);
        step = (maxlog - minlog) / M;

        stats.resize(2);
        for ( auto& stat : stats ) {
            stat.resize(M);
        }

        std::regex rxInputFileName(R"(Fdata\..+\.np)"); // "Fdata.E.524288.10.np"
        for ( const auto& entry : fs::directory_iterator(dirpath) ) {
            auto filename = entry.path().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                if ( !ProcessSamples( entry.path() ) ) {
                    return false;
                }
            }
        }
        for ( size_t i = 0; i != 2; i++ ) {
            fs::path outfilepath = dirpath / ("FDBins."s + (i == 0 ? "pos" : "neg") + ".np");
            std::ofstream fOut( outfilepath, std::ios::binary );
            for ( const auto& st : stats[i] ) {
                fOut << st;
            }
        }
        std::cout << zeros_count << " zero samples\n";
        return true;
    }

    bool ProcessSamples( fs::path filepath ) {
        fs::path dirpath = filepath.parent_path();
        std::ifstream fIn( filepath, std::ios::binary );
        if ( !fIn ) {
            std::cerr << "Couldn't open file " << filepath << std::endl;
            return false;
        }
        const auto filesize = fs::file_size(filepath);
        const auto T = filesize / sizeof(double) / 3;
        std::vector<Sample> samples(T);
        bool ok;
        fIn.read( (char*) &samples[0], filesize );

        for ( size_t j = 0; j != samples.size(); j++ ) {
            Sample& sample = samples[j];
            double o_o = sample.field[2];
            if (!( std::isfinite(sample.field[0]) && std::isfinite(sample.field[1]) && std::isfinite(o_o) && sample.field[0] > 0 && sample.field[1] > 0 ))
            {
                std::cout << "Bad sample: " << sample.field[0] << ", " << sample.field[1] << ", " << o_o << std::endl;
                ok = false;
                continue;
            }
            int i;
            if ( o_o > 0 ) {
                i = 0;
            } else if ( o_o < 0 ) {
                i = 1;
            } else {
                zeros_count++;
                continue;
            }
            double log_fabs = log( fabs( sample.field[0] ) );
            if ( !std::isfinite(log_fabs) ) {
                std::cout << "Infinite value detected on samples #" << j << ": " << sample.field[0] << ", " << sample.field[1] << ", " << o_o << std::endl;
            }

            long long bin = static_cast<long long>( (log_fabs - minlog) / step );
            bin = std::max( 0LL, std::min( (long long)M - 1, bin ) );
            stats[i][bin].n++;
            stats[i][bin].acc[0].Add( log_fabs );
            stats[i][bin].acc[1].Add( log( fabs( sample.field[1] ) ) );
            stats[i][bin].acc[2].Add( log( fabs( o_o ) ) );
        }
        return true;
    }
private:
    size_t M;
    size_t zeros_count = 0;
    double minlog;
    double step;
    std::vector<std::vector<Stats>> stats;
};


int main( int argc, const char* argv[] ) {
    if ( argc != 3 ) {
        std::cout << "Usage: " << argv[0] << " <file path> <M>" << std::endl;
        return 1;
    }
    fs::path dirpath( argv[1] );
    const size_t M = std::atoi( argv[2] );
    SamplesBinner binner;
    return binner.ProcessSamplesDir( dirpath, M ) ? 0 : -1;
}
