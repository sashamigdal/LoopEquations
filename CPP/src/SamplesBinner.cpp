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
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace std::string_literals;

#pragma pack(push, 1)
struct Sample {
    double ctg, ds, oo;
    bool operator <(const Sample *other) const{
        return ctg < other->ctg;
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
    accum acc[3];
    Stats() : n(0) {
        std::memset( acc, 0, sizeof acc );
    }
    void Add(const Stat &other){
        n += other.n;
        for (int i =0; i < 3; i++) {
            acc[i] += other.acc[i] ;
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

enum SIGN { POS, NEG };

class SamplesBinner {
public:
    bool ProcessSamplesDir( fs::path dirpath, size_t M ) {
        this->M = M;
        if ( !dirpath.empty() && dirpath.generic_string().back() != '/' ) {
            dirpath += '/';
        }
        auto last_dirname = dirpath.parent_path().filename().string();
        std::smatch m;
        std::regex rxDataDirName(R"(VorticityCorr\.(\d+)\..+\.\d+)"); // "VorticityCorr.100000000.GPU.1"
        if ( !std::regex_match( last_dirname, m, rxDataDirName ) ) {
            std::cerr << "[ERROR] Couldn't get N from directory path." << std::endl;
            return false;
        }
        unsigned long long N = std::stoull( m[1] );

        rxInputFileName.assign(R"(Fdata\..+\.np)"); // "Fdata.E.524288.10.np"
        if ( !LoadSamplesDir(dirpath) ) { return false; }

        for ( size_t i = 0; i != 2; i++ ) {
            stats.clear();
            stats.resize( 1 << LEVELS );
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
                nSamples += std::filesystem::file_size( entry.path() ) / sizeof(double) / 3;
            }
        }

        for ( auto& smps : samples ) {
            smps.reserve(nSamples * 0.6); // avoid dynamic alloc
        }

        for ( const auto& entry : fs::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                if ( !LoadSamplesFile( entry.path() ) ) {
                    return false;
                }
            }
        }
    }

    bool LoadSamplesFile( fs::path filepath ) {
        size_t nSamples = std::filesystem::file_size(filepath) / sizeof(double) / 3;
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
    }

    bool ProcessSamples( std::vector<Sample>& smpls ) {
        std::vector<Sample*> V;
        //array of pointers to samples
        for(auto s: smpls){
            V.push_back(&s);
        }
        //we pass V.begin() which is a pointer to the first element of pointers to Sample, i.e. pointer to a pointer to Sample
        ProcessSamplesRecur(V.begin(), V.size(), 0 ,0);

        /*fs::path dirpath = filepath.parent_path();
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
        }*/
        return true;
    }
    bool UpdateStat(Stat &stat, const Sample &sample){
        double o_o = sample.field[2];
        if (!( std::isfinite(sample.field[0]) && std::isfinite(sample.field[1]) && std::isfinite(o_o) && sample.field[0] > 0 && sample.field[1] > 0 ))
        {
            std::cout << "Bad sample: " << sample.field[0] << ", " << sample.field[1] << ", " << o_o << std::endl;
            ok = false;
            return false;
        }
        int i;
        if ( o_o > 0 ) {
            i = 0;
        } else if ( o_o < 0 ) {
            i = 1;
        } else {
            return false;
        }
        double log_fabs = log( fabs( sample.field[0] ) );
        if ( !std::isfinite(log_fabs) ) {
            std::cout << "Infinite value detected on samples #" << j << ": " << sample.field[0] << ", " << sample.field[1] << ", " << o_o << std::endl;
            return false;
        }

        stat.n++;
        stat.acc[0].Add( log_fabs );
        stat.acc[1].Add( log( fabs( sample.field[1] ) ) );
        stat.acc[2].Add( log( fabs( o_o ) ) );
        return true;
    }
    bool ProcessSamplesRecur(Sample** v, size_t len, int level, size_t index ) {
            //we pass v =V.begin() which is a pointer to the first element of pointers to Sample, 
            //i.e. pointer to a pointer to Sample
        #pragma omp parallel
        {
        if ( level == LEVELS ) {
            // TODO: collect stat : DONE, AM
            Stats &stat = stats[index];
            for(size_t i =0; i < len; i++){
                UpdateStat(stat, *(v[i]));
            }
            return true;
        }
    
        auto mid = v + len/2;
        std::nth_element(v, mid, v+len);
        if ( level < 5 ) {
            // TODO: parallel recur : DONE, AM
            #pragma omp task 
            {
                ProcessSamplesRecur(v, len/2, level+1,index*2);
            }
            #pragma omp task 
            {
                ProcessSamplesRecur(mid,len - len/2, level+1,index*2+1);
            }
            return false;
        }
        // TODO: sequencial, no recur : DONE, AM
        #pragma omp single{ 
            ProcessSamplesRecur(v, len/2, level+1,index*2);
            ProcessSamplesRecur(mid,len - len/2, level+1,index*2+1);
        }
        }
        return false;
    }
private:
    size_t M;
    size_t zeros_count = 0;
    double minlog;
    double step;
    std::vector<Stats> stats;
    std::vector<Sample> samples[2];
    std::regex rxInputFileName;
    const int LEVELS = 20; // stats[treepath] bins
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