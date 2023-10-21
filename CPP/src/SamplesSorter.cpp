#include <algorithm>
#include <cmath>
#include <cstring>
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
        return sqrt((sum2 - sum * sum / n) / n);
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

int main( int argc, const char* argv[] ) {
    if ( argc != 3 ) {
        std::cout << "Usage: " << argv[0] << " <file path> <M>" << std::endl;
        return 1;
    }
    fs::path filepath( argv[1] );
    const size_t M = std::atoi( argv[2] );
    fs::path dirpath = filepath.parent_path();
    std::ifstream fIn( filepath, std::ios::binary );
    if ( !fIn ) {
        std::cerr << "Couldn't open file " << filepath << std::endl;
        return 1;
    }
    const auto filesize = fs::file_size(filepath);
    const auto T = filesize / sizeof(double) / 3;
    std::vector<Sample> samples(T);
    std::vector<bool> ok(T, true);
    fIn.read( (char*) &samples[0], filesize );

    std::vector<std::vector<Stats>> stats(2);
    for ( auto& stat : stats ) {
        stat.resize(M);
    }
    double minlog[2] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    double maxlog[2] = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
    double step[2] = {};
    int i;
    size_t zeros_count = 0;
    for ( size_t j = 0; j != samples.size(); j++ ) {
        Sample& sample = samples[j];
        double oldField0 = sample.field[0];
        if (!( std::isfinite(sample.field[0]) && std::isfinite(sample.field[1]) && std::isfinite(sample.field[2]) && sample.field[0] > 0 && sample.field[1] > 0 ))
        {
            std::cout << "Bad sample: " << sample.field[0] << ", " << sample.field[1] << ", " << sample.field[2] << std::endl;
            ok[j] = false;
            continue;
        }
        if ( sample.field[2] > 0 ) {
            i = 0;
        } else if ( sample.field[2] < 0 ) {
            i = 1;
        } else {
            zeros_count++;
            continue;
        }
        sample.field[0] = log( abs( sample.field[0] ) );
        minlog[i] = std::min( minlog[i], sample.field[0] );
        maxlog[i] = std::max( maxlog[i], sample.field[0] );
        if ( !std::isfinite(minlog[i]) ) {
            std::cout << "Infinite value detected on samples #" << j << ": " << oldField0 << ", " << sample.field[1] << ", " << sample.field[2] << std::endl;
        }
    }
    for ( i = 0; i != 2; i++ ) {
        step[i] = (maxlog[i] - minlog[i]) / M;
    }
    std::cout << ">0: log(abs(ctg)) = [" << minlog[0] << "; " << maxlog[0] << "], step=" << step[0] << "\n";
    std::cout << "<0: log(abs(ctg)) = [" << minlog[1] << "; " << maxlog[1] << "], step=" << step[1] << "\n";
    for ( size_t j = 0; j != samples.size(); j++ ) {
        Sample& sample = samples[j];
        if (!ok[j]) { continue; }
        if ( sample.field[2] > 0 ) {
            i = 0;
        } else if ( sample.field[2] < 0 ) {
            i = 1;
        } else {
            continue;
        }
        long long bin = static_cast<long long>( (sample.field[0] - minlog[i]) / step[i] );
        bin = std::max( 0LL, std::min( (long long)M - 1, bin ) );
        stats[i][bin].n++;
        stats[i][bin].acc[0].Add( sample.field[0] );
        stats[i][bin].acc[1].Add( log( abs( sample.field[1] ) ) );
        stats[i][bin].acc[2].Add( log( abs( sample.field[2] ) ) );
    }
    for ( size_t i = 0; i != 2; i++ ) {
        fs::path outfilepath = dirpath / ("FDBins."s + (i == 0 ? "pos" : "neg") + ".np");
        std::ofstream fOut( outfilepath, std::ios::binary );
        for ( const auto& st : stats[i] ) {
            fOut << st;
        }
    }
    std::cout << zeros_count << " zero samples\n";
    return 0;
}
