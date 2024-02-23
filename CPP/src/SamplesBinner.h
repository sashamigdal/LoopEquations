#pragma once
#include <cmath>
#include <regex>
#include <iostream>
#include <future>
#include <fstream>

// Corresponds to columns in "Fdata.*.*.*.np"
// PACK because we read them from binary.
#pragma pack(push, 1)
struct Sample {
    static constexpr int NUM_FIELDS = 4;
    double ctg2q2;  // ctg^2(beta/2) / q^2
    double ds;      // |(S_nm - S_mn) / 2sin(beta/2)|
    double oo;      // -1/4 sigma_n sigma_m ctg^2(beta/2)
    double q;

    bool operator< ( const Sample& other ) const {
        return ds < other.ds;
    }

    bool ReadFromStream( std::istream& stream ) {
        stream.read( (char*) this, sizeof(Sample) );
        return (bool)stream;
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

struct Stats1 {
    static constexpr int NUM_FIELDS = 4;

    Stats1() : n(0) {}

    void Add( double a, double b, double c, double d ) {
        n++;
        acc[0].Add(a);
        acc[1].Add(b);
        acc[2].Add(c);
        acc[3].Add(d);
    }

    void Add( const Stats1& other ) {
        n += other.n;
        for ( int i = 0; i != NUM_FIELDS; i++ ) {
            acc[i].Add( other.acc[i] );
        }
    }

    void Clear() {
        n = 0;
        for ( accum& a : acc ) {
            a.Clear();
        }
    }

    void Write( std::ostream& out ) const {
        double dblN = static_cast<double>(n);
        out.write( (char*)&dblN, sizeof dblN );
        for ( size_t i = 0; i != NUM_FIELDS; i++ ) {
            double val = acc[i].Mean(n);
            out.write( (char*)&val, sizeof val );
            val = acc[i].Stdev(n);
            out.write( (char*)&val, sizeof val );
        }
    }

    // Dumps raw state of accums
    void Save( std::ostream& out ) const {
        out.write( (char*)this, sizeof(Stats1) );
    }

    void Load( std::istream& stream ) {
        stream.read( (char*)this, sizeof(Stats1) );
    }
private:
    size_t n;
    accum acc[NUM_FIELDS];
};

struct accum2 {
    double mean, n, sigma;

    accum2() : mean(0), sigma(0) ,n(0){}
    void Add( double x ) {
        n++;
        double delta = (x-mean);
        double newmean = mean + delta/n; 
        sigma += (x-newmean)*delta;
    }
    void Add( const accum2 &other) {
        double oldn =n;
        n += other.n;
        double delta = other.mean-mean;
        double tmp = delta * other.n/n;
        mean += tmp;
        sigma += other.sigma + delta*tmp*oldn;
    }

    double Mean() const {
        return mean;
    }

    double Stdev( ) const {
        return n == 1 ? 0 : sqrt(sigma / (n - 1));
    }

    void Clear() {
        mean = sigma=n =0;
    }
};

struct Stats2 {
    static constexpr int NUM_FIELDS = 4;

    Stats2() {}

    void Add( double a, double b, double c, double d ) {
        acc[0].Add(a);
        acc[1].Add(b);
        acc[2].Add(c);
        acc[3].Add(d);
    }

    void Add( const Stats2& other ) {
        for ( int i = 0; i != NUM_FIELDS; i++ ) {
            acc[i].Add( other.acc[i] );
        }
    }

    void Clear() {
        for ( accum2& a : acc ) {
            a.Clear();
        }
    }

    void Write( std::ostream& out ) const {
        double dblN = acc[0].n;
        out.write( (char*)&dblN, sizeof dblN );
        for ( size_t i = 0; i != NUM_FIELDS; i++ ) {
            double val = acc[i].Mean();
            out.write( (char*)&val, sizeof val );
            val = acc[i].Stdev();
            out.write( (char*)&val, sizeof val );
        }
    }

    // Dumps raw state of accums
    void Save( std::ostream& out ) const {
        out.write( (char*)this, sizeof(*this) );
    }

    void Load( std::istream& stream ) {
        stream.read( (char*)this, sizeof(*this) );
    }
private:
    accum2 acc[NUM_FIELDS];
};
#pragma pack(pop)

using Stats = Stats2;

class SamplesBinner {
public:
    bool ProcessSamplesDir( std::filesystem::path dirpath, size_t levels );
    static bool ProcessSortedSamplesFile( std::filesystem::path filepath, size_t nFiles, size_t nBins );

    static size_t GetNumSamples( std::filesystem::path filepath ) {
        return std::filesystem::file_size(filepath) / sizeof(double) / Sample::NUM_FIELDS;
    }

    bool SortSamplesFile( std::filesystem::path filepath ) {
        samples.clear();
        samples.reserve( GetNumSamples(filepath) );
        if ( !AppendFromSamplesFile( samples, filepath ) ) { return false; }
        std::sort( std::begin(samples), std::end(samples) );
        filepath.replace_extension( "sorted.np" );
        SaveSamplesFile(filepath);
        return true;
    }

    static bool AppendFromSamplesFile( std::vector<Sample>& samples, std::filesystem::path filepath ) {
        size_t nSamples = GetNumSamples(filepath);
        Sample sample;
        std::ifstream fIn( filepath, std::ios::binary );
        if ( !fIn ) {
            std::cerr << "Couldn't open file " << filepath << std::endl;
            return false;
        }
        for ( size_t i = 0; i != nSamples; i++ ) {
            sample.ReadFromStream(fIn);
            samples.push_back(sample);
        }
        return true;
    }

    void SaveSamplesFile( std::filesystem::path filepath ) {
        std::ofstream fOut( filepath, std::ios::binary );
        for ( size_t i = 0; i != samples.size(); i++ ) {
            fOut.write( (char*) &samples[i], sizeof samples[i] );
        }
    }

    static bool UpdateStat( Stats& stat, const Sample& sample ) {
        if ( !( std::isfinite(sample.ctg2q2) && sample.ctg2q2 > 0
             && std::isfinite(sample.ds)     && sample.ds > 0
             && std::isfinite(sample.oo)
             && std::isfinite(sample.q)      && sample.q > 0 ) )
        {
            std::cout << "Bad sample: " << sample.ctg2q2 << ", " << sample.ds << ", " << sample.oo << ", " << sample.q << std::endl;
            return false;
        }

        stat.Add( log( sample.ctg2q2 ), log( sample.ds ), sample.oo, log( sample.q ) );
        return true;
    }
private:
    std::vector<Stats> stats;
    std::vector<Sample> samples;
    size_t levels; // stats[treepath] bins
};
