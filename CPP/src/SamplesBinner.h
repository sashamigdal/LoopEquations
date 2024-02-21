#pragma once
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
        stream.read( (char*) this, sizeof Sample );
        return (bool)stream;
    }
};
#pragma pack(pop)

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
    static constexpr int NUM_FIELDS = 4;
    size_t n;
    accum acc[NUM_FIELDS];

    Stats() : n(0) {}

    void Add( const Stats& other ) {
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
};

class SamplesBinner {
public:
    SamplesBinner() {
        auto maxThreads = std::thread::hardware_concurrency();
        if ( maxThreads == 0 ) {
            maxThreads = 32;
        }
        maxThreadsLevel = int( log(maxThreads) / log(2) );
    }

    bool ProcessSamplesDir( std::filesystem::path dirpath, size_t levels );

    size_t GetNumSamples( std::filesystem::path filepath ) {
        return std::filesystem::file_size(filepath) / sizeof(double) / Sample::NUM_FIELDS;
    }

    bool PartialSortSamplesFiles( std::filesystem::path dirpath ) {
        std::smatch m;
        for ( const auto& entry : std::filesystem::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                if ( !SortSamplesFile( entry.path() ) ) {
                    return false;
                }
            }
        }
        return true;
    }

    bool SortSamplesFile( std::filesystem::path filepath ) {
        samples.clear();
        samples.reserve( GetNumSamples(filepath) );
        if ( !LoadSamplesFile(filepath) ) { return false; }
        std::sort( std::begin(samples), std::end(samples) );
        filepath.replace_extension( "sorted.np" );
        SaveSamplesFile(filepath);
        return true;
    }

    bool LoadSamplesDir( std::filesystem::path dirpath ) {
        std::smatch m;
        size_t nSamples = 0;
        for ( const auto& entry : std::filesystem::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                nSamples += GetNumSamples( entry.path() );
            }
        }
        samples.reserve(nSamples);

        for ( const auto& entry : std::filesystem::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                if ( !LoadSamplesFile( entry.path() ) ) {
                    return false;
                }
            }
        }
        return true;
    }

    bool LoadSamplesFile( std::filesystem::path filepath ) {
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

    bool ProcessSamples( const std::vector<Sample>& smpls ) {
        std::vector<const Sample*> V; // array of pointers to samples
        V.reserve( smpls.size() );
        std::transform( std::begin(smpls), std::end(smpls), std::back_inserter(V), [](const Sample& sample){ return &sample; } );

        //we pass V.begin() which is a pointer to the first element of pointers to Sample, i.e. pointer to a pointer to Sample
        ProcessSamplesRecur(V.begin(), V.size(), 0 ,0);
        return true;
    }

    void ProcessSortedSamples();

    bool UpdateStat( Stats& stat, const Sample& sample ) {
        if ( !( std::isfinite(sample.ctg2q2) && sample.ctg2q2 > 0
             && std::isfinite(sample.ds)     && sample.ds > 0
             && std::isfinite(sample.oo)
             && std::isfinite(sample.q)      && sample.q > 0 ) )
        {
            std::cout << "Bad sample: " << sample.ctg2q2 << ", " << sample.ds << ", " << sample.oo << ", " << sample.q << std::endl;
            return false;
        }

        stat.n++;
        stat.acc[0].Add( log( sample.ctg2q2 ) );
        stat.acc[1].Add( log( sample.ds ) );
        stat.acc[2].Add( sample.oo );
        stat.acc[3].Add( log( sample.q ) );

        return true;
    }

    void ProcessSamplesRecur( std::vector<const Sample*>::iterator v, size_t len, size_t level, size_t index ) {
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
    void ProcessSamplesBySort( std::vector<Sample>& samples );

    std::vector<Stats> stats;
    std::vector<Sample> samples;
    std::regex rxInputFileName;
    size_t levels; // stats[treepath] bins
    unsigned int maxThreadsLevel;
};
