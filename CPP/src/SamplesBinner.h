#pragma once
#include <iostream>

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
        return stream;
    }
};
#pragma pack(pop)
