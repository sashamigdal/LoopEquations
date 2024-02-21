#pragma once
#include <map>
#include <regex>
#include <fstream>
#include <filesystem>
#include "SamplesBinner.h"

// Reads sorted files, produces sorted output.
class SortedMerger {
    std::filesystem::path dirpath;
    size_t nSamples;
public:
    class iterator {
        friend class SortedMerger;

        iterator() {}

        iterator( std::filesystem::path dirpath ) {
            std::regex rxInputFileName(R"(Fdata\..+\.sorted\.np)"); // "Fdata.E.524288.10.sorted.np"
            std::smatch m;
            for ( const auto& entry : std::filesystem::directory_iterator(dirpath) ) {
                auto filename = entry.path().filename().string();
                if ( std::regex_match( filename, m, rxInputFileName ) ) {
                    std::ifstream fIn( entry.path(), std::ios::binary );
                    Sample sample;
                    if ( sample.ReadFromStream(fIn) ) {
                        buffer.emplace( sample, std::move(fIn) );
                    }
                }
            }
        }

        std::map<Sample,std::ifstream> buffer;
    public:
        Sample operator*() {
            return buffer.begin()->first;
        }

        void operator++() {
            if ( buffer.empty() ) { return; }
            std::ifstream stream( std::move( buffer.begin()->second ) );
            buffer.erase( buffer.begin() );

            Sample sample;
            if ( sample.ReadFromStream(stream) ) {
                buffer.emplace( sample, std::move(stream) );
            }
        }

        bool operator== ( const iterator& other ) {
            return buffer.empty() && other.buffer.empty(); // all end() iterators are equal. Any other are different.
        }

        bool operator!= ( const iterator& other ) {
            return !(*this == other);
        }
    };

    SortedMerger( std::filesystem::path dirpath ) : dirpath(dirpath) {
        std::regex rxInputFileName(R"(Fdata\..+\.sorted\.np)"); // "Fdata.E.524288.10.sorted.np"
        std::smatch m;
        nSamples = 0;
        for ( const auto& entry : std::filesystem::directory_iterator(dirpath) ) {
            auto filename = entry.path().filename().string();
            if ( std::regex_match( filename, m, rxInputFileName ) ) {
                nSamples += SamplesBinner::GetNumSamples( entry.path() );
            }
        }
    }

    size_t size() const {
        return nSamples;
    }

    iterator begin() const {
        return iterator(dirpath);
    }

    iterator end() const {
        return {};
    }
};
