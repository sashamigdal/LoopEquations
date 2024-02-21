#pragma once
#include <set>
#include <regex>
#include <fstream>
#include <filesystem>
#include "SamplesBinner.h"

// Reads sorted files, produces sorted output.
class MergeSorter {
    std::filesystem::path dirpath;
public:
    class iterator {
        iterator( std::filesystem::path dirpath ) {
            std::regex rxInputFileName(R"(Fdata\..+\.sorted\.np)"); // "Fdata.E.524288.10.sorted.np"
            std::smatch m;
            for ( const auto& entry : std::filesystem::directory_iterator(dirpath) ) {
                auto filename = entry.path().filename().string();
                if ( std::regex_match( filename, m, rxInputFileName ) ) {
                    std::ifstream fIn( entry.path(), std::ios::binary );
                    Sample sample;
                    if ( sample.ReadFromStream(fIn) ) {
                        buffer.emplace( sample, sources.emplace( std::move(fIn) ).first );
                    }
                }
            }
        }

        struct SampleInfo {
            Sample sample;
            std::set<std::ifstream>::iterator source;
        };

        std::set<std::ifstream> sources; // because iterators are not invalidated on emplace()/erase()
        std::set<SampleInfo> buffer;
    public:
        Sample operator*() {
            return buffer.begin()->sample;
        }

        void operator++() {
            if ( buffer.empty() ) { return; }
            SampleInfo elem = *buffer.begin();
            buffer.erase( buffer.begin() );
            auto source = elem.source;

            Sample sample;
            if ( sample.ReadFromStream( *(elem.source) ) ) {
                buffer.emplace( sample, elem.source );
            } else {
                sources.erase( elem.source );
            }
        }
    };

    MergeSorter( std::filesystem::path dirpath ) : dirpath(dirpath) {}

    iterator begin() {
        return iterator(dirpath);
    }

    iterator end() {
        return {};
    }
};
