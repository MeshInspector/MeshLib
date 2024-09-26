#include "MRScanHelpers.h"

#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRStringConvert.h"

namespace MR
{

void sortScansByOrder( std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder )
{
    std::sort( zOrder.begin(), zOrder.end() );
    auto filesSorted = scans;
    for ( int i = 0; i < scans.size(); ++i )
        filesSorted[i] = scans[zOrder[i].fileNum];
    scans = std::move( filesSorted );
}

void putScanFileNameInZ( const std::vector<std::filesystem::path>& scans, std::vector<SliceInfo>& zOrder )
{
    assert( zOrder.size() == scans.size() );
    tbb::parallel_for( tbb::blocked_range( 0, int( scans.size() ) ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            std::string name = utf8string( scans[i].stem() );
            auto pos = name.find_last_of( "-0123456789" );
            double res = 0.0;
            if ( pos != std::string::npos )
            {
                // find the start of last number in file name
                for ( ; pos > 0; --pos )
                {
                    auto c = name[pos-1];
                    if ( c != '-' && c != '.' && ( c < '0' || c > '9' ) )
                        break;
                }
                res = std::atof( name.c_str() + pos );
            }
            assert( zOrder[i].fileNum == i );
            zOrder[i].z = res;
        }
    } );
}

void sortScanFilesByName( std::vector<std::filesystem::path>& scans )
{
    const auto sz = scans.size();
    std::vector<SliceInfo> zOrder( sz );
    for ( int i = 0; i < sz; ++i )
        zOrder[i].fileNum = i;
    putScanFileNameInZ( scans, zOrder );
    sortScansByOrder( scans, zOrder );
}

} // namespace MR
