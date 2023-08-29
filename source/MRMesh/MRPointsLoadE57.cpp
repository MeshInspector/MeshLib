#include "MRPointsLoad.h"
#include "MRPointCloud.h"
#include "MRStringConvert.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4251) // class needs to have dll-interface to be used by clients of another class
#include <E57Format/E57SimpleReader.h>
#pragma warning(pop)

#pragma warning(disable: 4996) // allow using deprecated functions

namespace MR
{

namespace PointsLoad
{

Expected<PointCloud, std::string> fromE57( std::istream& /*in*/, ProgressCallback /*callback*/ )
{
    assert( false );
    PointCloud res;
    return res;
}

Expected<PointCloud, std::string> fromE57( const std::filesystem::path& file, ProgressCallback /*callback*/ )
{
    MR_TIMER

    try
    {
        e57::Reader eReader( utf8string( file ) );

        int scanIndex = 0;

        e57::Data3D scanHeader;
        eReader.ReadData3D( scanIndex, scanHeader );

        int64_t nColumn = 0;
        int64_t nRow = 0;
        int64_t nPointsSize = 0;
        int64_t nGroupsSize = 0;
        int64_t nCountSize = 0;
        bool bColumnIndex = false;

        if ( !eReader.GetData3DSizes( scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex) )
            return MR::unexpected( std::string( "GetData3DSizes failed during reading of " + utf8string( file ) ) );
    
        int64_t nSize = (nRow > 0) ? nRow : 1024;

        e57::Data3DPointsFloat buffers;
        std::vector<float> xs( nSize ), ys( nSize ), zs( nSize );
        buffers.cartesianX = xs.data();
        buffers.cartesianY = ys.data();
        buffers.cartesianZ = zs.data();

        e57::CompressedVectorReader dataReader = eReader.SetUpData3DPointsData( scanIndex, nSize, buffers );

        PointCloud res;
        res.points.reserve( nPointsSize );
        unsigned long size = 0;
        while ( ( size = dataReader.read() ) > 0 )
        {
            for ( unsigned long i = 0; i < size; ++i )
            {
                res.points.emplace_back( buffers.cartesianX[i], buffers.cartesianY[i], buffers.cartesianZ[i] );
            }
        }
        assert( res.points.size() == (size_t)nPointsSize );
        res.validPoints.resize( res.points.size(), true );

        dataReader.close();

        return res;
    }
    catch( const e57::E57Exception & e )
    {
        return MR::unexpected( std::string( "Error '") + e.errorStr() + "' during reading of " + utf8string( file ) );
    }
}

} //namespace PointsLoad

} //namespace MR
