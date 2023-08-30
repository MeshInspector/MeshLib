#include "MRPointsLoad.h"
#include "MRPointCloud.h"
#include "MRStringConvert.h"
#include "MRColor.h"
#include "MRTimer.h"

#pragma warning(push)
#pragma warning(disable: 4251) // class needs to have dll-interface to be used by clients of another class
#include <E57Format/E57SimpleReader.h>
#pragma warning(pop)

namespace MR
{

namespace PointsLoad
{

Expected<PointCloud, std::string> fromE57( const std::filesystem::path& file, VertColors* colors, ProgressCallback progress )
{
    MR_TIMER

    try
    {
        e57::Reader eReader( utf8string( file ), {} );

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
    
        const int64_t nSize = std::min( nPointsSize, int64_t( 1024 ) * 128 );

        e57::Data3DPointsFloat buffers;
        std::vector<float> xs( nSize ), ys( nSize ), zs( nSize );
        buffers.cartesianX = xs.data();
        buffers.cartesianY = ys.data();
        buffers.cartesianZ = zs.data();
        std::vector<uint16_t> rs, gs, bs;
        std::vector<int8_t> invalidColors;
        if ( colors )
        {
            rs.resize( nSize );
            gs.resize( nSize );
            bs.resize( nSize );
            buffers.colorRed =   rs.data();
            buffers.colorGreen = gs.data();
            buffers.colorBlue =  bs.data();
            invalidColors.resize( nSize );
            buffers.isColorInvalid = invalidColors.data();
        }

        e57::CompressedVectorReader dataReader = eReader.SetUpData3DPointsData( scanIndex, nSize, buffers );

        PointCloud res;
        res.points.reserve( nPointsSize );
        unsigned long size = 0;
        bool hasInputColors = false;
        if ( colors )
            colors->clear();
        while ( ( size = dataReader.read() ) > 0 )
        {
            reportProgress( progress, float( res.points.size() ) / nPointsSize );
            if ( res.points.empty() )
            {
                hasInputColors = invalidColors.front() == 0;
                if ( colors && hasInputColors )
                    colors->reserve( nPointsSize );
            }
            for ( unsigned long i = 0; i < size; ++i )
            {
                res.points.emplace_back( buffers.cartesianX[i], buffers.cartesianY[i], buffers.cartesianZ[i] );
                if ( colors && hasInputColors )
                    colors->emplace_back( buffers.colorRed[i], buffers.colorGreen[i], buffers.colorBlue[i] );
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
