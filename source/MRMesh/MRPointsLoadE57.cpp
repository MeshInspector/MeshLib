#include "MRPointsLoad.h"
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
#include "MRAffineXf3.h"
#include "MRBox.h"
#include "MRColor.h"
#include "MRPointCloud.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include <MRPch/MRSpdlog.h> //fmt

#pragma warning(push)
#pragma warning(disable: 4251) // class needs to have dll-interface to be used by clients of another class
#pragma warning(disable: 4275) // vcpkg `2022.11.14`: non dll-interface class 'std::exception' used as base for dll-interface class 'e57::E57Exception'
#include <E57Format/E57SimpleReader.h>
#if !__has_include(<E57Format/E57Version.h>)
#define  MR_OLD_E57
#endif

#pragma warning(pop)

namespace MR
{

namespace PointsLoad
{

Expected<PointCloud, std::string> fromE57( const std::filesystem::path& file, VertColors* colors, AffineXf3f* outXf,
                                           ProgressCallback progress )
{
    MR_TIMER

    try
    {
#ifdef MR_OLD_E57
        e57::Reader eReader( utf8string( file ) );
#else
        e57::Reader eReader( utf8string( file ), {} );
#endif
        int scanIndex = 0;

        e57::Data3D scanHeader;
        eReader.ReadData3D( scanIndex, scanHeader );

        std::optional<Vector3d> offset;
        if ( outXf )
        {
            const auto& bounds = scanHeader.cartesianBounds;
            const Box3d box {
                { bounds.xMinimum, bounds.yMinimum, bounds.zMinimum },
                { bounds.xMaximum, bounds.yMaximum, bounds.zMaximum },
            };
            if ( box.valid() )
            {
                offset = box.center();
                *outXf = AffineXf3f::translation( Vector3f( *offset ) );
            }
        }
        else
        {
            offset = Vector3d();
        }

        int64_t nColumn = 0;
        int64_t nRow = 0;
        int64_t nPointsSize = 0;
        int64_t nGroupsSize = 0;
        int64_t nCountSize = 0;
        bool bColumnIndex = false;

        if ( !eReader.GetData3DSizes( scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex) )
            return MR::unexpected( std::string( "GetData3DSizes failed during reading of " + utf8string( file ) ) );
    
        // how many points to read in a time
        const int64_t nSize = std::min( nPointsSize, int64_t( 1024 ) * 128 );

#ifdef MR_OLD_E57
        e57::Data3DPointsData_d buffers;
#else
        e57::Data3DPointsDouble buffers;
#endif
        std::vector<double> xs( nSize ), ys( nSize ), zs( nSize );
        buffers.cartesianX = xs.data();
        buffers.cartesianY = ys.data();
        buffers.cartesianZ = zs.data();
#ifdef MR_OLD_E57
        std::vector<uint8_t> rs, gs, bs;
#else
        std::vector<uint16_t> rs, gs, bs;
#endif
        std::vector<int8_t> invalidColors;
        if ( colors )
        {
            rs.resize( nSize );
            gs.resize( nSize );
            bs.resize( nSize );
            buffers.colorRed = rs.data();
            buffers.colorGreen = gs.data();
            buffers.colorBlue = bs.data();
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
            if ( outXf && !offset )
            {
                offset = {
                    buffers.cartesianX[0],
                    buffers.cartesianY[0],
                    buffers.cartesianZ[0],
                };
                *outXf = AffineXf3f::translation( Vector3f( *offset ) );
            }
            for ( unsigned long i = 0; i < size; ++i )
            {
                res.points.emplace_back(
                    buffers.cartesianX[i] - offset->x,
                    buffers.cartesianY[i] - offset->y,
                    buffers.cartesianZ[i] - offset->z
                );
                if ( colors && hasInputColors )
                {
                    colors->emplace_back(
                        buffers.colorRed[i],
                        buffers.colorGreen[i],
                        buffers.colorBlue[i]
                    );
                }
            }
        }
        assert( res.points.size() == (size_t)nPointsSize );
        res.validPoints.resize( res.points.size(), true );

        dataReader.close();

        return res;
    }
    catch( const e57::E57Exception & e )
    {
        return MR::unexpected( fmt::format( "Error '{}' during reading of {}", 
            e57::Utilities::errorCodeToString( e.errorCode() ), utf8string( file ) ) );
    }
}

} //namespace PointsLoad

} //namespace MR

#endif // !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
