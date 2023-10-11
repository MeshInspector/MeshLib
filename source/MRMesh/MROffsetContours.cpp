#include "MROffsetContours.h"
#include "MRVector2.h"
#include "MRTimer.h"
#include "MRAffineXf.h"
#include "MRMatrix2.h"
#include "MRMesh.h"
#include "MR2DContoursTriangulation.h"
#include "MRRegionBoundary.h"
#include "MR2to3.h"

namespace MR
{

Contour2f offsetOneDirectionContour( const Contour2f& cont, float offset, const OffsetContoursParams& params )
{
    MR_TIMER;
    bool isClosed = cont.front() == cont.back();

    auto contNorm = [&] ( int i )
    {
        auto norm = cont[i + 1] - cont[i];
        std::swap( norm.x, norm.y );
        norm.x = -norm.x;
        norm = norm.normalized();
        return norm;
    };

    Contour2f res;
    res.emplace_back( isClosed ?
        cont[0] + offset * contNorm( int( cont.size() ) - 2 ) :
        cont[0] + offset * contNorm( 0 ) );
    for ( int i = 0; i + 1 < cont.size(); ++i )
    {
        auto orgPt = cont[i];
        auto destPt = cont[i + 1];
        auto norm = contNorm( i );

        auto nextPoint = orgPt + norm * offset;
        bool sameAsPrev = false;
        // interpolation
        auto prevPoint = res.back();
        auto a = prevPoint - orgPt;
        auto b = nextPoint - orgPt;
        auto crossRes = cross( a, b );
        auto dotRes = dot( a, b );
        float ang = 0.0f;
        if ( crossRes == 0.0f )
            ang = dotRes >= 0.0f ? 0.0f : PI_F;
        else
            ang = std::atan2( crossRes, dotRes );

        sameAsPrev = std::abs( ang ) < PI_F / 360.0f;
        if ( !sameAsPrev )
        {
            if ( params.cornerType == OffsetContoursParams::CornerType::Round )
            {
                int numSteps = int( std::floor( std::abs( ang ) / ( params.minAnglePrecision ) ) );
                for ( int s = 0; s < numSteps; ++s )
                {
                    float stepAng = ( ang / ( numSteps + 1 ) ) * ( s + 1 );
                    auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( stepAng ), orgPt );
                    res.emplace_back( rotXf( prevPoint ) );
                }
            }
            else if ( params.cornerType == OffsetContoursParams::CornerType::Sharp )
            {

            }
            res.emplace_back( std::move( nextPoint ) );
        }
        res.emplace_back( destPt + norm * offset );
    }
    return res;
}

Contours2f offsetContours( const Contours2f& contours, float offset, const OffsetContoursParams& params /*= {} */ )
{
    MR_TIMER;

    Contours2f intermediateRes;

    for ( int i = 0; i < contours.size(); ++i )
    {
        bool isClosed = contours[i].front() == contours[i].back();
        if ( isClosed )
        {
            intermediateRes.push_back( offsetOneDirectionContour( contours[i], offset, params ) );
            if ( params.type == OffsetContoursParams::Type::Shell )
            {
                intermediateRes.push_back( offsetOneDirectionContour( contours[i], -offset, params ) );
                std::reverse( intermediateRes.back().begin(), intermediateRes.back().end() );
            }
        }
        else
        {

        }
    }

    auto mesh = PlanarTriangulation::triangulateContours( std::move( intermediateRes ), nullptr, PlanarTriangulation::WindingMode::NonZero );

    auto bourndaries = findLeftBoundary( mesh.topology );
    Contours2f res;
    for ( const auto& loop : bourndaries )
    {
        res.push_back( {} );
        for ( auto e : loop )
            res.back().push_back( to2dim( mesh.orgPnt( e ) ) );
        res.back().push_back( to2dim( mesh.destPnt( loop.back() ) ) );
    }
    return res;
}

}