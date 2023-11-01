#include "MROffsetContours.h"
#include "MRVector2.h"
#include "MRTimer.h"
#include "MRAffineXf.h"
#include "MRMatrix2.h"
#include "MRMesh.h"
#include "MR2DContoursTriangulation.h"
#include "MRRegionBoundary.h"
#include "MR2to3.h"
#include "MRPolyline.h"
#include "MRLinesSave.h"
#include <numeric>

namespace MR
{

float findAngle( const Vector2f& prev, const Vector2f& org, const Vector2f& next )
{
    auto a = prev - org;
    auto b = next - org;
    auto crossRes = cross( a, b );
    auto dotRes = dot( a, b );
    if ( crossRes == 0.0f )
        return dotRes >= 0.0f ? 0.0f : PI_F;
    else
        return std::atan2( crossRes, dotRes );
}

void insertRoundCorner( Contour2f& cont, Vector2f prevPoint, Vector2f orgPt, float ang, float minAnglePrecision, int* shiftMap )
{
    int numSteps = int( std::floor( std::abs( ang ) / minAnglePrecision ) );
    for ( int s = 0; s < numSteps; ++s )
    {
        float stepAng = ( ang / ( numSteps + 1 ) ) * ( s + 1 );
        auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( stepAng ), orgPt );
        cont.emplace_back( rotXf( prevPoint ) );
        if ( shiftMap )
            ++( *shiftMap );
    }
}

void insertSharpCorner( Contour2f& cont, Vector2f prevPoint, Vector2f orgPt, float ang, float maxSharpAngle, int* shiftMap )
{
    if ( maxSharpAngle <= 0.0f )
        return;
    if ( std::abs( ang ) <= maxSharpAngle )
    {
        auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( ang * 0.5f ), orgPt );
        auto rotPoint = rotXf( prevPoint );
        auto mod = 1.0f / std::max( std::cos( std::abs( ang ) * 0.5f ), 1e-2f );
        cont.emplace_back( rotPoint * mod + orgPt * ( 1.0f - mod ) );
        if ( shiftMap )
            ++( *shiftMap );
    }
    else
    {
        auto tmpAng = maxSharpAngle;
        float mod = 1.0f / std::max( std::cos( tmpAng * 0.5f ), 1e-2f );
        tmpAng = std::copysign( tmpAng, ang );


        auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( tmpAng * 0.5f ), orgPt );
        auto rotPoint = rotXf( prevPoint );
        cont.emplace_back( rotPoint * mod + orgPt * ( 1.0f - mod ) );

        rotXf = AffineXf2f::xfAround( Matrix2f::rotation( ang - tmpAng * 0.5f ), orgPt );
        rotPoint = rotXf( prevPoint );
        cont.emplace_back( rotPoint * mod + orgPt * ( 1.0f - mod ) );

        if ( shiftMap )
            ( *shiftMap ) += 2;
    }
}

Contour2f offsetOneDirectionContour( const Contour2f& cont, float offset, const OffsetContoursParams& params, 
    int* shiftMap )
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
    res.reserve( 3 * cont.size() );
    if ( shiftMap )
        ++shiftMap[0];

    res.emplace_back( isClosed ?
        cont[0] + offset * contNorm( int( cont.size() ) - 2 ) :
        cont[0] + offset * contNorm( 0 ) );
    for ( int i = 0; i + 1 < cont.size(); ++i )
    {
        auto orgPt = cont[i];
        auto destPt = cont[i + 1];
        auto norm = contNorm( i );

        auto nextPoint = orgPt + norm * offset;

        if ( shiftMap && i > 0 )
            shiftMap[i] += shiftMap[i - 1];

        // interpolation
        auto prevPoint = res.back();
        auto ang = findAngle( prevPoint, orgPt, nextPoint );
        bool sameAsPrev = std::abs( ang ) < PI_F / 360.0f;
        if ( !sameAsPrev )
        {
            bool needCorner = ( ang * offset ) < 0.0f;
            if ( needCorner )
            {
                if ( params.cornerType == OffsetContoursParams::CornerType::Round )
                {
                    insertRoundCorner( res, prevPoint, orgPt, ang, params.minAnglePrecision, shiftMap ? &shiftMap[i] : nullptr );
                }
                else if ( params.cornerType == OffsetContoursParams::CornerType::Sharp )
                {
                    insertSharpCorner( res, prevPoint, orgPt, ang, params.maxSharpAngle, shiftMap ? &shiftMap[i] : nullptr );
                }
            }
            else
            {
                res.push_back( orgPt );
                if ( shiftMap )
                    ++shiftMap[i];
            }
            res.emplace_back( std::move( nextPoint ) );
            if ( shiftMap )
                ++shiftMap[i];
        }
        res.emplace_back( destPt + norm * offset );
        if ( shiftMap )
            ++shiftMap[i + 1];
    }
    if ( shiftMap && cont.size() > 1 )
        shiftMap[int( cont.size() ) - 1] += shiftMap[int( cont.size() ) - 2];
    return res;
}

Contours2f offsetContours( const Contours2f& contours, float offset, const OffsetContoursParams& params /*= {} */ )
{
    MR_TIMER;

    ContoursIndicesMap shiftsMap;

    Contours2f intermediateRes;

    for ( int i = 0; i < contours.size(); ++i )
    {
        if ( contours[i].empty() )
            continue;

        bool isClosed = contours[i].front() == contours[i].back();
        if ( offset == 0.0f )
        {
            intermediateRes.push_back( contours[i] );
            if ( params.indicesMap )
            {
                shiftsMap.push_back( { i,std::vector<int>( contours[i].size(), 0 ) } );
                std::iota( shiftsMap.back().map.begin(), shiftsMap.back().map.end(), 1 );
            }
            if ( !isClosed || params.type == OffsetContoursParams::Type::Shell )
            {
                if ( isClosed )
                {
                    intermediateRes.push_back( contours[i] );
                    std::reverse( intermediateRes.back().begin(), intermediateRes.back().end() );
                    if ( params.indicesMap )
                    {
                        shiftsMap.push_back( { i,std::vector<int>( contours[i].size(), 0 ) } );
                        std::iota( shiftsMap.back().map.rbegin(), shiftsMap.back().map.rend(), 1 );
                    }
                }
                else
                {
                    intermediateRes.back().insert( intermediateRes.back().end(), contours[i].rbegin(), contours[i].rend() );
                    if ( params.indicesMap )
                        shiftsMap.back().map.insert( shiftsMap.back().map.end(), shiftsMap.back().map.rbegin(), shiftsMap.back().map.rend() );
                }
            }
            continue;
        }

        if ( isClosed )
        {
            if ( params.indicesMap )
                shiftsMap.push_back( { i,std::vector<int>( contours[i].size(), 0 ) } );
            intermediateRes.push_back( offsetOneDirectionContour( contours[i], offset, params, 
                params.indicesMap ? shiftsMap.back().map.data() : nullptr ) );
            if ( params.type == OffsetContoursParams::Type::Shell )
            {
                if ( params.indicesMap )
                    shiftsMap.push_back( { i,std::vector<int>( contours[i].size(), 0 ) } );
                intermediateRes.push_back( offsetOneDirectionContour( contours[i], -offset, params,
                    params.indicesMap ? shiftsMap.back().map.data() : nullptr ) );
                if ( params.indicesMap )
                    std::reverse( shiftsMap.back().map.begin(), shiftsMap.back().map.end() );
                std::reverse( intermediateRes.back().begin(), intermediateRes.back().end() );
            }
        }
        else
        {
            if ( params.indicesMap )
                shiftsMap.push_back( { i,std::vector<int>( contours[i].size() * 2, 0 ) } );

            auto tmpOffset = std::abs( offset );
            intermediateRes.push_back( offsetOneDirectionContour( contours[i], tmpOffset, params,
                params.indicesMap ? shiftsMap.back().map.data() : nullptr ) );
            auto backward = offsetOneDirectionContour( contours[i], -tmpOffset, params,
                params.indicesMap ? &shiftsMap.back().map[contours[i].size()] : nullptr );
            if ( params.indicesMap )
                std::reverse( shiftsMap.back().map.begin() + contours[i].size(), shiftsMap.back().map.end() );
            std::reverse( backward.begin(), backward.end() );
            if ( params.endType == OffsetContoursParams::EndType::Round )
            {
                int lastAddiction = 0;
                insertRoundCorner( intermediateRes.back(), intermediateRes.back().back(), contours[i].back(), -PI_F, params.minAnglePrecision, 
                    params.indicesMap ? &lastAddiction : nullptr );
                if ( params.indicesMap )
                    for ( int smi = int( contours[i].size() ) - 1; smi < shiftsMap.back().map.size(); ++smi )
                        shiftsMap.back().map[smi] += lastAddiction;
                intermediateRes.back().insert( intermediateRes.back().end(), backward.begin(), backward.end() );
                insertRoundCorner( intermediateRes.back(), intermediateRes.back().back(), contours[i].front(), -PI_F, params.minAnglePrecision, nullptr );
            }
            else if ( params.endType == OffsetContoursParams::EndType::Cut )
            {
                intermediateRes.back().insert(intermediateRes.back().end(),
                    std::make_move_iterator( backward.begin() ), std::make_move_iterator( backward.end() ) );
            }
            intermediateRes.back().push_back( intermediateRes.back().front() );
        }
    }
    if ( params.indicesMap )
    {
        auto findIndex = [&] ( int contId, int newIndex )->int
        {
            const auto& sm = shiftsMap[contId].map;
            const auto& cont = contours[shiftsMap[contId].contourId];
            const auto& resCont = intermediateRes[contId];

            bool isClosed = cont.front() == cont.back();
            if ( isClosed )
            {
                bool forward = params.type == OffsetContoursParams::Type::Offset || 
                    ( sm.size() > 1 && sm[1] - sm[0] > 0 );
                int result = 0;
                if ( forward )
                {
                    for ( int i = 0; i < cont.size(); ++i )
                        if ( newIndex < sm[i] )
                        {
                            result = i;
                            break;
                        }
                }
                else
                {
                    newIndex = int( resCont.size() ) - 1 - newIndex;
                    for ( int i = 0; i < cont.size(); ++i )
                        if ( newIndex < sm[int( cont.size() ) - 1 - i] )
                        {
                            result = i;
                            break;
                        }
                }
                return result == int( cont.size() ) - 1 ? 0 : result;
            }
            // if not closed
            auto firstPartSize = sm[int( cont.size() ) - 1];
            bool forward = newIndex < firstPartSize;
            if ( forward )
            {
                for ( int i = 0; i < cont.size(); ++i )
                    if ( newIndex < sm[i] )
                        return i;
            }
            else
            {
                newIndex = int( resCont.size() ) - 2 - newIndex;
                for ( int i = 0; i < cont.size(); ++i )
                    if ( newIndex < sm[int( sm.size() ) - 1 - i] )
                        return i;
            }
            return 0;
        };

        params.indicesMap->resize( intermediateRes.size() );
        for ( int i = 0; i < intermediateRes.size(); ++i )
        {
            auto& map = ( *params.indicesMap )[i];
            map.contourId = shiftsMap[i].contourId;
            map.map.resize( intermediateRes[i].size() );
            for ( int j = 0; j < intermediateRes[i].size(); ++j )
                map.map[j] = findIndex( i, j );
        }
    }
    return intermediateRes;// PlanarTriangulation::getOutline( intermediateRes );
}

}