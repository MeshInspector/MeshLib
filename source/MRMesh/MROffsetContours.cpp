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
#include "MRParallelFor.h"
#include "MRLine.h"
#include <numeric>

namespace MR
{
struct IntermediateIndicesMap
{
    int contourId{ -1 };
    std::vector<int> map;
};

using IntermediateIndicesMaps = std::vector<IntermediateIndicesMap>;
using SingleOffset = std::function<float( int )>;

void fillIntermediateIndicesMap(
    const Contours2f& contours, 
    const Contours2f& intermediateRes,
    const IntermediateIndicesMaps& shiftsMap,
    OffsetContoursParams::Type type,
    IntermediateIndicesMaps& outMap )
{
    auto findIndex = [&] ( int contId, int newIndex )->int
    {
        const auto& sm = shiftsMap[contId].map;
        const auto& cont = contours[shiftsMap[contId].contourId];
        const auto& resCont = intermediateRes[contId];

        bool isClosed = cont.front() == cont.back();
        if ( isClosed )
        {
            bool forward = type == OffsetContoursParams::Type::Offset ||
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

    outMap.resize( intermediateRes.size() );
    for ( int i = 0; i < intermediateRes.size(); ++i )
    {
        auto& map = outMap[i];
        map.contourId = shiftsMap[i].contourId;
        map.map.resize( intermediateRes[i].size() );
        ParallelFor( intermediateRes[i], [&] ( size_t j )
        {
            map.map[j] = findIndex( i, int( j ) );
        } );
    }
}

void fillResultIndicesMap(
    const Contours2f& intermediateRes,
    const IntermediateIndicesMaps& intermediateMap,
    const PlanarTriangulation::ContoursIdMap& outlineMap,
    OffsetContoursVertMaps& outMaps )
{
    std::vector<int> intermediateToIdShifts( intermediateRes.size() );
    for ( int i = 0; i < intermediateToIdShifts.size(); ++i )
        intermediateToIdShifts[i] = int( intermediateRes[i].size() ) - 1 + ( i > 0 ? intermediateToIdShifts[i - 1] : 0 );

    auto outlineVertIdToContoursVertId = [&] ( int vertId )->OffsetContourIndex
    {
        OffsetContourIndex res;
        if ( vertId == -1 )
        {
            assert( false );
            return res;
        }
        for ( int i = 0; i < intermediateToIdShifts.size(); ++i )
        {
            if ( vertId < intermediateToIdShifts[i] )
            {
                res.contourId = i;
                if ( i > 0 )
                    vertId -= intermediateToIdShifts[i - 1];
                break;
            }
        }
        res.vertId = intermediateMap[res.contourId].map[vertId];
        res.contourId = intermediateMap[res.contourId].contourId;
        return res;
    };
    outMaps.resize( outlineMap.size() );
    for ( int i = 0; i < outMaps.size(); ++i )
    {
        auto& outMap = outMaps[i];
        const auto& inMap = outlineMap[i];
        outMap.resize( inMap.size() );
        ParallelFor( outMap, [&] ( size_t ind )
        {
            auto inVal = inMap[ind];
            auto& outMapI = outMap[ind];
            outMapI.lOrg = outlineVertIdToContoursVertId( int( inVal.lOrg ) );
            if ( inVal.isIntersection() )
            {
                outMapI.lDest = outlineVertIdToContoursVertId( int( inVal.lDest ) );
                outMapI.uOrg = outlineVertIdToContoursVertId( int( inVal.uOrg ) );
                outMapI.uDest = outlineVertIdToContoursVertId( int( inVal.uDest ) );
                outMapI.lRatio = inVal.lRatio;
                outMapI.uRatio = inVal.uRatio;
            }
        } );
    }
}

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

struct CornerParameters
{
    // left prev
    Vector2f lp;
    // left current
    Vector2f lc;
    // right current
    Vector2f rc;
    // right next
    Vector2f rn;
    // org point
    Vector2f org;

    // angle lc,org,rc
    float lrAng;
};

// returns intersection point of (a->b)x(c->d)
// nullopt if parallel
std::optional<Vector2f> findIntersection( const Vector2f& a, const Vector2f& b, const Vector2f& c, const Vector2f& d )
{
    auto vecA = b - a;
    if ( cross( vecA, d - c ) == 0.0f )
        return std::nullopt;

    auto abcS = cross( c - a, vecA );
    auto abdS = cross( vecA, d - a );

    auto sum = abcS + abdS;
    if ( sum == 0.0f )
        return std::nullopt;
    auto ratio = abcS / sum;
    return c * ( 1.0f - ratio ) + d * ratio;
}

void insertRoundCorner( Contour2f& cont, const CornerParameters& params, float minAnglePrecision, int* shiftMap )
{
    int numSteps = int( std::floor( std::abs( params.lrAng ) / minAnglePrecision ) );
    if ( numSteps == 0 )
        return;

    auto lVec = params.lc - params.lp;
    auto rVec = params.rc - params.rn;
    bool round = std::abs( dot( params.lc - params.org, lVec ) / ( params.lc - params.org ).lengthSq() ) < std::numeric_limits<float>::epsilon() * 10.0f;
    round = round && std::abs( dot( params.rc - params.org, rVec ) / ( params.rc - params.org ).lengthSq() ) < std::numeric_limits<float>::epsilon() * 10.0f;
    
    if ( !round )
    {
        auto offset = ( params.org - params.lc ).length();
        auto width = std::abs( params.lrAng ) / PI_F * 1.5f;

        auto ln = params.lc + lVec.normalized() * width * offset;
        auto rp = params.rc + rVec.normalized() * width * offset;

        for ( int s = 0; s < numSteps; ++s )
        {
            float t = float( s + 1 ) / float( numSteps + 1 );
            auto invt = ( 1 - t );
            auto tSq = t * t;
            auto invtSq = invt * invt;
            auto tCb = tSq * t;
            auto invtCb = invtSq * invt;
            const auto& p1 = params.lc;
            const auto& p2 = ln;
            const auto& p3 = rp;
            const auto& p4 = params.rc;
            cont.emplace_back( p1 * invtCb + 3.0f * p2 * invtSq * t + 3.0f * p3 * invt * tSq + p4 * tCb );
            if ( shiftMap )
                ++( *shiftMap );
        }
    }
    else
    {
        for ( int s = 0; s < numSteps; ++s )
        {
            float stepAng = ( params.lrAng / ( numSteps + 1 ) ) * ( s + 1 );
            auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( stepAng ), params.org );
            cont.emplace_back( rotXf( params.lc ) );
            if ( shiftMap )
                ++( *shiftMap );
        }
    }
}

void insertSharpCorner( Contour2f& cont, const CornerParameters& params, float maxSharpAngle, int* shiftMap )
{
    if ( maxSharpAngle <= 0.0f )
        return;

    bool openAng = cross( params.rc - params.lc, params.rn - params.lc ) * params.lrAng < 0.0f || cross( params.lp - params.rc, params.lc - params.rc ) * params.lrAng < 0.0f;
    if ( openAng )
        return;

    auto realAng = findAngle( params.rn, params.rc, params.rc + params.lp - params.lc );
    if ( params.lrAng < 0.0f )
        realAng = -PI_F - realAng;
    else
        realAng = -PI_F + realAng;

    if ( cross( params.rc - params.rn, params.lc - params.lp ) * params.lrAng < 0.0f )
        return;

    auto intersection = findIntersection( params.lp, params.lc, params.rn, params.rc );
    if ( intersection && std::abs( realAng ) <= maxSharpAngle )
    {
        cont.emplace_back( *intersection );
        if ( shiftMap )
            ++( *shiftMap );
    }
    else
    {
        float leftAngRat = params.lrAng * 0.5f;
        if ( intersection )
            leftAngRat = findAngle( params.lc, params.org, *intersection );

        auto leftAng = leftAngRat - std::copysign( ( std::abs( realAng ) - maxSharpAngle ), realAng ) * leftAngRat / realAng;
        auto rotXf = AffineXf2f::xfAround( Matrix2f::rotation( leftAng ), params.org );
        auto rotPoint = rotXf( params.lc );
        auto interLeft = findIntersection( params.lp, params.lc, params.org, rotPoint );
        if ( interLeft )
        {
            cont.emplace_back( *interLeft );
            if ( shiftMap )
                ++( *shiftMap );
        }

        auto rightAng = ( params.lrAng - leftAngRat ) - std::copysign( ( std::abs( realAng ) - maxSharpAngle ), realAng ) * ( params.lrAng - leftAngRat ) / realAng;
        rotXf = AffineXf2f::xfAround( Matrix2f::rotation( -rightAng ), params.org );
        rotPoint = rotXf( params.rc );
        auto interRight = findIntersection( params.rn, params.rc, params.org, rotPoint );
        if ( interRight )
        {
            cont.emplace_back( *interRight );
            if ( shiftMap )
                ++( *shiftMap );
        }
    }
}

Contour2f offsetOneDirectionContour( const Contour2f& cont, SingleOffset offset, const OffsetContoursParams& params,
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
        cont[0] + offset( 0 ) * contNorm( int( cont.size() ) - 2 ) :
        cont[0] + offset( 0 ) * contNorm( 0 ) );

    CornerParameters cParams;
    int lastIndex = int( cont.size() ) - 2;
    cParams.rc = isClosed ? cont[lastIndex] + offset( lastIndex ) * contNorm( lastIndex ) : res.back();
    cParams.rn = res.back();
    for ( int i = 0; i + 1 < cont.size(); ++i )
    {
        auto norm = contNorm( i );        
        auto iOffset = offset( i );
        auto iNextOffset = offset( i + 1 );

        cParams.org = cont[i];
        cParams.lp = cParams.rc;
        cParams.lc = cParams.rn;
        cParams.rc = cParams.org + norm * iOffset;
        cParams.rn = cont[i + 1] + norm * iNextOffset;

        if ( shiftMap && i > 0 )
            shiftMap[i] += shiftMap[i - 1];

        // interpolation
        cParams.lrAng = findAngle( cParams.lc, cParams.org, cParams.rc );
        bool sameAsPrev = std::abs( cParams.lrAng ) < PI_F / 360.0f;
        if ( !sameAsPrev )
        {
            bool needCorner = ( cParams.lrAng * iOffset ) < 0.0f;
            if ( needCorner )
            {
                if ( params.cornerType == OffsetContoursParams::CornerType::Round )
                {
                    insertRoundCorner( res, cParams, params.minAnglePrecision, shiftMap ? &shiftMap[i] : nullptr );
                }
                else if ( params.cornerType == OffsetContoursParams::CornerType::Sharp )
                {
                    insertSharpCorner( res, cParams, params.maxSharpAngle, shiftMap ? &shiftMap[i] : nullptr );
                }
            }
            else
            {
                res.push_back( cParams.org );
                if ( shiftMap )
                    ++shiftMap[i];
            }
            res.emplace_back( cParams.rc );
            if ( shiftMap )
                ++shiftMap[i];
        }
        res.emplace_back( cParams.rn );
        if ( shiftMap )
            ++shiftMap[i + 1];
    }
    if ( shiftMap && cont.size() > 1 )
        shiftMap[int( cont.size() ) - 1] += shiftMap[int( cont.size() ) - 2];
    return res;
}

Contours2f offsetContours( const Contours2f& contours, float offset, 
    const OffsetContoursParams& params /*= {} */ )
{
    return offsetContours( contours, [offset] ( int, int ) { return offset; }, params );
}

Contours2f offsetContours( const Contours2f& contours, ContoursVariableOffset offsetFn, const OffsetContoursParams& params /*= {} */ )
{
    MR_TIMER;

    IntermediateIndicesMaps shiftsMap;

    Contours2f intermediateRes;

    enum class Mode
    {
        Raw,
        NegativeRaw,
        Abs,
        Negative
    };
    auto getSOffset = [&] ( int i, Mode mode )->SingleOffset
    {
        switch ( mode )
        {
        default:
        case Mode::Raw:
            return [offsetFn,i] ( int j ) { return offsetFn( i, j ); };
        case Mode::NegativeRaw:
            return [offsetFn,i] ( int j ) { return -offsetFn( i, j ); };
        case Mode::Abs:
            return [offsetFn,i] ( int j ) { return std::abs( offsetFn( i, j ) ); };
        case Mode::Negative:
            return [offsetFn,i] ( int j ) { return -std::abs( offsetFn( i, j ) ); };
        }
    };

    for ( int i = 0; i < contours.size(); ++i )
    {
        if ( contours[i].empty() )
            continue;

        bool isClosed = contours[i].front() == contours[i].back();
        if ( isClosed )
        {
            if ( params.indicesMap )
                shiftsMap.push_back( { i,std::vector<int>( contours[i].size(), 0 ) } );
            intermediateRes.push_back( offsetOneDirectionContour( contours[i], getSOffset( i, Mode::Raw ), params,
                params.indicesMap ? shiftsMap.back().map.data() : nullptr ) );
            if ( params.type == OffsetContoursParams::Type::Shell )
            {
                if ( params.indicesMap )
                    shiftsMap.push_back( { i,std::vector<int>( contours[i].size(), 0 ) } );
                intermediateRes.push_back( offsetOneDirectionContour( contours[i], getSOffset( i, Mode::NegativeRaw ), params,
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

            intermediateRes.push_back( offsetOneDirectionContour( contours[i], getSOffset( i, Mode::Abs ), params,
                params.indicesMap ? shiftsMap.back().map.data() : nullptr ) );
            auto backward = offsetOneDirectionContour( contours[i], getSOffset( i, Mode::Negative ), params,
                params.indicesMap ? &shiftsMap.back().map[contours[i].size()] : nullptr );
            if ( params.indicesMap )
                std::reverse( shiftsMap.back().map.begin() + contours[i].size(), shiftsMap.back().map.end() );
            std::reverse( backward.begin(), backward.end() );

            if ( params.endType == OffsetContoursParams::EndType::Round )
            {
                int lastAddition = 0;
                assert( intermediateRes.back().size() > 1 );
                assert( backward.size() > 1 );
                CornerParameters cParams;

                cParams.lp = intermediateRes.back()[intermediateRes.back().size() - 2];
                cParams.lc = intermediateRes.back().back();
                cParams.org = contours[i].back();
                cParams.rc = backward.front();
                cParams.rn = backward[1];
                cParams.lrAng = -PI_F;
                if ( cParams.lc != cParams.org )
                    insertRoundCorner( intermediateRes.back(), cParams, params.minAnglePrecision, params.indicesMap ? &lastAddition : nullptr );

                if ( params.indicesMap )
                    for ( int smi = int( contours[i].size() ) - 1; smi < shiftsMap.back().map.size(); ++smi )
                        shiftsMap.back().map[smi] += lastAddition;
                intermediateRes.back().insert( intermediateRes.back().end(), backward.begin(), backward.end() );

                cParams.lp = intermediateRes.back()[intermediateRes.back().size() - 2];
                cParams.lc = intermediateRes.back().back();
                cParams.org = contours[i].front();
                cParams.rc = intermediateRes.back().front();
                cParams.rn = intermediateRes.back()[1];
                cParams.lrAng = -PI_F;
                if ( cParams.lc != cParams.org )
                    insertRoundCorner( intermediateRes.back(), cParams, params.minAnglePrecision, nullptr );
            }
            else if ( params.endType == OffsetContoursParams::EndType::Cut )
            {
                intermediateRes.back().insert( intermediateRes.back().end(),
                    std::make_move_iterator( backward.begin() ), std::make_move_iterator( backward.end() ) );
            }
            intermediateRes.back().push_back( intermediateRes.back().front() );
        }
    }

    IntermediateIndicesMaps intermediateMap;
    if ( params.indicesMap )
        fillIntermediateIndicesMap( contours, intermediateRes, shiftsMap, params.type, intermediateMap );

    PlanarTriangulation::ContoursIdMap outlineMap;
    auto res = PlanarTriangulation::getOutline( intermediateRes, { .indicesMap = params.indicesMap ? &outlineMap : nullptr } );

    if ( params.indicesMap )
        fillResultIndicesMap( intermediateRes, intermediateMap, outlineMap, *params.indicesMap );

    return res;
}

Contours3f offsetContours( const Contours3f& contours, float offset, const OffsetContoursParams& params /*= {} */, const OffsetContoursRestoreZParams& zParmas /*= {} */)
{
    return offsetContours( contours, [offset] ( int, int )
    {
        return offset;
    }, params, zParmas );
}

Contours3f offsetContours( const Contours3f& contours, ContoursVariableOffset offset, const OffsetContoursParams& params /*= {} */, const OffsetContoursRestoreZParams& zParmas /*= {} */ )
{
    MR_TIMER;

    // copy contours to 2d
    float maxOffset = 0.0f;
    Contours2f conts2d( contours.size() );
    for ( int i = 0; i < contours.size(); ++i )
    {
        const auto& cont3d = contours[i];
        auto& cont2d = conts2d[i];
        cont2d.resize( cont3d.size() );
        for ( int j = 0; j < cont3d.size(); ++j )
        {
            cont2d[j] = to2dim( cont3d[j] );
            auto offsetVal = std::abs( offset( i, j ) );
            if ( offsetVal > maxOffset )
                maxOffset = offsetVal;
        }
    }
    auto paramsCpy = params;
    OffsetContoursVertMaps tempMap;
    if ( !paramsCpy.indicesMap )
        paramsCpy.indicesMap = &tempMap;
    // create 2d offset
    auto offset2d = offsetContours( conts2d, offset, paramsCpy );

    const auto& map = *paramsCpy.indicesMap;
    Contours3f result( offset2d.size() );
    for ( int i = 0; i < result.size(); ++i )
    {
        auto& res3I = result[i];
        const auto& res2I = offset2d[i];
        res3I.resize( res2I.size() );
        ParallelFor( 0, int( res3I.size() ), [&] ( int j )
        {
            res3I[j] = to3dim( res2I[j] );
            const auto& mapVal = map[i][j];
            if ( zParmas.zCallback )
            {
                res3I[j].z = zParmas.zCallback( offset2d, { .contourId = i,.vertId = j }, mapVal );
            }
            else
            {
                Line3f line( res3I[j], Vector3f::plusZ() );
                assert( mapVal.valid() );
                if ( !mapVal.isIntersection() )
                {
                    res3I[j].z = contours[mapVal.lOrg.contourId][mapVal.lOrg.vertId].z;
                }
                else
                {
                    float lzorg = contours[mapVal.lOrg.contourId][mapVal.lOrg.vertId].z;
                    float lzdest = contours[mapVal.lDest.contourId][mapVal.lDest.vertId].z;
                    float uzorg = contours[mapVal.uOrg.contourId][mapVal.uOrg.vertId].z;
                    float uzdest = contours[mapVal.uDest.contourId][mapVal.uDest.vertId].z;
                    res3I[j].z =
                        ( ( 1 - mapVal.lRatio ) * lzorg + mapVal.lRatio * lzdest +
                        ( 1 - mapVal.uRatio ) * uzorg + mapVal.uRatio * uzdest ) * 0.5f;
                }
            }
        } );
    }

    if ( zParmas.relaxIterations <= 0 )
        return result;
    for ( int i = 0; i < result.size(); ++i )
    {
        for ( int it = 0; it < zParmas.relaxIterations; ++it )
        {
            auto points = result[i];
            auto& refPoints = result[i];
            std::swap( points, refPoints );
            ParallelFor( 0, int( points.size() ), [&] ( int j )
            {
                int prevInd = ( j - 1 + int( points.size() ) ) % int( points.size() );
                int nextInd = ( j + 1 ) % int( points.size() );
                if ( prevInd + 1 == points.size() )
                    --prevInd;
                if ( nextInd == 0 )
                    ++nextInd;
                const auto& prevP = to2dim( points[prevInd] );
                const auto& curP = to2dim( points[j] );
                const auto& nextP = to2dim( points[nextInd] );
                float ratio = std::clamp( dot( curP - prevP, nextP - prevP ) / ( nextP - prevP ).lengthSq(), 0.0f, 1.0f );
                float targetZ = ( 1 - ratio ) * points[prevInd].z + ratio * points[nextInd].z;
                refPoints[j].z = ( targetZ + points[j].z ) * 0.5f;
            } );
        }
    }
    return result;
}

}