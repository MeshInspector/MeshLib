#include "MRDistanceMap.h"
#include "MRMeshIntersect.h"
#include "MRBox.h"
#include "MRImageSave.h"
#include "MRImageLoad.h"
#include "MRImage.h"
#include "MRTriangleIntersection.h"
#include "MRLine3.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRRegularGridMesh.h"
#include "MRPolyline2Project.h"
#include "MRBitSetParallelFor.h"
#include "MRPolyline2Intersect.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <vector>

namespace MR
{

static constexpr float NOT_VALID_VALUE = std::numeric_limits<float>::lowest();

DistanceMap::DistanceMap( const MR::Matrix<float>& m ) 
    : RectIndexer( m ) 
    , data_( m.data() )
{
}

DistanceMap::DistanceMap( size_t resX, size_t resY )
    : RectIndexer( { (int)resX, (int)resY } )
    , data_( size(), NOT_VALID_VALUE )
{
    invalidateAll();
}

bool DistanceMap::isValid( size_t x, size_t y ) const
{
    return data_[ toIndex( { int( x ), int( y ) } ) ] != NOT_VALID_VALUE;
}

bool DistanceMap::isValid( size_t i ) const
{
    return data_[i] != NOT_VALID_VALUE;
}

std::optional<float> DistanceMap::get( size_t x, size_t y ) const
{
    if ( isValid( x, y ) )
        return data_[ toIndex( { int( x ), int( y ) } ) ];
    else
        return std::nullopt;
}

std::optional<float> DistanceMap::get( size_t i ) const
{
    if ( isValid( i ) )
        return data_[i];
    else
        return std::nullopt;
}

std::optional<float> DistanceMap::getInterpolated( float x, float y ) const
{
    if ( x < 0.f )
    {
        return std::nullopt;
    }
    else if ( x < 0.5f )
    {
        x = 0.f;
    }
    else if ( x > float( resX() ) )
    {
        return std::nullopt;
    }
    else if( x > float( resX() ) - 0.5f )
    {
        x = float( resX() ) - 1.f;
    }
    else
    {
        x -= 0.5f;
    }

    if ( y < 0.f )
    {
        return std::nullopt;
    }
    else if ( y < 0.5f )
    {
        y = 0.f;
    }
    else if ( y > float( resY() ) )
    {
        return std::nullopt;
    }
    else if ( y > float( resY() ) - 0.5f )
    {
        y = float( resY() ) - 1.f;
    }
    else
    {
        y -= 0.5f;
    }

    float xlowf = float( std::floor( x ) );
    float ylowf = float( std::floor( y ) );
    float xhighf = xlowf + 1.f;
    float yhighf = ylowf + 1.f;
    int xlow = int( xlowf );
    int ylow = int( ylowf );
    int xhigh = xlow + 1;
    int yhigh = ylow + 1;

    auto lowlow = get( xlow, ylow );
    auto lowhigh = get( xlow, yhigh );
    auto highlow = get( xhigh, ylow );
    auto highhigh = get( xhigh, yhigh );
    if ( lowlow && lowhigh && highlow && highhigh )
    {
        // bilinear interpolation
        // https://en.wikipedia.org/wiki/Bilinear_interpolation
        return ( ( *lowlow ) * ( yhighf - y ) + ( *lowhigh ) * ( y - ylowf ) ) * ( xhighf - x ) +
            ( ( *highlow ) * ( yhighf - y ) + ( *highhigh ) * ( y - ylowf ) ) * ( x - xlowf );
    }
    else
    {
        return std::nullopt;
    }
}

void DistanceMap::set( size_t x, size_t y, float val )
{
    data_[ toIndex( { int( x ), int( y ) } ) ] = val;
}

void DistanceMap::set( size_t i, float val )
{
    data_[i] = val;
}

void DistanceMap::unset( size_t x, size_t y )
{
    data_[ toIndex( { int( x ), int( y ) } ) ] = NOT_VALID_VALUE;
}

void DistanceMap::unset( size_t i )
{
    data_[i] = NOT_VALID_VALUE;
}

void DistanceMap::invalidateAll()
{
    for( auto& elem : data_ )
        elem = NOT_VALID_VALUE;
}

void DistanceMap::clear()
{
    RectIndexer::resize( {0, 0} );
    data_.clear();
}

std::optional<Vector3f> DistanceMap::unproject( size_t x, size_t y, const DistanceMapToWorld& toWorldStruct ) const
{
    auto val = get( x, y );
    if ( !val )
        return {};
    return toWorldStruct.toWorld( x + 0.5f, y + 0.5f, *val );
}

std::optional<Vector3f> DistanceMap::unprojectInterpolated( float x, float y, const DistanceMapToWorld& toWorldStruct ) const
{
    auto val = getInterpolated( x, y );
    if ( !val )
        return {};
    return toWorldStruct.toWorld( x, y, *val );
}

Mesh distanceMapToMesh( const DistanceMap& distMap, const AffineXf3f& xf )
{
    DistanceMapToWorld toWorldParams;
    toWorldParams.direction = xf.A.z;
    toWorldParams.orgPoint = xf.b;
    toWorldParams.pixelXVec = xf.A.x;
    toWorldParams.pixelYVec = xf.A.y;

    return distanceMapToMesh( distMap, toWorldParams );
}

Mesh distanceMapToMesh( const DistanceMap& distMap, const DistanceMapToWorld& toWorldStruct )
{
    auto resX = distMap.resX();
    auto resY = distMap.resY();

    if (resX < 2 || resY < 2)
    {
        return Mesh();
    }

    return makeRegularGridMesh( resX, resY, [&]( size_t x, size_t y )
    {
        return distMap.isValid( x, y );
    },
                                            [&]( size_t x, size_t y )
    {
        return distMap.unproject( x, y, toWorldStruct ).value_or( Vector3f{} );
    } );
}

tl::expected<void, std::string> saveDistanceMapToImage( const DistanceMap& dm, const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    threshold = std::clamp( threshold, 0.f, 1.f );
    auto size = dm.numPoints();
    std::vector<Color> pixels( size );
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    // find min-max
    for ( int i = 0; i < size; ++i )
    {
        const auto val = dm.get( i );
        if ( val )
        {
            if ( *val < min )
                min = *val;
            if ( val > max )
                max = *val;
        }
    }

    for ( int i = 0; i < size; ++i )
    {
        const auto val = dm.get( i );
        pixels[i] = val ?
            Color( Vector3f::diagonal( ( max - *val ) / ( max - min ) * ( 1 - threshold ) + threshold ) ) :
            Color::black();
    }

    return ImageSave::toAnySupportedFormat( { pixels, { int( dm.resX() ), int( dm.resY() ) } }, filename );
}


tl::expected<MR::DistanceMap, std::string> convertImageToDistanceMap( const Image& image, float threshold /*= 1.f / 255*/ )
{
    threshold = std::clamp( threshold * 255, 0.f, 255.f );
    DistanceMap dm( image.resolution.x, image.resolution.y );
    const auto& pixels = image.pixels;
    for ( int i = 0; i < image.pixels.size(); ++i )
    {
        const bool monochrome = pixels[i].r == pixels[i].g && pixels[i].g == pixels[i].b;
        assert( monochrome );
        if ( !monochrome )
            return tl::make_unexpected( "Error convert Image to DistanceMap: image isn't monochrome" );
        if ( pixels[i].r < threshold )
            continue;
        dm.set( i, 255.0f - pixels[i].r );
    }
    return dm;
}

tl::expected<MR::DistanceMap, std::string> loadDistanceMapFromImage( const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    auto resLoad = ImageLoad::fromAnySupportedFormat( filename );
    if ( !resLoad.has_value() )
        return tl::make_unexpected( resLoad.error() );
    return convertImageToDistanceMap( *resLoad, threshold );
}

template <typename T = float>
DistanceMap computeDistanceMap_( const MeshPart& mp, const MeshToDistanceMapParams& params )
{
    DistanceMap distMap( params.resolution.x, params.resolution.y );

    // precomputed some values
    IntersectionPrecomputes<T> prec( Vector3<T>( params.direction ) );

    auto ori = params.orgPoint;
    float shift = 0.f;
    if ( params.allowNegativeValues )
    {
        AffineXf3f xf( Matrix3f( params.xRange.normalized(), params.yRange.normalized(), params.direction.normalized() ), Vector3f() );
        Box box = mp.mesh.computeBoundingBox( mp.region,&xf );
        shift = dot( params.direction, ori - box.min );
        if ( shift > 0.f )
        {
            ori -= params.direction * shift;
        }
        else
        {
            shift = T( 0 );
        }
    }

    T xStep_1 = T( 1 ) / T( params.resolution.x );
    T yStep_1 = T( 1 ) / T( params.resolution.y );
    if ( params.useDistanceLimits )
    {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, params.resolution.x ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t x = range.begin(); x < range.end(); x++ )
            {
                for ( size_t y = 0; y < params.resolution.y; y++ )
                {
                    Vector3<T> rayOri = Vector3<T>( ori ) +
                        Vector3<T>( params.xRange ) * ( ( T( x ) + T( 0.5 ) ) * xStep_1 ) +
                        Vector3<T>( params.yRange ) * ( ( T( y ) + T( 0.5 ) ) * yStep_1 );
                    if ( auto meshIntersectionRes = rayMeshIntersect( mp, Line3<T>( Vector3<T>( rayOri ), Vector3<T>( params.direction ) ),
                        -std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), &prec ) )
                    {
                        if ( ( meshIntersectionRes->distanceAlongLine < params.minValue ) || ( meshIntersectionRes->distanceAlongLine > params.maxValue ) )
                            distMap.set( x, y, meshIntersectionRes->distanceAlongLine );
                    }
                }
            }
        } );
    }
    else
    {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, params.resolution.x ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t x = range.begin(); x < range.end(); x++ )
            //debug line
            //for ( size_t x = 0; x < params.resX; x++ )
            {
                for ( size_t y = 0; y < params.resolution.y; y++ )
                {
                    Vector3<T> rayOri = Vector3<T>( ori ) +
                        Vector3<T>( params.xRange ) * ( ( T( x ) + T( 0.5 ) ) * xStep_1 ) +
                        Vector3<T>( params.yRange ) * ( ( T( y ) + T( 0.5 ) ) * yStep_1 );
                    if ( auto meshIntersectionRes = rayMeshIntersect( mp, Line3<T>( Vector3<T>( rayOri ), Vector3<T>( params.direction ) ),
                        -std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), &prec ) )
                    {
                        distMap.set( x, y, meshIntersectionRes->distanceAlongLine );
                    }
                }
            }
        } );
    }

    if ( params.allowNegativeValues )
    {
        for ( int i = 0; i < distMap.numPoints(); i++ )
        {
            const auto val = distMap.get( i );
            if ( val )
                distMap.set( i, *val - shift );
        }
    }

    return distMap;
}

DistanceMap computeDistanceMap( const MeshPart& mp, const MeshToDistanceMapParams& params )
{
    return computeDistanceMap_<float>( mp, params );
}

DistanceMap computeDistanceMapD( const MeshPart& mp, const MeshToDistanceMapParams& params )
{
    return computeDistanceMap_<double>( mp, params );
}

DistanceMap distanceMapFromContours( const Polyline2& polyline, const ContourToDistanceMapParams& params,
    const ContoursDistanceMapOptions& options )
{
    assert( polyline.topology.isConsistentlyOriented() );

    if ( options.offsetParameters )
    {
        bool goodSize = options.offsetParameters->perEdgeOffset.size() >= polyline.topology.undirectedEdgeSize();
        if ( !goodSize )
        {
            assert( false );
            spdlog::error( "Offset per edges should contain offset for all edges" );
            return {};
        }
    }

    const Vector3f originPoint = Vector3f{ params.orgPoint.x, params.orgPoint.y, 0.F } +
        Vector3f{ params.pixelSize.x / 2.F, params.pixelSize.y / 2.F, 0.F };

    size_t size = size_t( params.resolution.x ) * params.resolution.y;
    if ( options.outClosestEdges )
        options.outClosestEdges->resize( size );

    DistanceMap distMap( params.resolution.x, params.resolution.y );
    if ( !polyline.topology.lastNotLoneEdge().valid())
        return distMap;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, size ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( options.region && !options.region->test( PixelId( int( i ) ) ) )
                continue;
            size_t x = i % params.resolution.x;
            size_t y = i / params.resolution.x;
            Vector2f p;
            p.x = params.pixelSize.x * x + originPoint.x;
            p.y = params.pixelSize.y * y + originPoint.y;
            Polyline2ProjectionWithOffsetResult res;
            if ( options.offsetParameters )
            {
                res = findProjectionOnPolyline2WithOffset( p, polyline, options.offsetParameters->perEdgeOffset );
            }
            else
            {
                auto noOffsetRes = findProjectionOnPolyline2( p, polyline );
                res.line = noOffsetRes.line;
                res.point = noOffsetRes.point;
                res.dist = std::sqrt( noOffsetRes.distSq );
            }
            
            if ( options.outClosestEdges )
                ( *options.outClosestEdges )[i] = res.line;

            if ( params.withSign && ( !options.offsetParameters || options.offsetParameters->type != ContoursDistanceMapOffset::OffsetType::Shell ) )
            {
                bool positive = true;
                if ( options.signMethod == ContoursDistanceMapOptions::SignedDetectionMethod::ContourOrientation )
                {
                    const EdgeId e = res.line;
                    const auto& v0 = polyline.points[polyline.topology.org( e )];
                    const auto& v1 = polyline.points[polyline.topology.dest( e )];
                    auto vecA = v1 - v0;
                    auto ray = res.point - p;
                    float ratio = dot( res.point - v0, vecA ) / vecA.lengthSq();
                    if ( ratio <= 0.0f || ratio >= 1.0f )
                    {
                        Vector2f vecB;
                        const EdgeId prevEdge = polyline.topology.next( e ).sym();
                        const EdgeId nextEdge = polyline.topology.next( e.sym() );
                        if ( ratio <= 0.0f && e.sym() != prevEdge )
                        {
                            const auto& v2 = polyline.points[polyline.topology.org( prevEdge )];
                            vecB = v0 - v2;
                        }
                        else if ( ratio >= 1.0f && e.sym() != nextEdge )
                        {
                            const auto& v2 = polyline.points[polyline.topology.dest( nextEdge )];
                            vecB = v2 - v1;
                        }
                        vecA = ( vecA.normalized() + vecB.normalized() ) * 0.5f;
                    }
                    if ( cross( vecA, ray ) > 0.0f )
                        positive = false;
                }
                else if ( options.signMethod == ContoursDistanceMapOptions::SignedDetectionMethod::WindingRule )
                {
                    if ( isPointInsidePolyline( polyline, p ) )
                        positive = false;
                }
                if ( !positive )
                {
                    res.dist *= -1.0f;
                    if ( options.offsetParameters )
                        res.dist -= 2.0f * options.offsetParameters->perEdgeOffset[res.line];
                }
            }
            if ( !params.withSign && options.offsetParameters && options.offsetParameters->type == ContoursDistanceMapOffset::OffsetType::Shell )
                res.dist = std::abs( res.dist );
            distMap.set( x, y, res.dist );
        }
    } );
    return distMap;
}

std::vector<Vector3f> edgePointsFromContours( const Polyline2& polyline, float pixelSize, float threshold )
{
    assert( polyline.topology.isConsistentlyOriented() );
    std::vector<Vector3f> edgePoints;
    auto box = polyline.getBoundingBox();
    assert( box.valid() );
    auto resX = ( int )std::ceil( ( box.max.x - box.min.x ) / pixelSize );
    auto resY = ( int )std::ceil( ( box.max.y - box.min.y ) / pixelSize );

    std::vector<Vector2f> prevLine;
    Vector2f prevPix;
    prevLine.resize( resX );
    for ( auto x = 0; x < resX; x++ )
    {
        Vector2f p = box.min + Vector2f( pixelSize * ( float( x ) + 0.5f ), 0.f );
        auto res = findProjectionOnPolyline2( p, polyline );
        prevLine[x] = res.point;
    }

    float distLimitSq = threshold * threshold;
    for ( auto y = 1; y < resY; y++ )
    {
        {
            Vector2f p = box.min + Vector2f( 0.f, pixelSize * ( float( y ) + 0.5f ) );
            auto res = findProjectionOnPolyline2( p, polyline );
            prevPix = res.point;
        }
        for ( auto x = 1; x < resX; x++ )
        {
            Vector2f p = box.min + Vector2f( pixelSize * ( float( x ) + 0.5f ), pixelSize * ( float( y ) + 0.5f ) );
            auto res = findProjectionOnPolyline2( p, polyline );
            if ( ( res.point - prevPix ).lengthSq() > distLimitSq ||
                ( res.point - prevLine[x] ).lengthSq() > distLimitSq )
            {
                auto z = std::sqrt( res.distSq );
                edgePoints.emplace_back( p.x, p.y, z );
            }
            prevPix = res.point;
            prevLine[x] = res.point;
        }
    }
    return edgePoints;
}

Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float isoValue )
{
    MR_NAMED_TIMER( "distanceMapTo2DIsoPolyline" );
    const size_t resX = distMap.resX();
    const size_t resY = distMap.resY();
    if ( resX == 0 || resY == 0 )
        return {};
    const size_t size = resX * resY;

    struct SeparationPoint 
    {
        Vector2f coord;
        VertId id;
        bool low{ false }; // true means that left/down vert of edge is lower than iso when right/up is higher
                           // false means that left/down vert of edge is higher than iso when right/up is lower
    };
    auto horizontalEdgesSize = ( resX - 1 ) * resY;
    std::vector<SeparationPoint> separationPoints( horizontalEdgesSize + resX * ( resY - 1 ) );
    tbb::enumerable_thread_specific<size_t> numValidVertsPerThread( 0 );
    auto setupSeparation = [&] ( size_t x0, size_t y0, size_t x1, size_t y1 )
    {
        const auto v0 = distMap.getValue( x0, y0 );
        const auto v1 = distMap.getValue( x1, y1 );
        if ( v0 == NOT_VALID_VALUE || v1 == NOT_VALID_VALUE )
            return false;
        const bool low0 = v0 < isoValue;
        const bool low1 = v1 < isoValue;
        if ( low0 == low1 )
            return false;

        const float ratio = std::abs( isoValue - v0 ) / std::abs( v1 - v0 );
        size_t index = 0;
        if ( x1 == x0 )
        {
            // vertical edge
            assert( y1 == y0 + 1 );
            index = horizontalEdgesSize + x0 + y0 * resX;
        }
        else
        {
            // horizontal edge
            assert( y1 == y0 );
            assert( x1 == x0 + 1 );
            index = x0 + y0 * ( resX - 1 );
        }
        separationPoints[index].id = VertId( 0 );// real number now is not important, only that it is valid
        separationPoints[index].low = low0;
        separationPoints[index].coord = ( 1.0f - ratio ) * Vector2f( float( x0 ), float( y0 ) ) +
            ratio * Vector2f( float( x1 ), float( y1 ) ) + Vector2f::diagonal( 0.5f );
        return true;
    };
    // fill separationPoints
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, resY ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        size_t counter{ 0 };
        for ( size_t y = range.begin(); y < range.end() && y + 1 < resY; ++y )
            for ( size_t x = 0; x < resX; x++ )
                counter += setupSeparation( x, y, x, y + 1 );
        for ( size_t y = range.begin(); y < range.end(); ++y )
            for ( size_t x = 0; x + 1 < resX; x++ )
                counter += setupSeparation( x, y, x + 1, y );
        numValidVertsPerThread.local() += counter;
    } );
    size_t numValidVerts = 0;
    for ( const auto& num : numValidVertsPerThread )
        numValidVerts += num;

    Polyline2 polyline;
    polyline.points.resize( numValidVerts );
    VertId indexVert;
    for ( auto& sp : separationPoints )
    {
        if ( !sp.id )
            continue;
        sp.id = ++indexVert;
        polyline.points[indexVert] = sp.coord;
    }

    struct Line
    {
        VertId begin, end;
    };
    std::vector<Line> lines( size * 2 );

    SeparationPoint dummyPoint;

    auto findSeparation = [&] ( size_t x0, size_t y0, size_t x1, size_t y1 )->const SeparationPoint&
    {
        if ( x1 == resX || y1 == resY )
            return dummyPoint;
        size_t index = 0;
        if ( x1 == x0 )
        {
            // vertical edge
            assert( y1 == y0 + 1 );
            index = horizontalEdgesSize + x0 + y0 * resX;
        }
        else
        {
            // horizontal edge
            assert( y1 == y0 );
            assert( x1 == x0 + 1 );
            index = x0 + y0 * ( resX - 1 );
        }
        return separationPoints[index];
    };
    // find lines in each pixel independently
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, size ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            size_t x = i % resX;
            size_t y = i / resX;
            const auto& lowEdgeRes = findSeparation( x, y, x + 1, y );
            const auto& rightEdgeRes = findSeparation( x + 1, y, x + 1, y + 1 );
            const auto& upEdgeRes = findSeparation( x, y + 1, x + 1, y + 1 );
            const auto& leftEdgeRes = findSeparation( x, y, x, y + 1 );
            if ( !lowEdgeRes.id && !rightEdgeRes.id && !upEdgeRes.id && !leftEdgeRes.id )
                continue; // none
            auto& firstLine = lines[UndirectedEdgeId( 2 * i )];
            if ( lowEdgeRes.id && rightEdgeRes.id && upEdgeRes.id && leftEdgeRes.id )
            {
                // undetermined case, prefer left-lower and right-upper
                firstLine.begin = leftEdgeRes.id;
                firstLine.end = lowEdgeRes.id;
				
                lines[UndirectedEdgeId( 2 * i + 1 )].begin = rightEdgeRes.id;
                lines[UndirectedEdgeId( 2 * i + 1 )].end = upEdgeRes.id;
                if ( !leftEdgeRes.low )
                {
                    std::swap( firstLine.begin, firstLine.end );
                    std::swap( lines[UndirectedEdgeId( 2 * i + 1 )].begin, lines[UndirectedEdgeId( 2 * i + 1 )].end );
                }
                continue;
            }
            if ( lowEdgeRes.id && ( leftEdgeRes.id || upEdgeRes.id || rightEdgeRes.id ) )
            {
                if ( leftEdgeRes.id )
                    firstLine.begin = leftEdgeRes.id;
                else if ( upEdgeRes.id )
                    firstLine.begin = upEdgeRes.id;
                else if ( rightEdgeRes.id )
                    firstLine.begin = rightEdgeRes.id;
                firstLine.end = lowEdgeRes.id;
                if ( !lowEdgeRes.low )
                    std::swap( firstLine.begin, firstLine.end );
                continue;
            }
            if ( leftEdgeRes.id && ( upEdgeRes.id || rightEdgeRes.id ) )
            {
                if ( upEdgeRes.id )
                    firstLine.begin = upEdgeRes.id;
                else if ( rightEdgeRes.id )
                    firstLine.begin = rightEdgeRes.id;
                firstLine.end = leftEdgeRes.id;
                if ( leftEdgeRes.low )
                    std::swap( firstLine.begin, firstLine.end );
                continue;
            }
            if ( upEdgeRes.id && rightEdgeRes.id )
            {
                firstLine.begin = upEdgeRes.id;
                firstLine.end = rightEdgeRes.id;
                if ( !upEdgeRes.low )
                    std::swap( firstLine.begin, firstLine.end );
                continue;
            }
        }
    } );

    polyline.topology.vertResize( polyline.points.size() );
    for ( const auto & l : lines )
    {
        if ( l.begin && l.end )
            polyline.topology.makeEdge( l.begin, l.end );
    }
    assert( polyline.topology.isConsistentlyOriented() );

    return polyline;
}

Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, const ContourToDistanceMapParams& params, float isoValue )
{
    Polyline2 res = distanceMapTo2DIsoPolyline( distMap, isoValue );

    BitSetParallelFor( res.topology.getValidVerts(), [&] ( VertId v )
    {
        res.points[v] = params.toWorld( res.points[v] );
    } );

    return res;
}

std::pair<MR::Polyline2, MR::AffineXf3f> distanceMapTo2DIsoPolyline( const DistanceMap& distMap,
    const DistanceMapToWorld& params, float isoValue, bool useDepth /* = false */)
{
    Polyline2 resContours = distanceMapTo2DIsoPolyline( distMap, isoValue );

    const float depth = useDepth ? isoValue : 0.F;
    const Matrix3f m = Matrix3f::fromColumns( params.pixelXVec, params.pixelYVec, params.direction );
    const AffineXf3f resXf{ m, params.toWorld( 0.F, 0.F, depth ) };
    const AffineXf3f resInv = resXf.inverse();

    BitSetParallelFor( resContours.topology.getValidVerts(), [&] ( VertId v )
    {
        const Vector3f p = params.toWorld( resContours.points[v].x, resContours.points[v].y, 0.F );
        const Vector3f pInv = resInv( p );
        resContours.points[v] = Vector2f{ pInv.x, pInv.y };
    } );

    return { resContours, resXf };
}

Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float pixelSize, float isoValue )
{
    ContourToDistanceMapParams params;
    params.pixelSize = Vector2f{ pixelSize, pixelSize };
    params.resolution = Vector2i{ int( distMap.resX() ), int( distMap.resY() ) };
    auto res = distanceMapTo2DIsoPolyline( distMap, params, isoValue );
    return res;
}

void DistanceMap::negate()
{
    for( auto & v : data_ )
        if ( v != NOT_VALID_VALUE )
            v = -v;
}

// boolean operators
DistanceMap DistanceMap::max( const DistanceMap& rhs ) const
{
    DistanceMap res( resX(), resY() );
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( val )
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, std::max( *val, *valrhs ) );
                    }
                    else
                    {
                        res.set( iX, iY, *val );
                    }
                }
                else
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, *valrhs );
                    }
                }
            }
            else
            {
                res.set( iX, iY, *val );
            }
        }
    }
    return res;
}

const DistanceMap& DistanceMap::mergeMax( const DistanceMap& rhs)
{
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for ( auto iY = 0; iY < resY(); iY++ )
        {
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( valrhs )
                {
                    const auto val = get( iX, iY );
                    if ( !val || ( *val < *valrhs ) )
                        set( iX, iY, *valrhs );
                }
            }
        }
    }
    return *this;
}

DistanceMap DistanceMap::min( const DistanceMap& rhs) const
{
    DistanceMap res( resX(), resY() );
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for ( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( val )
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, std::min( *val, *valrhs ) );
                    }
                    else
                    {
                        res.set( iX, iY, *val );
                    }
                }
                else
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, *valrhs );
                    }
                }
            }
            else
            {
                res.set( iX, iY, *val );
            }
        }
    }
    return res;
}

const DistanceMap& DistanceMap::mergeMin( const DistanceMap& rhs )
{
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( valrhs )
                {
                    const auto val = get( iX, iY );
                    if ( !val || ( *val > *valrhs ) )
                        set( iX, iY, *valrhs );
                }
            }
        }
    }
    return *this;
}

DistanceMap DistanceMap::operator-( const DistanceMap& rhs) const
{
    DistanceMap res( resX(), resY() );
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( val )
            {
                if ( iX < rhs.resX() && iY < rhs.resY() )
                {
                    const auto valrhs = rhs.get( iX, iY );
                    if ( valrhs )
                    {
                        res.set( iX, iY, *val - *valrhs );
                    }
                }
                else
                {
                    res.set( iX, iY, *val );
                }
            }
        }
    }
    return res;
}

const DistanceMap& DistanceMap::operator-=( const DistanceMap& rhs )
{
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( val )
            {
                if ( iX < rhs.resX() && iY < rhs.resY() )
                {
                    const auto valrhs = rhs.get( iX, iY );
                    if ( valrhs )
                    {
                        set( iX, iY, *val - *valrhs );
                    }
                }
            }
        }
    }
    return *this;
}

DistanceMap DistanceMap::getDerivativeMap() const
{
    auto XYmaps = getXYDerivativeMaps();
    return combineXYderivativeMaps( XYmaps );
}

std::pair< DistanceMap, DistanceMap > DistanceMap::getXYDerivativeMaps() const
{
    auto res = std::make_pair( DistanceMap( resX(), resY() ), DistanceMap( resX(), resY() ) );
    DistanceMap& dx = res.first;
    DistanceMap& dy = res.second;
    if (resX() < 3 || resY() < 3)
    {
        return res;
    }
    tbb::parallel_for( tbb::blocked_range<size_t>( 1, resX() - 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t x = range.begin(); x < range.end(); x++ )
        {
            for ( size_t y = 1; y + 1 < resY(); y++ )
            {
                const auto val = get( x, y );
                if ( val )
                {
                    const auto valxpos = get( x + 1, y );
                    const auto valxneg = get( x - 1, y );
                    if ( valxpos )
                        if ( valxneg )
                            dx.set( x, y, ( ( *valxpos ) - ( *valxneg ) ) / 2.f );
                        else
                            dx.set( x, y, ( *valxpos ) - ( *val ) );
                    else
                        if ( valxneg )
                            dx.set( x, y, ( *val ) - ( *valxneg ) );
                        else
                            dx.unset( x, y );

                    const auto valypos = get( x, y + 1 );
                    const auto valyneg = get( x, y - 1 );
                    if ( valypos )
                        if ( valyneg )
                            dy.set( x, y, ( ( *valypos ) - ( *valyneg ) ) / 2.f );
                        else
                            dy.set( x, y, ( *valypos ) - ( *val ) );
                    else
                        if ( valyneg )
                            dy.set( x, y, ( *val ) - ( *valyneg ) );
                        else
                            dy.unset( x, y );
                }
            }
        }
    } );
    return res;
}

DistanceMap combineXYderivativeMaps( std::pair<DistanceMap, DistanceMap> XYderivativeMaps )
{
    assert( XYderivativeMaps.first.resX() == XYderivativeMaps.second.resX() );
    assert( XYderivativeMaps.first.resY() == XYderivativeMaps.second.resY() );

    DistanceMap dMap( XYderivativeMaps.first.resX(), XYderivativeMaps.second.resY() );
    const auto& dx = XYderivativeMaps.first;
    const auto& dy = XYderivativeMaps.second;

    if ( dx.resX() < 3 || dx.resY() < 3 )
    {
        return dMap;
    }
    // fill the central area
    tbb::parallel_for( tbb::blocked_range<size_t>( 1, dx.resX() - 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t x = range.begin(); x < range.end(); x++ )
        {
            for ( size_t y = 1; y + 1 < dx.resY(); y++ )
            {
                const auto valX = dx.get( x, y );
                const auto valY = dy.get( x, y );
                
                if (valX)
                {
                    if (valY)
                    {
                        dMap.set( x, y, std::sqrt( ( *valX ) * ( *valX ) + ( *valY ) * ( *valY ) ) );
                    }
                    else
                    {
                        dMap.set( x, y, *valY );
                    }
                }
                else
                {
                    if (valY)
                    {
                        dMap.set( x, y, *valY );
                    }
                    else
                    {
                        dMap.unset( x, y );
                    }
                }
            }
        }
    } );
    return dMap;
}

std::vector<std::pair<size_t, size_t>>  DistanceMap::getLocalMaximums() const
{
    typedef std::vector<std::pair<size_t, size_t>> LocalMaxAcc;
    LocalMaxAcc acc;
    // ignore border cases (y==0 or y==resY()-1)
    auto min = tbb::parallel_reduce( tbb::blocked_range<size_t>( resX(), numPoints() - resX() ), acc,
    [&] ( const tbb::blocked_range<size_t> range, LocalMaxAcc localAcc )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            // ignore border cases (x==0 or x==resX()-1)
            if ( ( i % resX() == 0 ) || ( ( i + 1 ) % resX() == 0 ) )
            {
                continue;
            }
            const auto& val = data_[i];
            if ( ( data_[i - 1 - resX()] < val ) &&
                 ( data_[i - 1] < val ) &&
                 ( data_[i - 1 + resX()] < val ) &&
                 ( data_[i - resX()] < val ) &&
                 ( data_[i + resX()] < val ) &&
                 ( data_[i + 1 - resX()] < val ) &&
                 ( data_[i + 1] < val ) &&
                 ( data_[i + 1 + resX()] < val ) )
            {
                localAcc.push_back( { i % resX(), i / resX() } );
            }
        }
        return localAcc;
    },
    [&] ( const LocalMaxAcc& a, const LocalMaxAcc& b )
    {
        LocalMaxAcc res = a;
        res.insert( res.end(), b.begin(), b.end() );
        return res;
    } );

    return acc;
}

std::pair<float, float> DistanceMap::getMinMaxValues() const
{
    struct MinMax
    {
        float min;
        float max;
        size_t minI;
        size_t maxI;
    };
    MinMax minElem{ std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 0, 0 };
    auto minmax = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, numPoints() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MinMax curMinMax )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            auto val = get( i );
            if ( val )
            {
                if ( *val < curMinMax.min )
                {
                    curMinMax.min = *val;
                    curMinMax.minI = i;
                }
                if ( *val > curMinMax.max )
                {
                    curMinMax.max = *val;
                    curMinMax.maxI = i;
                }
            }
        }
        return curMinMax;
    },
    [&] ( const MinMax& a, const MinMax& b )
    {
        MinMax res;
        if ( a.min < b.min )
        {
            res.min = a.min;
            res.minI = a.minI;
        }
        else
        {
            res.min = b.min;
            res.minI = b.minI;
        }
        if ( a.max > b.max )
        {
            res.max = a.max;
            res.maxI = a.maxI;
        }
        else
        {
            res.max = b.max;
            res.maxI = b.maxI;
        }
        return res;
    } );

    return { minmax.min, minmax.max };
}

std::pair<size_t, size_t> DistanceMap::getMinIndex() const
{
    typedef std::pair<float, size_t> MinElem;
    MinElem minElem{ std::numeric_limits<float>::max(), 0 };
    auto min = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, numPoints() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MinElem curMin )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            auto val = get( i );
            if ( val && ( *val < curMin.first ) )
            {
                curMin.first = *val;
                curMin.second = i;
            }
        }
        return curMin;
    },
    [&] ( const MinElem& a, const MinElem& b )
    {
        return a.first < b.first ? a : b;
    } );

    return { min.second / resY(), min.second % resY() };
}

std::pair<size_t, size_t> DistanceMap::getMaxIndex() const
{
    typedef std::pair<float, size_t> MaxElem;
    MaxElem minElem{ -std::numeric_limits<float>::max(), 0 };
    auto max = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, numPoints() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MaxElem curMin )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            auto val = get( i );
            if ( val && ( *val > curMin.first ) )
            {
                curMin.first = *val;
                curMin.second = i;
            }
        }
        return curMin;
    },
    [&] ( const MaxElem& a, const MaxElem& b )
    {
        return a.first > b.first ? a : b;
    } );

    return { max.second / resY(), max.second % resY() };
}

Polyline2 contourUnion( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside )
{
    assert( params.withSign );
    auto mapA = distanceMapFromContours( contoursA, params );
    auto mapB = distanceMapFromContours( contoursB, params );
    mapA.mergeMin( mapB );
    return distanceMapTo2DIsoPolyline( mapA, params, offsetInside );
}

Polyline2 contourIntersection( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside )
{
    assert( params.withSign );
    auto mapA = distanceMapFromContours( contoursA, params );
    auto mapB = distanceMapFromContours( contoursB, params );
    mapA.mergeMax( mapB );
    return distanceMapTo2DIsoPolyline( mapA, params, offsetInside );
}

Polyline2 contourSubtract( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside )
{
    assert( params.withSign );
    auto mapA = distanceMapFromContours( contoursA, params );
    auto mapB = distanceMapFromContours( contoursB, params );
    mapB.negate();
    mapA.mergeMax( mapB );
    return distanceMapTo2DIsoPolyline( mapA, params, offsetInside );
}

} //namespace MR
