#include "MRFillContours2D.h"
#include "MRMesh.h"
#include "MRVector2.h"
#include "MR2DContoursTriangulation.h"
#include "MRRingIterator.h"
#include "MREdgePaths.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRFillContour.h"
#include "MRObjectMeshData.h"
#include "MRColor.h"
#include "MRMeshFillHole.h"
#include "MRBox.h"
#include <cfloat>

namespace MR
{

class FromOxyPlaneCalculator
{
public:
    void addLineSegm( const Vector3d & a, const Vector3d & b )
    {
        sumPts_ += a;
        sumPts_ += b;
        numPts_ += 2;
        sumCross_ += cross( a, b );
    }
    void addLineSegm( const Vector3f & a, const Vector3f & b )
    {
        addLineSegm( Vector3d( a ), Vector3d( b ) );
    }
    AffineXf3d getXf() const
    {
        if ( numPts_ <= 0 )
            return {};
        auto planeNormal = sumCross_.normalized();
        auto center = sumPts_ / double( numPts_ );
        return { Matrix3d::rotation( Vector3d::plusZ(), planeNormal ), center };
    }

private:
    Vector3d sumPts_;
    Vector3d sumCross_;
    int numPts_ = 0;
};

AffineXf3f getXfFromOxyPlane( const Mesh& mesh, const std::vector<EdgePath>& paths )
{
    FromOxyPlaneCalculator c;
    for ( const auto& path : paths )
    {
        for ( const auto& edge : path )
            c.addLineSegm( mesh.orgPnt( edge ), mesh.destPnt( edge ) );
    }
    return AffineXf3f( c.getXf() );
}

AffineXf3f getXfFromOxyPlane( const Contours3f& contours )
{
    FromOxyPlaneCalculator c;
    for ( const auto& contour : contours )
    {
        for ( int i = 0; i + 1 < contour.size(); ++i )
            c.addLineSegm( contour[i], contour[i + 1] );
    }
    return AffineXf3f( c.getXf() );
}

struct ProjectFillInput
{
    std::vector<EdgeLoop> paths;
    AffineXf3f fromPlaneXf;
    Contours2f holes2d;
};

Expected<ProjectFillInput> projectHoles( const Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges )
{
    MR_TIMER;
    assert( !holeRepresentativeEdges.empty() );
    if ( holeRepresentativeEdges.empty() )
        return unexpected( "No hole edges are given" );

    // reorder to make edges ring with hole on left side
    bool badEdge = false;
    auto& meshTopology = mesh.topology;
    for ( const auto& edge : holeRepresentativeEdges )
    {
        if ( meshTopology.left( edge ) )
        {
            badEdge = true;
            break;
        }
    }
    assert( !badEdge );
    if ( badEdge )
        return unexpected( "Some hole edges have left face" );

    ProjectFillInput res;
    // make border rings
    res.paths.resize( holeRepresentativeEdges.size() );
    for ( int i = 0; i < res.paths.size(); ++i )
        res.paths[i] = trackRightBoundaryLoop( meshTopology, holeRepresentativeEdges[i] );

    // find transformation from world to plane space and back
    res.fromPlaneXf = getXfFromOxyPlane( mesh, res.paths );
    const auto toPlane = res.fromPlaneXf.inverse();

    // make contours2D (on plane) from border rings (in world)
    res.holes2d.reserve( res.paths.size() );
    for ( const auto& path : res.paths )
    {
        res.holes2d.emplace_back();
        auto& contour = res.holes2d.back();
        contour.reserve( path.size() + 1 );
        for ( const auto& edge : path )
        {
            const auto localPoint = toPlane( mesh.orgPnt( edge ) );
            contour.emplace_back( Vector2f( localPoint.x, localPoint.y ) );
        }
        contour.emplace_back( contour.front() );
    }

    return res;
}

struct ProjectedFillMesh
{
    Mesh mesh;
    std::vector<EdgeLoop> paths;
};

// checks that patch boundary loops match the input hole loops one-to-one and re-orients inverted
// degenerate holes (expected sometimes as far as planar triangulation does not now about input topology)
static Expected<void> validateAndFixPatch( MeshTopology& patchTopology, const std::vector<EdgeLoop>& inputPaths, std::vector<EdgePath>& patchPaths )
{
    if ( inputPaths.size() != patchPaths.size() )
        return unexpected( "Patch surface borders size different from original mesh borders size" );

    std::vector<EdgePath> invertedHoles;
    invertedHoles.reserve( patchPaths.size() );
    for ( int i = 0; i < patchPaths.size(); ++i )
    {
        if ( inputPaths[i].size() != patchPaths[i].size() )
            return unexpected( "Patch surface borders size different from original mesh borders size" );

        if ( patchPaths[i].empty() || patchTopology.right( patchPaths[i].front() ) )
            if ( !patchPaths[i].empty() )
                MR::reverse( invertedHoles.emplace_back( patchPaths[i] ) );
    }
    if ( !invertedHoles.empty() )
    {
        auto invertedParts = fillContourLeft( patchTopology, invertedHoles );
        auto invertedEdges = getIncidentEdges( patchTopology, invertedParts );
        patchTopology.flipOrientation( &invertedEdges );

        // validate one more time
        for ( int i = 0; i < patchPaths.size(); ++i )
            if ( patchPaths[i].empty() || patchTopology.right( patchPaths[i].front() ) )
                if ( !patchPaths[i].empty() )
                    return unexpected( "Patch surface borders are incompatible with mesh borders" );
    }
    return {};
}

Expected<ProjectedFillMesh> fillProjected( const MeshTopology& tp, const ProjectFillInput& input )
{
    MR_TIMER;
    ProjectedFillMesh res;

    auto holeVertIds = std::make_unique<PlanarTriangulation::HolesVertIds>(
        PlanarTriangulation::findHoleVertIdsByHoleEdges( tp, input.paths ) );

    auto fillResult = PlanarTriangulation::triangulateDisjointContours( input.holes2d, holeVertIds.get(), &res.paths );
    holeVertIds.reset();
    if ( !fillResult )
        return unexpected( "Cannot triangulate contours with self-intersections" );

    res.mesh = std::move( *fillResult );

    if ( auto v = validateAndFixPatch( res.mesh.topology, input.paths, res.paths ); !v )
        return unexpected( std::move( v.error() ) );
    return res;
}

Expected<void> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges )
{
    MR_TIMER;

    auto projInput = projectHoles( mesh, holeRepresentativeEdges );
    if ( !projInput.has_value() )
        return unexpected( std::move( projInput.error() ) );

    auto fillRes = fillProjected( mesh.topology, *projInput );
    if ( !fillRes.has_value() )
        return unexpected( std::move( fillRes.error() ) );

    // move patch surface border points to original position (according original mesh)
    auto& patchMeshPoints = fillRes->mesh.points;
    auto& patchMeshTopology = fillRes->mesh.topology;
    auto& meshPoints = mesh.points;
    auto& meshTopology = mesh.topology;
    for ( int i = 0; i < projInput->paths.size(); ++i )
    {
        auto& path = projInput->paths[i];
        auto& newPath = fillRes->paths[i];
        for ( int j = 0; j < path.size(); ++j )
            patchMeshPoints[patchMeshTopology.org( newPath[j] )] = meshPoints[meshTopology.org( path[j] )];
    }

    // add patch surface to original mesh
    mesh.addMeshPart( fillRes->mesh, false, projInput->paths, fillRes->paths );
    return {};
}

Expected<HoleFillPlan> fillContours2DPlan( const Mesh& mesh, EdgeId holeEdgeId )
{
    assert( !mesh.topology.left( holeEdgeId ) );
    if ( mesh.topology.left( holeEdgeId ) )
        return unexpected( "Hole edge has left face" );

    // triangulate the hole in the mesh's own 3d space: only the plan is needed, so the projection
    // round-trip of the mesh-filling path above is avoided
    EdgeLoops loops( 1 );
    loops.front() = trackRightBoundaryLoop( mesh.topology, holeEdgeId );

    // the hole loop winds counterclockwise around this direction, same as around the fitted plane's normal it replaces
    Vector3d sumCross;
    Box3f loopBox;
    float minEdgeLenSq = FLT_MAX;
    for ( EdgeId e : loops.front() )
    {
        const auto o = mesh.orgPnt( e ), d = mesh.destPnt( e );
        sumCross += cross( Vector3d( o ), Vector3d( d ) );
        loopBox.include( o );
        minEdgeLenSq = std::min( minEdgeLenSq, ( d - o ).lengthSq() );
    }

    // A hairline boundary edge forces every triangulation to emit a zero-area needle whose placement
    // in 3d is arbitrary; two holes sharing such an edge pick their needles independently and can make
    // them cross. Leave those contours to the metric fill, which weighs triangle shape.
    if ( minEdgeLenSq < 1e-12f * loopBox.size().lengthSq() )
        return unexpected( "Hole boundary has a degenerate edge" );

    std::vector<EdgePath> newPaths;
    auto patch = PlanarTriangulation::triangulateDisjointContours( mesh, loops, Vector3f( sumCross.normalized() ), &newPaths );
    if ( !patch )
        return unexpected( "Cannot triangulate contours with self-intersections" );

    if ( auto v = validateAndFixPatch( patch->topology, loops, newPaths ); !v )
        return unexpected( std::move( v.error() ) );

    const auto& pTp = patch->topology;
    const auto& ip = loops.front();
    auto& np = newPaths.front();
    HoleFillPlan res;
    res.numTris = pTp.numValidFaces();
    if ( res.numTris == 1 )
        return res;
    auto size = int( np.size() );
    assert( size > 3 );
    res.items.reserve( size - 3 );

    for ( ;;)
    {
        for ( int i0 = 0; i0 < np.size(); ++i0 )
        {
            auto e0 = np[i0];
            if ( !e0 )
                continue; // skip unused/encoded 
            auto i1 = int( pTp.dest( np[i0] ) );
            if ( i1 < 0 )
                return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
            auto e1 = np[i1];
            if ( !e1 )
                return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
            auto ne = pTp.next( e0 );
            auto dest = pTp.dest( ne );
            if ( dest != pTp.dest( e1 ) )
                continue;
            FillHoleItem fhi;
            int i01 = ( i0 + 1 ) % size;
            fhi.edgeCode1 = i1 == i01 ? ip[i0] : int( np[i01] );
            i1 = dest;
            e1 = np[i1];
            if ( !e1 )
                return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
            int i11 = ( i1 + 1 ) % size;
            fhi.edgeCode2 = ( pTp.dest( e1 ) == i11 ) ? ip[i1] : int( np[i11] );
            res.items.push_back( std::move( fhi ) );
            if ( res.items.size() == size - 3 )
                return res;
            np[i0] = ne;
            np[i01] = EdgeId( fillHoleItemCode( { .item = int( res.items.size() ) - 1 } ) ); // encode the just pushed plan edge in free slot
        }
    }
}

Expected<void> fillPlanarHole( ObjectMeshData& data, std::vector<EdgeLoop>& holeContours )
{
    MR_TIMER;

    if ( !data.mesh )
        return unexpected( "fillPlanarHole: no input mesh" );

    auto& mesh = *data.mesh;
    auto& tp = mesh.topology;

    // take first edge from each contour and check that it is a hole boundary
    EdgePath holesEdges;
    for ( const auto& path : holeContours )
    {
        if ( path.empty() )
            continue;
        for ( auto e : path )
            if ( tp.right( e ).valid() )
                return unexpected( "fillPlanarHole: not hole contour given" );
        holesEdges.push_back( path.front().sym() );
    }

    for ( auto& loop : holeContours )
    {
        // if not closed, add edge to enclose
        if ( loop.empty() )
            continue;
        if ( tp.org( loop.front() ) == tp.dest( loop.back() ) )
            continue;
        auto newEdge = makeBridgeEdge( tp, loop.back().sym(), tp.prev( loop.front() ) );
        if ( !newEdge )
            continue;
        loop.emplace_back( newEdge );
    }

    const auto fsz0 = tp.faceSize();
    if ( !holesEdges.empty() )
    {
        auto fillSuccess = fillContours2D( mesh, holesEdges );
        if ( !fillSuccess.has_value() )
        {
            return unexpected( "Cannot fill section: " + fillSuccess.error() );
        }
    }

    const auto fsz = tp.faceSize();
    data.selectedFaces.resize( fsz );
    data.selectedFaces.set( FaceId{ fsz0 }, fsz - fsz0, true );
    data.selectedFaces &= tp.getValidFaces();

    tp.excludeLoneEdges( data.selectedEdges );
    tp.excludeLoneEdges( data.creases );

    auto& fcm = data.faceColors;
    auto& tpf = data.texturePerFace;
    if ( fcm.empty() && tpf.empty() )
        return {};
    if ( !fcm.empty() )
        fcm.resize( fsz );
    if ( !tpf.empty() )
        tpf.resize( fsz );
    for ( FaceId f = FaceId{ fsz0 }; f < fsz; ++f )
    {
        VertId v[3];
        tp.getTriVerts( f, v );
        float sumNeighColorWeight = 0;
        Vector4f sum;
        FaceId maxAreaF;
        float maxArea = 0.0f;
        for ( size_t i = 0; i < 3; ++i )
        {
            for ( auto e : orgRing( tp, v[i] ) )
            {
                const auto tmpFace = tp.left( e );
                if ( tmpFace >= fsz0 )
                    continue;

                const float area = mesh.area( tmpFace );
                if ( !tpf.empty() )
                {
                    if ( area > maxArea )
                    {
                        maxArea = area;
                        maxAreaF = tmpFace;
                    }
                }
                if ( !fcm.empty() )
                {
                    const auto& color = fcm[tmpFace];
                    sum += Vector4f( color ) * area;
                    sumNeighColorWeight += area;
                }
            }
        }
        if ( !fcm.empty() )
            fcm[f] = Color( sum / float( sumNeighColorWeight ) );
        if ( !tpf.empty() )
            tpf[f] = tpf[maxAreaF];
    }

    return {};
}

}
