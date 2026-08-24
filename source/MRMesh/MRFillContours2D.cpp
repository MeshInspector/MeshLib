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
#include "MRMapEdge.h"
#include "MRBitSet.h"
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

    // patch boundary edge (by undirected id) -> the mesh edge it copies; the peel anchors through it
    WholeEdgeMap bd2mesh;
    auto patch = PlanarTriangulation::triangulateDisjointContours( mesh, loops, Vector3f( sumCross.normalized() ), nullptr, &bd2mesh );
    if ( !patch )
        return unexpected( "Cannot triangulate contours with self-intersections" );

    const auto& pTp = patch->topology;
    HoleFillPlan res;
    res.numTris = pTp.numValidFaces();
    if ( res.numTris == 1 )
        return res;

    // faces must lie on the left of the boundary; a degenerate (zero area) hole can be filled on the wrong side
    if ( pTp.right( EdgeId( 0 ) ) )
        return unexpected( "Incorrect filling" );

    const int n = int( loops.front().size() );
    assert( n > 3 );
    // interior edges: of the 3 * numTris face sides, each boundary edge covers one and each interior edge two
    const int numChords = ( 3 * res.numTris - int( bd2mesh.size() ) ) / 2;

    // the peel's current polygon: one slot per boundary edge, the rings implicit in succ
    struct Slot
    {
        EdgeId cur;   // current polygon edge in the patch, invalid = consumed position
        int refCode;  // plan code of cur: not-negative absolute mesh EdgeId, negative - earlier plan edge
        int succ;     // next slot around the polygon
    };
    std::vector<Slot> slots;
    slots.reserve( n );
    // EdgeId( 0 ) is the first boundary edge the copy created, so a plain loop yields the loop order.
    // Turn::Leftmost keeps the walk in the same region corner at a pinch vertex (the input loop's own
    // pairing); the default tight-right turn would pair the arrival with the other cavity's exit
    auto walkRing = [&]( EdgeId b0, UndirectedEdgeBitSet* visited )
    {
        const int first = int( slots.size() );
        EdgeId b = b0;
        do
        {
            if ( int( slots.size() ) >= n )
                return false;
            if ( visited )
                visited->set( b.undirected() );
            slots.push_back( { b, int( mapEdge( bd2mesh, b ) ), int( slots.size() ) + 1 } );
            b = pTp.nextLeftBd( b, nullptr, Turn::Leftmost );
        } while ( b != b0 );
        slots.back().succ = first;
        return true;
    };
    if ( !walkRing( EdgeId( 0 ), nullptr ) )
        return unexpected( "Incorrect filling" );
    if ( int( slots.size() ) != n )
    {
        // a pinched hole: the boundary is several rings sharing vertices; reseed recording visits
        slots.clear();
        UndirectedEdgeBitSet visited( bd2mesh.size() );
        for ( UndirectedEdgeId ue{ 0 }; ue < bd2mesh.size(); ++ue )
        {
            if ( visited.test( ue ) )
                continue;
            const EdgeId b{ ue };
            const bool bdFwd = !pTp.right( b ), bdBack = !pTp.left( b );
            if ( bdFwd == bdBack ) // an edge the hole boundary traverses twice: no face side to seed from
                return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
            if ( !walkRing( bdFwd ? b : b.sym(), &visited ) )
                return unexpected( "Incorrect filling" );
        }
        if ( int( slots.size() ) != n )
            return unexpected( "Incorrect filling" );
    }

    res.items.reserve( numChords );
    while ( int( res.items.size() ) < numChords )
    {
        bool progress = false;
        for ( int s0 = 0; s0 < int( slots.size() ) && int( res.items.size() ) < numChords; ++s0 )
        {
            if ( !slots[s0].cur )
                continue; // consumed position
            const int s1 = slots[s0].succ;
            const int s2 = slots[s1].succ;
            if ( slots[s2].succ == s0 )
                continue; // triangle ring: its last face needs no new edge
            const EdgeId ne = pTp.next( slots[s0].cur );
            if ( pTp.dest( ne ) != pTp.dest( slots[s1].cur ) )
                continue; // left face of cur[s0] is not an ear here
            if ( ne.undirected() < bd2mesh.size() )
                slots[s0] = { ne, int( mapEdge( bd2mesh, ne ) ), s2 }; // pinched ring: the closing edge exists in the mesh
            else
            {
                res.items.push_back( { slots[s0].refCode, slots[s2].refCode } );
                slots[s0] = { ne, FillHoleItemEdge{ .item = int( res.items.size() ) - 1 }.encode(), s2 };
            }
            slots[s1].cur = {};
            progress = true;
        }
        if ( !progress )
            return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
    }
    return res;
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
