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

// converts a triangulated patch into a plan of new edges for executeHoleFillPlan.
// The patch is peeled ear by ear, each ear recording its closing edge as a chord between the polygon
// edges currently at its ends. executeHoleFillPlan re-creates such a chord by splicing it right after
// those edges in their vertex rings, and every later chord at the same vertex lands closer to the
// anchor - which is exactly the patch's angular order, also where several holes meet in one vertex.
static Expected<HoleFillPlan> patchToPlan( const MeshTopology& pTp, const std::vector<EdgeLoop>& meshLoops,
    const std::vector<EdgePath>& patchBds, std::vector<EdgeId>* outSingleFaceHoles )
{
    assert( meshLoops.size() == patchBds.size() ); // validated in validateAndFixPatch

    // peeling state per boundary point; each current polygon ring is implicit in succ
    int n = 0;
    for ( const auto& loop : meshLoops )
        n += int( loop.size() );
    std::vector<EdgeId> cur;  // current polygon edge in the patch, invalid = consumed position
    std::vector<int> refCode; // plan code of cur: not-negative absolute EdgeId in the mesh, negative - plan edge
    std::vector<int> succ;    // next position around the polygon
    cur.reserve( n + patchBds.size() ); // joining two rings adds a position
    refCode.reserve( n + patchBds.size() );
    succ.reserve( n + patchBds.size() );
    Vector<EdgeId, UndirectedEdgeId> bd2mesh( n ); // patch boundary edge (even direction) -> mesh edge
    for ( int j = 0, base = 0; j < int( patchBds.size() ); base += int( patchBds[j].size() ), ++j )
    {
        for ( int i = 0; i < int( patchBds[j].size() ); ++i )
        {
            const EdgeId b = patchBds[j][i];
            if ( !pTp.left( b ) )
                return unexpected( "Patch surface borders are incompatible with mesh borders" );
            cur.push_back( b );
            refCode.push_back( int( meshLoops[j][i] ) );
            succ.push_back( i + 1 < int( patchBds[j].size() ) ? int( cur.size() ) : base );
            bd2mesh.autoResizeSet( b.undirected(), b.even() ? meshLoops[j][i] : meshLoops[j][i].sym() );
        }
    }
    auto meshEdge = [&] ( EdgeId patchBd )
    {
        return patchBd.even() ? bd2mesh[patchBd.undirected()] : bd2mesh[patchBd.undirected()].sym();
    };

    UndirectedEdgeBitSet bdEdges( pTp.undirectedEdgeSize() );
    for ( auto e : cur )
        bdEdges.set( e.undirected() );

    // the plan has to create every interior edge of the patch, and every filled face must be adjacent
    // to one of them (or belong to a hole filled with a single face, see below)
    int numChords = 0;
    FaceBitSet chordFaces( pTp.faceSize() );
    for ( UndirectedEdgeId ue{ 0 }; ue < pTp.undirectedEdgeSize(); ++ue )
    {
        if ( bdEdges.test( ue ) )
            continue;
        const auto l = pTp.left( ue ), r = pTp.right( EdgeId( ue ) );
        if ( !l && !r )
            continue; // lone edge
        if ( !l || !r )
            return unexpected( "Incorrect filling" );
        ++numChords;
        chordFaces.set( l );
        chordFaces.set( r );
    }

    HoleFillPlan res;
    res.items.reserve( numChords );
    UndirectedEdgeBitSet emitted( pTp.undirectedEdgeSize() ); // patch edges already present in the plan
    auto clip = [&] ( int p0, EdgeId ne, int code )
    {
        const int p1 = succ[p0];
        cur[p1] = {};
        cur[p0] = ne;
        refCode[p0] = code;
        succ[p0] = succ[p1];
    };
    while ( int( res.items.size() ) < numChords )
    {
        bool progress = false;
        for ( int p0 = 0; p0 < int( cur.size() ) && int( res.items.size() ) < numChords; ++p0 )
        {
            if ( !cur[p0] )
                continue;
            const int p1 = succ[p0];
            const int p2 = succ[p1];
            if ( succ[p2] == p0 )
                continue; // triangle ring: its last face needs no new edge
            const EdgeId ne = pTp.next( cur[p0] );
            if ( pTp.dest( ne ) != pTp.dest( cur[p1] ) )
                continue; // left face of cur[p0] is not an ear here
            if ( emitted.test( ne.undirected() ) )
                continue; // this face is clipped from the position holding ne as its polygon edge
            if ( bdEdges.test( ne.undirected() ) )
                clip( p0, ne, int( meshEdge( ne ) ) ); // pinched ring: the closing edge exists in the mesh
            else
            {
                res.items.push_back( { refCode[p0], refCode[p2] } );
                emitted.set( ne.undirected() );
                clip( p0, ne, -int( res.items.size() ) );
            }
            progress = true;
        }
        if ( progress || int( res.items.size() ) >= numChords )
            continue;

        // no ears left: the rings have to be joined through a bridge edge first. A bridge is expressible
        // only if it is the first new edge after the current polygon edge at both of its ends, and only
        // when the face left of its reversed direction is clipped before it: that clip lands right after
        // the bridge in the shared vertex ring, so it has to be created earlier (see anchoring above)
        bool joined = false;
        for ( int s = 0; s < int( cur.size() ) && !joined; ++s )
        {
            if ( !cur[s] )
                continue;
            const EdgeId m = pTp.next( cur[s] );
            if ( bdEdges.test( m.undirected() ) || emitted.test( m.undirected() ) )
                continue;
            int q = -1;
            for ( int c = 0; c < int( cur.size() ) && q < 0; ++c )
                if ( cur[c] && c != s && pTp.next( cur[c] ) == m.sym() )
                    q = c;
            if ( q < 0 || q == succ[s] || succ[q] == s )
                continue;
            const EdgeId x = pTp.next( m.sym() ); // closes the face left of m.sym() together with cur[s]
            assert( pTp.dest( x ) == pTp.dest( cur[s] ) );
            int codeX;
            if ( bdEdges.test( x.undirected() ) )
                codeX = int( meshEdge( x ) );
            else
            {
                res.items.push_back( { refCode[q], refCode[succ[s]] } );
                emitted.set( x.undirected() );
                codeX = -int( res.items.size() );
            }
            res.items.push_back( { refCode[s], refCode[q] } );
            emitted.set( m.undirected() );
            // join the rings: ... -> s( cur = m ) -> q -> ... -> q's old predecessor -> t( cur = x ) -> old succ[s] -> ...
            const int t = int( cur.size() );
            cur.push_back( x );
            refCode.push_back( codeX );
            succ.push_back( succ[s] );
            for ( int c = 0; c < t; ++c )
                if ( cur[c] && succ[c] == q )
                {
                    succ[c] = t;
                    break;
                }
            cur[s] = m;
            refCode[s] = -int( res.items.size() );
            succ[s] = q;
            joined = true;
        }
        if ( !joined )
            return unexpected( "Incorrect filling" ); // most likely due to ties in input contour
    }

    // an untouched triangle ring got no new edges, so executeHoleFillPlan cannot reach its face:
    // such a hole is reported to be filled with one face separately
    int numSingleFaceHoles = 0;
    for ( int j = 0, base = 0; j < int( patchBds.size() ); base += int( patchBds[j].size() ), ++j )
    {
        if ( patchBds[j].size() != 3 || !cur[base] || !cur[base + 1] || !cur[base + 2] ||
            succ[base] != base + 1 || succ[base + 1] != base + 2 || succ[base + 2] != base )
            continue;
        if ( !pTp.isLeftTri( cur[base] ) )
            return unexpected( "Incorrect filling" );
        ++numSingleFaceHoles;
        if ( !outSingleFaceHoles )
            return unexpected( "Hole needs no new edges to be filled" );
        outSingleFaceHoles->push_back( meshLoops[j].front() );
    }
    if ( int( chordFaces.count() ) + numSingleFaceHoles != pTp.numValidFaces() )
        return unexpected( "Incorrect filling" );
    res.numTris = pTp.numValidFaces() - numSingleFaceHoles;
    return res;
}

Expected<HoleFillPlan> fillContours2DPlan( const Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges,
    std::vector<EdgeId>* outSingleFaceHoles )
{
    assert( !holeRepresentativeEdges.empty() );
    if ( holeRepresentativeEdges.empty() )
        return unexpected( "No hole edges are given" );

    // triangulate the holes in the mesh's own 3d space: only the plan is needed, so the projection
    // round-trip of the mesh-filling path above is avoided
    EdgeLoops loops( holeRepresentativeEdges.size() );
    for ( int i = 0; i < int( holeRepresentativeEdges.size() ); ++i )
    {
        assert( !mesh.topology.left( holeRepresentativeEdges[i] ) );
        if ( mesh.topology.left( holeRepresentativeEdges[i] ) )
            return unexpected( "Hole edge has left face" );
        loops[i] = trackRightBoundaryLoop( mesh.topology, holeRepresentativeEdges[i] );
    }

    // the hole loops wind counterclockwise around this direction, same as around the fitted plane's normal it replaces
    Vector3d sumCross;
    Box3f loopBox;
    float minEdgeLenSq = FLT_MAX;
    for ( const auto& loop : loops )
        for ( EdgeId e : loop )
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

    return patchToPlan( patch->topology, loops, newPaths, outSingleFaceHoles );
}

Expected<HoleFillPlan> fillContours2DPlan( const Mesh& mesh, EdgeId holeEdgeId )
{
    std::vector<EdgeId> singleFaceHoles;
    auto res = fillContours2DPlan( mesh, std::vector<EdgeId>{ holeEdgeId }, &singleFaceHoles );
    if ( res && !singleFaceHoles.empty() )
        res->numTris = 1; // the only hole is a triangle, executeHoleFillPlan fills it by the empty plan
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
