#include "MRSharpenMarchingCubesMesh.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRParallelFor.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRReducePath.h"
#include "MRLineSegm.h"
#include "MRVolumeIndexer.h"
#include "MRBox.h"
#include "MRMeshIntersect.h"
#include "MRMeshProject.h"

namespace MR
{

namespace
{
/// returns the two lattice nodes bounding the lattice edge, on which the vertex (v) of marching cubes mesh is located;
/// valid for original marching cubes vertices only (all faces around a vertex introduced later are in one voxel);
/// returns degenerate segment if the vertex is not surrounded by four voxels
LineSegm3f findLatticeEdgeEnds( const MeshTopology& t, VertId v, const Vector<VoxelId, FaceId>& face2voxel,
    const VolumeIndexer& vi, const AffineXf3f& gridToMeshXf )
{
    // every marching cubes vertex is located on a lattice edge, which is shared by exactly four voxels
    VoxelId v0, v1, v2, v3;
    for ( auto e : orgRing( t, v ) )
    {
        auto f = t.left( e );
        assert( f );
        auto vc = face2voxel[f];
        if ( vc == v0 || vc == v1 || vc == v2 )
            continue;
        if ( !v0 ) v0 = vc;
        else if ( !v1 ) v1 = vc;
        else if ( !v2 ) v2 = vc;
        else if ( !v3 )
        {
            v3 = vc;
            break;
        }
    }
    assert( v0 && v1 && v2 && v3 );
    if ( !v3 )
        return {};

    // the origins of the four voxels are the corners of a unit square orthogonal to the lattice edge
    Box3i box;
    for ( VoxelId vc : { v0, v1, v2, v3 } )
        box.include( vi.toPos( vc ) );

    // the lattice edge is directed along the only axis, where all four voxels have equal coordinate
    const auto sz = box.size();
    int dir = -1, numEqualAxes = 0;
    for ( int i = 0; i < 3; ++i )
    {
        if ( sz[i] != 0 )
            continue;
        dir = i;
        ++numEqualAxes;
    }
    assert( numEqualAxes == 1 ); // otherwise the voxels do not share one lattice edge
    if ( numEqualAxes != 1 )
        return {};

    // and it starts in the node with maximal coordinates among the voxels' origins
    auto n1 = box.max;
    ++n1[dir];
    return { gridToMeshXf( Vector3f( box.max ) ), gridToMeshXf( Vector3f( n1 ) ) };
}

/// moves point (p) along the lattice segment by Newton's method to the location, where the signed distance
/// to the reference mesh is equal to given offset; returns false if the method did not converge
bool moveAlongLatticeEdgeToOffset( const MeshPart& ref, const LineSegm3f& segm, float offset, Vector3f& p, MeshProjectionResult& proj )
{
    constexpr int maxSteps = 3;
    const auto d = segm.dir();
    const auto len = d.length();
    float t = dot( p - segm.a, d ) / sqr( len );
    for ( int i = 0; ; ++i )
    {
        proj = findProjection( p, ref );
        const auto dist = std::sqrt( proj.distSq );
        if ( dist <= 0 )
            return false;
        const auto toP = p - proj.proj.point;
        const float side = dot( toP, ref.mesh.pseudonormal( proj.mtp, ref.region ) ) >= 0 ? 1.0f : -1.0f;
        const auto f = side * dist - offset;
        if ( std::abs( f ) <= 0.01f * len )
            return true;
        const auto df = side * dot( toP, d ) / dist; // derivative of the signed distance along the segment
        if ( i == maxSteps || std::abs( df ) <= 1e-6f * len )
            return false;
        const auto nextT = std::clamp( t - f / df, 0.0f, 1.0f );
        if ( nextT == t )
            return false;
        t = nextT;
        p = segm.a + t * d;
    }
}

/// shortens the displacement from (from) to (to), keeping its direction, until the result is within the voxel's box;
/// (from) must be within the box
Vector3f clampToVoxel( const VolumeIndexer& vi, const AffineXf3f& gridToMeshXf, VoxelId voxel, const Vector3f& from, const Vector3f& to )
{
    const auto pos = vi.toPos( voxel );
    Box3f voxBox;
    voxBox.include( gridToMeshXf( Vector3f( pos ) + Vector3f::diagonal( 1e-3f ) ) );
    voxBox.include( gridToMeshXf( Vector3f( pos ) + Vector3f::diagonal( 1.0f - 1e-3f ) ) );
    const auto shift = to - from;
    float part = 1;
    for ( int i = 0; i < 3; ++i )
    {
        if ( shift[i] > 0 )
            part = std::min( part, ( voxBox.max[i] - from[i] ) / shift[i] );
        else if ( shift[i] < 0 )
            part = std::min( part, ( voxBox.min[i] - from[i] ) / shift[i] );
    }
    return from + std::max( 0.0f, part ) * shift;
}

/// returns the number of faces in the voxel of original marching cubes face (f);
/// marching cubes outputs all faces of a voxel consecutively, and the original faces precede (endFace)
int numVoxelFaces( const Vector<VoxelId, FaceId>& face2voxel, FaceId f, FaceId endFace )
{
    const auto voxel = face2voxel[f];
    FaceId b = f, e = f;
    while ( b > 0 && face2voxel[FaceId( b - 1 )] == voxel )
        --b;
    while ( e + 1 < endFace && face2voxel[FaceId( e + 1 )] == voxel )
        ++e;
    return e - b + 1;
}
}

void sharpenMarchingCubesMesh( const MeshPart& ref, Mesh& vox, Vector<VoxelId, FaceId>& face2voxel,
    const SharpenMarchingCubesMeshSettings& settings )
{
    MR_TIMER;
    assert( settings.minNewVertDev < settings.maxNewRank2VertDev );
    assert( settings.minNewVertDev < settings.maxNewRank3VertDev );
    VertNormals normals( vox.topology.vertSize() );
    VolumeIndexer vi( settings.dims );
    // find normals and correct points
    ParallelFor( normals, [&] ( VertId v )
    {
        if ( !vox.topology.hasVert( v ) )
            return;

        MeshTriPoint rp;
        Vector3f refPt;
        // the vertex is located on the lattice edge, and its position there was found by linear interpolation
        // of the values in the edge's ends; the vertex is corrected only along that edge, so the triangles
        // of every voxel stay within it and cannot intersect the triangles of other voxels
        const auto latticeSegm = findLatticeEdgeEnds( vox.topology, v, face2voxel, vi, settings.gridToMeshXf );
        if ( latticeSegm.lengthSq() > 0 ) // otherwise no lattice edge was found for the vertex
        {
            const auto p = vox.points[v];
            if ( settings.offset == 0 )
            {
                // the crossing of the edge with the reference mesh is the exact point, which the interpolation approximates;
                // the ray starts in the vertex and spans the edge in both directions,
                // so the crossing closest to the vertex is found
                const auto d = latticeSegm.dir();
                const auto rayStart = dot( latticeSegm.a - p, d ) / d.lengthSq();
                if ( const auto isect = rayMeshIntersect( ref, Line3f{ p, d }, rayStart, rayStart + 1 ) )
                {
                    rp = isect.mtp;
                    refPt = ref.mesh.triPoint( rp );
                    // a correction of any length is trustworthy here
                    if ( settings.maxOldVertPosCorrection > 0 )
                        vox.points[v] = refPt;
                }
            }
            else if ( settings.maxOldVertPosCorrection > 0 )
            {
                auto newPos = p;
                MeshProjectionResult proj;
                // a large correction can be wrong and result from self-intersections in the reference mesh
                if ( moveAlongLatticeEdgeToOffset( ref, latticeSegm, settings.offset, newPos, proj )
                    && ( newPos - p ).lengthSq() <= sqr( settings.maxOldVertPosCorrection ) )
                {
                    rp = proj.mtp;
                    refPt = proj.proj.point;
                    vox.points[v] = newPos;
                }
            }
        }
        if ( !rp.valid() )
        {
            const auto proj = findProjection( vox.points[v], ref );
            rp = proj.mtp;
            refPt = proj.proj.point;
        }

        Vector3f n = ( vox.points[v] - refPt ).normalized();
        Vector3f np = ref.mesh.pseudonormal( rp, ref.region );
        if ( settings.offset == 0 || n.lengthSq() <= 0 )
            n = np;
        else if ( dot( n, np ) < 0 )
            n = -n;
        normals[v] = n;
    } );

    auto facesToProcess = vox.topology.getValidFaces();
    VertId firstNewVert( vox.topology.vertSize() );
    const FaceId firstNewFace( vox.topology.faceSize() );
    // line directions and voxels of new vertices, dirs[i] and newVertVoxels[i] contain the data for vertex firstNewVert+i
    std::vector<Vector3f> dirs;
    std::vector<VoxelId> newVertVoxels;
    std::vector<EdgeId> boundary; // the edges with the voxel's surface patch on the left
    for ( auto f : facesToProcess )
    {
        const auto voxel = face2voxel[f];
        EdgeId e0 = vox.topology.edgeWithLeft( f );
        EdgeId e = e0;
        for (;;)
        {
            auto r = vox.topology.right( e );
            if ( !r || face2voxel[r] != voxel )
                break;
            e = vox.topology.prev( e );
            if ( e == e0 )
            {
                assert( false ); //not found after full cycle
                break;
            }
        }
        e0 = e; // an edge with this voxel on the left and another voxel on the right
        Vector3f sumAC;
        float sumArea = 0;
        Vector3f sumDirArea; // area-weighted normal sum of the voxel's faces
        int numPatchFaces = 0;
        PlaneAccumulator pacc;
        boundary.clear();
        do
        {
            boundary.push_back( e );
            auto v = vox.topology.org( e );
            pacc.addPlane( Plane3f::fromDirAndPt( normals[v], vox.points[v] ) );

            auto ei = e;
            for ( ;; )
            {
                auto l = vox.topology.left( ei );
                if ( !l || face2voxel[l] != voxel )
                    break;
                if ( facesToProcess.test_set( l, false ) )
                {
                    ++numPatchFaces;
                    const auto da = vox.dirDblArea( l ); // length = 2x face area
                    const auto a = da.length();
                    sumArea += a;
                    sumAC += a * vox.triCenter( l );
                    sumDirArea += da;
                }
                ei = vox.topology.next( ei );
            }

            e = vox.topology.prev( e.sym() );
            for (;;)
            {
                auto r = vox.topology.right( e );
                if ( !r || face2voxel[r] != voxel )
                    break;
                e = vox.topology.prev( e );
            }
        } while ( e != e0 );

        if ( sumArea <= 0 )
            continue; //degenerate triangles within voxel
        if ( numPatchFaces != numVoxelFaces( face2voxel, f, firstNewFace ) )
            continue; //a new vertex can intersect another surface patch in the same voxel

        Vector3f avgPt = sumAC / sumArea;
        int rank = 0;
        Vector3f dir;
        constexpr float tol = 0.01f; // tolerance for comparing eigenvalues
        auto sharpPt = pacc.findBestCrossPoint( avgPt, tol, &rank, &dir );
        if ( rank <= 1 )
            continue; // the surface is planar within the voxel

        if ( settings.voxelClamp )
        {
            // every marching cubes triangle is located within its own voxel, and the new vertex must not
            // break that, so the geometry of a voxel can never reach another one
            sharpPt = clampToVoxel( vi, settings.gridToMeshXf, voxel, avgPt, sharpPt );
        }

        const auto distSq = ( avgPt - sharpPt ).lengthSq();
        if ( distSq < sqr( settings.minNewVertDev ) )
            continue; //too little deviation of new point to introduce a vertex in mesh
        if ( rank == 2 && distSq > sqr( settings.maxNewRank2VertDev ) )
            continue; //new point is too from existing mesh triangles
        if ( rank == 3 && distSq > sqr( settings.maxNewRank3VertDev ) )
            continue; //new point is too from existing mesh triangles

        if ( !settings.voxelClamp )
        {
            // forbid in-plane shifts: the displacement to sharpPt must rise off the voxel's mean
            // surface plane, not slide along it; an in-plane sharpPt is a phantom feature and a fold source
            constexpr float minElevSin = 0.1f; // ~6 deg minimal angle between the displacement and the plane
            const Vector3f d = sharpPt - avgPt;
            if ( sqr( dot( d, sumDirArea ) ) < sqr( minElevSin ) * d.lengthSq() * sumDirArea.lengthSq() )
                continue; //sharpPt shifts (nearly) within the surface plane
        }

        // the triangles connecting the new vertex with the patch boundary must not fold:
        // find the range of the displacement fraction, where every triangle faces along sumDirArea, and shorten the displacement
        const auto shift = sharpPt - avgPt;
        float minPart = 0, maxPart = FLT_MAX;
        for ( auto be : boundary )
        {
            const auto o = vox.orgPnt( be );
            const auto m = cross( sumDirArea, vox.destPnt( be ) - o );
            const auto a = dot( avgPt - o, m );
            const auto b = dot( shift, m );
            if ( b > 0 )
                minPart = std::max( minPart, -a / b );
            else if ( b < 0 )
                maxPart = std::min( maxPart, a / -b );
            else if ( a <= 0 )
                maxPart = -1;
        }
        if ( maxPart <= minPart || minPart >= 1 )
            continue;
        if ( maxPart < 1 )
        {
            sharpPt = avgPt + ( minPart + 0.9f * ( maxPart - minPart ) ) * shift;
            if ( ( sharpPt - avgPt ).lengthSq() < sqr( settings.minNewVertDev ) )
                continue; //too little deviation of new point to introduce a vertex in mesh
        }

        auto v = vox.splitFace( f, sharpPt );
        assert( v == dirs.size() + firstNewVert );
        dirs.push_back( dir );
        newVertVoxels.push_back( voxel );
        for ( auto ei : orgRing( vox.topology, v ) )
            face2voxel.autoResizeSet( vox.topology.left( ei ), voxel );

        // connect new vertex with every vertex from the voxel
        vox.topology.flipEdgesIn( v, [&]( EdgeId e )
        {
            auto r = vox.topology.right( e );
            assert( r );
            if ( face2voxel[r] == voxel )
            {
                [[maybe_unused]] auto b = vox.topology.dest( vox.topology.prev( e ) );
                assert( !vox.topology.findEdge( v, b ) ); //there is no edge between v and b yet
                return true;
            }
            return false;
        } );

        // make triangles from old voxel vertices if all 3 vertices have similar normals;
        // this reduces self-intersections appeared after previous flip
        vox.topology.flipEdgesOut( v, [&]( EdgeId e )
        {
            assert( vox.topology.org( e ) == v );
            auto b = vox.topology.dest( vox.topology.prev( e ) );
            auto c = vox.topology.dest( e );
            auto d = vox.topology.dest( vox.topology.next( e ) );
            if ( !isUnfoldQuadrangleConvex( vox.points[v], vox.points[b], vox.points[c], vox.points[d] ) )
                return false;
            SymMatrix3f mat;
            mat += outerSquare( normals[b] );
            mat += outerSquare( normals[c] );
            mat += outerSquare( normals[d] );
            const auto eigenvalues = mat.eigens();
            if ( eigenvalues[1] > eigenvalues[2] * tol )
                return false; // normals in the vertices are not equal for given tolerance
            if ( vox.topology.findEdge( d, b ) )
                return false; // multiple edges between b and d will appear
            return true;
        } );
    }

    // find edges between voxels with new vertices
    UndirectedEdgeBitSet sharpEdges( vox.topology.undirectedEdgeSize() );
    for ( auto v = firstNewVert; v < vox.topology.vertSize(); ++v )
    {
        const auto vDir = dirs[ v - firstNewVert ];
        const bool vIsCorner = vDir.lengthSq() < 0.1f; // unit length for edges
        // not-corner vertices can have at most two sharp edges with other new vertices
        struct CanditeEdge
        {
            float metric = 0;
            UndirectedEdgeId edge;
        };
        CanditeEdge best, secondBest; // first maximal metrics
        for ( auto ei : orgRing( vox.topology, v ) )
        {
            if ( !vox.topology.left( ei ) )
                continue;
            EdgeId e = vox.topology.prev( ei.sym() );
            if ( !vox.topology.right( e ) )
                continue;
            auto b = vox.topology.dest( vox.topology.prev( e ) );
            if ( b >= firstNewVert )
            {
                if ( vIsCorner )
                    sharpEdges.set( e.undirected() );
                else
                {
                    const auto bDir = dirs[ b - firstNewVert ];
                    const bool bIsCorner = bDir.lengthSq() < 0.1f; // unit length for edges
                    const auto bvDir = ( vox.points[b] - vox.points[v] ).normalized();
                    // dot( vDir, bDir ) worked bad for cone vertex
                    const auto metric = bIsCorner ? 10.0f : std::abs( dot( vDir, bvDir ) );
                    if ( metric > 0.5f ) // avoid connection with vertex not along v-line
                    {
                        CanditeEdge c{ .metric = metric, .edge = e.undirected() };
                        if ( c.metric > best.metric )
                        {
                            secondBest = best;
                            best = c;
                        }
                        else if ( c.metric > secondBest.metric )
                        {
                            secondBest = c;
                        }
                    }
                }
            }
        }
        if ( best.edge )
            sharpEdges.set( best.edge );
        if ( secondBest.edge )
            sharpEdges.set( secondBest.edge );
    }

    // flip edges between voxels with new vertices to form sharp ridges
    for ( EdgeId e : sharpEdges )
    {
        auto b = vox.topology.dest( vox.topology.prev( e ) );
        auto d = vox.topology.dest( vox.topology.next( e ) );
        if ( !vox.topology.findEdge( b, d ) )
            vox.topology.flipEdge( e );
    }

    // best position new vertices on found lines
    std::vector<Vector3f> newPos;
    newPos.reserve( vox.topology.vertSize() - firstNewVert );
    for ( int iPosSel = 0; iPosSel < settings.posSelIters; ++iPosSel )
    {
        // calculate optimal position of each vertex independently
        newPos.clear();
        for ( auto iv = firstNewVert; iv < vox.topology.vertSize(); ++iv )
        {
            const auto p = vox.points[iv];
            newPos.push_back( p );
            const auto vDir = dirs[ iv - firstNewVert ];
            const bool vIsCorner = vDir.lengthSq() < 0.1f; // unit length for edges
            if ( vIsCorner )
                continue;
            float uv = 0, vv = 0;
            for ( auto ei : orgRing( vox.topology, iv ) )
            {
                if ( !vox.topology.left( ei ) )
                    continue;
                auto ap = vox.destPnt( ei );
                auto bp = vox.destPnt( vox.topology.next( ei ) );
                auto u = cross( bp - ap, p - ap );
                auto v = cross( bp - ap, vDir );
                uv += dot( u, v );
                vv += dot( v, v );
            }
            if ( vv > 0 )
                newPos.back() -= uv / vv * vDir;
        }

        // move each vertex half way toward its optimal position
        for ( auto v = firstNewVert; v < vox.topology.vertSize(); ++v )
        {
            const auto pOld = vox.points[v];
            const auto pNew = 0.5f * ( pOld + newPos[ v - firstNewVert ] );
            vox.points[v] = settings.voxelClamp ? clampToVoxel( vi, settings.gridToMeshXf, newVertVoxels[ v - firstNewVert ], pOld, pNew ) : pNew;
        }
    }

    if ( settings.outSharpEdges )
        *settings.outSharpEdges = std::move( sharpEdges );
}

} //namespace MR
