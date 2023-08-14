#include "MRMeshIntersect.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRRayBoxIntersection.h"
#include "MRTriangleIntersection.h"
#include "MRPrecisePredicates3.h"
#include "MRVector3.h"
#include "MRLine3.h"
#include "MRUVSphere.h"
#include "MRGTest.h"
#include "MRMeshLoad.h"
#include "MRMeshBuilder.h"
#include "MRMeshSave.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

template<typename T>
std::optional<MeshIntersectionResult> meshRayIntersect_( const MeshPart& meshPart, const Line3<T>& line,
    T rayStart, T rayEnd, const IntersectionPrecomputes<T>& prec, bool closestIntersect, const FacePredicate & validFaces )
{
    const auto& m = meshPart.mesh;
    constexpr int maxTreeDepth = 32;
    const auto& tree = m.getAABBTree();
    if( tree.nodes().size() == 0 )
        return std::nullopt;

    RayOrigin<T> rayOrigin{ line.p };
    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( Box3<T>{ tree[tree.rootNodeId()].box }, rayOrigin, s, e, prec ) )
    {
        return std::nullopt;
    }

    std::pair< AABBTree::NodeId,T> nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = { tree.rootNodeId(), rayStart };

    FaceId faceId;
    TriPointf triP;
    while( currentNode >= 0 && ( closestIntersect || !faceId ) )
    {
        if( currentNode >= maxTreeDepth ) // max depth exceeded
        {
            spdlog::critical( "Maximal AABBTree depth reached!" );
            assert( false );
            break;
        }

        const auto& node = tree[nodesStack[currentNode].first];
        if( nodesStack[currentNode--].second < rayEnd )
        {
            if( node.leaf() )
            {
                auto face = node.leafId();
                if( ( !meshPart.region || meshPart.region->test( face ) ) && ( !validFaces || validFaces( face ) ) )
                {
                    VertId a, b, c;
                    m.topology.getTriVerts( face, a, b, c );

                    const Vector3<T> vA = Vector3<T>( m.points[a] ) - line.p;
                    const Vector3<T> vB = Vector3<T>( m.points[b] ) - line.p;
                    const Vector3<T> vC = Vector3<T>( m.points[c] ) - line.p;
                    if ( auto triIsect = rayTriangleIntersect( vA, vB, vC, prec ) )
                    {
                        if ( triIsect->t < rayEnd && triIsect->t > rayStart )
                        {
                            faceId = face;
                            triP = triIsect->bary;
                            rayEnd = triIsect->t;
                        }
                    }
                }
            }
            else
            {
                T lStart = rayStart, lEnd = rayEnd;
                T rStart = rayStart, rEnd = rayEnd;
                if( rayBoxIntersect( Box3<T>{ tree[node.l].box }, rayOrigin, lStart, lEnd, prec ) )
                {
                    if( rayBoxIntersect( Box3<T>{ tree[node.r].box }, rayOrigin, rStart, rEnd, prec ) )
                    {
                        if( lStart > rStart )
                        {
                            nodesStack[++currentNode] = { node.l,lStart };
                            nodesStack[++currentNode] = { node.r,rStart };
                        }
                        else
                        {
                            nodesStack[++currentNode] = { node.r,rStart };
                            nodesStack[++currentNode] = { node.l,lStart };
                        }
                    }
                    else
                    {
                        nodesStack[++currentNode] = { node.l,lStart };
                    }
                }
                else
                {
                    if( rayBoxIntersect( Box3<T>{ tree[node.r].box }, rayOrigin, rStart, rEnd, prec ) )
                    {
                        nodesStack[++currentNode] = { node.r,rStart };
                    }
                }
            }
        }
    }

    if( faceId.valid() )
    {
        MeshIntersectionResult res;
        res.proj.face = faceId;
        res.proj.point = Vector3f( line.p + rayEnd * line.d );
        res.mtp = MeshTriPoint( m.topology.edgeWithLeft( faceId ), triP );
        res.distanceAlongLine = float( rayEnd );
        return res;
    }
    else
    {
        return std::nullopt;
    }
}

std::optional<MeshIntersectionResult> rayMeshIntersect( const MeshPart& meshPart, const Line3f& line,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec, bool closestIntersect, const FacePredicate & validFaces )
{
    if( prec )
    {
        return meshRayIntersect_<float>( meshPart, line, rayStart, rayEnd, *prec, closestIntersect, validFaces );
    }
    else
    {
        const IntersectionPrecomputes<float> precNew( line.d );
        return meshRayIntersect_<float>( meshPart, line, rayStart, rayEnd, precNew, closestIntersect, validFaces );
    }
}

std::optional<MeshIntersectionResult> rayMeshIntersect( const MeshPart& meshPart, const Line3d& line,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec, bool closestIntersect, const FacePredicate & validFaces )
{
    if( prec )
    {
        return meshRayIntersect_<double>( meshPart, line, rayStart, rayEnd, *prec, closestIntersect, validFaces );
    }
    else
    {
        const IntersectionPrecomputes<double> precNew( line.d );
        return meshRayIntersect_<double>( meshPart, line, rayStart, rayEnd, precNew, closestIntersect, validFaces );
    }
}

template<typename T>
std::optional<MultiMeshIntersectionResult> rayMultiMeshAnyIntersect_( const std::vector<Line3Mesh<T>> & lineMeshes,
    T rayStart /*= 0.0f*/, T rayEnd /*= FLT_MAX */ )
{
    std::optional<MultiMeshIntersectionResult> res;
    for ( const auto & lm : lineMeshes )
    {
        assert( lm.mesh );
        auto prec = lm.prec;
        IntersectionPrecomputes<T> myPrec;
        if ( !prec )
        {
            myPrec = { lm.line.d };
            prec = &myPrec;
        }
        if ( auto r = meshRayIntersect_( { *lm.mesh, lm.region }, lm.line, rayStart, rayEnd, *prec, false, {} ) )
        {
            res = MultiMeshIntersectionResult{ *r };
            break;
        }
    }
    return res;
}

std::optional<MultiMeshIntersectionResult> rayMultiMeshAnyIntersect( const std::vector<Line3fMesh> & lineMeshes,
    float rayStart, float rayEnd )
{
    return rayMultiMeshAnyIntersect_( lineMeshes, rayStart, rayEnd );
}

std::optional<MultiMeshIntersectionResult> rayMultiMeshAnyIntersect( const std::vector<Line3dMesh> & lineMeshes,
    double rayStart, double rayEnd )
{
    return rayMultiMeshAnyIntersect_( lineMeshes, rayStart, rayEnd );
}

template<typename T>
void rayMeshIntersectAll_( const MeshPart& meshPart, const Line3<T>& line, MeshIntersectionCallback callback,
    T rayStart /*= 0.0f*/, T rayEnd /*= FLT_MAX */, const IntersectionPrecomputes<T>& prec )
{
    assert( callback );
    if ( !callback )
        return;

    const auto& m = meshPart.mesh;
    constexpr int maxTreeDepth = 32;
    const auto& tree = m.getAABBTree();
    if( tree.nodes().size() == 0 )
        return;

    RayOrigin<T> rayOrigin{ line.p };
    T s = rayStart, e = rayEnd;
    if( !rayBoxIntersect( Box3<T>{ tree[tree.rootNodeId()].box }, rayOrigin, s, e, prec ) )
    {
        return;
    }

    AABBTree::NodeId nodesStack[maxTreeDepth];
    int currentNode = 0;
    nodesStack[0] = tree.rootNodeId();
    ConvertToIntVector convToInt;
    ConvertToFloatVector convToFloat;
    Vector3f dP, eP;
    std::array<PreciseVertCoords, 5> pvc;

    if constexpr ( std::is_same_v<T, double> )
    {
        convToInt = getToIntConverter( Box3d( tree[tree.rootNodeId()].box ) );
        convToFloat = getToFloatConverter( Box3d( tree[tree.rootNodeId()].box ) );
        dP = Vector3f( line( s ) );
        eP = Vector3f( line( e ) );
        pvc[3].pt = convToInt( dP );
        pvc[3].id = VertId( m.topology.vertSize() );
        pvc[4].pt = convToInt( eP );
        pvc[4].id = pvc[3].id + 1;
    }

    while( currentNode >= 0 )
    {
        if( currentNode >= maxTreeDepth ) // max depth exceeded
        {
            spdlog::critical( "Maximal AABBTree depth reached!" );
            assert( false );
            break;
        }

        const auto& node = tree[nodesStack[currentNode--]];
        if( node.leaf() )
        {
            auto face = node.leafId();
            if( !meshPart.region || meshPart.region->test( face ) )
            {
                m.topology.getTriVerts( face, pvc[0].id, pvc[1].id, pvc[2].id );
                if constexpr ( std::is_same_v<T, double> )
                {
                    // double version (more precise)
                    for ( int i = 0; i < 3; ++i )
                        pvc[i].pt = convToInt( m.points[pvc[i].id] );
                    if ( doTriangleSegmentIntersect( pvc ) )
                    {
                        MeshIntersectionResult found;
                        found.proj.face = face;
                        found.proj.point = findTriangleSegmentIntersectionPrecise( m.points[pvc[0].id], m.points[pvc[1].id], m.points[pvc[2].id], dP, eP, { convToInt,convToFloat } );
                        found.distanceAlongLine = dot( found.proj.point - Vector3f( line.p ), Vector3f( line.d ) );
                        if ( found.distanceAlongLine < rayEnd && found.distanceAlongLine > rayStart )
                        {
                            found.mtp = MeshTriPoint( m.topology.edgeWithLeft( face ), TriPointf( found.proj.point, m.points[pvc[0].id], m.points[pvc[1].id], m.points[pvc[2].id] ) );
                            if ( !callback( found ) )
                                return;
                        }
                    }
                }
                else
                {
                    // float version (faster)
                    const Vector3<T> vA = Vector3<T>( m.points[pvc[0].id] ) - line.p;
                    const Vector3<T> vB = Vector3<T>( m.points[pvc[1].id] ) - line.p;
                    const Vector3<T> vC = Vector3<T>( m.points[pvc[2].id] ) - line.p;
                    const auto triIsect = rayTriangleIntersect( vA, vB, vC, prec );
                    if ( triIsect && triIsect->t < rayEnd && triIsect->t > rayStart )
                    {
                        MeshIntersectionResult found;
                        found.proj.face = face;
                        found.proj.point = Vector3f( line( triIsect->t ) );
                        found.mtp = MeshTriPoint( m.topology.edgeWithLeft( face ), triIsect->bary );
                        found.distanceAlongLine = float( triIsect->t );
                        if ( !callback( found ) )
                            return;
                    }
                }
            }
        }
        else
        {
            s = rayStart, e = rayEnd;
            if( rayBoxIntersect( Box3<T>{ tree[node.l].box }, rayOrigin, s, e, prec ) )
            {
                nodesStack[++currentNode] = node.l;
            }
            s = rayStart, e = rayEnd;
            if( rayBoxIntersect( Box3<T>{ tree[node.r].box }, rayOrigin, s, e, prec ) )
            {
                nodesStack[++currentNode] = node.r;
            }
        }
    }
}

void rayMeshIntersectAll( const MeshPart& meshPart, const Line3f& line, MeshIntersectionCallback callback,
    float rayStart, float rayEnd, const IntersectionPrecomputes<float>* prec )
{
    if( prec )
    {
        return rayMeshIntersectAll_<float>( meshPart, line, callback, rayStart, rayEnd, *prec );
    }
    else
    {
        const IntersectionPrecomputes<float> precNew( line.d );
        return rayMeshIntersectAll_<float>( meshPart, line, callback, rayStart, rayEnd, precNew );
    }
}

void rayMeshIntersectAll( const MeshPart& meshPart, const Line3d& line, MeshIntersectionCallback callback,
    double rayStart, double rayEnd, const IntersectionPrecomputes<double>* prec )
{
    if( prec )
    {
        return rayMeshIntersectAll_<double>( meshPart, line, callback, rayStart, rayEnd, *prec );
    }
    else
    {
        const IntersectionPrecomputes<double> precNew( line.d );
        return rayMeshIntersectAll_<double>( meshPart, line, callback, rayStart, rayEnd, precNew );
    }
}

void xyPlaneMeshIntersect( const MeshPart& meshPart, float zLevel,
    FaceBitSet * fs, UndirectedEdgeBitSet * ues, VertBitSet * vs )
{
    assert( fs || ues || vs );

    const auto& m = meshPart.mesh;
    constexpr int maxTreeDepth = 32;
    const auto& tree = m.getAABBTree();
    if( tree.nodes().size() == 0 )
        return;

    assert( !fs  || fs->size()  >= m.topology.faceSize() );
    assert( !ues || ues->size() >= m.topology.undirectedEdgeSize() );
    assert( !vs  || vs->size()  >= m.topology.vertSize() );

    AABBTree::NodeId nodesStack[maxTreeDepth];
    int currentNode = -1;

    auto addNode = [&]( AABBTree::NodeId nid )
    {
        const auto & box = tree[nid].box;
        if ( box.min.z <= zLevel && box.max.z >= zLevel )
            nodesStack[++currentNode] = nid;
    };
    addNode( tree.rootNodeId() );

    while( currentNode >= 0 )
    {
        if( currentNode >= maxTreeDepth ) // max depth exceeded
        {
            spdlog::critical( "Maximal AABBTree depth reached!" );
            assert( false );
            break;
        }

        const auto& node = tree[nodesStack[currentNode--]];
        if( node.leaf() )
        {
            auto face = node.leafId();
            if( !meshPart.region || meshPart.region->test( face ) )
            {
                if ( fs )
                    fs->set( face );
                if ( ues || vs )
                {
                    EdgeId e0, e1, e2;
                    m.topology.getTriEdges( face, e0, e1, e2 );
                    if ( ues )
                    {
                        ues->set( e0 );
                        ues->set( e1 );
                        ues->set( e2 );
                    }
                    if ( vs )
                    {
                        vs->set( m.topology.org( e0 ) );
                        vs->set( m.topology.org( e1 ) );
                        vs->set( m.topology.org( e2 ) );
                    }
                }
            }
            continue;
        }
        addNode( node.r ); // push first to go there later
        addNode( node.l );
    }
}

std::optional<MeshIntersectionResult> rayInsideIntersect( const Mesh& mesh, VertId v )
{
    const auto dir = -mesh.pseudonormal( v );
    return rayMeshIntersect( mesh, { mesh.points[v], dir }, 0.0f, FLT_MAX, nullptr, true,
        [v, &top = mesh.topology]( FaceId f )
        {
            // ignore intersections with incident faces of (v)
            VertId a, b, c;
            top.getTriVerts( f, a, b, c );
            return v != a && v != b && v != c;
        } );
}

VertScalars computeThicknessAtVertices( const Mesh& mesh )
{
    MR_TIMER
    VertScalars res( mesh.points.size(), FLT_MAX );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        auto isec = rayInsideIntersect( mesh, v );
        if ( isec )
            res[v] = isec->distanceAlongLine;
    } );
    return res;
}

TEST(MRMesh, MeshIntersect) 
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );

    std::vector<MeshIntersectionResult> allFound;
    auto callback = [&allFound]( const MeshIntersectionResult & found ) -> bool
    {
        allFound.push_back( found );
        return true;
    };

    Vector3f d{ 1, 2, 3 };
    rayMeshIntersectAll( sphere, { 2.0f * d, -d.normalized() }, callback );
    ASSERT_EQ( allFound.size(), 2 );
    for ( const auto & found : allFound )
    {
        ASSERT_NEAR( found.proj.point.length(), 1.0f, 0.05f ); //our sphere is very approximate
    }
}

} //namespace MR
