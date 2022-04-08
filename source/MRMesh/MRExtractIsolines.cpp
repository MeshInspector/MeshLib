#include "MRExtractIsolines.h"
#include "MRBitSet.h"
#include "MREdgeIterator.h"
#include "MRMeshEdgePoint.h"
#include "MRPlane3.h"
#include "MRMesh.h"
#include "MRAffineXf3.h"
#include "MRVector2.h"
#include "MRTimer.h"

namespace MR
{

using ValueInVertex = std::function<float(VertId)>;

class Isoliner
{
public:
    Isoliner( const MeshTopology & topology, ValueInVertex valueInVertex, const FaceBitSet * region )
        : topology_( topology ), region_( region ), valueInVertex_( valueInVertex ) { }

    std::vector<std::vector<MeshEdgePoint>> extract();

private:
    std::vector<MeshEdgePoint> extractOneLine_( const MeshEdgePoint & first );
    MeshEdgePoint toEdgePoint_( EdgeId e, float vo, float vd ) const;
    std::optional<MeshEdgePoint> findNextEdgePoint_( EdgeId e ) const;

private:
    const MeshTopology & topology_;
    const FaceBitSet * region_ = nullptr;
    ValueInVertex valueInVertex_;
    UndirectedEdgeBitSet seenEdges_;
};

std::vector<std::vector<MeshEdgePoint>> Isoliner::extract()
{
    std::vector<std::vector<MeshEdgePoint>> res;
    for ( auto ue : undirectedEdges( topology_ ) )
    {
        if ( region_ && !contains( *region_, topology_.left( ue ) ) && !contains( *region_, topology_.right( ue ) ) )
            continue;
        if ( seenEdges_.test( ue ) )
            continue;
        EdgeId e = ue;
        VertId o = topology_.org( e );
        VertId d = topology_.dest( e );
        float vo = valueInVertex_( o );
        float vd = valueInVertex_( d );
        if ( vo < 0 && 0 <= vd )
            res.push_back( extractOneLine_( toEdgePoint_( e, vo, vd ) ) );
        else if ( vd < 0 && 0 <= vo )
            res.push_back( extractOneLine_( toEdgePoint_( e.sym(), vd, vo ) ) );
    }
    return res;
}

inline MeshEdgePoint Isoliner::toEdgePoint_( EdgeId e, float vo, float vd ) const
{
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    const float x = vo / ( vo - vd );
    return MeshEdgePoint( e, x );
}

std::optional<MeshEdgePoint> Isoliner::findNextEdgePoint_( EdgeId e ) const
{
    if ( !topology_.isLeftInRegion( e, region_ ) )
        return {};
    VertId o, d, x;
    topology_.getLeftTriVerts( e, o, d, x );
    const float vo = valueInVertex_( o );
    const float vd = valueInVertex_( d );
    const float vx = valueInVertex_( x );
    assert( ( vo < 0 && 0 <= vd ) || ( vd < 0 && 0 <= vo ) );
    if ( ( vo < 0 && vx < 0 ) || ( vd < 0 && vx >= 0 ) )
        return toEdgePoint_( topology_.prev( e.sym() ).sym(), vx, vd );
    else
        return toEdgePoint_( topology_.next( e ), vo, vx );
}

std::vector<MeshEdgePoint> Isoliner::extractOneLine_( const MeshEdgePoint & first )
{
    std::vector<MeshEdgePoint> res;
    res.push_back( first );
    seenEdges_.autoResizeSet( first.e.undirected() );

    bool closed = false;
    while ( auto next = findNextEdgePoint_( res.back().e ) )
    {
        if ( first.e == next->e )
        {
            res.push_back( first );
            closed = true;
            break;
        }
        res.push_back( *next );
        seenEdges_.autoResizeSet( next->e.undirected() );
    }

    if ( !closed )
    {
        auto firstSym = first;
        firstSym = firstSym.sym(); // go backward
        std::vector<MeshEdgePoint> back;
        back.push_back( firstSym );
        while ( auto next = findNextEdgePoint_( back.back().e ) )
        {
            back.push_back( *next );
            seenEdges_.autoResizeSet( next->e.undirected() );
        }
        std::reverse( back.begin(), back.end() );
        back.pop_back(); // remove extra copy of firstSym
        for ( auto & i : back )
            i = i.sym(); // make consistent edge orientations of forward and backward passes
        res.insert( res.begin(), back.begin(), back.end() );
    }

    return res;
}

std::vector<std::vector<MeshEdgePoint>> extractIsolines( const MeshTopology & topology, 
    const Vector<float,VertId> & vertValues, float isoValue, const FaceBitSet * region )
{
    MR_TIMER;

    Isoliner s( topology, [&]( VertId v ) { return vertValues[v] - isoValue; }, region );
    return s.extract();
}

std::vector<std::vector<MeshEdgePoint>> extractPlaneSections( const MeshPart & mp, const Plane3f & plane )
{
    MR_TIMER;

    Isoliner s( mp.mesh.topology, [&]( VertId v ) { return plane.distance( mp.mesh.points[v] ); }, mp.region );
    return s.extract();
}

Contour2f planeSectionToContour2f( const Mesh & mesh, const PlaneSection & section, const AffineXf3f & meshToPlane )
{
    MR_TIMER;
    Contour2f res;
    res.reserve( section.size() );
    for ( const auto & s : section )
    {
        auto p = meshToPlane( mesh.edgePoint( s ) );
        res.emplace_back( p.x, p.y );
    }

    return res;
}

Contours2f planeSectionsToContours2f( const Mesh & mesh, const PlaneSections & sections, const AffineXf3f & meshToPlane )
{
    MR_TIMER;
    Contours2f res;
    res.reserve( sections.size() );
    for ( const auto & s : sections )
        res.push_back( planeSectionToContour2f( mesh, s, meshToPlane ) );
    return res;
}

} //namespace MR
