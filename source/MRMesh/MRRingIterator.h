#pragma once

#include "MRMeshTopology.h"
#include "MRIteratorRange.h"
#include <iterator>

namespace MR
{

// The iterator to find all edges in a ring of edges (e.g. all edges with same origin or all edges with same left face)
template <typename N>
class RingIterator : public N
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = EdgeId;
    using difference_type   = std::ptrdiff_t;

    RingIterator( const MeshTopology & topology, EdgeId edge, bool first )
        : N( topology ), edge_( edge ), first_( first )
    {
    }
    RingIterator & operator++( )
    {
        first_ = false;
        edge_ = N::next( edge_ );
        return * this;
    }
    EdgeId operator *( ) const { return edge_; }
    bool first() const { return first_; }

private:
    EdgeId edge_;
    bool first_ = false;
};

template <typename N>
inline bool operator ==( const RingIterator<N> & a, const RingIterator<N> & b )
    { return *a == *b && a.first() == b.first(); }

template <typename N>
inline bool operator !=( const RingIterator<N> & a, const RingIterator<N> & b )
    { return *a != *b || a.first() != b.first(); }

class NextEdgeSameOrigin
{
    const MeshTopology * topology_ = nullptr;

public:
    NextEdgeSameOrigin( const MeshTopology & topology ) : topology_( &topology ) { }
    EdgeId next( EdgeId e ) const { return topology_->next( e ); }
};

using OrgRingIterator = RingIterator<NextEdgeSameOrigin>;

class NextEdgeSameLeft
{
    const MeshTopology * topology_ = nullptr;

public:
    NextEdgeSameLeft( const MeshTopology & topology ) : topology_( &topology ) { }
    EdgeId next( EdgeId e ) const { return topology_->prev( e.sym() ); }
};

using LeftRingIterator = RingIterator<NextEdgeSameLeft>;


// to iterate over all edges with same origin vertex  as firstEdge (INCLUDING firstEdge)
// for ( Edge e : orgRing( topology, firstEdge ) ) ...
inline IteratorRange<OrgRingIterator> orgRing( const MeshTopology & topology, EdgeId edge )
    { return { OrgRingIterator( topology, edge, edge.valid() ), OrgRingIterator( topology, edge, false ) }; }
inline IteratorRange<OrgRingIterator> orgRing( const MeshTopology & topology, VertId v )
    { return orgRing( topology, topology.edgeWithOrg( v ) ); }

// to iterate over all edges with same origin vertex as firstEdge (EXCLUDING firstEdge)
// for ( Edge e : orgRing0( topology, firstEdge ) ) ...
inline IteratorRange<OrgRingIterator> orgRing0( const MeshTopology & topology, EdgeId edge )
    { return { ++OrgRingIterator( topology, edge, true ), OrgRingIterator( topology, edge, false ) }; }


// to iterate over all edges with same left face as firstEdge (INCLUDING firstEdge)
// for ( Edge e : leftRing( topology, firstEdge ) ) ...
inline IteratorRange<LeftRingIterator> leftRing( const MeshTopology & topology, EdgeId edge )
    { return { LeftRingIterator( topology, edge, edge.valid() ), LeftRingIterator( topology, edge, false ) }; }
inline IteratorRange<LeftRingIterator> leftRing( const MeshTopology & topology, FaceId f )
    { return leftRing( topology, topology.edgeWithLeft( f ) ); }

// to iterate over all edges with same left face as firstEdge (EXCLUDING firstEdge)
// for ( Edge e : leftRing0( topology, firstEdge ) ) ...
inline IteratorRange<LeftRingIterator> leftRing0( const MeshTopology & topology, EdgeId edge )
    { return { ++LeftRingIterator( topology, edge, true ), LeftRingIterator( topology, edge, false ) }; }

} //namespace MR
