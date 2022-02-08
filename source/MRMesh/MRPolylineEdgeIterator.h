#pragma once

#include "MRPolylineTopology.h"
#include "MRIteratorRange.h"
#include <iterator>

namespace MR
{
 
// The iterator to find all not-lone undirected edges in the polyline topology
class PolylineUndirectedEdgeIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = UndirectedEdgeId;

    // creates begin iterator
    PolylineUndirectedEdgeIterator( const PolylineTopology & topology ) : topology_( &topology )
    {
        if ( topology_->undirectedEdgeSize() == 0 )
            return; // end has reached
        edge_ = UndirectedEdgeId{0};
        if ( topology_->isLoneEdge( edge_ ) )
            operator ++();
    }
    // creates end iterator
    PolylineUndirectedEdgeIterator() = default;

    PolylineUndirectedEdgeIterator & operator++( )
    {
        assert( edge_.valid() );
        for (;;)
        {
            ++edge_;
            if ( edge_ >= topology_->undirectedEdgeSize() )
            {
                edge_ = UndirectedEdgeId{};
                break;
            }
            if ( !topology_->isLoneEdge( edge_ ) )
                break;
        }
        return * this;
    }
    UndirectedEdgeId operator *( ) const { return edge_; }

private:
    const PolylineTopology * topology_ = nullptr;
    UndirectedEdgeId edge_;
};

inline bool operator ==( const PolylineUndirectedEdgeIterator & a, const PolylineUndirectedEdgeIterator & b )
    { return *a == *b; }

inline bool operator !=( const PolylineUndirectedEdgeIterator & a, const PolylineUndirectedEdgeIterator & b )
    { return *a != *b; }

inline IteratorRange<PolylineUndirectedEdgeIterator> undirectedEdges( const PolylineTopology & topology )
    { return { PolylineUndirectedEdgeIterator( topology ), PolylineUndirectedEdgeIterator() }; }

} //namespace MR
