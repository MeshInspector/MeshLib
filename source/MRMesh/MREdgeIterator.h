#pragma once

#include "MRMeshTopology.h"
#include "MRIteratorRange.h"
#include <iterator>

namespace MR
{
 
// The iterator to find all not-lone undirected edges in the mesh
class UndirectedEdgeIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = UndirectedEdgeId;

    // creates begin iterator
    UndirectedEdgeIterator( const MeshTopology & topology ) : topology_( &topology )
    {
        if ( topology_->undirectedEdgeSize() == 0 )
            return; // end has reached
        edge_ = UndirectedEdgeId{0};
        if ( topology_->isLoneEdge( edge_ ) )
            operator ++();
    }
    // creates end iterator
    UndirectedEdgeIterator() = default;

    UndirectedEdgeIterator & operator++( )
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
    const MeshTopology * topology_ = nullptr;
    UndirectedEdgeId edge_;
};

inline bool operator ==( const UndirectedEdgeIterator & a, const UndirectedEdgeIterator & b )
    { return *a == *b; }

inline bool operator !=( const UndirectedEdgeIterator & a, const UndirectedEdgeIterator & b )
    { return *a != *b; }

inline IteratorRange<UndirectedEdgeIterator> undirectedEdges( const MeshTopology & topology )
    { return { UndirectedEdgeIterator( topology ), UndirectedEdgeIterator() }; }

} //namespace MR
