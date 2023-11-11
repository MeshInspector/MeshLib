#include "MRIdentifyVertices.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

namespace MeshBuilder
{

void VertexIdentifier::reserve( size_t numTris )
{
    hmap_.reserve( numTris / 2 ); // there should be about twice more triangles than vertices
    t_.reserve( numTris );
}

void VertexIdentifier::addTriangles( const std::vector<Triangle3f> & buffer )
{
    MR_TIMER
    assert ( t_.size() + buffer.size() <= t_.capacity() );
    vertsInHMap_.resize( buffer.size() );

    for (;;)
    {
        auto buckets0 = hmap_.bucket_count();

        const auto subcnt = hmap_.subcnt();
        ParallelFor( size_t( 0 ), subcnt, [&]( size_t myPartId )
        {
            for ( size_t j = 0; j < buffer.size(); ++j )
            {
                const auto & st = buffer[j];
                auto & it = vertsInHMap_[j];
                for ( int k = 0; k < 3; ++k )
                {
                    const auto & p = st[k];
                    auto hashval = hmap_.hash( p );
                    auto idx = hmap_.subidx( hashval );
                    if ( idx != myPartId )
                        continue;
                    it[k] = &hmap_[ p ];
                }
            }
        } );

        if ( buckets0 == hmap_.bucket_count() )
            break; // the number of buckets has not changed - all pointers are valid
    }

    for ( size_t j = 0; j < buffer.size(); ++j )
    {
        const auto & st = buffer[j];
        const auto & it = vertsInHMap_[j];
        for ( int k = 0; k < 3; ++k )
        {
            if ( !it[k]->valid() )
            {
                *it[k] = VertId( (int)points_.size() );
                points_.push_back( st[k] );
            }
        }
        t_.push_back( { *it[0], *it[1], *it[2] } );
    }
}

} //namespace MeshBuilder

} //namespace MR
