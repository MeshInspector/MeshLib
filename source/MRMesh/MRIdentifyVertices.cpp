#include "MRIdentifyVertices.h"
#include "MRphmap.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace MeshBuilder
{

void VertexIdentifier::reserve( size_t numTris )
{
    hmap_.reserve( numTris / 2 ); // there should be about twice more triangles than vertices
    tris_.reserve( numTris );
}

void VertexIdentifier::addTriangles( const std::vector<ThreePoints> & buffer )
{
    assert ( tris_.size() + buffer.size() <= tris_.capacity() );
    vertsInHMap_.resize( buffer.size() );

    for (;;)
    {
        auto buckets0 = hmap_.bucket_count();

        const auto subcnt = hmap_.subcnt();
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ), [&]( const tbb::blocked_range<size_t> & range )
        {
            assert( range.begin() + 1 == range.end() );
            for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
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
        tris_.emplace_back( *it[0], *it[1], *it[2], FaceId( int( tris_.size() ) ) );
    }
}

} //namespace MeshBuilder

} //namespace MR
