#include "MRWasmBindings.h"

#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

#include <cstdint>
#include <vector>

using namespace MR;

namespace
{

emscripten::val bitSetToIndices( const BitSet& bs )
{
    std::vector<uint32_t> idx;
    idx.reserve( bs.count() );
    for ( size_t i = bs.find_first(); i != BitSet::npos; i = bs.find_next( i ) )
        idx.push_back( static_cast<uint32_t>( i ) );
    return Wasm::makeTypedArray<uint32_t>( idx.data(), idx.size() );
}

template <typename BS>
BS bitSetFromIndices( emscripten::val arr )
{
    const size_t len = arr["length"].as<size_t>();
    std::vector<uint32_t> idx( len );
    if ( len != 0 )
    {
        emscripten::val view = emscripten::val( emscripten::typed_memory_view( len, idx.data() ) );
        view.call<void>( "set", arr );
    }
    BS bs;
    for ( uint32_t i : idx )
        bs.autoResizeSet( typename BS::IndexType( i ) );
    return bs;
}

template <typename BS>
void registerTypedBitSet( const char* name )
{
    emscripten::class_<BS, emscripten::base<BitSet>>( name )
        .template constructor<>()
        .class_function( "fromIndices", &bitSetFromIndices<BS> );
}

}

EMSCRIPTEN_BINDINGS( meshlib_bitset )
{
    emscripten::class_<BitSet>( "BitSet" )
        .constructor<>()
        .function( "size", &BitSet::size )
        .function( "count", &BitSet::count )
        .function( "empty", &BitSet::empty )
        .function( "test", &BitSet::test )
        .function( "set", +[]( BitSet& bs, int n, bool val ) { bs.set( BitSet::IndexType( n ), val ); } )
        .function( "resize", +[]( BitSet& bs, int numBits, bool fillValue ) { bs.resize( BitSet::size_type( numBits ), fillValue ); } )
        .function( "find_first", +[]( const BitSet& bs ) { return (int)bs.find_first(); } )
        .function( "find_last", +[]( const BitSet& bs ) { return (int)bs.find_last(); } )
        .function( "toIndices", &bitSetToIndices )
        .class_function( "fromIndices", &bitSetFromIndices<BitSet> );

    registerTypedBitSet<FaceBitSet>( "FaceBitSet" );
    registerTypedBitSet<VertBitSet>( "VertBitSet" );
    registerTypedBitSet<EdgeBitSet>( "EdgeBitSet" );
    registerTypedBitSet<UndirectedEdgeBitSet>( "UndirectedEdgeBitSet" );
}
