#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRphmap.h"
#include "MRVector.h"
#include "MRPch/MRBindingMacros.h"
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#pragma warning(push)
#pragma warning(disable: 4643) //Forward declaring in namespace std is not permitted by the C++ Standard.
#include <boost/dynamic_bitset.hpp>
#pragma warning(pop)
#include <iterator>
#include <functional>

namespace MR
{

/**
 * \defgroup BasicGroup Basic elements overview
 * \brief This chapter represents documentation about basic elements
 * \{
 */

/// std::vector<bool> like container  (random-access, size_t - index type, bool - value type)
/// with all bits after size() considered off during testing
class BitSet : public boost::dynamic_bitset<std::uint64_t>
{
public:
    using base = boost::dynamic_bitset<std::uint64_t>;
    using base::base;
    using IndexType = size_t;

    /// creates bitset of given size filled with given value
    explicit BitSet( size_t numBits, bool fillValue ) { resize( numBits, fillValue ); }

    /// prohibit these constructors inherited from boost::dynamic_bitset, which can initialize only few initial bits
    explicit BitSet( size_t, unsigned long ) = delete;
    template<class T, std::enable_if_t<std::is_arithmetic<T>::value, std::nullptr_t> = nullptr>
    explicit BitSet( T, T ) = delete;

    // all bits after size() we silently consider as not-set
    [[nodiscard]] bool test( IndexType n ) const { return n < size() && base::test( n ); }
    [[nodiscard]] bool test_set( IndexType n, bool val = true ) { return ( val || n < size() ) ? base::test_set( n, val ) : false; }

    BitSet & set( IndexType n, size_type len, bool val ) { base::set( n, len, val ); return * this; }
    BitSet & set( IndexType n, bool val ) { base::set( n, val ); return * this; } // Not using a default argument for `val` to get better C bindings.
    BitSet & set( IndexType n ) { base::set( n ); return * this; }
    BitSet & set() { base::set(); return * this; }
    BitSet & reset( IndexType n, size_type len ) { if ( n < size() ) base::reset( n, len ); return * this; }
    BitSet & reset( IndexType n ) { if ( n < size() ) base::reset( n ); return * this; }
    BitSet & reset() { base::reset(); return * this; }
    BitSet & flip( IndexType n, size_type len ) { base::flip( n, len ); return * this; }
    BitSet & flip( IndexType n ) { base::flip( n ); return * this; }
    BitSet & flip() { base::flip(); return * this; }

    /// read-only access to all bits stored as a vector of uint64 blocks
    const auto & bits() const { return m_bits; }

    MRMESH_API BitSet & operator &= ( const BitSet & b );
    MRMESH_API BitSet & operator |= ( const BitSet & b );
    MRMESH_API BitSet & operator ^= ( const BitSet & b );

    MRMESH_API BitSet & operator -= ( const BitSet & b );
    /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
    MRMESH_API BitSet & subtract( const BitSet & b, int bShiftInBlocks );

    /// return the highest index i such as bit i is set, or npos if *this has no on bits.
    [[nodiscard]] MRMESH_API IndexType find_last() const;

    /// returns the location of nth set bit (where the first bit corresponds to n=0) or npos if there are less bit set
    [[nodiscard]] MRMESH_API size_t nthSetBit( size_t n ) const;

    /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
    [[nodiscard]] MRMESH_API bool is_subset_of( const BitSet& a ) const;

    /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set and if this->count() < a.count(). Otherwise this function returns false.
    bool is_proper_subset_of( const BitSet& a ) const = delete; // base implementation does not support bitsets of different sizes

    /// doubles reserved memory until resize(newSize) can be done without reallocation
    void resizeWithReserve( size_t newSize )
    {
        auto reserved = capacity();
        if ( reserved > 0 && newSize > reserved )
        {
            while ( newSize > reserved )
                reserved <<= 1;
            reserve( reserved );
        }
        resize( newSize );
    }

    /// sets elements [pos, pos+len) to given value, adjusting the size of the set to include new elements
    void autoResizeSet( size_t pos, size_type len, bool val = true )
    {
        if ( pos + len > size() )
            resizeWithReserve( pos + len );
        set( pos, len, val );
    }
    void autoResizeSet( size_t pos, bool val = true ) { autoResizeSet( pos, 1, val ); }

    /// same as \ref autoResizeSet and returns previous value of pos-bit
    [[nodiscard]] bool autoResizeTestSet( size_t pos, bool val = true )
    {
        bool const b = test( pos );
        if ( b != val )
            autoResizeSet( pos, val );
        return b;
    }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return capacity() / 8; }

    /// returns the identifier of the back() element
    [[nodiscard]] IndexType backId() const { assert( !empty() ); return IndexType{ size() - 1 }; }

    /// [beginId(), endId()) is the range of all bits in the set
    [[nodiscard]] static IndexType beginId() { return IndexType{ 0 }; }
    [[nodiscard]] IndexType endId() const { return IndexType{ size() }; }

    // Normally those are inherited from `boost::dynamic_bitset`, but MRBind currently chokes on it, so we provide those manually.
    #if defined(MR_PARSING_FOR_ANY_BINDINGS) || defined(MR_COMPILING_ANY_BINDINGS)
    std::size_t size() const { return dynamic_bitset::size(); }
    std::size_t count() const { return dynamic_bitset::count(); }
    void resize( std::size_t num_bits, bool value = false ) { dynamic_bitset::resize( num_bits, value ); }
    void clear() { dynamic_bitset::clear(); }
    void push_back( bool bit ) { dynamic_bitset::push_back( bit ); }
    void pop_back() { dynamic_bitset::pop_back(); }
    #endif

private:
    using base::m_highest_block;
    using base::m_bits;
    using base::m_num_bits;
};

/// Vector<bool, I> like container (random-access, I - index type, bool - value type)
/// with all bits after size() considered off during testing
template <typename I>
class TypedBitSet : public BitSet
{
    using base = BitSet;
public:
    using base::base;
    using IndexType = I;

    /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
    explicit TypedBitSet( const BitSet & src ) : BitSet( src ) {}

    /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
    explicit TypedBitSet( BitSet && src ) : BitSet( std::move( src ) ) {}

    TypedBitSet & set( IndexType n, size_type len, bool val ) { base::set( n, len, val ); return * this; }
    TypedBitSet & set( IndexType n, bool val ) { base::set( n, val ); return * this; } // Not using a default argument for `val` to get better C bindings.
    TypedBitSet & set( IndexType n ) { base::set( n ); return * this; }
    TypedBitSet & set() { base::set(); return * this; }
    TypedBitSet & reset( IndexType n, size_type len ) { base::reset( n, len ); return * this; }
    TypedBitSet & reset( IndexType n ) { base::reset( n ); return * this; }
    TypedBitSet & reset() { base::reset(); return * this; }
    TypedBitSet & flip( IndexType n, size_type len ) { base::flip( n, len ); return * this; }
    TypedBitSet & flip( IndexType n ) { base::flip( n ); return * this; }
    TypedBitSet & flip() { base::flip(); return * this; }
    [[nodiscard]] bool test( IndexType n ) const { return base::test( n ); }
    [[nodiscard]] bool test_set( IndexType n, bool val = true ) { return base::test_set( n, val ); }

    // Disable for python bindings, because MRBind chokes on `boost::dynamic_bitset::reference`.
    [[nodiscard]] MR_BIND_IGNORE reference operator[]( IndexType pos ) { return base::operator[]( pos ); }
    [[nodiscard]] bool operator[]( IndexType pos ) const { return base::operator[]( pos ); }

    [[nodiscard]] IndexType find_first() const { return IndexType( base::find_first() ); }
    [[nodiscard]] IndexType find_next( IndexType pos ) const { return IndexType( base::find_next( pos ) ); }
    [[nodiscard]] IndexType find_last() const { return IndexType( base::find_last() ); }
    /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
    [[nodiscard]] IndexType nthSetBit( size_t n ) const { return IndexType( base::nthSetBit( n ) ); }

    TypedBitSet & operator &= ( const TypedBitSet & b ) { base::operator &= ( b ); return * this; }
    TypedBitSet & operator |= ( const TypedBitSet & b ) { base::operator |= ( b ); return * this; }
    TypedBitSet & operator ^= ( const TypedBitSet & b ) { base::operator ^= ( b ); return * this; }
    TypedBitSet & operator -= ( const TypedBitSet & b ) { base::operator -= ( b ); return * this; }

    [[nodiscard]] friend TypedBitSet operator & ( const TypedBitSet & a, const TypedBitSet & b ) { auto res{ a }; res &= b; return res; }
    [[nodiscard]] friend TypedBitSet operator | ( const TypedBitSet & a, const TypedBitSet & b ) { auto res{ a }; res |= b; return res; }
    [[nodiscard]] friend TypedBitSet operator ^ ( const TypedBitSet & a, const TypedBitSet & b ) { auto res{ a }; res ^= b; return res; }
    [[nodiscard]] friend TypedBitSet operator - ( const TypedBitSet & a, const TypedBitSet & b ) { auto res{ a }; res -= b; return res; }

    /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
    TypedBitSet & subtract( const TypedBitSet & b, int bShiftInBlocks ) { base::subtract( b, bShiftInBlocks ); return * this; }

    /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
    [[nodiscard]] bool is_subset_of( const TypedBitSet& a ) const { return base::is_subset_of( a ); }

    /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
    [[nodiscard]] bool intersects( const TypedBitSet & a ) const { return base::intersects( a ); }

    void autoResizeSet( IndexType pos, size_type len, bool val = true ) { base::autoResizeSet( pos, len, val ); }
    void autoResizeSet( IndexType pos, bool val = true ) { base::autoResizeSet( pos, val ); }
    [[nodiscard]] bool autoResizeTestSet( IndexType pos, bool val = true ) { return base::autoResizeTestSet( pos, val ); }

    /// constructs another bit set from this where every set bit index is transformed using given map
    template <typename M>
    [[nodiscard]] TypedBitSet getMapping( const M & map ) const;
    [[nodiscard]] TypedBitSet getMapping( const Vector<IndexType, IndexType> & map ) const
        { return getMapping( [&map]( IndexType i ) { return map[i]; } ); }
    [[nodiscard]] TypedBitSet getMapping( const BMap<IndexType, IndexType> & map ) const
        { return getMapping( [&map]( IndexType i ) { return map.b[i]; }, map.tsize ); }
    [[nodiscard]] TypedBitSet getMapping( const HashMap<IndexType, IndexType> & map ) const
        { return getMapping( [&map]( IndexType i ) { return getAt( map, i ); } ); }
    /// this is a faster version if the result size is known beforehand
    template <typename M>
    [[nodiscard]] TypedBitSet getMapping( const M & map, size_t resSize ) const;
    [[nodiscard]] TypedBitSet getMapping( const Vector<IndexType, IndexType> & map, size_t resSize ) const
        { return getMapping( [&map]( IndexType i ) { return map[i]; }, resSize ); }
    [[nodiscard]] TypedBitSet getMapping( const HashMap<IndexType, IndexType> & map, size_t resSize ) const
        { return getMapping( [&map]( IndexType i ) { return getAt( map, i ); }, resSize ); }

    /// returns the identifier of the back() element
    [[nodiscard]] IndexType backId() const { assert( !empty() ); return IndexType{ size() - 1 }; }

    /// [beginId(), endId()) is the range of all bits in the set
    [[nodiscard]] static IndexType beginId() { return IndexType{ size_t( 0 ) }; }
    [[nodiscard]] IndexType endId() const { return IndexType{ size() }; }
};


/// returns the amount of memory given BitSet occupies on heap
[[nodiscard]] inline size_t heapBytes( const BitSet& bs )
{
    return bs.heapBytes();
}

/// compare that two bit sets have the same set bits (they can be equal even if sizes are distinct but last bits are off)
[[nodiscard]] MRMESH_API bool operator == ( const BitSet & a, const BitSet & b );
template <typename I>
[[nodiscard]] inline bool operator == ( const TypedBitSet<I> & a, const TypedBitSet<I> & b )
    { return static_cast<const BitSet &>( a ) == static_cast<const BitSet &>( b ); }
/// prohibit comparison of unrelated sets
template <typename T, typename U>
void operator == ( const TypedBitSet<T> & a, const TypedBitSet<U> & b ) = delete;

template <typename I>
[[nodiscard]] inline std::function<bool( I )> makePredicate( const TypedBitSet<I> * bitset )
{
    std::function<bool( I )> res;
    if ( bitset )
        res = [bitset]( I id ) { return bitset->test( id ); };
    return res;
}

template <typename I>
[[nodiscard]] inline std::function<bool( I )> makePredicate( const TypedBitSet<I> & bitset )
    { return makePredicate( &bitset ); }

template <typename I>
[[nodiscard]] inline bool contains( const TypedBitSet<I> * bitset, I id )
{
    return id.valid() && ( !bitset || bitset->test( id ) );
}

template <typename I>
[[nodiscard]] inline bool contains( const TypedBitSet<I> & bitset, I id )
{
    return id.valid() && bitset.test( id );
}

/// iterator to enumerate all indices with set bits in BitSet class or its derivatives
template <typename T>
class MR_BIND_IGNORE SetBitIteratorT
{
public:
    using IndexType = typename T::IndexType;

    using iterator_category = std::forward_iterator_tag;
    using value_type        = IndexType;
    using difference_type   = std::ptrdiff_t;
    using reference         = const IndexType; ///< intentionally not a reference
    using pointer           = const IndexType *;

    /// constructs end iterator
    SetBitIteratorT() = default;
    /// constructs begin iterator
    SetBitIteratorT( const T & bitset )
        : bitset_( &bitset ), index_( bitset.find_first() )
    {
    }
    SetBitIteratorT & operator++( )
    {
        index_ = bitset_->find_next( index_ );
        return * this;
    }
    [[nodiscard]] SetBitIteratorT operator++( int )
    {
        SetBitIteratorT ret = *this;
        operator++();
        return ret;
    }

    [[nodiscard]] const T * bitset() const { return bitset_; }
    [[nodiscard]] reference operator *() const { return index_; }

    [[nodiscard]] friend bool operator ==( const SetBitIteratorT<T> & a, const SetBitIteratorT<T> & b ) { return *a == *b; }

private:
    const T * bitset_ = nullptr;
    IndexType index_ = IndexType( ~size_t( 0 ) );
};


[[nodiscard]] MR_BIND_IGNORE inline auto begin( const BitSet & a )
    { return SetBitIteratorT<BitSet>(a); }
[[nodiscard]] MR_BIND_IGNORE inline auto end( const BitSet & )
    { return SetBitIteratorT<BitSet>(); }

template <typename I>
[[nodiscard]] MR_BIND_IGNORE inline auto begin( const TypedBitSet<I> & a )
    { return SetBitIteratorT<TypedBitSet<I>>(a); }
template <typename I>
[[nodiscard]] MR_BIND_IGNORE inline auto end( const TypedBitSet<I> & )
    { return SetBitIteratorT<TypedBitSet<I>>(); }

/// creates a Vector where for each set bit of input bitset its sequential number starting from 0 is returned; and -1 for reset bits
template <typename I>
[[nodiscard]] Vector<int, I> makeVectorWithSeqNums( const TypedBitSet<I> & bs )
{
    Vector<int, I> res( bs.size(), -1 );
    int n = 0;
    for ( auto v : bs )
        res[v] = n++;
    return res;
}

/// creates a HashMap where for each set bit of input bitset its sequential number starting from 0 is returned
template <typename I>
[[nodiscard]] HashMap<I, int> makeHashMapWithSeqNums( const TypedBitSet<I> & bs )
{
    HashMap<I, int> res;
    int n = 0;
    for ( auto v : bs )
        res[v] = n++;
    return res;
}

template <typename I>
template <typename M>
[[nodiscard]] TypedBitSet<I> TypedBitSet<I>::getMapping( const M & map ) const
{
    TypedBitSet<I> res;
    for ( auto b : *this )
        if ( auto mapped = map( b ) )
            res.autoResizeSet( mapped );
    return res;
}

template <typename I>
template <typename M>
[[nodiscard]] TypedBitSet<I> TypedBitSet<I>::getMapping( const M & map, size_t resSize ) const
{
    TypedBitSet<I> res;
    if ( !any() )
        return res;
    res.resize( resSize );
    for ( auto b : *this )
        if ( auto mapped = map( b ) )
            res.set( mapped );
    return res;
}

[[nodiscard]] inline BitSet operator & ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res &= b; return res; }
[[nodiscard]] inline BitSet operator | ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res |= b; return res; }
[[nodiscard]] inline BitSet operator ^ ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res ^= b; return res; }
[[nodiscard]] inline BitSet operator - ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res -= b; return res; }

/// \}

} // namespace MR
