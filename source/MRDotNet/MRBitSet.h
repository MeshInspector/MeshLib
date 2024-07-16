#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// container of bits
public ref class BitSetReadOnly
{
public:
    /// test if given bit is set
    bool Test( int index );
    /// returns index of the first set bit
    int FindFirst();
    /// returns index of the last set bit
    int FindLast();
    /// returns total number of bits
    int Size();
    /// returns number of set bits
    int Count();

internal:
    BitSetReadOnly( MR::BitSet* bs );
    ~BitSetReadOnly();
protected:
    BitSetReadOnly();
    MR::BitSet* bs_;
};

public ref class BitSet : public BitSetReadOnly
{
public:
    /// creates empty bitset
    BitSet();

    /// creates bitset with given size
    BitSet(int size);

    /// sets the given bit to true
    void Set( int index );
    /// sets the given bit to given value
    void Set( int index, bool value );

    /// changes the size of the bitset
    void Resize( int size );
    
    /// sets the given bit to true. If index is out of range, it will be automatically resized
    void AutoResizeSet( int index );
    /// sets the given bit to given value. If index is out of range, it will be automatically resized
    void AutoResizeSet( int index, bool value );

internal:
    BitSet( MR::BitSet* bs );
    MR::BitSet* bitSet() { return bs_; }
};

MR_DOTNET_NAMESPACE_END
