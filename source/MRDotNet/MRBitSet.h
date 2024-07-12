#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class BitSetReadOnly
{
public:
    bool Test( int index );
    int FindFirst();
    int FindLast();
    int Size();
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
    BitSet();
    BitSet(int size);

    void Set( int index );
    void Set( int index, bool value );

    void Resize( int size );
    
    void AutoResizeSet( int index );
    void AutoResizeSet( int index, bool value );

internal:
    BitSet( MR::BitSet* bs );
};

MR_DOTNET_NAMESPACE_END
