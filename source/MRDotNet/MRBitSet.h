#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public ref class BitSetReadOnly
{
public:
    bool test( int index );
    int findLast();
    int size();
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

    void set( int index );
    void set( int index, bool value );

    void resize( int size );
    
    void autoResizeSet( int index );
    void autoResizeSet( int index, bool value );

internal:
    BitSet( MR::BitSet* bs );
};

MR_DOTNET_NAMESPACE_END
