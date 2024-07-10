#include "MRBitSet.h"

#pragma managed( push, off )
#include <MRMesh/MRBitSet.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

BitSetReadOnly::BitSetReadOnly()
{
    bs_ = new MR::BitSet();
}

BitSetReadOnly::BitSetReadOnly( MR::BitSet* bs )
{
    bs_ = bs;
}

BitSetReadOnly::~BitSetReadOnly()
{
    delete bs_;
}

BitSet::BitSet()
:BitSetReadOnly()
{}

BitSet::BitSet(int size)
:BitSetReadOnly()
{
    bs_->resize( size );
}

BitSet::BitSet(MR::BitSet* bs)
:BitSetReadOnly( bs )
{}

int BitSetReadOnly::size()
{
    return int( bs_->size() );
}

//call of MR::BitSet::find_last somehow breaks the build, so we provide an alternative implementation here
int BitSetReadOnly::findLast()
{
    if ( !bs_->any() )
        return -1;

    for ( int i = int( bs_->size() ); i-- >= 1; )
    {
        if ( bs_->test( i ) )
            return i;
    }
    return -1;
}

bool BitSetReadOnly::test(int i)
{
    return bs_->test( i );
}

void BitSet::set(int i)
{
    bs_->set( i, true );
}

void BitSet::set( int i, bool value )
{
    bs_->set( i, value );
}

void BitSet::resize( int size )
{
    bs_->resize( size );
}

void BitSet::autoResizeSet( int i )
{
    bs_->autoResizeSet( i, true );
}

void BitSet::autoResizeSet( int i, bool value )
{
    bs_->autoResizeSet( i, value );
}

MR_DOTNET_NAMESPACE_END
