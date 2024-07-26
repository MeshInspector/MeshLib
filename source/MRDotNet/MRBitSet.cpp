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

bool BitSetReadOnly::operator==( BitSetReadOnly^ a, BitSetReadOnly^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return a->IsEqualTo( b );
}

bool BitSetReadOnly::operator!=( BitSetReadOnly^ a, BitSetReadOnly^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return !a->IsEqualTo( b );
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

int BitSetReadOnly::Size()
{
    return int( bs_->size() );
}

int BitSetReadOnly::Count()
{
    return int( bs_->count() );
}

int BitSetReadOnly::FindLast()
{
    return int( bs_->find_last() );
}

int BitSetReadOnly::FindFirst()
{
    return int( bs_->find_first() );
}

bool BitSetReadOnly::Test(int i)
{
    return bs_->test( i );
}

void BitSet::Set(int i)
{
    bs_->set( i, true );
}

void BitSet::Set( int i, bool value )
{
    bs_->set( i, value );
}

void BitSet::Resize( int size )
{
    bs_->resize( size );
}

void BitSet::AutoResizeSet( int i )
{
    bs_->autoResizeSet( i, true );
}

void BitSet::AutoResizeSet( int i, bool value )
{
    bs_->autoResizeSet( i, value );
}

BitSetReadOnly^ BitSet::Clone()
{
    return gcnew BitSet( new MR::BitSet( *bs_ ) );
}

bool BitSet::IsEqualTo( BitSetReadOnly^ other )
{
    if ( !other )
        throw gcnew System::ArgumentNullException();

    return *bs_ == *((BitSet^)other)->bs_;
}

MR_DOTNET_NAMESPACE_END
