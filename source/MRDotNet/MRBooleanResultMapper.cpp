#include "MRBooleanResultMapper.h"
#include "MRBitSet.h"

#pragma managed( push, off )
#include <MRMesh/MRBitSet.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

BooleanMaps::BooleanMaps( MR::BooleanResultMapper::Maps* maps )
{
    if ( !maps )
        throw gcnew System::ArgumentNullException( "maps" );

    maps_ = maps;
}

BooleanMaps::~BooleanMaps()
{
    delete maps_;
}

FaceMapReadOnly^ BooleanMaps::Cut2Origin::get()
{
    if ( !cut2origin_ )
    {
        cut2origin_ = gcnew FaceMap( int( maps_->cut2origin.size() ) );
        for ( auto face : maps_->cut2origin )
            cut2origin_->Add( FaceId( face ) );
    }

    return cut2origin_->AsReadOnly();
}

FaceMapReadOnly^ BooleanMaps::Cut2NewFaces::get()
{
    if ( !cut2newFaces_ )
    {
        cut2newFaces_ = gcnew FaceMap( int( maps_->cut2newFaces.size() ) );
        for ( auto face : maps_->cut2newFaces )
            cut2newFaces_->Add( FaceId( face ) );
    }

    return cut2newFaces_->AsReadOnly();
}

VertMapReadOnly^ BooleanMaps::Old2NewVerts::get()
{
    if ( !old2newVerts_ )
    {
        old2newVerts_ = gcnew VertMap( int( maps_->old2newVerts.size() ) );
        for ( auto vert : maps_->old2newVerts )
            old2newVerts_->Add( VertId( vert ) );
    }

    return old2newVerts_->AsReadOnly();
}

bool BooleanMaps::Identity::get()
{
    return maps_->identity;
}

BooleanResultMapper::BooleanResultMapper()
{
    mapper_ = new MR::BooleanResultMapper();
}

BooleanResultMapper::BooleanResultMapper( MR::BooleanResultMapper* mapper )
{
    if ( !mapper )
        throw gcnew System::ArgumentNullException( "mapper" );

    mapper_ = mapper;
}

BooleanResultMapper::~BooleanResultMapper()
{
    delete mapper_;
}

FaceBitSet^ BooleanResultMapper::FaceMap( FaceBitSet^ oldBS, MapObject obj )
{
    if ( !oldBS )
        throw gcnew System::ArgumentNullException( "oldBS" );

    if ( !maps_ )
        maps_ = gcnew array<BooleanMaps^>( 2 );

    auto% map = maps_[(int)obj];
    if ( !map )
        map = gcnew BooleanMaps( &mapper_->maps[(int)obj] );

    MR::FaceBitSet nativeOldBitSet( oldBS->bitSet()->m_bits.begin(), oldBS->bitSet()->m_bits.end() );        
    auto nativeRes = mapper_->map( nativeOldBitSet, MR::BooleanResultMapper::MapObject( obj ) );

    return gcnew FaceBitSet( new MR::BitSet( nativeRes.m_bits.begin(), nativeRes.m_bits.end() ) );    
}

VertBitSet^ BooleanResultMapper::VertMap( VertBitSet^ oldBS, MapObject obj )
{
    if ( !oldBS )
        throw gcnew System::ArgumentNullException( "oldBS" );

    if ( !maps_ )
        maps_ = gcnew array<BooleanMaps^>( 2 );

    auto% map = maps_[(int)obj];
    if ( !map )
        map = gcnew BooleanMaps( &mapper_->maps[(int)obj] );

    MR::VertBitSet nativeOldBitSet( oldBS->bitSet()->m_bits.begin(), oldBS->bitSet()->m_bits.end() );        
    auto nativeRes = mapper_->map( nativeOldBitSet, MR::BooleanResultMapper::MapObject( obj ) );

    return gcnew VertBitSet( new MR::BitSet( nativeRes.m_bits.begin(), nativeRes.m_bits.end() ) );
}

FaceBitSet^ BooleanResultMapper::NewFaces()
{
    auto nativeRes = mapper_->newFaces();
    return gcnew FaceBitSet( new MR::BitSet( nativeRes.m_bits.begin(), nativeRes.m_bits.end() ) );
}

BooleanMaps^ BooleanResultMapper::GetMaps( MapObject obj )
{
    if ( !maps_ )
        maps_ = gcnew array<BooleanMaps^>( 2 );

    auto% map = maps_[( int )obj];
    if ( !map )
        map = gcnew BooleanMaps( &mapper_->maps[( int )obj] );

    return map;
}

FaceBitSet^ BooleanResultMapper::FilteredOldFaceBitSet( FaceBitSet^ oldBS, MapObject obj )
{
    if ( !oldBS )
        throw gcnew System::ArgumentNullException( "oldBS" );

    if ( !maps_ )
        maps_ = gcnew array<BooleanMaps^>( 2 );

    auto% map = maps_[(int)obj];
    if ( !map )
        map = gcnew BooleanMaps( &mapper_->maps[(int)obj] );

    MR::FaceBitSet nativeOldBitSet( oldBS->bitSet()->m_bits.begin(), oldBS->bitSet()->m_bits.end() );        
    auto nativeRes = mapper_->filteredOldFaceBitSet( nativeOldBitSet, MR::BooleanResultMapper::MapObject( obj ) );
    return gcnew FaceBitSet( new MR::BitSet( nativeRes.m_bits.begin(), nativeRes.m_bits.end() ) );
}

MR_DOTNET_NAMESPACE_END