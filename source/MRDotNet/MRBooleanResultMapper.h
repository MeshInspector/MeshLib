#include "MRMeshFwd.h"
#pragma managed( push, off )
#include <MRMesh/MRBooleanOperation.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

public ref class BooleanMaps
{
public:
    ~BooleanMaps();

    property FaceMapReadOnly^ Cut2Origin { FaceMapReadOnly^ get(); }
    property FaceMapReadOnly^ Cut2NewFaces { FaceMapReadOnly^ get(); }
    property VertMapReadOnly^ Old2NewVerts { VertMapReadOnly^ get(); }
    property bool Identity { bool get(); }

internal:
    BooleanMaps( MR::BooleanResultMapper::Maps* maps );

private:
    FaceMap^ cut2origin_;
    FaceMap^ cut2newFaces_;
    VertMap^ old2newVerts_;

    MR::BooleanResultMapper::Maps* maps_;
};

public enum class MapObject
{
    A,
    B
};

public ref class BooleanResultMapper
{
public:
    BooleanResultMapper();
    ~BooleanResultMapper();
    /// Returns faces bitset of result mesh corresponding input one
    FaceBitSet^ FaceMap( FaceBitSet^ oldBS, MapObject obj );
    /// Returns vertices bitset of result mesh corresponding input one
    VertBitSet^ VertMap( VertBitSet^ oldBS, MapObject obj );
    /// Returns only new faces that are created during boolean operation
    FaceBitSet^ NewFaces();

    BooleanMaps^ GetMaps( MapObject obj );
    /// returns updated oldBS leaving only faces that has corresponding ones in result mesh
    FaceBitSet^ FilteredOldFaceBitSet( FaceBitSet^ oldBS, MapObject obj );

internal:
    BooleanResultMapper( MR::BooleanResultMapper* mapper );
    MR::BooleanResultMapper* getMapper() { return mapper_; }

private:
    MR::BooleanResultMapper* mapper_;
    array<BooleanMaps^>^ maps_;
};



MR_DOTNET_NAMESPACE_END