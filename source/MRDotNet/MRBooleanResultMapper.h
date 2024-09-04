#include "MRMeshFwd.h"
#pragma managed( push, off )
#include <MRMesh/MRBooleanOperation.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

public ref class BooleanMaps
{
public:
    ~BooleanMaps();

    /// "after cut" faces to "origin" faces
    /// this map is not 1-1, but N-1
    property FaceMapReadOnly^ Cut2Origin { FaceMapReadOnly^ get(); }
    /// "after cut" faces to "after stitch" faces (1-1)
    property FaceMapReadOnly^ Cut2NewFaces { FaceMapReadOnly^ get(); }
    /// "origin" vertices to "after stitch" vertices (1-1)
    property VertMapReadOnly^ Old2NewVerts { VertMapReadOnly^ get(); }
    /// old topology indexes are valid if true
    property bool Identity { bool get(); }

internal:
    BooleanMaps( MR::BooleanResultMapper::Maps* maps );

private:
    FaceMap^ cut2origin_;
    FaceMap^ cut2newFaces_;
    VertMap^ old2newVerts_;

    MR::BooleanResultMapper::Maps* maps_;
};

/// input object index enum
public enum class MapObject
{
    A,
    B
};

///this class allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MeshBoolean to result mesh topology primitives
public ref class BooleanResultMapper
{
public:
    BooleanResultMapper();
    ~BooleanResultMapper();
    /// returns faces bitset of result mesh corresponding input one
    FaceBitSet^ FaceMap( FaceBitSet^ oldBS, MapObject obj );
    /// returns vertices bitset of result mesh corresponding input one
    VertBitSet^ VertMap( VertBitSet^ oldBS, MapObject obj );
    /// returns only new faces that are created during boolean operation
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