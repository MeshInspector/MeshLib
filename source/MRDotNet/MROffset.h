#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

public enum class SignDetectionMode
{
    Unsigned,         ///< unsigned distance, useful for bidirectional `Shell` offset
    OpenVDB,          ///< sign detection from OpenVDB library, which is good and fast if input geometry is closed
    ProjectionNormal, ///< the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
    WindingRule,      ///< ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh
    HoleWindingRule   ///< computes winding number generalization with support of holes in mesh, slower than WindingRule
};

public ref class OffsetParameters
{
public:
    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0.0f;
    /// determines the method to compute distance sign
    SignDetectionMode signDetectionMode = SignDetectionMode::OpenVDB;
    /// use FunctionVolume for voxel grid representation:
    ///  - memory consumption is approx. (z / (2 * thread_count)) less
    ///  - computation is about 2-3 times slower
    /// used only by \ref McOffsetMesh and \ref SharpOffsetMesh methods
    bool memoryEfficient = false;
};

public ref class SharpOffsetParameters : public OffsetParameters
{
public:
    /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
    float minNewVertDev = 1.0f / 25;
    /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
    float maxNewRank2VertDev = 5;
    /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
    float maxNewRank3VertDev = 2;
    /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
    /// big correction can be wrong and result from self-intersections in the reference mesh
    float maxOldVertPosCorrection = 0.5f;
};

public enum class GeneralOffsetMode
{
#ifndef MRMESH_NO_OPENVDB
    Smooth,     ///< create mesh using dual marching cubes from OpenVDB library
#endif
    Standard,   ///< create mesh using standard marching cubes implemented in MeshLib
    Sharpening  ///< create mesh using standard marching cubes with additional sharpening implemented in MeshLib
};

/// allows the user to select in the parameters which offset algorithm to call
public ref class GeneralOffsetParameters : public SharpOffsetParameters
{
public:
    GeneralOffsetMode mode = GeneralOffsetMode::Standard;
};

public ref class Offset
{
public:
    /// computes size of a cubical voxel to get approximately given number of voxels during rasterization
    static float SuggestVoxelSize( MeshPart mp, float approxNumVoxels );
#ifndef MRMESH_NO_OPENVDB
    /// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
    /// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
    /// and then converts back using OpenVDB library (dual marching cubes),
    /// so result mesh is always closed
    static Mesh^ OffsetMesh( MeshPart mp, float offset, OffsetParameters^ parameters );
    /// Offsets mesh by converting it to voxels and back two times
    /// only closed meshes allowed (only Offset mode)
    /// typically offsetA and offsetB have distinct signs
    static Mesh^ DoubleOffsetMesh( MeshPart mp, float offsetA, float offsetB, OffsetParameters^ parameters );
#endif
    /// Offsets mesh by converting it to distance field in voxels (using OpenVDB library if SignDetectionMode.OpenVDB or our implementation otherwise)
    /// and back using standard Marching Cubes, as opposed to Dual Marching Cubes in OffsetMesh(...)
    static Mesh^ McOffsetMesh( MeshPart mp, float offset, OffsetParameters^ parameters );
    /// Constructs a shell around selected mesh region with the properties that every point on the shall must
    ///  1. be located not further than given distance from selected mesh part,
    ///  2. be located not closer to not-selected mesh part than to selected mesh part.
    static Mesh^ McShellMeshRegion( MeshPart mp, float offset, float voxelSize );
    /// Offsets mesh by converting it to voxels and back
    /// post process result using reference mesh to sharpen features
    static Mesh^ SharpOffsetMesh( MeshPart mp, float offset, SharpOffsetParameters^ parameters );
    /// Offsets mesh by converting it to voxels and back using one of three modes specified in the parameters
    static Mesh^ GeneralOffsetMesh( MeshPart mp, float offset, GeneralOffsetParameters^ parameters );
    /// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
    /// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
    /// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode.Unsigned, and you will get open mesh (with several components) on output
    /// if your input mesh is closed then please specify another sign detection mode, and you will get closed mesh (with several components) on output;
    static Mesh^ ThickenMesh( Mesh^ mesh, float offset, GeneralOffsetParameters^ parameters );
};

MR_DOTNET_NAMESPACE_END

