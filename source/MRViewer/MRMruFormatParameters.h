#pragma once

namespace MR
{

// Parameters of mesh / point / voxel object storage formats inside an .mru file
struct MruFormatParameters
{
    enum class MeshFormat
    {
        Ctm,
        Ply,
        Mrmesh
    } meshFormat = MeshFormat::Ply; ///< ObjectMesh storage format

    enum class PointsFormat
    {
        Ctm,
        Ply
    } pointsFormat = PointsFormat::Ply;  ///< ObjectPoints storage format

    enum class VoxelsFormat
    {
        Vdb,
        Raw
    } voxelsFormat = VoxelsFormat::Vdb; ///< ObjectVoxels storage format

};

}
