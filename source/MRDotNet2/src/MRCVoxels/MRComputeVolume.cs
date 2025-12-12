public static partial class MR
{
    // Computes summary volume of given meshes converting it to voxels of given size
    // note that each mesh should have closed topology
    // speed and precision depends on voxelSize (smaller voxel - faster, less precise; bigger voxel - slower, more precise)
    /// Generated from function `MR::voxelizeAndComputeVolume`.
    public static unsafe float VoxelizeAndComputeVolume(MR.Std.Const_Vector_StdSharedPtrMRMesh meshes, MR.Const_AffineXf3f xf, MR.Const_Vector3f voxelSize)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_voxelizeAndComputeVolume", ExactSpelling = true)]
        extern static float __MR_voxelizeAndComputeVolume(MR.Std.Const_Vector_StdSharedPtrMRMesh._Underlying *meshes, MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector3f._Underlying *voxelSize);
        return __MR_voxelizeAndComputeVolume(meshes._UnderlyingPtr, xf._UnderlyingPtr, voxelSize._UnderlyingPtr);
    }
}
