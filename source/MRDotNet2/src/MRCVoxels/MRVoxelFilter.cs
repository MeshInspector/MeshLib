public static partial class MR
{
    public enum VoxelFilterType : int
    {
        Median = 0,
        Mean = 1,
        Gaussian = 2,
    }

    /// Performs voxels filtering.
    /// @param type Type of fitler
    /// @param width Width of the filtering window, must be an odd number greater or equal to 1.
    /// Generated from function `MR::voxelFilter`.
    public static unsafe MR.Misc._Moved<MR.VdbVolume> VoxelFilter(MR.Const_VdbVolume volume, MR.VoxelFilterType type, int width)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_voxelFilter", ExactSpelling = true)]
        extern static MR.VdbVolume._Underlying *__MR_voxelFilter(MR.Const_VdbVolume._Underlying *volume, MR.VoxelFilterType type, int width);
        return MR.Misc.Move(new MR.VdbVolume(__MR_voxelFilter(volume._UnderlyingPtr, type, width), is_owning: true));
    }
}
