public static partial class MR
{
    public static partial class FloatGridComponents
    {
        /// finds separated by iso-value components in grid space (0 voxel id is minimum active voxel in grid)
        /// Generated from function `MR::FloatGridComponents::getAllComponents`.
        /// Parameter `isoValue` defaults to `0.0f`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVoxelBitSet> GetAllComponents(MR.Const_FloatGrid grid, float? isoValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGridComponents_getAllComponents", ExactSpelling = true)]
            extern static MR.Std.Vector_MRVoxelBitSet._Underlying *__MR_FloatGridComponents_getAllComponents(MR.Const_FloatGrid._Underlying *grid, float *isoValue);
            float __deref_isoValue = isoValue.GetValueOrDefault();
            return MR.Misc.Move(new MR.Std.Vector_MRVoxelBitSet(__MR_FloatGridComponents_getAllComponents(grid._UnderlyingPtr, isoValue.HasValue ? &__deref_isoValue : null), is_owning: true));
        }
    }
}
