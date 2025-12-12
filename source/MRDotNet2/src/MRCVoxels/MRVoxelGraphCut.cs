public static partial class MR
{
    /**
    * \brief Segment voxels of given volume on two sets using graph-cut, returning source set
    *
    * \param k - coefficient in the exponent of the metric affecting edge capacity:\n
    *        increasing k you force to find a higher steps in the density on the boundary, decreasing k you ask for smoother boundary
    * \param sourceSeeds - these voxels will be included in the result
    * \param sinkSeeds - these voxels will be excluded from the result
    * 
    * \sa \ref VolumeSegmenter
    */
    /// Generated from function `MR::segmentVolumeByGraphCut`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVoxelBitSet_StdString> SegmentVolumeByGraphCut(MR.Const_SimpleVolume densityVolume, float k, MR.Const_VoxelBitSet sourceSeeds, MR.Const_VoxelBitSet sinkSeeds, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_segmentVolumeByGraphCut", ExactSpelling = true)]
        extern static MR.Expected_MRVoxelBitSet_StdString._Underlying *__MR_segmentVolumeByGraphCut(MR.Const_SimpleVolume._Underlying *densityVolume, float k, MR.Const_VoxelBitSet._Underlying *sourceSeeds, MR.Const_VoxelBitSet._Underlying *sinkSeeds, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRVoxelBitSet_StdString(__MR_segmentVolumeByGraphCut(densityVolume._UnderlyingPtr, k, sourceSeeds._UnderlyingPtr, sinkSeeds._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }
}
