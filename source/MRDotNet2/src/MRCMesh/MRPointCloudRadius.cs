public static partial class MR
{
    /// \brief Finds the radius of ball, so on average that ball contained avgPoints excluding the central point
    /// \param samples the number of test points to find given number of samples in each
    /// Generated from function `MR::findAvgPointsRadius`.
    /// Parameter `samples` defaults to `1024`.
    public static unsafe float FindAvgPointsRadius(MR.Const_PointCloud pointCloud, int avgPoints, int? samples = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findAvgPointsRadius", ExactSpelling = true)]
        extern static float __MR_findAvgPointsRadius(MR.Const_PointCloud._Underlying *pointCloud, int avgPoints, int *samples);
        int __deref_samples = samples.GetValueOrDefault();
        return __MR_findAvgPointsRadius(pointCloud._UnderlyingPtr, avgPoints, samples.HasValue ? &__deref_samples : null);
    }

    /// expands the region on given euclidian distance. returns false if callback also returns false
    /// Generated from function `MR::dilateRegion`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_PointCloud pointCloud, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_5_const_MR_PointCloud_ref", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_5_const_MR_PointCloud_ref(MR.Const_PointCloud._Underlying *pointCloud, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Const_AffineXf3f._Underlying *xf);
        return __MR_dilateRegion_5_const_MR_PointCloud_ref(pointCloud._UnderlyingPtr, region._UnderlyingPtr, dilation, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null) != 0;
    }

    /// shrinks the region on given euclidian distance. returns false if callback also returns false
    /// Generated from function `MR::erodeRegion`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_PointCloud pointCloud, MR.VertBitSet region, float erosion, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_5_const_MR_PointCloud_ref", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_5_const_MR_PointCloud_ref(MR.Const_PointCloud._Underlying *pointCloud, MR.VertBitSet._Underlying *region, float erosion, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Const_AffineXf3f._Underlying *xf);
        return __MR_erodeRegion_5_const_MR_PointCloud_ref(pointCloud._UnderlyingPtr, region._UnderlyingPtr, erosion, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null) != 0;
    }
}
