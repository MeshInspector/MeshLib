public static partial class MR
{
    /// performs sampling of cloud points by iteratively removing one point with minimal metric (describing distance to the closest point and previous nearby removals),
    /// thus allowing stopping at any given number of samples;
    /// returns std::nullopt if it was terminated by the callback
    /// Generated from function `MR::pointIterativeSampling`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> PointIterativeSampling(MR.Const_PointCloud cloud, int numSamples, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointIterativeSampling", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_pointIterativeSampling(MR.Const_PointCloud._Underlying *cloud, int numSamples, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_pointIterativeSampling(cloud._UnderlyingPtr, numSamples, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }
}
