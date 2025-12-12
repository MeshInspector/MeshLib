public static partial class MR
{
    /**
    * \brief returns the maximum of the squared distances from each B-point to A-cloud
    * \param rigidB2A rigid transformation from B-cloud space to A-cloud space, nullptr considered as identity transformation
    * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
    */
    /// Generated from function `MR::findMaxDistanceSqOneWay`.
    /// Parameter `maxDistanceSq` defaults to `3.40282347e38f`.
    public static unsafe float FindMaxDistanceSqOneWay(MR.Const_PointCloud a, MR.Const_PointCloud b, MR.Const_AffineXf3f? rigidB2A = null, float? maxDistanceSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMaxDistanceSqOneWay_MR_PointCloud", ExactSpelling = true)]
        extern static float __MR_findMaxDistanceSqOneWay_MR_PointCloud(MR.Const_PointCloud._Underlying *a, MR.Const_PointCloud._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, float *maxDistanceSq);
        float __deref_maxDistanceSq = maxDistanceSq.GetValueOrDefault();
        return __MR_findMaxDistanceSqOneWay_MR_PointCloud(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, maxDistanceSq.HasValue ? &__deref_maxDistanceSq : null);
    }

    /**
    * \brief returns the squared Hausdorff distance between two point clouds, that is
    the maximum of squared distances from each point to the other cloud (in both directions)
    * \param rigidB2A rigid transformation from B-cloud space to A-cloud space, nullptr considered as identity transformation
    * \param maxDistanceSq upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq
    */
    /// Generated from function `MR::findMaxDistanceSq`.
    /// Parameter `maxDistanceSq` defaults to `3.40282347e38f`.
    public static unsafe float FindMaxDistanceSq(MR.Const_PointCloud a, MR.Const_PointCloud b, MR.Const_AffineXf3f? rigidB2A = null, float? maxDistanceSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMaxDistanceSq_MR_PointCloud", ExactSpelling = true)]
        extern static float __MR_findMaxDistanceSq_MR_PointCloud(MR.Const_PointCloud._Underlying *a, MR.Const_PointCloud._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, float *maxDistanceSq);
        float __deref_maxDistanceSq = maxDistanceSq.GetValueOrDefault();
        return __MR_findMaxDistanceSq_MR_PointCloud(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, maxDistanceSq.HasValue ? &__deref_maxDistanceSq : null);
    }
}
