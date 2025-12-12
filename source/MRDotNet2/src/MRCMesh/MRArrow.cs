public static partial class MR
{
    // creates hollow arrow from the 'base' to the 'vert'. Number of points on the circle 'qual' is between 3 and 256
    /// Generated from function `MR::makeArrow`.
    /// Parameter `thickness` defaults to `0.0500000007f`.
    /// Parameter `coneRadius` defaults to `0.100000001f`.
    /// Parameter `coneSize` defaults to `0.200000003f`.
    /// Parameter `qual` defaults to `32`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeArrow(MR.Const_Vector3f base_, MR.Const_Vector3f vert, float? thickness = null, float? coneRadius = null, float? coneSize = null, int? qual = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeArrow", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeArrow(MR.Const_Vector3f._Underlying *base_, MR.Const_Vector3f._Underlying *vert, float *thickness, float *coneRadius, float *coneSize, int *qual);
        float __deref_thickness = thickness.GetValueOrDefault();
        float __deref_coneRadius = coneRadius.GetValueOrDefault();
        float __deref_coneSize = coneSize.GetValueOrDefault();
        int __deref_qual = qual.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeArrow(base_._UnderlyingPtr, vert._UnderlyingPtr, thickness.HasValue ? &__deref_thickness : null, coneRadius.HasValue ? &__deref_coneRadius : null, coneSize.HasValue ? &__deref_coneSize : null, qual.HasValue ? &__deref_qual : null), is_owning: true));
    }

    // creates the mesh with 3 axis arrows
    /// Generated from function `MR::makeBasisAxes`.
    /// Parameter `size` defaults to `1.0f`.
    /// Parameter `thickness` defaults to `0.0500000007f`.
    /// Parameter `coneRadius` defaults to `0.100000001f`.
    /// Parameter `coneSize` defaults to `0.200000003f`.
    /// Parameter `qual` defaults to `32`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeBasisAxes(float? size = null, float? thickness = null, float? coneRadius = null, float? coneSize = null, int? qual = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeBasisAxes", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeBasisAxes(float *size, float *thickness, float *coneRadius, float *coneSize, int *qual);
        float __deref_size = size.GetValueOrDefault();
        float __deref_thickness = thickness.GetValueOrDefault();
        float __deref_coneRadius = coneRadius.GetValueOrDefault();
        float __deref_coneSize = coneSize.GetValueOrDefault();
        int __deref_qual = qual.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeBasisAxes(size.HasValue ? &__deref_size : null, thickness.HasValue ? &__deref_thickness : null, coneRadius.HasValue ? &__deref_coneRadius : null, coneSize.HasValue ? &__deref_coneSize : null, qual.HasValue ? &__deref_qual : null), is_owning: true));
    }
}
