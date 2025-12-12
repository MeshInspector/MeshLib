public static partial class MR
{
    /// finds an intersection between a segm1 and a segm2
    /// \return nullopt if they don't intersect (even if they match)
    /// Generated from function `MR::intersection`.
    public static unsafe MR.Std.Optional_MRVector2f Intersection(MR.Const_LineSegm2f segm1, MR.Const_LineSegm2f segm2)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_intersection", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVector2f._Underlying *__MR_intersection(MR.Const_LineSegm2f._Underlying *segm1, MR.Const_LineSegm2f._Underlying *segm2);
        return new(__MR_intersection(segm1._UnderlyingPtr, segm2._UnderlyingPtr), is_owning: true);
    }
}
