public static partial class MR
{
    /// \brief computes the closest points on two triangles, and returns the 
    /// squared distance between them.
    /// 
    /// \param s,t are the triangles, stored tri[point][dimension].
    ///
    /// \details If the triangles are disjoint, p and q give the closest points of 
    /// s and t respectively. However, if the triangles overlap, p and q 
    /// are basically a random pair of points from the triangles, not 
    /// coincident points on the intersection of the triangles, as might 
    /// be expected.
    /// Generated from function `MR::TriDist`.
    public static unsafe float TriDist(MR.Mut_Vector3f p, MR.Mut_Vector3f q, MR.Const_Vector3f? s, MR.Const_Vector3f? t)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriDist", ExactSpelling = true)]
        extern static float __MR_TriDist(MR.Mut_Vector3f._Underlying *p, MR.Mut_Vector3f._Underlying *q, MR.Const_Vector3f._Underlying *s, MR.Const_Vector3f._Underlying *t);
        return __MR_TriDist(p._UnderlyingPtr, q._UnderlyingPtr, s is not null ? s._UnderlyingPtr : null, t is not null ? t._UnderlyingPtr : null);
    }

    /// Returns closest points between an segment pair.
    /// Generated from function `MR::SegPoints`.
    public static unsafe void SegPoints(MR.Mut_Vector3f VEC, MR.Mut_Vector3f X, MR.Mut_Vector3f Y, MR.Const_Vector3f P, MR.Const_Vector3f A, MR.Const_Vector3f Q, MR.Const_Vector3f B)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SegPoints", ExactSpelling = true)]
        extern static void __MR_SegPoints(MR.Mut_Vector3f._Underlying *VEC, MR.Mut_Vector3f._Underlying *X, MR.Mut_Vector3f._Underlying *Y, MR.Const_Vector3f._Underlying *P, MR.Const_Vector3f._Underlying *A, MR.Const_Vector3f._Underlying *Q, MR.Const_Vector3f._Underlying *B);
        __MR_SegPoints(VEC._UnderlyingPtr, X._UnderlyingPtr, Y._UnderlyingPtr, P._UnderlyingPtr, A._UnderlyingPtr, Q._UnderlyingPtr, B._UnderlyingPtr);
    }
}
