public static partial class MR
{
    /// Finds all valid points of pointCloud that are inside or on the surface of given ball until callback returns Stop;
    /// the ball can shrink (new ball is always within the previous one) during the search but never expand
    /// \param xf points-to-center transformation, if not specified then identity transformation is assumed
    /// Generated from function `MR::findPointsInBall`.
    public static unsafe void FindPointsInBall(MR.Const_PointCloud pointCloud, MR.Const_Ball3f ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRPointsProjectionResultRefConstMRVector3fRefMRBall3fRef foundCallback, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findPointsInBall_MR_PointCloud", ExactSpelling = true)]
        extern static void __MR_findPointsInBall_MR_PointCloud(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_Ball3f._Underlying *ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRPointsProjectionResultRefConstMRVector3fRefMRBall3fRef._Underlying *foundCallback, MR.Const_AffineXf3f._Underlying *xf);
        __MR_findPointsInBall_MR_PointCloud(pointCloud._UnderlyingPtr, ball._UnderlyingPtr, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Finds all valid vertices of the mesh that are inside or on the surface of given ball until callback returns Stop;
    /// the ball can shrink (new ball is always within the previous one) during the search but never expand
    /// \param xf points-to-center transformation, if not specified then identity transformation is assumed
    /// Generated from function `MR::findPointsInBall`.
    public static unsafe void FindPointsInBall(MR.Const_Mesh mesh, MR.Const_Ball3f ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRPointsProjectionResultRefConstMRVector3fRefMRBall3fRef foundCallback, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findPointsInBall_MR_Mesh", ExactSpelling = true)]
        extern static void __MR_findPointsInBall_MR_Mesh(MR.Const_Mesh._Underlying *mesh, MR.Const_Ball3f._Underlying *ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRPointsProjectionResultRefConstMRVector3fRefMRBall3fRef._Underlying *foundCallback, MR.Const_AffineXf3f._Underlying *xf);
        __MR_findPointsInBall_MR_Mesh(mesh._UnderlyingPtr, ball._UnderlyingPtr, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Finds all points in tree that are inside or on the surface of given ball until callback returns Stop;
    /// the ball can shrink (new ball is always within the previous one) during the search but never expand
    /// \param xf points-to-center transformation, if not specified then identity transformation is assumed
    /// Generated from function `MR::findPointsInBall`.
    public static unsafe void FindPointsInBall(MR.Const_AABBTreePoints tree, MR.Const_Ball3f ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRPointsProjectionResultRefConstMRVector3fRefMRBall3fRef foundCallback, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findPointsInBall_MR_AABBTreePoints", ExactSpelling = true)]
        extern static void __MR_findPointsInBall_MR_AABBTreePoints(MR.Const_AABBTreePoints._Underlying *tree, MR.Ball3f._Underlying *ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRPointsProjectionResultRefConstMRVector3fRefMRBall3fRef._Underlying *foundCallback, MR.Const_AffineXf3f._Underlying *xf);
        __MR_findPointsInBall_MR_AABBTreePoints(tree._UnderlyingPtr, ball._UnderlyingPtr, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }
}
