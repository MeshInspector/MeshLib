public static partial class MR
{
    /// Finds all valid points of pointCloud that are inside or on the surface of given box
    /// \param xf points-to-center transformation, if not specified then identity transformation is assumed
    /// Generated from function `MR::findPointsInBox`.
    public static unsafe void FindPointsInBox(MR.Const_PointCloud pointCloud, MR.Const_Box3f box, MR.Std.Const_Function_VoidFuncFromMRVertIdConstMRVector3fRef foundCallback, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findPointsInBox_MR_PointCloud", ExactSpelling = true)]
        extern static void __MR_findPointsInBox_MR_PointCloud(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_Box3f._Underlying *box, MR.Std.Const_Function_VoidFuncFromMRVertIdConstMRVector3fRef._Underlying *foundCallback, MR.Const_AffineXf3f._Underlying *xf);
        __MR_findPointsInBox_MR_PointCloud(pointCloud._UnderlyingPtr, box._UnderlyingPtr, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Finds all valid vertices of the mesh that are inside or on the surface of given box
    /// \param xf points-to-center transformation, if not specified then identity transformation is assumed
    /// Generated from function `MR::findPointsInBox`.
    public static unsafe void FindPointsInBox(MR.Const_Mesh mesh, MR.Const_Box3f box, MR.Std.Const_Function_VoidFuncFromMRVertIdConstMRVector3fRef foundCallback, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findPointsInBox_MR_Mesh", ExactSpelling = true)]
        extern static void __MR_findPointsInBox_MR_Mesh(MR.Const_Mesh._Underlying *mesh, MR.Const_Box3f._Underlying *box, MR.Std.Const_Function_VoidFuncFromMRVertIdConstMRVector3fRef._Underlying *foundCallback, MR.Const_AffineXf3f._Underlying *xf);
        __MR_findPointsInBox_MR_Mesh(mesh._UnderlyingPtr, box._UnderlyingPtr, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Finds all points in tree that are inside or on the surface of given box
    /// \param xf points-to-center transformation, if not specified then identity transformation is assumed
    /// Generated from function `MR::findPointsInBox`.
    public static unsafe void FindPointsInBox(MR.Const_AABBTreePoints tree, MR.Const_Box3f box, MR.Std.Const_Function_VoidFuncFromMRVertIdConstMRVector3fRef foundCallback, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findPointsInBox_MR_AABBTreePoints", ExactSpelling = true)]
        extern static void __MR_findPointsInBox_MR_AABBTreePoints(MR.Const_AABBTreePoints._Underlying *tree, MR.Const_Box3f._Underlying *box, MR.Std.Const_Function_VoidFuncFromMRVertIdConstMRVector3fRef._Underlying *foundCallback, MR.Const_AffineXf3f._Underlying *xf);
        __MR_findPointsInBox_MR_AABBTreePoints(tree._UnderlyingPtr, box._UnderlyingPtr, foundCallback._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }
}
