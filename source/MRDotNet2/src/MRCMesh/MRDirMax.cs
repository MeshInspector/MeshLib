public static partial class MR
{
    /// finds the vertex in the mesh part having the largest projection on given direction,
    /// optionally uses aabb-tree inside for faster computation
    /// Generated from function `MR::findDirMax`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector3f dir, MR.Const_Mesh m, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_3_const_MR_Vector3f_ref_MR_Mesh", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_3_const_MR_Vector3f_ref_MR_Mesh(MR.Const_Vector3f._Underlying *dir, MR.Const_Mesh._Underlying *m, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_findDirMax_3_const_MR_Vector3f_ref_MR_Mesh(dir._UnderlyingPtr, m._UnderlyingPtr, u.HasValue ? &__deref_u : null);
    }

    /// finds the vertex in the mesh part having the largest projection on given direction,
    /// optionally uses aabb-tree inside for faster computation
    /// Generated from function `MR::findDirMax`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector3f dir, MR.Const_MeshPart mp, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_3_const_MR_Vector3f_ref_MR_MeshPart", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_3_const_MR_Vector3f_ref_MR_MeshPart(MR.Const_Vector3f._Underlying *dir, MR.Const_MeshPart._Underlying *mp, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_findDirMax_3_const_MR_Vector3f_ref_MR_MeshPart(dir._UnderlyingPtr, mp._UnderlyingPtr, u.HasValue ? &__deref_u : null);
    }

    /// finds the vertex in the mesh part having the largest projection on given direction,
    /// optionally uses aabb-points-tree inside for faster computation
    /// Generated from function `MR::findDirMax`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector3f dir, MR.Const_MeshVertPart mp, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_3_const_MR_Vector3f_ref_MR_MeshVertPart", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_3_const_MR_Vector3f_ref_MR_MeshVertPart(MR.Const_Vector3f._Underlying *dir, MR.Const_MeshVertPart._Underlying *mp, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_findDirMax_3_const_MR_Vector3f_ref_MR_MeshVertPart(dir._UnderlyingPtr, mp._UnderlyingPtr, u.HasValue ? &__deref_u : null);
    }

    /// finds the vertex in the polyline having the largest projection on given direction,
    /// optionally uses aabb-tree inside for faster computation
    /// Generated from function `MR::findDirMax`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector3f dir, MR.Const_Polyline3 polyline, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_3_const_MR_Vector3f_ref_MR_Polyline3", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_3_const_MR_Vector3f_ref_MR_Polyline3(MR.Const_Vector3f._Underlying *dir, MR.Const_Polyline3._Underlying *polyline, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_findDirMax_3_const_MR_Vector3f_ref_MR_Polyline3(dir._UnderlyingPtr, polyline._UnderlyingPtr, u.HasValue ? &__deref_u : null);
    }

    /// finds the vertex in the polyline having the largest projection on given direction,
    /// optionally uses aabb-tree inside for faster computation
    /// Generated from function `MR::findDirMax`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector2f dir, MR.Const_Polyline2 polyline, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_3_const_MR_Vector2f_ref", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_3_const_MR_Vector2f_ref(MR.Const_Vector2f._Underlying *dir, MR.Const_Polyline2._Underlying *polyline, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_findDirMax_3_const_MR_Vector2f_ref(dir._UnderlyingPtr, polyline._UnderlyingPtr, u.HasValue ? &__deref_u : null);
    }

    /// finds the point in the cloud having the largest projection on given direction,
    /// optionally uses aabb-tree inside for faster computation
    /// Generated from function `MR::findDirMax`.
    /// Parameter `u` defaults to `UseAABBTree::Yes`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector3f dir, MR.Const_PointCloud cloud, MR.Const_VertBitSet? region = null, MR.UseAABBTree? u = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_4", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_4(MR.Const_Vector3f._Underlying *dir, MR.Const_PointCloud._Underlying *cloud, MR.Const_VertBitSet._Underlying *region, MR.UseAABBTree *u);
        MR.UseAABBTree __deref_u = u.GetValueOrDefault();
        return __MR_findDirMax_4(dir._UnderlyingPtr, cloud._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, u.HasValue ? &__deref_u : null);
    }

    /// finds the point in the tree having the largest projection on given direction
    /// Generated from function `MR::findDirMax`.
    public static unsafe MR.VertId FindDirMax(MR.Const_Vector3f dir, MR.Const_AABBTreePoints tree, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDirMax_3_const_MR_Vector3f_ref_MR_AABBTreePoints", ExactSpelling = true)]
        extern static MR.VertId __MR_findDirMax_3_const_MR_Vector3f_ref_MR_AABBTreePoints(MR.Const_Vector3f._Underlying *dir, MR.Const_AABBTreePoints._Underlying *tree, MR.Const_VertBitSet._Underlying *region);
        return __MR_findDirMax_3_const_MR_Vector3f_ref_MR_AABBTreePoints(dir._UnderlyingPtr, tree._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }
}
