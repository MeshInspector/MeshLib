public static partial class MR
{
    // computes the mesh of convex hull from given input points
    /// Generated from function `MR::makeConvexHull`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeConvexHull(MR.Const_VertCoords points, MR.Const_VertBitSet validPoints)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeConvexHull_2", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeConvexHull_2(MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *validPoints);
        return MR.Misc.Move(new MR.Mesh(__MR_makeConvexHull_2(points._UnderlyingPtr, validPoints._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::makeConvexHull`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeConvexHull(MR.Const_Mesh in_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeConvexHull_1_MR_Mesh", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeConvexHull_1_MR_Mesh(MR.Const_Mesh._Underlying *in_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeConvexHull_1_MR_Mesh(in_._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::makeConvexHull`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeConvexHull(MR.Const_PointCloud in_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeConvexHull_1_MR_PointCloud", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeConvexHull_1_MR_PointCloud(MR.Const_PointCloud._Underlying *in_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeConvexHull_1_MR_PointCloud(in_._UnderlyingPtr), is_owning: true));
    }

    // computes the contour of convex hull from given input points
    /// Generated from function `MR::makeConvexHull`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector2f> MakeConvexHull(MR.Std._ByValue_Vector_MRVector2f points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeConvexHull_1_std_vector_MR_Vector2f", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector2f._Underlying *__MR_makeConvexHull_1_std_vector_MR_Vector2f(MR.Misc._PassBy points_pass_by, MR.Std.Vector_MRVector2f._Underlying *points);
        return MR.Misc.Move(new MR.Std.Vector_MRVector2f(__MR_makeConvexHull_1_std_vector_MR_Vector2f(points.PassByMode, points.Value is not null ? points.Value._UnderlyingPtr : null), is_owning: true));
    }
}
