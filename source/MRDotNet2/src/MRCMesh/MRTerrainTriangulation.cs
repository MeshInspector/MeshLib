public static partial class MR
{
    /// Creates Delaunay triangulation using only XY components of points 
    /// points will be changed inside this function take argument by value
    /// Generated from function `MR::terrainTriangulation`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> TerrainTriangulation(MR.Std._ByValue_Vector_MRVector3f points, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_terrainTriangulation", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_terrainTriangulation(MR.Misc._PassBy points_pass_by, MR.Std.Vector_MRVector3f._Underlying *points, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_terrainTriangulation(points.PassByMode, points.Value is not null ? points.Value._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }
}
