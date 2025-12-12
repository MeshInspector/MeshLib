public static partial class MR
{
    // Creates regular mesh with points in valid grid lattice
    /// Generated from function `MR::makeRegularGridMesh`.
    /// Parameter `faceValidator` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MakeRegularGridMesh(ulong width, ulong height, MR.Std.Const_Function_BoolFuncFromMRUint64TMRUint64T validator, MR.Std.Const_Function_MRVector3fFuncFromMRUint64TMRUint64T positioner, MR.Std.Const_Function_BoolFuncFromMRUint64TMRUint64TMRUint64TMRUint64TMRUint64TMRUint64T? faceValidator = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeRegularGridMesh_6", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_makeRegularGridMesh_6(ulong width, ulong height, MR.Std.Const_Function_BoolFuncFromMRUint64TMRUint64T._Underlying *validator, MR.Std.Const_Function_MRVector3fFuncFromMRUint64TMRUint64T._Underlying *positioner, MR.Std.Const_Function_BoolFuncFromMRUint64TMRUint64TMRUint64TMRUint64TMRUint64TMRUint64T._Underlying *faceValidator, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_makeRegularGridMesh_6(width, height, validator._UnderlyingPtr, positioner._UnderlyingPtr, faceValidator is not null ? faceValidator._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    // Creates regular mesh from monotone (connects point with closed x, y neighbors) points
    /// Generated from function `MR::makeRegularGridMesh`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MakeRegularGridMesh(MR._ByValue_VertCoords pc, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeRegularGridMesh_2", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_makeRegularGridMesh_2(MR.Misc._PassBy pc_pass_by, MR.VertCoords._Underlying *pc, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_makeRegularGridMesh_2(pc.PassByMode, pc.Value is not null ? pc.Value._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }
}
