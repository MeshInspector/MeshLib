public static partial class MR
{
    /// <summary>
    /// Returns contours in \p mesh space that are offset from \p surfaceLine on \p offset amount in all directions
    /// </summary>
    /// <param name="mesh">mesh to perform offset on</param>
    /// <param name="surfaceLine">surface line to perofrm offset from</param>
    /// <param name="offset">amount of offset, note that absolute value is used</param>
    /// <returns>resulting offset contours or error if something goes wrong</returns>
    /// Generated from function `MR::offsetSurfaceLine`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdVectorMRVector3f_StdString> OffsetSurfaceLine(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint surfaceLine, float offset)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetSurfaceLine_float", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *__MR_offsetSurfaceLine_float(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *surfaceLine, float offset);
        return MR.Misc.Move(new MR.Expected_StdVectorStdVectorMRVector3f_StdString(__MR_offsetSurfaceLine_float(mesh._UnderlyingPtr, surfaceLine._UnderlyingPtr, offset), is_owning: true));
    }

    /// <summary>
    /// Returns contours in \p mesh space that are offset from \p surfaceLine on \p offsetAtPoint amount in all directions
    /// </summary>
    /// <param name="mesh">mesh to perform offset on</param>
    /// <param name="surfaceLine">surface line to perofrm offset from</param>
    /// <param name="offsetAtPoint">function that can return different amount of offset in different point (argument is index of point in \p surfaceLine), note that absolute value is used</param>
    /// <returns>resulting offset contours or error if something goes wrong</returns>
    /// Generated from function `MR::offsetSurfaceLine`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdVectorMRVector3f_StdString> OffsetSurfaceLine(MR.Const_Mesh mesh, MR.Std.Const_Vector_MRMeshTriPoint surfaceLine, MR.Std.Const_Function_FloatFuncFromInt offsetAtPoint)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetSurfaceLine_std_function_float_func_from_int", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *__MR_offsetSurfaceLine_std_function_float_func_from_int(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MRMeshTriPoint._Underlying *surfaceLine, MR.Std.Const_Function_FloatFuncFromInt._Underlying *offsetAtPoint);
        return MR.Misc.Move(new MR.Expected_StdVectorStdVectorMRVector3f_StdString(__MR_offsetSurfaceLine_std_function_float_func_from_int(mesh._UnderlyingPtr, surfaceLine._UnderlyingPtr, offsetAtPoint._UnderlyingPtr), is_owning: true));
    }
}
