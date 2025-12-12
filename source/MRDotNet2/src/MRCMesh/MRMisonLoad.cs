public static partial class MR
{
    /// load scene from MISON file \n
    /// JSON file with array named "Objects" or root array: \n
    /// element fields:\n
    ///    "Filename" : required full path to file for loading object
    ///    "XF": optional xf for loaded object
    ///    "Name": optional name for loaded object
    /// Generated from function `MR::fromSceneMison`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> FromSceneMison(ReadOnlySpan<char> path, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fromSceneMison_std_filesystem_path", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_fromSceneMison_std_filesystem_path(byte *path, byte *path_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_fromSceneMison_std_filesystem_path(__ptr_path, __ptr_path + __len_path, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
        }
    }

    /// Generated from function `MR::fromSceneMison`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> FromSceneMison(MR.Std.Istream in_, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fromSceneMison_std_istream", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_fromSceneMison_std_istream(MR.Std.Istream._Underlying *in_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_fromSceneMison_std_istream(in_._UnderlyingPtr, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
    }
}
