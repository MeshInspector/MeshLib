public static partial class MR
{
    // loads scene from 3MF file in a new container object
    /// Generated from function `MR::deserializeObjectTreeFrom3mf`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> DeserializeObjectTreeFrom3mf(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deserializeObjectTreeFrom3mf", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_deserializeObjectTreeFrom3mf(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_deserializeObjectTreeFrom3mf(__ptr_file, __ptr_file + __len_file, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
        }
    }

    // loads scene from .model file in a new container object
    /// Generated from function `MR::deserializeObjectTreeFromModel`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRLoadedObjectT_StdString> DeserializeObjectTreeFromModel(ReadOnlySpan<char> file, MR.Std.Const_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deserializeObjectTreeFromModel", ExactSpelling = true)]
        extern static MR.Expected_MRLoadedObjectT_StdString._Underlying *__MR_deserializeObjectTreeFromModel(byte *file, byte *file_end, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_MRLoadedObjectT_StdString(__MR_deserializeObjectTreeFromModel(__ptr_file, __ptr_file + __len_file, callback is not null ? callback._UnderlyingPtr : null), is_owning: true));
        }
    }
}
