public static partial class MR
{
    // loads scene from glTF file in a new container object
    /// Generated from function `MR::deserializeObjectTreeFromGltf`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdSharedPtrMRObject_StdString> DeserializeObjectTreeFromGltf(ReadOnlySpan<char> file, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deserializeObjectTreeFromGltf", ExactSpelling = true)]
        extern static MR.Expected_StdSharedPtrMRObject_StdString._Underlying *__MR_deserializeObjectTreeFromGltf(byte *file, byte *file_end, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_StdSharedPtrMRObject_StdString(__MR_deserializeObjectTreeFromGltf(__ptr_file, __ptr_file + __len_file, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }
    }

    // saves scene to a glTF file
    /// Generated from function `MR::serializeObjectTreeToGltf`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SerializeObjectTreeToGltf(MR.Const_Object root, ReadOnlySpan<char> file, MR.ObjectSave.Const_Settings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_serializeObjectTreeToGltf", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_serializeObjectTreeToGltf(MR.Const_Object._Underlying *root, byte *file, byte *file_end, MR.ObjectSave.Const_Settings._Underlying *settings);
        byte[] __bytes_file = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(file.Length)];
        int __len_file = System.Text.Encoding.UTF8.GetBytes(file, __bytes_file);
        fixed (byte *__ptr_file = __bytes_file)
        {
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_serializeObjectTreeToGltf(root._UnderlyingPtr, __ptr_file, __ptr_file + __len_file, settings._UnderlyingPtr), is_owning: true));
        }
    }
}
