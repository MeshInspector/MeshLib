public static partial class MR
{
    /// saves mesh with optional selection to mru format;
    /// this is very convenient for saving intermediate states during algorithm debugging;
    /// ".mrmesh" save mesh format is not space efficient, but guaranties no changes in the topology after loading
    /// Generated from function `MR::serializeMesh`.
    /// Parameter `serializeFormat` defaults to `".mrmesh"`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SerializeMesh(MR.Const_Mesh mesh, ReadOnlySpan<char> path, MR.Const_FaceBitSet? selection = null, MR.Misc._InOpt<byte>? serializeFormat = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_serializeMesh", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_serializeMesh(MR.Const_Mesh._Underlying *mesh, byte *path, byte *path_end, MR.Const_FaceBitSet._Underlying *selection, byte **serializeFormat);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            byte __value_serializeFormat = serializeFormat is not null && serializeFormat.Opt is not null ? serializeFormat.Opt.Value : default(byte);
            byte *__valueptr_serializeFormat = serializeFormat is not null && serializeFormat.Opt is not null ? &__value_serializeFormat : null;
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_serializeMesh(mesh._UnderlyingPtr, __ptr_path, __ptr_path + __len_path, selection is not null ? selection._UnderlyingPtr : null, serializeFormat is not null ? &__valueptr_serializeFormat : null), is_owning: true));
        }
    }
}
