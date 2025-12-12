public static partial class MR
{
    /// encodes binary data into textual Base64 format
    /// Generated from function `MR::encode64`.
    public static unsafe MR.Misc._Moved<MR.Std.String> Encode64(byte? data, ulong size)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_encode64", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_encode64(byte *data, ulong size);
        byte __deref_data = data.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.String(__MR_encode64(data.HasValue ? &__deref_data : null, size), is_owning: true));
    }

    /// decodes Base64 format into binary data
    /// Generated from function `MR::decode64`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_UnsignedChar> Decode64(ReadOnlySpan<char> val)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decode64", ExactSpelling = true)]
        extern static MR.Std.Vector_UnsignedChar._Underlying *__MR_decode64(byte *val, byte *val_end);
        byte[] __bytes_val = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(val.Length)];
        int __len_val = System.Text.Encoding.UTF8.GetBytes(val, __bytes_val);
        fixed (byte *__ptr_val = __bytes_val)
        {
            return MR.Misc.Move(new MR.Std.Vector_UnsignedChar(__MR_decode64(__ptr_val, __ptr_val + __len_val), is_owning: true));
        }
    }
}
