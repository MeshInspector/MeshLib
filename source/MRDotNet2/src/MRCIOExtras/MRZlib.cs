public static partial class MR
{
    /**
    * @brief compress the input data using the Deflate algorithm
    * @param in - input data stream
    * @param out - output data stream
    * @param level - compression level (0 - no compression, 1 - the fastest but the most inefficient compression, 9 - the most efficient but the slowest compression)
    * @return nothing or error string
    */
    /// Generated from function `MR::zlibCompressStream`.
    /// Parameter `level` defaults to `-1`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ZlibCompressStream(MR.Std.Istream in_, MR.Std.Ostream out_, int? level = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_zlibCompressStream", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_zlibCompressStream(MR.Std.Istream._Underlying *in_, MR.Std.Ostream._Underlying *out_, int *level);
        int __deref_level = level.GetValueOrDefault();
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_zlibCompressStream(in_._UnderlyingPtr, out_._UnderlyingPtr, level.HasValue ? &__deref_level : null), is_owning: true));
    }

    /**
    * /brief decompress the input data compressed using the Deflate algorithm
    * @param in - input data stream
    * @param out - output data stream
    * @return nothing or error string
    */
    /// Generated from function `MR::zlibDecompressStream`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ZlibDecompressStream(MR.Std.Istream in_, MR.Std.Ostream out_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_zlibDecompressStream", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_zlibDecompressStream(MR.Std.Istream._Underlying *in_, MR.Std.Ostream._Underlying *out_);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_zlibDecompressStream(in_._UnderlyingPtr, out_._UnderlyingPtr), is_owning: true));
    }
}
