public static partial class MR
{
    /**
    * \brief write dataSize bytes from data to out stream by blocks blockSize bytes
    * \details if progress callback is not set, write all data by one block
    * \return false if process was canceled (callback is set and return false )
    */
    /// Generated from function `MR::writeByBlocks`.
    /// Parameter `callback` defaults to `{}`.
    /// Parameter `blockSize` defaults to `(size_t(1)<<16)`.
    public static unsafe bool WriteByBlocks(MR.Std.Ostream out_, byte? data, ulong dataSize, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null, ulong? blockSize = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_writeByBlocks", ExactSpelling = true)]
        extern static byte __MR_writeByBlocks(MR.Std.Ostream._Underlying *out_, byte *data, ulong dataSize, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback, ulong *blockSize);
        byte __deref_data = data.GetValueOrDefault();
        ulong __deref_blockSize = blockSize.GetValueOrDefault();
        return __MR_writeByBlocks(out_._UnderlyingPtr, data.HasValue ? &__deref_data : null, dataSize, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null, blockSize.HasValue ? &__deref_blockSize : null) != 0;
    }

    /**
    * \brief read dataSize bytes from in stream to data by blocks blockSize bytes
    * \details if progress callback is not set, read all data by one block
    * \return false if process was canceled (callback is set and return false )
    */
    /// Generated from function `MR::readByBlocks`.
    /// Parameter `callback` defaults to `{}`.
    /// Parameter `blockSize` defaults to `(size_t(1)<<16)`.
    public static unsafe bool ReadByBlocks(MR.Std.Istream in_, MR.Misc.InOut<byte>? data, ulong dataSize, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null, ulong? blockSize = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_readByBlocks", ExactSpelling = true)]
        extern static byte __MR_readByBlocks(MR.Std.Istream._Underlying *in_, byte *data, ulong dataSize, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback, ulong *blockSize);
        byte __value_data = data is not null ? data.Value : default(byte);
        ulong __deref_blockSize = blockSize.GetValueOrDefault();
        var __ret = __MR_readByBlocks(in_._UnderlyingPtr, data is not null ? &__value_data : null, dataSize, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null, blockSize.HasValue ? &__deref_blockSize : null);
        if (data is not null) data.Value = __value_data;
        return __ret != 0;
    }
}
