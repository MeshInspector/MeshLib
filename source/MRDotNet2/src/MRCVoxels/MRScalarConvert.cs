public static partial class MR
{
    /// scalar value's binary format type
    public enum ScalarType : int
    {
        UInt8 = 0,
        Int8 = 1,
        UInt16 = 2,
        Int16 = 3,
        UInt32 = 4,
        Int32 = 5,
        UInt64 = 6,
        Int64 = 7,
        Float32 = 8,
        Float64 = 9,
        ///< the last value from float[4]
        Float32_4 = 10,
        Unknown = 11,
        Count = 12,
    }

    /// get a function to convert binary data of specified format type to a scalar value
    /// \param scalarType - binary format type
    /// \param range - (for integer types only) the range of possible values
    /// \param min - (for integer types only) the minimal value
    /// Generated from function `MR::getTypeConverter`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromConstCharPtr> GetTypeConverter(MR.ScalarType scalarType, ulong range, long min)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getTypeConverter", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromConstCharPtr._Underlying *__MR_getTypeConverter(MR.ScalarType scalarType, ulong range, long min);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromConstCharPtr(__MR_getTypeConverter(scalarType, range, min), is_owning: true));
    }
}
