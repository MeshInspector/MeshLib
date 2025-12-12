public static partial class MR
{
    // returns offsets for each new line in monolith char block
    /// Generated from function `MR::splitByLines`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUint64T> SplitByLines(byte? data, ulong size)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_splitByLines", ExactSpelling = true)]
        extern static MR.Std.Vector_MRUint64T._Underlying *__MR_splitByLines(byte *data, ulong size);
        byte __deref_data = data.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MRUint64T(__MR_splitByLines(data.HasValue ? &__deref_data : null, size), is_owning: true));
    }

    // reads the first integer number in the line
    /// Generated from function `MR::parseFirstNum`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ParseFirstNum(ReadOnlySpan<char> str, ref int num)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_parseFirstNum", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_parseFirstNum(byte *str, byte *str_end, int *num);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            fixed (int *__ptr_num = &num)
            {
                return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_parseFirstNum(__ptr_str, __ptr_str + __len_str, __ptr_num), is_owning: true));
            }
        }
    }

    // reads the polygon points and optional number of polygon points
    // example
    // N vertex0 vertex1 ... vertexN
    /// Generated from function `MR::parsePolygon`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ParsePolygon(ReadOnlySpan<char> str, MR.Mut_VertId? vertId, MR.Misc.InOut<int>? numPoints)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_parsePolygon", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_parsePolygon(byte *str, byte *str_end, MR.Mut_VertId._Underlying *vertId, int *numPoints);
        byte[] __bytes_str = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(str.Length)];
        int __len_str = System.Text.Encoding.UTF8.GetBytes(str, __bytes_str);
        fixed (byte *__ptr_str = __bytes_str)
        {
            int __value_numPoints = numPoints is not null ? numPoints.Value : default(int);
            var __ret = __MR_parsePolygon(__ptr_str, __ptr_str + __len_str, vertId is not null ? vertId._UnderlyingPtr : null, numPoints is not null ? &__value_numPoints : null);
            if (numPoints is not null) numPoints.Value = __value_numPoints;
            return MR.Misc.Move(new MR.Expected_Void_StdString(__ret, is_owning: true));
        }
    }
}
