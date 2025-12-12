public static partial class MR
{
    /// quickly tests whether given float is not-a-number
    /// Generated from function `MR::isNanFast`.
    public static bool IsNanFast(float f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isNanFast", ExactSpelling = true)]
        extern static byte __MR_isNanFast(float f);
        return __MR_isNanFast(f) != 0;
    }
}
