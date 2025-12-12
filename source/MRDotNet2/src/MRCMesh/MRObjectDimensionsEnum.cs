public static partial class MR
{
    // This controls which dimensions are shown on top of an object as arrows.
    public enum DimensionsVisualizePropertyType : int
    {
        Diameter = 0,
        Angle = 1,
        Length = 2,
        Count = 3,
    }

    /// Generated from function `MR::toString`.
    public static unsafe MR.Std.StringView ToString(MR.DimensionsVisualizePropertyType value)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_toString_MR_DimensionsVisualizePropertyType", ExactSpelling = true)]
        extern static MR.Std.StringView._Underlying *__MR_toString_MR_DimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType value);
        return new(__MR_toString_MR_DimensionsVisualizePropertyType(value), is_owning: true);
    }
}
