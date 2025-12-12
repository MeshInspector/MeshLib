public static partial class MR
{
    /// Makes a solid-of-revolution mesh. The resulting mesh is symmetrical about the z-axis.
    /// The profile points must be in the format { distance to the z-axis; z value }.
    /// Generated from function `MR::makeSolidOfRevolution`.
    /// Parameter `resolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeSolidOfRevolution(MR.Std.Const_Vector_MRVector2f profile, int? resolution = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSolidOfRevolution", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeSolidOfRevolution(MR.Std.Const_Vector_MRVector2f._Underlying *profile, int *resolution);
        int __deref_resolution = resolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeSolidOfRevolution(profile._UnderlyingPtr, resolution.HasValue ? &__deref_resolution : null), is_owning: true));
    }
}
