public static partial class MR
{
    //Draws cylinder with radius 'radius', height - 'length', its base have 'resolution' sides
    /// Generated from function `MR::makeCylinder`.
    /// Parameter `radius` defaults to `0.100000001f`.
    /// Parameter `length` defaults to `1.0f`.
    /// Parameter `resolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeCylinder(float? radius = null, float? length = null, int? resolution = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeCylinder", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeCylinder(float *radius, float *length, int *resolution);
        float __deref_radius = radius.GetValueOrDefault();
        float __deref_length = length.GetValueOrDefault();
        int __deref_resolution = resolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeCylinder(radius.HasValue ? &__deref_radius : null, length.HasValue ? &__deref_length : null, resolution.HasValue ? &__deref_resolution : null), is_owning: true));
    }

    // A hollow cylinder.
    /// Generated from function `MR::makeOpenCylinder`.
    /// Parameter `radius` defaults to `1`.
    /// Parameter `z1` defaults to `-1`.
    /// Parameter `z2` defaults to `1`.
    /// Parameter `numCircleSegments` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeOpenCylinder(float? radius = null, float? z1 = null, float? z2 = null, int? numCircleSegments = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeOpenCylinder", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeOpenCylinder(float *radius, float *z1, float *z2, int *numCircleSegments);
        float __deref_radius = radius.GetValueOrDefault();
        float __deref_z1 = z1.GetValueOrDefault();
        float __deref_z2 = z2.GetValueOrDefault();
        int __deref_numCircleSegments = numCircleSegments.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeOpenCylinder(radius.HasValue ? &__deref_radius : null, z1.HasValue ? &__deref_z1 : null, z2.HasValue ? &__deref_z2 : null, numCircleSegments.HasValue ? &__deref_numCircleSegments : null), is_owning: true));
    }

    // A hollow cone.
    /// Generated from function `MR::makeOpenCone`.
    /// Parameter `radius` defaults to `1`.
    /// Parameter `zApex` defaults to `0`.
    /// Parameter `zBase` defaults to `1`.
    /// Parameter `numCircleSegments` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeOpenCone(float? radius = null, float? zApex = null, float? zBase = null, int? numCircleSegments = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeOpenCone", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeOpenCone(float *radius, float *zApex, float *zBase, int *numCircleSegments);
        float __deref_radius = radius.GetValueOrDefault();
        float __deref_zApex = zApex.GetValueOrDefault();
        float __deref_zBase = zBase.GetValueOrDefault();
        int __deref_numCircleSegments = numCircleSegments.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeOpenCone(radius.HasValue ? &__deref_radius : null, zApex.HasValue ? &__deref_zApex : null, zBase.HasValue ? &__deref_zBase : null, numCircleSegments.HasValue ? &__deref_numCircleSegments : null), is_owning: true));
    }

    /// Generated from function `MR::makeCylinderAdvanced`.
    /// Parameter `radius0` defaults to `0.100000001f`.
    /// Parameter `radius1` defaults to `0.100000001f`.
    /// Parameter `start_angle` defaults to `0.0f`.
    /// Parameter `arc_size` defaults to `2.0f*PI_F`.
    /// Parameter `length` defaults to `1.0f`.
    /// Parameter `resolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeCylinderAdvanced(float? radius0 = null, float? radius1 = null, float? start_angle = null, float? arc_size = null, float? length = null, int? resolution = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeCylinderAdvanced", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeCylinderAdvanced(float *radius0, float *radius1, float *start_angle, float *arc_size, float *length, int *resolution);
        float __deref_radius0 = radius0.GetValueOrDefault();
        float __deref_radius1 = radius1.GetValueOrDefault();
        float __deref_start_angle = start_angle.GetValueOrDefault();
        float __deref_arc_size = arc_size.GetValueOrDefault();
        float __deref_length = length.GetValueOrDefault();
        int __deref_resolution = resolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeCylinderAdvanced(radius0.HasValue ? &__deref_radius0 : null, radius1.HasValue ? &__deref_radius1 : null, start_angle.HasValue ? &__deref_start_angle : null, arc_size.HasValue ? &__deref_arc_size : null, length.HasValue ? &__deref_length : null, resolution.HasValue ? &__deref_resolution : null), is_owning: true));
    }

    // Makes cone mesh by calling makeCylinderAdvanced with the top radius 0.
    /// Generated from function `MR::makeCone`.
    /// Parameter `radius0` defaults to `0.100000001f`.
    /// Parameter `length` defaults to `1.0f`.
    /// Parameter `resolution` defaults to `32`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeCone(float? radius0 = null, float? length = null, int? resolution = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeCone", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeCone(float *radius0, float *length, int *resolution);
        float __deref_radius0 = radius0.GetValueOrDefault();
        float __deref_length = length.GetValueOrDefault();
        int __deref_resolution = resolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeCone(radius0.HasValue ? &__deref_radius0 : null, length.HasValue ? &__deref_length : null, resolution.HasValue ? &__deref_resolution : null), is_owning: true));
    }
}
