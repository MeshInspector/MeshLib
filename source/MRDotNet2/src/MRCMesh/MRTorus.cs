public static partial class MR
{
    // Z is symmetry axis of this torus
    // points - optional out points of main circle
    /// Generated from function `MR::makeTorus`.
    /// Parameter `primaryRadius` defaults to `1.0f`.
    /// Parameter `secondaryRadius` defaults to `0.100000001f`.
    /// Parameter `primaryResolution` defaults to `16`.
    /// Parameter `secondaryResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeTorus(float? primaryRadius = null, float? secondaryRadius = null, int? primaryResolution = null, int? secondaryResolution = null, MR.Std.Vector_MRVector3f? points = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeTorus", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeTorus(float *primaryRadius, float *secondaryRadius, int *primaryResolution, int *secondaryResolution, MR.Std.Vector_MRVector3f._Underlying *points);
        float __deref_primaryRadius = primaryRadius.GetValueOrDefault();
        float __deref_secondaryRadius = secondaryRadius.GetValueOrDefault();
        int __deref_primaryResolution = primaryResolution.GetValueOrDefault();
        int __deref_secondaryResolution = secondaryResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeTorus(primaryRadius.HasValue ? &__deref_primaryRadius : null, secondaryRadius.HasValue ? &__deref_secondaryRadius : null, primaryResolution.HasValue ? &__deref_primaryResolution : null, secondaryResolution.HasValue ? &__deref_secondaryResolution : null, points is not null ? points._UnderlyingPtr : null), is_owning: true));
    }

    // creates torus without inner half faces
    // main application - testing fillHole and Stitch
    /// Generated from function `MR::makeOuterHalfTorus`.
    /// Parameter `primaryRadius` defaults to `1.0f`.
    /// Parameter `secondaryRadius` defaults to `0.100000001f`.
    /// Parameter `primaryResolution` defaults to `16`.
    /// Parameter `secondaryResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeOuterHalfTorus(float? primaryRadius = null, float? secondaryRadius = null, int? primaryResolution = null, int? secondaryResolution = null, MR.Std.Vector_MRVector3f? points = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeOuterHalfTorus", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeOuterHalfTorus(float *primaryRadius, float *secondaryRadius, int *primaryResolution, int *secondaryResolution, MR.Std.Vector_MRVector3f._Underlying *points);
        float __deref_primaryRadius = primaryRadius.GetValueOrDefault();
        float __deref_secondaryRadius = secondaryRadius.GetValueOrDefault();
        int __deref_primaryResolution = primaryResolution.GetValueOrDefault();
        int __deref_secondaryResolution = secondaryResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeOuterHalfTorus(primaryRadius.HasValue ? &__deref_primaryRadius : null, secondaryRadius.HasValue ? &__deref_secondaryRadius : null, primaryResolution.HasValue ? &__deref_primaryResolution : null, secondaryResolution.HasValue ? &__deref_secondaryResolution : null, points is not null ? points._UnderlyingPtr : null), is_owning: true));
    }

    // creates torus with inner protruding half as undercut
    // main application - testing fixUndercuts
    /// Generated from function `MR::makeTorusWithUndercut`.
    /// Parameter `primaryRadius` defaults to `1.0f`.
    /// Parameter `secondaryRadiusInner` defaults to `0.100000001f`.
    /// Parameter `secondaryRadiusOuter` defaults to `0.200000003f`.
    /// Parameter `primaryResolution` defaults to `16`.
    /// Parameter `secondaryResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeTorusWithUndercut(float? primaryRadius = null, float? secondaryRadiusInner = null, float? secondaryRadiusOuter = null, int? primaryResolution = null, int? secondaryResolution = null, MR.Std.Vector_MRVector3f? points = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeTorusWithUndercut", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeTorusWithUndercut(float *primaryRadius, float *secondaryRadiusInner, float *secondaryRadiusOuter, int *primaryResolution, int *secondaryResolution, MR.Std.Vector_MRVector3f._Underlying *points);
        float __deref_primaryRadius = primaryRadius.GetValueOrDefault();
        float __deref_secondaryRadiusInner = secondaryRadiusInner.GetValueOrDefault();
        float __deref_secondaryRadiusOuter = secondaryRadiusOuter.GetValueOrDefault();
        int __deref_primaryResolution = primaryResolution.GetValueOrDefault();
        int __deref_secondaryResolution = secondaryResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeTorusWithUndercut(primaryRadius.HasValue ? &__deref_primaryRadius : null, secondaryRadiusInner.HasValue ? &__deref_secondaryRadiusInner : null, secondaryRadiusOuter.HasValue ? &__deref_secondaryRadiusOuter : null, primaryResolution.HasValue ? &__deref_primaryResolution : null, secondaryResolution.HasValue ? &__deref_secondaryResolution : null, points is not null ? points._UnderlyingPtr : null), is_owning: true));
    }

    // creates torus with some handed-up points
    // main application - testing fixSpikes and Relax
    /// Generated from function `MR::makeTorusWithSpikes`.
    /// Parameter `primaryRadius` defaults to `1.0f`.
    /// Parameter `secondaryRadiusInner` defaults to `0.100000001f`.
    /// Parameter `secondaryRadiusOuter` defaults to `0.5f`.
    /// Parameter `primaryResolution` defaults to `16`.
    /// Parameter `secondaryResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeTorusWithSpikes(float? primaryRadius = null, float? secondaryRadiusInner = null, float? secondaryRadiusOuter = null, int? primaryResolution = null, int? secondaryResolution = null, MR.Std.Vector_MRVector3f? points = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeTorusWithSpikes", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeTorusWithSpikes(float *primaryRadius, float *secondaryRadiusInner, float *secondaryRadiusOuter, int *primaryResolution, int *secondaryResolution, MR.Std.Vector_MRVector3f._Underlying *points);
        float __deref_primaryRadius = primaryRadius.GetValueOrDefault();
        float __deref_secondaryRadiusInner = secondaryRadiusInner.GetValueOrDefault();
        float __deref_secondaryRadiusOuter = secondaryRadiusOuter.GetValueOrDefault();
        int __deref_primaryResolution = primaryResolution.GetValueOrDefault();
        int __deref_secondaryResolution = secondaryResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeTorusWithSpikes(primaryRadius.HasValue ? &__deref_primaryRadius : null, secondaryRadiusInner.HasValue ? &__deref_secondaryRadiusInner : null, secondaryRadiusOuter.HasValue ? &__deref_secondaryRadiusOuter : null, primaryResolution.HasValue ? &__deref_primaryResolution : null, secondaryResolution.HasValue ? &__deref_secondaryResolution : null, points is not null ? points._UnderlyingPtr : null), is_owning: true));
    }

    // creates torus with empty sectors
    // main application - testing Components
    /// Generated from function `MR::makeTorusWithComponents`.
    /// Parameter `primaryRadius` defaults to `1.0f`.
    /// Parameter `secondaryRadius` defaults to `0.100000001f`.
    /// Parameter `primaryResolution` defaults to `16`.
    /// Parameter `secondaryResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeTorusWithComponents(float? primaryRadius = null, float? secondaryRadius = null, int? primaryResolution = null, int? secondaryResolution = null, MR.Std.Vector_MRVector3f? points = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeTorusWithComponents", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeTorusWithComponents(float *primaryRadius, float *secondaryRadius, int *primaryResolution, int *secondaryResolution, MR.Std.Vector_MRVector3f._Underlying *points);
        float __deref_primaryRadius = primaryRadius.GetValueOrDefault();
        float __deref_secondaryRadius = secondaryRadius.GetValueOrDefault();
        int __deref_primaryResolution = primaryResolution.GetValueOrDefault();
        int __deref_secondaryResolution = secondaryResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeTorusWithComponents(primaryRadius.HasValue ? &__deref_primaryRadius : null, secondaryRadius.HasValue ? &__deref_secondaryRadius : null, primaryResolution.HasValue ? &__deref_primaryResolution : null, secondaryResolution.HasValue ? &__deref_secondaryResolution : null, points is not null ? points._UnderlyingPtr : null), is_owning: true));
    }

    // creates torus with empty sectors
    // main application - testing Components
    /// Generated from function `MR::makeTorusWithSelfIntersections`.
    /// Parameter `primaryRadius` defaults to `1.0f`.
    /// Parameter `secondaryRadius` defaults to `0.100000001f`.
    /// Parameter `primaryResolution` defaults to `16`.
    /// Parameter `secondaryResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeTorusWithSelfIntersections(float? primaryRadius = null, float? secondaryRadius = null, int? primaryResolution = null, int? secondaryResolution = null, MR.Std.Vector_MRVector3f? points = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeTorusWithSelfIntersections", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeTorusWithSelfIntersections(float *primaryRadius, float *secondaryRadius, int *primaryResolution, int *secondaryResolution, MR.Std.Vector_MRVector3f._Underlying *points);
        float __deref_primaryRadius = primaryRadius.GetValueOrDefault();
        float __deref_secondaryRadius = secondaryRadius.GetValueOrDefault();
        int __deref_primaryResolution = primaryResolution.GetValueOrDefault();
        int __deref_secondaryResolution = secondaryResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeTorusWithSelfIntersections(primaryRadius.HasValue ? &__deref_primaryRadius : null, secondaryRadius.HasValue ? &__deref_secondaryRadius : null, primaryResolution.HasValue ? &__deref_primaryResolution : null, secondaryResolution.HasValue ? &__deref_secondaryResolution : null, points is not null ? points._UnderlyingPtr : null), is_owning: true));
    }
}
