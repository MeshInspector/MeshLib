public static partial class MR
{
    /**
    * \brief finds all pairs of colliding triangles from two meshes or two mesh regions
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param firstIntersectionOnly if true then the function returns at most one pair of intersecting triangles and returns faster
    */
    /// Generated from function `MR::findCollidingTriangles`.
    /// Parameter `firstIntersectionOnly` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRFaceFace> FindCollidingTriangles(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null, bool? firstIntersectionOnly = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingTriangles", ExactSpelling = true)]
        extern static MR.Std.Vector_MRFaceFace._Underlying *__MR_findCollidingTriangles(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A, byte *firstIntersectionOnly);
        byte __deref_firstIntersectionOnly = firstIntersectionOnly.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Vector_MRFaceFace(__MR_findCollidingTriangles(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null, firstIntersectionOnly.HasValue ? &__deref_firstIntersectionOnly : null), is_owning: true));
    }

    /// the same as \ref findCollidingTriangles, but returns one bite set per mesh with colliding triangles
    /// Generated from function `MR::findCollidingTriangleBitsets`.
    public static unsafe MR.Misc._Moved<MR.Std.Pair_MRFaceBitSet_MRFaceBitSet> FindCollidingTriangleBitsets(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findCollidingTriangleBitsets", ExactSpelling = true)]
        extern static MR.Std.Pair_MRFaceBitSet_MRFaceBitSet._Underlying *__MR_findCollidingTriangleBitsets(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return MR.Misc.Move(new MR.Std.Pair_MRFaceBitSet_MRFaceBitSet(__MR_findCollidingTriangleBitsets(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all pairs (or the fact of any self-collision) of colliding triangles from one mesh or a region
    /// Generated from function `MR::findSelfCollidingTriangles`.
    /// Parameter `cb` defaults to `{}`.
    /// Parameter `touchIsIntersection` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Expected_Bool_StdString> FindSelfCollidingTriangles(MR.Const_MeshPart mp, MR.Std.Vector_MRFaceFace? outCollidingPairs, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Const_Face2RegionMap? regionMap = null, bool? touchIsIntersection = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingTriangles_5", ExactSpelling = true)]
        extern static MR.Expected_Bool_StdString._Underlying *__MR_findSelfCollidingTriangles_5(MR.Const_MeshPart._Underlying *mp, MR.Std.Vector_MRFaceFace._Underlying *outCollidingPairs, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Const_Face2RegionMap._Underlying *regionMap, byte *touchIsIntersection);
        byte __deref_touchIsIntersection = touchIsIntersection.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Expected_Bool_StdString(__MR_findSelfCollidingTriangles_5(mp._UnderlyingPtr, outCollidingPairs is not null ? outCollidingPairs._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, regionMap is not null ? regionMap._UnderlyingPtr : null, touchIsIntersection.HasValue ? &__deref_touchIsIntersection : null), is_owning: true));
    }

    /// finds all pairs of colliding triangles from one mesh or a region
    /// Generated from function `MR::findSelfCollidingTriangles`.
    /// Parameter `cb` defaults to `{}`.
    /// Parameter `touchIsIntersection` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRFaceFace_StdString> FindSelfCollidingTriangles(MR.Const_MeshPart mp, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Const_Face2RegionMap? regionMap = null, bool? touchIsIntersection = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingTriangles_4", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMRFaceFace_StdString._Underlying *__MR_findSelfCollidingTriangles_4(MR.Const_MeshPart._Underlying *mp, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Const_Face2RegionMap._Underlying *regionMap, byte *touchIsIntersection);
        byte __deref_touchIsIntersection = touchIsIntersection.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Expected_StdVectorMRFaceFace_StdString(__MR_findSelfCollidingTriangles_4(mp._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, regionMap is not null ? regionMap._UnderlyingPtr : null, touchIsIntersection.HasValue ? &__deref_touchIsIntersection : null), is_owning: true));
    }

    /// the same \ref findSelfCollidingTriangles but returns the union of all self-intersecting faces
    /// Generated from function `MR::findSelfCollidingTrianglesBS`.
    /// Parameter `cb` defaults to `{}`.
    /// Parameter `touchIsIntersection` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> FindSelfCollidingTrianglesBS(MR.Const_MeshPart mp, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Const_Face2RegionMap? regionMap = null, bool? touchIsIntersection = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSelfCollidingTrianglesBS", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_findSelfCollidingTrianglesBS(MR.Const_MeshPart._Underlying *mp, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Const_Face2RegionMap._Underlying *regionMap, byte *touchIsIntersection);
        byte __deref_touchIsIntersection = touchIsIntersection.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_findSelfCollidingTrianglesBS(mp._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, regionMap is not null ? regionMap._UnderlyingPtr : null, touchIsIntersection.HasValue ? &__deref_touchIsIntersection : null), is_owning: true));
    }

    /**
    * \brief checks that arbitrary mesh part A is inside of closed mesh part B
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    */
    /// Generated from function `MR::isInside`.
    public static unsafe bool IsInside(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isInside_MR_MeshPart", ExactSpelling = true)]
        extern static byte __MR_isInside_MR_MeshPart(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return __MR_isInside_MR_MeshPart(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null) != 0;
    }

    /**
    * \brief checks that arbitrary mesh part A is inside of closed mesh part B
    * The version of `isInside` without collision check; it is user's responsibility to guarantee that the meshes don't collide
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    */
    /// Generated from function `MR::isNonIntersectingInside`.
    public static unsafe bool IsNonIntersectingInside(MR.Const_MeshPart a, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isNonIntersectingInside_3", ExactSpelling = true)]
        extern static byte __MR_isNonIntersectingInside_3(MR.Const_MeshPart._Underlying *a, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return __MR_isNonIntersectingInside_3(a._UnderlyingPtr, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null) != 0;
    }

    /**
    * \brief checks that arbitrary mesh A part (whole part is represented by one face `partFace`) is inside of closed mesh part B
    * The version of `isInside` without collision check; it is user's responsibility to guarantee that the meshes don't collide
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    */
    /// Generated from function `MR::isNonIntersectingInside`.
    public static unsafe bool IsNonIntersectingInside(MR.Const_Mesh a, MR.FaceId partFace, MR.Const_MeshPart b, MR.Const_AffineXf3f? rigidB2A = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isNonIntersectingInside_4", ExactSpelling = true)]
        extern static byte __MR_isNonIntersectingInside_4(MR.Const_Mesh._Underlying *a, MR.FaceId partFace, MR.Const_MeshPart._Underlying *b, MR.Const_AffineXf3f._Underlying *rigidB2A);
        return __MR_isNonIntersectingInside_4(a._UnderlyingPtr, partFace, b._UnderlyingPtr, rigidB2A is not null ? rigidB2A._UnderlyingPtr : null) != 0;
    }
}
