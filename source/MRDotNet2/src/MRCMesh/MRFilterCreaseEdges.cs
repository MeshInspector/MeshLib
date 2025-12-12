public static partial class MR
{
    /// filters given edges using the following criteria:
    /// if \param filterComponents is true then connected components with summary length of their edges less than \param critLength will be excluded
    /// if \param filterBranches is true then branches shorter than \param critLength will be excluded
    /// Generated from function `MR::filterCreaseEdges`.
    /// Parameter `filterComponents` defaults to `true`.
    /// Parameter `filterBranches` defaults to `false`.
    public static unsafe void FilterCreaseEdges(MR.Const_Mesh mesh, MR.UndirectedEdgeBitSet creaseEdges, float critLength, bool? filterComponents = null, bool? filterBranches = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_filterCreaseEdges", ExactSpelling = true)]
        extern static void __MR_filterCreaseEdges(MR.Const_Mesh._Underlying *mesh, MR.UndirectedEdgeBitSet._Underlying *creaseEdges, float critLength, byte *filterComponents, byte *filterBranches);
        byte __deref_filterComponents = filterComponents.GetValueOrDefault() ? (byte)1 : (byte)0;
        byte __deref_filterBranches = filterBranches.GetValueOrDefault() ? (byte)1 : (byte)0;
        __MR_filterCreaseEdges(mesh._UnderlyingPtr, creaseEdges._UnderlyingPtr, critLength, filterComponents.HasValue ? &__deref_filterComponents : null, filterBranches.HasValue ? &__deref_filterBranches : null);
    }
}
