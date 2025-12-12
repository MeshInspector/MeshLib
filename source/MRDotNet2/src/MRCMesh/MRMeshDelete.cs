public static partial class MR
{
    /// deletes object faces with normals pointed to the target geometry center
    /// Generated from function `MR::deleteTargetFaces`.
    public static unsafe void DeleteTargetFaces(MR.Mesh obj, MR.Const_Vector3f targetCenter)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deleteTargetFaces_MR_Vector3f", ExactSpelling = true)]
        extern static void __MR_deleteTargetFaces_MR_Vector3f(MR.Mesh._Underlying *obj, MR.Const_Vector3f._Underlying *targetCenter);
        __MR_deleteTargetFaces_MR_Vector3f(obj._UnderlyingPtr, targetCenter._UnderlyingPtr);
    }

    /// Generated from function `MR::deleteTargetFaces`.
    public static unsafe void DeleteTargetFaces(MR.Mesh obj, MR.Const_Mesh target)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deleteTargetFaces_MR_Mesh", ExactSpelling = true)]
        extern static void __MR_deleteTargetFaces_MR_Mesh(MR.Mesh._Underlying *obj, MR.Const_Mesh._Underlying *target);
        __MR_deleteTargetFaces_MR_Mesh(obj._UnderlyingPtr, target._UnderlyingPtr);
    }
}
