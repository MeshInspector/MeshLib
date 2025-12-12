public static partial class MR
{
    /// divides given mesh into hierarchy of mesh objects:
    /// the deepest level of the hierarchy has all details from the original mesh;
    /// top levels have progressively less number of objects and less details in each;
    /// the number of faces in any object on any level is about the same.
    /// Generated from function `MR::makeLevelOfDetails`.
    public static unsafe MR.Misc._Moved<MR.Object> MakeLevelOfDetails(MR.Misc._Moved<MR.Mesh> mesh, int maxDepth)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeLevelOfDetails", ExactSpelling = true)]
        extern static MR.Object._UnderlyingShared *__MR_makeLevelOfDetails(MR.Mesh._Underlying *mesh, int maxDepth);
        return MR.Misc.Move(new MR.Object(__MR_makeLevelOfDetails(mesh.Value._UnderlyingPtr, maxDepth), is_owning: true));
    }
}
