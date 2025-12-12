public static partial class MR
{
    /// given a double-layer mesh with one layer having normals outside and the other layer - inside,
    /// finds all faces of the outer layer;
    /// the algorithm first detects some seed faces of each layer by casting a ray from triangle's center in both directions along the normal;
    /// then remaining faces are redistributed toward the closest seed face
    /// Generated from function `MR::findOuterLayer`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FindOuterLayer(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findOuterLayer", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_findOuterLayer(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_findOuterLayer(mesh._UnderlyingPtr), is_owning: true));
    }
}
