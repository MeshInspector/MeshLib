public static partial class MR
{
    ///  Mesh to PointCloud
    /// Generated from function `MR::meshToPointCloud`.
    /// Parameter `saveNormals` defaults to `true`.
    public static unsafe MR.Misc._Moved<MR.PointCloud> MeshToPointCloud(MR.Const_Mesh mesh, bool? saveNormals = null, MR.Const_VertBitSet? verts = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToPointCloud", ExactSpelling = true)]
        extern static MR.PointCloud._Underlying *__MR_meshToPointCloud(MR.Const_Mesh._Underlying *mesh, byte *saveNormals, MR.Const_VertBitSet._Underlying *verts);
        byte __deref_saveNormals = saveNormals.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.PointCloud(__MR_meshToPointCloud(mesh._UnderlyingPtr, saveNormals.HasValue ? &__deref_saveNormals : null, verts is not null ? verts._UnderlyingPtr : null), is_owning: true));
    }
}
