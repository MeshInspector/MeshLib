public static partial class MR
{
    /// Offsets mesh part by converting it to voxels and back
    /// and unite it with original mesh (via boolean)
    /// note: only OffsetParameters.signDetectionMode = SignDetectionMode::Unsigned will work in this function
    /// Generated from function `MR::partialOffsetMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> PartialOffsetMesh(MR.Const_MeshPart mp, float offset, MR.Const_GeneralOffsetParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_partialOffsetMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_partialOffsetMesh(MR.Const_MeshPart._Underlying *mp, float offset, MR.Const_GeneralOffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_partialOffsetMesh(mp._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
