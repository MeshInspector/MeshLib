public static partial class MR
{
    /// given a mesh part and its arbitrary transformation, computes and returns
    /// the rigid transformation that best approximates meshXf
    /// Generated from function `MR::makeRigidXf`.
    public static unsafe MR.AffineXf3d MakeRigidXf(MR.Const_MeshPart mp, MR.Const_AffineXf3d meshXf)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeRigidXf_MR_AffineXf3d", ExactSpelling = true)]
        extern static MR.AffineXf3d __MR_makeRigidXf_MR_AffineXf3d(MR.Const_MeshPart._Underlying *mp, MR.Const_AffineXf3d._Underlying *meshXf);
        return __MR_makeRigidXf_MR_AffineXf3d(mp._UnderlyingPtr, meshXf._UnderlyingPtr);
    }

    /// Generated from function `MR::makeRigidXf`.
    public static unsafe MR.AffineXf3f MakeRigidXf(MR.Const_MeshPart mp, MR.Const_AffineXf3f meshXf)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeRigidXf_MR_AffineXf3f", ExactSpelling = true)]
        extern static MR.AffineXf3f __MR_makeRigidXf_MR_AffineXf3f(MR.Const_MeshPart._Underlying *mp, MR.Const_AffineXf3f._Underlying *meshXf);
        return __MR_makeRigidXf_MR_AffineXf3f(mp._UnderlyingPtr, meshXf._UnderlyingPtr);
    }
}
