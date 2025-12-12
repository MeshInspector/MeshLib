public static partial class MR
{
    /// updates a2b map to a2c map using b2c map
    /// Generated from function `MR::vertMapsComposition`.
    public static unsafe void VertMapsComposition(MR.VertMap a2b, MR.Const_VertMap b2c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vertMapsComposition_MR_VertMap_ref", ExactSpelling = true)]
        extern static void __MR_vertMapsComposition_MR_VertMap_ref(MR.VertMap._Underlying *a2b, MR.Const_VertMap._Underlying *b2c);
        __MR_vertMapsComposition_MR_VertMap_ref(a2b._UnderlyingPtr, b2c._UnderlyingPtr);
    }

    /// returns map a2c from a2b and b2c maps
    /// Generated from function `MR::vertMapsComposition`.
    public static unsafe MR.Misc._Moved<MR.VertMap> VertMapsComposition(MR.Const_VertMap a2b, MR.Const_VertMap b2c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vertMapsComposition_const_MR_VertMap_ref", ExactSpelling = true)]
        extern static MR.VertMap._Underlying *__MR_vertMapsComposition_const_MR_VertMap_ref(MR.Const_VertMap._Underlying *a2b, MR.Const_VertMap._Underlying *b2c);
        return MR.Misc.Move(new MR.VertMap(__MR_vertMapsComposition_const_MR_VertMap_ref(a2b._UnderlyingPtr, b2c._UnderlyingPtr), is_owning: true));
    }

    /// updates a2b map to a2c map using b2c map
    /// Generated from function `MR::edgeMapsComposition`.
    public static unsafe void EdgeMapsComposition(MR.EdgeMap a2b, MR.Const_EdgeMap b2c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeMapsComposition_MR_EdgeMap_ref", ExactSpelling = true)]
        extern static void __MR_edgeMapsComposition_MR_EdgeMap_ref(MR.EdgeMap._Underlying *a2b, MR.Const_EdgeMap._Underlying *b2c);
        __MR_edgeMapsComposition_MR_EdgeMap_ref(a2b._UnderlyingPtr, b2c._UnderlyingPtr);
    }

    /// returns map a2c from a2b and b2c maps
    /// Generated from function `MR::edgeMapsComposition`.
    public static unsafe MR.Misc._Moved<MR.EdgeMap> EdgeMapsComposition(MR.Const_EdgeMap a2b, MR.Const_EdgeMap b2c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeMapsComposition_const_MR_EdgeMap_ref", ExactSpelling = true)]
        extern static MR.EdgeMap._Underlying *__MR_edgeMapsComposition_const_MR_EdgeMap_ref(MR.Const_EdgeMap._Underlying *a2b, MR.Const_EdgeMap._Underlying *b2c);
        return MR.Misc.Move(new MR.EdgeMap(__MR_edgeMapsComposition_const_MR_EdgeMap_ref(a2b._UnderlyingPtr, b2c._UnderlyingPtr), is_owning: true));
    }

    /// updates a2b map to a2c map using b2c map
    /// Generated from function `MR::faceMapsComposition`.
    public static unsafe void FaceMapsComposition(MR.FaceMap a2b, MR.Const_FaceMap b2c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_faceMapsComposition_MR_FaceMap_ref", ExactSpelling = true)]
        extern static void __MR_faceMapsComposition_MR_FaceMap_ref(MR.FaceMap._Underlying *a2b, MR.Const_FaceMap._Underlying *b2c);
        __MR_faceMapsComposition_MR_FaceMap_ref(a2b._UnderlyingPtr, b2c._UnderlyingPtr);
    }

    /// returns map a2c from a2b and b2c maps
    /// Generated from function `MR::faceMapsComposition`.
    public static unsafe MR.Misc._Moved<MR.FaceMap> FaceMapsComposition(MR.Const_FaceMap a2b, MR.Const_FaceMap b2c)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_faceMapsComposition_const_MR_FaceMap_ref", ExactSpelling = true)]
        extern static MR.FaceMap._Underlying *__MR_faceMapsComposition_const_MR_FaceMap_ref(MR.Const_FaceMap._Underlying *a2b, MR.Const_FaceMap._Underlying *b2c);
        return MR.Misc.Move(new MR.FaceMap(__MR_faceMapsComposition_const_MR_FaceMap_ref(a2b._UnderlyingPtr, b2c._UnderlyingPtr), is_owning: true));
    }
}
