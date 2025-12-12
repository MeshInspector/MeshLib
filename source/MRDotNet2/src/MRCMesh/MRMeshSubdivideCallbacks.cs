public static partial class MR
{
    /**
    * auto uvCoords = obj_->getUVCoords();
    * auto texturePerFace = obj_->getTexturePerFace();
    * MeshAttributesToUpdate meshParams;
    * if ( !uvCoords.empty() )
    *     meshParams.uvCoords = &uvCoords;
    * if ( !texturePerFace.empty() )
    *     meshParams.texturePerFace = &texturePerFace;
    * subs.onEdgeSplit = meshOnEdgeSplitAttribute( *obj_->varMesh(), meshParams );
    * subdivideMesh( *obj_->varMesh(), subs );
    */
    /// Generated from function `MR::meshOnEdgeSplitAttribute`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId> MeshOnEdgeSplitAttribute(MR.Const_Mesh mesh, MR.Const_MeshAttributesToUpdate params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshOnEdgeSplitAttribute", ExactSpelling = true)]
        extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_meshOnEdgeSplitAttribute(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshAttributesToUpdate._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId(__MR_meshOnEdgeSplitAttribute(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::meshOnEdgeSplitVertAttribute`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId> MeshOnEdgeSplitVertAttribute(MR.Const_Mesh mesh, MR.Const_MeshAttributesToUpdate params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshOnEdgeSplitVertAttribute", ExactSpelling = true)]
        extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_meshOnEdgeSplitVertAttribute(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshAttributesToUpdate._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId(__MR_meshOnEdgeSplitVertAttribute(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::meshOnEdgeSplitFaceAttribute`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId> MeshOnEdgeSplitFaceAttribute(MR.Const_Mesh mesh, MR.Const_MeshAttributesToUpdate params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshOnEdgeSplitFaceAttribute", ExactSpelling = true)]
        extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_meshOnEdgeSplitFaceAttribute(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshAttributesToUpdate._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId(__MR_meshOnEdgeSplitFaceAttribute(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
