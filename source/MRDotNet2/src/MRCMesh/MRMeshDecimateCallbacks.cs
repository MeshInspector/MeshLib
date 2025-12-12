public static partial class MR
{
    /**
    * Please use this callback when you decimate a mesh with associated data with each vertex
    * recalculate texture coordinates and mesh vertex colors for vertices moved during decimation
    * usage example
    *   MeshAttributesToUpdate meshParams;
    *   auto uvCoords = obj->getUVCoords();
    *   auto colorMap = obj->getVertsColorMap();
    *   if ( needUpdateUV )
    *       meshParams.uvCoords = &uvCoords;
    *   if ( needUpdateColorMap )
    *       meshParams.colorMap = &colorMap;
    *   auto preCollapse = meshPreCollapseVertAttribute( mesh, meshParams );
    *   decimateMesh( mesh, DecimateSettings{ .preCollapse = preCollapse } );
    */
    /// Generated from function `MR::meshPreCollapseVertAttribute`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef> MeshPreCollapseVertAttribute(MR.Const_Mesh mesh, MR.Const_MeshAttributesToUpdate params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshPreCollapseVertAttribute", ExactSpelling = true)]
        extern static MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_meshPreCollapseVertAttribute(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshAttributesToUpdate._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef(__MR_meshPreCollapseVertAttribute(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
