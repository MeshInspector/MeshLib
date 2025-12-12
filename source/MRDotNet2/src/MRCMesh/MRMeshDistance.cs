public static partial class MR
{
    /// This enum is intended to be boolean.
    public enum ProcessOneResult : byte
    {
        StopProcessing = 0,
        ContinueProcessing = 1,
    }

    /// invokes given callback for all triangles from given mesh part located not further than
    /// given squared distance from t-triangle
    /// Generated from function `MR::processCloseTriangles`.
    public static unsafe void ProcessCloseTriangles(MR.Const_MeshPart mp, MR.Std.Const_Array_MRVector3f_3 t, float rangeSq, MR.Std.Const_Function_MRProcessOneResultFuncFromConstMRVector3fRefMRFaceIdConstMRVector3fRefFloat call)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_processCloseTriangles", ExactSpelling = true)]
        extern static void __MR_processCloseTriangles(MR.Const_MeshPart._Underlying *mp, MR.Std.Const_Array_MRVector3f_3._Underlying *t, float rangeSq, MR.Std.Const_Function_MRProcessOneResultFuncFromConstMRVector3fRefMRFaceIdConstMRVector3fRefFloat._Underlying *call);
        __MR_processCloseTriangles(mp._UnderlyingPtr, t._UnderlyingPtr, rangeSq, call._UnderlyingPtr);
    }

    /// computes signed distance from point (p) to mesh part (mp) following options (op);
    /// returns std::nullopt if distance is smaller than op.minDist or larger than op.maxDist (except for op.signMode == HoleWindingRule)
    /// Generated from function `MR::signedDistanceToMesh`.
    public static unsafe MR.Std.Optional_Float SignedDistanceToMesh(MR.Const_MeshPart mp, MR.Const_Vector3f p, MR.Const_SignedDistanceToMeshOptions op)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_signedDistanceToMesh", ExactSpelling = true)]
        extern static MR.Std.Optional_Float._Underlying *__MR_signedDistanceToMesh(MR.Const_MeshPart._Underlying *mp, MR.Const_Vector3f._Underlying *p, MR.Const_SignedDistanceToMeshOptions._Underlying *op);
        return new(__MR_signedDistanceToMesh(mp._UnderlyingPtr, p._UnderlyingPtr, op._UnderlyingPtr), is_owning: true);
    }
}
