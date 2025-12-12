public static partial class MR
{
    /// metric returning 1 for every edge
    /// Generated from function `MR::identityMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> IdentityMetric()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_identityMetric", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_identityMetric();
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_identityMetric(), is_owning: true));
    }

    /// returns edge's length as a metric;
    /// this metric is symmetric: m(e) == m(e.sym())
    /// Generated from function `MR::edgeLengthMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> EdgeLengthMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeLengthMetric_1", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_edgeLengthMetric_1(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_edgeLengthMetric_1(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::edgeLengthMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> EdgeLengthMetric(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeLengthMetric_2", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_edgeLengthMetric_2(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_edgeLengthMetric_2(topology._UnderlyingPtr, points._UnderlyingPtr), is_owning: true));
    }

    /// returns edge's absolute discrete mean curvature as a metric;
    /// the metric is minimal in the planar regions of mesh;
    /// this metric is symmetric: m(e) == m(e.sym())
    /// Generated from function `MR::discreteAbsMeanCurvatureMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> DiscreteAbsMeanCurvatureMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteAbsMeanCurvatureMetric_1", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_discreteAbsMeanCurvatureMetric_1(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_discreteAbsMeanCurvatureMetric_1(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::discreteAbsMeanCurvatureMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> DiscreteAbsMeanCurvatureMetric(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteAbsMeanCurvatureMetric_2", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_discreteAbsMeanCurvatureMetric_2(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_discreteAbsMeanCurvatureMetric_2(topology._UnderlyingPtr, points._UnderlyingPtr), is_owning: true));
    }

    /// returns minus of edge's absolute discrete mean curvature as a metric;
    /// the metric is minimal in the most curved regions of mesh;
    /// this metric is symmetric: m(e) == m(e.sym())
    /// Generated from function `MR::discreteMinusAbsMeanCurvatureMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> DiscreteMinusAbsMeanCurvatureMetric(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteMinusAbsMeanCurvatureMetric_1", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_discreteMinusAbsMeanCurvatureMetric_1(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_discreteMinusAbsMeanCurvatureMetric_1(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::discreteMinusAbsMeanCurvatureMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> DiscreteMinusAbsMeanCurvatureMetric(MR.Const_MeshTopology topology, MR.Const_VertCoords points)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_discreteMinusAbsMeanCurvatureMetric_2", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_discreteMinusAbsMeanCurvatureMetric_2(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_discreteMinusAbsMeanCurvatureMetric_2(topology._UnderlyingPtr, points._UnderlyingPtr), is_owning: true));
    }

    /// returns edge's metric that depends both on edge's length and on the angle between its left and right faces
    /// \param angleSinFactor multiplier before dihedral angle sine in edge metric calculation (positive to prefer concave angles, negative - convex)
    /// \param angleSinForBoundary consider this dihedral angle sine for boundary edges;
    /// this metric is symmetric: m(e) == m(e.sym())
    /// Generated from function `MR::edgeCurvMetric`.
    /// Parameter `angleSinFactor` defaults to `2`.
    /// Parameter `angleSinForBoundary` defaults to `0`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> EdgeCurvMetric(MR.Const_Mesh mesh, float? angleSinFactor = null, float? angleSinForBoundary = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeCurvMetric_3", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_edgeCurvMetric_3(MR.Const_Mesh._Underlying *mesh, float *angleSinFactor, float *angleSinForBoundary);
        float __deref_angleSinFactor = angleSinFactor.GetValueOrDefault();
        float __deref_angleSinForBoundary = angleSinForBoundary.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_edgeCurvMetric_3(mesh._UnderlyingPtr, angleSinFactor.HasValue ? &__deref_angleSinFactor : null, angleSinForBoundary.HasValue ? &__deref_angleSinForBoundary : null), is_owning: true));
    }

    /// Generated from function `MR::edgeCurvMetric`.
    /// Parameter `angleSinFactor` defaults to `2`.
    /// Parameter `angleSinForBoundary` defaults to `0`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> EdgeCurvMetric(MR.Const_MeshTopology topology, MR.Const_VertCoords points, float? angleSinFactor = null, float? angleSinForBoundary = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeCurvMetric_4", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_edgeCurvMetric_4(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, float *angleSinFactor, float *angleSinForBoundary);
        float __deref_angleSinFactor = angleSinFactor.GetValueOrDefault();
        float __deref_angleSinForBoundary = angleSinForBoundary.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_edgeCurvMetric_4(topology._UnderlyingPtr, points._UnderlyingPtr, angleSinFactor.HasValue ? &__deref_angleSinFactor : null, angleSinForBoundary.HasValue ? &__deref_angleSinForBoundary : null), is_owning: true));
    }

    /// pre-computes the metric for all mesh edges to quickly return it later for any edge;
    /// input metric must be symmetric: metric(e) == metric(e.sym())
    /// Generated from function `MR::edgeTableSymMetric`.
    public static unsafe MR.Misc._Moved<MR.Std.Function_FloatFuncFromMREdgeId> EdgeTableSymMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgeTableSymMetric", ExactSpelling = true)]
        extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_edgeTableSymMetric(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric);
        return MR.Misc.Move(new MR.Std.Function_FloatFuncFromMREdgeId(__MR_edgeTableSymMetric(topology._UnderlyingPtr, metric._UnderlyingPtr), is_owning: true));
    }
}
