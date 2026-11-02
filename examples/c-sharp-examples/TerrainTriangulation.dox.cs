public static class TerrainTriangulationExample
{
    public static void Main(string[] args)
    {
        // load points
        MR.VertColors colors = new();
        MR.PointsLoadSettings pls = new();
        pls.colors = colors;
        var pc = MR.PointsLoad.fromAnySupportedFormat("TerrainPoints.ply", pls);

        var mesh = MR.delaunayTriangulationXY(pc);

        MR.SaveSettings ss = new();
        if (pc.points.size() == colors.size())
            ss.colors = colors;
        MR.MeshSave.toAnySupportedFormat(mesh, "TerrainMesh.ctm", ss);
    }
}
