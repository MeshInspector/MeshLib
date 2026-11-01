public static class MeshStitchHoleExample
{
    public static void Main(string[] args)
    {
        // Load meshes
        var mesh = MR.MeshLoad.fromAnySupportedFormat("meshAwithHole.stl");
        var meshB = MR.MeshLoad.fromAnySupportedFormat("meshBwithHole.stl");

        // Unite meshes
        mesh.addMesh(meshB);

        // Find holes (expect that there are exactly 2 holes)
        var edges = mesh.topology.findHoleRepresentiveEdges();
        if (edges.size() != 2)
            Environment.Exit(1);

        // Connect two holes
        MR.StitchHolesParams parameters = new();
        parameters.metric.assign(MR.getUniversalMetric(mesh));
        MR.stitchHoles(mesh, edges[0], edges[1], parameters);

        // Save result
        MR.MeshSave.toAnySupportedFormat(mesh, "stitchedMesh.stl");
    }
}
