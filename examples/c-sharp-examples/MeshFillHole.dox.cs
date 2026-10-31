public static class MeshFillHoleExample
{
    public static void Main(string[] args)
    {
        // Load mesh
        var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

        // Find single edge for each hole in mesh
        var holeEdges = mesh.topology.findHoleRepresentiveEdges();

        // Setup filling parameters
        MR.FillHoleParams parameters = new();
        parameters.metric.assign(MR.getUniversalMetric(mesh));

        // Alternatively, MR.fillHoles(mesh, holeEdges, parameters) fills all holes at once.
        for (nuint i = 0; i < holeEdges.size(); ++i)
            MR.fillHole(mesh, holeEdges[i], parameters);

        // Save result
        MR.MeshSave.toAnySupportedFormat(mesh, "filledMesh.stl");
    }
}
