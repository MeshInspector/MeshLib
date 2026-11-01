public static class MeshFixDegeneraciesExample
{
    public static void Main(string[] args)
    {
        var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

        // you can set various parameters for the resolving process; see the documentation for more info
        MR.FixMeshDegeneraciesParams parameters = new();
        parameters.maxDeviation = 1e-5f * mesh.computeBoundingBox().diagonal();
        parameters.tinyEdgeLength = 1e-3f;
        MR.fixMeshDegeneracies(mesh, parameters);

        // Save result
        MR.MeshSave.toAnySupportedFormat(mesh, "fixedMesh.stl");
    }
}
