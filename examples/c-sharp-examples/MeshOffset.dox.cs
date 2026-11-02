public static class MeshOffsetExample
{
    public static void Main(string[] args)
    {
        // Create mesh
        var mesh = MR.makeCube();

        // Setup parameters
        MR.GeneralOffsetParameters parameters = new();
        // calculate voxel size depending on desired accuracy and/or memory consumption
        parameters.voxelSize = MR.suggestVoxelSize(mesh, 10000000.0f);
        if (!MR.findRightBoundary(mesh.topology).empty())
            parameters.signDetectionMode = MR.SignDetectionMode.HoleWindingRule; // use if you have holes in mesh

        // Make offset mesh
        float offset = mesh.computeBoundingBox().diagonal() * 0.1f;
        var result = MR.generalOffsetMesh(mesh, offset, parameters);

        // Save result
        MR.MeshSave.toAnySupportedFormat(result, "mesh_offset.stl");
    }
}
