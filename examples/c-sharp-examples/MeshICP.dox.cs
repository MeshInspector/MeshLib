public static class MeshICPExample
{
    public static void Main(string[] args)
    {
        // Load meshes
        var meshFloating = MR.MeshLoad.fromAnySupportedFormat("meshA.stl");
        var meshFixed = MR.MeshLoad.fromAnySupportedFormat("meshB.stl");

        // Prepare ICP parameters
        float diagonal = meshFixed.getBoundingBox().diagonal();
        float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
        MR.ICPProperties icpParams = new();
        icpParams.distThresholdSq = (diagonal * 0.1f) * (diagonal * 0.1f); // Use points pairs with maximum distance specified
        icpParams.exitVal = diagonal * 0.003f; // Stop when distance reached

        // Calculate transformation
        MR.ICP icp = new(meshFloating, meshFixed, new MR.AffineXf3f(), new MR.AffineXf3f(), icpSamplingVoxelSize);
        icp.setParams(icpParams);
        var xf = icp.calculateTransformation();

        // Transform floating mesh
        meshFloating.transform(xf);

        // Output information string
        Console.Error.WriteLine(icp.getStatusInfo());

        // Save result
        MR.MeshSave.toAnySupportedFormat(meshFloating, "meshA_icp.stl");
    }
}
