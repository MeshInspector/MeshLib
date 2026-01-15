public class FreeFormDeformationExample
{
    public static void Run(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

            // Compute mesh bounding box
            MR.Box3f box = mesh.getBoundingBox();

            // Construct deformer on mesh vertices
            MR.FreeFormDeformer ffDeformer = new(mesh);

            // Init deformer with 3x3 grid on mesh box
            ffDeformer.init(MR.Vector3i.diagonal(3), box);

            // Move some control points of the grid to the center
            ffDeformer.setRefGridPointPosition(new MR.Vector3i(1, 1, 0), box.center());
            ffDeformer.setRefGridPointPosition(new MR.Vector3i(1, 1, 2), box.center());
            ffDeformer.setRefGridPointPosition(new MR.Vector3i(0, 1, 1), box.center());
            ffDeformer.setRefGridPointPosition(new MR.Vector3i(2, 1, 1), box.center());
            ffDeformer.setRefGridPointPosition(new MR.Vector3i(1, 0, 1), box.center());
            ffDeformer.setRefGridPointPosition(new MR.Vector3i(1, 2, 1), box.center());

            // Apply the deformation to the mesh vertices
            ffDeformer.apply();

            // Invalidate the mesh because of external vertex changes
            mesh.invalidateCaches();

            // Save deformed mesh
            MR.MeshSave.toAnySupportedFormat(mesh, "deformed_mesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
