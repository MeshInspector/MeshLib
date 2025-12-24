public class FreeFormDeformationExample
{
    public static void Run(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");

            // Compute mesh bounding box
            MR.Box3f box = mesh.GetBoundingBox();

            // Construct deformer on mesh vertices
            MR.FreeFormDeformer ffDeformer = new(mesh);

            // Init deformer with 3x3 grid on mesh box
            ffDeformer.Init(MR.Vector3i.Diagonal(3), box);

            // Move some control points of the grid to the center
            ffDeformer.SetRefGridPointPosition(new MR.Vector3i(1, 1, 0), box.Center());
            ffDeformer.SetRefGridPointPosition(new MR.Vector3i(1, 1, 2), box.Center());
            ffDeformer.SetRefGridPointPosition(new MR.Vector3i(0, 1, 1), box.Center());
            ffDeformer.SetRefGridPointPosition(new MR.Vector3i(2, 1, 1), box.Center());
            ffDeformer.SetRefGridPointPosition(new MR.Vector3i(1, 0, 1), box.Center());
            ffDeformer.SetRefGridPointPosition(new MR.Vector3i(1, 2, 1), box.Center());

            // Apply the deformation to the mesh vertices
            ffDeformer.Apply();

            // Invalidate the mesh because of external vertex changes
            mesh.InvalidateCaches();

            // Save deformed mesh
            MR.MeshSave.ToAnySupportedFormat(mesh, "deformed_mesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
