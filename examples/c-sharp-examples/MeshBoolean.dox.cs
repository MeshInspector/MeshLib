public class MeshBooleanExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Create the first sphere with a radius of 1 unit
            // (to use your own meshes, load them with MR.MeshLoad.fromAnySupportedFormat instead)
            var mesh_a = MR.makeUVSphere(1.0f, 64, 64);

            // Create the second sphere and shift it along X, so the two spheres overlap
            var mesh_b = MR.makeUVSphere(1.0f, 64, 64);
            mesh_b.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.7f, 0.0f, 0.0f)));

            // perform boolean operation
            MR.BooleanResult res = MR.boolean(mesh_a, mesh_b, MR.BooleanOperation.Intersection);
            if (!res.valid())
            {
                Console.WriteLine("Error: {0}", res.errorString);
                return;
            }

            // save result to STL file
            MR.MeshSave.toAnySupportedFormat(res.mesh, "out_boolean.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
