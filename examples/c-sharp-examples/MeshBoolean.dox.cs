public class MeshBooleanExample
{
    public static void Main(string[] args)
    {
        try
        {
            MR.Mesh mesh_a, mesh_b;
            if (args.Length >= 2)
            {
                // load the two meshes given on the command line
                mesh_a = MR.MeshLoad.fromAnySupportedFormat(args[0]);
                mesh_b = MR.MeshLoad.fromAnySupportedFormat(args[1]);
            }
            else
            {
                // no input given: make two unit spheres shifted along X, so that they overlap
                mesh_a = MR.makeUVSphere(1.0f, 64, 64);
                mesh_b = MR.makeUVSphere(1.0f, 64, 64);
                mesh_b.transform(MR.AffineXf3f.translation(new MR.Vector3f(0.7f, 0.0f, 0.0f)));
            }

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
