using NUnitLite;
using static MR;

namespace MRTest
{
    class Program
    {
        public static int Main(string[] args)
        {
            // Create a sphere mesh.
            Mesh sphere1 = MakeUVSphere(radius: 1, horisontalResolution: 64, verticalResolution: 64);

            // Copy it into another mesh.
            Mesh sphere2 = new Mesh(sphere1);
            // Apply some offset.
            sphere2.Transform(AffineXf3f.Translation(new Vector3f { x = 0.7f }));

            BooleanResult boolean_result = Boolean(sphere1, sphere2, BooleanOperation.Intersection);
            if (!boolean_result)
            {
                Console.WriteLine($"Failed to perform boolean: {(string)boolean_result.errorString}");
                Environment.Exit(1);
            }

            MeshSave.ToAnySupportedFormat(boolean_result.mesh, "out_boolean.stl");

            /// ...

            Console.WriteLine("Starting tests...");
            return new AutoRun().Execute(args);
        }
    }
}
