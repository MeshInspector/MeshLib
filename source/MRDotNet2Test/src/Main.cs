using NUnitLite;
using static MR;

namespace MRTest
{
    class Program
    {
        public static int Main(string[] args)
        {
            // Create a sphere mesh.
            Mesh sphere1 = makeUVSphere(radius: 1, horisontalResolution: 64, verticalResolution: 64);

            // Copy it into another mesh.
            Mesh sphere2 = new Mesh(sphere1);
            // Apply some offset.
            sphere2.transform(AffineXf3f.translation(new Vector3f { x = 0.7f }));

            BooleanResult boolean_result = boolean(sphere1, sphere2, BooleanOperation.Intersection);
            if (!boolean_result)
            {
                Console.WriteLine($"Failed to perform boolean: {(string)boolean_result.errorString}");
                Environment.Exit(1);
            }

            MeshSave.toAnySupportedFormat(boolean_result.mesh, "out_boolean.stl");

            /// ...

            Console.WriteLine("Starting tests...");
            return new AutoRun().Execute(args);
        }
    }
}
