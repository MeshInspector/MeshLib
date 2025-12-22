class Program
{
    public static void Main()
    {
        // Create a sphere mesh.
        MR.Mesh sphere1 = MR.MakeUVSphere(radius: 1, horisontalResolution: 64, verticalResolution: 64);

        // Copy it into another mesh.
        MR.Mesh sphere2 = new(sphere1);
        // Apply some offset.
        sphere2.Transform(MR.AffineXf3f.Translation(new MR.Vector3f{X = 0.7f}));

        MR.BooleanResult boolean_result = MR.Boolean(sphere1, sphere2, MR.BooleanOperation.Intersection);
        if (!boolean_result)
        {
            Console.WriteLine($"Failed to perform boolean: {(string)boolean_result.ErrorString}");
            Environment.Exit(1);
        }

        MR.Expected_Void_StdString save_result = MR.MeshSave.ToAnySupportedFormat(boolean_result.Mesh, "out_boolean.stl");
        if (save_result.GetError() is var error and not null)
        {
            Console.WriteLine($"Failed to save file: {(string)error}");
            Environment.Exit(1);
        }
    }
}
