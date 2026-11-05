public static class BooleanBenchmarkExample
{
    public static void Main(string[] args)
    {
        // the number of Booleans can be given in the command line
        int numBooleans = args.Length > 0 ? int.Parse(args[0]) : 2000;

        // two identical spheres of radius 1
        var sphereA = MR.makeSphere(new MR.SphereParams(1.0f, 3366));
        var sphereB = new MR.Mesh(sphereA);

        // subtracts sphereB rotated around Z-axis by i * 0.1 degrees from sphereA;
        // sphereB itself is not modified, so the AABB trees of both meshes are built once and reused
        void SubtractRotated(int i)
        {
            var rigidB2A = MR.AffineXf3f.linear(MR.Matrix3f.rotation(MR.Vector3f.plusZ(), i * 0.1f * MathF.PI / 180));
            var result = MR.boolean(sphereA, sphereB, MR.BooleanOperation.DifferenceAB, rigidB2A);
            if (!result.valid())
                throw new Exception(result.errorString);
        }

        // untimed warm-up Boolean, it builds the AABB trees
        SubtractRotated(1);

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 1; i <= numBooleans; ++i)
            SubtractRotated(i);
        double seconds = stopwatch.Elapsed.TotalSeconds;

        Console.WriteLine("{0} Booleans in {1:F3} s, {2:F1} Booleans/s", numBooleans, seconds, numBooleans / seconds);
    }
}
