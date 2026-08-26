// Smoke test for the MeshLib NuGet package: builds a cube through the native
// library, so the managed wrapper, the netstandard2.0 facades and the win-x64
// natives all have to load for this to print the expected counts.
using System;

internal static class Program
{
    private const int ExpectedPoints = 8;
    private const int ExpectedFaces = 12;

    private static int Main()
    {
        Console.WriteLine("CLR {0}, {1}-bit process", Environment.Version, IntPtr.Size * 8);

        MR.Mesh cube = MR.makeCube(MR.Vector3f.diagonal(1), MR.Vector3f.diagonal(-0.5f));
        int points = (int)cube.points.size();
        int faces = (int)cube.topology.getValidFaces().count();
        Console.WriteLine("{0} points, {1} faces", points, faces);

        if (points != ExpectedPoints || faces != ExpectedFaces)
        {
            Console.Error.WriteLine("expected {0} points, {1} faces", ExpectedPoints, ExpectedFaces);
            return 1;
        }
        return 0;
    }
}
