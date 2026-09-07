using System;

internal static class Program
{
    private static int Main()
    {
        MR.Mesh cube = MR.makeCube(MR.Vector3f.diagonal(1), MR.Vector3f.diagonal(-0.5f));
        int points = (int)cube.points.size();
        int faces = (int)cube.topology.getValidFaces().count();
        Console.WriteLine("{0} points, {1} faces", points, faces);
        if (points == 8 && faces == 12)
            return 0;
        Console.Error.WriteLine("expected 8 points, 12 faces");
        return 1;
    }
}
