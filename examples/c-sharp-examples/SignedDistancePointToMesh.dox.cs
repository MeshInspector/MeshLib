public static class SignedDistancePointToMeshExample
{
    public static void Main(string[] args)
    {
        var mesh1 = MR.MeshLoad.fromAnySupportedFormat("mesh1.ctm");
        var point = new MR.Vector3f(1.5f, 1.0f, 0.5f);

        var dist = MR.findSignedDistance(point, mesh1).value();
        if (dist is not null)
            Console.WriteLine("Signed distance from point to mesh {0}", dist.dist);
    }
}
