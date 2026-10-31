public static class SignedDistanceExample
{
    public static void Main(string[] args)
    {
        try
        {
            var mesh1 = MR.MeshLoad.fromAnySupportedFormat("mesh1.ctm");
            var mesh2 = MR.MeshLoad.fromAnySupportedFormat("mesh2.ctm");

            var dist = MR.findSignedDistance(mesh1, mesh2);
            Console.WriteLine("Signed distance between meshes is {0}", dist.signedDist);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
