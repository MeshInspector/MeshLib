public static class SignedDistancesExample
{
    public static void Main(string[] args)
    {
        try
        {
            var refMesh = MR.MeshLoad.fromAnySupportedFormat("mesh1.ctm");
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh2.ctm");

            // get object of VertScalars - set of distances between points of target mesh and reference mesh
            var vertDistances = MR.findSignedDistances(refMesh, mesh);
            float minDist = float.MaxValue, maxDist = float.MinValue;
            for (int i = 0; i < (int)vertDistances.size(); ++i)
            {
                float d = vertDistances[new MR.VertId(i)];
                minDist = MathF.Min(minDist, d);
                maxDist = MathF.Max(maxDist, d);
            }
            Console.WriteLine("Distance between reference mesh and the closest point of target mesh is {0}", minDist);
            Console.WriteLine("Distance between reference mesh and the farthest point of target mesh is {0}", maxDist);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
