public static class ContourTriangulationExample
{
    public static void Main(string[] args)
    {
        try
        {
            // load image as distance map
            var image = MR.ImageLoad.fromAnySupportedFormat("logo.jpg");
            var dm = MR.convertImageToDistanceMap(image);

            var pl2 = MR.distanceMapTo2DIsoPolyline(dm, 210.0f);
            var triangulationRes = MR.PlanarTriangulation.triangulateContours(pl2.contours());

            MR.MeshSave.toAnySupportedFormat(triangulationRes, "LogoMesh.ctm");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
