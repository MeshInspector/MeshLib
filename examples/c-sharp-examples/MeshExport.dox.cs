public static class MeshExportExample
{
    public static void Run(string[] args)
    {
        try
        {
            MR.Mesh mesh = MR.makeCube(MR.Vector3f.diagonal(1), MR.Vector3f.diagonal(-0.5f));
            Console.WriteLine("Vertices coordinates:");
            for (ulong i = 0; i < mesh.points.size(); ++i)
            {
                var p = mesh.points[new MR.VertId(i)];
                Console.WriteLine("Vertex {0} coordinates: {1}; {2}; {3}", i, p.x, p.y, p.z);
            }

            MR.Triangulation tri = mesh.topology.getTriangulation();

            for (ulong i = 0; i < tri.size(); ++i)
            {
                var t = tri[new MR.FaceId(i)];
                Console.WriteLine("Triangle {0} vertices: {1}; {2}; {3}", i, t.elems._0.id, t.elems._1.id, t.elems._2.id);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
