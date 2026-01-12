public static class MeshExportExample
{
    public static void Run(string[] args)
    {
        try
        {
            MR.Mesh mesh = MR.MakeCube(MR.Vector3f.Diagonal(1), MR.Vector3f.Diagonal(-0.5f));
            Console.WriteLine("Vertices coordinates:");
            for (ulong i = 0; i < mesh.points.Size(); ++i)
            {
                var p = mesh.points.Index(new MR.VertId(i));
                Console.WriteLine("Vertex {0} coordinates: {1}; {2}; {3}", i, p.x, p.y, p.z);
            }

            MR.Triangulation tri = mesh.topology.GetTriangulation();

            for (ulong i = 0; i < tri.Size(); ++i)
            {
                var t = tri.Index(new MR.FaceId(i));
                Console.WriteLine("Triangle {0} vertices: {1}; {2}; {3}", i, t.elems._0.id, t.elems._1.id, t.elems._2.id);
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
