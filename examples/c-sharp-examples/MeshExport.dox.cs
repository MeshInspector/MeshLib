using static MR.DotNet;

public static class MeshExportExample
{
    public static void Run(string[] args)
    {
        try
        {
            var mesh = Mesh.MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );
            Console.WriteLine("Vertices coordinates:");
            for ( int i = 0; i < mesh.Points.Count; ++i )
            {
                var p = mesh.Points[i];
                Console.WriteLine( "Vertex {0} coordinates: {1}; {2}; {3}", i, p.X, p.Y, p.Z );
            }

            for ( int i = 0; i < mesh.Triangulation.Count; ++i )
            {
                var t = mesh.Triangulation[i];
                Console.WriteLine( "Triangle {0} vertices: {1}; {2}; {3}", i, t.v0.Id, t.v1.Id, t.v2.Id );
            }
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
