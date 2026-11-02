public static class MeshExportExample
{
    public static void Main(string[] args)
    {
        // create some mesh
        var mesh = MR.makeCube();

        // all vertices of valid triangles
        var triangles = mesh.topology.getAllTriVerts();

        // all point coordinates
        var points = mesh.points;
        // triangle vertices as triples of ints (pointing to elements in points vector)
        var vertexTriples = new int[(int)triangles.size() * 3];
        for (int t = 0; t < (int)triangles.size(); ++t)
        {
            var tri = triangles[(nuint)t].elems;
            vertexTriples[3 * t + 0] = tri._0.id;
            vertexTriples[3 * t + 1] = tri._1.id;
            vertexTriples[3 * t + 2] = tri._2.id;
        }

        // TODO: export in your format
    }
}
