public static class MeshExtrudeExample
{
    public static void Main(string[] args)
    {
        try
        {
            // Load mesh
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

            // Select faces to extrude
            MR.FaceBitSet facesToExtrude = new();
            facesToExtrude.autoResizeSet(new MR.FaceId(1));
            facesToExtrude.autoResizeSet(new MR.FaceId(2));

            // Create duplicated verts on region boundary
            MR.makeDegenerateBandAroundRegion(mesh, facesToExtrude);

            // Find vertices that will be moved
            var vertsForMove = MR.getIncidentVerts(mesh.topology, facesToExtrude);

            foreach (var v in vertsForMove)
            {
                // Move each vertex
                mesh.points[v] += MR.Vector3f.plusZ();
            }

            // Invalidate internal caches after manual changing
            mesh.invalidateCaches();

            // Save mesh
            MR.MeshSave.toAnySupportedFormat(mesh, "extrudedMesh.stl");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
