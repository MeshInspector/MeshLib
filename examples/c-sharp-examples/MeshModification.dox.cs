public static class MeshModificationExample
{
    public static void Main(string[] args)
    {
        var mesh = MR.makeTorus();

        // Relax mesh (5 iterations)
        MR.MeshRelaxParams relaxParams = new();
        relaxParams.iterations = 5;
        MR.relax(mesh, relaxParams);

        // Subdivide mesh
        MR.SubdivideSettings props = new();
        props.maxDeviationAfterFlip = 0.5f;
        MR.subdivideMesh(mesh, props);

        // Rotate mesh
        var rotationXf = MR.AffineXf3f.linear(MR.Matrix3f.rotation(MR.Vector3f.plusZ(), MathF.PI * 0.5f));
        mesh.transform(rotationXf);
    }
}
