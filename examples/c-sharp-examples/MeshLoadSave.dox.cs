public static class MeshLoadSaveExample
{
    public static void Main(string[] args)
    {
        // Load mesh
        var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

        // Save mesh
        MR.MeshSave.toAnySupportedFormat(mesh, "mesh.ply");

        // More mesh formats are available in the MRIOExtras library
        MR.MeshSave.toCtm(mesh, "mesh.ctm");
    }
}
