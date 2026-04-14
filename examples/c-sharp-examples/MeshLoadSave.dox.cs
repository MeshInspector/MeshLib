public class MeshLoadSaveExample
{
    public static void Run(string[] args)
    {
        try
        {
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");
            MR.MeshSave.toAnySupportedFormat(mesh, "mesh.ply");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
