public class MeshLoadSaveExample
{
    public static void Run(string[] args)
    {
        try
        {
            var mesh = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");
            MR.MeshSave.ToAnySupportedFormat(mesh, "mesh.ply");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
