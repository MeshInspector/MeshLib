using static MR.DotNet;

public class MeshLoadSaveExample
{
    public static void Run(string[] args)
    {
        try
        {
            var mesh = MeshLoad.FromAnySupportedFormat("mesh.stl");
            MeshSave.ToAnySupportedFormat(mesh, "mesh.ply");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}

