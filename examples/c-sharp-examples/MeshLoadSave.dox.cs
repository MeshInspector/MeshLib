public class MeshLoadSaveExample
{
    public static void Run(string[] args)
    {
        try
        {
            MR.Expected_MRMesh_StdString mesh_ex = MR.MeshLoad.FromAnySupportedFormat("mesh.stl");
            if (mesh_ex.GetError() is var mesh_error and not null)
                throw new Exception(mesh_error);

            MR.Expected_Void_StdString save_ex = MR.MeshSave.ToAnySupportedFormat(mesh_ex.GetValue()!, "mesh.ply");
            if (save_ex.GetError() is var save_error and not null)
                throw new Exception(save_error);
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
