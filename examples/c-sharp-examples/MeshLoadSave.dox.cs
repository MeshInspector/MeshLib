public class MeshLoadSaveExample
{
    public static void Main(string[] args)
    {
        try
        {
            // This example needs an input mesh; create one if you do not have it already
            if (!File.Exists("mesh.stl"))
                MR.MeshSave.toAnySupportedFormat(MR.makeCube(), "mesh.stl");

            // Load mesh
            var mesh = MR.MeshLoad.fromAnySupportedFormat("mesh.stl");

            // Save it in another format
            MR.MeshSave.toAnySupportedFormat(mesh, "mesh.ply");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
