public static class MeshFromTextExample
{
    public static void Main(string[] args)
    {
        try
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: MeshFromText fontpath text");
                return;
            }
            string fontPath = args[0];
            string text = args[1];

            // set text-to-mesh parameters
            MR.SymbolMeshParams parameters = new();
            // input text
            parameters.text.assign(text);
            // font file name
            parameters.pathToFontFile.assign(fontPath);

            var mesh = MR.createSymbolsMesh(parameters);

            MR.MeshSave.toAnySupportedFormat(mesh, "mesh.ply");
        }
        catch (Exception e)
        {
            Console.WriteLine("Error: {0}", e.Message);
        }
    }
}
