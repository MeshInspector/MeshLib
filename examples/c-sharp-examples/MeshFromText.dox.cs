public static class MeshFromTextExample
{
    public static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("Usage: MeshFromText fontpath text");
            Environment.Exit(1);
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
}
