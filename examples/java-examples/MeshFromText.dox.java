import meshlib.Functions;
import meshlib.Mesh;
import meshlib.SymbolMeshParams;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshFromText
{
    public static void main(String[] args)
    {
        if (args.length < 2)
        {
            System.err.println("Usage: MeshFromText fontpath text");
            System.exit(1);
        }
        String fontPath = args[0];
        String text = args[1];

        // set text-to-mesh parameters
        SymbolMeshParams params = new SymbolMeshParams();
        // input text
        params.setText(text);
        // font file name
        params.setPathToFontFile(fontPath);
        // An error is thrown as `UnexpectedResultException`.
        Mesh mesh = Functions.createSymbolsMesh(params);

        meshlib.meshsave.Functions.toAnySupportedFormat(mesh, "mesh.ply", null);
    }
}
