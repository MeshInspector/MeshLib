import meshlib.Functions;
import meshlib.Mesh;
import meshlib.VertCoords;
import meshlib.std_vector_std_array_MR_VertId_3;

// The class isn't public because of the `.dox.java` file name, the launcher doesn't mind.
class MeshExport
{
    public static void main(String[] args)
    {
        // create some mesh
        Mesh mesh = Functions.makeCube(null, null);

        // all vertices of valid triangles
        std_vector_std_array_MR_VertId_3 triangles = mesh.topology().getAllTriVerts();

        // all point coordinates
        VertCoords points = mesh.points();
        // triangle vertices as triples of ints (pointing to elements in points vector)
        int[] vertexTriples = new int[(int)triangles.size() * 3];
        for (int t = 0; t < triangles.size(); t++)
            for (int i = 0; i < 3; i++)
                vertexTriples[3 * t + i] = triangles.at(t).elems(i).id_();

        // TODO: export in your format
    }
}
