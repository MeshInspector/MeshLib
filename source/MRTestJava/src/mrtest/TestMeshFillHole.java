package mrtest;

import static mrtest.Common.checkEq;

import meshlib.mr.FaceBitSet;
import meshlib.mr.FaceId;
import meshlib.mr.FillHoleNicelySettings;
import meshlib.mr.FillHoleParams;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.Triangulation;
import meshlib.mr.Vector3f;
import meshlib.mr.VertCoords;
import meshlib.mr.VertId;
import meshlib.mr.std_vector_MR_EdgeId;
import meshlib.mr.std_vector_std_vector_MR_EdgeId;

public final class TestMeshFillHole
{
    static Mesh createMeshWithHoles()
    {
        VertCoords points = new VertCoords(6);
        points.index_mut(new VertId(0)).set(new Vector3f(0.0f, 0.0f, 0.0f));
        points.index_mut(new VertId(1)).set(new Vector3f(1.0f, 0.0f, 0.0f));
        points.index_mut(new VertId(2)).set(new Vector3f(0.0f, 1.0f, 0.0f));
        points.index_mut(new VertId(3)).set(new Vector3f(0.0f, 0.0f, 1.0f));
        points.index_mut(new VertId(4)).set(new Vector3f(1.0f, 0.0f, 1.0f));
        points.index_mut(new VertId(5)).set(new Vector3f(0.0f, 1.0f, 1.0f));
        Triangulation triangulation = new Triangulation(5);
        triangulation.index_mut(new FaceId(0)).set(TestMeshBoolean.tri(0, 2, 1));
        triangulation.index_mut(new FaceId(1)).set(TestMeshBoolean.tri(3, 4, 5));
        triangulation.index_mut(new FaceId(2)).set(TestMeshBoolean.tri(0, 1, 3));
        triangulation.index_mut(new FaceId(3)).set(TestMeshBoolean.tri(2, 5, 4));
        triangulation.index_mut(new FaceId(4)).set(TestMeshBoolean.tri(2, 3, 5));
        return Mesh.fromTriangles(points, triangulation, null, null);
    }

    public static void meshFillHole()
    {
        Mesh mesh = createMeshWithHoles();
        std_vector_MR_EdgeId oldBdEdges = mesh.topology().findHoleRepresentiveEdges(null);
        checkEq(oldBdEdges.size(), 2, "holes");

        FaceBitSet newFaces = new FaceBitSet();
        {
            FillHoleParams params = new FillHoleParams();
            params.setOutNewFaces(newFaces);
            Functions.fillHoles(mesh, oldBdEdges, params);
        }
        checkEq(newFaces.count(), 3, "new faces");

        std_vector_MR_EdgeId newBdEdges = mesh.topology().findHoleRepresentiveEdges(null);
        checkEq(newBdEdges.size(), 0, "no holes");
    }

    public static void meshFillHoleNicely()
    {
        Mesh mesh = createMeshWithHoles();
        std_vector_MR_EdgeId oldBdEdges = mesh.topology().findHoleRepresentiveEdges(null);
        checkEq(oldBdEdges.size(), 2, "holes");

        FillHoleNicelySettings settings = new FillHoleNicelySettings();
        FaceBitSet patch = Functions.fillHoleNicely(mesh, oldBdEdges.front(), settings);
        checkEq(patch.count(), 1887, "patch");

        std_vector_MR_EdgeId newBdEdges = mesh.topology().findHoleRepresentiveEdges(null);
        checkEq(newBdEdges.size(), 1, "one hole left");
    }

    public static void rightBoundary()
    {
        Mesh mesh = createMeshWithHoles();
        std_vector_std_vector_MR_EdgeId loops = Functions.findRightBoundary(mesh.topology(), null, null);
        checkEq(loops.size(), 2, "loops");
        checkEq(loops.at(0).size(), 3, "loop 0");
        checkEq(loops.at(1).size(), 4, "loop 1");
    }

    public static void findHoleComplicatingFaces()
    {
        Mesh mesh = createMeshWithHoles();
        FaceBitSet faces = Functions.findHoleComplicatingFaces(mesh);
        checkEq(faces.count(), 0, "complicating faces");
    }
}
