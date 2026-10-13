package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.FaceBitSet;
import meshlib.mr.FaceId;
import meshlib.mr.Functions;
import meshlib.mr.Mesh;
import meshlib.mr.MeshPart;
import meshlib.mr.UndirectedEdgeBitSet;
import meshlib.mr.UndirectedEdgeId;
import meshlib.mr.VertBitSet;
import meshlib.mr.VertId;
import meshlib.mr.expected_MR_UndirectedEdgeBitSet_std_string;

public final class TestMesh
{
    public static void area()
    {
        Mesh mesh = Common.createCube();
        double area = mesh.area((FaceBitSet)null);
        check(area > 5.999 && area < 6.001, "whole area");

        FaceBitSet faces = new FaceBitSet();
        faces.resize(12, true);
        for (int i = 6; i < 12; i++)
            faces.set(new FaceId(i), false);
        area = mesh.area(faces);
        check(area > 2.999 && area < 3.001, "region area");

        mesh.deleteFaces(faces, null);
        area = mesh.area((FaceBitSet)null);
        check(area > 2.999 && area < 3.001, "area after delete");
    }

    public static void shortEdges()
    {
        Mesh mesh = Common.makeTorus(1.0f, 0.05f, 16, 16);
        MeshPart mp = new MeshPart(mesh, null);
        expected_MR_UndirectedEdgeBitSet_std_string shortEdges = Functions.findShortEdges(mp, 0.1f, null);
        check(shortEdges.success(), "findShortEdges failed");
        checkEq(shortEdges.value().count(), 256, "short edges");
    }

    public static void incidentFacesFromVerts()
    {
        Mesh mesh = Common.createCube();
        VertBitSet verts = new VertBitSet();
        verts.resize(8, false);
        verts.set(new VertId(0), true);
        FaceBitSet faces = Functions.getIncidentFaces(mesh.topology(), verts);
        checkEq(faces.count(), 6, "incident faces");
    }

    public static void incidentFacesFromEdges()
    {
        Mesh mesh = Common.createCube();
        UndirectedEdgeBitSet edges = new UndirectedEdgeBitSet();
        edges.resize(12, false);
        edges.set(new UndirectedEdgeId(0), true);
        FaceBitSet faces = Functions.getIncidentFaces(mesh.topology(), edges);
        checkEq(faces.count(), 8, "incident faces");
    }
}
