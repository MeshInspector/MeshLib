package mrtest;

import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.AffineXf3f;
import meshlib.CoordinateConverters;
import meshlib.Functions;
import meshlib.Matrix3f;
import meshlib.Mesh;
import meshlib.MeshPart;
import meshlib.Vector3f;
import meshlib.Vector_MR_OneMeshContour;
import meshlib.Vector_MR_Vector_MR_VarEdgeTri;
import meshlib.Vector_MR_VarEdgeTri;

public final class TestMeshCollidePrecise
{
    public static void meshCollidePrecise()
    {
        Mesh meshA = Common.makeTorus(1.1f, 0.5f, 8, 8);
        Mesh meshB = Common.makeTorus(1.1f, 0.5f, 8, 8);
        Matrix3f rot = Matrix3f.rotation(Vector3f.plusZ(), new Vector3f(0.1f, 0.8f, 0.2f));
        meshB.transform(AffineXf3f.linear(rot), null);

        MeshPart partA = new MeshPart(meshA, null);
        MeshPart partB = new MeshPart(meshB, null);
        CoordinateConverters conv = Functions.getVectorConverters(partA, partB, null);
        Vector_MR_VarEdgeTri intersections = Functions.findCollidingEdgeTrisPrecise(partA, partB, conv.toInt(), null, null);
        checkEq(intersections.size(), 152, "intersections");

        Vector_MR_Vector_MR_VarEdgeTri contours = Functions.orderIntersectionContours(meshA.topology(), meshB.topology(), intersections);
        checkEq(contours.size(), 4, "contours");
        long[] sizes = new long[4];
        for (int i = 0; i < 4; i++)
            sizes[i] = contours.index(i).size();
        check(sizes[0] == 69 || sizes[0] == 71, "contour 0: " + sizes[0]); // Apple Debug
        check(sizes[1] == 9 || sizes[1] == 7, "contour 1: " + sizes[1]); // Apple Debug
        check(sizes[2] == 69 || sizes[2] == 71, "contour 2: " + sizes[2]); // with or without the FMA instruction
        check(sizes[3] == 9 || sizes[3] == 7, "contour 3: " + sizes[3]); // with or without the FMA instruction

        Vector_MR_OneMeshContour contoursA = new Vector_MR_OneMeshContour();
        Vector_MR_OneMeshContour contoursB = new Vector_MR_OneMeshContour();
        Functions.getOneMeshIntersectionContours(meshA, meshB, contours, contoursA, contoursB, conv, null, null, null);
        checkEq(contoursA.size(), 4, "contours A");
        checkEq(contoursB.size(), 4, "contours B");
        long posCount = 0;
        for (long i = 0; i < contoursA.size(); i++)
            posCount += contoursA.index(i).intersections().size();
        checkEq(posCount, 156, "intersections in contours");
    }
}
