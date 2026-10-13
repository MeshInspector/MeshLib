package mrtest;

import static mrtest.Common.PI_F;
import static mrtest.Common.check;
import static mrtest.Common.checkEq;

import meshlib.mr.AffineXf3f;
import meshlib.mr.BooleanOperation;
import meshlib.mr.BooleanParameters;
import meshlib.mr.BooleanResult;
import meshlib.mr.BooleanResultMapper;
import meshlib.mr.BooleanResultMapper_MapObject;
import meshlib.mr.BooleanResultMapper_Maps;
import meshlib.mr.FaceBitSet;
import meshlib.mr.FaceId;
import meshlib.mr.Functions;
import meshlib.mr.Matrix3f;
import meshlib.mr.Mesh;
import meshlib.mr.MeshPart;
import meshlib.mr.Triangulation;
import meshlib.mr.Turn;
import meshlib.mr.Vector3f;
import meshlib.mr.VertBitSet;
import meshlib.mr.VertCoords;
import meshlib.mr.VertId;
import meshlib.mr.std_array_MR_VertId_3;
import meshlib.mr.std_vector_MR_EdgeId;
import meshlib.mr.std_vector_std_vector_MR_EdgeId;

public final class TestMeshBoolean
{
    public static void meshBoolean()
    {
        Mesh meshA = Common.makeTorus(1.1f, 0.5f, 8, 8);
        Mesh meshB = Common.makeTorus(1.0f, 0.2f, 8, 8);
        {
            Matrix3f rot = Matrix3f.rotation(Vector3f.plusZ(), Vector3f.plusY());
            meshB.transform(AffineXf3f.linear(rot), null);
        }

        float shiftStep = 0.2f;
        float angleStep = PI_F;
        Vector3f[] baseAxis = {Vector3f.plusX(), Vector3f.plusY(), Vector3f.plusZ()};
        for (int maskTrans = 0; maskTrans < 8; maskTrans++)
        {
            for (int maskRot = 0; maskRot < 8; maskRot++)
            {
                for (float shift = 0.01f; shift < 0.2f; shift += shiftStep)
                {
                    Vector3f shiftVec = new Vector3f(0.0f, 0.0f, 0.0f);
                    for (int i = 0; i < 3; i++)
                    {
                        if ((maskTrans & (1 << i)) != 0)
                            shiftVec = Functions.opAdd(shiftVec, baseAxis[i]);
                    }
                    for (float angle = PI_F * 0.01f; angle < PI_F * 7.0f / 18.0f; angle += angleStep)
                    {
                        Matrix3f rotation = Matrix3f.identity();
                        for (int i = 0; i < 3; i++)
                        {
                            if ((maskRot & (1 << i)) != 0)
                            {
                                Matrix3f rot = Matrix3f.rotation(baseAxis[i], angle);
                                rotation = Functions.opMul(rot, rotation);
                            }
                        }
                        AffineXf3f xf = Functions.opMul(AffineXf3f.translation(shiftVec), AffineXf3f.linear(rotation));
                        BooleanParameters params = new BooleanParameters();
                        params.setRigidB2A(xf);
                        BooleanResult resultAB = Functions.boolean_4_const_MR_Mesh_ref(meshA, meshB, BooleanOperation.Union, params);
                        BooleanResult resultBA = Functions.boolean_4_const_MR_Mesh_ref(meshB, meshA, BooleanOperation.Union, params);
                        check(resultAB.valid(), "AB");
                        check(resultBA.valid(), "BA");
                    }
                }
            }
        }
    }

    static std_array_MR_VertId_3 tri(int a, int b, int c)
    {
        std_array_MR_VertId_3 ret = new std_array_MR_VertId_3();
        ret.setElems(0, new VertId(a));
        ret.setElems(1, new VertId(b));
        ret.setElems(2, new VertId(c));
        return ret;
    }

    public static void booleanMultipleEdgePropogationSort()
    {
        VertCoords points = new VertCoords(6);
        points.index_mut(new VertId(0)).set(new Vector3f(0.0f, 0.0f, 0.0f));
        points.index_mut(new VertId(1)).set(new Vector3f(-0.5f, 1.0f, 0.0f));
        points.index_mut(new VertId(2)).set(new Vector3f(0.5f, 1.0f, 0.0f));
        points.index_mut(new VertId(3)).set(new Vector3f(0.0f, 1.5f, 0.5f));
        points.index_mut(new VertId(4)).set(new Vector3f(-1.0f, 1.5f, 0.0f));
        points.index_mut(new VertId(5)).set(new Vector3f(1.0f, 1.5f, 0.0f));
        Triangulation triangulation = new Triangulation(5);
        triangulation.index_mut(new FaceId(0)).set(tri(0, 2, 1));
        triangulation.index_mut(new FaceId(1)).set(tri(1, 2, 3));
        triangulation.index_mut(new FaceId(2)).set(tri(3, 4, 1));
        triangulation.index_mut(new FaceId(3)).set(tri(2, 5, 3));
        triangulation.index_mut(new FaceId(4)).set(tri(3, 5, 4));
        Mesh meshA = Mesh.fromTriangles(points, triangulation, null, null);
        {
            Mesh meshASup = new Mesh(meshA);
            meshASup.points().index_mut(new VertId(3)).set(new Vector3f(0.0f, 1.5f, -0.5f));
            std_vector_MR_EdgeId holes = meshA.topology().findHoleRepresentiveEdges(null);
            std_vector_MR_EdgeId border = Functions.trackRightBoundaryLoop(meshA.topology(), holes.at(0), null, Turn.Rightmost);
            std_vector_std_vector_MR_EdgeId borderVec = new std_vector_std_vector_MR_EdgeId();
            borderVec.push_back(border);
            MeshPart mp = new MeshPart(meshASup, meshASup.topology().getValidFaces());
            meshA.addMeshPart(mp, true, borderVec, borderVec, null, null);
        }

        Mesh meshB = Functions.makeCube(Vector3f.diagonal(2.0f), Vector3f.diagonal(-0.5f));
        meshB.transform(AffineXf3f.translation(new Vector3f(-1.5f, -0.2f, -0.5f)), null);

        for (int i = 0; i < BooleanOperation.Count.value; i++)
        {
            BooleanResult resultAB = Functions.boolean_4_const_MR_Mesh_ref(meshA, meshB, new BooleanOperation(i), null);
            BooleanResult resultBA = Functions.boolean_4_const_MR_Mesh_ref(meshB, meshA, new BooleanOperation(i), null);
            check(resultAB.valid(), "AB " + i);
            check(resultBA.valid(), "BA " + i);
        }
    }

    public static void booleanMapper()
    {
        Mesh meshA = Common.makeTorus(1.1f, 0.5f, 8, 8);
        Mesh meshB = Common.makeTorus(1.0f, 0.2f, 8, 8);
        {
            Matrix3f rot = Matrix3f.rotation(Vector3f.plusZ(), Vector3f.plusY());
            meshB.transform(AffineXf3f.linear(rot), null);
        }

        BooleanResultMapper mapper = new BooleanResultMapper();
        BooleanResult result;
        {
            BooleanParameters parameters = new BooleanParameters();
            parameters.setMapper(mapper);
            result = Functions.boolean_4_const_MR_Mesh_ref(meshA, meshB, BooleanOperation.Union, parameters);
        }
        check(result.valid(), "valid");

        VertBitSet vMapA = mapper.map(meshA.topology().getValidVerts(), BooleanResultMapper_MapObject.A);
        VertBitSet vMapB = mapper.map(meshB.topology().getValidVerts(), BooleanResultMapper_MapObject.B);
        checkEq(Functions.opSub(vMapA, vMapB).count(), vMapA.count(), "disjoint verts");
        checkEq(vMapA.count(), 60, "verts A");
        checkEq(vMapB.count(), 48, "verts B");

        FaceBitSet fMapA = mapper.map(meshA.topology().getValidFaces(), BooleanResultMapper_MapObject.A);
        FaceBitSet fMapB = mapper.map(meshB.topology().getValidFaces(), BooleanResultMapper_MapObject.B);
        checkEq(Functions.opSub(fMapA, fMapB).count(), fMapA.count(), "disjoint faces");
        checkEq(fMapA.count(), 224, "faces A");
        checkEq(fMapB.count(), 192, "faces B");

        FaceBitSet newFaces = mapper.newFaces();
        checkEq(newFaces.size(), 416, "new faces size");
        checkEq(newFaces.count(), 252, "new faces count");

        BooleanResultMapper_Maps mapsA = mapper.getMaps(BooleanResultMapper_MapObject.A);
        checkEq(mapsA.old2newVerts().size(), 160, "old2newVerts A");
        checkEq(mapsA.cut2newFaces().size(), 280, "cut2newFaces A");
        checkEq(mapsA.cut2origin().size(), 280, "cut2origin A");
        BooleanResultMapper_Maps mapsB = mapper.getMaps(BooleanResultMapper_MapObject.B);
        checkEq(mapsB.old2newVerts().size(), 160, "old2newVerts B");
        checkEq(mapsB.cut2newFaces().size(), 320, "cut2newFaces B");
        checkEq(mapsB.cut2origin().size(), 320, "cut2origin B");
    }
}
