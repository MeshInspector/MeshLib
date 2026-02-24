using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class MeshTests
    {
        [Test]
        public void TestDoubleAssignment()
        {
            var mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            Assert.That(mesh.points.size() == 8);
            Assert.That(mesh.topology.getValidFaces().count() == 12);
        }

        [Test]
        public void TestSaveLoad()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".mrmesh";
            MeshSave.toAnySupportedFormat(cubeMesh, tempFile);

            var readMesh = MeshLoad.fromAnySupportedFormat(tempFile);
            Assert.That(cubeMesh.points.size() == readMesh.points.size());
            Assert.That(cubeMesh.topology.getValidFaces().count() == readMesh.topology.getValidFaces().count());

            File.Delete(tempFile);
        }

        [Test]
        public void TestSaveLoadException()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".fakeextension";

            try
            {
                MeshSave.toAnySupportedFormat(cubeMesh, tempFile);
            }
            catch (System.Exception e)
            {
                Assert.That(e.Message.Contains("Unsupported file extension"));
            }

            try
            {
                MeshLoad.fromAnySupportedFormat(tempFile);
            }
            catch (System.Exception e)
            {
                Assert.That(e.Message.Contains("Unsupported file extension"));
            }
        }

        [Test]
        public void TestSaveLoadCtm()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".ctm";
            MeshSave.toAnySupportedFormat(cubeMesh, tempFile);

            var readMesh = MeshLoad.fromAnySupportedFormat(tempFile);
            Assert.That(readMesh.points.size() == 8);
            Assert.That(readMesh.topology.getValidFaces().count() == 12);

            File.Delete(tempFile);
        }

        private static Std.Array_MRVertId_3 makeTri(int v0, int v1, int v2)
        {
            // TODO: array constructor
            Std.Array_MRVertId_3 tri;
            tri.elems._0 = new VertId(v0);
            tri.elems._1 = new VertId(v1);
            tri.elems._2 = new VertId(v2);
            return tri;
        }

        [Test]
        public void TestFromTriangles()
        {
            var points = new VertCoords();
            points.pushBack(new Vector3f(0, 0, 0));
            points.pushBack(new Vector3f(0, 1, 0));
            points.pushBack(new Vector3f(1, 1, 0));
            points.pushBack(new Vector3f(1, 0, 0));
            points.pushBack(new Vector3f(0, 0, 1));
            points.pushBack(new Vector3f(0, 1, 1));
            points.pushBack(new Vector3f(1, 1, 1));
            points.pushBack(new Vector3f(1, 0, 1));

            var triangles = new Triangulation();
            triangles.pushBack(makeTri(0, 1, 2));
            triangles.pushBack(makeTri(2, 3, 0));
            triangles.pushBack(makeTri(0, 4, 5));
            triangles.pushBack(makeTri(5, 1, 0));
            triangles.pushBack(makeTri(0, 3, 7));
            triangles.pushBack(makeTri(7, 4, 0));
            triangles.pushBack(makeTri(6, 5, 4));
            triangles.pushBack(makeTri(4, 7, 6));
            triangles.pushBack(makeTri(1, 5, 6));
            triangles.pushBack(makeTri(6, 2, 1));
            triangles.pushBack(makeTri(6, 7, 3));
            triangles.pushBack(makeTri(3, 2, 6));

            var mesh = Mesh.fromTriangles(points, triangles);
            Assert.That(mesh.points.size() == 8);
            Assert.That(mesh.topology.getValidFaces().size() == 12);
        }

        [Test]
        public void TestFromTrianglesDuplicating()
        {
            var points = new VertCoords();
            points.pushBack(new Vector3f(0, 0, 0));
            points.pushBack(new Vector3f(0, 1, 0));
            points.pushBack(new Vector3f(1, 1, 0));
            points.pushBack(new Vector3f(1, 0, 0));
            points.pushBack(new Vector3f(0, 0, 1));
            points.pushBack(new Vector3f(0, 1, 1));
            points.pushBack(new Vector3f(1, 1, 1));
            points.pushBack(new Vector3f(1, 0, 1));

            var triangles = new Triangulation();
            triangles.pushBack(makeTri(0, 1, 2));
            triangles.pushBack(makeTri(2, 3, 0));
            triangles.pushBack(makeTri(0, 4, 5));
            triangles.pushBack(makeTri(5, 1, 0));
            triangles.pushBack(makeTri(0, 3, 7));
            triangles.pushBack(makeTri(7, 4, 0));
            triangles.pushBack(makeTri(6, 5, 4));
            triangles.pushBack(makeTri(4, 7, 6));
            triangles.pushBack(makeTri(1, 5, 6));
            triangles.pushBack(makeTri(6, 2, 1));
            triangles.pushBack(makeTri(6, 7, 3));
            triangles.pushBack(makeTri(3, 2, 6));

            var mesh = Mesh.fromTrianglesDuplicatingNonManifoldVertices(points, triangles);
            Assert.That(mesh.points.size() == 8);
            Assert.That(mesh.topology.getValidFaces().size() == 12);
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".mrmesh";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<Misc.UnexpectedResultException>(() => MeshLoad.fromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestTransform()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var xf = AffineXf3f.translation(Vector3f.diagonal(1.0f));
            cubeMesh.transform(xf);

            Assert.That(cubeMesh.points.vec[0] == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[1] == new Vector3f(0.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[2] == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[3] == new Vector3f(1.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[4] == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.points.vec[5] == new Vector3f(0.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec[6] == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec[7] == new Vector3f(1.5f, 0.5f, 1.5f));
        }

        [Test]
        public void TestTransformWithRegion()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var region = new VertBitSet(8);
            region.set(new VertId(0));
            region.set(new VertId(2));
            region.set(new VertId(4));
            region.set(new VertId(6));

            var xf = AffineXf3f.translation(Vector3f.diagonal(1.0f));
            cubeMesh.transform(xf, region);

            Assert.That(cubeMesh.points.vec[0] == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[1] == new Vector3f(-0.5f, 0.5f, -0.5f));
            Assert.That(cubeMesh.points.vec[2] == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[3] == new Vector3f(0.5f, -0.5f, -0.5f));
            Assert.That(cubeMesh.points.vec[4] == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.points.vec[5] == new Vector3f(-0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec[6] == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec[7] == new Vector3f(0.5f, -0.5f, 0.5f));
        }

        [Test]
        public void TestLeftTriVerts()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var triVerts = cubeMesh.topology.getLeftTriVerts(new EdgeId(0));
            Assert.That(triVerts.elems[0].id, Is.EqualTo(0));
            Assert.That(triVerts.elems[1].id, Is.EqualTo(1));
            Assert.That(triVerts.elems[2].id, Is.EqualTo(2));

            triVerts = cubeMesh.topology.getLeftTriVerts(new EdgeId(6));
            Assert.That(triVerts.elems[0].id, Is.EqualTo(2));
            Assert.That(triVerts.elems[1].id, Is.EqualTo(3));
            Assert.That(triVerts.elems[2].id, Is.EqualTo(0));
        }

        [Test]
        public void TestSaveLoadToObj()
        {
            Assert.DoesNotThrow(() =>
            {
                var objects = new Std.Vector_MRMeshSaveNamedXfMesh();

                // TODO: fix empty NamedXfMesh construction
                // TODO: fix field assignment
                //var obj1 = new MeshSave.NamedXfMesh();
                //obj1.mesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
                //obj1.name = "Cube";
                //obj1.toWorld = AffineXf3f.translation(Vector3f.diagonal(1));
                var obj1 = new MeshSave.NamedXfMesh("Cube", AffineXf3f.translation(Vector3f.diagonal(1)), makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f)));
                objects.pushBack(Misc.Move(obj1));

                // TODO: fix field assignment
                //var obj2 = new MeshSave.NamedXfMesh();
                //obj2.mesh = makeSphere(new SphereParams(1.0f, 100));
                //obj2.name = "LongSphereName"; // must be long enough to deactivate short string optimization (SSO) in C++
                //obj2.toWorld = AffineXf3f.translation(Vector3f.diagonal(-2));
                var obj2 = new MeshSave.NamedXfMesh("LongSphereName", AffineXf3f.translation(Vector3f.diagonal(-2)), makeSphere(new SphereParams(1.0f, 100)));
                objects.pushBack(Misc.Move(obj2));

                var tempFile = Path.GetTempFileName() + ".obj";
                MeshSave.sceneToObj(objects, tempFile);

                var settings = new MeshLoad.ObjLoadSettings();
                var loadedObjs = MeshLoad.fromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.size() == 2);

                Assert.That(loadedObjs[0].mesh.points.size() == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedObjs[0].xf.b.x == 0.0f);

                Assert.That(loadedObjs[1].mesh.points.size() == 100);
                Assert.That(loadedObjs[1].name == "LongSphereName");
                Assert.That(loadedObjs[1].xf.b.x == 0.0f);

                settings.customXf = true;
                loadedObjs = MeshLoad.fromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.size() == 2);

                Assert.That(loadedObjs[0].mesh.points.size() == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedObjs[0].xf.b.x == 1.0f);

                Assert.That(loadedObjs[1].mesh.points.size() == 100);
                Assert.That(loadedObjs[1].name == "LongSphereName");
                Assert.That(loadedObjs[1].xf.b.x == -2.0f);

                File.Delete(tempFile);
            });
        }

        [Test]
        public void TestToTriPoint()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var triVerts = cubeMesh.topology.getTriVerts(new FaceId(0));
            var centerPoint = (cubeMesh.points[triVerts.elems[1]] + cubeMesh.points[triVerts.elems[2]]) * 0.5f;
            var triPoint = cubeMesh.toTriPoint(new FaceId(0), centerPoint);
            Assert.That(triPoint.bary.a, Is.EqualTo(0.5f));
            Assert.That(triPoint.bary.b, Is.EqualTo(0.5f));
        }

        [Test]
        public void TestCalculatingVolume()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            Assert.That(cubeMesh.volume(), Is.EqualTo(1.0).Within(1e-6));

            var validPoints = new FaceBitSet(8);
            validPoints.set(new FaceId(0));
            validPoints.set(new FaceId(1));
            validPoints.set(new FaceId(3));
            validPoints.set(new FaceId(4));
            validPoints.set(new FaceId(5));
            validPoints.set(new FaceId(7));

            Assert.That(cubeMesh.volume(validPoints), Is.EqualTo(0.5).Within(1e-6));
        }

        [Test]
        public void TestAddMesh()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            Assert.That(cubeMesh.topology.getValidFaces().count() == 12);
            var cpyMesh = new Mesh(cubeMesh);
            cubeMesh.addMesh(cpyMesh);
            Assert.That(cubeMesh.topology.getValidFaces().count() == 24);
            FaceBitSet bs = new FaceBitSet(12);
            bs.set(new FaceId(0));
            MeshPart mp = new MeshPart(cpyMesh, bs);
            cubeMesh.addMeshPart(mp);
            Assert.That(cubeMesh.topology.getValidFaces().count() == 25);
        }

        [Test]
        public void TestProjection()
        {
            var p = new Vector3f(1, 2, 3);
            var mesh = makeSphere(new SphereParams(1.0f, 1000));
            var projRes = findProjection(p, mesh);
            Assert.That(projRes.distSq, Is.EqualTo(7.529f).Within(1e-3));

            Assert.That(projRes.proj.face.id, Is.EqualTo(904));
            Assert.That(projRes.proj.point.x, Is.EqualTo(0.310).Within(1e-3));
            Assert.That(projRes.proj.point.y, Is.EqualTo(0.507).Within(1e-3));
            Assert.That(projRes.proj.point.z, Is.EqualTo(0.803).Within(1e-3));

            Assert.That(projRes.mtp.e.id, Is.EqualTo(1640));
            Assert.That(projRes.mtp.bary.a, Is.EqualTo(0.053).Within(1e-3));
            Assert.That(projRes.mtp.bary.b, Is.EqualTo(0.946).Within(1e-3));

            var xf = AffineXf3f.translation(Vector3f.diagonal(1.0f));
            projRes = findProjection(p, mesh, float.MaxValue, xf);

            Assert.That(projRes.proj.face.id, Is.EqualTo(632));
            Assert.That(projRes.proj.point.x, Is.EqualTo(1.000).Within(1e-3));
            Assert.That(projRes.proj.point.y, Is.EqualTo(1.439).Within(1e-3));
            Assert.That(projRes.proj.point.z, Is.EqualTo(1.895).Within(1e-3));

            Assert.That(projRes.mtp.e.id, Is.EqualTo(1898));
            Assert.That(projRes.mtp.bary.a, Is.EqualTo(0.5).Within(1e-3));
            Assert.That(projRes.mtp.bary.b, Is.EqualTo(0.0).Within(1e-3));
        }

        [Test]
        public void TestMeshMeshDistance()
        {
            var sphere1 = makeUVSphere(1, 8, 8);

            var d11 = findDistance(sphere1, sphere1);
            Assert.That(d11.distSq, Is.EqualTo(0));

            var zShift = AffineXf3f.translation(new Vector3f(0.0f, 0.0f, 3.0f));
            var d1z = findDistance(sphere1, sphere1, zShift);
            Assert.That(d1z.distSq, Is.EqualTo(1));

            Mesh sphere2 = makeUVSphere(2, 8, 8);

            var d12 = findDistance(sphere1, sphere2);
            var dist12 = Math.Sqrt(d12.distSq);
            Assert.That(dist12, Is.InRange(0.9, 1.0));
        }

        [Test]
        public void TestValidPoints()
        {
            Assert.DoesNotThrow(() =>
            {
                var mesh = makeSphere(new SphereParams(1.0f, 3000));
                var count = mesh.topology.getValidVerts().count();
                Assert.That(count, Is.EqualTo(3000));
                mesh.Dispose();
            });
        }

        [Test]
        public void TestClone()
        {
            var mesh = makeSphere(new SphereParams(1.0f, 3000));
            var clone = new Mesh(mesh);
            Assert.That(clone, Is.Not.SameAs(mesh));
            Assert.That(clone.points.size(), Is.EqualTo(mesh.points.size()));
            Assert.That(clone.topology.getValidFaces().count(), Is.EqualTo(mesh.topology.getValidFaces().count()));
            mesh.Dispose();
            clone.Dispose();
        }

        [Test]
        public void TestUniteCloseVertices()
        {
            var mesh = makeSphere(new SphereParams(1.0f, 3000));
            Assert.That(mesh.topology.getValidVerts().count() == 3000);
            var old2new = new VertMap();
            var unitedCount = MeshBuilder.uniteCloseVertices(mesh, 0.1f, false, old2new);
            Assert.That(unitedCount, Is.EqualTo(2230));
            Assert.That(old2new[new VertId(1000)].id, Is.EqualTo(42));
            Assert.That(mesh.topology.getValidVerts().count() < 3000);
            mesh.Dispose();
        }

        [Test]
        public void TestArea()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            Assert.That(cubeMesh.area(), Is.EqualTo(6.0).Within(0.001));

            var faces = new FaceBitSet(12, true);
            for (int i = 0; i < 6; ++i)
                faces.set(new FaceId(i), false);

            Assert.That(cubeMesh.area(faces), Is.EqualTo(3.0).Within(0.001));

            cubeMesh.deleteFaces(faces);
            Assert.That(cubeMesh.area(), Is.EqualTo(3.0).Within(0.001));

            var holes = findRightBoundary(cubeMesh.topology);
            Assert.That(holes.size(), Is.EqualTo(1));
            Assert.That(holes[0].size(), Is.EqualTo(6));

            var hole0 = trackRightBoundaryLoop(cubeMesh.topology, holes[0][0]);
            Assert.That(hole0.size(), Is.EqualTo(holes[0].size()));
            for (ulong i = 0; i < hole0.size(); i++)
            {
                Assert.That(hole0[i].id, Is.EqualTo(holes[0][i].id));
            }
        }

        [Test]
        public void TestIncidentFacesFromVerts()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var verts = new VertBitSet(8, false);
            verts.set(new VertId(0), true);
            var faces = getIncidentFaces(cubeMesh.topology, verts);
            Assert.That(faces.count(), Is.EqualTo(6));
        }

        [Test]
        public void TestIncidentFacesFromEdges()
        {
            var cubeMesh = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var edges = new UndirectedEdgeBitSet(12, false);
            edges.set(new EdgeId(0), true);
            var faces = getIncidentFaces(cubeMesh.topology, edges);
            Assert.That(faces.count(), Is.EqualTo(8));
        }

        [Test]
        public void TestShortEdges()
        {
            var mesh = makeTorus(1.0f, 0.05f, 16, 16);
            var shortEdges = findShortEdges(mesh, 0.1f);
            Assert.That(shortEdges.count(), Is.EqualTo(256));
        }
    }
}
