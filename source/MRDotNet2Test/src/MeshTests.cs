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

        /*
        [Test]
        public void TestFromTriangles()
        {
            List<Vector3f> points = new List<Vector3f>();
            points.Add(new Vector3f(0, 0, 0));
            points.Add(new Vector3f(0, 1, 0));
            points.Add(new Vector3f(1, 1, 0));
            points.Add(new Vector3f(1, 0, 0));
            points.Add(new Vector3f(0, 0, 1));
            points.Add(new Vector3f(0, 1, 1));
            points.Add(new Vector3f(1, 1, 1));
            points.Add(new Vector3f(1, 0, 1));

            List<ThreeVertIds> triangles = new List<ThreeVertIds>();
            triangles.Add(new ThreeVertIds(0, 1, 2));
            triangles.Add(new ThreeVertIds(2, 3, 0));
            triangles.Add(new ThreeVertIds(0, 4, 5));
            triangles.Add(new ThreeVertIds(5, 1, 0));
            triangles.Add(new ThreeVertIds(0, 3, 7));
            triangles.Add(new ThreeVertIds(7, 4, 0));
            triangles.Add(new ThreeVertIds(6, 5, 4));
            triangles.Add(new ThreeVertIds(4, 7, 6));
            triangles.Add(new ThreeVertIds(1, 5, 6));
            triangles.Add(new ThreeVertIds(6, 2, 1));
            triangles.Add(new ThreeVertIds(6, 7, 3));
            triangles.Add(new ThreeVertIds(3, 2, 6));

            var mesh = Mesh.FromTriangles(points, triangles);
            Assert.That(mesh.points.count == 8);
            Assert.That(mesh.Triangulation.count == 12);
        }

        [Test]
        public void TestFromTrianglesDuplicating()
        {
            List<Vector3f> points = new List<Vector3f>();
            points.Add(new Vector3f(0, 0, 0));
            points.Add(new Vector3f(0, 1, 0));
            points.Add(new Vector3f(1, 1, 0));
            points.Add(new Vector3f(1, 0, 0));
            points.Add(new Vector3f(0, 0, 1));
            points.Add(new Vector3f(0, 1, 1));
            points.Add(new Vector3f(1, 1, 1));
            points.Add(new Vector3f(1, 0, 1));

            List<ThreeVertIds> triangles = new List<ThreeVertIds>();
            triangles.Add(new ThreeVertIds(0, 1, 2));
            triangles.Add(new ThreeVertIds(2, 3, 0));
            triangles.Add(new ThreeVertIds(0, 4, 5));
            triangles.Add(new ThreeVertIds(5, 1, 0));
            triangles.Add(new ThreeVertIds(0, 3, 7));
            triangles.Add(new ThreeVertIds(7, 4, 0));
            triangles.Add(new ThreeVertIds(6, 5, 4));
            triangles.Add(new ThreeVertIds(4, 7, 6));
            triangles.Add(new ThreeVertIds(1, 5, 6));
            triangles.Add(new ThreeVertIds(6, 2, 1));
            triangles.Add(new ThreeVertIds(6, 7, 3));
            triangles.Add(new ThreeVertIds(3, 2, 6));

            var mesh = Mesh.FromTrianglesDuplicatingNonManifoldVertices(points, triangles);
            Assert.That(mesh.points.count == 8);
            Assert.That(mesh.Triangulation.count == 12);
        }
        */

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

            Assert.That(cubeMesh.points.vec.at(0) == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(1) == new Vector3f(0.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(2) == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(3) == new Vector3f(1.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(4) == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.at(5) == new Vector3f(0.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.at(6) == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.at(7) == new Vector3f(1.5f, 0.5f, 1.5f));
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

            Assert.That(cubeMesh.points.vec.at(0) == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(1) == new Vector3f(-0.5f, 0.5f, -0.5f));
            Assert.That(cubeMesh.points.vec.at(2) == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(3) == new Vector3f(0.5f, -0.5f, -0.5f));
            Assert.That(cubeMesh.points.vec.at(4) == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.at(5) == new Vector3f(-0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.at(6) == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.at(7) == new Vector3f(0.5f, -0.5f, 0.5f));
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

        /*
        [Test]
        public void TestSaveLoadToObj()
        {
            Assert.DoesNotThrow(() =>
            {
                var objects = new List<NamedMeshXf>();
                var obj = new NamedMeshXf();
                obj.mesh = Mesh.makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
                obj.name = "Cube";
                obj.toWorld = new AffineXf3f(Vector3f.diagonal(1));
                objects.Add(obj);

                obj.mesh = Mesh.makeSphere(1.0f, 100);
                obj.name = "LongSphereName"; // must be long enough to deactivate short string optimization (SSO) in C++
                obj.toWorld = new AffineXf3f(Vector3f.diagonal(-2));
                objects.Add(obj);

                var tempFile = Path.GetTempFileName() + ".obj";
                MeshSave.SceneToObj(objects, tempFile);

                var settings = new ObjLoadSettings();
                var loadedObjs = MeshLoad.FromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.count == 2);

                var loadedMesh = loadedObjs[0].mesh;
                var loadedXf = loadedObjs[0].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if ( loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.count == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedXf.B.X == 0.0f);

                loadedMesh = loadedObjs[1].mesh;
                loadedXf = loadedObjs[1].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.count == 100);
                Assert.That(loadedObjs[1].name == "LongSphereName");
                Assert.That(loadedXf.B.X == 0.0f);

                settings.customXf = true;
                loadedObjs = MeshLoad.FromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.count == 2);

                loadedMesh = loadedObjs[0].mesh;
                loadedXf = loadedObjs[0].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.count == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedXf.B.X == 1.0f);

                loadedMesh = loadedObjs[1].mesh;
                loadedXf = loadedObjs[1].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.count == 100);
                Assert.That(loadedObjs[1].name == "LongSphereName");
                Assert.That(loadedXf.B.X == -2.0f);

                loadedMesh = loadedObjs[0].mesh;
                if (loadedMesh is not null)
                    loadedMesh.Dispose();

                loadedMesh = loadedObjs[1].mesh;
                if (loadedMesh is not null)
                    loadedMesh.Dispose();

                File.Delete(tempFile);
            });
        }
        */

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
            var mp = new MeshPart(makeSphere(new SphereParams(1.0f, 1000)));
            var projRes = findProjection(p, mp);
            Assert.That(projRes.distSq, Is.EqualTo(7.529f).Within(1e-3));

            Assert.That(projRes.proj.face.id, Is.EqualTo(904));
            Assert.That(projRes.proj.point.x, Is.EqualTo(0.310).Within(1e-3));
            Assert.That(projRes.proj.point.y, Is.EqualTo(0.507).Within(1e-3));
            Assert.That(projRes.proj.point.z, Is.EqualTo(0.803).Within(1e-3));

            Assert.That(projRes.mtp.e.id, Is.EqualTo(1640));
            Assert.That(projRes.mtp.bary.a, Is.EqualTo(0.053).Within(1e-3));
            Assert.That(projRes.mtp.bary.b, Is.EqualTo(0.946).Within(1e-3));

            var xf = AffineXf3f.translation(Vector3f.diagonal(1.0f));
            projRes = findProjection(p, mp, float.MaxValue, xf);

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

            var wholeSphere1 = new MeshPart(sphere1);
            var d11 = findDistance(wholeSphere1, wholeSphere1);
            Assert.That(d11.distSq, Is.EqualTo(0));

            var zShift = AffineXf3f.translation(new Vector3f(0.0f, 0.0f, 3.0f));
            var d1z = findDistance(wholeSphere1, wholeSphere1, zShift);
            Assert.That(d1z.distSq, Is.EqualTo(1));

            Mesh sphere2 = makeUVSphere(2, 8, 8);

            var wholeSphere2 = new MeshPart(sphere2);
            var d12 = findDistance(wholeSphere1, wholeSphere2);
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
            Assert.That(holes.at(0).size(), Is.EqualTo(6));

            var hole0 = trackRightBoundaryLoop(cubeMesh.topology, holes.at(0).at(0));
            Assert.That(hole0.size(), Is.EqualTo(holes.at(0).size()));
            for (ulong i = 0; i < hole0.size(); i++)
            {
                Assert.That(hole0.at(i).id, Is.EqualTo(holes.at(0).at(i).id));
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
            var shortEdges = findShortEdges(new MeshPart( mesh ), 0.1f);
            Assert.That(shortEdges.count(), Is.EqualTo(256));
        }
    }
}
