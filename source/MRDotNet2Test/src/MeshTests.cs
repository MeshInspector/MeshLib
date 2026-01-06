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
            var mesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            mesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(mesh.points.Size() == 8);
            Assert.That(mesh.topology.GetValidFaces().Count() == 12);
        }

        [Test]
        public void TestSaveLoad()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".mrmesh";
            MeshSave.ToAnySupportedFormat(cubeMesh, tempFile);

            var readMesh = MeshLoad.FromAnySupportedFormat(tempFile);
            Assert.That(cubeMesh.points.Size() == readMesh.points.Size());
            Assert.That(cubeMesh.topology.GetValidFaces().Count() == readMesh.topology.GetValidFaces().Count());

            File.Delete(tempFile);
        }

        [Test]
        public void TestSaveLoadException()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".fakeextension";

            try
            {
                MeshSave.ToAnySupportedFormat(cubeMesh, tempFile);
            }
            catch (System.Exception e)
            {
                Assert.That(e.Message.Contains("Unsupported file extension"));
            }

            try
            {
                MeshLoad.FromAnySupportedFormat(tempFile);
            }
            catch (System.Exception e)
            {
                Assert.That(e.Message.Contains("Unsupported file extension"));
            }
        }

        [Test]
        public void TestSaveLoadCtm()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".ctm";
            MeshSave.ToAnySupportedFormat(cubeMesh, tempFile);

            var readMesh = MeshLoad.FromAnySupportedFormat(tempFile);
            Assert.That(readMesh.points.Size() == 8);
            Assert.That(readMesh.topology.GetValidFaces().Count() == 12);

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
            Assert.That(mesh.points.Count == 8);
            Assert.That(mesh.Triangulation.Count == 12);
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
            Assert.That(mesh.points.Count == 8);
            Assert.That(mesh.Triangulation.Count == 12);
        }
        */

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".mrmesh";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<Misc.UnexpectedResultException>(() => MeshLoad.FromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestTransform()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var xf = AffineXf3f.Translation(Vector3f.Diagonal(1.0f));
            cubeMesh.Transform(xf);

            Assert.That(cubeMesh.points.vec.At(0) == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(1) == new Vector3f(0.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(2) == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(3) == new Vector3f(1.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(4) == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.At(5) == new Vector3f(0.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.At(6) == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.At(7) == new Vector3f(1.5f, 0.5f, 1.5f));
        }

        [Test]
        public void TestTransformWithRegion()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var region = new VertBitSet(8);
            region.Set(new VertId(0));
            region.Set(new VertId(2));
            region.Set(new VertId(4));
            region.Set(new VertId(6));

            var xf = AffineXf3f.Translation(Vector3f.Diagonal(1.0f));
            cubeMesh.Transform(xf, region);

            Assert.That(cubeMesh.points.vec.At(0) == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(1) == new Vector3f(-0.5f, 0.5f, -0.5f));
            Assert.That(cubeMesh.points.vec.At(2) == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(3) == new Vector3f(0.5f, -0.5f, -0.5f));
            Assert.That(cubeMesh.points.vec.At(4) == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.At(5) == new Vector3f(-0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.points.vec.At(6) == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.points.vec.At(7) == new Vector3f(0.5f, -0.5f, 0.5f));
        }

        [Test]
        public void TestLeftTriVerts()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var triVerts = cubeMesh.topology.GetLeftTriVerts(new EdgeId(0));
            Assert.That(triVerts.elems[0].id, Is.EqualTo(0));
            Assert.That(triVerts.elems[1].id, Is.EqualTo(1));
            Assert.That(triVerts.elems[2].id, Is.EqualTo(2));

            triVerts = cubeMesh.topology.GetLeftTriVerts(new EdgeId(6));
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
                obj.mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
                obj.name = "Cube";
                obj.toWorld = new AffineXf3f(Vector3f.Diagonal(1));
                objects.Add(obj);

                obj.mesh = Mesh.MakeSphere(1.0f, 100);
                obj.name = "LongSphereName"; // must be long enough to deactivate short string optimization (SSO) in C++
                obj.toWorld = new AffineXf3f(Vector3f.Diagonal(-2));
                objects.Add(obj);

                var tempFile = Path.GetTempFileName() + ".obj";
                MeshSave.SceneToObj(objects, tempFile);

                var settings = new ObjLoadSettings();
                var loadedObjs = MeshLoad.FromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.Count == 2);

                var loadedMesh = loadedObjs[0].mesh;
                var loadedXf = loadedObjs[0].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if ( loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.Count == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedXf.B.X == 0.0f);

                loadedMesh = loadedObjs[1].mesh;
                loadedXf = loadedObjs[1].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.Count == 100);
                Assert.That(loadedObjs[1].name == "LongSphereName");
                Assert.That(loadedXf.B.X == 0.0f);

                settings.customXf = true;
                loadedObjs = MeshLoad.FromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.Count == 2);

                loadedMesh = loadedObjs[0].mesh;
                loadedXf = loadedObjs[0].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.Count == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedXf.B.X == 1.0f);

                loadedMesh = loadedObjs[1].mesh;
                loadedXf = loadedObjs[1].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.points.Count == 100);
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
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var triVerts = cubeMesh.topology.GetTriVerts(new FaceId(0));
            var centerPoint = (cubeMesh.points.Index(triVerts.elems[1]) + cubeMesh.points.Index(triVerts.elems[2])) * 0.5f;
            var triPoint = cubeMesh.ToTriPoint(new FaceId(0), centerPoint);
            Assert.That(triPoint.bary.a, Is.EqualTo(0.5f));
            Assert.That(triPoint.bary.b, Is.EqualTo(0.5f));
        }

        [Test]
        public void TestCalculatingVolume()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(cubeMesh.Volume(), Is.EqualTo(1.0).Within(1e-6));

            var validPoints = new FaceBitSet(8);
            validPoints.Set(new FaceId(0));
            validPoints.Set(new FaceId(1));
            validPoints.Set(new FaceId(3));
            validPoints.Set(new FaceId(4));
            validPoints.Set(new FaceId(5));
            validPoints.Set(new FaceId(7));

            Assert.That(cubeMesh.Volume(validPoints), Is.EqualTo(0.5).Within(1e-6));
        }

        [Test]
        public void TestAddMesh()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(cubeMesh.topology.GetValidFaces().Count() == 12);
            var cpyMesh = new Mesh(cubeMesh);
            cubeMesh.AddMesh(cpyMesh);
            Assert.That(cubeMesh.topology.GetValidFaces().Count() == 24);
            FaceBitSet bs = new FaceBitSet(12);
            bs.Set(new FaceId(0));
            MeshPart mp = new MeshPart(cpyMesh, bs);
            cubeMesh.AddMeshPart(mp);
            Assert.That(cubeMesh.topology.GetValidFaces().Count() == 25);
        }

        [Test]
        public void TestProjection()
        {
            var p = new Vector3f(1, 2, 3);
            var mp = new MeshPart(MakeSphere(new SphereParams(1.0f, 1000)));
            var projRes = FindProjection(p, mp);
            Assert.That(projRes.distSq, Is.EqualTo(7.529f).Within(1e-3));

            Assert.That(projRes.proj.face.id, Is.EqualTo(904));
            Assert.That(projRes.proj.point.x, Is.EqualTo(0.310).Within(1e-3));
            Assert.That(projRes.proj.point.y, Is.EqualTo(0.507).Within(1e-3));
            Assert.That(projRes.proj.point.z, Is.EqualTo(0.803).Within(1e-3));

            Assert.That(projRes.mtp.e.id, Is.EqualTo(1640));
            Assert.That(projRes.mtp.bary.a, Is.EqualTo(0.053).Within(1e-3));
            Assert.That(projRes.mtp.bary.b, Is.EqualTo(0.946).Within(1e-3));

            var xf = AffineXf3f.Translation(Vector3f.Diagonal(1.0f));
            projRes = FindProjection(p, mp, float.MaxValue, xf);

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
            var sphere1 = MakeUVSphere(1, 8, 8);

            var wholeSphere1 = new MeshPart(sphere1);
            var d11 = FindDistance(wholeSphere1, wholeSphere1);
            Assert.That(d11.distSq, Is.EqualTo(0));

            var zShift = AffineXf3f.Translation(new Vector3f(0.0f, 0.0f, 3.0f));
            var d1z = FindDistance(wholeSphere1, wholeSphere1, zShift);
            Assert.That(d1z.distSq, Is.EqualTo(1));

            Mesh sphere2 = MakeUVSphere(2, 8, 8);

            var wholeSphere2 = new MeshPart(sphere2);
            var d12 = FindDistance(wholeSphere1, wholeSphere2);
            var dist12 = Math.Sqrt(d12.distSq);
            Assert.That(dist12, Is.InRange(0.9, 1.0));
        }

        [Test]
        public void TestValidPoints()
        {
            Assert.DoesNotThrow(() =>
            {
                var mesh = MakeSphere(new SphereParams(1.0f, 3000));
                var count = mesh.topology.GetValidVerts().Count();
                Assert.That(count, Is.EqualTo(3000));
                mesh.Dispose();
            });
        }

        [Test]
        public void TestClone()
        {
            var mesh = MakeSphere(new SphereParams(1.0f, 3000));
            var clone = new Mesh(mesh);
            Assert.That(clone, Is.Not.SameAs(mesh));
            Assert.That(clone.points.Size(), Is.EqualTo(mesh.points.Size()));
            Assert.That(clone.topology.GetValidFaces().Count(), Is.EqualTo(mesh.topology.GetValidFaces().Count()));
            mesh.Dispose();
            clone.Dispose();
        }

        [Test]
        public void TestUniteCloseVertices()
        {
            var mesh = MakeSphere(new SphereParams(1.0f, 3000));
            Assert.That(mesh.topology.GetValidVerts().Count() == 3000);
            var old2new = new VertMap();
            var unitedCount = MeshBuilder.UniteCloseVertices(mesh, 0.1f, false, old2new);
            Assert.That(unitedCount, Is.EqualTo(2230));
            Assert.That(old2new.Index(new VertId(1000)).id, Is.EqualTo(42));
            Assert.That(mesh.topology.GetValidVerts().Count() < 3000);
            mesh.Dispose();
        }

        [Test]
        public void TestArea()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(cubeMesh.Area(), Is.EqualTo(6.0).Within(0.001));

            var faces = new FaceBitSet(12, true);
            for (int i = 0; i < 6; ++i)
                faces.Set(new FaceId(i), false);

            Assert.That(cubeMesh.Area(faces), Is.EqualTo(3.0).Within(0.001));

            cubeMesh.DeleteFaces(faces);
            Assert.That(cubeMesh.Area(), Is.EqualTo(3.0).Within(0.001));

            var holes = FindRightBoundary(cubeMesh.topology);
            Assert.That(holes.Size(), Is.EqualTo(1));
            Assert.That(holes.At(0).Size(), Is.EqualTo(6));

            var hole0 = TrackRightBoundaryLoop(cubeMesh.topology, holes.At(0).At(0));
            Assert.That(hole0.Size(), Is.EqualTo(holes.At(0).Size()));
            for (ulong i = 0; i < hole0.Size(); i++)
            {
                Assert.That(hole0.At(i).id, Is.EqualTo(holes.At(0).At(i).id));
            }
        }

        [Test]
        public void TestIncidentFacesFromVerts()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var verts = new VertBitSet(8, false);
            verts.Set(new VertId(0), true);
            var faces = GetIncidentFaces(cubeMesh.topology, verts);
            Assert.That(faces.Count(), Is.EqualTo(6));
        }

        [Test]
        public void TestIncidentFacesFromEdges()
        {
            var cubeMesh = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var edges = new UndirectedEdgeBitSet(12, false);
            edges.Set(new EdgeId(0), true);
            var faces = GetIncidentFaces(cubeMesh.topology, edges);
            Assert.That(faces.Count(), Is.EqualTo(8));
        }

        [Test]
        public void TestShortEdges()
        {
            var mesh = MakeTorus(1.0f, 0.05f, 16, 16);
            var shortEdges = FindShortEdges(new MeshPart( mesh ), 0.1f);
            Assert.That(shortEdges.Count(), Is.EqualTo(256));
        }
    }
}
