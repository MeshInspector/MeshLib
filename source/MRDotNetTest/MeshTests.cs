using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class MeshTests
    {
        [Test]
        public void TestDoubleAssignment()
        {
            var mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That( mesh.Points.Count == 8 );
            Assert.That( mesh.Triangulation.Count == 12 );
        }

        [Test]
        public void TestSaveLoad()
        {
            var cubeMesh = Mesh.MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );
            var tempFile = Path.GetTempFileName() + ".mrmesh";
            Mesh.ToAnySupportedFormat( cubeMesh, tempFile );

            var readMesh = Mesh.FromAnySupportedFormat( tempFile );
            Assert.That( cubeMesh == readMesh );

            File.Delete( tempFile );
        }

        [Test]
        public void TestFromTriangles()
        {
            List<Vector3f> points = new List<Vector3f>();
            points.Add( new Vector3f( 0, 0, 0 ) );
            points.Add(new Vector3f(0, 1, 0));
            points.Add(new Vector3f(1, 1, 0));
            points.Add(new Vector3f(1, 0, 0));
            points.Add(new Vector3f(0, 0, 1));
            points.Add(new Vector3f(0, 1, 1));
            points.Add(new Vector3f(1, 1, 1));
            points.Add(new Vector3f(1, 0, 1));

            List<ThreeVertIds> triangles = new List<ThreeVertIds>();
            triangles.Add( new ThreeVertIds( 0, 1, 2 ) );
            triangles.Add( new ThreeVertIds( 2, 3, 0 ) );
            triangles.Add( new ThreeVertIds( 0, 4, 5 ) );
            triangles.Add( new ThreeVertIds( 5, 1, 0 ) );
            triangles.Add( new ThreeVertIds( 0, 3, 7 ) );
            triangles.Add( new ThreeVertIds( 7, 4, 0 ) );
            triangles.Add( new ThreeVertIds( 6, 5, 4 ) );
            triangles.Add( new ThreeVertIds( 4, 7, 6 ) );
            triangles.Add( new ThreeVertIds( 1, 5, 6 ) );
            triangles.Add( new ThreeVertIds( 6, 2, 1 ) );
            triangles.Add( new ThreeVertIds( 6, 7, 3 ) );
            triangles.Add( new ThreeVertIds( 3, 2, 6 ) );
            
            var mesh = Mesh.FromTriangles( points, triangles );
            Assert.That( mesh.Points.Count == 8 );
            Assert.That( mesh.Triangulation.Count == 12 );
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".mrmesh";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<SystemException>( () => Mesh.FromAnySupportedFormat(path)) ;
            File.Delete(path);
        }

        [Test]
        public void TestNullArgs()
        {
            Assert.Throws<ArgumentNullException>( () => Mesh.FromAnySupportedFormat(null));
            Assert.Throws<ArgumentNullException>( () => Mesh.ToAnySupportedFormat(null, null));
            Assert.Throws<ArgumentNullException>( () => Mesh.MakeCube(null, null));
            Assert.Throws<ArgumentException>( () => Mesh.MakeSphere( 0.0f, -50 ) );

            Assert.Throws<ArgumentNullException>( () => Mesh.FromTriangles(null, null) );
            Assert.Throws<ArgumentNullException>( () => Mesh.FromTrianglesDuplicatingNonManifoldVertices( null, null ) );
        }

        [Test]
        public void TestTransform()
        {
            var cubeMesh = Mesh.MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );
            var xf = new AffineXf3f(Vector3f.Diagonal(1.0f));
            cubeMesh.Transform( xf );

            Assert.That(cubeMesh.Points[0] == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[1] == new Vector3f(0.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.Points[2] == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.Points[3] == new Vector3f(1.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[4] == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.Points[5] == new Vector3f(0.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.Points[6] == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.Points[7] == new Vector3f(1.5f, 0.5f, 1.5f));
        }

        [Test]
        public void TestTransformWithRegion()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var region = new BitSet(8);
            region.Set(0);
            region.Set(2);
            region.Set(4);
            region.Set(6);

            var xf = new AffineXf3f(Vector3f.Diagonal(1.0f));
            cubeMesh.Transform(xf, region);

            Assert.That(cubeMesh.Points[0] == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[1] == new Vector3f(-0.5f, 0.5f, -0.5f));
            Assert.That(cubeMesh.Points[2] == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.Points[3] == new Vector3f(0.5f, -0.5f, -0.5f));
            Assert.That(cubeMesh.Points[4] == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.Points[5] == new Vector3f(-0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[6] == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.Points[7] == new Vector3f(0.5f, -0.5f, 0.5f));
        }

    }
}
