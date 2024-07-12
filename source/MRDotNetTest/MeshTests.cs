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
        public void TestMeshFromFile()
        {
            var path = TestTools.GetPathToTestFile( "cube.mrmesh" );
            var mesh = Mesh.FromFile( path );
            Assert.That( mesh.Points.Count == 8 );
            Assert.That( mesh.Triangulation.Count == 12 );
        }

        [Test]
        public void TestSaveLoad()
        {
            var cubeMesh = Mesh.MakeCube( Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f) );
            var tempFile = Path.GetTempFileName() + ".mrmesh";
            Mesh.ToFile( cubeMesh, tempFile );

            var readMesh = Mesh.FromFile( tempFile );
            Assert.That( cubeMesh == readMesh );

            File.Delete( tempFile );
        }

        [Test]
        public void TestSpartan()
        {
            var mesh = Mesh.FromFile( TestTools.GetPathToTestFile( "spartan.mrmesh" ) );
            Assert.That( mesh.Points.Count == 68230 );            
            Assert.That( mesh.Triangulation.Count == 136948 );
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
            Assert.That( TestTools.AreMeshesEqual( mesh, TestTools.GetPathToPattern( "cube_from_triangles.mrmesh" ) ) );
        }
    }
}
