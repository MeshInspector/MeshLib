using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class MeshShapesTest
    {
        [Test]
        public void TestCubeShape()
        {
            var cube = Mesh.MakeCube(new Vector3f(1.0f, 1.0f, 1.0f), new Vector3f(0,0,0));
            Assert.That(cube.Points.Count == 8);
        }

        [Test]
        public void TestCylinderShape()
        {
            var cylinder = Mesh.MakeCylinder(1.0f, 1.0f, 0.0f, (float)Math.PI * 2, 2.0f, 16);
            Assert.That(cylinder.Points.Count == 34 );
        }

        [Test]
        public void TestSphereShape()
        {
            var sphere = Mesh.MakeSphere(1.0f, 64); 
            Assert.That(sphere.Points.Count == 64);
        }

        [Test]
        public void TestTorusShape()
        {
            var torus = Mesh.MakeTorus(1.0f, 2.0f, 16, 16);
            Assert.That(torus.Points.Count == 256);
        }
    }
}
