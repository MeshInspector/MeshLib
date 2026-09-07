using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class MeshShapesTest
    {
        [Test]
        public void TestCubeShape()
        {
            var cube = makeCube(new Vector3f(1.0f, 1.0f, 1.0f), new Vector3f(0, 0, 0));
            Assert.That(cube.points.size() == 8);
        }

        [Test]
        public void TestCylinderShape()
        {
            var cylinder = makeCylinderAdvanced(1.0f, 1.0f, 0.0f, (float)Math.PI * 2, 2.0f, 16);
            Assert.That(cylinder.points.size() == 34 );
        }

        [Test]
        public void TestSphereShape()
        {
            var sphere = makeSphere(new SphereParams(1.0f, 64));
            Assert.That(sphere.points.size() == 64);
        }

        [Test]
        public void TestTorusShape()
        {
            var torus = makeTorus(1.0f, 2.0f, 16, 16);
            Assert.That(torus.points.size() == 256);
        }
    }
}
