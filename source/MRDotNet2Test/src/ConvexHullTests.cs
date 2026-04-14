using System;
using System.IO;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class ConvexHullTests
    {
        [Test]
        public void TestConvexHull()
        {
            var meshA = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var meshB = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(0.0f));
            var union = boolean(meshA, meshB, BooleanOperation.Union);
            var convexHull = makeConvexHull(union.mesh);

            Assert.That(convexHull.points.size() == 14);
        }
    }
}
