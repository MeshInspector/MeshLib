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
            var meshA = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var meshB = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(0.0f));
            var union = Boolean(meshA, meshB, BooleanOperation.Union).Value; // TODO: replace _Moved
            var convexHull = MakeConvexHull(union.Mesh).Value; // TODO: replace _Moved

            Assert.That(convexHull.Points.Size() == 14);
        }
    }
}
