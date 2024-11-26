using System;
using System.IO;
using NUnit.Framework;

using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class ConvexHullTests
    {
        [Test]
        public void TestConvexHull()
        {
            var meshA = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var meshB = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(0.0f));
            var union = Boolean(meshA, meshB, BooleanOperation.Union);
            var convexHull = MakeConvexHull(union.mesh);

            Assert.That(convexHull.Points.Count == 14);
        }
    }
}
