﻿using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class MeshCollideTests
    {
        [Test]
        public void TestMeshCollidePrecise()
        {
            MeshPart meshA = new MeshPart( Mesh.MakeTorus(1.1f, 0.5f, 8, 8) );
            MeshPart meshB = new MeshPart(Mesh.MakeTorus(1.1f, 0.5f, 8, 8));

            meshB.mesh.Transform(new AffineXf3f(Matrix3f.Rotation(Vector3f.PlusZ(), new Vector3f( 0.1f, 0.8f, 0.2f))));
            var conv = new CoordinateConverters(meshA, meshB);

            var intersections = MeshCollidePrecise.FindCollidingEdgeTrisPrecise(meshA, meshB, conv);
            Assert.That(intersections.EdgesAtrisB.Count, Is.EqualTo(80));
            Assert.That(intersections.EdgesBtrisA.Count, Is.EqualTo(72));
            var orderedIntersections = IntersectionContour.OrderIntersectionContours(meshA.mesh, meshB.mesh, intersections);
            Assert.That(orderedIntersections.Count, Is.EqualTo(4));
            Assert.That(orderedIntersections[0].Count, Is.EqualTo(69));
            Assert.That(orderedIntersections[1].Count, Is.EqualTo(71));
            Assert.That(orderedIntersections[2].Count, Is.EqualTo(7));
            Assert.That(orderedIntersections[3].Count, Is.EqualTo(9));
            var aConts = ContoursCut.GetOneMeshIntersectionContours(meshA.mesh, meshB.mesh, orderedIntersections, true, conv);
            Assert.That(aConts.Count, Is.EqualTo(4));
            var bConts = ContoursCut.GetOneMeshIntersectionContours(meshA.mesh, meshB.mesh, orderedIntersections, false, conv);
            Assert.That(bConts.Count, Is.EqualTo(4));

            int posCount = 0;
            for (int i = 0; i < aConts.Count; i++)
            {
                posCount += aConts[i].intersections.Count;
            }

            Assert.That(posCount, Is.EqualTo(156) );

        }
    }
}
