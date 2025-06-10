using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
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

            var intersections = FindCollidingEdgeTrisPrecise(meshA, meshB, conv);
            Assert.That(intersections.List.Count, Is.EqualTo(152));
            var edgeATriBCount = 0;
            var edgeBTriACount = 0;
            foreach ( var ver in intersections.List )
            {
                if ( ver.isEdgeATriB )
                    edgeATriBCount++;
                else
                    edgeBTriACount++;
            }
            Assert.That(edgeATriBCount, Is.EqualTo(80));
            Assert.That(edgeBTriACount, Is.EqualTo(72));

            var orderedIntersections = IntersectionContour.OrderIntersectionContours(meshA.mesh, meshB.mesh, intersections);
            var contours = orderedIntersections.Contours;

            Assert.That(contours.Count, Is.EqualTo(4));
            Assert.That(contours[0].Count, Is.EqualTo(71));
            Assert.That(contours[1].Count, Is.EqualTo(7));
            Assert.That(contours[2].Count, Is.EqualTo(69));
            Assert.That(contours[3].Count, Is.EqualTo(9));
            var aConts = GetOneMeshIntersectionContours(meshA.mesh, meshB.mesh, orderedIntersections, true, conv);
            Assert.That(aConts.Count, Is.EqualTo(4));
            var bConts = GetOneMeshIntersectionContours(meshA.mesh, meshB.mesh, orderedIntersections, false, conv);
            Assert.That(bConts.Count, Is.EqualTo(4));

            int posCount = 0;
            for (int i = 0; i < aConts.Count; i++)
            {
                posCount += aConts[i].intersections.Count;
            }

            Assert.That(posCount, Is.EqualTo(156) );

        }

        [Test]
        public void TestMeshCollide()
        {
            MeshPart meshA = new MeshPart(Mesh.MakeTorus(1.1f, 0.5f, 8, 8));
            MeshPart meshB = new MeshPart(Mesh.MakeTorus(1.1f, 0.2f, 8, 8));
            Assert.That(IsInside(meshB, meshA));
            Assert.That(!IsInside(meshA, meshB));
        }
    }
}
