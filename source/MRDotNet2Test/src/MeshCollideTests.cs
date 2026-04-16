using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class MeshCollideTests
    {
        [Test]
        public void TestMeshCollidePrecise()
        {
            var meshA = makeTorus(1.1f, 0.5f, 8, 8);
            var meshB = makeTorus(1.1f, 0.5f, 8, 8);
            meshB.transform(AffineXf3f.linear(Matrix3f.rotation(Vector3f.plusZ(), new Vector3f(0.1f, 0.8f, 0.2f))));

            var conv = getVectorConverters(meshA, meshB);

            var intersections = findCollidingEdgeTrisPrecise(meshA, meshB, conv.toInt);
            Assert.That(intersections.size(), Is.EqualTo(152));
            var edgeATriBCount = 0;
            var edgeBTriACount = 0;
            for (ulong i = 0; i < intersections.size(); i++)
            {
                if ( intersections[i].isEdgeATriB() )
                    edgeATriBCount++;
                else
                    edgeBTriACount++;
            }
            Assert.That(edgeATriBCount, Is.EqualTo(80));
            Assert.That(edgeBTriACount, Is.EqualTo(72));

            var contours = orderIntersectionContours(meshA.topology, meshB.topology, intersections);

            Assert.That(contours.size(), Is.EqualTo(4));
            Assert.That(contours[0].size(), Is.EqualTo(71));
            Assert.That(contours[1].size(), Is.EqualTo(7));
            Assert.That(contours[2].size(), Is.EqualTo(69));
            Assert.That(contours[3].size(), Is.EqualTo(9));
            var aConts = new Std.Vector_MROneMeshContour();
            getOneMeshIntersectionContours(meshA, meshB, contours, aConts, null, conv);
            Assert.That(aConts.size(), Is.EqualTo(4));
            var bConts = new Std.Vector_MROneMeshContour();
            getOneMeshIntersectionContours(meshA, meshB, contours, null, bConts, conv);
            Assert.That(bConts.size(), Is.EqualTo(4));

            ulong posCount = 0;
            for (ulong i = 0; i < aConts.size(); i++)
            {
                posCount += aConts[i].intersections.size();
            }

            Assert.That(posCount, Is.EqualTo(156) );

        }

        [Test]
        public void TestMeshCollide()
        {
            Mesh meshA = makeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = makeTorus(1.1f, 0.2f, 8, 8);
            Assert.That(isInside(meshB, meshA));
            Assert.That(!isInside(meshA, meshB));
        }
    }
}
