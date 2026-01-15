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

            MeshPart mpA = new MeshPart( meshA );
            MeshPart mpB = new MeshPart( meshB );

            var conv = getVectorConverters(mpA, mpB);

            var intersections = findCollidingEdgeTrisPrecise(mpA, mpB, conv.toInt);
            Assert.That(intersections.size(), Is.EqualTo(152));
            var edgeATriBCount = 0;
            var edgeBTriACount = 0;
            for (ulong i = 0; i < intersections.size(); i++)
            {
                if ( intersections.at(i).isEdgeATriB() )
                    edgeATriBCount++;
                else
                    edgeBTriACount++;
            }
            Assert.That(edgeATriBCount, Is.EqualTo(80));
            Assert.That(edgeBTriACount, Is.EqualTo(72));

            var contours = orderIntersectionContours(meshA.topology, meshB.topology, intersections);

            Assert.That(contours.size(), Is.EqualTo(4));
            Assert.That(contours.at(0).size(), Is.EqualTo(71));
            Assert.That(contours.at(1).size(), Is.EqualTo(7));
            Assert.That(contours.at(2).size(), Is.EqualTo(69));
            Assert.That(contours.at(3).size(), Is.EqualTo(9));
            var aConts = new Std.Vector_MROneMeshContour();
            getOneMeshIntersectionContours(meshA, meshB, contours, aConts, null, conv);
            Assert.That(aConts.size(), Is.EqualTo(4));
            var bConts = new Std.Vector_MROneMeshContour();
            getOneMeshIntersectionContours(meshA, meshB, contours, null, bConts, conv);
            Assert.That(bConts.size(), Is.EqualTo(4));

            ulong posCount = 0;
            for (ulong i = 0; i < aConts.size(); i++)
            {
                posCount += aConts.at(i).intersections.size();
            }

            Assert.That(posCount, Is.EqualTo(156) );

        }

        [Test]
        public void TestMeshCollide()
        {
            MeshPart meshA = new MeshPart(makeTorus(1.1f, 0.5f, 8, 8));
            MeshPart meshB = new MeshPart(makeTorus(1.1f, 0.2f, 8, 8));
            Assert.That(isInside(meshB, meshA));
            Assert.That(!isInside(meshA, meshB));
        }
    }
}
