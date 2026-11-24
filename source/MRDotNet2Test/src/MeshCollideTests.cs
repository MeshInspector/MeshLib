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
            Assert.That(intersections.size(), Is.EqualTo((nuint)152));
            var edgeATriBCount = 0;
            var edgeBTriACount = 0;
            for (nuint i = 0; i < intersections.size(); i++)
            {
                if ( intersections[i].isEdgeATriB() )
                    edgeATriBCount++;
                else
                    edgeBTriACount++;
            }
            // the second values are with FMA instructions (clang-cl on arm64), as in the C++ test of MRMeshBooleanTests.cpp
            Assert.That(edgeATriBCount, Is.EqualTo(80).Or.EqualTo(76));
            Assert.That(edgeBTriACount, Is.EqualTo(72).Or.EqualTo(76));

            var contours = orderIntersectionContours(meshA.topology, meshB.topology, intersections);

            Assert.That(contours.size(), Is.EqualTo((nuint)4));
            // the order of the contours follows the order of the intersections found in the trees
            var sizes = new nuint[4];
            for (nuint i = 0; i < 4; i++)
                sizes[(int)i] = contours[i].size();
            Array.Sort(sizes);
            Assert.That(sizes[0], Is.EqualTo((nuint)7).Or.EqualTo((nuint)9));
            Assert.That(sizes[1], Is.EqualTo((nuint)7).Or.EqualTo((nuint)9));
            Assert.That(sizes[2], Is.EqualTo((nuint)69).Or.EqualTo((nuint)71));
            Assert.That(sizes[3], Is.EqualTo((nuint)69).Or.EqualTo((nuint)71));
            var aConts = new OneMeshContours();
            getOneMeshIntersectionContours(meshA, meshB, contours, aConts, null, conv);
            Assert.That(aConts.size(), Is.EqualTo((nuint)4));
            var bConts = new OneMeshContours();
            getOneMeshIntersectionContours(meshA, meshB, contours, null, bConts, conv);
            Assert.That(bConts.size(), Is.EqualTo((nuint)4));

            // the intersections of all contours are in one array
            Assert.That(aConts.intersections.size(), Is.EqualTo((nuint)156) );

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
