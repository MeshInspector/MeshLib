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
            var meshA = MakeTorus(1.1f, 0.5f, 8, 8);
            var meshB = MakeTorus(1.1f, 0.5f, 8, 8);
            meshB.Transform(AffineXf3f.Linear(Matrix3f.Rotation(Vector3f.PlusZ(), new Vector3f(0.1f, 0.8f, 0.2f))));

            MeshPart mpA = new MeshPart( meshA );
            MeshPart mpB = new MeshPart( meshB );

            var conv = GetVectorConverters(mpA, mpB);

            var intersections = FindCollidingEdgeTrisPrecise(mpA, mpB, conv.toInt);
            Assert.That(intersections.Size(), Is.EqualTo(152));
            var edgeATriBCount = 0;
            var edgeBTriACount = 0;
            for (ulong i = 0; i < intersections.Size(); i++)
            {
                if ( intersections.At(i).IsEdgeATriB() )
                    edgeATriBCount++;
                else
                    edgeBTriACount++;
            }
            Assert.That(edgeATriBCount, Is.EqualTo(80));
            Assert.That(edgeBTriACount, Is.EqualTo(72));

            var contours = OrderIntersectionContours(meshA.topology, meshB.topology, intersections);

            Assert.That(contours.Size(), Is.EqualTo(4));
            Assert.That(contours.At(0).Size(), Is.EqualTo(71));
            Assert.That(contours.At(1).Size(), Is.EqualTo(7));
            Assert.That(contours.At(2).Size(), Is.EqualTo(69));
            Assert.That(contours.At(3).Size(), Is.EqualTo(9));
            var aConts = new Std.Vector_MROneMeshContour();
            GetOneMeshIntersectionContours(meshA, meshB, contours, aConts, null, conv);
            Assert.That(aConts.Size(), Is.EqualTo(4));
            var bConts = new Std.Vector_MROneMeshContour();
            GetOneMeshIntersectionContours(meshA, meshB, contours, null, bConts, conv);
            Assert.That(bConts.Size(), Is.EqualTo(4));

            ulong posCount = 0;
            for (ulong i = 0; i < aConts.Size(); i++)
            {
                posCount += aConts.At(i).intersections.Size();
            }

            Assert.That(posCount, Is.EqualTo(156) );

        }

        [Test]
        public void TestMeshCollide()
        {
            MeshPart meshA = new MeshPart(MakeTorus(1.1f, 0.5f, 8, 8));
            MeshPart meshB = new MeshPart(MakeTorus(1.1f, 0.2f, 8, 8));
            Assert.That(IsInside(meshB, meshA));
            Assert.That(!IsInside(meshA, meshB));
        }
    }
}
