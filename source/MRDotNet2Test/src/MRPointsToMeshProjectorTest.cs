using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class PointsToMeshProjectorTest
    {
        [Test]
        public void TestPointsToMeshProjector()
        {
            var meshA = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(0));
            var meshB = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(0));

            var parameters = new MeshProjectionParameters();
            var shift = new Vector3f(1, 2, 3);

            parameters.xf = AffineXf3f.Translation(shift);

            var res = FindSignedDistances(meshA, meshB, parameters);
            Assert.That(res.Size(), Is.EqualTo(meshB.topology.GetValidVerts().Count()));

            // TODO: iteration
            float resMax = 0f;
            for (ulong i = 0; i < res.Size(); i++)
            {
                resMax = Math.Max(resMax, res.Index(new VertId(i)));
            }
            Assert.That(resMax, Is.InRange(shift.Length()-1e-6, shift.Length() + 1e-6));
        }
    }
}
