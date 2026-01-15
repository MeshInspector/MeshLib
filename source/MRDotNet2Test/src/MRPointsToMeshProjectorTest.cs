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
            var meshA = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(0));
            var meshB = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(0));

            var parameters = new MeshProjectionParameters();
            var shift = new Vector3f(1, 2, 3);

            parameters.xf = AffineXf3f.translation(shift);

            var res = findSignedDistances(meshA, meshB, parameters);
            Assert.That(res.size(), Is.EqualTo(meshB.topology.getValidVerts().count()));

            // TODO: iteration
            float resMax = 0f;
            for (ulong i = 0; i < res.size(); i++)
            {
                resMax = Math.Max(resMax, res[new VertId(i)]);
            }
            Assert.That(resMax, Is.InRange(shift.length()-1e-6, shift.length() + 1e-6));
        }
    }
}
