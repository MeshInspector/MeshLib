using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class PointsToMeshProjectorTest
    {
        [Test]
        public void TestPointsToMeshProjector()
        {
            var meshA = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(0));
            var meshB = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(0));
            
            var parameters = new MeshProjectionParameters();
            var shift = new Vector3f(1, 2, 3);

            parameters.xf = new AffineXf3f(shift);

            var res = FindSignedDistances(meshA, meshB, parameters);
            Assert.That(res.Count,Is.EqualTo(meshB.ValidPoints.Count()));
            Assert.That(res.Max(),Is.InRange(shift.Length()-1e-6, shift.Length() + 1e-6));
        }

    }
}
