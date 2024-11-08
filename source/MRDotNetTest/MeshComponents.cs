using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    public class MeshComponentsTests
    {

        [Test]
        public void TestComponentsMap()
        {
            var cube = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var cubeCopy = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(1.5f));
            var boolResult = MeshBoolean.Boolean(cube, cubeCopy, BooleanOperation.Union);
            
            
            var map = MeshComponents.GetAllComponentsMap(new MeshPart( boolResult.mesh ), FaceIncidence.PerEdge);

            Assert.That(map.NumComponents == 2);
            Assert.That(map.Count == 24);
            Assert.That(map[0].Id == 0);
            Assert.That(map[12].Id == 1);
        }

        [Test]
        public void TestLargeRegions()
        {
            var bigCube = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var smallCube = Mesh.MakeCube(Vector3f.Diagonal(0.1f), Vector3f.Diagonal(1.0f));
            var boolResult = MeshBoolean.Boolean(bigCube, smallCube, BooleanOperation.Union);

            var mp = new MeshPart(boolResult.mesh);
            var map = MeshComponents.GetAllComponentsMap(mp, FaceIncidence.PerEdge);
            var res = MeshComponents.GetLargeByAreaRegions(mp, map, map.NumComponents, 0.1f);

            Assert.That(res.numRegions == 1);
            Assert.That(res.faces.Test(0));
            Assert.That(!res.faces.Test(12));
        }
    }
}
