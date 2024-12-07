using NUnit.Framework;
using static MR.DotNet;
using static MR.DotNet.MeshComponents;

namespace MR.Test
{
    [TestFixture]
    public class MeshComponentsTests
    {

        static MeshPart CreateMesh()
        {
            var bigCube = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var smallCube = Mesh.MakeCube(Vector3f.Diagonal(0.1f), Vector3f.Diagonal(1.0f));
            var boolResult = Boolean(bigCube, smallCube, BooleanOperation.Union);
            return new MeshPart(boolResult.mesh);
        }

        [Test]
        public void TestComponentsMap()
        {
            var mp = CreateMesh();
            var map = MeshComponents.GetAllComponentsMap(mp, FaceIncidence.PerEdge);

            Assert.That(map.NumComponents == 2);
            Assert.That(map.Count == 24);
            Assert.That(map[0].Id == 0);
            Assert.That(map[12].Id == 1);
        }

        [Test]
        public void TestLargeRegions()
        {
            var mp = CreateMesh();
            var map = MeshComponents.GetAllComponentsMap(mp, FaceIncidence.PerEdge);
            var res = MeshComponents.GetLargeByAreaRegions(mp, map, map.NumComponents, 0.1f);

            Assert.That(res.numRegions == 1);
            Assert.That(res.faces.Test(0));
            Assert.That(!res.faces.Test(12));
        }

        [Test]
        public void TestLargestComponent()
        {
            var mp = CreateMesh();
            int numSmallerComponents = 0;
            var components = MeshComponents.GetLargestComponent(mp, FaceIncidence.PerEdge, 0.1f, out numSmallerComponents);
            Assert.That(numSmallerComponents == 1);
            Assert.That(components.Test(0));
            Assert.That(!components.Test(12));
        }

        [Test]
        public void TestLargeComponents()
        {
            var mp = CreateMesh();
            var components = MeshComponents.GetLargeByAreaComponents(mp, 0.1f);
            Assert.That(components.Test(0));
            Assert.That(!components.Test(12));
        }

        [Test]
        public void TestComponent()
        {
            var mp = CreateMesh();
            var component = MeshComponents.GetComponent(mp, new FaceId(12), FaceIncidence.PerEdge);
            Assert.That(!component.Test(0));
            Assert.That(component.Test(12));
        }
    }
}
