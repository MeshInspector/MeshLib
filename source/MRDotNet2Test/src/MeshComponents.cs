using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    public class MeshComponentsTests
    {

        static MeshPart CreateMesh()
        {
            var bigCube = makeCube(Vector3f.diagonal(1), Vector3f.diagonal(-0.5f));
            var smallCube = makeCube(Vector3f.diagonal(0.1f), Vector3f.diagonal(1.0f));
            var boolResult = boolean(bigCube, smallCube, BooleanOperation.Union);
            return new MeshPart(boolResult.mesh);
        }

        [Test]
        public void TestComponentsMap()
        {
            var mp = CreateMesh();
            var res = MeshComponents.getAllComponentsMap(mp, MeshComponents.FaceIncidence.PerEdge);
            var map = res.first();
            var numComponents = res.second();

            Assert.That(numComponents == 2);
            Assert.That(map.size() == 24);
            Assert.That(map[new FaceId(0)].id == 0);
            Assert.That(map[new FaceId(12)].id == 1);
        }

        [Test]
        public void TestLargeRegions()
        {
            var mp = CreateMesh();
            var res1 = MeshComponents.getAllComponentsMap(mp, MeshComponents.FaceIncidence.PerEdge);
            var map = res1.first();
            var numComponents = res1.second();
            var res2 = MeshComponents.getLargeByAreaRegions(mp, map, numComponents, 0.1f);
            var region = res2.first();
            var numRegions = res2.second();

            Assert.That(numRegions == 1);
            Assert.That(region.test(new FaceId(0)));
            Assert.That(!region.test(new FaceId(12)));
        }

        [Test]
        public void TestLargestComponent()
        {
            var mp = CreateMesh();
            var numSmallerComponents = new Misc.InOut<int>(0);
            var components = MeshComponents.getLargestComponent(mp, MeshComponents.FaceIncidence.PerEdge, null, 0.1f, numSmallerComponents);
            Assert.That(numSmallerComponents.Value == 1);
            Assert.That(components.test(new FaceId(0)));
            Assert.That(!components.test(new FaceId(12)));
        }

        [Test]
        public void TestLargeComponents()
        {
            var mp = CreateMesh();
            var components = MeshComponents.getLargeByAreaComponents(mp, 0.1f, null);
            Assert.That(components.test(new FaceId(0)));
            Assert.That(!components.test(new FaceId(12)));
        }

        [Test]
        public void TestComponent()
        {
            var mp = CreateMesh();
            var component = MeshComponents.getComponent(mp, new FaceId(12), MeshComponents.FaceIncidence.PerEdge);
            Assert.That(!component.test(new FaceId(0)));
            Assert.That(component.test(new FaceId(12)));
        }
    }
}
