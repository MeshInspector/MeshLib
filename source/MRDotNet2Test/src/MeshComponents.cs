using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    public class MeshComponentsTests
    {

        static MeshPart CreateMesh()
        {
            var bigCube = MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f)).Value; // TODO: replace _Moved
            var smallCube = MakeCube(Vector3f.Diagonal(0.1f), Vector3f.Diagonal(1.0f)).Value; // TODO: replace _Moved
            var boolResult = Boolean(bigCube, smallCube, BooleanOperation.Union).Value; // TODO: replace _Moved
            return new MeshPart(boolResult.Mesh);
        }

        [Test]
        public void TestComponentsMap()
        {
            var mp = CreateMesh();
            var res = MeshComponents.GetAllComponentsMap(mp, MeshComponents.FaceIncidence.PerEdge).Value; // TODO: replace _Moved
            var map = res.First();
            var numComponents = res.Second();

            Assert.That(numComponents == 2);
            Assert.That(map.Size() == 24);
            Assert.That(map.Index(new FaceId(0)).Id == 0);
            Assert.That(map.Index(new FaceId(12)).Id == 1);
        }

        [Test]
        public void TestLargeRegions()
        {
            var mp = CreateMesh();
            var res1 = MeshComponents.GetAllComponentsMap(mp, MeshComponents.FaceIncidence.PerEdge).Value; // TODO: replace _Moved
            var map = res1.First();
            var numComponents = res1.Second();
            var res2 = MeshComponents.GetLargeByAreaRegions(mp, map, numComponents, 0.1f).Value; // TODO: replace _Moved
            var region = res2.First();
            var numRegions = res2.Second();

            Assert.That(numRegions == 1);
            Assert.That(region.Test(new FaceId(0)));
            Assert.That(!region.Test(new FaceId(12)));
        }

        [Test]
        public void TestLargestComponent()
        {
            var mp = CreateMesh();
            var numSmallerComponents = new Misc.InOut<int>(0);
            var components = MeshComponents.GetLargestComponent(mp, MeshComponents.FaceIncidence.PerEdge, null, 0.1f, numSmallerComponents).Value; // TODO: replace _Moved
            Assert.That(numSmallerComponents.Value == 1);
            Assert.That(components.Test(new FaceId(0)));
            Assert.That(!components.Test(new FaceId(12)));
        }

        [Test]
        public void TestLargeComponents()
        {
            var mp = CreateMesh();
            var components = MeshComponents.GetLargeByAreaComponents(mp, 0.1f, null).Value; // TODO: replace _Moved
            Assert.That(components.Test(new FaceId(0)));
            Assert.That(!components.Test(new FaceId(12)));
        }

        [Test]
        public void TestComponent()
        {
            var mp = CreateMesh();
            var component = MeshComponents.GetComponent(mp, new FaceId(12), MeshComponents.FaceIncidence.PerEdge).Value; // TODO: replace _Moved
            Assert.That(!component.Test(new FaceId(0)));
            Assert.That(component.Test(new FaceId(12)));
        }
    }
}
