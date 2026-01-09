using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class ExpandShrinkTests
    {
        [Test]
        public void TestExpandShrink()
        {
            var mesh = MakeSphere(new SphereParams(1.0f, 3000));
            var region = Expand(mesh.Topology, new FaceId(0), 3);
            Assert.That(region.Count, Is.EqualTo(75));
            Expand(mesh.Topology, region, 3);
            Assert.That(region.Count, Is.GreaterThan(75));
            Shrink(mesh.Topology, region, 3);
            Assert.That(region.Count, Is.EqualTo(75));
        }

        [Test]
        public void TestExpandShrinkVerts()
        {
            var mesh = MakeSphere(new SphereParams(1.0f, 3000));
            var region = Expand(mesh.Topology, new VertId(0), 3);
            Assert.That(region.Count, Is.EqualTo(37));
            Expand(mesh.Topology, region, 3);
            Assert.That(region.Count, Is.GreaterThan(37));
            Shrink(mesh.Topology, region, 3);
            Assert.That(region.Count, Is.EqualTo(37));
        }

    }
}
