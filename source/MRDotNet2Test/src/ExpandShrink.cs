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
            var mesh = makeSphere(new SphereParams(1.0f, 3000));
            var region = expand(mesh.topology, new FaceId(0), 3);
            Assert.That(region.count, Is.EqualTo(75));
            expand(mesh.topology, region, 3);
            Assert.That(region.count, Is.GreaterThan(75));
            shrink(mesh.topology, region, 3);
            Assert.That(region.count, Is.EqualTo(75));
        }

        [Test]
        public void TestExpandShrinkVerts()
        {
            var mesh = makeSphere(new SphereParams(1.0f, 3000));
            var region = expand(mesh.topology, new VertId(0), 3);
            Assert.That(region.count, Is.EqualTo(37));
            expand(mesh.topology, region, 3);
            Assert.That(region.count, Is.GreaterThan(37));
            shrink(mesh.topology, region, 3);
            Assert.That(region.count, Is.EqualTo(37));
        }

    }
}
