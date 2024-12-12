using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class ExpandShrinkTests
    {
        [Test]
        public void TestExpandShrink()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var region = Expand(mesh, new FaceId(0), 3);
            Assert.That(region.Count, Is.EqualTo(75));
            Expand(mesh, region, 3);
            Assert.That(region.Count, Is.GreaterThan(75));
            Shrink(mesh, region, 3);
            Assert.That(region.Count, Is.EqualTo(75));
        }

        [Test]
        public void TestExpandShrinkVerts()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var region = Expand(mesh, new VertId(0), 3);
            Assert.That(region.Count, Is.EqualTo(37));
            Expand(mesh, region, 3);
            Assert.That(region.Count, Is.GreaterThan(37));
            Shrink(mesh, region, 3);
            Assert.That(region.Count, Is.EqualTo(37));
        }

    }
}
