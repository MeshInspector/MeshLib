using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class SelfIntersectionsTests
    {
        [Test]
        public void TestSelfIntersections()
        {
            var mesh = makeTorusWithSelfIntersections(1.0f, 0.2f, 32, 16);
            Assert.That( mesh.topology.getValidFaces().count(), Is.EqualTo(1024) );

            var intersections = SelfIntersections.getFaces(mesh);
            Assert.That( intersections.count, Is.EqualTo(128) );

            var settings = new SelfIntersections.Settings();
            settings.method = SelfIntersections.Settings.Method.CutAndFill;

            Assert.DoesNotThrow(() => SelfIntersections.fix(mesh, settings) );
            Assert.That(mesh.topology.getValidFaces().count(), Is.EqualTo(1194));

            intersections = SelfIntersections.getFaces(mesh);
            Assert.That(intersections.count, Is.EqualTo(0));
        }
    }
}
