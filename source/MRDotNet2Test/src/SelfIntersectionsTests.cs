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
            var mesh = MakeTorusWithSelfIntersections(1.0f, 0.2f, 32, 16);
            Assert.That( mesh.topology.GetValidFaces().Count(), Is.EqualTo(1024) );

            var intersections = SelfIntersections.GetFaces(mesh);
            Assert.That( intersections.Count, Is.EqualTo(128) );

            var settings = new SelfIntersections.Settings();
            settings.method = SelfIntersections.Settings.Method.CutAndFill;

            Assert.DoesNotThrow(() => SelfIntersections.Fix(mesh, settings) );
            Assert.That(mesh.topology.GetValidFaces().Count(), Is.EqualTo(1194));

            intersections = SelfIntersections.GetFaces(mesh);
            Assert.That(intersections.Count, Is.EqualTo(0));
        }
    }
}
