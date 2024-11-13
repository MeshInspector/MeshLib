using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class SelfIntersectionsTests
    {
        [Test]
        public void TestSelfIntersections() 
        {
            var mesh = Mesh.MakeTorusWithSelfIntersections(1.0f, 0.2f, 32, 16);
            Assert.That( mesh.ValidFaces.Count(), Is.EqualTo(1024) );

            var intersections = FixSelfIntersections.GetFaces(mesh);
            Assert.That( intersections.Count, Is.EqualTo(128) );

            FixSelfIntersections.Settings settings = new FixSelfIntersections.Settings();
            settings.method = FixSelfIntersections.Method.CutAndFill;

            Assert.DoesNotThrow(() => FixSelfIntersections.Fix(ref mesh, settings) );
            Assert.That(mesh.ValidFaces.Count(), Is.EqualTo(1194));
            
            intersections = FixSelfIntersections.GetFaces(mesh);
            Assert.That(intersections.Count, Is.EqualTo(0));
        }
    }
}
