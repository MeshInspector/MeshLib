using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class SelfIntersectionsTests
    {
        [Test]
        public void TestSelfIntersections() 
        {
            var mesh = Mesh.MakeTorusWithSelfIntersections(1.0f, 0.2f, 32, 16);
            Assert.That( mesh.ValidFaces.Count(), Is.EqualTo(1024) );

            var intersections = SelfIntersections.GetFaces(mesh);
            Assert.That( intersections.Count, Is.EqualTo(128) );

            var settings = new SelfIntersections.Settings();
            settings.method = SelfIntersections.Method.CutAndFill;

            Assert.DoesNotThrow(() => SelfIntersections.Fix(ref mesh, settings) );
            Assert.That(mesh.ValidFaces.Count(), Is.EqualTo(1194));
            
            intersections = SelfIntersections.GetFaces(mesh);
            Assert.That(intersections.Count, Is.EqualTo(0));
        }
    }
}
