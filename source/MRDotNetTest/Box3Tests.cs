using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{ 
    [TestFixture]
    internal class Box3Tests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var box = new Box3f();
            Assert.That(!box.Valid());
        }

        [Test]
        public void TestBox()
        {
            var box = new Box3f(new Vector3f(1, 2, 3), new Vector3f(4, 5, 6));
            Assert.That(box.Min.X == 1);
            Assert.That(box.Min.Y == 2);
            Assert.That(box.Min.Z == 3);

            Assert.That(box.Max.X == 4);
            Assert.That(box.Max.Y == 5);
            Assert.That(box.Max.Z == 6);

            var center = box.Center();
            Assert.That(center.X == 2.5);
            Assert.That(center.Y == 3.5);
            Assert.That(center.Z == 4.5);

            var size = box.Size();
            Assert.That(size.X == 3);
            Assert.That(size.Y == 3);
            Assert.That(size.Z == 3);

            float diagonal = box.Diagonal();
            Assert.That(diagonal, Is.EqualTo(5.19).Within(0.01));

            float volume = box.Volume();
            Assert.That(volume, Is.EqualTo(27).Within(0.01));
        }
    }
}
