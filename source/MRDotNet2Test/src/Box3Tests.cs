using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class Box3Tests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var box = new Box3f();
            Assert.That(!box.valid());
        }

        [Test]
        public void TestBox()
        {
            var box = new Box3f(new Vector3f(1, 2, 3), new Vector3f(4, 5, 6));
            Assert.That(box.min.x == 1);
            Assert.That(box.min.y == 2);
            Assert.That(box.min.z == 3);

            Assert.That(box.max.x == 4);
            Assert.That(box.max.y == 5);
            Assert.That(box.max.z == 6);

            var center = box.center();
            Assert.That(center.x == 2.5);
            Assert.That(center.y == 3.5);
            Assert.That(center.z == 4.5);

            var size = box.size();
            Assert.That(size.x == 3);
            Assert.That(size.y == 3);
            Assert.That(size.z == 3);

            float diagonal = box.diagonal();
            Assert.That(diagonal, Is.EqualTo(5.19).Within(0.01));

            float volume = box.volume();
            Assert.That(volume, Is.EqualTo(27).Within(0.01));
        }
    }
}
