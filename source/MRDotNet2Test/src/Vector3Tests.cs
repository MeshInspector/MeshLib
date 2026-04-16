using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class Vector3Tests
    {
        [Test]
        public void TestConstructor()
        {
            var a = new Vector3f(1, 2, 3);
            Assert.That(1 == a.x);
            Assert.That(2 == a.y);
            Assert.That(3 == a.z);
        }

        [Test]
        public void TestDefaultConstructor()
        {
            var a = new Vector3f();
            Assert.That(0 == a.x);
            Assert.That(0 == a.y);
            Assert.That(0 == a.z);
        }

        [Test]
        public void TestAddition()
        {
            var a = new Vector3f(1, 2, 3);
            var b = new Vector3f(4, 5, 6);
            var c = a + b;

            Assert.That(5 == c.x);
            Assert.That(7 == c.y);
            Assert.That(9 == c.z);
        }

        [Test]
        public void TestSubtraction()
        {
            var a = new Vector3f(1, 2, 3);
            var b = new Vector3f(6, 5, 4);
            var c = b - a;

            Assert.That(5 == c.x);
            Assert.That(3 == c.y);
            Assert.That(1 == c.z);
        }

        [Test]
        public void TestMultiplication()
        {
            var a = new Vector3f(1, 2, 3);
            float k = 2;
            var c = a * k;

            Assert.That(2 == c.x);
            Assert.That(4 == c.y);
            Assert.That(6 == c.z);

            c = k * a;
            Assert.That(2 == c.x);
            Assert.That(4 == c.y);
            Assert.That(6 == c.z);
        }
    }
}
