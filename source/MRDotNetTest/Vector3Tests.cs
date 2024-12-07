using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class Vector3Tests
    {
        [Test]
        public void TestConstructor()
        {
            var a = new Vector3f(1, 2, 3);
            Assert.That(1 == a.X);
            Assert.That(2 == a.Y);
            Assert.That(3 == a.Z);
        }

        [Test]
        public void TestDefaultConstructor()
        {
            var a = new Vector3f();
            Assert.That(0 == a.X);
            Assert.That(0 == a.Y);
            Assert.That(0 == a.Z);
        }

        [Test]
        public void TestAddition()
        {
            var a = new Vector3f(1, 2, 3);
            var b = new Vector3f(4, 5, 6);
            var c = a + b;

            Assert.That(5 == c.X);
            Assert.That(7 == c.Y);
            Assert.That(9 == c.Z);
        }

        [Test]
        public void TestSubtraction()
        {
            var a = new Vector3f(1, 2, 3);
            var b = new Vector3f(6, 5, 4);
            var c = b - a;

            Assert.That(5 == c.X);
            Assert.That(3 == c.Y);
            Assert.That(1 == c.Z);            
        }

        [Test]
        public void TestMultiplication()
        {
            var a = new Vector3f(1, 2, 3);
            float k = 2;
            var c = a * k;

            Assert.That(2 == c.X);
            Assert.That(4 == c.Y);
            Assert.That(6 == c.Z);

            c = k * a;
            Assert.That(2 == c.X);
            Assert.That(4 == c.Y);
            Assert.That(6 == c.Z);
        }        
    }
}
