using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class Matrix3Tests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new MR.DotNet.Matrix3f();

            Assert.That(1 == a.x.x);
            Assert.That(0 == a.x.y);
            Assert.That(0 == a.x.z);
            Assert.That(0 == a.y.x);
            Assert.That(1 == a.y.y);
            Assert.That(0 == a.y.z);
            Assert.That(0 == a.z.x);
            Assert.That(0 == a.z.y);
            Assert.That(1 == a.z.z);
        }

        [Test]
        public void TestConstructor()
        {
            var a = new MR.DotNet.Matrix3f( new Vector3f(1, 2, 3), new Vector3f(4, 5, 6), new Vector3f(7, 8, 9) );

            Assert.That(1 == a.x.x);
            Assert.That(2 == a.x.y);
            Assert.That(3 == a.x.z);
            Assert.That(4 == a.y.x);
            Assert.That(5 == a.y.y);
            Assert.That(6 == a.y.z);
            Assert.That(7 == a.z.x);
            Assert.That(8 == a.z.y);
            Assert.That(9 == a.z.z);
        }

        [Test]
        public void TestZeroMatrix()
        {
            var a = Matrix3f.zero();

            Assert.That(0 == a.x.x);
            Assert.That(0 == a.x.y);
            Assert.That(0 == a.x.z);
            Assert.That(0 == a.y.x);
            Assert.That(0 == a.y.y);
            Assert.That(0 == a.y.z);
            Assert.That(0 == a.z.x);
            Assert.That(0 == a.z.y);
            Assert.That(0 == a.z.z);
        }

        [Test]
        public void TestRotation()
        {
            Vector3f axis = new Vector3f( 1, 1, 1 );
            float angle = 1.0f;
            var a = Matrix3f.rotation(axis, angle);
            Assert.That(1 == a.x.x);
            Assert.That(0 == a.x.y);
            Assert.That(0 == a.x.z);
            Assert.That(0 == a.y.x);
            Assert.That(1 == a.y.y);
            Assert.That(0 == a.y.z);
            Assert.That(0 == a.z.x);
            Assert.That(0 == a.z.y);
            Assert.That(1 == a.z.z);
        }
    }
}
