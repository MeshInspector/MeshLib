using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class Matrix3Tests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new Matrix3f();

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
            var a = new Matrix3f( new Vector3f(1, 2, 3), new Vector3f(4, 5, 6), new Vector3f(7, 8, 9) );

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
            var a = Matrix3f.Zero();

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
        public void TestRotationAroundAxis()
        {
            var a = Matrix3f.Rotation( new Vector3f( 1, 1, 1 ), 1.0f ) ;
            Assert.That( 0.69, Is.EqualTo( a.x.x ).Within( 0.01 ) );
            Assert.That( -0.33, Is.EqualTo( a.x.y ).Within( 0.01 ) );
            Assert.That( 0.64, Is.EqualTo( a.x.z ).Within( 0.01 ) );
            Assert.That( 0.64, Is.EqualTo( a.y.x ).Within( 0.01 ) );
            Assert.That( 0.69, Is.EqualTo( a.y.y ).Within( 0.01 ) );
            Assert.That( -0.33, Is.EqualTo( a.y.z ).Within( 0.01 ) );
            Assert.That( -0.33, Is.EqualTo( a.z.x ).Within( 0.01 ) );
            Assert.That( 0.64, Is.EqualTo( a.z.y ).Within( 0.01 ) );
            Assert.That( 0.69, Is.EqualTo( a.z.z ).Within( 0.01 ) );
        }

        [Test]
        public void TestRotationBetweenTwoVectors()
        {
            var a = Matrix3f.Rotation( new Vector3f( 1, 0, 1), new Vector3f( 0, 1, 1 ) );
            Assert.That(0.67, Is.EqualTo(a.x.x).Within(0.01));
            Assert.That(-0.33, Is.EqualTo(a.x.y).Within(0.01));
            Assert.That(-0.67, Is.EqualTo(a.x.z).Within(0.01));
            Assert.That(0.67, Is.EqualTo(a.y.x).Within(0.01));
            Assert.That(0.67, Is.EqualTo(a.y.y).Within(0.01));
            Assert.That(0.33, Is.EqualTo(a.y.z).Within(0.01));
            Assert.That(0.33, Is.EqualTo(a.z.x).Within(0.01));
            Assert.That(-0.67, Is.EqualTo(a.z.y).Within(0.01));
            Assert.That(0.67, Is.EqualTo(a.z.z).Within(0.01));
        }

        [Test]
        public void TestMultiplication()
        {
            var a = new Matrix3f( new Vector3f( 1, 2, 3 ), new Vector3f( 4, 5, 6 ), new Vector3f( 7, 8, 9 ) );
            var b = new Matrix3f( new Vector3f( 10, 11, 12 ), new Vector3f( 13, 14, 15 ), new Vector3f( 16, 17, 18 ) );

            var c = a * b;

            Assert.That( 84, Is.EqualTo( c.x.x ).Within( 0.01 ) );
            Assert.That( 90, Is.EqualTo( c.x.y ).Within( 0.01 ) );
            Assert.That( 96, Is.EqualTo( c.x.z ).Within( 0.01 ) );
            Assert.That( 201, Is.EqualTo( c.y.x ).Within( 0.01 ) );
            Assert.That( 216, Is.EqualTo( c.y.y ).Within( 0.01 ) );
            Assert.That( 231, Is.EqualTo( c.y.z ).Within( 0.01 ) );
            Assert.That( 318, Is.EqualTo( c.z.x ).Within( 0.01 ) );
            Assert.That( 342, Is.EqualTo( c.z.y ).Within( 0.01 ) );
            Assert.That( 366, Is.EqualTo( c.z.z ).Within( 0.01 ) );
        }

        [Test]
        public void TestMultiplicationWithVector()
        {
            var a = new Matrix3f(new Vector3f(1, 2, 3), new Vector3f(4, 5, 6), new Vector3f(7, 8, 9));
            var b = new Vector3f(10, 11, 12);

            var c = a * b;

            Assert.That( 68, Is.EqualTo( c.x ).Within( 0.01 ) );
            Assert.That( 167, Is.EqualTo( c.y ).Within( 0.01 ) );
            Assert.That( 266, Is.EqualTo( c.z ).Within( 0.01 ) );
        }
    }
}
