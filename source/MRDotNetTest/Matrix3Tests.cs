using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class Matrix3Tests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new Matrix3f();

            Assert.That(1 == a.X.X);
            Assert.That(0 == a.X.Y);
            Assert.That(0 == a.X.Z);
            Assert.That(0 == a.Y.X);
            Assert.That(1 == a.Y.Y);
            Assert.That(0 == a.Y.Z);
            Assert.That(0 == a.Z.X);
            Assert.That(0 == a.Z.Y);
            Assert.That(1 == a.Z.Z);
        }

        [Test]
        public void TestConstructor()
        {
            var a = new Matrix3f( new Vector3f(1, 2, 3), new Vector3f(4, 5, 6), new Vector3f(7, 8, 9) );

            Assert.That(1 == a.X.X);
            Assert.That(2 == a.X.Y);
            Assert.That(3 == a.X.Z);
            Assert.That(4 == a.Y.X);
            Assert.That(5 == a.Y.Y);
            Assert.That(6 == a.Y.Z);
            Assert.That(7 == a.Z.X);
            Assert.That(8 == a.Z.Y);
            Assert.That(9 == a.Z.Z);
        }

        [Test]
        public void TestZeroMatrix()
        {
            var a = Matrix3f.Zero();

            Assert.That(0 == a.X.X);
            Assert.That(0 == a.X.Y);
            Assert.That(0 == a.X.Z);
            Assert.That(0 == a.Y.X);
            Assert.That(0 == a.Y.Y);
            Assert.That(0 == a.Y.Z);
            Assert.That(0 == a.Z.X);
            Assert.That(0 == a.Z.Y);
            Assert.That(0 == a.Z.Z);
        }

        [Test]
        public void TestRotationAroundAxis()
        {
            var a = Matrix3f.Rotation( new Vector3f( 1, 1, 1 ), 1.0f ) ;
            Assert.That( 0.69, Is.EqualTo( a.X.X ).Within( 0.01 ) );
            Assert.That( -0.33, Is.EqualTo( a.X.Y ).Within( 0.01 ) );
            Assert.That( 0.64, Is.EqualTo( a.X.Z ).Within( 0.01 ) );
            Assert.That( 0.64, Is.EqualTo( a.Y.X ).Within( 0.01 ) );
            Assert.That( 0.69, Is.EqualTo( a.Y.Y ).Within( 0.01 ) );
            Assert.That( -0.33, Is.EqualTo( a.Y.Z ).Within( 0.01 ) );
            Assert.That( -0.33, Is.EqualTo( a.Z.X ).Within( 0.01 ) );
            Assert.That( 0.64, Is.EqualTo( a.Z.Y ).Within( 0.01 ) );
            Assert.That( 0.69, Is.EqualTo( a.Z.Z ).Within( 0.01 ) );
        }

        [Test]
        public void TestRotationBetweenTwoVectors()
        {
            var a = Matrix3f.Rotation( new Vector3f( 1, 0, 1), new Vector3f( 0, 1, 1 ) );
            Assert.That(0.67, Is.EqualTo(a.X.X).Within(0.01));
            Assert.That(-0.33, Is.EqualTo(a.X.Y).Within(0.01));
            Assert.That(-0.67, Is.EqualTo(a.X.Z).Within(0.01));
            Assert.That(0.67, Is.EqualTo(a.Y.X).Within(0.01));
            Assert.That(0.67, Is.EqualTo(a.Y.Y).Within(0.01));
            Assert.That(0.33, Is.EqualTo(a.Y.Z).Within(0.01));
            Assert.That(0.33, Is.EqualTo(a.Z.X).Within(0.01));
            Assert.That(-0.67, Is.EqualTo(a.Z.Y).Within(0.01));
            Assert.That(0.67, Is.EqualTo(a.Z.Z).Within(0.01));
        }

        [Test]
        public void TestMultiplication()
        {
            var a = new Matrix3f( new Vector3f( 1, 2, 3 ), new Vector3f( 4, 5, 6 ), new Vector3f( 7, 8, 9 ) );
            var b = new Matrix3f( new Vector3f( 10, 11, 12 ), new Vector3f( 13, 14, 15 ), new Vector3f( 16, 17, 18 ) );

            var c = a * b;

            Assert.That( 84, Is.EqualTo( c.X.X ).Within( 0.01 ) );
            Assert.That( 90, Is.EqualTo( c.X.Y ).Within( 0.01 ) );
            Assert.That( 96, Is.EqualTo( c.X.Z ).Within( 0.01 ) );
            Assert.That( 201, Is.EqualTo( c.Y.X ).Within( 0.01 ) );
            Assert.That( 216, Is.EqualTo( c.Y.Y ).Within( 0.01 ) );
            Assert.That( 231, Is.EqualTo( c.Y.Z ).Within( 0.01 ) );
            Assert.That( 318, Is.EqualTo( c.Z.X ).Within( 0.01 ) );
            Assert.That( 342, Is.EqualTo( c.Z.Y ).Within( 0.01 ) );
            Assert.That( 366, Is.EqualTo( c.Z.Z ).Within( 0.01 ) );
        }

        [Test]
        public void TestMultiplicationWithVector()
        {
            var a = new Matrix3f(new Vector3f(1, 2, 3), new Vector3f(4, 5, 6), new Vector3f(7, 8, 9));
            var b = new Vector3f(10, 11, 12);

            var c = a * b;

            Assert.That( 68, Is.EqualTo( c.X ).Within( 0.01 ) );
            Assert.That( 167, Is.EqualTo( c.Y ).Within( 0.01 ) );
            Assert.That( 266, Is.EqualTo( c.Z ).Within( 0.01 ) );
        }        
    }
}
