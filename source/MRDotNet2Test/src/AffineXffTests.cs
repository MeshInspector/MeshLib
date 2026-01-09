using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class AffineXfTests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new AffineXf3f();
            Assert.That( a.A == new Matrix3f() );
            Assert.That( a.B == new Vector3f() );
        }

        [Test]
        public void TestConstructor()
        {
            var A = new Matrix3f( new Vector3f( 1, 2, 3 ), new Vector3f( 4, 5, 6 ), new Vector3f( 7, 8, 9 ) );
            var b = new Vector3f( 10, 11, 12 );

            var a = new AffineXf3f( A, b );
            Assert.That( a.A == A );
            Assert.That( a.B == b );
        }

        [Test]
        public void TestLinearConstructor()
        {
            var A = new Matrix3f( new Vector3f( 1, 2, 3 ), new Vector3f( 4, 5, 6 ), new Vector3f( 7, 8, 9 ) );
            var a = AffineXf3f.Linear( A );
            Assert.That( a.A == A );
            Assert.That( a.B == new Vector3f() );
        }

        [Test]
        public void TestTranslationConstructor()
        {
            var b = new Vector3f( 10, 11, 12 );
            var a = AffineXf3f.Translation( b );
            Assert.That( a.A == new Matrix3f() );
            Assert.That( a.B == b );
        }

        [Test]
        public void TestMultiplication()
        {
            var A = new Matrix3f( new Vector3f( 1, 2, 3 ), new Vector3f( 4, 5, 6 ), new Vector3f( 7, 8, 9 ) );
            var b = new Vector3f( 10, 11, 12 );
            var xf = new AffineXf3f( A, b );

            var res = xf * xf;
            Assert.That( res.A == A * A );
            Assert.That( res.B == A * b + b );
        }

    }
}
