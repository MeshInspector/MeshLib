using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class BitSetTests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new BitSet();
            Assert.That( a.size(), Is.EqualTo((nuint)0) );
        }

        [Test]
        public void TestConstructor()
        {
            var a = new BitSet( 20 );
            Assert.That(a.size(), Is.EqualTo((nuint)20));
        }

        [Test]
        public void TestSet()
        {
            var a = new BitSet( 10 );
            a.set( 5 );
            Assert.That( a.test( 5 ) );
            Assert.That( !a.test( 4 ) );
        }

        [Test]
        public void TestFindLast()
        {
            var a = new BitSet( 10 );
            a.set( 5 );
            Assert.That( a.findLast(), Is.EqualTo((nuint)5) );
        }

        [Test]
        public void TestResize()
        {
            var a = new BitSet();
            a.resize(7);
            a.set(6);
            Assert.That( a.size() == 7 );
            Assert.That( a.test( 6 ) );
        }

        [Test]
        public void TestSubtraction()
        {
            var a = new BitSet( 10 );
            a.set( 5 );
            a.set(6);
            var b = new BitSet( 10 );
            b.set( 6 );
            var c = a - b;
            Assert.That( c.test( 5 ) );
            Assert.That( !c.test( 6 ) );
        }

        [Test]
        public void TestUnion()
        {
            var a = new BitSet( 10 );
            a.set( 5 );
            var b = new BitSet( 10 );
            b.set( 6 );
            var c = a | b;
            Assert.That( c.test( 5 ) );
            Assert.That( c.test( 6 ) );
        }
    }
}
