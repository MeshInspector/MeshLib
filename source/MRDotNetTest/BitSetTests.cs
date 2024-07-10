using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class BitSetTests
    {
        [Test]
        public void TestDefaultConstructor()
        {
            var a = new MR.DotNet.BitSet();
            Assert.That( a.size() == 0 );
        }

        [Test]
        public void TestConstructor()
        {
            var a = new MR.DotNet.BitSet( 10 );
            Assert.That( a.size() == 10 );
        }

        [Test]
        public void TestSet()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.set( 5 );
            Assert.That( a.test( 5 ) );
            Assert.That( !a.test( 4 ) );
        }

        [Test]
        public void TestFindLast()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.set( 5 );
            Assert.That( a.findLast() == 5 );
        }

        [Test]
        public void TestAutoResize()
        {
            var a = new MR.DotNet.BitSet();
            a.autoResizeSet(6);
            Assert.That( a.size() == 7 );
        }
    }
}
