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
            Assert.That( a.Size() == 0 );
        }

        [Test]
        public void TestConstructor()
        {
            var a = new MR.DotNet.BitSet( 10 );
            Assert.That( a.Size() == 10 );
        }

        [Test]
        public void TestSet()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.Set( 5 );
            Assert.That( a.Test( 5 ) );
            Assert.That( !a.Test( 4 ) );
        }

        [Test]
        public void TestFindLast()
        {
            var a = new MR.DotNet.BitSet( 10 );
            a.Set( 5 );
            Assert.That( a.FindLast() == 5 );
        }

        [Test]
        public void TestAutoResize()
        {
            var a = new MR.DotNet.BitSet();
            a.AutoResizeSet(6);
            Assert.That( a.Size() == 7 );
        }
    }
}
