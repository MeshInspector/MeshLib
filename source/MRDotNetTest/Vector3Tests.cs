using NUnit.Framework;
using MR.DotNet;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class Vector3Tests
    {
        [Test]
        public void TestAddition()
        {
            var a = new Vector3f(1, 2, 3);
            var b = new Vector3f(4, 5, 6);
            var c = a + b;

            Assert.That(5 == c.x);
            Assert.That(7 == c.y);
        }
    }
}
