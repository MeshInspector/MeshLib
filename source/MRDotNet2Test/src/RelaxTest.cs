using System;
using System.IO;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class RelaxTests
    {
        [Test]
        public void TestRelax()
        {
            var sphere = makeSphere(new SphereParams(1.0f, 200));

            var parameters = new MeshRelaxParams();
            parameters.iterations = 20;

            var oldVolume = sphere.volume();

            relax(sphere, parameters);
            Assert.That( sphere.volume() / oldVolume < 0.7f );
        }

        [Test]
        public void TestRelaxKeppVolume()
        {
            var sphere = makeSphere(new SphereParams(1.0f, 200));

            var parameters = new MeshRelaxParams();
            parameters.iterations = 20;

            var oldVolume = sphere.volume();

            relaxKeepVolume(sphere, parameters);
            Assert.That(sphere.volume() / oldVolume > 0.7f);
        }
    }
}
