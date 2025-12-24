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
            var sphere = MakeSphere(new SphereParams(1.0f, 200));

            var parameters = new MeshRelaxParams();
            parameters.Iterations = 20;

            var oldVolume = sphere.Volume();

            Relax(sphere, parameters);
            Assert.That( sphere.Volume() / oldVolume < 0.7f );
        }

        [Test]
        public void TestRelaxKeppVolume()
        {
            var sphere = MakeSphere(new SphereParams(1.0f, 200));

            var parameters = new MeshRelaxParams();
            parameters.Iterations = 20;

            var oldVolume = sphere.Volume();

            RelaxKeepVolume(sphere, parameters);
            Assert.That(sphere.Volume() / oldVolume > 0.7f);
        }
    }
}
