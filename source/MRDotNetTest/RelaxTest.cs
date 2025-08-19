using System;
using System.IO;
using NUnit.Framework;
using static MR.DotNet;

namespace MR.Test
{
    [TestFixture]
    internal class RelaxTests
    {
        [Test]
        public void TestRelax()
        {
            var sphere = Mesh.MakeSphere(1.0f, 200);

            BitSetReadOnly savedRegion = sphere.ValidFaces.Clone();

            var parameters = new RelaxParams();
            parameters.iterations = 20;

            var oldVolume = sphere.Volume();

            Relax(ref sphere, parameters);
            Assert.That( sphere.Volume() / oldVolume < 0.7f );
        }

        [Test]
        public void TestRelaxKeppVolume()
        {
            var sphere = Mesh.MakeSphere(1.0f, 200);

            BitSetReadOnly savedRegion = sphere.ValidFaces.Clone();

            var parameters = new RelaxParams();
            parameters.iterations = 20;

            var oldVolume = sphere.Volume();

            RelaxKeepVolume(ref sphere, parameters);
            Assert.That(sphere.Volume() / oldVolume > 0.7f);
        }
    }
}
