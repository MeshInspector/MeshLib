using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class ICPTests
    {
        [Test]
        public void TestICP()
        {
            var torusRef = Mesh.MakeTorus(2, 1, 32, 32);
            var torusMove = Mesh.MakeTorus(2, 1, 32, 32);

            var axis = Vector3f.PlusX();
            var trans = new Vector3f(0.0f, 0.2f, 0.105f);
            var xf = new AffineXf3f(Matrix3f.Rotation(axis, 0.2f), trans);

            MeshOrPointsXf flt = new MeshOrPointsXf();
            flt.obj = torusMove;
            flt.xf = xf;

            MeshOrPointsXf refer = new MeshOrPointsXf();
            refer.xf = new AffineXf3f();
            refer.obj = torusRef;

            var icp = new ICP(flt, refer, torusMove.ValidPoints as BitSet, torusRef.ValidPoints as BitSet );

            var newXf = icp.CalculateTransformation();
            Console.WriteLine(icp.GetStatusInfo());

            var diffXf = new AffineXf3f();
            diffXf.A -= newXf.A;
            diffXf.B -= newXf.B;

            Assert.That(Math.Abs(diffXf.A.X.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.X.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.X.Z), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Y.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Y.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Y.Z), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Z.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Z.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Z.Z), Is.LessThan(1e-6f));

            Assert.That(Math.Abs(diffXf.B.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.B.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.B.Z), Is.LessThan(1e-6f));

            var pairs = icp.GetRef2FltPairs();
            Assert.That(pairs.pairs.Count, Is.EqualTo(1024));

            pairs = icp.GetFlt2RefPairs();
            Assert.That(pairs.pairs.Count, Is.EqualTo(1024));
        }
    }
}
