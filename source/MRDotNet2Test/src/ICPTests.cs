using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;
using static MR;

namespace MRTest
{
    [TestFixture]
    internal class ICPTests
    {
        [Test]
        public void TestMultiwayICP()
        {
            var inputs = new Std.Vector_MRMeshOrPointsXf();
            Box3f maxBBox = new Box3f();

            inputs.PushBack( new MeshOrPointsXf(MakeSphere(new SphereParams(1.0f, 1000)).Value, new AffineXf3f())); // TODO: replace _Moved
            Box3f bbox = inputs.At(0).Obj.GetObjBoundingBox();
            if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                maxBBox = bbox;

            inputs.PushBack(new MeshOrPointsXf(MakeSphere(new SphereParams(1.0f, 1000)).Value, AffineXf3f.Linear(Matrix3f.Rotation(Vector3f.PlusZ(), 0.1f)))); // TODO: replace _Moved
            bbox = inputs.At(1).Obj.GetObjBoundingBox();
            if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                maxBBox = bbox;

            MultiwayICPSamplingParameters samplingParams = new MultiwayICPSamplingParameters();
            samplingParams.SamplingVoxelSize = maxBBox.Diagonal() * 0.03f;

            MultiwayICP icp = new MultiwayICP(new Vector_MRMeshOrPointsXf_MRObjId(Misc.Move(inputs)), samplingParams);
            ICPProperties iCPProperties = new ICPProperties();
            icp.SetParams(iCPProperties);
            icp.UpdateAllPointPairs();

            Assert.That(icp.GetNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.GetNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.GetMeanSqDistToPoint(), Is.EqualTo(0.00254).Within(1e-5));

            Console.WriteLine("Calculating transformations...");
            var xfs = icp.CalculateTransformations();
            Assert.That(icp.GetNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.GetNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.GetMeanSqDistToPoint(), Is.EqualTo(0).Within(1e-5));
        }

        [Test]
        public void TestICP()
        {
            var torusRef = MakeTorus(2, 1, 32, 32).Value; // TODO: replace _Moved
            var torusMove = MakeTorus(2, 1, 32, 32).Value; // TODO: replace _Moved

            var axis = Vector3f.PlusX();
            var trans = new Vector3f(0.0f, 0.2f, 0.105f);
            var xf = new AffineXf3f(Matrix3f.Rotation(axis, 0.2f), trans);

            MeshOrPointsXf flt = new MeshOrPointsXf(torusMove, xf );
            MeshOrPointsXf refer = new MeshOrPointsXf( torusRef, new AffineXf3f());

            var fltSamples = torusMove.Topology.GetValidVerts();
            var referSamples = torusRef.Topology.GetValidVerts();
            Assert.That(fltSamples is not null);
            Assert.That(referSamples is not null);

            if (fltSamples is null || referSamples is null)
                return;

            var icp = new ICP(flt, refer, fltSamples, referSamples);

            var newXf = icp.CalculateTransformation();
            Console.WriteLine(icp.GetStatusInfo().Value); // TODO: replace _Moved

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
            Assert.That(pairs.Size(), Is.EqualTo(1024));

            pairs = icp.GetFlt2RefPairs();
            Assert.That(pairs.Size(), Is.EqualTo(1024));
        }
    }
}
