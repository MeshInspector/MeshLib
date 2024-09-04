﻿using System;
using System.IO;
using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class BooleanTests
    {
        [Test]
        public void TestOperations()
        {
            const float PI = 3.14159265f;
            Mesh meshA = Mesh.MakeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = Mesh.MakeTorus(1.0f, 0.2f, 8, 8);
            meshB.Transform( new AffineXf3f( Matrix3f.Rotation( Vector3f.PlusZ(), Vector3f.PlusY() ) ) );

            const float shiftStep = 0.2f;
            const float angleStep = PI;
            var baseAxis = new Vector3f[] { Vector3f.PlusX(), Vector3f.PlusY(), Vector3f.PlusZ() };

            for (int maskTrans = 0; maskTrans < 8; ++maskTrans)
            {
                for (int maskRot = 0; maskRot < 8; ++maskRot)
                {
                    for (float shift = 0.01f; shift < 0.2f; shift += shiftStep)
                    {
                        Vector3f shiftVec = new Vector3f();
                        for (int i = 0; i < 3; ++i)
                            if ( ( maskTrans & ( 1 << i ) ) > 0 )
                                shiftVec += shift * baseAxis[i];

                        for (float angle = PI * 0.01f; angle < PI * 7.0f / 18.0f; angle += angleStep)
                        {
                            Matrix3f rotation = new Matrix3f();
                            for (int i = 0; i < 3; ++i)
                                if ( ( maskRot & (1 << i) ) > 0 )
                                    rotation = Matrix3f.Rotation( baseAxis[i], angle ) * rotation; 

                            BooleanParameters parameters = new BooleanParameters();
                            parameters.rigidB2A = new AffineXf3f(shiftVec) * new AffineXf3f(rotation);

                            Assert.DoesNotThrow(() => MeshBoolean.Boolean(meshA, meshB, BooleanOperation.Union, parameters));
                            Assert.DoesNotThrow(() => MeshBoolean.Boolean(meshA, meshB, BooleanOperation.Intersection, parameters));
                        }
                    }
                }
            }
        }

        [Test]
        public void TestMapper()
        {
            Mesh meshA = Mesh.MakeTorus(1.1f, 0.5f, 8, 8);
            Mesh meshB = Mesh.MakeTorus(1.0f, 0.2f, 8, 8);
            meshB.Transform(new AffineXf3f(Matrix3f.Rotation(Vector3f.PlusZ(), Vector3f.PlusY())));

            var parameters = new BooleanParameters();
            parameters.mapper = new BooleanResultMapper();
            var booleanResult = MeshBoolean.Boolean(meshA, meshB, BooleanOperation.Union, parameters );
            var vMapA = parameters.mapper.VertMap(meshA.ValidPoints as BitSet, MapObject.A);
            var vMapB = parameters.mapper.VertMap(meshB.ValidPoints as BitSet, MapObject.B);
            Assert.That(vMapA.Size(), Is.EqualTo(64) );
            Assert.That(vMapA.Count(), Is.EqualTo(60));
            Assert.That(vMapB.Size(), Is.EqualTo(256) );
            Assert.That(vMapB.Count(), Is.EqualTo(48));

            var fMapA = parameters.mapper.FaceMap(meshA.ValidFaces as BitSet, MapObject.A);
            var fMapB = parameters.mapper.FaceMap(meshB.ValidFaces as BitSet, MapObject.B);
            Assert.That(fMapA.Size(), Is.EqualTo(256) );
            Assert.That(fMapA.Count(), Is.EqualTo(224));
            Assert.That(fMapB.Size(), Is.EqualTo(448) );
            Assert.That(fMapB.Count(), Is.EqualTo(192));

            var newFaces = parameters.mapper.NewFaces();
            Assert.That(newFaces.Size(), Is.EqualTo(448) );
            Assert.That(newFaces.Count(), Is.EqualTo(252));

            var mapsA = parameters.mapper.GetMaps( MapObject.A );
            Assert.That(!mapsA.Identity);
            Assert.That( mapsA.Old2NewVerts.Count, Is.EqualTo(160) );
            Assert.That( mapsA.Cut2NewFaces.Count, Is.EqualTo(348) );
            Assert.That( mapsA.Cut2Origin.Count, Is.EqualTo(348) );

            var mapsB = parameters.mapper.GetMaps( MapObject.B );
            Assert.That(!mapsB.Identity);
            Assert.That( mapsB.Old2NewVerts.Count, Is.EqualTo(160) );
            Assert.That( mapsB.Cut2NewFaces.Count, Is.EqualTo(384) );
            Assert.That( mapsB.Cut2Origin.Count, Is.EqualTo(384) );
        }

        [Test]
        public void TestNullArgs()
        {
            Assert.Throws<ArgumentNullException>( () => MeshBoolean.Boolean( null, null, BooleanOperation.Union ) );
        }
    }
}
