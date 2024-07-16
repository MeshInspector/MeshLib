using System;
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
        public void TestNullArgs()
        {
            Assert.Throws<ArgumentNullException>( () => MeshBoolean.Boolean( null, null, BooleanOperation.Union ) );
        }
    }
}
