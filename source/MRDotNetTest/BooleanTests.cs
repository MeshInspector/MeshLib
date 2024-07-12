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
            Mesh meshA = Mesh.MakeCube( Vector3f.Diagonal(1), new Vector3f( 0.5f, -0.5f, -0.5f ) );
            Mesh meshB = Mesh.MakeSphere(1.0f, 100);

            for ( int i = 0; i < (int)BooleanOperation.Count; ++i )
            {
                BooleanOperation op = (BooleanOperation)i;
                var res = MeshBoolean.Boolean( meshA, meshB, op );
                Assert.That( TestTools.AreMeshesEqual( res.mesh, TestTools.GetPathToPattern( op.ToString() + ".mrmesh" ) ) );
            }
        }

        [Test]
        public void TestSpartan()
        {
            var meshA = Mesh.FromFile( TestTools.GetPathToTestFile( "spartan.mrmesh" ) );
            Mesh meshB = Mesh.MakeSphere( 20.0f, 100 );
            var res = MeshBoolean.Boolean( meshA, meshB, BooleanOperation.Union );
            Assert.That( TestTools.AreMeshesEqual( res.mesh, TestTools.GetPathToPattern( "spartan_union.mrmesh" ) ) );
        }
    }
}
