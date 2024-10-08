﻿using System.Runtime.InteropServices;
using static MR.DotNet.Vector3f;
using static MR.DotNet.Matrix3f;

namespace MR.DotNet
{
    public class AffineXf3f
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRAffineXf3f
        {
            public MRMatrix3f A;
            public MRVector3f b;
        };


        /*[DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRAffineXf3f mrAffineXf3fNew();

        /// creates translation-only transformation (with identity linear component)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRAffineXf3f mrAffineXf3fTranslation( ref MRVector3f b );

        /// creates linear-only transformation (without translation)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRAffineXf3f mrAffineXf3fLinear( ref MRMatrix3f A );*/

        /// composition of two transformations:
        /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRAffineXf3f mrAffineXf3fMul( ref MRAffineXf3f a, ref MRAffineXf3f b );

        /// application of the transformation to a point
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRVector3f mrAffineXf3fApply( ref MRAffineXf3f xf, ref MRVector3f v );

        internal MRAffineXf3f xf_;

        private Matrix3f A_;
        private Vector3f b_;

        internal AffineXf3f( MRAffineXf3f xf )
        {
            xf_ = xf;
            A_ = new Matrix3f( xf.A );
            b_ = new Vector3f( xf.b );
        }

        public AffineXf3f() 
            : this( new Matrix3f(), new Vector3f() )
        { }
        public AffineXf3f(Matrix3f A)
             : this(A, new Vector3f())
        { }
        public AffineXf3f( Vector3f b )
            : this( new Matrix3f(), b )
        { }

        public AffineXf3f( Matrix3f A, Vector3f b )
        {
            A_ = A;
            b_ = b;
            xf_.A = A.mat_;
            xf_.b = b.vec_;
        }

        public Vector3f Apply( Vector3f v )
        {
            return new Vector3f( mrAffineXf3fApply( ref xf_, ref v.vec_ ) );
        }

        static public AffineXf3f operator*( AffineXf3f a, AffineXf3f b )
        {
            return new AffineXf3f( mrAffineXf3fMul( ref a.xf_, ref b.xf_ ) );
        }

        public Matrix3f A { get { return A_; } set { A_ = value; xf_.A = value.mat_; } }
        public Vector3f b { get { return b_; } set { b_ = value; xf_.b = value.vec_; } }
    }
}
