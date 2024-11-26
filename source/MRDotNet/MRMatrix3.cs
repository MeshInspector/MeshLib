using System.Data.Common;
using System.Runtime.InteropServices;
using static MR.DotNet.Vector3f;

namespace MR
{
    public partial class DotNet
    {
        /// arbitrary row-major 3x3 matrix
        public class Matrix3f
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRMatrix3f
            {
                public Vector3f.MRVector3f x;
                public Vector3f.MRVector3f y;
                public Vector3f.MRVector3f z;
            };

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRMatrix3f mrMatrix3fIdentity();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRMatrix3f mrMatrix3fRotationScalar(ref MRVector3f axis, float angle);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRMatrix3f mrMatrix3fRotationVector(ref MRVector3f from, ref MRVector3f to);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRMatrix3f mrMatrix3fAdd(ref MRMatrix3f a, ref MRMatrix3f b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRMatrix3f mrMatrix3fSub(ref MRMatrix3f a, ref MRMatrix3f b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRMatrix3f mrMatrix3fMul(ref MRMatrix3f a, ref MRMatrix3f b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrMatrix3fMulVector(ref MRMatrix3f a, ref MRVector3f b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            [return: MarshalAs(UnmanagedType.I1)]
            private static extern bool mrMatrix3fEqual(ref MRMatrix3f a, ref MRMatrix3f b);

            internal MRMatrix3f mat_;
            /// creates the identity matrix
            public Matrix3f()
            {
                mat_ = mrMatrix3fIdentity();
                x_ = Vector3f.PlusX();
                y_ = Vector3f.PlusY();
                z_ = Vector3f.PlusZ();
            }

            internal Matrix3f(MRMatrix3f mat)
            {
                mat_ = mat;

                x_ = new Vector3f(mat_.x);
                y_ = new Vector3f(mat_.y);
                z_ = new Vector3f(mat_.z);
            }
            /// creates matrix with given rows
            public Matrix3f(Vector3f x, Vector3f y, Vector3f z)
            {
                mat_.x = x.vec_;
                mat_.y = y.vec_;
                mat_.z = z.vec_;

                x_ = x;
                y_ = y;
                z_ = z;
            }
            /// creates zero matrix
            public static Matrix3f Zero()
            {
                return new Matrix3f(new Vector3f(), new Vector3f(), new Vector3f());
            }
            /// creates rotation matrix around given axis with given angle
            public static Matrix3f Rotation(Vector3f axis, float angle)
            {
                return new Matrix3f(mrMatrix3fRotationScalar(ref axis.vec_, angle));
            }
            /// creates rotation matrix from one vector to another
            public static Matrix3f Rotation(Vector3f from, Vector3f to)
            {
                return new Matrix3f(mrMatrix3fRotationVector(ref from.vec_, ref to.vec_));
            }
            public static Matrix3f operator +(Matrix3f a, Matrix3f b)
            {
                return new Matrix3f(mrMatrix3fAdd(ref a.mat_, ref b.mat_));
            }

            public static Matrix3f operator -(Matrix3f a, Matrix3f b)
            {
                return new Matrix3f(mrMatrix3fSub(ref a.mat_, ref b.mat_));
            }
            public static Matrix3f operator *(Matrix3f a, Matrix3f b)
            {
                return new Matrix3f(mrMatrix3fMul(ref a.mat_, ref b.mat_));
            }

            public static Vector3f operator *(Matrix3f a, Vector3f b)
            {
                return new Vector3f(mrMatrix3fMulVector(ref a.mat_, ref b.vec_));
            }

            public static bool operator ==(Matrix3f a, Matrix3f b)
            {
                return mrMatrix3fEqual(ref a.mat_, ref b.mat_);
            }

            public static bool operator !=(Matrix3f a, Matrix3f b)
            {
                return !mrMatrix3fEqual(ref a.mat_, ref b.mat_);
            }

            public override bool Equals(object obj)
            {
                return (obj is Matrix3f) ? this == (Matrix3f)obj : false;
            }

            public override int GetHashCode() => mat_.x.GetHashCode() ^ mat_.y.GetHashCode() ^ mat_.z.GetHashCode();

            private Vector3f x_;
            private Vector3f y_;
            private Vector3f z_;
            /// first row
            public Vector3f X { get { return x_; } set { mat_.x = value.vec_; x_ = value; } }
            /// second row
            public Vector3f Y { get { return y_; } set { mat_.y = value.vec_; y_ = value; } }
            /// third row
            public Vector3f Z { get { return z_; } set { mat_.z = value.vec_; z_ = value; } }
        }
    }
}