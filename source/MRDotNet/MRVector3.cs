using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {
        /// represents a 3-dimentional float-typed vector
        public class Vector3f
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRVector3f
            {
                public float x = 0.0f;
                public float y = 0.0f;
                public float z = 0.0f;
                public MRVector3f() { }
            };

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fDiagonal(float a);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fPlusX();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fPlusY();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fPlusZ();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fAdd(ref MRVector3f a, ref MRVector3f b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fSub(ref MRVector3f a, ref MRVector3f b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrVector3fMulScalar(ref MRVector3f a, float b);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern float mrVector3fLength(ref MRVector3f a);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern float mrVector3fLengthSq(ref MRVector3f a);

            internal MRVector3f vec_;
            /// creates a new vector with zero coordinates
            public Vector3f()
            {
                vec_ = mrVector3fDiagonal(0.0f);
            }

            internal Vector3f(MRVector3f vec)
            {
                vec_ = vec;
            }
            /// creates a new vector with specified coordinates
            public Vector3f(float x, float y, float z)
            {
                vec_.x = x;
                vec_.y = y;
                vec_.z = z;
            }
            /// creates a new vector with same coordinates
            static public Vector3f Diagonal(float a)
            {
                return new Vector3f(mrVector3fDiagonal(a));
            }
            /// returns a new vector with (1, 0, 0) coordinates
            static public Vector3f PlusX()
            {
                return new Vector3f(mrVector3fPlusX());
            }
            /// returns a new vector with (0, 1, 0) coordinates
            static public Vector3f PlusY()
            {
                return new Vector3f(mrVector3fPlusY());
            }
            /// returns a new vector with (0, 0, 1) coordinates
            static public Vector3f PlusZ()
            {
                return new Vector3f(mrVector3fPlusZ());
            }
            ///returns sum of two vectors
            static public Vector3f operator +(Vector3f a, Vector3f b) => new Vector3f(mrVector3fAdd(ref a.vec_, ref b.vec_));
            ///returns difference of two vectors
            static public Vector3f operator -(Vector3f a, Vector3f b) => new Vector3f(mrVector3fSub(ref a.vec_, ref b.vec_));
            ///returns product of vector and scalar
            static public Vector3f operator *(Vector3f a, float b) => new Vector3f(mrVector3fMulScalar(ref a.vec_, b));
            ///returns product of vector and scalar
            static public Vector3f operator *(float a, Vector3f b) => new Vector3f(mrVector3fMulScalar(ref b.vec_, a));

            static public bool operator ==(Vector3f a, Vector3f b) => a.vec_.x == b.vec_.x && a.vec_.y == b.vec_.y && a.vec_.z == b.vec_.z;
            static public bool operator !=(Vector3f a, Vector3f b) => a.vec_.x != b.vec_.x || a.vec_.y != b.vec_.y || a.vec_.z != b.vec_.z;

            public override bool Equals(object obj)
            {
                return (obj is Vector3f) ? this == (Vector3f)obj : false;
            }

            public override int GetHashCode() => vec_.x.GetHashCode() ^ vec_.y.GetHashCode() ^ vec_.z.GetHashCode();
            /// returns first coordinate
            public float X { get => vec_.x; set => vec_.x = value; }
            /// returns second coordinate
            public float Y { get => vec_.y; set => vec_.y = value; }
            /// returns third coordinate
            public float Z { get => vec_.z; set => vec_.z = value; }
            /// returns Euclidean length of the vector
            public float Length() => mrVector3fLength(ref vec_);
            /// returns squared Euclidean length of the vector
            public float LengthSq() => mrVector3fLengthSq(ref vec_);

        }
    }
}
