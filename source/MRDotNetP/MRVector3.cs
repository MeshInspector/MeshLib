using System.Runtime.InteropServices;

namespace MR.DotNet
{
    /// represents a 3-dimentional float-typed vector
    public class Vector3f
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRVector3f
        {
            public float x;
            public float y;
            public float z;
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
        private static extern float mrVector3fLength(ref MRVector3f a );

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

        static public Vector3f PlusX()
        {
            return new Vector3f(mrVector3fPlusX());
        }

        static public Vector3f PlusY()
        {
            return new Vector3f(mrVector3fPlusY());
        }

        static public Vector3f PlusZ()
        {
            return new Vector3f(mrVector3fPlusZ());
        }

        static public Vector3f operator +(Vector3f a, Vector3f b) => new Vector3f(mrVector3fAdd(ref a.vec_, ref b.vec_));

        static public Vector3f operator -(Vector3f a, Vector3f b) => new Vector3f(mrVector3fSub(ref a.vec_, ref b.vec_));

        static public Vector3f operator *(Vector3f a, float b) => new Vector3f(mrVector3fMulScalar(ref a.vec_, b));

        static public Vector3f operator *(float a, Vector3f b) => new Vector3f(mrVector3fMulScalar(ref b.vec_, a));

        static public bool operator ==(Vector3f a, Vector3f b) => a.vec_.x == b.vec_.x && a.vec_.y == b.vec_.y && a.vec_.z == b.vec_.z;
        static public bool operator !=(Vector3f a, Vector3f b) => a.vec_.x != b.vec_.x || a.vec_.y != b.vec_.y || a.vec_.z != b.vec_.z;

        public override bool Equals(object obj)
        {
            return (obj is Vector3f) ? this == (Vector3f)obj : false;
        }

        public override int GetHashCode() => vec_.x.GetHashCode() ^ vec_.y.GetHashCode() ^ vec_.z.GetHashCode();

        public float X { get => vec_.x; set => vec_.x = value; }
        public float Y { get => vec_.y; set => vec_.y = value; }
        public float Z { get => vec_.z; set => vec_.z = value; }

        public float Length() => mrVector3fLength(ref vec_);

        public float LengthSq() => mrVector3fLengthSq(ref vec_);

    }
}
