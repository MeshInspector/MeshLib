using System.Runtime.InteropServices;
using static MR.DotNet;
using static MR.DotNet.Box3i;
using static MR.DotNet.Vector3f;
using static MR.DotNet.Vector3i;

namespace MR
{
    public partial class DotNet
    {
        public class Box3f
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRBox3f
            {
                public MRVector3f min;
                public MRVector3f max;
            };


            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRBox3f mrBox3fNew();


            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            [return: MarshalAs(UnmanagedType.I1)]
            private static extern bool mrBox3fValid(ref MRBox3f box);


            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrBox3fSize(ref MRBox3f box);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern float mrBox3fDiagonal(ref MRBox3f box);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern float mrBox3fVolume(ref MRBox3f box);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3f mrBox3fCenter(ref MRBox3f box);

            MRBox3f box_;

            Vector3f min_;
            Vector3f max_;

            internal void invalidateBox()
            {
                box_.min = min_.vec_;
                box_.max = max_.vec_;
            }

            internal ref MRBox3f boxRef()
            {
                invalidateBox();
                return ref box_;
            }

            internal Box3f(MRBox3f box)
            {
                box_ = box;
                min_ = new Vector3f(box_.min);
                max_ = new Vector3f(box_.max);
            }

            /// creates invalid box by default
            public Box3f()
            {
                box_ = mrBox3fNew();
                min_ = new Vector3f(box_.min);
                max_ = new Vector3f(box_.max);
            }
            /// creates box with given min and max
            public Box3f(Vector3f min, Vector3f max)
            {
                box_.min = min.vec_;
                box_.max = max.vec_;

                min_ = min;
                max_ = max;
            }
            /// true if the box contains at least one point
            public bool Valid() => mrBox3fValid(ref box_);
            /// computes size of the box in all dimensions
            public Vector3f Size() => new Vector3f(mrBox3fSize(ref box_));
            /// computes length from min to max
            public float Diagonal() => mrBox3fDiagonal(ref box_);
            /// computes the volume of this box
            public float Volume() => mrBox3fVolume(ref box_);
            /// computes the center of this box
            public Vector3f Center() => new Vector3f(mrBox3fCenter(ref box_));
            /// returns the min point
            public Vector3f Min { get { return min_; } set { min_ = value; box_.min = value.vec_; } }
            /// returns the max point
            public Vector3f Max { get { return max_; } set { max_ = value; box_.max = value.vec_; } }
        }

        public class Box3i
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRBox3i
            {
                public MRVector3i min;
                public MRVector3i max;
            };


            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRBox3i mrBox3iNew();


            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            [return: MarshalAs(UnmanagedType.I1)]
            private static extern bool mrBox3iValid(ref MRBox3i box);


            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3i mrBox3iSize(ref MRBox3i box);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern int mrBox3iVolume(ref MRBox3i box);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRVector3i mrBox3iCenter(ref MRBox3i box);

            MRBox3i box_;
            Vector3i min_;
            Vector3i max_;

            internal void invalidateBox()
            {
                box_.min = min_.vec_;
                box_.max = max_.vec_;
            }

            internal ref MRBox3i boxRef()
            {
                invalidateBox();
                return ref box_;
            }

            internal Box3i(MRBox3i box)
            {
                box_ = box;
                min_ = new Vector3i(box.min);
                max_ = new Vector3i(box.max);
            }

            /// creates invalid box by default
            public Box3i()
            {
                var box = mrBox3iNew();
                min_ = new Vector3i(box.min);
                max_ = new Vector3i(box.max);
            }
            /// creates box with given min and max
            public Box3i(Vector3i min, Vector3i max)
            {
                box_.min = min.vec_;
                box_.max = max.vec_;

                min_ = min;
                max_ = max;
            }


            /// true if the box contains at least one point
            public bool Valid() => mrBox3iValid(ref box_);
            /// computes size of the box in all dimensions
            public Vector3i Size() => new Vector3i(mrBox3iSize(ref box_));
            /// computes the volume of this box
            public int Volume() => mrBox3iVolume(ref box_);
            /// computes the center of this box
            public Vector3i Center() => new Vector3i(mrBox3iCenter(ref box_));
            /// returns the min point
            public Vector3i Min { get { return min_; } set { min_ = value; box_.min = value.vec_; } }
            /// returns the max point
            public Vector3i Max { get { return max_; } set { max_ = value; box_.max = value.vec_; } }
        }
    }
}
