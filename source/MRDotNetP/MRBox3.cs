using System.Runtime.InteropServices;
using static MR.DotNet.Vector3f;

namespace MR.DotNet
{
    public class Box3f
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRBox3f
        {
            public MRVector3f min;
            public MRVector3f max;
        };

        /// creates invalid box by default
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRBox3f mrBox3fNew();

        /// true if the box contains at least one point
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern bool mrBox3fValid( ref MRBox3f box );

        /// computes size of the box in all dimensions
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRVector3f mrBox3fSize(ref MRBox3f box);

        /// computes length from min to max
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern float mrBox3fDiagonal(ref MRBox3f box);

        /// computes the volume of this box
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern float mrBox3fVolume(ref MRBox3f box);

        MRBox3f box_;

        Vector3f min_;
        Vector3f max_;

        internal Box3f(MRBox3f box)
        {
            box_ = box;
            min_ = new Vector3f(box_.min);
            max_ = new Vector3f(box_.max);
        }

        public Box3f()
        {
            box_ = mrBox3fNew();
            min_ = new Vector3f(box_.min);
            max_ = new Vector3f(box_.max);
        }

        public Box3f( Vector3f min, Vector3f max )
        {
            box_.min = min.vec_;
            box_.max = max.vec_;

            min_ = min;
            max_ = max;
        }

        public bool Valid() => mrBox3fValid(ref box_);
        public Vector3f Size() => new Vector3f(mrBox3fSize(ref box_));
        public float Diagonal() => mrBox3fDiagonal(ref box_);
        public float Volume() => mrBox3fVolume(ref box_);
    }
}
