using System;
using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRPointsLoadSettings
        {
            public IntPtr colors = IntPtr.Zero;
            public IntPtr outXf = IntPtr.Zero;
            public IntPtr callback = IntPtr.Zero;

            public MRPointsLoadSettings() { }
        }
        /// structure with settings and side output parameters for loading point cloud
        public struct PointsLoadSettings
        {
            /// points where to load point color map
            public VertColors? colors = null;
            /// transform for the loaded point cloud
            public AffineXf3f? outXf = null;

            public PointsLoadSettings() { }

            internal MRPointsLoadSettings ToNative()
            {
                MRPointsLoadSettings res = new MRPointsLoadSettings();
                res.colors = colors is null ? IntPtr.Zero : colors.data_;
                res.outXf = outXf is null ? IntPtr.Zero : outXf.XfAddr();
                return res;
            }
        }
    }
}