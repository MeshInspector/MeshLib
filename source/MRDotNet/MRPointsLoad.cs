using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace MR
{
    public partial class DotNet
    {
        public class PointsLoad
        {
            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern IntPtr mrPointsLoadFromAnySupportedFormat(string filename, ref MRPointsLoadSettings settings, ref IntPtr errorString);

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrLoadIOExtras();

            /// loads point cloud from file of any supported format
            public static PointCloud FromAnySupportedFormat(string path, PointsLoadSettings? settings = null)
            {
                mrLoadIOExtras();

                MRPointsLoadSettings mrSettings = settings is null ? new MRPointsLoadSettings() : settings.Value.ToNative();
                IntPtr errString = IntPtr.Zero;
                var mesh = mrPointsLoadFromAnySupportedFormat(path, ref mrSettings, ref errString);

                if (errString != IntPtr.Zero)
                {
                    var errData = mrStringData(errString);
                    string errorMessage = MarshalNativeUtf8ToManagedString(errData);
                    throw new SystemException(errorMessage);
                }

                return new PointCloud(mesh);
            }
        }
    }
}