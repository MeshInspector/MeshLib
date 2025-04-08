using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using static MR.DotNet.SelfIntersections;

namespace MR
{
    public partial class DotNet
    {
        public class PointsSave
        {
            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrPointsSaveToAnySupportedFormat(IntPtr pc, string file, ref MRSaveSettings settings, ref IntPtr errorString);

            [DllImport("MRMeshC", CharSet = CharSet.Ansi)]
            private static extern void mrLoadIOExtras();

            /// saves point cloud to file of any supported format
            unsafe public static void ToAnySupportedFormat(PointCloud pc, string path, SaveSettings? settings = null)
            {
                mrLoadIOExtras();

                IntPtr errString = IntPtr.Zero;
                MRSaveSettings mrSettings = settings is null ? new MRSaveSettings() : settings.Value.ToNative();
                mrPointsSaveToAnySupportedFormat(pc.pc_, path, ref mrSettings, ref errString);

                if (errString != IntPtr.Zero)
                {
                    var errData = mrStringData(errString);
                    string errorMessage = MarshalNativeUtf8ToManagedString(errData);
                    throw new SystemException(errorMessage);
                }
            }
        }
    }
}