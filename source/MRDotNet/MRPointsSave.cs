using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace MR
{
    public partial class DotNet
    {
        public class PointsSave
        {
            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            unsafe private static extern void mrPointsSaveToAnySupportedFormat(IntPtr pc, string file, IntPtr* errorString);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
            private static extern void mrLoadIOExtras();

            /// saves point cloud to file of any supported format
            unsafe public static void ToAnySupportedFormat(PointCloud pc, string path)
            {
                mrLoadIOExtras();

                IntPtr errString = new IntPtr();
                mrPointsSaveToAnySupportedFormat(pc.pc_, path, &errString);
                if (errString != IntPtr.Zero)
                {
                    var errData = mrStringData(errString);
                    string errorMessage = Marshal.PtrToStringAnsi(errData);
                    throw new SystemException(errorMessage);
                }
            }
        }
    }
}