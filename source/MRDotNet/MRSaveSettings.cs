using System;
using System.Runtime.InteropServices;

namespace MR
{
    public partial class DotNet
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRSaveSettings
        {
            public byte saveOnly = 0;
            public byte rearrangeTriangles = 1;
            public IntPtr colors = IntPtr.Zero;
            public IntPtr callback = IntPtr.Zero;

            public MRSaveSettings() { }
        }

        /// determines how to save points/lines/mesh
        public struct SaveSettings
        {            
            /// true - save valid points/vertices only (pack them);
            /// false - save all points/vertices preserving their indices
            public bool saveOnly = false;
            /// if it is turned on, then higher compression ratios are reached but the order of triangles is changed;
            /// currently affects .ctm format only
            public bool rearrangeTriangles = true;
            /// optional per-vertex color to save with the geometry
            public VertColors? colors = null;

            public SaveSettings() { }

            internal MRSaveSettings ToNative()
            {
                MRSaveSettings saveSettings = new MRSaveSettings();
                saveSettings.saveOnly = (byte)( saveOnly ? 1 : 0 );
                saveSettings.rearrangeTriangles = (byte)(saveOnly ? 1 : 0);
                if (colors != null) 
                    saveSettings.colors = colors.data_;

                return saveSettings;
            }
        }
    }
}
