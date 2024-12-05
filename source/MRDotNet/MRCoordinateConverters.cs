using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace MR
{
    public partial class DotNet
    {
        /// this class contains coordinate converters float-int-float
        public class CoordinateConverters : IDisposable
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRCoordinateConverters
            {
                public IntPtr toInt = IntPtr.Zero;
                public IntPtr toFloat = IntPtr.Zero;
                public MRCoordinateConverters() { }
            }

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRCoordinateConverters mrGetVectorConverters(ref MRMeshPart a, ref MRMeshPart b, IntPtr rigidB2A);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern void mrConvertToIntVectorFree(IntPtr conv);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern void mrConvertToFloatVectorFree(IntPtr conv);

            /// creates new converters for given pair of meshes
            public CoordinateConverters(MeshPart meshA, MeshPart meshB)
            {
                conv_ = mrGetVectorConverters(ref meshA.mrMeshPart, ref meshB.mrMeshPart, IntPtr.Zero);
            }

            private bool disposed = false;
            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!disposed)
                {
                    if (conv_.toInt != IntPtr.Zero)
                    {
                        mrConvertToIntVectorFree(conv_.toInt);
                        conv_.toInt = IntPtr.Zero;
                    }

                    if (conv_.toFloat != IntPtr.Zero)
                    {
                        mrConvertToFloatVectorFree(conv_.toFloat);
                        conv_.toFloat = IntPtr.Zero;
                    }

                    disposed = true;
                }
            }

            ~CoordinateConverters()
            {
                Dispose(false);
            }

            internal IntPtr GetConvertToFloatVector()
            {
                return conv_.toFloat;
            }

            internal IntPtr GetConvertToIntVector()
            {
                return conv_.toInt;
            }

            internal MRCoordinateConverters conv_;
        }
    }
}
