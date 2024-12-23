using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using static MR.DotNet;

namespace MR
{
    public partial class DotNet
    {
        /// stores a color in 32-bit RGBA format
        public class Color
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRColor
            {
                public byte r = 0, g = 0, b = 0, a = 255;
                public MRColor() { }
            };

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRColor mrColorFromComponents(byte r, byte g, byte b, byte a);            
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern MRColor mrColorFromFloatComponents(float r, float g, float b, float a);
            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern uint mrColorGetUInt32( ref MRColor color );

            internal MRColor color_;
            internal Color(MRColor color) { color_ = color; }

            /// creates opaque black color by default
            public Color() => color_ = mrColorFromComponents(0, 0, 0, 255);
            /// creates color from components [0..255]
            public Color(byte r, byte g, byte b, byte a = 255) => color_ = mrColorFromComponents(r, g, b, a);
            /// creates color from float components [0..1]
            public Color(float r, float g, float b, float a = 1.0f) => color_ = mrColorFromFloatComponents(r, g, b, a);
            /// returns color as unsigned int
            public uint GetUInt32() => mrColorGetUInt32(ref color_);

           public override bool Equals(object obj) => (obj is Color) ? this == (Color)obj : false;
           public override int GetHashCode() => GetUInt32().GetHashCode();
           static public bool operator == (Color a, Color b) => a.color_.r == b.color_.r && a.color_.g == b.color_.g && a.color_.b == b.color_.b && a.color_.a == b.color_.a;
           static public bool operator != (Color a, Color b) => !(a == b);


            public byte R => color_.r;
            public byte G => color_.g;
            public byte B => color_.b;
            public byte A => color_.a;
        }
        /// stores native pointer to native array of vertex colors
        public class VertColors : IDisposable
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRVertColors
            {
                public IntPtr data;
                public ulong size;
                public IntPtr reserved;
            }

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern IntPtr mrVertColorsNew();

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern IntPtr mrVertColorsNewSized(ulong size);

            [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
            private static extern void mrVertColorsFree(IntPtr colors);


            internal IntPtr data_;
            private bool disposed_ = false;
            
            /// creates empty array, to pass as optional output parameter
            public VertColors()
            {
                data_ = mrVertColorsNew();
            }
            /// creates array from given list of colors
            unsafe public VertColors( List<Color> colors )
            {
                data_ = mrVertColorsNewSized((ulong)colors.Count);
                MRVertColors* data = (MRVertColors*)data_;
                for (int i = 0; i < colors.Count; i++)
                {
                    Marshal.StructureToPtr(colors[i].color_, IntPtr.Add(data->data, i * sizeof(Color.MRColor)), false);
                }
            }

            internal VertColors(IntPtr colors) => data_ = colors;

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            unsafe protected virtual void Dispose(bool disposing)
            {
                if (!disposed_)
                {
                    if (data_ != IntPtr.Zero)
                    {
                        mrVertColorsFree(data_);
                        data_ = IntPtr.Zero;
                    }

                    disposed_ = true;
                }
            }

            ~VertColors()
            {
                Dispose(false);
            }
            /// returns list of colors
            unsafe public List<Color> ToList()
            {
                List<Color> colors = new List<Color>();
                MRVertColors* data = (MRVertColors*)data_;
                for (int i = 0; i < (int)data->size; i++)
                {
                    colors.Add(new Color(Marshal.PtrToStructure<Color.MRColor>(IntPtr.Add(data->data, i * sizeof(Color.MRColor)))));
                }
                return colors;
            }            
        }

    }
}