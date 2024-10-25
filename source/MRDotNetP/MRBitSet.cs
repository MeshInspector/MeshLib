using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    public abstract class BitSetReadOnly
    {
        /// gets total length of a bitset
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetSize( IntPtr bs );

        /// returns the number of bits in this bitset that are set
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetCount( IntPtr bs );

        /// checks if two bitsets are equal (have the same length and identical bit values)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool mrBitSetEq( IntPtr a, IntPtr b );

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool mrBitSetTest( IntPtr bs, UInt64 index );       

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetFindFirst( IntPtr bs );

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetFindLast( IntPtr bs );

        internal IntPtr bs_;

        public bool Test( int i )
        {
            var result = mrBitSetTest( bs_, (UInt64)i );
            return result;
        }
        public int FindFirst()
        {
            return (int)mrBitSetFindFirst( bs_ );
        }

        public int FindLast()
        {
            return (int)mrBitSetFindLast( bs_ );
        }        
        public int Size()
        {
            return (int)mrBitSetSize( bs_ );
        }
        public int Count()
        {
            return (int)mrBitSetCount( bs_ );
        }

        public abstract BitSetReadOnly Clone();

        public static bool operator ==(BitSetReadOnly a, BitSetReadOnly b )
        {
            return mrBitSetEq( a.bs_, b.bs_ );
        }

        public static bool operator !=(BitSetReadOnly a, BitSetReadOnly b )
        {
            return !mrBitSetEq( a.bs_, b.bs_ );
        }

        public override bool Equals( object obj )
        {
            return (obj is BitSetReadOnly) ? this == (BitSetReadOnly)obj : false;
        }

        public override int GetHashCode()
        {
            throw new NotImplementedException();
        }
    }

    public class BitSet : BitSetReadOnly, IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetCopy(IntPtr bs);
        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetNew(UInt64 numBits, bool fillValue);

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetSet(IntPtr bs, UInt64 index, bool value);

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetResize(IntPtr bs, UInt64 size, bool value);

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetAutoResizeSet(IntPtr bs, UInt64 pos, bool value);

        /// ...
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrBitSetSub(IntPtr a, IntPtr b);

        /// deallocates a BitSet object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetFree(IntPtr bs);

        public BitSet() : this( 0, false )
        {
        }

        internal BitSet(IntPtr bs)
        {
            bs_ = bs;
        }

        public BitSet( int size ) : this( size, false )
        {
        }

        public BitSet( int size, bool fillValue )
        {
            bs_ = mrBitSetNew((UInt64)size, fillValue);
            needDispose = true;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (needDispose)
            {
                if (bs_ != IntPtr.Zero)
                {
                    Console.WriteLine("mrBitSetFree start");
                    mrBitSetFree(bs_);
                    Console.WriteLine("mrBitSetFree end");
                    bs_ = IntPtr.Zero;
                }

                needDispose = false;
            }
        }

        ~BitSet()
        {
            Dispose(false);
        }
        public void Set(int index)
        {
            mrBitSetSet(bs_, (UInt64)index, true);
        }
        public void Set( int index, bool value )
        {
            mrBitSetSet( bs_, (UInt64)index, value );
        }

        public void Resize( int size )
        {
            mrBitSetResize( bs_, (UInt64)size, false );
        }

        public void Resize( int size, bool value )
        {
            mrBitSetResize( bs_, (UInt64)size, value );
        }

        public void AutoResizeSet( int pos )
        {
            mrBitSetAutoResizeSet( bs_, (UInt64)pos, true );
        }
        public void AutoResizeSet( int pos, bool value )
        {
            mrBitSetAutoResizeSet( bs_, (UInt64)pos, value );
        }

        public override BitSetReadOnly Clone()
        {
            IntPtr bsCopy = mrBitSetCopy( bs_ );
            return new BitSet(bsCopy);
        }

        public static BitSet operator -(BitSet a, BitSet b)
        {
            return new BitSet(mrBitSetSub(a.bs_, b.bs_));
        }

        bool needDispose = false;
    }
}
