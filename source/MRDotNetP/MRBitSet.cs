using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    public abstract class BitSetReadOnly
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRBitSet { };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern bool mrBitSetTest( ref MRBitSet bs, UInt64 i );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetFindFirst( ref MRBitSet bs );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetFindLast( ref MRBitSet bs );

        /*[DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetSet(ref MRBitSet bs, UInt64 i, bool val);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrBitSetAutoResizeSet( ref MRBitSet bs, UInt64 i, bool val);

        /// gets read-only access to the underlying blocks of a bitset
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

        /// gets count of the underlying blocks of a bitset
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern size_t mrBitSetBlocksNum( const MRBitSet* bs );*/

        /// gets total length of a bitset
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetSize( ref MRBitSet bs );

        /// returns the number of bits in this bitset that are set
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern UInt64 mrBitSetCount( ref MRBitSet bs );

        /// checks if two bitsets are equal (have the same length and identical bit values)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern bool mrBitSetEq( ref MRBitSet a, ref MRBitSet b );

        internal MRBitSet bs_;

        public bool Test( int i )
        {
            return mrBitSetTest( ref bs_, (UInt64)i );
        }
        public int FindFirst()
        {
            return (int)mrBitSetFindFirst( ref bs_ );
        }

        public int FindLast()
        {
            return (int)mrBitSetFindLast( ref bs_ );
        }        
        public int Size()
        {
            return (int)mrBitSetSize( ref bs_ );
        }
        public int Count()
        {
            return (int)mrBitSetCount( ref bs_ );
        }

        public abstract BitSetReadOnly Clone();

        public static bool operator ==(BitSetReadOnly a, BitSetReadOnly b )
        {
            return mrBitSetEq( ref a.bs_, ref b.bs_ );
        }

        public static bool operator !=(BitSetReadOnly a, BitSetReadOnly b )
        {
            return !mrBitSetEq( ref a.bs_, ref b.bs_ );
        }
    }
}
