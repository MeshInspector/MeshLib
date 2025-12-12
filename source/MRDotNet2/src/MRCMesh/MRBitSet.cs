public static partial class MR
{
    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::FaceBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_FaceBitSet : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_FaceBitSet>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FaceBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_Destroy(_Underlying *_this);
            __MR_FaceBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_FaceBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_FaceBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_FaceBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_FaceBitSet_Get_bits_per_block();
                return *__MR_FaceBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_FaceBitSet_Get_npos();
                return *__MR_FaceBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        public unsafe Const_FaceBitSet(MR._ByValue_FaceBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_FaceBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_FaceBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_FaceBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        public unsafe Const_FaceBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_FaceBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        public unsafe Const_FaceBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_FaceBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::test`.
        public unsafe bool Test(MR.FaceId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_test", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_test(_Underlying *_this, MR.FaceId n);
            return __MR_FaceBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::FaceBitSet::find_first`.
        public unsafe MR.FaceId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_find_first", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_find_first(_Underlying *_this);
            return __MR_FaceBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::find_next`.
        public unsafe MR.FaceId FindNext(MR.FaceId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_find_next", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_find_next(_Underlying *_this, MR.FaceId pos);
            return __MR_FaceBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::FaceBitSet::find_last`.
        public unsafe MR.FaceId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_find_last", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_find_last(_Underlying *_this);
            return __MR_FaceBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::FaceBitSet::nthSetBit`.
        public unsafe MR.FaceId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_FaceBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::FaceBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_FaceBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_is_subset_of(_Underlying *_this, MR.Const_FaceBitSet._Underlying *a);
            return __MR_FaceBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::FaceBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_FaceBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_intersects(_Underlying *_this, MR.Const_FaceBitSet._Underlying *a);
            return __MR_FaceBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FaceBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetMapping(MR.Const_FaceMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_getMapping_1_MR_FaceMap", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_getMapping_1_MR_FaceMap(_Underlying *_this, MR.Const_FaceMap._Underlying *map);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_FaceBitSet_getMapping_1_MR_FaceMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FaceBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetMapping(MR.Const_FaceBMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_getMapping_1_MR_FaceBMap", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_getMapping_1_MR_FaceBMap(_Underlying *_this, MR.Const_FaceBMap._Underlying *map);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_FaceBitSet_getMapping_1_MR_FaceBMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FaceBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_getMapping_1_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_getMapping_1_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId._Underlying *map);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_FaceBitSet_getMapping_1_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FaceBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetMapping(MR.Const_FaceMap map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_getMapping_2_MR_FaceMap", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_getMapping_2_MR_FaceMap(_Underlying *_this, MR.Const_FaceMap._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_FaceBitSet_getMapping_2_MR_FaceMap(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::FaceBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_getMapping_2_phmap_flat_hash_map_MR_FaceId_MR_FaceId", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_getMapping_2_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRFaceId_MRFaceId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_FaceBitSet_getMapping_2_phmap_flat_hash_map_MR_FaceId_MR_FaceId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::FaceBitSet::backId`.
        public unsafe MR.FaceId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_backId", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_backId(_Underlying *_this);
            return __MR_FaceBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::FaceBitSet::beginId`.
        public static MR.FaceId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_beginId", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_beginId();
            return __MR_FaceBitSet_beginId();
        }

        /// Generated from method `MR::FaceBitSet::endId`.
        public unsafe MR.FaceId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_endId", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceBitSet_endId(_Underlying *_this);
            return __MR_FaceBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::FaceBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_FaceBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_FaceBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FaceBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_empty(_Underlying *_this);
            return __MR_FaceBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FaceBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_FaceBitSet_size(_Underlying *_this);
            return __MR_FaceBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_FaceBitSet_num_blocks(_Underlying *_this);
            return __MR_FaceBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_FaceBitSet_capacity(_Underlying *_this);
            return __MR_FaceBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_FaceBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::FaceBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_FaceBitSet_bits(_Underlying *_this);
            return new(__MR_FaceBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::FaceBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_all", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_all(_Underlying *_this);
            return __MR_FaceBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::FaceBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_any", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_any(_Underlying *_this);
            return __MR_FaceBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::FaceBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_none", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_none(_Underlying *_this);
            return __MR_FaceBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::FaceBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_FaceBitSet_count(_Underlying *_this);
            return __MR_FaceBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::FaceBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_FaceBitSet_heapBytes(_Underlying *_this);
            return __MR_FaceBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> operator&(MR.Const_FaceBitSet a, MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_FaceBitSet", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_bitand_MR_FaceBitSet(MR.Const_FaceBitSet._Underlying *a, MR.Const_FaceBitSet._Underlying *b);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_bitand_MR_FaceBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> operator|(MR.Const_FaceBitSet a, MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_FaceBitSet", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_bitor_MR_FaceBitSet(MR.Const_FaceBitSet._Underlying *a, MR.Const_FaceBitSet._Underlying *b);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_bitor_MR_FaceBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> operator^(MR.Const_FaceBitSet a, MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_FaceBitSet", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_xor_MR_FaceBitSet(MR.Const_FaceBitSet._Underlying *a, MR.Const_FaceBitSet._Underlying *b);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_xor_MR_FaceBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> operator-(MR.Const_FaceBitSet a, MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_FaceBitSet", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_sub_MR_FaceBitSet(MR.Const_FaceBitSet._Underlying *a, MR.Const_FaceBitSet._Underlying *b);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_sub_MR_FaceBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator==<MR::FaceId>`.
        public static unsafe bool operator==(MR.Const_FaceBitSet a, MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_FaceBitSet", ExactSpelling = true)]
            extern static byte __MR_equal_MR_FaceBitSet(MR.Const_FaceBitSet._Underlying *a, MR.Const_FaceBitSet._Underlying *b);
            return __MR_equal_MR_FaceBitSet(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_FaceBitSet a, MR.Const_FaceBitSet b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_FaceBitSet? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_FaceBitSet)
                return this == (MR.Const_FaceBitSet)other;
            return false;
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::FaceBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class FaceBitSet : Const_FaceBitSet
    {
        internal unsafe FaceBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(FaceBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_FaceBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_FaceBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        public unsafe FaceBitSet(MR._ByValue_FaceBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_FaceBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe FaceBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_FaceBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        public unsafe FaceBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_FaceBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::FaceBitSet::FaceBitSet`.
        public unsafe FaceBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_FaceBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::operator=`.
        public unsafe MR.FaceBitSet Assign(MR._ByValue_FaceBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FaceBitSet._Underlying *_other);
            return new(__MR_FaceBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::set`.
        public unsafe MR.FaceBitSet Set(MR.FaceId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_set_3", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_set_3(_Underlying *_this, MR.FaceId n, ulong len, byte val);
            return new(__MR_FaceBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::set`.
        public unsafe MR.FaceBitSet Set(MR.FaceId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_set_2", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_set_2(_Underlying *_this, MR.FaceId n, byte val);
            return new(__MR_FaceBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::set`.
        public unsafe MR.FaceBitSet Set(MR.FaceId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_set_1", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_set_1(_Underlying *_this, MR.FaceId n);
            return new(__MR_FaceBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::set`.
        public unsafe MR.FaceBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_set_0", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_set_0(_Underlying *_this);
            return new(__MR_FaceBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::reset`.
        public unsafe MR.FaceBitSet Reset(MR.FaceId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_reset_2", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_reset_2(_Underlying *_this, MR.FaceId n, ulong len);
            return new(__MR_FaceBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::reset`.
        public unsafe MR.FaceBitSet Reset(MR.FaceId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_reset_1", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_reset_1(_Underlying *_this, MR.FaceId n);
            return new(__MR_FaceBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::reset`.
        public unsafe MR.FaceBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_reset_0", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_reset_0(_Underlying *_this);
            return new(__MR_FaceBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::flip`.
        public unsafe MR.FaceBitSet Flip(MR.FaceId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_flip_2", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_flip_2(_Underlying *_this, MR.FaceId n, ulong len);
            return new(__MR_FaceBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::flip`.
        public unsafe MR.FaceBitSet Flip(MR.FaceId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_flip_1", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_flip_1(_Underlying *_this, MR.FaceId n);
            return new(__MR_FaceBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::flip`.
        public unsafe MR.FaceBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_flip_0", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_flip_0(_Underlying *_this);
            return new(__MR_FaceBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.FaceId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_test_set(_Underlying *_this, MR.FaceId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_FaceBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::FaceBitSet::operator&=`.
        public unsafe MR.FaceBitSet BitandAssign(MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_bitand_assign(_Underlying *_this, MR.Const_FaceBitSet._Underlying *b);
            return new(__MR_FaceBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::operator|=`.
        public unsafe MR.FaceBitSet BitorAssign(MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_bitor_assign(_Underlying *_this, MR.Const_FaceBitSet._Underlying *b);
            return new(__MR_FaceBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::operator^=`.
        public unsafe MR.FaceBitSet XorAssign(MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_xor_assign(_Underlying *_this, MR.Const_FaceBitSet._Underlying *b);
            return new(__MR_FaceBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::operator-=`.
        public unsafe MR.FaceBitSet SubAssign(MR.Const_FaceBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_sub_assign(_Underlying *_this, MR.Const_FaceBitSet._Underlying *b);
            return new(__MR_FaceBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::FaceBitSet::subtract`.
        public unsafe MR.FaceBitSet Subtract(MR.Const_FaceBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_subtract", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_FaceBitSet_subtract(_Underlying *_this, MR.Const_FaceBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_FaceBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::FaceBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.FaceId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_autoResizeSet_3(_Underlying *_this, MR.FaceId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_FaceBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::FaceBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.FaceId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_autoResizeSet_2(_Underlying *_this, MR.FaceId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_FaceBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::FaceBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.FaceId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_autoResizeTestSet(_Underlying *_this, MR.FaceId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_FaceBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::FaceBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_FaceBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::FaceBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_resize", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_FaceBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::FaceBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_clear", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_clear(_Underlying *_this);
            __MR_FaceBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_shrink_to_fit(_Underlying *_this);
            __MR_FaceBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_FaceBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_FaceBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::FaceBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_reverse(_Underlying *_this);
            __MR_FaceBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::FaceBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_push_back(_Underlying *_this, byte val);
            __MR_FaceBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::FaceBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_pop_back(_Underlying *_this);
            __MR_FaceBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::FaceBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_FaceBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_FaceBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FaceBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FaceBitSet`/`Const_FaceBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FaceBitSet
    {
        internal readonly Const_FaceBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FaceBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FaceBitSet(Const_FaceBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FaceBitSet(Const_FaceBitSet arg) {return new(arg);}
        public _ByValue_FaceBitSet(MR.Misc._Moved<FaceBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FaceBitSet(MR.Misc._Moved<FaceBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FaceBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceBitSet`/`Const_FaceBitSet` directly.
    public class _InOptMut_FaceBitSet
    {
        public FaceBitSet? Opt;

        public _InOptMut_FaceBitSet() {}
        public _InOptMut_FaceBitSet(FaceBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_FaceBitSet(FaceBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `FaceBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceBitSet`/`Const_FaceBitSet` to pass it to the function.
    public class _InOptConst_FaceBitSet
    {
        public Const_FaceBitSet? Opt;

        public _InOptConst_FaceBitSet() {}
        public _InOptConst_FaceBitSet(Const_FaceBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_FaceBitSet(Const_FaceBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::VertBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_VertBitSet : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_VertBitSet>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_VertBitSet_Destroy(_Underlying *_this);
            __MR_VertBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_VertBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_VertBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_VertBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_VertBitSet_Get_bits_per_block();
                return *__MR_VertBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_VertBitSet_Get_npos();
                return *__MR_VertBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_VertBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        public unsafe Const_VertBitSet(MR._ByValue_VertBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_VertBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_VertBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_VertBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        public unsafe Const_VertBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VertBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        public unsafe Const_VertBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VertBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::test`.
        public unsafe bool Test(MR.VertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_test", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_test(_Underlying *_this, MR.VertId n);
            return __MR_VertBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::VertBitSet::find_first`.
        public unsafe MR.VertId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_find_first", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_find_first(_Underlying *_this);
            return __MR_VertBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::find_next`.
        public unsafe MR.VertId FindNext(MR.VertId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_find_next", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_find_next(_Underlying *_this, MR.VertId pos);
            return __MR_VertBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::VertBitSet::find_last`.
        public unsafe MR.VertId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_find_last", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_find_last(_Underlying *_this);
            return __MR_VertBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::VertBitSet::nthSetBit`.
        public unsafe MR.VertId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_VertBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::VertBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_VertBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_is_subset_of(_Underlying *_this, MR.Const_VertBitSet._Underlying *a);
            return __MR_VertBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::VertBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_VertBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_intersects(_Underlying *_this, MR.Const_VertBitSet._Underlying *a);
            return __MR_VertBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetMapping(MR.Const_VertMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_getMapping_1_MR_VertMap", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_getMapping_1_MR_VertMap(_Underlying *_this, MR.Const_VertMap._Underlying *map);
            return MR.Misc.Move(new MR.VertBitSet(__MR_VertBitSet_getMapping_1_MR_VertMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetMapping(MR.Const_VertBMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_getMapping_1_MR_VertBMap", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_getMapping_1_MR_VertBMap(_Underlying *_this, MR.Const_VertBMap._Underlying *map);
            return MR.Misc.Move(new MR.VertBitSet(__MR_VertBitSet_getMapping_1_MR_VertBMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_getMapping_1_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_getMapping_1_phmap_flat_hash_map_MR_VertId_MR_VertId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId._Underlying *map);
            return MR.Misc.Move(new MR.VertBitSet(__MR_VertBitSet_getMapping_1_phmap_flat_hash_map_MR_VertId_MR_VertId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetMapping(MR.Const_VertMap map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_getMapping_2_MR_VertMap", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_getMapping_2_MR_VertMap(_Underlying *_this, MR.Const_VertMap._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.VertBitSet(__MR_VertBitSet_getMapping_2_MR_VertMap(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::VertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_getMapping_2_phmap_flat_hash_map_MR_VertId_MR_VertId", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_getMapping_2_phmap_flat_hash_map_MR_VertId_MR_VertId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRVertId_MRVertId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.VertBitSet(__MR_VertBitSet_getMapping_2_phmap_flat_hash_map_MR_VertId_MR_VertId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::VertBitSet::backId`.
        public unsafe MR.VertId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_backId", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_backId(_Underlying *_this);
            return __MR_VertBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::VertBitSet::beginId`.
        public static MR.VertId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_beginId", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_beginId();
            return __MR_VertBitSet_beginId();
        }

        /// Generated from method `MR::VertBitSet::endId`.
        public unsafe MR.VertId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_endId", ExactSpelling = true)]
            extern static MR.VertId __MR_VertBitSet_endId(_Underlying *_this);
            return __MR_VertBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::VertBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_VertBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_VertBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VertBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_empty(_Underlying *_this);
            return __MR_VertBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VertBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_VertBitSet_size(_Underlying *_this);
            return __MR_VertBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_VertBitSet_num_blocks(_Underlying *_this);
            return __MR_VertBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_VertBitSet_capacity(_Underlying *_this);
            return __MR_VertBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_VertBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::VertBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_VertBitSet_bits(_Underlying *_this);
            return new(__MR_VertBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::VertBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_all", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_all(_Underlying *_this);
            return __MR_VertBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::VertBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_any", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_any(_Underlying *_this);
            return __MR_VertBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::VertBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_none", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_none(_Underlying *_this);
            return __MR_VertBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::VertBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_VertBitSet_count(_Underlying *_this);
            return __MR_VertBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::VertBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_VertBitSet_heapBytes(_Underlying *_this);
            return __MR_VertBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> operator&(MR.Const_VertBitSet a, MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_VertBitSet", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_bitand_MR_VertBitSet(MR.Const_VertBitSet._Underlying *a, MR.Const_VertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VertBitSet(__MR_bitand_MR_VertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> operator|(MR.Const_VertBitSet a, MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_VertBitSet", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_bitor_MR_VertBitSet(MR.Const_VertBitSet._Underlying *a, MR.Const_VertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VertBitSet(__MR_bitor_MR_VertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> operator^(MR.Const_VertBitSet a, MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_VertBitSet", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_xor_MR_VertBitSet(MR.Const_VertBitSet._Underlying *a, MR.Const_VertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VertBitSet(__MR_xor_MR_VertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> operator-(MR.Const_VertBitSet a, MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VertBitSet", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_sub_MR_VertBitSet(MR.Const_VertBitSet._Underlying *a, MR.Const_VertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VertBitSet(__MR_sub_MR_VertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator==<MR::VertId>`.
        public static unsafe bool operator==(MR.Const_VertBitSet a, MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VertBitSet", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VertBitSet(MR.Const_VertBitSet._Underlying *a, MR.Const_VertBitSet._Underlying *b);
            return __MR_equal_MR_VertBitSet(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_VertBitSet a, MR.Const_VertBitSet b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_VertBitSet? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_VertBitSet)
                return this == (MR.Const_VertBitSet)other;
            return false;
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::VertBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class VertBitSet : Const_VertBitSet
    {
        internal unsafe VertBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(VertBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_VertBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_VertBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_VertBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        public unsafe VertBitSet(MR._ByValue_VertBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_VertBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe VertBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_VertBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        public unsafe VertBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VertBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VertBitSet::VertBitSet`.
        public unsafe VertBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VertBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::operator=`.
        public unsafe MR.VertBitSet Assign(MR._ByValue_VertBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VertBitSet._Underlying *_other);
            return new(__MR_VertBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::set`.
        public unsafe MR.VertBitSet Set(MR.VertId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_set_3", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_set_3(_Underlying *_this, MR.VertId n, ulong len, byte val);
            return new(__MR_VertBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::set`.
        public unsafe MR.VertBitSet Set(MR.VertId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_set_2", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_set_2(_Underlying *_this, MR.VertId n, byte val);
            return new(__MR_VertBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::set`.
        public unsafe MR.VertBitSet Set(MR.VertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_set_1", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_set_1(_Underlying *_this, MR.VertId n);
            return new(__MR_VertBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::set`.
        public unsafe MR.VertBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_set_0", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_set_0(_Underlying *_this);
            return new(__MR_VertBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::reset`.
        public unsafe MR.VertBitSet Reset(MR.VertId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_reset_2", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_reset_2(_Underlying *_this, MR.VertId n, ulong len);
            return new(__MR_VertBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::reset`.
        public unsafe MR.VertBitSet Reset(MR.VertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_reset_1", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_reset_1(_Underlying *_this, MR.VertId n);
            return new(__MR_VertBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::reset`.
        public unsafe MR.VertBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_reset_0", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_reset_0(_Underlying *_this);
            return new(__MR_VertBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::flip`.
        public unsafe MR.VertBitSet Flip(MR.VertId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_flip_2", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_flip_2(_Underlying *_this, MR.VertId n, ulong len);
            return new(__MR_VertBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::flip`.
        public unsafe MR.VertBitSet Flip(MR.VertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_flip_1", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_flip_1(_Underlying *_this, MR.VertId n);
            return new(__MR_VertBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::flip`.
        public unsafe MR.VertBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_flip_0", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_flip_0(_Underlying *_this);
            return new(__MR_VertBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.VertId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_test_set(_Underlying *_this, MR.VertId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VertBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::VertBitSet::operator&=`.
        public unsafe MR.VertBitSet BitandAssign(MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_bitand_assign(_Underlying *_this, MR.Const_VertBitSet._Underlying *b);
            return new(__MR_VertBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::operator|=`.
        public unsafe MR.VertBitSet BitorAssign(MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_bitor_assign(_Underlying *_this, MR.Const_VertBitSet._Underlying *b);
            return new(__MR_VertBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::operator^=`.
        public unsafe MR.VertBitSet XorAssign(MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_xor_assign(_Underlying *_this, MR.Const_VertBitSet._Underlying *b);
            return new(__MR_VertBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::operator-=`.
        public unsafe MR.VertBitSet SubAssign(MR.Const_VertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_sub_assign(_Underlying *_this, MR.Const_VertBitSet._Underlying *b);
            return new(__MR_VertBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::VertBitSet::subtract`.
        public unsafe MR.VertBitSet Subtract(MR.Const_VertBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_subtract", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_VertBitSet_subtract(_Underlying *_this, MR.Const_VertBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_VertBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::VertBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.VertId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_VertBitSet_autoResizeSet_3(_Underlying *_this, MR.VertId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VertBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::VertBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.VertId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_VertBitSet_autoResizeSet_2(_Underlying *_this, MR.VertId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VertBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::VertBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.VertId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_autoResizeTestSet(_Underlying *_this, MR.VertId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VertBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::VertBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_VertBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_VertBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::VertBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_resize", ExactSpelling = true)]
            extern static void __MR_VertBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VertBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::VertBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_clear", ExactSpelling = true)]
            extern static void __MR_VertBitSet_clear(_Underlying *_this);
            __MR_VertBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_VertBitSet_shrink_to_fit(_Underlying *_this);
            __MR_VertBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_VertBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VertBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::VertBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_VertBitSet_reverse(_Underlying *_this);
            __MR_VertBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::VertBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_VertBitSet_push_back(_Underlying *_this, byte val);
            __MR_VertBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::VertBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_VertBitSet_pop_back(_Underlying *_this);
            __MR_VertBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::VertBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_VertBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_VertBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VertBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VertBitSet`/`Const_VertBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VertBitSet
    {
        internal readonly Const_VertBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VertBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VertBitSet(Const_VertBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VertBitSet(Const_VertBitSet arg) {return new(arg);}
        public _ByValue_VertBitSet(MR.Misc._Moved<VertBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VertBitSet(MR.Misc._Moved<VertBitSet> arg) {return new(arg);}
    }

    /// This is used as a function parameter when the underlying function receives an optional `VertBitSet` by value,
    ///   and also has a default argument, meaning it has two different null states.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VertBitSet`/`Const_VertBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument.
    /// * Pass `MR.Misc.NullOptType` to pass no object.
    public class _ByValueOptOpt_VertBitSet
    {
        internal readonly Const_VertBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValueOptOpt_VertBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValueOptOpt_VertBitSet(Const_VertBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValueOptOpt_VertBitSet(Const_VertBitSet arg) {return new(arg);}
        public _ByValueOptOpt_VertBitSet(MR.Misc._Moved<VertBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValueOptOpt_VertBitSet(MR.Misc._Moved<VertBitSet> arg) {return new(arg);}
        public _ByValueOptOpt_VertBitSet(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
        public static implicit operator _ByValueOptOpt_VertBitSet(MR.Misc.NullOptType nullopt) {return new(nullopt);}
    }

    /// This is used for optional parameters of class `VertBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertBitSet`/`Const_VertBitSet` directly.
    public class _InOptMut_VertBitSet
    {
        public VertBitSet? Opt;

        public _InOptMut_VertBitSet() {}
        public _InOptMut_VertBitSet(VertBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_VertBitSet(VertBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertBitSet`/`Const_VertBitSet` to pass it to the function.
    public class _InOptConst_VertBitSet
    {
        public Const_VertBitSet? Opt;

        public _InOptConst_VertBitSet() {}
        public _InOptConst_VertBitSet(Const_VertBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_VertBitSet(Const_VertBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::EdgeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_EdgeBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgeBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_Destroy(_Underlying *_this);
            __MR_EdgeBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgeBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_EdgeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_EdgeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_EdgeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_EdgeBitSet_Get_bits_per_block();
                return *__MR_EdgeBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_EdgeBitSet_Get_npos();
                return *__MR_EdgeBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        public unsafe Const_EdgeBitSet(MR._ByValue_EdgeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_EdgeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_EdgeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        public unsafe Const_EdgeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_EdgeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        public unsafe Const_EdgeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_EdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::test`.
        public unsafe bool Test(MR.EdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_test", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_test(_Underlying *_this, MR.EdgeId n);
            return __MR_EdgeBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::EdgeBitSet::find_first`.
        public unsafe MR.EdgeId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_find_first", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_find_first(_Underlying *_this);
            return __MR_EdgeBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::find_next`.
        public unsafe MR.EdgeId FindNext(MR.EdgeId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_find_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_find_next(_Underlying *_this, MR.EdgeId pos);
            return __MR_EdgeBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::EdgeBitSet::find_last`.
        public unsafe MR.EdgeId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_find_last", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_find_last(_Underlying *_this);
            return __MR_EdgeBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::EdgeBitSet::nthSetBit`.
        public unsafe MR.EdgeId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_EdgeBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::EdgeBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_EdgeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_is_subset_of(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *a);
            return __MR_EdgeBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::EdgeBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_EdgeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_intersects(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *a);
            return __MR_EdgeBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::EdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> GetMapping(MR.Const_EdgeMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_getMapping_1_MR_EdgeMap", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_getMapping_1_MR_EdgeMap(_Underlying *_this, MR.Const_EdgeMap._Underlying *map);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_EdgeBitSet_getMapping_1_MR_EdgeMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::EdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> GetMapping(MR.Const_EdgeBMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_getMapping_1_MR_EdgeBMap", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_getMapping_1_MR_EdgeBMap(_Underlying *_this, MR.Const_EdgeBMap._Underlying *map);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_EdgeBitSet_getMapping_1_MR_EdgeBMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::EdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId._Underlying *map);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_EdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::EdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> GetMapping(MR.Const_EdgeMap map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_getMapping_2_MR_EdgeMap", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_getMapping_2_MR_EdgeMap(_Underlying *_this, MR.Const_EdgeMap._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_EdgeBitSet_getMapping_2_MR_EdgeMap(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::EdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MREdgeId_MREdgeId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_EdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_EdgeId_MR_EdgeId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::EdgeBitSet::backId`.
        public unsafe MR.EdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_backId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_backId(_Underlying *_this);
            return __MR_EdgeBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::EdgeBitSet::beginId`.
        public static MR.EdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_beginId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_beginId();
            return __MR_EdgeBitSet_beginId();
        }

        /// Generated from method `MR::EdgeBitSet::endId`.
        public unsafe MR.EdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_endId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeBitSet_endId(_Underlying *_this);
            return __MR_EdgeBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::EdgeBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_EdgeBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_EdgeBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::EdgeBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_empty(_Underlying *_this);
            return __MR_EdgeBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::EdgeBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_EdgeBitSet_size(_Underlying *_this);
            return __MR_EdgeBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_EdgeBitSet_num_blocks(_Underlying *_this);
            return __MR_EdgeBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_EdgeBitSet_capacity(_Underlying *_this);
            return __MR_EdgeBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_EdgeBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::EdgeBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_EdgeBitSet_bits(_Underlying *_this);
            return new(__MR_EdgeBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::EdgeBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_all", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_all(_Underlying *_this);
            return __MR_EdgeBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::EdgeBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_any", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_any(_Underlying *_this);
            return __MR_EdgeBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::EdgeBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_none", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_none(_Underlying *_this);
            return __MR_EdgeBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::EdgeBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_EdgeBitSet_count(_Underlying *_this);
            return __MR_EdgeBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::EdgeBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_EdgeBitSet_heapBytes(_Underlying *_this);
            return __MR_EdgeBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.EdgeBitSet> operator&(MR.Const_EdgeBitSet a, MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_EdgeBitSet", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_bitand_MR_EdgeBitSet(MR.Const_EdgeBitSet._Underlying *a, MR.Const_EdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_bitand_MR_EdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.EdgeBitSet> operator|(MR.Const_EdgeBitSet a, MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_EdgeBitSet", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_bitor_MR_EdgeBitSet(MR.Const_EdgeBitSet._Underlying *a, MR.Const_EdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_bitor_MR_EdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.EdgeBitSet> operator^(MR.Const_EdgeBitSet a, MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_EdgeBitSet", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_xor_MR_EdgeBitSet(MR.Const_EdgeBitSet._Underlying *a, MR.Const_EdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_xor_MR_EdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.EdgeBitSet> operator-(MR.Const_EdgeBitSet a, MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_EdgeBitSet", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_sub_MR_EdgeBitSet(MR.Const_EdgeBitSet._Underlying *a, MR.Const_EdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_sub_MR_EdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::EdgeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class EdgeBitSet : Const_EdgeBitSet
    {
        internal unsafe EdgeBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(EdgeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_EdgeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_EdgeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        public unsafe EdgeBitSet(MR._ByValue_EdgeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe EdgeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_EdgeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        public unsafe EdgeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_EdgeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::EdgeBitSet::EdgeBitSet`.
        public unsafe EdgeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_EdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::operator=`.
        public unsafe MR.EdgeBitSet Assign(MR._ByValue_EdgeBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.EdgeBitSet._Underlying *_other);
            return new(__MR_EdgeBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::set`.
        public unsafe MR.EdgeBitSet Set(MR.EdgeId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_set_3", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_set_3(_Underlying *_this, MR.EdgeId n, ulong len, byte val);
            return new(__MR_EdgeBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::set`.
        public unsafe MR.EdgeBitSet Set(MR.EdgeId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_set_2", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_set_2(_Underlying *_this, MR.EdgeId n, byte val);
            return new(__MR_EdgeBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::set`.
        public unsafe MR.EdgeBitSet Set(MR.EdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_set_1", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_set_1(_Underlying *_this, MR.EdgeId n);
            return new(__MR_EdgeBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::set`.
        public unsafe MR.EdgeBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_set_0", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_set_0(_Underlying *_this);
            return new(__MR_EdgeBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::reset`.
        public unsafe MR.EdgeBitSet Reset(MR.EdgeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_reset_2", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_reset_2(_Underlying *_this, MR.EdgeId n, ulong len);
            return new(__MR_EdgeBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::reset`.
        public unsafe MR.EdgeBitSet Reset(MR.EdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_reset_1", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_reset_1(_Underlying *_this, MR.EdgeId n);
            return new(__MR_EdgeBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::reset`.
        public unsafe MR.EdgeBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_reset_0", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_reset_0(_Underlying *_this);
            return new(__MR_EdgeBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::flip`.
        public unsafe MR.EdgeBitSet Flip(MR.EdgeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_flip_2", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_flip_2(_Underlying *_this, MR.EdgeId n, ulong len);
            return new(__MR_EdgeBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::flip`.
        public unsafe MR.EdgeBitSet Flip(MR.EdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_flip_1", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_flip_1(_Underlying *_this, MR.EdgeId n);
            return new(__MR_EdgeBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::flip`.
        public unsafe MR.EdgeBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_flip_0", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_flip_0(_Underlying *_this);
            return new(__MR_EdgeBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.EdgeId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_test_set(_Underlying *_this, MR.EdgeId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_EdgeBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::EdgeBitSet::operator&=`.
        public unsafe MR.EdgeBitSet BitandAssign(MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_bitand_assign(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *b);
            return new(__MR_EdgeBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::operator|=`.
        public unsafe MR.EdgeBitSet BitorAssign(MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_bitor_assign(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *b);
            return new(__MR_EdgeBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::operator^=`.
        public unsafe MR.EdgeBitSet XorAssign(MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_xor_assign(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *b);
            return new(__MR_EdgeBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::operator-=`.
        public unsafe MR.EdgeBitSet SubAssign(MR.Const_EdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_sub_assign(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *b);
            return new(__MR_EdgeBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::EdgeBitSet::subtract`.
        public unsafe MR.EdgeBitSet Subtract(MR.Const_EdgeBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_subtract", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_EdgeBitSet_subtract(_Underlying *_this, MR.Const_EdgeBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_EdgeBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::EdgeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.EdgeId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_autoResizeSet_3(_Underlying *_this, MR.EdgeId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_EdgeBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::EdgeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.EdgeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_autoResizeSet_2(_Underlying *_this, MR.EdgeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_EdgeBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::EdgeBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.EdgeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_autoResizeTestSet(_Underlying *_this, MR.EdgeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_EdgeBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::EdgeBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_EdgeBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::EdgeBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_resize", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_EdgeBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::EdgeBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_clear", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_clear(_Underlying *_this);
            __MR_EdgeBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_shrink_to_fit(_Underlying *_this);
            __MR_EdgeBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_EdgeBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_EdgeBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::EdgeBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_reverse(_Underlying *_this);
            __MR_EdgeBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::EdgeBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_push_back(_Underlying *_this, byte val);
            __MR_EdgeBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::EdgeBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_pop_back(_Underlying *_this);
            __MR_EdgeBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::EdgeBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_EdgeBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_EdgeBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EdgeBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EdgeBitSet`/`Const_EdgeBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EdgeBitSet
    {
        internal readonly Const_EdgeBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EdgeBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_EdgeBitSet(Const_EdgeBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EdgeBitSet(Const_EdgeBitSet arg) {return new(arg);}
        public _ByValue_EdgeBitSet(MR.Misc._Moved<EdgeBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EdgeBitSet(MR.Misc._Moved<EdgeBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EdgeBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeBitSet`/`Const_EdgeBitSet` directly.
    public class _InOptMut_EdgeBitSet
    {
        public EdgeBitSet? Opt;

        public _InOptMut_EdgeBitSet() {}
        public _InOptMut_EdgeBitSet(EdgeBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_EdgeBitSet(EdgeBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgeBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeBitSet`/`Const_EdgeBitSet` to pass it to the function.
    public class _InOptConst_EdgeBitSet
    {
        public Const_EdgeBitSet? Opt;

        public _InOptConst_EdgeBitSet() {}
        public _InOptConst_EdgeBitSet(Const_EdgeBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_EdgeBitSet(Const_EdgeBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::UndirectedEdgeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_UndirectedEdgeBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UndirectedEdgeBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_Destroy(_Underlying *_this);
            __MR_UndirectedEdgeBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UndirectedEdgeBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_UndirectedEdgeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_UndirectedEdgeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_UndirectedEdgeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_UndirectedEdgeBitSet_Get_bits_per_block();
                return *__MR_UndirectedEdgeBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_UndirectedEdgeBitSet_Get_npos();
                return *__MR_UndirectedEdgeBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UndirectedEdgeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        public unsafe Const_UndirectedEdgeBitSet(MR._ByValue_UndirectedEdgeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UndirectedEdgeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_UndirectedEdgeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        public unsafe Const_UndirectedEdgeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        public unsafe Const_UndirectedEdgeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::test`.
        public unsafe bool Test(MR.UndirectedEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_test", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_test(_Underlying *_this, MR.UndirectedEdgeId n);
            return __MR_UndirectedEdgeBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::find_first`.
        public unsafe MR.UndirectedEdgeId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_find_first", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_find_first(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::find_next`.
        public unsafe MR.UndirectedEdgeId FindNext(MR.UndirectedEdgeId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_find_next", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_find_next(_Underlying *_this, MR.UndirectedEdgeId pos);
            return __MR_UndirectedEdgeBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::find_last`.
        public unsafe MR.UndirectedEdgeId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_find_last", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_find_last(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::UndirectedEdgeBitSet::nthSetBit`.
        public unsafe MR.UndirectedEdgeId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_UndirectedEdgeBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::UndirectedEdgeBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_UndirectedEdgeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_is_subset_of(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *a);
            return __MR_UndirectedEdgeBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::UndirectedEdgeBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_UndirectedEdgeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_intersects(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *a);
            return __MR_UndirectedEdgeBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetMapping(MR.Const_UndirectedEdgeMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_getMapping_1_MR_UndirectedEdgeMap", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_getMapping_1_MR_UndirectedEdgeMap(_Underlying *_this, MR.Const_UndirectedEdgeMap._Underlying *map);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_UndirectedEdgeBitSet_getMapping_1_MR_UndirectedEdgeMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetMapping(MR.Const_UndirectedEdgeBMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_getMapping_1_MR_UndirectedEdgeBMap", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_getMapping_1_MR_UndirectedEdgeBMap(_Underlying *_this, MR.Const_UndirectedEdgeBMap._Underlying *map);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_UndirectedEdgeBitSet_getMapping_1_MR_UndirectedEdgeBMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *map);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_UndirectedEdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetMapping(MR.Const_UndirectedEdgeMap map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_getMapping_2_MR_UndirectedEdgeMap", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_getMapping_2_MR_UndirectedEdgeMap(_Underlying *_this, MR.Const_UndirectedEdgeMap._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_UndirectedEdgeBitSet_getMapping_2_MR_UndirectedEdgeMap(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_UndirectedEdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_UndirectedEdgeId_MR_UndirectedEdgeId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::UndirectedEdgeBitSet::backId`.
        public unsafe MR.UndirectedEdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_backId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_backId(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::UndirectedEdgeBitSet::beginId`.
        public static MR.UndirectedEdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_beginId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_beginId();
            return __MR_UndirectedEdgeBitSet_beginId();
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::endId`.
        public unsafe MR.UndirectedEdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_endId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeBitSet_endId(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::UndirectedEdgeBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_UndirectedEdgeBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_UndirectedEdgeBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_empty(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_UndirectedEdgeBitSet_size(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_UndirectedEdgeBitSet_num_blocks(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_UndirectedEdgeBitSet_capacity(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_UndirectedEdgeBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::UndirectedEdgeBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_UndirectedEdgeBitSet_bits(_Underlying *_this);
            return new(__MR_UndirectedEdgeBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::UndirectedEdgeBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_all", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_all(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::UndirectedEdgeBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_any", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_any(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::UndirectedEdgeBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_none", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_none(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::UndirectedEdgeBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_UndirectedEdgeBitSet_count(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::UndirectedEdgeBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_UndirectedEdgeBitSet_heapBytes(_Underlying *_this);
            return __MR_UndirectedEdgeBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> operator&(MR.Const_UndirectedEdgeBitSet a, MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_bitand_MR_UndirectedEdgeBitSet(MR.Const_UndirectedEdgeBitSet._Underlying *a, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_bitand_MR_UndirectedEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> operator|(MR.Const_UndirectedEdgeBitSet a, MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_bitor_MR_UndirectedEdgeBitSet(MR.Const_UndirectedEdgeBitSet._Underlying *a, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_bitor_MR_UndirectedEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> operator^(MR.Const_UndirectedEdgeBitSet a, MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_xor_MR_UndirectedEdgeBitSet(MR.Const_UndirectedEdgeBitSet._Underlying *a, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_xor_MR_UndirectedEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> operator-(MR.Const_UndirectedEdgeBitSet a, MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_sub_MR_UndirectedEdgeBitSet(MR.Const_UndirectedEdgeBitSet._Underlying *a, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_sub_MR_UndirectedEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::UndirectedEdgeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class UndirectedEdgeBitSet : Const_UndirectedEdgeBitSet
    {
        internal unsafe UndirectedEdgeBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(UndirectedEdgeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_UndirectedEdgeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_UndirectedEdgeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe UndirectedEdgeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        public unsafe UndirectedEdgeBitSet(MR._ByValue_UndirectedEdgeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UndirectedEdgeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe UndirectedEdgeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        public unsafe UndirectedEdgeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::UndirectedEdgeBitSet::UndirectedEdgeBitSet`.
        public unsafe UndirectedEdgeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_UndirectedEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::operator=`.
        public unsafe MR.UndirectedEdgeBitSet Assign(MR._ByValue_UndirectedEdgeBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UndirectedEdgeBitSet._Underlying *_other);
            return new(__MR_UndirectedEdgeBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::set`.
        public unsafe MR.UndirectedEdgeBitSet Set(MR.UndirectedEdgeId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_set_3", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_set_3(_Underlying *_this, MR.UndirectedEdgeId n, ulong len, byte val);
            return new(__MR_UndirectedEdgeBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::set`.
        public unsafe MR.UndirectedEdgeBitSet Set(MR.UndirectedEdgeId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_set_2", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_set_2(_Underlying *_this, MR.UndirectedEdgeId n, byte val);
            return new(__MR_UndirectedEdgeBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::set`.
        public unsafe MR.UndirectedEdgeBitSet Set(MR.UndirectedEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_set_1", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_set_1(_Underlying *_this, MR.UndirectedEdgeId n);
            return new(__MR_UndirectedEdgeBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::set`.
        public unsafe MR.UndirectedEdgeBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_set_0", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_set_0(_Underlying *_this);
            return new(__MR_UndirectedEdgeBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::reset`.
        public unsafe MR.UndirectedEdgeBitSet Reset(MR.UndirectedEdgeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_reset_2", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_reset_2(_Underlying *_this, MR.UndirectedEdgeId n, ulong len);
            return new(__MR_UndirectedEdgeBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::reset`.
        public unsafe MR.UndirectedEdgeBitSet Reset(MR.UndirectedEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_reset_1", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_reset_1(_Underlying *_this, MR.UndirectedEdgeId n);
            return new(__MR_UndirectedEdgeBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::reset`.
        public unsafe MR.UndirectedEdgeBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_reset_0", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_reset_0(_Underlying *_this);
            return new(__MR_UndirectedEdgeBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::flip`.
        public unsafe MR.UndirectedEdgeBitSet Flip(MR.UndirectedEdgeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_flip_2", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_flip_2(_Underlying *_this, MR.UndirectedEdgeId n, ulong len);
            return new(__MR_UndirectedEdgeBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::flip`.
        public unsafe MR.UndirectedEdgeBitSet Flip(MR.UndirectedEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_flip_1", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_flip_1(_Underlying *_this, MR.UndirectedEdgeId n);
            return new(__MR_UndirectedEdgeBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::flip`.
        public unsafe MR.UndirectedEdgeBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_flip_0", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_flip_0(_Underlying *_this);
            return new(__MR_UndirectedEdgeBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.UndirectedEdgeId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_test_set(_Underlying *_this, MR.UndirectedEdgeId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_UndirectedEdgeBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::operator&=`.
        public unsafe MR.UndirectedEdgeBitSet BitandAssign(MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_bitand_assign(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return new(__MR_UndirectedEdgeBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::operator|=`.
        public unsafe MR.UndirectedEdgeBitSet BitorAssign(MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_bitor_assign(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return new(__MR_UndirectedEdgeBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::operator^=`.
        public unsafe MR.UndirectedEdgeBitSet XorAssign(MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_xor_assign(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return new(__MR_UndirectedEdgeBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::operator-=`.
        public unsafe MR.UndirectedEdgeBitSet SubAssign(MR.Const_UndirectedEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_sub_assign(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *b);
            return new(__MR_UndirectedEdgeBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::UndirectedEdgeBitSet::subtract`.
        public unsafe MR.UndirectedEdgeBitSet Subtract(MR.Const_UndirectedEdgeBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_subtract", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_UndirectedEdgeBitSet_subtract(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_UndirectedEdgeBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.UndirectedEdgeId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_autoResizeSet_3(_Underlying *_this, MR.UndirectedEdgeId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_UndirectedEdgeBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.UndirectedEdgeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_autoResizeSet_2(_Underlying *_this, MR.UndirectedEdgeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_UndirectedEdgeBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.UndirectedEdgeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_autoResizeTestSet(_Underlying *_this, MR.UndirectedEdgeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_UndirectedEdgeBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_UndirectedEdgeBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_resize", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_UndirectedEdgeBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_clear", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_clear(_Underlying *_this);
            __MR_UndirectedEdgeBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_shrink_to_fit(_Underlying *_this);
            __MR_UndirectedEdgeBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_UndirectedEdgeBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::UndirectedEdgeBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_reverse(_Underlying *_this);
            __MR_UndirectedEdgeBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::UndirectedEdgeBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_push_back(_Underlying *_this, byte val);
            __MR_UndirectedEdgeBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::UndirectedEdgeBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_pop_back(_Underlying *_this);
            __MR_UndirectedEdgeBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::UndirectedEdgeBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_UndirectedEdgeBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UndirectedEdgeBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UndirectedEdgeBitSet`/`Const_UndirectedEdgeBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UndirectedEdgeBitSet
    {
        internal readonly Const_UndirectedEdgeBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UndirectedEdgeBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UndirectedEdgeBitSet(Const_UndirectedEdgeBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UndirectedEdgeBitSet(Const_UndirectedEdgeBitSet arg) {return new(arg);}
        public _ByValue_UndirectedEdgeBitSet(MR.Misc._Moved<UndirectedEdgeBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UndirectedEdgeBitSet(MR.Misc._Moved<UndirectedEdgeBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UndirectedEdgeBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UndirectedEdgeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeBitSet`/`Const_UndirectedEdgeBitSet` directly.
    public class _InOptMut_UndirectedEdgeBitSet
    {
        public UndirectedEdgeBitSet? Opt;

        public _InOptMut_UndirectedEdgeBitSet() {}
        public _InOptMut_UndirectedEdgeBitSet(UndirectedEdgeBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_UndirectedEdgeBitSet(UndirectedEdgeBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `UndirectedEdgeBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UndirectedEdgeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeBitSet`/`Const_UndirectedEdgeBitSet` to pass it to the function.
    public class _InOptConst_UndirectedEdgeBitSet
    {
        public Const_UndirectedEdgeBitSet? Opt;

        public _InOptConst_UndirectedEdgeBitSet() {}
        public _InOptConst_UndirectedEdgeBitSet(Const_UndirectedEdgeBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_UndirectedEdgeBitSet(Const_UndirectedEdgeBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::PixelBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_PixelBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PixelBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_Destroy(_Underlying *_this);
            __MR_PixelBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PixelBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_PixelBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_PixelBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_PixelBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_PixelBitSet_Get_bits_per_block();
                return *__MR_PixelBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_PixelBitSet_Get_npos();
                return *__MR_PixelBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PixelBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_PixelBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        public unsafe Const_PixelBitSet(MR._ByValue_PixelBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PixelBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_PixelBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_PixelBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_PixelBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        public unsafe Const_PixelBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_PixelBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        public unsafe Const_PixelBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_PixelBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::test`.
        public unsafe bool Test(MR.PixelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_test", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_test(_Underlying *_this, MR.PixelId n);
            return __MR_PixelBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::PixelBitSet::find_first`.
        public unsafe MR.PixelId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_find_first", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_find_first(_Underlying *_this);
            return __MR_PixelBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::find_next`.
        public unsafe MR.PixelId FindNext(MR.PixelId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_find_next", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_find_next(_Underlying *_this, MR.PixelId pos);
            return __MR_PixelBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::PixelBitSet::find_last`.
        public unsafe MR.PixelId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_find_last", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_find_last(_Underlying *_this);
            return __MR_PixelBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::PixelBitSet::nthSetBit`.
        public unsafe MR.PixelId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_PixelBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::PixelBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_PixelBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_is_subset_of(_Underlying *_this, MR.Const_PixelBitSet._Underlying *a);
            return __MR_PixelBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::PixelBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_PixelBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_intersects(_Underlying *_this, MR.Const_PixelBitSet._Underlying *a);
            return __MR_PixelBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PixelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.PixelBitSet> GetMapping(MR.Const_Vector_MRPixelId_MRPixelId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_getMapping_1_MR_Vector_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_getMapping_1_MR_Vector_MR_PixelId_MR_PixelId(_Underlying *_this, MR.Const_Vector_MRPixelId_MRPixelId._Underlying *map);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_PixelBitSet_getMapping_1_MR_Vector_MR_PixelId_MR_PixelId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PixelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.PixelBitSet> GetMapping(MR.Const_BMap_MRPixelId_MRPixelId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_getMapping_1_MR_BMap_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_getMapping_1_MR_BMap_MR_PixelId_MR_PixelId(_Underlying *_this, MR.Const_BMap_MRPixelId_MRPixelId._Underlying *map);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_PixelBitSet_getMapping_1_MR_BMap_MR_PixelId_MR_PixelId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PixelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.PixelBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRPixelId_MRPixelId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_getMapping_1_phmap_flat_hash_map_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_getMapping_1_phmap_flat_hash_map_MR_PixelId_MR_PixelId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRPixelId_MRPixelId._Underlying *map);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_PixelBitSet_getMapping_1_phmap_flat_hash_map_MR_PixelId_MR_PixelId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PixelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.PixelBitSet> GetMapping(MR.Const_Vector_MRPixelId_MRPixelId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_getMapping_2_MR_Vector_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_getMapping_2_MR_Vector_MR_PixelId_MR_PixelId(_Underlying *_this, MR.Const_Vector_MRPixelId_MRPixelId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_PixelBitSet_getMapping_2_MR_Vector_MR_PixelId_MR_PixelId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::PixelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.PixelBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRPixelId_MRPixelId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_getMapping_2_phmap_flat_hash_map_MR_PixelId_MR_PixelId", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_getMapping_2_phmap_flat_hash_map_MR_PixelId_MR_PixelId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRPixelId_MRPixelId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_PixelBitSet_getMapping_2_phmap_flat_hash_map_MR_PixelId_MR_PixelId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::PixelBitSet::backId`.
        public unsafe MR.PixelId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_backId", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_backId(_Underlying *_this);
            return __MR_PixelBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::PixelBitSet::beginId`.
        public static MR.PixelId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_beginId", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_beginId();
            return __MR_PixelBitSet_beginId();
        }

        /// Generated from method `MR::PixelBitSet::endId`.
        public unsafe MR.PixelId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_endId", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelBitSet_endId(_Underlying *_this);
            return __MR_PixelBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::PixelBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_PixelBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_PixelBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::PixelBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_empty(_Underlying *_this);
            return __MR_PixelBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PixelBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_PixelBitSet_size(_Underlying *_this);
            return __MR_PixelBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_PixelBitSet_num_blocks(_Underlying *_this);
            return __MR_PixelBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_PixelBitSet_capacity(_Underlying *_this);
            return __MR_PixelBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_PixelBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::PixelBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_PixelBitSet_bits(_Underlying *_this);
            return new(__MR_PixelBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::PixelBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_all", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_all(_Underlying *_this);
            return __MR_PixelBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::PixelBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_any", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_any(_Underlying *_this);
            return __MR_PixelBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::PixelBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_none", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_none(_Underlying *_this);
            return __MR_PixelBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::PixelBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_PixelBitSet_count(_Underlying *_this);
            return __MR_PixelBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::PixelBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PixelBitSet_heapBytes(_Underlying *_this);
            return __MR_PixelBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.PixelBitSet> operator&(MR.Const_PixelBitSet a, MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_PixelBitSet", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_bitand_MR_PixelBitSet(MR.Const_PixelBitSet._Underlying *a, MR.Const_PixelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_bitand_MR_PixelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.PixelBitSet> operator|(MR.Const_PixelBitSet a, MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_PixelBitSet", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_bitor_MR_PixelBitSet(MR.Const_PixelBitSet._Underlying *a, MR.Const_PixelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_bitor_MR_PixelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.PixelBitSet> operator^(MR.Const_PixelBitSet a, MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_PixelBitSet", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_xor_MR_PixelBitSet(MR.Const_PixelBitSet._Underlying *a, MR.Const_PixelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_xor_MR_PixelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.PixelBitSet> operator-(MR.Const_PixelBitSet a, MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_PixelBitSet", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_sub_MR_PixelBitSet(MR.Const_PixelBitSet._Underlying *a, MR.Const_PixelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.PixelBitSet(__MR_sub_MR_PixelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::PixelBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class PixelBitSet : Const_PixelBitSet
    {
        internal unsafe PixelBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(PixelBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_PixelBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_PixelBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PixelBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_PixelBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        public unsafe PixelBitSet(MR._ByValue_PixelBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PixelBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_PixelBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe PixelBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_PixelBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        public unsafe PixelBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_PixelBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::PixelBitSet::PixelBitSet`.
        public unsafe PixelBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_PixelBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::operator=`.
        public unsafe MR.PixelBitSet Assign(MR._ByValue_PixelBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PixelBitSet._Underlying *_other);
            return new(__MR_PixelBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::set`.
        public unsafe MR.PixelBitSet Set(MR.PixelId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_set_3", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_set_3(_Underlying *_this, MR.PixelId n, ulong len, byte val);
            return new(__MR_PixelBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::set`.
        public unsafe MR.PixelBitSet Set(MR.PixelId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_set_2", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_set_2(_Underlying *_this, MR.PixelId n, byte val);
            return new(__MR_PixelBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::set`.
        public unsafe MR.PixelBitSet Set(MR.PixelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_set_1", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_set_1(_Underlying *_this, MR.PixelId n);
            return new(__MR_PixelBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::set`.
        public unsafe MR.PixelBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_set_0", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_set_0(_Underlying *_this);
            return new(__MR_PixelBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::reset`.
        public unsafe MR.PixelBitSet Reset(MR.PixelId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_reset_2", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_reset_2(_Underlying *_this, MR.PixelId n, ulong len);
            return new(__MR_PixelBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::reset`.
        public unsafe MR.PixelBitSet Reset(MR.PixelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_reset_1", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_reset_1(_Underlying *_this, MR.PixelId n);
            return new(__MR_PixelBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::reset`.
        public unsafe MR.PixelBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_reset_0", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_reset_0(_Underlying *_this);
            return new(__MR_PixelBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::flip`.
        public unsafe MR.PixelBitSet Flip(MR.PixelId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_flip_2", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_flip_2(_Underlying *_this, MR.PixelId n, ulong len);
            return new(__MR_PixelBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::flip`.
        public unsafe MR.PixelBitSet Flip(MR.PixelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_flip_1", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_flip_1(_Underlying *_this, MR.PixelId n);
            return new(__MR_PixelBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::flip`.
        public unsafe MR.PixelBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_flip_0", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_flip_0(_Underlying *_this);
            return new(__MR_PixelBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.PixelId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_test_set(_Underlying *_this, MR.PixelId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_PixelBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::PixelBitSet::operator&=`.
        public unsafe MR.PixelBitSet BitandAssign(MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_bitand_assign(_Underlying *_this, MR.Const_PixelBitSet._Underlying *b);
            return new(__MR_PixelBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::operator|=`.
        public unsafe MR.PixelBitSet BitorAssign(MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_bitor_assign(_Underlying *_this, MR.Const_PixelBitSet._Underlying *b);
            return new(__MR_PixelBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::operator^=`.
        public unsafe MR.PixelBitSet XorAssign(MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_xor_assign(_Underlying *_this, MR.Const_PixelBitSet._Underlying *b);
            return new(__MR_PixelBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::operator-=`.
        public unsafe MR.PixelBitSet SubAssign(MR.Const_PixelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_sub_assign(_Underlying *_this, MR.Const_PixelBitSet._Underlying *b);
            return new(__MR_PixelBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::PixelBitSet::subtract`.
        public unsafe MR.PixelBitSet Subtract(MR.Const_PixelBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_subtract", ExactSpelling = true)]
            extern static MR.PixelBitSet._Underlying *__MR_PixelBitSet_subtract(_Underlying *_this, MR.Const_PixelBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_PixelBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::PixelBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.PixelId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_autoResizeSet_3(_Underlying *_this, MR.PixelId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PixelBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::PixelBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.PixelId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_autoResizeSet_2(_Underlying *_this, MR.PixelId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PixelBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::PixelBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.PixelId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_autoResizeTestSet(_Underlying *_this, MR.PixelId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_PixelBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::PixelBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_PixelBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::PixelBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_resize", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_PixelBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::PixelBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_clear", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_clear(_Underlying *_this);
            __MR_PixelBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_shrink_to_fit(_Underlying *_this);
            __MR_PixelBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_PixelBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_PixelBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::PixelBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_reverse(_Underlying *_this);
            __MR_PixelBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::PixelBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_push_back(_Underlying *_this, byte val);
            __MR_PixelBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::PixelBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_pop_back(_Underlying *_this);
            __MR_PixelBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::PixelBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_PixelBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_PixelBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PixelBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PixelBitSet`/`Const_PixelBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PixelBitSet
    {
        internal readonly Const_PixelBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PixelBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PixelBitSet(Const_PixelBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PixelBitSet(Const_PixelBitSet arg) {return new(arg);}
        public _ByValue_PixelBitSet(MR.Misc._Moved<PixelBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PixelBitSet(MR.Misc._Moved<PixelBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PixelBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PixelBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PixelBitSet`/`Const_PixelBitSet` directly.
    public class _InOptMut_PixelBitSet
    {
        public PixelBitSet? Opt;

        public _InOptMut_PixelBitSet() {}
        public _InOptMut_PixelBitSet(PixelBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_PixelBitSet(PixelBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `PixelBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PixelBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PixelBitSet`/`Const_PixelBitSet` to pass it to the function.
    public class _InOptConst_PixelBitSet
    {
        public Const_PixelBitSet? Opt;

        public _InOptConst_PixelBitSet() {}
        public _InOptConst_PixelBitSet(Const_PixelBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_PixelBitSet(Const_PixelBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::VoxelBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_VoxelBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_Destroy(_Underlying *_this);
            __MR_VoxelBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_VoxelBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_VoxelBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_VoxelBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_VoxelBitSet_Get_bits_per_block();
                return *__MR_VoxelBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_VoxelBitSet_Get_npos();
                return *__MR_VoxelBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        public unsafe Const_VoxelBitSet(MR._ByValue_VoxelBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_VoxelBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_VoxelBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        public unsafe Const_VoxelBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VoxelBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        public unsafe Const_VoxelBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VoxelBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::test`.
        public unsafe bool Test(MR.VoxelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_test", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_test(_Underlying *_this, MR.VoxelId n);
            return __MR_VoxelBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::VoxelBitSet::find_first`.
        public unsafe MR.VoxelId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_find_first", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_find_first(_Underlying *_this);
            return __MR_VoxelBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::find_next`.
        public unsafe MR.VoxelId FindNext(MR.VoxelId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_find_next", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_find_next(_Underlying *_this, MR.VoxelId pos);
            return __MR_VoxelBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::VoxelBitSet::find_last`.
        public unsafe MR.VoxelId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_find_last", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_find_last(_Underlying *_this);
            return __MR_VoxelBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::VoxelBitSet::nthSetBit`.
        public unsafe MR.VoxelId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_VoxelBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::VoxelBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_VoxelBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_is_subset_of(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *a);
            return __MR_VoxelBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::VoxelBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_VoxelBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_intersects(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *a);
            return __MR_VoxelBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VoxelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VoxelBitSet> GetMapping(MR.Const_Vector_MRVoxelId_MRVoxelId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_getMapping_1_MR_Vector_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_getMapping_1_MR_Vector_MR_VoxelId_MR_VoxelId(_Underlying *_this, MR.Const_Vector_MRVoxelId_MRVoxelId._Underlying *map);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_VoxelBitSet_getMapping_1_MR_Vector_MR_VoxelId_MR_VoxelId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VoxelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VoxelBitSet> GetMapping(MR.Const_BMap_MRVoxelId_MRVoxelId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_getMapping_1_MR_BMap_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_getMapping_1_MR_BMap_MR_VoxelId_MR_VoxelId(_Underlying *_this, MR.Const_BMap_MRVoxelId_MRVoxelId._Underlying *map);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_VoxelBitSet_getMapping_1_MR_BMap_MR_VoxelId_MR_VoxelId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VoxelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VoxelBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRVoxelId_MRVoxelId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_getMapping_1_phmap_flat_hash_map_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_getMapping_1_phmap_flat_hash_map_MR_VoxelId_MR_VoxelId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRVoxelId_MRVoxelId._Underlying *map);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_VoxelBitSet_getMapping_1_phmap_flat_hash_map_MR_VoxelId_MR_VoxelId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VoxelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VoxelBitSet> GetMapping(MR.Const_Vector_MRVoxelId_MRVoxelId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_getMapping_2_MR_Vector_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_getMapping_2_MR_Vector_MR_VoxelId_MR_VoxelId(_Underlying *_this, MR.Const_Vector_MRVoxelId_MRVoxelId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_VoxelBitSet_getMapping_2_MR_Vector_MR_VoxelId_MR_VoxelId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::VoxelBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.VoxelBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRVoxelId_MRVoxelId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_getMapping_2_phmap_flat_hash_map_MR_VoxelId_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_getMapping_2_phmap_flat_hash_map_MR_VoxelId_MR_VoxelId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRVoxelId_MRVoxelId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_VoxelBitSet_getMapping_2_phmap_flat_hash_map_MR_VoxelId_MR_VoxelId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::VoxelBitSet::backId`.
        public unsafe MR.VoxelId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_backId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_backId(_Underlying *_this);
            return __MR_VoxelBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::VoxelBitSet::beginId`.
        public static MR.VoxelId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_beginId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_beginId();
            return __MR_VoxelBitSet_beginId();
        }

        /// Generated from method `MR::VoxelBitSet::endId`.
        public unsafe MR.VoxelId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_endId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelBitSet_endId(_Underlying *_this);
            return __MR_VoxelBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::VoxelBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_VoxelBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_VoxelBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VoxelBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_empty(_Underlying *_this);
            return __MR_VoxelBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VoxelBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_VoxelBitSet_size(_Underlying *_this);
            return __MR_VoxelBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_VoxelBitSet_num_blocks(_Underlying *_this);
            return __MR_VoxelBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_VoxelBitSet_capacity(_Underlying *_this);
            return __MR_VoxelBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_VoxelBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::VoxelBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_VoxelBitSet_bits(_Underlying *_this);
            return new(__MR_VoxelBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::VoxelBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_all", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_all(_Underlying *_this);
            return __MR_VoxelBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::VoxelBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_any", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_any(_Underlying *_this);
            return __MR_VoxelBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::VoxelBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_none", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_none(_Underlying *_this);
            return __MR_VoxelBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::VoxelBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_VoxelBitSet_count(_Underlying *_this);
            return __MR_VoxelBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::VoxelBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_VoxelBitSet_heapBytes(_Underlying *_this);
            return __MR_VoxelBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.VoxelBitSet> operator&(MR.Const_VoxelBitSet a, MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_VoxelBitSet", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_bitand_MR_VoxelBitSet(MR.Const_VoxelBitSet._Underlying *a, MR.Const_VoxelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_bitand_MR_VoxelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.VoxelBitSet> operator|(MR.Const_VoxelBitSet a, MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_VoxelBitSet", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_bitor_MR_VoxelBitSet(MR.Const_VoxelBitSet._Underlying *a, MR.Const_VoxelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_bitor_MR_VoxelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.VoxelBitSet> operator^(MR.Const_VoxelBitSet a, MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_VoxelBitSet", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_xor_MR_VoxelBitSet(MR.Const_VoxelBitSet._Underlying *a, MR.Const_VoxelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_xor_MR_VoxelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.VoxelBitSet> operator-(MR.Const_VoxelBitSet a, MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelBitSet", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_sub_MR_VoxelBitSet(MR.Const_VoxelBitSet._Underlying *a, MR.Const_VoxelBitSet._Underlying *b);
            return MR.Misc.Move(new MR.VoxelBitSet(__MR_sub_MR_VoxelBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::VoxelBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class VoxelBitSet : Const_VoxelBitSet
    {
        internal unsafe VoxelBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(VoxelBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_VoxelBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_VoxelBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        public unsafe VoxelBitSet(MR._ByValue_VoxelBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe VoxelBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_VoxelBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        public unsafe VoxelBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VoxelBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::VoxelBitSet::VoxelBitSet`.
        public unsafe VoxelBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_VoxelBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::operator=`.
        public unsafe MR.VoxelBitSet Assign(MR._ByValue_VoxelBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelBitSet._Underlying *_other);
            return new(__MR_VoxelBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::set`.
        public unsafe MR.VoxelBitSet Set(MR.VoxelId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_set_3", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_set_3(_Underlying *_this, MR.VoxelId n, ulong len, byte val);
            return new(__MR_VoxelBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::set`.
        public unsafe MR.VoxelBitSet Set(MR.VoxelId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_set_2", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_set_2(_Underlying *_this, MR.VoxelId n, byte val);
            return new(__MR_VoxelBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::set`.
        public unsafe MR.VoxelBitSet Set(MR.VoxelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_set_1", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_set_1(_Underlying *_this, MR.VoxelId n);
            return new(__MR_VoxelBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::set`.
        public unsafe MR.VoxelBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_set_0", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_set_0(_Underlying *_this);
            return new(__MR_VoxelBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::reset`.
        public unsafe MR.VoxelBitSet Reset(MR.VoxelId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_reset_2", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_reset_2(_Underlying *_this, MR.VoxelId n, ulong len);
            return new(__MR_VoxelBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::reset`.
        public unsafe MR.VoxelBitSet Reset(MR.VoxelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_reset_1", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_reset_1(_Underlying *_this, MR.VoxelId n);
            return new(__MR_VoxelBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::reset`.
        public unsafe MR.VoxelBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_reset_0", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_reset_0(_Underlying *_this);
            return new(__MR_VoxelBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::flip`.
        public unsafe MR.VoxelBitSet Flip(MR.VoxelId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_flip_2", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_flip_2(_Underlying *_this, MR.VoxelId n, ulong len);
            return new(__MR_VoxelBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::flip`.
        public unsafe MR.VoxelBitSet Flip(MR.VoxelId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_flip_1", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_flip_1(_Underlying *_this, MR.VoxelId n);
            return new(__MR_VoxelBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::flip`.
        public unsafe MR.VoxelBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_flip_0", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_flip_0(_Underlying *_this);
            return new(__MR_VoxelBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.VoxelId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_test_set(_Underlying *_this, MR.VoxelId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VoxelBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::VoxelBitSet::operator&=`.
        public unsafe MR.VoxelBitSet BitandAssign(MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_bitand_assign(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *b);
            return new(__MR_VoxelBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::operator|=`.
        public unsafe MR.VoxelBitSet BitorAssign(MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_bitor_assign(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *b);
            return new(__MR_VoxelBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::operator^=`.
        public unsafe MR.VoxelBitSet XorAssign(MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_xor_assign(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *b);
            return new(__MR_VoxelBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::operator-=`.
        public unsafe MR.VoxelBitSet SubAssign(MR.Const_VoxelBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_sub_assign(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *b);
            return new(__MR_VoxelBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::VoxelBitSet::subtract`.
        public unsafe MR.VoxelBitSet Subtract(MR.Const_VoxelBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_subtract", ExactSpelling = true)]
            extern static MR.VoxelBitSet._Underlying *__MR_VoxelBitSet_subtract(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_VoxelBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::VoxelBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.VoxelId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_autoResizeSet_3(_Underlying *_this, MR.VoxelId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VoxelBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::VoxelBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.VoxelId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_autoResizeSet_2(_Underlying *_this, MR.VoxelId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VoxelBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::VoxelBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.VoxelId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_autoResizeTestSet(_Underlying *_this, MR.VoxelId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VoxelBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::VoxelBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_VoxelBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::VoxelBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_resize", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VoxelBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::VoxelBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_clear", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_clear(_Underlying *_this);
            __MR_VoxelBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_shrink_to_fit(_Underlying *_this);
            __MR_VoxelBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_VoxelBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VoxelBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::VoxelBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_reverse(_Underlying *_this);
            __MR_VoxelBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::VoxelBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_push_back(_Underlying *_this, byte val);
            __MR_VoxelBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::VoxelBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_pop_back(_Underlying *_this);
            __MR_VoxelBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::VoxelBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_VoxelBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_VoxelBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VoxelBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VoxelBitSet`/`Const_VoxelBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VoxelBitSet
    {
        internal readonly Const_VoxelBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VoxelBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VoxelBitSet(Const_VoxelBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VoxelBitSet(Const_VoxelBitSet arg) {return new(arg);}
        public _ByValue_VoxelBitSet(MR.Misc._Moved<VoxelBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VoxelBitSet(MR.Misc._Moved<VoxelBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VoxelBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelBitSet`/`Const_VoxelBitSet` directly.
    public class _InOptMut_VoxelBitSet
    {
        public VoxelBitSet? Opt;

        public _InOptMut_VoxelBitSet() {}
        public _InOptMut_VoxelBitSet(VoxelBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelBitSet(VoxelBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelBitSet`/`Const_VoxelBitSet` to pass it to the function.
    public class _InOptConst_VoxelBitSet
    {
        public Const_VoxelBitSet? Opt;

        public _InOptConst_VoxelBitSet() {}
        public _InOptConst_VoxelBitSet(Const_VoxelBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelBitSet(Const_VoxelBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::RegionBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_RegionBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RegionBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_Destroy(_Underlying *_this);
            __MR_RegionBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RegionBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_RegionBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_RegionBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_RegionBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_RegionBitSet_Get_bits_per_block();
                return *__MR_RegionBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_RegionBitSet_Get_npos();
                return *__MR_RegionBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RegionBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_RegionBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        public unsafe Const_RegionBitSet(MR._ByValue_RegionBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RegionBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_RegionBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_RegionBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_RegionBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        public unsafe Const_RegionBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_RegionBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        public unsafe Const_RegionBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_RegionBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::test`.
        public unsafe bool Test(MR.RegionId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_test", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_test(_Underlying *_this, MR.RegionId n);
            return __MR_RegionBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::RegionBitSet::find_first`.
        public unsafe MR.RegionId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_find_first", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_find_first(_Underlying *_this);
            return __MR_RegionBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::find_next`.
        public unsafe MR.RegionId FindNext(MR.RegionId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_find_next", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_find_next(_Underlying *_this, MR.RegionId pos);
            return __MR_RegionBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::RegionBitSet::find_last`.
        public unsafe MR.RegionId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_find_last", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_find_last(_Underlying *_this);
            return __MR_RegionBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::RegionBitSet::nthSetBit`.
        public unsafe MR.RegionId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_RegionBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::RegionBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_RegionBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_is_subset_of(_Underlying *_this, MR.Const_RegionBitSet._Underlying *a);
            return __MR_RegionBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::RegionBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_RegionBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_intersects(_Underlying *_this, MR.Const_RegionBitSet._Underlying *a);
            return __MR_RegionBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RegionBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.RegionBitSet> GetMapping(MR.Const_Vector_MRRegionId_MRRegionId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_getMapping_1_MR_Vector_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_getMapping_1_MR_Vector_MR_RegionId_MR_RegionId(_Underlying *_this, MR.Const_Vector_MRRegionId_MRRegionId._Underlying *map);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_RegionBitSet_getMapping_1_MR_Vector_MR_RegionId_MR_RegionId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RegionBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.RegionBitSet> GetMapping(MR.Const_BMap_MRRegionId_MRRegionId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_getMapping_1_MR_BMap_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_getMapping_1_MR_BMap_MR_RegionId_MR_RegionId(_Underlying *_this, MR.Const_BMap_MRRegionId_MRRegionId._Underlying *map);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_RegionBitSet_getMapping_1_MR_BMap_MR_RegionId_MR_RegionId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RegionBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.RegionBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRRegionId_MRRegionId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_getMapping_1_phmap_flat_hash_map_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_getMapping_1_phmap_flat_hash_map_MR_RegionId_MR_RegionId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRRegionId_MRRegionId._Underlying *map);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_RegionBitSet_getMapping_1_phmap_flat_hash_map_MR_RegionId_MR_RegionId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RegionBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.RegionBitSet> GetMapping(MR.Const_Vector_MRRegionId_MRRegionId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_getMapping_2_MR_Vector_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_getMapping_2_MR_Vector_MR_RegionId_MR_RegionId(_Underlying *_this, MR.Const_Vector_MRRegionId_MRRegionId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_RegionBitSet_getMapping_2_MR_Vector_MR_RegionId_MR_RegionId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::RegionBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.RegionBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRRegionId_MRRegionId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_getMapping_2_phmap_flat_hash_map_MR_RegionId_MR_RegionId", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_getMapping_2_phmap_flat_hash_map_MR_RegionId_MR_RegionId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRRegionId_MRRegionId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_RegionBitSet_getMapping_2_phmap_flat_hash_map_MR_RegionId_MR_RegionId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::RegionBitSet::backId`.
        public unsafe MR.RegionId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_backId", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_backId(_Underlying *_this);
            return __MR_RegionBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::RegionBitSet::beginId`.
        public static MR.RegionId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_beginId", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_beginId();
            return __MR_RegionBitSet_beginId();
        }

        /// Generated from method `MR::RegionBitSet::endId`.
        public unsafe MR.RegionId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_endId", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionBitSet_endId(_Underlying *_this);
            return __MR_RegionBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::RegionBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_RegionBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_RegionBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::RegionBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_empty(_Underlying *_this);
            return __MR_RegionBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RegionBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_RegionBitSet_size(_Underlying *_this);
            return __MR_RegionBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_RegionBitSet_num_blocks(_Underlying *_this);
            return __MR_RegionBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_RegionBitSet_capacity(_Underlying *_this);
            return __MR_RegionBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_RegionBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::RegionBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_RegionBitSet_bits(_Underlying *_this);
            return new(__MR_RegionBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::RegionBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_all", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_all(_Underlying *_this);
            return __MR_RegionBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::RegionBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_any", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_any(_Underlying *_this);
            return __MR_RegionBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::RegionBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_none", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_none(_Underlying *_this);
            return __MR_RegionBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::RegionBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_RegionBitSet_count(_Underlying *_this);
            return __MR_RegionBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::RegionBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_RegionBitSet_heapBytes(_Underlying *_this);
            return __MR_RegionBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.RegionBitSet> operator&(MR.Const_RegionBitSet a, MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_RegionBitSet", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_bitand_MR_RegionBitSet(MR.Const_RegionBitSet._Underlying *a, MR.Const_RegionBitSet._Underlying *b);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_bitand_MR_RegionBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.RegionBitSet> operator|(MR.Const_RegionBitSet a, MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_RegionBitSet", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_bitor_MR_RegionBitSet(MR.Const_RegionBitSet._Underlying *a, MR.Const_RegionBitSet._Underlying *b);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_bitor_MR_RegionBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.RegionBitSet> operator^(MR.Const_RegionBitSet a, MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_RegionBitSet", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_xor_MR_RegionBitSet(MR.Const_RegionBitSet._Underlying *a, MR.Const_RegionBitSet._Underlying *b);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_xor_MR_RegionBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.RegionBitSet> operator-(MR.Const_RegionBitSet a, MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_RegionBitSet", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_sub_MR_RegionBitSet(MR.Const_RegionBitSet._Underlying *a, MR.Const_RegionBitSet._Underlying *b);
            return MR.Misc.Move(new MR.RegionBitSet(__MR_sub_MR_RegionBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::RegionBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class RegionBitSet : Const_RegionBitSet
    {
        internal unsafe RegionBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(RegionBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_RegionBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_RegionBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RegionBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_RegionBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        public unsafe RegionBitSet(MR._ByValue_RegionBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RegionBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_RegionBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe RegionBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_RegionBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        public unsafe RegionBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_RegionBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::RegionBitSet::RegionBitSet`.
        public unsafe RegionBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_RegionBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::operator=`.
        public unsafe MR.RegionBitSet Assign(MR._ByValue_RegionBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RegionBitSet._Underlying *_other);
            return new(__MR_RegionBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::set`.
        public unsafe MR.RegionBitSet Set(MR.RegionId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_set_3", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_set_3(_Underlying *_this, MR.RegionId n, ulong len, byte val);
            return new(__MR_RegionBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::set`.
        public unsafe MR.RegionBitSet Set(MR.RegionId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_set_2", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_set_2(_Underlying *_this, MR.RegionId n, byte val);
            return new(__MR_RegionBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::set`.
        public unsafe MR.RegionBitSet Set(MR.RegionId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_set_1", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_set_1(_Underlying *_this, MR.RegionId n);
            return new(__MR_RegionBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::set`.
        public unsafe MR.RegionBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_set_0", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_set_0(_Underlying *_this);
            return new(__MR_RegionBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::reset`.
        public unsafe MR.RegionBitSet Reset(MR.RegionId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_reset_2", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_reset_2(_Underlying *_this, MR.RegionId n, ulong len);
            return new(__MR_RegionBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::reset`.
        public unsafe MR.RegionBitSet Reset(MR.RegionId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_reset_1", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_reset_1(_Underlying *_this, MR.RegionId n);
            return new(__MR_RegionBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::reset`.
        public unsafe MR.RegionBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_reset_0", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_reset_0(_Underlying *_this);
            return new(__MR_RegionBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::flip`.
        public unsafe MR.RegionBitSet Flip(MR.RegionId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_flip_2", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_flip_2(_Underlying *_this, MR.RegionId n, ulong len);
            return new(__MR_RegionBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::flip`.
        public unsafe MR.RegionBitSet Flip(MR.RegionId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_flip_1", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_flip_1(_Underlying *_this, MR.RegionId n);
            return new(__MR_RegionBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::flip`.
        public unsafe MR.RegionBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_flip_0", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_flip_0(_Underlying *_this);
            return new(__MR_RegionBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.RegionId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_test_set(_Underlying *_this, MR.RegionId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_RegionBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::RegionBitSet::operator&=`.
        public unsafe MR.RegionBitSet BitandAssign(MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_bitand_assign(_Underlying *_this, MR.Const_RegionBitSet._Underlying *b);
            return new(__MR_RegionBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::operator|=`.
        public unsafe MR.RegionBitSet BitorAssign(MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_bitor_assign(_Underlying *_this, MR.Const_RegionBitSet._Underlying *b);
            return new(__MR_RegionBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::operator^=`.
        public unsafe MR.RegionBitSet XorAssign(MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_xor_assign(_Underlying *_this, MR.Const_RegionBitSet._Underlying *b);
            return new(__MR_RegionBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::operator-=`.
        public unsafe MR.RegionBitSet SubAssign(MR.Const_RegionBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_sub_assign(_Underlying *_this, MR.Const_RegionBitSet._Underlying *b);
            return new(__MR_RegionBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::RegionBitSet::subtract`.
        public unsafe MR.RegionBitSet Subtract(MR.Const_RegionBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_subtract", ExactSpelling = true)]
            extern static MR.RegionBitSet._Underlying *__MR_RegionBitSet_subtract(_Underlying *_this, MR.Const_RegionBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_RegionBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::RegionBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.RegionId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_autoResizeSet_3(_Underlying *_this, MR.RegionId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_RegionBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::RegionBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.RegionId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_autoResizeSet_2(_Underlying *_this, MR.RegionId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_RegionBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::RegionBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.RegionId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_autoResizeTestSet(_Underlying *_this, MR.RegionId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_RegionBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::RegionBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_RegionBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::RegionBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_resize", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_RegionBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::RegionBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_clear", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_clear(_Underlying *_this);
            __MR_RegionBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_shrink_to_fit(_Underlying *_this);
            __MR_RegionBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_RegionBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_RegionBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::RegionBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_reverse(_Underlying *_this);
            __MR_RegionBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::RegionBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_push_back(_Underlying *_this, byte val);
            __MR_RegionBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::RegionBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_pop_back(_Underlying *_this);
            __MR_RegionBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::RegionBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_RegionBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_RegionBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `RegionBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RegionBitSet`/`Const_RegionBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RegionBitSet
    {
        internal readonly Const_RegionBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RegionBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RegionBitSet(Const_RegionBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_RegionBitSet(Const_RegionBitSet arg) {return new(arg);}
        public _ByValue_RegionBitSet(MR.Misc._Moved<RegionBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RegionBitSet(MR.Misc._Moved<RegionBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `RegionBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RegionBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RegionBitSet`/`Const_RegionBitSet` directly.
    public class _InOptMut_RegionBitSet
    {
        public RegionBitSet? Opt;

        public _InOptMut_RegionBitSet() {}
        public _InOptMut_RegionBitSet(RegionBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_RegionBitSet(RegionBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `RegionBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RegionBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RegionBitSet`/`Const_RegionBitSet` to pass it to the function.
    public class _InOptConst_RegionBitSet
    {
        public Const_RegionBitSet? Opt;

        public _InOptConst_RegionBitSet() {}
        public _InOptConst_RegionBitSet(Const_RegionBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_RegionBitSet(Const_RegionBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::NodeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_NodeBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NodeBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_Destroy(_Underlying *_this);
            __MR_NodeBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NodeBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_NodeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_NodeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_NodeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_NodeBitSet_Get_bits_per_block();
                return *__MR_NodeBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_NodeBitSet_Get_npos();
                return *__MR_NodeBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NodeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_NodeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        public unsafe Const_NodeBitSet(MR._ByValue_NodeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NodeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_NodeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_NodeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_NodeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        public unsafe Const_NodeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_NodeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        public unsafe Const_NodeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_NodeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::test`.
        public unsafe bool Test(MR.NodeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_test", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_test(_Underlying *_this, MR.NodeId n);
            return __MR_NodeBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::NodeBitSet::find_first`.
        public unsafe MR.NodeId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_find_first", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_find_first(_Underlying *_this);
            return __MR_NodeBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::find_next`.
        public unsafe MR.NodeId FindNext(MR.NodeId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_find_next", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_find_next(_Underlying *_this, MR.NodeId pos);
            return __MR_NodeBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::NodeBitSet::find_last`.
        public unsafe MR.NodeId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_find_last", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_find_last(_Underlying *_this);
            return __MR_NodeBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::NodeBitSet::nthSetBit`.
        public unsafe MR.NodeId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_NodeBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::NodeBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_NodeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_is_subset_of(_Underlying *_this, MR.Const_NodeBitSet._Underlying *a);
            return __MR_NodeBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::NodeBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_NodeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_intersects(_Underlying *_this, MR.Const_NodeBitSet._Underlying *a);
            return __MR_NodeBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NodeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetMapping(MR.Const_Vector_MRNodeId_MRNodeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_getMapping_1_MR_Vector_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_getMapping_1_MR_Vector_MR_NodeId_MR_NodeId(_Underlying *_this, MR.Const_Vector_MRNodeId_MRNodeId._Underlying *map);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_NodeBitSet_getMapping_1_MR_Vector_MR_NodeId_MR_NodeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::NodeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetMapping(MR.Const_BMap_MRNodeId_MRNodeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_getMapping_1_MR_BMap_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_getMapping_1_MR_BMap_MR_NodeId_MR_NodeId(_Underlying *_this, MR.Const_BMap_MRNodeId_MRNodeId._Underlying *map);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_NodeBitSet_getMapping_1_MR_BMap_MR_NodeId_MR_NodeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::NodeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRNodeId_MRNodeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_getMapping_1_phmap_flat_hash_map_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_getMapping_1_phmap_flat_hash_map_MR_NodeId_MR_NodeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRNodeId_MRNodeId._Underlying *map);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_NodeBitSet_getMapping_1_phmap_flat_hash_map_MR_NodeId_MR_NodeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::NodeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetMapping(MR.Const_Vector_MRNodeId_MRNodeId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_getMapping_2_MR_Vector_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_getMapping_2_MR_Vector_MR_NodeId_MR_NodeId(_Underlying *_this, MR.Const_Vector_MRNodeId_MRNodeId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_NodeBitSet_getMapping_2_MR_Vector_MR_NodeId_MR_NodeId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::NodeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRNodeId_MRNodeId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_getMapping_2_phmap_flat_hash_map_MR_NodeId_MR_NodeId", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_getMapping_2_phmap_flat_hash_map_MR_NodeId_MR_NodeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRNodeId_MRNodeId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_NodeBitSet_getMapping_2_phmap_flat_hash_map_MR_NodeId_MR_NodeId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::NodeBitSet::backId`.
        public unsafe MR.NodeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_backId", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_backId(_Underlying *_this);
            return __MR_NodeBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::NodeBitSet::beginId`.
        public static MR.NodeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_beginId", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_beginId();
            return __MR_NodeBitSet_beginId();
        }

        /// Generated from method `MR::NodeBitSet::endId`.
        public unsafe MR.NodeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_endId", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeBitSet_endId(_Underlying *_this);
            return __MR_NodeBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::NodeBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_NodeBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_NodeBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::NodeBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_empty(_Underlying *_this);
            return __MR_NodeBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NodeBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_NodeBitSet_size(_Underlying *_this);
            return __MR_NodeBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_NodeBitSet_num_blocks(_Underlying *_this);
            return __MR_NodeBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_NodeBitSet_capacity(_Underlying *_this);
            return __MR_NodeBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_NodeBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::NodeBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_NodeBitSet_bits(_Underlying *_this);
            return new(__MR_NodeBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::NodeBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_all", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_all(_Underlying *_this);
            return __MR_NodeBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::NodeBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_any", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_any(_Underlying *_this);
            return __MR_NodeBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::NodeBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_none", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_none(_Underlying *_this);
            return __MR_NodeBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::NodeBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_NodeBitSet_count(_Underlying *_this);
            return __MR_NodeBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::NodeBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_NodeBitSet_heapBytes(_Underlying *_this);
            return __MR_NodeBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.NodeBitSet> operator&(MR.Const_NodeBitSet a, MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_NodeBitSet", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_bitand_MR_NodeBitSet(MR.Const_NodeBitSet._Underlying *a, MR.Const_NodeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_bitand_MR_NodeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.NodeBitSet> operator|(MR.Const_NodeBitSet a, MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_NodeBitSet", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_bitor_MR_NodeBitSet(MR.Const_NodeBitSet._Underlying *a, MR.Const_NodeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_bitor_MR_NodeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.NodeBitSet> operator^(MR.Const_NodeBitSet a, MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_NodeBitSet", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_xor_MR_NodeBitSet(MR.Const_NodeBitSet._Underlying *a, MR.Const_NodeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_xor_MR_NodeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.NodeBitSet> operator-(MR.Const_NodeBitSet a, MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_NodeBitSet", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_sub_MR_NodeBitSet(MR.Const_NodeBitSet._Underlying *a, MR.Const_NodeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_sub_MR_NodeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::NodeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class NodeBitSet : Const_NodeBitSet
    {
        internal unsafe NodeBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(NodeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_NodeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_NodeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NodeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_NodeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        public unsafe NodeBitSet(MR._ByValue_NodeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NodeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_NodeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe NodeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_NodeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        public unsafe NodeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_NodeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::NodeBitSet::NodeBitSet`.
        public unsafe NodeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_NodeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::operator=`.
        public unsafe MR.NodeBitSet Assign(MR._ByValue_NodeBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.NodeBitSet._Underlying *_other);
            return new(__MR_NodeBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::set`.
        public unsafe MR.NodeBitSet Set(MR.NodeId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_set_3", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_set_3(_Underlying *_this, MR.NodeId n, ulong len, byte val);
            return new(__MR_NodeBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::set`.
        public unsafe MR.NodeBitSet Set(MR.NodeId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_set_2", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_set_2(_Underlying *_this, MR.NodeId n, byte val);
            return new(__MR_NodeBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::set`.
        public unsafe MR.NodeBitSet Set(MR.NodeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_set_1", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_set_1(_Underlying *_this, MR.NodeId n);
            return new(__MR_NodeBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::set`.
        public unsafe MR.NodeBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_set_0", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_set_0(_Underlying *_this);
            return new(__MR_NodeBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::reset`.
        public unsafe MR.NodeBitSet Reset(MR.NodeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_reset_2", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_reset_2(_Underlying *_this, MR.NodeId n, ulong len);
            return new(__MR_NodeBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::reset`.
        public unsafe MR.NodeBitSet Reset(MR.NodeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_reset_1", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_reset_1(_Underlying *_this, MR.NodeId n);
            return new(__MR_NodeBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::reset`.
        public unsafe MR.NodeBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_reset_0", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_reset_0(_Underlying *_this);
            return new(__MR_NodeBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::flip`.
        public unsafe MR.NodeBitSet Flip(MR.NodeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_flip_2", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_flip_2(_Underlying *_this, MR.NodeId n, ulong len);
            return new(__MR_NodeBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::flip`.
        public unsafe MR.NodeBitSet Flip(MR.NodeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_flip_1", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_flip_1(_Underlying *_this, MR.NodeId n);
            return new(__MR_NodeBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::flip`.
        public unsafe MR.NodeBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_flip_0", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_flip_0(_Underlying *_this);
            return new(__MR_NodeBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.NodeId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_test_set(_Underlying *_this, MR.NodeId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_NodeBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::NodeBitSet::operator&=`.
        public unsafe MR.NodeBitSet BitandAssign(MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_bitand_assign(_Underlying *_this, MR.Const_NodeBitSet._Underlying *b);
            return new(__MR_NodeBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::operator|=`.
        public unsafe MR.NodeBitSet BitorAssign(MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_bitor_assign(_Underlying *_this, MR.Const_NodeBitSet._Underlying *b);
            return new(__MR_NodeBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::operator^=`.
        public unsafe MR.NodeBitSet XorAssign(MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_xor_assign(_Underlying *_this, MR.Const_NodeBitSet._Underlying *b);
            return new(__MR_NodeBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::operator-=`.
        public unsafe MR.NodeBitSet SubAssign(MR.Const_NodeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_sub_assign(_Underlying *_this, MR.Const_NodeBitSet._Underlying *b);
            return new(__MR_NodeBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::NodeBitSet::subtract`.
        public unsafe MR.NodeBitSet Subtract(MR.Const_NodeBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_subtract", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_NodeBitSet_subtract(_Underlying *_this, MR.Const_NodeBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_NodeBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::NodeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.NodeId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_autoResizeSet_3(_Underlying *_this, MR.NodeId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_NodeBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::NodeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.NodeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_autoResizeSet_2(_Underlying *_this, MR.NodeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_NodeBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::NodeBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.NodeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_autoResizeTestSet(_Underlying *_this, MR.NodeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_NodeBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::NodeBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_NodeBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::NodeBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_resize", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_NodeBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::NodeBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_clear", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_clear(_Underlying *_this);
            __MR_NodeBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_shrink_to_fit(_Underlying *_this);
            __MR_NodeBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_NodeBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_NodeBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::NodeBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_reverse(_Underlying *_this);
            __MR_NodeBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::NodeBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_push_back(_Underlying *_this, byte val);
            __MR_NodeBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::NodeBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_pop_back(_Underlying *_this);
            __MR_NodeBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::NodeBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_NodeBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_NodeBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `NodeBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `NodeBitSet`/`Const_NodeBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_NodeBitSet
    {
        internal readonly Const_NodeBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_NodeBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_NodeBitSet(Const_NodeBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_NodeBitSet(Const_NodeBitSet arg) {return new(arg);}
        public _ByValue_NodeBitSet(MR.Misc._Moved<NodeBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_NodeBitSet(MR.Misc._Moved<NodeBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `NodeBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NodeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NodeBitSet`/`Const_NodeBitSet` directly.
    public class _InOptMut_NodeBitSet
    {
        public NodeBitSet? Opt;

        public _InOptMut_NodeBitSet() {}
        public _InOptMut_NodeBitSet(NodeBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_NodeBitSet(NodeBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `NodeBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NodeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NodeBitSet`/`Const_NodeBitSet` to pass it to the function.
    public class _InOptConst_NodeBitSet
    {
        public Const_NodeBitSet? Opt;

        public _InOptConst_NodeBitSet() {}
        public _InOptConst_NodeBitSet(Const_NodeBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_NodeBitSet(Const_NodeBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::ObjBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_ObjBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_Destroy(_Underlying *_this);
            __MR_ObjBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_ObjBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_ObjBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_ObjBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_ObjBitSet_Get_bits_per_block();
                return *__MR_ObjBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_ObjBitSet_Get_npos();
                return *__MR_ObjBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        public unsafe Const_ObjBitSet(MR._ByValue_ObjBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_ObjBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_ObjBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ObjBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        public unsafe Const_ObjBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_ObjBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        public unsafe Const_ObjBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_ObjBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::test`.
        public unsafe bool Test(MR.ObjId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_test", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_test(_Underlying *_this, MR.ObjId n);
            return __MR_ObjBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::ObjBitSet::find_first`.
        public unsafe MR.ObjId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_find_first", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_find_first(_Underlying *_this);
            return __MR_ObjBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::find_next`.
        public unsafe MR.ObjId FindNext(MR.ObjId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_find_next", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_find_next(_Underlying *_this, MR.ObjId pos);
            return __MR_ObjBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::ObjBitSet::find_last`.
        public unsafe MR.ObjId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_find_last", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_find_last(_Underlying *_this);
            return __MR_ObjBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::ObjBitSet::nthSetBit`.
        public unsafe MR.ObjId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_ObjBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::ObjBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_ObjBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_is_subset_of(_Underlying *_this, MR.Const_ObjBitSet._Underlying *a);
            return __MR_ObjBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::ObjBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_ObjBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_intersects(_Underlying *_this, MR.Const_ObjBitSet._Underlying *a);
            return __MR_ObjBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetMapping(MR.Const_ObjMap map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_getMapping_1_MR_ObjMap", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_getMapping_1_MR_ObjMap(_Underlying *_this, MR.Const_ObjMap._Underlying *map);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_ObjBitSet_getMapping_1_MR_ObjMap(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetMapping(MR.Const_BMap_MRObjId_MRObjId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_getMapping_1_MR_BMap_MR_ObjId_MR_ObjId", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_getMapping_1_MR_BMap_MR_ObjId_MR_ObjId(_Underlying *_this, MR.Const_BMap_MRObjId_MRObjId._Underlying *map);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_ObjBitSet_getMapping_1_MR_BMap_MR_ObjId_MR_ObjId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRObjId_MRObjId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_getMapping_1_phmap_flat_hash_map_MR_ObjId_MR_ObjId", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_getMapping_1_phmap_flat_hash_map_MR_ObjId_MR_ObjId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRObjId_MRObjId._Underlying *map);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_ObjBitSet_getMapping_1_phmap_flat_hash_map_MR_ObjId_MR_ObjId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetMapping(MR.Const_ObjMap map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_getMapping_2_MR_ObjMap", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_getMapping_2_MR_ObjMap(_Underlying *_this, MR.Const_ObjMap._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_ObjBitSet_getMapping_2_MR_ObjMap(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::ObjBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRObjId_MRObjId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_getMapping_2_phmap_flat_hash_map_MR_ObjId_MR_ObjId", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_getMapping_2_phmap_flat_hash_map_MR_ObjId_MR_ObjId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRObjId_MRObjId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_ObjBitSet_getMapping_2_phmap_flat_hash_map_MR_ObjId_MR_ObjId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::ObjBitSet::backId`.
        public unsafe MR.ObjId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_backId", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_backId(_Underlying *_this);
            return __MR_ObjBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::ObjBitSet::beginId`.
        public static MR.ObjId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_beginId", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_beginId();
            return __MR_ObjBitSet_beginId();
        }

        /// Generated from method `MR::ObjBitSet::endId`.
        public unsafe MR.ObjId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_endId", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjBitSet_endId(_Underlying *_this);
            return __MR_ObjBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::ObjBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_ObjBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_ObjBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_empty(_Underlying *_this);
            return __MR_ObjBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_ObjBitSet_size(_Underlying *_this);
            return __MR_ObjBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_ObjBitSet_num_blocks(_Underlying *_this);
            return __MR_ObjBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_ObjBitSet_capacity(_Underlying *_this);
            return __MR_ObjBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_ObjBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::ObjBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_ObjBitSet_bits(_Underlying *_this);
            return new(__MR_ObjBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::ObjBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_all", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_all(_Underlying *_this);
            return __MR_ObjBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::ObjBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_any", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_any(_Underlying *_this);
            return __MR_ObjBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::ObjBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_none", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_none(_Underlying *_this);
            return __MR_ObjBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::ObjBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_ObjBitSet_count(_Underlying *_this);
            return __MR_ObjBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjBitSet_heapBytes(_Underlying *_this);
            return __MR_ObjBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.ObjBitSet> operator&(MR.Const_ObjBitSet a, MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_ObjBitSet", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_bitand_MR_ObjBitSet(MR.Const_ObjBitSet._Underlying *a, MR.Const_ObjBitSet._Underlying *b);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_bitand_MR_ObjBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.ObjBitSet> operator|(MR.Const_ObjBitSet a, MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_ObjBitSet", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_bitor_MR_ObjBitSet(MR.Const_ObjBitSet._Underlying *a, MR.Const_ObjBitSet._Underlying *b);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_bitor_MR_ObjBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.ObjBitSet> operator^(MR.Const_ObjBitSet a, MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_ObjBitSet", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_xor_MR_ObjBitSet(MR.Const_ObjBitSet._Underlying *a, MR.Const_ObjBitSet._Underlying *b);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_xor_MR_ObjBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.ObjBitSet> operator-(MR.Const_ObjBitSet a, MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_ObjBitSet", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_sub_MR_ObjBitSet(MR.Const_ObjBitSet._Underlying *a, MR.Const_ObjBitSet._Underlying *b);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_sub_MR_ObjBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::ObjBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class ObjBitSet : Const_ObjBitSet
    {
        internal unsafe ObjBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(ObjBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_ObjBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_ObjBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        public unsafe ObjBitSet(MR._ByValue_ObjBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_ObjBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe ObjBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ObjBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        public unsafe ObjBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_ObjBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::ObjBitSet::ObjBitSet`.
        public unsafe ObjBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_ObjBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::operator=`.
        public unsafe MR.ObjBitSet Assign(MR._ByValue_ObjBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjBitSet._Underlying *_other);
            return new(__MR_ObjBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::set`.
        public unsafe MR.ObjBitSet Set(MR.ObjId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_set_3", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_set_3(_Underlying *_this, MR.ObjId n, ulong len, byte val);
            return new(__MR_ObjBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::set`.
        public unsafe MR.ObjBitSet Set(MR.ObjId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_set_2", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_set_2(_Underlying *_this, MR.ObjId n, byte val);
            return new(__MR_ObjBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::set`.
        public unsafe MR.ObjBitSet Set(MR.ObjId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_set_1", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_set_1(_Underlying *_this, MR.ObjId n);
            return new(__MR_ObjBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::set`.
        public unsafe MR.ObjBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_set_0", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_set_0(_Underlying *_this);
            return new(__MR_ObjBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::reset`.
        public unsafe MR.ObjBitSet Reset(MR.ObjId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_reset_2", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_reset_2(_Underlying *_this, MR.ObjId n, ulong len);
            return new(__MR_ObjBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::reset`.
        public unsafe MR.ObjBitSet Reset(MR.ObjId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_reset_1", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_reset_1(_Underlying *_this, MR.ObjId n);
            return new(__MR_ObjBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::reset`.
        public unsafe MR.ObjBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_reset_0", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_reset_0(_Underlying *_this);
            return new(__MR_ObjBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::flip`.
        public unsafe MR.ObjBitSet Flip(MR.ObjId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_flip_2", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_flip_2(_Underlying *_this, MR.ObjId n, ulong len);
            return new(__MR_ObjBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::flip`.
        public unsafe MR.ObjBitSet Flip(MR.ObjId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_flip_1", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_flip_1(_Underlying *_this, MR.ObjId n);
            return new(__MR_ObjBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::flip`.
        public unsafe MR.ObjBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_flip_0", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_flip_0(_Underlying *_this);
            return new(__MR_ObjBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.ObjId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_test_set(_Underlying *_this, MR.ObjId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::ObjBitSet::operator&=`.
        public unsafe MR.ObjBitSet BitandAssign(MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_bitand_assign(_Underlying *_this, MR.Const_ObjBitSet._Underlying *b);
            return new(__MR_ObjBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::operator|=`.
        public unsafe MR.ObjBitSet BitorAssign(MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_bitor_assign(_Underlying *_this, MR.Const_ObjBitSet._Underlying *b);
            return new(__MR_ObjBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::operator^=`.
        public unsafe MR.ObjBitSet XorAssign(MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_xor_assign(_Underlying *_this, MR.Const_ObjBitSet._Underlying *b);
            return new(__MR_ObjBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::operator-=`.
        public unsafe MR.ObjBitSet SubAssign(MR.Const_ObjBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_sub_assign(_Underlying *_this, MR.Const_ObjBitSet._Underlying *b);
            return new(__MR_ObjBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::ObjBitSet::subtract`.
        public unsafe MR.ObjBitSet Subtract(MR.Const_ObjBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_subtract", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_ObjBitSet_subtract(_Underlying *_this, MR.Const_ObjBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_ObjBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::ObjBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.ObjId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_autoResizeSet_3(_Underlying *_this, MR.ObjId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::ObjBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.ObjId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_autoResizeSet_2(_Underlying *_this, MR.ObjId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::ObjBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.ObjId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_autoResizeTestSet(_Underlying *_this, MR.ObjId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::ObjBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_ObjBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::ObjBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_resize", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::ObjBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_clear", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_clear(_Underlying *_this);
            __MR_ObjBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_shrink_to_fit(_Underlying *_this);
            __MR_ObjBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_ObjBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::ObjBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_reverse(_Underlying *_this);
            __MR_ObjBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::ObjBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_push_back(_Underlying *_this, byte val);
            __MR_ObjBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::ObjBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_pop_back(_Underlying *_this);
            __MR_ObjBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::ObjBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_ObjBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_ObjBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjBitSet`/`Const_ObjBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjBitSet
    {
        internal readonly Const_ObjBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjBitSet(Const_ObjBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ObjBitSet(Const_ObjBitSet arg) {return new(arg);}
        public _ByValue_ObjBitSet(MR.Misc._Moved<ObjBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjBitSet(MR.Misc._Moved<ObjBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjBitSet`/`Const_ObjBitSet` directly.
    public class _InOptMut_ObjBitSet
    {
        public ObjBitSet? Opt;

        public _InOptMut_ObjBitSet() {}
        public _InOptMut_ObjBitSet(ObjBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_ObjBitSet(ObjBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjBitSet`/`Const_ObjBitSet` to pass it to the function.
    public class _InOptConst_ObjBitSet
    {
        public Const_ObjBitSet? Opt;

        public _InOptConst_ObjBitSet() {}
        public _InOptConst_ObjBitSet(Const_ObjBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_ObjBitSet(Const_ObjBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::TextureBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_TextureBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TextureBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_Destroy(_Underlying *_this);
            __MR_TextureBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TextureBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_TextureBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_TextureBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_TextureBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_TextureBitSet_Get_bits_per_block();
                return *__MR_TextureBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_TextureBitSet_Get_npos();
                return *__MR_TextureBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TextureBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_TextureBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        public unsafe Const_TextureBitSet(MR._ByValue_TextureBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TextureBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_TextureBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_TextureBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_TextureBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        public unsafe Const_TextureBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TextureBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        public unsafe Const_TextureBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TextureBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::test`.
        public unsafe bool Test(MR.TextureId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_test", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_test(_Underlying *_this, MR.TextureId n);
            return __MR_TextureBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::TextureBitSet::find_first`.
        public unsafe MR.TextureId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_find_first", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_find_first(_Underlying *_this);
            return __MR_TextureBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::find_next`.
        public unsafe MR.TextureId FindNext(MR.TextureId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_find_next", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_find_next(_Underlying *_this, MR.TextureId pos);
            return __MR_TextureBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::TextureBitSet::find_last`.
        public unsafe MR.TextureId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_find_last", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_find_last(_Underlying *_this);
            return __MR_TextureBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::TextureBitSet::nthSetBit`.
        public unsafe MR.TextureId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_TextureBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::TextureBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_TextureBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_is_subset_of(_Underlying *_this, MR.Const_TextureBitSet._Underlying *a);
            return __MR_TextureBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::TextureBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_TextureBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_intersects(_Underlying *_this, MR.Const_TextureBitSet._Underlying *a);
            return __MR_TextureBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TextureBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.TextureBitSet> GetMapping(MR.Const_Vector_MRTextureId_MRTextureId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_getMapping_1_MR_Vector_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_getMapping_1_MR_Vector_MR_TextureId_MR_TextureId(_Underlying *_this, MR.Const_Vector_MRTextureId_MRTextureId._Underlying *map);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_TextureBitSet_getMapping_1_MR_Vector_MR_TextureId_MR_TextureId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TextureBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.TextureBitSet> GetMapping(MR.Const_BMap_MRTextureId_MRTextureId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_getMapping_1_MR_BMap_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_getMapping_1_MR_BMap_MR_TextureId_MR_TextureId(_Underlying *_this, MR.Const_BMap_MRTextureId_MRTextureId._Underlying *map);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_TextureBitSet_getMapping_1_MR_BMap_MR_TextureId_MR_TextureId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TextureBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.TextureBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRTextureId_MRTextureId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_getMapping_1_phmap_flat_hash_map_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_getMapping_1_phmap_flat_hash_map_MR_TextureId_MR_TextureId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRTextureId_MRTextureId._Underlying *map);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_TextureBitSet_getMapping_1_phmap_flat_hash_map_MR_TextureId_MR_TextureId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TextureBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.TextureBitSet> GetMapping(MR.Const_Vector_MRTextureId_MRTextureId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_getMapping_2_MR_Vector_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_getMapping_2_MR_Vector_MR_TextureId_MR_TextureId(_Underlying *_this, MR.Const_Vector_MRTextureId_MRTextureId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_TextureBitSet_getMapping_2_MR_Vector_MR_TextureId_MR_TextureId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::TextureBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.TextureBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRTextureId_MRTextureId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_getMapping_2_phmap_flat_hash_map_MR_TextureId_MR_TextureId", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_getMapping_2_phmap_flat_hash_map_MR_TextureId_MR_TextureId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRTextureId_MRTextureId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_TextureBitSet_getMapping_2_phmap_flat_hash_map_MR_TextureId_MR_TextureId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::TextureBitSet::backId`.
        public unsafe MR.TextureId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_backId", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_backId(_Underlying *_this);
            return __MR_TextureBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::TextureBitSet::beginId`.
        public static MR.TextureId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_beginId", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_beginId();
            return __MR_TextureBitSet_beginId();
        }

        /// Generated from method `MR::TextureBitSet::endId`.
        public unsafe MR.TextureId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_endId", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureBitSet_endId(_Underlying *_this);
            return __MR_TextureBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::TextureBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_TextureBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_TextureBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TextureBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_empty(_Underlying *_this);
            return __MR_TextureBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TextureBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_TextureBitSet_size(_Underlying *_this);
            return __MR_TextureBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_TextureBitSet_num_blocks(_Underlying *_this);
            return __MR_TextureBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_TextureBitSet_capacity(_Underlying *_this);
            return __MR_TextureBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_TextureBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::TextureBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_TextureBitSet_bits(_Underlying *_this);
            return new(__MR_TextureBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::TextureBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_all", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_all(_Underlying *_this);
            return __MR_TextureBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::TextureBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_any", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_any(_Underlying *_this);
            return __MR_TextureBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::TextureBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_none", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_none(_Underlying *_this);
            return __MR_TextureBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::TextureBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_TextureBitSet_count(_Underlying *_this);
            return __MR_TextureBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::TextureBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_TextureBitSet_heapBytes(_Underlying *_this);
            return __MR_TextureBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.TextureBitSet> operator&(MR.Const_TextureBitSet a, MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_TextureBitSet", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_bitand_MR_TextureBitSet(MR.Const_TextureBitSet._Underlying *a, MR.Const_TextureBitSet._Underlying *b);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_bitand_MR_TextureBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.TextureBitSet> operator|(MR.Const_TextureBitSet a, MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_TextureBitSet", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_bitor_MR_TextureBitSet(MR.Const_TextureBitSet._Underlying *a, MR.Const_TextureBitSet._Underlying *b);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_bitor_MR_TextureBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.TextureBitSet> operator^(MR.Const_TextureBitSet a, MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_TextureBitSet", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_xor_MR_TextureBitSet(MR.Const_TextureBitSet._Underlying *a, MR.Const_TextureBitSet._Underlying *b);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_xor_MR_TextureBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.TextureBitSet> operator-(MR.Const_TextureBitSet a, MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_TextureBitSet", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_sub_MR_TextureBitSet(MR.Const_TextureBitSet._Underlying *a, MR.Const_TextureBitSet._Underlying *b);
            return MR.Misc.Move(new MR.TextureBitSet(__MR_sub_MR_TextureBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::TextureBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class TextureBitSet : Const_TextureBitSet
    {
        internal unsafe TextureBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(TextureBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_TextureBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_TextureBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TextureBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_TextureBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        public unsafe TextureBitSet(MR._ByValue_TextureBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TextureBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_TextureBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe TextureBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_TextureBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        public unsafe TextureBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TextureBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TextureBitSet::TextureBitSet`.
        public unsafe TextureBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TextureBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::operator=`.
        public unsafe MR.TextureBitSet Assign(MR._ByValue_TextureBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TextureBitSet._Underlying *_other);
            return new(__MR_TextureBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::set`.
        public unsafe MR.TextureBitSet Set(MR.TextureId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_set_3", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_set_3(_Underlying *_this, MR.TextureId n, ulong len, byte val);
            return new(__MR_TextureBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::set`.
        public unsafe MR.TextureBitSet Set(MR.TextureId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_set_2", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_set_2(_Underlying *_this, MR.TextureId n, byte val);
            return new(__MR_TextureBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::set`.
        public unsafe MR.TextureBitSet Set(MR.TextureId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_set_1", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_set_1(_Underlying *_this, MR.TextureId n);
            return new(__MR_TextureBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::set`.
        public unsafe MR.TextureBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_set_0", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_set_0(_Underlying *_this);
            return new(__MR_TextureBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::reset`.
        public unsafe MR.TextureBitSet Reset(MR.TextureId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_reset_2", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_reset_2(_Underlying *_this, MR.TextureId n, ulong len);
            return new(__MR_TextureBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::reset`.
        public unsafe MR.TextureBitSet Reset(MR.TextureId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_reset_1", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_reset_1(_Underlying *_this, MR.TextureId n);
            return new(__MR_TextureBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::reset`.
        public unsafe MR.TextureBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_reset_0", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_reset_0(_Underlying *_this);
            return new(__MR_TextureBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::flip`.
        public unsafe MR.TextureBitSet Flip(MR.TextureId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_flip_2", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_flip_2(_Underlying *_this, MR.TextureId n, ulong len);
            return new(__MR_TextureBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::flip`.
        public unsafe MR.TextureBitSet Flip(MR.TextureId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_flip_1", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_flip_1(_Underlying *_this, MR.TextureId n);
            return new(__MR_TextureBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::flip`.
        public unsafe MR.TextureBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_flip_0", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_flip_0(_Underlying *_this);
            return new(__MR_TextureBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.TextureId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_test_set(_Underlying *_this, MR.TextureId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_TextureBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::TextureBitSet::operator&=`.
        public unsafe MR.TextureBitSet BitandAssign(MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_bitand_assign(_Underlying *_this, MR.Const_TextureBitSet._Underlying *b);
            return new(__MR_TextureBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::operator|=`.
        public unsafe MR.TextureBitSet BitorAssign(MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_bitor_assign(_Underlying *_this, MR.Const_TextureBitSet._Underlying *b);
            return new(__MR_TextureBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::operator^=`.
        public unsafe MR.TextureBitSet XorAssign(MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_xor_assign(_Underlying *_this, MR.Const_TextureBitSet._Underlying *b);
            return new(__MR_TextureBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::operator-=`.
        public unsafe MR.TextureBitSet SubAssign(MR.Const_TextureBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_sub_assign(_Underlying *_this, MR.Const_TextureBitSet._Underlying *b);
            return new(__MR_TextureBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::TextureBitSet::subtract`.
        public unsafe MR.TextureBitSet Subtract(MR.Const_TextureBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_subtract", ExactSpelling = true)]
            extern static MR.TextureBitSet._Underlying *__MR_TextureBitSet_subtract(_Underlying *_this, MR.Const_TextureBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_TextureBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::TextureBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.TextureId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_autoResizeSet_3(_Underlying *_this, MR.TextureId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_TextureBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::TextureBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.TextureId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_autoResizeSet_2(_Underlying *_this, MR.TextureId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_TextureBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::TextureBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.TextureId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_autoResizeTestSet(_Underlying *_this, MR.TextureId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_TextureBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::TextureBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_TextureBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::TextureBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_resize", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_TextureBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::TextureBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_clear", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_clear(_Underlying *_this);
            __MR_TextureBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_shrink_to_fit(_Underlying *_this);
            __MR_TextureBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_TextureBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_TextureBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::TextureBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_reverse(_Underlying *_this);
            __MR_TextureBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::TextureBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_push_back(_Underlying *_this, byte val);
            __MR_TextureBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::TextureBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_pop_back(_Underlying *_this);
            __MR_TextureBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::TextureBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_TextureBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_TextureBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TextureBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TextureBitSet`/`Const_TextureBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TextureBitSet
    {
        internal readonly Const_TextureBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TextureBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TextureBitSet(Const_TextureBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TextureBitSet(Const_TextureBitSet arg) {return new(arg);}
        public _ByValue_TextureBitSet(MR.Misc._Moved<TextureBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TextureBitSet(MR.Misc._Moved<TextureBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TextureBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TextureBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TextureBitSet`/`Const_TextureBitSet` directly.
    public class _InOptMut_TextureBitSet
    {
        public TextureBitSet? Opt;

        public _InOptMut_TextureBitSet() {}
        public _InOptMut_TextureBitSet(TextureBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_TextureBitSet(TextureBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `TextureBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TextureBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TextureBitSet`/`Const_TextureBitSet` to pass it to the function.
    public class _InOptConst_TextureBitSet
    {
        public Const_TextureBitSet? Opt;

        public _InOptConst_TextureBitSet() {}
        public _InOptConst_TextureBitSet(Const_TextureBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_TextureBitSet(Const_TextureBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::GraphVertBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_GraphVertBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GraphVertBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_Destroy(_Underlying *_this);
            __MR_GraphVertBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GraphVertBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_GraphVertBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_GraphVertBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_GraphVertBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_GraphVertBitSet_Get_bits_per_block();
                return *__MR_GraphVertBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_GraphVertBitSet_Get_npos();
                return *__MR_GraphVertBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GraphVertBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_GraphVertBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        public unsafe Const_GraphVertBitSet(MR._ByValue_GraphVertBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GraphVertBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_GraphVertBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_GraphVertBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_GraphVertBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        public unsafe Const_GraphVertBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphVertBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        public unsafe Const_GraphVertBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphVertBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::test`.
        public unsafe bool Test(MR.GraphVertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_test", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_test(_Underlying *_this, MR.GraphVertId n);
            return __MR_GraphVertBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::GraphVertBitSet::find_first`.
        public unsafe MR.GraphVertId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_find_first", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_find_first(_Underlying *_this);
            return __MR_GraphVertBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::find_next`.
        public unsafe MR.GraphVertId FindNext(MR.GraphVertId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_find_next", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_find_next(_Underlying *_this, MR.GraphVertId pos);
            return __MR_GraphVertBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::GraphVertBitSet::find_last`.
        public unsafe MR.GraphVertId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_find_last", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_find_last(_Underlying *_this);
            return __MR_GraphVertBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::GraphVertBitSet::nthSetBit`.
        public unsafe MR.GraphVertId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_GraphVertBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::GraphVertBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_GraphVertBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_is_subset_of(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *a);
            return __MR_GraphVertBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::GraphVertBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_GraphVertBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_intersects(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *a);
            return __MR_GraphVertBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphVertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphVertBitSet> GetMapping(MR.Const_Vector_MRGraphVertId_MRGraphVertId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_getMapping_1_MR_Vector_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_getMapping_1_MR_Vector_MR_GraphVertId_MR_GraphVertId(_Underlying *_this, MR.Const_Vector_MRGraphVertId_MRGraphVertId._Underlying *map);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_GraphVertBitSet_getMapping_1_MR_Vector_MR_GraphVertId_MR_GraphVertId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphVertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphVertBitSet> GetMapping(MR.Const_BMap_MRGraphVertId_MRGraphVertId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_getMapping_1_MR_BMap_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_getMapping_1_MR_BMap_MR_GraphVertId_MR_GraphVertId(_Underlying *_this, MR.Const_BMap_MRGraphVertId_MRGraphVertId._Underlying *map);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_GraphVertBitSet_getMapping_1_MR_BMap_MR_GraphVertId_MR_GraphVertId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphVertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphVertBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRGraphVertId_MRGraphVertId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_getMapping_1_phmap_flat_hash_map_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_getMapping_1_phmap_flat_hash_map_MR_GraphVertId_MR_GraphVertId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRGraphVertId_MRGraphVertId._Underlying *map);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_GraphVertBitSet_getMapping_1_phmap_flat_hash_map_MR_GraphVertId_MR_GraphVertId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphVertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphVertBitSet> GetMapping(MR.Const_Vector_MRGraphVertId_MRGraphVertId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_getMapping_2_MR_Vector_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_getMapping_2_MR_Vector_MR_GraphVertId_MR_GraphVertId(_Underlying *_this, MR.Const_Vector_MRGraphVertId_MRGraphVertId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_GraphVertBitSet_getMapping_2_MR_Vector_MR_GraphVertId_MR_GraphVertId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::GraphVertBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphVertBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRGraphVertId_MRGraphVertId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_getMapping_2_phmap_flat_hash_map_MR_GraphVertId_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_getMapping_2_phmap_flat_hash_map_MR_GraphVertId_MR_GraphVertId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRGraphVertId_MRGraphVertId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_GraphVertBitSet_getMapping_2_phmap_flat_hash_map_MR_GraphVertId_MR_GraphVertId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::GraphVertBitSet::backId`.
        public unsafe MR.GraphVertId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_backId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_backId(_Underlying *_this);
            return __MR_GraphVertBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::GraphVertBitSet::beginId`.
        public static MR.GraphVertId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_beginId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_beginId();
            return __MR_GraphVertBitSet_beginId();
        }

        /// Generated from method `MR::GraphVertBitSet::endId`.
        public unsafe MR.GraphVertId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_endId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertBitSet_endId(_Underlying *_this);
            return __MR_GraphVertBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::GraphVertBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_GraphVertBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_GraphVertBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphVertBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_empty(_Underlying *_this);
            return __MR_GraphVertBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphVertBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_GraphVertBitSet_size(_Underlying *_this);
            return __MR_GraphVertBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_GraphVertBitSet_num_blocks(_Underlying *_this);
            return __MR_GraphVertBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_GraphVertBitSet_capacity(_Underlying *_this);
            return __MR_GraphVertBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_GraphVertBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::GraphVertBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_GraphVertBitSet_bits(_Underlying *_this);
            return new(__MR_GraphVertBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::GraphVertBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_all", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_all(_Underlying *_this);
            return __MR_GraphVertBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::GraphVertBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_any", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_any(_Underlying *_this);
            return __MR_GraphVertBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::GraphVertBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_none", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_none(_Underlying *_this);
            return __MR_GraphVertBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::GraphVertBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_GraphVertBitSet_count(_Underlying *_this);
            return __MR_GraphVertBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::GraphVertBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_GraphVertBitSet_heapBytes(_Underlying *_this);
            return __MR_GraphVertBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.GraphVertBitSet> operator&(MR.Const_GraphVertBitSet a, MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_GraphVertBitSet", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_bitand_MR_GraphVertBitSet(MR.Const_GraphVertBitSet._Underlying *a, MR.Const_GraphVertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_bitand_MR_GraphVertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.GraphVertBitSet> operator|(MR.Const_GraphVertBitSet a, MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_GraphVertBitSet", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_bitor_MR_GraphVertBitSet(MR.Const_GraphVertBitSet._Underlying *a, MR.Const_GraphVertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_bitor_MR_GraphVertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.GraphVertBitSet> operator^(MR.Const_GraphVertBitSet a, MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_GraphVertBitSet", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_xor_MR_GraphVertBitSet(MR.Const_GraphVertBitSet._Underlying *a, MR.Const_GraphVertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_xor_MR_GraphVertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.GraphVertBitSet> operator-(MR.Const_GraphVertBitSet a, MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_GraphVertBitSet", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_sub_MR_GraphVertBitSet(MR.Const_GraphVertBitSet._Underlying *a, MR.Const_GraphVertBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphVertBitSet(__MR_sub_MR_GraphVertBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::GraphVertBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class GraphVertBitSet : Const_GraphVertBitSet
    {
        internal unsafe GraphVertBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(GraphVertBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_GraphVertBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_GraphVertBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe GraphVertBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_GraphVertBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        public unsafe GraphVertBitSet(MR._ByValue_GraphVertBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GraphVertBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_GraphVertBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe GraphVertBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_GraphVertBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        public unsafe GraphVertBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphVertBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphVertBitSet::GraphVertBitSet`.
        public unsafe GraphVertBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphVertBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::operator=`.
        public unsafe MR.GraphVertBitSet Assign(MR._ByValue_GraphVertBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GraphVertBitSet._Underlying *_other);
            return new(__MR_GraphVertBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::set`.
        public unsafe MR.GraphVertBitSet Set(MR.GraphVertId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_set_3", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_set_3(_Underlying *_this, MR.GraphVertId n, ulong len, byte val);
            return new(__MR_GraphVertBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::set`.
        public unsafe MR.GraphVertBitSet Set(MR.GraphVertId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_set_2", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_set_2(_Underlying *_this, MR.GraphVertId n, byte val);
            return new(__MR_GraphVertBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::set`.
        public unsafe MR.GraphVertBitSet Set(MR.GraphVertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_set_1", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_set_1(_Underlying *_this, MR.GraphVertId n);
            return new(__MR_GraphVertBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::set`.
        public unsafe MR.GraphVertBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_set_0", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_set_0(_Underlying *_this);
            return new(__MR_GraphVertBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::reset`.
        public unsafe MR.GraphVertBitSet Reset(MR.GraphVertId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_reset_2", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_reset_2(_Underlying *_this, MR.GraphVertId n, ulong len);
            return new(__MR_GraphVertBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::reset`.
        public unsafe MR.GraphVertBitSet Reset(MR.GraphVertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_reset_1", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_reset_1(_Underlying *_this, MR.GraphVertId n);
            return new(__MR_GraphVertBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::reset`.
        public unsafe MR.GraphVertBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_reset_0", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_reset_0(_Underlying *_this);
            return new(__MR_GraphVertBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::flip`.
        public unsafe MR.GraphVertBitSet Flip(MR.GraphVertId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_flip_2", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_flip_2(_Underlying *_this, MR.GraphVertId n, ulong len);
            return new(__MR_GraphVertBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::flip`.
        public unsafe MR.GraphVertBitSet Flip(MR.GraphVertId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_flip_1", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_flip_1(_Underlying *_this, MR.GraphVertId n);
            return new(__MR_GraphVertBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::flip`.
        public unsafe MR.GraphVertBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_flip_0", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_flip_0(_Underlying *_this);
            return new(__MR_GraphVertBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.GraphVertId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_test_set(_Underlying *_this, MR.GraphVertId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_GraphVertBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::GraphVertBitSet::operator&=`.
        public unsafe MR.GraphVertBitSet BitandAssign(MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_bitand_assign(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *b);
            return new(__MR_GraphVertBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::operator|=`.
        public unsafe MR.GraphVertBitSet BitorAssign(MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_bitor_assign(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *b);
            return new(__MR_GraphVertBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::operator^=`.
        public unsafe MR.GraphVertBitSet XorAssign(MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_xor_assign(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *b);
            return new(__MR_GraphVertBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::operator-=`.
        public unsafe MR.GraphVertBitSet SubAssign(MR.Const_GraphVertBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_sub_assign(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *b);
            return new(__MR_GraphVertBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::GraphVertBitSet::subtract`.
        public unsafe MR.GraphVertBitSet Subtract(MR.Const_GraphVertBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_subtract", ExactSpelling = true)]
            extern static MR.GraphVertBitSet._Underlying *__MR_GraphVertBitSet_subtract(_Underlying *_this, MR.Const_GraphVertBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_GraphVertBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::GraphVertBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.GraphVertId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_autoResizeSet_3(_Underlying *_this, MR.GraphVertId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_GraphVertBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::GraphVertBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.GraphVertId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_autoResizeSet_2(_Underlying *_this, MR.GraphVertId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_GraphVertBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::GraphVertBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.GraphVertId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_autoResizeTestSet(_Underlying *_this, MR.GraphVertId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_GraphVertBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::GraphVertBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_GraphVertBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::GraphVertBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_resize", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_GraphVertBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::GraphVertBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_clear", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_clear(_Underlying *_this);
            __MR_GraphVertBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_shrink_to_fit(_Underlying *_this);
            __MR_GraphVertBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_GraphVertBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_GraphVertBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::GraphVertBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_reverse(_Underlying *_this);
            __MR_GraphVertBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::GraphVertBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_push_back(_Underlying *_this, byte val);
            __MR_GraphVertBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::GraphVertBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_pop_back(_Underlying *_this);
            __MR_GraphVertBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::GraphVertBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_GraphVertBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_GraphVertBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `GraphVertBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `GraphVertBitSet`/`Const_GraphVertBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_GraphVertBitSet
    {
        internal readonly Const_GraphVertBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_GraphVertBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_GraphVertBitSet(Const_GraphVertBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_GraphVertBitSet(Const_GraphVertBitSet arg) {return new(arg);}
        public _ByValue_GraphVertBitSet(MR.Misc._Moved<GraphVertBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_GraphVertBitSet(MR.Misc._Moved<GraphVertBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `GraphVertBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GraphVertBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GraphVertBitSet`/`Const_GraphVertBitSet` directly.
    public class _InOptMut_GraphVertBitSet
    {
        public GraphVertBitSet? Opt;

        public _InOptMut_GraphVertBitSet() {}
        public _InOptMut_GraphVertBitSet(GraphVertBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_GraphVertBitSet(GraphVertBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `GraphVertBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GraphVertBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GraphVertBitSet`/`Const_GraphVertBitSet` to pass it to the function.
    public class _InOptConst_GraphVertBitSet
    {
        public Const_GraphVertBitSet? Opt;

        public _InOptConst_GraphVertBitSet() {}
        public _InOptConst_GraphVertBitSet(Const_GraphVertBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_GraphVertBitSet(Const_GraphVertBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::GraphEdgeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_GraphEdgeBitSet : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GraphEdgeBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_Destroy(_Underlying *_this);
            __MR_GraphEdgeBitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GraphEdgeBitSet() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_GraphEdgeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_GraphEdgeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_GraphEdgeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_GraphEdgeBitSet_Get_bits_per_block();
                return *__MR_GraphEdgeBitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_GraphEdgeBitSet_Get_npos();
                return *__MR_GraphEdgeBitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GraphEdgeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_GraphEdgeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        public unsafe Const_GraphEdgeBitSet(MR._ByValue_GraphEdgeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GraphEdgeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_GraphEdgeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_GraphEdgeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_GraphEdgeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        public unsafe Const_GraphEdgeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphEdgeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        public unsafe Const_GraphEdgeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::test`.
        public unsafe bool Test(MR.GraphEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_test", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_test(_Underlying *_this, MR.GraphEdgeId n);
            return __MR_GraphEdgeBitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// Generated from method `MR::GraphEdgeBitSet::find_first`.
        public unsafe MR.GraphEdgeId FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_find_first", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_find_first(_Underlying *_this);
            return __MR_GraphEdgeBitSet_find_first(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::find_next`.
        public unsafe MR.GraphEdgeId FindNext(MR.GraphEdgeId pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_find_next", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_find_next(_Underlying *_this, MR.GraphEdgeId pos);
            return __MR_GraphEdgeBitSet_find_next(_UnderlyingPtr, pos);
        }

        /// Generated from method `MR::GraphEdgeBitSet::find_last`.
        public unsafe MR.GraphEdgeId FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_find_last", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_find_last(_Underlying *_this);
            return __MR_GraphEdgeBitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::GraphEdgeBitSet::nthSetBit`.
        public unsafe MR.GraphEdgeId NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_nthSetBit", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_GraphEdgeBitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::GraphEdgeBitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_GraphEdgeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_is_subset_of(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *a);
            return __MR_GraphEdgeBitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::GraphEdgeBitSet::intersects`.
        public unsafe bool Intersects(MR.Const_GraphEdgeBitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_intersects(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *a);
            return __MR_GraphEdgeBitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> GetMapping(MR.Const_Vector_MRGraphEdgeId_MRGraphEdgeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_getMapping_1_MR_Vector_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_getMapping_1_MR_Vector_MR_GraphEdgeId_MR_GraphEdgeId(_Underlying *_this, MR.Const_Vector_MRGraphEdgeId_MRGraphEdgeId._Underlying *map);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_GraphEdgeBitSet_getMapping_1_MR_Vector_MR_GraphEdgeId_MR_GraphEdgeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> GetMapping(MR.Const_BMap_MRGraphEdgeId_MRGraphEdgeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_getMapping_1_MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_getMapping_1_MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId(_Underlying *_this, MR.Const_BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *map);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_GraphEdgeBitSet_getMapping_1_MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRGraphEdgeId_MRGraphEdgeId map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_GraphEdgeId_MR_GraphEdgeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *map);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_GraphEdgeBitSet_getMapping_1_phmap_flat_hash_map_MR_GraphEdgeId_MR_GraphEdgeId(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> GetMapping(MR.Const_Vector_MRGraphEdgeId_MRGraphEdgeId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_getMapping_2_MR_Vector_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_getMapping_2_MR_Vector_MR_GraphEdgeId_MR_GraphEdgeId(_Underlying *_this, MR.Const_Vector_MRGraphEdgeId_MRGraphEdgeId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_GraphEdgeBitSet_getMapping_2_MR_Vector_MR_GraphEdgeId_MR_GraphEdgeId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::GraphEdgeBitSet::getMapping`.
        public unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> GetMapping(MR.Phmap.Const_FlatHashMap_MRGraphEdgeId_MRGraphEdgeId map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_GraphEdgeId_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_GraphEdgeId_MR_GraphEdgeId(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_GraphEdgeBitSet_getMapping_2_phmap_flat_hash_map_MR_GraphEdgeId_MR_GraphEdgeId(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::GraphEdgeBitSet::backId`.
        public unsafe MR.GraphEdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_backId", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_backId(_Underlying *_this);
            return __MR_GraphEdgeBitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::GraphEdgeBitSet::beginId`.
        public static MR.GraphEdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_beginId", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_beginId();
            return __MR_GraphEdgeBitSet_beginId();
        }

        /// Generated from method `MR::GraphEdgeBitSet::endId`.
        public unsafe MR.GraphEdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_endId", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeBitSet_endId(_Underlying *_this);
            return __MR_GraphEdgeBitSet_endId(_UnderlyingPtr);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::GraphEdgeBitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_GraphEdgeBitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_GraphEdgeBitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::GraphEdgeBitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_empty", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_empty(_Underlying *_this);
            return __MR_GraphEdgeBitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphEdgeBitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_size", ExactSpelling = true)]
            extern static ulong __MR_GraphEdgeBitSet_size(_Underlying *_this);
            return __MR_GraphEdgeBitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_GraphEdgeBitSet_num_blocks(_Underlying *_this);
            return __MR_GraphEdgeBitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_GraphEdgeBitSet_capacity(_Underlying *_this);
            return __MR_GraphEdgeBitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_GraphEdgeBitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::GraphEdgeBitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_GraphEdgeBitSet_bits(_Underlying *_this);
            return new(__MR_GraphEdgeBitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::GraphEdgeBitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_all", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_all(_Underlying *_this);
            return __MR_GraphEdgeBitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::GraphEdgeBitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_any", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_any(_Underlying *_this);
            return __MR_GraphEdgeBitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::GraphEdgeBitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_none", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_none(_Underlying *_this);
            return __MR_GraphEdgeBitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::GraphEdgeBitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_count", ExactSpelling = true)]
            extern static ulong __MR_GraphEdgeBitSet_count(_Underlying *_this);
            return __MR_GraphEdgeBitSet_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::GraphEdgeBitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_GraphEdgeBitSet_heapBytes(_Underlying *_this);
            return __MR_GraphEdgeBitSet_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> operator&(MR.Const_GraphEdgeBitSet a, MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_GraphEdgeBitSet", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_bitand_MR_GraphEdgeBitSet(MR.Const_GraphEdgeBitSet._Underlying *a, MR.Const_GraphEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_bitand_MR_GraphEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> operator|(MR.Const_GraphEdgeBitSet a, MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_GraphEdgeBitSet", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_bitor_MR_GraphEdgeBitSet(MR.Const_GraphEdgeBitSet._Underlying *a, MR.Const_GraphEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_bitor_MR_GraphEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> operator^(MR.Const_GraphEdgeBitSet a, MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_GraphEdgeBitSet", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_xor_MR_GraphEdgeBitSet(MR.Const_GraphEdgeBitSet._Underlying *a, MR.Const_GraphEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_xor_MR_GraphEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.GraphEdgeBitSet> operator-(MR.Const_GraphEdgeBitSet a, MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_GraphEdgeBitSet", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_sub_MR_GraphEdgeBitSet(MR.Const_GraphEdgeBitSet._Underlying *a, MR.Const_GraphEdgeBitSet._Underlying *b);
            return MR.Misc.Move(new MR.GraphEdgeBitSet(__MR_sub_MR_GraphEdgeBitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::GraphEdgeBitSet`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class GraphEdgeBitSet : Const_GraphEdgeBitSet
    {
        internal unsafe GraphEdgeBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(GraphEdgeBitSet self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_GraphEdgeBitSet_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_GraphEdgeBitSet_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe GraphEdgeBitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_GraphEdgeBitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        public unsafe GraphEdgeBitSet(MR._ByValue_GraphEdgeBitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GraphEdgeBitSet._Underlying *_other);
            _UnderlyingPtr = __MR_GraphEdgeBitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe GraphEdgeBitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Construct_2", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_GraphEdgeBitSet_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        public unsafe GraphEdgeBitSet(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphEdgeBitSet_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::GraphEdgeBitSet::GraphEdgeBitSet`.
        public unsafe GraphEdgeBitSet(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_GraphEdgeBitSet_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::operator=`.
        public unsafe MR.GraphEdgeBitSet Assign(MR._ByValue_GraphEdgeBitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GraphEdgeBitSet._Underlying *_other);
            return new(__MR_GraphEdgeBitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::set`.
        public unsafe MR.GraphEdgeBitSet Set(MR.GraphEdgeId n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_set_3", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_set_3(_Underlying *_this, MR.GraphEdgeId n, ulong len, byte val);
            return new(__MR_GraphEdgeBitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::set`.
        public unsafe MR.GraphEdgeBitSet Set(MR.GraphEdgeId n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_set_2", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_set_2(_Underlying *_this, MR.GraphEdgeId n, byte val);
            return new(__MR_GraphEdgeBitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::set`.
        public unsafe MR.GraphEdgeBitSet Set(MR.GraphEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_set_1", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_set_1(_Underlying *_this, MR.GraphEdgeId n);
            return new(__MR_GraphEdgeBitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::set`.
        public unsafe MR.GraphEdgeBitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_set_0", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_set_0(_Underlying *_this);
            return new(__MR_GraphEdgeBitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::reset`.
        public unsafe MR.GraphEdgeBitSet Reset(MR.GraphEdgeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_reset_2", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_reset_2(_Underlying *_this, MR.GraphEdgeId n, ulong len);
            return new(__MR_GraphEdgeBitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::reset`.
        public unsafe MR.GraphEdgeBitSet Reset(MR.GraphEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_reset_1", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_reset_1(_Underlying *_this, MR.GraphEdgeId n);
            return new(__MR_GraphEdgeBitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::reset`.
        public unsafe MR.GraphEdgeBitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_reset_0", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_reset_0(_Underlying *_this);
            return new(__MR_GraphEdgeBitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::flip`.
        public unsafe MR.GraphEdgeBitSet Flip(MR.GraphEdgeId n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_flip_2", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_flip_2(_Underlying *_this, MR.GraphEdgeId n, ulong len);
            return new(__MR_GraphEdgeBitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::flip`.
        public unsafe MR.GraphEdgeBitSet Flip(MR.GraphEdgeId n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_flip_1", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_flip_1(_Underlying *_this, MR.GraphEdgeId n);
            return new(__MR_GraphEdgeBitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::flip`.
        public unsafe MR.GraphEdgeBitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_flip_0", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_flip_0(_Underlying *_this);
            return new(__MR_GraphEdgeBitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.GraphEdgeId n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_test_set(_Underlying *_this, MR.GraphEdgeId n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_GraphEdgeBitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::GraphEdgeBitSet::operator&=`.
        public unsafe MR.GraphEdgeBitSet BitandAssign(MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_bitand_assign(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *b);
            return new(__MR_GraphEdgeBitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::operator|=`.
        public unsafe MR.GraphEdgeBitSet BitorAssign(MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_bitor_assign(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *b);
            return new(__MR_GraphEdgeBitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::operator^=`.
        public unsafe MR.GraphEdgeBitSet XorAssign(MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_xor_assign", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_xor_assign(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *b);
            return new(__MR_GraphEdgeBitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::operator-=`.
        public unsafe MR.GraphEdgeBitSet SubAssign(MR.Const_GraphEdgeBitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_sub_assign", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_sub_assign(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *b);
            return new(__MR_GraphEdgeBitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::GraphEdgeBitSet::subtract`.
        public unsafe MR.GraphEdgeBitSet Subtract(MR.Const_GraphEdgeBitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_subtract", ExactSpelling = true)]
            extern static MR.GraphEdgeBitSet._Underlying *__MR_GraphEdgeBitSet_subtract(_Underlying *_this, MR.Const_GraphEdgeBitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_GraphEdgeBitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.GraphEdgeId pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_autoResizeSet_3(_Underlying *_this, MR.GraphEdgeId pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_GraphEdgeBitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::GraphEdgeBitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.GraphEdgeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_autoResizeSet_2(_Underlying *_this, MR.GraphEdgeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_GraphEdgeBitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::GraphEdgeBitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.GraphEdgeId pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_autoResizeTestSet(_Underlying *_this, MR.GraphEdgeId pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_GraphEdgeBitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::GraphEdgeBitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_reserve", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_GraphEdgeBitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::GraphEdgeBitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_resize", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_GraphEdgeBitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::GraphEdgeBitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_clear", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_clear(_Underlying *_this);
            __MR_GraphEdgeBitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_shrink_to_fit(_Underlying *_this);
            __MR_GraphEdgeBitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeBitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeBitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_GraphEdgeBitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::GraphEdgeBitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_reverse", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_reverse(_Underlying *_this);
            __MR_GraphEdgeBitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::GraphEdgeBitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_push_back", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_push_back(_Underlying *_this, byte val);
            __MR_GraphEdgeBitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::GraphEdgeBitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_pop_back(_Underlying *_this);
            __MR_GraphEdgeBitSet_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::GraphEdgeBitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeBitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_GraphEdgeBitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_GraphEdgeBitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `GraphEdgeBitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `GraphEdgeBitSet`/`Const_GraphEdgeBitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_GraphEdgeBitSet
    {
        internal readonly Const_GraphEdgeBitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_GraphEdgeBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_GraphEdgeBitSet(Const_GraphEdgeBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_GraphEdgeBitSet(Const_GraphEdgeBitSet arg) {return new(arg);}
        public _ByValue_GraphEdgeBitSet(MR.Misc._Moved<GraphEdgeBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_GraphEdgeBitSet(MR.Misc._Moved<GraphEdgeBitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `GraphEdgeBitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GraphEdgeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GraphEdgeBitSet`/`Const_GraphEdgeBitSet` directly.
    public class _InOptMut_GraphEdgeBitSet
    {
        public GraphEdgeBitSet? Opt;

        public _InOptMut_GraphEdgeBitSet() {}
        public _InOptMut_GraphEdgeBitSet(GraphEdgeBitSet value) {Opt = value;}
        public static implicit operator _InOptMut_GraphEdgeBitSet(GraphEdgeBitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `GraphEdgeBitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GraphEdgeBitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GraphEdgeBitSet`/`Const_GraphEdgeBitSet` to pass it to the function.
    public class _InOptConst_GraphEdgeBitSet
    {
        public Const_GraphEdgeBitSet? Opt;

        public _InOptConst_GraphEdgeBitSet() {}
        public _InOptConst_GraphEdgeBitSet(Const_GraphEdgeBitSet value) {Opt = value;}
        public static implicit operator _InOptConst_GraphEdgeBitSet(Const_GraphEdgeBitSet value) {return new(value);}
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the const half of the class.
    public class Const_TypedBitSet_MRIdMRICPElemtTag : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TypedBitSet_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Destroy", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Destroy(_Underlying *_this);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TypedBitSet_MRIdMRICPElemtTag() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BitSet(Const_TypedBitSet_MRIdMRICPElemtTag self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.Const_BitSet._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.Const_BitSet ret = new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Get_bits_per_block();
                return *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Get_npos();
                return *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TypedBitSet_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        public unsafe Const_TypedBitSet_MRIdMRICPElemtTag(MR._ByValue_TypedBitSet_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_TypedBitSet_MRIdMRICPElemtTag(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_2", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        public unsafe Const_TypedBitSet_MRIdMRICPElemtTag(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        public unsafe Const_TypedBitSet_MRIdMRICPElemtTag(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::test`.
        public unsafe bool Test(MR.Const_Id_MRICPElemtTag n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_test", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_test(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_test(_UnderlyingPtr, n._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::find_first`.
        public unsafe MR.Id_MRICPElemtTag FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_first", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_first(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_first(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::find_next`.
        public unsafe MR.Id_MRICPElemtTag FindNext(MR.Const_Id_MRICPElemtTag pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_next", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_next(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *pos);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_next(_UnderlyingPtr, pos._UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::find_last`.
        public unsafe MR.Id_MRICPElemtTag FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_last", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_last(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_find_last(_UnderlyingPtr), is_owning: true);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or IndexType(npos) if there are less bit set
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::nthSetBit`.
        public unsafe MR.Id_MRICPElemtTag NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_nthSetBit", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_nthSetBit(_Underlying *_this, ulong n);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_nthSetBit(_UnderlyingPtr, n), is_owning: true);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_TypedBitSet_MRIdMRICPElemtTag a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_is_subset_of(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *a);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::intersects`.
        public unsafe bool Intersects(MR.Const_TypedBitSet_MRIdMRICPElemtTag a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_intersects", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_intersects(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *a);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::getMapping`.
        public unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> GetMapping(MR.Const_Vector_MRIdMRICPElemtTag_MRIdMRICPElemtTag map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_MR_Vector_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_MR_Vector_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Const_Vector_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *map);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_MR_Vector_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::getMapping`.
        public unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> GetMapping(MR.Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *map);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::getMapping`.
        public unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> GetMapping(MR.Phmap.Const_FlatHashMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_phmap_flat_hash_map_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_phmap_flat_hash_map_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *map);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_1_phmap_flat_hash_map_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, map._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::getMapping`.
        public unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> GetMapping(MR.Const_Vector_MRIdMRICPElemtTag_MRIdMRICPElemtTag map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_2_MR_Vector_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_2_MR_Vector_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Const_Vector_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_2_MR_Vector_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::getMapping`.
        public unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> GetMapping(MR.Phmap.Const_FlatHashMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag map, ulong resSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_2_phmap_flat_hash_map_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_2_phmap_flat_hash_map_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Phmap.Const_FlatHashMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *map, ulong resSize);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_getMapping_2_phmap_flat_hash_map_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, map._UnderlyingPtr, resSize), is_owning: true));
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::backId`.
        public unsafe MR.Id_MRICPElemtTag BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_backId", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_backId(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_backId(_UnderlyingPtr), is_owning: true);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::beginId`.
        public static unsafe MR.Id_MRICPElemtTag BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_beginId", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_beginId();
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_beginId(), is_owning: true);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::endId`.
        public unsafe MR.Id_MRICPElemtTag EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_endId", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_endId(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_endId(_UnderlyingPtr), is_owning: true);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_empty", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_empty(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_size", ExactSpelling = true)]
            extern static ulong __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_size(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_num_blocks(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_capacity", ExactSpelling = true)]
            extern static ulong __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_capacity(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bits(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_all", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_all(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_any", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_any(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_none", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_none(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_count", ExactSpelling = true)]
            extern static ulong __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_count(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_count(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_heapBytes(_Underlying *_this);
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_heapBytes(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> operator&(MR.Const_TypedBitSet_MRIdMRICPElemtTag a, MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_TypedBitSet_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_bitand_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *a, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_bitand_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> operator|(MR.Const_TypedBitSet_MRIdMRICPElemtTag a, MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_TypedBitSet_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_bitor_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *a, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_bitor_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> operator^(MR.Const_TypedBitSet_MRIdMRICPElemtTag a, MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_TypedBitSet_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_xor_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *a, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_xor_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.TypedBitSet_MRIdMRICPElemtTag> operator-(MR.Const_TypedBitSet_MRIdMRICPElemtTag a, MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_TypedBitSet_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_sub_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *a, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return MR.Misc.Move(new MR.TypedBitSet_MRIdMRICPElemtTag(__MR_sub_MR_TypedBitSet_MR_Id_MR_ICPElemtTag(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// Vector<bool, I> like container (random-access, I - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BitSet`
    /// This is the non-const half of the class.
    public class TypedBitSet_MRIdMRICPElemtTag : Const_TypedBitSet_MRIdMRICPElemtTag
    {
        internal unsafe TypedBitSet_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BitSet(TypedBitSet_MRIdMRICPElemtTag self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_UpcastTo_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_UpcastTo_MR_BitSet(_Underlying *_this);
            MR.BitSet ret = new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_UpcastTo_MR_BitSet(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TypedBitSet_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        public unsafe TypedBitSet_MRIdMRICPElemtTag(MR._ByValue_TypedBitSet_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe TypedBitSet_MRIdMRICPElemtTag(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_2", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_2(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_2(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// copies all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        public unsafe TypedBitSet_MRIdMRICPElemtTag(MR.Const_BitSet src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_const_MR_BitSet_ref", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_const_MR_BitSet_ref(MR.Const_BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_const_MR_BitSet_ref(src._UnderlyingPtr);
        }

        /// moves all bits from another BitSet (or a descending class, e.g. TypedBitSet<U>)
        /// Generated from constructor `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::TypedBitSet`.
        public unsafe TypedBitSet_MRIdMRICPElemtTag(MR.Misc._Moved<MR.BitSet> src) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_MR_BitSet_rvalue_ref", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_MR_BitSet_rvalue_ref(MR.BitSet._Underlying *src);
            _UnderlyingPtr = __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_Construct_1_MR_BitSet_rvalue_ref(src.Value._UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::operator=`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Assign(MR._ByValue_TypedBitSet_MRIdMRICPElemtTag _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *_other);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::set`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Set(MR.Const_Id_MRICPElemtTag n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_3", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_3(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n, ulong len, byte val);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_3(_UnderlyingPtr, n._UnderlyingPtr, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::set`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Set(MR.Const_Id_MRICPElemtTag n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_2", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_2(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n, byte val);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_2(_UnderlyingPtr, n._UnderlyingPtr, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::set`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Set(MR.Const_Id_MRICPElemtTag n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_1", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_1(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_1(_UnderlyingPtr, n._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::set`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_0", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_0(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::reset`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Reset(MR.Const_Id_MRICPElemtTag n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_2", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_2(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n, ulong len);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_2(_UnderlyingPtr, n._UnderlyingPtr, len), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::reset`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Reset(MR.Const_Id_MRICPElemtTag n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_1", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_1(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_1(_UnderlyingPtr, n._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::reset`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_0", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_0(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::flip`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Flip(MR.Const_Id_MRICPElemtTag n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_2", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_2(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n, ulong len);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_2(_UnderlyingPtr, n._UnderlyingPtr, len), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::flip`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Flip(MR.Const_Id_MRICPElemtTag n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_1", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_1(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_1(_UnderlyingPtr, n._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::flip`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_0", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_0(_Underlying *_this);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(MR.Const_Id_MRICPElemtTag n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_test_set", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_test_set(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_test_set(_UnderlyingPtr, n._UnderlyingPtr, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::operator&=`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag BitandAssign(MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bitand_assign", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bitand_assign(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::operator|=`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag BitorAssign(MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bitor_assign", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bitor_assign(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::operator^=`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag XorAssign(MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_xor_assign", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_xor_assign(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::operator-=`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag SubAssign(MR.Const_TypedBitSet_MRIdMRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_sub_assign", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_sub_assign(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::subtract`.
        public unsafe MR.TypedBitSet_MRIdMRICPElemtTag Subtract(MR.Const_TypedBitSet_MRIdMRICPElemtTag b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_subtract", ExactSpelling = true)]
            extern static MR.TypedBitSet_MRIdMRICPElemtTag._Underlying *__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_subtract(_Underlying *_this, MR.Const_TypedBitSet_MRIdMRICPElemtTag._Underlying *b, int bShiftInBlocks);
            return new(__MR_TypedBitSet_MR_Id_MR_ICPElemtTag_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.Const_Id_MRICPElemtTag pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeSet_3(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeSet_3(_UnderlyingPtr, pos._UnderlyingPtr, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(MR.Const_Id_MRICPElemtTag pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeSet_2(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeSet_2(_UnderlyingPtr, pos._UnderlyingPtr, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(MR.Const_Id_MRICPElemtTag pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeTestSet(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_autoResizeTestSet(_UnderlyingPtr, pos._UnderlyingPtr, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reserve", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reserve(_Underlying *_this, ulong numBits);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_resize", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_clear", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_clear(_Underlying *_this);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_shrink_to_fit(_Underlying *_this);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reverse", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reverse(_Underlying *_this);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_push_back", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_push_back(_Underlying *_this, byte val);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_pop_back", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_pop_back(_Underlying *_this);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_pop_back(_UnderlyingPtr);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TypedBitSet_MR_Id_MR_ICPElemtTag_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_TypedBitSet_MR_Id_MR_ICPElemtTag_resizeWithReserve(_UnderlyingPtr, newSize);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TypedBitSet_MRIdMRICPElemtTag` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TypedBitSet_MRIdMRICPElemtTag`/`Const_TypedBitSet_MRIdMRICPElemtTag` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TypedBitSet_MRIdMRICPElemtTag
    {
        internal readonly Const_TypedBitSet_MRIdMRICPElemtTag? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TypedBitSet_MRIdMRICPElemtTag() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TypedBitSet_MRIdMRICPElemtTag(Const_TypedBitSet_MRIdMRICPElemtTag new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TypedBitSet_MRIdMRICPElemtTag(Const_TypedBitSet_MRIdMRICPElemtTag arg) {return new(arg);}
        public _ByValue_TypedBitSet_MRIdMRICPElemtTag(MR.Misc._Moved<TypedBitSet_MRIdMRICPElemtTag> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TypedBitSet_MRIdMRICPElemtTag(MR.Misc._Moved<TypedBitSet_MRIdMRICPElemtTag> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TypedBitSet_MRIdMRICPElemtTag` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TypedBitSet_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TypedBitSet_MRIdMRICPElemtTag`/`Const_TypedBitSet_MRIdMRICPElemtTag` directly.
    public class _InOptMut_TypedBitSet_MRIdMRICPElemtTag
    {
        public TypedBitSet_MRIdMRICPElemtTag? Opt;

        public _InOptMut_TypedBitSet_MRIdMRICPElemtTag() {}
        public _InOptMut_TypedBitSet_MRIdMRICPElemtTag(TypedBitSet_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptMut_TypedBitSet_MRIdMRICPElemtTag(TypedBitSet_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// This is used for optional parameters of class `TypedBitSet_MRIdMRICPElemtTag` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TypedBitSet_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TypedBitSet_MRIdMRICPElemtTag`/`Const_TypedBitSet_MRIdMRICPElemtTag` to pass it to the function.
    public class _InOptConst_TypedBitSet_MRIdMRICPElemtTag
    {
        public Const_TypedBitSet_MRIdMRICPElemtTag? Opt;

        public _InOptConst_TypedBitSet_MRIdMRICPElemtTag() {}
        public _InOptConst_TypedBitSet_MRIdMRICPElemtTag(Const_TypedBitSet_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptConst_TypedBitSet_MRIdMRICPElemtTag(Const_TypedBitSet_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// std::vector<bool> like container (random-access, size_t - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::BitSet`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgeBitSet`
    ///     `MR::FaceBitSet`
    ///     `MR::GraphEdgeBitSet`
    ///     `MR::GraphVertBitSet`
    ///     `MR::NodeBitSet`
    ///     `MR::ObjBitSet`
    ///     `MR::PixelBitSet`
    ///     `MR::RegionBitSet`
    ///     `MR::TextureBitSet`
    ///     `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>`
    ///     `MR::UndirectedEdgeBitSet`
    ///     `MR::VertBitSet`
    ///     `MR::VoxelBitSet`
    /// This is the const half of the class.
    public class Const_BitSet : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_BitSet>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_Destroy", ExactSpelling = true)]
            extern static void __MR_BitSet_Destroy(_Underlying *_this);
            __MR_BitSet_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BitSet() {Dispose(false);}

        public static unsafe ulong BitsPerBlock
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_Get_bits_per_block", ExactSpelling = true)]
                extern static ulong *__MR_BitSet_Get_bits_per_block();
                return *__MR_BitSet_Get_bits_per_block();
            }
        }

        public static unsafe ulong Npos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_Get_npos", ExactSpelling = true)]
                extern static ulong *__MR_BitSet_Get_npos();
                return *__MR_BitSet_Get_npos();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_BitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::BitSet::BitSet`.
        public unsafe Const_BitSet(MR._ByValue_BitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BitSet._Underlying *_other);
            _UnderlyingPtr = __MR_BitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::BitSet::BitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe Const_BitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_Construct", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_Construct(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_BitSet_Construct(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// creates bitset from the given blocks of bits
        /// Generated from method `MR::BitSet::fromBlocks`.
        public static unsafe MR.Misc._Moved<MR.BitSet> FromBlocks(MR.Misc._Moved<MR.Std.Vector_MRUint64T> blocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_fromBlocks", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_fromBlocks(MR.Std.Vector_MRUint64T._Underlying *blocks);
            return MR.Misc.Move(new MR.BitSet(__MR_BitSet_fromBlocks(blocks.Value._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::BitSet::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_empty", ExactSpelling = true)]
            extern static byte __MR_BitSet_empty(_Underlying *_this);
            return __MR_BitSet_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::BitSet::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_size", ExactSpelling = true)]
            extern static ulong __MR_BitSet_size(_Underlying *_this);
            return __MR_BitSet_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::BitSet::num_blocks`.
        public unsafe ulong NumBlocks()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_num_blocks", ExactSpelling = true)]
            extern static ulong __MR_BitSet_num_blocks(_Underlying *_this);
            return __MR_BitSet_num_blocks(_UnderlyingPtr);
        }

        /// Generated from method `MR::BitSet::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_capacity", ExactSpelling = true)]
            extern static ulong __MR_BitSet_capacity(_Underlying *_this);
            return __MR_BitSet_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::BitSet::uncheckedTest`.
        public unsafe bool UncheckedTest(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_uncheckedTest", ExactSpelling = true)]
            extern static byte __MR_BitSet_uncheckedTest(_Underlying *_this, ulong n);
            return __MR_BitSet_uncheckedTest(_UnderlyingPtr, n) != 0;
        }

        // all bits after size() we silently consider as not-set
        /// Generated from method `MR::BitSet::test`.
        public unsafe bool Test(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_test", ExactSpelling = true)]
            extern static byte __MR_BitSet_test(_Underlying *_this, ulong n);
            return __MR_BitSet_test(_UnderlyingPtr, n) != 0;
        }

        /// read-only access to all bits stored as a vector of uint64 blocks
        /// Generated from method `MR::BitSet::bits`.
        public unsafe MR.Std.Const_Vector_MRUint64T Bits()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_bits", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_BitSet_bits(_Underlying *_this);
            return new(__MR_BitSet_bits(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if all bits in this container are set
        /// Generated from method `MR::BitSet::all`.
        public unsafe bool All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_all", ExactSpelling = true)]
            extern static byte __MR_BitSet_all(_Underlying *_this);
            return __MR_BitSet_all(_UnderlyingPtr) != 0;
        }

        /// returns true if at least one bits in this container is set
        /// Generated from method `MR::BitSet::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_any", ExactSpelling = true)]
            extern static byte __MR_BitSet_any(_Underlying *_this);
            return __MR_BitSet_any(_UnderlyingPtr) != 0;
        }

        /// returns true if all bits in this container are reset
        /// Generated from method `MR::BitSet::none`.
        public unsafe bool None()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_none", ExactSpelling = true)]
            extern static byte __MR_BitSet_none(_Underlying *_this);
            return __MR_BitSet_none(_UnderlyingPtr) != 0;
        }

        /// computes the number of set bits in the whole set
        /// Generated from method `MR::BitSet::count`.
        public unsafe ulong Count()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_count", ExactSpelling = true)]
            extern static ulong __MR_BitSet_count(_Underlying *_this);
            return __MR_BitSet_count(_UnderlyingPtr);
        }

        /// return the smallest index i such that bit i is set, or npos if *this has no on bits.
        /// Generated from method `MR::BitSet::find_first`.
        public unsafe ulong FindFirst()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_find_first", ExactSpelling = true)]
            extern static ulong __MR_BitSet_find_first(_Underlying *_this);
            return __MR_BitSet_find_first(_UnderlyingPtr);
        }

        /// return the smallest index i>n such that bit i is set, or npos if *this has no on bits.
        /// Generated from method `MR::BitSet::find_next`.
        public unsafe ulong FindNext(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_find_next", ExactSpelling = true)]
            extern static ulong __MR_BitSet_find_next(_Underlying *_this, ulong n);
            return __MR_BitSet_find_next(_UnderlyingPtr, n);
        }

        /// return the highest index i such that bit i is set, or npos if *this has no on bits.
        /// Generated from method `MR::BitSet::find_last`.
        public unsafe ulong FindLast()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_find_last", ExactSpelling = true)]
            extern static ulong __MR_BitSet_find_last(_Underlying *_this);
            return __MR_BitSet_find_last(_UnderlyingPtr);
        }

        /// returns the location of nth set bit (where the first bit corresponds to n=0) or npos if there are less bit set
        /// Generated from method `MR::BitSet::nthSetBit`.
        public unsafe ulong NthSetBit(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_nthSetBit", ExactSpelling = true)]
            extern static ulong __MR_BitSet_nthSetBit(_Underlying *_this, ulong n);
            return __MR_BitSet_nthSetBit(_UnderlyingPtr, n);
        }

        /// returns true if, for every bit that is set in this bitset, the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::BitSet::is_subset_of`.
        public unsafe bool IsSubsetOf(MR.Const_BitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_is_subset_of", ExactSpelling = true)]
            extern static byte __MR_BitSet_is_subset_of(_Underlying *_this, MR.Const_BitSet._Underlying *a);
            return __MR_BitSet_is_subset_of(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns true if, there is a bit which is set in this bitset, such that the corresponding bit in bitset a is also set. Otherwise this function returns false.
        /// Generated from method `MR::BitSet::intersects`.
        public unsafe bool Intersects(MR.Const_BitSet a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_intersects", ExactSpelling = true)]
            extern static byte __MR_BitSet_intersects(_Underlying *_this, MR.Const_BitSet._Underlying *a);
            return __MR_BitSet_intersects(_UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::BitSet::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_BitSet_heapBytes(_Underlying *_this);
            return __MR_BitSet_heapBytes(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::BitSet::backId`.
        public unsafe ulong BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_backId", ExactSpelling = true)]
            extern static ulong __MR_BitSet_backId(_Underlying *_this);
            return __MR_BitSet_backId(_UnderlyingPtr);
        }

        /// [beginId(), endId()) is the range of all bits in the set
        /// Generated from method `MR::BitSet::beginId`.
        public static ulong BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_beginId", ExactSpelling = true)]
            extern static ulong __MR_BitSet_beginId();
            return __MR_BitSet_beginId();
        }

        /// Generated from method `MR::BitSet::endId`.
        public unsafe ulong EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_endId", ExactSpelling = true)]
            extern static ulong __MR_BitSet_endId(_Underlying *_this);
            return __MR_BitSet_endId(_UnderlyingPtr);
        }

        /// compare that two bit sets have the same set bits (they can be equal even if sizes are distinct but last bits are off)
        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_BitSet a, MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_BitSet", ExactSpelling = true)]
            extern static byte __MR_equal_MR_BitSet(MR.Const_BitSet._Underlying *a, MR.Const_BitSet._Underlying *b);
            return __MR_equal_MR_BitSet(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_BitSet a, MR.Const_BitSet b)
        {
            return !(a == b);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.Misc._Moved<MR.BitSet> operator&(MR.Const_BitSet a, MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_bitand_MR_BitSet(MR.Const_BitSet._Underlying *a, MR.Const_BitSet._Underlying *b);
            return MR.Misc.Move(new MR.BitSet(__MR_bitand_MR_BitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.Misc._Moved<MR.BitSet> operator|(MR.Const_BitSet a, MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_bitor_MR_BitSet(MR.Const_BitSet._Underlying *a, MR.Const_BitSet._Underlying *b);
            return MR.Misc.Move(new MR.BitSet(__MR_bitor_MR_BitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.Misc._Moved<MR.BitSet> operator^(MR.Const_BitSet a, MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_xor_MR_BitSet(MR.Const_BitSet._Underlying *a, MR.Const_BitSet._Underlying *b);
            return MR.Misc.Move(new MR.BitSet(__MR_xor_MR_BitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        /// Generated from function `MR::operator-`.
        public static unsafe MR.Misc._Moved<MR.BitSet> operator-(MR.Const_BitSet a, MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_BitSet", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_sub_MR_BitSet(MR.Const_BitSet._Underlying *a, MR.Const_BitSet._Underlying *b);
            return MR.Misc.Move(new MR.BitSet(__MR_sub_MR_BitSet(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        // IEquatable:

        public bool Equals(MR.Const_BitSet? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_BitSet)
                return this == (MR.Const_BitSet)other;
            return false;
        }
    }

    /// std::vector<bool> like container (random-access, size_t - index type, bool - value type)
    /// with all bits after size() considered off during testing
    /// Generated from class `MR::BitSet`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgeBitSet`
    ///     `MR::FaceBitSet`
    ///     `MR::GraphEdgeBitSet`
    ///     `MR::GraphVertBitSet`
    ///     `MR::NodeBitSet`
    ///     `MR::ObjBitSet`
    ///     `MR::PixelBitSet`
    ///     `MR::RegionBitSet`
    ///     `MR::TextureBitSet`
    ///     `MR::TypedBitSet<MR::Id<MR::ICPElemtTag>>`
    ///     `MR::UndirectedEdgeBitSet`
    ///     `MR::VertBitSet`
    ///     `MR::VoxelBitSet`
    /// This is the non-const half of the class.
    public class BitSet : Const_BitSet
    {
        internal unsafe BitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe BitSet() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_DefaultConstruct();
            _UnderlyingPtr = __MR_BitSet_DefaultConstruct();
        }

        /// Generated from constructor `MR::BitSet::BitSet`.
        public unsafe BitSet(MR._ByValue_BitSet _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BitSet._Underlying *_other);
            _UnderlyingPtr = __MR_BitSet_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates bitset of given size filled with given value
        /// Generated from constructor `MR::BitSet::BitSet`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe BitSet(ulong numBits, bool? fillValue = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_Construct", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_Construct(ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_BitSet_Construct(numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::BitSet::operator=`.
        public unsafe MR.BitSet Assign(MR._ByValue_BitSet _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BitSet._Underlying *_other);
            return new(__MR_BitSet_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::BitSet::reserve`.
        public unsafe void Reserve(ulong numBits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_reserve", ExactSpelling = true)]
            extern static void __MR_BitSet_reserve(_Underlying *_this, ulong numBits);
            __MR_BitSet_reserve(_UnderlyingPtr, numBits);
        }

        /// Generated from method `MR::BitSet::resize`.
        /// Parameter `fillValue` defaults to `false`.
        public unsafe void Resize(ulong numBits, bool? fillValue = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_resize", ExactSpelling = true)]
            extern static void __MR_BitSet_resize(_Underlying *_this, ulong numBits, byte *fillValue);
            byte __deref_fillValue = fillValue.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_BitSet_resize(_UnderlyingPtr, numBits, fillValue.HasValue ? &__deref_fillValue : null);
        }

        /// Generated from method `MR::BitSet::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_clear", ExactSpelling = true)]
            extern static void __MR_BitSet_clear(_Underlying *_this);
            __MR_BitSet_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::BitSet::shrink_to_fit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_shrink_to_fit", ExactSpelling = true)]
            extern static void __MR_BitSet_shrink_to_fit(_Underlying *_this);
            __MR_BitSet_shrink_to_fit(_UnderlyingPtr);
        }

        /// Generated from method `MR::BitSet::uncheckedTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool UncheckedTestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_uncheckedTestSet", ExactSpelling = true)]
            extern static byte __MR_BitSet_uncheckedTestSet(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_BitSet_uncheckedTestSet(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::BitSet::test_set`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool TestSet(ulong n, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_test_set", ExactSpelling = true)]
            extern static byte __MR_BitSet_test_set(_Underlying *_this, ulong n, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_BitSet_test_set(_UnderlyingPtr, n, val.HasValue ? &__deref_val : null) != 0;
        }

        /// Generated from method `MR::BitSet::set`.
        public unsafe MR.BitSet Set(ulong n, ulong len, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_set_3", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_set_3(_Underlying *_this, ulong n, ulong len, byte val);
            return new(__MR_BitSet_set_3(_UnderlyingPtr, n, len, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::BitSet::set`.
        public unsafe MR.BitSet Set(ulong n, bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_set_2", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_set_2(_Underlying *_this, ulong n, byte val);
            return new(__MR_BitSet_set_2(_UnderlyingPtr, n, val ? (byte)1 : (byte)0), is_owning: false);
        }

        /// Generated from method `MR::BitSet::set`.
        public unsafe MR.BitSet Set(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_set_1", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_set_1(_Underlying *_this, ulong n);
            return new(__MR_BitSet_set_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::BitSet::set`.
        public unsafe MR.BitSet Set()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_set_0", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_set_0(_Underlying *_this);
            return new(__MR_BitSet_set_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BitSet::reset`.
        public unsafe MR.BitSet Reset(ulong n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_reset_2", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_reset_2(_Underlying *_this, ulong n, ulong len);
            return new(__MR_BitSet_reset_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::BitSet::reset`.
        public unsafe MR.BitSet Reset(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_reset_1", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_reset_1(_Underlying *_this, ulong n);
            return new(__MR_BitSet_reset_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::BitSet::reset`.
        public unsafe MR.BitSet Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_reset_0", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_reset_0(_Underlying *_this);
            return new(__MR_BitSet_reset_0(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BitSet::flip`.
        public unsafe MR.BitSet Flip(ulong n, ulong len)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_flip_2", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_flip_2(_Underlying *_this, ulong n, ulong len);
            return new(__MR_BitSet_flip_2(_UnderlyingPtr, n, len), is_owning: false);
        }

        /// Generated from method `MR::BitSet::flip`.
        public unsafe MR.BitSet Flip(ulong n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_flip_1", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_flip_1(_Underlying *_this, ulong n);
            return new(__MR_BitSet_flip_1(_UnderlyingPtr, n), is_owning: false);
        }

        /// Generated from method `MR::BitSet::flip`.
        public unsafe MR.BitSet Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_flip_0", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_flip_0(_Underlying *_this);
            return new(__MR_BitSet_flip_0(_UnderlyingPtr), is_owning: false);
        }

        /// changes the order of bits on the opposite
        /// Generated from method `MR::BitSet::reverse`.
        public unsafe void Reverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_reverse", ExactSpelling = true)]
            extern static void __MR_BitSet_reverse(_Underlying *_this);
            __MR_BitSet_reverse(_UnderlyingPtr);
        }

        /// adds one more bit with the given value in the container, increasing its size on 1
        /// Generated from method `MR::BitSet::push_back`.
        public unsafe void PushBack(bool val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_push_back", ExactSpelling = true)]
            extern static void __MR_BitSet_push_back(_Underlying *_this, byte val);
            __MR_BitSet_push_back(_UnderlyingPtr, val ? (byte)1 : (byte)0);
        }

        /// removes last bit from the container, decreasing its size on 1
        /// Generated from method `MR::BitSet::pop_back`.
        public unsafe void PopBack()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_pop_back", ExactSpelling = true)]
            extern static void __MR_BitSet_pop_back(_Underlying *_this);
            __MR_BitSet_pop_back(_UnderlyingPtr);
        }

        /// Generated from method `MR::BitSet::operator&=`.
        public unsafe MR.BitSet BitandAssign(MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_bitand_assign", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_bitand_assign(_Underlying *_this, MR.Const_BitSet._Underlying *b);
            return new(__MR_BitSet_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BitSet::operator|=`.
        public unsafe MR.BitSet BitorAssign(MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_bitor_assign", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_bitor_assign(_Underlying *_this, MR.Const_BitSet._Underlying *b);
            return new(__MR_BitSet_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BitSet::operator^=`.
        public unsafe MR.BitSet XorAssign(MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_xor_assign", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_xor_assign(_Underlying *_this, MR.Const_BitSet._Underlying *b);
            return new(__MR_BitSet_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::BitSet::operator-=`.
        public unsafe MR.BitSet SubAssign(MR.Const_BitSet b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_sub_assign", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_sub_assign(_Underlying *_this, MR.Const_BitSet._Underlying *b);
            return new(__MR_BitSet_sub_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// subtracts b from this, considering that bits in b are shifted right on bShiftInBlocks*bits_per_block
        /// Generated from method `MR::BitSet::subtract`.
        public unsafe MR.BitSet Subtract(MR.Const_BitSet b, int bShiftInBlocks)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_subtract", ExactSpelling = true)]
            extern static MR.BitSet._Underlying *__MR_BitSet_subtract(_Underlying *_this, MR.Const_BitSet._Underlying *b, int bShiftInBlocks);
            return new(__MR_BitSet_subtract(_UnderlyingPtr, b._UnderlyingPtr, bShiftInBlocks), is_owning: false);
        }

        /// doubles reserved memory until resize(newSize) can be done without reallocation
        /// Generated from method `MR::BitSet::resizeWithReserve`.
        public unsafe void ResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_resizeWithReserve", ExactSpelling = true)]
            extern static void __MR_BitSet_resizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_BitSet_resizeWithReserve(_UnderlyingPtr, newSize);
        }

        /// sets elements [pos, pos+len) to given value, adjusting the size of the set to include new elements
        /// Generated from method `MR::BitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(ulong pos, ulong len, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_autoResizeSet_3", ExactSpelling = true)]
            extern static void __MR_BitSet_autoResizeSet_3(_Underlying *_this, ulong pos, ulong len, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_BitSet_autoResizeSet_3(_UnderlyingPtr, pos, len, val.HasValue ? &__deref_val : null);
        }

        /// Generated from method `MR::BitSet::autoResizeSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe void AutoResizeSet(ulong pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_autoResizeSet_2", ExactSpelling = true)]
            extern static void __MR_BitSet_autoResizeSet_2(_Underlying *_this, ulong pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_BitSet_autoResizeSet_2(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null);
        }

        /// same as \ref autoResizeSet and returns previous value of pos-bit
        /// Generated from method `MR::BitSet::autoResizeTestSet`.
        /// Parameter `val` defaults to `true`.
        public unsafe bool AutoResizeTestSet(ulong pos, bool? val = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BitSet_autoResizeTestSet", ExactSpelling = true)]
            extern static byte __MR_BitSet_autoResizeTestSet(_Underlying *_this, ulong pos, byte *val);
            byte __deref_val = val.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_BitSet_autoResizeTestSet(_UnderlyingPtr, pos, val.HasValue ? &__deref_val : null) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `BitSet` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BitSet`/`Const_BitSet` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BitSet
    {
        internal readonly Const_BitSet? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BitSet(Const_BitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BitSet(Const_BitSet arg) {return new(arg);}
        public _ByValue_BitSet(MR.Misc._Moved<BitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BitSet(MR.Misc._Moved<BitSet> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BitSet` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BitSet`/`Const_BitSet` directly.
    public class _InOptMut_BitSet
    {
        public BitSet? Opt;

        public _InOptMut_BitSet() {}
        public _InOptMut_BitSet(BitSet value) {Opt = value;}
        public static implicit operator _InOptMut_BitSet(BitSet value) {return new(value);}
    }

    /// This is used for optional parameters of class `BitSet` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BitSet`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BitSet`/`Const_BitSet` to pass it to the function.
    public class _InOptConst_BitSet
    {
        public Const_BitSet? Opt;

        public _InOptConst_BitSet() {}
        public _InOptConst_BitSet(Const_BitSet value) {Opt = value;}
        public static implicit operator _InOptConst_BitSet(Const_BitSet value) {return new(value);}
    }

    /// returns the amount of memory given BitSet occupies on heap
    /// Generated from function `MR::heapBytes`.
    public static unsafe ulong HeapBytes(MR.Const_BitSet bs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_BitSet", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_BitSet(MR.Const_BitSet._Underlying *bs);
        return __MR_heapBytes_MR_BitSet(bs._UnderlyingPtr);
    }

    /// Generated from function `MR::contains<MR::FaceId>`.
    public static unsafe bool Contains(MR.Const_FaceBitSet? bitset, MR.FaceId id)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_contains", ExactSpelling = true)]
        extern static byte __MR_contains(MR.Const_FaceBitSet._Underlying *bitset, MR.FaceId id);
        return __MR_contains(bitset is not null ? bitset._UnderlyingPtr : null, id) != 0;
    }
}
