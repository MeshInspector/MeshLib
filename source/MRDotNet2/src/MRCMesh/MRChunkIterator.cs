public static partial class MR
{
    /// array chunk representation
    /// Generated from class `MR::Chunk`.
    /// This is the const half of the class.
    public class Const_Chunk : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Chunk(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_Destroy", ExactSpelling = true)]
            extern static void __MR_Chunk_Destroy(_Underlying *_this);
            __MR_Chunk_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Chunk() {Dispose(false);}

        /// chunk offset
        public unsafe ulong Offset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_Get_offset", ExactSpelling = true)]
                extern static ulong *__MR_Chunk_Get_offset(_Underlying *_this);
                return *__MR_Chunk_Get_offset(_UnderlyingPtr);
            }
        }

        /// chunk size; the last chunk's size may be smaller than other chunk's ones
        public unsafe ulong Size
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_Get_size", ExactSpelling = true)]
                extern static ulong *__MR_Chunk_Get_size(_Underlying *_this);
                return *__MR_Chunk_Get_size(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Chunk() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_DefaultConstruct();
            _UnderlyingPtr = __MR_Chunk_DefaultConstruct();
        }

        /// Constructs `MR::Chunk` elementwise.
        public unsafe Const_Chunk(ulong offset, ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_ConstructFrom", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_ConstructFrom(ulong offset, ulong size);
            _UnderlyingPtr = __MR_Chunk_ConstructFrom(offset, size);
        }

        /// Generated from constructor `MR::Chunk::Chunk`.
        public unsafe Const_Chunk(MR.Const_Chunk _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_ConstructFromAnother(MR.Chunk._Underlying *_other);
            _UnderlyingPtr = __MR_Chunk_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// array chunk representation
    /// Generated from class `MR::Chunk`.
    /// This is the non-const half of the class.
    public class Chunk : Const_Chunk
    {
        internal unsafe Chunk(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// chunk offset
        public new unsafe ref ulong Offset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_GetMutable_offset", ExactSpelling = true)]
                extern static ulong *__MR_Chunk_GetMutable_offset(_Underlying *_this);
                return ref *__MR_Chunk_GetMutable_offset(_UnderlyingPtr);
            }
        }

        /// chunk size; the last chunk's size may be smaller than other chunk's ones
        public new unsafe ref ulong Size
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_GetMutable_size", ExactSpelling = true)]
                extern static ulong *__MR_Chunk_GetMutable_size(_Underlying *_this);
                return ref *__MR_Chunk_GetMutable_size(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Chunk() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_DefaultConstruct();
            _UnderlyingPtr = __MR_Chunk_DefaultConstruct();
        }

        /// Constructs `MR::Chunk` elementwise.
        public unsafe Chunk(ulong offset, ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_ConstructFrom", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_ConstructFrom(ulong offset, ulong size);
            _UnderlyingPtr = __MR_Chunk_ConstructFrom(offset, size);
        }

        /// Generated from constructor `MR::Chunk::Chunk`.
        public unsafe Chunk(MR.Const_Chunk _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_ConstructFromAnother(MR.Chunk._Underlying *_other);
            _UnderlyingPtr = __MR_Chunk_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Chunk::operator=`.
        public unsafe MR.Chunk Assign(MR.Const_Chunk _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Chunk_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_Chunk_AssignFromAnother(_Underlying *_this, MR.Chunk._Underlying *_other);
            return new(__MR_Chunk_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Chunk` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Chunk`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Chunk`/`Const_Chunk` directly.
    public class _InOptMut_Chunk
    {
        public Chunk? Opt;

        public _InOptMut_Chunk() {}
        public _InOptMut_Chunk(Chunk value) {Opt = value;}
        public static implicit operator _InOptMut_Chunk(Chunk value) {return new(value);}
    }

    /// This is used for optional parameters of class `Chunk` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Chunk`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Chunk`/`Const_Chunk` to pass it to the function.
    public class _InOptConst_Chunk
    {
        public Const_Chunk? Opt;

        public _InOptConst_Chunk() {}
        public _InOptConst_Chunk(Const_Chunk value) {Opt = value;}
        public static implicit operator _InOptConst_Chunk(Const_Chunk value) {return new(value);}
    }

    /// iterator class for array chunks
    /// Generated from class `MR::ChunkIterator`.
    /// This is the const half of the class.
    public class Const_ChunkIterator : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_ChunkIterator>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ChunkIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_ChunkIterator_Destroy(_Underlying *_this);
            __MR_ChunkIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ChunkIterator() {Dispose(false);}

        public unsafe ulong TotalSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_Get_totalSize", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_Get_totalSize(_Underlying *_this);
                return *__MR_ChunkIterator_Get_totalSize(_UnderlyingPtr);
            }
        }

        public unsafe ulong ChunkSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_Get_chunkSize", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_Get_chunkSize(_Underlying *_this);
                return *__MR_ChunkIterator_Get_chunkSize(_UnderlyingPtr);
            }
        }

        public unsafe ulong Overlap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_Get_overlap", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_Get_overlap(_Underlying *_this);
                return *__MR_ChunkIterator_Get_overlap(_UnderlyingPtr);
            }
        }

        public unsafe ulong Index
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_Get_index", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_Get_index(_Underlying *_this);
                return *__MR_ChunkIterator_Get_index(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ChunkIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_ChunkIterator_DefaultConstruct();
        }

        /// Constructs `MR::ChunkIterator` elementwise.
        public unsafe Const_ChunkIterator(ulong totalSize, ulong chunkSize, ulong overlap, ulong index) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_ConstructFrom", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_ConstructFrom(ulong totalSize, ulong chunkSize, ulong overlap, ulong index);
            _UnderlyingPtr = __MR_ChunkIterator_ConstructFrom(totalSize, chunkSize, overlap, index);
        }

        /// Generated from constructor `MR::ChunkIterator::ChunkIterator`.
        public unsafe Const_ChunkIterator(MR.Const_ChunkIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_ConstructFromAnother(MR.ChunkIterator._Underlying *_other);
            _UnderlyingPtr = __MR_ChunkIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ChunkIterator::operator==`.
        public static unsafe bool operator==(MR.Const_ChunkIterator _this, MR.Const_ChunkIterator other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ChunkIterator", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ChunkIterator(MR.Const_ChunkIterator._Underlying *_this, MR.Const_ChunkIterator._Underlying *other);
            return __MR_equal_MR_ChunkIterator(_this._UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ChunkIterator _this, MR.Const_ChunkIterator other)
        {
            return !(_this == other);
        }

        /// Generated from method `MR::ChunkIterator::operator++`.
        public static unsafe ChunkIterator operator++(MR.Const_ChunkIterator _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ChunkIterator", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_incr_MR_ChunkIterator(MR.Const_ChunkIterator._Underlying *_this);
            ChunkIterator _this_copy = new(_this);
            MR.ChunkIterator _unused_ret = new(__MR_incr_MR_ChunkIterator(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::ChunkIterator::operator*`.
        public unsafe MR.Chunk Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_ChunkIterator", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_deref_MR_ChunkIterator(_Underlying *_this);
            return new(__MR_deref_MR_ChunkIterator(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ChunkIterator::operator->`.
        public unsafe MR.Chunk Arrow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_arrow", ExactSpelling = true)]
            extern static MR.Chunk._Underlying *__MR_ChunkIterator_arrow(_Underlying *_this);
            return new(__MR_ChunkIterator_arrow(_UnderlyingPtr), is_owning: true);
        }

        // IEquatable:

        public bool Equals(MR.Const_ChunkIterator? other)
        {
            if (other is null)
                return false;
            return this == other;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_ChunkIterator)
                return this == (MR.Const_ChunkIterator)other;
            return false;
        }
    }

    /// iterator class for array chunks
    /// Generated from class `MR::ChunkIterator`.
    /// This is the non-const half of the class.
    public class ChunkIterator : Const_ChunkIterator
    {
        internal unsafe ChunkIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref ulong TotalSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_GetMutable_totalSize", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_GetMutable_totalSize(_Underlying *_this);
                return ref *__MR_ChunkIterator_GetMutable_totalSize(_UnderlyingPtr);
            }
        }

        public new unsafe ref ulong ChunkSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_GetMutable_chunkSize", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_GetMutable_chunkSize(_Underlying *_this);
                return ref *__MR_ChunkIterator_GetMutable_chunkSize(_UnderlyingPtr);
            }
        }

        public new unsafe ref ulong Overlap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_GetMutable_overlap", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_GetMutable_overlap(_Underlying *_this);
                return ref *__MR_ChunkIterator_GetMutable_overlap(_UnderlyingPtr);
            }
        }

        public new unsafe ref ulong Index
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_GetMutable_index", ExactSpelling = true)]
                extern static ulong *__MR_ChunkIterator_GetMutable_index(_Underlying *_this);
                return ref *__MR_ChunkIterator_GetMutable_index(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ChunkIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_ChunkIterator_DefaultConstruct();
        }

        /// Constructs `MR::ChunkIterator` elementwise.
        public unsafe ChunkIterator(ulong totalSize, ulong chunkSize, ulong overlap, ulong index) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_ConstructFrom", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_ConstructFrom(ulong totalSize, ulong chunkSize, ulong overlap, ulong index);
            _UnderlyingPtr = __MR_ChunkIterator_ConstructFrom(totalSize, chunkSize, overlap, index);
        }

        /// Generated from constructor `MR::ChunkIterator::ChunkIterator`.
        public unsafe ChunkIterator(MR.Const_ChunkIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_ConstructFromAnother(MR.ChunkIterator._Underlying *_other);
            _UnderlyingPtr = __MR_ChunkIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ChunkIterator::operator=`.
        public unsafe MR.ChunkIterator Assign(MR.Const_ChunkIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ChunkIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_ChunkIterator_AssignFromAnother(_Underlying *_this, MR.ChunkIterator._Underlying *_other);
            return new(__MR_ChunkIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ChunkIterator::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ChunkIterator", ExactSpelling = true)]
            extern static MR.ChunkIterator._Underlying *__MR_incr_MR_ChunkIterator(_Underlying *_this);
            MR.ChunkIterator _unused_ret = new(__MR_incr_MR_ChunkIterator(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ChunkIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ChunkIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChunkIterator`/`Const_ChunkIterator` directly.
    public class _InOptMut_ChunkIterator
    {
        public ChunkIterator? Opt;

        public _InOptMut_ChunkIterator() {}
        public _InOptMut_ChunkIterator(ChunkIterator value) {Opt = value;}
        public static implicit operator _InOptMut_ChunkIterator(ChunkIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `ChunkIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ChunkIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ChunkIterator`/`Const_ChunkIterator` to pass it to the function.
    public class _InOptConst_ChunkIterator
    {
        public Const_ChunkIterator? Opt;

        public _InOptConst_ChunkIterator() {}
        public _InOptConst_ChunkIterator(Const_ChunkIterator value) {Opt = value;}
        public static implicit operator _InOptConst_ChunkIterator(Const_ChunkIterator value) {return new(value);}
    }

    /// returns the amount of chunks of given size required to cover the full array
    /// Generated from function `MR::chunkCount`.
    /// Parameter `overlap` defaults to `0`.
    public static unsafe ulong ChunkCount(ulong totalSize, ulong chunkSize, ulong? overlap = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_chunkCount", ExactSpelling = true)]
        extern static ulong __MR_chunkCount(ulong totalSize, ulong chunkSize, ulong *overlap);
        ulong __deref_overlap = overlap.GetValueOrDefault();
        return __MR_chunkCount(totalSize, chunkSize, overlap.HasValue ? &__deref_overlap : null);
    }

    /// returns a pair of iterators for chunks covering the array of given size
    /// Generated from function `MR::splitByChunks`.
    /// Parameter `overlap` defaults to `0`.
    public static unsafe MR.IteratorRange_MRChunkIterator SplitByChunks(ulong totalSize, ulong chunkSize, ulong? overlap = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_splitByChunks", ExactSpelling = true)]
        extern static MR.IteratorRange_MRChunkIterator._Underlying *__MR_splitByChunks(ulong totalSize, ulong chunkSize, ulong *overlap);
        ulong __deref_overlap = overlap.GetValueOrDefault();
        return new(__MR_splitByChunks(totalSize, chunkSize, overlap.HasValue ? &__deref_overlap : null), is_owning: true);
    }
}
