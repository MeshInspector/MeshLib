public static partial class MR
{
    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::ChunkIterator>`.
    /// This is the const half of the class.
    public class Const_IteratorRange_MRChunkIterator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IteratorRange_MRChunkIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_IteratorRange_MR_ChunkIterator_Destroy(_Underlying *_this);
            __MR_IteratorRange_MR_ChunkIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IteratorRange_MRChunkIterator() {Dispose(false);}

        public unsafe MR.Const_ChunkIterator Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_Get_begin_", ExactSpelling = true)]
                extern static MR.Const_ChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_Get_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_ChunkIterator_Get_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_ChunkIterator End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_Get_end_", ExactSpelling = true)]
                extern static MR.Const_ChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_Get_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_ChunkIterator_Get_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::ChunkIterator>::IteratorRange`.
        public unsafe Const_IteratorRange_MRChunkIterator(MR.Const_IteratorRange_MRChunkIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_ConstructFromAnother(MR.IteratorRange_MRChunkIterator._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_ChunkIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::ChunkIterator>::IteratorRange`.
        public unsafe Const_IteratorRange_MRChunkIterator(MR.Const_ChunkIterator begin, MR.Const_ChunkIterator end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_Construct(MR.ChunkIterator._Underlying *begin, MR.ChunkIterator._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_ChunkIterator_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::ChunkIterator>`.
    /// This is the non-const half of the class.
    public class IteratorRange_MRChunkIterator : Const_IteratorRange_MRChunkIterator
    {
        internal unsafe IteratorRange_MRChunkIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.ChunkIterator Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_GetMutable_begin_", ExactSpelling = true)]
                extern static MR.ChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_GetMutable_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_ChunkIterator_GetMutable_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.ChunkIterator End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_GetMutable_end_", ExactSpelling = true)]
                extern static MR.ChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_GetMutable_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_ChunkIterator_GetMutable_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::ChunkIterator>::IteratorRange`.
        public unsafe IteratorRange_MRChunkIterator(MR.Const_IteratorRange_MRChunkIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_ConstructFromAnother(MR.IteratorRange_MRChunkIterator._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_ChunkIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::ChunkIterator>::IteratorRange`.
        public unsafe IteratorRange_MRChunkIterator(MR.Const_ChunkIterator begin, MR.Const_ChunkIterator end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_Construct(MR.ChunkIterator._Underlying *begin, MR.ChunkIterator._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_ChunkIterator_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }

        /// Generated from method `MR::IteratorRange<MR::ChunkIterator>::operator=`.
        public unsafe MR.IteratorRange_MRChunkIterator Assign(MR.Const_IteratorRange_MRChunkIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_ChunkIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRChunkIterator._Underlying *__MR_IteratorRange_MR_ChunkIterator_AssignFromAnother(_Underlying *_this, MR.IteratorRange_MRChunkIterator._Underlying *_other);
            return new(__MR_IteratorRange_MR_ChunkIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IteratorRange_MRChunkIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IteratorRange_MRChunkIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRChunkIterator`/`Const_IteratorRange_MRChunkIterator` directly.
    public class _InOptMut_IteratorRange_MRChunkIterator
    {
        public IteratorRange_MRChunkIterator? Opt;

        public _InOptMut_IteratorRange_MRChunkIterator() {}
        public _InOptMut_IteratorRange_MRChunkIterator(IteratorRange_MRChunkIterator value) {Opt = value;}
        public static implicit operator _InOptMut_IteratorRange_MRChunkIterator(IteratorRange_MRChunkIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `IteratorRange_MRChunkIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IteratorRange_MRChunkIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRChunkIterator`/`Const_IteratorRange_MRChunkIterator` to pass it to the function.
    public class _InOptConst_IteratorRange_MRChunkIterator
    {
        public Const_IteratorRange_MRChunkIterator? Opt;

        public _InOptConst_IteratorRange_MRChunkIterator() {}
        public _InOptConst_IteratorRange_MRChunkIterator(Const_IteratorRange_MRChunkIterator value) {Opt = value;}
        public static implicit operator _InOptConst_IteratorRange_MRChunkIterator(Const_IteratorRange_MRChunkIterator value) {return new(value);}
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::UndirectedEdgeIterator>`.
    /// This is the const half of the class.
    public class Const_IteratorRange_MRUndirectedEdgeIterator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IteratorRange_MRUndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_IteratorRange_MR_UndirectedEdgeIterator_Destroy(_Underlying *_this);
            __MR_IteratorRange_MR_UndirectedEdgeIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IteratorRange_MRUndirectedEdgeIterator() {Dispose(false);}

        public unsafe MR.Const_UndirectedEdgeIterator Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_Get_begin_", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_Get_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_UndirectedEdgeIterator_Get_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_UndirectedEdgeIterator End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_Get_end_", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_Get_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_UndirectedEdgeIterator_Get_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::UndirectedEdgeIterator>::IteratorRange`.
        public unsafe Const_IteratorRange_MRUndirectedEdgeIterator(MR.Const_IteratorRange_MRUndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_ConstructFromAnother(MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_UndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::UndirectedEdgeIterator>::IteratorRange`.
        public unsafe Const_IteratorRange_MRUndirectedEdgeIterator(MR.Const_UndirectedEdgeIterator begin, MR.Const_UndirectedEdgeIterator end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_Construct(MR.UndirectedEdgeIterator._Underlying *begin, MR.UndirectedEdgeIterator._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_UndirectedEdgeIterator_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::UndirectedEdgeIterator>`.
    /// This is the non-const half of the class.
    public class IteratorRange_MRUndirectedEdgeIterator : Const_IteratorRange_MRUndirectedEdgeIterator
    {
        internal unsafe IteratorRange_MRUndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.UndirectedEdgeIterator Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_GetMutable_begin_", ExactSpelling = true)]
                extern static MR.UndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_GetMutable_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_UndirectedEdgeIterator_GetMutable_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.UndirectedEdgeIterator End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_GetMutable_end_", ExactSpelling = true)]
                extern static MR.UndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_GetMutable_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_UndirectedEdgeIterator_GetMutable_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::UndirectedEdgeIterator>::IteratorRange`.
        public unsafe IteratorRange_MRUndirectedEdgeIterator(MR.Const_IteratorRange_MRUndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_ConstructFromAnother(MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_UndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::UndirectedEdgeIterator>::IteratorRange`.
        public unsafe IteratorRange_MRUndirectedEdgeIterator(MR.Const_UndirectedEdgeIterator begin, MR.Const_UndirectedEdgeIterator end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_Construct(MR.UndirectedEdgeIterator._Underlying *begin, MR.UndirectedEdgeIterator._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_UndirectedEdgeIterator_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }

        /// Generated from method `MR::IteratorRange<MR::UndirectedEdgeIterator>::operator=`.
        public unsafe MR.IteratorRange_MRUndirectedEdgeIterator Assign(MR.Const_IteratorRange_MRUndirectedEdgeIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_UndirectedEdgeIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_UndirectedEdgeIterator_AssignFromAnother(_Underlying *_this, MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *_other);
            return new(__MR_IteratorRange_MR_UndirectedEdgeIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IteratorRange_MRUndirectedEdgeIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IteratorRange_MRUndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRUndirectedEdgeIterator`/`Const_IteratorRange_MRUndirectedEdgeIterator` directly.
    public class _InOptMut_IteratorRange_MRUndirectedEdgeIterator
    {
        public IteratorRange_MRUndirectedEdgeIterator? Opt;

        public _InOptMut_IteratorRange_MRUndirectedEdgeIterator() {}
        public _InOptMut_IteratorRange_MRUndirectedEdgeIterator(IteratorRange_MRUndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptMut_IteratorRange_MRUndirectedEdgeIterator(IteratorRange_MRUndirectedEdgeIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `IteratorRange_MRUndirectedEdgeIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IteratorRange_MRUndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRUndirectedEdgeIterator`/`Const_IteratorRange_MRUndirectedEdgeIterator` to pass it to the function.
    public class _InOptConst_IteratorRange_MRUndirectedEdgeIterator
    {
        public Const_IteratorRange_MRUndirectedEdgeIterator? Opt;

        public _InOptConst_IteratorRange_MRUndirectedEdgeIterator() {}
        public _InOptConst_IteratorRange_MRUndirectedEdgeIterator(Const_IteratorRange_MRUndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptConst_IteratorRange_MRUndirectedEdgeIterator(Const_IteratorRange_MRUndirectedEdgeIterator value) {return new(value);}
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>`.
    /// This is the const half of the class.
    public class Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Destroy", ExactSpelling = true)]
            extern static void __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Destroy(_Underlying *_this);
            __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin() {Dispose(false);}

        public unsafe MR.Const_RingIterator_MRNextEdgeSameOrigin Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Get_begin_", ExactSpelling = true)]
                extern static MR.Const_RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Get_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Get_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_RingIterator_MRNextEdgeSameOrigin End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Get_end_", ExactSpelling = true)]
                extern static MR.Const_RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Get_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Get_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>::IteratorRange`.
        public unsafe Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>::IteratorRange`.
        public unsafe Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(MR.Const_RingIterator_MRNextEdgeSameOrigin begin, MR.Const_RingIterator_MRNextEdgeSameOrigin end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Construct(MR.RingIterator_MRNextEdgeSameOrigin._Underlying *begin, MR.RingIterator_MRNextEdgeSameOrigin._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>`.
    /// This is the non-const half of the class.
    public class IteratorRange_MRRingIteratorMRNextEdgeSameOrigin : Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin
    {
        internal unsafe IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.RingIterator_MRNextEdgeSameOrigin Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_GetMutable_begin_", ExactSpelling = true)]
                extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_GetMutable_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_GetMutable_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.RingIterator_MRNextEdgeSameOrigin End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_GetMutable_end_", ExactSpelling = true)]
                extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_GetMutable_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_GetMutable_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>::IteratorRange`.
        public unsafe IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>::IteratorRange`.
        public unsafe IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(MR.Const_RingIterator_MRNextEdgeSameOrigin begin, MR.Const_RingIterator_MRNextEdgeSameOrigin end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Construct(MR.RingIterator_MRNextEdgeSameOrigin._Underlying *begin, MR.RingIterator_MRNextEdgeSameOrigin._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }

        /// Generated from method `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameOrigin>>::operator=`.
        public unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin Assign(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_AssignFromAnother(_Underlying *_this, MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *_other);
            return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IteratorRange_MRRingIteratorMRNextEdgeSameOrigin` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRRingIteratorMRNextEdgeSameOrigin`/`Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin` directly.
    public class _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin
    {
        public IteratorRange_MRRingIteratorMRNextEdgeSameOrigin? Opt;

        public _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin() {}
        public _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(IteratorRange_MRRingIteratorMRNextEdgeSameOrigin value) {Opt = value;}
        public static implicit operator _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(IteratorRange_MRRingIteratorMRNextEdgeSameOrigin value) {return new(value);}
    }

    /// This is used for optional parameters of class `IteratorRange_MRRingIteratorMRNextEdgeSameOrigin` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRRingIteratorMRNextEdgeSameOrigin`/`Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin` to pass it to the function.
    public class _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin
    {
        public Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin? Opt;

        public _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin() {}
        public _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin value) {Opt = value;}
        public static implicit operator _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin(Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin value) {return new(value);}
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>`.
    /// This is the const half of the class.
    public class Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Destroy", ExactSpelling = true)]
            extern static void __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Destroy(_Underlying *_this);
            __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft() {Dispose(false);}

        public unsafe MR.Const_RingIterator_MRNextEdgeSameLeft Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Get_begin_", ExactSpelling = true)]
                extern static MR.Const_RingIterator_MRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Get_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Get_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_RingIterator_MRNextEdgeSameLeft End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Get_end_", ExactSpelling = true)]
                extern static MR.Const_RingIterator_MRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Get_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Get_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>::IteratorRange`.
        public unsafe Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>::IteratorRange`.
        public unsafe Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(MR.Const_RingIterator_MRNextEdgeSameLeft begin, MR.Const_RingIterator_MRNextEdgeSameLeft end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Construct(MR.RingIterator_MRNextEdgeSameLeft._Underlying *begin, MR.RingIterator_MRNextEdgeSameLeft._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>`.
    /// This is the non-const half of the class.
    public class IteratorRange_MRRingIteratorMRNextEdgeSameLeft : Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft
    {
        internal unsafe IteratorRange_MRRingIteratorMRNextEdgeSameLeft(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.RingIterator_MRNextEdgeSameLeft Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_GetMutable_begin_", ExactSpelling = true)]
                extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_GetMutable_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_GetMutable_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.RingIterator_MRNextEdgeSameLeft End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_GetMutable_end_", ExactSpelling = true)]
                extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_GetMutable_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_GetMutable_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>::IteratorRange`.
        public unsafe IteratorRange_MRRingIteratorMRNextEdgeSameLeft(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>::IteratorRange`.
        public unsafe IteratorRange_MRRingIteratorMRNextEdgeSameLeft(MR.Const_RingIterator_MRNextEdgeSameLeft begin, MR.Const_RingIterator_MRNextEdgeSameLeft end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Construct(MR.RingIterator_MRNextEdgeSameLeft._Underlying *begin, MR.RingIterator_MRNextEdgeSameLeft._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }

        /// Generated from method `MR::IteratorRange<MR::RingIterator<MR::NextEdgeSameLeft>>::operator=`.
        public unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft Assign(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_AssignFromAnother(_Underlying *_this, MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *_other);
            return new(__MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IteratorRange_MRRingIteratorMRNextEdgeSameLeft` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameLeft`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRRingIteratorMRNextEdgeSameLeft`/`Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft` directly.
    public class _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameLeft
    {
        public IteratorRange_MRRingIteratorMRNextEdgeSameLeft? Opt;

        public _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameLeft() {}
        public _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(IteratorRange_MRRingIteratorMRNextEdgeSameLeft value) {Opt = value;}
        public static implicit operator _InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(IteratorRange_MRRingIteratorMRNextEdgeSameLeft value) {return new(value);}
    }

    /// This is used for optional parameters of class `IteratorRange_MRRingIteratorMRNextEdgeSameLeft` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IteratorRange_MRRingIteratorMRNextEdgeSameLeft`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRRingIteratorMRNextEdgeSameLeft`/`Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft` to pass it to the function.
    public class _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameLeft
    {
        public Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft? Opt;

        public _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameLeft() {}
        public _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft value) {Opt = value;}
        public static implicit operator _InOptConst_IteratorRange_MRRingIteratorMRNextEdgeSameLeft(Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft value) {return new(value);}
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>`.
    /// This is the const half of the class.
    public class Const_IteratorRange_MRPolylineUndirectedEdgeIterator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IteratorRange_MRPolylineUndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Destroy(_Underlying *_this);
            __MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IteratorRange_MRPolylineUndirectedEdgeIterator() {Dispose(false);}

        public unsafe MR.Const_PolylineUndirectedEdgeIterator Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Get_begin_", ExactSpelling = true)]
                extern static MR.Const_PolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Get_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Get_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_PolylineUndirectedEdgeIterator End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Get_end_", ExactSpelling = true)]
                extern static MR.Const_PolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Get_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Get_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>::IteratorRange`.
        public unsafe Const_IteratorRange_MRPolylineUndirectedEdgeIterator(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>::IteratorRange`.
        public unsafe Const_IteratorRange_MRPolylineUndirectedEdgeIterator(MR.Const_PolylineUndirectedEdgeIterator begin, MR.Const_PolylineUndirectedEdgeIterator end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Construct(MR.PolylineUndirectedEdgeIterator._Underlying *begin, MR.PolylineUndirectedEdgeIterator._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }
    }

    /// \brief simple alternative to boost/iterator_range
    /// Generated from class `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>`.
    /// This is the non-const half of the class.
    public class IteratorRange_MRPolylineUndirectedEdgeIterator : Const_IteratorRange_MRPolylineUndirectedEdgeIterator
    {
        internal unsafe IteratorRange_MRPolylineUndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.PolylineUndirectedEdgeIterator Begin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_GetMutable_begin_", ExactSpelling = true)]
                extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_GetMutable_begin_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_GetMutable_begin_(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.PolylineUndirectedEdgeIterator End
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_GetMutable_end_", ExactSpelling = true)]
                extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_GetMutable_end_(_Underlying *_this);
                return new(__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_GetMutable_end_(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>::IteratorRange`.
        public unsafe IteratorRange_MRPolylineUndirectedEdgeIterator(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>::IteratorRange`.
        public unsafe IteratorRange_MRPolylineUndirectedEdgeIterator(MR.Const_PolylineUndirectedEdgeIterator begin, MR.Const_PolylineUndirectedEdgeIterator end) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Construct(MR.PolylineUndirectedEdgeIterator._Underlying *begin, MR.PolylineUndirectedEdgeIterator._Underlying *end);
            _UnderlyingPtr = __MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_Construct(begin._UnderlyingPtr, end._UnderlyingPtr);
        }

        /// Generated from method `MR::IteratorRange<MR::PolylineUndirectedEdgeIterator>::operator=`.
        public unsafe MR.IteratorRange_MRPolylineUndirectedEdgeIterator Assign(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_AssignFromAnother(_Underlying *_this, MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *_other);
            return new(__MR_IteratorRange_MR_PolylineUndirectedEdgeIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IteratorRange_MRPolylineUndirectedEdgeIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IteratorRange_MRPolylineUndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRPolylineUndirectedEdgeIterator`/`Const_IteratorRange_MRPolylineUndirectedEdgeIterator` directly.
    public class _InOptMut_IteratorRange_MRPolylineUndirectedEdgeIterator
    {
        public IteratorRange_MRPolylineUndirectedEdgeIterator? Opt;

        public _InOptMut_IteratorRange_MRPolylineUndirectedEdgeIterator() {}
        public _InOptMut_IteratorRange_MRPolylineUndirectedEdgeIterator(IteratorRange_MRPolylineUndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptMut_IteratorRange_MRPolylineUndirectedEdgeIterator(IteratorRange_MRPolylineUndirectedEdgeIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `IteratorRange_MRPolylineUndirectedEdgeIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IteratorRange_MRPolylineUndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IteratorRange_MRPolylineUndirectedEdgeIterator`/`Const_IteratorRange_MRPolylineUndirectedEdgeIterator` to pass it to the function.
    public class _InOptConst_IteratorRange_MRPolylineUndirectedEdgeIterator
    {
        public Const_IteratorRange_MRPolylineUndirectedEdgeIterator? Opt;

        public _InOptConst_IteratorRange_MRPolylineUndirectedEdgeIterator() {}
        public _InOptConst_IteratorRange_MRPolylineUndirectedEdgeIterator(Const_IteratorRange_MRPolylineUndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptConst_IteratorRange_MRPolylineUndirectedEdgeIterator(Const_IteratorRange_MRPolylineUndirectedEdgeIterator value) {return new(value);}
    }

    /// Generated from function `MR::begin`.
    public static unsafe MR.ChunkIterator Begin(MR.Const_IteratorRange_MRChunkIterator range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_begin_MR_IteratorRange_MR_ChunkIterator", ExactSpelling = true)]
        extern static MR.ChunkIterator._Underlying *__MR_begin_MR_IteratorRange_MR_ChunkIterator(MR.Const_IteratorRange_MRChunkIterator._Underlying *range);
        return new(__MR_begin_MR_IteratorRange_MR_ChunkIterator(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::end`.
    public static unsafe MR.ChunkIterator End(MR.Const_IteratorRange_MRChunkIterator range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_end_MR_IteratorRange_MR_ChunkIterator", ExactSpelling = true)]
        extern static MR.ChunkIterator._Underlying *__MR_end_MR_IteratorRange_MR_ChunkIterator(MR.Const_IteratorRange_MRChunkIterator._Underlying *range);
        return new(__MR_end_MR_IteratorRange_MR_ChunkIterator(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::begin`.
    public static unsafe MR.UndirectedEdgeIterator Begin(MR.Const_IteratorRange_MRUndirectedEdgeIterator range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_begin_MR_IteratorRange_MR_UndirectedEdgeIterator", ExactSpelling = true)]
        extern static MR.UndirectedEdgeIterator._Underlying *__MR_begin_MR_IteratorRange_MR_UndirectedEdgeIterator(MR.Const_IteratorRange_MRUndirectedEdgeIterator._Underlying *range);
        return new(__MR_begin_MR_IteratorRange_MR_UndirectedEdgeIterator(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::end`.
    public static unsafe MR.UndirectedEdgeIterator End(MR.Const_IteratorRange_MRUndirectedEdgeIterator range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_end_MR_IteratorRange_MR_UndirectedEdgeIterator", ExactSpelling = true)]
        extern static MR.UndirectedEdgeIterator._Underlying *__MR_end_MR_IteratorRange_MR_UndirectedEdgeIterator(MR.Const_IteratorRange_MRUndirectedEdgeIterator._Underlying *range);
        return new(__MR_end_MR_IteratorRange_MR_UndirectedEdgeIterator(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::begin`.
    public static unsafe MR.RingIterator_MRNextEdgeSameOrigin Begin(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_begin_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin", ExactSpelling = true)]
        extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_begin_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *range);
        return new(__MR_begin_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::end`.
    public static unsafe MR.RingIterator_MRNextEdgeSameOrigin End(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_end_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin", ExactSpelling = true)]
        extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_end_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *range);
        return new(__MR_end_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameOrigin(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::begin`.
    public static unsafe MR.RingIterator_MRNextEdgeSameLeft Begin(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_begin_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft", ExactSpelling = true)]
        extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_begin_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *range);
        return new(__MR_begin_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::end`.
    public static unsafe MR.RingIterator_MRNextEdgeSameLeft End(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_end_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft", ExactSpelling = true)]
        extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_end_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft(MR.Const_IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *range);
        return new(__MR_end_MR_IteratorRange_MR_RingIterator_MR_NextEdgeSameLeft(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::begin`.
    public static unsafe MR.PolylineUndirectedEdgeIterator Begin(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_begin_MR_IteratorRange_MR_PolylineUndirectedEdgeIterator", ExactSpelling = true)]
        extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_begin_MR_IteratorRange_MR_PolylineUndirectedEdgeIterator(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *range);
        return new(__MR_begin_MR_IteratorRange_MR_PolylineUndirectedEdgeIterator(range._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::end`.
    public static unsafe MR.PolylineUndirectedEdgeIterator End(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator range)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_end_MR_IteratorRange_MR_PolylineUndirectedEdgeIterator", ExactSpelling = true)]
        extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_end_MR_IteratorRange_MR_PolylineUndirectedEdgeIterator(MR.Const_IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *range);
        return new(__MR_end_MR_IteratorRange_MR_PolylineUndirectedEdgeIterator(range._UnderlyingPtr), is_owning: true);
    }
}
