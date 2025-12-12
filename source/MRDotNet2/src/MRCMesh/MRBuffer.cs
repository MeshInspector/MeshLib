public static partial class MR
{
    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(MR._ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_size(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_empty(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRUndirectedEdgeId Index(MR.UndirectedEdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_index_const(_Underlying *_this, MR.UndirectedEdgeId i);
            return new(__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::data`.
        public unsafe MR.NoDefInit_MRUndirectedEdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRUndirectedEdgeId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::beginId`.
        public unsafe MR.UndirectedEdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_beginId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::backId`.
        public unsafe MR.UndirectedEdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_backId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_backId(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::endId`.
        public unsafe MR.UndirectedEdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_endId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_endId(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId : Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        internal unsafe Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(MR._ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId Assign(MR._ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_clear(_Underlying *_this);
            __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::operator[]`.
        public unsafe new MR.NoDefInit_MRUndirectedEdgeId Index(MR.UndirectedEdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_index(_Underlying *_this, MR.UndirectedEdgeId i);
            return new(__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId, MR::UndirectedEdgeId>::data`.
        public unsafe new MR.NoDefInit_MRUndirectedEdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_UndirectedEdgeId_MR_UndirectedEdgeId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRUndirectedEdgeId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId`/`Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        internal readonly Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(MR.Misc._Moved<Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(MR.Misc._Moved<Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId`/`Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId` directly.
    public class _InOptMut_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        public Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId? Opt;

        public _InOptMut_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId() {}
        public _InOptMut_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId`/`Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId
    {
        public Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId? Opt;

        public _InOptConst_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId() {}
        public _InOptConst_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId(Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::FaceId, MR::FaceId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRFaceId_MRFaceId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRFaceId_MRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_FaceId_MR_FaceId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_FaceId_MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRFaceId_MRFaceId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRFaceId_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId, MR::FaceId>::Buffer`.
        public unsafe Const_Buffer_MRFaceId_MRFaceId(MR._ByValue_Buffer_MRFaceId_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRFaceId_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId, MR::FaceId>::Buffer`.
        public unsafe Const_Buffer_MRFaceId_MRFaceId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_MR_FaceId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_MR_FaceId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_MR_FaceId_size(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_FaceId_MR_FaceId_empty(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRFaceId Index(MR.FaceId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_index_const(_Underlying *_this, MR.FaceId i);
            return new(__MR_Buffer_MR_FaceId_MR_FaceId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::data`.
        public unsafe MR.NoDefInit_MRFaceId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_FaceId_MR_FaceId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRFaceId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::beginId`.
        public unsafe MR.FaceId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_beginId", ExactSpelling = true)]
            extern static MR.FaceId __MR_Buffer_MR_FaceId_MR_FaceId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::backId`.
        public unsafe MR.FaceId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_backId", ExactSpelling = true)]
            extern static MR.FaceId __MR_Buffer_MR_FaceId_MR_FaceId_backId(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::endId`.
        public unsafe MR.FaceId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_endId", ExactSpelling = true)]
            extern static MR.FaceId __MR_Buffer_MR_FaceId_MR_FaceId_endId(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_MR_FaceId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_MR_FaceId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::FaceId, MR::FaceId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRFaceId_MRFaceId : Const_Buffer_MRFaceId_MRFaceId
    {
        internal unsafe Buffer_MRFaceId_MRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRFaceId_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId, MR::FaceId>::Buffer`.
        public unsafe Buffer_MRFaceId_MRFaceId(MR._ByValue_Buffer_MRFaceId_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRFaceId_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId, MR::FaceId>::Buffer`.
        public unsafe Buffer_MRFaceId_MRFaceId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_MR_FaceId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::operator=`.
        public unsafe MR.Buffer_MRFaceId_MRFaceId Assign(MR._ByValue_Buffer_MRFaceId_MRFaceId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRFaceId_MRFaceId._Underlying *_other);
            return new(__MR_Buffer_MR_FaceId_MR_FaceId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_FaceId_MR_FaceId_clear(_Underlying *_this);
            __MR_Buffer_MR_FaceId_MR_FaceId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_FaceId_MR_FaceId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_FaceId_MR_FaceId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::operator[]`.
        public unsafe new MR.NoDefInit_MRFaceId Index(MR.FaceId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_index(_Underlying *_this, MR.FaceId i);
            return new(__MR_Buffer_MR_FaceId_MR_FaceId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::FaceId, MR::FaceId>::data`.
        public unsafe new MR.NoDefInit_MRFaceId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_MR_FaceId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_MR_FaceId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_FaceId_MR_FaceId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRFaceId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRFaceId_MRFaceId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRFaceId_MRFaceId`/`Const_Buffer_MRFaceId_MRFaceId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRFaceId_MRFaceId
    {
        internal readonly Const_Buffer_MRFaceId_MRFaceId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRFaceId_MRFaceId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRFaceId_MRFaceId(MR.Misc._Moved<Buffer_MRFaceId_MRFaceId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRFaceId_MRFaceId(MR.Misc._Moved<Buffer_MRFaceId_MRFaceId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRFaceId_MRFaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRFaceId_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRFaceId_MRFaceId`/`Const_Buffer_MRFaceId_MRFaceId` directly.
    public class _InOptMut_Buffer_MRFaceId_MRFaceId
    {
        public Buffer_MRFaceId_MRFaceId? Opt;

        public _InOptMut_Buffer_MRFaceId_MRFaceId() {}
        public _InOptMut_Buffer_MRFaceId_MRFaceId(Buffer_MRFaceId_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRFaceId_MRFaceId(Buffer_MRFaceId_MRFaceId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRFaceId_MRFaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRFaceId_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRFaceId_MRFaceId`/`Const_Buffer_MRFaceId_MRFaceId` to pass it to the function.
    public class _InOptConst_Buffer_MRFaceId_MRFaceId
    {
        public Const_Buffer_MRFaceId_MRFaceId? Opt;

        public _InOptConst_Buffer_MRFaceId_MRFaceId() {}
        public _InOptConst_Buffer_MRFaceId_MRFaceId(Const_Buffer_MRFaceId_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRFaceId_MRFaceId(Const_Buffer_MRFaceId_MRFaceId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::VertId, MR::VertId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRVertId_MRVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRVertId_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VertId_MR_VertId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_VertId_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRVertId_MRVertId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRVertId_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_VertId_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::VertId, MR::VertId>::Buffer`.
        public unsafe Const_Buffer_MRVertId_MRVertId(MR._ByValue_Buffer_MRVertId_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVertId_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::VertId, MR::VertId>::Buffer`.
        public unsafe Const_Buffer_MRVertId_MRVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_MR_VertId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_MR_VertId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_MR_VertId_size(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_VertId_MR_VertId_empty(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRVertId Index(MR.VertId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_index_const(_Underlying *_this, MR.VertId i);
            return new(__MR_Buffer_MR_VertId_MR_VertId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::data`.
        public unsafe MR.NoDefInit_MRVertId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_VertId_MR_VertId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRVertId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::beginId`.
        public unsafe MR.VertId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_beginId", ExactSpelling = true)]
            extern static MR.VertId __MR_Buffer_MR_VertId_MR_VertId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::backId`.
        public unsafe MR.VertId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_backId", ExactSpelling = true)]
            extern static MR.VertId __MR_Buffer_MR_VertId_MR_VertId_backId(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::endId`.
        public unsafe MR.VertId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_endId", ExactSpelling = true)]
            extern static MR.VertId __MR_Buffer_MR_VertId_MR_VertId_endId(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_MR_VertId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_VertId_MR_VertId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::VertId, MR::VertId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRVertId_MRVertId : Const_Buffer_MRVertId_MRVertId
    {
        internal unsafe Buffer_MRVertId_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRVertId_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_VertId_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::VertId, MR::VertId>::Buffer`.
        public unsafe Buffer_MRVertId_MRVertId(MR._ByValue_Buffer_MRVertId_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVertId_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::VertId, MR::VertId>::Buffer`.
        public unsafe Buffer_MRVertId_MRVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_MR_VertId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::operator=`.
        public unsafe MR.Buffer_MRVertId_MRVertId Assign(MR._ByValue_Buffer_MRVertId_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVertId_MRVertId._Underlying *_other);
            return new(__MR_Buffer_MR_VertId_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VertId_MR_VertId_clear(_Underlying *_this);
            __MR_Buffer_MR_VertId_MR_VertId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VertId_MR_VertId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_VertId_MR_VertId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::operator[]`.
        public unsafe new MR.NoDefInit_MRVertId Index(MR.VertId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_index(_Underlying *_this, MR.VertId i);
            return new(__MR_Buffer_MR_VertId_MR_VertId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VertId, MR::VertId>::data`.
        public unsafe new MR.NoDefInit_MRVertId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_MR_VertId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_MR_VertId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_VertId_MR_VertId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRVertId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRVertId_MRVertId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRVertId_MRVertId`/`Const_Buffer_MRVertId_MRVertId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRVertId_MRVertId
    {
        internal readonly Const_Buffer_MRVertId_MRVertId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRVertId_MRVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRVertId_MRVertId(MR.Misc._Moved<Buffer_MRVertId_MRVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRVertId_MRVertId(MR.Misc._Moved<Buffer_MRVertId_MRVertId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRVertId_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRVertId_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRVertId_MRVertId`/`Const_Buffer_MRVertId_MRVertId` directly.
    public class _InOptMut_Buffer_MRVertId_MRVertId
    {
        public Buffer_MRVertId_MRVertId? Opt;

        public _InOptMut_Buffer_MRVertId_MRVertId() {}
        public _InOptMut_Buffer_MRVertId_MRVertId(Buffer_MRVertId_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRVertId_MRVertId(Buffer_MRVertId_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRVertId_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRVertId_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRVertId_MRVertId`/`Const_Buffer_MRVertId_MRVertId` to pass it to the function.
    public class _InOptConst_Buffer_MRVertId_MRVertId
    {
        public Const_Buffer_MRVertId_MRVertId? Opt;

        public _InOptConst_Buffer_MRVertId_MRVertId() {}
        public _InOptConst_Buffer_MRVertId_MRVertId(Const_Buffer_MRVertId_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRVertId_MRVertId(Const_Buffer_MRVertId_MRVertId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<unsigned char>`.
    /// This is the const half of the class.
    public class Const_Buffer_UnsignedChar : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_unsigned_char_Destroy(_Underlying *_this);
            __MR_Buffer_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_UnsignedChar() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<unsigned char>::Buffer`.
        public unsafe Const_Buffer_UnsignedChar(MR._ByValue_Buffer_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_unsigned_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<unsigned char>::Buffer`.
        public unsafe Const_Buffer_UnsignedChar(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_unsigned_char_Construct(size);
        }

        /// Generated from method `MR::Buffer<unsigned char>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_unsigned_char_capacity(_Underlying *_this);
            return __MR_Buffer_unsigned_char_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<unsigned char>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_unsigned_char_size(_Underlying *_this);
            return __MR_Buffer_unsigned_char_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<unsigned char>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_unsigned_char_empty(_Underlying *_this);
            return __MR_Buffer_unsigned_char_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<unsigned char>::operator[]`.
        public unsafe byte Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_index_const", ExactSpelling = true)]
            extern static byte *__MR_Buffer_unsigned_char_index_const(_Underlying *_this, ulong i);
            return *__MR_Buffer_unsigned_char_index_const(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::Buffer<unsigned char>::data`.
        public unsafe MR.Misc.Ref<byte>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_data_const", ExactSpelling = true)]
            extern static byte *__MR_Buffer_unsigned_char_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_unsigned_char_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<byte>(__ret) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<unsigned char>::beginId`.
        public unsafe ulong BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_beginId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_unsigned_char_beginId(_Underlying *_this);
            return __MR_Buffer_unsigned_char_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<unsigned char>::backId`.
        public unsafe ulong BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_backId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_unsigned_char_backId(_Underlying *_this);
            return __MR_Buffer_unsigned_char_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<unsigned char>::endId`.
        public unsafe ulong EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_endId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_unsigned_char_endId(_Underlying *_this);
            return __MR_Buffer_unsigned_char_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<unsigned char>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_unsigned_char_heapBytes(_Underlying *_this);
            return __MR_Buffer_unsigned_char_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<unsigned char>`.
    /// This is the non-const half of the class.
    public class Buffer_UnsignedChar : Const_Buffer_UnsignedChar
    {
        internal unsafe Buffer_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<unsigned char>::Buffer`.
        public unsafe Buffer_UnsignedChar(MR._ByValue_Buffer_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_unsigned_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<unsigned char>::Buffer`.
        public unsafe Buffer_UnsignedChar(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_Construct", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_unsigned_char_Construct(size);
        }

        /// Generated from method `MR::Buffer<unsigned char>::operator=`.
        public unsafe MR.Buffer_UnsignedChar Assign(MR._ByValue_Buffer_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_UnsignedChar._Underlying *__MR_Buffer_unsigned_char_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_UnsignedChar._Underlying *_other);
            return new(__MR_Buffer_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<unsigned char>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_unsigned_char_clear(_Underlying *_this);
            __MR_Buffer_unsigned_char_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<unsigned char>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_unsigned_char_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_unsigned_char_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<unsigned char>::operator[]`.
        public unsafe new ref byte Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_index", ExactSpelling = true)]
            extern static byte *__MR_Buffer_unsigned_char_index(_Underlying *_this, ulong i);
            return ref *__MR_Buffer_unsigned_char_index(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::Buffer<unsigned char>::data`.
        public unsafe new MR.Misc.Ref<byte>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_unsigned_char_data", ExactSpelling = true)]
            extern static byte *__MR_Buffer_unsigned_char_data(_Underlying *_this);
            var __ret = __MR_Buffer_unsigned_char_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<byte>(__ret) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_UnsignedChar` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_UnsignedChar`/`Const_Buffer_UnsignedChar` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_UnsignedChar
    {
        internal readonly Const_Buffer_UnsignedChar? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_UnsignedChar() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_UnsignedChar(MR.Misc._Moved<Buffer_UnsignedChar> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_UnsignedChar(MR.Misc._Moved<Buffer_UnsignedChar> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_UnsignedChar`/`Const_Buffer_UnsignedChar` directly.
    public class _InOptMut_Buffer_UnsignedChar
    {
        public Buffer_UnsignedChar? Opt;

        public _InOptMut_Buffer_UnsignedChar() {}
        public _InOptMut_Buffer_UnsignedChar(Buffer_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_UnsignedChar(Buffer_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_UnsignedChar`/`Const_Buffer_UnsignedChar` to pass it to the function.
    public class _InOptConst_Buffer_UnsignedChar
    {
        public Const_Buffer_UnsignedChar? Opt;

        public _InOptConst_Buffer_UnsignedChar() {}
        public _InOptConst_Buffer_UnsignedChar(Const_Buffer_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_UnsignedChar(Const_Buffer_UnsignedChar value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::VertId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VertId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRVertId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::VertId>::Buffer`.
        public unsafe Const_Buffer_MRVertId(MR._ByValue_Buffer_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::VertId>::Buffer`.
        public unsafe Const_Buffer_MRVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_VertId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_size(_Underlying *_this);
            return __MR_Buffer_MR_VertId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_VertId_empty(_Underlying *_this);
            return __MR_Buffer_MR_VertId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::VertId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRVertId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_index_const(_Underlying *_this, ulong i);
            return new(__MR_Buffer_MR_VertId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::data`.
        public unsafe MR.NoDefInit_MRVertId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_VertId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRVertId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::VertId>::beginId`.
        public unsafe ulong BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_beginId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_VertId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::VertId>::backId`.
        public unsafe ulong BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_backId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_backId(_Underlying *_this);
            return __MR_Buffer_MR_VertId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::VertId>::endId`.
        public unsafe ulong EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_endId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_endId(_Underlying *_this);
            return __MR_Buffer_MR_VertId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::VertId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VertId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_VertId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::VertId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRVertId : Const_Buffer_MRVertId
    {
        internal unsafe Buffer_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::VertId>::Buffer`.
        public unsafe Buffer_MRVertId(MR._ByValue_Buffer_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::VertId>::Buffer`.
        public unsafe Buffer_MRVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_VertId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::operator=`.
        public unsafe MR.Buffer_MRVertId Assign(MR._ByValue_Buffer_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVertId._Underlying *__MR_Buffer_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVertId._Underlying *_other);
            return new(__MR_Buffer_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VertId_clear(_Underlying *_this);
            __MR_Buffer_MR_VertId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VertId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_VertId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::operator[]`.
        public unsafe new MR.NoDefInit_MRVertId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_index(_Underlying *_this, ulong i);
            return new(__MR_Buffer_MR_VertId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VertId>::data`.
        public unsafe new MR.NoDefInit_MRVertId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VertId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVertId._Underlying *__MR_Buffer_MR_VertId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_VertId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRVertId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRVertId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRVertId`/`Const_Buffer_MRVertId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRVertId
    {
        internal readonly Const_Buffer_MRVertId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRVertId(MR.Misc._Moved<Buffer_MRVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRVertId(MR.Misc._Moved<Buffer_MRVertId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRVertId`/`Const_Buffer_MRVertId` directly.
    public class _InOptMut_Buffer_MRVertId
    {
        public Buffer_MRVertId? Opt;

        public _InOptMut_Buffer_MRVertId() {}
        public _InOptMut_Buffer_MRVertId(Buffer_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRVertId(Buffer_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRVertId`/`Const_Buffer_MRVertId` to pass it to the function.
    public class _InOptConst_Buffer_MRVertId
    {
        public Const_Buffer_MRVertId? Opt;

        public _InOptConst_Buffer_MRVertId() {}
        public _InOptConst_Buffer_MRVertId(Const_Buffer_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRVertId(Const_Buffer_MRVertId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::UndirectedEdgeId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRUndirectedEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId>::Buffer`.
        public unsafe Const_Buffer_MRUndirectedEdgeId(MR._ByValue_Buffer_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId>::Buffer`.
        public unsafe Const_Buffer_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_size(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_UndirectedEdgeId_empty(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRUndirectedEdgeId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_index_const(_Underlying *_this, ulong i);
            return new(__MR_Buffer_MR_UndirectedEdgeId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::data`.
        public unsafe MR.NoDefInit_MRUndirectedEdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_UndirectedEdgeId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRUndirectedEdgeId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::beginId`.
        public unsafe ulong BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_beginId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::backId`.
        public unsafe ulong BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_backId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_backId(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::endId`.
        public unsafe ulong EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_endId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_endId(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_UndirectedEdgeId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_UndirectedEdgeId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::UndirectedEdgeId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRUndirectedEdgeId : Const_Buffer_MRUndirectedEdgeId
    {
        internal unsafe Buffer_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId>::Buffer`.
        public unsafe Buffer_MRUndirectedEdgeId(MR._ByValue_Buffer_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::UndirectedEdgeId>::Buffer`.
        public unsafe Buffer_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.Buffer_MRUndirectedEdgeId Assign(MR._ByValue_Buffer_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_Buffer_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_UndirectedEdgeId_clear(_Underlying *_this);
            __MR_Buffer_MR_UndirectedEdgeId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_UndirectedEdgeId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_UndirectedEdgeId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::operator[]`.
        public unsafe new MR.NoDefInit_MRUndirectedEdgeId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_index(_Underlying *_this, ulong i);
            return new(__MR_Buffer_MR_UndirectedEdgeId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::UndirectedEdgeId>::data`.
        public unsafe new MR.NoDefInit_MRUndirectedEdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_UndirectedEdgeId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_UndirectedEdgeId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_UndirectedEdgeId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRUndirectedEdgeId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRUndirectedEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRUndirectedEdgeId`/`Const_Buffer_MRUndirectedEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRUndirectedEdgeId
    {
        internal readonly Const_Buffer_MRUndirectedEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRUndirectedEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRUndirectedEdgeId(MR.Misc._Moved<Buffer_MRUndirectedEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRUndirectedEdgeId(MR.Misc._Moved<Buffer_MRUndirectedEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRUndirectedEdgeId`/`Const_Buffer_MRUndirectedEdgeId` directly.
    public class _InOptMut_Buffer_MRUndirectedEdgeId
    {
        public Buffer_MRUndirectedEdgeId? Opt;

        public _InOptMut_Buffer_MRUndirectedEdgeId() {}
        public _InOptMut_Buffer_MRUndirectedEdgeId(Buffer_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRUndirectedEdgeId(Buffer_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRUndirectedEdgeId`/`Const_Buffer_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_Buffer_MRUndirectedEdgeId
    {
        public Const_Buffer_MRUndirectedEdgeId? Opt;

        public _InOptConst_Buffer_MRUndirectedEdgeId() {}
        public _InOptConst_Buffer_MRUndirectedEdgeId(Const_Buffer_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRUndirectedEdgeId(Const_Buffer_MRUndirectedEdgeId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::FaceId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRFaceId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_FaceId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRFaceId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId>::Buffer`.
        public unsafe Const_Buffer_MRFaceId(MR._ByValue_Buffer_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId>::Buffer`.
        public unsafe Const_Buffer_MRFaceId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_size(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_FaceId_empty(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRFaceId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_index_const(_Underlying *_this, ulong i);
            return new(__MR_Buffer_MR_FaceId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::data`.
        public unsafe MR.NoDefInit_MRFaceId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_FaceId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRFaceId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::FaceId>::beginId`.
        public unsafe ulong BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_beginId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::FaceId>::backId`.
        public unsafe ulong BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_backId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_backId(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::FaceId>::endId`.
        public unsafe ulong EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_endId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_endId(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::FaceId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_FaceId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_FaceId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::FaceId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRFaceId : Const_Buffer_MRFaceId
    {
        internal unsafe Buffer_MRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId>::Buffer`.
        public unsafe Buffer_MRFaceId(MR._ByValue_Buffer_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::FaceId>::Buffer`.
        public unsafe Buffer_MRFaceId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_FaceId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::operator=`.
        public unsafe MR.Buffer_MRFaceId Assign(MR._ByValue_Buffer_MRFaceId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRFaceId._Underlying *_other);
            return new(__MR_Buffer_MR_FaceId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_FaceId_clear(_Underlying *_this);
            __MR_Buffer_MR_FaceId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_FaceId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_FaceId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::operator[]`.
        public unsafe new MR.NoDefInit_MRFaceId Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_index(_Underlying *_this, ulong i);
            return new(__MR_Buffer_MR_FaceId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::FaceId>::data`.
        public unsafe new MR.NoDefInit_MRFaceId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_FaceId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRFaceId._Underlying *__MR_Buffer_MR_FaceId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_FaceId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRFaceId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRFaceId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRFaceId`/`Const_Buffer_MRFaceId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRFaceId
    {
        internal readonly Const_Buffer_MRFaceId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRFaceId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRFaceId(MR.Misc._Moved<Buffer_MRFaceId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRFaceId(MR.Misc._Moved<Buffer_MRFaceId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRFaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRFaceId`/`Const_Buffer_MRFaceId` directly.
    public class _InOptMut_Buffer_MRFaceId
    {
        public Buffer_MRFaceId? Opt;

        public _InOptMut_Buffer_MRFaceId() {}
        public _InOptMut_Buffer_MRFaceId(Buffer_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRFaceId(Buffer_MRFaceId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRFaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRFaceId`/`Const_Buffer_MRFaceId` to pass it to the function.
    public class _InOptConst_Buffer_MRFaceId
    {
        public Const_Buffer_MRFaceId? Opt;

        public _InOptConst_Buffer_MRFaceId() {}
        public _InOptConst_Buffer_MRFaceId(Const_Buffer_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRFaceId(Const_Buffer_MRFaceId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<char>`.
    /// This is the const half of the class.
    public class Const_Buffer_Char : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_Char(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_char_Destroy(_Underlying *_this);
            __MR_Buffer_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_Char() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_Char() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<char>::Buffer`.
        public unsafe Const_Buffer_Char(MR._ByValue_Buffer_Char _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_Char._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<char>::Buffer`.
        public unsafe Const_Buffer_Char(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_Construct", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_char_Construct(size);
        }

        /// Generated from method `MR::Buffer<char>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_char_capacity(_Underlying *_this);
            return __MR_Buffer_char_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<char>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_char_size(_Underlying *_this);
            return __MR_Buffer_char_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<char>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_char_empty(_Underlying *_this);
            return __MR_Buffer_char_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<char>::operator[]`.
        public unsafe byte Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_index_const", ExactSpelling = true)]
            extern static byte *__MR_Buffer_char_index_const(_Underlying *_this, ulong i);
            return *__MR_Buffer_char_index_const(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::Buffer<char>::data`.
        public unsafe MR.Misc.Ref<byte>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_data_const", ExactSpelling = true)]
            extern static byte *__MR_Buffer_char_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_char_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<byte>(__ret) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<char>::beginId`.
        public unsafe ulong BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_beginId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_char_beginId(_Underlying *_this);
            return __MR_Buffer_char_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<char>::backId`.
        public unsafe ulong BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_backId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_char_backId(_Underlying *_this);
            return __MR_Buffer_char_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<char>::endId`.
        public unsafe ulong EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_endId", ExactSpelling = true)]
            extern static ulong __MR_Buffer_char_endId(_Underlying *_this);
            return __MR_Buffer_char_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<char>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_char_heapBytes(_Underlying *_this);
            return __MR_Buffer_char_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<char>`.
    /// This is the non-const half of the class.
    public class Buffer_Char : Const_Buffer_Char
    {
        internal unsafe Buffer_Char(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_Char() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<char>::Buffer`.
        public unsafe Buffer_Char(MR._ByValue_Buffer_Char _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_Char._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_char_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<char>::Buffer`.
        public unsafe Buffer_Char(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_Construct", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_char_Construct(size);
        }

        /// Generated from method `MR::Buffer<char>::operator=`.
        public unsafe MR.Buffer_Char Assign(MR._ByValue_Buffer_Char _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_Char._Underlying *__MR_Buffer_char_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_Char._Underlying *_other);
            return new(__MR_Buffer_char_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<char>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_char_clear(_Underlying *_this);
            __MR_Buffer_char_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<char>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_char_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_char_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<char>::operator[]`.
        public unsafe new ref byte Index(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_index", ExactSpelling = true)]
            extern static byte *__MR_Buffer_char_index(_Underlying *_this, ulong i);
            return ref *__MR_Buffer_char_index(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::Buffer<char>::data`.
        public unsafe new MR.Misc.Ref<byte>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_char_data", ExactSpelling = true)]
            extern static byte *__MR_Buffer_char_data(_Underlying *_this);
            var __ret = __MR_Buffer_char_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<byte>(__ret) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_Char` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_Char`/`Const_Buffer_Char` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_Char
    {
        internal readonly Const_Buffer_Char? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_Char() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_Char(MR.Misc._Moved<Buffer_Char> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_Char(MR.Misc._Moved<Buffer_Char> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_Char` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_Char`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_Char`/`Const_Buffer_Char` directly.
    public class _InOptMut_Buffer_Char
    {
        public Buffer_Char? Opt;

        public _InOptMut_Buffer_Char() {}
        public _InOptMut_Buffer_Char(Buffer_Char value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_Char(Buffer_Char value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_Char` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_Char`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_Char`/`Const_Buffer_Char` to pass it to the function.
    public class _InOptConst_Buffer_Char
    {
        public Const_Buffer_Char? Opt;

        public _InOptConst_Buffer_Char() {}
        public _InOptConst_Buffer_Char(Const_Buffer_Char value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_Char(Const_Buffer_Char value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::EdgeId, MR::EdgeId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MREdgeId_MREdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MREdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_EdgeId_MR_EdgeId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_EdgeId_MR_EdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MREdgeId_MREdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MREdgeId_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::EdgeId>::Buffer`.
        public unsafe Const_Buffer_MREdgeId_MREdgeId(MR._ByValue_Buffer_MREdgeId_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MREdgeId_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_EdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::EdgeId>::Buffer`.
        public unsafe Const_Buffer_MREdgeId_MREdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_EdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_EdgeId_MR_EdgeId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_EdgeId_MR_EdgeId_size(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_EdgeId_MR_EdgeId_empty(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MREdgeId Index(MR.EdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_index_const(_Underlying *_this, MR.EdgeId i);
            return new(__MR_Buffer_MR_EdgeId_MR_EdgeId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::data`.
        public unsafe MR.NoDefInit_MREdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_EdgeId_MR_EdgeId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MREdgeId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::beginId`.
        public unsafe MR.EdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_beginId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Buffer_MR_EdgeId_MR_EdgeId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::backId`.
        public unsafe MR.EdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_backId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Buffer_MR_EdgeId_MR_EdgeId_backId(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::endId`.
        public unsafe MR.EdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_endId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Buffer_MR_EdgeId_MR_EdgeId_endId(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_EdgeId_MR_EdgeId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_EdgeId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::EdgeId, MR::EdgeId>`.
    /// This is the non-const half of the class.
    public class Buffer_MREdgeId_MREdgeId : Const_Buffer_MREdgeId_MREdgeId
    {
        internal unsafe Buffer_MREdgeId_MREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MREdgeId_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::EdgeId>::Buffer`.
        public unsafe Buffer_MREdgeId_MREdgeId(MR._ByValue_Buffer_MREdgeId_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MREdgeId_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_EdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::EdgeId>::Buffer`.
        public unsafe Buffer_MREdgeId_MREdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_EdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::operator=`.
        public unsafe MR.Buffer_MREdgeId_MREdgeId Assign(MR._ByValue_Buffer_MREdgeId_MREdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MREdgeId_MREdgeId._Underlying *_other);
            return new(__MR_Buffer_MR_EdgeId_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_EdgeId_MR_EdgeId_clear(_Underlying *_this);
            __MR_Buffer_MR_EdgeId_MR_EdgeId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_EdgeId_MR_EdgeId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_EdgeId_MR_EdgeId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::operator[]`.
        public unsafe new MR.NoDefInit_MREdgeId Index(MR.EdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_index(_Underlying *_this, MR.EdgeId i);
            return new(__MR_Buffer_MR_EdgeId_MR_EdgeId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::EdgeId>::data`.
        public unsafe new MR.NoDefInit_MREdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_EdgeId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_EdgeId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_EdgeId_MR_EdgeId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MREdgeId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MREdgeId_MREdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MREdgeId_MREdgeId`/`Const_Buffer_MREdgeId_MREdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MREdgeId_MREdgeId
    {
        internal readonly Const_Buffer_MREdgeId_MREdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MREdgeId_MREdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MREdgeId_MREdgeId(MR.Misc._Moved<Buffer_MREdgeId_MREdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MREdgeId_MREdgeId(MR.Misc._Moved<Buffer_MREdgeId_MREdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MREdgeId_MREdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MREdgeId_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MREdgeId_MREdgeId`/`Const_Buffer_MREdgeId_MREdgeId` directly.
    public class _InOptMut_Buffer_MREdgeId_MREdgeId
    {
        public Buffer_MREdgeId_MREdgeId? Opt;

        public _InOptMut_Buffer_MREdgeId_MREdgeId() {}
        public _InOptMut_Buffer_MREdgeId_MREdgeId(Buffer_MREdgeId_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MREdgeId_MREdgeId(Buffer_MREdgeId_MREdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MREdgeId_MREdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MREdgeId_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MREdgeId_MREdgeId`/`Const_Buffer_MREdgeId_MREdgeId` to pass it to the function.
    public class _InOptConst_Buffer_MREdgeId_MREdgeId
    {
        public Const_Buffer_MREdgeId_MREdgeId? Opt;

        public _InOptConst_Buffer_MREdgeId_MREdgeId() {}
        public _InOptConst_Buffer_MREdgeId_MREdgeId(Const_Buffer_MREdgeId_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MREdgeId_MREdgeId(Const_Buffer_MREdgeId_MREdgeId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MREdgeId_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MREdgeId_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MREdgeId_MRUndirectedEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MREdgeId_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Const_Buffer_MREdgeId_MRUndirectedEdgeId(MR._ByValue_Buffer_MREdgeId_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Const_Buffer_MREdgeId_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_size(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_empty(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MREdgeId Index(MR.UndirectedEdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_index_const(_Underlying *_this, MR.UndirectedEdgeId i);
            return new(__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::data`.
        public unsafe MR.NoDefInit_MREdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MREdgeId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::beginId`.
        public unsafe MR.UndirectedEdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_beginId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::backId`.
        public unsafe MR.UndirectedEdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_backId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_backId(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::endId`.
        public unsafe MR.UndirectedEdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_endId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_endId(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>`.
    /// This is the non-const half of the class.
    public class Buffer_MREdgeId_MRUndirectedEdgeId : Const_Buffer_MREdgeId_MRUndirectedEdgeId
    {
        internal unsafe Buffer_MREdgeId_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MREdgeId_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Buffer_MREdgeId_MRUndirectedEdgeId(MR._ByValue_Buffer_MREdgeId_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::Buffer`.
        public unsafe Buffer_MREdgeId_MRUndirectedEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.Buffer_MREdgeId_MRUndirectedEdgeId Assign(MR._ByValue_Buffer_MREdgeId_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_clear(_Underlying *_this);
            __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::operator[]`.
        public unsafe new MR.NoDefInit_MREdgeId Index(MR.UndirectedEdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_index(_Underlying *_this, MR.UndirectedEdgeId i);
            return new(__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::EdgeId, MR::UndirectedEdgeId>::data`.
        public unsafe new MR.NoDefInit_MREdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MREdgeId._Underlying *__MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_EdgeId_MR_UndirectedEdgeId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MREdgeId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MREdgeId_MRUndirectedEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MREdgeId_MRUndirectedEdgeId`/`Const_Buffer_MREdgeId_MRUndirectedEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MREdgeId_MRUndirectedEdgeId
    {
        internal readonly Const_Buffer_MREdgeId_MRUndirectedEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MREdgeId_MRUndirectedEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MREdgeId_MRUndirectedEdgeId(MR.Misc._Moved<Buffer_MREdgeId_MRUndirectedEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MREdgeId_MRUndirectedEdgeId(MR.Misc._Moved<Buffer_MREdgeId_MRUndirectedEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MREdgeId_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MREdgeId_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MREdgeId_MRUndirectedEdgeId`/`Const_Buffer_MREdgeId_MRUndirectedEdgeId` directly.
    public class _InOptMut_Buffer_MREdgeId_MRUndirectedEdgeId
    {
        public Buffer_MREdgeId_MRUndirectedEdgeId? Opt;

        public _InOptMut_Buffer_MREdgeId_MRUndirectedEdgeId() {}
        public _InOptMut_Buffer_MREdgeId_MRUndirectedEdgeId(Buffer_MREdgeId_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MREdgeId_MRUndirectedEdgeId(Buffer_MREdgeId_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MREdgeId_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MREdgeId_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MREdgeId_MRUndirectedEdgeId`/`Const_Buffer_MREdgeId_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_Buffer_MREdgeId_MRUndirectedEdgeId
    {
        public Const_Buffer_MREdgeId_MRUndirectedEdgeId? Opt;

        public _InOptConst_Buffer_MREdgeId_MRUndirectedEdgeId() {}
        public _InOptConst_Buffer_MREdgeId_MRUndirectedEdgeId(Const_Buffer_MREdgeId_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MREdgeId_MRUndirectedEdgeId(Const_Buffer_MREdgeId_MRUndirectedEdgeId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::ObjId, MR::ObjId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRObjId_MRObjId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRObjId_MRObjId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_ObjId_MR_ObjId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_ObjId_MR_ObjId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRObjId_MRObjId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRObjId_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_ObjId_MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::ObjId, MR::ObjId>::Buffer`.
        public unsafe Const_Buffer_MRObjId_MRObjId(MR._ByValue_Buffer_MRObjId_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRObjId_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_ObjId_MR_ObjId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::ObjId, MR::ObjId>::Buffer`.
        public unsafe Const_Buffer_MRObjId_MRObjId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_ObjId_MR_ObjId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_ObjId_MR_ObjId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_ObjId_MR_ObjId_size(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_ObjId_MR_ObjId_empty(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRObjId Index(MR.ObjId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_index_const(_Underlying *_this, MR.ObjId i);
            return new(__MR_Buffer_MR_ObjId_MR_ObjId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::data`.
        public unsafe MR.NoDefInit_MRObjId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_ObjId_MR_ObjId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRObjId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::beginId`.
        public unsafe MR.ObjId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_beginId", ExactSpelling = true)]
            extern static MR.ObjId __MR_Buffer_MR_ObjId_MR_ObjId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::backId`.
        public unsafe MR.ObjId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_backId", ExactSpelling = true)]
            extern static MR.ObjId __MR_Buffer_MR_ObjId_MR_ObjId_backId(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::endId`.
        public unsafe MR.ObjId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_endId", ExactSpelling = true)]
            extern static MR.ObjId __MR_Buffer_MR_ObjId_MR_ObjId_endId(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_ObjId_MR_ObjId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_ObjId_MR_ObjId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::ObjId, MR::ObjId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRObjId_MRObjId : Const_Buffer_MRObjId_MRObjId
    {
        internal unsafe Buffer_MRObjId_MRObjId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRObjId_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_ObjId_MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::ObjId, MR::ObjId>::Buffer`.
        public unsafe Buffer_MRObjId_MRObjId(MR._ByValue_Buffer_MRObjId_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRObjId_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_ObjId_MR_ObjId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::ObjId, MR::ObjId>::Buffer`.
        public unsafe Buffer_MRObjId_MRObjId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_ObjId_MR_ObjId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::operator=`.
        public unsafe MR.Buffer_MRObjId_MRObjId Assign(MR._ByValue_Buffer_MRObjId_MRObjId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRObjId_MRObjId._Underlying *_other);
            return new(__MR_Buffer_MR_ObjId_MR_ObjId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_ObjId_MR_ObjId_clear(_Underlying *_this);
            __MR_Buffer_MR_ObjId_MR_ObjId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_ObjId_MR_ObjId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_ObjId_MR_ObjId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::operator[]`.
        public unsafe new MR.NoDefInit_MRObjId Index(MR.ObjId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_index(_Underlying *_this, MR.ObjId i);
            return new(__MR_Buffer_MR_ObjId_MR_ObjId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::ObjId, MR::ObjId>::data`.
        public unsafe new MR.NoDefInit_MRObjId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_ObjId_MR_ObjId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRObjId._Underlying *__MR_Buffer_MR_ObjId_MR_ObjId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_ObjId_MR_ObjId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRObjId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRObjId_MRObjId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRObjId_MRObjId`/`Const_Buffer_MRObjId_MRObjId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRObjId_MRObjId
    {
        internal readonly Const_Buffer_MRObjId_MRObjId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRObjId_MRObjId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRObjId_MRObjId(MR.Misc._Moved<Buffer_MRObjId_MRObjId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRObjId_MRObjId(MR.Misc._Moved<Buffer_MRObjId_MRObjId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRObjId_MRObjId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRObjId_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRObjId_MRObjId`/`Const_Buffer_MRObjId_MRObjId` directly.
    public class _InOptMut_Buffer_MRObjId_MRObjId
    {
        public Buffer_MRObjId_MRObjId? Opt;

        public _InOptMut_Buffer_MRObjId_MRObjId() {}
        public _InOptMut_Buffer_MRObjId_MRObjId(Buffer_MRObjId_MRObjId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRObjId_MRObjId(Buffer_MRObjId_MRObjId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRObjId_MRObjId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRObjId_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRObjId_MRObjId`/`Const_Buffer_MRObjId_MRObjId` to pass it to the function.
    public class _InOptConst_Buffer_MRObjId_MRObjId
    {
        public Const_Buffer_MRObjId_MRObjId? Opt;

        public _InOptConst_Buffer_MRObjId_MRObjId() {}
        public _InOptConst_Buffer_MRObjId_MRObjId(Const_Buffer_MRObjId_MRObjId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRObjId_MRObjId(Const_Buffer_MRObjId_MRObjId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::GraphVertId, MR::GraphVertId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRGraphVertId_MRGraphVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRGraphVertId_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_GraphVertId_MR_GraphVertId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_GraphVertId_MR_GraphVertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRGraphVertId_MRGraphVertId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRGraphVertId_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::Buffer`.
        public unsafe Const_Buffer_MRGraphVertId_MRGraphVertId(MR._ByValue_Buffer_MRGraphVertId_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::Buffer`.
        public unsafe Const_Buffer_MRGraphVertId_MRGraphVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_GraphVertId_MR_GraphVertId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_GraphVertId_MR_GraphVertId_size(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_GraphVertId_MR_GraphVertId_empty(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRGraphVertId Index(MR.GraphVertId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_index_const(_Underlying *_this, MR.GraphVertId i);
            return new(__MR_Buffer_MR_GraphVertId_MR_GraphVertId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::data`.
        public unsafe MR.NoDefInit_MRGraphVertId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRGraphVertId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::beginId`.
        public unsafe MR.GraphVertId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_beginId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_Buffer_MR_GraphVertId_MR_GraphVertId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::backId`.
        public unsafe MR.GraphVertId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_backId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_Buffer_MR_GraphVertId_MR_GraphVertId_backId(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::endId`.
        public unsafe MR.GraphVertId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_endId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_Buffer_MR_GraphVertId_MR_GraphVertId_endId(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_GraphVertId_MR_GraphVertId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_GraphVertId_MR_GraphVertId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::GraphVertId, MR::GraphVertId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRGraphVertId_MRGraphVertId : Const_Buffer_MRGraphVertId_MRGraphVertId
    {
        internal unsafe Buffer_MRGraphVertId_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRGraphVertId_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::Buffer`.
        public unsafe Buffer_MRGraphVertId_MRGraphVertId(MR._ByValue_Buffer_MRGraphVertId_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::Buffer`.
        public unsafe Buffer_MRGraphVertId_MRGraphVertId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::operator=`.
        public unsafe MR.Buffer_MRGraphVertId_MRGraphVertId Assign(MR._ByValue_Buffer_MRGraphVertId_MRGraphVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *_other);
            return new(__MR_Buffer_MR_GraphVertId_MR_GraphVertId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_GraphVertId_MR_GraphVertId_clear(_Underlying *_this);
            __MR_Buffer_MR_GraphVertId_MR_GraphVertId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_GraphVertId_MR_GraphVertId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_GraphVertId_MR_GraphVertId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::operator[]`.
        public unsafe new MR.NoDefInit_MRGraphVertId Index(MR.GraphVertId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_index(_Underlying *_this, MR.GraphVertId i);
            return new(__MR_Buffer_MR_GraphVertId_MR_GraphVertId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::GraphVertId, MR::GraphVertId>::data`.
        public unsafe new MR.NoDefInit_MRGraphVertId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphVertId_MR_GraphVertId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphVertId._Underlying *__MR_Buffer_MR_GraphVertId_MR_GraphVertId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_GraphVertId_MR_GraphVertId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRGraphVertId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRGraphVertId_MRGraphVertId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRGraphVertId_MRGraphVertId`/`Const_Buffer_MRGraphVertId_MRGraphVertId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRGraphVertId_MRGraphVertId
    {
        internal readonly Const_Buffer_MRGraphVertId_MRGraphVertId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRGraphVertId_MRGraphVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRGraphVertId_MRGraphVertId(MR.Misc._Moved<Buffer_MRGraphVertId_MRGraphVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRGraphVertId_MRGraphVertId(MR.Misc._Moved<Buffer_MRGraphVertId_MRGraphVertId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRGraphVertId_MRGraphVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRGraphVertId_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRGraphVertId_MRGraphVertId`/`Const_Buffer_MRGraphVertId_MRGraphVertId` directly.
    public class _InOptMut_Buffer_MRGraphVertId_MRGraphVertId
    {
        public Buffer_MRGraphVertId_MRGraphVertId? Opt;

        public _InOptMut_Buffer_MRGraphVertId_MRGraphVertId() {}
        public _InOptMut_Buffer_MRGraphVertId_MRGraphVertId(Buffer_MRGraphVertId_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRGraphVertId_MRGraphVertId(Buffer_MRGraphVertId_MRGraphVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRGraphVertId_MRGraphVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRGraphVertId_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRGraphVertId_MRGraphVertId`/`Const_Buffer_MRGraphVertId_MRGraphVertId` to pass it to the function.
    public class _InOptConst_Buffer_MRGraphVertId_MRGraphVertId
    {
        public Const_Buffer_MRGraphVertId_MRGraphVertId? Opt;

        public _InOptConst_Buffer_MRGraphVertId_MRGraphVertId() {}
        public _InOptConst_Buffer_MRGraphVertId_MRGraphVertId(Const_Buffer_MRGraphVertId_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRGraphVertId_MRGraphVertId(Const_Buffer_MRGraphVertId_MRGraphVertId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRGraphEdgeId_MRGraphEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRGraphEdgeId_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRGraphEdgeId_MRGraphEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRGraphEdgeId_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::Buffer`.
        public unsafe Const_Buffer_MRGraphEdgeId_MRGraphEdgeId(MR._ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::Buffer`.
        public unsafe Const_Buffer_MRGraphEdgeId_MRGraphEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_size(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_empty(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRGraphEdgeId Index(MR.GraphEdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_index_const(_Underlying *_this, MR.GraphEdgeId i);
            return new(__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::data`.
        public unsafe MR.NoDefInit_MRGraphEdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRGraphEdgeId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::beginId`.
        public unsafe MR.GraphEdgeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_beginId", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::backId`.
        public unsafe MR.GraphEdgeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_backId", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_backId(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::endId`.
        public unsafe MR.GraphEdgeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_endId", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_endId(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRGraphEdgeId_MRGraphEdgeId : Const_Buffer_MRGraphEdgeId_MRGraphEdgeId
    {
        internal unsafe Buffer_MRGraphEdgeId_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRGraphEdgeId_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::Buffer`.
        public unsafe Buffer_MRGraphEdgeId_MRGraphEdgeId(MR._ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::Buffer`.
        public unsafe Buffer_MRGraphEdgeId_MRGraphEdgeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::operator=`.
        public unsafe MR.Buffer_MRGraphEdgeId_MRGraphEdgeId Assign(MR._ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *_other);
            return new(__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_clear(_Underlying *_this);
            __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::operator[]`.
        public unsafe new MR.NoDefInit_MRGraphEdgeId Index(MR.GraphEdgeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_index(_Underlying *_this, MR.GraphEdgeId i);
            return new(__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::GraphEdgeId, MR::GraphEdgeId>::data`.
        public unsafe new MR.NoDefInit_MRGraphEdgeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRGraphEdgeId._Underlying *__MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_GraphEdgeId_MR_GraphEdgeId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRGraphEdgeId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRGraphEdgeId_MRGraphEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRGraphEdgeId_MRGraphEdgeId`/`Const_Buffer_MRGraphEdgeId_MRGraphEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId
    {
        internal readonly Const_Buffer_MRGraphEdgeId_MRGraphEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId(MR.Misc._Moved<Buffer_MRGraphEdgeId_MRGraphEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId(MR.Misc._Moved<Buffer_MRGraphEdgeId_MRGraphEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRGraphEdgeId_MRGraphEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRGraphEdgeId_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRGraphEdgeId_MRGraphEdgeId`/`Const_Buffer_MRGraphEdgeId_MRGraphEdgeId` directly.
    public class _InOptMut_Buffer_MRGraphEdgeId_MRGraphEdgeId
    {
        public Buffer_MRGraphEdgeId_MRGraphEdgeId? Opt;

        public _InOptMut_Buffer_MRGraphEdgeId_MRGraphEdgeId() {}
        public _InOptMut_Buffer_MRGraphEdgeId_MRGraphEdgeId(Buffer_MRGraphEdgeId_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRGraphEdgeId_MRGraphEdgeId(Buffer_MRGraphEdgeId_MRGraphEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRGraphEdgeId_MRGraphEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRGraphEdgeId_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRGraphEdgeId_MRGraphEdgeId`/`Const_Buffer_MRGraphEdgeId_MRGraphEdgeId` to pass it to the function.
    public class _InOptConst_Buffer_MRGraphEdgeId_MRGraphEdgeId
    {
        public Const_Buffer_MRGraphEdgeId_MRGraphEdgeId? Opt;

        public _InOptConst_Buffer_MRGraphEdgeId_MRGraphEdgeId() {}
        public _InOptConst_Buffer_MRGraphEdgeId_MRGraphEdgeId(Const_Buffer_MRGraphEdgeId_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRGraphEdgeId_MRGraphEdgeId(Const_Buffer_MRGraphEdgeId_MRGraphEdgeId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::VoxelId, MR::VoxelId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRVoxelId_MRVoxelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRVoxelId_MRVoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VoxelId_MR_VoxelId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_VoxelId_MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRVoxelId_MRVoxelId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRVoxelId_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_VoxelId_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::VoxelId, MR::VoxelId>::Buffer`.
        public unsafe Const_Buffer_MRVoxelId_MRVoxelId(MR._ByValue_Buffer_MRVoxelId_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVoxelId_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_VoxelId_MR_VoxelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::VoxelId, MR::VoxelId>::Buffer`.
        public unsafe Const_Buffer_MRVoxelId_MRVoxelId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_VoxelId_MR_VoxelId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VoxelId_MR_VoxelId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VoxelId_MR_VoxelId_size(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_VoxelId_MR_VoxelId_empty(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRVoxelId Index(MR.VoxelId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_index_const(_Underlying *_this, MR.VoxelId i);
            return new(__MR_Buffer_MR_VoxelId_MR_VoxelId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::data`.
        public unsafe MR.NoDefInit_MRVoxelId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_VoxelId_MR_VoxelId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRVoxelId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::beginId`.
        public unsafe MR.VoxelId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_beginId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_Buffer_MR_VoxelId_MR_VoxelId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::backId`.
        public unsafe MR.VoxelId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_backId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_Buffer_MR_VoxelId_MR_VoxelId_backId(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::endId`.
        public unsafe MR.VoxelId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_endId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_Buffer_MR_VoxelId_MR_VoxelId_endId(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_VoxelId_MR_VoxelId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_VoxelId_MR_VoxelId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::VoxelId, MR::VoxelId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRVoxelId_MRVoxelId : Const_Buffer_MRVoxelId_MRVoxelId
    {
        internal unsafe Buffer_MRVoxelId_MRVoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRVoxelId_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_VoxelId_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::VoxelId, MR::VoxelId>::Buffer`.
        public unsafe Buffer_MRVoxelId_MRVoxelId(MR._ByValue_Buffer_MRVoxelId_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVoxelId_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_VoxelId_MR_VoxelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::VoxelId, MR::VoxelId>::Buffer`.
        public unsafe Buffer_MRVoxelId_MRVoxelId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_VoxelId_MR_VoxelId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::operator=`.
        public unsafe MR.Buffer_MRVoxelId_MRVoxelId Assign(MR._ByValue_Buffer_MRVoxelId_MRVoxelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRVoxelId_MRVoxelId._Underlying *_other);
            return new(__MR_Buffer_MR_VoxelId_MR_VoxelId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VoxelId_MR_VoxelId_clear(_Underlying *_this);
            __MR_Buffer_MR_VoxelId_MR_VoxelId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_VoxelId_MR_VoxelId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_VoxelId_MR_VoxelId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::operator[]`.
        public unsafe new MR.NoDefInit_MRVoxelId Index(MR.VoxelId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_index(_Underlying *_this, MR.VoxelId i);
            return new(__MR_Buffer_MR_VoxelId_MR_VoxelId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::VoxelId, MR::VoxelId>::data`.
        public unsafe new MR.NoDefInit_MRVoxelId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_VoxelId_MR_VoxelId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRVoxelId._Underlying *__MR_Buffer_MR_VoxelId_MR_VoxelId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_VoxelId_MR_VoxelId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRVoxelId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRVoxelId_MRVoxelId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRVoxelId_MRVoxelId`/`Const_Buffer_MRVoxelId_MRVoxelId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRVoxelId_MRVoxelId
    {
        internal readonly Const_Buffer_MRVoxelId_MRVoxelId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRVoxelId_MRVoxelId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRVoxelId_MRVoxelId(MR.Misc._Moved<Buffer_MRVoxelId_MRVoxelId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRVoxelId_MRVoxelId(MR.Misc._Moved<Buffer_MRVoxelId_MRVoxelId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRVoxelId_MRVoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRVoxelId_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRVoxelId_MRVoxelId`/`Const_Buffer_MRVoxelId_MRVoxelId` directly.
    public class _InOptMut_Buffer_MRVoxelId_MRVoxelId
    {
        public Buffer_MRVoxelId_MRVoxelId? Opt;

        public _InOptMut_Buffer_MRVoxelId_MRVoxelId() {}
        public _InOptMut_Buffer_MRVoxelId_MRVoxelId(Buffer_MRVoxelId_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRVoxelId_MRVoxelId(Buffer_MRVoxelId_MRVoxelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRVoxelId_MRVoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRVoxelId_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRVoxelId_MRVoxelId`/`Const_Buffer_MRVoxelId_MRVoxelId` to pass it to the function.
    public class _InOptConst_Buffer_MRVoxelId_MRVoxelId
    {
        public Const_Buffer_MRVoxelId_MRVoxelId? Opt;

        public _InOptConst_Buffer_MRVoxelId_MRVoxelId() {}
        public _InOptConst_Buffer_MRVoxelId_MRVoxelId(Const_Buffer_MRVoxelId_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRVoxelId_MRVoxelId(Const_Buffer_MRVoxelId_MRVoxelId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::PixelId, MR::PixelId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRPixelId_MRPixelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRPixelId_MRPixelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_PixelId_MR_PixelId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_PixelId_MR_PixelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRPixelId_MRPixelId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRPixelId_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_PixelId_MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::PixelId, MR::PixelId>::Buffer`.
        public unsafe Const_Buffer_MRPixelId_MRPixelId(MR._ByValue_Buffer_MRPixelId_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRPixelId_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_PixelId_MR_PixelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::PixelId, MR::PixelId>::Buffer`.
        public unsafe Const_Buffer_MRPixelId_MRPixelId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_PixelId_MR_PixelId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_PixelId_MR_PixelId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_PixelId_MR_PixelId_size(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_PixelId_MR_PixelId_empty(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRPixelId Index(MR.PixelId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_index_const(_Underlying *_this, MR.PixelId i);
            return new(__MR_Buffer_MR_PixelId_MR_PixelId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::data`.
        public unsafe MR.NoDefInit_MRPixelId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_PixelId_MR_PixelId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRPixelId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::beginId`.
        public unsafe MR.PixelId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_beginId", ExactSpelling = true)]
            extern static MR.PixelId __MR_Buffer_MR_PixelId_MR_PixelId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::backId`.
        public unsafe MR.PixelId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_backId", ExactSpelling = true)]
            extern static MR.PixelId __MR_Buffer_MR_PixelId_MR_PixelId_backId(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::endId`.
        public unsafe MR.PixelId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_endId", ExactSpelling = true)]
            extern static MR.PixelId __MR_Buffer_MR_PixelId_MR_PixelId_endId(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_PixelId_MR_PixelId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_PixelId_MR_PixelId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::PixelId, MR::PixelId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRPixelId_MRPixelId : Const_Buffer_MRPixelId_MRPixelId
    {
        internal unsafe Buffer_MRPixelId_MRPixelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRPixelId_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_PixelId_MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::PixelId, MR::PixelId>::Buffer`.
        public unsafe Buffer_MRPixelId_MRPixelId(MR._ByValue_Buffer_MRPixelId_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRPixelId_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_PixelId_MR_PixelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::PixelId, MR::PixelId>::Buffer`.
        public unsafe Buffer_MRPixelId_MRPixelId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_PixelId_MR_PixelId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::operator=`.
        public unsafe MR.Buffer_MRPixelId_MRPixelId Assign(MR._ByValue_Buffer_MRPixelId_MRPixelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRPixelId_MRPixelId._Underlying *_other);
            return new(__MR_Buffer_MR_PixelId_MR_PixelId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_PixelId_MR_PixelId_clear(_Underlying *_this);
            __MR_Buffer_MR_PixelId_MR_PixelId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_PixelId_MR_PixelId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_PixelId_MR_PixelId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::operator[]`.
        public unsafe new MR.NoDefInit_MRPixelId Index(MR.PixelId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_index(_Underlying *_this, MR.PixelId i);
            return new(__MR_Buffer_MR_PixelId_MR_PixelId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::PixelId, MR::PixelId>::data`.
        public unsafe new MR.NoDefInit_MRPixelId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_PixelId_MR_PixelId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRPixelId._Underlying *__MR_Buffer_MR_PixelId_MR_PixelId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_PixelId_MR_PixelId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRPixelId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRPixelId_MRPixelId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRPixelId_MRPixelId`/`Const_Buffer_MRPixelId_MRPixelId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRPixelId_MRPixelId
    {
        internal readonly Const_Buffer_MRPixelId_MRPixelId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRPixelId_MRPixelId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRPixelId_MRPixelId(MR.Misc._Moved<Buffer_MRPixelId_MRPixelId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRPixelId_MRPixelId(MR.Misc._Moved<Buffer_MRPixelId_MRPixelId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRPixelId_MRPixelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRPixelId_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRPixelId_MRPixelId`/`Const_Buffer_MRPixelId_MRPixelId` directly.
    public class _InOptMut_Buffer_MRPixelId_MRPixelId
    {
        public Buffer_MRPixelId_MRPixelId? Opt;

        public _InOptMut_Buffer_MRPixelId_MRPixelId() {}
        public _InOptMut_Buffer_MRPixelId_MRPixelId(Buffer_MRPixelId_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRPixelId_MRPixelId(Buffer_MRPixelId_MRPixelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRPixelId_MRPixelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRPixelId_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRPixelId_MRPixelId`/`Const_Buffer_MRPixelId_MRPixelId` to pass it to the function.
    public class _InOptConst_Buffer_MRPixelId_MRPixelId
    {
        public Const_Buffer_MRPixelId_MRPixelId? Opt;

        public _InOptConst_Buffer_MRPixelId_MRPixelId() {}
        public _InOptConst_Buffer_MRPixelId_MRPixelId(Const_Buffer_MRPixelId_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRPixelId_MRPixelId(Const_Buffer_MRPixelId_MRPixelId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::RegionId, MR::RegionId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRRegionId_MRRegionId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRRegionId_MRRegionId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_RegionId_MR_RegionId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_RegionId_MR_RegionId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRRegionId_MRRegionId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRRegionId_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_RegionId_MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::RegionId, MR::RegionId>::Buffer`.
        public unsafe Const_Buffer_MRRegionId_MRRegionId(MR._ByValue_Buffer_MRRegionId_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRRegionId_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_RegionId_MR_RegionId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::RegionId, MR::RegionId>::Buffer`.
        public unsafe Const_Buffer_MRRegionId_MRRegionId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_RegionId_MR_RegionId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_RegionId_MR_RegionId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_RegionId_MR_RegionId_size(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_RegionId_MR_RegionId_empty(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRRegionId Index(MR.RegionId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_index_const(_Underlying *_this, MR.RegionId i);
            return new(__MR_Buffer_MR_RegionId_MR_RegionId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::data`.
        public unsafe MR.NoDefInit_MRRegionId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_RegionId_MR_RegionId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRRegionId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::beginId`.
        public unsafe MR.RegionId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_beginId", ExactSpelling = true)]
            extern static MR.RegionId __MR_Buffer_MR_RegionId_MR_RegionId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::backId`.
        public unsafe MR.RegionId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_backId", ExactSpelling = true)]
            extern static MR.RegionId __MR_Buffer_MR_RegionId_MR_RegionId_backId(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::endId`.
        public unsafe MR.RegionId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_endId", ExactSpelling = true)]
            extern static MR.RegionId __MR_Buffer_MR_RegionId_MR_RegionId_endId(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_RegionId_MR_RegionId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_RegionId_MR_RegionId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::RegionId, MR::RegionId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRRegionId_MRRegionId : Const_Buffer_MRRegionId_MRRegionId
    {
        internal unsafe Buffer_MRRegionId_MRRegionId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRRegionId_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_RegionId_MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::RegionId, MR::RegionId>::Buffer`.
        public unsafe Buffer_MRRegionId_MRRegionId(MR._ByValue_Buffer_MRRegionId_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRRegionId_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_RegionId_MR_RegionId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::RegionId, MR::RegionId>::Buffer`.
        public unsafe Buffer_MRRegionId_MRRegionId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_RegionId_MR_RegionId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::operator=`.
        public unsafe MR.Buffer_MRRegionId_MRRegionId Assign(MR._ByValue_Buffer_MRRegionId_MRRegionId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRRegionId_MRRegionId._Underlying *_other);
            return new(__MR_Buffer_MR_RegionId_MR_RegionId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_RegionId_MR_RegionId_clear(_Underlying *_this);
            __MR_Buffer_MR_RegionId_MR_RegionId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_RegionId_MR_RegionId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_RegionId_MR_RegionId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::operator[]`.
        public unsafe new MR.NoDefInit_MRRegionId Index(MR.RegionId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_index(_Underlying *_this, MR.RegionId i);
            return new(__MR_Buffer_MR_RegionId_MR_RegionId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::RegionId, MR::RegionId>::data`.
        public unsafe new MR.NoDefInit_MRRegionId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_RegionId_MR_RegionId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRRegionId._Underlying *__MR_Buffer_MR_RegionId_MR_RegionId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_RegionId_MR_RegionId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRRegionId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRRegionId_MRRegionId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRRegionId_MRRegionId`/`Const_Buffer_MRRegionId_MRRegionId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRRegionId_MRRegionId
    {
        internal readonly Const_Buffer_MRRegionId_MRRegionId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRRegionId_MRRegionId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRRegionId_MRRegionId(MR.Misc._Moved<Buffer_MRRegionId_MRRegionId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRRegionId_MRRegionId(MR.Misc._Moved<Buffer_MRRegionId_MRRegionId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRRegionId_MRRegionId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRRegionId_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRRegionId_MRRegionId`/`Const_Buffer_MRRegionId_MRRegionId` directly.
    public class _InOptMut_Buffer_MRRegionId_MRRegionId
    {
        public Buffer_MRRegionId_MRRegionId? Opt;

        public _InOptMut_Buffer_MRRegionId_MRRegionId() {}
        public _InOptMut_Buffer_MRRegionId_MRRegionId(Buffer_MRRegionId_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRRegionId_MRRegionId(Buffer_MRRegionId_MRRegionId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRRegionId_MRRegionId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRRegionId_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRRegionId_MRRegionId`/`Const_Buffer_MRRegionId_MRRegionId` to pass it to the function.
    public class _InOptConst_Buffer_MRRegionId_MRRegionId
    {
        public Const_Buffer_MRRegionId_MRRegionId? Opt;

        public _InOptConst_Buffer_MRRegionId_MRRegionId() {}
        public _InOptConst_Buffer_MRRegionId_MRRegionId(Const_Buffer_MRRegionId_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRRegionId_MRRegionId(Const_Buffer_MRRegionId_MRRegionId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::NodeId, MR::NodeId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRNodeId_MRNodeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRNodeId_MRNodeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_NodeId_MR_NodeId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_NodeId_MR_NodeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRNodeId_MRNodeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRNodeId_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_NodeId_MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::NodeId, MR::NodeId>::Buffer`.
        public unsafe Const_Buffer_MRNodeId_MRNodeId(MR._ByValue_Buffer_MRNodeId_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRNodeId_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_NodeId_MR_NodeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::NodeId, MR::NodeId>::Buffer`.
        public unsafe Const_Buffer_MRNodeId_MRNodeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_NodeId_MR_NodeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_NodeId_MR_NodeId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_NodeId_MR_NodeId_size(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_NodeId_MR_NodeId_empty(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRNodeId Index(MR.NodeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_index_const(_Underlying *_this, MR.NodeId i);
            return new(__MR_Buffer_MR_NodeId_MR_NodeId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::data`.
        public unsafe MR.NoDefInit_MRNodeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_NodeId_MR_NodeId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRNodeId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::beginId`.
        public unsafe MR.NodeId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_beginId", ExactSpelling = true)]
            extern static MR.NodeId __MR_Buffer_MR_NodeId_MR_NodeId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::backId`.
        public unsafe MR.NodeId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_backId", ExactSpelling = true)]
            extern static MR.NodeId __MR_Buffer_MR_NodeId_MR_NodeId_backId(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::endId`.
        public unsafe MR.NodeId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_endId", ExactSpelling = true)]
            extern static MR.NodeId __MR_Buffer_MR_NodeId_MR_NodeId_endId(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_NodeId_MR_NodeId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_NodeId_MR_NodeId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::NodeId, MR::NodeId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRNodeId_MRNodeId : Const_Buffer_MRNodeId_MRNodeId
    {
        internal unsafe Buffer_MRNodeId_MRNodeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRNodeId_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_NodeId_MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::NodeId, MR::NodeId>::Buffer`.
        public unsafe Buffer_MRNodeId_MRNodeId(MR._ByValue_Buffer_MRNodeId_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRNodeId_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_NodeId_MR_NodeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::NodeId, MR::NodeId>::Buffer`.
        public unsafe Buffer_MRNodeId_MRNodeId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_NodeId_MR_NodeId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::operator=`.
        public unsafe MR.Buffer_MRNodeId_MRNodeId Assign(MR._ByValue_Buffer_MRNodeId_MRNodeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRNodeId_MRNodeId._Underlying *_other);
            return new(__MR_Buffer_MR_NodeId_MR_NodeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_NodeId_MR_NodeId_clear(_Underlying *_this);
            __MR_Buffer_MR_NodeId_MR_NodeId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_NodeId_MR_NodeId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_NodeId_MR_NodeId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::operator[]`.
        public unsafe new MR.NoDefInit_MRNodeId Index(MR.NodeId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_index(_Underlying *_this, MR.NodeId i);
            return new(__MR_Buffer_MR_NodeId_MR_NodeId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::NodeId, MR::NodeId>::data`.
        public unsafe new MR.NoDefInit_MRNodeId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_NodeId_MR_NodeId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRNodeId._Underlying *__MR_Buffer_MR_NodeId_MR_NodeId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_NodeId_MR_NodeId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRNodeId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRNodeId_MRNodeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRNodeId_MRNodeId`/`Const_Buffer_MRNodeId_MRNodeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRNodeId_MRNodeId
    {
        internal readonly Const_Buffer_MRNodeId_MRNodeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRNodeId_MRNodeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRNodeId_MRNodeId(MR.Misc._Moved<Buffer_MRNodeId_MRNodeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRNodeId_MRNodeId(MR.Misc._Moved<Buffer_MRNodeId_MRNodeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRNodeId_MRNodeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRNodeId_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRNodeId_MRNodeId`/`Const_Buffer_MRNodeId_MRNodeId` directly.
    public class _InOptMut_Buffer_MRNodeId_MRNodeId
    {
        public Buffer_MRNodeId_MRNodeId? Opt;

        public _InOptMut_Buffer_MRNodeId_MRNodeId() {}
        public _InOptMut_Buffer_MRNodeId_MRNodeId(Buffer_MRNodeId_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRNodeId_MRNodeId(Buffer_MRNodeId_MRNodeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRNodeId_MRNodeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRNodeId_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRNodeId_MRNodeId`/`Const_Buffer_MRNodeId_MRNodeId` to pass it to the function.
    public class _InOptConst_Buffer_MRNodeId_MRNodeId
    {
        public Const_Buffer_MRNodeId_MRNodeId? Opt;

        public _InOptConst_Buffer_MRNodeId_MRNodeId() {}
        public _InOptConst_Buffer_MRNodeId_MRNodeId(Const_Buffer_MRNodeId_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRNodeId_MRNodeId(Const_Buffer_MRNodeId_MRNodeId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::TextureId, MR::TextureId>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRTextureId_MRTextureId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRTextureId_MRTextureId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_TextureId_MR_TextureId_Destroy(_Underlying *_this);
            __MR_Buffer_MR_TextureId_MR_TextureId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRTextureId_MRTextureId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRTextureId_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_TextureId_MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::TextureId, MR::TextureId>::Buffer`.
        public unsafe Const_Buffer_MRTextureId_MRTextureId(MR._ByValue_Buffer_MRTextureId_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRTextureId_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_TextureId_MR_TextureId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::TextureId, MR::TextureId>::Buffer`.
        public unsafe Const_Buffer_MRTextureId_MRTextureId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_TextureId_MR_TextureId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_TextureId_MR_TextureId_capacity(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_TextureId_MR_TextureId_size(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_TextureId_MR_TextureId_empty(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRTextureId Index(MR.TextureId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_index_const(_Underlying *_this, MR.TextureId i);
            return new(__MR_Buffer_MR_TextureId_MR_TextureId_index_const(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::data`.
        public unsafe MR.NoDefInit_MRTextureId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_TextureId_MR_TextureId_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRTextureId(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::beginId`.
        public unsafe MR.TextureId BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_beginId", ExactSpelling = true)]
            extern static MR.TextureId __MR_Buffer_MR_TextureId_MR_TextureId_beginId(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_beginId(_UnderlyingPtr);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::backId`.
        public unsafe MR.TextureId BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_backId", ExactSpelling = true)]
            extern static MR.TextureId __MR_Buffer_MR_TextureId_MR_TextureId_backId(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_backId(_UnderlyingPtr);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::endId`.
        public unsafe MR.TextureId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_endId", ExactSpelling = true)]
            extern static MR.TextureId __MR_Buffer_MR_TextureId_MR_TextureId_endId(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_endId(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_TextureId_MR_TextureId_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_TextureId_MR_TextureId_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::TextureId, MR::TextureId>`.
    /// This is the non-const half of the class.
    public class Buffer_MRTextureId_MRTextureId : Const_Buffer_MRTextureId_MRTextureId
    {
        internal unsafe Buffer_MRTextureId_MRTextureId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRTextureId_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_TextureId_MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::TextureId, MR::TextureId>::Buffer`.
        public unsafe Buffer_MRTextureId_MRTextureId(MR._ByValue_Buffer_MRTextureId_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRTextureId_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_TextureId_MR_TextureId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::TextureId, MR::TextureId>::Buffer`.
        public unsafe Buffer_MRTextureId_MRTextureId(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_TextureId_MR_TextureId_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::operator=`.
        public unsafe MR.Buffer_MRTextureId_MRTextureId Assign(MR._ByValue_Buffer_MRTextureId_MRTextureId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRTextureId_MRTextureId._Underlying *_other);
            return new(__MR_Buffer_MR_TextureId_MR_TextureId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_TextureId_MR_TextureId_clear(_Underlying *_this);
            __MR_Buffer_MR_TextureId_MR_TextureId_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_TextureId_MR_TextureId_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_TextureId_MR_TextureId_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::operator[]`.
        public unsafe new MR.NoDefInit_MRTextureId Index(MR.TextureId i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_index(_Underlying *_this, MR.TextureId i);
            return new(__MR_Buffer_MR_TextureId_MR_TextureId_index(_UnderlyingPtr, i), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::TextureId, MR::TextureId>::data`.
        public unsafe new MR.NoDefInit_MRTextureId? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_TextureId_MR_TextureId_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRTextureId._Underlying *__MR_Buffer_MR_TextureId_MR_TextureId_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_TextureId_MR_TextureId_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRTextureId(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRTextureId_MRTextureId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRTextureId_MRTextureId`/`Const_Buffer_MRTextureId_MRTextureId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRTextureId_MRTextureId
    {
        internal readonly Const_Buffer_MRTextureId_MRTextureId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRTextureId_MRTextureId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRTextureId_MRTextureId(MR.Misc._Moved<Buffer_MRTextureId_MRTextureId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRTextureId_MRTextureId(MR.Misc._Moved<Buffer_MRTextureId_MRTextureId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRTextureId_MRTextureId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRTextureId_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRTextureId_MRTextureId`/`Const_Buffer_MRTextureId_MRTextureId` directly.
    public class _InOptMut_Buffer_MRTextureId_MRTextureId
    {
        public Buffer_MRTextureId_MRTextureId? Opt;

        public _InOptMut_Buffer_MRTextureId_MRTextureId() {}
        public _InOptMut_Buffer_MRTextureId_MRTextureId(Buffer_MRTextureId_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRTextureId_MRTextureId(Buffer_MRTextureId_MRTextureId value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRTextureId_MRTextureId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRTextureId_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRTextureId_MRTextureId`/`Const_Buffer_MRTextureId_MRTextureId` to pass it to the function.
    public class _InOptConst_Buffer_MRTextureId_MRTextureId
    {
        public Const_Buffer_MRTextureId_MRTextureId? Opt;

        public _InOptConst_Buffer_MRTextureId_MRTextureId() {}
        public _InOptConst_Buffer_MRTextureId_MRTextureId(Const_Buffer_MRTextureId_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRTextureId_MRTextureId(Const_Buffer_MRTextureId_MRTextureId value) {return new(value);}
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>`.
    /// This is the const half of the class.
    public class Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Destroy", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Destroy(_Underlying *_this);
            __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::Buffer`.
        public unsafe Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR._ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::Buffer`.
        public unsafe Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::capacity`.
        public unsafe ulong Capacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_capacity", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_capacity(_Underlying *_this);
            return __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_capacity(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_size", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_size(_Underlying *_this);
            return __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_empty", ExactSpelling = true)]
            extern static byte __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_empty(_Underlying *_this);
            return __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::operator[]`.
        public unsafe MR.Const_NoDefInit_MRIdMRICPElemtTag Index(MR.Const_Id_MRICPElemtTag i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_index_const", ExactSpelling = true)]
            extern static MR.Const_NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_index_const(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *i);
            return new(__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_index_const(_UnderlyingPtr, i._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::data`.
        public unsafe MR.NoDefInit_MRIdMRICPElemtTag? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_data_const", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_data_const(_Underlying *_this);
            var __ret = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_data_const(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRIdMRICPElemtTag(__ret, is_owning: false) : null;
        }

        /// returns the identifier of the first element
        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::beginId`.
        public unsafe MR.Id_MRICPElemtTag BeginId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_beginId", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_beginId(_Underlying *_this);
            return new(__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_beginId(_UnderlyingPtr), is_owning: true);
        }

        /// returns the identifier of the back() element
        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::backId`.
        public unsafe MR.Id_MRICPElemtTag BackId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_backId", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_backId(_Underlying *_this);
            return new(__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_backId(_UnderlyingPtr), is_owning: true);
        }

        /// returns backId() + 1
        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::endId`.
        public unsafe MR.Id_MRICPElemtTag EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_endId", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_endId(_Underlying *_this);
            return new(__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_endId(_UnderlyingPtr), is_owning: true);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_heapBytes(_Underlying *_this);
            return __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_heapBytes(_UnderlyingPtr);
        }
    }

    /**
    * \brief std::vector<V>-like container that is
    *  1) resized without initialization of its elements,
    *  2) much simplified: no push_back and many other methods
    * \tparam V type of stored elements
    * \tparam I type of index (shall be convertible to size_t)
    *
    */
    /// Generated from class `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>`.
    /// This is the non-const half of the class.
    public class Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag : Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        internal unsafe Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::Buffer`.
        public unsafe Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR._ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::Buffer`.
        public unsafe Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Construct", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Construct(ulong size);
            _UnderlyingPtr = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Construct(size);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::operator=`.
        public unsafe MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag Assign(MR._ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *_other);
            return new(__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_clear", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_clear(_Underlying *_this);
            __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::resize`.
        public unsafe void Resize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_resize", ExactSpelling = true)]
            extern static void __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_resize(_Underlying *_this, ulong newSize);
            __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_resize(_UnderlyingPtr, newSize);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::operator[]`.
        public unsafe new MR.NoDefInit_MRIdMRICPElemtTag Index(MR.Const_Id_MRICPElemtTag i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_index", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_index(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *i);
            return new(__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_index(_UnderlyingPtr, i._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Buffer<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::data`.
        public unsafe new MR.NoDefInit_MRIdMRICPElemtTag? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_data", ExactSpelling = true)]
            extern static MR.NoDefInit_MRIdMRICPElemtTag._Underlying *__MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_data(_Underlying *_this);
            var __ret = __MR_Buffer_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_data(_UnderlyingPtr);
            return __ret is not null ? new MR.NoDefInit_MRIdMRICPElemtTag(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag`/`Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        internal readonly Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR.Misc._Moved<Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR.Misc._Moved<Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag`/`Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag` directly.
    public class _InOptMut_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        public Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag? Opt;

        public _InOptMut_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {}
        public _InOptMut_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptMut_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// This is used for optional parameters of class `Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag`/`Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag` to pass it to the function.
    public class _InOptConst_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        public Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag? Opt;

        public _InOptConst_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {}
        public _InOptConst_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptConst_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag(Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::FaceBMap`.
    /// This is the const half of the class.
    public class Const_FaceBMap : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FaceBMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceBMap_Destroy(_Underlying *_this);
            __MR_FaceBMap_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceBMap() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRFaceId_MRFaceId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRFaceId_MRFaceId._Underlying *__MR_FaceBMap_Get_b(_Underlying *_this);
                return new(__MR_FaceBMap_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_FaceBMap_Get_tsize(_Underlying *_this);
                return *__MR_FaceBMap_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceBMap_DefaultConstruct();
        }

        /// Constructs `MR::FaceBMap` elementwise.
        public unsafe Const_FaceBMap(MR._ByValue_Buffer_MRFaceId_MRFaceId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRFaceId_MRFaceId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_FaceBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::FaceBMap::FaceBMap`.
        public unsafe Const_FaceBMap(MR._ByValue_FaceBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceBMap._Underlying *_other);
            _UnderlyingPtr = __MR_FaceBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::FaceBMap`.
    /// This is the non-const half of the class.
    public class FaceBMap : Const_FaceBMap
    {
        internal unsafe FaceBMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRFaceId_MRFaceId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRFaceId_MRFaceId._Underlying *__MR_FaceBMap_GetMutable_b(_Underlying *_this);
                return new(__MR_FaceBMap_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_FaceBMap_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_FaceBMap_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceBMap_DefaultConstruct();
        }

        /// Constructs `MR::FaceBMap` elementwise.
        public unsafe FaceBMap(MR._ByValue_Buffer_MRFaceId_MRFaceId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRFaceId_MRFaceId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_FaceBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::FaceBMap::FaceBMap`.
        public unsafe FaceBMap(MR._ByValue_FaceBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceBMap._Underlying *_other);
            _UnderlyingPtr = __MR_FaceBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FaceBMap::operator=`.
        public unsafe MR.FaceBMap Assign(MR._ByValue_FaceBMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceBMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FaceBMap._Underlying *__MR_FaceBMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FaceBMap._Underlying *_other);
            return new(__MR_FaceBMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FaceBMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FaceBMap`/`Const_FaceBMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FaceBMap
    {
        internal readonly Const_FaceBMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FaceBMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FaceBMap(MR.Misc._Moved<FaceBMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FaceBMap(MR.Misc._Moved<FaceBMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FaceBMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceBMap`/`Const_FaceBMap` directly.
    public class _InOptMut_FaceBMap
    {
        public FaceBMap? Opt;

        public _InOptMut_FaceBMap() {}
        public _InOptMut_FaceBMap(FaceBMap value) {Opt = value;}
        public static implicit operator _InOptMut_FaceBMap(FaceBMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `FaceBMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceBMap`/`Const_FaceBMap` to pass it to the function.
    public class _InOptConst_FaceBMap
    {
        public Const_FaceBMap? Opt;

        public _InOptConst_FaceBMap() {}
        public _InOptConst_FaceBMap(Const_FaceBMap value) {Opt = value;}
        public static implicit operator _InOptConst_FaceBMap(Const_FaceBMap value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::VertBMap`.
    /// This is the const half of the class.
    public class Const_VertBMap : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertBMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_Destroy", ExactSpelling = true)]
            extern static void __MR_VertBMap_Destroy(_Underlying *_this);
            __MR_VertBMap_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertBMap() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRVertId_MRVertId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRVertId_MRVertId._Underlying *__MR_VertBMap_Get_b(_Underlying *_this);
                return new(__MR_VertBMap_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_VertBMap_Get_tsize(_Underlying *_this);
                return *__MR_VertBMap_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_VertBMap_DefaultConstruct();
        }

        /// Constructs `MR::VertBMap` elementwise.
        public unsafe Const_VertBMap(MR._ByValue_Buffer_MRVertId_MRVertId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRVertId_MRVertId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_VertBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::VertBMap::VertBMap`.
        public unsafe Const_VertBMap(MR._ByValue_VertBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertBMap._Underlying *_other);
            _UnderlyingPtr = __MR_VertBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::VertBMap`.
    /// This is the non-const half of the class.
    public class VertBMap : Const_VertBMap
    {
        internal unsafe VertBMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRVertId_MRVertId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRVertId_MRVertId._Underlying *__MR_VertBMap_GetMutable_b(_Underlying *_this);
                return new(__MR_VertBMap_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_VertBMap_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_VertBMap_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_VertBMap_DefaultConstruct();
        }

        /// Constructs `MR::VertBMap` elementwise.
        public unsafe VertBMap(MR._ByValue_Buffer_MRVertId_MRVertId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRVertId_MRVertId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_VertBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::VertBMap::VertBMap`.
        public unsafe VertBMap(MR._ByValue_VertBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertBMap._Underlying *_other);
            _UnderlyingPtr = __MR_VertBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VertBMap::operator=`.
        public unsafe MR.VertBMap Assign(MR._ByValue_VertBMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertBMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_VertBMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VertBMap._Underlying *_other);
            return new(__MR_VertBMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VertBMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VertBMap`/`Const_VertBMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VertBMap
    {
        internal readonly Const_VertBMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VertBMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VertBMap(MR.Misc._Moved<VertBMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VertBMap(MR.Misc._Moved<VertBMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VertBMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertBMap`/`Const_VertBMap` directly.
    public class _InOptMut_VertBMap
    {
        public VertBMap? Opt;

        public _InOptMut_VertBMap() {}
        public _InOptMut_VertBMap(VertBMap value) {Opt = value;}
        public static implicit operator _InOptMut_VertBMap(VertBMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertBMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertBMap`/`Const_VertBMap` to pass it to the function.
    public class _InOptConst_VertBMap
    {
        public Const_VertBMap? Opt;

        public _InOptConst_VertBMap() {}
        public _InOptConst_VertBMap(Const_VertBMap value) {Opt = value;}
        public static implicit operator _InOptConst_VertBMap(Const_VertBMap value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::EdgeBMap`.
    /// This is the const half of the class.
    public class Const_EdgeBMap : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgeBMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgeBMap_Destroy(_Underlying *_this);
            __MR_EdgeBMap_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgeBMap() {Dispose(false);}

        public unsafe MR.Const_Buffer_MREdgeId_MREdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MREdgeId_MREdgeId._Underlying *__MR_EdgeBMap_Get_b(_Underlying *_this);
                return new(__MR_EdgeBMap_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_EdgeBMap_Get_tsize(_Underlying *_this);
                return *__MR_EdgeBMap_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgeBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeBMap_DefaultConstruct();
        }

        /// Constructs `MR::EdgeBMap` elementwise.
        public unsafe Const_EdgeBMap(MR._ByValue_Buffer_MREdgeId_MREdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MREdgeId_MREdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_EdgeBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::EdgeBMap::EdgeBMap`.
        public unsafe Const_EdgeBMap(MR._ByValue_EdgeBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgeBMap._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::EdgeBMap`.
    /// This is the non-const half of the class.
    public class EdgeBMap : Const_EdgeBMap
    {
        internal unsafe EdgeBMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MREdgeId_MREdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MREdgeId_MREdgeId._Underlying *__MR_EdgeBMap_GetMutable_b(_Underlying *_this);
                return new(__MR_EdgeBMap_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_EdgeBMap_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_EdgeBMap_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgeBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeBMap_DefaultConstruct();
        }

        /// Constructs `MR::EdgeBMap` elementwise.
        public unsafe EdgeBMap(MR._ByValue_Buffer_MREdgeId_MREdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MREdgeId_MREdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_EdgeBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::EdgeBMap::EdgeBMap`.
        public unsafe EdgeBMap(MR._ByValue_EdgeBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgeBMap._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::EdgeBMap::operator=`.
        public unsafe MR.EdgeBMap Assign(MR._ByValue_EdgeBMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeBMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgeBMap._Underlying *__MR_EdgeBMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.EdgeBMap._Underlying *_other);
            return new(__MR_EdgeBMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EdgeBMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EdgeBMap`/`Const_EdgeBMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EdgeBMap
    {
        internal readonly Const_EdgeBMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EdgeBMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_EdgeBMap(MR.Misc._Moved<EdgeBMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EdgeBMap(MR.Misc._Moved<EdgeBMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EdgeBMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgeBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeBMap`/`Const_EdgeBMap` directly.
    public class _InOptMut_EdgeBMap
    {
        public EdgeBMap? Opt;

        public _InOptMut_EdgeBMap() {}
        public _InOptMut_EdgeBMap(EdgeBMap value) {Opt = value;}
        public static implicit operator _InOptMut_EdgeBMap(EdgeBMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgeBMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgeBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeBMap`/`Const_EdgeBMap` to pass it to the function.
    public class _InOptConst_EdgeBMap
    {
        public Const_EdgeBMap? Opt;

        public _InOptConst_EdgeBMap() {}
        public _InOptConst_EdgeBMap(Const_EdgeBMap value) {Opt = value;}
        public static implicit operator _InOptConst_EdgeBMap(Const_EdgeBMap value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::UndirectedEdgeBMap`.
    /// This is the const half of the class.
    public class Const_UndirectedEdgeBMap : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UndirectedEdgeBMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_Destroy", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeBMap_Destroy(_Underlying *_this);
            __MR_UndirectedEdgeBMap_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UndirectedEdgeBMap() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_UndirectedEdgeBMap_Get_b(_Underlying *_this);
                return new(__MR_UndirectedEdgeBMap_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_UndirectedEdgeBMap_Get_tsize(_Underlying *_this);
                return *__MR_UndirectedEdgeBMap_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UndirectedEdgeBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeBMap_DefaultConstruct();
        }

        /// Constructs `MR::UndirectedEdgeBMap` elementwise.
        public unsafe Const_UndirectedEdgeBMap(MR._ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_UndirectedEdgeBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::UndirectedEdgeBMap::UndirectedEdgeBMap`.
        public unsafe Const_UndirectedEdgeBMap(MR._ByValue_UndirectedEdgeBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UndirectedEdgeBMap._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::UndirectedEdgeBMap`.
    /// This is the non-const half of the class.
    public class UndirectedEdgeBMap : Const_UndirectedEdgeBMap
    {
        internal unsafe UndirectedEdgeBMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *__MR_UndirectedEdgeBMap_GetMutable_b(_Underlying *_this);
                return new(__MR_UndirectedEdgeBMap_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_UndirectedEdgeBMap_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_UndirectedEdgeBMap_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe UndirectedEdgeBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeBMap_DefaultConstruct();
        }

        /// Constructs `MR::UndirectedEdgeBMap` elementwise.
        public unsafe UndirectedEdgeBMap(MR._ByValue_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_UndirectedEdgeBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::UndirectedEdgeBMap::UndirectedEdgeBMap`.
        public unsafe UndirectedEdgeBMap(MR._ByValue_UndirectedEdgeBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UndirectedEdgeBMap._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::UndirectedEdgeBMap::operator=`.
        public unsafe MR.UndirectedEdgeBMap Assign(MR._ByValue_UndirectedEdgeBMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeBMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBMap._Underlying *__MR_UndirectedEdgeBMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UndirectedEdgeBMap._Underlying *_other);
            return new(__MR_UndirectedEdgeBMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UndirectedEdgeBMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UndirectedEdgeBMap`/`Const_UndirectedEdgeBMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UndirectedEdgeBMap
    {
        internal readonly Const_UndirectedEdgeBMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UndirectedEdgeBMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UndirectedEdgeBMap(MR.Misc._Moved<UndirectedEdgeBMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UndirectedEdgeBMap(MR.Misc._Moved<UndirectedEdgeBMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UndirectedEdgeBMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UndirectedEdgeBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeBMap`/`Const_UndirectedEdgeBMap` directly.
    public class _InOptMut_UndirectedEdgeBMap
    {
        public UndirectedEdgeBMap? Opt;

        public _InOptMut_UndirectedEdgeBMap() {}
        public _InOptMut_UndirectedEdgeBMap(UndirectedEdgeBMap value) {Opt = value;}
        public static implicit operator _InOptMut_UndirectedEdgeBMap(UndirectedEdgeBMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `UndirectedEdgeBMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UndirectedEdgeBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeBMap`/`Const_UndirectedEdgeBMap` to pass it to the function.
    public class _InOptConst_UndirectedEdgeBMap
    {
        public Const_UndirectedEdgeBMap? Opt;

        public _InOptConst_UndirectedEdgeBMap() {}
        public _InOptConst_UndirectedEdgeBMap(Const_UndirectedEdgeBMap value) {Opt = value;}
        public static implicit operator _InOptConst_UndirectedEdgeBMap(Const_UndirectedEdgeBMap value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::WholeEdgeBMap`.
    /// This is the const half of the class.
    public class Const_WholeEdgeBMap : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_WholeEdgeBMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_Destroy", ExactSpelling = true)]
            extern static void __MR_WholeEdgeBMap_Destroy(_Underlying *_this);
            __MR_WholeEdgeBMap_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_WholeEdgeBMap() {Dispose(false);}

        public unsafe MR.Const_Buffer_MREdgeId_MRUndirectedEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_WholeEdgeBMap_Get_b(_Underlying *_this);
                return new(__MR_WholeEdgeBMap_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_WholeEdgeBMap_Get_tsize(_Underlying *_this);
                return *__MR_WholeEdgeBMap_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_WholeEdgeBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_WholeEdgeBMap_DefaultConstruct();
        }

        /// Constructs `MR::WholeEdgeBMap` elementwise.
        public unsafe Const_WholeEdgeBMap(MR._ByValue_Buffer_MREdgeId_MRUndirectedEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_WholeEdgeBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::WholeEdgeBMap::WholeEdgeBMap`.
        public unsafe Const_WholeEdgeBMap(MR._ByValue_WholeEdgeBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WholeEdgeBMap._Underlying *_other);
            _UnderlyingPtr = __MR_WholeEdgeBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::WholeEdgeBMap`.
    /// This is the non-const half of the class.
    public class WholeEdgeBMap : Const_WholeEdgeBMap
    {
        internal unsafe WholeEdgeBMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MREdgeId_MRUndirectedEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *__MR_WholeEdgeBMap_GetMutable_b(_Underlying *_this);
                return new(__MR_WholeEdgeBMap_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_WholeEdgeBMap_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_WholeEdgeBMap_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe WholeEdgeBMap() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_DefaultConstruct();
            _UnderlyingPtr = __MR_WholeEdgeBMap_DefaultConstruct();
        }

        /// Constructs `MR::WholeEdgeBMap` elementwise.
        public unsafe WholeEdgeBMap(MR._ByValue_Buffer_MREdgeId_MRUndirectedEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_ConstructFrom", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MREdgeId_MRUndirectedEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_WholeEdgeBMap_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::WholeEdgeBMap::WholeEdgeBMap`.
        public unsafe WholeEdgeBMap(MR._ByValue_WholeEdgeBMap _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WholeEdgeBMap._Underlying *_other);
            _UnderlyingPtr = __MR_WholeEdgeBMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::WholeEdgeBMap::operator=`.
        public unsafe MR.WholeEdgeBMap Assign(MR._ByValue_WholeEdgeBMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WholeEdgeBMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.WholeEdgeBMap._Underlying *__MR_WholeEdgeBMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WholeEdgeBMap._Underlying *_other);
            return new(__MR_WholeEdgeBMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `WholeEdgeBMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `WholeEdgeBMap`/`Const_WholeEdgeBMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_WholeEdgeBMap
    {
        internal readonly Const_WholeEdgeBMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_WholeEdgeBMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_WholeEdgeBMap(MR.Misc._Moved<WholeEdgeBMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_WholeEdgeBMap(MR.Misc._Moved<WholeEdgeBMap> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `WholeEdgeBMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_WholeEdgeBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WholeEdgeBMap`/`Const_WholeEdgeBMap` directly.
    public class _InOptMut_WholeEdgeBMap
    {
        public WholeEdgeBMap? Opt;

        public _InOptMut_WholeEdgeBMap() {}
        public _InOptMut_WholeEdgeBMap(WholeEdgeBMap value) {Opt = value;}
        public static implicit operator _InOptMut_WholeEdgeBMap(WholeEdgeBMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `WholeEdgeBMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_WholeEdgeBMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WholeEdgeBMap`/`Const_WholeEdgeBMap` to pass it to the function.
    public class _InOptConst_WholeEdgeBMap
    {
        public Const_WholeEdgeBMap? Opt;

        public _InOptConst_WholeEdgeBMap() {}
        public _InOptConst_WholeEdgeBMap(Const_WholeEdgeBMap value) {Opt = value;}
        public static implicit operator _InOptConst_WholeEdgeBMap(Const_WholeEdgeBMap value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::ObjId, MR::ObjId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRObjId_MRObjId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRObjId_MRObjId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_ObjId_MR_ObjId_Destroy(_Underlying *_this);
            __MR_BMap_MR_ObjId_MR_ObjId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRObjId_MRObjId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRObjId_MRObjId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_ObjId_MR_ObjId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_ObjId_MR_ObjId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_ObjId_MR_ObjId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRObjId_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_ObjId_MR_ObjId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::ObjId, MR::ObjId>` elementwise.
        public unsafe Const_BMap_MRObjId_MRObjId(MR._ByValue_Buffer_MRObjId_MRObjId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRObjId_MRObjId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_ObjId_MR_ObjId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::ObjId, MR::ObjId>::BMap`.
        public unsafe Const_BMap_MRObjId_MRObjId(MR._ByValue_BMap_MRObjId_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRObjId_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_ObjId_MR_ObjId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::ObjId, MR::ObjId>`.
    /// This is the non-const half of the class.
    public class BMap_MRObjId_MRObjId : Const_BMap_MRObjId_MRObjId
    {
        internal unsafe BMap_MRObjId_MRObjId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRObjId_MRObjId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_ObjId_MR_ObjId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_ObjId_MR_ObjId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_ObjId_MR_ObjId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRObjId_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_ObjId_MR_ObjId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::ObjId, MR::ObjId>` elementwise.
        public unsafe BMap_MRObjId_MRObjId(MR._ByValue_Buffer_MRObjId_MRObjId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRObjId_MRObjId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_ObjId_MR_ObjId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::ObjId, MR::ObjId>::BMap`.
        public unsafe BMap_MRObjId_MRObjId(MR._ByValue_BMap_MRObjId_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRObjId_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_ObjId_MR_ObjId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::ObjId, MR::ObjId>::operator=`.
        public unsafe MR.BMap_MRObjId_MRObjId Assign(MR._ByValue_BMap_MRObjId_MRObjId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_ObjId_MR_ObjId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRObjId_MRObjId._Underlying *__MR_BMap_MR_ObjId_MR_ObjId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRObjId_MRObjId._Underlying *_other);
            return new(__MR_BMap_MR_ObjId_MR_ObjId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRObjId_MRObjId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRObjId_MRObjId`/`Const_BMap_MRObjId_MRObjId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRObjId_MRObjId
    {
        internal readonly Const_BMap_MRObjId_MRObjId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRObjId_MRObjId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRObjId_MRObjId(MR.Misc._Moved<BMap_MRObjId_MRObjId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRObjId_MRObjId(MR.Misc._Moved<BMap_MRObjId_MRObjId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRObjId_MRObjId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRObjId_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRObjId_MRObjId`/`Const_BMap_MRObjId_MRObjId` directly.
    public class _InOptMut_BMap_MRObjId_MRObjId
    {
        public BMap_MRObjId_MRObjId? Opt;

        public _InOptMut_BMap_MRObjId_MRObjId() {}
        public _InOptMut_BMap_MRObjId_MRObjId(BMap_MRObjId_MRObjId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRObjId_MRObjId(BMap_MRObjId_MRObjId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRObjId_MRObjId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRObjId_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRObjId_MRObjId`/`Const_BMap_MRObjId_MRObjId` to pass it to the function.
    public class _InOptConst_BMap_MRObjId_MRObjId
    {
        public Const_BMap_MRObjId_MRObjId? Opt;

        public _InOptConst_BMap_MRObjId_MRObjId() {}
        public _InOptConst_BMap_MRObjId_MRObjId(Const_BMap_MRObjId_MRObjId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRObjId_MRObjId(Const_BMap_MRObjId_MRObjId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::GraphVertId, MR::GraphVertId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRGraphVertId_MRGraphVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRGraphVertId_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_GraphVertId_MR_GraphVertId_Destroy(_Underlying *_this);
            __MR_BMap_MR_GraphVertId_MR_GraphVertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRGraphVertId_MRGraphVertId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRGraphVertId_MRGraphVertId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_GraphVertId_MR_GraphVertId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_GraphVertId_MR_GraphVertId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_GraphVertId_MR_GraphVertId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRGraphVertId_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::GraphVertId, MR::GraphVertId>` elementwise.
        public unsafe Const_BMap_MRGraphVertId_MRGraphVertId(MR._ByValue_Buffer_MRGraphVertId_MRGraphVertId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::GraphVertId, MR::GraphVertId>::BMap`.
        public unsafe Const_BMap_MRGraphVertId_MRGraphVertId(MR._ByValue_BMap_MRGraphVertId_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::GraphVertId, MR::GraphVertId>`.
    /// This is the non-const half of the class.
    public class BMap_MRGraphVertId_MRGraphVertId : Const_BMap_MRGraphVertId_MRGraphVertId
    {
        internal unsafe BMap_MRGraphVertId_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRGraphVertId_MRGraphVertId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_GraphVertId_MR_GraphVertId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_GraphVertId_MR_GraphVertId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_GraphVertId_MR_GraphVertId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRGraphVertId_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_GraphVertId_MR_GraphVertId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::GraphVertId, MR::GraphVertId>` elementwise.
        public unsafe BMap_MRGraphVertId_MRGraphVertId(MR._ByValue_Buffer_MRGraphVertId_MRGraphVertId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRGraphVertId_MRGraphVertId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::GraphVertId, MR::GraphVertId>::BMap`.
        public unsafe BMap_MRGraphVertId_MRGraphVertId(MR._ByValue_BMap_MRGraphVertId_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_GraphVertId_MR_GraphVertId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::GraphVertId, MR::GraphVertId>::operator=`.
        public unsafe MR.BMap_MRGraphVertId_MRGraphVertId Assign(MR._ByValue_BMap_MRGraphVertId_MRGraphVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphVertId_MR_GraphVertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *__MR_BMap_MR_GraphVertId_MR_GraphVertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRGraphVertId_MRGraphVertId._Underlying *_other);
            return new(__MR_BMap_MR_GraphVertId_MR_GraphVertId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRGraphVertId_MRGraphVertId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRGraphVertId_MRGraphVertId`/`Const_BMap_MRGraphVertId_MRGraphVertId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRGraphVertId_MRGraphVertId
    {
        internal readonly Const_BMap_MRGraphVertId_MRGraphVertId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRGraphVertId_MRGraphVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRGraphVertId_MRGraphVertId(MR.Misc._Moved<BMap_MRGraphVertId_MRGraphVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRGraphVertId_MRGraphVertId(MR.Misc._Moved<BMap_MRGraphVertId_MRGraphVertId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRGraphVertId_MRGraphVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRGraphVertId_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRGraphVertId_MRGraphVertId`/`Const_BMap_MRGraphVertId_MRGraphVertId` directly.
    public class _InOptMut_BMap_MRGraphVertId_MRGraphVertId
    {
        public BMap_MRGraphVertId_MRGraphVertId? Opt;

        public _InOptMut_BMap_MRGraphVertId_MRGraphVertId() {}
        public _InOptMut_BMap_MRGraphVertId_MRGraphVertId(BMap_MRGraphVertId_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRGraphVertId_MRGraphVertId(BMap_MRGraphVertId_MRGraphVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRGraphVertId_MRGraphVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRGraphVertId_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRGraphVertId_MRGraphVertId`/`Const_BMap_MRGraphVertId_MRGraphVertId` to pass it to the function.
    public class _InOptConst_BMap_MRGraphVertId_MRGraphVertId
    {
        public Const_BMap_MRGraphVertId_MRGraphVertId? Opt;

        public _InOptConst_BMap_MRGraphVertId_MRGraphVertId() {}
        public _InOptConst_BMap_MRGraphVertId_MRGraphVertId(Const_BMap_MRGraphVertId_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRGraphVertId_MRGraphVertId(Const_BMap_MRGraphVertId_MRGraphVertId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRGraphEdgeId_MRGraphEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRGraphEdgeId_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Destroy(_Underlying *_this);
            __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRGraphEdgeId_MRGraphEdgeId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRGraphEdgeId_MRGraphEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRGraphEdgeId_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>` elementwise.
        public unsafe Const_BMap_MRGraphEdgeId_MRGraphEdgeId(MR._ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>::BMap`.
        public unsafe Const_BMap_MRGraphEdgeId_MRGraphEdgeId(MR._ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>`.
    /// This is the non-const half of the class.
    public class BMap_MRGraphEdgeId_MRGraphEdgeId : Const_BMap_MRGraphEdgeId_MRGraphEdgeId
    {
        internal unsafe BMap_MRGraphEdgeId_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRGraphEdgeId_MRGraphEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRGraphEdgeId_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>` elementwise.
        public unsafe BMap_MRGraphEdgeId_MRGraphEdgeId(MR._ByValue_Buffer_MRGraphEdgeId_MRGraphEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRGraphEdgeId_MRGraphEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>::BMap`.
        public unsafe BMap_MRGraphEdgeId_MRGraphEdgeId(MR._ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::GraphEdgeId, MR::GraphEdgeId>::operator=`.
        public unsafe MR.BMap_MRGraphEdgeId_MRGraphEdgeId Assign(MR._ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRGraphEdgeId_MRGraphEdgeId._Underlying *_other);
            return new(__MR_BMap_MR_GraphEdgeId_MR_GraphEdgeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRGraphEdgeId_MRGraphEdgeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRGraphEdgeId_MRGraphEdgeId`/`Const_BMap_MRGraphEdgeId_MRGraphEdgeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId
    {
        internal readonly Const_BMap_MRGraphEdgeId_MRGraphEdgeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId(MR.Misc._Moved<BMap_MRGraphEdgeId_MRGraphEdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRGraphEdgeId_MRGraphEdgeId(MR.Misc._Moved<BMap_MRGraphEdgeId_MRGraphEdgeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRGraphEdgeId_MRGraphEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRGraphEdgeId_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRGraphEdgeId_MRGraphEdgeId`/`Const_BMap_MRGraphEdgeId_MRGraphEdgeId` directly.
    public class _InOptMut_BMap_MRGraphEdgeId_MRGraphEdgeId
    {
        public BMap_MRGraphEdgeId_MRGraphEdgeId? Opt;

        public _InOptMut_BMap_MRGraphEdgeId_MRGraphEdgeId() {}
        public _InOptMut_BMap_MRGraphEdgeId_MRGraphEdgeId(BMap_MRGraphEdgeId_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRGraphEdgeId_MRGraphEdgeId(BMap_MRGraphEdgeId_MRGraphEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRGraphEdgeId_MRGraphEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRGraphEdgeId_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRGraphEdgeId_MRGraphEdgeId`/`Const_BMap_MRGraphEdgeId_MRGraphEdgeId` to pass it to the function.
    public class _InOptConst_BMap_MRGraphEdgeId_MRGraphEdgeId
    {
        public Const_BMap_MRGraphEdgeId_MRGraphEdgeId? Opt;

        public _InOptConst_BMap_MRGraphEdgeId_MRGraphEdgeId() {}
        public _InOptConst_BMap_MRGraphEdgeId_MRGraphEdgeId(Const_BMap_MRGraphEdgeId_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRGraphEdgeId_MRGraphEdgeId(Const_BMap_MRGraphEdgeId_MRGraphEdgeId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::VertId, MR_uint64_t>`.
    /// This is the const half of the class.
    public class Const_BMap_MRVertId_MRUint64T : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRVertId_MRUint64T(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_VertId_uint64_t_Destroy(_Underlying *_this);
            __MR_BMap_MR_VertId_uint64_t_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRVertId_MRUint64T() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRVertId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRVertId._Underlying *__MR_BMap_MR_VertId_uint64_t_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_VertId_uint64_t_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_VertId_uint64_t_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_VertId_uint64_t_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRVertId_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_VertId_uint64_t_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::VertId, MR_uint64_t>` elementwise.
        public unsafe Const_BMap_MRVertId_MRUint64T(MR._ByValue_Buffer_MRVertId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRVertId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_VertId_uint64_t_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::VertId, MR_uint64_t>::BMap`.
        public unsafe Const_BMap_MRVertId_MRUint64T(MR._ByValue_BMap_MRVertId_MRUint64T _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRVertId_MRUint64T._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_VertId_uint64_t_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::VertId, MR_uint64_t>`.
    /// This is the non-const half of the class.
    public class BMap_MRVertId_MRUint64T : Const_BMap_MRVertId_MRUint64T
    {
        internal unsafe BMap_MRVertId_MRUint64T(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRVertId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRVertId._Underlying *__MR_BMap_MR_VertId_uint64_t_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_VertId_uint64_t_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_VertId_uint64_t_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_VertId_uint64_t_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRVertId_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_VertId_uint64_t_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::VertId, MR_uint64_t>` elementwise.
        public unsafe BMap_MRVertId_MRUint64T(MR._ByValue_Buffer_MRVertId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRVertId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_VertId_uint64_t_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::VertId, MR_uint64_t>::BMap`.
        public unsafe BMap_MRVertId_MRUint64T(MR._ByValue_BMap_MRVertId_MRUint64T _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRVertId_MRUint64T._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_VertId_uint64_t_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::VertId, MR_uint64_t>::operator=`.
        public unsafe MR.BMap_MRVertId_MRUint64T Assign(MR._ByValue_BMap_MRVertId_MRUint64T _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VertId_uint64_t_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRVertId_MRUint64T._Underlying *__MR_BMap_MR_VertId_uint64_t_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRVertId_MRUint64T._Underlying *_other);
            return new(__MR_BMap_MR_VertId_uint64_t_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRVertId_MRUint64T` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRVertId_MRUint64T`/`Const_BMap_MRVertId_MRUint64T` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRVertId_MRUint64T
    {
        internal readonly Const_BMap_MRVertId_MRUint64T? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRVertId_MRUint64T() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRVertId_MRUint64T(MR.Misc._Moved<BMap_MRVertId_MRUint64T> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRVertId_MRUint64T(MR.Misc._Moved<BMap_MRVertId_MRUint64T> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRVertId_MRUint64T` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRVertId_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRVertId_MRUint64T`/`Const_BMap_MRVertId_MRUint64T` directly.
    public class _InOptMut_BMap_MRVertId_MRUint64T
    {
        public BMap_MRVertId_MRUint64T? Opt;

        public _InOptMut_BMap_MRVertId_MRUint64T() {}
        public _InOptMut_BMap_MRVertId_MRUint64T(BMap_MRVertId_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRVertId_MRUint64T(BMap_MRVertId_MRUint64T value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRVertId_MRUint64T` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRVertId_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRVertId_MRUint64T`/`Const_BMap_MRVertId_MRUint64T` to pass it to the function.
    public class _InOptConst_BMap_MRVertId_MRUint64T
    {
        public Const_BMap_MRVertId_MRUint64T? Opt;

        public _InOptConst_BMap_MRVertId_MRUint64T() {}
        public _InOptConst_BMap_MRVertId_MRUint64T(Const_BMap_MRVertId_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRVertId_MRUint64T(Const_BMap_MRVertId_MRUint64T value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>`.
    /// This is the const half of the class.
    public class Const_BMap_MRUndirectedEdgeId_MRUint64T : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRUndirectedEdgeId_MRUint64T(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_UndirectedEdgeId_uint64_t_Destroy(_Underlying *_this);
            __MR_BMap_MR_UndirectedEdgeId_uint64_t_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRUndirectedEdgeId_MRUint64T() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRUndirectedEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRUndirectedEdgeId._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_UndirectedEdgeId_uint64_t_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_UndirectedEdgeId_uint64_t_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_UndirectedEdgeId_uint64_t_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRUndirectedEdgeId_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_UndirectedEdgeId_uint64_t_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>` elementwise.
        public unsafe Const_BMap_MRUndirectedEdgeId_MRUint64T(MR._ByValue_Buffer_MRUndirectedEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRUndirectedEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>::BMap`.
        public unsafe Const_BMap_MRUndirectedEdgeId_MRUint64T(MR._ByValue_BMap_MRUndirectedEdgeId_MRUint64T _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>`.
    /// This is the non-const half of the class.
    public class BMap_MRUndirectedEdgeId_MRUint64T : Const_BMap_MRUndirectedEdgeId_MRUint64T
    {
        internal unsafe BMap_MRUndirectedEdgeId_MRUint64T(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRUndirectedEdgeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRUndirectedEdgeId._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_UndirectedEdgeId_uint64_t_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_UndirectedEdgeId_uint64_t_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_UndirectedEdgeId_uint64_t_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRUndirectedEdgeId_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_UndirectedEdgeId_uint64_t_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>` elementwise.
        public unsafe BMap_MRUndirectedEdgeId_MRUint64T(MR._ByValue_Buffer_MRUndirectedEdgeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRUndirectedEdgeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>::BMap`.
        public unsafe BMap_MRUndirectedEdgeId_MRUint64T(MR._ByValue_BMap_MRUndirectedEdgeId_MRUint64T _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_UndirectedEdgeId_uint64_t_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::UndirectedEdgeId, MR_uint64_t>::operator=`.
        public unsafe MR.BMap_MRUndirectedEdgeId_MRUint64T Assign(MR._ByValue_BMap_MRUndirectedEdgeId_MRUint64T _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_UndirectedEdgeId_uint64_t_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *__MR_BMap_MR_UndirectedEdgeId_uint64_t_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRUndirectedEdgeId_MRUint64T._Underlying *_other);
            return new(__MR_BMap_MR_UndirectedEdgeId_uint64_t_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRUndirectedEdgeId_MRUint64T` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRUndirectedEdgeId_MRUint64T`/`Const_BMap_MRUndirectedEdgeId_MRUint64T` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRUndirectedEdgeId_MRUint64T
    {
        internal readonly Const_BMap_MRUndirectedEdgeId_MRUint64T? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRUndirectedEdgeId_MRUint64T() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRUndirectedEdgeId_MRUint64T(MR.Misc._Moved<BMap_MRUndirectedEdgeId_MRUint64T> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRUndirectedEdgeId_MRUint64T(MR.Misc._Moved<BMap_MRUndirectedEdgeId_MRUint64T> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRUndirectedEdgeId_MRUint64T` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRUndirectedEdgeId_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRUndirectedEdgeId_MRUint64T`/`Const_BMap_MRUndirectedEdgeId_MRUint64T` directly.
    public class _InOptMut_BMap_MRUndirectedEdgeId_MRUint64T
    {
        public BMap_MRUndirectedEdgeId_MRUint64T? Opt;

        public _InOptMut_BMap_MRUndirectedEdgeId_MRUint64T() {}
        public _InOptMut_BMap_MRUndirectedEdgeId_MRUint64T(BMap_MRUndirectedEdgeId_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRUndirectedEdgeId_MRUint64T(BMap_MRUndirectedEdgeId_MRUint64T value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRUndirectedEdgeId_MRUint64T` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRUndirectedEdgeId_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRUndirectedEdgeId_MRUint64T`/`Const_BMap_MRUndirectedEdgeId_MRUint64T` to pass it to the function.
    public class _InOptConst_BMap_MRUndirectedEdgeId_MRUint64T
    {
        public Const_BMap_MRUndirectedEdgeId_MRUint64T? Opt;

        public _InOptConst_BMap_MRUndirectedEdgeId_MRUint64T() {}
        public _InOptConst_BMap_MRUndirectedEdgeId_MRUint64T(Const_BMap_MRUndirectedEdgeId_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRUndirectedEdgeId_MRUint64T(Const_BMap_MRUndirectedEdgeId_MRUint64T value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::FaceId, MR_uint64_t>`.
    /// This is the const half of the class.
    public class Const_BMap_MRFaceId_MRUint64T : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRFaceId_MRUint64T(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_FaceId_uint64_t_Destroy(_Underlying *_this);
            __MR_BMap_MR_FaceId_uint64_t_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRFaceId_MRUint64T() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRFaceId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRFaceId._Underlying *__MR_BMap_MR_FaceId_uint64_t_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_FaceId_uint64_t_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_FaceId_uint64_t_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_FaceId_uint64_t_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRFaceId_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_FaceId_uint64_t_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::FaceId, MR_uint64_t>` elementwise.
        public unsafe Const_BMap_MRFaceId_MRUint64T(MR._ByValue_Buffer_MRFaceId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRFaceId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_FaceId_uint64_t_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::FaceId, MR_uint64_t>::BMap`.
        public unsafe Const_BMap_MRFaceId_MRUint64T(MR._ByValue_BMap_MRFaceId_MRUint64T _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRFaceId_MRUint64T._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_FaceId_uint64_t_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::FaceId, MR_uint64_t>`.
    /// This is the non-const half of the class.
    public class BMap_MRFaceId_MRUint64T : Const_BMap_MRFaceId_MRUint64T
    {
        internal unsafe BMap_MRFaceId_MRUint64T(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRFaceId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRFaceId._Underlying *__MR_BMap_MR_FaceId_uint64_t_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_FaceId_uint64_t_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_FaceId_uint64_t_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_FaceId_uint64_t_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRFaceId_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_FaceId_uint64_t_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::FaceId, MR_uint64_t>` elementwise.
        public unsafe BMap_MRFaceId_MRUint64T(MR._ByValue_Buffer_MRFaceId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRFaceId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_FaceId_uint64_t_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::FaceId, MR_uint64_t>::BMap`.
        public unsafe BMap_MRFaceId_MRUint64T(MR._ByValue_BMap_MRFaceId_MRUint64T _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRFaceId_MRUint64T._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_FaceId_uint64_t_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::FaceId, MR_uint64_t>::operator=`.
        public unsafe MR.BMap_MRFaceId_MRUint64T Assign(MR._ByValue_BMap_MRFaceId_MRUint64T _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_FaceId_uint64_t_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRFaceId_MRUint64T._Underlying *__MR_BMap_MR_FaceId_uint64_t_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRFaceId_MRUint64T._Underlying *_other);
            return new(__MR_BMap_MR_FaceId_uint64_t_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRFaceId_MRUint64T` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRFaceId_MRUint64T`/`Const_BMap_MRFaceId_MRUint64T` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRFaceId_MRUint64T
    {
        internal readonly Const_BMap_MRFaceId_MRUint64T? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRFaceId_MRUint64T() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRFaceId_MRUint64T(MR.Misc._Moved<BMap_MRFaceId_MRUint64T> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRFaceId_MRUint64T(MR.Misc._Moved<BMap_MRFaceId_MRUint64T> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRFaceId_MRUint64T` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRFaceId_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRFaceId_MRUint64T`/`Const_BMap_MRFaceId_MRUint64T` directly.
    public class _InOptMut_BMap_MRFaceId_MRUint64T
    {
        public BMap_MRFaceId_MRUint64T? Opt;

        public _InOptMut_BMap_MRFaceId_MRUint64T() {}
        public _InOptMut_BMap_MRFaceId_MRUint64T(BMap_MRFaceId_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRFaceId_MRUint64T(BMap_MRFaceId_MRUint64T value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRFaceId_MRUint64T` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRFaceId_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRFaceId_MRUint64T`/`Const_BMap_MRFaceId_MRUint64T` to pass it to the function.
    public class _InOptConst_BMap_MRFaceId_MRUint64T
    {
        public Const_BMap_MRFaceId_MRUint64T? Opt;

        public _InOptConst_BMap_MRFaceId_MRUint64T() {}
        public _InOptConst_BMap_MRFaceId_MRUint64T(Const_BMap_MRFaceId_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRFaceId_MRUint64T(Const_BMap_MRFaceId_MRUint64T value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::VoxelId, MR::VoxelId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRVoxelId_MRVoxelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRVoxelId_MRVoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_VoxelId_MR_VoxelId_Destroy(_Underlying *_this);
            __MR_BMap_MR_VoxelId_MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRVoxelId_MRVoxelId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRVoxelId_MRVoxelId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_VoxelId_MR_VoxelId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_VoxelId_MR_VoxelId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_VoxelId_MR_VoxelId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRVoxelId_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_VoxelId_MR_VoxelId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::VoxelId, MR::VoxelId>` elementwise.
        public unsafe Const_BMap_MRVoxelId_MRVoxelId(MR._ByValue_Buffer_MRVoxelId_MRVoxelId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRVoxelId_MRVoxelId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::VoxelId, MR::VoxelId>::BMap`.
        public unsafe Const_BMap_MRVoxelId_MRVoxelId(MR._ByValue_BMap_MRVoxelId_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRVoxelId_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::VoxelId, MR::VoxelId>`.
    /// This is the non-const half of the class.
    public class BMap_MRVoxelId_MRVoxelId : Const_BMap_MRVoxelId_MRVoxelId
    {
        internal unsafe BMap_MRVoxelId_MRVoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRVoxelId_MRVoxelId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_VoxelId_MR_VoxelId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_VoxelId_MR_VoxelId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_VoxelId_MR_VoxelId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRVoxelId_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_VoxelId_MR_VoxelId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::VoxelId, MR::VoxelId>` elementwise.
        public unsafe BMap_MRVoxelId_MRVoxelId(MR._ByValue_Buffer_MRVoxelId_MRVoxelId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRVoxelId_MRVoxelId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::VoxelId, MR::VoxelId>::BMap`.
        public unsafe BMap_MRVoxelId_MRVoxelId(MR._ByValue_BMap_MRVoxelId_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRVoxelId_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_VoxelId_MR_VoxelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::VoxelId, MR::VoxelId>::operator=`.
        public unsafe MR.BMap_MRVoxelId_MRVoxelId Assign(MR._ByValue_BMap_MRVoxelId_MRVoxelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_VoxelId_MR_VoxelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRVoxelId_MRVoxelId._Underlying *__MR_BMap_MR_VoxelId_MR_VoxelId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRVoxelId_MRVoxelId._Underlying *_other);
            return new(__MR_BMap_MR_VoxelId_MR_VoxelId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRVoxelId_MRVoxelId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRVoxelId_MRVoxelId`/`Const_BMap_MRVoxelId_MRVoxelId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRVoxelId_MRVoxelId
    {
        internal readonly Const_BMap_MRVoxelId_MRVoxelId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRVoxelId_MRVoxelId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRVoxelId_MRVoxelId(MR.Misc._Moved<BMap_MRVoxelId_MRVoxelId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRVoxelId_MRVoxelId(MR.Misc._Moved<BMap_MRVoxelId_MRVoxelId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRVoxelId_MRVoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRVoxelId_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRVoxelId_MRVoxelId`/`Const_BMap_MRVoxelId_MRVoxelId` directly.
    public class _InOptMut_BMap_MRVoxelId_MRVoxelId
    {
        public BMap_MRVoxelId_MRVoxelId? Opt;

        public _InOptMut_BMap_MRVoxelId_MRVoxelId() {}
        public _InOptMut_BMap_MRVoxelId_MRVoxelId(BMap_MRVoxelId_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRVoxelId_MRVoxelId(BMap_MRVoxelId_MRVoxelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRVoxelId_MRVoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRVoxelId_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRVoxelId_MRVoxelId`/`Const_BMap_MRVoxelId_MRVoxelId` to pass it to the function.
    public class _InOptConst_BMap_MRVoxelId_MRVoxelId
    {
        public Const_BMap_MRVoxelId_MRVoxelId? Opt;

        public _InOptConst_BMap_MRVoxelId_MRVoxelId() {}
        public _InOptConst_BMap_MRVoxelId_MRVoxelId(Const_BMap_MRVoxelId_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRVoxelId_MRVoxelId(Const_BMap_MRVoxelId_MRVoxelId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::PixelId, MR::PixelId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRPixelId_MRPixelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRPixelId_MRPixelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_PixelId_MR_PixelId_Destroy(_Underlying *_this);
            __MR_BMap_MR_PixelId_MR_PixelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRPixelId_MRPixelId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRPixelId_MRPixelId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_PixelId_MR_PixelId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_PixelId_MR_PixelId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_PixelId_MR_PixelId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRPixelId_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_PixelId_MR_PixelId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::PixelId, MR::PixelId>` elementwise.
        public unsafe Const_BMap_MRPixelId_MRPixelId(MR._ByValue_Buffer_MRPixelId_MRPixelId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRPixelId_MRPixelId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_PixelId_MR_PixelId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::PixelId, MR::PixelId>::BMap`.
        public unsafe Const_BMap_MRPixelId_MRPixelId(MR._ByValue_BMap_MRPixelId_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRPixelId_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_PixelId_MR_PixelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::PixelId, MR::PixelId>`.
    /// This is the non-const half of the class.
    public class BMap_MRPixelId_MRPixelId : Const_BMap_MRPixelId_MRPixelId
    {
        internal unsafe BMap_MRPixelId_MRPixelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRPixelId_MRPixelId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_PixelId_MR_PixelId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_PixelId_MR_PixelId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_PixelId_MR_PixelId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRPixelId_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_PixelId_MR_PixelId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::PixelId, MR::PixelId>` elementwise.
        public unsafe BMap_MRPixelId_MRPixelId(MR._ByValue_Buffer_MRPixelId_MRPixelId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRPixelId_MRPixelId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_PixelId_MR_PixelId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::PixelId, MR::PixelId>::BMap`.
        public unsafe BMap_MRPixelId_MRPixelId(MR._ByValue_BMap_MRPixelId_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRPixelId_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_PixelId_MR_PixelId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::PixelId, MR::PixelId>::operator=`.
        public unsafe MR.BMap_MRPixelId_MRPixelId Assign(MR._ByValue_BMap_MRPixelId_MRPixelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_PixelId_MR_PixelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRPixelId_MRPixelId._Underlying *__MR_BMap_MR_PixelId_MR_PixelId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRPixelId_MRPixelId._Underlying *_other);
            return new(__MR_BMap_MR_PixelId_MR_PixelId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRPixelId_MRPixelId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRPixelId_MRPixelId`/`Const_BMap_MRPixelId_MRPixelId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRPixelId_MRPixelId
    {
        internal readonly Const_BMap_MRPixelId_MRPixelId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRPixelId_MRPixelId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRPixelId_MRPixelId(MR.Misc._Moved<BMap_MRPixelId_MRPixelId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRPixelId_MRPixelId(MR.Misc._Moved<BMap_MRPixelId_MRPixelId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRPixelId_MRPixelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRPixelId_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRPixelId_MRPixelId`/`Const_BMap_MRPixelId_MRPixelId` directly.
    public class _InOptMut_BMap_MRPixelId_MRPixelId
    {
        public BMap_MRPixelId_MRPixelId? Opt;

        public _InOptMut_BMap_MRPixelId_MRPixelId() {}
        public _InOptMut_BMap_MRPixelId_MRPixelId(BMap_MRPixelId_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRPixelId_MRPixelId(BMap_MRPixelId_MRPixelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRPixelId_MRPixelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRPixelId_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRPixelId_MRPixelId`/`Const_BMap_MRPixelId_MRPixelId` to pass it to the function.
    public class _InOptConst_BMap_MRPixelId_MRPixelId
    {
        public Const_BMap_MRPixelId_MRPixelId? Opt;

        public _InOptConst_BMap_MRPixelId_MRPixelId() {}
        public _InOptConst_BMap_MRPixelId_MRPixelId(Const_BMap_MRPixelId_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRPixelId_MRPixelId(Const_BMap_MRPixelId_MRPixelId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::RegionId, MR::RegionId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRRegionId_MRRegionId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRRegionId_MRRegionId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_RegionId_MR_RegionId_Destroy(_Underlying *_this);
            __MR_BMap_MR_RegionId_MR_RegionId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRRegionId_MRRegionId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRRegionId_MRRegionId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_RegionId_MR_RegionId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_RegionId_MR_RegionId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_RegionId_MR_RegionId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRRegionId_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_RegionId_MR_RegionId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::RegionId, MR::RegionId>` elementwise.
        public unsafe Const_BMap_MRRegionId_MRRegionId(MR._ByValue_Buffer_MRRegionId_MRRegionId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRRegionId_MRRegionId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_RegionId_MR_RegionId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::RegionId, MR::RegionId>::BMap`.
        public unsafe Const_BMap_MRRegionId_MRRegionId(MR._ByValue_BMap_MRRegionId_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRRegionId_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_RegionId_MR_RegionId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::RegionId, MR::RegionId>`.
    /// This is the non-const half of the class.
    public class BMap_MRRegionId_MRRegionId : Const_BMap_MRRegionId_MRRegionId
    {
        internal unsafe BMap_MRRegionId_MRRegionId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRRegionId_MRRegionId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_RegionId_MR_RegionId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_RegionId_MR_RegionId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_RegionId_MR_RegionId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRRegionId_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_RegionId_MR_RegionId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::RegionId, MR::RegionId>` elementwise.
        public unsafe BMap_MRRegionId_MRRegionId(MR._ByValue_Buffer_MRRegionId_MRRegionId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRRegionId_MRRegionId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_RegionId_MR_RegionId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::RegionId, MR::RegionId>::BMap`.
        public unsafe BMap_MRRegionId_MRRegionId(MR._ByValue_BMap_MRRegionId_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRRegionId_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_RegionId_MR_RegionId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::RegionId, MR::RegionId>::operator=`.
        public unsafe MR.BMap_MRRegionId_MRRegionId Assign(MR._ByValue_BMap_MRRegionId_MRRegionId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_RegionId_MR_RegionId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRRegionId_MRRegionId._Underlying *__MR_BMap_MR_RegionId_MR_RegionId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRRegionId_MRRegionId._Underlying *_other);
            return new(__MR_BMap_MR_RegionId_MR_RegionId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRRegionId_MRRegionId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRRegionId_MRRegionId`/`Const_BMap_MRRegionId_MRRegionId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRRegionId_MRRegionId
    {
        internal readonly Const_BMap_MRRegionId_MRRegionId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRRegionId_MRRegionId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRRegionId_MRRegionId(MR.Misc._Moved<BMap_MRRegionId_MRRegionId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRRegionId_MRRegionId(MR.Misc._Moved<BMap_MRRegionId_MRRegionId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRRegionId_MRRegionId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRRegionId_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRRegionId_MRRegionId`/`Const_BMap_MRRegionId_MRRegionId` directly.
    public class _InOptMut_BMap_MRRegionId_MRRegionId
    {
        public BMap_MRRegionId_MRRegionId? Opt;

        public _InOptMut_BMap_MRRegionId_MRRegionId() {}
        public _InOptMut_BMap_MRRegionId_MRRegionId(BMap_MRRegionId_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRRegionId_MRRegionId(BMap_MRRegionId_MRRegionId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRRegionId_MRRegionId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRRegionId_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRRegionId_MRRegionId`/`Const_BMap_MRRegionId_MRRegionId` to pass it to the function.
    public class _InOptConst_BMap_MRRegionId_MRRegionId
    {
        public Const_BMap_MRRegionId_MRRegionId? Opt;

        public _InOptConst_BMap_MRRegionId_MRRegionId() {}
        public _InOptConst_BMap_MRRegionId_MRRegionId(Const_BMap_MRRegionId_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRRegionId_MRRegionId(Const_BMap_MRRegionId_MRRegionId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::NodeId, MR::NodeId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRNodeId_MRNodeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRNodeId_MRNodeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_NodeId_MR_NodeId_Destroy(_Underlying *_this);
            __MR_BMap_MR_NodeId_MR_NodeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRNodeId_MRNodeId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRNodeId_MRNodeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_NodeId_MR_NodeId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_NodeId_MR_NodeId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_NodeId_MR_NodeId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRNodeId_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_NodeId_MR_NodeId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::NodeId, MR::NodeId>` elementwise.
        public unsafe Const_BMap_MRNodeId_MRNodeId(MR._ByValue_Buffer_MRNodeId_MRNodeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRNodeId_MRNodeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_NodeId_MR_NodeId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::NodeId, MR::NodeId>::BMap`.
        public unsafe Const_BMap_MRNodeId_MRNodeId(MR._ByValue_BMap_MRNodeId_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRNodeId_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_NodeId_MR_NodeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::NodeId, MR::NodeId>`.
    /// This is the non-const half of the class.
    public class BMap_MRNodeId_MRNodeId : Const_BMap_MRNodeId_MRNodeId
    {
        internal unsafe BMap_MRNodeId_MRNodeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRNodeId_MRNodeId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_NodeId_MR_NodeId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_NodeId_MR_NodeId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_NodeId_MR_NodeId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRNodeId_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_NodeId_MR_NodeId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::NodeId, MR::NodeId>` elementwise.
        public unsafe BMap_MRNodeId_MRNodeId(MR._ByValue_Buffer_MRNodeId_MRNodeId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRNodeId_MRNodeId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_NodeId_MR_NodeId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::NodeId, MR::NodeId>::BMap`.
        public unsafe BMap_MRNodeId_MRNodeId(MR._ByValue_BMap_MRNodeId_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRNodeId_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_NodeId_MR_NodeId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::NodeId, MR::NodeId>::operator=`.
        public unsafe MR.BMap_MRNodeId_MRNodeId Assign(MR._ByValue_BMap_MRNodeId_MRNodeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_NodeId_MR_NodeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRNodeId_MRNodeId._Underlying *__MR_BMap_MR_NodeId_MR_NodeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRNodeId_MRNodeId._Underlying *_other);
            return new(__MR_BMap_MR_NodeId_MR_NodeId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRNodeId_MRNodeId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRNodeId_MRNodeId`/`Const_BMap_MRNodeId_MRNodeId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRNodeId_MRNodeId
    {
        internal readonly Const_BMap_MRNodeId_MRNodeId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRNodeId_MRNodeId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRNodeId_MRNodeId(MR.Misc._Moved<BMap_MRNodeId_MRNodeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRNodeId_MRNodeId(MR.Misc._Moved<BMap_MRNodeId_MRNodeId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRNodeId_MRNodeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRNodeId_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRNodeId_MRNodeId`/`Const_BMap_MRNodeId_MRNodeId` directly.
    public class _InOptMut_BMap_MRNodeId_MRNodeId
    {
        public BMap_MRNodeId_MRNodeId? Opt;

        public _InOptMut_BMap_MRNodeId_MRNodeId() {}
        public _InOptMut_BMap_MRNodeId_MRNodeId(BMap_MRNodeId_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRNodeId_MRNodeId(BMap_MRNodeId_MRNodeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRNodeId_MRNodeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRNodeId_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRNodeId_MRNodeId`/`Const_BMap_MRNodeId_MRNodeId` to pass it to the function.
    public class _InOptConst_BMap_MRNodeId_MRNodeId
    {
        public Const_BMap_MRNodeId_MRNodeId? Opt;

        public _InOptConst_BMap_MRNodeId_MRNodeId() {}
        public _InOptConst_BMap_MRNodeId_MRNodeId(Const_BMap_MRNodeId_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRNodeId_MRNodeId(Const_BMap_MRNodeId_MRNodeId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::TextureId, MR::TextureId>`.
    /// This is the const half of the class.
    public class Const_BMap_MRTextureId_MRTextureId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRTextureId_MRTextureId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_TextureId_MR_TextureId_Destroy(_Underlying *_this);
            __MR_BMap_MR_TextureId_MR_TextureId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRTextureId_MRTextureId() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRTextureId_MRTextureId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_TextureId_MR_TextureId_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_TextureId_MR_TextureId_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_TextureId_MR_TextureId_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRTextureId_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_TextureId_MR_TextureId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::TextureId, MR::TextureId>` elementwise.
        public unsafe Const_BMap_MRTextureId_MRTextureId(MR._ByValue_Buffer_MRTextureId_MRTextureId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRTextureId_MRTextureId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_TextureId_MR_TextureId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::TextureId, MR::TextureId>::BMap`.
        public unsafe Const_BMap_MRTextureId_MRTextureId(MR._ByValue_BMap_MRTextureId_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRTextureId_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_TextureId_MR_TextureId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::TextureId, MR::TextureId>`.
    /// This is the non-const half of the class.
    public class BMap_MRTextureId_MRTextureId : Const_BMap_MRTextureId_MRTextureId
    {
        internal unsafe BMap_MRTextureId_MRTextureId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRTextureId_MRTextureId B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_TextureId_MR_TextureId_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_TextureId_MR_TextureId_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_TextureId_MR_TextureId_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRTextureId_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_TextureId_MR_TextureId_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::TextureId, MR::TextureId>` elementwise.
        public unsafe BMap_MRTextureId_MRTextureId(MR._ByValue_Buffer_MRTextureId_MRTextureId b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRTextureId_MRTextureId._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_TextureId_MR_TextureId_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::TextureId, MR::TextureId>::BMap`.
        public unsafe BMap_MRTextureId_MRTextureId(MR._ByValue_BMap_MRTextureId_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRTextureId_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_TextureId_MR_TextureId_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::TextureId, MR::TextureId>::operator=`.
        public unsafe MR.BMap_MRTextureId_MRTextureId Assign(MR._ByValue_BMap_MRTextureId_MRTextureId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_TextureId_MR_TextureId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRTextureId_MRTextureId._Underlying *__MR_BMap_MR_TextureId_MR_TextureId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRTextureId_MRTextureId._Underlying *_other);
            return new(__MR_BMap_MR_TextureId_MR_TextureId_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRTextureId_MRTextureId` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRTextureId_MRTextureId`/`Const_BMap_MRTextureId_MRTextureId` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRTextureId_MRTextureId
    {
        internal readonly Const_BMap_MRTextureId_MRTextureId? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRTextureId_MRTextureId() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRTextureId_MRTextureId(MR.Misc._Moved<BMap_MRTextureId_MRTextureId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRTextureId_MRTextureId(MR.Misc._Moved<BMap_MRTextureId_MRTextureId> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRTextureId_MRTextureId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRTextureId_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRTextureId_MRTextureId`/`Const_BMap_MRTextureId_MRTextureId` directly.
    public class _InOptMut_BMap_MRTextureId_MRTextureId
    {
        public BMap_MRTextureId_MRTextureId? Opt;

        public _InOptMut_BMap_MRTextureId_MRTextureId() {}
        public _InOptMut_BMap_MRTextureId_MRTextureId(BMap_MRTextureId_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRTextureId_MRTextureId(BMap_MRTextureId_MRTextureId value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRTextureId_MRTextureId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRTextureId_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRTextureId_MRTextureId`/`Const_BMap_MRTextureId_MRTextureId` to pass it to the function.
    public class _InOptConst_BMap_MRTextureId_MRTextureId
    {
        public Const_BMap_MRTextureId_MRTextureId? Opt;

        public _InOptConst_BMap_MRTextureId_MRTextureId() {}
        public _InOptConst_BMap_MRTextureId_MRTextureId(Const_BMap_MRTextureId_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRTextureId_MRTextureId(Const_BMap_MRTextureId_MRTextureId value) {return new(value);}
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>`.
    /// This is the const half of the class.
    public class Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Destroy", ExactSpelling = true)]
            extern static void __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Destroy(_Underlying *_this);
            __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {Dispose(false);}

        public unsafe MR.Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Get_b", ExactSpelling = true)]
                extern static MR.Const_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Get_b(_Underlying *_this);
                return new(__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public unsafe ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Get_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Get_tsize(_Underlying *_this);
                return *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_Get_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>` elementwise.
        public unsafe Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR._ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::BMap`.
        public unsafe Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR._ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// flat map: I -> T
    /// Generated from class `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>`.
    /// This is the non-const half of the class.
    public class BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag : Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        internal unsafe BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_GetMutable_b", ExactSpelling = true)]
                extern static MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_GetMutable_b(_Underlying *_this);
                return new(__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< target size, all values inside b must be less than this value
        public new unsafe ref ulong Tsize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_GetMutable_tsize", ExactSpelling = true)]
                extern static ulong *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_GetMutable_tsize(_Underlying *_this);
                return ref *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_GetMutable_tsize(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Constructs `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>` elementwise.
        public unsafe BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR._ByValue_Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag b, ulong tsize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFrom", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFrom(MR.Misc._PassBy b_pass_by, MR.Buffer_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *b, ulong tsize);
            _UnderlyingPtr = __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFrom(b.PassByMode, b.Value is not null ? b.Value._UnderlyingPtr : null, tsize);
        }

        /// Generated from constructor `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::BMap`.
        public unsafe BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR._ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BMap<MR::Id<MR::ICPElemtTag>, MR::Id<MR::ICPElemtTag>>::operator=`.
        public unsafe MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag Assign(MR._ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag._Underlying *_other);
            return new(__MR_BMap_MR_Id_MR_ICPElemtTag_MR_Id_MR_ICPElemtTag_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag`/`Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        internal readonly Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR.Misc._Moved<BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(MR.Misc._Moved<BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag`/`Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag` directly.
    public class _InOptMut_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        public BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag? Opt;

        public _InOptMut_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {}
        public _InOptMut_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptMut_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// This is used for optional parameters of class `BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag`/`Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag` to pass it to the function.
    public class _InOptConst_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag
    {
        public Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag? Opt;

        public _InOptConst_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag() {}
        public _InOptConst_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptConst_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag(Const_BMap_MRIdMRICPElemtTag_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// Generated from class `MR::ZeroOnMove<MR_uint64_t>`.
    /// This is the const half of the class.
    public class Const_ZeroOnMove_MRUint64T : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ZeroOnMove_MRUint64T(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_Destroy", ExactSpelling = true)]
            extern static void __MR_ZeroOnMove_uint64_t_Destroy(_Underlying *_this);
            __MR_ZeroOnMove_uint64_t_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ZeroOnMove_MRUint64T() {Dispose(false);}

        public unsafe ulong Val
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_Get_val", ExactSpelling = true)]
                extern static ulong *__MR_ZeroOnMove_uint64_t_Get_val(_Underlying *_this);
                return *__MR_ZeroOnMove_uint64_t_Get_val(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ZeroOnMove_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ZeroOnMove_MRUint64T._Underlying *__MR_ZeroOnMove_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_ZeroOnMove_uint64_t_DefaultConstruct();
        }

        /// Generated from constructor `MR::ZeroOnMove<MR_uint64_t>::ZeroOnMove`.
        public unsafe Const_ZeroOnMove_MRUint64T(MR._ByValue_ZeroOnMove_MRUint64T z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ZeroOnMove_MRUint64T._Underlying *__MR_ZeroOnMove_uint64_t_ConstructFromAnother(MR.Misc._PassBy z_pass_by, MR.ZeroOnMove_MRUint64T._Underlying *z);
            _UnderlyingPtr = __MR_ZeroOnMove_uint64_t_ConstructFromAnother(z.PassByMode, z.Value is not null ? z.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ZeroOnMove<MR_uint64_t>`.
    /// This is the non-const half of the class.
    public class ZeroOnMove_MRUint64T : Const_ZeroOnMove_MRUint64T
    {
        internal unsafe ZeroOnMove_MRUint64T(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref ulong Val
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_GetMutable_val", ExactSpelling = true)]
                extern static ulong *__MR_ZeroOnMove_uint64_t_GetMutable_val(_Underlying *_this);
                return ref *__MR_ZeroOnMove_uint64_t_GetMutable_val(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ZeroOnMove_MRUint64T() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ZeroOnMove_MRUint64T._Underlying *__MR_ZeroOnMove_uint64_t_DefaultConstruct();
            _UnderlyingPtr = __MR_ZeroOnMove_uint64_t_DefaultConstruct();
        }

        /// Generated from constructor `MR::ZeroOnMove<MR_uint64_t>::ZeroOnMove`.
        public unsafe ZeroOnMove_MRUint64T(MR._ByValue_ZeroOnMove_MRUint64T z) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ZeroOnMove_MRUint64T._Underlying *__MR_ZeroOnMove_uint64_t_ConstructFromAnother(MR.Misc._PassBy z_pass_by, MR.ZeroOnMove_MRUint64T._Underlying *z);
            _UnderlyingPtr = __MR_ZeroOnMove_uint64_t_ConstructFromAnother(z.PassByMode, z.Value is not null ? z.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ZeroOnMove<MR_uint64_t>::operator=`.
        public unsafe MR.ZeroOnMove_MRUint64T Assign(MR._ByValue_ZeroOnMove_MRUint64T z)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZeroOnMove_uint64_t_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ZeroOnMove_MRUint64T._Underlying *__MR_ZeroOnMove_uint64_t_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy z_pass_by, MR.ZeroOnMove_MRUint64T._Underlying *z);
            return new(__MR_ZeroOnMove_uint64_t_AssignFromAnother(_UnderlyingPtr, z.PassByMode, z.Value is not null ? z.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ZeroOnMove_MRUint64T` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ZeroOnMove_MRUint64T`/`Const_ZeroOnMove_MRUint64T` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ZeroOnMove_MRUint64T
    {
        internal readonly Const_ZeroOnMove_MRUint64T? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ZeroOnMove_MRUint64T() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ZeroOnMove_MRUint64T(MR.Misc._Moved<ZeroOnMove_MRUint64T> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ZeroOnMove_MRUint64T(MR.Misc._Moved<ZeroOnMove_MRUint64T> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ZeroOnMove_MRUint64T` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ZeroOnMove_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ZeroOnMove_MRUint64T`/`Const_ZeroOnMove_MRUint64T` directly.
    public class _InOptMut_ZeroOnMove_MRUint64T
    {
        public ZeroOnMove_MRUint64T? Opt;

        public _InOptMut_ZeroOnMove_MRUint64T() {}
        public _InOptMut_ZeroOnMove_MRUint64T(ZeroOnMove_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptMut_ZeroOnMove_MRUint64T(ZeroOnMove_MRUint64T value) {return new(value);}
    }

    /// This is used for optional parameters of class `ZeroOnMove_MRUint64T` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ZeroOnMove_MRUint64T`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ZeroOnMove_MRUint64T`/`Const_ZeroOnMove_MRUint64T` to pass it to the function.
    public class _InOptConst_ZeroOnMove_MRUint64T
    {
        public Const_ZeroOnMove_MRUint64T? Opt;

        public _InOptConst_ZeroOnMove_MRUint64T() {}
        public _InOptConst_ZeroOnMove_MRUint64T(Const_ZeroOnMove_MRUint64T value) {Opt = value;}
        public static implicit operator _InOptConst_ZeroOnMove_MRUint64T(Const_ZeroOnMove_MRUint64T value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::UndirectedEdgeId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRUndirectedEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRUndirectedEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRUndirectedEdgeId._Underlying *__MR_NoCtor_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::UndirectedEdgeId>::NoCtor`.
        public unsafe Const_NoCtor_MRUndirectedEdgeId(MR.Const_NoCtor_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRUndirectedEdgeId._Underlying *__MR_NoCtor_MR_UndirectedEdgeId_ConstructFromAnother(MR.NoCtor_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_UndirectedEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::UndirectedEdgeId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRUndirectedEdgeId : Const_NoCtor_MRUndirectedEdgeId
    {
        internal unsafe NoCtor_MRUndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRUndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRUndirectedEdgeId._Underlying *__MR_NoCtor_MR_UndirectedEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::UndirectedEdgeId>::NoCtor`.
        public unsafe NoCtor_MRUndirectedEdgeId(MR.Const_NoCtor_MRUndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_UndirectedEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRUndirectedEdgeId._Underlying *__MR_NoCtor_MR_UndirectedEdgeId_ConstructFromAnother(MR.NoCtor_MRUndirectedEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_UndirectedEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::UndirectedEdgeId>::operator=`.
        public unsafe MR.NoCtor_MRUndirectedEdgeId Assign(MR.Const_NoCtor_MRUndirectedEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_UndirectedEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRUndirectedEdgeId._Underlying *__MR_NoCtor_MR_UndirectedEdgeId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRUndirectedEdgeId._Underlying *_other);
            return new(__MR_NoCtor_MR_UndirectedEdgeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRUndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRUndirectedEdgeId`/`Const_NoCtor_MRUndirectedEdgeId` directly.
    public class _InOptMut_NoCtor_MRUndirectedEdgeId
    {
        public NoCtor_MRUndirectedEdgeId? Opt;

        public _InOptMut_NoCtor_MRUndirectedEdgeId() {}
        public _InOptMut_NoCtor_MRUndirectedEdgeId(NoCtor_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRUndirectedEdgeId(NoCtor_MRUndirectedEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRUndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRUndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRUndirectedEdgeId`/`Const_NoCtor_MRUndirectedEdgeId` to pass it to the function.
    public class _InOptConst_NoCtor_MRUndirectedEdgeId
    {
        public Const_NoCtor_MRUndirectedEdgeId? Opt;

        public _InOptConst_NoCtor_MRUndirectedEdgeId() {}
        public _InOptConst_NoCtor_MRUndirectedEdgeId(Const_NoCtor_MRUndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRUndirectedEdgeId(Const_NoCtor_MRUndirectedEdgeId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::FaceId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRFaceId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRFaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_FaceId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRFaceId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRFaceId._Underlying *__MR_NoCtor_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::FaceId>::NoCtor`.
        public unsafe Const_NoCtor_MRFaceId(MR.Const_NoCtor_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRFaceId._Underlying *__MR_NoCtor_MR_FaceId_ConstructFromAnother(MR.NoCtor_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_FaceId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::FaceId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRFaceId : Const_NoCtor_MRFaceId
    {
        internal unsafe NoCtor_MRFaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRFaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRFaceId._Underlying *__MR_NoCtor_MR_FaceId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::FaceId>::NoCtor`.
        public unsafe NoCtor_MRFaceId(MR.Const_NoCtor_MRFaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_FaceId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRFaceId._Underlying *__MR_NoCtor_MR_FaceId_ConstructFromAnother(MR.NoCtor_MRFaceId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_FaceId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::FaceId>::operator=`.
        public unsafe MR.NoCtor_MRFaceId Assign(MR.Const_NoCtor_MRFaceId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_FaceId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRFaceId._Underlying *__MR_NoCtor_MR_FaceId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRFaceId._Underlying *_other);
            return new(__MR_NoCtor_MR_FaceId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRFaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRFaceId`/`Const_NoCtor_MRFaceId` directly.
    public class _InOptMut_NoCtor_MRFaceId
    {
        public NoCtor_MRFaceId? Opt;

        public _InOptMut_NoCtor_MRFaceId() {}
        public _InOptMut_NoCtor_MRFaceId(NoCtor_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRFaceId(NoCtor_MRFaceId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRFaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRFaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRFaceId`/`Const_NoCtor_MRFaceId` to pass it to the function.
    public class _InOptConst_NoCtor_MRFaceId
    {
        public Const_NoCtor_MRFaceId? Opt;

        public _InOptConst_NoCtor_MRFaceId() {}
        public _InOptConst_NoCtor_MRFaceId(Const_NoCtor_MRFaceId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRFaceId(Const_NoCtor_MRFaceId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::VertId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_VertId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRVertId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRVertId._Underlying *__MR_NoCtor_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::VertId>::NoCtor`.
        public unsafe Const_NoCtor_MRVertId(MR.Const_NoCtor_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRVertId._Underlying *__MR_NoCtor_MR_VertId_ConstructFromAnother(MR.NoCtor_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_VertId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::VertId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRVertId : Const_NoCtor_MRVertId
    {
        internal unsafe NoCtor_MRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRVertId._Underlying *__MR_NoCtor_MR_VertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::VertId>::NoCtor`.
        public unsafe NoCtor_MRVertId(MR.Const_NoCtor_MRVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRVertId._Underlying *__MR_NoCtor_MR_VertId_ConstructFromAnother(MR.NoCtor_MRVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_VertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::VertId>::operator=`.
        public unsafe MR.NoCtor_MRVertId Assign(MR.Const_NoCtor_MRVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRVertId._Underlying *__MR_NoCtor_MR_VertId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRVertId._Underlying *_other);
            return new(__MR_NoCtor_MR_VertId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRVertId`/`Const_NoCtor_MRVertId` directly.
    public class _InOptMut_NoCtor_MRVertId
    {
        public NoCtor_MRVertId? Opt;

        public _InOptMut_NoCtor_MRVertId() {}
        public _InOptMut_NoCtor_MRVertId(NoCtor_MRVertId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRVertId(NoCtor_MRVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRVertId`/`Const_NoCtor_MRVertId` to pass it to the function.
    public class _InOptConst_NoCtor_MRVertId
    {
        public Const_NoCtor_MRVertId? Opt;

        public _InOptConst_NoCtor_MRVertId() {}
        public _InOptConst_NoCtor_MRVertId(Const_NoCtor_MRVertId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRVertId(Const_NoCtor_MRVertId value) {return new(value);}
    }

    // for trivial types, return the type itself
    /// Generated from class `MR::NoCtor<unsigned char>`.
    /// This is the const half of the class.
    public class Const_NoCtor_UnsignedChar : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_UnsignedChar(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_unsigned_char_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_unsigned_char_Destroy(_Underlying *_this);
            __MR_NoCtor_unsigned_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_UnsignedChar() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_UnsignedChar._Underlying *__MR_NoCtor_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<unsigned char>::NoCtor`.
        public unsafe Const_NoCtor_UnsignedChar(MR.Const_NoCtor_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_UnsignedChar._Underlying *__MR_NoCtor_unsigned_char_ConstructFromAnother(MR.NoCtor_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for trivial types, return the type itself
    /// Generated from class `MR::NoCtor<unsigned char>`.
    /// This is the non-const half of the class.
    public class NoCtor_UnsignedChar : Const_NoCtor_UnsignedChar
    {
        internal unsafe NoCtor_UnsignedChar(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_UnsignedChar() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_unsigned_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_UnsignedChar._Underlying *__MR_NoCtor_unsigned_char_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_unsigned_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<unsigned char>::NoCtor`.
        public unsafe NoCtor_UnsignedChar(MR.Const_NoCtor_UnsignedChar _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_unsigned_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_UnsignedChar._Underlying *__MR_NoCtor_unsigned_char_ConstructFromAnother(MR.NoCtor_UnsignedChar._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_unsigned_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<unsigned char>::operator=`.
        public unsafe MR.NoCtor_UnsignedChar Assign(MR.Const_NoCtor_UnsignedChar _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_unsigned_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_UnsignedChar._Underlying *__MR_NoCtor_unsigned_char_AssignFromAnother(_Underlying *_this, MR.NoCtor_UnsignedChar._Underlying *_other);
            return new(__MR_NoCtor_unsigned_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_UnsignedChar` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_UnsignedChar`/`Const_NoCtor_UnsignedChar` directly.
    public class _InOptMut_NoCtor_UnsignedChar
    {
        public NoCtor_UnsignedChar? Opt;

        public _InOptMut_NoCtor_UnsignedChar() {}
        public _InOptMut_NoCtor_UnsignedChar(NoCtor_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_UnsignedChar(NoCtor_UnsignedChar value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_UnsignedChar` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_UnsignedChar`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_UnsignedChar`/`Const_NoCtor_UnsignedChar` to pass it to the function.
    public class _InOptConst_NoCtor_UnsignedChar
    {
        public Const_NoCtor_UnsignedChar? Opt;

        public _InOptConst_NoCtor_UnsignedChar() {}
        public _InOptConst_NoCtor_UnsignedChar(Const_NoCtor_UnsignedChar value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_UnsignedChar(Const_NoCtor_UnsignedChar value) {return new(value);}
    }

    // for trivial types, return the type itself
    /// Generated from class `MR::NoCtor<char>`.
    /// This is the const half of the class.
    public class Const_NoCtor_Char : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_Char(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_char_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_char_Destroy(_Underlying *_this);
            __MR_NoCtor_char_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_Char() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_Char() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_Char._Underlying *__MR_NoCtor_char_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<char>::NoCtor`.
        public unsafe Const_NoCtor_Char(MR.Const_NoCtor_Char _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_Char._Underlying *__MR_NoCtor_char_ConstructFromAnother(MR.NoCtor_Char._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_char_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for trivial types, return the type itself
    /// Generated from class `MR::NoCtor<char>`.
    /// This is the non-const half of the class.
    public class NoCtor_Char : Const_NoCtor_Char
    {
        internal unsafe NoCtor_Char(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_Char() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_char_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_Char._Underlying *__MR_NoCtor_char_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_char_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<char>::NoCtor`.
        public unsafe NoCtor_Char(MR.Const_NoCtor_Char _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_char_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_Char._Underlying *__MR_NoCtor_char_ConstructFromAnother(MR.NoCtor_Char._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_char_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<char>::operator=`.
        public unsafe MR.NoCtor_Char Assign(MR.Const_NoCtor_Char _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_char_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_Char._Underlying *__MR_NoCtor_char_AssignFromAnother(_Underlying *_this, MR.NoCtor_Char._Underlying *_other);
            return new(__MR_NoCtor_char_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_Char` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_Char`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_Char`/`Const_NoCtor_Char` directly.
    public class _InOptMut_NoCtor_Char
    {
        public NoCtor_Char? Opt;

        public _InOptMut_NoCtor_Char() {}
        public _InOptMut_NoCtor_Char(NoCtor_Char value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_Char(NoCtor_Char value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_Char` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_Char`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_Char`/`Const_NoCtor_Char` to pass it to the function.
    public class _InOptConst_NoCtor_Char
    {
        public Const_NoCtor_Char? Opt;

        public _InOptConst_NoCtor_Char() {}
        public _InOptConst_NoCtor_Char(Const_NoCtor_Char value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_Char(Const_NoCtor_Char value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::EdgeId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MREdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_EdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_EdgeId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_EdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MREdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MREdgeId._Underlying *__MR_NoCtor_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::EdgeId>::NoCtor`.
        public unsafe Const_NoCtor_MREdgeId(MR.Const_NoCtor_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MREdgeId._Underlying *__MR_NoCtor_MR_EdgeId_ConstructFromAnother(MR.NoCtor_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_EdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::EdgeId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MREdgeId : Const_NoCtor_MREdgeId
    {
        internal unsafe NoCtor_MREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MREdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MREdgeId._Underlying *__MR_NoCtor_MR_EdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::EdgeId>::NoCtor`.
        public unsafe NoCtor_MREdgeId(MR.Const_NoCtor_MREdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MREdgeId._Underlying *__MR_NoCtor_MR_EdgeId_ConstructFromAnother(MR.NoCtor_MREdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_EdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::EdgeId>::operator=`.
        public unsafe MR.NoCtor_MREdgeId Assign(MR.Const_NoCtor_MREdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MREdgeId._Underlying *__MR_NoCtor_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MREdgeId._Underlying *_other);
            return new(__MR_NoCtor_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MREdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MREdgeId`/`Const_NoCtor_MREdgeId` directly.
    public class _InOptMut_NoCtor_MREdgeId
    {
        public NoCtor_MREdgeId? Opt;

        public _InOptMut_NoCtor_MREdgeId() {}
        public _InOptMut_NoCtor_MREdgeId(NoCtor_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MREdgeId(NoCtor_MREdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MREdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MREdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MREdgeId`/`Const_NoCtor_MREdgeId` to pass it to the function.
    public class _InOptConst_NoCtor_MREdgeId
    {
        public Const_NoCtor_MREdgeId? Opt;

        public _InOptConst_NoCtor_MREdgeId() {}
        public _InOptConst_NoCtor_MREdgeId(Const_NoCtor_MREdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MREdgeId(Const_NoCtor_MREdgeId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::ObjId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRObjId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRObjId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_ObjId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_ObjId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_ObjId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRObjId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRObjId._Underlying *__MR_NoCtor_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::ObjId>::NoCtor`.
        public unsafe Const_NoCtor_MRObjId(MR.Const_NoCtor_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRObjId._Underlying *__MR_NoCtor_MR_ObjId_ConstructFromAnother(MR.NoCtor_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_ObjId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::ObjId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRObjId : Const_NoCtor_MRObjId
    {
        internal unsafe NoCtor_MRObjId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRObjId._Underlying *__MR_NoCtor_MR_ObjId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::ObjId>::NoCtor`.
        public unsafe NoCtor_MRObjId(MR.Const_NoCtor_MRObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_ObjId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRObjId._Underlying *__MR_NoCtor_MR_ObjId_ConstructFromAnother(MR.NoCtor_MRObjId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_ObjId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::ObjId>::operator=`.
        public unsafe MR.NoCtor_MRObjId Assign(MR.Const_NoCtor_MRObjId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_ObjId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRObjId._Underlying *__MR_NoCtor_MR_ObjId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRObjId._Underlying *_other);
            return new(__MR_NoCtor_MR_ObjId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRObjId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRObjId`/`Const_NoCtor_MRObjId` directly.
    public class _InOptMut_NoCtor_MRObjId
    {
        public NoCtor_MRObjId? Opt;

        public _InOptMut_NoCtor_MRObjId() {}
        public _InOptMut_NoCtor_MRObjId(NoCtor_MRObjId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRObjId(NoCtor_MRObjId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRObjId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRObjId`/`Const_NoCtor_MRObjId` to pass it to the function.
    public class _InOptConst_NoCtor_MRObjId
    {
        public Const_NoCtor_MRObjId? Opt;

        public _InOptConst_NoCtor_MRObjId() {}
        public _InOptConst_NoCtor_MRObjId(Const_NoCtor_MRObjId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRObjId(Const_NoCtor_MRObjId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::GraphVertId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRGraphVertId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphVertId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_GraphVertId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_GraphVertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRGraphVertId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphVertId._Underlying *__MR_NoCtor_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::GraphVertId>::NoCtor`.
        public unsafe Const_NoCtor_MRGraphVertId(MR.Const_NoCtor_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphVertId._Underlying *__MR_NoCtor_MR_GraphVertId_ConstructFromAnother(MR.NoCtor_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_GraphVertId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::GraphVertId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRGraphVertId : Const_NoCtor_MRGraphVertId
    {
        internal unsafe NoCtor_MRGraphVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRGraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphVertId._Underlying *__MR_NoCtor_MR_GraphVertId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::GraphVertId>::NoCtor`.
        public unsafe NoCtor_MRGraphVertId(MR.Const_NoCtor_MRGraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphVertId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphVertId._Underlying *__MR_NoCtor_MR_GraphVertId_ConstructFromAnother(MR.NoCtor_MRGraphVertId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_GraphVertId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::GraphVertId>::operator=`.
        public unsafe MR.NoCtor_MRGraphVertId Assign(MR.Const_NoCtor_MRGraphVertId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphVertId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphVertId._Underlying *__MR_NoCtor_MR_GraphVertId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRGraphVertId._Underlying *_other);
            return new(__MR_NoCtor_MR_GraphVertId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRGraphVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRGraphVertId`/`Const_NoCtor_MRGraphVertId` directly.
    public class _InOptMut_NoCtor_MRGraphVertId
    {
        public NoCtor_MRGraphVertId? Opt;

        public _InOptMut_NoCtor_MRGraphVertId() {}
        public _InOptMut_NoCtor_MRGraphVertId(NoCtor_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRGraphVertId(NoCtor_MRGraphVertId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRGraphVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRGraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRGraphVertId`/`Const_NoCtor_MRGraphVertId` to pass it to the function.
    public class _InOptConst_NoCtor_MRGraphVertId
    {
        public Const_NoCtor_MRGraphVertId? Opt;

        public _InOptConst_NoCtor_MRGraphVertId() {}
        public _InOptConst_NoCtor_MRGraphVertId(Const_NoCtor_MRGraphVertId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRGraphVertId(Const_NoCtor_MRGraphVertId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::GraphEdgeId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRGraphEdgeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_GraphEdgeId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_GraphEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRGraphEdgeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphEdgeId._Underlying *__MR_NoCtor_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::GraphEdgeId>::NoCtor`.
        public unsafe Const_NoCtor_MRGraphEdgeId(MR.Const_NoCtor_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphEdgeId._Underlying *__MR_NoCtor_MR_GraphEdgeId_ConstructFromAnother(MR.NoCtor_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_GraphEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::GraphEdgeId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRGraphEdgeId : Const_NoCtor_MRGraphEdgeId
    {
        internal unsafe NoCtor_MRGraphEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRGraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphEdgeId._Underlying *__MR_NoCtor_MR_GraphEdgeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::GraphEdgeId>::NoCtor`.
        public unsafe NoCtor_MRGraphEdgeId(MR.Const_NoCtor_MRGraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphEdgeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphEdgeId._Underlying *__MR_NoCtor_MR_GraphEdgeId_ConstructFromAnother(MR.NoCtor_MRGraphEdgeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_GraphEdgeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::GraphEdgeId>::operator=`.
        public unsafe MR.NoCtor_MRGraphEdgeId Assign(MR.Const_NoCtor_MRGraphEdgeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_GraphEdgeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRGraphEdgeId._Underlying *__MR_NoCtor_MR_GraphEdgeId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRGraphEdgeId._Underlying *_other);
            return new(__MR_NoCtor_MR_GraphEdgeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRGraphEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRGraphEdgeId`/`Const_NoCtor_MRGraphEdgeId` directly.
    public class _InOptMut_NoCtor_MRGraphEdgeId
    {
        public NoCtor_MRGraphEdgeId? Opt;

        public _InOptMut_NoCtor_MRGraphEdgeId() {}
        public _InOptMut_NoCtor_MRGraphEdgeId(NoCtor_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRGraphEdgeId(NoCtor_MRGraphEdgeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRGraphEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRGraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRGraphEdgeId`/`Const_NoCtor_MRGraphEdgeId` to pass it to the function.
    public class _InOptConst_NoCtor_MRGraphEdgeId
    {
        public Const_NoCtor_MRGraphEdgeId? Opt;

        public _InOptConst_NoCtor_MRGraphEdgeId() {}
        public _InOptConst_NoCtor_MRGraphEdgeId(Const_NoCtor_MRGraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRGraphEdgeId(Const_NoCtor_MRGraphEdgeId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::VoxelId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRVoxelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRVoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_VoxelId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRVoxelId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRVoxelId._Underlying *__MR_NoCtor_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::VoxelId>::NoCtor`.
        public unsafe Const_NoCtor_MRVoxelId(MR.Const_NoCtor_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRVoxelId._Underlying *__MR_NoCtor_MR_VoxelId_ConstructFromAnother(MR.NoCtor_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::VoxelId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRVoxelId : Const_NoCtor_MRVoxelId
    {
        internal unsafe NoCtor_MRVoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRVoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRVoxelId._Underlying *__MR_NoCtor_MR_VoxelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::VoxelId>::NoCtor`.
        public unsafe NoCtor_MRVoxelId(MR.Const_NoCtor_MRVoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VoxelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRVoxelId._Underlying *__MR_NoCtor_MR_VoxelId_ConstructFromAnother(MR.NoCtor_MRVoxelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_VoxelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::VoxelId>::operator=`.
        public unsafe MR.NoCtor_MRVoxelId Assign(MR.Const_NoCtor_MRVoxelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_VoxelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRVoxelId._Underlying *__MR_NoCtor_MR_VoxelId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRVoxelId._Underlying *_other);
            return new(__MR_NoCtor_MR_VoxelId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRVoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRVoxelId`/`Const_NoCtor_MRVoxelId` directly.
    public class _InOptMut_NoCtor_MRVoxelId
    {
        public NoCtor_MRVoxelId? Opt;

        public _InOptMut_NoCtor_MRVoxelId() {}
        public _InOptMut_NoCtor_MRVoxelId(NoCtor_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRVoxelId(NoCtor_MRVoxelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRVoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRVoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRVoxelId`/`Const_NoCtor_MRVoxelId` to pass it to the function.
    public class _InOptConst_NoCtor_MRVoxelId
    {
        public Const_NoCtor_MRVoxelId? Opt;

        public _InOptConst_NoCtor_MRVoxelId() {}
        public _InOptConst_NoCtor_MRVoxelId(Const_NoCtor_MRVoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRVoxelId(Const_NoCtor_MRVoxelId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::PixelId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRPixelId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRPixelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_PixelId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_PixelId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_PixelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRPixelId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRPixelId._Underlying *__MR_NoCtor_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::PixelId>::NoCtor`.
        public unsafe Const_NoCtor_MRPixelId(MR.Const_NoCtor_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRPixelId._Underlying *__MR_NoCtor_MR_PixelId_ConstructFromAnother(MR.NoCtor_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_PixelId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::PixelId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRPixelId : Const_NoCtor_MRPixelId
    {
        internal unsafe NoCtor_MRPixelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRPixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRPixelId._Underlying *__MR_NoCtor_MR_PixelId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::PixelId>::NoCtor`.
        public unsafe NoCtor_MRPixelId(MR.Const_NoCtor_MRPixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_PixelId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRPixelId._Underlying *__MR_NoCtor_MR_PixelId_ConstructFromAnother(MR.NoCtor_MRPixelId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_PixelId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::PixelId>::operator=`.
        public unsafe MR.NoCtor_MRPixelId Assign(MR.Const_NoCtor_MRPixelId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_PixelId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRPixelId._Underlying *__MR_NoCtor_MR_PixelId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRPixelId._Underlying *_other);
            return new(__MR_NoCtor_MR_PixelId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRPixelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRPixelId`/`Const_NoCtor_MRPixelId` directly.
    public class _InOptMut_NoCtor_MRPixelId
    {
        public NoCtor_MRPixelId? Opt;

        public _InOptMut_NoCtor_MRPixelId() {}
        public _InOptMut_NoCtor_MRPixelId(NoCtor_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRPixelId(NoCtor_MRPixelId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRPixelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRPixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRPixelId`/`Const_NoCtor_MRPixelId` to pass it to the function.
    public class _InOptConst_NoCtor_MRPixelId
    {
        public Const_NoCtor_MRPixelId? Opt;

        public _InOptConst_NoCtor_MRPixelId() {}
        public _InOptConst_NoCtor_MRPixelId(Const_NoCtor_MRPixelId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRPixelId(Const_NoCtor_MRPixelId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::RegionId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRRegionId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRRegionId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_RegionId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_RegionId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_RegionId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRRegionId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRRegionId._Underlying *__MR_NoCtor_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::RegionId>::NoCtor`.
        public unsafe Const_NoCtor_MRRegionId(MR.Const_NoCtor_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRRegionId._Underlying *__MR_NoCtor_MR_RegionId_ConstructFromAnother(MR.NoCtor_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_RegionId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::RegionId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRRegionId : Const_NoCtor_MRRegionId
    {
        internal unsafe NoCtor_MRRegionId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRRegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRRegionId._Underlying *__MR_NoCtor_MR_RegionId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::RegionId>::NoCtor`.
        public unsafe NoCtor_MRRegionId(MR.Const_NoCtor_MRRegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_RegionId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRRegionId._Underlying *__MR_NoCtor_MR_RegionId_ConstructFromAnother(MR.NoCtor_MRRegionId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_RegionId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::RegionId>::operator=`.
        public unsafe MR.NoCtor_MRRegionId Assign(MR.Const_NoCtor_MRRegionId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_RegionId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRRegionId._Underlying *__MR_NoCtor_MR_RegionId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRRegionId._Underlying *_other);
            return new(__MR_NoCtor_MR_RegionId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRRegionId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRRegionId`/`Const_NoCtor_MRRegionId` directly.
    public class _InOptMut_NoCtor_MRRegionId
    {
        public NoCtor_MRRegionId? Opt;

        public _InOptMut_NoCtor_MRRegionId() {}
        public _InOptMut_NoCtor_MRRegionId(NoCtor_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRRegionId(NoCtor_MRRegionId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRRegionId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRRegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRRegionId`/`Const_NoCtor_MRRegionId` to pass it to the function.
    public class _InOptConst_NoCtor_MRRegionId
    {
        public Const_NoCtor_MRRegionId? Opt;

        public _InOptConst_NoCtor_MRRegionId() {}
        public _InOptConst_NoCtor_MRRegionId(Const_NoCtor_MRRegionId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRRegionId(Const_NoCtor_MRRegionId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::NodeId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRNodeId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRNodeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_NodeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_NodeId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_NodeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRNodeId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRNodeId._Underlying *__MR_NoCtor_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::NodeId>::NoCtor`.
        public unsafe Const_NoCtor_MRNodeId(MR.Const_NoCtor_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRNodeId._Underlying *__MR_NoCtor_MR_NodeId_ConstructFromAnother(MR.NoCtor_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_NodeId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::NodeId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRNodeId : Const_NoCtor_MRNodeId
    {
        internal unsafe NoCtor_MRNodeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRNodeId._Underlying *__MR_NoCtor_MR_NodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::NodeId>::NoCtor`.
        public unsafe NoCtor_MRNodeId(MR.Const_NoCtor_MRNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_NodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRNodeId._Underlying *__MR_NoCtor_MR_NodeId_ConstructFromAnother(MR.NoCtor_MRNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_NodeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::NodeId>::operator=`.
        public unsafe MR.NoCtor_MRNodeId Assign(MR.Const_NoCtor_MRNodeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_NodeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRNodeId._Underlying *__MR_NoCtor_MR_NodeId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRNodeId._Underlying *_other);
            return new(__MR_NoCtor_MR_NodeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRNodeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRNodeId`/`Const_NoCtor_MRNodeId` directly.
    public class _InOptMut_NoCtor_MRNodeId
    {
        public NoCtor_MRNodeId? Opt;

        public _InOptMut_NoCtor_MRNodeId() {}
        public _InOptMut_NoCtor_MRNodeId(NoCtor_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRNodeId(NoCtor_MRNodeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRNodeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRNodeId`/`Const_NoCtor_MRNodeId` to pass it to the function.
    public class _InOptConst_NoCtor_MRNodeId
    {
        public Const_NoCtor_MRNodeId? Opt;

        public _InOptConst_NoCtor_MRNodeId() {}
        public _InOptConst_NoCtor_MRNodeId(Const_NoCtor_MRNodeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRNodeId(Const_NoCtor_MRNodeId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::TextureId>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRTextureId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRTextureId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_TextureId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_TextureId_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_TextureId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRTextureId() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRTextureId._Underlying *__MR_NoCtor_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::TextureId>::NoCtor`.
        public unsafe Const_NoCtor_MRTextureId(MR.Const_NoCtor_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRTextureId._Underlying *__MR_NoCtor_MR_TextureId_ConstructFromAnother(MR.NoCtor_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_TextureId_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::TextureId>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRTextureId : Const_NoCtor_MRTextureId
    {
        internal unsafe NoCtor_MRTextureId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRTextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRTextureId._Underlying *__MR_NoCtor_MR_TextureId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::TextureId>::NoCtor`.
        public unsafe NoCtor_MRTextureId(MR.Const_NoCtor_MRTextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_TextureId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRTextureId._Underlying *__MR_NoCtor_MR_TextureId_ConstructFromAnother(MR.NoCtor_MRTextureId._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_TextureId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::TextureId>::operator=`.
        public unsafe MR.NoCtor_MRTextureId Assign(MR.Const_NoCtor_MRTextureId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_TextureId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRTextureId._Underlying *__MR_NoCtor_MR_TextureId_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRTextureId._Underlying *_other);
            return new(__MR_NoCtor_MR_TextureId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRTextureId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRTextureId`/`Const_NoCtor_MRTextureId` directly.
    public class _InOptMut_NoCtor_MRTextureId
    {
        public NoCtor_MRTextureId? Opt;

        public _InOptMut_NoCtor_MRTextureId() {}
        public _InOptMut_NoCtor_MRTextureId(NoCtor_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRTextureId(NoCtor_MRTextureId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRTextureId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRTextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRTextureId`/`Const_NoCtor_MRTextureId` to pass it to the function.
    public class _InOptConst_NoCtor_MRTextureId
    {
        public Const_NoCtor_MRTextureId? Opt;

        public _InOptConst_NoCtor_MRTextureId() {}
        public _InOptConst_NoCtor_MRTextureId(Const_NoCtor_MRTextureId value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRTextureId(Const_NoCtor_MRTextureId value) {return new(value);}
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::Id<MR::ICPElemtTag>>`.
    /// This is the const half of the class.
    public class Const_NoCtor_MRIdMRICPElemtTag : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoCtor_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_Id_MR_ICPElemtTag_Destroy", ExactSpelling = true)]
            extern static void __MR_NoCtor_MR_Id_MR_ICPElemtTag_Destroy(_Underlying *_this);
            __MR_NoCtor_MR_Id_MR_ICPElemtTag_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoCtor_MRIdMRICPElemtTag() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoCtor_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRIdMRICPElemtTag._Underlying *__MR_NoCtor_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::Id<MR::ICPElemtTag>>::NoCtor`.
        public unsafe Const_NoCtor_MRIdMRICPElemtTag(MR.Const_NoCtor_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRIdMRICPElemtTag._Underlying *__MR_NoCtor_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.NoCtor_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // for our complex types, return wrapped type with default constructor doing nothing
    /// Generated from class `MR::NoCtor<MR::Id<MR::ICPElemtTag>>`.
    /// This is the non-const half of the class.
    public class NoCtor_MRIdMRICPElemtTag : Const_NoCtor_MRIdMRICPElemtTag
    {
        internal unsafe NoCtor_MRIdMRICPElemtTag(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoCtor_MRIdMRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoCtor_MRIdMRICPElemtTag._Underlying *__MR_NoCtor_MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_NoCtor_MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoCtor<MR::Id<MR::ICPElemtTag>>::NoCtor`.
        public unsafe NoCtor_MRIdMRICPElemtTag(MR.Const_NoCtor_MRIdMRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRIdMRICPElemtTag._Underlying *__MR_NoCtor_MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.NoCtor_MRIdMRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_NoCtor_MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::NoCtor<MR::Id<MR::ICPElemtTag>>::operator=`.
        public unsafe MR.NoCtor_MRIdMRICPElemtTag Assign(MR.Const_NoCtor_MRIdMRICPElemtTag _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoCtor_MR_Id_MR_ICPElemtTag_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoCtor_MRIdMRICPElemtTag._Underlying *__MR_NoCtor_MR_Id_MR_ICPElemtTag_AssignFromAnother(_Underlying *_this, MR.NoCtor_MRIdMRICPElemtTag._Underlying *_other);
            return new(__MR_NoCtor_MR_Id_MR_ICPElemtTag_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoCtor_MRIdMRICPElemtTag` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoCtor_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRIdMRICPElemtTag`/`Const_NoCtor_MRIdMRICPElemtTag` directly.
    public class _InOptMut_NoCtor_MRIdMRICPElemtTag
    {
        public NoCtor_MRIdMRICPElemtTag? Opt;

        public _InOptMut_NoCtor_MRIdMRICPElemtTag() {}
        public _InOptMut_NoCtor_MRIdMRICPElemtTag(NoCtor_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptMut_NoCtor_MRIdMRICPElemtTag(NoCtor_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoCtor_MRIdMRICPElemtTag` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoCtor_MRIdMRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoCtor_MRIdMRICPElemtTag`/`Const_NoCtor_MRIdMRICPElemtTag` to pass it to the function.
    public class _InOptConst_NoCtor_MRIdMRICPElemtTag
    {
        public Const_NoCtor_MRIdMRICPElemtTag? Opt;

        public _InOptConst_NoCtor_MRIdMRICPElemtTag() {}
        public _InOptConst_NoCtor_MRIdMRICPElemtTag(Const_NoCtor_MRIdMRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptConst_NoCtor_MRIdMRICPElemtTag(Const_NoCtor_MRIdMRICPElemtTag value) {return new(value);}
    }

    /// mapping of mesh elements: old -> new,
    /// the mapping is tight (or packing) in the sense that there are no unused new elements within [0, (e/f/v).tsize)
    /// Generated from class `MR::PackMapping`.
    /// This is the const half of the class.
    public class Const_PackMapping : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PackMapping(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_Destroy", ExactSpelling = true)]
            extern static void __MR_PackMapping_Destroy(_Underlying *_this);
            __MR_PackMapping_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PackMapping() {Dispose(false);}

        public unsafe MR.Const_UndirectedEdgeBMap E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_Get_e", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeBMap._Underlying *__MR_PackMapping_Get_e(_Underlying *_this);
                return new(__MR_PackMapping_Get_e(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_FaceBMap F
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_Get_f", ExactSpelling = true)]
                extern static MR.Const_FaceBMap._Underlying *__MR_PackMapping_Get_f(_Underlying *_this);
                return new(__MR_PackMapping_Get_f(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertBMap V
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_Get_v", ExactSpelling = true)]
                extern static MR.Const_VertBMap._Underlying *__MR_PackMapping_Get_v(_Underlying *_this);
                return new(__MR_PackMapping_Get_v(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PackMapping() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_DefaultConstruct();
            _UnderlyingPtr = __MR_PackMapping_DefaultConstruct();
        }

        /// Constructs `MR::PackMapping` elementwise.
        public unsafe Const_PackMapping(MR._ByValue_UndirectedEdgeBMap e, MR._ByValue_FaceBMap f, MR._ByValue_VertBMap v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_ConstructFrom", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_ConstructFrom(MR.Misc._PassBy e_pass_by, MR.UndirectedEdgeBMap._Underlying *e, MR.Misc._PassBy f_pass_by, MR.FaceBMap._Underlying *f, MR.Misc._PassBy v_pass_by, MR.VertBMap._Underlying *v);
            _UnderlyingPtr = __MR_PackMapping_ConstructFrom(e.PassByMode, e.Value is not null ? e.Value._UnderlyingPtr : null, f.PassByMode, f.Value is not null ? f.Value._UnderlyingPtr : null, v.PassByMode, v.Value is not null ? v.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PackMapping::PackMapping`.
        public unsafe Const_PackMapping(MR._ByValue_PackMapping _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PackMapping._Underlying *_other);
            _UnderlyingPtr = __MR_PackMapping_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// mapping of mesh elements: old -> new,
    /// the mapping is tight (or packing) in the sense that there are no unused new elements within [0, (e/f/v).tsize)
    /// Generated from class `MR::PackMapping`.
    /// This is the non-const half of the class.
    public class PackMapping : Const_PackMapping
    {
        internal unsafe PackMapping(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.UndirectedEdgeBMap E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_GetMutable_e", ExactSpelling = true)]
                extern static MR.UndirectedEdgeBMap._Underlying *__MR_PackMapping_GetMutable_e(_Underlying *_this);
                return new(__MR_PackMapping_GetMutable_e(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.FaceBMap F
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_GetMutable_f", ExactSpelling = true)]
                extern static MR.FaceBMap._Underlying *__MR_PackMapping_GetMutable_f(_Underlying *_this);
                return new(__MR_PackMapping_GetMutable_f(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertBMap V
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_GetMutable_v", ExactSpelling = true)]
                extern static MR.VertBMap._Underlying *__MR_PackMapping_GetMutable_v(_Underlying *_this);
                return new(__MR_PackMapping_GetMutable_v(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PackMapping() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_DefaultConstruct();
            _UnderlyingPtr = __MR_PackMapping_DefaultConstruct();
        }

        /// Constructs `MR::PackMapping` elementwise.
        public unsafe PackMapping(MR._ByValue_UndirectedEdgeBMap e, MR._ByValue_FaceBMap f, MR._ByValue_VertBMap v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_ConstructFrom", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_ConstructFrom(MR.Misc._PassBy e_pass_by, MR.UndirectedEdgeBMap._Underlying *e, MR.Misc._PassBy f_pass_by, MR.FaceBMap._Underlying *f, MR.Misc._PassBy v_pass_by, MR.VertBMap._Underlying *v);
            _UnderlyingPtr = __MR_PackMapping_ConstructFrom(e.PassByMode, e.Value is not null ? e.Value._UnderlyingPtr : null, f.PassByMode, f.Value is not null ? f.Value._UnderlyingPtr : null, v.PassByMode, v.Value is not null ? v.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PackMapping::PackMapping`.
        public unsafe PackMapping(MR._ByValue_PackMapping _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PackMapping._Underlying *_other);
            _UnderlyingPtr = __MR_PackMapping_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PackMapping::operator=`.
        public unsafe MR.PackMapping Assign(MR._ByValue_PackMapping _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PackMapping_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_PackMapping_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PackMapping._Underlying *_other);
            return new(__MR_PackMapping_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PackMapping` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PackMapping`/`Const_PackMapping` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PackMapping
    {
        internal readonly Const_PackMapping? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PackMapping() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PackMapping(MR.Misc._Moved<PackMapping> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PackMapping(MR.Misc._Moved<PackMapping> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PackMapping` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PackMapping`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PackMapping`/`Const_PackMapping` directly.
    public class _InOptMut_PackMapping
    {
        public PackMapping? Opt;

        public _InOptMut_PackMapping() {}
        public _InOptMut_PackMapping(PackMapping value) {Opt = value;}
        public static implicit operator _InOptMut_PackMapping(PackMapping value) {return new(value);}
    }

    /// This is used for optional parameters of class `PackMapping` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PackMapping`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PackMapping`/`Const_PackMapping` to pass it to the function.
    public class _InOptConst_PackMapping
    {
        public Const_PackMapping? Opt;

        public _InOptConst_PackMapping() {}
        public _InOptConst_PackMapping(Const_PackMapping value) {Opt = value;}
        public static implicit operator _InOptConst_PackMapping(Const_PackMapping value) {return new(value);}
    }

    /// given some buffer map and a key, returns the value associated with the key, or default value if key is invalid
    /// Generated from function `MR::getAt<MR::UndirectedEdgeId, MR::UndirectedEdgeId>`.
    /// Parameter `def` defaults to `{}`.
    public static unsafe MR.UndirectedEdgeId GetAt(MR.Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId bmap, MR.UndirectedEdgeId key, MR._InOpt_UndirectedEdgeId def = default)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getAt", ExactSpelling = true)]
        extern static MR.UndirectedEdgeId __MR_getAt(MR.Const_Buffer_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *bmap, MR.UndirectedEdgeId key, MR.UndirectedEdgeId *def);
        return __MR_getAt(bmap._UnderlyingPtr, key, def.HasValue ? &def.Object : null);
    }
}
