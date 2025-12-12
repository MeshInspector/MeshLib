public static partial class MR
{
    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::UndirectedEdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::UndirectedEdgeId>`
    /// This is the const reference to the struct.
    public class Const_UndirectedEdgeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.UndirectedEdgeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly UndirectedEdgeId UnderlyingStruct => ref *(UndirectedEdgeId *)_UnderlyingPtr;

        internal unsafe Const_UndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeId_Destroy(_Underlying *_this);
            __MR_UndirectedEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UndirectedEdgeId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_UndirectedEdgeId(Const_UndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe Const_UndirectedEdgeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe Const_UndirectedEdgeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe Const_UndirectedEdgeId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::UndirectedEdgeId::operator int`.
        public static unsafe implicit operator int(MR.Const_UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_UndirectedEdgeId_ConvertTo_int(MR.Const_UndirectedEdgeId._Underlying *_this);
            return __MR_UndirectedEdgeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::UndirectedEdgeId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeId_ConvertTo_bool(MR.Const_UndirectedEdgeId._Underlying *_this);
            return __MR_UndirectedEdgeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeId_valid(_Underlying *_this);
            return __MR_UndirectedEdgeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeId::operator==`.
        public static unsafe bool operator==(MR.Const_UndirectedEdgeId _this, MR.UndirectedEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_UndirectedEdgeId(MR.Const_UndirectedEdgeId._Underlying *_this, MR.UndirectedEdgeId b);
            return __MR_equal_MR_UndirectedEdgeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_UndirectedEdgeId _this, MR.UndirectedEdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::UndirectedEdgeId::operator<`.
        public unsafe bool Less(MR.UndirectedEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_UndirectedEdgeId(_Underlying *_this, MR.UndirectedEdgeId b);
            return __MR_less_MR_UndirectedEdgeId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeId::operator--`.
        public static unsafe Mut_UndirectedEdgeId operator--(MR.Const_UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_decr_MR_UndirectedEdgeId(MR.Const_UndirectedEdgeId._Underlying *_this);
            Mut_UndirectedEdgeId _this_copy = new(_this);
            MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_decr_MR_UndirectedEdgeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::UndirectedEdgeId::operator++`.
        public static unsafe Mut_UndirectedEdgeId operator++(MR.Const_UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_incr_MR_UndirectedEdgeId(MR.Const_UndirectedEdgeId._Underlying *_this);
            Mut_UndirectedEdgeId _this_copy = new(_this);
            MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_incr_MR_UndirectedEdgeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.UndirectedEdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.UndirectedEdgeId)
                return this == (MR.UndirectedEdgeId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::UndirectedEdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::UndirectedEdgeId>`
    /// This is the non-const reference to the struct.
    public class Mut_UndirectedEdgeId : Const_UndirectedEdgeId
    {
        /// Get the underlying struct.
        public unsafe new ref UndirectedEdgeId UnderlyingStruct => ref *(UndirectedEdgeId *)_UnderlyingPtr;

        internal unsafe Mut_UndirectedEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_UndirectedEdgeId(Const_UndirectedEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_UndirectedEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe Mut_UndirectedEdgeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe Mut_UndirectedEdgeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe Mut_UndirectedEdgeId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.UndirectedEdgeId _ctor_result = __MR_UndirectedEdgeId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::UndirectedEdgeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_get", ExactSpelling = true)]
            extern static int *__MR_UndirectedEdgeId_get(_Underlying *_this);
            return ref *__MR_UndirectedEdgeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::UndirectedEdgeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_decr_MR_UndirectedEdgeId(_Underlying *_this);
            MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_decr_MR_UndirectedEdgeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_incr_MR_UndirectedEdgeId(_Underlying *_this);
            MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_incr_MR_UndirectedEdgeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeId::operator-=`.
        public unsafe MR.Mut_UndirectedEdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_UndirectedEdgeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeId::operator+=`.
        public unsafe MR.Mut_UndirectedEdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeId_add_assign(_Underlying *_this, int a);
            return new(__MR_UndirectedEdgeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::UndirectedEdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::UndirectedEdgeId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct UndirectedEdgeId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator UndirectedEdgeId(Const_UndirectedEdgeId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_UndirectedEdgeId(UndirectedEdgeId other) => new(new Mut_UndirectedEdgeId((Mut_UndirectedEdgeId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public UndirectedEdgeId(UndirectedEdgeId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UndirectedEdgeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_DefaultConstruct();
            this = __MR_UndirectedEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe UndirectedEdgeId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_UndirectedEdgeId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe UndirectedEdgeId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct_int(int i);
            this = __MR_UndirectedEdgeId_Construct_int(i);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::UndirectedEdgeId::UndirectedEdgeId`.
        public unsafe UndirectedEdgeId(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_UndirectedEdgeId_Construct_uint64_t(ulong i);
            this = __MR_UndirectedEdgeId_Construct_uint64_t(i);
        }

        /// Generated from conversion operator `MR::UndirectedEdgeId::operator int`.
        public static unsafe implicit operator int(MR.UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_UndirectedEdgeId_ConvertTo_int(MR.Const_UndirectedEdgeId._Underlying *_this);
            return __MR_UndirectedEdgeId_ConvertTo_int((MR.Mut_UndirectedEdgeId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::UndirectedEdgeId::operator bool`.
        public static unsafe explicit operator bool(MR.UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeId_ConvertTo_bool(MR.Const_UndirectedEdgeId._Underlying *_this);
            return __MR_UndirectedEdgeId_ConvertTo_bool((MR.Mut_UndirectedEdgeId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::UndirectedEdgeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_UndirectedEdgeId_valid(MR.UndirectedEdgeId *_this);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                return __MR_UndirectedEdgeId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::UndirectedEdgeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_get", ExactSpelling = true)]
            extern static int *__MR_UndirectedEdgeId_get(MR.UndirectedEdgeId *_this);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                return ref *__MR_UndirectedEdgeId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::UndirectedEdgeId::operator==`.
        public static unsafe bool operator==(MR.UndirectedEdgeId _this, MR.UndirectedEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_UndirectedEdgeId(MR.Const_UndirectedEdgeId._Underlying *_this, MR.UndirectedEdgeId b);
            return __MR_equal_MR_UndirectedEdgeId((MR.Mut_UndirectedEdgeId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.UndirectedEdgeId _this, MR.UndirectedEdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::UndirectedEdgeId::operator<`.
        public unsafe bool Less(MR.UndirectedEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_UndirectedEdgeId(MR.UndirectedEdgeId *_this, MR.UndirectedEdgeId b);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                return __MR_less_MR_UndirectedEdgeId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::UndirectedEdgeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_decr_MR_UndirectedEdgeId(MR.UndirectedEdgeId *_this);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_decr_MR_UndirectedEdgeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::UndirectedEdgeId::operator--`.
        public unsafe UndirectedEdgeId Decr(MR.UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_decr_MR_UndirectedEdgeId(UndirectedEdgeId *_this);
            UndirectedEdgeId _this_copy = new(_this);
            MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_decr_MR_UndirectedEdgeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::UndirectedEdgeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_incr_MR_UndirectedEdgeId(MR.UndirectedEdgeId *_this);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_incr_MR_UndirectedEdgeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::UndirectedEdgeId::operator++`.
        public unsafe UndirectedEdgeId Incr(MR.UndirectedEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_incr_MR_UndirectedEdgeId(UndirectedEdgeId *_this);
            UndirectedEdgeId _this_copy = new(_this);
            MR.Mut_UndirectedEdgeId _unused_ret = new(__MR_incr_MR_UndirectedEdgeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::UndirectedEdgeId::operator-=`.
        public unsafe MR.Mut_UndirectedEdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeId_sub_assign(MR.UndirectedEdgeId *_this, int a);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                return new(__MR_UndirectedEdgeId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::UndirectedEdgeId::operator+=`.
        public unsafe MR.Mut_UndirectedEdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_UndirectedEdgeId._Underlying *__MR_UndirectedEdgeId_add_assign(MR.UndirectedEdgeId *_this, int a);
            fixed (MR.UndirectedEdgeId *__ptr__this = &this)
            {
                return new(__MR_UndirectedEdgeId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.UndirectedEdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.UndirectedEdgeId)
                return this == (MR.UndirectedEdgeId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_UndirectedEdgeId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_UndirectedEdgeId`/`Const_UndirectedEdgeId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_UndirectedEdgeId
    {
        public readonly bool HasValue;
        internal readonly UndirectedEdgeId Object;
        public UndirectedEdgeId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_UndirectedEdgeId() {HasValue = false;}
        public _InOpt_UndirectedEdgeId(UndirectedEdgeId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_UndirectedEdgeId(UndirectedEdgeId new_value) {return new(new_value);}
        public _InOpt_UndirectedEdgeId(Const_UndirectedEdgeId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_UndirectedEdgeId(Const_UndirectedEdgeId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_UndirectedEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_UndirectedEdgeId`/`Const_UndirectedEdgeId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `UndirectedEdgeId`.
    public class _InOptMut_UndirectedEdgeId
    {
        public Mut_UndirectedEdgeId? Opt;

        public _InOptMut_UndirectedEdgeId() {}
        public _InOptMut_UndirectedEdgeId(Mut_UndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_UndirectedEdgeId(Mut_UndirectedEdgeId value) {return new(value);}
        public unsafe _InOptMut_UndirectedEdgeId(ref UndirectedEdgeId value)
        {
            fixed (UndirectedEdgeId *value_ptr = &value)
            {
                Opt = new((Const_UndirectedEdgeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_UndirectedEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UndirectedEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_UndirectedEdgeId`/`Const_UndirectedEdgeId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `UndirectedEdgeId`.
    public class _InOptConst_UndirectedEdgeId
    {
        public Const_UndirectedEdgeId? Opt;

        public _InOptConst_UndirectedEdgeId() {}
        public _InOptConst_UndirectedEdgeId(Const_UndirectedEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_UndirectedEdgeId(Const_UndirectedEdgeId value) {return new(value);}
        public unsafe _InOptConst_UndirectedEdgeId(ref readonly UndirectedEdgeId value)
        {
            fixed (UndirectedEdgeId *value_ptr = &value)
            {
                Opt = new((Const_UndirectedEdgeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::FaceId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::FaceId>`
    /// This is the const reference to the struct.
    public class Const_FaceId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.FaceId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly FaceId UnderlyingStruct => ref *(FaceId *)_UnderlyingPtr;

        internal unsafe Const_FaceId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceId_Destroy(_Underlying *_this);
            __MR_FaceId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_FaceId(Const_FaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Const_FaceId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Const_FaceId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_int", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Const_FaceId(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_unsigned_int(uint i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct_unsigned_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Const_FaceId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::FaceId::operator int`.
        public static unsafe implicit operator int(MR.Const_FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_FaceId_ConvertTo_int(MR.Const_FaceId._Underlying *_this);
            return __MR_FaceId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::FaceId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_FaceId_ConvertTo_bool(MR.Const_FaceId._Underlying *_this);
            return __MR_FaceId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FaceId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_valid", ExactSpelling = true)]
            extern static byte __MR_FaceId_valid(_Underlying *_this);
            return __MR_FaceId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FaceId::operator==`.
        public static unsafe bool operator==(MR.Const_FaceId _this, MR.FaceId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_FaceId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_FaceId(MR.Const_FaceId._Underlying *_this, MR.FaceId b);
            return __MR_equal_MR_FaceId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_FaceId _this, MR.FaceId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::FaceId::operator<`.
        public unsafe bool Less(MR.FaceId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_FaceId", ExactSpelling = true)]
            extern static byte __MR_less_MR_FaceId(_Underlying *_this, MR.FaceId b);
            return __MR_less_MR_FaceId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::FaceId::operator--`.
        public static unsafe Mut_FaceId operator--(MR.Const_FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_decr_MR_FaceId(MR.Const_FaceId._Underlying *_this);
            Mut_FaceId _this_copy = new(_this);
            MR.Mut_FaceId _unused_ret = new(__MR_decr_MR_FaceId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::FaceId::operator++`.
        public static unsafe Mut_FaceId operator++(MR.Const_FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_incr_MR_FaceId(MR.Const_FaceId._Underlying *_this);
            Mut_FaceId _this_copy = new(_this);
            MR.Mut_FaceId _unused_ret = new(__MR_incr_MR_FaceId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.FaceId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.FaceId)
                return this == (MR.FaceId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::FaceId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::FaceId>`
    /// This is the non-const reference to the struct.
    public class Mut_FaceId : Const_FaceId
    {
        /// Get the underlying struct.
        public unsafe new ref FaceId UnderlyingStruct => ref *(FaceId *)_UnderlyingPtr;

        internal unsafe Mut_FaceId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_FaceId(Const_FaceId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_FaceId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Mut_FaceId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Mut_FaceId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_int", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Mut_FaceId(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_unsigned_int(uint i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct_unsigned_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe Mut_FaceId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.FaceId _ctor_result = __MR_FaceId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::FaceId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_get", ExactSpelling = true)]
            extern static int *__MR_FaceId_get(_Underlying *_this);
            return ref *__MR_FaceId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::FaceId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_decr_MR_FaceId(_Underlying *_this);
            MR.Mut_FaceId _unused_ret = new(__MR_decr_MR_FaceId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_incr_MR_FaceId(_Underlying *_this);
            MR.Mut_FaceId _unused_ret = new(__MR_incr_MR_FaceId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FaceId::operator-=`.
        public unsafe MR.Mut_FaceId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_FaceId_sub_assign(_Underlying *_this, int a);
            return new(__MR_FaceId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::FaceId::operator+=`.
        public unsafe MR.Mut_FaceId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_FaceId_add_assign(_Underlying *_this, int a);
            return new(__MR_FaceId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::FaceId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::FaceId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct FaceId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator FaceId(Const_FaceId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_FaceId(FaceId other) => new(new Mut_FaceId((Mut_FaceId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public FaceId(FaceId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_DefaultConstruct();
            this = __MR_FaceId_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe FaceId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_FaceId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe FaceId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_int", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_int(int i);
            this = __MR_FaceId_Construct_int(i);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe FaceId(uint i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_unsigned_int(uint i);
            this = __MR_FaceId_Construct_unsigned_int(i);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::FaceId::FaceId`.
        public unsafe FaceId(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.FaceId __MR_FaceId_Construct_uint64_t(ulong i);
            this = __MR_FaceId_Construct_uint64_t(i);
        }

        /// Generated from conversion operator `MR::FaceId::operator int`.
        public static unsafe implicit operator int(MR.FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_FaceId_ConvertTo_int(MR.Const_FaceId._Underlying *_this);
            return __MR_FaceId_ConvertTo_int((MR.Mut_FaceId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::FaceId::operator bool`.
        public static unsafe explicit operator bool(MR.FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_FaceId_ConvertTo_bool(MR.Const_FaceId._Underlying *_this);
            return __MR_FaceId_ConvertTo_bool((MR.Mut_FaceId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::FaceId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_valid", ExactSpelling = true)]
            extern static byte __MR_FaceId_valid(MR.FaceId *_this);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                return __MR_FaceId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::FaceId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_get", ExactSpelling = true)]
            extern static int *__MR_FaceId_get(MR.FaceId *_this);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                return ref *__MR_FaceId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::FaceId::operator==`.
        public static unsafe bool operator==(MR.FaceId _this, MR.FaceId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_FaceId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_FaceId(MR.Const_FaceId._Underlying *_this, MR.FaceId b);
            return __MR_equal_MR_FaceId((MR.Mut_FaceId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.FaceId _this, MR.FaceId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::FaceId::operator<`.
        public unsafe bool Less(MR.FaceId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_FaceId", ExactSpelling = true)]
            extern static byte __MR_less_MR_FaceId(MR.FaceId *_this, MR.FaceId b);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                return __MR_less_MR_FaceId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::FaceId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_decr_MR_FaceId(MR.FaceId *_this);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                MR.Mut_FaceId _unused_ret = new(__MR_decr_MR_FaceId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::FaceId::operator--`.
        public unsafe FaceId Decr(MR.FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_decr_MR_FaceId(FaceId *_this);
            FaceId _this_copy = new(_this);
            MR.Mut_FaceId _unused_ret = new(__MR_decr_MR_FaceId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::FaceId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_incr_MR_FaceId(MR.FaceId *_this);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                MR.Mut_FaceId _unused_ret = new(__MR_incr_MR_FaceId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::FaceId::operator++`.
        public unsafe FaceId Incr(MR.FaceId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_FaceId", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_incr_MR_FaceId(FaceId *_this);
            FaceId _this_copy = new(_this);
            MR.Mut_FaceId _unused_ret = new(__MR_incr_MR_FaceId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::FaceId::operator-=`.
        public unsafe MR.Mut_FaceId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_FaceId_sub_assign(MR.FaceId *_this, int a);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                return new(__MR_FaceId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::FaceId::operator+=`.
        public unsafe MR.Mut_FaceId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_FaceId._Underlying *__MR_FaceId_add_assign(MR.FaceId *_this, int a);
            fixed (MR.FaceId *__ptr__this = &this)
            {
                return new(__MR_FaceId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.FaceId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.FaceId)
                return this == (MR.FaceId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_FaceId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_FaceId`/`Const_FaceId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_FaceId
    {
        public readonly bool HasValue;
        internal readonly FaceId Object;
        public FaceId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_FaceId() {HasValue = false;}
        public _InOpt_FaceId(FaceId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_FaceId(FaceId new_value) {return new(new_value);}
        public _InOpt_FaceId(Const_FaceId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_FaceId(Const_FaceId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_FaceId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_FaceId`/`Const_FaceId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `FaceId`.
    public class _InOptMut_FaceId
    {
        public Mut_FaceId? Opt;

        public _InOptMut_FaceId() {}
        public _InOptMut_FaceId(Mut_FaceId value) {Opt = value;}
        public static implicit operator _InOptMut_FaceId(Mut_FaceId value) {return new(value);}
        public unsafe _InOptMut_FaceId(ref FaceId value)
        {
            fixed (FaceId *value_ptr = &value)
            {
                Opt = new((Const_FaceId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_FaceId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_FaceId`/`Const_FaceId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `FaceId`.
    public class _InOptConst_FaceId
    {
        public Const_FaceId? Opt;

        public _InOptConst_FaceId() {}
        public _InOptConst_FaceId(Const_FaceId value) {Opt = value;}
        public static implicit operator _InOptConst_FaceId(Const_FaceId value) {return new(value);}
        public unsafe _InOptConst_FaceId(ref readonly FaceId value)
        {
            fixed (FaceId *value_ptr = &value)
            {
                Opt = new((Const_FaceId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::VertId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::VertId>`
    /// This is the const reference to the struct.
    public class Const_VertId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.VertId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly VertId UnderlyingStruct => ref *(VertId *)_UnderlyingPtr;

        internal unsafe Const_VertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Destroy", ExactSpelling = true)]
            extern static void __MR_VertId_Destroy(_Underlying *_this);
            __MR_VertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_VertId(Const_VertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe Const_VertId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe Const_VertId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct_int", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe Const_VertId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::VertId::operator int`.
        public static unsafe implicit operator int(MR.Const_VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_VertId_ConvertTo_int(MR.Const_VertId._Underlying *_this);
            return __MR_VertId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::VertId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_VertId_ConvertTo_bool(MR.Const_VertId._Underlying *_this);
            return __MR_VertId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VertId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_valid", ExactSpelling = true)]
            extern static byte __MR_VertId_valid(_Underlying *_this);
            return __MR_VertId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VertId::operator==`.
        public static unsafe bool operator==(MR.Const_VertId _this, MR.VertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VertId(MR.Const_VertId._Underlying *_this, MR.VertId b);
            return __MR_equal_MR_VertId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_VertId _this, MR.VertId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::VertId::operator<`.
        public unsafe bool Less(MR.VertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_VertId", ExactSpelling = true)]
            extern static byte __MR_less_MR_VertId(_Underlying *_this, MR.VertId b);
            return __MR_less_MR_VertId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::VertId::operator--`.
        public static unsafe Mut_VertId operator--(MR.Const_VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_decr_MR_VertId(MR.Const_VertId._Underlying *_this);
            Mut_VertId _this_copy = new(_this);
            MR.Mut_VertId _unused_ret = new(__MR_decr_MR_VertId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::VertId::operator++`.
        public static unsafe Mut_VertId operator++(MR.Const_VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_incr_MR_VertId(MR.Const_VertId._Underlying *_this);
            Mut_VertId _this_copy = new(_this);
            MR.Mut_VertId _unused_ret = new(__MR_incr_MR_VertId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.VertId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.VertId)
                return this == (MR.VertId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::VertId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::VertId>`
    /// This is the non-const reference to the struct.
    public class Mut_VertId : Const_VertId
    {
        /// Get the underlying struct.
        public unsafe new ref VertId UnderlyingStruct => ref *(VertId *)_UnderlyingPtr;

        internal unsafe Mut_VertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_VertId(Const_VertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_VertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe Mut_VertId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe Mut_VertId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct_int", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe Mut_VertId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.VertId _ctor_result = __MR_VertId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::VertId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_get", ExactSpelling = true)]
            extern static int *__MR_VertId_get(_Underlying *_this);
            return ref *__MR_VertId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::VertId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_decr_MR_VertId(_Underlying *_this);
            MR.Mut_VertId _unused_ret = new(__MR_decr_MR_VertId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_incr_MR_VertId(_Underlying *_this);
            MR.Mut_VertId _unused_ret = new(__MR_incr_MR_VertId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VertId::operator-=`.
        public unsafe MR.Mut_VertId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_VertId_sub_assign(_Underlying *_this, int a);
            return new(__MR_VertId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::VertId::operator+=`.
        public unsafe MR.Mut_VertId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_VertId_add_assign(_Underlying *_this, int a);
            return new(__MR_VertId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::VertId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::VertId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct VertId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator VertId(Const_VertId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_VertId(VertId other) => new(new Mut_VertId((Mut_VertId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public VertId(VertId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_DefaultConstruct();
            this = __MR_VertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe VertId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_VertId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe VertId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct_int", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct_int(int i);
            this = __MR_VertId_Construct_int(i);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::VertId::VertId`.
        public unsafe VertId(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.VertId __MR_VertId_Construct_uint64_t(ulong i);
            this = __MR_VertId_Construct_uint64_t(i);
        }

        /// Generated from conversion operator `MR::VertId::operator int`.
        public static unsafe implicit operator int(MR.VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_VertId_ConvertTo_int(MR.Const_VertId._Underlying *_this);
            return __MR_VertId_ConvertTo_int((MR.Mut_VertId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::VertId::operator bool`.
        public static unsafe explicit operator bool(MR.VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_VertId_ConvertTo_bool(MR.Const_VertId._Underlying *_this);
            return __MR_VertId_ConvertTo_bool((MR.Mut_VertId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::VertId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_valid", ExactSpelling = true)]
            extern static byte __MR_VertId_valid(MR.VertId *_this);
            fixed (MR.VertId *__ptr__this = &this)
            {
                return __MR_VertId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::VertId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_get", ExactSpelling = true)]
            extern static int *__MR_VertId_get(MR.VertId *_this);
            fixed (MR.VertId *__ptr__this = &this)
            {
                return ref *__MR_VertId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::VertId::operator==`.
        public static unsafe bool operator==(MR.VertId _this, MR.VertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VertId(MR.Const_VertId._Underlying *_this, MR.VertId b);
            return __MR_equal_MR_VertId((MR.Mut_VertId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.VertId _this, MR.VertId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::VertId::operator<`.
        public unsafe bool Less(MR.VertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_VertId", ExactSpelling = true)]
            extern static byte __MR_less_MR_VertId(MR.VertId *_this, MR.VertId b);
            fixed (MR.VertId *__ptr__this = &this)
            {
                return __MR_less_MR_VertId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::VertId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_decr_MR_VertId(MR.VertId *_this);
            fixed (MR.VertId *__ptr__this = &this)
            {
                MR.Mut_VertId _unused_ret = new(__MR_decr_MR_VertId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::VertId::operator--`.
        public unsafe VertId Decr(MR.VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_decr_MR_VertId(VertId *_this);
            VertId _this_copy = new(_this);
            MR.Mut_VertId _unused_ret = new(__MR_decr_MR_VertId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::VertId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_incr_MR_VertId(MR.VertId *_this);
            fixed (MR.VertId *__ptr__this = &this)
            {
                MR.Mut_VertId _unused_ret = new(__MR_incr_MR_VertId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::VertId::operator++`.
        public unsafe VertId Incr(MR.VertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VertId", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_incr_MR_VertId(VertId *_this);
            VertId _this_copy = new(_this);
            MR.Mut_VertId _unused_ret = new(__MR_incr_MR_VertId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::VertId::operator-=`.
        public unsafe MR.Mut_VertId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_VertId_sub_assign(MR.VertId *_this, int a);
            fixed (MR.VertId *__ptr__this = &this)
            {
                return new(__MR_VertId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::VertId::operator+=`.
        public unsafe MR.Mut_VertId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_VertId._Underlying *__MR_VertId_add_assign(MR.VertId *_this, int a);
            fixed (MR.VertId *__ptr__this = &this)
            {
                return new(__MR_VertId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.VertId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.VertId)
                return this == (MR.VertId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_VertId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_VertId`/`Const_VertId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_VertId
    {
        public readonly bool HasValue;
        internal readonly VertId Object;
        public VertId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_VertId() {HasValue = false;}
        public _InOpt_VertId(VertId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_VertId(VertId new_value) {return new(new_value);}
        public _InOpt_VertId(Const_VertId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_VertId(Const_VertId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_VertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_VertId`/`Const_VertId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `VertId`.
    public class _InOptMut_VertId
    {
        public Mut_VertId? Opt;

        public _InOptMut_VertId() {}
        public _InOptMut_VertId(Mut_VertId value) {Opt = value;}
        public static implicit operator _InOptMut_VertId(Mut_VertId value) {return new(value);}
        public unsafe _InOptMut_VertId(ref VertId value)
        {
            fixed (VertId *value_ptr = &value)
            {
                Opt = new((Const_VertId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_VertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_VertId`/`Const_VertId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `VertId`.
    public class _InOptConst_VertId
    {
        public Const_VertId? Opt;

        public _InOptConst_VertId() {}
        public _InOptConst_VertId(Const_VertId value) {Opt = value;}
        public static implicit operator _InOptConst_VertId(Const_VertId value) {return new(value);}
        public unsafe _InOptConst_VertId(ref readonly VertId value)
        {
            fixed (VertId *value_ptr = &value)
            {
                Opt = new((Const_VertId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::PixelId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::PixelId>`
    /// This is the const reference to the struct.
    public class Const_PixelId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.PixelId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly PixelId UnderlyingStruct => ref *(PixelId *)_UnderlyingPtr;

        internal unsafe Const_PixelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Destroy", ExactSpelling = true)]
            extern static void __MR_PixelId_Destroy(_Underlying *_this);
            __MR_PixelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PixelId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_PixelId(Const_PixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.PixelId _ctor_result = __MR_PixelId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::PixelId::PixelId`.
        public unsafe Const_PixelId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Construct", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.PixelId _ctor_result = __MR_PixelId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::PixelId::PixelId`.
        public unsafe Const_PixelId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Construct_int", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.PixelId _ctor_result = __MR_PixelId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::PixelId::operator int`.
        public static unsafe implicit operator int(MR.Const_PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_PixelId_ConvertTo_int(MR.Const_PixelId._Underlying *_this);
            return __MR_PixelId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::PixelId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PixelId_ConvertTo_bool(MR.Const_PixelId._Underlying *_this);
            return __MR_PixelId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PixelId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_valid", ExactSpelling = true)]
            extern static byte __MR_PixelId_valid(_Underlying *_this);
            return __MR_PixelId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PixelId::operator==`.
        public static unsafe bool operator==(MR.Const_PixelId _this, MR.PixelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_PixelId(MR.Const_PixelId._Underlying *_this, MR.PixelId b);
            return __MR_equal_MR_PixelId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_PixelId _this, MR.PixelId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::PixelId::operator<`.
        public unsafe bool Less(MR.PixelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_less_MR_PixelId(_Underlying *_this, MR.PixelId b);
            return __MR_less_MR_PixelId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::PixelId::operator--`.
        public static unsafe Mut_PixelId operator--(MR.Const_PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_decr_MR_PixelId(MR.Const_PixelId._Underlying *_this);
            Mut_PixelId _this_copy = new(_this);
            MR.Mut_PixelId _unused_ret = new(__MR_decr_MR_PixelId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::PixelId::operator++`.
        public static unsafe Mut_PixelId operator++(MR.Const_PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_incr_MR_PixelId(MR.Const_PixelId._Underlying *_this);
            Mut_PixelId _this_copy = new(_this);
            MR.Mut_PixelId _unused_ret = new(__MR_incr_MR_PixelId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.PixelId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.PixelId)
                return this == (MR.PixelId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::PixelId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::PixelId>`
    /// This is the non-const reference to the struct.
    public class Mut_PixelId : Const_PixelId
    {
        /// Get the underlying struct.
        public unsafe new ref PixelId UnderlyingStruct => ref *(PixelId *)_UnderlyingPtr;

        internal unsafe Mut_PixelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_PixelId(Const_PixelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_PixelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.PixelId _ctor_result = __MR_PixelId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::PixelId::PixelId`.
        public unsafe Mut_PixelId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Construct", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.PixelId _ctor_result = __MR_PixelId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::PixelId::PixelId`.
        public unsafe Mut_PixelId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Construct_int", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.PixelId _ctor_result = __MR_PixelId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::PixelId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_get", ExactSpelling = true)]
            extern static int *__MR_PixelId_get(_Underlying *_this);
            return ref *__MR_PixelId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::PixelId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_decr_MR_PixelId(_Underlying *_this);
            MR.Mut_PixelId _unused_ret = new(__MR_decr_MR_PixelId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_incr_MR_PixelId(_Underlying *_this);
            MR.Mut_PixelId _unused_ret = new(__MR_incr_MR_PixelId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PixelId::operator-=`.
        public unsafe MR.Mut_PixelId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_PixelId_sub_assign(_Underlying *_this, int a);
            return new(__MR_PixelId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::PixelId::operator+=`.
        public unsafe MR.Mut_PixelId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_PixelId_add_assign(_Underlying *_this, int a);
            return new(__MR_PixelId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::PixelId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::PixelId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct PixelId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator PixelId(Const_PixelId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_PixelId(PixelId other) => new(new Mut_PixelId((Mut_PixelId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public PixelId(PixelId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PixelId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_DefaultConstruct();
            this = __MR_PixelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::PixelId::PixelId`.
        public unsafe PixelId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Construct", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_PixelId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::PixelId::PixelId`.
        public unsafe PixelId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_Construct_int", ExactSpelling = true)]
            extern static MR.PixelId __MR_PixelId_Construct_int(int i);
            this = __MR_PixelId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::PixelId::operator int`.
        public static unsafe implicit operator int(MR.PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_PixelId_ConvertTo_int(MR.Const_PixelId._Underlying *_this);
            return __MR_PixelId_ConvertTo_int((MR.Mut_PixelId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::PixelId::operator bool`.
        public static unsafe explicit operator bool(MR.PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_PixelId_ConvertTo_bool(MR.Const_PixelId._Underlying *_this);
            return __MR_PixelId_ConvertTo_bool((MR.Mut_PixelId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::PixelId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_valid", ExactSpelling = true)]
            extern static byte __MR_PixelId_valid(MR.PixelId *_this);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                return __MR_PixelId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::PixelId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_get", ExactSpelling = true)]
            extern static int *__MR_PixelId_get(MR.PixelId *_this);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                return ref *__MR_PixelId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::PixelId::operator==`.
        public static unsafe bool operator==(MR.PixelId _this, MR.PixelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_PixelId(MR.Const_PixelId._Underlying *_this, MR.PixelId b);
            return __MR_equal_MR_PixelId((MR.Mut_PixelId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.PixelId _this, MR.PixelId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::PixelId::operator<`.
        public unsafe bool Less(MR.PixelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_PixelId", ExactSpelling = true)]
            extern static byte __MR_less_MR_PixelId(MR.PixelId *_this, MR.PixelId b);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                return __MR_less_MR_PixelId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::PixelId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_decr_MR_PixelId(MR.PixelId *_this);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                MR.Mut_PixelId _unused_ret = new(__MR_decr_MR_PixelId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::PixelId::operator--`.
        public unsafe PixelId Decr(MR.PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_decr_MR_PixelId(PixelId *_this);
            PixelId _this_copy = new(_this);
            MR.Mut_PixelId _unused_ret = new(__MR_decr_MR_PixelId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::PixelId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_incr_MR_PixelId(MR.PixelId *_this);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                MR.Mut_PixelId _unused_ret = new(__MR_incr_MR_PixelId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::PixelId::operator++`.
        public unsafe PixelId Incr(MR.PixelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_PixelId", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_incr_MR_PixelId(PixelId *_this);
            PixelId _this_copy = new(_this);
            MR.Mut_PixelId _unused_ret = new(__MR_incr_MR_PixelId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::PixelId::operator-=`.
        public unsafe MR.Mut_PixelId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_PixelId_sub_assign(MR.PixelId *_this, int a);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                return new(__MR_PixelId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::PixelId::operator+=`.
        public unsafe MR.Mut_PixelId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PixelId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_PixelId._Underlying *__MR_PixelId_add_assign(MR.PixelId *_this, int a);
            fixed (MR.PixelId *__ptr__this = &this)
            {
                return new(__MR_PixelId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.PixelId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.PixelId)
                return this == (MR.PixelId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_PixelId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_PixelId`/`Const_PixelId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_PixelId
    {
        public readonly bool HasValue;
        internal readonly PixelId Object;
        public PixelId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_PixelId() {HasValue = false;}
        public _InOpt_PixelId(PixelId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_PixelId(PixelId new_value) {return new(new_value);}
        public _InOpt_PixelId(Const_PixelId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_PixelId(Const_PixelId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_PixelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_PixelId`/`Const_PixelId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `PixelId`.
    public class _InOptMut_PixelId
    {
        public Mut_PixelId? Opt;

        public _InOptMut_PixelId() {}
        public _InOptMut_PixelId(Mut_PixelId value) {Opt = value;}
        public static implicit operator _InOptMut_PixelId(Mut_PixelId value) {return new(value);}
        public unsafe _InOptMut_PixelId(ref PixelId value)
        {
            fixed (PixelId *value_ptr = &value)
            {
                Opt = new((Const_PixelId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_PixelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PixelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_PixelId`/`Const_PixelId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `PixelId`.
    public class _InOptConst_PixelId
    {
        public Const_PixelId? Opt;

        public _InOptConst_PixelId() {}
        public _InOptConst_PixelId(Const_PixelId value) {Opt = value;}
        public static implicit operator _InOptConst_PixelId(Const_PixelId value) {return new(value);}
        public unsafe _InOptConst_PixelId(ref readonly PixelId value)
        {
            fixed (PixelId *value_ptr = &value)
            {
                Opt = new((Const_PixelId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::RegionId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::RegionId>`
    /// This is the const reference to the struct.
    public class Const_RegionId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.RegionId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly RegionId UnderlyingStruct => ref *(RegionId *)_UnderlyingPtr;

        internal unsafe Const_RegionId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Destroy", ExactSpelling = true)]
            extern static void __MR_RegionId_Destroy(_Underlying *_this);
            __MR_RegionId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RegionId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_RegionId(Const_RegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.RegionId _ctor_result = __MR_RegionId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::RegionId::RegionId`.
        public unsafe Const_RegionId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Construct", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.RegionId _ctor_result = __MR_RegionId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::RegionId::RegionId`.
        public unsafe Const_RegionId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Construct_int", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.RegionId _ctor_result = __MR_RegionId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::RegionId::operator int`.
        public static unsafe implicit operator int(MR.Const_RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_RegionId_ConvertTo_int(MR.Const_RegionId._Underlying *_this);
            return __MR_RegionId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::RegionId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_RegionId_ConvertTo_bool(MR.Const_RegionId._Underlying *_this);
            return __MR_RegionId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RegionId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_valid", ExactSpelling = true)]
            extern static byte __MR_RegionId_valid(_Underlying *_this);
            return __MR_RegionId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RegionId::operator==`.
        public static unsafe bool operator==(MR.Const_RegionId _this, MR.RegionId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_RegionId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_RegionId(MR.Const_RegionId._Underlying *_this, MR.RegionId b);
            return __MR_equal_MR_RegionId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_RegionId _this, MR.RegionId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::RegionId::operator<`.
        public unsafe bool Less(MR.RegionId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_RegionId", ExactSpelling = true)]
            extern static byte __MR_less_MR_RegionId(_Underlying *_this, MR.RegionId b);
            return __MR_less_MR_RegionId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::RegionId::operator--`.
        public static unsafe Mut_RegionId operator--(MR.Const_RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_decr_MR_RegionId(MR.Const_RegionId._Underlying *_this);
            Mut_RegionId _this_copy = new(_this);
            MR.Mut_RegionId _unused_ret = new(__MR_decr_MR_RegionId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::RegionId::operator++`.
        public static unsafe Mut_RegionId operator++(MR.Const_RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_incr_MR_RegionId(MR.Const_RegionId._Underlying *_this);
            Mut_RegionId _this_copy = new(_this);
            MR.Mut_RegionId _unused_ret = new(__MR_incr_MR_RegionId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.RegionId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.RegionId)
                return this == (MR.RegionId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::RegionId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::RegionId>`
    /// This is the non-const reference to the struct.
    public class Mut_RegionId : Const_RegionId
    {
        /// Get the underlying struct.
        public unsafe new ref RegionId UnderlyingStruct => ref *(RegionId *)_UnderlyingPtr;

        internal unsafe Mut_RegionId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_RegionId(Const_RegionId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_RegionId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.RegionId _ctor_result = __MR_RegionId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::RegionId::RegionId`.
        public unsafe Mut_RegionId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Construct", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.RegionId _ctor_result = __MR_RegionId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::RegionId::RegionId`.
        public unsafe Mut_RegionId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Construct_int", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.RegionId _ctor_result = __MR_RegionId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::RegionId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_get", ExactSpelling = true)]
            extern static int *__MR_RegionId_get(_Underlying *_this);
            return ref *__MR_RegionId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::RegionId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_decr_MR_RegionId(_Underlying *_this);
            MR.Mut_RegionId _unused_ret = new(__MR_decr_MR_RegionId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_incr_MR_RegionId(_Underlying *_this);
            MR.Mut_RegionId _unused_ret = new(__MR_incr_MR_RegionId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RegionId::operator-=`.
        public unsafe MR.Mut_RegionId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_RegionId_sub_assign(_Underlying *_this, int a);
            return new(__MR_RegionId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::RegionId::operator+=`.
        public unsafe MR.Mut_RegionId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_RegionId_add_assign(_Underlying *_this, int a);
            return new(__MR_RegionId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::RegionId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::RegionId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct RegionId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator RegionId(Const_RegionId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_RegionId(RegionId other) => new(new Mut_RegionId((Mut_RegionId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public RegionId(RegionId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe RegionId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_DefaultConstruct();
            this = __MR_RegionId_DefaultConstruct();
        }

        /// Generated from constructor `MR::RegionId::RegionId`.
        public unsafe RegionId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Construct", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_RegionId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::RegionId::RegionId`.
        public unsafe RegionId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_Construct_int", ExactSpelling = true)]
            extern static MR.RegionId __MR_RegionId_Construct_int(int i);
            this = __MR_RegionId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::RegionId::operator int`.
        public static unsafe implicit operator int(MR.RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_RegionId_ConvertTo_int(MR.Const_RegionId._Underlying *_this);
            return __MR_RegionId_ConvertTo_int((MR.Mut_RegionId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::RegionId::operator bool`.
        public static unsafe explicit operator bool(MR.RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_RegionId_ConvertTo_bool(MR.Const_RegionId._Underlying *_this);
            return __MR_RegionId_ConvertTo_bool((MR.Mut_RegionId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::RegionId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_valid", ExactSpelling = true)]
            extern static byte __MR_RegionId_valid(MR.RegionId *_this);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                return __MR_RegionId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::RegionId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_get", ExactSpelling = true)]
            extern static int *__MR_RegionId_get(MR.RegionId *_this);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                return ref *__MR_RegionId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::RegionId::operator==`.
        public static unsafe bool operator==(MR.RegionId _this, MR.RegionId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_RegionId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_RegionId(MR.Const_RegionId._Underlying *_this, MR.RegionId b);
            return __MR_equal_MR_RegionId((MR.Mut_RegionId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.RegionId _this, MR.RegionId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::RegionId::operator<`.
        public unsafe bool Less(MR.RegionId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_RegionId", ExactSpelling = true)]
            extern static byte __MR_less_MR_RegionId(MR.RegionId *_this, MR.RegionId b);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                return __MR_less_MR_RegionId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::RegionId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_decr_MR_RegionId(MR.RegionId *_this);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                MR.Mut_RegionId _unused_ret = new(__MR_decr_MR_RegionId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::RegionId::operator--`.
        public unsafe RegionId Decr(MR.RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_decr_MR_RegionId(RegionId *_this);
            RegionId _this_copy = new(_this);
            MR.Mut_RegionId _unused_ret = new(__MR_decr_MR_RegionId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::RegionId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_incr_MR_RegionId(MR.RegionId *_this);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                MR.Mut_RegionId _unused_ret = new(__MR_incr_MR_RegionId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::RegionId::operator++`.
        public unsafe RegionId Incr(MR.RegionId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RegionId", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_incr_MR_RegionId(RegionId *_this);
            RegionId _this_copy = new(_this);
            MR.Mut_RegionId _unused_ret = new(__MR_incr_MR_RegionId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::RegionId::operator-=`.
        public unsafe MR.Mut_RegionId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_RegionId_sub_assign(MR.RegionId *_this, int a);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                return new(__MR_RegionId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::RegionId::operator+=`.
        public unsafe MR.Mut_RegionId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RegionId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_RegionId._Underlying *__MR_RegionId_add_assign(MR.RegionId *_this, int a);
            fixed (MR.RegionId *__ptr__this = &this)
            {
                return new(__MR_RegionId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.RegionId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.RegionId)
                return this == (MR.RegionId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_RegionId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_RegionId`/`Const_RegionId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_RegionId
    {
        public readonly bool HasValue;
        internal readonly RegionId Object;
        public RegionId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_RegionId() {HasValue = false;}
        public _InOpt_RegionId(RegionId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_RegionId(RegionId new_value) {return new(new_value);}
        public _InOpt_RegionId(Const_RegionId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_RegionId(Const_RegionId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_RegionId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_RegionId`/`Const_RegionId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `RegionId`.
    public class _InOptMut_RegionId
    {
        public Mut_RegionId? Opt;

        public _InOptMut_RegionId() {}
        public _InOptMut_RegionId(Mut_RegionId value) {Opt = value;}
        public static implicit operator _InOptMut_RegionId(Mut_RegionId value) {return new(value);}
        public unsafe _InOptMut_RegionId(ref RegionId value)
        {
            fixed (RegionId *value_ptr = &value)
            {
                Opt = new((Const_RegionId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_RegionId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RegionId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_RegionId`/`Const_RegionId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `RegionId`.
    public class _InOptConst_RegionId
    {
        public Const_RegionId? Opt;

        public _InOptConst_RegionId() {}
        public _InOptConst_RegionId(Const_RegionId value) {Opt = value;}
        public static implicit operator _InOptConst_RegionId(Const_RegionId value) {return new(value);}
        public unsafe _InOptConst_RegionId(ref readonly RegionId value)
        {
            fixed (RegionId *value_ptr = &value)
            {
                Opt = new((Const_RegionId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::NodeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::NodeId>`
    ///     `MR::NoInitNodeId`
    /// This is the const reference to the struct.
    public class Const_NodeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.NodeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly NodeId UnderlyingStruct => ref *(NodeId *)_UnderlyingPtr;

        internal unsafe Const_NodeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NodeId_Destroy(_Underlying *_this);
            __MR_NodeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NodeId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_NodeId(Const_NodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.NodeId _ctor_result = __MR_NodeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::NodeId::NodeId`.
        public unsafe Const_NodeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Construct", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.NodeId _ctor_result = __MR_NodeId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::NodeId::NodeId`.
        public unsafe Const_NodeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Construct_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.NodeId _ctor_result = __MR_NodeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::NodeId::operator int`.
        public static unsafe implicit operator int(MR.Const_NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NodeId_ConvertTo_int(MR.Const_NodeId._Underlying *_this);
            return __MR_NodeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NodeId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NodeId_ConvertTo_bool(MR.Const_NodeId._Underlying *_this);
            return __MR_NodeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NodeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_valid", ExactSpelling = true)]
            extern static byte __MR_NodeId_valid(_Underlying *_this);
            return __MR_NodeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NodeId::operator==`.
        public static unsafe bool operator==(MR.Const_NodeId _this, MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NodeId(MR.Const_NodeId._Underlying *_this, MR.NodeId b);
            return __MR_equal_MR_NodeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NodeId _this, MR.NodeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NodeId::operator<`.
        public unsafe bool Less(MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NodeId(_Underlying *_this, MR.NodeId b);
            return __MR_less_MR_NodeId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::NodeId::operator--`.
        public static unsafe Mut_NodeId operator--(MR.Const_NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_decr_MR_NodeId(MR.Const_NodeId._Underlying *_this);
            Mut_NodeId _this_copy = new(_this);
            MR.Mut_NodeId _unused_ret = new(__MR_decr_MR_NodeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::NodeId::operator++`.
        public static unsafe Mut_NodeId operator++(MR.Const_NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_incr_MR_NodeId(MR.Const_NodeId._Underlying *_this);
            Mut_NodeId _this_copy = new(_this);
            MR.Mut_NodeId _unused_ret = new(__MR_incr_MR_NodeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from function `MR::operator+<MR::NodeTag>`.
        public static unsafe MR.NodeId operator+(Const_NodeId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NodeId_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_add_MR_NodeId_int(MR.NodeId id, int a);
            return __MR_add_MR_NodeId_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator+<MR::NodeTag>`.
        public static unsafe MR.NodeId operator+(Const_NodeId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NodeId_unsigned_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_add_MR_NodeId_unsigned_int(MR.NodeId id, uint a);
            return __MR_add_MR_NodeId_unsigned_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator+<MR::NodeTag>`.
        public static unsafe MR.NodeId operator+(Const_NodeId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NodeId_uint64_t", ExactSpelling = true)]
            extern static MR.NodeId __MR_add_MR_NodeId_uint64_t(MR.NodeId id, ulong a);
            return __MR_add_MR_NodeId_uint64_t(id.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.NodeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.NodeId)
                return this == (MR.NodeId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::NodeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::NodeId>`
    ///     `MR::NoInitNodeId`
    /// This is the non-const reference to the struct.
    public class Mut_NodeId : Const_NodeId
    {
        /// Get the underlying struct.
        public unsafe new ref NodeId UnderlyingStruct => ref *(NodeId *)_UnderlyingPtr;

        internal unsafe Mut_NodeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_NodeId(Const_NodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_NodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.NodeId _ctor_result = __MR_NodeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::NodeId::NodeId`.
        public unsafe Mut_NodeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Construct", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.NodeId _ctor_result = __MR_NodeId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::NodeId::NodeId`.
        public unsafe Mut_NodeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Construct_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.NodeId _ctor_result = __MR_NodeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::NodeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_get", ExactSpelling = true)]
            extern static int *__MR_NodeId_get(_Underlying *_this);
            return ref *__MR_NodeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NodeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_decr_MR_NodeId(_Underlying *_this);
            MR.Mut_NodeId _unused_ret = new(__MR_decr_MR_NodeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_incr_MR_NodeId(_Underlying *_this);
            MR.Mut_NodeId _unused_ret = new(__MR_incr_MR_NodeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NodeId::operator-=`.
        public unsafe MR.Mut_NodeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NodeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NodeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NodeId::operator+=`.
        public unsafe MR.Mut_NodeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NodeId_add_assign(_Underlying *_this, int a);
            return new(__MR_NodeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::NodeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::NodeId>`
    ///     `MR::NoInitNodeId`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct NodeId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator NodeId(Const_NodeId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_NodeId(NodeId other) => new(new Mut_NodeId((Mut_NodeId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public NodeId(NodeId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_DefaultConstruct();
            this = __MR_NodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NodeId::NodeId`.
        public unsafe NodeId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Construct", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_NodeId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::NodeId::NodeId`.
        public unsafe NodeId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_Construct_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_NodeId_Construct_int(int i);
            this = __MR_NodeId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::NodeId::operator int`.
        public static unsafe implicit operator int(MR.NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NodeId_ConvertTo_int(MR.Const_NodeId._Underlying *_this);
            return __MR_NodeId_ConvertTo_int((MR.Mut_NodeId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::NodeId::operator bool`.
        public static unsafe explicit operator bool(MR.NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NodeId_ConvertTo_bool(MR.Const_NodeId._Underlying *_this);
            return __MR_NodeId_ConvertTo_bool((MR.Mut_NodeId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::NodeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_valid", ExactSpelling = true)]
            extern static byte __MR_NodeId_valid(MR.NodeId *_this);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                return __MR_NodeId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::NodeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_get", ExactSpelling = true)]
            extern static int *__MR_NodeId_get(MR.NodeId *_this);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                return ref *__MR_NodeId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::NodeId::operator==`.
        public static unsafe bool operator==(MR.NodeId _this, MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NodeId(MR.Const_NodeId._Underlying *_this, MR.NodeId b);
            return __MR_equal_MR_NodeId((MR.Mut_NodeId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.NodeId _this, MR.NodeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NodeId::operator<`.
        public unsafe bool Less(MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NodeId(MR.NodeId *_this, MR.NodeId b);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                return __MR_less_MR_NodeId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::NodeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_decr_MR_NodeId(MR.NodeId *_this);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                MR.Mut_NodeId _unused_ret = new(__MR_decr_MR_NodeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::NodeId::operator--`.
        public unsafe NodeId Decr(MR.NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_decr_MR_NodeId(NodeId *_this);
            NodeId _this_copy = new(_this);
            MR.Mut_NodeId _unused_ret = new(__MR_decr_MR_NodeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::NodeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_incr_MR_NodeId(MR.NodeId *_this);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                MR.Mut_NodeId _unused_ret = new(__MR_incr_MR_NodeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::NodeId::operator++`.
        public unsafe NodeId Incr(MR.NodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_incr_MR_NodeId(NodeId *_this);
            NodeId _this_copy = new(_this);
            MR.Mut_NodeId _unused_ret = new(__MR_incr_MR_NodeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::NodeId::operator-=`.
        public unsafe MR.Mut_NodeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NodeId_sub_assign(MR.NodeId *_this, int a);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                return new(__MR_NodeId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::NodeId::operator+=`.
        public unsafe MR.Mut_NodeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NodeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NodeId_add_assign(MR.NodeId *_this, int a);
            fixed (MR.NodeId *__ptr__this = &this)
            {
                return new(__MR_NodeId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from function `MR::operator+<MR::NodeTag>`.
        public static MR.NodeId operator+(MR.NodeId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NodeId_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_add_MR_NodeId_int(MR.NodeId id, int a);
            return __MR_add_MR_NodeId_int(id, a);
        }

        /// Generated from function `MR::operator+<MR::NodeTag>`.
        public static MR.NodeId operator+(MR.NodeId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NodeId_unsigned_int", ExactSpelling = true)]
            extern static MR.NodeId __MR_add_MR_NodeId_unsigned_int(MR.NodeId id, uint a);
            return __MR_add_MR_NodeId_unsigned_int(id, a);
        }

        /// Generated from function `MR::operator+<MR::NodeTag>`.
        public static MR.NodeId operator+(MR.NodeId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_NodeId_uint64_t", ExactSpelling = true)]
            extern static MR.NodeId __MR_add_MR_NodeId_uint64_t(MR.NodeId id, ulong a);
            return __MR_add_MR_NodeId_uint64_t(id, a);
        }

        // IEquatable:

        public bool Equals(MR.NodeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.NodeId)
                return this == (MR.NodeId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_NodeId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_NodeId`/`Const_NodeId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_NodeId
    {
        public readonly bool HasValue;
        internal readonly NodeId Object;
        public NodeId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_NodeId() {HasValue = false;}
        public _InOpt_NodeId(NodeId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_NodeId(NodeId new_value) {return new(new_value);}
        public _InOpt_NodeId(Const_NodeId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_NodeId(Const_NodeId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_NodeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_NodeId`/`Const_NodeId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `NodeId`.
    public class _InOptMut_NodeId
    {
        public Mut_NodeId? Opt;

        public _InOptMut_NodeId() {}
        public _InOptMut_NodeId(Mut_NodeId value) {Opt = value;}
        public static implicit operator _InOptMut_NodeId(Mut_NodeId value) {return new(value);}
        public unsafe _InOptMut_NodeId(ref NodeId value)
        {
            fixed (NodeId *value_ptr = &value)
            {
                Opt = new((Const_NodeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_NodeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_NodeId`/`Const_NodeId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `NodeId`.
    public class _InOptConst_NodeId
    {
        public Const_NodeId? Opt;

        public _InOptConst_NodeId() {}
        public _InOptConst_NodeId(Const_NodeId value) {Opt = value;}
        public static implicit operator _InOptConst_NodeId(Const_NodeId value) {return new(value);}
        public unsafe _InOptConst_NodeId(ref readonly NodeId value)
        {
            fixed (NodeId *value_ptr = &value)
            {
                Opt = new((Const_NodeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::ObjId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::ObjId>`
    /// This is the const reference to the struct.
    public class Const_ObjId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.ObjId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly ObjId UnderlyingStruct => ref *(ObjId *)_UnderlyingPtr;

        internal unsafe Const_ObjId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjId_Destroy(_Underlying *_this);
            __MR_ObjId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_ObjId(Const_ObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ObjId _ctor_result = __MR_ObjId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::ObjId::ObjId`.
        public unsafe Const_ObjId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Construct", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ObjId _ctor_result = __MR_ObjId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::ObjId::ObjId`.
        public unsafe Const_ObjId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Construct_int", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ObjId _ctor_result = __MR_ObjId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::ObjId::operator int`.
        public static unsafe implicit operator int(MR.Const_ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_ObjId_ConvertTo_int(MR.Const_ObjId._Underlying *_this);
            return __MR_ObjId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::ObjId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_ObjId_ConvertTo_bool(MR.Const_ObjId._Underlying *_this);
            return __MR_ObjId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_valid", ExactSpelling = true)]
            extern static byte __MR_ObjId_valid(_Underlying *_this);
            return __MR_ObjId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjId::operator==`.
        public static unsafe bool operator==(MR.Const_ObjId _this, MR.ObjId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ObjId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ObjId(MR.Const_ObjId._Underlying *_this, MR.ObjId b);
            return __MR_equal_MR_ObjId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ObjId _this, MR.ObjId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::ObjId::operator<`.
        public unsafe bool Less(MR.ObjId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_ObjId", ExactSpelling = true)]
            extern static byte __MR_less_MR_ObjId(_Underlying *_this, MR.ObjId b);
            return __MR_less_MR_ObjId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::ObjId::operator--`.
        public static unsafe Mut_ObjId operator--(MR.Const_ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_decr_MR_ObjId(MR.Const_ObjId._Underlying *_this);
            Mut_ObjId _this_copy = new(_this);
            MR.Mut_ObjId _unused_ret = new(__MR_decr_MR_ObjId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::ObjId::operator++`.
        public static unsafe Mut_ObjId operator++(MR.Const_ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_incr_MR_ObjId(MR.Const_ObjId._Underlying *_this);
            Mut_ObjId _this_copy = new(_this);
            MR.Mut_ObjId _unused_ret = new(__MR_incr_MR_ObjId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.ObjId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.ObjId)
                return this == (MR.ObjId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::ObjId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::ObjId>`
    /// This is the non-const reference to the struct.
    public class Mut_ObjId : Const_ObjId
    {
        /// Get the underlying struct.
        public unsafe new ref ObjId UnderlyingStruct => ref *(ObjId *)_UnderlyingPtr;

        internal unsafe Mut_ObjId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_ObjId(Const_ObjId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_ObjId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ObjId _ctor_result = __MR_ObjId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::ObjId::ObjId`.
        public unsafe Mut_ObjId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Construct", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ObjId _ctor_result = __MR_ObjId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::ObjId::ObjId`.
        public unsafe Mut_ObjId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Construct_int", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ObjId _ctor_result = __MR_ObjId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::ObjId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_get", ExactSpelling = true)]
            extern static int *__MR_ObjId_get(_Underlying *_this);
            return ref *__MR_ObjId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_decr_MR_ObjId(_Underlying *_this);
            MR.Mut_ObjId _unused_ret = new(__MR_decr_MR_ObjId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_incr_MR_ObjId(_Underlying *_this);
            MR.Mut_ObjId _unused_ret = new(__MR_incr_MR_ObjId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjId::operator-=`.
        public unsafe MR.Mut_ObjId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_ObjId_sub_assign(_Underlying *_this, int a);
            return new(__MR_ObjId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::ObjId::operator+=`.
        public unsafe MR.Mut_ObjId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_ObjId_add_assign(_Underlying *_this, int a);
            return new(__MR_ObjId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::ObjId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::ObjId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct ObjId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator ObjId(Const_ObjId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_ObjId(ObjId other) => new(new Mut_ObjId((Mut_ObjId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public ObjId(ObjId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_DefaultConstruct();
            this = __MR_ObjId_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjId::ObjId`.
        public unsafe ObjId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Construct", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_ObjId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::ObjId::ObjId`.
        public unsafe ObjId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_Construct_int", ExactSpelling = true)]
            extern static MR.ObjId __MR_ObjId_Construct_int(int i);
            this = __MR_ObjId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::ObjId::operator int`.
        public static unsafe implicit operator int(MR.ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_ObjId_ConvertTo_int(MR.Const_ObjId._Underlying *_this);
            return __MR_ObjId_ConvertTo_int((MR.Mut_ObjId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::ObjId::operator bool`.
        public static unsafe explicit operator bool(MR.ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_ObjId_ConvertTo_bool(MR.Const_ObjId._Underlying *_this);
            return __MR_ObjId_ConvertTo_bool((MR.Mut_ObjId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::ObjId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_valid", ExactSpelling = true)]
            extern static byte __MR_ObjId_valid(MR.ObjId *_this);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                return __MR_ObjId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::ObjId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_get", ExactSpelling = true)]
            extern static int *__MR_ObjId_get(MR.ObjId *_this);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                return ref *__MR_ObjId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::ObjId::operator==`.
        public static unsafe bool operator==(MR.ObjId _this, MR.ObjId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ObjId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ObjId(MR.Const_ObjId._Underlying *_this, MR.ObjId b);
            return __MR_equal_MR_ObjId((MR.Mut_ObjId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.ObjId _this, MR.ObjId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::ObjId::operator<`.
        public unsafe bool Less(MR.ObjId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_ObjId", ExactSpelling = true)]
            extern static byte __MR_less_MR_ObjId(MR.ObjId *_this, MR.ObjId b);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                return __MR_less_MR_ObjId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::ObjId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_decr_MR_ObjId(MR.ObjId *_this);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                MR.Mut_ObjId _unused_ret = new(__MR_decr_MR_ObjId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::ObjId::operator--`.
        public unsafe ObjId Decr(MR.ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_decr_MR_ObjId(ObjId *_this);
            ObjId _this_copy = new(_this);
            MR.Mut_ObjId _unused_ret = new(__MR_decr_MR_ObjId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::ObjId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_incr_MR_ObjId(MR.ObjId *_this);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                MR.Mut_ObjId _unused_ret = new(__MR_incr_MR_ObjId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::ObjId::operator++`.
        public unsafe ObjId Incr(MR.ObjId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ObjId", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_incr_MR_ObjId(ObjId *_this);
            ObjId _this_copy = new(_this);
            MR.Mut_ObjId _unused_ret = new(__MR_incr_MR_ObjId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::ObjId::operator-=`.
        public unsafe MR.Mut_ObjId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_ObjId_sub_assign(MR.ObjId *_this, int a);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                return new(__MR_ObjId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::ObjId::operator+=`.
        public unsafe MR.Mut_ObjId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_ObjId._Underlying *__MR_ObjId_add_assign(MR.ObjId *_this, int a);
            fixed (MR.ObjId *__ptr__this = &this)
            {
                return new(__MR_ObjId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.ObjId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.ObjId)
                return this == (MR.ObjId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_ObjId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_ObjId`/`Const_ObjId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_ObjId
    {
        public readonly bool HasValue;
        internal readonly ObjId Object;
        public ObjId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_ObjId() {HasValue = false;}
        public _InOpt_ObjId(ObjId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_ObjId(ObjId new_value) {return new(new_value);}
        public _InOpt_ObjId(Const_ObjId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_ObjId(Const_ObjId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_ObjId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_ObjId`/`Const_ObjId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `ObjId`.
    public class _InOptMut_ObjId
    {
        public Mut_ObjId? Opt;

        public _InOptMut_ObjId() {}
        public _InOptMut_ObjId(Mut_ObjId value) {Opt = value;}
        public static implicit operator _InOptMut_ObjId(Mut_ObjId value) {return new(value);}
        public unsafe _InOptMut_ObjId(ref ObjId value)
        {
            fixed (ObjId *value_ptr = &value)
            {
                Opt = new((Const_ObjId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_ObjId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_ObjId`/`Const_ObjId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `ObjId`.
    public class _InOptConst_ObjId
    {
        public Const_ObjId? Opt;

        public _InOptConst_ObjId() {}
        public _InOptConst_ObjId(Const_ObjId value) {Opt = value;}
        public static implicit operator _InOptConst_ObjId(Const_ObjId value) {return new(value);}
        public unsafe _InOptConst_ObjId(ref readonly ObjId value)
        {
            fixed (ObjId *value_ptr = &value)
            {
                Opt = new((Const_ObjId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::TextureId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::TextureId>`
    /// This is the const reference to the struct.
    public class Const_TextureId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.TextureId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly TextureId UnderlyingStruct => ref *(TextureId *)_UnderlyingPtr;

        internal unsafe Const_TextureId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Destroy", ExactSpelling = true)]
            extern static void __MR_TextureId_Destroy(_Underlying *_this);
            __MR_TextureId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TextureId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_TextureId(Const_TextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.TextureId _ctor_result = __MR_TextureId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::TextureId::TextureId`.
        public unsafe Const_TextureId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Construct", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.TextureId _ctor_result = __MR_TextureId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::TextureId::TextureId`.
        public unsafe Const_TextureId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Construct_int", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.TextureId _ctor_result = __MR_TextureId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::TextureId::operator int`.
        public static unsafe implicit operator int(MR.Const_TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_TextureId_ConvertTo_int(MR.Const_TextureId._Underlying *_this);
            return __MR_TextureId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::TextureId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_TextureId_ConvertTo_bool(MR.Const_TextureId._Underlying *_this);
            return __MR_TextureId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TextureId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_valid", ExactSpelling = true)]
            extern static byte __MR_TextureId_valid(_Underlying *_this);
            return __MR_TextureId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TextureId::operator==`.
        public static unsafe bool operator==(MR.Const_TextureId _this, MR.TextureId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_TextureId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_TextureId(MR.Const_TextureId._Underlying *_this, MR.TextureId b);
            return __MR_equal_MR_TextureId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_TextureId _this, MR.TextureId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::TextureId::operator<`.
        public unsafe bool Less(MR.TextureId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_TextureId", ExactSpelling = true)]
            extern static byte __MR_less_MR_TextureId(_Underlying *_this, MR.TextureId b);
            return __MR_less_MR_TextureId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::TextureId::operator--`.
        public static unsafe Mut_TextureId operator--(MR.Const_TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_decr_MR_TextureId(MR.Const_TextureId._Underlying *_this);
            Mut_TextureId _this_copy = new(_this);
            MR.Mut_TextureId _unused_ret = new(__MR_decr_MR_TextureId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::TextureId::operator++`.
        public static unsafe Mut_TextureId operator++(MR.Const_TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_incr_MR_TextureId(MR.Const_TextureId._Underlying *_this);
            Mut_TextureId _this_copy = new(_this);
            MR.Mut_TextureId _unused_ret = new(__MR_incr_MR_TextureId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.TextureId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.TextureId)
                return this == (MR.TextureId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::TextureId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::TextureId>`
    /// This is the non-const reference to the struct.
    public class Mut_TextureId : Const_TextureId
    {
        /// Get the underlying struct.
        public unsafe new ref TextureId UnderlyingStruct => ref *(TextureId *)_UnderlyingPtr;

        internal unsafe Mut_TextureId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_TextureId(Const_TextureId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_TextureId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.TextureId _ctor_result = __MR_TextureId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::TextureId::TextureId`.
        public unsafe Mut_TextureId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Construct", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.TextureId _ctor_result = __MR_TextureId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::TextureId::TextureId`.
        public unsafe Mut_TextureId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Construct_int", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.TextureId _ctor_result = __MR_TextureId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::TextureId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_get", ExactSpelling = true)]
            extern static int *__MR_TextureId_get(_Underlying *_this);
            return ref *__MR_TextureId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::TextureId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_decr_MR_TextureId(_Underlying *_this);
            MR.Mut_TextureId _unused_ret = new(__MR_decr_MR_TextureId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_incr_MR_TextureId(_Underlying *_this);
            MR.Mut_TextureId _unused_ret = new(__MR_incr_MR_TextureId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::TextureId::operator-=`.
        public unsafe MR.Mut_TextureId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_TextureId_sub_assign(_Underlying *_this, int a);
            return new(__MR_TextureId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::TextureId::operator+=`.
        public unsafe MR.Mut_TextureId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_TextureId_add_assign(_Underlying *_this, int a);
            return new(__MR_TextureId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::TextureId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::TextureId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct TextureId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator TextureId(Const_TextureId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_TextureId(TextureId other) => new(new Mut_TextureId((Mut_TextureId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public TextureId(TextureId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe TextureId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_DefaultConstruct();
            this = __MR_TextureId_DefaultConstruct();
        }

        /// Generated from constructor `MR::TextureId::TextureId`.
        public unsafe TextureId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Construct", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_TextureId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::TextureId::TextureId`.
        public unsafe TextureId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_Construct_int", ExactSpelling = true)]
            extern static MR.TextureId __MR_TextureId_Construct_int(int i);
            this = __MR_TextureId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::TextureId::operator int`.
        public static unsafe implicit operator int(MR.TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_TextureId_ConvertTo_int(MR.Const_TextureId._Underlying *_this);
            return __MR_TextureId_ConvertTo_int((MR.Mut_TextureId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::TextureId::operator bool`.
        public static unsafe explicit operator bool(MR.TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_TextureId_ConvertTo_bool(MR.Const_TextureId._Underlying *_this);
            return __MR_TextureId_ConvertTo_bool((MR.Mut_TextureId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::TextureId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_valid", ExactSpelling = true)]
            extern static byte __MR_TextureId_valid(MR.TextureId *_this);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                return __MR_TextureId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::TextureId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_get", ExactSpelling = true)]
            extern static int *__MR_TextureId_get(MR.TextureId *_this);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                return ref *__MR_TextureId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::TextureId::operator==`.
        public static unsafe bool operator==(MR.TextureId _this, MR.TextureId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_TextureId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_TextureId(MR.Const_TextureId._Underlying *_this, MR.TextureId b);
            return __MR_equal_MR_TextureId((MR.Mut_TextureId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.TextureId _this, MR.TextureId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::TextureId::operator<`.
        public unsafe bool Less(MR.TextureId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_TextureId", ExactSpelling = true)]
            extern static byte __MR_less_MR_TextureId(MR.TextureId *_this, MR.TextureId b);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                return __MR_less_MR_TextureId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::TextureId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_decr_MR_TextureId(MR.TextureId *_this);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                MR.Mut_TextureId _unused_ret = new(__MR_decr_MR_TextureId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::TextureId::operator--`.
        public unsafe TextureId Decr(MR.TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_decr_MR_TextureId(TextureId *_this);
            TextureId _this_copy = new(_this);
            MR.Mut_TextureId _unused_ret = new(__MR_decr_MR_TextureId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::TextureId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_incr_MR_TextureId(MR.TextureId *_this);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                MR.Mut_TextureId _unused_ret = new(__MR_incr_MR_TextureId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::TextureId::operator++`.
        public unsafe TextureId Incr(MR.TextureId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_TextureId", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_incr_MR_TextureId(TextureId *_this);
            TextureId _this_copy = new(_this);
            MR.Mut_TextureId _unused_ret = new(__MR_incr_MR_TextureId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::TextureId::operator-=`.
        public unsafe MR.Mut_TextureId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_TextureId_sub_assign(MR.TextureId *_this, int a);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                return new(__MR_TextureId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::TextureId::operator+=`.
        public unsafe MR.Mut_TextureId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextureId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_TextureId._Underlying *__MR_TextureId_add_assign(MR.TextureId *_this, int a);
            fixed (MR.TextureId *__ptr__this = &this)
            {
                return new(__MR_TextureId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.TextureId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.TextureId)
                return this == (MR.TextureId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_TextureId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_TextureId`/`Const_TextureId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_TextureId
    {
        public readonly bool HasValue;
        internal readonly TextureId Object;
        public TextureId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_TextureId() {HasValue = false;}
        public _InOpt_TextureId(TextureId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_TextureId(TextureId new_value) {return new(new_value);}
        public _InOpt_TextureId(Const_TextureId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_TextureId(Const_TextureId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_TextureId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_TextureId`/`Const_TextureId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `TextureId`.
    public class _InOptMut_TextureId
    {
        public Mut_TextureId? Opt;

        public _InOptMut_TextureId() {}
        public _InOptMut_TextureId(Mut_TextureId value) {Opt = value;}
        public static implicit operator _InOptMut_TextureId(Mut_TextureId value) {return new(value);}
        public unsafe _InOptMut_TextureId(ref TextureId value)
        {
            fixed (TextureId *value_ptr = &value)
            {
                Opt = new((Const_TextureId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_TextureId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TextureId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_TextureId`/`Const_TextureId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `TextureId`.
    public class _InOptConst_TextureId
    {
        public Const_TextureId? Opt;

        public _InOptConst_TextureId() {}
        public _InOptConst_TextureId(Const_TextureId value) {Opt = value;}
        public static implicit operator _InOptConst_TextureId(Const_TextureId value) {return new(value);}
        public unsafe _InOptConst_TextureId(ref readonly TextureId value)
        {
            fixed (TextureId *value_ptr = &value)
            {
                Opt = new((Const_TextureId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::GraphVertId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::GraphVertId>`
    /// This is the const reference to the struct.
    public class Const_GraphVertId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.GraphVertId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly GraphVertId UnderlyingStruct => ref *(GraphVertId *)_UnderlyingPtr;

        internal unsafe Const_GraphVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Destroy", ExactSpelling = true)]
            extern static void __MR_GraphVertId_Destroy(_Underlying *_this);
            __MR_GraphVertId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GraphVertId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_GraphVertId(Const_GraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphVertId _ctor_result = __MR_GraphVertId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::GraphVertId::GraphVertId`.
        public unsafe Const_GraphVertId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Construct", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphVertId _ctor_result = __MR_GraphVertId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::GraphVertId::GraphVertId`.
        public unsafe Const_GraphVertId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Construct_int", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphVertId _ctor_result = __MR_GraphVertId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::GraphVertId::operator int`.
        public static unsafe implicit operator int(MR.Const_GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_GraphVertId_ConvertTo_int(MR.Const_GraphVertId._Underlying *_this);
            return __MR_GraphVertId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::GraphVertId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_GraphVertId_ConvertTo_bool(MR.Const_GraphVertId._Underlying *_this);
            return __MR_GraphVertId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphVertId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_valid", ExactSpelling = true)]
            extern static byte __MR_GraphVertId_valid(_Underlying *_this);
            return __MR_GraphVertId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphVertId::operator==`.
        public static unsafe bool operator==(MR.Const_GraphVertId _this, MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_GraphVertId(MR.Const_GraphVertId._Underlying *_this, MR.GraphVertId b);
            return __MR_equal_MR_GraphVertId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_GraphVertId _this, MR.GraphVertId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::GraphVertId::operator<`.
        public unsafe bool Less(MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_less_MR_GraphVertId(_Underlying *_this, MR.GraphVertId b);
            return __MR_less_MR_GraphVertId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::GraphVertId::operator--`.
        public static unsafe Mut_GraphVertId operator--(MR.Const_GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_decr_MR_GraphVertId(MR.Const_GraphVertId._Underlying *_this);
            Mut_GraphVertId _this_copy = new(_this);
            MR.Mut_GraphVertId _unused_ret = new(__MR_decr_MR_GraphVertId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::GraphVertId::operator++`.
        public static unsafe Mut_GraphVertId operator++(MR.Const_GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_incr_MR_GraphVertId(MR.Const_GraphVertId._Underlying *_this);
            Mut_GraphVertId _this_copy = new(_this);
            MR.Mut_GraphVertId _unused_ret = new(__MR_incr_MR_GraphVertId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.GraphVertId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.GraphVertId)
                return this == (MR.GraphVertId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::GraphVertId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::GraphVertId>`
    /// This is the non-const reference to the struct.
    public class Mut_GraphVertId : Const_GraphVertId
    {
        /// Get the underlying struct.
        public unsafe new ref GraphVertId UnderlyingStruct => ref *(GraphVertId *)_UnderlyingPtr;

        internal unsafe Mut_GraphVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_GraphVertId(Const_GraphVertId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_GraphVertId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphVertId _ctor_result = __MR_GraphVertId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::GraphVertId::GraphVertId`.
        public unsafe Mut_GraphVertId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Construct", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphVertId _ctor_result = __MR_GraphVertId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::GraphVertId::GraphVertId`.
        public unsafe Mut_GraphVertId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Construct_int", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphVertId _ctor_result = __MR_GraphVertId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::GraphVertId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_get", ExactSpelling = true)]
            extern static int *__MR_GraphVertId_get(_Underlying *_this);
            return ref *__MR_GraphVertId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphVertId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_decr_MR_GraphVertId(_Underlying *_this);
            MR.Mut_GraphVertId _unused_ret = new(__MR_decr_MR_GraphVertId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_incr_MR_GraphVertId(_Underlying *_this);
            MR.Mut_GraphVertId _unused_ret = new(__MR_incr_MR_GraphVertId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphVertId::operator-=`.
        public unsafe MR.Mut_GraphVertId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_GraphVertId_sub_assign(_Underlying *_this, int a);
            return new(__MR_GraphVertId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::GraphVertId::operator+=`.
        public unsafe MR.Mut_GraphVertId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_GraphVertId_add_assign(_Underlying *_this, int a);
            return new(__MR_GraphVertId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::GraphVertId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::GraphVertId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct GraphVertId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator GraphVertId(Const_GraphVertId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_GraphVertId(GraphVertId other) => new(new Mut_GraphVertId((Mut_GraphVertId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public GraphVertId(GraphVertId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe GraphVertId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_DefaultConstruct();
            this = __MR_GraphVertId_DefaultConstruct();
        }

        /// Generated from constructor `MR::GraphVertId::GraphVertId`.
        public unsafe GraphVertId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Construct", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_GraphVertId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::GraphVertId::GraphVertId`.
        public unsafe GraphVertId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_Construct_int", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_GraphVertId_Construct_int(int i);
            this = __MR_GraphVertId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::GraphVertId::operator int`.
        public static unsafe implicit operator int(MR.GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_GraphVertId_ConvertTo_int(MR.Const_GraphVertId._Underlying *_this);
            return __MR_GraphVertId_ConvertTo_int((MR.Mut_GraphVertId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::GraphVertId::operator bool`.
        public static unsafe explicit operator bool(MR.GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_GraphVertId_ConvertTo_bool(MR.Const_GraphVertId._Underlying *_this);
            return __MR_GraphVertId_ConvertTo_bool((MR.Mut_GraphVertId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::GraphVertId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_valid", ExactSpelling = true)]
            extern static byte __MR_GraphVertId_valid(MR.GraphVertId *_this);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                return __MR_GraphVertId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::GraphVertId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_get", ExactSpelling = true)]
            extern static int *__MR_GraphVertId_get(MR.GraphVertId *_this);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                return ref *__MR_GraphVertId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::GraphVertId::operator==`.
        public static unsafe bool operator==(MR.GraphVertId _this, MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_GraphVertId(MR.Const_GraphVertId._Underlying *_this, MR.GraphVertId b);
            return __MR_equal_MR_GraphVertId((MR.Mut_GraphVertId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.GraphVertId _this, MR.GraphVertId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::GraphVertId::operator<`.
        public unsafe bool Less(MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_less_MR_GraphVertId(MR.GraphVertId *_this, MR.GraphVertId b);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                return __MR_less_MR_GraphVertId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::GraphVertId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_decr_MR_GraphVertId(MR.GraphVertId *_this);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                MR.Mut_GraphVertId _unused_ret = new(__MR_decr_MR_GraphVertId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::GraphVertId::operator--`.
        public unsafe GraphVertId Decr(MR.GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_decr_MR_GraphVertId(GraphVertId *_this);
            GraphVertId _this_copy = new(_this);
            MR.Mut_GraphVertId _unused_ret = new(__MR_decr_MR_GraphVertId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::GraphVertId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_incr_MR_GraphVertId(MR.GraphVertId *_this);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                MR.Mut_GraphVertId _unused_ret = new(__MR_incr_MR_GraphVertId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::GraphVertId::operator++`.
        public unsafe GraphVertId Incr(MR.GraphVertId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphVertId", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_incr_MR_GraphVertId(GraphVertId *_this);
            GraphVertId _this_copy = new(_this);
            MR.Mut_GraphVertId _unused_ret = new(__MR_incr_MR_GraphVertId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::GraphVertId::operator-=`.
        public unsafe MR.Mut_GraphVertId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_GraphVertId_sub_assign(MR.GraphVertId *_this, int a);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                return new(__MR_GraphVertId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::GraphVertId::operator+=`.
        public unsafe MR.Mut_GraphVertId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphVertId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphVertId._Underlying *__MR_GraphVertId_add_assign(MR.GraphVertId *_this, int a);
            fixed (MR.GraphVertId *__ptr__this = &this)
            {
                return new(__MR_GraphVertId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.GraphVertId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.GraphVertId)
                return this == (MR.GraphVertId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_GraphVertId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_GraphVertId`/`Const_GraphVertId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_GraphVertId
    {
        public readonly bool HasValue;
        internal readonly GraphVertId Object;
        public GraphVertId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_GraphVertId() {HasValue = false;}
        public _InOpt_GraphVertId(GraphVertId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_GraphVertId(GraphVertId new_value) {return new(new_value);}
        public _InOpt_GraphVertId(Const_GraphVertId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_GraphVertId(Const_GraphVertId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_GraphVertId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_GraphVertId`/`Const_GraphVertId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `GraphVertId`.
    public class _InOptMut_GraphVertId
    {
        public Mut_GraphVertId? Opt;

        public _InOptMut_GraphVertId() {}
        public _InOptMut_GraphVertId(Mut_GraphVertId value) {Opt = value;}
        public static implicit operator _InOptMut_GraphVertId(Mut_GraphVertId value) {return new(value);}
        public unsafe _InOptMut_GraphVertId(ref GraphVertId value)
        {
            fixed (GraphVertId *value_ptr = &value)
            {
                Opt = new((Const_GraphVertId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_GraphVertId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GraphVertId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_GraphVertId`/`Const_GraphVertId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `GraphVertId`.
    public class _InOptConst_GraphVertId
    {
        public Const_GraphVertId? Opt;

        public _InOptConst_GraphVertId() {}
        public _InOptConst_GraphVertId(Const_GraphVertId value) {Opt = value;}
        public static implicit operator _InOptConst_GraphVertId(Const_GraphVertId value) {return new(value);}
        public unsafe _InOptConst_GraphVertId(ref readonly GraphVertId value)
        {
            fixed (GraphVertId *value_ptr = &value)
            {
                Opt = new((Const_GraphVertId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::GraphEdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::GraphEdgeId>`
    /// This is the const reference to the struct.
    public class Const_GraphEdgeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.GraphEdgeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly GraphEdgeId UnderlyingStruct => ref *(GraphEdgeId *)_UnderlyingPtr;

        internal unsafe Const_GraphEdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_GraphEdgeId_Destroy(_Underlying *_this);
            __MR_GraphEdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GraphEdgeId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_GraphEdgeId(Const_GraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphEdgeId _ctor_result = __MR_GraphEdgeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::GraphEdgeId::GraphEdgeId`.
        public unsafe Const_GraphEdgeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Construct", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphEdgeId _ctor_result = __MR_GraphEdgeId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::GraphEdgeId::GraphEdgeId`.
        public unsafe Const_GraphEdgeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphEdgeId _ctor_result = __MR_GraphEdgeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::GraphEdgeId::operator int`.
        public static unsafe implicit operator int(MR.Const_GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_GraphEdgeId_ConvertTo_int(MR.Const_GraphEdgeId._Underlying *_this);
            return __MR_GraphEdgeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::GraphEdgeId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeId_ConvertTo_bool(MR.Const_GraphEdgeId._Underlying *_this);
            return __MR_GraphEdgeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphEdgeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeId_valid(_Underlying *_this);
            return __MR_GraphEdgeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::GraphEdgeId::operator==`.
        public static unsafe bool operator==(MR.Const_GraphEdgeId _this, MR.GraphEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_GraphEdgeId(MR.Const_GraphEdgeId._Underlying *_this, MR.GraphEdgeId b);
            return __MR_equal_MR_GraphEdgeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_GraphEdgeId _this, MR.GraphEdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::GraphEdgeId::operator<`.
        public unsafe bool Less(MR.GraphEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_GraphEdgeId(_Underlying *_this, MR.GraphEdgeId b);
            return __MR_less_MR_GraphEdgeId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::GraphEdgeId::operator--`.
        public static unsafe Mut_GraphEdgeId operator--(MR.Const_GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_decr_MR_GraphEdgeId(MR.Const_GraphEdgeId._Underlying *_this);
            Mut_GraphEdgeId _this_copy = new(_this);
            MR.Mut_GraphEdgeId _unused_ret = new(__MR_decr_MR_GraphEdgeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::GraphEdgeId::operator++`.
        public static unsafe Mut_GraphEdgeId operator++(MR.Const_GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_incr_MR_GraphEdgeId(MR.Const_GraphEdgeId._Underlying *_this);
            Mut_GraphEdgeId _this_copy = new(_this);
            MR.Mut_GraphEdgeId _unused_ret = new(__MR_incr_MR_GraphEdgeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.GraphEdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.GraphEdgeId)
                return this == (MR.GraphEdgeId)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::GraphEdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::GraphEdgeId>`
    /// This is the non-const reference to the struct.
    public class Mut_GraphEdgeId : Const_GraphEdgeId
    {
        /// Get the underlying struct.
        public unsafe new ref GraphEdgeId UnderlyingStruct => ref *(GraphEdgeId *)_UnderlyingPtr;

        internal unsafe Mut_GraphEdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_GraphEdgeId(Const_GraphEdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_GraphEdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphEdgeId _ctor_result = __MR_GraphEdgeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::GraphEdgeId::GraphEdgeId`.
        public unsafe Mut_GraphEdgeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Construct", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_Construct(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphEdgeId _ctor_result = __MR_GraphEdgeId_Construct(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::GraphEdgeId::GraphEdgeId`.
        public unsafe Mut_GraphEdgeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.GraphEdgeId _ctor_result = __MR_GraphEdgeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::GraphEdgeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_get", ExactSpelling = true)]
            extern static int *__MR_GraphEdgeId_get(_Underlying *_this);
            return ref *__MR_GraphEdgeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::GraphEdgeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_decr_MR_GraphEdgeId(_Underlying *_this);
            MR.Mut_GraphEdgeId _unused_ret = new(__MR_decr_MR_GraphEdgeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_incr_MR_GraphEdgeId(_Underlying *_this);
            MR.Mut_GraphEdgeId _unused_ret = new(__MR_incr_MR_GraphEdgeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeId::operator-=`.
        public unsafe MR.Mut_GraphEdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_GraphEdgeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_GraphEdgeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::GraphEdgeId::operator+=`.
        public unsafe MR.Mut_GraphEdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_GraphEdgeId_add_assign(_Underlying *_this, int a);
            return new(__MR_GraphEdgeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::GraphEdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::GraphEdgeId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct GraphEdgeId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator GraphEdgeId(Const_GraphEdgeId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_GraphEdgeId(GraphEdgeId other) => new(new Mut_GraphEdgeId((Mut_GraphEdgeId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public GraphEdgeId(GraphEdgeId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe GraphEdgeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_DefaultConstruct();
            this = __MR_GraphEdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::GraphEdgeId::GraphEdgeId`.
        public unsafe GraphEdgeId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Construct", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_Construct(MR.NoInit._Underlying *_1);
            this = __MR_GraphEdgeId_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::GraphEdgeId::GraphEdgeId`.
        public unsafe GraphEdgeId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_GraphEdgeId_Construct_int(int i);
            this = __MR_GraphEdgeId_Construct_int(i);
        }

        /// Generated from conversion operator `MR::GraphEdgeId::operator int`.
        public static unsafe implicit operator int(MR.GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_GraphEdgeId_ConvertTo_int(MR.Const_GraphEdgeId._Underlying *_this);
            return __MR_GraphEdgeId_ConvertTo_int((MR.Mut_GraphEdgeId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::GraphEdgeId::operator bool`.
        public static unsafe explicit operator bool(MR.GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeId_ConvertTo_bool(MR.Const_GraphEdgeId._Underlying *_this);
            return __MR_GraphEdgeId_ConvertTo_bool((MR.Mut_GraphEdgeId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::GraphEdgeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_GraphEdgeId_valid(MR.GraphEdgeId *_this);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                return __MR_GraphEdgeId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::GraphEdgeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_get", ExactSpelling = true)]
            extern static int *__MR_GraphEdgeId_get(MR.GraphEdgeId *_this);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                return ref *__MR_GraphEdgeId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::GraphEdgeId::operator==`.
        public static unsafe bool operator==(MR.GraphEdgeId _this, MR.GraphEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_GraphEdgeId(MR.Const_GraphEdgeId._Underlying *_this, MR.GraphEdgeId b);
            return __MR_equal_MR_GraphEdgeId((MR.Mut_GraphEdgeId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.GraphEdgeId _this, MR.GraphEdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::GraphEdgeId::operator<`.
        public unsafe bool Less(MR.GraphEdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_GraphEdgeId(MR.GraphEdgeId *_this, MR.GraphEdgeId b);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                return __MR_less_MR_GraphEdgeId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::GraphEdgeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_decr_MR_GraphEdgeId(MR.GraphEdgeId *_this);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                MR.Mut_GraphEdgeId _unused_ret = new(__MR_decr_MR_GraphEdgeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::GraphEdgeId::operator--`.
        public unsafe GraphEdgeId Decr(MR.GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_decr_MR_GraphEdgeId(GraphEdgeId *_this);
            GraphEdgeId _this_copy = new(_this);
            MR.Mut_GraphEdgeId _unused_ret = new(__MR_decr_MR_GraphEdgeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::GraphEdgeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_incr_MR_GraphEdgeId(MR.GraphEdgeId *_this);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                MR.Mut_GraphEdgeId _unused_ret = new(__MR_incr_MR_GraphEdgeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::GraphEdgeId::operator++`.
        public unsafe GraphEdgeId Incr(MR.GraphEdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_GraphEdgeId", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_incr_MR_GraphEdgeId(GraphEdgeId *_this);
            GraphEdgeId _this_copy = new(_this);
            MR.Mut_GraphEdgeId _unused_ret = new(__MR_incr_MR_GraphEdgeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::GraphEdgeId::operator-=`.
        public unsafe MR.Mut_GraphEdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_GraphEdgeId_sub_assign(MR.GraphEdgeId *_this, int a);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                return new(__MR_GraphEdgeId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::GraphEdgeId::operator+=`.
        public unsafe MR.Mut_GraphEdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GraphEdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_GraphEdgeId._Underlying *__MR_GraphEdgeId_add_assign(MR.GraphEdgeId *_this, int a);
            fixed (MR.GraphEdgeId *__ptr__this = &this)
            {
                return new(__MR_GraphEdgeId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        // IEquatable:

        public bool Equals(MR.GraphEdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.GraphEdgeId)
                return this == (MR.GraphEdgeId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_GraphEdgeId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_GraphEdgeId`/`Const_GraphEdgeId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_GraphEdgeId
    {
        public readonly bool HasValue;
        internal readonly GraphEdgeId Object;
        public GraphEdgeId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_GraphEdgeId() {HasValue = false;}
        public _InOpt_GraphEdgeId(GraphEdgeId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_GraphEdgeId(GraphEdgeId new_value) {return new(new_value);}
        public _InOpt_GraphEdgeId(Const_GraphEdgeId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_GraphEdgeId(Const_GraphEdgeId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_GraphEdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_GraphEdgeId`/`Const_GraphEdgeId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `GraphEdgeId`.
    public class _InOptMut_GraphEdgeId
    {
        public Mut_GraphEdgeId? Opt;

        public _InOptMut_GraphEdgeId() {}
        public _InOptMut_GraphEdgeId(Mut_GraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_GraphEdgeId(Mut_GraphEdgeId value) {return new(value);}
        public unsafe _InOptMut_GraphEdgeId(ref GraphEdgeId value)
        {
            fixed (GraphEdgeId *value_ptr = &value)
            {
                Opt = new((Const_GraphEdgeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_GraphEdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GraphEdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_GraphEdgeId`/`Const_GraphEdgeId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `GraphEdgeId`.
    public class _InOptConst_GraphEdgeId
    {
        public Const_GraphEdgeId? Opt;

        public _InOptConst_GraphEdgeId() {}
        public _InOptConst_GraphEdgeId(Const_GraphEdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_GraphEdgeId(Const_GraphEdgeId value) {return new(value);}
        public unsafe _InOptConst_GraphEdgeId(ref readonly GraphEdgeId value)
        {
            fixed (GraphEdgeId *value_ptr = &value)
            {
                Opt = new((Const_GraphEdgeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::Id<MR::ICPElemtTag>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>`
    /// This is the const half of the class.
    public class Const_Id_MRICPElemtTag : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Id_MRICPElemtTag>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Id_MRICPElemtTag(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_Destroy", ExactSpelling = true)]
            extern static void __MR_Id_MR_ICPElemtTag_Destroy(_Underlying *_this);
            __MR_Id_MR_ICPElemtTag_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Id_MRICPElemtTag() {Dispose(false);}

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_Get_id_", ExactSpelling = true)]
                extern static int *__MR_Id_MR_ICPElemtTag_Get_id_(_Underlying *_this);
                return *__MR_Id_MR_ICPElemtTag_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Id_MRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::Id<MR::ICPElemtTag>::Id`.
        public unsafe Const_Id_MRICPElemtTag(MR.Const_Id_MRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Id_MRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Id<MR::ICPElemtTag>::Id`.
        public unsafe Const_Id_MRICPElemtTag(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_Construct", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_Construct(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::Id<MR::ICPElemtTag>::Id`.
        public unsafe Const_Id_MRICPElemtTag(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_Construct_int", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_Construct_int(int i);
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_Construct_int(i);
        }

        /// Generated from conversion operator `MR::Id<MR::ICPElemtTag>::operator int`.
        public static unsafe implicit operator int(MR.Const_Id_MRICPElemtTag _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_Id_MR_ICPElemtTag_ConvertTo_int(MR.Const_Id_MRICPElemtTag._Underlying *_this);
            return __MR_Id_MR_ICPElemtTag_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::Id<MR::ICPElemtTag>::operator bool`.
        public static unsafe explicit operator bool(MR.Const_Id_MRICPElemtTag _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_Id_MR_ICPElemtTag_ConvertTo_bool(MR.Const_Id_MRICPElemtTag._Underlying *_this);
            return __MR_Id_MR_ICPElemtTag_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_valid", ExactSpelling = true)]
            extern static byte __MR_Id_MR_ICPElemtTag_valid(_Underlying *_this);
            return __MR_Id_MR_ICPElemtTag_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator==`.
        public static unsafe bool operator==(MR.Const_Id_MRICPElemtTag _this, MR.Const_Id_MRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Id_MR_ICPElemtTag(MR.Const_Id_MRICPElemtTag._Underlying *_this, MR.Id_MRICPElemtTag._Underlying *b);
            return __MR_equal_MR_Id_MR_ICPElemtTag(_this._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Id_MRICPElemtTag _this, MR.Const_Id_MRICPElemtTag b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator<`.
        public unsafe bool Less(MR.Const_Id_MRICPElemtTag b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static byte __MR_less_MR_Id_MR_ICPElemtTag(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *b);
            return __MR_less_MR_Id_MR_ICPElemtTag(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator--`.
        public static unsafe Id_MRICPElemtTag operator--(MR.Const_Id_MRICPElemtTag _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_decr_MR_Id_MR_ICPElemtTag(MR.Const_Id_MRICPElemtTag._Underlying *_this);
            Id_MRICPElemtTag _this_copy = new(_this);
            MR.Id_MRICPElemtTag _unused_ret = new(__MR_decr_MR_Id_MR_ICPElemtTag(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator++`.
        public static unsafe Id_MRICPElemtTag operator++(MR.Const_Id_MRICPElemtTag _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_incr_MR_Id_MR_ICPElemtTag(MR.Const_Id_MRICPElemtTag._Underlying *_this);
            Id_MRICPElemtTag _this_copy = new(_this);
            MR.Id_MRICPElemtTag _unused_ret = new(__MR_incr_MR_Id_MR_ICPElemtTag(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        // IEquatable:

        public bool Equals(MR.Const_Id_MRICPElemtTag? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Id_MRICPElemtTag)
                return this == (MR.Const_Id_MRICPElemtTag)other;
            return false;
        }
    }

    // stores index of some element, it is made as template class to avoid mixing faces, edges and vertices
    /// Generated from class `MR::Id<MR::ICPElemtTag>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::Id<MR::ICPElemtTag>>`
    /// This is the non-const half of the class.
    public class Id_MRICPElemtTag : Const_Id_MRICPElemtTag
    {
        internal unsafe Id_MRICPElemtTag(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_Id_MR_ICPElemtTag_GetMutable_id_(_Underlying *_this);
                return ref *__MR_Id_MR_ICPElemtTag_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Id_MRICPElemtTag() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_DefaultConstruct();
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_DefaultConstruct();
        }

        /// Generated from constructor `MR::Id<MR::ICPElemtTag>::Id`.
        public unsafe Id_MRICPElemtTag(MR.Const_Id_MRICPElemtTag _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_ConstructFromAnother(MR.Id_MRICPElemtTag._Underlying *_other);
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Id<MR::ICPElemtTag>::Id`.
        public unsafe Id_MRICPElemtTag(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_Construct", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_Construct(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_Construct(_1._UnderlyingPtr);
        }

        // Allow constructing from `int` and other integral types.
        // This constructor is written like this instead of a plain `Id(int)`, because we also wish to disable construction
        //   from other unrelated `Id<U>` specializations, which themselves have implicit conversions to `int`.
        // We could also achieve that using `template <typename U> Id(Id<U>) = delete;`, but it turns out that that causes issues
        //   for the `EdgeId::operator UndirectedEdgeId` below. There, while `UndirectedEdgeId x = EdgeId{};` compiles with this approach,
        //   but `UndirectedEdgeId x(EdgeId{});` doesn't. So to allow both forms, this constructor must be written this way, as a template.
        // The `= int` is there only to make the bindings emit this constructor, I don't think it affects anything else.
        /// Generated from constructor `MR::Id<MR::ICPElemtTag>::Id`.
        public unsafe Id_MRICPElemtTag(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_Construct_int", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_Construct_int(int i);
            _UnderlyingPtr = __MR_Id_MR_ICPElemtTag_Construct_int(i);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator=`.
        public unsafe MR.Id_MRICPElemtTag Assign(MR.Const_Id_MRICPElemtTag _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_AssignFromAnother(_Underlying *_this, MR.Id_MRICPElemtTag._Underlying *_other);
            return new(__MR_Id_MR_ICPElemtTag_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_get", ExactSpelling = true)]
            extern static int *__MR_Id_MR_ICPElemtTag_get(_Underlying *_this);
            return ref *__MR_Id_MR_ICPElemtTag_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_decr_MR_Id_MR_ICPElemtTag(_Underlying *_this);
            MR.Id_MRICPElemtTag _unused_ret = new(__MR_decr_MR_Id_MR_ICPElemtTag(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_Id_MR_ICPElemtTag", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_incr_MR_Id_MR_ICPElemtTag(_Underlying *_this);
            MR.Id_MRICPElemtTag _unused_ret = new(__MR_incr_MR_Id_MR_ICPElemtTag(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator-=`.
        public unsafe MR.Id_MRICPElemtTag SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_sub_assign", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_sub_assign(_Underlying *_this, int a);
            return new(__MR_Id_MR_ICPElemtTag_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::Id<MR::ICPElemtTag>::operator+=`.
        public unsafe MR.Id_MRICPElemtTag AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Id_MR_ICPElemtTag_add_assign", ExactSpelling = true)]
            extern static MR.Id_MRICPElemtTag._Underlying *__MR_Id_MR_ICPElemtTag_add_assign(_Underlying *_this, int a);
            return new(__MR_Id_MR_ICPElemtTag_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Id_MRICPElemtTag` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Id_MRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Id_MRICPElemtTag`/`Const_Id_MRICPElemtTag` directly.
    public class _InOptMut_Id_MRICPElemtTag
    {
        public Id_MRICPElemtTag? Opt;

        public _InOptMut_Id_MRICPElemtTag() {}
        public _InOptMut_Id_MRICPElemtTag(Id_MRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptMut_Id_MRICPElemtTag(Id_MRICPElemtTag value) {return new(value);}
    }

    /// This is used for optional parameters of class `Id_MRICPElemtTag` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Id_MRICPElemtTag`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Id_MRICPElemtTag`/`Const_Id_MRICPElemtTag` to pass it to the function.
    public class _InOptConst_Id_MRICPElemtTag
    {
        public Const_Id_MRICPElemtTag? Opt;

        public _InOptConst_Id_MRICPElemtTag() {}
        public _InOptConst_Id_MRICPElemtTag(Const_Id_MRICPElemtTag value) {Opt = value;}
        public static implicit operator _InOptConst_Id_MRICPElemtTag(Const_Id_MRICPElemtTag value) {return new(value);}
    }

    // Variant of Id<T> with omitted initialization by default. Useful for containers.
    /// Generated from class `MR::NoInitNodeId`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NodeId`
    /// This is the const half of the class.
    public class Const_NoInitNodeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.NodeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoInitNodeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_Destroy", ExactSpelling = true)]
            extern static void __MR_NoInitNodeId_Destroy(_Underlying *_this);
            __MR_NoInitNodeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoInitNodeId() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_NodeId(Const_NoInitNodeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_UpcastTo_MR_NodeId", ExactSpelling = true)]
            extern static MR.Const_NodeId._Underlying *__MR_NoInitNodeId_UpcastTo_MR_NodeId(_Underlying *_this);
            MR.Const_NodeId ret = new(__MR_NoInitNodeId_UpcastTo_MR_NodeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_Get_id_", ExactSpelling = true)]
                extern static int *__MR_NoInitNodeId_Get_id_(_Underlying *_this);
                return *__MR_NoInitNodeId_Get_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoInitNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoInitNodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public unsafe Const_NoInitNodeId(MR.Const_NoInitNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_ConstructFromAnother(MR.NoInitNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoInitNodeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public unsafe Const_NoInitNodeId(MR.NodeId id) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_Construct", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_Construct(MR.NodeId id);
            _UnderlyingPtr = __MR_NoInitNodeId_Construct(id);
        }

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public static unsafe implicit operator Const_NoInitNodeId(MR.NodeId id) {return new(id);}

        /// Generated from conversion operator `MR::NoInitNodeId::operator int`.
        public static unsafe implicit operator int(MR.Const_NoInitNodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_NoInitNodeId_ConvertTo_int(MR.Const_NoInitNodeId._Underlying *_this);
            return __MR_NoInitNodeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::NoInitNodeId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_NoInitNodeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_NoInitNodeId_ConvertTo_bool(MR.Const_NoInitNodeId._Underlying *_this);
            return __MR_NoInitNodeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoInitNodeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_valid", ExactSpelling = true)]
            extern static byte __MR_NoInitNodeId_valid(_Underlying *_this);
            return __MR_NoInitNodeId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::NoInitNodeId::operator==`.
        public static unsafe bool operator==(MR.Const_NoInitNodeId _this, MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_NoInitNodeId_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_NoInitNodeId_MR_NodeId(MR.Const_NoInitNodeId._Underlying *_this, MR.NodeId b);
            return __MR_equal_MR_NoInitNodeId_MR_NodeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_NoInitNodeId _this, MR.NodeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::NoInitNodeId::operator<`.
        public unsafe bool Less(MR.NodeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_NoInitNodeId_MR_NodeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_NoInitNodeId_MR_NodeId(_Underlying *_this, MR.NodeId b);
            return __MR_less_MR_NoInitNodeId_MR_NodeId(_UnderlyingPtr, b) != 0;
        }

        // IEquatable:

        public bool Equals(MR.NodeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.NodeId)
                return this == (MR.NodeId)other;
            return false;
        }
    }

    // Variant of Id<T> with omitted initialization by default. Useful for containers.
    /// Generated from class `MR::NoInitNodeId`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NodeId`
    /// This is the non-const half of the class.
    public class NoInitNodeId : Const_NoInitNodeId
    {
        internal unsafe NoInitNodeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Mut_NodeId(NoInitNodeId self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_UpcastTo_MR_NodeId", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NoInitNodeId_UpcastTo_MR_NodeId(_Underlying *_this);
            MR.Mut_NodeId ret = new(__MR_NoInitNodeId_UpcastTo_MR_NodeId(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref int Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_GetMutable_id_", ExactSpelling = true)]
                extern static int *__MR_NoInitNodeId_GetMutable_id_(_Underlying *_this);
                return ref *__MR_NoInitNodeId_GetMutable_id_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoInitNodeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_DefaultConstruct();
            _UnderlyingPtr = __MR_NoInitNodeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public unsafe NoInitNodeId(MR.Const_NoInitNodeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_ConstructFromAnother(MR.NoInitNodeId._Underlying *_other);
            _UnderlyingPtr = __MR_NoInitNodeId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public unsafe NoInitNodeId(MR.NodeId id) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_Construct", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_Construct(MR.NodeId id);
            _UnderlyingPtr = __MR_NoInitNodeId_Construct(id);
        }

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public static unsafe implicit operator NoInitNodeId(MR.NodeId id) {return new(id);}

        /// Generated from method `MR::NoInitNodeId::operator=`.
        public unsafe MR.NoInitNodeId Assign(MR.Const_NoInitNodeId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoInitNodeId._Underlying *__MR_NoInitNodeId_AssignFromAnother(_Underlying *_this, MR.NoInitNodeId._Underlying *_other);
            return new(__MR_NoInitNodeId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::NoInitNodeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_get", ExactSpelling = true)]
            extern static int *__MR_NoInitNodeId_get(_Underlying *_this);
            return ref *__MR_NoInitNodeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::NoInitNodeId::operator-=`.
        public unsafe MR.Mut_NodeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NoInitNodeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_NoInitNodeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::NoInitNodeId::operator+=`.
        public unsafe MR.Mut_NodeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoInitNodeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_NodeId._Underlying *__MR_NoInitNodeId_add_assign(_Underlying *_this, int a);
            return new(__MR_NoInitNodeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NoInitNodeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoInitNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoInitNodeId`/`Const_NoInitNodeId` directly.
    public class _InOptMut_NoInitNodeId
    {
        public NoInitNodeId? Opt;

        public _InOptMut_NoInitNodeId() {}
        public _InOptMut_NoInitNodeId(NoInitNodeId value) {Opt = value;}
        public static implicit operator _InOptMut_NoInitNodeId(NoInitNodeId value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoInitNodeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoInitNodeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoInitNodeId`/`Const_NoInitNodeId` to pass it to the function.
    public class _InOptConst_NoInitNodeId
    {
        public Const_NoInitNodeId? Opt;

        public _InOptConst_NoInitNodeId() {}
        public _InOptConst_NoInitNodeId(Const_NoInitNodeId value) {Opt = value;}
        public static implicit operator _InOptConst_NoInitNodeId(Const_NoInitNodeId value) {return new(value);}

        /// Generated from constructor `MR::NoInitNodeId::NoInitNodeId`.
        public static unsafe implicit operator _InOptConst_NoInitNodeId(MR.NodeId id) {return new MR.NoInitNodeId(id);}
    }

    // Those are full specializations in `MRId.h`, so `MR_CANONICAL_TYPEDEFS` doesn't work on them.
    // Have to add this too.
    /// Generated from class `MR::EdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::EdgeId>`
    /// This is the const reference to the struct.
    public class Const_EdgeId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.EdgeId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly EdgeId UnderlyingStruct => ref *(EdgeId *)_UnderlyingPtr;

        internal unsafe Const_EdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgeId_Destroy(_Underlying *_this);
            __MR_EdgeId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgeId() {Dispose(false);}

        public ref readonly int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_EdgeId(Const_EdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Const_EdgeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_MR_NoInit", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_MR_NoInit(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_MR_NoInit(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Const_EdgeId(MR.UndirectedEdgeId u) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_MR_UndirectedEdgeId(MR.UndirectedEdgeId u);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_MR_UndirectedEdgeId(u);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public static unsafe implicit operator Const_EdgeId(MR.UndirectedEdgeId u) {return new(u);}

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Const_EdgeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Const_EdgeId(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_unsigned_int(uint i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_unsigned_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Const_EdgeId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::EdgeId::operator int`.
        public static unsafe implicit operator int(MR.Const_EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_EdgeId_ConvertTo_int(MR.Const_EdgeId._Underlying *_this);
            return __MR_EdgeId_ConvertTo_int(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::EdgeId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_EdgeId_ConvertTo_bool(MR.Const_EdgeId._Underlying *_this);
            return __MR_EdgeId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from conversion operator `MR::EdgeId::operator MR::UndirectedEdgeId`.
        public static unsafe implicit operator MR.UndirectedEdgeId(MR.Const_EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_ConvertTo_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_EdgeId_ConvertTo_MR_UndirectedEdgeId(MR.Const_EdgeId._Underlying *_this);
            return __MR_EdgeId_ConvertTo_MR_UndirectedEdgeId(_this._UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_EdgeId_valid(_Underlying *_this);
            return __MR_EdgeId_valid(_UnderlyingPtr) != 0;
        }

        // returns identifier of the edge with same ends but opposite orientation
        /// Generated from method `MR::EdgeId::sym`.
        public unsafe MR.EdgeId Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_sym", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_sym(_Underlying *_this);
            return __MR_EdgeId_sym(_UnderlyingPtr);
        }

        // among each pair of sym-edges: one is always even and the other is odd
        /// Generated from method `MR::EdgeId::even`.
        public unsafe bool Even()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_even", ExactSpelling = true)]
            extern static byte __MR_EdgeId_even(_Underlying *_this);
            return __MR_EdgeId_even(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::EdgeId::odd`.
        public unsafe bool Odd()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_odd", ExactSpelling = true)]
            extern static byte __MR_EdgeId_odd(_Underlying *_this);
            return __MR_EdgeId_odd(_UnderlyingPtr) != 0;
        }

        // returns unique identifier of the edge ignoring its direction
        /// Generated from method `MR::EdgeId::undirected`.
        public unsafe MR.UndirectedEdgeId Undirected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_undirected", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_EdgeId_undirected(_Underlying *_this);
            return __MR_EdgeId_undirected(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeId::operator==`.
        public static unsafe bool operator==(MR.Const_EdgeId _this, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_EdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_EdgeId(MR.Const_EdgeId._Underlying *_this, MR.EdgeId b);
            return __MR_equal_MR_EdgeId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_EdgeId _this, MR.EdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::EdgeId::operator<`.
        public unsafe bool Less(MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_EdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_EdgeId(_Underlying *_this, MR.EdgeId b);
            return __MR_less_MR_EdgeId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::EdgeId::operator--`.
        public static unsafe Mut_EdgeId operator--(MR.Const_EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_decr_MR_EdgeId(MR.Const_EdgeId._Underlying *_this);
            Mut_EdgeId _this_copy = new(_this);
            MR.Mut_EdgeId _unused_ret = new(__MR_decr_MR_EdgeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::EdgeId::operator++`.
        public static unsafe Mut_EdgeId operator++(MR.Const_EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_incr_MR_EdgeId(MR.Const_EdgeId._Underlying *_this);
            Mut_EdgeId _this_copy = new(_this);
            MR.Mut_EdgeId _unused_ret = new(__MR_incr_MR_EdgeId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from function `MR::operator+<MR::EdgeTag>`.
        public static unsafe MR.EdgeId operator+(Const_EdgeId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_EdgeId_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_add_MR_EdgeId_int(MR.EdgeId id, int a);
            return __MR_add_MR_EdgeId_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator+<MR::EdgeTag>`.
        public static unsafe MR.EdgeId operator+(Const_EdgeId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_EdgeId_unsigned_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_add_MR_EdgeId_unsigned_int(MR.EdgeId id, uint a);
            return __MR_add_MR_EdgeId_unsigned_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator+<MR::EdgeTag>`.
        public static unsafe MR.EdgeId operator+(Const_EdgeId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_EdgeId_uint64_t", ExactSpelling = true)]
            extern static MR.EdgeId __MR_add_MR_EdgeId_uint64_t(MR.EdgeId id, ulong a);
            return __MR_add_MR_EdgeId_uint64_t(id.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.EdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.EdgeId)
                return this == (MR.EdgeId)other;
            return false;
        }
    }

    // Those are full specializations in `MRId.h`, so `MR_CANONICAL_TYPEDEFS` doesn't work on them.
    // Have to add this too.
    /// Generated from class `MR::EdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::EdgeId>`
    /// This is the non-const reference to the struct.
    public class Mut_EdgeId : Const_EdgeId
    {
        /// Get the underlying struct.
        public unsafe new ref EdgeId UnderlyingStruct => ref *(EdgeId *)_UnderlyingPtr;

        internal unsafe Mut_EdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_EdgeId(Const_EdgeId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_EdgeId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Mut_EdgeId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_MR_NoInit", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_MR_NoInit(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_MR_NoInit(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Mut_EdgeId(MR.UndirectedEdgeId u) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_MR_UndirectedEdgeId(MR.UndirectedEdgeId u);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_MR_UndirectedEdgeId(u);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public static unsafe implicit operator Mut_EdgeId(MR.UndirectedEdgeId u) {return new(u);}

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Mut_EdgeId(int i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_int(int i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Mut_EdgeId(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_unsigned_int(uint i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_unsigned_int(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe Mut_EdgeId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.EdgeId _ctor_result = __MR_EdgeId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from method `MR::EdgeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_get", ExactSpelling = true)]
            extern static int *__MR_EdgeId_get(_Underlying *_this);
            return ref *__MR_EdgeId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::EdgeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_decr_MR_EdgeId(_Underlying *_this);
            MR.Mut_EdgeId _unused_ret = new(__MR_decr_MR_EdgeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_incr_MR_EdgeId(_Underlying *_this);
            MR.Mut_EdgeId _unused_ret = new(__MR_incr_MR_EdgeId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::EdgeId::operator-=`.
        public unsafe MR.Mut_EdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_EdgeId_sub_assign(_Underlying *_this, int a);
            return new(__MR_EdgeId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::EdgeId::operator+=`.
        public unsafe MR.Mut_EdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_EdgeId_add_assign(_Underlying *_this, int a);
            return new(__MR_EdgeId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    // Those are full specializations in `MRId.h`, so `MR_CANONICAL_TYPEDEFS` doesn't work on them.
    // Have to add this too.
    /// Generated from class `MR::EdgeId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::EdgeId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct EdgeId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator EdgeId(Const_EdgeId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_EdgeId(EdgeId other) => new(new Mut_EdgeId((Mut_EdgeId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Id;

        /// Generated copy constructor.
        public EdgeId(EdgeId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_DefaultConstruct();
            this = __MR_EdgeId_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe EdgeId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_MR_NoInit", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_MR_NoInit(MR.NoInit._Underlying *_1);
            this = __MR_EdgeId_Construct_MR_NoInit(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe EdgeId(MR.UndirectedEdgeId u)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_MR_UndirectedEdgeId(MR.UndirectedEdgeId u);
            this = __MR_EdgeId_Construct_MR_UndirectedEdgeId(u);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public static unsafe implicit operator EdgeId(MR.UndirectedEdgeId u) {return new(u);}

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe EdgeId(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_int(int i);
            this = __MR_EdgeId_Construct_int(i);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe EdgeId(uint i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_unsigned_int(uint i);
            this = __MR_EdgeId_Construct_unsigned_int(i);
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public unsafe EdgeId(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_Construct_uint64_t(ulong i);
            this = __MR_EdgeId_Construct_uint64_t(i);
        }

        /// Generated from conversion operator `MR::EdgeId::operator int`.
        public static unsafe implicit operator int(MR.EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_ConvertTo_int", ExactSpelling = true)]
            extern static int __MR_EdgeId_ConvertTo_int(MR.Const_EdgeId._Underlying *_this);
            return __MR_EdgeId_ConvertTo_int((MR.Mut_EdgeId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::EdgeId::operator bool`.
        public static unsafe explicit operator bool(MR.EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_EdgeId_ConvertTo_bool(MR.Const_EdgeId._Underlying *_this);
            return __MR_EdgeId_ConvertTo_bool((MR.Mut_EdgeId._Underlying *)&_this) != 0;
        }

        /// Generated from conversion operator `MR::EdgeId::operator MR::UndirectedEdgeId`.
        public static unsafe implicit operator MR.UndirectedEdgeId(MR.EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_ConvertTo_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_EdgeId_ConvertTo_MR_UndirectedEdgeId(MR.Const_EdgeId._Underlying *_this);
            return __MR_EdgeId_ConvertTo_MR_UndirectedEdgeId((MR.Mut_EdgeId._Underlying *)&_this);
        }

        /// Generated from method `MR::EdgeId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_valid", ExactSpelling = true)]
            extern static byte __MR_EdgeId_valid(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return __MR_EdgeId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::EdgeId::get`.
        public unsafe ref int Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_get", ExactSpelling = true)]
            extern static int *__MR_EdgeId_get(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return ref *__MR_EdgeId_get(__ptr__this);
            }
        }

        // returns identifier of the edge with same ends but opposite orientation
        /// Generated from method `MR::EdgeId::sym`.
        public unsafe MR.EdgeId Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_sym", ExactSpelling = true)]
            extern static MR.EdgeId __MR_EdgeId_sym(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return __MR_EdgeId_sym(__ptr__this);
            }
        }

        // among each pair of sym-edges: one is always even and the other is odd
        /// Generated from method `MR::EdgeId::even`.
        public unsafe bool Even()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_even", ExactSpelling = true)]
            extern static byte __MR_EdgeId_even(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return __MR_EdgeId_even(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::EdgeId::odd`.
        public unsafe bool Odd()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_odd", ExactSpelling = true)]
            extern static byte __MR_EdgeId_odd(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return __MR_EdgeId_odd(__ptr__this) != 0;
            }
        }

        // returns unique identifier of the edge ignoring its direction
        /// Generated from method `MR::EdgeId::undirected`.
        public unsafe MR.UndirectedEdgeId Undirected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_undirected", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_EdgeId_undirected(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return __MR_EdgeId_undirected(__ptr__this);
            }
        }

        /// Generated from method `MR::EdgeId::operator==`.
        public static unsafe bool operator==(MR.EdgeId _this, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_EdgeId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_EdgeId(MR.Const_EdgeId._Underlying *_this, MR.EdgeId b);
            return __MR_equal_MR_EdgeId((MR.Mut_EdgeId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.EdgeId _this, MR.EdgeId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::EdgeId::operator<`.
        public unsafe bool Less(MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_EdgeId", ExactSpelling = true)]
            extern static byte __MR_less_MR_EdgeId(MR.EdgeId *_this, MR.EdgeId b);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return __MR_less_MR_EdgeId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::EdgeId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_decr_MR_EdgeId(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                MR.Mut_EdgeId _unused_ret = new(__MR_decr_MR_EdgeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::EdgeId::operator--`.
        public unsafe EdgeId Decr(MR.EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_decr_MR_EdgeId(EdgeId *_this);
            EdgeId _this_copy = new(_this);
            MR.Mut_EdgeId _unused_ret = new(__MR_decr_MR_EdgeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::EdgeId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_incr_MR_EdgeId(MR.EdgeId *_this);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                MR.Mut_EdgeId _unused_ret = new(__MR_incr_MR_EdgeId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::EdgeId::operator++`.
        public unsafe EdgeId Incr(MR.EdgeId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_EdgeId", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_incr_MR_EdgeId(EdgeId *_this);
            EdgeId _this_copy = new(_this);
            MR.Mut_EdgeId _unused_ret = new(__MR_incr_MR_EdgeId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::EdgeId::operator-=`.
        public unsafe MR.Mut_EdgeId SubAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_EdgeId_sub_assign(MR.EdgeId *_this, int a);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return new(__MR_EdgeId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::EdgeId::operator+=`.
        public unsafe MR.Mut_EdgeId AddAssign(int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_EdgeId._Underlying *__MR_EdgeId_add_assign(MR.EdgeId *_this, int a);
            fixed (MR.EdgeId *__ptr__this = &this)
            {
                return new(__MR_EdgeId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from function `MR::operator+<MR::EdgeTag>`.
        public static MR.EdgeId operator+(MR.EdgeId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_EdgeId_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_add_MR_EdgeId_int(MR.EdgeId id, int a);
            return __MR_add_MR_EdgeId_int(id, a);
        }

        /// Generated from function `MR::operator+<MR::EdgeTag>`.
        public static MR.EdgeId operator+(MR.EdgeId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_EdgeId_unsigned_int", ExactSpelling = true)]
            extern static MR.EdgeId __MR_add_MR_EdgeId_unsigned_int(MR.EdgeId id, uint a);
            return __MR_add_MR_EdgeId_unsigned_int(id, a);
        }

        /// Generated from function `MR::operator+<MR::EdgeTag>`.
        public static MR.EdgeId operator+(MR.EdgeId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_EdgeId_uint64_t", ExactSpelling = true)]
            extern static MR.EdgeId __MR_add_MR_EdgeId_uint64_t(MR.EdgeId id, ulong a);
            return __MR_add_MR_EdgeId_uint64_t(id, a);
        }

        // IEquatable:

        public bool Equals(MR.EdgeId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.EdgeId)
                return this == (MR.EdgeId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_EdgeId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_EdgeId`/`Const_EdgeId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_EdgeId
    {
        public readonly bool HasValue;
        internal readonly EdgeId Object;
        public EdgeId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_EdgeId() {HasValue = false;}
        public _InOpt_EdgeId(EdgeId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_EdgeId(EdgeId new_value) {return new(new_value);}
        public _InOpt_EdgeId(Const_EdgeId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_EdgeId(Const_EdgeId new_value) {return new(new_value);}

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public static unsafe implicit operator _InOpt_EdgeId(MR.UndirectedEdgeId u) {return new MR.EdgeId(u);}
    }

    /// This is used for optional parameters of class `Mut_EdgeId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_EdgeId`/`Const_EdgeId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `EdgeId`.
    public class _InOptMut_EdgeId
    {
        public Mut_EdgeId? Opt;

        public _InOptMut_EdgeId() {}
        public _InOptMut_EdgeId(Mut_EdgeId value) {Opt = value;}
        public static implicit operator _InOptMut_EdgeId(Mut_EdgeId value) {return new(value);}
        public unsafe _InOptMut_EdgeId(ref EdgeId value)
        {
            fixed (EdgeId *value_ptr = &value)
            {
                Opt = new((Const_EdgeId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_EdgeId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgeId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_EdgeId`/`Const_EdgeId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `EdgeId`.
    public class _InOptConst_EdgeId
    {
        public Const_EdgeId? Opt;

        public _InOptConst_EdgeId() {}
        public _InOptConst_EdgeId(Const_EdgeId value) {Opt = value;}
        public static implicit operator _InOptConst_EdgeId(Const_EdgeId value) {return new(value);}
        public unsafe _InOptConst_EdgeId(ref readonly EdgeId value)
        {
            fixed (EdgeId *value_ptr = &value)
            {
                Opt = new((Const_EdgeId._Underlying *)value_ptr, is_owning: false);
            }
        }

        /// Generated from constructor `MR::EdgeId::EdgeId`.
        public static unsafe implicit operator _InOptConst_EdgeId(MR.UndirectedEdgeId u) {return new MR.EdgeId(u);}
    }

    /// Generated from class `MR::VoxelId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::VoxelId>`
    /// This is the const reference to the struct.
    public class Const_VoxelId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.VoxelId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly VoxelId UnderlyingStruct => ref *(VoxelId *)_UnderlyingPtr;

        internal unsafe Const_VoxelId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelId_Destroy(_Underlying *_this);
            __MR_VoxelId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelId() {Dispose(false);}

        public ref readonly ulong Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_VoxelId(Const_VoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.VoxelId _ctor_result = __MR_VoxelId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::VoxelId::VoxelId`.
        public unsafe Const_VoxelId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Construct_MR_NoInit", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_Construct_MR_NoInit(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.VoxelId _ctor_result = __MR_VoxelId_Construct_MR_NoInit(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::VoxelId::VoxelId`.
        public unsafe Const_VoxelId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.VoxelId _ctor_result = __MR_VoxelId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from conversion operator `MR::VoxelId::operator MR_uint64_t`.
        public static unsafe implicit operator ulong(MR.Const_VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_ConvertTo_uint64_t", ExactSpelling = true)]
            extern static ulong __MR_VoxelId_ConvertTo_uint64_t(MR.Const_VoxelId._Underlying *_this);
            return __MR_VoxelId_ConvertTo_uint64_t(_this._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::VoxelId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_VoxelId_ConvertTo_bool(MR.Const_VoxelId._Underlying *_this);
            return __MR_VoxelId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VoxelId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_valid", ExactSpelling = true)]
            extern static byte __MR_VoxelId_valid(_Underlying *_this);
            return __MR_VoxelId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VoxelId::operator==`.
        public static unsafe bool operator==(MR.Const_VoxelId _this, MR.VoxelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VoxelId(MR.Const_VoxelId._Underlying *_this, MR.VoxelId b);
            return __MR_equal_MR_VoxelId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_VoxelId _this, MR.VoxelId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::VoxelId::operator<`.
        public unsafe bool Less(MR.VoxelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_less_MR_VoxelId(_Underlying *_this, MR.VoxelId b);
            return __MR_less_MR_VoxelId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::VoxelId::operator--`.
        public static unsafe Mut_VoxelId operator--(MR.Const_VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_decr_MR_VoxelId(MR.Const_VoxelId._Underlying *_this);
            Mut_VoxelId _this_copy = new(_this);
            MR.Mut_VoxelId _unused_ret = new(__MR_decr_MR_VoxelId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::VoxelId::operator++`.
        public static unsafe Mut_VoxelId operator++(MR.Const_VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_incr_MR_VoxelId(MR.Const_VoxelId._Underlying *_this);
            Mut_VoxelId _this_copy = new(_this);
            MR.Mut_VoxelId _unused_ret = new(__MR_incr_MR_VoxelId(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from function `MR::operator+<MR::VoxelTag>`.
        public static unsafe MR.VoxelId operator+(Const_VoxelId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_VoxelId_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_add_MR_VoxelId_int(MR.VoxelId id, int a);
            return __MR_add_MR_VoxelId_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator+<MR::VoxelTag>`.
        public static unsafe MR.VoxelId operator+(Const_VoxelId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_VoxelId_unsigned_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_add_MR_VoxelId_unsigned_int(MR.VoxelId id, uint a);
            return __MR_add_MR_VoxelId_unsigned_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator+<MR::VoxelTag>`.
        public static unsafe MR.VoxelId operator+(Const_VoxelId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_VoxelId_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_add_MR_VoxelId_uint64_t(MR.VoxelId id, ulong a);
            return __MR_add_MR_VoxelId_uint64_t(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator-<MR::VoxelTag>`.
        public static unsafe MR.VoxelId operator-(Const_VoxelId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelId_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_sub_MR_VoxelId_int(MR.VoxelId id, int a);
            return __MR_sub_MR_VoxelId_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator-<MR::VoxelTag>`.
        public static unsafe MR.VoxelId operator-(Const_VoxelId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelId_unsigned_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_sub_MR_VoxelId_unsigned_int(MR.VoxelId id, uint a);
            return __MR_sub_MR_VoxelId_unsigned_int(id.UnderlyingStruct, a);
        }

        /// Generated from function `MR::operator-<MR::VoxelTag>`.
        public static unsafe MR.VoxelId operator-(Const_VoxelId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelId_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_sub_MR_VoxelId_uint64_t(MR.VoxelId id, ulong a);
            return __MR_sub_MR_VoxelId_uint64_t(id.UnderlyingStruct, a);
        }

        // IEquatable:

        public bool Equals(MR.VoxelId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.VoxelId)
                return this == (MR.VoxelId)other;
            return false;
        }
    }

    /// Generated from class `MR::VoxelId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::VoxelId>`
    /// This is the non-const reference to the struct.
    public class Mut_VoxelId : Const_VoxelId
    {
        /// Get the underlying struct.
        public unsafe new ref VoxelId UnderlyingStruct => ref *(VoxelId *)_UnderlyingPtr;

        internal unsafe Mut_VoxelId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref ulong Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_VoxelId(Const_VoxelId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_VoxelId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.VoxelId _ctor_result = __MR_VoxelId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::VoxelId::VoxelId`.
        public unsafe Mut_VoxelId(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Construct_MR_NoInit", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_Construct_MR_NoInit(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.VoxelId _ctor_result = __MR_VoxelId_Construct_MR_NoInit(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::VoxelId::VoxelId`.
        public unsafe Mut_VoxelId(ulong i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_Construct_uint64_t(ulong i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.VoxelId _ctor_result = __MR_VoxelId_Construct_uint64_t(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::VoxelId::get`.
        public unsafe ref ulong Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_get", ExactSpelling = true)]
            extern static ulong *__MR_VoxelId_get(_Underlying *_this);
            return ref *__MR_VoxelId_get(_UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_decr_MR_VoxelId(_Underlying *_this);
            MR.Mut_VoxelId _unused_ret = new(__MR_decr_MR_VoxelId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_incr_MR_VoxelId(_Underlying *_this);
            MR.Mut_VoxelId _unused_ret = new(__MR_incr_MR_VoxelId(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::VoxelId::operator-=`.
        public unsafe MR.Mut_VoxelId SubAssign(ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_VoxelId_sub_assign(_Underlying *_this, ulong a);
            return new(__MR_VoxelId_sub_assign(_UnderlyingPtr, a), is_owning: false);
        }

        /// Generated from method `MR::VoxelId::operator+=`.
        public unsafe MR.Mut_VoxelId AddAssign(ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_VoxelId_add_assign(_Underlying *_this, ulong a);
            return new(__MR_VoxelId_add_assign(_UnderlyingPtr, a), is_owning: false);
        }
    }

    /// Generated from class `MR::VoxelId`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::NoDefInit<MR::VoxelId>`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
    public struct VoxelId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator VoxelId(Const_VoxelId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_VoxelId(VoxelId other) => new(new Mut_VoxelId((Mut_VoxelId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public ulong Id;

        /// Generated copy constructor.
        public VoxelId(VoxelId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_DefaultConstruct();
            this = __MR_VoxelId_DefaultConstruct();
        }

        /// Generated from constructor `MR::VoxelId::VoxelId`.
        public unsafe VoxelId(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Construct_MR_NoInit", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_Construct_MR_NoInit(MR.NoInit._Underlying *_1);
            this = __MR_VoxelId_Construct_MR_NoInit(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelId::VoxelId`.
        public unsafe VoxelId(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_Construct_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VoxelId_Construct_uint64_t(ulong i);
            this = __MR_VoxelId_Construct_uint64_t(i);
        }

        /// Generated from conversion operator `MR::VoxelId::operator MR_uint64_t`.
        public static unsafe implicit operator ulong(MR.VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_ConvertTo_uint64_t", ExactSpelling = true)]
            extern static ulong __MR_VoxelId_ConvertTo_uint64_t(MR.Const_VoxelId._Underlying *_this);
            return __MR_VoxelId_ConvertTo_uint64_t((MR.Mut_VoxelId._Underlying *)&_this);
        }

        /// Generated from conversion operator `MR::VoxelId::operator bool`.
        public static unsafe explicit operator bool(MR.VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_VoxelId_ConvertTo_bool(MR.Const_VoxelId._Underlying *_this);
            return __MR_VoxelId_ConvertTo_bool((MR.Mut_VoxelId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::VoxelId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_valid", ExactSpelling = true)]
            extern static byte __MR_VoxelId_valid(MR.VoxelId *_this);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                return __MR_VoxelId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::VoxelId::get`.
        public unsafe ref ulong Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_get", ExactSpelling = true)]
            extern static ulong *__MR_VoxelId_get(MR.VoxelId *_this);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                return ref *__MR_VoxelId_get(__ptr__this);
            }
        }

        /// Generated from method `MR::VoxelId::operator==`.
        public static unsafe bool operator==(MR.VoxelId _this, MR.VoxelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_VoxelId(MR.Const_VoxelId._Underlying *_this, MR.VoxelId b);
            return __MR_equal_MR_VoxelId((MR.Mut_VoxelId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.VoxelId _this, MR.VoxelId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::VoxelId::operator<`.
        public unsafe bool Less(MR.VoxelId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_less_MR_VoxelId(MR.VoxelId *_this, MR.VoxelId b);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                return __MR_less_MR_VoxelId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::VoxelId::operator--`.
        public unsafe void Decr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_decr_MR_VoxelId(MR.VoxelId *_this);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                MR.Mut_VoxelId _unused_ret = new(__MR_decr_MR_VoxelId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::VoxelId::operator--`.
        public unsafe VoxelId Decr(MR.VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_decr_MR_VoxelId(VoxelId *_this);
            VoxelId _this_copy = new(_this);
            MR.Mut_VoxelId _unused_ret = new(__MR_decr_MR_VoxelId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::VoxelId::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_incr_MR_VoxelId(MR.VoxelId *_this);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                MR.Mut_VoxelId _unused_ret = new(__MR_incr_MR_VoxelId(__ptr__this), is_owning: false);
            }
        }

        /// Generated from method `MR::VoxelId::operator++`.
        public unsafe VoxelId Incr(MR.VoxelId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_VoxelId", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_incr_MR_VoxelId(VoxelId *_this);
            VoxelId _this_copy = new(_this);
            MR.Mut_VoxelId _unused_ret = new(__MR_incr_MR_VoxelId(&_this_copy), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::VoxelId::operator-=`.
        public unsafe MR.Mut_VoxelId SubAssign(ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_sub_assign", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_VoxelId_sub_assign(MR.VoxelId *_this, ulong a);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                return new(__MR_VoxelId_sub_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from method `MR::VoxelId::operator+=`.
        public unsafe MR.Mut_VoxelId AddAssign(ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelId_add_assign", ExactSpelling = true)]
            extern static MR.Mut_VoxelId._Underlying *__MR_VoxelId_add_assign(MR.VoxelId *_this, ulong a);
            fixed (MR.VoxelId *__ptr__this = &this)
            {
                return new(__MR_VoxelId_add_assign(__ptr__this, a), is_owning: false);
            }
        }

        /// Generated from function `MR::operator+<MR::VoxelTag>`.
        public static MR.VoxelId operator+(MR.VoxelId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_VoxelId_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_add_MR_VoxelId_int(MR.VoxelId id, int a);
            return __MR_add_MR_VoxelId_int(id, a);
        }

        /// Generated from function `MR::operator+<MR::VoxelTag>`.
        public static MR.VoxelId operator+(MR.VoxelId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_VoxelId_unsigned_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_add_MR_VoxelId_unsigned_int(MR.VoxelId id, uint a);
            return __MR_add_MR_VoxelId_unsigned_int(id, a);
        }

        /// Generated from function `MR::operator+<MR::VoxelTag>`.
        public static MR.VoxelId operator+(MR.VoxelId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_MR_VoxelId_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_add_MR_VoxelId_uint64_t(MR.VoxelId id, ulong a);
            return __MR_add_MR_VoxelId_uint64_t(id, a);
        }

        /// Generated from function `MR::operator-<MR::VoxelTag>`.
        public static MR.VoxelId operator-(MR.VoxelId id, int a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelId_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_sub_MR_VoxelId_int(MR.VoxelId id, int a);
            return __MR_sub_MR_VoxelId_int(id, a);
        }

        /// Generated from function `MR::operator-<MR::VoxelTag>`.
        public static MR.VoxelId operator-(MR.VoxelId id, uint a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelId_unsigned_int", ExactSpelling = true)]
            extern static MR.VoxelId __MR_sub_MR_VoxelId_unsigned_int(MR.VoxelId id, uint a);
            return __MR_sub_MR_VoxelId_unsigned_int(id, a);
        }

        /// Generated from function `MR::operator-<MR::VoxelTag>`.
        public static MR.VoxelId operator-(MR.VoxelId id, ulong a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_VoxelId_uint64_t", ExactSpelling = true)]
            extern static MR.VoxelId __MR_sub_MR_VoxelId_uint64_t(MR.VoxelId id, ulong a);
            return __MR_sub_MR_VoxelId_uint64_t(id, a);
        }

        // IEquatable:

        public bool Equals(MR.VoxelId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.VoxelId)
                return this == (MR.VoxelId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_VoxelId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_VoxelId`/`Const_VoxelId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_VoxelId
    {
        public readonly bool HasValue;
        internal readonly VoxelId Object;
        public VoxelId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_VoxelId() {HasValue = false;}
        public _InOpt_VoxelId(VoxelId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_VoxelId(VoxelId new_value) {return new(new_value);}
        public _InOpt_VoxelId(Const_VoxelId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_VoxelId(Const_VoxelId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_VoxelId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_VoxelId`/`Const_VoxelId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `VoxelId`.
    public class _InOptMut_VoxelId
    {
        public Mut_VoxelId? Opt;

        public _InOptMut_VoxelId() {}
        public _InOptMut_VoxelId(Mut_VoxelId value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelId(Mut_VoxelId value) {return new(value);}
        public unsafe _InOptMut_VoxelId(ref VoxelId value)
        {
            fixed (VoxelId *value_ptr = &value)
            {
                Opt = new((Const_VoxelId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_VoxelId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_VoxelId`/`Const_VoxelId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `VoxelId`.
    public class _InOptConst_VoxelId
    {
        public Const_VoxelId? Opt;

        public _InOptConst_VoxelId() {}
        public _InOptConst_VoxelId(Const_VoxelId value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelId(Const_VoxelId value) {return new(value);}
        public unsafe _InOptConst_VoxelId(ref readonly VoxelId value)
        {
            fixed (VoxelId *value_ptr = &value)
            {
                Opt = new((Const_VoxelId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }
}
