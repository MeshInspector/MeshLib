public static partial class MR
{
    /// stores unique identifier of a viewport, which is power of two;
    /// id=0 has a special meaning of default viewport in some contexts
    /// Generated from class `MR::ViewportId`.
    /// This is the const reference to the struct.
    public class Const_ViewportId : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.ViewportId>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly ViewportId UnderlyingStruct => ref *(ViewportId *)_UnderlyingPtr;

        internal unsafe Const_ViewportId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportId_Destroy(_Underlying *_this);
            __MR_ViewportId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportId() {Dispose(false);}

        public ref readonly uint Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Const_ViewportId(Const_ViewportId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ViewportId _ctor_result = __MR_ViewportId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::ViewportId::ViewportId`.
        public unsafe Const_ViewportId(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_Construct", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_Construct(uint i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ViewportId _ctor_result = __MR_ViewportId_Construct(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from conversion operator `MR::ViewportId::operator bool`.
        public static unsafe explicit operator bool(MR.Const_ViewportId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_ViewportId_ConvertTo_bool(MR.Const_ViewportId._Underlying *_this);
            return __MR_ViewportId_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ViewportId::value`.
        public unsafe uint Value()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_value", ExactSpelling = true)]
            extern static uint __MR_ViewportId_value(_Underlying *_this);
            return __MR_ViewportId_value(_UnderlyingPtr);
        }

        /// Generated from method `MR::ViewportId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_valid", ExactSpelling = true)]
            extern static byte __MR_ViewportId_valid(_Underlying *_this);
            return __MR_ViewportId_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ViewportId::operator==`.
        public static unsafe bool operator==(MR.Const_ViewportId _this, MR.ViewportId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ViewportId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ViewportId(MR.Const_ViewportId._Underlying *_this, MR.ViewportId b);
            return __MR_equal_MR_ViewportId(_this._UnderlyingPtr, b) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ViewportId _this, MR.ViewportId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::ViewportId::operator<`.
        public unsafe bool Less(MR.ViewportId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_ViewportId", ExactSpelling = true)]
            extern static byte __MR_less_MR_ViewportId(_Underlying *_this, MR.ViewportId b);
            return __MR_less_MR_ViewportId(_UnderlyingPtr, b) != 0;
        }

        /// Generated from method `MR::ViewportId::next`.
        public unsafe MR.ViewportId Next()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_next", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_next(_Underlying *_this);
            return __MR_ViewportId_next(_UnderlyingPtr);
        }

        /// Generated from method `MR::ViewportId::prev`.
        public unsafe MR.ViewportId Prev()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_prev", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_prev(_Underlying *_this);
            return __MR_ViewportId_prev(_UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.ViewportId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.ViewportId)
                return this == (MR.ViewportId)other;
            return false;
        }
    }

    /// stores unique identifier of a viewport, which is power of two;
    /// id=0 has a special meaning of default viewport in some contexts
    /// Generated from class `MR::ViewportId`.
    /// This is the non-const reference to the struct.
    public class Mut_ViewportId : Const_ViewportId
    {
        /// Get the underlying struct.
        public unsafe new ref ViewportId UnderlyingStruct => ref *(ViewportId *)_UnderlyingPtr;

        internal unsafe Mut_ViewportId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref uint Id => ref UnderlyingStruct.Id;

        /// Generated copy constructor.
        public unsafe Mut_ViewportId(Const_ViewportId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 4);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_ViewportId() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ViewportId _ctor_result = __MR_ViewportId_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }

        /// Generated from constructor `MR::ViewportId::ViewportId`.
        public unsafe Mut_ViewportId(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_Construct", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_Construct(uint i);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(4);
            MR.ViewportId _ctor_result = __MR_ViewportId_Construct(i);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 4);
        }
    }

    /// stores unique identifier of a viewport, which is power of two;
    /// id=0 has a special meaning of default viewport in some contexts
    /// Generated from class `MR::ViewportId`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 4)]
    public struct ViewportId
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator ViewportId(Const_ViewportId other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_ViewportId(ViewportId other) => new(new Mut_ViewportId((Mut_ViewportId._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public uint Id;

        /// Generated copy constructor.
        public ViewportId(ViewportId _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_DefaultConstruct();
            this = __MR_ViewportId_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportId::ViewportId`.
        public unsafe ViewportId(uint i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_Construct", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_Construct(uint i);
            this = __MR_ViewportId_Construct(i);
        }

        /// Generated from conversion operator `MR::ViewportId::operator bool`.
        public static unsafe explicit operator bool(MR.ViewportId _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_ViewportId_ConvertTo_bool(MR.Const_ViewportId._Underlying *_this);
            return __MR_ViewportId_ConvertTo_bool((MR.Mut_ViewportId._Underlying *)&_this) != 0;
        }

        /// Generated from method `MR::ViewportId::value`.
        public unsafe uint Value()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_value", ExactSpelling = true)]
            extern static uint __MR_ViewportId_value(MR.ViewportId *_this);
            fixed (MR.ViewportId *__ptr__this = &this)
            {
                return __MR_ViewportId_value(__ptr__this);
            }
        }

        /// Generated from method `MR::ViewportId::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_valid", ExactSpelling = true)]
            extern static byte __MR_ViewportId_valid(MR.ViewportId *_this);
            fixed (MR.ViewportId *__ptr__this = &this)
            {
                return __MR_ViewportId_valid(__ptr__this) != 0;
            }
        }

        /// Generated from method `MR::ViewportId::operator==`.
        public static unsafe bool operator==(MR.ViewportId _this, MR.ViewportId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ViewportId", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ViewportId(MR.Const_ViewportId._Underlying *_this, MR.ViewportId b);
            return __MR_equal_MR_ViewportId((MR.Mut_ViewportId._Underlying *)&_this, b) != 0;
        }

        public static unsafe bool operator!=(MR.ViewportId _this, MR.ViewportId b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::ViewportId::operator<`.
        public unsafe bool Less(MR.ViewportId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_ViewportId", ExactSpelling = true)]
            extern static byte __MR_less_MR_ViewportId(MR.ViewportId *_this, MR.ViewportId b);
            fixed (MR.ViewportId *__ptr__this = &this)
            {
                return __MR_less_MR_ViewportId(__ptr__this, b) != 0;
            }
        }

        /// Generated from method `MR::ViewportId::next`.
        public unsafe MR.ViewportId Next()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_next", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_next(MR.ViewportId *_this);
            fixed (MR.ViewportId *__ptr__this = &this)
            {
                return __MR_ViewportId_next(__ptr__this);
            }
        }

        /// Generated from method `MR::ViewportId::prev`.
        public unsafe MR.ViewportId Prev()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportId_prev", ExactSpelling = true)]
            extern static MR.ViewportId __MR_ViewportId_prev(MR.ViewportId *_this);
            fixed (MR.ViewportId *__ptr__this = &this)
            {
                return __MR_ViewportId_prev(__ptr__this);
            }
        }

        // IEquatable:

        public bool Equals(MR.ViewportId b)
        {
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.ViewportId)
                return this == (MR.ViewportId)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_ViewportId` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_ViewportId`/`Const_ViewportId` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_ViewportId
    {
        public readonly bool HasValue;
        internal readonly ViewportId Object;
        public ViewportId Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_ViewportId() {HasValue = false;}
        public _InOpt_ViewportId(ViewportId new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_ViewportId(ViewportId new_value) {return new(new_value);}
        public _InOpt_ViewportId(Const_ViewportId new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_ViewportId(Const_ViewportId new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_ViewportId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_ViewportId`/`Const_ViewportId` directly.
    /// * Pass `new(ref ...)` to pass a reference to `ViewportId`.
    public class _InOptMut_ViewportId
    {
        public Mut_ViewportId? Opt;

        public _InOptMut_ViewportId() {}
        public _InOptMut_ViewportId(Mut_ViewportId value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportId(Mut_ViewportId value) {return new(value);}
        public unsafe _InOptMut_ViewportId(ref ViewportId value)
        {
            fixed (ViewportId *value_ptr = &value)
            {
                Opt = new((Const_ViewportId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_ViewportId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_ViewportId`/`Const_ViewportId` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `ViewportId`.
    public class _InOptConst_ViewportId
    {
        public Const_ViewportId? Opt;

        public _InOptConst_ViewportId() {}
        public _InOptConst_ViewportId(Const_ViewportId value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportId(Const_ViewportId value) {return new(value);}
        public unsafe _InOptConst_ViewportId(ref readonly ViewportId value)
        {
            fixed (ViewportId *value_ptr = &value)
            {
                Opt = new((Const_ViewportId._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// stores mask of viewport unique identifiers
    /// Generated from class `MR::ViewportMask`.
    /// This is the const half of the class.
    public class Const_ViewportMask : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_ViewportMask>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportMask(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportMask_Destroy(_Underlying *_this);
            __MR_ViewportMask_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportMask() {Dispose(false);}

        public unsafe uint Mask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_Get_mask_", ExactSpelling = true)]
                extern static uint *__MR_ViewportMask_Get_mask_(_Underlying *_this);
                return *__MR_ViewportMask_Get_mask_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportMask() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportMask_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public unsafe Const_ViewportMask(MR.Const_ViewportMask _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_ConstructFromAnother(MR.ViewportMask._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportMask_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public unsafe Const_ViewportMask(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_Construct_unsigned_int(uint i);
            _UnderlyingPtr = __MR_ViewportMask_Construct_unsigned_int(i);
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public unsafe Const_ViewportMask(MR.ViewportId i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_Construct_MR_ViewportId", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_Construct_MR_ViewportId(MR.ViewportId i);
            _UnderlyingPtr = __MR_ViewportMask_Construct_MR_ViewportId(i);
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public static unsafe implicit operator Const_ViewportMask(MR.ViewportId i) {return new(i);}

        /// mask meaning all or any viewports
        /// Generated from method `MR::ViewportMask::all`.
        public static unsafe MR.ViewportMask All()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_all", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_all();
            return new(__MR_ViewportMask_all(), is_owning: true);
        }

        /// Generated from method `MR::ViewportMask::any`.
        public static unsafe MR.ViewportMask Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_any", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_any();
            return new(__MR_ViewportMask_any(), is_owning: true);
        }

        /// Generated from method `MR::ViewportMask::value`.
        public unsafe uint Value()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_value", ExactSpelling = true)]
            extern static uint __MR_ViewportMask_value(_Underlying *_this);
            return __MR_ViewportMask_value(_UnderlyingPtr);
        }

        /// Generated from method `MR::ViewportMask::empty`.
        public unsafe bool Empty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_empty", ExactSpelling = true)]
            extern static byte __MR_ViewportMask_empty(_Underlying *_this);
            return __MR_ViewportMask_empty(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ViewportMask::contains`.
        public unsafe bool Contains(MR.ViewportId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_contains", ExactSpelling = true)]
            extern static byte __MR_ViewportMask_contains(_Underlying *_this, MR.ViewportId id);
            return __MR_ViewportMask_contains(_UnderlyingPtr, id) != 0;
        }

        /// Generated from method `MR::ViewportMask::operator==`.
        public static unsafe bool operator==(MR.Const_ViewportMask _this, MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ViewportMask", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ViewportMask(MR.Const_ViewportMask._Underlying *_this, MR.ViewportMask._Underlying *b);
            return __MR_equal_MR_ViewportMask(_this._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ViewportMask _this, MR.Const_ViewportMask b)
        {
            return !(_this == b);
        }

        /// Generated from method `MR::ViewportMask::operator<`.
        public unsafe bool Less(MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_ViewportMask", ExactSpelling = true)]
            extern static byte __MR_less_MR_ViewportMask(_Underlying *_this, MR.ViewportMask._Underlying *b);
            return __MR_less_MR_ViewportMask(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ViewportMask::operator~`.
        public static unsafe MR.ViewportMask operator~(MR.Const_ViewportMask _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compl_MR_ViewportMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_compl_MR_ViewportMask(MR.Const_ViewportMask._Underlying *_this);
            return new(__MR_compl_MR_ViewportMask(_this._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator&`.
        public static unsafe MR.ViewportMask operator&(Const_ViewportMask a, MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_ViewportMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_bitand_MR_ViewportMask(MR.ViewportMask._Underlying *a, MR.ViewportMask._Underlying *b);
            return new(__MR_bitand_MR_ViewportMask(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator|`.
        public static unsafe MR.ViewportMask operator|(Const_ViewportMask a, MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_ViewportMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_bitor_MR_ViewportMask(MR.ViewportMask._Underlying *a, MR.ViewportMask._Underlying *b);
            return new(__MR_bitor_MR_ViewportMask(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::operator^`.
        public static unsafe MR.ViewportMask operator^(Const_ViewportMask a, MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_xor_MR_ViewportMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_xor_MR_ViewportMask(MR.ViewportMask._Underlying *a, MR.ViewportMask._Underlying *b);
            return new(__MR_xor_MR_ViewportMask(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true);
        }

        // IEquatable:

        public bool Equals(MR.Const_ViewportMask? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_ViewportMask)
                return this == (MR.Const_ViewportMask)other;
            return false;
        }
    }

    /// stores mask of viewport unique identifiers
    /// Generated from class `MR::ViewportMask`.
    /// This is the non-const half of the class.
    public class ViewportMask : Const_ViewportMask
    {
        internal unsafe ViewportMask(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref uint Mask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_GetMutable_mask_", ExactSpelling = true)]
                extern static uint *__MR_ViewportMask_GetMutable_mask_(_Underlying *_this);
                return ref *__MR_ViewportMask_GetMutable_mask_(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportMask() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportMask_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public unsafe ViewportMask(MR.Const_ViewportMask _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_ConstructFromAnother(MR.ViewportMask._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportMask_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public unsafe ViewportMask(uint i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_Construct_unsigned_int", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_Construct_unsigned_int(uint i);
            _UnderlyingPtr = __MR_ViewportMask_Construct_unsigned_int(i);
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public unsafe ViewportMask(MR.ViewportId i) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_Construct_MR_ViewportId", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_Construct_MR_ViewportId(MR.ViewportId i);
            _UnderlyingPtr = __MR_ViewportMask_Construct_MR_ViewportId(i);
        }

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public static unsafe implicit operator ViewportMask(MR.ViewportId i) {return new(i);}

        /// Generated from method `MR::ViewportMask::operator=`.
        public unsafe MR.ViewportMask Assign(MR.Const_ViewportMask _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_AssignFromAnother(_Underlying *_this, MR.ViewportMask._Underlying *_other);
            return new(__MR_ViewportMask_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ViewportMask::operator&=`.
        public unsafe MR.ViewportMask BitandAssign(MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_bitand_assign", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_bitand_assign(_Underlying *_this, MR.ViewportMask._Underlying *b);
            return new(__MR_ViewportMask_bitand_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ViewportMask::operator|=`.
        public unsafe MR.ViewportMask BitorAssign(MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_bitor_assign", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_bitor_assign(_Underlying *_this, MR.ViewportMask._Underlying *b);
            return new(__MR_ViewportMask_bitor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ViewportMask::operator^=`.
        public unsafe MR.ViewportMask XorAssign(MR.Const_ViewportMask b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_xor_assign", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportMask_xor_assign(_Underlying *_this, MR.ViewportMask._Underlying *b);
            return new(__MR_ViewportMask_xor_assign(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ViewportMask::set`.
        /// Parameter `on` defaults to `true`.
        public unsafe void Set(MR.ViewportId id, bool? on = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportMask_set", ExactSpelling = true)]
            extern static void __MR_ViewportMask_set(_Underlying *_this, MR.ViewportId id, byte *on);
            byte __deref_on = on.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ViewportMask_set(_UnderlyingPtr, id, on.HasValue ? &__deref_on : null);
        }
    }

    /// This is used for optional parameters of class `ViewportMask` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportMask`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportMask`/`Const_ViewportMask` directly.
    public class _InOptMut_ViewportMask
    {
        public ViewportMask? Opt;

        public _InOptMut_ViewportMask() {}
        public _InOptMut_ViewportMask(ViewportMask value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportMask(ViewportMask value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportMask` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportMask`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportMask`/`Const_ViewportMask` to pass it to the function.
    public class _InOptConst_ViewportMask
    {
        public Const_ViewportMask? Opt;

        public _InOptConst_ViewportMask() {}
        public _InOptConst_ViewportMask(Const_ViewportMask value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportMask(Const_ViewportMask value) {return new(value);}

        /// Generated from constructor `MR::ViewportMask::ViewportMask`.
        public static unsafe implicit operator _InOptConst_ViewportMask(MR.ViewportId i) {return new MR.ViewportMask(i);}
    }

    /// iterates over all ViewportIds in given ViewportMask
    /// Generated from class `MR::ViewportIterator`.
    /// This is the const half of the class.
    public class Const_ViewportIterator : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_ViewportIterator>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ViewportIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_ViewportIterator_Destroy(_Underlying *_this);
            __MR_ViewportIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ViewportIterator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ViewportIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportIterator_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public unsafe Const_ViewportIterator(MR.Const_ViewportIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_ConstructFromAnother(MR.ViewportIterator._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// constructs begin iterator
        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public unsafe Const_ViewportIterator(MR.Const_ViewportMask mask) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_Construct", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_Construct(MR.ViewportMask._Underlying *mask);
            _UnderlyingPtr = __MR_ViewportIterator_Construct(mask._UnderlyingPtr);
        }

        /// constructs begin iterator
        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public static unsafe implicit operator Const_ViewportIterator(MR.Const_ViewportMask mask) {return new(mask);}

        /// Generated from method `MR::ViewportIterator::operator++`.
        public static unsafe ViewportIterator operator++(MR.Const_ViewportIterator _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ViewportIterator", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_incr_MR_ViewportIterator(MR.Const_ViewportIterator._Underlying *_this);
            ViewportIterator _this_copy = new(_this);
            MR.ViewportIterator _unused_ret = new(__MR_incr_MR_ViewportIterator(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::ViewportIterator::mask`.
        public unsafe MR.ViewportMask Mask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_mask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ViewportIterator_mask(_Underlying *_this);
            return new(__MR_ViewportIterator_mask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ViewportIterator::operator*`.
        public unsafe MR.ViewportId Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_ViewportIterator", ExactSpelling = true)]
            extern static MR.ViewportId __MR_deref_MR_ViewportIterator(_Underlying *_this);
            return __MR_deref_MR_ViewportIterator(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_ViewportIterator a, MR.Const_ViewportIterator b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ViewportIterator", ExactSpelling = true)]
            extern static byte __MR_equal_MR_ViewportIterator(MR.Const_ViewportIterator._Underlying *a, MR.Const_ViewportIterator._Underlying *b);
            return __MR_equal_MR_ViewportIterator(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_ViewportIterator a, MR.Const_ViewportIterator b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_ViewportIterator? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_ViewportIterator)
                return this == (MR.Const_ViewportIterator)other;
            return false;
        }
    }

    /// iterates over all ViewportIds in given ViewportMask
    /// Generated from class `MR::ViewportIterator`.
    /// This is the non-const half of the class.
    public class ViewportIterator : Const_ViewportIterator
    {
        internal unsafe ViewportIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ViewportIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_ViewportIterator_DefaultConstruct();
        }

        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public unsafe ViewportIterator(MR.Const_ViewportIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_ConstructFromAnother(MR.ViewportIterator._Underlying *_other);
            _UnderlyingPtr = __MR_ViewportIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// constructs begin iterator
        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public unsafe ViewportIterator(MR.Const_ViewportMask mask) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_Construct", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_Construct(MR.ViewportMask._Underlying *mask);
            _UnderlyingPtr = __MR_ViewportIterator_Construct(mask._UnderlyingPtr);
        }

        /// constructs begin iterator
        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public static unsafe implicit operator ViewportIterator(MR.Const_ViewportMask mask) {return new(mask);}

        /// Generated from method `MR::ViewportIterator::operator=`.
        public unsafe MR.ViewportIterator Assign(MR.Const_ViewportIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ViewportIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_ViewportIterator_AssignFromAnother(_Underlying *_this, MR.ViewportIterator._Underlying *_other);
            return new(__MR_ViewportIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ViewportIterator::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_ViewportIterator", ExactSpelling = true)]
            extern static MR.ViewportIterator._Underlying *__MR_incr_MR_ViewportIterator(_Underlying *_this);
            MR.ViewportIterator _unused_ret = new(__MR_incr_MR_ViewportIterator(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ViewportIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ViewportIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportIterator`/`Const_ViewportIterator` directly.
    public class _InOptMut_ViewportIterator
    {
        public ViewportIterator? Opt;

        public _InOptMut_ViewportIterator() {}
        public _InOptMut_ViewportIterator(ViewportIterator value) {Opt = value;}
        public static implicit operator _InOptMut_ViewportIterator(ViewportIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `ViewportIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ViewportIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ViewportIterator`/`Const_ViewportIterator` to pass it to the function.
    public class _InOptConst_ViewportIterator
    {
        public Const_ViewportIterator? Opt;

        public _InOptConst_ViewportIterator() {}
        public _InOptConst_ViewportIterator(Const_ViewportIterator value) {Opt = value;}
        public static implicit operator _InOptConst_ViewportIterator(Const_ViewportIterator value) {return new(value);}

        /// constructs begin iterator
        /// Generated from constructor `MR::ViewportIterator::ViewportIterator`.
        public static unsafe implicit operator _InOptConst_ViewportIterator(MR.Const_ViewportMask mask) {return new MR.ViewportIterator(mask);}
    }

    /// Generated from function `MR::begin`.
    public static unsafe MR.ViewportIterator Begin(MR.Const_ViewportMask mask)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_begin_MR_ViewportMask", ExactSpelling = true)]
        extern static MR.ViewportIterator._Underlying *__MR_begin_MR_ViewportMask(MR.ViewportMask._Underlying *mask);
        return new(__MR_begin_MR_ViewportMask(mask._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::end`.
    public static unsafe MR.ViewportIterator End(MR.Const_ViewportMask _1)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_end_MR_ViewportMask", ExactSpelling = true)]
        extern static MR.ViewportIterator._Underlying *__MR_end_MR_ViewportMask(MR.ViewportMask._Underlying *_1);
        return new(__MR_end_MR_ViewportMask(_1._UnderlyingPtr), is_owning: true);
    }
}
