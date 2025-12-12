public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::Color` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRColor : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRColor(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Color_Destroy(_Underlying *_this);
                __MR_std_optional_MR_Color_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRColor() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRColor() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_std_optional_MR_Color_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Color_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRColor(MR.Std.Const_Optional_MRColor other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_std_optional_MR_Color_ConstructFromAnother(MR.Std.Optional_MRColor._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Color_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRColor(MR._InOpt_Color other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_std_optional_MR_Color_ConstructFrom(MR.Color *other);
                _UnderlyingPtr = __MR_std_optional_MR_Color_ConstructFrom(other.HasValue ? &other.Object : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRColor(MR._InOpt_Color other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_Color? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_Value", ExactSpelling = true)]
                extern static MR.Const_Color._Underlying *__MR_std_optional_MR_Color_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Color_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Color(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::Color` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRColor : Const_Optional_MRColor
        {
            internal unsafe Optional_MRColor(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRColor() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_std_optional_MR_Color_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Color_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRColor(MR.Std.Const_Optional_MRColor other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_std_optional_MR_Color_ConstructFromAnother(MR.Std.Optional_MRColor._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Color_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRColor(MR._InOpt_Color other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRColor._Underlying *__MR_std_optional_MR_Color_ConstructFrom(MR.Color *other);
                _UnderlyingPtr = __MR_std_optional_MR_Color_ConstructFrom(other.HasValue ? &other.Object : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRColor(MR._InOpt_Color other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRColor other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Color_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRColor._Underlying *other);
                __MR_std_optional_MR_Color_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR._InOpt_Color other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Color_AssignFrom(_Underlying *_this, MR.Color *other);
                __MR_std_optional_MR_Color_AssignFrom(_UnderlyingPtr, other.HasValue ? &other.Object : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Mut_Color? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Color_MutableValue", ExactSpelling = true)]
                extern static MR.Mut_Color._Underlying *__MR_std_optional_MR_Color_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Color_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Color(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRColor` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRColor`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRColor`/`Const_Optional_MRColor` directly.
        public class _InOptMut_Optional_MRColor
        {
            public Optional_MRColor? Opt;

            public _InOptMut_Optional_MRColor() {}
            public _InOptMut_Optional_MRColor(Optional_MRColor value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRColor(Optional_MRColor value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRColor` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRColor`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRColor`/`Const_Optional_MRColor` to pass it to the function.
        public class _InOptConst_Optional_MRColor
        {
            public Const_Optional_MRColor? Opt;

            public _InOptConst_Optional_MRColor() {}
            public _InOptConst_Optional_MRColor(Const_Optional_MRColor value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRColor(Const_Optional_MRColor value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRColor(MR._InOpt_Color other) {return new MR.Std.Optional_MRColor(other);}
        }
    }
}
