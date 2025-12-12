public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::Box1f` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRBox1f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRBox1f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Box1f_Destroy(_Underlying *_this);
                __MR_std_optional_MR_Box1f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRBox1f() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRBox1f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRBox1f._Underlying *__MR_std_optional_MR_Box1f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Box1f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRBox1f(MR.Std.Const_Optional_MRBox1f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRBox1f._Underlying *__MR_std_optional_MR_Box1f_ConstructFromAnother(MR.Std.Optional_MRBox1f._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Box1f_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRBox1f(MR._InOpt_Box1f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRBox1f._Underlying *__MR_std_optional_MR_Box1f_ConstructFrom(MR.Box1f *other);
                _UnderlyingPtr = __MR_std_optional_MR_Box1f_ConstructFrom(other.HasValue ? &other.Object : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRBox1f(MR._InOpt_Box1f other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_Box1f? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_Value", ExactSpelling = true)]
                extern static MR.Const_Box1f._Underlying *__MR_std_optional_MR_Box1f_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Box1f_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Box1f(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::Box1f` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRBox1f : Const_Optional_MRBox1f
        {
            internal unsafe Optional_MRBox1f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRBox1f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRBox1f._Underlying *__MR_std_optional_MR_Box1f_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Box1f_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRBox1f(MR.Std.Const_Optional_MRBox1f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRBox1f._Underlying *__MR_std_optional_MR_Box1f_ConstructFromAnother(MR.Std.Optional_MRBox1f._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Box1f_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRBox1f(MR._InOpt_Box1f other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRBox1f._Underlying *__MR_std_optional_MR_Box1f_ConstructFrom(MR.Box1f *other);
                _UnderlyingPtr = __MR_std_optional_MR_Box1f_ConstructFrom(other.HasValue ? &other.Object : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRBox1f(MR._InOpt_Box1f other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRBox1f other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Box1f_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRBox1f._Underlying *other);
                __MR_std_optional_MR_Box1f_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR._InOpt_Box1f other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Box1f_AssignFrom(_Underlying *_this, MR.Box1f *other);
                __MR_std_optional_MR_Box1f_AssignFrom(_UnderlyingPtr, other.HasValue ? &other.Object : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Mut_Box1f? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Box1f_MutableValue", ExactSpelling = true)]
                extern static MR.Mut_Box1f._Underlying *__MR_std_optional_MR_Box1f_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Box1f_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Box1f(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRBox1f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRBox1f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRBox1f`/`Const_Optional_MRBox1f` directly.
        public class _InOptMut_Optional_MRBox1f
        {
            public Optional_MRBox1f? Opt;

            public _InOptMut_Optional_MRBox1f() {}
            public _InOptMut_Optional_MRBox1f(Optional_MRBox1f value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRBox1f(Optional_MRBox1f value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRBox1f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRBox1f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRBox1f`/`Const_Optional_MRBox1f` to pass it to the function.
        public class _InOptConst_Optional_MRBox1f
        {
            public Const_Optional_MRBox1f? Opt;

            public _InOptConst_Optional_MRBox1f() {}
            public _InOptConst_Optional_MRBox1f(Const_Optional_MRBox1f value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRBox1f(Const_Optional_MRBox1f value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRBox1f(MR._InOpt_Box1f other) {return new MR.Std.Optional_MRBox1f(other);}
        }
    }
}
