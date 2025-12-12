public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::Vector3i` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRVector3i : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRVector3i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Vector3i_Destroy(_Underlying *_this);
                __MR_std_optional_MR_Vector3i_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRVector3i() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRVector3i() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3i._Underlying *__MR_std_optional_MR_Vector3i_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Vector3i_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRVector3i(MR.Std.Const_Optional_MRVector3i other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3i._Underlying *__MR_std_optional_MR_Vector3i_ConstructFromAnother(MR.Std.Optional_MRVector3i._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Vector3i_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRVector3i(MR._InOpt_Vector3i other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3i._Underlying *__MR_std_optional_MR_Vector3i_ConstructFrom(MR.Vector3i *other);
                _UnderlyingPtr = __MR_std_optional_MR_Vector3i_ConstructFrom(other.HasValue ? &other.Object : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRVector3i(MR._InOpt_Vector3i other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_Vector3i? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_Value", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_std_optional_MR_Vector3i_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Vector3i_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Vector3i(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::Vector3i` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRVector3i : Const_Optional_MRVector3i
        {
            internal unsafe Optional_MRVector3i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRVector3i() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3i._Underlying *__MR_std_optional_MR_Vector3i_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Vector3i_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRVector3i(MR.Std.Const_Optional_MRVector3i other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3i._Underlying *__MR_std_optional_MR_Vector3i_ConstructFromAnother(MR.Std.Optional_MRVector3i._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Vector3i_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRVector3i(MR._InOpt_Vector3i other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3i._Underlying *__MR_std_optional_MR_Vector3i_ConstructFrom(MR.Vector3i *other);
                _UnderlyingPtr = __MR_std_optional_MR_Vector3i_ConstructFrom(other.HasValue ? &other.Object : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRVector3i(MR._InOpt_Vector3i other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRVector3i other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Vector3i_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRVector3i._Underlying *other);
                __MR_std_optional_MR_Vector3i_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR._InOpt_Vector3i other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Vector3i_AssignFrom(_Underlying *_this, MR.Vector3i *other);
                __MR_std_optional_MR_Vector3i_AssignFrom(_UnderlyingPtr, other.HasValue ? &other.Object : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Mut_Vector3i? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Vector3i_MutableValue", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_std_optional_MR_Vector3i_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Vector3i_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Mut_Vector3i(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRVector3i` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRVector3i`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRVector3i`/`Const_Optional_MRVector3i` directly.
        public class _InOptMut_Optional_MRVector3i
        {
            public Optional_MRVector3i? Opt;

            public _InOptMut_Optional_MRVector3i() {}
            public _InOptMut_Optional_MRVector3i(Optional_MRVector3i value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRVector3i(Optional_MRVector3i value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRVector3i` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRVector3i`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRVector3i`/`Const_Optional_MRVector3i` to pass it to the function.
        public class _InOptConst_Optional_MRVector3i
        {
            public Const_Optional_MRVector3i? Opt;

            public _InOptConst_Optional_MRVector3i() {}
            public _InOptConst_Optional_MRVector3i(Const_Optional_MRVector3i value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRVector3i(Const_Optional_MRVector3i value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRVector3i(MR._InOpt_Vector3i other) {return new MR.Std.Optional_MRVector3i(other);}
        }
    }
}
