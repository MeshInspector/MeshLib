public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::SignedDistanceToMeshResult` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRSignedDistanceToMeshResult : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRSignedDistanceToMeshResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_SignedDistanceToMeshResult_Destroy(_Underlying *_this);
                __MR_std_optional_MR_SignedDistanceToMeshResult_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRSignedDistanceToMeshResult() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRSignedDistanceToMeshResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_SignedDistanceToMeshResult_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRSignedDistanceToMeshResult(MR.Std.Const_Optional_MRSignedDistanceToMeshResult other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFromAnother(MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRSignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFrom(MR.SignedDistanceToMeshResult._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRSignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_SignedDistanceToMeshResult? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_Value", ExactSpelling = true)]
                extern static MR.Const_SignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_SignedDistanceToMeshResult_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_SignedDistanceToMeshResult(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::SignedDistanceToMeshResult` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRSignedDistanceToMeshResult : Const_Optional_MRSignedDistanceToMeshResult
        {
            internal unsafe Optional_MRSignedDistanceToMeshResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRSignedDistanceToMeshResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_SignedDistanceToMeshResult_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRSignedDistanceToMeshResult(MR.Std.Const_Optional_MRSignedDistanceToMeshResult other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFromAnother(MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRSignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFrom(MR.SignedDistanceToMeshResult._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_SignedDistanceToMeshResult_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRSignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRSignedDistanceToMeshResult other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_SignedDistanceToMeshResult_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *other);
                __MR_std_optional_MR_SignedDistanceToMeshResult_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.Const_SignedDistanceToMeshResult? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_SignedDistanceToMeshResult_AssignFrom(_Underlying *_this, MR.SignedDistanceToMeshResult._Underlying *other);
                __MR_std_optional_MR_SignedDistanceToMeshResult_AssignFrom(_UnderlyingPtr, other is not null ? other._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.SignedDistanceToMeshResult? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_SignedDistanceToMeshResult_MutableValue", ExactSpelling = true)]
                extern static MR.SignedDistanceToMeshResult._Underlying *__MR_std_optional_MR_SignedDistanceToMeshResult_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_SignedDistanceToMeshResult_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.SignedDistanceToMeshResult(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRSignedDistanceToMeshResult` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRSignedDistanceToMeshResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRSignedDistanceToMeshResult`/`Const_Optional_MRSignedDistanceToMeshResult` directly.
        public class _InOptMut_Optional_MRSignedDistanceToMeshResult
        {
            public Optional_MRSignedDistanceToMeshResult? Opt;

            public _InOptMut_Optional_MRSignedDistanceToMeshResult() {}
            public _InOptMut_Optional_MRSignedDistanceToMeshResult(Optional_MRSignedDistanceToMeshResult value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRSignedDistanceToMeshResult(Optional_MRSignedDistanceToMeshResult value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRSignedDistanceToMeshResult` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRSignedDistanceToMeshResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRSignedDistanceToMeshResult`/`Const_Optional_MRSignedDistanceToMeshResult` to pass it to the function.
        public class _InOptConst_Optional_MRSignedDistanceToMeshResult
        {
            public Const_Optional_MRSignedDistanceToMeshResult? Opt;

            public _InOptConst_Optional_MRSignedDistanceToMeshResult() {}
            public _InOptConst_Optional_MRSignedDistanceToMeshResult(Const_Optional_MRSignedDistanceToMeshResult value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRSignedDistanceToMeshResult(Const_Optional_MRSignedDistanceToMeshResult value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRSignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult? other) {return new MR.Std.Optional_MRSignedDistanceToMeshResult(other);}
        }
    }
}
