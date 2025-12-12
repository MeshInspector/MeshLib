public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::PolylineIntersectionResult2` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRPolylineIntersectionResult2 : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRPolylineIntersectionResult2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_PolylineIntersectionResult2_Destroy(_Underlying *_this);
                __MR_std_optional_MR_PolylineIntersectionResult2_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRPolylineIntersectionResult2() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRPolylineIntersectionResult2() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_PolylineIntersectionResult2_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRPolylineIntersectionResult2(MR.Std.Const_Optional_MRPolylineIntersectionResult2 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_ConstructFromAnother(MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_PolylineIntersectionResult2_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRPolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_ConstructFrom(MR.PolylineIntersectionResult2._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_PolylineIntersectionResult2_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRPolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_PolylineIntersectionResult2? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_Value", ExactSpelling = true)]
                extern static MR.Const_PolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_PolylineIntersectionResult2_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_PolylineIntersectionResult2(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::PolylineIntersectionResult2` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRPolylineIntersectionResult2 : Const_Optional_MRPolylineIntersectionResult2
        {
            internal unsafe Optional_MRPolylineIntersectionResult2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRPolylineIntersectionResult2() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_PolylineIntersectionResult2_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRPolylineIntersectionResult2(MR.Std.Const_Optional_MRPolylineIntersectionResult2 other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_ConstructFromAnother(MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_PolylineIntersectionResult2_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRPolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_ConstructFrom(MR.PolylineIntersectionResult2._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_PolylineIntersectionResult2_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRPolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRPolylineIntersectionResult2 other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_PolylineIntersectionResult2_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *other);
                __MR_std_optional_MR_PolylineIntersectionResult2_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.Const_PolylineIntersectionResult2? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_PolylineIntersectionResult2_AssignFrom(_Underlying *_this, MR.PolylineIntersectionResult2._Underlying *other);
                __MR_std_optional_MR_PolylineIntersectionResult2_AssignFrom(_UnderlyingPtr, other is not null ? other._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.PolylineIntersectionResult2? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_PolylineIntersectionResult2_MutableValue", ExactSpelling = true)]
                extern static MR.PolylineIntersectionResult2._Underlying *__MR_std_optional_MR_PolylineIntersectionResult2_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_PolylineIntersectionResult2_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.PolylineIntersectionResult2(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRPolylineIntersectionResult2` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRPolylineIntersectionResult2`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRPolylineIntersectionResult2`/`Const_Optional_MRPolylineIntersectionResult2` directly.
        public class _InOptMut_Optional_MRPolylineIntersectionResult2
        {
            public Optional_MRPolylineIntersectionResult2? Opt;

            public _InOptMut_Optional_MRPolylineIntersectionResult2() {}
            public _InOptMut_Optional_MRPolylineIntersectionResult2(Optional_MRPolylineIntersectionResult2 value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRPolylineIntersectionResult2(Optional_MRPolylineIntersectionResult2 value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRPolylineIntersectionResult2` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRPolylineIntersectionResult2`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRPolylineIntersectionResult2`/`Const_Optional_MRPolylineIntersectionResult2` to pass it to the function.
        public class _InOptConst_Optional_MRPolylineIntersectionResult2
        {
            public Const_Optional_MRPolylineIntersectionResult2? Opt;

            public _InOptConst_Optional_MRPolylineIntersectionResult2() {}
            public _InOptConst_Optional_MRPolylineIntersectionResult2(Const_Optional_MRPolylineIntersectionResult2 value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRPolylineIntersectionResult2(Const_Optional_MRPolylineIntersectionResult2 value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRPolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2? other) {return new MR.Std.Optional_MRPolylineIntersectionResult2(other);}
        }
    }
}
