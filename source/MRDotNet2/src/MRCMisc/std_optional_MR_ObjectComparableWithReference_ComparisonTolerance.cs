public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::ObjectComparableWithReference::ComparisonTolerance` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRObjectComparableWithReferenceComparisonTolerance : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRObjectComparableWithReferenceComparisonTolerance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_Destroy(_Underlying *_this);
                __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRObjectComparableWithReferenceComparisonTolerance() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRObjectComparableWithReferenceComparisonTolerance() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.Std.Const_Optional_MRObjectComparableWithReferenceComparisonTolerance other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.ObjectComparableWithReference.Const_ComparisonTolerance? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_Value", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.Const_ComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.ObjectComparableWithReference.Const_ComparisonTolerance(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::ObjectComparableWithReference::ComparisonTolerance` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRObjectComparableWithReferenceComparisonTolerance : Const_Optional_MRObjectComparableWithReferenceComparisonTolerance
        {
            internal unsafe Optional_MRObjectComparableWithReferenceComparisonTolerance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRObjectComparableWithReferenceComparisonTolerance() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.Std.Const_Optional_MRObjectComparableWithReferenceComparisonTolerance other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(other is not null ? other._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRObjectComparableWithReferenceComparisonTolerance other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *other);
                __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.ObjectComparableWithReference.Const_ComparisonTolerance? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_AssignFrom(_Underlying *_this, MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *other);
                __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_AssignFrom(_UnderlyingPtr, other is not null ? other._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.ObjectComparableWithReference.ComparisonTolerance? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_MutableValue", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_ObjectComparableWithReference_ComparisonTolerance_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.ObjectComparableWithReference.ComparisonTolerance(__ret, is_owning: false) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRObjectComparableWithReferenceComparisonTolerance` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRObjectComparableWithReferenceComparisonTolerance`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRObjectComparableWithReferenceComparisonTolerance`/`Const_Optional_MRObjectComparableWithReferenceComparisonTolerance` directly.
        public class _InOptMut_Optional_MRObjectComparableWithReferenceComparisonTolerance
        {
            public Optional_MRObjectComparableWithReferenceComparisonTolerance? Opt;

            public _InOptMut_Optional_MRObjectComparableWithReferenceComparisonTolerance() {}
            public _InOptMut_Optional_MRObjectComparableWithReferenceComparisonTolerance(Optional_MRObjectComparableWithReferenceComparisonTolerance value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRObjectComparableWithReferenceComparisonTolerance(Optional_MRObjectComparableWithReferenceComparisonTolerance value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRObjectComparableWithReferenceComparisonTolerance` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRObjectComparableWithReferenceComparisonTolerance`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRObjectComparableWithReferenceComparisonTolerance`/`Const_Optional_MRObjectComparableWithReferenceComparisonTolerance` to pass it to the function.
        public class _InOptConst_Optional_MRObjectComparableWithReferenceComparisonTolerance
        {
            public Const_Optional_MRObjectComparableWithReferenceComparisonTolerance? Opt;

            public _InOptConst_Optional_MRObjectComparableWithReferenceComparisonTolerance() {}
            public _InOptConst_Optional_MRObjectComparableWithReferenceComparisonTolerance(Const_Optional_MRObjectComparableWithReferenceComparisonTolerance value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRObjectComparableWithReferenceComparisonTolerance(Const_Optional_MRObjectComparableWithReferenceComparisonTolerance value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRObjectComparableWithReferenceComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance? other) {return new MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance(other);}
        }
    }
}
