public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::DimensionsVisualizePropertyType` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRDimensionsVisualizePropertyType : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRDimensionsVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_DimensionsVisualizePropertyType_Destroy(_Underlying *_this);
                __MR_std_optional_MR_DimensionsVisualizePropertyType_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRDimensionsVisualizePropertyType() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRDimensionsVisualizePropertyType() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_std_optional_MR_DimensionsVisualizePropertyType_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_DimensionsVisualizePropertyType_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRDimensionsVisualizePropertyType(MR.Std.Const_Optional_MRDimensionsVisualizePropertyType other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFromAnother(MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRDimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFrom(MR.DimensionsVisualizePropertyType *other);
                MR.DimensionsVisualizePropertyType __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRDimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.DimensionsVisualizePropertyType? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_Value", ExactSpelling = true)]
                extern static MR.DimensionsVisualizePropertyType *__MR_std_optional_MR_DimensionsVisualizePropertyType_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_DimensionsVisualizePropertyType_Value(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Stores either a single `MR::DimensionsVisualizePropertyType` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRDimensionsVisualizePropertyType : Const_Optional_MRDimensionsVisualizePropertyType
        {
            internal unsafe Optional_MRDimensionsVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRDimensionsVisualizePropertyType() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_std_optional_MR_DimensionsVisualizePropertyType_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_DimensionsVisualizePropertyType_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRDimensionsVisualizePropertyType(MR.Std.Const_Optional_MRDimensionsVisualizePropertyType other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFromAnother(MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRDimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFrom(MR.DimensionsVisualizePropertyType *other);
                MR.DimensionsVisualizePropertyType __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_MR_DimensionsVisualizePropertyType_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRDimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRDimensionsVisualizePropertyType other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_DimensionsVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *other);
                __MR_std_optional_MR_DimensionsVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.DimensionsVisualizePropertyType? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_DimensionsVisualizePropertyType_AssignFrom(_Underlying *_this, MR.DimensionsVisualizePropertyType *other);
                MR.DimensionsVisualizePropertyType __deref_other = other.GetValueOrDefault();
                __MR_std_optional_MR_DimensionsVisualizePropertyType_AssignFrom(_UnderlyingPtr, other.HasValue ? &__deref_other : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Misc.Ref<MR.DimensionsVisualizePropertyType>? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_DimensionsVisualizePropertyType_MutableValue", ExactSpelling = true)]
                extern static MR.DimensionsVisualizePropertyType *__MR_std_optional_MR_DimensionsVisualizePropertyType_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_DimensionsVisualizePropertyType_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<MR.DimensionsVisualizePropertyType>(__ret) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRDimensionsVisualizePropertyType` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRDimensionsVisualizePropertyType`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRDimensionsVisualizePropertyType`/`Const_Optional_MRDimensionsVisualizePropertyType` directly.
        public class _InOptMut_Optional_MRDimensionsVisualizePropertyType
        {
            public Optional_MRDimensionsVisualizePropertyType? Opt;

            public _InOptMut_Optional_MRDimensionsVisualizePropertyType() {}
            public _InOptMut_Optional_MRDimensionsVisualizePropertyType(Optional_MRDimensionsVisualizePropertyType value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRDimensionsVisualizePropertyType(Optional_MRDimensionsVisualizePropertyType value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRDimensionsVisualizePropertyType` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRDimensionsVisualizePropertyType`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRDimensionsVisualizePropertyType`/`Const_Optional_MRDimensionsVisualizePropertyType` to pass it to the function.
        public class _InOptConst_Optional_MRDimensionsVisualizePropertyType
        {
            public Const_Optional_MRDimensionsVisualizePropertyType? Opt;

            public _InOptConst_Optional_MRDimensionsVisualizePropertyType() {}
            public _InOptConst_Optional_MRDimensionsVisualizePropertyType(Const_Optional_MRDimensionsVisualizePropertyType value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRDimensionsVisualizePropertyType(Const_Optional_MRDimensionsVisualizePropertyType value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRDimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType? other) {return new MR.Std.Optional_MRDimensionsVisualizePropertyType(other);}
        }
    }
}
