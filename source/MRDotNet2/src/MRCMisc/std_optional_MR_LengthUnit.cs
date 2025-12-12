public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::LengthUnit` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRLengthUnit : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRLengthUnit(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_LengthUnit_Destroy(_Underlying *_this);
                __MR_std_optional_MR_LengthUnit_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRLengthUnit() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRLengthUnit() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_std_optional_MR_LengthUnit_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_LengthUnit_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRLengthUnit(MR.Std.Const_Optional_MRLengthUnit other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_std_optional_MR_LengthUnit_ConstructFromAnother(MR.Std.Optional_MRLengthUnit._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_LengthUnit_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRLengthUnit(MR.LengthUnit? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_std_optional_MR_LengthUnit_ConstructFrom(MR.LengthUnit *other);
                MR.LengthUnit __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_MR_LengthUnit_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRLengthUnit(MR.LengthUnit? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.LengthUnit? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_Value", ExactSpelling = true)]
                extern static MR.LengthUnit *__MR_std_optional_MR_LengthUnit_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_LengthUnit_Value(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Stores either a single `MR::LengthUnit` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRLengthUnit : Const_Optional_MRLengthUnit
        {
            internal unsafe Optional_MRLengthUnit(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRLengthUnit() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_std_optional_MR_LengthUnit_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_LengthUnit_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRLengthUnit(MR.Std.Const_Optional_MRLengthUnit other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_std_optional_MR_LengthUnit_ConstructFromAnother(MR.Std.Optional_MRLengthUnit._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_LengthUnit_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRLengthUnit(MR.LengthUnit? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_std_optional_MR_LengthUnit_ConstructFrom(MR.LengthUnit *other);
                MR.LengthUnit __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_MR_LengthUnit_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRLengthUnit(MR.LengthUnit? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_MRLengthUnit other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_LengthUnit_AssignFromAnother(_Underlying *_this, MR.Std.Optional_MRLengthUnit._Underlying *other);
                __MR_std_optional_MR_LengthUnit_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.LengthUnit? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_LengthUnit_AssignFrom(_Underlying *_this, MR.LengthUnit *other);
                MR.LengthUnit __deref_other = other.GetValueOrDefault();
                __MR_std_optional_MR_LengthUnit_AssignFrom(_UnderlyingPtr, other.HasValue ? &__deref_other : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Misc.Ref<MR.LengthUnit>? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_LengthUnit_MutableValue", ExactSpelling = true)]
                extern static MR.LengthUnit *__MR_std_optional_MR_LengthUnit_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_LengthUnit_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<MR.LengthUnit>(__ret) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_MRLengthUnit` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRLengthUnit`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRLengthUnit`/`Const_Optional_MRLengthUnit` directly.
        public class _InOptMut_Optional_MRLengthUnit
        {
            public Optional_MRLengthUnit? Opt;

            public _InOptMut_Optional_MRLengthUnit() {}
            public _InOptMut_Optional_MRLengthUnit(Optional_MRLengthUnit value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRLengthUnit(Optional_MRLengthUnit value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRLengthUnit` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRLengthUnit`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRLengthUnit`/`Const_Optional_MRLengthUnit` to pass it to the function.
        public class _InOptConst_Optional_MRLengthUnit
        {
            public Const_Optional_MRLengthUnit? Opt;

            public _InOptConst_Optional_MRLengthUnit() {}
            public _InOptConst_Optional_MRLengthUnit(Const_Optional_MRLengthUnit value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRLengthUnit(Const_Optional_MRLengthUnit value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRLengthUnit(MR.LengthUnit? other) {return new MR.Std.Optional_MRLengthUnit(other);}
        }
    }
}
