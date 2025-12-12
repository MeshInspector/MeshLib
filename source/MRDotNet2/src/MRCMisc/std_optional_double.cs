public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `double` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_Double : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_Double(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_double_Destroy(_Underlying *_this);
                __MR_std_optional_double_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_Double() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_Double() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_Double._Underlying *__MR_std_optional_double_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_double_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_Double(MR.Std.Const_Optional_Double other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_Double._Underlying *__MR_std_optional_double_ConstructFromAnother(MR.Std.Optional_Double._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_double_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_Double(double? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_Double._Underlying *__MR_std_optional_double_ConstructFrom(double *other);
                double __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_double_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_Double(double? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe double? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_Value", ExactSpelling = true)]
                extern static double *__MR_std_optional_double_Value(_Underlying *_this);
                var __ret = __MR_std_optional_double_Value(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Stores either a single `double` or nothing.
        /// This is the non-const half of the class.
        public class Optional_Double : Const_Optional_Double
        {
            internal unsafe Optional_Double(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_Double() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_Double._Underlying *__MR_std_optional_double_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_double_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_Double(MR.Std.Const_Optional_Double other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_Double._Underlying *__MR_std_optional_double_ConstructFromAnother(MR.Std.Optional_Double._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_double_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_Double(double? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_Double._Underlying *__MR_std_optional_double_ConstructFrom(double *other);
                double __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_double_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_Double(double? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_Double other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_double_AssignFromAnother(_Underlying *_this, MR.Std.Optional_Double._Underlying *other);
                __MR_std_optional_double_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(double? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_double_AssignFrom(_Underlying *_this, double *other);
                double __deref_other = other.GetValueOrDefault();
                __MR_std_optional_double_AssignFrom(_UnderlyingPtr, other.HasValue ? &__deref_other : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Misc.Ref<double>? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_double_MutableValue", ExactSpelling = true)]
                extern static double *__MR_std_optional_double_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_double_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<double>(__ret) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_Double` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_Double`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_Double`/`Const_Optional_Double` directly.
        public class _InOptMut_Optional_Double
        {
            public Optional_Double? Opt;

            public _InOptMut_Optional_Double() {}
            public _InOptMut_Optional_Double(Optional_Double value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_Double(Optional_Double value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_Double` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_Double`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_Double`/`Const_Optional_Double` to pass it to the function.
        public class _InOptConst_Optional_Double
        {
            public Const_Optional_Double? Opt;

            public _InOptConst_Optional_Double() {}
            public _InOptConst_Optional_Double(Const_Optional_Double value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_Double(Const_Optional_Double value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_Double(double? other) {return new MR.Std.Optional_Double(other);}
        }
    }
}
