public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `float` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_Float : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_float_Destroy(_Underlying *_this);
                __MR_std_optional_float_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_Float() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_Float() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_std_optional_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_Float(MR.Std.Const_Optional_Float other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_std_optional_float_ConstructFromAnother(MR.Std.Optional_Float._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_float_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_Float(float? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_std_optional_float_ConstructFrom(float *other);
                float __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_float_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_Float(float? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe float? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_Value", ExactSpelling = true)]
                extern static float *__MR_std_optional_float_Value(_Underlying *_this);
                var __ret = __MR_std_optional_float_Value(_UnderlyingPtr);
                return __ret is not null ? *__ret : null;
            }
        }

        /// Stores either a single `float` or nothing.
        /// This is the non-const half of the class.
        public class Optional_Float : Const_Optional_Float
        {
            internal unsafe Optional_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_Float() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_std_optional_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_Float(MR.Std.Const_Optional_Float other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_std_optional_float_ConstructFromAnother(MR.Std.Optional_Float._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_float_ConstructFromAnother(other._UnderlyingPtr);
            }

            /// Constructs a new instance.
            public unsafe Optional_Float(float? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_std_optional_float_ConstructFrom(float *other);
                float __deref_other = other.GetValueOrDefault();
                _UnderlyingPtr = __MR_std_optional_float_ConstructFrom(other.HasValue ? &__deref_other : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_Float(float? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std.Const_Optional_Float other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_float_AssignFromAnother(_Underlying *_this, MR.Std.Optional_Float._Underlying *other);
                __MR_std_optional_float_AssignFromAnother(_UnderlyingPtr, other._UnderlyingPtr);
            }

            /// Assigns the contents.
            public unsafe void Assign(float? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_float_AssignFrom(_Underlying *_this, float *other);
                float __deref_other = other.GetValueOrDefault();
                __MR_std_optional_float_AssignFrom(_UnderlyingPtr, other.HasValue ? &__deref_other : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Misc.Ref<float>? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_float_MutableValue", ExactSpelling = true)]
                extern static float *__MR_std_optional_float_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_float_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Misc.Ref<float>(__ret) : null;
            }
        }

        /// This is used for optional parameters of class `Optional_Float` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_Float`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_Float`/`Const_Optional_Float` directly.
        public class _InOptMut_Optional_Float
        {
            public Optional_Float? Opt;

            public _InOptMut_Optional_Float() {}
            public _InOptMut_Optional_Float(Optional_Float value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_Float(Optional_Float value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_Float` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_Float`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_Float`/`Const_Optional_Float` to pass it to the function.
        public class _InOptConst_Optional_Float
        {
            public Const_Optional_Float? Opt;

            public _InOptConst_Optional_Float() {}
            public _InOptConst_Optional_Float(Const_Optional_Float value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_Float(Const_Optional_Float value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_Float(float? other) {return new MR.Std.Optional_Float(other);}
        }
    }
}
