public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::VertBitSet` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRVertBitSet : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRVertBitSet(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_VertBitSet_Destroy(_Underlying *_this);
                __MR_std_optional_MR_VertBitSet_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRVertBitSet() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRVertBitSet() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_VertBitSet_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRVertBitSet(MR.Std._ByValue_Optional_MRVertBitSet other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRVertBitSet._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertBitSet_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRVertBitSet(MR._ByValue_VertBitSet? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.VertBitSet._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertBitSet_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRVertBitSet(MR._ByValue_VertBitSet? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_VertBitSet? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_Value", ExactSpelling = true)]
                extern static MR.Const_VertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_VertBitSet_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_VertBitSet(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::VertBitSet` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRVertBitSet : Const_Optional_MRVertBitSet
        {
            internal unsafe Optional_MRVertBitSet(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRVertBitSet() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_VertBitSet_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRVertBitSet(MR.Std._ByValue_Optional_MRVertBitSet other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRVertBitSet._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertBitSet_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRVertBitSet(MR._ByValue_VertBitSet? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.VertBitSet._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertBitSet_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRVertBitSet(MR._ByValue_VertBitSet? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Optional_MRVertBitSet other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_VertBitSet_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRVertBitSet._Underlying *other);
                __MR_std_optional_MR_VertBitSet_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR._ByValue_VertBitSet? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_VertBitSet_AssignFrom(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.VertBitSet._Underlying *other);
                __MR_std_optional_MR_VertBitSet_AssignFrom(_UnderlyingPtr, other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.VertBitSet? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertBitSet_MutableValue", ExactSpelling = true)]
                extern static MR.VertBitSet._Underlying *__MR_std_optional_MR_VertBitSet_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_VertBitSet_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.VertBitSet(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Optional_MRVertBitSet` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Optional_MRVertBitSet`/`Const_Optional_MRVertBitSet` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Optional_MRVertBitSet
        {
            internal readonly Const_Optional_MRVertBitSet? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Optional_MRVertBitSet() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Optional_MRVertBitSet(Const_Optional_MRVertBitSet new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Optional_MRVertBitSet(Const_Optional_MRVertBitSet arg) {return new(arg);}
            public _ByValue_Optional_MRVertBitSet(MR.Misc._Moved<Optional_MRVertBitSet> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Optional_MRVertBitSet(MR.Misc._Moved<Optional_MRVertBitSet> arg) {return new(arg);}

            /// Constructs a new instance.
            public static unsafe implicit operator _ByValue_Optional_MRVertBitSet(MR._ByValue_VertBitSet? other) {return new MR.Std.Optional_MRVertBitSet(other);}
        }

        /// This is used for optional parameters of class `Optional_MRVertBitSet` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRVertBitSet`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRVertBitSet`/`Const_Optional_MRVertBitSet` directly.
        public class _InOptMut_Optional_MRVertBitSet
        {
            public Optional_MRVertBitSet? Opt;

            public _InOptMut_Optional_MRVertBitSet() {}
            public _InOptMut_Optional_MRVertBitSet(Optional_MRVertBitSet value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRVertBitSet(Optional_MRVertBitSet value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRVertBitSet` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRVertBitSet`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRVertBitSet`/`Const_Optional_MRVertBitSet` to pass it to the function.
        public class _InOptConst_Optional_MRVertBitSet
        {
            public Const_Optional_MRVertBitSet? Opt;

            public _InOptConst_Optional_MRVertBitSet() {}
            public _InOptConst_Optional_MRVertBitSet(Const_Optional_MRVertBitSet value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRVertBitSet(Const_Optional_MRVertBitSet value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRVertBitSet(MR._ByValue_VertBitSet? other) {return new MR.Std.Optional_MRVertBitSet(other);}
        }
    }
}
