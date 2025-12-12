public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::VertCoords` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRVertCoords : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRVertCoords(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_VertCoords_Destroy(_Underlying *_this);
                __MR_std_optional_MR_VertCoords_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRVertCoords() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRVertCoords() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_std_optional_MR_VertCoords_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_VertCoords_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRVertCoords(MR.Std._ByValue_Optional_MRVertCoords other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_std_optional_MR_VertCoords_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRVertCoords._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertCoords_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRVertCoords(MR._ByValue_VertCoords? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_std_optional_MR_VertCoords_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.VertCoords._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertCoords_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRVertCoords(MR._ByValue_VertCoords? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_VertCoords? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_Value", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_std_optional_MR_VertCoords_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_VertCoords_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_VertCoords(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::VertCoords` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRVertCoords : Const_Optional_MRVertCoords
        {
            internal unsafe Optional_MRVertCoords(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRVertCoords() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_std_optional_MR_VertCoords_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_VertCoords_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRVertCoords(MR.Std._ByValue_Optional_MRVertCoords other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_std_optional_MR_VertCoords_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRVertCoords._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertCoords_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRVertCoords(MR._ByValue_VertCoords? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_std_optional_MR_VertCoords_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.VertCoords._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_VertCoords_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRVertCoords(MR._ByValue_VertCoords? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Optional_MRVertCoords other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_VertCoords_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRVertCoords._Underlying *other);
                __MR_std_optional_MR_VertCoords_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR._ByValue_VertCoords? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_VertCoords_AssignFrom(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.VertCoords._Underlying *other);
                __MR_std_optional_MR_VertCoords_AssignFrom(_UnderlyingPtr, other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.VertCoords? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_VertCoords_MutableValue", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_std_optional_MR_VertCoords_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_VertCoords_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.VertCoords(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Optional_MRVertCoords` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Optional_MRVertCoords`/`Const_Optional_MRVertCoords` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Optional_MRVertCoords
        {
            internal readonly Const_Optional_MRVertCoords? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Optional_MRVertCoords() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Optional_MRVertCoords(Const_Optional_MRVertCoords new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Optional_MRVertCoords(Const_Optional_MRVertCoords arg) {return new(arg);}
            public _ByValue_Optional_MRVertCoords(MR.Misc._Moved<Optional_MRVertCoords> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Optional_MRVertCoords(MR.Misc._Moved<Optional_MRVertCoords> arg) {return new(arg);}

            /// Constructs a new instance.
            public static unsafe implicit operator _ByValue_Optional_MRVertCoords(MR._ByValue_VertCoords? other) {return new MR.Std.Optional_MRVertCoords(other);}
        }

        /// This is used for optional parameters of class `Optional_MRVertCoords` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRVertCoords`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRVertCoords`/`Const_Optional_MRVertCoords` directly.
        public class _InOptMut_Optional_MRVertCoords
        {
            public Optional_MRVertCoords? Opt;

            public _InOptMut_Optional_MRVertCoords() {}
            public _InOptMut_Optional_MRVertCoords(Optional_MRVertCoords value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRVertCoords(Optional_MRVertCoords value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRVertCoords` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRVertCoords`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRVertCoords`/`Const_Optional_MRVertCoords` to pass it to the function.
        public class _InOptConst_Optional_MRVertCoords
        {
            public Const_Optional_MRVertCoords? Opt;

            public _InOptConst_Optional_MRVertCoords() {}
            public _InOptConst_Optional_MRVertCoords(Const_Optional_MRVertCoords value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRVertCoords(Const_Optional_MRVertCoords value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRVertCoords(MR._ByValue_VertCoords? other) {return new MR.Std.Optional_MRVertCoords(other);}
        }
    }
}
