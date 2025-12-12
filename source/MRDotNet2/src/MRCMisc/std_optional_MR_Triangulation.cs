public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `MR::Triangulation` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_MRTriangulation : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_MRTriangulation(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Triangulation_Destroy(_Underlying *_this);
                __MR_std_optional_MR_Triangulation_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_MRTriangulation() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_MRTriangulation() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRTriangulation._Underlying *__MR_std_optional_MR_Triangulation_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Triangulation_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_MRTriangulation(MR.Std._ByValue_Optional_MRTriangulation other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRTriangulation._Underlying *__MR_std_optional_MR_Triangulation_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRTriangulation._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Triangulation_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_MRTriangulation(MR._ByValue_Triangulation? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRTriangulation._Underlying *__MR_std_optional_MR_Triangulation_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.Triangulation._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Triangulation_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_MRTriangulation(MR._ByValue_Triangulation? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Const_Triangulation? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_Value", ExactSpelling = true)]
                extern static MR.Const_Triangulation._Underlying *__MR_std_optional_MR_Triangulation_Value(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Triangulation_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Const_Triangulation(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `MR::Triangulation` or nothing.
        /// This is the non-const half of the class.
        public class Optional_MRTriangulation : Const_Optional_MRTriangulation
        {
            internal unsafe Optional_MRTriangulation(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_MRTriangulation() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_MRTriangulation._Underlying *__MR_std_optional_MR_Triangulation_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_MR_Triangulation_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_MRTriangulation(MR.Std._ByValue_Optional_MRTriangulation other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_MRTriangulation._Underlying *__MR_std_optional_MR_Triangulation_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRTriangulation._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Triangulation_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Optional_MRTriangulation(MR._ByValue_Triangulation? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_MRTriangulation._Underlying *__MR_std_optional_MR_Triangulation_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.Triangulation._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_MR_Triangulation_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_MRTriangulation(MR._ByValue_Triangulation? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Optional_MRTriangulation other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Triangulation_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Optional_MRTriangulation._Underlying *other);
                __MR_std_optional_MR_Triangulation_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR._ByValue_Triangulation? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_MR_Triangulation_AssignFrom(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Triangulation._Underlying *other);
                __MR_std_optional_MR_Triangulation_AssignFrom(_UnderlyingPtr, other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Triangulation? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_MR_Triangulation_MutableValue", ExactSpelling = true)]
                extern static MR.Triangulation._Underlying *__MR_std_optional_MR_Triangulation_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_MR_Triangulation_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Triangulation(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Optional_MRTriangulation` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Optional_MRTriangulation`/`Const_Optional_MRTriangulation` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Optional_MRTriangulation
        {
            internal readonly Const_Optional_MRTriangulation? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Optional_MRTriangulation() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Optional_MRTriangulation(Const_Optional_MRTriangulation new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Optional_MRTriangulation(Const_Optional_MRTriangulation arg) {return new(arg);}
            public _ByValue_Optional_MRTriangulation(MR.Misc._Moved<Optional_MRTriangulation> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Optional_MRTriangulation(MR.Misc._Moved<Optional_MRTriangulation> arg) {return new(arg);}

            /// Constructs a new instance.
            public static unsafe implicit operator _ByValue_Optional_MRTriangulation(MR._ByValue_Triangulation? other) {return new MR.Std.Optional_MRTriangulation(other);}
        }

        /// This is used for optional parameters of class `Optional_MRTriangulation` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_MRTriangulation`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRTriangulation`/`Const_Optional_MRTriangulation` directly.
        public class _InOptMut_Optional_MRTriangulation
        {
            public Optional_MRTriangulation? Opt;

            public _InOptMut_Optional_MRTriangulation() {}
            public _InOptMut_Optional_MRTriangulation(Optional_MRTriangulation value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_MRTriangulation(Optional_MRTriangulation value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_MRTriangulation` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_MRTriangulation`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_MRTriangulation`/`Const_Optional_MRTriangulation` to pass it to the function.
        public class _InOptConst_Optional_MRTriangulation
        {
            public Const_Optional_MRTriangulation? Opt;

            public _InOptConst_Optional_MRTriangulation() {}
            public _InOptConst_Optional_MRTriangulation(Const_Optional_MRTriangulation value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_MRTriangulation(Const_Optional_MRTriangulation value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_MRTriangulation(MR._ByValue_Triangulation? other) {return new MR.Std.Optional_MRTriangulation(other);}
        }
    }
}
