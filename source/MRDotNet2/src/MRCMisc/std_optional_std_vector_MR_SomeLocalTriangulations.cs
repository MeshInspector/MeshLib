public static partial class MR
{
    public static partial class Std
    {
        /// Stores either a single `std::vector<MR::SomeLocalTriangulations>` or nothing.
        /// This is the const half of the class.
        public class Const_Optional_StdVectorMRSomeLocalTriangulations : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Optional_StdVectorMRSomeLocalTriangulations(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_Destroy", ExactSpelling = true)]
                extern static void __MR_std_optional_std_vector_MR_SomeLocalTriangulations_Destroy(_Underlying *_this);
                __MR_std_optional_std_vector_MR_SomeLocalTriangulations_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Optional_StdVectorMRSomeLocalTriangulations() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Optional_StdVectorMRSomeLocalTriangulations() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Optional_StdVectorMRSomeLocalTriangulations other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Const_Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRSomeLocalTriangulations._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Const_Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other) {return new(other);}

            /// The stored element or null if none, read-only.
            public unsafe MR.Std.Const_Vector_MRSomeLocalTriangulations? Value()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_Value", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_Value(_Underlying *_this);
                var __ret = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_Value(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Const_Vector_MRSomeLocalTriangulations(__ret, is_owning: false) : null;
            }
        }

        /// Stores either a single `std::vector<MR::SomeLocalTriangulations>` or nothing.
        /// This is the non-const half of the class.
        public class Optional_StdVectorMRSomeLocalTriangulations : Const_Optional_StdVectorMRSomeLocalTriangulations
        {
            internal unsafe Optional_StdVectorMRSomeLocalTriangulations(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Optional_StdVectorMRSomeLocalTriangulations() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_DefaultConstruct();
                _UnderlyingPtr = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Optional_StdVectorMRSomeLocalTriangulations other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public unsafe Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFrom", ExactSpelling = true)]
                extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFrom(MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRSomeLocalTriangulations._Underlying *other);
                _UnderlyingPtr = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_ConstructFrom(other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Constructs a new instance.
            public static unsafe implicit operator Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other) {return new(other);}

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Optional_StdVectorMRSomeLocalTriangulations other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_optional_std_vector_MR_SomeLocalTriangulations_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *other);
                __MR_std_optional_std_vector_MR_SomeLocalTriangulations_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents.
            public unsafe void Assign(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_AssignFrom", ExactSpelling = true)]
                extern static void __MR_std_optional_std_vector_MR_SomeLocalTriangulations_AssignFrom(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Vector_MRSomeLocalTriangulations._Underlying *other);
                __MR_std_optional_std_vector_MR_SomeLocalTriangulations_AssignFrom(_UnderlyingPtr, other is not null ? other.PassByMode : MR.Misc._PassBy.no_object, other is not null && other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// The stored element or null if none, mutable.
            public unsafe MR.Std.Vector_MRSomeLocalTriangulations? MutableValue()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_optional_std_vector_MR_SomeLocalTriangulations_MutableValue", ExactSpelling = true)]
                extern static MR.Std.Vector_MRSomeLocalTriangulations._Underlying *__MR_std_optional_std_vector_MR_SomeLocalTriangulations_MutableValue(_Underlying *_this);
                var __ret = __MR_std_optional_std_vector_MR_SomeLocalTriangulations_MutableValue(_UnderlyingPtr);
                return __ret is not null ? new MR.Std.Vector_MRSomeLocalTriangulations(__ret, is_owning: false) : null;
            }
        }

        /// This is used as a function parameter when the underlying function receives `Optional_StdVectorMRSomeLocalTriangulations` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Optional_StdVectorMRSomeLocalTriangulations`/`Const_Optional_StdVectorMRSomeLocalTriangulations` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Optional_StdVectorMRSomeLocalTriangulations
        {
            internal readonly Const_Optional_StdVectorMRSomeLocalTriangulations? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Optional_StdVectorMRSomeLocalTriangulations() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Optional_StdVectorMRSomeLocalTriangulations(Const_Optional_StdVectorMRSomeLocalTriangulations new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Optional_StdVectorMRSomeLocalTriangulations(Const_Optional_StdVectorMRSomeLocalTriangulations arg) {return new(arg);}
            public _ByValue_Optional_StdVectorMRSomeLocalTriangulations(MR.Misc._Moved<Optional_StdVectorMRSomeLocalTriangulations> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Optional_StdVectorMRSomeLocalTriangulations(MR.Misc._Moved<Optional_StdVectorMRSomeLocalTriangulations> arg) {return new(arg);}

            /// Constructs a new instance.
            public static unsafe implicit operator _ByValue_Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other) {return new MR.Std.Optional_StdVectorMRSomeLocalTriangulations(other);}
        }

        /// This is used for optional parameters of class `Optional_StdVectorMRSomeLocalTriangulations` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Optional_StdVectorMRSomeLocalTriangulations`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_StdVectorMRSomeLocalTriangulations`/`Const_Optional_StdVectorMRSomeLocalTriangulations` directly.
        public class _InOptMut_Optional_StdVectorMRSomeLocalTriangulations
        {
            public Optional_StdVectorMRSomeLocalTriangulations? Opt;

            public _InOptMut_Optional_StdVectorMRSomeLocalTriangulations() {}
            public _InOptMut_Optional_StdVectorMRSomeLocalTriangulations(Optional_StdVectorMRSomeLocalTriangulations value) {Opt = value;}
            public static implicit operator _InOptMut_Optional_StdVectorMRSomeLocalTriangulations(Optional_StdVectorMRSomeLocalTriangulations value) {return new(value);}
        }

        /// This is used for optional parameters of class `Optional_StdVectorMRSomeLocalTriangulations` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Optional_StdVectorMRSomeLocalTriangulations`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Optional_StdVectorMRSomeLocalTriangulations`/`Const_Optional_StdVectorMRSomeLocalTriangulations` to pass it to the function.
        public class _InOptConst_Optional_StdVectorMRSomeLocalTriangulations
        {
            public Const_Optional_StdVectorMRSomeLocalTriangulations? Opt;

            public _InOptConst_Optional_StdVectorMRSomeLocalTriangulations() {}
            public _InOptConst_Optional_StdVectorMRSomeLocalTriangulations(Const_Optional_StdVectorMRSomeLocalTriangulations value) {Opt = value;}
            public static implicit operator _InOptConst_Optional_StdVectorMRSomeLocalTriangulations(Const_Optional_StdVectorMRSomeLocalTriangulations value) {return new(value);}

            /// Constructs a new instance.
            public static unsafe implicit operator _InOptConst_Optional_StdVectorMRSomeLocalTriangulations(MR.Std._ByValue_Vector_MRSomeLocalTriangulations? other) {return new MR.Std.Optional_StdVectorMRSomeLocalTriangulations(other);}
        }
    }
}
