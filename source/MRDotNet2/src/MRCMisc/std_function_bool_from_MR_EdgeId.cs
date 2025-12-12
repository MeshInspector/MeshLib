public static partial class MR
{
    public static partial class Std
    {
        /// Stores a functor of type: `bool(MR::EdgeId)`. Possibly stateful.
        /// This is the const half of the class.
        public class Const_Function_BoolFuncFromMREdgeId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Function_BoolFuncFromMREdgeId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_MR_EdgeId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_function_bool_from_MR_EdgeId_Destroy(_Underlying *_this);
                __MR_std_function_bool_from_MR_EdgeId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Function_BoolFuncFromMREdgeId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Function_BoolFuncFromMREdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeId._Underlying *__MR_std_function_bool_from_MR_EdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_bool_from_MR_EdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Function_BoolFuncFromMREdgeId(MR.Std._ByValue_Function_BoolFuncFromMREdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeId._Underlying *__MR_std_function_bool_from_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_function_bool_from_MR_EdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// Stores a functor of type: `bool(MR::EdgeId)`. Possibly stateful.
        /// This is the non-const half of the class.
        public class Function_BoolFuncFromMREdgeId : Const_Function_BoolFuncFromMREdgeId
        {
            internal unsafe Function_BoolFuncFromMREdgeId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Function_BoolFuncFromMREdgeId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_MR_EdgeId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeId._Underlying *__MR_std_function_bool_from_MR_EdgeId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_bool_from_MR_EdgeId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Function_BoolFuncFromMREdgeId(MR.Std._ByValue_Function_BoolFuncFromMREdgeId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_MR_EdgeId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeId._Underlying *__MR_std_function_bool_from_MR_EdgeId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *other);
                _UnderlyingPtr = __MR_std_function_bool_from_MR_EdgeId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Function_BoolFuncFromMREdgeId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_MR_EdgeId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_function_bool_from_MR_EdgeId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *other);
                __MR_std_function_bool_from_MR_EdgeId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Function_BoolFuncFromMREdgeId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Function_BoolFuncFromMREdgeId`/`Const_Function_BoolFuncFromMREdgeId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Function_BoolFuncFromMREdgeId
        {
            internal readonly Const_Function_BoolFuncFromMREdgeId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Function_BoolFuncFromMREdgeId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Function_BoolFuncFromMREdgeId(Const_Function_BoolFuncFromMREdgeId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Function_BoolFuncFromMREdgeId(Const_Function_BoolFuncFromMREdgeId arg) {return new(arg);}
            public _ByValue_Function_BoolFuncFromMREdgeId(MR.Misc._Moved<Function_BoolFuncFromMREdgeId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Function_BoolFuncFromMREdgeId(MR.Misc._Moved<Function_BoolFuncFromMREdgeId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Function_BoolFuncFromMREdgeId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Function_BoolFuncFromMREdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_BoolFuncFromMREdgeId`/`Const_Function_BoolFuncFromMREdgeId` directly.
        public class _InOptMut_Function_BoolFuncFromMREdgeId
        {
            public Function_BoolFuncFromMREdgeId? Opt;

            public _InOptMut_Function_BoolFuncFromMREdgeId() {}
            public _InOptMut_Function_BoolFuncFromMREdgeId(Function_BoolFuncFromMREdgeId value) {Opt = value;}
            public static implicit operator _InOptMut_Function_BoolFuncFromMREdgeId(Function_BoolFuncFromMREdgeId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Function_BoolFuncFromMREdgeId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Function_BoolFuncFromMREdgeId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_BoolFuncFromMREdgeId`/`Const_Function_BoolFuncFromMREdgeId` to pass it to the function.
        public class _InOptConst_Function_BoolFuncFromMREdgeId
        {
            public Const_Function_BoolFuncFromMREdgeId? Opt;

            public _InOptConst_Function_BoolFuncFromMREdgeId() {}
            public _InOptConst_Function_BoolFuncFromMREdgeId(Const_Function_BoolFuncFromMREdgeId value) {Opt = value;}
            public static implicit operator _InOptConst_Function_BoolFuncFromMREdgeId(Const_Function_BoolFuncFromMREdgeId value) {return new(value);}
        }
    }
}
