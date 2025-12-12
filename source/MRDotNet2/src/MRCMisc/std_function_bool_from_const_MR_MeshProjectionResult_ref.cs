public static partial class MR
{
    public static partial class Std
    {
        /// Stores a functor of type: `bool(const MR::MeshProjectionResult &)`. Possibly stateful.
        /// This is the const half of the class.
        public class Const_Function_BoolFuncFromConstMRMeshProjectionResultRef : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Function_BoolFuncFromConstMRMeshProjectionResultRef(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_Destroy", ExactSpelling = true)]
                extern static void __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_Destroy(_Underlying *_this);
                __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Function_BoolFuncFromConstMRMeshProjectionResultRef() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Function_BoolFuncFromConstMRMeshProjectionResultRef() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *__MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Function_BoolFuncFromConstMRMeshProjectionResultRef(MR.Std._ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *__MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *other);
                _UnderlyingPtr = __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// Stores a functor of type: `bool(const MR::MeshProjectionResult &)`. Possibly stateful.
        /// This is the non-const half of the class.
        public class Function_BoolFuncFromConstMRMeshProjectionResultRef : Const_Function_BoolFuncFromConstMRMeshProjectionResultRef
        {
            internal unsafe Function_BoolFuncFromConstMRMeshProjectionResultRef(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Function_BoolFuncFromConstMRMeshProjectionResultRef() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *__MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Function_BoolFuncFromConstMRMeshProjectionResultRef(MR.Std._ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *__MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *other);
                _UnderlyingPtr = __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *other);
                __MR_std_function_bool_from_const_MR_MeshProjectionResult_ref_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Function_BoolFuncFromConstMRMeshProjectionResultRef` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Function_BoolFuncFromConstMRMeshProjectionResultRef`/`Const_Function_BoolFuncFromConstMRMeshProjectionResultRef` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef
        {
            internal readonly Const_Function_BoolFuncFromConstMRMeshProjectionResultRef? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef(Const_Function_BoolFuncFromConstMRMeshProjectionResultRef new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef(Const_Function_BoolFuncFromConstMRMeshProjectionResultRef arg) {return new(arg);}
            public _ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef(MR.Misc._Moved<Function_BoolFuncFromConstMRMeshProjectionResultRef> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Function_BoolFuncFromConstMRMeshProjectionResultRef(MR.Misc._Moved<Function_BoolFuncFromConstMRMeshProjectionResultRef> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Function_BoolFuncFromConstMRMeshProjectionResultRef` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Function_BoolFuncFromConstMRMeshProjectionResultRef`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_BoolFuncFromConstMRMeshProjectionResultRef`/`Const_Function_BoolFuncFromConstMRMeshProjectionResultRef` directly.
        public class _InOptMut_Function_BoolFuncFromConstMRMeshProjectionResultRef
        {
            public Function_BoolFuncFromConstMRMeshProjectionResultRef? Opt;

            public _InOptMut_Function_BoolFuncFromConstMRMeshProjectionResultRef() {}
            public _InOptMut_Function_BoolFuncFromConstMRMeshProjectionResultRef(Function_BoolFuncFromConstMRMeshProjectionResultRef value) {Opt = value;}
            public static implicit operator _InOptMut_Function_BoolFuncFromConstMRMeshProjectionResultRef(Function_BoolFuncFromConstMRMeshProjectionResultRef value) {return new(value);}
        }

        /// This is used for optional parameters of class `Function_BoolFuncFromConstMRMeshProjectionResultRef` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Function_BoolFuncFromConstMRMeshProjectionResultRef`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_BoolFuncFromConstMRMeshProjectionResultRef`/`Const_Function_BoolFuncFromConstMRMeshProjectionResultRef` to pass it to the function.
        public class _InOptConst_Function_BoolFuncFromConstMRMeshProjectionResultRef
        {
            public Const_Function_BoolFuncFromConstMRMeshProjectionResultRef? Opt;

            public _InOptConst_Function_BoolFuncFromConstMRMeshProjectionResultRef() {}
            public _InOptConst_Function_BoolFuncFromConstMRMeshProjectionResultRef(Const_Function_BoolFuncFromConstMRMeshProjectionResultRef value) {Opt = value;}
            public static implicit operator _InOptConst_Function_BoolFuncFromConstMRMeshProjectionResultRef(Const_Function_BoolFuncFromConstMRMeshProjectionResultRef value) {return new(value);}
        }
    }
}
