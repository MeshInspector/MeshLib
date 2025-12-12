public static partial class MR
{
    public static partial class Std
    {
        /// Stores a functor of type: `void(MR::Mesh &, float, float)`. Possibly stateful.
        /// This is the const half of the class.
        public class Const_Function_VoidFuncFromMRMeshRefFloatFloat : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Function_VoidFuncFromMRMeshRefFloatFloat(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_void_from_MR_Mesh_ref_float_float_Destroy", ExactSpelling = true)]
                extern static void __MR_std_function_void_from_MR_Mesh_ref_float_float_Destroy(_Underlying *_this);
                __MR_std_function_void_from_MR_Mesh_ref_float_float_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Function_VoidFuncFromMRMeshRefFloatFloat() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Function_VoidFuncFromMRMeshRefFloatFloat() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_void_from_MR_Mesh_ref_float_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *__MR_std_function_void_from_MR_Mesh_ref_float_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_void_from_MR_Mesh_ref_float_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Function_VoidFuncFromMRMeshRefFloatFloat(MR.Std._ByValue_Function_VoidFuncFromMRMeshRefFloatFloat other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_void_from_MR_Mesh_ref_float_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *__MR_std_function_void_from_MR_Mesh_ref_float_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *other);
                _UnderlyingPtr = __MR_std_function_void_from_MR_Mesh_ref_float_float_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// Stores a functor of type: `void(MR::Mesh &, float, float)`. Possibly stateful.
        /// This is the non-const half of the class.
        public class Function_VoidFuncFromMRMeshRefFloatFloat : Const_Function_VoidFuncFromMRMeshRefFloatFloat
        {
            internal unsafe Function_VoidFuncFromMRMeshRefFloatFloat(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Function_VoidFuncFromMRMeshRefFloatFloat() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_void_from_MR_Mesh_ref_float_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *__MR_std_function_void_from_MR_Mesh_ref_float_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_void_from_MR_Mesh_ref_float_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Function_VoidFuncFromMRMeshRefFloatFloat(MR.Std._ByValue_Function_VoidFuncFromMRMeshRefFloatFloat other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_void_from_MR_Mesh_ref_float_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *__MR_std_function_void_from_MR_Mesh_ref_float_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *other);
                _UnderlyingPtr = __MR_std_function_void_from_MR_Mesh_ref_float_float_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Function_VoidFuncFromMRMeshRefFloatFloat other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_void_from_MR_Mesh_ref_float_float_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_function_void_from_MR_Mesh_ref_float_float_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Function_VoidFuncFromMRMeshRefFloatFloat._Underlying *other);
                __MR_std_function_void_from_MR_Mesh_ref_float_float_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Function_VoidFuncFromMRMeshRefFloatFloat` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Function_VoidFuncFromMRMeshRefFloatFloat`/`Const_Function_VoidFuncFromMRMeshRefFloatFloat` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Function_VoidFuncFromMRMeshRefFloatFloat
        {
            internal readonly Const_Function_VoidFuncFromMRMeshRefFloatFloat? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Function_VoidFuncFromMRMeshRefFloatFloat() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Function_VoidFuncFromMRMeshRefFloatFloat(Const_Function_VoidFuncFromMRMeshRefFloatFloat new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Function_VoidFuncFromMRMeshRefFloatFloat(Const_Function_VoidFuncFromMRMeshRefFloatFloat arg) {return new(arg);}
            public _ByValue_Function_VoidFuncFromMRMeshRefFloatFloat(MR.Misc._Moved<Function_VoidFuncFromMRMeshRefFloatFloat> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Function_VoidFuncFromMRMeshRefFloatFloat(MR.Misc._Moved<Function_VoidFuncFromMRMeshRefFloatFloat> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Function_VoidFuncFromMRMeshRefFloatFloat` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Function_VoidFuncFromMRMeshRefFloatFloat`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_VoidFuncFromMRMeshRefFloatFloat`/`Const_Function_VoidFuncFromMRMeshRefFloatFloat` directly.
        public class _InOptMut_Function_VoidFuncFromMRMeshRefFloatFloat
        {
            public Function_VoidFuncFromMRMeshRefFloatFloat? Opt;

            public _InOptMut_Function_VoidFuncFromMRMeshRefFloatFloat() {}
            public _InOptMut_Function_VoidFuncFromMRMeshRefFloatFloat(Function_VoidFuncFromMRMeshRefFloatFloat value) {Opt = value;}
            public static implicit operator _InOptMut_Function_VoidFuncFromMRMeshRefFloatFloat(Function_VoidFuncFromMRMeshRefFloatFloat value) {return new(value);}
        }

        /// This is used for optional parameters of class `Function_VoidFuncFromMRMeshRefFloatFloat` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Function_VoidFuncFromMRMeshRefFloatFloat`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_VoidFuncFromMRMeshRefFloatFloat`/`Const_Function_VoidFuncFromMRMeshRefFloatFloat` to pass it to the function.
        public class _InOptConst_Function_VoidFuncFromMRMeshRefFloatFloat
        {
            public Const_Function_VoidFuncFromMRMeshRefFloatFloat? Opt;

            public _InOptConst_Function_VoidFuncFromMRMeshRefFloatFloat() {}
            public _InOptConst_Function_VoidFuncFromMRMeshRefFloatFloat(Const_Function_VoidFuncFromMRMeshRefFloatFloat value) {Opt = value;}
            public static implicit operator _InOptConst_Function_VoidFuncFromMRMeshRefFloatFloat(Const_Function_VoidFuncFromMRMeshRefFloatFloat value) {return new(value);}
        }
    }
}
