public static partial class MR
{
    public static partial class Std
    {
        /// Stores a functor of type: `double(const MR::FaceBitSet &, const MR::FixUndercuts::FindParams &)`. Possibly stateful.
        /// This is the const half of the class.
        public class Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_Destroy", ExactSpelling = true)]
                extern static void __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_Destroy(_Underlying *_this);
                __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *__MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(MR.Std._ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *__MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *other);
                _UnderlyingPtr = __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// Stores a functor of type: `double(const MR::FaceBitSet &, const MR::FixUndercuts::FindParams &)`. Possibly stateful.
        /// This is the non-const half of the class.
        public class Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef : Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef
        {
            internal unsafe Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *__MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(MR.Std._ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *__MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *other);
                _UnderlyingPtr = __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef._Underlying *other);
                __MR_std_function_double_from_const_MR_FaceBitSet_ref_const_MR_FixUndercuts_FindParams_ref_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef`/`Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef
        {
            internal readonly Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef arg) {return new(arg);}
            public _ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(MR.Misc._Moved<Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(MR.Misc._Moved<Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef`/`Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef` directly.
        public class _InOptMut_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef
        {
            public Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef? Opt;

            public _InOptMut_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef() {}
            public _InOptMut_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef value) {Opt = value;}
            public static implicit operator _InOptMut_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef value) {return new(value);}
        }

        /// This is used for optional parameters of class `Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef`/`Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef` to pass it to the function.
        public class _InOptConst_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef
        {
            public Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef? Opt;

            public _InOptConst_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef() {}
            public _InOptConst_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef value) {Opt = value;}
            public static implicit operator _InOptConst_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef(Const_Function_DoubleFuncFromConstMRFaceBitSetRefConstMRFixUndercutsFindParamsRef value) {return new(value);}
        }
    }
}
