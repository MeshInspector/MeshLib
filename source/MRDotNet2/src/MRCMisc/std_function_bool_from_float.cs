public static partial class MR
{
    public static partial class Std
    {
        /// Stores a functor of type: `bool(float)`. Possibly stateful.
        /// This is the const half of the class.
        public class Const_Function_BoolFuncFromFloat : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Function_BoolFuncFromFloat(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_float_Destroy", ExactSpelling = true)]
                extern static void __MR_std_function_bool_from_float_Destroy(_Underlying *_this);
                __MR_std_function_bool_from_float_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Function_BoolFuncFromFloat() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Function_BoolFuncFromFloat() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_std_function_bool_from_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_bool_from_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Function_BoolFuncFromFloat(MR.Std._ByValue_Function_BoolFuncFromFloat other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_std_function_bool_from_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *other);
                _UnderlyingPtr = __MR_std_function_bool_from_float_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// Stores a functor of type: `bool(float)`. Possibly stateful.
        /// This is the non-const half of the class.
        public class Function_BoolFuncFromFloat : Const_Function_BoolFuncFromFloat
        {
            internal unsafe Function_BoolFuncFromFloat(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Function_BoolFuncFromFloat() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_float_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_std_function_bool_from_float_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_bool_from_float_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Function_BoolFuncFromFloat(MR.Std._ByValue_Function_BoolFuncFromFloat other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_float_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_std_function_bool_from_float_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *other);
                _UnderlyingPtr = __MR_std_function_bool_from_float_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Function_BoolFuncFromFloat other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_bool_from_float_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_function_bool_from_float_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *other);
                __MR_std_function_bool_from_float_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Function_BoolFuncFromFloat` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Function_BoolFuncFromFloat`/`Const_Function_BoolFuncFromFloat` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Function_BoolFuncFromFloat
        {
            internal readonly Const_Function_BoolFuncFromFloat? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Function_BoolFuncFromFloat() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Function_BoolFuncFromFloat(Const_Function_BoolFuncFromFloat new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Function_BoolFuncFromFloat(Const_Function_BoolFuncFromFloat arg) {return new(arg);}
            public _ByValue_Function_BoolFuncFromFloat(MR.Misc._Moved<Function_BoolFuncFromFloat> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Function_BoolFuncFromFloat(MR.Misc._Moved<Function_BoolFuncFromFloat> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Function_BoolFuncFromFloat` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Function_BoolFuncFromFloat`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_BoolFuncFromFloat`/`Const_Function_BoolFuncFromFloat` directly.
        public class _InOptMut_Function_BoolFuncFromFloat
        {
            public Function_BoolFuncFromFloat? Opt;

            public _InOptMut_Function_BoolFuncFromFloat() {}
            public _InOptMut_Function_BoolFuncFromFloat(Function_BoolFuncFromFloat value) {Opt = value;}
            public static implicit operator _InOptMut_Function_BoolFuncFromFloat(Function_BoolFuncFromFloat value) {return new(value);}
        }

        /// This is used for optional parameters of class `Function_BoolFuncFromFloat` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Function_BoolFuncFromFloat`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_BoolFuncFromFloat`/`Const_Function_BoolFuncFromFloat` to pass it to the function.
        public class _InOptConst_Function_BoolFuncFromFloat
        {
            public Const_Function_BoolFuncFromFloat? Opt;

            public _InOptConst_Function_BoolFuncFromFloat() {}
            public _InOptConst_Function_BoolFuncFromFloat(Const_Function_BoolFuncFromFloat value) {Opt = value;}
            public static implicit operator _InOptConst_Function_BoolFuncFromFloat(Const_Function_BoolFuncFromFloat value) {return new(value);}
        }
    }
}
