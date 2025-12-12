public static partial class MR
{
    public static partial class Std
    {
        /// Stores a functor of type: `double(MR::VertId, MR::VertId, MR::VertId)`. Possibly stateful.
        /// This is the const half of the class.
        public class Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_Destroy", ExactSpelling = true)]
                extern static void __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_Destroy(_Underlying *_this);
                __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId() {Dispose(false);}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *__MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *__MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// Stores a functor of type: `double(MR::VertId, MR::VertId, MR::VertId)`. Possibly stateful.
        /// This is the non-const half of the class.
        public class Function_DoubleFuncFromMRVertIdMRVertIdMRVertId : Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId
        {
            internal unsafe Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Constructs an empty (default-constructed) instance.
            public unsafe Function_DoubleFuncFromMRVertIdMRVertIdMRVertId() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *__MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_DefaultConstruct();
                _UnderlyingPtr = __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_DefaultConstruct();
            }

            /// Constructs a copy of another instance. The source remains alive.
            public unsafe Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *__MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *other);
                _UnderlyingPtr = __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }

            /// Assigns the contents from another instance. Both objects remain alive after the call.
            public unsafe void Assign(MR.Std._ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_AssignFromAnother", ExactSpelling = true)]
                extern static void __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Std.Function_DoubleFuncFromMRVertIdMRVertIdMRVertId._Underlying *other);
                __MR_std_function_double_from_MR_VertId_MR_VertId_MR_VertId_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Function_DoubleFuncFromMRVertIdMRVertIdMRVertId` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Function_DoubleFuncFromMRVertIdMRVertIdMRVertId`/`Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId
        {
            internal readonly Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId arg) {return new(arg);}
            public _ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(MR.Misc._Moved<Function_DoubleFuncFromMRVertIdMRVertIdMRVertId> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(MR.Misc._Moved<Function_DoubleFuncFromMRVertIdMRVertIdMRVertId> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Function_DoubleFuncFromMRVertIdMRVertIdMRVertId` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_DoubleFuncFromMRVertIdMRVertIdMRVertId`/`Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId` directly.
        public class _InOptMut_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId
        {
            public Function_DoubleFuncFromMRVertIdMRVertIdMRVertId? Opt;

            public _InOptMut_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId() {}
            public _InOptMut_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(Function_DoubleFuncFromMRVertIdMRVertIdMRVertId value) {Opt = value;}
            public static implicit operator _InOptMut_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(Function_DoubleFuncFromMRVertIdMRVertIdMRVertId value) {return new(value);}
        }

        /// This is used for optional parameters of class `Function_DoubleFuncFromMRVertIdMRVertIdMRVertId` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Function_DoubleFuncFromMRVertIdMRVertIdMRVertId`/`Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId` to pass it to the function.
        public class _InOptConst_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId
        {
            public Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId? Opt;

            public _InOptConst_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId() {}
            public _InOptConst_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId value) {Opt = value;}
            public static implicit operator _InOptConst_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId(Const_Function_DoubleFuncFromMRVertIdMRVertIdMRVertId value) {return new(value);}
        }
    }
}
