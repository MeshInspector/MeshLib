public static partial class MR
{
    /// Stores either a `std::vector<MR::EdgePoint>` that represents success or a `MR::PathError` that represents an error.
    /// This is the const half of the class.
    public class Const_Expected_StdVectorMREdgePoint_MRPathError : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Expected_StdVectorMREdgePoint_MRPathError(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_Destroy", ExactSpelling = true)]
            extern static void __MR_expected_std_vector_MR_EdgePoint_MR_PathError_Destroy(_Underlying *_this);
            __MR_expected_std_vector_MR_EdgePoint_MR_PathError_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Expected_StdVectorMREdgePoint_MRPathError() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Expected_StdVectorMREdgePoint_MRPathError() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_DefaultConstruct();
            _UnderlyingPtr = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_DefaultConstruct();
        }

        /// Constructs a copy of another instance. The source remains alive.
        public unsafe Const_Expected_StdVectorMREdgePoint_MRPathError(MR._ByValue_Expected_StdVectorMREdgePoint_MRPathError other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *other);
            _UnderlyingPtr = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// If this instance represents success, returns the stored `std::vector<MR::EdgePoint>`. Otherwise null.
        public unsafe MR.Std.Const_Vector_MREdgePoint? GetValue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetValue", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MREdgePoint._Underlying *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetValue(_Underlying *_this);
            var __ret = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetValue(_UnderlyingPtr);
            return __ret is not null ? new MR.Std.Const_Vector_MREdgePoint(__ret, is_owning: false) : null;
        }

        /// If this instance represents an error, returns the stored `MR::PathError`. Otherwise null.
        public unsafe MR.PathError? GetError()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetError", ExactSpelling = true)]
            extern static MR.PathError *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetError(_Underlying *_this);
            var __ret = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetError(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }
    }

    /// Stores either a `std::vector<MR::EdgePoint>` that represents success or a `MR::PathError` that represents an error.
    /// This is the non-const half of the class.
    public class Expected_StdVectorMREdgePoint_MRPathError : Const_Expected_StdVectorMREdgePoint_MRPathError
    {
        internal unsafe Expected_StdVectorMREdgePoint_MRPathError(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Expected_StdVectorMREdgePoint_MRPathError() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_DefaultConstruct();
            _UnderlyingPtr = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_DefaultConstruct();
        }

        /// Constructs a copy of another instance. The source remains alive.
        public unsafe Expected_StdVectorMREdgePoint_MRPathError(MR._ByValue_Expected_StdVectorMREdgePoint_MRPathError other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *other);
            _UnderlyingPtr = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Assigns the contents from another instance. Both objects remain alive after the call.
        public unsafe void Assign(MR._ByValue_Expected_StdVectorMREdgePoint_MRPathError other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_AssignFromAnother", ExactSpelling = true)]
            extern static void __MR_expected_std_vector_MR_EdgePoint_MR_PathError_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Expected_StdVectorMREdgePoint_MRPathError._Underlying *other);
            __MR_expected_std_vector_MR_EdgePoint_MR_PathError_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// If this instance represents success, returns the stored `std::vector<MR::EdgePoint>`. Otherwise null. This version returns a mutable pointer.
        public unsafe MR.Std.Vector_MREdgePoint? GetMutableValue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetMutableValue", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgePoint._Underlying *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetMutableValue(_Underlying *_this);
            var __ret = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetMutableValue(_UnderlyingPtr);
            return __ret is not null ? new MR.Std.Vector_MREdgePoint(__ret, is_owning: false) : null;
        }

        /// If this instance represents an error, returns the stored `MR::PathError`. Otherwise null. This version returns a mutable pointer.
        public unsafe MR.Misc.Ref<MR.PathError>? GetMutableError()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetMutableError", ExactSpelling = true)]
            extern static MR.PathError *__MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetMutableError(_Underlying *_this);
            var __ret = __MR_expected_std_vector_MR_EdgePoint_MR_PathError_GetMutableError(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<MR.PathError>(__ret) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Expected_StdVectorMREdgePoint_MRPathError` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Expected_StdVectorMREdgePoint_MRPathError`/`Const_Expected_StdVectorMREdgePoint_MRPathError` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Expected_StdVectorMREdgePoint_MRPathError
    {
        internal readonly Const_Expected_StdVectorMREdgePoint_MRPathError? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Expected_StdVectorMREdgePoint_MRPathError() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Expected_StdVectorMREdgePoint_MRPathError(Const_Expected_StdVectorMREdgePoint_MRPathError new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Expected_StdVectorMREdgePoint_MRPathError(Const_Expected_StdVectorMREdgePoint_MRPathError arg) {return new(arg);}
        public _ByValue_Expected_StdVectorMREdgePoint_MRPathError(MR.Misc._Moved<Expected_StdVectorMREdgePoint_MRPathError> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Expected_StdVectorMREdgePoint_MRPathError(MR.Misc._Moved<Expected_StdVectorMREdgePoint_MRPathError> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Expected_StdVectorMREdgePoint_MRPathError` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Expected_StdVectorMREdgePoint_MRPathError`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Expected_StdVectorMREdgePoint_MRPathError`/`Const_Expected_StdVectorMREdgePoint_MRPathError` directly.
    public class _InOptMut_Expected_StdVectorMREdgePoint_MRPathError
    {
        public Expected_StdVectorMREdgePoint_MRPathError? Opt;

        public _InOptMut_Expected_StdVectorMREdgePoint_MRPathError() {}
        public _InOptMut_Expected_StdVectorMREdgePoint_MRPathError(Expected_StdVectorMREdgePoint_MRPathError value) {Opt = value;}
        public static implicit operator _InOptMut_Expected_StdVectorMREdgePoint_MRPathError(Expected_StdVectorMREdgePoint_MRPathError value) {return new(value);}
    }

    /// This is used for optional parameters of class `Expected_StdVectorMREdgePoint_MRPathError` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Expected_StdVectorMREdgePoint_MRPathError`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Expected_StdVectorMREdgePoint_MRPathError`/`Const_Expected_StdVectorMREdgePoint_MRPathError` to pass it to the function.
    public class _InOptConst_Expected_StdVectorMREdgePoint_MRPathError
    {
        public Const_Expected_StdVectorMREdgePoint_MRPathError? Opt;

        public _InOptConst_Expected_StdVectorMREdgePoint_MRPathError() {}
        public _InOptConst_Expected_StdVectorMREdgePoint_MRPathError(Const_Expected_StdVectorMREdgePoint_MRPathError value) {Opt = value;}
        public static implicit operator _InOptConst_Expected_StdVectorMREdgePoint_MRPathError(Const_Expected_StdVectorMREdgePoint_MRPathError value) {return new(value);}
    }
}
