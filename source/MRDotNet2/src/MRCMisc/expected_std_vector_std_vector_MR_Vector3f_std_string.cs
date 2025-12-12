public static partial class MR
{
    /// Stores either a `std::vector<std::vector<MR::Vector3f>>` that represents success or a `std::string` that represents an error.
    /// This is the const half of the class.
    public class Const_Expected_StdVectorStdVectorMRVector3f_StdString : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Expected_StdVectorStdVectorMRVector3f_StdString(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_Destroy", ExactSpelling = true)]
            extern static void __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_Destroy(_Underlying *_this);
            __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Expected_StdVectorStdVectorMRVector3f_StdString() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Expected_StdVectorStdVectorMRVector3f_StdString() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_DefaultConstruct();
            _UnderlyingPtr = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_DefaultConstruct();
        }

        /// Constructs a copy of another instance. The source remains alive.
        public unsafe Const_Expected_StdVectorStdVectorMRVector3f_StdString(MR._ByValue_Expected_StdVectorStdVectorMRVector3f_StdString other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *other);
            _UnderlyingPtr = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// If this instance represents success, returns the stored `std::vector<std::vector<MR::Vector3f>>`. Otherwise null.
        public unsafe MR.Std.Const_Vector_StdVectorMRVector3f? GetValue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetValue", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetValue(_Underlying *_this);
            var __ret = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetValue(_UnderlyingPtr);
            return __ret is not null ? new MR.Std.Const_Vector_StdVectorMRVector3f(__ret, is_owning: false) : null;
        }

        /// If this instance represents an error, returns the stored `std::string`. Otherwise null.
        public unsafe MR.Std.Const_String? GetError()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetError", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetError(_Underlying *_this);
            var __ret = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetError(_UnderlyingPtr);
            return __ret is not null ? new MR.Std.Const_String(__ret, is_owning: false) : null;
        }
    }

    /// Stores either a `std::vector<std::vector<MR::Vector3f>>` that represents success or a `std::string` that represents an error.
    /// This is the non-const half of the class.
    public class Expected_StdVectorStdVectorMRVector3f_StdString : Const_Expected_StdVectorStdVectorMRVector3f_StdString
    {
        internal unsafe Expected_StdVectorStdVectorMRVector3f_StdString(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Expected_StdVectorStdVectorMRVector3f_StdString() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_DefaultConstruct();
            _UnderlyingPtr = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_DefaultConstruct();
        }

        /// Constructs a copy of another instance. The source remains alive.
        public unsafe Expected_StdVectorStdVectorMRVector3f_StdString(MR._ByValue_Expected_StdVectorStdVectorMRVector3f_StdString other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *other);
            _UnderlyingPtr = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Assigns the contents from another instance. Both objects remain alive after the call.
        public unsafe void Assign(MR._ByValue_Expected_StdVectorStdVectorMRVector3f_StdString other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_AssignFromAnother", ExactSpelling = true)]
            extern static void __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Expected_StdVectorStdVectorMRVector3f_StdString._Underlying *other);
            __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// If this instance represents success, returns the stored `std::vector<std::vector<MR::Vector3f>>`. Otherwise null. This version returns a mutable pointer.
        public unsafe MR.Std.Vector_StdVectorMRVector3f? GetMutableValue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetMutableValue", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetMutableValue(_Underlying *_this);
            var __ret = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetMutableValue(_UnderlyingPtr);
            return __ret is not null ? new MR.Std.Vector_StdVectorMRVector3f(__ret, is_owning: false) : null;
        }

        /// If this instance represents an error, returns the stored `std::string`. Otherwise null. This version returns a mutable pointer.
        public unsafe MR.Std.String? GetMutableError()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetMutableError", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetMutableError(_Underlying *_this);
            var __ret = __MR_expected_std_vector_std_vector_MR_Vector3f_std_string_GetMutableError(_UnderlyingPtr);
            return __ret is not null ? new MR.Std.String(__ret, is_owning: false) : null;
        }
    }

    /// This is used as a function parameter when the underlying function receives `Expected_StdVectorStdVectorMRVector3f_StdString` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Expected_StdVectorStdVectorMRVector3f_StdString`/`Const_Expected_StdVectorStdVectorMRVector3f_StdString` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Expected_StdVectorStdVectorMRVector3f_StdString
    {
        internal readonly Const_Expected_StdVectorStdVectorMRVector3f_StdString? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Expected_StdVectorStdVectorMRVector3f_StdString() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Expected_StdVectorStdVectorMRVector3f_StdString(Const_Expected_StdVectorStdVectorMRVector3f_StdString new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Expected_StdVectorStdVectorMRVector3f_StdString(Const_Expected_StdVectorStdVectorMRVector3f_StdString arg) {return new(arg);}
        public _ByValue_Expected_StdVectorStdVectorMRVector3f_StdString(MR.Misc._Moved<Expected_StdVectorStdVectorMRVector3f_StdString> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Expected_StdVectorStdVectorMRVector3f_StdString(MR.Misc._Moved<Expected_StdVectorStdVectorMRVector3f_StdString> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Expected_StdVectorStdVectorMRVector3f_StdString` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Expected_StdVectorStdVectorMRVector3f_StdString`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Expected_StdVectorStdVectorMRVector3f_StdString`/`Const_Expected_StdVectorStdVectorMRVector3f_StdString` directly.
    public class _InOptMut_Expected_StdVectorStdVectorMRVector3f_StdString
    {
        public Expected_StdVectorStdVectorMRVector3f_StdString? Opt;

        public _InOptMut_Expected_StdVectorStdVectorMRVector3f_StdString() {}
        public _InOptMut_Expected_StdVectorStdVectorMRVector3f_StdString(Expected_StdVectorStdVectorMRVector3f_StdString value) {Opt = value;}
        public static implicit operator _InOptMut_Expected_StdVectorStdVectorMRVector3f_StdString(Expected_StdVectorStdVectorMRVector3f_StdString value) {return new(value);}
    }

    /// This is used for optional parameters of class `Expected_StdVectorStdVectorMRVector3f_StdString` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Expected_StdVectorStdVectorMRVector3f_StdString`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Expected_StdVectorStdVectorMRVector3f_StdString`/`Const_Expected_StdVectorStdVectorMRVector3f_StdString` to pass it to the function.
    public class _InOptConst_Expected_StdVectorStdVectorMRVector3f_StdString
    {
        public Const_Expected_StdVectorStdVectorMRVector3f_StdString? Opt;

        public _InOptConst_Expected_StdVectorStdVectorMRVector3f_StdString() {}
        public _InOptConst_Expected_StdVectorStdVectorMRVector3f_StdString(Const_Expected_StdVectorStdVectorMRVector3f_StdString value) {Opt = value;}
        public static implicit operator _InOptConst_Expected_StdVectorStdVectorMRVector3f_StdString(Const_Expected_StdVectorStdVectorMRVector3f_StdString value) {return new(value);}
    }
}
