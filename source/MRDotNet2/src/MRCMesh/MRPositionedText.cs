public static partial class MR
{
    /// Generated from class `MR::PositionedText`.
    /// This is the const half of the class.
    public class Const_PositionedText : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_PositionedText>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PositionedText(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_Destroy", ExactSpelling = true)]
            extern static void __MR_PositionedText_Destroy(_Underlying *_this);
            __MR_PositionedText_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PositionedText() {Dispose(false);}

        public unsafe MR.Std.Const_String Text
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_Get_text", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_PositionedText_Get_text(_Underlying *_this);
                return new(__MR_PositionedText_Get_text(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f Position
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_Get_position", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PositionedText_Get_position(_Underlying *_this);
                return new(__MR_PositionedText_Get_position(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PositionedText() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_DefaultConstruct();
            _UnderlyingPtr = __MR_PositionedText_DefaultConstruct();
        }

        /// Generated from constructor `MR::PositionedText::PositionedText`.
        public unsafe Const_PositionedText(MR._ByValue_PositionedText _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PositionedText._Underlying *_other);
            _UnderlyingPtr = __MR_PositionedText_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PositionedText::PositionedText`.
        public unsafe Const_PositionedText(ReadOnlySpan<char> text, MR.Const_Vector3f position) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_Construct", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_Construct(byte *text, byte *text_end, MR.Const_Vector3f._Underlying *position);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                _UnderlyingPtr = __MR_PositionedText_Construct(__ptr_text, __ptr_text + __len_text, position._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::PositionedText::operator==`.
        public static unsafe bool operator==(MR.Const_PositionedText _this, MR.Const_PositionedText _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_PositionedText", ExactSpelling = true)]
            extern static byte __MR_equal_MR_PositionedText(MR.Const_PositionedText._Underlying *_this, MR.Const_PositionedText._Underlying *_1);
            return __MR_equal_MR_PositionedText(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_PositionedText _this, MR.Const_PositionedText _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_PositionedText? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_PositionedText)
                return this == (MR.Const_PositionedText)other;
            return false;
        }
    }

    /// Generated from class `MR::PositionedText`.
    /// This is the non-const half of the class.
    public class PositionedText : Const_PositionedText
    {
        internal unsafe PositionedText(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.String Text
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_GetMutable_text", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_PositionedText_GetMutable_text(_Underlying *_this);
                return new(__MR_PositionedText_GetMutable_text(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f Position
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_GetMutable_position", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PositionedText_GetMutable_position(_Underlying *_this);
                return new(__MR_PositionedText_GetMutable_position(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PositionedText() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_DefaultConstruct();
            _UnderlyingPtr = __MR_PositionedText_DefaultConstruct();
        }

        /// Generated from constructor `MR::PositionedText::PositionedText`.
        public unsafe PositionedText(MR._ByValue_PositionedText _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PositionedText._Underlying *_other);
            _UnderlyingPtr = __MR_PositionedText_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PositionedText::PositionedText`.
        public unsafe PositionedText(ReadOnlySpan<char> text, MR.Const_Vector3f position) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_Construct", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_Construct(byte *text, byte *text_end, MR.Const_Vector3f._Underlying *position);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                _UnderlyingPtr = __MR_PositionedText_Construct(__ptr_text, __ptr_text + __len_text, position._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::PositionedText::operator=`.
        public unsafe MR.PositionedText Assign(MR._ByValue_PositionedText _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionedText_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PositionedText._Underlying *__MR_PositionedText_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PositionedText._Underlying *_other);
            return new(__MR_PositionedText_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PositionedText` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PositionedText`/`Const_PositionedText` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PositionedText
    {
        internal readonly Const_PositionedText? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PositionedText() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PositionedText(Const_PositionedText new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PositionedText(Const_PositionedText arg) {return new(arg);}
        public _ByValue_PositionedText(MR.Misc._Moved<PositionedText> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PositionedText(MR.Misc._Moved<PositionedText> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PositionedText` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PositionedText`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PositionedText`/`Const_PositionedText` directly.
    public class _InOptMut_PositionedText
    {
        public PositionedText? Opt;

        public _InOptMut_PositionedText() {}
        public _InOptMut_PositionedText(PositionedText value) {Opt = value;}
        public static implicit operator _InOptMut_PositionedText(PositionedText value) {return new(value);}
    }

    /// This is used for optional parameters of class `PositionedText` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PositionedText`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PositionedText`/`Const_PositionedText` to pass it to the function.
    public class _InOptConst_PositionedText
    {
        public Const_PositionedText? Opt;

        public _InOptConst_PositionedText() {}
        public _InOptConst_PositionedText(Const_PositionedText value) {Opt = value;}
        public static implicit operator _InOptConst_PositionedText(Const_PositionedText value) {return new(value);}
    }
}
