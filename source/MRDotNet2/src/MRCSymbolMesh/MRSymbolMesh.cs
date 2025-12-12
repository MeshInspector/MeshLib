public static partial class MR
{
    public enum AlignType : int
    {
        Left = 0,
        Center = 1,
        Right = 2,
    }

    /// Generated from class `MR::SymbolMeshParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::TextMeshAlignParams`
    /// This is the const half of the class.
    public class Const_SymbolMeshParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SymbolMeshParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Destroy", ExactSpelling = true)]
            extern static void __MR_SymbolMeshParams_Destroy(_Underlying *_this);
            __MR_SymbolMeshParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SymbolMeshParams() {Dispose(false);}

        // max font size with 128 << 6 FT_F26Dot6 font size
        public static unsafe float MaxGeneratedFontHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_MaxGeneratedFontHeight", ExactSpelling = true)]
                extern static float *__MR_SymbolMeshParams_Get_MaxGeneratedFontHeight();
                return *__MR_SymbolMeshParams_Get_MaxGeneratedFontHeight();
            }
        }

        // Text that will be made mesh
        public unsafe MR.Std.Const_String Text
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_text", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_SymbolMeshParams_Get_text(_Underlying *_this);
                return new(__MR_SymbolMeshParams_Get_text(_UnderlyingPtr), is_owning: false);
            }
        }

        // Detailization of Bezier curves on font glyphs
        public unsafe int FontDetalization
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_fontDetalization", ExactSpelling = true)]
                extern static int *__MR_SymbolMeshParams_Get_fontDetalization(_Underlying *_this);
                return *__MR_SymbolMeshParams_Get_fontDetalization(_UnderlyingPtr);
            }
        }

        // Additional offset between symbols
        // X: In symbol size: 1.0f adds one "space", 0.5 adds half "space".
        // Y: In symbol size: 1.0f adds one base height, 0.5 adds half base height
        public unsafe MR.Const_Vector2f SymbolsDistanceAdditionalOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_symbolsDistanceAdditionalOffset", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_SymbolMeshParams_Get_symbolsDistanceAdditionalOffset(_Underlying *_this);
                return new(__MR_SymbolMeshParams_Get_symbolsDistanceAdditionalOffset(_UnderlyingPtr), is_owning: false);
            }
        }

        // Symbols thickness will be modified by this value (newThickness = modifier*baseSymbolHeight + defaultThickness)
        // note: changing this to non-zero values cause costly calculations
        public unsafe float SymbolsThicknessOffsetModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_symbolsThicknessOffsetModifier", ExactSpelling = true)]
                extern static float *__MR_SymbolMeshParams_Get_symbolsThicknessOffsetModifier(_Underlying *_this);
                return *__MR_SymbolMeshParams_Get_symbolsThicknessOffsetModifier(_UnderlyingPtr);
            }
        }

        // alignment of the text inside bbox
        public unsafe MR.AlignType Align
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_align", ExactSpelling = true)]
                extern static MR.AlignType *__MR_SymbolMeshParams_Get_align(_Underlying *_this);
                return *__MR_SymbolMeshParams_Get_align(_UnderlyingPtr);
            }
        }

        // Path to font file
        public unsafe MR.Std.Filesystem.Const_Path PathToFontFile
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_Get_pathToFontFile", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_SymbolMeshParams_Get_pathToFontFile(_Underlying *_this);
                return new(__MR_SymbolMeshParams_Get_pathToFontFile(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SymbolMeshParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_DefaultConstruct();
            _UnderlyingPtr = __MR_SymbolMeshParams_DefaultConstruct();
        }

        /// Constructs `MR::SymbolMeshParams` elementwise.
        public unsafe Const_SymbolMeshParams(ReadOnlySpan<char> text, int fontDetalization, MR.Vector2f symbolsDistanceAdditionalOffset, float symbolsThicknessOffsetModifier, MR.AlignType align, ReadOnlySpan<char> pathToFontFile) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_ConstructFrom(byte *text, byte *text_end, int fontDetalization, MR.Vector2f symbolsDistanceAdditionalOffset, float symbolsThicknessOffsetModifier, MR.AlignType align, byte *pathToFontFile, byte *pathToFontFile_end);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                byte[] __bytes_pathToFontFile = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(pathToFontFile.Length)];
                int __len_pathToFontFile = System.Text.Encoding.UTF8.GetBytes(pathToFontFile, __bytes_pathToFontFile);
                fixed (byte *__ptr_pathToFontFile = __bytes_pathToFontFile)
                {
                    _UnderlyingPtr = __MR_SymbolMeshParams_ConstructFrom(__ptr_text, __ptr_text + __len_text, fontDetalization, symbolsDistanceAdditionalOffset, symbolsThicknessOffsetModifier, align, __ptr_pathToFontFile, __ptr_pathToFontFile + __len_pathToFontFile);
                }
            }
        }

        /// Generated from constructor `MR::SymbolMeshParams::SymbolMeshParams`.
        public unsafe Const_SymbolMeshParams(MR._ByValue_SymbolMeshParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SymbolMeshParams._Underlying *_other);
            _UnderlyingPtr = __MR_SymbolMeshParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::SymbolMeshParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::TextMeshAlignParams`
    /// This is the non-const half of the class.
    public class SymbolMeshParams : Const_SymbolMeshParams
    {
        internal unsafe SymbolMeshParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Text that will be made mesh
        public new unsafe MR.Std.String Text
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_GetMutable_text", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_SymbolMeshParams_GetMutable_text(_Underlying *_this);
                return new(__MR_SymbolMeshParams_GetMutable_text(_UnderlyingPtr), is_owning: false);
            }
        }

        // Detailization of Bezier curves on font glyphs
        public new unsafe ref int FontDetalization
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_GetMutable_fontDetalization", ExactSpelling = true)]
                extern static int *__MR_SymbolMeshParams_GetMutable_fontDetalization(_Underlying *_this);
                return ref *__MR_SymbolMeshParams_GetMutable_fontDetalization(_UnderlyingPtr);
            }
        }

        // Additional offset between symbols
        // X: In symbol size: 1.0f adds one "space", 0.5 adds half "space".
        // Y: In symbol size: 1.0f adds one base height, 0.5 adds half base height
        public new unsafe MR.Mut_Vector2f SymbolsDistanceAdditionalOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_GetMutable_symbolsDistanceAdditionalOffset", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_SymbolMeshParams_GetMutable_symbolsDistanceAdditionalOffset(_Underlying *_this);
                return new(__MR_SymbolMeshParams_GetMutable_symbolsDistanceAdditionalOffset(_UnderlyingPtr), is_owning: false);
            }
        }

        // Symbols thickness will be modified by this value (newThickness = modifier*baseSymbolHeight + defaultThickness)
        // note: changing this to non-zero values cause costly calculations
        public new unsafe ref float SymbolsThicknessOffsetModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_GetMutable_symbolsThicknessOffsetModifier", ExactSpelling = true)]
                extern static float *__MR_SymbolMeshParams_GetMutable_symbolsThicknessOffsetModifier(_Underlying *_this);
                return ref *__MR_SymbolMeshParams_GetMutable_symbolsThicknessOffsetModifier(_UnderlyingPtr);
            }
        }

        // alignment of the text inside bbox
        public new unsafe ref MR.AlignType Align
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_GetMutable_align", ExactSpelling = true)]
                extern static MR.AlignType *__MR_SymbolMeshParams_GetMutable_align(_Underlying *_this);
                return ref *__MR_SymbolMeshParams_GetMutable_align(_UnderlyingPtr);
            }
        }

        // Path to font file
        public new unsafe MR.Std.Filesystem.Path PathToFontFile
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_GetMutable_pathToFontFile", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Path._Underlying *__MR_SymbolMeshParams_GetMutable_pathToFontFile(_Underlying *_this);
                return new(__MR_SymbolMeshParams_GetMutable_pathToFontFile(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SymbolMeshParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_DefaultConstruct();
            _UnderlyingPtr = __MR_SymbolMeshParams_DefaultConstruct();
        }

        /// Constructs `MR::SymbolMeshParams` elementwise.
        public unsafe SymbolMeshParams(ReadOnlySpan<char> text, int fontDetalization, MR.Vector2f symbolsDistanceAdditionalOffset, float symbolsThicknessOffsetModifier, MR.AlignType align, ReadOnlySpan<char> pathToFontFile) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_ConstructFrom(byte *text, byte *text_end, int fontDetalization, MR.Vector2f symbolsDistanceAdditionalOffset, float symbolsThicknessOffsetModifier, MR.AlignType align, byte *pathToFontFile, byte *pathToFontFile_end);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                byte[] __bytes_pathToFontFile = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(pathToFontFile.Length)];
                int __len_pathToFontFile = System.Text.Encoding.UTF8.GetBytes(pathToFontFile, __bytes_pathToFontFile);
                fixed (byte *__ptr_pathToFontFile = __bytes_pathToFontFile)
                {
                    _UnderlyingPtr = __MR_SymbolMeshParams_ConstructFrom(__ptr_text, __ptr_text + __len_text, fontDetalization, symbolsDistanceAdditionalOffset, symbolsThicknessOffsetModifier, align, __ptr_pathToFontFile, __ptr_pathToFontFile + __len_pathToFontFile);
                }
            }
        }

        /// Generated from constructor `MR::SymbolMeshParams::SymbolMeshParams`.
        public unsafe SymbolMeshParams(MR._ByValue_SymbolMeshParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SymbolMeshParams._Underlying *_other);
            _UnderlyingPtr = __MR_SymbolMeshParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SymbolMeshParams::operator=`.
        public unsafe MR.SymbolMeshParams Assign(MR._ByValue_SymbolMeshParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SymbolMeshParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_SymbolMeshParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SymbolMeshParams._Underlying *_other);
            return new(__MR_SymbolMeshParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SymbolMeshParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SymbolMeshParams`/`Const_SymbolMeshParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SymbolMeshParams
    {
        internal readonly Const_SymbolMeshParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SymbolMeshParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SymbolMeshParams(Const_SymbolMeshParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SymbolMeshParams(Const_SymbolMeshParams arg) {return new(arg);}
        public _ByValue_SymbolMeshParams(MR.Misc._Moved<SymbolMeshParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SymbolMeshParams(MR.Misc._Moved<SymbolMeshParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SymbolMeshParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SymbolMeshParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymbolMeshParams`/`Const_SymbolMeshParams` directly.
    public class _InOptMut_SymbolMeshParams
    {
        public SymbolMeshParams? Opt;

        public _InOptMut_SymbolMeshParams() {}
        public _InOptMut_SymbolMeshParams(SymbolMeshParams value) {Opt = value;}
        public static implicit operator _InOptMut_SymbolMeshParams(SymbolMeshParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `SymbolMeshParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SymbolMeshParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SymbolMeshParams`/`Const_SymbolMeshParams` to pass it to the function.
    public class _InOptConst_SymbolMeshParams
    {
        public Const_SymbolMeshParams? Opt;

        public _InOptConst_SymbolMeshParams() {}
        public _InOptConst_SymbolMeshParams(Const_SymbolMeshParams value) {Opt = value;}
        public static implicit operator _InOptConst_SymbolMeshParams(Const_SymbolMeshParams value) {return new(value);}
    }

    // converts text string into set of contours
    /// Generated from function `MR::createSymbolContours`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdVectorMRVector2f_StdString> CreateSymbolContours(MR.Const_SymbolMeshParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_createSymbolContours", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdVectorMRVector2f_StdString._Underlying *__MR_createSymbolContours(MR.Const_SymbolMeshParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_StdVectorStdVectorMRVector2f_StdString(__MR_createSymbolContours(params_._UnderlyingPtr), is_owning: true));
    }

    // converts text string into Z-facing symbol mesh
    /// Generated from function `MR::createSymbolsMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> CreateSymbolsMesh(MR.Const_SymbolMeshParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_createSymbolsMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_createSymbolsMesh(MR.Const_SymbolMeshParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_createSymbolsMesh(params_._UnderlyingPtr), is_owning: true));
    }
}
