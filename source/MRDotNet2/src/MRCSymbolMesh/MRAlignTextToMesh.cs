public static partial class MR
{
    /// Generated from class `MR::TextMeshAlignParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SymbolMeshParams`
    /// This is the const half of the class.
    public class Const_TextMeshAlignParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TextMeshAlignParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Destroy", ExactSpelling = true)]
            extern static void __MR_TextMeshAlignParams_Destroy(_Underlying *_this);
            __MR_TextMeshAlignParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TextMeshAlignParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_SymbolMeshParams(Const_TextMeshAlignParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_UpcastTo_MR_SymbolMeshParams", ExactSpelling = true)]
            extern static MR.Const_SymbolMeshParams._Underlying *__MR_TextMeshAlignParams_UpcastTo_MR_SymbolMeshParams(_Underlying *_this);
            MR.Const_SymbolMeshParams ret = new(__MR_TextMeshAlignParams_UpcastTo_MR_SymbolMeshParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Start coordinate on mesh, represent lowest left corner of text
        public unsafe MR.Const_MeshTriPoint StartPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_startPoint", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_TextMeshAlignParams_Get_startPoint(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_Get_startPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        // Position of the startPoint in a text bounding box
        // (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
        public unsafe MR.Const_Vector2f PivotPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_pivotPoint", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_TextMeshAlignParams_Get_pivotPoint(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_Get_pivotPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        // Direction of text
        public unsafe MR.Const_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_direction", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_TextMeshAlignParams_Get_direction(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_Get_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        // Text normal to surface, if nullptr - use mesh normal at `startPoint`
        public unsafe ref readonly MR.Vector3f * TextNormal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_textNormal", ExactSpelling = true)]
                extern static MR.Vector3f **__MR_TextMeshAlignParams_Get_textNormal(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_Get_textNormal(_UnderlyingPtr);
            }
        }

        // Font height, meters
        public unsafe float FontHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_fontHeight", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_Get_fontHeight(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_fontHeight(_UnderlyingPtr);
            }
        }

        // Text mesh inside and outside offset of input mesh
        public unsafe float SurfaceOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_surfaceOffset", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_Get_surfaceOffset(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_surfaceOffset(_UnderlyingPtr);
            }
        }

        // Maximum possible movement of text mesh alignment, meters
        public unsafe float TextMaximumMovement
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_textMaximumMovement", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_Get_textMaximumMovement(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_textMaximumMovement(_UnderlyingPtr);
            }
        }

        // If true then size of each symbol will be calculated from font height, otherwise - on bounding box of the text
        public unsafe bool FontBasedSizeCalc
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_fontBasedSizeCalc", ExactSpelling = true)]
                extern static bool *__MR_TextMeshAlignParams_Get_fontBasedSizeCalc(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_fontBasedSizeCalc(_UnderlyingPtr);
            }
        }

        // max font size with 128 << 6 FT_F26Dot6 font size
        public static unsafe float MaxGeneratedFontHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_MaxGeneratedFontHeight", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_Get_MaxGeneratedFontHeight();
                return *__MR_TextMeshAlignParams_Get_MaxGeneratedFontHeight();
            }
        }

        // Text that will be made mesh
        public unsafe MR.Std.Const_String Text
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_text", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_TextMeshAlignParams_Get_text(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_Get_text(_UnderlyingPtr), is_owning: false);
            }
        }

        // Detailization of Bezier curves on font glyphs
        public unsafe int FontDetalization
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_fontDetalization", ExactSpelling = true)]
                extern static int *__MR_TextMeshAlignParams_Get_fontDetalization(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_fontDetalization(_UnderlyingPtr);
            }
        }

        // Additional offset between symbols
        // X: In symbol size: 1.0f adds one "space", 0.5 adds half "space".
        // Y: In symbol size: 1.0f adds one base height, 0.5 adds half base height
        public unsafe MR.Const_Vector2f SymbolsDistanceAdditionalOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_symbolsDistanceAdditionalOffset", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_TextMeshAlignParams_Get_symbolsDistanceAdditionalOffset(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_Get_symbolsDistanceAdditionalOffset(_UnderlyingPtr), is_owning: false);
            }
        }

        // Symbols thickness will be modified by this value (newThickness = modifier*baseSymbolHeight + defaultThickness)
        // note: changing this to non-zero values cause costly calculations
        public unsafe float SymbolsThicknessOffsetModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_symbolsThicknessOffsetModifier", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_Get_symbolsThicknessOffsetModifier(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_symbolsThicknessOffsetModifier(_UnderlyingPtr);
            }
        }

        // alignment of the text inside bbox
        public unsafe MR.AlignType Align
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_align", ExactSpelling = true)]
                extern static MR.AlignType *__MR_TextMeshAlignParams_Get_align(_Underlying *_this);
                return *__MR_TextMeshAlignParams_Get_align(_UnderlyingPtr);
            }
        }

        // Path to font file
        public unsafe MR.Std.Filesystem.Const_Path PathToFontFile
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_Get_pathToFontFile", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Const_Path._Underlying *__MR_TextMeshAlignParams_Get_pathToFontFile(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_Get_pathToFontFile(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TextMeshAlignParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextMeshAlignParams._Underlying *__MR_TextMeshAlignParams_DefaultConstruct();
            _UnderlyingPtr = __MR_TextMeshAlignParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::TextMeshAlignParams::TextMeshAlignParams`.
        public unsafe Const_TextMeshAlignParams(MR._ByValue_TextMeshAlignParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TextMeshAlignParams._Underlying *__MR_TextMeshAlignParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TextMeshAlignParams._Underlying *_other);
            _UnderlyingPtr = __MR_TextMeshAlignParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::TextMeshAlignParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SymbolMeshParams`
    /// This is the non-const half of the class.
    public class TextMeshAlignParams : Const_TextMeshAlignParams
    {
        internal unsafe TextMeshAlignParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.SymbolMeshParams(TextMeshAlignParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_UpcastTo_MR_SymbolMeshParams", ExactSpelling = true)]
            extern static MR.SymbolMeshParams._Underlying *__MR_TextMeshAlignParams_UpcastTo_MR_SymbolMeshParams(_Underlying *_this);
            MR.SymbolMeshParams ret = new(__MR_TextMeshAlignParams_UpcastTo_MR_SymbolMeshParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Start coordinate on mesh, represent lowest left corner of text
        public new unsafe MR.MeshTriPoint StartPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_startPoint", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_TextMeshAlignParams_GetMutable_startPoint(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_GetMutable_startPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        // Position of the startPoint in a text bounding box
        // (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
        public new unsafe MR.Mut_Vector2f PivotPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_pivotPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_TextMeshAlignParams_GetMutable_pivotPoint(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_GetMutable_pivotPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        // Direction of text
        public new unsafe MR.Mut_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_direction", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_TextMeshAlignParams_GetMutable_direction(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_GetMutable_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        // Text normal to surface, if nullptr - use mesh normal at `startPoint`
        public new unsafe ref readonly MR.Vector3f * TextNormal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_textNormal", ExactSpelling = true)]
                extern static MR.Vector3f **__MR_TextMeshAlignParams_GetMutable_textNormal(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_textNormal(_UnderlyingPtr);
            }
        }

        // Font height, meters
        public new unsafe ref float FontHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_fontHeight", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_GetMutable_fontHeight(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_fontHeight(_UnderlyingPtr);
            }
        }

        // Text mesh inside and outside offset of input mesh
        public new unsafe ref float SurfaceOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_surfaceOffset", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_GetMutable_surfaceOffset(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_surfaceOffset(_UnderlyingPtr);
            }
        }

        // Maximum possible movement of text mesh alignment, meters
        public new unsafe ref float TextMaximumMovement
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_textMaximumMovement", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_GetMutable_textMaximumMovement(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_textMaximumMovement(_UnderlyingPtr);
            }
        }

        // If true then size of each symbol will be calculated from font height, otherwise - on bounding box of the text
        public new unsafe ref bool FontBasedSizeCalc
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_fontBasedSizeCalc", ExactSpelling = true)]
                extern static bool *__MR_TextMeshAlignParams_GetMutable_fontBasedSizeCalc(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_fontBasedSizeCalc(_UnderlyingPtr);
            }
        }

        // Text that will be made mesh
        public new unsafe MR.Std.String Text
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_text", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_TextMeshAlignParams_GetMutable_text(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_GetMutable_text(_UnderlyingPtr), is_owning: false);
            }
        }

        // Detailization of Bezier curves on font glyphs
        public new unsafe ref int FontDetalization
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_fontDetalization", ExactSpelling = true)]
                extern static int *__MR_TextMeshAlignParams_GetMutable_fontDetalization(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_fontDetalization(_UnderlyingPtr);
            }
        }

        // Additional offset between symbols
        // X: In symbol size: 1.0f adds one "space", 0.5 adds half "space".
        // Y: In symbol size: 1.0f adds one base height, 0.5 adds half base height
        public new unsafe MR.Mut_Vector2f SymbolsDistanceAdditionalOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_symbolsDistanceAdditionalOffset", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_TextMeshAlignParams_GetMutable_symbolsDistanceAdditionalOffset(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_GetMutable_symbolsDistanceAdditionalOffset(_UnderlyingPtr), is_owning: false);
            }
        }

        // Symbols thickness will be modified by this value (newThickness = modifier*baseSymbolHeight + defaultThickness)
        // note: changing this to non-zero values cause costly calculations
        public new unsafe ref float SymbolsThicknessOffsetModifier
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_symbolsThicknessOffsetModifier", ExactSpelling = true)]
                extern static float *__MR_TextMeshAlignParams_GetMutable_symbolsThicknessOffsetModifier(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_symbolsThicknessOffsetModifier(_UnderlyingPtr);
            }
        }

        // alignment of the text inside bbox
        public new unsafe ref MR.AlignType Align
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_align", ExactSpelling = true)]
                extern static MR.AlignType *__MR_TextMeshAlignParams_GetMutable_align(_Underlying *_this);
                return ref *__MR_TextMeshAlignParams_GetMutable_align(_UnderlyingPtr);
            }
        }

        // Path to font file
        public new unsafe MR.Std.Filesystem.Path PathToFontFile
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_GetMutable_pathToFontFile", ExactSpelling = true)]
                extern static MR.Std.Filesystem.Path._Underlying *__MR_TextMeshAlignParams_GetMutable_pathToFontFile(_Underlying *_this);
                return new(__MR_TextMeshAlignParams_GetMutable_pathToFontFile(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TextMeshAlignParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TextMeshAlignParams._Underlying *__MR_TextMeshAlignParams_DefaultConstruct();
            _UnderlyingPtr = __MR_TextMeshAlignParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::TextMeshAlignParams::TextMeshAlignParams`.
        public unsafe TextMeshAlignParams(MR._ByValue_TextMeshAlignParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TextMeshAlignParams._Underlying *__MR_TextMeshAlignParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TextMeshAlignParams._Underlying *_other);
            _UnderlyingPtr = __MR_TextMeshAlignParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::TextMeshAlignParams::operator=`.
        public unsafe MR.TextMeshAlignParams Assign(MR._ByValue_TextMeshAlignParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TextMeshAlignParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TextMeshAlignParams._Underlying *__MR_TextMeshAlignParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TextMeshAlignParams._Underlying *_other);
            return new(__MR_TextMeshAlignParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TextMeshAlignParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TextMeshAlignParams`/`Const_TextMeshAlignParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TextMeshAlignParams
    {
        internal readonly Const_TextMeshAlignParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TextMeshAlignParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TextMeshAlignParams(Const_TextMeshAlignParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TextMeshAlignParams(Const_TextMeshAlignParams arg) {return new(arg);}
        public _ByValue_TextMeshAlignParams(MR.Misc._Moved<TextMeshAlignParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TextMeshAlignParams(MR.Misc._Moved<TextMeshAlignParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TextMeshAlignParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TextMeshAlignParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TextMeshAlignParams`/`Const_TextMeshAlignParams` directly.
    public class _InOptMut_TextMeshAlignParams
    {
        public TextMeshAlignParams? Opt;

        public _InOptMut_TextMeshAlignParams() {}
        public _InOptMut_TextMeshAlignParams(TextMeshAlignParams value) {Opt = value;}
        public static implicit operator _InOptMut_TextMeshAlignParams(TextMeshAlignParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `TextMeshAlignParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TextMeshAlignParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TextMeshAlignParams`/`Const_TextMeshAlignParams` to pass it to the function.
    public class _InOptConst_TextMeshAlignParams
    {
        public Const_TextMeshAlignParams? Opt;

        public _InOptConst_TextMeshAlignParams() {}
        public _InOptConst_TextMeshAlignParams(Const_TextMeshAlignParams value) {Opt = value;}
        public static implicit operator _InOptConst_TextMeshAlignParams(Const_TextMeshAlignParams value) {return new(value);}
    }

    // Creates symbol mesh and aligns it to given surface
    /// Generated from function `MR::alignTextToMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> AlignTextToMesh(MR.Const_Mesh mesh, MR.Const_TextMeshAlignParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_alignTextToMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_alignTextToMesh(MR.Const_Mesh._Underlying *mesh, MR.Const_TextMeshAlignParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_alignTextToMesh(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
