public static partial class MR
{
    /// Fonts included in libharu
    /// please note that using default font does not allow UTF-8 encoding
    public enum PdfBuildinFont : int
    {
        Courier = 0,
        CourierBold = 1,
        CourierOblique = 2,
        CourierBoldOblique = 3,
        Helvetica = 4,
        HelveticaBold = 5,
        HelveticaOblique = 6,
        HelveticaBoldOblique = 7,
        TimesRoman = 8,
        TimesBold = 9,
        TimesItalic = 10,
        TimesBoldItalic = 11,
        Symbol = 12,
        ZapfDingbats = 13,
        Count = 14,
    }

    /**
    * @brief Parameters of document style
    */
    /// Generated from class `MR::PdfParameters`.
    /// This is the const half of the class.
    public class Const_PdfParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PdfParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_PdfParameters_Destroy(_Underlying *_this);
            __MR_PdfParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PdfParameters() {Dispose(false);}

        public unsafe float TitleSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Get_titleSize", ExactSpelling = true)]
                extern static float *__MR_PdfParameters_Get_titleSize(_Underlying *_this);
                return *__MR_PdfParameters_Get_titleSize(_UnderlyingPtr);
            }
        }

        public unsafe float TextSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Get_textSize", ExactSpelling = true)]
                extern static float *__MR_PdfParameters_Get_textSize(_Underlying *_this);
                return *__MR_PdfParameters_Get_textSize(_UnderlyingPtr);
            }
        }

        /**
        * @brief Font name
        */
        public unsafe MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath DefaultFont
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Get_defaultFont", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_Get_defaultFont(_Underlying *_this);
                return new(__MR_PdfParameters_Get_defaultFont(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath DefaultFontBold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Get_defaultFontBold", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_Get_defaultFontBold(_Underlying *_this);
                return new(__MR_PdfParameters_Get_defaultFontBold(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * Font name for table (monospaced)
        */
        public unsafe MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath TableFont
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Get_tableFont", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_Get_tableFont(_Underlying *_this);
                return new(__MR_PdfParameters_Get_tableFont(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath TableFontBold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_Get_tableFontBold", ExactSpelling = true)]
                extern static MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_Get_tableFontBold(_Underlying *_this);
                return new(__MR_PdfParameters_Get_tableFontBold(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PdfParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_PdfParameters_DefaultConstruct();
        }

        /// Constructs `MR::PdfParameters` elementwise.
        public unsafe Const_PdfParameters(float titleSize, float textSize, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath defaultFont, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath defaultFontBold, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath tableFont, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath tableFontBold) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_ConstructFrom(float titleSize, float textSize, MR.Misc._PassBy defaultFont_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *defaultFont, MR.Misc._PassBy defaultFontBold_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *defaultFontBold, MR.Misc._PassBy tableFont_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *tableFont, MR.Misc._PassBy tableFontBold_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *tableFontBold);
            _UnderlyingPtr = __MR_PdfParameters_ConstructFrom(titleSize, textSize, defaultFont.PassByMode, defaultFont.Value is not null ? defaultFont.Value._UnderlyingPtr : null, defaultFontBold.PassByMode, defaultFontBold.Value is not null ? defaultFontBold.Value._UnderlyingPtr : null, tableFont.PassByMode, tableFont.Value is not null ? tableFont.Value._UnderlyingPtr : null, tableFontBold.PassByMode, tableFontBold.Value is not null ? tableFontBold.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PdfParameters::PdfParameters`.
        public unsafe Const_PdfParameters(MR._ByValue_PdfParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PdfParameters._Underlying *_other);
            _UnderlyingPtr = __MR_PdfParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * @brief Parameters of document style
    */
    /// Generated from class `MR::PdfParameters`.
    /// This is the non-const half of the class.
    public class PdfParameters : Const_PdfParameters
    {
        internal unsafe PdfParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref float TitleSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_GetMutable_titleSize", ExactSpelling = true)]
                extern static float *__MR_PdfParameters_GetMutable_titleSize(_Underlying *_this);
                return ref *__MR_PdfParameters_GetMutable_titleSize(_UnderlyingPtr);
            }
        }

        public new unsafe ref float TextSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_GetMutable_textSize", ExactSpelling = true)]
                extern static float *__MR_PdfParameters_GetMutable_textSize(_Underlying *_this);
                return ref *__MR_PdfParameters_GetMutable_textSize(_UnderlyingPtr);
            }
        }

        /**
        * @brief Font name
        */
        public new unsafe MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath DefaultFont
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_GetMutable_defaultFont", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_GetMutable_defaultFont(_Underlying *_this);
                return new(__MR_PdfParameters_GetMutable_defaultFont(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath DefaultFontBold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_GetMutable_defaultFontBold", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_GetMutable_defaultFontBold(_Underlying *_this);
                return new(__MR_PdfParameters_GetMutable_defaultFontBold(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * Font name for table (monospaced)
        */
        public new unsafe MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath TableFont
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_GetMutable_tableFont", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_GetMutable_tableFont(_Underlying *_this);
                return new(__MR_PdfParameters_GetMutable_tableFont(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath TableFontBold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_GetMutable_tableFontBold", ExactSpelling = true)]
                extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_PdfParameters_GetMutable_tableFontBold(_Underlying *_this);
                return new(__MR_PdfParameters_GetMutable_tableFontBold(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PdfParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_PdfParameters_DefaultConstruct();
        }

        /// Constructs `MR::PdfParameters` elementwise.
        public unsafe PdfParameters(float titleSize, float textSize, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath defaultFont, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath defaultFontBold, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath tableFont, MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath tableFontBold) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_ConstructFrom(float titleSize, float textSize, MR.Misc._PassBy defaultFont_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *defaultFont, MR.Misc._PassBy defaultFontBold_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *defaultFontBold, MR.Misc._PassBy tableFont_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *tableFont, MR.Misc._PassBy tableFontBold_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *tableFontBold);
            _UnderlyingPtr = __MR_PdfParameters_ConstructFrom(titleSize, textSize, defaultFont.PassByMode, defaultFont.Value is not null ? defaultFont.Value._UnderlyingPtr : null, defaultFontBold.PassByMode, defaultFontBold.Value is not null ? defaultFontBold.Value._UnderlyingPtr : null, tableFont.PassByMode, tableFont.Value is not null ? tableFont.Value._UnderlyingPtr : null, tableFontBold.PassByMode, tableFontBold.Value is not null ? tableFontBold.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PdfParameters::PdfParameters`.
        public unsafe PdfParameters(MR._ByValue_PdfParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PdfParameters._Underlying *_other);
            _UnderlyingPtr = __MR_PdfParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PdfParameters::operator=`.
        public unsafe MR.PdfParameters Assign(MR._ByValue_PdfParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PdfParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PdfParameters._Underlying *__MR_PdfParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PdfParameters._Underlying *_other);
            return new(__MR_PdfParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PdfParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PdfParameters`/`Const_PdfParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PdfParameters
    {
        internal readonly Const_PdfParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PdfParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PdfParameters(Const_PdfParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PdfParameters(Const_PdfParameters arg) {return new(arg);}
        public _ByValue_PdfParameters(MR.Misc._Moved<PdfParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PdfParameters(MR.Misc._Moved<PdfParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PdfParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PdfParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PdfParameters`/`Const_PdfParameters` directly.
    public class _InOptMut_PdfParameters
    {
        public PdfParameters? Opt;

        public _InOptMut_PdfParameters() {}
        public _InOptMut_PdfParameters(PdfParameters value) {Opt = value;}
        public static implicit operator _InOptMut_PdfParameters(PdfParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `PdfParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PdfParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PdfParameters`/`Const_PdfParameters` to pass it to the function.
    public class _InOptConst_PdfParameters
    {
        public Const_PdfParameters? Opt;

        public _InOptConst_PdfParameters() {}
        public _InOptConst_PdfParameters(Const_PdfParameters value) {Opt = value;}
        public static implicit operator _InOptConst_PdfParameters(Const_PdfParameters value) {return new(value);}
    }

    /**
    * Class for simple creation pdf.
    */
    /// Generated from class `MR::Pdf`.
    /// This is the const half of the class.
    public class Const_Pdf : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Pdf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Destroy", ExactSpelling = true)]
            extern static void __MR_Pdf_Destroy(_Underlying *_this);
            __MR_Pdf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Pdf() {Dispose(false);}

        /// Ctor. Create a document, but not a page. To create a new page use newPage() method
        /// Generated from constructor `MR::Pdf::Pdf`.
        /// Parameter `params_` defaults to `MR::PdfParameters()`.
        public unsafe Const_Pdf(MR.Const_PdfParameters? params_ = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Construct", ExactSpelling = true)]
            extern static MR.Pdf._Underlying *__MR_Pdf_Construct(MR.Const_PdfParameters._Underlying *params_);
            _UnderlyingPtr = __MR_Pdf_Construct(params_ is not null ? params_._UnderlyingPtr : null);
        }

        /// Ctor. Create a document, but not a page. To create a new page use newPage() method
        /// Generated from constructor `MR::Pdf::Pdf`.
        /// Parameter `params_` defaults to `MR::PdfParameters()`.
        public static unsafe implicit operator Const_Pdf(MR.Const_PdfParameters? params_) {return new(params_);}

        /// Generated from constructor `MR::Pdf::Pdf`.
        public unsafe Const_Pdf(MR._ByValue_Pdf other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Pdf._Underlying *__MR_Pdf_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Pdf._Underlying *other);
            _UnderlyingPtr = __MR_Pdf_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Checking the ability to work with a document
        /// Generated from conversion operator `MR::Pdf::operator bool`.
        public static unsafe implicit operator bool(MR.Const_Pdf _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_Pdf_ConvertTo_bool(MR.Const_Pdf._Underlying *_this);
            return __MR_Pdf_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Pdf::getCursorPosX`.
        public unsafe float GetCursorPosX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_getCursorPosX", ExactSpelling = true)]
            extern static float __MR_Pdf_getCursorPosX(_Underlying *_this);
            return __MR_Pdf_getCursorPosX(_UnderlyingPtr);
        }

        /// Generated from method `MR::Pdf::getCursorPosY`.
        public unsafe float GetCursorPosY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_getCursorPosY", ExactSpelling = true)]
            extern static float __MR_Pdf_getCursorPosY(_Underlying *_this);
            return __MR_Pdf_getCursorPosY(_UnderlyingPtr);
        }

        /// Generated from method `MR::Pdf::getPageSize`.
        public unsafe MR.Vector2f GetPageSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_getPageSize", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Pdf_getPageSize(_Underlying *_this);
            return __MR_Pdf_getPageSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::Pdf::getPageWorkArea`.
        public unsafe MR.Box2f GetPageWorkArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_getPageWorkArea", ExactSpelling = true)]
            extern static MR.Box2f __MR_Pdf_getPageWorkArea(_Underlying *_this);
            return __MR_Pdf_getPageWorkArea(_UnderlyingPtr);
        }

        public enum AlignmentHorizontal : int
        {
            Left = 0,
            Center = 1,
            Right = 2,
        }

        // Table part
        // class to convert values to string with set format
        /// Generated from class `MR::Pdf::Cell`.
        /// This is the const half of the class.
        public class Const_Cell : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Cell(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Destroy", ExactSpelling = true)]
                extern static void __MR_Pdf_Cell_Destroy(_Underlying *_this);
                __MR_Pdf_Cell_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Cell() {Dispose(false);}

            public unsafe MR.Std.Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty Data
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Get_data", ExactSpelling = true)]
                    extern static MR.Std.Const_Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_Pdf_Cell_Get_data(_Underlying *_this);
                    return new(__MR_Pdf_Cell_Get_data(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Cell() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.Cell._Underlying *__MR_Pdf_Cell_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_Cell_DefaultConstruct();
            }

            /// Generated from constructor `MR::Pdf::Cell::Cell`.
            public unsafe Const_Cell(MR.Pdf._ByValue_Cell _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.Cell._Underlying *__MR_Pdf_Cell_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.Cell._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_Cell_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            // get strang from contained value
            // \param fmtStr format string like fmt::format
            /// Generated from method `MR::Pdf::Cell::toString`.
            /// Parameter `fmtStr` defaults to `"{}"`.
            public unsafe MR.Misc._Moved<MR.Std.String> ToString(MR.Misc.ReadOnlyCharSpanOpt fmtStr = new())
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_toString", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_Pdf_Cell_toString(_Underlying *_this, byte *fmtStr, byte *fmtStr_end);
                byte[] __bytes_fmtStr;
                int __len_fmtStr = 0;
                if (fmtStr.HasValue)
                {
                    __bytes_fmtStr = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(fmtStr.Value.Length)];
                    __len_fmtStr = System.Text.Encoding.UTF8.GetBytes(fmtStr.Value, __bytes_fmtStr);
                }
                fixed (byte *__ptr_fmtStr = __bytes_fmtStr)
                {
                    return MR.Misc.Move(new MR.Std.String(__MR_Pdf_Cell_toString(_UnderlyingPtr, fmtStr.HasValue ? __ptr_fmtStr : null, fmtStr.HasValue ? __ptr_fmtStr + __len_fmtStr : null), is_owning: true));
                }
            }

            /// Generated from class `MR::Pdf::Cell::Empty`.
            /// This is the const half of the class.
            public class Const_Empty : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Empty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Empty_Destroy", ExactSpelling = true)]
                    extern static void __MR_Pdf_Cell_Empty_Destroy(_Underlying *_this);
                    __MR_Pdf_Cell_Empty_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Empty() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Empty() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Empty_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Pdf.Cell.Empty._Underlying *__MR_Pdf_Cell_Empty_DefaultConstruct();
                    _UnderlyingPtr = __MR_Pdf_Cell_Empty_DefaultConstruct();
                }

                /// Generated from constructor `MR::Pdf::Cell::Empty::Empty`.
                public unsafe Const_Empty(MR.Pdf.Cell.Const_Empty _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Empty_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Pdf.Cell.Empty._Underlying *__MR_Pdf_Cell_Empty_ConstructFromAnother(MR.Pdf.Cell.Empty._Underlying *_other);
                    _UnderlyingPtr = __MR_Pdf_Cell_Empty_ConstructFromAnother(_other._UnderlyingPtr);
                }
            }

            /// Generated from class `MR::Pdf::Cell::Empty`.
            /// This is the non-const half of the class.
            public class Empty : Const_Empty
            {
                internal unsafe Empty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Empty() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Empty_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Pdf.Cell.Empty._Underlying *__MR_Pdf_Cell_Empty_DefaultConstruct();
                    _UnderlyingPtr = __MR_Pdf_Cell_Empty_DefaultConstruct();
                }

                /// Generated from constructor `MR::Pdf::Cell::Empty::Empty`.
                public unsafe Empty(MR.Pdf.Cell.Const_Empty _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Empty_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Pdf.Cell.Empty._Underlying *__MR_Pdf_Cell_Empty_ConstructFromAnother(MR.Pdf.Cell.Empty._Underlying *_other);
                    _UnderlyingPtr = __MR_Pdf_Cell_Empty_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Pdf::Cell::Empty::operator=`.
                public unsafe MR.Pdf.Cell.Empty Assign(MR.Pdf.Cell.Const_Empty _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_Empty_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Pdf.Cell.Empty._Underlying *__MR_Pdf_Cell_Empty_AssignFromAnother(_Underlying *_this, MR.Pdf.Cell.Empty._Underlying *_other);
                    return new(__MR_Pdf_Cell_Empty_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Empty` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Empty`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Empty`/`Const_Empty` directly.
            public class _InOptMut_Empty
            {
                public Empty? Opt;

                public _InOptMut_Empty() {}
                public _InOptMut_Empty(Empty value) {Opt = value;}
                public static implicit operator _InOptMut_Empty(Empty value) {return new(value);}
            }

            /// This is used for optional parameters of class `Empty` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Empty`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Empty`/`Const_Empty` to pass it to the function.
            public class _InOptConst_Empty
            {
                public Const_Empty? Opt;

                public _InOptConst_Empty() {}
                public _InOptConst_Empty(Const_Empty value) {Opt = value;}
                public static implicit operator _InOptConst_Empty(Const_Empty value) {return new(value);}
            }
        }

        // Table part
        // class to convert values to string with set format
        /// Generated from class `MR::Pdf::Cell`.
        /// This is the non-const half of the class.
        public class Cell : Const_Cell
        {
            internal unsafe Cell(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty Data
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_GetMutable_data", ExactSpelling = true)]
                    extern static MR.Std.Variant_Int_Float_Bool_StdString_MRPdfCellEmpty._Underlying *__MR_Pdf_Cell_GetMutable_data(_Underlying *_this);
                    return new(__MR_Pdf_Cell_GetMutable_data(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Cell() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.Cell._Underlying *__MR_Pdf_Cell_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_Cell_DefaultConstruct();
            }

            /// Generated from constructor `MR::Pdf::Cell::Cell`.
            public unsafe Cell(MR.Pdf._ByValue_Cell _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.Cell._Underlying *__MR_Pdf_Cell_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.Cell._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_Cell_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Pdf::Cell::operator=`.
            public unsafe MR.Pdf.Cell Assign(MR.Pdf._ByValue_Cell _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Cell_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.Cell._Underlying *__MR_Pdf_Cell_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Pdf.Cell._Underlying *_other);
                return new(__MR_Pdf_Cell_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Cell` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Cell`/`Const_Cell` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Cell
        {
            internal readonly Const_Cell? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Cell() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Cell(Const_Cell new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Cell(Const_Cell arg) {return new(arg);}
            public _ByValue_Cell(MR.Misc._Moved<Cell> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Cell(MR.Misc._Moved<Cell> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Cell` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Cell`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Cell`/`Const_Cell` directly.
        public class _InOptMut_Cell
        {
            public Cell? Opt;

            public _InOptMut_Cell() {}
            public _InOptMut_Cell(Cell value) {Opt = value;}
            public static implicit operator _InOptMut_Cell(Cell value) {return new(value);}
        }

        /// This is used for optional parameters of class `Cell` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Cell`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Cell`/`Const_Cell` to pass it to the function.
        public class _InOptConst_Cell
        {
            public Const_Cell? Opt;

            public _InOptConst_Cell() {}
            public _InOptConst_Cell(Const_Cell value) {Opt = value;}
            public static implicit operator _InOptConst_Cell(Const_Cell value) {return new(value);}
        }

        /// Generated from class `MR::Pdf::CellCustomParams`.
        /// This is the const half of the class.
        public class Const_CellCustomParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_CellCustomParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_Destroy", ExactSpelling = true)]
                extern static void __MR_Pdf_CellCustomParams_Destroy(_Underlying *_this);
                __MR_Pdf_CellCustomParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_CellCustomParams() {Dispose(false);}

            public unsafe MR.Std.Const_Optional_MRColor ColorText
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_Get_colorText", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_Pdf_CellCustomParams_Get_colorText(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_Get_colorText(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_Optional_MRColor ColorCellBg
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_Get_colorCellBg", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_Pdf_CellCustomParams_Get_colorCellBg(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_Get_colorCellBg(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_Optional_MRColor ColorCellBorder
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_Get_colorCellBorder", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRColor._Underlying *__MR_Pdf_CellCustomParams_Get_colorCellBorder(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_Get_colorCellBorder(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_Optional_StdString Text
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_Get_text", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_StdString._Underlying *__MR_Pdf_CellCustomParams_Get_text(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_Get_text(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_CellCustomParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_CellCustomParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::CellCustomParams` elementwise.
            public unsafe Const_CellCustomParams(MR._InOpt_Color colorText, MR._InOpt_Color colorCellBg, MR._InOpt_Color colorCellBorder, MR.Misc.ReadOnlyCharSpanOpt text) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_ConstructFrom(MR.Color *colorText, MR.Color *colorCellBg, MR.Color *colorCellBorder, byte *text, byte *text_end);
                byte[] __bytes_text;
                int __len_text = 0;
                if (text.HasValue)
                {
                    __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Value.Length)];
                    __len_text = System.Text.Encoding.UTF8.GetBytes(text.Value, __bytes_text);
                }
                fixed (byte *__ptr_text = __bytes_text)
                {
                    _UnderlyingPtr = __MR_Pdf_CellCustomParams_ConstructFrom(colorText.HasValue ? &colorText.Object : null, colorCellBg.HasValue ? &colorCellBg.Object : null, colorCellBorder.HasValue ? &colorCellBorder.Object : null, text.HasValue ? __ptr_text : null, text.HasValue ? __ptr_text + __len_text : null);
                }
            }

            /// Generated from constructor `MR::Pdf::CellCustomParams::CellCustomParams`.
            public unsafe Const_CellCustomParams(MR.Pdf._ByValue_CellCustomParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.CellCustomParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_CellCustomParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::Pdf::CellCustomParams`.
        /// This is the non-const half of the class.
        public class CellCustomParams : Const_CellCustomParams
        {
            internal unsafe CellCustomParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.Optional_MRColor ColorText
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_GetMutable_colorText", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRColor._Underlying *__MR_Pdf_CellCustomParams_GetMutable_colorText(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_GetMutable_colorText(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.Optional_MRColor ColorCellBg
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_GetMutable_colorCellBg", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRColor._Underlying *__MR_Pdf_CellCustomParams_GetMutable_colorCellBg(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_GetMutable_colorCellBg(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.Optional_MRColor ColorCellBorder
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_GetMutable_colorCellBorder", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRColor._Underlying *__MR_Pdf_CellCustomParams_GetMutable_colorCellBorder(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_GetMutable_colorCellBorder(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.Optional_StdString Text
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_GetMutable_text", ExactSpelling = true)]
                    extern static MR.Std.Optional_StdString._Underlying *__MR_Pdf_CellCustomParams_GetMutable_text(_Underlying *_this);
                    return new(__MR_Pdf_CellCustomParams_GetMutable_text(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe CellCustomParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_CellCustomParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::CellCustomParams` elementwise.
            public unsafe CellCustomParams(MR._InOpt_Color colorText, MR._InOpt_Color colorCellBg, MR._InOpt_Color colorCellBorder, MR.Misc.ReadOnlyCharSpanOpt text) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_ConstructFrom(MR.Color *colorText, MR.Color *colorCellBg, MR.Color *colorCellBorder, byte *text, byte *text_end);
                byte[] __bytes_text;
                int __len_text = 0;
                if (text.HasValue)
                {
                    __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Value.Length)];
                    __len_text = System.Text.Encoding.UTF8.GetBytes(text.Value, __bytes_text);
                }
                fixed (byte *__ptr_text = __bytes_text)
                {
                    _UnderlyingPtr = __MR_Pdf_CellCustomParams_ConstructFrom(colorText.HasValue ? &colorText.Object : null, colorCellBg.HasValue ? &colorCellBg.Object : null, colorCellBorder.HasValue ? &colorCellBorder.Object : null, text.HasValue ? __ptr_text : null, text.HasValue ? __ptr_text + __len_text : null);
                }
            }

            /// Generated from constructor `MR::Pdf::CellCustomParams::CellCustomParams`.
            public unsafe CellCustomParams(MR.Pdf._ByValue_CellCustomParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.CellCustomParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_CellCustomParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Pdf::CellCustomParams::operator=`.
            public unsafe MR.Pdf.CellCustomParams Assign(MR.Pdf._ByValue_CellCustomParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_CellCustomParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.CellCustomParams._Underlying *__MR_Pdf_CellCustomParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Pdf.CellCustomParams._Underlying *_other);
                return new(__MR_Pdf_CellCustomParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `CellCustomParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `CellCustomParams`/`Const_CellCustomParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_CellCustomParams
        {
            internal readonly Const_CellCustomParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_CellCustomParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_CellCustomParams(Const_CellCustomParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_CellCustomParams(Const_CellCustomParams arg) {return new(arg);}
            public _ByValue_CellCustomParams(MR.Misc._Moved<CellCustomParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_CellCustomParams(MR.Misc._Moved<CellCustomParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `CellCustomParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_CellCustomParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CellCustomParams`/`Const_CellCustomParams` directly.
        public class _InOptMut_CellCustomParams
        {
            public CellCustomParams? Opt;

            public _InOptMut_CellCustomParams() {}
            public _InOptMut_CellCustomParams(CellCustomParams value) {Opt = value;}
            public static implicit operator _InOptMut_CellCustomParams(CellCustomParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `CellCustomParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_CellCustomParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `CellCustomParams`/`Const_CellCustomParams` to pass it to the function.
        public class _InOptConst_CellCustomParams
        {
            public Const_CellCustomParams? Opt;

            public _InOptConst_CellCustomParams() {}
            public _InOptConst_CellCustomParams(Const_CellCustomParams value) {Opt = value;}
            public static implicit operator _InOptConst_CellCustomParams(Const_CellCustomParams value) {return new(value);}
        }

        /// Parameters to adding image from file
        /// Generated from class `MR::Pdf::ImageParams`.
        /// This is the const half of the class.
        public class Const_ImageParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ImageParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_Destroy", ExactSpelling = true)]
                extern static void __MR_Pdf_ImageParams_Destroy(_Underlying *_this);
                __MR_Pdf_ImageParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ImageParams() {Dispose(false);}

            /// image size in page space
            /// if == {0, 0} - use image size
            /// if .x or .y < 0 use the available page size from the current cursor position (caption size is also accounted for)
            public unsafe MR.Const_Vector2f Size
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_Get_size", ExactSpelling = true)]
                    extern static MR.Const_Vector2f._Underlying *__MR_Pdf_ImageParams_Get_size(_Underlying *_this);
                    return new(__MR_Pdf_ImageParams_Get_size(_UnderlyingPtr), is_owning: false);
                }
            }

            /// caption if not empty - add caption under marks (if exist) or image.
            public unsafe MR.Std.Const_String Caption
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_Get_caption", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_Pdf_ImageParams_Get_caption(_Underlying *_this);
                    return new(__MR_Pdf_ImageParams_Get_caption(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Pdf.ImageParams.UniformScale UniformScale_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_Get_uniformScale", ExactSpelling = true)]
                    extern static MR.Pdf.ImageParams.UniformScale *__MR_Pdf_ImageParams_Get_uniformScale(_Underlying *_this);
                    return *__MR_Pdf_ImageParams_Get_uniformScale(_UnderlyingPtr);
                }
            }

            public unsafe MR.Pdf.ImageParams.AlignmentVertical AlignmentVertical_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_Get_alignmentVertical", ExactSpelling = true)]
                    extern static MR.Pdf.ImageParams.AlignmentVertical *__MR_Pdf_ImageParams_Get_alignmentVertical(_Underlying *_this);
                    return *__MR_Pdf_ImageParams_Get_alignmentVertical(_UnderlyingPtr);
                }
            }

            public unsafe MR.Pdf.AlignmentHorizontal AlignmentHorizontal
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_Get_alignmentHorizontal", ExactSpelling = true)]
                    extern static MR.Pdf.AlignmentHorizontal *__MR_Pdf_ImageParams_Get_alignmentHorizontal(_Underlying *_this);
                    return *__MR_Pdf_ImageParams_Get_alignmentHorizontal(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ImageParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_ImageParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::ImageParams` elementwise.
            public unsafe Const_ImageParams(MR.Vector2f size, ReadOnlySpan<char> caption, MR.Pdf.ImageParams.UniformScale uniformScale, MR.Pdf.ImageParams.AlignmentVertical alignmentVertical, MR.Pdf.AlignmentHorizontal alignmentHorizontal) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_ConstructFrom(MR.Vector2f size, byte *caption, byte *caption_end, MR.Pdf.ImageParams.UniformScale uniformScale, MR.Pdf.ImageParams.AlignmentVertical alignmentVertical, MR.Pdf.AlignmentHorizontal alignmentHorizontal);
                byte[] __bytes_caption = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(caption.Length)];
                int __len_caption = System.Text.Encoding.UTF8.GetBytes(caption, __bytes_caption);
                fixed (byte *__ptr_caption = __bytes_caption)
                {
                    _UnderlyingPtr = __MR_Pdf_ImageParams_ConstructFrom(size, __ptr_caption, __ptr_caption + __len_caption, uniformScale, alignmentVertical, alignmentHorizontal);
                }
            }

            /// Generated from constructor `MR::Pdf::ImageParams::ImageParams`.
            public unsafe Const_ImageParams(MR.Pdf._ByValue_ImageParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.ImageParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_ImageParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            public enum AlignmentVertical : int
            {
                Top = 0,
                Center = 1,
                Bottom = 2,
            }

            /// set height to keep same scale as width scale
            public enum UniformScale : int
            {
                None = 0,
                FromWidth = 1,
                FromHeight = 2,
                Auto = 3,
            }
        }

        /// Parameters to adding image from file
        /// Generated from class `MR::Pdf::ImageParams`.
        /// This is the non-const half of the class.
        public class ImageParams : Const_ImageParams
        {
            internal unsafe ImageParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// image size in page space
            /// if == {0, 0} - use image size
            /// if .x or .y < 0 use the available page size from the current cursor position (caption size is also accounted for)
            public new unsafe MR.Mut_Vector2f Size
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_GetMutable_size", ExactSpelling = true)]
                    extern static MR.Mut_Vector2f._Underlying *__MR_Pdf_ImageParams_GetMutable_size(_Underlying *_this);
                    return new(__MR_Pdf_ImageParams_GetMutable_size(_UnderlyingPtr), is_owning: false);
                }
            }

            /// caption if not empty - add caption under marks (if exist) or image.
            public new unsafe MR.Std.String Caption
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_GetMutable_caption", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Pdf_ImageParams_GetMutable_caption(_Underlying *_this);
                    return new(__MR_Pdf_ImageParams_GetMutable_caption(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe ref MR.Pdf.ImageParams.UniformScale UniformScale_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_GetMutable_uniformScale", ExactSpelling = true)]
                    extern static MR.Pdf.ImageParams.UniformScale *__MR_Pdf_ImageParams_GetMutable_uniformScale(_Underlying *_this);
                    return ref *__MR_Pdf_ImageParams_GetMutable_uniformScale(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.Pdf.ImageParams.AlignmentVertical AlignmentVertical_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_GetMutable_alignmentVertical", ExactSpelling = true)]
                    extern static MR.Pdf.ImageParams.AlignmentVertical *__MR_Pdf_ImageParams_GetMutable_alignmentVertical(_Underlying *_this);
                    return ref *__MR_Pdf_ImageParams_GetMutable_alignmentVertical(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.Pdf.AlignmentHorizontal AlignmentHorizontal
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_GetMutable_alignmentHorizontal", ExactSpelling = true)]
                    extern static MR.Pdf.AlignmentHorizontal *__MR_Pdf_ImageParams_GetMutable_alignmentHorizontal(_Underlying *_this);
                    return ref *__MR_Pdf_ImageParams_GetMutable_alignmentHorizontal(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ImageParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_ImageParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::ImageParams` elementwise.
            public unsafe ImageParams(MR.Vector2f size, ReadOnlySpan<char> caption, MR.Pdf.ImageParams.UniformScale uniformScale, MR.Pdf.ImageParams.AlignmentVertical alignmentVertical, MR.Pdf.AlignmentHorizontal alignmentHorizontal) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_ConstructFrom(MR.Vector2f size, byte *caption, byte *caption_end, MR.Pdf.ImageParams.UniformScale uniformScale, MR.Pdf.ImageParams.AlignmentVertical alignmentVertical, MR.Pdf.AlignmentHorizontal alignmentHorizontal);
                byte[] __bytes_caption = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(caption.Length)];
                int __len_caption = System.Text.Encoding.UTF8.GetBytes(caption, __bytes_caption);
                fixed (byte *__ptr_caption = __bytes_caption)
                {
                    _UnderlyingPtr = __MR_Pdf_ImageParams_ConstructFrom(size, __ptr_caption, __ptr_caption + __len_caption, uniformScale, alignmentVertical, alignmentHorizontal);
                }
            }

            /// Generated from constructor `MR::Pdf::ImageParams::ImageParams`.
            public unsafe ImageParams(MR.Pdf._ByValue_ImageParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.ImageParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_ImageParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Pdf::ImageParams::operator=`.
            public unsafe MR.Pdf.ImageParams Assign(MR.Pdf._ByValue_ImageParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ImageParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.ImageParams._Underlying *__MR_Pdf_ImageParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Pdf.ImageParams._Underlying *_other);
                return new(__MR_Pdf_ImageParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ImageParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ImageParams`/`Const_ImageParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ImageParams
        {
            internal readonly Const_ImageParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ImageParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ImageParams(Const_ImageParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ImageParams(Const_ImageParams arg) {return new(arg);}
            public _ByValue_ImageParams(MR.Misc._Moved<ImageParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ImageParams(MR.Misc._Moved<ImageParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ImageParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ImageParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ImageParams`/`Const_ImageParams` directly.
        public class _InOptMut_ImageParams
        {
            public ImageParams? Opt;

            public _InOptMut_ImageParams() {}
            public _InOptMut_ImageParams(ImageParams value) {Opt = value;}
            public static implicit operator _InOptMut_ImageParams(ImageParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `ImageParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ImageParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ImageParams`/`Const_ImageParams` to pass it to the function.
        public class _InOptConst_ImageParams
        {
            public Const_ImageParams? Opt;

            public _InOptConst_ImageParams() {}
            public _InOptConst_ImageParams(Const_ImageParams value) {Opt = value;}
            public static implicit operator _InOptConst_ImageParams(Const_ImageParams value) {return new(value);}
        }

        /// Generated from class `MR::Pdf::PaletteRowStats`.
        /// This is the const half of the class.
        public class Const_PaletteRowStats : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_PaletteRowStats(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_Destroy", ExactSpelling = true)]
                extern static void __MR_Pdf_PaletteRowStats_Destroy(_Underlying *_this);
                __MR_Pdf_PaletteRowStats_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_PaletteRowStats() {Dispose(false);}

            public unsafe MR.Const_Color Color
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_Get_color", ExactSpelling = true)]
                    extern static MR.Const_Color._Underlying *__MR_Pdf_PaletteRowStats_Get_color(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_Get_color(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_String RangeMin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_Get_rangeMin", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_Pdf_PaletteRowStats_Get_rangeMin(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_Get_rangeMin(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_String RangeMax
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_Get_rangeMax", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_Pdf_PaletteRowStats_Get_rangeMax(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_Get_rangeMax(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_String Percent
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_Get_percent", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_Pdf_PaletteRowStats_Get_percent(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_Get_percent(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_PaletteRowStats() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_PaletteRowStats_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::PaletteRowStats` elementwise.
            public unsafe Const_PaletteRowStats(MR.Color color, ReadOnlySpan<char> rangeMin, ReadOnlySpan<char> rangeMax, ReadOnlySpan<char> percent) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_ConstructFrom(MR.Color color, byte *rangeMin, byte *rangeMin_end, byte *rangeMax, byte *rangeMax_end, byte *percent, byte *percent_end);
                byte[] __bytes_rangeMin = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(rangeMin.Length)];
                int __len_rangeMin = System.Text.Encoding.UTF8.GetBytes(rangeMin, __bytes_rangeMin);
                fixed (byte *__ptr_rangeMin = __bytes_rangeMin)
                {
                    byte[] __bytes_rangeMax = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(rangeMax.Length)];
                    int __len_rangeMax = System.Text.Encoding.UTF8.GetBytes(rangeMax, __bytes_rangeMax);
                    fixed (byte *__ptr_rangeMax = __bytes_rangeMax)
                    {
                        byte[] __bytes_percent = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(percent.Length)];
                        int __len_percent = System.Text.Encoding.UTF8.GetBytes(percent, __bytes_percent);
                        fixed (byte *__ptr_percent = __bytes_percent)
                        {
                            _UnderlyingPtr = __MR_Pdf_PaletteRowStats_ConstructFrom(color, __ptr_rangeMin, __ptr_rangeMin + __len_rangeMin, __ptr_rangeMax, __ptr_rangeMax + __len_rangeMax, __ptr_percent, __ptr_percent + __len_percent);
                        }
                    }
                }
            }

            /// Generated from constructor `MR::Pdf::PaletteRowStats::PaletteRowStats`.
            public unsafe Const_PaletteRowStats(MR.Pdf._ByValue_PaletteRowStats _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.PaletteRowStats._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_PaletteRowStats_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::Pdf::PaletteRowStats`.
        /// This is the non-const half of the class.
        public class PaletteRowStats : Const_PaletteRowStats
        {
            internal unsafe PaletteRowStats(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Color Color
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_GetMutable_color", ExactSpelling = true)]
                    extern static MR.Mut_Color._Underlying *__MR_Pdf_PaletteRowStats_GetMutable_color(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_GetMutable_color(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.String RangeMin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_GetMutable_rangeMin", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Pdf_PaletteRowStats_GetMutable_rangeMin(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_GetMutable_rangeMin(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.String RangeMax
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_GetMutable_rangeMax", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Pdf_PaletteRowStats_GetMutable_rangeMax(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_GetMutable_rangeMax(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.String Percent
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_GetMutable_percent", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Pdf_PaletteRowStats_GetMutable_percent(_Underlying *_this);
                    return new(__MR_Pdf_PaletteRowStats_GetMutable_percent(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe PaletteRowStats() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_PaletteRowStats_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::PaletteRowStats` elementwise.
            public unsafe PaletteRowStats(MR.Color color, ReadOnlySpan<char> rangeMin, ReadOnlySpan<char> rangeMax, ReadOnlySpan<char> percent) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_ConstructFrom(MR.Color color, byte *rangeMin, byte *rangeMin_end, byte *rangeMax, byte *rangeMax_end, byte *percent, byte *percent_end);
                byte[] __bytes_rangeMin = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(rangeMin.Length)];
                int __len_rangeMin = System.Text.Encoding.UTF8.GetBytes(rangeMin, __bytes_rangeMin);
                fixed (byte *__ptr_rangeMin = __bytes_rangeMin)
                {
                    byte[] __bytes_rangeMax = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(rangeMax.Length)];
                    int __len_rangeMax = System.Text.Encoding.UTF8.GetBytes(rangeMax, __bytes_rangeMax);
                    fixed (byte *__ptr_rangeMax = __bytes_rangeMax)
                    {
                        byte[] __bytes_percent = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(percent.Length)];
                        int __len_percent = System.Text.Encoding.UTF8.GetBytes(percent, __bytes_percent);
                        fixed (byte *__ptr_percent = __bytes_percent)
                        {
                            _UnderlyingPtr = __MR_Pdf_PaletteRowStats_ConstructFrom(color, __ptr_rangeMin, __ptr_rangeMin + __len_rangeMin, __ptr_rangeMax, __ptr_rangeMax + __len_rangeMax, __ptr_percent, __ptr_percent + __len_percent);
                        }
                    }
                }
            }

            /// Generated from constructor `MR::Pdf::PaletteRowStats::PaletteRowStats`.
            public unsafe PaletteRowStats(MR.Pdf._ByValue_PaletteRowStats _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.PaletteRowStats._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_PaletteRowStats_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Pdf::PaletteRowStats::operator=`.
            public unsafe MR.Pdf.PaletteRowStats Assign(MR.Pdf._ByValue_PaletteRowStats _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_PaletteRowStats_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.PaletteRowStats._Underlying *__MR_Pdf_PaletteRowStats_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Pdf.PaletteRowStats._Underlying *_other);
                return new(__MR_Pdf_PaletteRowStats_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `PaletteRowStats` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `PaletteRowStats`/`Const_PaletteRowStats` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_PaletteRowStats
        {
            internal readonly Const_PaletteRowStats? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_PaletteRowStats() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_PaletteRowStats(Const_PaletteRowStats new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_PaletteRowStats(Const_PaletteRowStats arg) {return new(arg);}
            public _ByValue_PaletteRowStats(MR.Misc._Moved<PaletteRowStats> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_PaletteRowStats(MR.Misc._Moved<PaletteRowStats> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `PaletteRowStats` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_PaletteRowStats`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PaletteRowStats`/`Const_PaletteRowStats` directly.
        public class _InOptMut_PaletteRowStats
        {
            public PaletteRowStats? Opt;

            public _InOptMut_PaletteRowStats() {}
            public _InOptMut_PaletteRowStats(PaletteRowStats value) {Opt = value;}
            public static implicit operator _InOptMut_PaletteRowStats(PaletteRowStats value) {return new(value);}
        }

        /// This is used for optional parameters of class `PaletteRowStats` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_PaletteRowStats`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PaletteRowStats`/`Const_PaletteRowStats` to pass it to the function.
        public class _InOptConst_PaletteRowStats
        {
            public Const_PaletteRowStats? Opt;

            public _InOptConst_PaletteRowStats() {}
            public _InOptConst_PaletteRowStats(Const_PaletteRowStats value) {Opt = value;}
            public static implicit operator _InOptConst_PaletteRowStats(Const_PaletteRowStats value) {return new(value);}
        }

        /// Generated from class `MR::Pdf::TextCellParams`.
        /// This is the const half of the class.
        public class Const_TextCellParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_TextCellParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_Destroy", ExactSpelling = true)]
                extern static void __MR_Pdf_TextCellParams_Destroy(_Underlying *_this);
                __MR_Pdf_TextCellParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_TextCellParams() {Dispose(false);}

            public unsafe MR.Pdf.Const_TextParams TextParams
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_Get_textParams", ExactSpelling = true)]
                    extern static MR.Pdf.Const_TextParams._Underlying *__MR_Pdf_TextCellParams_Get_textParams(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_Get_textParams(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Box2f Rect
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_Get_rect", ExactSpelling = true)]
                    extern static MR.Const_Box2f._Underlying *__MR_Pdf_TextCellParams_Get_rect(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_Get_rect(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Color ColorBorder
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_Get_colorBorder", ExactSpelling = true)]
                    extern static MR.Const_Color._Underlying *__MR_Pdf_TextCellParams_Get_colorBorder(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_Get_colorBorder(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Color ColorBackground
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_Get_colorBackground", ExactSpelling = true)]
                    extern static MR.Const_Color._Underlying *__MR_Pdf_TextCellParams_Get_colorBackground(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_Get_colorBackground(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_TextCellParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_TextCellParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::TextCellParams` elementwise.
            public unsafe Const_TextCellParams(MR.Pdf._ByValue_TextParams textParams, MR.Box2f rect, MR.Color colorBorder, MR.Color colorBackground) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_ConstructFrom(MR.Misc._PassBy textParams_pass_by, MR.Pdf.TextParams._Underlying *textParams, MR.Box2f rect, MR.Color colorBorder, MR.Color colorBackground);
                _UnderlyingPtr = __MR_Pdf_TextCellParams_ConstructFrom(textParams.PassByMode, textParams.Value is not null ? textParams.Value._UnderlyingPtr : null, rect, colorBorder, colorBackground);
            }

            /// Generated from constructor `MR::Pdf::TextCellParams::TextCellParams`.
            public unsafe Const_TextCellParams(MR.Pdf._ByValue_TextCellParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.TextCellParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_TextCellParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::Pdf::TextCellParams`.
        /// This is the non-const half of the class.
        public class TextCellParams : Const_TextCellParams
        {
            internal unsafe TextCellParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Pdf.TextParams TextParams
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_GetMutable_textParams", ExactSpelling = true)]
                    extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextCellParams_GetMutable_textParams(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_GetMutable_textParams(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Box2f Rect
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_GetMutable_rect", ExactSpelling = true)]
                    extern static MR.Mut_Box2f._Underlying *__MR_Pdf_TextCellParams_GetMutable_rect(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_GetMutable_rect(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Color ColorBorder
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_GetMutable_colorBorder", ExactSpelling = true)]
                    extern static MR.Mut_Color._Underlying *__MR_Pdf_TextCellParams_GetMutable_colorBorder(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_GetMutable_colorBorder(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_Color ColorBackground
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_GetMutable_colorBackground", ExactSpelling = true)]
                    extern static MR.Mut_Color._Underlying *__MR_Pdf_TextCellParams_GetMutable_colorBackground(_Underlying *_this);
                    return new(__MR_Pdf_TextCellParams_GetMutable_colorBackground(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe TextCellParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_TextCellParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::TextCellParams` elementwise.
            public unsafe TextCellParams(MR.Pdf._ByValue_TextParams textParams, MR.Box2f rect, MR.Color colorBorder, MR.Color colorBackground) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_ConstructFrom(MR.Misc._PassBy textParams_pass_by, MR.Pdf.TextParams._Underlying *textParams, MR.Box2f rect, MR.Color colorBorder, MR.Color colorBackground);
                _UnderlyingPtr = __MR_Pdf_TextCellParams_ConstructFrom(textParams.PassByMode, textParams.Value is not null ? textParams.Value._UnderlyingPtr : null, rect, colorBorder, colorBackground);
            }

            /// Generated from constructor `MR::Pdf::TextCellParams::TextCellParams`.
            public unsafe TextCellParams(MR.Pdf._ByValue_TextCellParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.TextCellParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_TextCellParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Pdf::TextCellParams::operator=`.
            public unsafe MR.Pdf.TextCellParams Assign(MR.Pdf._ByValue_TextCellParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextCellParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.TextCellParams._Underlying *__MR_Pdf_TextCellParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Pdf.TextCellParams._Underlying *_other);
                return new(__MR_Pdf_TextCellParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `TextCellParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `TextCellParams`/`Const_TextCellParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_TextCellParams
        {
            internal readonly Const_TextCellParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_TextCellParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_TextCellParams(Const_TextCellParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_TextCellParams(Const_TextCellParams arg) {return new(arg);}
            public _ByValue_TextCellParams(MR.Misc._Moved<TextCellParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_TextCellParams(MR.Misc._Moved<TextCellParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `TextCellParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_TextCellParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `TextCellParams`/`Const_TextCellParams` directly.
        public class _InOptMut_TextCellParams
        {
            public TextCellParams? Opt;

            public _InOptMut_TextCellParams() {}
            public _InOptMut_TextCellParams(TextCellParams value) {Opt = value;}
            public static implicit operator _InOptMut_TextCellParams(TextCellParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `TextCellParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_TextCellParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `TextCellParams`/`Const_TextCellParams` to pass it to the function.
        public class _InOptConst_TextCellParams
        {
            public Const_TextCellParams? Opt;

            public _InOptConst_TextCellParams() {}
            public _InOptConst_TextCellParams(Const_TextCellParams value) {Opt = value;}
            public static implicit operator _InOptConst_TextCellParams(Const_TextCellParams value) {return new(value);}
        }

        // parameters to drawing text
        /// Generated from class `MR::Pdf::TextParams`.
        /// This is the const half of the class.
        public class Const_TextParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_TextParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_Destroy", ExactSpelling = true)]
                extern static void __MR_Pdf_TextParams_Destroy(_Underlying *_this);
                __MR_Pdf_TextParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_TextParams() {Dispose(false);}

            public unsafe MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath FontName
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_Get_fontName", ExactSpelling = true)]
                    extern static MR.Std.Const_Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_Pdf_TextParams_Get_fontName(_Underlying *_this);
                    return new(__MR_Pdf_TextParams_Get_fontName(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe float FontSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_Get_fontSize", ExactSpelling = true)]
                    extern static float *__MR_Pdf_TextParams_Get_fontSize(_Underlying *_this);
                    return *__MR_Pdf_TextParams_Get_fontSize(_UnderlyingPtr);
                }
            }

            public unsafe MR.Pdf.AlignmentHorizontal Alignment
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_Get_alignment", ExactSpelling = true)]
                    extern static MR.Pdf.AlignmentHorizontal *__MR_Pdf_TextParams_Get_alignment(_Underlying *_this);
                    return *__MR_Pdf_TextParams_Get_alignment(_UnderlyingPtr);
                }
            }

            public unsafe MR.Const_Color ColorText
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_Get_colorText", ExactSpelling = true)]
                    extern static MR.Const_Color._Underlying *__MR_Pdf_TextParams_Get_colorText(_Underlying *_this);
                    return new(__MR_Pdf_TextParams_Get_colorText(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe bool Underline
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_Get_underline", ExactSpelling = true)]
                    extern static bool *__MR_Pdf_TextParams_Get_underline(_Underlying *_this);
                    return *__MR_Pdf_TextParams_Get_underline(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_TextParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_TextParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::TextParams` elementwise.
            public unsafe Const_TextParams(MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath fontName, float fontSize, MR.Pdf.AlignmentHorizontal alignment, MR.Color colorText, bool underline) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_ConstructFrom(MR.Misc._PassBy fontName_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *fontName, float fontSize, MR.Pdf.AlignmentHorizontal alignment, MR.Color colorText, byte underline);
                _UnderlyingPtr = __MR_Pdf_TextParams_ConstructFrom(fontName.PassByMode, fontName.Value is not null ? fontName.Value._UnderlyingPtr : null, fontSize, alignment, colorText, underline ? (byte)1 : (byte)0);
            }

            /// Generated from constructor `MR::Pdf::TextParams::TextParams`.
            public unsafe Const_TextParams(MR.Pdf._ByValue_TextParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.TextParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_TextParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        // parameters to drawing text
        /// Generated from class `MR::Pdf::TextParams`.
        /// This is the non-const half of the class.
        public class TextParams : Const_TextParams
        {
            internal unsafe TextParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath FontName
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_GetMutable_fontName", ExactSpelling = true)]
                    extern static MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *__MR_Pdf_TextParams_GetMutable_fontName(_Underlying *_this);
                    return new(__MR_Pdf_TextParams_GetMutable_fontName(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe ref float FontSize
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_GetMutable_fontSize", ExactSpelling = true)]
                    extern static float *__MR_Pdf_TextParams_GetMutable_fontSize(_Underlying *_this);
                    return ref *__MR_Pdf_TextParams_GetMutable_fontSize(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.Pdf.AlignmentHorizontal Alignment
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_GetMutable_alignment", ExactSpelling = true)]
                    extern static MR.Pdf.AlignmentHorizontal *__MR_Pdf_TextParams_GetMutable_alignment(_Underlying *_this);
                    return ref *__MR_Pdf_TextParams_GetMutable_alignment(_UnderlyingPtr);
                }
            }

            public new unsafe MR.Mut_Color ColorText
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_GetMutable_colorText", ExactSpelling = true)]
                    extern static MR.Mut_Color._Underlying *__MR_Pdf_TextParams_GetMutable_colorText(_Underlying *_this);
                    return new(__MR_Pdf_TextParams_GetMutable_colorText(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe ref bool Underline
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_GetMutable_underline", ExactSpelling = true)]
                    extern static bool *__MR_Pdf_TextParams_GetMutable_underline(_Underlying *_this);
                    return ref *__MR_Pdf_TextParams_GetMutable_underline(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe TextParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_DefaultConstruct();
                _UnderlyingPtr = __MR_Pdf_TextParams_DefaultConstruct();
            }

            /// Constructs `MR::Pdf::TextParams` elementwise.
            public unsafe TextParams(MR.Std._ByValue_Variant_MRPdfBuildinFont_StdFilesystemPath fontName, float fontSize, MR.Pdf.AlignmentHorizontal alignment, MR.Color colorText, bool underline) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_ConstructFrom(MR.Misc._PassBy fontName_pass_by, MR.Std.Variant_MRPdfBuildinFont_StdFilesystemPath._Underlying *fontName, float fontSize, MR.Pdf.AlignmentHorizontal alignment, MR.Color colorText, byte underline);
                _UnderlyingPtr = __MR_Pdf_TextParams_ConstructFrom(fontName.PassByMode, fontName.Value is not null ? fontName.Value._UnderlyingPtr : null, fontSize, alignment, colorText, underline ? (byte)1 : (byte)0);
            }

            /// Generated from constructor `MR::Pdf::TextParams::TextParams`.
            public unsafe TextParams(MR.Pdf._ByValue_TextParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Pdf.TextParams._Underlying *_other);
                _UnderlyingPtr = __MR_Pdf_TextParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Pdf::TextParams::operator=`.
            public unsafe MR.Pdf.TextParams Assign(MR.Pdf._ByValue_TextParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_TextParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Pdf.TextParams._Underlying *__MR_Pdf_TextParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Pdf.TextParams._Underlying *_other);
                return new(__MR_Pdf_TextParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `TextParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `TextParams`/`Const_TextParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_TextParams
        {
            internal readonly Const_TextParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_TextParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_TextParams(Const_TextParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_TextParams(Const_TextParams arg) {return new(arg);}
            public _ByValue_TextParams(MR.Misc._Moved<TextParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_TextParams(MR.Misc._Moved<TextParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `TextParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_TextParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `TextParams`/`Const_TextParams` directly.
        public class _InOptMut_TextParams
        {
            public TextParams? Opt;

            public _InOptMut_TextParams() {}
            public _InOptMut_TextParams(TextParams value) {Opt = value;}
            public static implicit operator _InOptMut_TextParams(TextParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `TextParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_TextParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `TextParams`/`Const_TextParams` to pass it to the function.
        public class _InOptConst_TextParams
        {
            public Const_TextParams? Opt;

            public _InOptConst_TextParams() {}
            public _InOptConst_TextParams(Const_TextParams value) {Opt = value;}
            public static implicit operator _InOptConst_TextParams(Const_TextParams value) {return new(value);}
        }
    }

    /**
    * Class for simple creation pdf.
    */
    /// Generated from class `MR::Pdf`.
    /// This is the non-const half of the class.
    public class Pdf : Const_Pdf
    {
        internal unsafe Pdf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Ctor. Create a document, but not a page. To create a new page use newPage() method
        /// Generated from constructor `MR::Pdf::Pdf`.
        /// Parameter `params_` defaults to `MR::PdfParameters()`.
        public unsafe Pdf(MR.Const_PdfParameters? params_ = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_Construct", ExactSpelling = true)]
            extern static MR.Pdf._Underlying *__MR_Pdf_Construct(MR.Const_PdfParameters._Underlying *params_);
            _UnderlyingPtr = __MR_Pdf_Construct(params_ is not null ? params_._UnderlyingPtr : null);
        }

        /// Ctor. Create a document, but not a page. To create a new page use newPage() method
        /// Generated from constructor `MR::Pdf::Pdf`.
        /// Parameter `params_` defaults to `MR::PdfParameters()`.
        public static unsafe implicit operator Pdf(MR.Const_PdfParameters? params_) {return new(params_);}

        /// Generated from constructor `MR::Pdf::Pdf`.
        public unsafe Pdf(MR._ByValue_Pdf other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Pdf._Underlying *__MR_Pdf_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.Pdf._Underlying *other);
            _UnderlyingPtr = __MR_Pdf_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Pdf::operator=`.
        public unsafe MR.Pdf Assign(MR._ByValue_Pdf other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Pdf._Underlying *__MR_Pdf_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy other_pass_by, MR.Pdf._Underlying *other);
            return new(__MR_Pdf_AssignFromAnother(_UnderlyingPtr, other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /**
        * Add text block in current cursor position.
        * Move cursor.
        * Box horizontal size is page width without offset.
        * Box vertical size is automatically for text.
        * horAlignment = left
        * if isTitle - horAlignment = center, use titleFontSize
        */
        /// Generated from method `MR::Pdf::addText`.
        /// Parameter `isTitle` defaults to `false`.
        public unsafe void AddText(ReadOnlySpan<char> text, bool? isTitle = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addText_bool", ExactSpelling = true)]
            extern static void __MR_Pdf_addText_bool(_Underlying *_this, byte *text, byte *text_end, byte *isTitle);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                byte __deref_isTitle = isTitle.GetValueOrDefault() ? (byte)1 : (byte)0;
                __MR_Pdf_addText_bool(_UnderlyingPtr, __ptr_text, __ptr_text + __len_text, isTitle.HasValue ? &__deref_isTitle : null);
            }
        }

        /// Generated from method `MR::Pdf::addText`.
        public unsafe void AddText(ReadOnlySpan<char> text, MR.Pdf.Const_TextParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addText_MR_Pdf_TextParams", ExactSpelling = true)]
            extern static void __MR_Pdf_addText_MR_Pdf_TextParams(_Underlying *_this, byte *text, byte *text_end, MR.Pdf.Const_TextParams._Underlying *params_);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                __MR_Pdf_addText_MR_Pdf_TextParams(_UnderlyingPtr, __ptr_text, __ptr_text + __len_text, params_._UnderlyingPtr);
            }
        }

        /// return text width
        /// Generated from method `MR::Pdf::getTextWidth`.
        public unsafe float GetTextWidth(ReadOnlySpan<char> text, MR.Pdf.Const_TextParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_getTextWidth", ExactSpelling = true)]
            extern static float __MR_Pdf_getTextWidth(_Underlying *_this, byte *text, byte *text_end, MR.Pdf.Const_TextParams._Underlying *params_);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                return __MR_Pdf_getTextWidth(_UnderlyingPtr, __ptr_text, __ptr_text + __len_text, params_._UnderlyingPtr);
            }
        }

        /**
        * Add set of pair string - value in current cursor position.
        * Move cursor.
        * Box horizontal size is page width without offset.
        * Box vertical size is automatically for text.
        */
        /// Generated from method `MR::Pdf::addTable`.
        public unsafe void AddTable(MR.Std.Const_Vector_StdPairStdStringFloat table)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addTable", ExactSpelling = true)]
            extern static void __MR_Pdf_addTable(_Underlying *_this, MR.Std.Const_Vector_StdPairStdStringFloat._Underlying *table);
            __MR_Pdf_addTable(_UnderlyingPtr, table._UnderlyingPtr);
        }

        /// Generated from method `MR::Pdf::addPaletteStatsTable`.
        public unsafe void AddPaletteStatsTable(MR.Std.Const_Vector_MRPdfPaletteRowStats paletteStats)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addPaletteStatsTable", ExactSpelling = true)]
            extern static void __MR_Pdf_addPaletteStatsTable(_Underlying *_this, MR.Std.Const_Vector_MRPdfPaletteRowStats._Underlying *paletteStats);
            __MR_Pdf_addPaletteStatsTable(_UnderlyingPtr, paletteStats._UnderlyingPtr);
        }

        /**
        * @brief Add image from file in current cursor position.
        * If image bigger than page size, autoscale image to page size.
        * Move cursor.
        */
        /// Generated from method `MR::Pdf::addImageFromFile`.
        public unsafe void AddImageFromFile(ReadOnlySpan<char> imagePath, MR.Pdf.Const_ImageParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addImageFromFile", ExactSpelling = true)]
            extern static void __MR_Pdf_addImageFromFile(_Underlying *_this, byte *imagePath, byte *imagePath_end, MR.Pdf.Const_ImageParams._Underlying *params_);
            byte[] __bytes_imagePath = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(imagePath.Length)];
            int __len_imagePath = System.Text.Encoding.UTF8.GetBytes(imagePath, __bytes_imagePath);
            fixed (byte *__ptr_imagePath = __bytes_imagePath)
            {
                __MR_Pdf_addImageFromFile(_UnderlyingPtr, __ptr_imagePath, __ptr_imagePath + __len_imagePath, params_._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Pdf::addImage`.
        public unsafe void AddImage(MR.Const_Image image, MR.Pdf.Const_ImageParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addImage", ExactSpelling = true)]
            extern static void __MR_Pdf_addImage(_Underlying *_this, MR.Const_Image._Underlying *image, MR.Pdf.Const_ImageParams._Underlying *params_);
            __MR_Pdf_addImage(_UnderlyingPtr, image._UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// Add new pageand move cursor on it
        /// Generated from method `MR::Pdf::newPage`.
        public unsafe void NewPage()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_newPage", ExactSpelling = true)]
            extern static void __MR_Pdf_newPage(_Underlying *_this);
            __MR_Pdf_newPage(_UnderlyingPtr);
        }

        /// set function to customize new page after creation
        /// Generated from method `MR::Pdf::setNewPageAction`.
        public unsafe void SetNewPageAction(MR.Std._ByValue_Function_VoidFuncFromMRPdfRef action)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_setNewPageAction", ExactSpelling = true)]
            extern static void __MR_Pdf_setNewPageAction(_Underlying *_this, MR.Misc._PassBy action_pass_by, MR.Std.Function_VoidFuncFromMRPdfRef._Underlying *action);
            __MR_Pdf_setNewPageAction(_UnderlyingPtr, action.PassByMode, action.Value is not null ? action.Value._UnderlyingPtr : null);
        }

        /// Save document to file
        /// Generated from method `MR::Pdf::saveToFile`.
        public unsafe void SaveToFile(ReadOnlySpan<char> documentPath)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_saveToFile", ExactSpelling = true)]
            extern static void __MR_Pdf_saveToFile(_Underlying *_this, byte *documentPath, byte *documentPath_end);
            byte[] __bytes_documentPath = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(documentPath.Length)];
            int __len_documentPath = System.Text.Encoding.UTF8.GetBytes(documentPath, __bytes_documentPath);
            fixed (byte *__ptr_documentPath = __bytes_documentPath)
            {
                __MR_Pdf_saveToFile(_UnderlyingPtr, __ptr_documentPath, __ptr_documentPath + __len_documentPath);
            }
        }

        /// Generated from method `MR::Pdf::setCursorPosX`.
        public unsafe void SetCursorPosX(float posX)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_setCursorPosX", ExactSpelling = true)]
            extern static void __MR_Pdf_setCursorPosX(_Underlying *_this, float posX);
            __MR_Pdf_setCursorPosX(_UnderlyingPtr, posX);
        }

        /// Generated from method `MR::Pdf::setCursorPosY`.
        public unsafe void SetCursorPosY(float posY)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_setCursorPosY", ExactSpelling = true)]
            extern static void __MR_Pdf_setCursorPosY(_Underlying *_this, float posY);
            __MR_Pdf_setCursorPosY(_UnderlyingPtr, posY);
        }

        // set up new table (clear table customization, reset parameters to default values)
        /// Generated from method `MR::Pdf::newTable`.
        public unsafe void NewTable(int columnCount)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_newTable", ExactSpelling = true)]
            extern static void __MR_Pdf_newTable(_Underlying *_this, int columnCount);
            __MR_Pdf_newTable(_UnderlyingPtr, columnCount);
        }

        // set table column widths
        /// Generated from method `MR::Pdf::setTableColumnWidths`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SetTableColumnWidths(MR.Std.Const_Vector_Float widths)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_setTableColumnWidths", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_Pdf_setTableColumnWidths(_Underlying *_this, MR.Std.Const_Vector_Float._Underlying *widths);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_Pdf_setTableColumnWidths(_UnderlyingPtr, widths._UnderlyingPtr), is_owning: true));
        }

        // add in pdf table row with titles
        /// Generated from method `MR::Pdf::addTableTitles`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> AddTableTitles(MR.Std.Const_Vector_StdString titles)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addTableTitles", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_Pdf_addTableTitles(_Underlying *_this, MR.Std.Const_Vector_StdString._Underlying *titles);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_Pdf_addTableTitles(_UnderlyingPtr, titles._UnderlyingPtr), is_owning: true));
        }

        // set format for conversion values to string for each column
        /// Generated from method `MR::Pdf::setColumnValuesFormat`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SetColumnValuesFormat(MR.Std.Const_Vector_StdString formats)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_setColumnValuesFormat", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_Pdf_setColumnValuesFormat(_Underlying *_this, MR.Std.Const_Vector_StdString._Underlying *formats);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_Pdf_setColumnValuesFormat(_UnderlyingPtr, formats._UnderlyingPtr), is_owning: true));
        }

        // add in pdf table row with values
        /// Generated from method `MR::Pdf::addRow`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> AddRow(MR.Std.Const_Vector_MRPdfCell cells)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_addRow", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_Pdf_addRow(_Underlying *_this, MR.Std.Const_Vector_MRPdfCell._Underlying *cells);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_Pdf_addRow(_UnderlyingPtr, cells._UnderlyingPtr), is_owning: true));
        }

        // parameters to customization table cell
        // return text width (for table font parameters)
        /// Generated from method `MR::Pdf::getTableTextWidth`.
        public unsafe float GetTableTextWidth(ReadOnlySpan<char> text)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_getTableTextWidth", ExactSpelling = true)]
            extern static float __MR_Pdf_getTableTextWidth(_Underlying *_this, byte *text, byte *text_end);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                return __MR_Pdf_getTableTextWidth(_UnderlyingPtr, __ptr_text, __ptr_text + __len_text);
            }
        }

        // add rule to customize table cells
        /// Generated from method `MR::Pdf::setTableCustomRule`.
        public unsafe void SetTableCustomRule(MR.Std._ByValue_Function_MRPdfCellCustomParamsFuncFromIntIntConstStdStringRef rule)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_setTableCustomRule", ExactSpelling = true)]
            extern static void __MR_Pdf_setTableCustomRule(_Underlying *_this, MR.Misc._PassBy rule_pass_by, MR.Std.Function_MRPdfCellCustomParamsFuncFromIntIntConstStdStringRef._Underlying *rule);
            __MR_Pdf_setTableCustomRule(_UnderlyingPtr, rule.PassByMode, rule.Value is not null ? rule.Value._UnderlyingPtr : null);
        }

        // draw text in specific rect on page
        // text will be cropped by rect
        /// Generated from method `MR::Pdf::drawTextInRect`.
        public unsafe void DrawTextInRect(ReadOnlySpan<char> text, MR.Const_Box2f rect, MR.Pdf.Const_TextParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_drawTextInRect", ExactSpelling = true)]
            extern static void __MR_Pdf_drawTextInRect(_Underlying *_this, byte *text, byte *text_end, MR.Const_Box2f._Underlying *rect, MR.Pdf.Const_TextParams._Underlying *params_);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                __MR_Pdf_drawTextInRect(_UnderlyingPtr, __ptr_text, __ptr_text + __len_text, rect._UnderlyingPtr, params_._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Pdf::drawTextCell`.
        public unsafe void DrawTextCell(ReadOnlySpan<char> text, MR.Pdf.Const_TextCellParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Pdf_drawTextCell", ExactSpelling = true)]
            extern static void __MR_Pdf_drawTextCell(_Underlying *_this, byte *text, byte *text_end, MR.Pdf.Const_TextCellParams._Underlying *params_);
            byte[] __bytes_text = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(text.Length)];
            int __len_text = System.Text.Encoding.UTF8.GetBytes(text, __bytes_text);
            fixed (byte *__ptr_text = __bytes_text)
            {
                __MR_Pdf_drawTextCell(_UnderlyingPtr, __ptr_text, __ptr_text + __len_text, params_._UnderlyingPtr);
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `Pdf` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Pdf`/`Const_Pdf` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Pdf
    {
        internal readonly Const_Pdf? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Pdf() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Pdf(MR.Misc._Moved<Pdf> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Pdf(MR.Misc._Moved<Pdf> arg) {return new(arg);}

        /// Ctor. Create a document, but not a page. To create a new page use newPage() method
        /// Generated from constructor `MR::Pdf::Pdf`.
        /// Parameter `params_` defaults to `MR::PdfParameters()`.
        public static unsafe implicit operator _ByValue_Pdf(MR.Const_PdfParameters? params_) {return new MR.Pdf(params_);}
    }

    /// This is used for optional parameters of class `Pdf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Pdf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Pdf`/`Const_Pdf` directly.
    public class _InOptMut_Pdf
    {
        public Pdf? Opt;

        public _InOptMut_Pdf() {}
        public _InOptMut_Pdf(Pdf value) {Opt = value;}
        public static implicit operator _InOptMut_Pdf(Pdf value) {return new(value);}
    }

    /// This is used for optional parameters of class `Pdf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Pdf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Pdf`/`Const_Pdf` to pass it to the function.
    public class _InOptConst_Pdf
    {
        public Const_Pdf? Opt;

        public _InOptConst_Pdf() {}
        public _InOptConst_Pdf(Const_Pdf value) {Opt = value;}
        public static implicit operator _InOptConst_Pdf(Const_Pdf value) {return new(value);}

        /// Ctor. Create a document, but not a page. To create a new page use newPage() method
        /// Generated from constructor `MR::Pdf::Pdf`.
        /// Parameter `params_` defaults to `MR::PdfParameters()`.
        public static unsafe implicit operator _InOptConst_Pdf(MR.Const_PdfParameters? params_) {return new MR.Pdf(params_);}
    }
}
