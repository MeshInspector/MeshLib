public static partial class MR
{
    /// Generated from class `MR::BaseTiffParameters`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::TiffParameters`
    /// This is the const half of the class.
    public class Const_BaseTiffParameters : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_BaseTiffParameters>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BaseTiffParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_BaseTiffParameters_Destroy(_Underlying *_this);
            __MR_BaseTiffParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BaseTiffParameters() {Dispose(false);}

        public unsafe MR.BaseTiffParameters.SampleType SampleType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_Get_sampleType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.SampleType *__MR_BaseTiffParameters_Get_sampleType(_Underlying *_this);
                return *__MR_BaseTiffParameters_Get_sampleType(_UnderlyingPtr);
            }
        }

        public unsafe MR.BaseTiffParameters.ValueType ValueType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_Get_valueType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.ValueType *__MR_BaseTiffParameters_Get_valueType(_Underlying *_this);
                return *__MR_BaseTiffParameters_Get_valueType(_UnderlyingPtr);
            }
        }

        // size of internal data in file
        public unsafe int BytesPerSample
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_Get_bytesPerSample", ExactSpelling = true)]
                extern static int *__MR_BaseTiffParameters_Get_bytesPerSample(_Underlying *_this);
                return *__MR_BaseTiffParameters_Get_bytesPerSample(_UnderlyingPtr);
            }
        }

        // size of image if not layered, otherwise size of layer
        public unsafe MR.Const_Vector2i ImageSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_Get_imageSize", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_BaseTiffParameters_Get_imageSize(_Underlying *_this);
                return new(__MR_BaseTiffParameters_Get_imageSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BaseTiffParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BaseTiffParameters_DefaultConstruct();
        }

        /// Constructs `MR::BaseTiffParameters` elementwise.
        public unsafe Const_BaseTiffParameters(MR.BaseTiffParameters.SampleType sampleType, MR.BaseTiffParameters.ValueType valueType, int bytesPerSample, MR.Vector2i imageSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_ConstructFrom(MR.BaseTiffParameters.SampleType sampleType, MR.BaseTiffParameters.ValueType valueType, int bytesPerSample, MR.Vector2i imageSize);
            _UnderlyingPtr = __MR_BaseTiffParameters_ConstructFrom(sampleType, valueType, bytesPerSample, imageSize);
        }

        /// Generated from constructor `MR::BaseTiffParameters::BaseTiffParameters`.
        public unsafe Const_BaseTiffParameters(MR.Const_BaseTiffParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_ConstructFromAnother(MR.BaseTiffParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BaseTiffParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::BaseTiffParameters::operator==`.
        public static unsafe bool operator==(MR.Const_BaseTiffParameters _this, MR.Const_BaseTiffParameters _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_BaseTiffParameters", ExactSpelling = true)]
            extern static byte __MR_equal_MR_BaseTiffParameters(MR.Const_BaseTiffParameters._Underlying *_this, MR.Const_BaseTiffParameters._Underlying *_1);
            return __MR_equal_MR_BaseTiffParameters(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_BaseTiffParameters _this, MR.Const_BaseTiffParameters _1)
        {
            return !(_this == _1);
        }

        public enum SampleType : int
        {
            Unknown = 0,
            Uint = 1,
            Int = 2,
            Float = 3,
        }

        public enum ValueType : int
        {
            Unknown = 0,
            Scalar = 1,
            RGB = 2,
            RGBA = 3,
        }

        // IEquatable:

        public bool Equals(MR.Const_BaseTiffParameters? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_BaseTiffParameters)
                return this == (MR.Const_BaseTiffParameters)other;
            return false;
        }
    }

    /// Generated from class `MR::BaseTiffParameters`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::TiffParameters`
    /// This is the non-const half of the class.
    public class BaseTiffParameters : Const_BaseTiffParameters
    {
        internal unsafe BaseTiffParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.BaseTiffParameters.SampleType SampleType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_GetMutable_sampleType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.SampleType *__MR_BaseTiffParameters_GetMutable_sampleType(_Underlying *_this);
                return ref *__MR_BaseTiffParameters_GetMutable_sampleType(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.BaseTiffParameters.ValueType ValueType_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_GetMutable_valueType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.ValueType *__MR_BaseTiffParameters_GetMutable_valueType(_Underlying *_this);
                return ref *__MR_BaseTiffParameters_GetMutable_valueType(_UnderlyingPtr);
            }
        }

        // size of internal data in file
        public new unsafe ref int BytesPerSample
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_GetMutable_bytesPerSample", ExactSpelling = true)]
                extern static int *__MR_BaseTiffParameters_GetMutable_bytesPerSample(_Underlying *_this);
                return ref *__MR_BaseTiffParameters_GetMutable_bytesPerSample(_UnderlyingPtr);
            }
        }

        // size of image if not layered, otherwise size of layer
        public new unsafe MR.Mut_Vector2i ImageSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_GetMutable_imageSize", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_BaseTiffParameters_GetMutable_imageSize(_Underlying *_this);
                return new(__MR_BaseTiffParameters_GetMutable_imageSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BaseTiffParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BaseTiffParameters_DefaultConstruct();
        }

        /// Constructs `MR::BaseTiffParameters` elementwise.
        public unsafe BaseTiffParameters(MR.BaseTiffParameters.SampleType sampleType, MR.BaseTiffParameters.ValueType valueType, int bytesPerSample, MR.Vector2i imageSize) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_ConstructFrom(MR.BaseTiffParameters.SampleType sampleType, MR.BaseTiffParameters.ValueType valueType, int bytesPerSample, MR.Vector2i imageSize);
            _UnderlyingPtr = __MR_BaseTiffParameters_ConstructFrom(sampleType, valueType, bytesPerSample, imageSize);
        }

        /// Generated from constructor `MR::BaseTiffParameters::BaseTiffParameters`.
        public unsafe BaseTiffParameters(MR.Const_BaseTiffParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_ConstructFromAnother(MR.BaseTiffParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BaseTiffParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::BaseTiffParameters::operator=`.
        public unsafe MR.BaseTiffParameters Assign(MR.Const_BaseTiffParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseTiffParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_BaseTiffParameters_AssignFromAnother(_Underlying *_this, MR.BaseTiffParameters._Underlying *_other);
            return new(__MR_BaseTiffParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `BaseTiffParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BaseTiffParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BaseTiffParameters`/`Const_BaseTiffParameters` directly.
    public class _InOptMut_BaseTiffParameters
    {
        public BaseTiffParameters? Opt;

        public _InOptMut_BaseTiffParameters() {}
        public _InOptMut_BaseTiffParameters(BaseTiffParameters value) {Opt = value;}
        public static implicit operator _InOptMut_BaseTiffParameters(BaseTiffParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `BaseTiffParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BaseTiffParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BaseTiffParameters`/`Const_BaseTiffParameters` to pass it to the function.
    public class _InOptConst_BaseTiffParameters
    {
        public Const_BaseTiffParameters? Opt;

        public _InOptConst_BaseTiffParameters() {}
        public _InOptConst_BaseTiffParameters(Const_BaseTiffParameters value) {Opt = value;}
        public static implicit operator _InOptConst_BaseTiffParameters(Const_BaseTiffParameters value) {return new(value);}
    }

    /// Generated from class `MR::TiffParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseTiffParameters`
    /// This is the const half of the class.
    public class Const_TiffParameters : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_TiffParameters>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TiffParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_TiffParameters_Destroy(_Underlying *_this);
            __MR_TiffParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TiffParameters() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseTiffParameters(Const_TiffParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_UpcastTo_MR_BaseTiffParameters", ExactSpelling = true)]
            extern static MR.Const_BaseTiffParameters._Underlying *__MR_TiffParameters_UpcastTo_MR_BaseTiffParameters(_Underlying *_this);
            MR.Const_BaseTiffParameters ret = new(__MR_TiffParameters_UpcastTo_MR_BaseTiffParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // true if tif file is tiled
        public unsafe bool Tiled
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_tiled", ExactSpelling = true)]
                extern static bool *__MR_TiffParameters_Get_tiled(_Underlying *_this);
                return *__MR_TiffParameters_Get_tiled(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Vector2i TileSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_tileSize", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_TiffParameters_Get_tileSize(_Underlying *_this);
                return new(__MR_TiffParameters_Get_tileSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe int Layers
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_layers", ExactSpelling = true)]
                extern static int *__MR_TiffParameters_Get_layers(_Underlying *_this);
                return *__MR_TiffParameters_Get_layers(_UnderlyingPtr);
            }
        }

        // tile depth (if several layers)
        public unsafe int Depth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_depth", ExactSpelling = true)]
                extern static int *__MR_TiffParameters_Get_depth(_Underlying *_this);
                return *__MR_TiffParameters_Get_depth(_UnderlyingPtr);
            }
        }

        public unsafe MR.BaseTiffParameters.SampleType SampleType
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_sampleType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.SampleType *__MR_TiffParameters_Get_sampleType(_Underlying *_this);
                return *__MR_TiffParameters_Get_sampleType(_UnderlyingPtr);
            }
        }

        public unsafe MR.BaseTiffParameters.ValueType ValueType
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_valueType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.ValueType *__MR_TiffParameters_Get_valueType(_Underlying *_this);
                return *__MR_TiffParameters_Get_valueType(_UnderlyingPtr);
            }
        }

        // size of internal data in file
        public unsafe int BytesPerSample
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_bytesPerSample", ExactSpelling = true)]
                extern static int *__MR_TiffParameters_Get_bytesPerSample(_Underlying *_this);
                return *__MR_TiffParameters_Get_bytesPerSample(_UnderlyingPtr);
            }
        }

        // size of image if not layered, otherwise size of layer
        public unsafe MR.Const_Vector2i ImageSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_Get_imageSize", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_TiffParameters_Get_imageSize(_Underlying *_this);
                return new(__MR_TiffParameters_Get_imageSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TiffParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TiffParameters._Underlying *__MR_TiffParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_TiffParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::TiffParameters::TiffParameters`.
        public unsafe Const_TiffParameters(MR.Const_TiffParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TiffParameters._Underlying *__MR_TiffParameters_ConstructFromAnother(MR.TiffParameters._Underlying *_other);
            _UnderlyingPtr = __MR_TiffParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TiffParameters::operator==`.
        public static unsafe bool operator==(MR.Const_TiffParameters _this, MR.Const_TiffParameters _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_TiffParameters", ExactSpelling = true)]
            extern static byte __MR_equal_MR_TiffParameters(MR.Const_TiffParameters._Underlying *_this, MR.Const_TiffParameters._Underlying *_1);
            return __MR_equal_MR_TiffParameters(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_TiffParameters _this, MR.Const_TiffParameters _1)
        {
            return !(_this == _1);
        }

        // IEquatable:

        public bool Equals(MR.Const_TiffParameters? _1)
        {
            if (_1 is null)
                return false;
            return this == _1;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_TiffParameters)
                return this == (MR.Const_TiffParameters)other;
            return false;
        }
    }

    /// Generated from class `MR::TiffParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseTiffParameters`
    /// This is the non-const half of the class.
    public class TiffParameters : Const_TiffParameters
    {
        internal unsafe TiffParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseTiffParameters(TiffParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_UpcastTo_MR_BaseTiffParameters", ExactSpelling = true)]
            extern static MR.BaseTiffParameters._Underlying *__MR_TiffParameters_UpcastTo_MR_BaseTiffParameters(_Underlying *_this);
            MR.BaseTiffParameters ret = new(__MR_TiffParameters_UpcastTo_MR_BaseTiffParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // true if tif file is tiled
        public new unsafe ref bool Tiled
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_tiled", ExactSpelling = true)]
                extern static bool *__MR_TiffParameters_GetMutable_tiled(_Underlying *_this);
                return ref *__MR_TiffParameters_GetMutable_tiled(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Mut_Vector2i TileSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_tileSize", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_TiffParameters_GetMutable_tileSize(_Underlying *_this);
                return new(__MR_TiffParameters_GetMutable_tileSize(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref int Layers
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_layers", ExactSpelling = true)]
                extern static int *__MR_TiffParameters_GetMutable_layers(_Underlying *_this);
                return ref *__MR_TiffParameters_GetMutable_layers(_UnderlyingPtr);
            }
        }

        // tile depth (if several layers)
        public new unsafe ref int Depth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_depth", ExactSpelling = true)]
                extern static int *__MR_TiffParameters_GetMutable_depth(_Underlying *_this);
                return ref *__MR_TiffParameters_GetMutable_depth(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.BaseTiffParameters.SampleType SampleType
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_sampleType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.SampleType *__MR_TiffParameters_GetMutable_sampleType(_Underlying *_this);
                return ref *__MR_TiffParameters_GetMutable_sampleType(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.BaseTiffParameters.ValueType ValueType
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_valueType", ExactSpelling = true)]
                extern static MR.BaseTiffParameters.ValueType *__MR_TiffParameters_GetMutable_valueType(_Underlying *_this);
                return ref *__MR_TiffParameters_GetMutable_valueType(_UnderlyingPtr);
            }
        }

        // size of internal data in file
        public new unsafe ref int BytesPerSample
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_bytesPerSample", ExactSpelling = true)]
                extern static int *__MR_TiffParameters_GetMutable_bytesPerSample(_Underlying *_this);
                return ref *__MR_TiffParameters_GetMutable_bytesPerSample(_UnderlyingPtr);
            }
        }

        // size of image if not layered, otherwise size of layer
        public new unsafe MR.Mut_Vector2i ImageSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_GetMutable_imageSize", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_TiffParameters_GetMutable_imageSize(_Underlying *_this);
                return new(__MR_TiffParameters_GetMutable_imageSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TiffParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TiffParameters._Underlying *__MR_TiffParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_TiffParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::TiffParameters::TiffParameters`.
        public unsafe TiffParameters(MR.Const_TiffParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TiffParameters._Underlying *__MR_TiffParameters_ConstructFromAnother(MR.TiffParameters._Underlying *_other);
            _UnderlyingPtr = __MR_TiffParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TiffParameters::operator=`.
        public unsafe MR.TiffParameters Assign(MR.Const_TiffParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TiffParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TiffParameters._Underlying *__MR_TiffParameters_AssignFromAnother(_Underlying *_this, MR.TiffParameters._Underlying *_other);
            return new(__MR_TiffParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TiffParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TiffParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TiffParameters`/`Const_TiffParameters` directly.
    public class _InOptMut_TiffParameters
    {
        public TiffParameters? Opt;

        public _InOptMut_TiffParameters() {}
        public _InOptMut_TiffParameters(TiffParameters value) {Opt = value;}
        public static implicit operator _InOptMut_TiffParameters(TiffParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `TiffParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TiffParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TiffParameters`/`Const_TiffParameters` to pass it to the function.
    public class _InOptConst_TiffParameters
    {
        public Const_TiffParameters? Opt;

        public _InOptConst_TiffParameters() {}
        public _InOptConst_TiffParameters(Const_TiffParameters value) {Opt = value;}
        public static implicit operator _InOptConst_TiffParameters(Const_TiffParameters value) {return new(value);}
    }

    /// Generated from class `MR::RawTiffOutput`.
    /// This is the const half of the class.
    public class Const_RawTiffOutput : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RawTiffOutput(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Destroy", ExactSpelling = true)]
            extern static void __MR_RawTiffOutput_Destroy(_Underlying *_this);
            __MR_RawTiffOutput_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RawTiffOutput() {Dispose(false);}

        // main output data, should be allocated
        public unsafe ref byte * Bytes
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_bytes", ExactSpelling = true)]
                extern static byte **__MR_RawTiffOutput_Get_bytes(_Underlying *_this);
                return ref *__MR_RawTiffOutput_Get_bytes(_UnderlyingPtr);
            }
        }

        // allocated data size
        public unsafe ulong Size
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_size", ExactSpelling = true)]
                extern static ulong *__MR_RawTiffOutput_Get_size(_Underlying *_this);
                return *__MR_RawTiffOutput_Get_size(_UnderlyingPtr);
            }
        }

        // optional params output
        public unsafe ref void * Params
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_params", ExactSpelling = true)]
                extern static void **__MR_RawTiffOutput_Get_params(_Underlying *_this);
                return ref *__MR_RawTiffOutput_Get_params(_UnderlyingPtr);
            }
        }

        // optional pixel to world transform
        public unsafe ref MR.AffineXf3f * P2wXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_p2wXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_RawTiffOutput_Get_p2wXf(_Underlying *_this);
                return ref *__MR_RawTiffOutput_Get_p2wXf(_UnderlyingPtr);
            }
        }

        // input if true loads tiff file as floats array
        public unsafe bool ConvertToFloat
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_convertToFloat", ExactSpelling = true)]
                extern static bool *__MR_RawTiffOutput_Get_convertToFloat(_Underlying *_this);
                return *__MR_RawTiffOutput_Get_convertToFloat(_UnderlyingPtr);
            }
        }

        // min max
        public unsafe ref float * Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_min", ExactSpelling = true)]
                extern static float **__MR_RawTiffOutput_Get_min(_Underlying *_this);
                return ref *__MR_RawTiffOutput_Get_min(_UnderlyingPtr);
            }
        }

        public unsafe ref float * Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_Get_max", ExactSpelling = true)]
                extern static float **__MR_RawTiffOutput_Get_max(_Underlying *_this);
                return ref *__MR_RawTiffOutput_Get_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RawTiffOutput() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_DefaultConstruct();
            _UnderlyingPtr = __MR_RawTiffOutput_DefaultConstruct();
        }

        /// Constructs `MR::RawTiffOutput` elementwise.
        public unsafe Const_RawTiffOutput(MR.Misc.InOut<byte>? bytes, ulong size, MR.TiffParameters? params_, MR.Mut_AffineXf3f? p2wXf, bool convertToFloat, MR.Misc.InOut<float>? min, MR.Misc.InOut<float>? max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_ConstructFrom", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_ConstructFrom(byte *bytes, ulong size, MR.TiffParameters._Underlying *params_, MR.Mut_AffineXf3f._Underlying *p2wXf, byte convertToFloat, float *min, float *max);
            byte __value_bytes = bytes is not null ? bytes.Value : default(byte);
            float __value_min = min is not null ? min.Value : default(float);
            float __value_max = max is not null ? max.Value : default(float);
            _UnderlyingPtr = __MR_RawTiffOutput_ConstructFrom(bytes is not null ? &__value_bytes : null, size, params_ is not null ? params_._UnderlyingPtr : null, p2wXf is not null ? p2wXf._UnderlyingPtr : null, convertToFloat ? (byte)1 : (byte)0, min is not null ? &__value_min : null, max is not null ? &__value_max : null);
            if (max is not null) max.Value = __value_max;
            if (min is not null) min.Value = __value_min;
            if (bytes is not null) bytes.Value = __value_bytes;
        }

        /// Generated from constructor `MR::RawTiffOutput::RawTiffOutput`.
        public unsafe Const_RawTiffOutput(MR.Const_RawTiffOutput _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_ConstructFromAnother(MR.RawTiffOutput._Underlying *_other);
            _UnderlyingPtr = __MR_RawTiffOutput_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::RawTiffOutput`.
    /// This is the non-const half of the class.
    public class RawTiffOutput : Const_RawTiffOutput
    {
        internal unsafe RawTiffOutput(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // main output data, should be allocated
        public new unsafe ref byte * Bytes
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_bytes", ExactSpelling = true)]
                extern static byte **__MR_RawTiffOutput_GetMutable_bytes(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_bytes(_UnderlyingPtr);
            }
        }

        // allocated data size
        public new unsafe ref ulong Size
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_size", ExactSpelling = true)]
                extern static ulong *__MR_RawTiffOutput_GetMutable_size(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_size(_UnderlyingPtr);
            }
        }

        // optional params output
        public new unsafe ref void * Params
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_params", ExactSpelling = true)]
                extern static void **__MR_RawTiffOutput_GetMutable_params(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_params(_UnderlyingPtr);
            }
        }

        // optional pixel to world transform
        public new unsafe ref MR.AffineXf3f * P2wXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_p2wXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_RawTiffOutput_GetMutable_p2wXf(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_p2wXf(_UnderlyingPtr);
            }
        }

        // input if true loads tiff file as floats array
        public new unsafe ref bool ConvertToFloat
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_convertToFloat", ExactSpelling = true)]
                extern static bool *__MR_RawTiffOutput_GetMutable_convertToFloat(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_convertToFloat(_UnderlyingPtr);
            }
        }

        // min max
        public new unsafe ref float * Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_min", ExactSpelling = true)]
                extern static float **__MR_RawTiffOutput_GetMutable_min(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_min(_UnderlyingPtr);
            }
        }

        public new unsafe ref float * Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_GetMutable_max", ExactSpelling = true)]
                extern static float **__MR_RawTiffOutput_GetMutable_max(_Underlying *_this);
                return ref *__MR_RawTiffOutput_GetMutable_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RawTiffOutput() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_DefaultConstruct();
            _UnderlyingPtr = __MR_RawTiffOutput_DefaultConstruct();
        }

        /// Constructs `MR::RawTiffOutput` elementwise.
        public unsafe RawTiffOutput(MR.Misc.InOut<byte>? bytes, ulong size, MR.TiffParameters? params_, MR.Mut_AffineXf3f? p2wXf, bool convertToFloat, MR.Misc.InOut<float>? min, MR.Misc.InOut<float>? max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_ConstructFrom", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_ConstructFrom(byte *bytes, ulong size, MR.TiffParameters._Underlying *params_, MR.Mut_AffineXf3f._Underlying *p2wXf, byte convertToFloat, float *min, float *max);
            byte __value_bytes = bytes is not null ? bytes.Value : default(byte);
            float __value_min = min is not null ? min.Value : default(float);
            float __value_max = max is not null ? max.Value : default(float);
            _UnderlyingPtr = __MR_RawTiffOutput_ConstructFrom(bytes is not null ? &__value_bytes : null, size, params_ is not null ? params_._UnderlyingPtr : null, p2wXf is not null ? p2wXf._UnderlyingPtr : null, convertToFloat ? (byte)1 : (byte)0, min is not null ? &__value_min : null, max is not null ? &__value_max : null);
            if (max is not null) max.Value = __value_max;
            if (min is not null) min.Value = __value_min;
            if (bytes is not null) bytes.Value = __value_bytes;
        }

        /// Generated from constructor `MR::RawTiffOutput::RawTiffOutput`.
        public unsafe RawTiffOutput(MR.Const_RawTiffOutput _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_ConstructFromAnother(MR.RawTiffOutput._Underlying *_other);
            _UnderlyingPtr = __MR_RawTiffOutput_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::RawTiffOutput::operator=`.
        public unsafe MR.RawTiffOutput Assign(MR.Const_RawTiffOutput _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RawTiffOutput_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RawTiffOutput._Underlying *__MR_RawTiffOutput_AssignFromAnother(_Underlying *_this, MR.RawTiffOutput._Underlying *_other);
            return new(__MR_RawTiffOutput_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RawTiffOutput` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RawTiffOutput`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RawTiffOutput`/`Const_RawTiffOutput` directly.
    public class _InOptMut_RawTiffOutput
    {
        public RawTiffOutput? Opt;

        public _InOptMut_RawTiffOutput() {}
        public _InOptMut_RawTiffOutput(RawTiffOutput value) {Opt = value;}
        public static implicit operator _InOptMut_RawTiffOutput(RawTiffOutput value) {return new(value);}
    }

    /// This is used for optional parameters of class `RawTiffOutput` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RawTiffOutput`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RawTiffOutput`/`Const_RawTiffOutput` to pass it to the function.
    public class _InOptConst_RawTiffOutput
    {
        public Const_RawTiffOutput? Opt;

        public _InOptConst_RawTiffOutput() {}
        public _InOptConst_RawTiffOutput(Const_RawTiffOutput value) {Opt = value;}
        public static implicit operator _InOptConst_RawTiffOutput(Const_RawTiffOutput value) {return new(value);}
    }

    /// Generated from class `MR::WriteRawTiffParams`.
    /// This is the const half of the class.
    public class Const_WriteRawTiffParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_WriteRawTiffParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_Destroy", ExactSpelling = true)]
            extern static void __MR_WriteRawTiffParams_Destroy(_Underlying *_this);
            __MR_WriteRawTiffParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_WriteRawTiffParams() {Dispose(false);}

        public unsafe MR.Const_BaseTiffParameters BaseParams
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_Get_baseParams", ExactSpelling = true)]
                extern static MR.Const_BaseTiffParameters._Underlying *__MR_WriteRawTiffParams_Get_baseParams(_Underlying *_this);
                return new(__MR_WriteRawTiffParams_Get_baseParams(_UnderlyingPtr), is_owning: false);
            }
        }

        // optional transformation data written to GeoTIFF's ModelTransformationTag
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_WriteRawTiffParams_Get_xf(_Underlying *_this);
                return ref *__MR_WriteRawTiffParams_Get_xf(_UnderlyingPtr);
            }
        }

        // optional NoData value written to GDAL_NODATA
        public unsafe MR.Std.Const_String NoData
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_Get_noData", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_WriteRawTiffParams_Get_noData(_Underlying *_this);
                return new(__MR_WriteRawTiffParams_Get_noData(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_WriteRawTiffParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_DefaultConstruct();
            _UnderlyingPtr = __MR_WriteRawTiffParams_DefaultConstruct();
        }

        /// Constructs `MR::WriteRawTiffParams` elementwise.
        public unsafe Const_WriteRawTiffParams(MR.Const_BaseTiffParameters baseParams, MR.Const_AffineXf3f? xf, ReadOnlySpan<char> noData) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_ConstructFrom(MR.BaseTiffParameters._Underlying *baseParams, MR.Const_AffineXf3f._Underlying *xf, byte *noData, byte *noData_end);
            byte[] __bytes_noData = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(noData.Length)];
            int __len_noData = System.Text.Encoding.UTF8.GetBytes(noData, __bytes_noData);
            fixed (byte *__ptr_noData = __bytes_noData)
            {
                _UnderlyingPtr = __MR_WriteRawTiffParams_ConstructFrom(baseParams._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null, __ptr_noData, __ptr_noData + __len_noData);
            }
        }

        /// Generated from constructor `MR::WriteRawTiffParams::WriteRawTiffParams`.
        public unsafe Const_WriteRawTiffParams(MR._ByValue_WriteRawTiffParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WriteRawTiffParams._Underlying *_other);
            _UnderlyingPtr = __MR_WriteRawTiffParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::WriteRawTiffParams`.
    /// This is the non-const half of the class.
    public class WriteRawTiffParams : Const_WriteRawTiffParams
    {
        internal unsafe WriteRawTiffParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.BaseTiffParameters BaseParams
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_GetMutable_baseParams", ExactSpelling = true)]
                extern static MR.BaseTiffParameters._Underlying *__MR_WriteRawTiffParams_GetMutable_baseParams(_Underlying *_this);
                return new(__MR_WriteRawTiffParams_GetMutable_baseParams(_UnderlyingPtr), is_owning: false);
            }
        }

        // optional transformation data written to GeoTIFF's ModelTransformationTag
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_WriteRawTiffParams_GetMutable_xf(_Underlying *_this);
                return ref *__MR_WriteRawTiffParams_GetMutable_xf(_UnderlyingPtr);
            }
        }

        // optional NoData value written to GDAL_NODATA
        public new unsafe MR.Std.String NoData
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_GetMutable_noData", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_WriteRawTiffParams_GetMutable_noData(_Underlying *_this);
                return new(__MR_WriteRawTiffParams_GetMutable_noData(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe WriteRawTiffParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_DefaultConstruct();
            _UnderlyingPtr = __MR_WriteRawTiffParams_DefaultConstruct();
        }

        /// Constructs `MR::WriteRawTiffParams` elementwise.
        public unsafe WriteRawTiffParams(MR.Const_BaseTiffParameters baseParams, MR.Const_AffineXf3f? xf, ReadOnlySpan<char> noData) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_ConstructFrom(MR.BaseTiffParameters._Underlying *baseParams, MR.Const_AffineXf3f._Underlying *xf, byte *noData, byte *noData_end);
            byte[] __bytes_noData = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(noData.Length)];
            int __len_noData = System.Text.Encoding.UTF8.GetBytes(noData, __bytes_noData);
            fixed (byte *__ptr_noData = __bytes_noData)
            {
                _UnderlyingPtr = __MR_WriteRawTiffParams_ConstructFrom(baseParams._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null, __ptr_noData, __ptr_noData + __len_noData);
            }
        }

        /// Generated from constructor `MR::WriteRawTiffParams::WriteRawTiffParams`.
        public unsafe WriteRawTiffParams(MR._ByValue_WriteRawTiffParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WriteRawTiffParams._Underlying *_other);
            _UnderlyingPtr = __MR_WriteRawTiffParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::WriteRawTiffParams::operator=`.
        public unsafe MR.WriteRawTiffParams Assign(MR._ByValue_WriteRawTiffParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WriteRawTiffParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.WriteRawTiffParams._Underlying *__MR_WriteRawTiffParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.WriteRawTiffParams._Underlying *_other);
            return new(__MR_WriteRawTiffParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `WriteRawTiffParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `WriteRawTiffParams`/`Const_WriteRawTiffParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_WriteRawTiffParams
    {
        internal readonly Const_WriteRawTiffParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_WriteRawTiffParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_WriteRawTiffParams(Const_WriteRawTiffParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_WriteRawTiffParams(Const_WriteRawTiffParams arg) {return new(arg);}
        public _ByValue_WriteRawTiffParams(MR.Misc._Moved<WriteRawTiffParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_WriteRawTiffParams(MR.Misc._Moved<WriteRawTiffParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `WriteRawTiffParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_WriteRawTiffParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WriteRawTiffParams`/`Const_WriteRawTiffParams` directly.
    public class _InOptMut_WriteRawTiffParams
    {
        public WriteRawTiffParams? Opt;

        public _InOptMut_WriteRawTiffParams() {}
        public _InOptMut_WriteRawTiffParams(WriteRawTiffParams value) {Opt = value;}
        public static implicit operator _InOptMut_WriteRawTiffParams(WriteRawTiffParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `WriteRawTiffParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_WriteRawTiffParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WriteRawTiffParams`/`Const_WriteRawTiffParams` to pass it to the function.
    public class _InOptConst_WriteRawTiffParams
    {
        public Const_WriteRawTiffParams? Opt;

        public _InOptConst_WriteRawTiffParams() {}
        public _InOptConst_WriteRawTiffParams(Const_WriteRawTiffParams value) {Opt = value;}
        public static implicit operator _InOptConst_WriteRawTiffParams(Const_WriteRawTiffParams value) {return new(value);}
    }

    // returns true if given file is tiff
    /// Generated from function `MR::isTIFFFile`.
    public static unsafe bool IsTIFFFile(ReadOnlySpan<char> path)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isTIFFFile", ExactSpelling = true)]
        extern static byte __MR_isTIFFFile(byte *path, byte *path_end);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return __MR_isTIFFFile(__ptr_path, __ptr_path + __len_path) != 0;
        }
    }

    // reads parameters of tiff file
    /// Generated from function `MR::readTiffParameters`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRTiffParameters_StdString> ReadTiffParameters(ReadOnlySpan<char> path)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_readTiffParameters", ExactSpelling = true)]
        extern static MR.Expected_MRTiffParameters_StdString._Underlying *__MR_readTiffParameters(byte *path, byte *path_end);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_MRTiffParameters_StdString(__MR_readTiffParameters(__ptr_path, __ptr_path + __len_path), is_owning: true));
        }
    }

    // load values from tiff to ouput.data
    /// Generated from function `MR::readRawTiff`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ReadRawTiff(ReadOnlySpan<char> path, MR.RawTiffOutput output)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_readRawTiff", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_readRawTiff(byte *path, byte *path_end, MR.RawTiffOutput._Underlying *output);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_readRawTiff(__ptr_path, __ptr_path + __len_path, output._UnderlyingPtr), is_owning: true));
        }
    }

    // writes bytes to tiff file
    /// Generated from function `MR::writeRawTiff`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> WriteRawTiff(byte? bytes, ReadOnlySpan<char> path, MR.Const_WriteRawTiffParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_writeRawTiff_3", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_writeRawTiff_3(byte *bytes, byte *path, byte *path_end, MR.Const_WriteRawTiffParams._Underlying *params_);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            byte __deref_bytes = bytes.GetValueOrDefault();
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_writeRawTiff_3(bytes.HasValue ? &__deref_bytes : null, __ptr_path, __ptr_path + __len_path, params_._UnderlyingPtr), is_owning: true));
        }
    }

    /// Generated from function `MR::writeRawTiff`.
    [Obsolete("use WriteRawTiffParams version instead")]
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> WriteRawTiff(byte? bytes, ReadOnlySpan<char> path, MR.Const_BaseTiffParameters params_, MR.Const_AffineXf3f? xf)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_writeRawTiff_4", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_writeRawTiff_4(byte *bytes, byte *path, byte *path_end, MR.Const_BaseTiffParameters._Underlying *params_, MR.Const_AffineXf3f._Underlying *xf);
        byte[] __bytes_path = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(path.Length)];
        int __len_path = System.Text.Encoding.UTF8.GetBytes(path, __bytes_path);
        fixed (byte *__ptr_path = __bytes_path)
        {
            byte __deref_bytes = bytes.GetValueOrDefault();
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_writeRawTiff_4(bytes.HasValue ? &__deref_bytes : null, __ptr_path, __ptr_path + __len_path, params_._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null), is_owning: true));
        }
    }
}
