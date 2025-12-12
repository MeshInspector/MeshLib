public static partial class MR
{
    /// Generated from class `MR::BaseShellParameters`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::OffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::GeneralOffsetParameters`
    ///     `MR::SharpOffsetParameters`
    /// This is the const half of the class.
    public class Const_BaseShellParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BaseShellParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_BaseShellParameters_Destroy(_Underlying *_this);
            __MR_BaseShellParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BaseShellParameters() {Dispose(false);}

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_BaseShellParameters_Get_voxelSize(_Underlying *_this);
                return *__MR_BaseShellParameters_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_Get_callBack", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_BaseShellParameters_Get_callBack(_Underlying *_this);
                return new(__MR_BaseShellParameters_Get_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BaseShellParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BaseShellParameters_DefaultConstruct();
        }

        /// Constructs `MR::BaseShellParameters` elementwise.
        public unsafe Const_BaseShellParameters(float voxelSize, MR.Std._ByValue_Function_BoolFuncFromFloat callBack) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_ConstructFrom(float voxelSize, MR.Misc._PassBy callBack_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callBack);
            _UnderlyingPtr = __MR_BaseShellParameters_ConstructFrom(voxelSize, callBack.PassByMode, callBack.Value is not null ? callBack.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BaseShellParameters::BaseShellParameters`.
        public unsafe Const_BaseShellParameters(MR._ByValue_BaseShellParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BaseShellParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BaseShellParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::BaseShellParameters`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::OffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::GeneralOffsetParameters`
    ///     `MR::SharpOffsetParameters`
    /// This is the non-const half of the class.
    public class BaseShellParameters : Const_BaseShellParameters
    {
        internal unsafe BaseShellParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_BaseShellParameters_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_BaseShellParameters_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_GetMutable_callBack", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_BaseShellParameters_GetMutable_callBack(_Underlying *_this);
                return new(__MR_BaseShellParameters_GetMutable_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe BaseShellParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_BaseShellParameters_DefaultConstruct();
        }

        /// Constructs `MR::BaseShellParameters` elementwise.
        public unsafe BaseShellParameters(float voxelSize, MR.Std._ByValue_Function_BoolFuncFromFloat callBack) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_ConstructFrom(float voxelSize, MR.Misc._PassBy callBack_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callBack);
            _UnderlyingPtr = __MR_BaseShellParameters_ConstructFrom(voxelSize, callBack.PassByMode, callBack.Value is not null ? callBack.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::BaseShellParameters::BaseShellParameters`.
        public unsafe BaseShellParameters(MR._ByValue_BaseShellParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.BaseShellParameters._Underlying *_other);
            _UnderlyingPtr = __MR_BaseShellParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::BaseShellParameters::operator=`.
        public unsafe MR.BaseShellParameters Assign(MR._ByValue_BaseShellParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BaseShellParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_BaseShellParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.BaseShellParameters._Underlying *_other);
            return new(__MR_BaseShellParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `BaseShellParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `BaseShellParameters`/`Const_BaseShellParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_BaseShellParameters
    {
        internal readonly Const_BaseShellParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_BaseShellParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_BaseShellParameters(Const_BaseShellParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_BaseShellParameters(Const_BaseShellParameters arg) {return new(arg);}
        public _ByValue_BaseShellParameters(MR.Misc._Moved<BaseShellParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_BaseShellParameters(MR.Misc._Moved<BaseShellParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `BaseShellParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BaseShellParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BaseShellParameters`/`Const_BaseShellParameters` directly.
    public class _InOptMut_BaseShellParameters
    {
        public BaseShellParameters? Opt;

        public _InOptMut_BaseShellParameters() {}
        public _InOptMut_BaseShellParameters(BaseShellParameters value) {Opt = value;}
        public static implicit operator _InOptMut_BaseShellParameters(BaseShellParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `BaseShellParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BaseShellParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BaseShellParameters`/`Const_BaseShellParameters` to pass it to the function.
    public class _InOptConst_BaseShellParameters
    {
        public Const_BaseShellParameters? Opt;

        public _InOptConst_BaseShellParameters() {}
        public _InOptConst_BaseShellParameters(Const_BaseShellParameters value) {Opt = value;}
        public static implicit operator _InOptConst_BaseShellParameters(Const_BaseShellParameters value) {return new(value);}
    }

    /// Generated from class `MR::OffsetParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseShellParameters`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SharpOffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::GeneralOffsetParameters`
    /// This is the const half of the class.
    public class Const_OffsetParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OffsetParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_OffsetParameters_Destroy(_Underlying *_this);
            __MR_OffsetParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OffsetParameters() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseShellParameters(Const_OffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_UpcastTo_MR_BaseShellParameters", ExactSpelling = true)]
            extern static MR.Const_BaseShellParameters._Underlying *__MR_OffsetParameters_UpcastTo_MR_BaseShellParameters(_Underlying *_this);
            MR.Const_BaseShellParameters ret = new(__MR_OffsetParameters_UpcastTo_MR_BaseShellParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// determines the method to compute distance sign
        public unsafe MR.SignDetectionMode SignDetectionMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_signDetectionMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_OffsetParameters_Get_signDetectionMode(_Underlying *_this);
                return *__MR_OffsetParameters_Get_signDetectionMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public unsafe bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_OffsetParameters_Get_closeHolesInHoleWindingNumber(_Underlying *_this);
                return *__MR_OffsetParameters_Get_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_OffsetParameters_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_OffsetParameters_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_OffsetParameters_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_OffsetParameters_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
        /// providing this will disable memoryEfficient (as if memoryEfficient == false)
        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_OffsetParameters_Get_fwn(_Underlying *_this);
                return new(__MR_OffsetParameters_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// use FunctionVolume for voxel grid representation:
        ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
        ///  - computations are about 15% slower (because some z-layers are computed twice)
        /// this setting is ignored (as if memoryEfficient == false) if
        ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
        ///  b) \ref fwn is provided (CUDA computations require full memory storage)
        /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
        public unsafe bool MemoryEfficient
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_memoryEfficient", ExactSpelling = true)]
                extern static bool *__MR_OffsetParameters_Get_memoryEfficient(_Underlying *_this);
                return *__MR_OffsetParameters_Get_memoryEfficient(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_OffsetParameters_Get_voxelSize(_Underlying *_this);
                return *__MR_OffsetParameters_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_Get_callBack", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_OffsetParameters_Get_callBack(_Underlying *_this);
                return new(__MR_OffsetParameters_Get_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OffsetParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_OffsetParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::OffsetParameters::OffsetParameters`.
        public unsafe Const_OffsetParameters(MR._ByValue_OffsetParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_OffsetParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OffsetParameters._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::OffsetParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::BaseShellParameters`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SharpOffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::GeneralOffsetParameters`
    /// This is the non-const half of the class.
    public class OffsetParameters : Const_OffsetParameters
    {
        internal unsafe OffsetParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseShellParameters(OffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_UpcastTo_MR_BaseShellParameters", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_OffsetParameters_UpcastTo_MR_BaseShellParameters(_Underlying *_this);
            MR.BaseShellParameters ret = new(__MR_OffsetParameters_UpcastTo_MR_BaseShellParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// determines the method to compute distance sign
        public new unsafe ref MR.SignDetectionMode SignDetectionMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_signDetectionMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_OffsetParameters_GetMutable_signDetectionMode(_Underlying *_this);
                return ref *__MR_OffsetParameters_GetMutable_signDetectionMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public new unsafe ref bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_OffsetParameters_GetMutable_closeHolesInHoleWindingNumber(_Underlying *_this);
                return ref *__MR_OffsetParameters_GetMutable_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_OffsetParameters_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_OffsetParameters_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_OffsetParameters_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_OffsetParameters_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
        /// providing this will disable memoryEfficient (as if memoryEfficient == false)
        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_OffsetParameters_GetMutable_fwn(_Underlying *_this);
                return new(__MR_OffsetParameters_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// use FunctionVolume for voxel grid representation:
        ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
        ///  - computations are about 15% slower (because some z-layers are computed twice)
        /// this setting is ignored (as if memoryEfficient == false) if
        ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
        ///  b) \ref fwn is provided (CUDA computations require full memory storage)
        /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
        public new unsafe ref bool MemoryEfficient
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_memoryEfficient", ExactSpelling = true)]
                extern static bool *__MR_OffsetParameters_GetMutable_memoryEfficient(_Underlying *_this);
                return ref *__MR_OffsetParameters_GetMutable_memoryEfficient(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_OffsetParameters_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_OffsetParameters_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_GetMutable_callBack", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_OffsetParameters_GetMutable_callBack(_Underlying *_this);
                return new(__MR_OffsetParameters_GetMutable_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OffsetParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_OffsetParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_OffsetParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::OffsetParameters::OffsetParameters`.
        public unsafe OffsetParameters(MR._ByValue_OffsetParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_OffsetParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OffsetParameters._Underlying *_other);
            _UnderlyingPtr = __MR_OffsetParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::OffsetParameters::operator=`.
        public unsafe MR.OffsetParameters Assign(MR._ByValue_OffsetParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OffsetParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_OffsetParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.OffsetParameters._Underlying *_other);
            return new(__MR_OffsetParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `OffsetParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `OffsetParameters`/`Const_OffsetParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_OffsetParameters
    {
        internal readonly Const_OffsetParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_OffsetParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_OffsetParameters(Const_OffsetParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_OffsetParameters(Const_OffsetParameters arg) {return new(arg);}
        public _ByValue_OffsetParameters(MR.Misc._Moved<OffsetParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_OffsetParameters(MR.Misc._Moved<OffsetParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `OffsetParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OffsetParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetParameters`/`Const_OffsetParameters` directly.
    public class _InOptMut_OffsetParameters
    {
        public OffsetParameters? Opt;

        public _InOptMut_OffsetParameters() {}
        public _InOptMut_OffsetParameters(OffsetParameters value) {Opt = value;}
        public static implicit operator _InOptMut_OffsetParameters(OffsetParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `OffsetParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OffsetParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OffsetParameters`/`Const_OffsetParameters` to pass it to the function.
    public class _InOptConst_OffsetParameters
    {
        public Const_OffsetParameters? Opt;

        public _InOptConst_OffsetParameters() {}
        public _InOptConst_OffsetParameters(Const_OffsetParameters value) {Opt = value;}
        public static implicit operator _InOptConst_OffsetParameters(Const_OffsetParameters value) {return new(value);}
    }

    /// Generated from class `MR::SharpOffsetParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::OffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::BaseShellParameters`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::GeneralOffsetParameters`
    /// This is the const half of the class.
    public class Const_SharpOffsetParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SharpOffsetParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_SharpOffsetParameters_Destroy(_Underlying *_this);
            __MR_SharpOffsetParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SharpOffsetParameters() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseShellParameters(Const_SharpOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_UpcastTo_MR_BaseShellParameters", ExactSpelling = true)]
            extern static MR.Const_BaseShellParameters._Underlying *__MR_SharpOffsetParameters_UpcastTo_MR_BaseShellParameters(_Underlying *_this);
            MR.Const_BaseShellParameters ret = new(__MR_SharpOffsetParameters_UpcastTo_MR_BaseShellParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_OffsetParameters(Const_SharpOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_UpcastTo_MR_OffsetParameters", ExactSpelling = true)]
            extern static MR.Const_OffsetParameters._Underlying *__MR_SharpOffsetParameters_UpcastTo_MR_OffsetParameters(_Underlying *_this);
            MR.Const_OffsetParameters ret = new(__MR_SharpOffsetParameters_UpcastTo_MR_OffsetParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// if non-null then created sharp edges will be saved here
        public unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_SharpOffsetParameters_Get_outSharpEdges(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_Get_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
        public unsafe float MinNewVertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_minNewVertDev", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_minNewVertDev(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_minNewVertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
        public unsafe float MaxNewRank2VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_maxNewRank2VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_maxNewRank2VertDev(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_maxNewRank2VertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
        public unsafe float MaxNewRank3VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_maxNewRank3VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_maxNewRank3VertDev(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_maxNewRank3VertDev(_UnderlyingPtr);
            }
        }

        /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
        /// big correction can be wrong and result from self-intersections in the reference mesh
        public unsafe float MaxOldVertPosCorrection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_maxOldVertPosCorrection", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_maxOldVertPosCorrection(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_maxOldVertPosCorrection(_UnderlyingPtr);
            }
        }

        /// determines the method to compute distance sign
        public unsafe MR.SignDetectionMode SignDetectionMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_signDetectionMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_SharpOffsetParameters_Get_signDetectionMode(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_signDetectionMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public unsafe bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_SharpOffsetParameters_Get_closeHolesInHoleWindingNumber(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
        /// providing this will disable memoryEfficient (as if memoryEfficient == false)
        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_SharpOffsetParameters_Get_fwn(_Underlying *_this);
                return new(__MR_SharpOffsetParameters_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// use FunctionVolume for voxel grid representation:
        ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
        ///  - computations are about 15% slower (because some z-layers are computed twice)
        /// this setting is ignored (as if memoryEfficient == false) if
        ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
        ///  b) \ref fwn is provided (CUDA computations require full memory storage)
        /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
        public unsafe bool MemoryEfficient
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_memoryEfficient", ExactSpelling = true)]
                extern static bool *__MR_SharpOffsetParameters_Get_memoryEfficient(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_memoryEfficient(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_Get_voxelSize(_Underlying *_this);
                return *__MR_SharpOffsetParameters_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_Get_callBack", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_SharpOffsetParameters_Get_callBack(_Underlying *_this);
                return new(__MR_SharpOffsetParameters_Get_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SharpOffsetParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SharpOffsetParameters._Underlying *__MR_SharpOffsetParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_SharpOffsetParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::SharpOffsetParameters::SharpOffsetParameters`.
        public unsafe Const_SharpOffsetParameters(MR._ByValue_SharpOffsetParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SharpOffsetParameters._Underlying *__MR_SharpOffsetParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SharpOffsetParameters._Underlying *_other);
            _UnderlyingPtr = __MR_SharpOffsetParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::SharpOffsetParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::OffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::BaseShellParameters`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::GeneralOffsetParameters`
    /// This is the non-const half of the class.
    public class SharpOffsetParameters : Const_SharpOffsetParameters
    {
        internal unsafe SharpOffsetParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseShellParameters(SharpOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_UpcastTo_MR_BaseShellParameters", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_SharpOffsetParameters_UpcastTo_MR_BaseShellParameters(_Underlying *_this);
            MR.BaseShellParameters ret = new(__MR_SharpOffsetParameters_UpcastTo_MR_BaseShellParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.OffsetParameters(SharpOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_UpcastTo_MR_OffsetParameters", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_SharpOffsetParameters_UpcastTo_MR_OffsetParameters(_Underlying *_this);
            MR.OffsetParameters ret = new(__MR_SharpOffsetParameters_UpcastTo_MR_OffsetParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// if non-null then created sharp edges will be saved here
        public new unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_SharpOffsetParameters_GetMutable_outSharpEdges(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
        public new unsafe ref float MinNewVertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_minNewVertDev", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_minNewVertDev(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_minNewVertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
        public new unsafe ref float MaxNewRank2VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_maxNewRank2VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_maxNewRank2VertDev(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_maxNewRank2VertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
        public new unsafe ref float MaxNewRank3VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_maxNewRank3VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_maxNewRank3VertDev(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_maxNewRank3VertDev(_UnderlyingPtr);
            }
        }

        /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
        /// big correction can be wrong and result from self-intersections in the reference mesh
        public new unsafe ref float MaxOldVertPosCorrection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_maxOldVertPosCorrection", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_maxOldVertPosCorrection(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_maxOldVertPosCorrection(_UnderlyingPtr);
            }
        }

        /// determines the method to compute distance sign
        public new unsafe ref MR.SignDetectionMode SignDetectionMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_signDetectionMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_SharpOffsetParameters_GetMutable_signDetectionMode(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_signDetectionMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public new unsafe ref bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_SharpOffsetParameters_GetMutable_closeHolesInHoleWindingNumber(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
        /// providing this will disable memoryEfficient (as if memoryEfficient == false)
        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_SharpOffsetParameters_GetMutable_fwn(_Underlying *_this);
                return new(__MR_SharpOffsetParameters_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// use FunctionVolume for voxel grid representation:
        ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
        ///  - computations are about 15% slower (because some z-layers are computed twice)
        /// this setting is ignored (as if memoryEfficient == false) if
        ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
        ///  b) \ref fwn is provided (CUDA computations require full memory storage)
        /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
        public new unsafe ref bool MemoryEfficient
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_memoryEfficient", ExactSpelling = true)]
                extern static bool *__MR_SharpOffsetParameters_GetMutable_memoryEfficient(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_memoryEfficient(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_SharpOffsetParameters_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_SharpOffsetParameters_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_GetMutable_callBack", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_SharpOffsetParameters_GetMutable_callBack(_Underlying *_this);
                return new(__MR_SharpOffsetParameters_GetMutable_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SharpOffsetParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SharpOffsetParameters._Underlying *__MR_SharpOffsetParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_SharpOffsetParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::SharpOffsetParameters::SharpOffsetParameters`.
        public unsafe SharpOffsetParameters(MR._ByValue_SharpOffsetParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SharpOffsetParameters._Underlying *__MR_SharpOffsetParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SharpOffsetParameters._Underlying *_other);
            _UnderlyingPtr = __MR_SharpOffsetParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SharpOffsetParameters::operator=`.
        public unsafe MR.SharpOffsetParameters Assign(MR._ByValue_SharpOffsetParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpOffsetParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SharpOffsetParameters._Underlying *__MR_SharpOffsetParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SharpOffsetParameters._Underlying *_other);
            return new(__MR_SharpOffsetParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SharpOffsetParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SharpOffsetParameters`/`Const_SharpOffsetParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SharpOffsetParameters
    {
        internal readonly Const_SharpOffsetParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SharpOffsetParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SharpOffsetParameters(Const_SharpOffsetParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SharpOffsetParameters(Const_SharpOffsetParameters arg) {return new(arg);}
        public _ByValue_SharpOffsetParameters(MR.Misc._Moved<SharpOffsetParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SharpOffsetParameters(MR.Misc._Moved<SharpOffsetParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SharpOffsetParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SharpOffsetParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SharpOffsetParameters`/`Const_SharpOffsetParameters` directly.
    public class _InOptMut_SharpOffsetParameters
    {
        public SharpOffsetParameters? Opt;

        public _InOptMut_SharpOffsetParameters() {}
        public _InOptMut_SharpOffsetParameters(SharpOffsetParameters value) {Opt = value;}
        public static implicit operator _InOptMut_SharpOffsetParameters(SharpOffsetParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `SharpOffsetParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SharpOffsetParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SharpOffsetParameters`/`Const_SharpOffsetParameters` to pass it to the function.
    public class _InOptConst_SharpOffsetParameters
    {
        public Const_SharpOffsetParameters? Opt;

        public _InOptConst_SharpOffsetParameters() {}
        public _InOptConst_SharpOffsetParameters(Const_SharpOffsetParameters value) {Opt = value;}
        public static implicit operator _InOptConst_SharpOffsetParameters(Const_SharpOffsetParameters value) {return new(value);}
    }

    /// allows the user to select in the parameters which offset algorithm to call
    /// Generated from class `MR::GeneralOffsetParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SharpOffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::BaseShellParameters`
    ///     `MR::OffsetParameters`
    /// This is the const half of the class.
    public class Const_GeneralOffsetParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GeneralOffsetParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_GeneralOffsetParameters_Destroy(_Underlying *_this);
            __MR_GeneralOffsetParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GeneralOffsetParameters() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_BaseShellParameters(Const_GeneralOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_UpcastTo_MR_BaseShellParameters", ExactSpelling = true)]
            extern static MR.Const_BaseShellParameters._Underlying *__MR_GeneralOffsetParameters_UpcastTo_MR_BaseShellParameters(_Underlying *_this);
            MR.Const_BaseShellParameters ret = new(__MR_GeneralOffsetParameters_UpcastTo_MR_BaseShellParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_OffsetParameters(Const_GeneralOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_UpcastTo_MR_OffsetParameters", ExactSpelling = true)]
            extern static MR.Const_OffsetParameters._Underlying *__MR_GeneralOffsetParameters_UpcastTo_MR_OffsetParameters(_Underlying *_this);
            MR.Const_OffsetParameters ret = new(__MR_GeneralOffsetParameters_UpcastTo_MR_OffsetParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_SharpOffsetParameters(Const_GeneralOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_UpcastTo_MR_SharpOffsetParameters", ExactSpelling = true)]
            extern static MR.Const_SharpOffsetParameters._Underlying *__MR_GeneralOffsetParameters_UpcastTo_MR_SharpOffsetParameters(_Underlying *_this);
            MR.Const_SharpOffsetParameters ret = new(__MR_GeneralOffsetParameters_UpcastTo_MR_SharpOffsetParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.OffsetMode Mode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_mode", ExactSpelling = true)]
                extern static MR.OffsetMode *__MR_GeneralOffsetParameters_Get_mode(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_mode(_UnderlyingPtr);
            }
        }

        /// if non-null then created sharp edges will be saved here
        public unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_GeneralOffsetParameters_Get_outSharpEdges(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_Get_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
        public unsafe float MinNewVertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_minNewVertDev", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_minNewVertDev(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_minNewVertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
        public unsafe float MaxNewRank2VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_maxNewRank2VertDev", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_maxNewRank2VertDev(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_maxNewRank2VertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
        public unsafe float MaxNewRank3VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_maxNewRank3VertDev", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_maxNewRank3VertDev(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_maxNewRank3VertDev(_UnderlyingPtr);
            }
        }

        /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
        /// big correction can be wrong and result from self-intersections in the reference mesh
        public unsafe float MaxOldVertPosCorrection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_maxOldVertPosCorrection", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_maxOldVertPosCorrection(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_maxOldVertPosCorrection(_UnderlyingPtr);
            }
        }

        /// determines the method to compute distance sign
        public unsafe MR.SignDetectionMode SignDetectionMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_signDetectionMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_GeneralOffsetParameters_Get_signDetectionMode(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_signDetectionMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public unsafe bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_GeneralOffsetParameters_Get_closeHolesInHoleWindingNumber(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
        /// providing this will disable memoryEfficient (as if memoryEfficient == false)
        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_GeneralOffsetParameters_Get_fwn(_Underlying *_this);
                return new(__MR_GeneralOffsetParameters_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// use FunctionVolume for voxel grid representation:
        ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
        ///  - computations are about 15% slower (because some z-layers are computed twice)
        /// this setting is ignored (as if memoryEfficient == false) if
        ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
        ///  b) \ref fwn is provided (CUDA computations require full memory storage)
        /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
        public unsafe bool MemoryEfficient
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_memoryEfficient", ExactSpelling = true)]
                extern static bool *__MR_GeneralOffsetParameters_Get_memoryEfficient(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_memoryEfficient(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_Get_voxelSize(_Underlying *_this);
                return *__MR_GeneralOffsetParameters_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_Get_callBack", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_GeneralOffsetParameters_Get_callBack(_Underlying *_this);
                return new(__MR_GeneralOffsetParameters_Get_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GeneralOffsetParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GeneralOffsetParameters._Underlying *__MR_GeneralOffsetParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_GeneralOffsetParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::GeneralOffsetParameters::GeneralOffsetParameters`.
        public unsafe Const_GeneralOffsetParameters(MR._ByValue_GeneralOffsetParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GeneralOffsetParameters._Underlying *__MR_GeneralOffsetParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GeneralOffsetParameters._Underlying *_other);
            _UnderlyingPtr = __MR_GeneralOffsetParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// allows the user to select in the parameters which offset algorithm to call
    /// Generated from class `MR::GeneralOffsetParameters`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::SharpOffsetParameters`
    ///   Indirect: (non-virtual)
    ///     `MR::BaseShellParameters`
    ///     `MR::OffsetParameters`
    /// This is the non-const half of the class.
    public class GeneralOffsetParameters : Const_GeneralOffsetParameters
    {
        internal unsafe GeneralOffsetParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.BaseShellParameters(GeneralOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_UpcastTo_MR_BaseShellParameters", ExactSpelling = true)]
            extern static MR.BaseShellParameters._Underlying *__MR_GeneralOffsetParameters_UpcastTo_MR_BaseShellParameters(_Underlying *_this);
            MR.BaseShellParameters ret = new(__MR_GeneralOffsetParameters_UpcastTo_MR_BaseShellParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.OffsetParameters(GeneralOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_UpcastTo_MR_OffsetParameters", ExactSpelling = true)]
            extern static MR.OffsetParameters._Underlying *__MR_GeneralOffsetParameters_UpcastTo_MR_OffsetParameters(_Underlying *_this);
            MR.OffsetParameters ret = new(__MR_GeneralOffsetParameters_UpcastTo_MR_OffsetParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.SharpOffsetParameters(GeneralOffsetParameters self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_UpcastTo_MR_SharpOffsetParameters", ExactSpelling = true)]
            extern static MR.SharpOffsetParameters._Underlying *__MR_GeneralOffsetParameters_UpcastTo_MR_SharpOffsetParameters(_Underlying *_this);
            MR.SharpOffsetParameters ret = new(__MR_GeneralOffsetParameters_UpcastTo_MR_SharpOffsetParameters(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref MR.OffsetMode Mode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_mode", ExactSpelling = true)]
                extern static MR.OffsetMode *__MR_GeneralOffsetParameters_GetMutable_mode(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_mode(_UnderlyingPtr);
            }
        }

        /// if non-null then created sharp edges will be saved here
        public new unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_GeneralOffsetParameters_GetMutable_outSharpEdges(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
        public new unsafe ref float MinNewVertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_minNewVertDev", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_minNewVertDev(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_minNewVertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
        public new unsafe ref float MaxNewRank2VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_maxNewRank2VertDev", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_maxNewRank2VertDev(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_maxNewRank2VertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
        public new unsafe ref float MaxNewRank3VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_maxNewRank3VertDev", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_maxNewRank3VertDev(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_maxNewRank3VertDev(_UnderlyingPtr);
            }
        }

        /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
        /// big correction can be wrong and result from self-intersections in the reference mesh
        public new unsafe ref float MaxOldVertPosCorrection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_maxOldVertPosCorrection", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_maxOldVertPosCorrection(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_maxOldVertPosCorrection(_UnderlyingPtr);
            }
        }

        /// determines the method to compute distance sign
        public new unsafe ref MR.SignDetectionMode SignDetectionMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_signDetectionMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_GeneralOffsetParameters_GetMutable_signDetectionMode(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_signDetectionMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public new unsafe ref bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_GeneralOffsetParameters_GetMutable_closeHolesInHoleWindingNumber(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
        /// providing this will disable memoryEfficient (as if memoryEfficient == false)
        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_GeneralOffsetParameters_GetMutable_fwn(_Underlying *_this);
                return new(__MR_GeneralOffsetParameters_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// use FunctionVolume for voxel grid representation:
        ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
        ///  - computations are about 15% slower (because some z-layers are computed twice)
        /// this setting is ignored (as if memoryEfficient == false) if
        ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
        ///  b) \ref fwn is provided (CUDA computations require full memory storage)
        /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
        public new unsafe ref bool MemoryEfficient
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_memoryEfficient", ExactSpelling = true)]
                extern static bool *__MR_GeneralOffsetParameters_GetMutable_memoryEfficient(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_memoryEfficient(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_GeneralOffsetParameters_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_GeneralOffsetParameters_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat CallBack
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_GetMutable_callBack", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_GeneralOffsetParameters_GetMutable_callBack(_Underlying *_this);
                return new(__MR_GeneralOffsetParameters_GetMutable_callBack(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe GeneralOffsetParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GeneralOffsetParameters._Underlying *__MR_GeneralOffsetParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_GeneralOffsetParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::GeneralOffsetParameters::GeneralOffsetParameters`.
        public unsafe GeneralOffsetParameters(MR._ByValue_GeneralOffsetParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GeneralOffsetParameters._Underlying *__MR_GeneralOffsetParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GeneralOffsetParameters._Underlying *_other);
            _UnderlyingPtr = __MR_GeneralOffsetParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::GeneralOffsetParameters::operator=`.
        public unsafe MR.GeneralOffsetParameters Assign(MR._ByValue_GeneralOffsetParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GeneralOffsetParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GeneralOffsetParameters._Underlying *__MR_GeneralOffsetParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GeneralOffsetParameters._Underlying *_other);
            return new(__MR_GeneralOffsetParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `GeneralOffsetParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `GeneralOffsetParameters`/`Const_GeneralOffsetParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_GeneralOffsetParameters
    {
        internal readonly Const_GeneralOffsetParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_GeneralOffsetParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_GeneralOffsetParameters(Const_GeneralOffsetParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_GeneralOffsetParameters(Const_GeneralOffsetParameters arg) {return new(arg);}
        public _ByValue_GeneralOffsetParameters(MR.Misc._Moved<GeneralOffsetParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_GeneralOffsetParameters(MR.Misc._Moved<GeneralOffsetParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `GeneralOffsetParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GeneralOffsetParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GeneralOffsetParameters`/`Const_GeneralOffsetParameters` directly.
    public class _InOptMut_GeneralOffsetParameters
    {
        public GeneralOffsetParameters? Opt;

        public _InOptMut_GeneralOffsetParameters() {}
        public _InOptMut_GeneralOffsetParameters(GeneralOffsetParameters value) {Opt = value;}
        public static implicit operator _InOptMut_GeneralOffsetParameters(GeneralOffsetParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `GeneralOffsetParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GeneralOffsetParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GeneralOffsetParameters`/`Const_GeneralOffsetParameters` to pass it to the function.
    public class _InOptConst_GeneralOffsetParameters
    {
        public Const_GeneralOffsetParameters? Opt;

        public _InOptConst_GeneralOffsetParameters() {}
        public _InOptConst_GeneralOffsetParameters(Const_GeneralOffsetParameters value) {Opt = value;}
        public static implicit operator _InOptConst_GeneralOffsetParameters(Const_GeneralOffsetParameters value) {return new(value);}
    }

    /// computes size of a cubical voxel to get approximately given number of voxels during rasterization
    /// Generated from function `MR::suggestVoxelSize`.
    public static unsafe float SuggestVoxelSize(MR.Const_MeshPart mp, float approxNumVoxels)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_suggestVoxelSize", ExactSpelling = true)]
        extern static float __MR_suggestVoxelSize(MR.Const_MeshPart._Underlying *mp, float approxNumVoxels);
        return __MR_suggestVoxelSize(mp._UnderlyingPtr, approxNumVoxels);
    }

    /// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
    /// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
    /// and then converts back using OpenVDB library (dual marching cubes),
    /// so result mesh is always closed
    /// Generated from function `MR::offsetMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> OffsetMesh(MR.Const_MeshPart mp, float offset, MR.Const_OffsetParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_offsetMesh(MR.Const_MeshPart._Underlying *mp, float offset, MR.Const_OffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_offsetMesh(mp._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Offsets mesh by converting it to voxels and back two times
    /// only closed meshes allowed (only Offset mode)
    /// typically offsetA and offsetB have distinct signs
    /// Generated from function `MR::doubleOffsetMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> DoubleOffsetMesh(MR.Const_MeshPart mp, float offsetA, float offsetB, MR.Const_OffsetParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_doubleOffsetMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_doubleOffsetMesh(MR.Const_MeshPart._Underlying *mp, float offsetA, float offsetB, MR.Const_OffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_doubleOffsetMesh(mp._UnderlyingPtr, offsetA, offsetB, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Offsets mesh by converting it to distance field in voxels (using OpenVDB library if SignDetectionMode::OpenVDB or our implementation otherwise)
    /// and back using standard Marching Cubes, as opposed to Dual Marching Cubes in offsetMesh(...)
    /// Generated from function `MR::mcOffsetMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> McOffsetMesh(MR.Const_MeshPart mp, float offset, MR.Const_OffsetParameters? params_ = null, MR.Vector_MRVoxelId_MRFaceId? outMap = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mcOffsetMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_mcOffsetMesh(MR.Const_MeshPart._Underlying *mp, float offset, MR.Const_OffsetParameters._Underlying *params_, MR.Vector_MRVoxelId_MRFaceId._Underlying *outMap);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_mcOffsetMesh(mp._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null, outMap is not null ? outMap._UnderlyingPtr : null), is_owning: true));
    }

    /// Constructs a shell around selected mesh region with the properties that every point on the shall must
    ///  1. be located not further than given distance from selected mesh part,
    ///  2. be located not closer to not-selected mesh part than to selected mesh part.
    /// Generated from function `MR::mcShellMeshRegion`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> McShellMeshRegion(MR.Const_Mesh mesh, MR.Const_FaceBitSet region, float offset, MR.Const_BaseShellParameters params_, MR.Vector_MRVoxelId_MRFaceId? outMap = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mcShellMeshRegion", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_mcShellMeshRegion(MR.Const_Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *region, float offset, MR.Const_BaseShellParameters._Underlying *params_, MR.Vector_MRVoxelId_MRFaceId._Underlying *outMap);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_mcShellMeshRegion(mesh._UnderlyingPtr, region._UnderlyingPtr, offset, params_._UnderlyingPtr, outMap is not null ? outMap._UnderlyingPtr : null), is_owning: true));
    }

    /// Offsets mesh by converting it to voxels and back
    /// post process result using reference mesh to sharpen features
    /// Generated from function `MR::sharpOffsetMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> SharpOffsetMesh(MR.Const_MeshPart mp, float offset, MR.Const_SharpOffsetParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sharpOffsetMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_sharpOffsetMesh(MR.Const_MeshPart._Underlying *mp, float offset, MR.Const_SharpOffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_sharpOffsetMesh(mp._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Offsets mesh by converting it to voxels and back using one of three modes specified in the parameters
    /// \snippet cpp-examples/MeshOffset.dox.cpp 0
    /// Generated from function `MR::generalOffsetMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> GeneralOffsetMesh(MR.Const_MeshPart mp, float offset, MR.Const_GeneralOffsetParameters params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_generalOffsetMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_generalOffsetMesh(MR.Const_MeshPart._Underlying *mp, float offset, MR.Const_GeneralOffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_generalOffsetMesh(mp._UnderlyingPtr, offset, params_._UnderlyingPtr), is_owning: true));
    }

    /// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
    /// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
    /// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode::Unsigned, and you will get open mesh (with several components) on output
    /// if your input mesh is closed then please specify another sign detection mode, and you will get closed mesh (with several components) on output;
    /// Generated from function `MR::thickenMesh`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `map` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> ThickenMesh(MR.Const_Mesh mesh, float offset, MR.Const_GeneralOffsetParameters? params_ = null, MR.Const_PartMapping? map = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_thickenMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_thickenMesh(MR.Const_Mesh._Underlying *mesh, float offset, MR.Const_GeneralOffsetParameters._Underlying *params_, MR.Const_PartMapping._Underlying *map);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_thickenMesh(mesh._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null, map is not null ? map._UnderlyingPtr : null), is_owning: true));
    }

    /// offsets given MeshPart in one direction only (positive or negative)
    /// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode::Unsigned
    /// if your input mesh is closed this function is equivalent to `generalOffsetMesh`, but in SignDetectionMode::Unsigned mode it will only keep one side (just like for open mesh)
    /// unlike `thickenMesh` this functions does not keep original mesh in result
    /// Generated from function `MR::offsetOneDirection`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> OffsetOneDirection(MR.Const_MeshPart mp, float offset, MR.Const_GeneralOffsetParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetOneDirection", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_offsetOneDirection(MR.Const_MeshPart._Underlying *mp, float offset, MR.Const_GeneralOffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_offsetOneDirection(mp._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Offsets polyline by converting it to voxels and building iso-surface
    /// do offset in all directions
    /// so result mesh is always closed
    /// params.signDetectionMode is ignored (always assumed SignDetectionMode::Unsigned)
    /// Generated from function `MR::offsetPolyline`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> OffsetPolyline(MR.Const_Polyline3 polyline, float offset, MR.Const_OffsetParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetPolyline", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_offsetPolyline(MR.Const_Polyline3._Underlying *polyline, float offset, MR.Const_OffsetParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_offsetPolyline(polyline._UnderlyingPtr, offset, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
