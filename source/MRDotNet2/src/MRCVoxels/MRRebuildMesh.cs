public static partial class MR
{
    /// Generated from class `MR::RebuildMeshSettings`.
    /// This is the const half of the class.
    public class Const_RebuildMeshSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RebuildMeshSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_RebuildMeshSettings_Destroy(_Underlying *_this);
            __MR_RebuildMeshSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RebuildMeshSettings() {Dispose(false);}

        /// whether to make subdivision of initial mesh before conversion to voxels,
        /// despite time and memory required for the subdivision, it typically makes the whole rebuilding faster (or even much faster in case of large initial triangles),
        /// because AABB tree contains small triangles in leaves, which is good for both
        /// 1) search for closest triangle because the closest box more frequently contains the closest triangle,
        /// 2) and winding number approximation because of more frequent usage of approximation for distant dipoles
        public unsafe bool PreSubdivide
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_preSubdivide", ExactSpelling = true)]
                extern static bool *__MR_RebuildMeshSettings_Get_preSubdivide(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_preSubdivide(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_Get_voxelSize(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_voxelSize(_UnderlyingPtr);
            }
        }

        public unsafe MR.SignDetectionModeShort SignMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_signMode", ExactSpelling = true)]
                extern static MR.SignDetectionModeShort *__MR_RebuildMeshSettings_Get_signMode(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_signMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public unsafe bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_RebuildMeshSettings_Get_closeHolesInHoleWindingNumber(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        public unsafe MR.OffsetMode OffsetMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_offsetMode", ExactSpelling = true)]
                extern static MR.OffsetMode *__MR_RebuildMeshSettings_Get_offsetMode(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_offsetMode(_UnderlyingPtr);
            }
        }

        /// if non-null then created sharp edges (only if offsetMode = OffsetMode::Sharpening) will be saved here
        public unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_RebuildMeshSettings_Get_outSharpEdges(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_Get_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// if general winding number is used to differentiate inside from outside:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// if general winding number is used to differentiate inside from outside:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings (if required).
        /// If it is not specified, default FastWindingNumber is used
        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_RebuildMeshSettings_Get_fwn(_Underlying *_this);
                return new(__MR_RebuildMeshSettings_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// whether to decimate resulting mesh
        public unsafe bool Decimate
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_decimate", ExactSpelling = true)]
                extern static bool *__MR_RebuildMeshSettings_Get_decimate(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_decimate(_UnderlyingPtr);
            }
        }

        /// only if decimate = true:
        /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
        public unsafe float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_Get_tinyEdgeLength(_Underlying *_this);
                return *__MR_RebuildMeshSettings_Get_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// To report algorithm's progress and cancel it on user demand
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_RebuildMeshSettings_Get_progress(_Underlying *_this);
                return new(__MR_RebuildMeshSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this callback is invoked when SignDetectionMode is determined (useful if signMode = SignDetectionModeShort::Auto),
        /// but before actual work begins
        public unsafe MR.Std.Const_Function_VoidFuncFromMRSignDetectionMode OnSignDetectionModeSelected
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_Get_onSignDetectionModeSelected", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRSignDetectionMode._Underlying *__MR_RebuildMeshSettings_Get_onSignDetectionModeSelected(_Underlying *_this);
                return new(__MR_RebuildMeshSettings_Get_onSignDetectionModeSelected(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RebuildMeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_RebuildMeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::RebuildMeshSettings` elementwise.
        public unsafe Const_RebuildMeshSettings(bool preSubdivide, float voxelSize, MR.SignDetectionModeShort signMode, bool closeHolesInHoleWindingNumber, MR.OffsetMode offsetMode, MR.UndirectedEdgeBitSet? outSharpEdges, float windingNumberThreshold, float windingNumberBeta, MR._ByValue_IFastWindingNumber fwn, bool decimate, float tinyEdgeLength, MR.Std._ByValue_Function_BoolFuncFromFloat progress, MR.Std._ByValue_Function_VoidFuncFromMRSignDetectionMode onSignDetectionModeSelected) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_ConstructFrom(byte preSubdivide, float voxelSize, MR.SignDetectionModeShort signMode, byte closeHolesInHoleWindingNumber, MR.OffsetMode offsetMode, MR.UndirectedEdgeBitSet._Underlying *outSharpEdges, float windingNumberThreshold, float windingNumberBeta, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn, byte decimate, float tinyEdgeLength, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress, MR.Misc._PassBy onSignDetectionModeSelected_pass_by, MR.Std.Function_VoidFuncFromMRSignDetectionMode._Underlying *onSignDetectionModeSelected);
            _UnderlyingPtr = __MR_RebuildMeshSettings_ConstructFrom(preSubdivide ? (byte)1 : (byte)0, voxelSize, signMode, closeHolesInHoleWindingNumber ? (byte)1 : (byte)0, offsetMode, outSharpEdges is not null ? outSharpEdges._UnderlyingPtr : null, windingNumberThreshold, windingNumberBeta, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null, decimate ? (byte)1 : (byte)0, tinyEdgeLength, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null, onSignDetectionModeSelected.PassByMode, onSignDetectionModeSelected.Value is not null ? onSignDetectionModeSelected.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::RebuildMeshSettings::RebuildMeshSettings`.
        public unsafe Const_RebuildMeshSettings(MR._ByValue_RebuildMeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RebuildMeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_RebuildMeshSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::RebuildMeshSettings`.
    /// This is the non-const half of the class.
    public class RebuildMeshSettings : Const_RebuildMeshSettings
    {
        internal unsafe RebuildMeshSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// whether to make subdivision of initial mesh before conversion to voxels,
        /// despite time and memory required for the subdivision, it typically makes the whole rebuilding faster (or even much faster in case of large initial triangles),
        /// because AABB tree contains small triangles in leaves, which is good for both
        /// 1) search for closest triangle because the closest box more frequently contains the closest triangle,
        /// 2) and winding number approximation because of more frequent usage of approximation for distant dipoles
        public new unsafe ref bool PreSubdivide
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_preSubdivide", ExactSpelling = true)]
                extern static bool *__MR_RebuildMeshSettings_GetMutable_preSubdivide(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_preSubdivide(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.SignDetectionModeShort SignMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_signMode", ExactSpelling = true)]
                extern static MR.SignDetectionModeShort *__MR_RebuildMeshSettings_GetMutable_signMode(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_signMode(_UnderlyingPtr);
            }
        }

        /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
        public new unsafe ref bool CloseHolesInHoleWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_closeHolesInHoleWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_RebuildMeshSettings_GetMutable_closeHolesInHoleWindingNumber(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_closeHolesInHoleWindingNumber(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.OffsetMode OffsetMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_offsetMode", ExactSpelling = true)]
                extern static MR.OffsetMode *__MR_RebuildMeshSettings_GetMutable_offsetMode(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_offsetMode(_UnderlyingPtr);
            }
        }

        /// if non-null then created sharp edges (only if offsetMode = OffsetMode::Sharpening) will be saved here
        public new unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_RebuildMeshSettings_GetMutable_outSharpEdges(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// if general winding number is used to differentiate inside from outside:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// if general winding number is used to differentiate inside from outside:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings (if required).
        /// If it is not specified, default FastWindingNumber is used
        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_RebuildMeshSettings_GetMutable_fwn(_Underlying *_this);
                return new(__MR_RebuildMeshSettings_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// whether to decimate resulting mesh
        public new unsafe ref bool Decimate
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_decimate", ExactSpelling = true)]
                extern static bool *__MR_RebuildMeshSettings_GetMutable_decimate(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_decimate(_UnderlyingPtr);
            }
        }

        /// only if decimate = true:
        /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
        public new unsafe ref float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_RebuildMeshSettings_GetMutable_tinyEdgeLength(_Underlying *_this);
                return ref *__MR_RebuildMeshSettings_GetMutable_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// To report algorithm's progress and cancel it on user demand
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_RebuildMeshSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_RebuildMeshSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this callback is invoked when SignDetectionMode is determined (useful if signMode = SignDetectionModeShort::Auto),
        /// but before actual work begins
        public new unsafe MR.Std.Function_VoidFuncFromMRSignDetectionMode OnSignDetectionModeSelected
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_GetMutable_onSignDetectionModeSelected", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRSignDetectionMode._Underlying *__MR_RebuildMeshSettings_GetMutable_onSignDetectionModeSelected(_Underlying *_this);
                return new(__MR_RebuildMeshSettings_GetMutable_onSignDetectionModeSelected(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RebuildMeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_RebuildMeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::RebuildMeshSettings` elementwise.
        public unsafe RebuildMeshSettings(bool preSubdivide, float voxelSize, MR.SignDetectionModeShort signMode, bool closeHolesInHoleWindingNumber, MR.OffsetMode offsetMode, MR.UndirectedEdgeBitSet? outSharpEdges, float windingNumberThreshold, float windingNumberBeta, MR._ByValue_IFastWindingNumber fwn, bool decimate, float tinyEdgeLength, MR.Std._ByValue_Function_BoolFuncFromFloat progress, MR.Std._ByValue_Function_VoidFuncFromMRSignDetectionMode onSignDetectionModeSelected) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_ConstructFrom(byte preSubdivide, float voxelSize, MR.SignDetectionModeShort signMode, byte closeHolesInHoleWindingNumber, MR.OffsetMode offsetMode, MR.UndirectedEdgeBitSet._Underlying *outSharpEdges, float windingNumberThreshold, float windingNumberBeta, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn, byte decimate, float tinyEdgeLength, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress, MR.Misc._PassBy onSignDetectionModeSelected_pass_by, MR.Std.Function_VoidFuncFromMRSignDetectionMode._Underlying *onSignDetectionModeSelected);
            _UnderlyingPtr = __MR_RebuildMeshSettings_ConstructFrom(preSubdivide ? (byte)1 : (byte)0, voxelSize, signMode, closeHolesInHoleWindingNumber ? (byte)1 : (byte)0, offsetMode, outSharpEdges is not null ? outSharpEdges._UnderlyingPtr : null, windingNumberThreshold, windingNumberBeta, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null, decimate ? (byte)1 : (byte)0, tinyEdgeLength, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null, onSignDetectionModeSelected.PassByMode, onSignDetectionModeSelected.Value is not null ? onSignDetectionModeSelected.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::RebuildMeshSettings::RebuildMeshSettings`.
        public unsafe RebuildMeshSettings(MR._ByValue_RebuildMeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RebuildMeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_RebuildMeshSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::RebuildMeshSettings::operator=`.
        public unsafe MR.RebuildMeshSettings Assign(MR._ByValue_RebuildMeshSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RebuildMeshSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RebuildMeshSettings._Underlying *__MR_RebuildMeshSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RebuildMeshSettings._Underlying *_other);
            return new(__MR_RebuildMeshSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `RebuildMeshSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RebuildMeshSettings`/`Const_RebuildMeshSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RebuildMeshSettings
    {
        internal readonly Const_RebuildMeshSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RebuildMeshSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RebuildMeshSettings(Const_RebuildMeshSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_RebuildMeshSettings(Const_RebuildMeshSettings arg) {return new(arg);}
        public _ByValue_RebuildMeshSettings(MR.Misc._Moved<RebuildMeshSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RebuildMeshSettings(MR.Misc._Moved<RebuildMeshSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `RebuildMeshSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RebuildMeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RebuildMeshSettings`/`Const_RebuildMeshSettings` directly.
    public class _InOptMut_RebuildMeshSettings
    {
        public RebuildMeshSettings? Opt;

        public _InOptMut_RebuildMeshSettings() {}
        public _InOptMut_RebuildMeshSettings(RebuildMeshSettings value) {Opt = value;}
        public static implicit operator _InOptMut_RebuildMeshSettings(RebuildMeshSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `RebuildMeshSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RebuildMeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RebuildMeshSettings`/`Const_RebuildMeshSettings` to pass it to the function.
    public class _InOptConst_RebuildMeshSettings
    {
        public Const_RebuildMeshSettings? Opt;

        public _InOptConst_RebuildMeshSettings() {}
        public _InOptConst_RebuildMeshSettings(Const_RebuildMeshSettings value) {Opt = value;}
        public static implicit operator _InOptConst_RebuildMeshSettings(Const_RebuildMeshSettings value) {return new(value);}
    }

    /// fixes all types of issues in input mesh (degenerations, holes, self-intersections, etc.)
    /// by first converting mesh in voxel representation, and then backward
    /// Generated from function `MR::rebuildMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> RebuildMesh(MR.Const_MeshPart mp, MR.Const_RebuildMeshSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rebuildMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_rebuildMesh(MR.Const_MeshPart._Underlying *mp, MR.Const_RebuildMeshSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_rebuildMesh(mp._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
