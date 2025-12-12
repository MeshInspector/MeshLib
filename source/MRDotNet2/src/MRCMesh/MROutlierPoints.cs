public static partial class MR
{
    /// Parameters of various criteria for detecting outlier points
    /// Generated from class `MR::OutlierParams`.
    /// This is the const half of the class.
    public class Const_OutlierParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OutlierParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_Destroy", ExactSpelling = true)]
            extern static void __MR_OutlierParams_Destroy(_Underlying *_this);
            __MR_OutlierParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OutlierParams() {Dispose(false);}

        /// Maximum points in the outlier component
        public unsafe int MaxClusterSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_Get_maxClusterSize", ExactSpelling = true)]
                extern static int *__MR_OutlierParams_Get_maxClusterSize(_Underlying *_this);
                return *__MR_OutlierParams_Get_maxClusterSize(_UnderlyingPtr);
            }
        }

        /// Maximum number of adjacent points for an outlier point
        public unsafe int MaxNeighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_Get_maxNeighbors", ExactSpelling = true)]
                extern static int *__MR_OutlierParams_Get_maxNeighbors(_Underlying *_this);
                return *__MR_OutlierParams_Get_maxNeighbors(_UnderlyingPtr);
            }
        }

        /// Minimum distance (as proportion of search radius) to the approximate surface from outliers point
        public unsafe float MinHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_Get_minHeight", ExactSpelling = true)]
                extern static float *__MR_OutlierParams_Get_minHeight(_Underlying *_this);
                return *__MR_OutlierParams_Get_minHeight(_UnderlyingPtr);
            }
        }

        /// Minimum angle of difference of normal at outlier points
        /// @note available only if there are normals
        public unsafe float MinAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_Get_minAngle", ExactSpelling = true)]
                extern static float *__MR_OutlierParams_Get_minAngle(_Underlying *_this);
                return *__MR_OutlierParams_Get_minAngle(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OutlierParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_DefaultConstruct();
            _UnderlyingPtr = __MR_OutlierParams_DefaultConstruct();
        }

        /// Constructs `MR::OutlierParams` elementwise.
        public unsafe Const_OutlierParams(int maxClusterSize, int maxNeighbors, float minHeight, float minAngle) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_ConstructFrom(int maxClusterSize, int maxNeighbors, float minHeight, float minAngle);
            _UnderlyingPtr = __MR_OutlierParams_ConstructFrom(maxClusterSize, maxNeighbors, minHeight, minAngle);
        }

        /// Generated from constructor `MR::OutlierParams::OutlierParams`.
        public unsafe Const_OutlierParams(MR.Const_OutlierParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_ConstructFromAnother(MR.OutlierParams._Underlying *_other);
            _UnderlyingPtr = __MR_OutlierParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Parameters of various criteria for detecting outlier points
    /// Generated from class `MR::OutlierParams`.
    /// This is the non-const half of the class.
    public class OutlierParams : Const_OutlierParams
    {
        internal unsafe OutlierParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Maximum points in the outlier component
        public new unsafe ref int MaxClusterSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_GetMutable_maxClusterSize", ExactSpelling = true)]
                extern static int *__MR_OutlierParams_GetMutable_maxClusterSize(_Underlying *_this);
                return ref *__MR_OutlierParams_GetMutable_maxClusterSize(_UnderlyingPtr);
            }
        }

        /// Maximum number of adjacent points for an outlier point
        public new unsafe ref int MaxNeighbors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_GetMutable_maxNeighbors", ExactSpelling = true)]
                extern static int *__MR_OutlierParams_GetMutable_maxNeighbors(_Underlying *_this);
                return ref *__MR_OutlierParams_GetMutable_maxNeighbors(_UnderlyingPtr);
            }
        }

        /// Minimum distance (as proportion of search radius) to the approximate surface from outliers point
        public new unsafe ref float MinHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_GetMutable_minHeight", ExactSpelling = true)]
                extern static float *__MR_OutlierParams_GetMutable_minHeight(_Underlying *_this);
                return ref *__MR_OutlierParams_GetMutable_minHeight(_UnderlyingPtr);
            }
        }

        /// Minimum angle of difference of normal at outlier points
        /// @note available only if there are normals
        public new unsafe ref float MinAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_GetMutable_minAngle", ExactSpelling = true)]
                extern static float *__MR_OutlierParams_GetMutable_minAngle(_Underlying *_this);
                return ref *__MR_OutlierParams_GetMutable_minAngle(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe OutlierParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_DefaultConstruct();
            _UnderlyingPtr = __MR_OutlierParams_DefaultConstruct();
        }

        /// Constructs `MR::OutlierParams` elementwise.
        public unsafe OutlierParams(int maxClusterSize, int maxNeighbors, float minHeight, float minAngle) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_ConstructFrom(int maxClusterSize, int maxNeighbors, float minHeight, float minAngle);
            _UnderlyingPtr = __MR_OutlierParams_ConstructFrom(maxClusterSize, maxNeighbors, minHeight, minAngle);
        }

        /// Generated from constructor `MR::OutlierParams::OutlierParams`.
        public unsafe OutlierParams(MR.Const_OutlierParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_ConstructFromAnother(MR.OutlierParams._Underlying *_other);
            _UnderlyingPtr = __MR_OutlierParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::OutlierParams::operator=`.
        public unsafe MR.OutlierParams Assign(MR.Const_OutlierParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutlierParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OutlierParams._Underlying *__MR_OutlierParams_AssignFromAnother(_Underlying *_this, MR.OutlierParams._Underlying *_other);
            return new(__MR_OutlierParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `OutlierParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OutlierParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OutlierParams`/`Const_OutlierParams` directly.
    public class _InOptMut_OutlierParams
    {
        public OutlierParams? Opt;

        public _InOptMut_OutlierParams() {}
        public _InOptMut_OutlierParams(OutlierParams value) {Opt = value;}
        public static implicit operator _InOptMut_OutlierParams(OutlierParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `OutlierParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OutlierParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OutlierParams`/`Const_OutlierParams` to pass it to the function.
    public class _InOptConst_OutlierParams
    {
        public Const_OutlierParams? Opt;

        public _InOptConst_OutlierParams() {}
        public _InOptConst_OutlierParams(Const_OutlierParams value) {Opt = value;}
        public static implicit operator _InOptConst_OutlierParams(Const_OutlierParams value) {return new(value);}
    }

    /// Types of outlier points
    public enum OutlierTypeMask : int
    {
        ///< Small groups of points that are far from the rest
        SmallComponents = 1,
        ///< Points that have too few neighbors within the radius
        WeaklyConnected = 2,
        ///< Points far from the surface approximating the nearest points
        FarSurface = 4,
        ///< Points whose normals differ from the average norm of their nearest neighbors
        AwayNormal = 8,
        All = 15,
    }

    /// A class for searching for outliers of points
    /// @detail The class caches the prepared search results, which allows to speed up the repeat search (while use same radius)
    /// Generated from class `MR::OutliersDetector`.
    /// This is the const half of the class.
    public class Const_OutliersDetector : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_OutliersDetector(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_Destroy", ExactSpelling = true)]
            extern static void __MR_OutliersDetector_Destroy(_Underlying *_this);
            __MR_OutliersDetector_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_OutliersDetector() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_OutliersDetector() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OutliersDetector._Underlying *__MR_OutliersDetector_DefaultConstruct();
            _UnderlyingPtr = __MR_OutliersDetector_DefaultConstruct();
        }

        /// Generated from constructor `MR::OutliersDetector::OutliersDetector`.
        public unsafe Const_OutliersDetector(MR._ByValue_OutliersDetector _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OutliersDetector._Underlying *__MR_OutliersDetector_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OutliersDetector._Underlying *_other);
            _UnderlyingPtr = __MR_OutliersDetector_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Get search parameters
        /// Generated from method `MR::OutliersDetector::getParams`.
        public unsafe MR.Const_OutlierParams GetParams()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_getParams", ExactSpelling = true)]
            extern static MR.Const_OutlierParams._Underlying *__MR_OutliersDetector_getParams(_Underlying *_this);
            return new(__MR_OutliersDetector_getParams(_UnderlyingPtr), is_owning: false);
        }
    }

    /// A class for searching for outliers of points
    /// @detail The class caches the prepared search results, which allows to speed up the repeat search (while use same radius)
    /// Generated from class `MR::OutliersDetector`.
    /// This is the non-const half of the class.
    public class OutliersDetector : Const_OutliersDetector
    {
        internal unsafe OutliersDetector(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe OutliersDetector() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_DefaultConstruct", ExactSpelling = true)]
            extern static MR.OutliersDetector._Underlying *__MR_OutliersDetector_DefaultConstruct();
            _UnderlyingPtr = __MR_OutliersDetector_DefaultConstruct();
        }

        /// Generated from constructor `MR::OutliersDetector::OutliersDetector`.
        public unsafe OutliersDetector(MR._ByValue_OutliersDetector _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.OutliersDetector._Underlying *__MR_OutliersDetector_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.OutliersDetector._Underlying *_other);
            _UnderlyingPtr = __MR_OutliersDetector_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::OutliersDetector::operator=`.
        public unsafe MR.OutliersDetector Assign(MR._ByValue_OutliersDetector _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_AssignFromAnother", ExactSpelling = true)]
            extern static MR.OutliersDetector._Underlying *__MR_OutliersDetector_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.OutliersDetector._Underlying *_other);
            return new(__MR_OutliersDetector_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Make a preliminary stage of outlier search. Caches the result
        /// 
        /// @param pc point cloud
        /// @param radius radius of the search for neighboring points for analysis
        /// @param mask mask of the types of outliers that are looking for
        /// @param progress progress callback function
        /// @return error text or nothing
        /// Generated from method `MR::OutliersDetector::prepare`.
        /// Parameter `progress` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> Prepare(MR.Const_PointCloud pc, float radius, MR.OutlierTypeMask mask, MR.Std._ByValue_Function_BoolFuncFromFloat? progress = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_prepare", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_OutliersDetector_prepare(_Underlying *_this, MR.Const_PointCloud._Underlying *pc, float radius, MR.OutlierTypeMask mask, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_OutliersDetector_prepare(_UnderlyingPtr, pc._UnderlyingPtr, radius, mask, progress is not null ? progress.PassByMode : MR.Misc._PassBy.default_arg, progress is not null && progress.Value is not null ? progress.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Set search parameters
        /// Generated from method `MR::OutliersDetector::setParams`.
        public unsafe void SetParams(MR.Const_OutlierParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_setParams", ExactSpelling = true)]
            extern static void __MR_OutliersDetector_setParams(_Underlying *_this, MR.Const_OutlierParams._Underlying *params_);
            __MR_OutliersDetector_setParams(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// Make an outlier search based on preliminary data
        /// @param mask mask of the types of outliers you are looking for
        /// Generated from method `MR::OutliersDetector::find`.
        /// Parameter `progress` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> Find(MR.OutlierTypeMask mask, MR.Std._ByValue_Function_BoolFuncFromFloat? progress = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_find", ExactSpelling = true)]
            extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_OutliersDetector_find(_Underlying *_this, MR.OutlierTypeMask mask, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_OutliersDetector_find(_UnderlyingPtr, mask, progress is not null ? progress.PassByMode : MR.Misc._PassBy.default_arg, progress is not null && progress.Value is not null ? progress.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Get statistics on the number of neighbors for each point
        /// Generated from method `MR::OutliersDetector::getWeaklyConnectedStat`.
        public unsafe MR.Std.Const_Vector_UnsignedChar GetWeaklyConnectedStat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_OutliersDetector_getWeaklyConnectedStat", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_UnsignedChar._Underlying *__MR_OutliersDetector_getWeaklyConnectedStat(_Underlying *_this);
            return new(__MR_OutliersDetector_getWeaklyConnectedStat(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `OutliersDetector` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `OutliersDetector`/`Const_OutliersDetector` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_OutliersDetector
    {
        internal readonly Const_OutliersDetector? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_OutliersDetector() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_OutliersDetector(Const_OutliersDetector new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_OutliersDetector(Const_OutliersDetector arg) {return new(arg);}
        public _ByValue_OutliersDetector(MR.Misc._Moved<OutliersDetector> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_OutliersDetector(MR.Misc._Moved<OutliersDetector> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `OutliersDetector` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_OutliersDetector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OutliersDetector`/`Const_OutliersDetector` directly.
    public class _InOptMut_OutliersDetector
    {
        public OutliersDetector? Opt;

        public _InOptMut_OutliersDetector() {}
        public _InOptMut_OutliersDetector(OutliersDetector value) {Opt = value;}
        public static implicit operator _InOptMut_OutliersDetector(OutliersDetector value) {return new(value);}
    }

    /// This is used for optional parameters of class `OutliersDetector` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_OutliersDetector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `OutliersDetector`/`Const_OutliersDetector` to pass it to the function.
    public class _InOptConst_OutliersDetector
    {
        public Const_OutliersDetector? Opt;

        public _InOptConst_OutliersDetector() {}
        public _InOptConst_OutliersDetector(Const_OutliersDetector value) {Opt = value;}
        public static implicit operator _InOptConst_OutliersDetector(Const_OutliersDetector value) {return new(value);}
    }

    /// Outlier point search parameters
    /// Generated from class `MR::FindOutliersParams`.
    /// This is the const half of the class.
    public class Const_FindOutliersParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FindOutliersParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_Destroy", ExactSpelling = true)]
            extern static void __MR_FindOutliersParams_Destroy(_Underlying *_this);
            __MR_FindOutliersParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FindOutliersParams() {Dispose(false);}

        ///< Parameters of various criteria for detecting outlier points
        public unsafe MR.Const_OutlierParams FinderParams
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_Get_finderParams", ExactSpelling = true)]
                extern static MR.Const_OutlierParams._Underlying *__MR_FindOutliersParams_Get_finderParams(_Underlying *_this);
                return new(__MR_FindOutliersParams_Get_finderParams(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< Radius of the search for neighboring points for analysis
        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_Get_radius", ExactSpelling = true)]
                extern static float *__MR_FindOutliersParams_Get_radius(_Underlying *_this);
                return *__MR_FindOutliersParams_Get_radius(_UnderlyingPtr);
            }
        }

        ///< Mask of the types of outliers that are looking for
        public unsafe MR.OutlierTypeMask Mask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_Get_mask", ExactSpelling = true)]
                extern static MR.OutlierTypeMask *__MR_FindOutliersParams_Get_mask(_Underlying *_this);
                return *__MR_FindOutliersParams_Get_mask(_UnderlyingPtr);
            }
        }

        ///< Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FindOutliersParams_Get_progress(_Underlying *_this);
                return new(__MR_FindOutliersParams_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FindOutliersParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FindOutliersParams_DefaultConstruct();
        }

        /// Constructs `MR::FindOutliersParams` elementwise.
        public unsafe Const_FindOutliersParams(MR.Const_OutlierParams finderParams, float radius, MR.OutlierTypeMask mask, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_ConstructFrom(MR.OutlierParams._Underlying *finderParams, float radius, MR.OutlierTypeMask mask, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_FindOutliersParams_ConstructFrom(finderParams._UnderlyingPtr, radius, mask, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindOutliersParams::FindOutliersParams`.
        public unsafe Const_FindOutliersParams(MR._ByValue_FindOutliersParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindOutliersParams._Underlying *_other);
            _UnderlyingPtr = __MR_FindOutliersParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Outlier point search parameters
    /// Generated from class `MR::FindOutliersParams`.
    /// This is the non-const half of the class.
    public class FindOutliersParams : Const_FindOutliersParams
    {
        internal unsafe FindOutliersParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< Parameters of various criteria for detecting outlier points
        public new unsafe MR.OutlierParams FinderParams
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_GetMutable_finderParams", ExactSpelling = true)]
                extern static MR.OutlierParams._Underlying *__MR_FindOutliersParams_GetMutable_finderParams(_Underlying *_this);
                return new(__MR_FindOutliersParams_GetMutable_finderParams(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< Radius of the search for neighboring points for analysis
        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_FindOutliersParams_GetMutable_radius(_Underlying *_this);
                return ref *__MR_FindOutliersParams_GetMutable_radius(_UnderlyingPtr);
            }
        }

        ///< Mask of the types of outliers that are looking for
        public new unsafe ref MR.OutlierTypeMask Mask
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_GetMutable_mask", ExactSpelling = true)]
                extern static MR.OutlierTypeMask *__MR_FindOutliersParams_GetMutable_mask(_Underlying *_this);
                return ref *__MR_FindOutliersParams_GetMutable_mask(_UnderlyingPtr);
            }
        }

        ///< Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FindOutliersParams_GetMutable_progress(_Underlying *_this);
                return new(__MR_FindOutliersParams_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FindOutliersParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FindOutliersParams_DefaultConstruct();
        }

        /// Constructs `MR::FindOutliersParams` elementwise.
        public unsafe FindOutliersParams(MR.Const_OutlierParams finderParams, float radius, MR.OutlierTypeMask mask, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_ConstructFrom(MR.OutlierParams._Underlying *finderParams, float radius, MR.OutlierTypeMask mask, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_FindOutliersParams_ConstructFrom(finderParams._UnderlyingPtr, radius, mask, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindOutliersParams::FindOutliersParams`.
        public unsafe FindOutliersParams(MR._ByValue_FindOutliersParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindOutliersParams._Underlying *_other);
            _UnderlyingPtr = __MR_FindOutliersParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FindOutliersParams::operator=`.
        public unsafe MR.FindOutliersParams Assign(MR._ByValue_FindOutliersParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOutliersParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FindOutliersParams._Underlying *__MR_FindOutliersParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FindOutliersParams._Underlying *_other);
            return new(__MR_FindOutliersParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FindOutliersParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FindOutliersParams`/`Const_FindOutliersParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FindOutliersParams
    {
        internal readonly Const_FindOutliersParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FindOutliersParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FindOutliersParams(Const_FindOutliersParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FindOutliersParams(Const_FindOutliersParams arg) {return new(arg);}
        public _ByValue_FindOutliersParams(MR.Misc._Moved<FindOutliersParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FindOutliersParams(MR.Misc._Moved<FindOutliersParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FindOutliersParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FindOutliersParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindOutliersParams`/`Const_FindOutliersParams` directly.
    public class _InOptMut_FindOutliersParams
    {
        public FindOutliersParams? Opt;

        public _InOptMut_FindOutliersParams() {}
        public _InOptMut_FindOutliersParams(FindOutliersParams value) {Opt = value;}
        public static implicit operator _InOptMut_FindOutliersParams(FindOutliersParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `FindOutliersParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FindOutliersParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindOutliersParams`/`Const_FindOutliersParams` to pass it to the function.
    public class _InOptConst_FindOutliersParams
    {
        public Const_FindOutliersParams? Opt;

        public _InOptConst_FindOutliersParams() {}
        public _InOptConst_FindOutliersParams(Const_FindOutliersParams value) {Opt = value;}
        public static implicit operator _InOptConst_FindOutliersParams(Const_FindOutliersParams value) {return new(value);}
    }

    /// Generated from function `MR::operator&`.
    public static MR.OutlierTypeMask Bitand(MR.OutlierTypeMask a, MR.OutlierTypeMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_OutlierTypeMask", ExactSpelling = true)]
        extern static MR.OutlierTypeMask __MR_bitand_MR_OutlierTypeMask(MR.OutlierTypeMask a, MR.OutlierTypeMask b);
        return __MR_bitand_MR_OutlierTypeMask(a, b);
    }

    /// Generated from function `MR::operator|`.
    public static MR.OutlierTypeMask Bitor(MR.OutlierTypeMask a, MR.OutlierTypeMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_OutlierTypeMask", ExactSpelling = true)]
        extern static MR.OutlierTypeMask __MR_bitor_MR_OutlierTypeMask(MR.OutlierTypeMask a, MR.OutlierTypeMask b);
        return __MR_bitor_MR_OutlierTypeMask(a, b);
    }

    /// Generated from function `MR::operator~`.
    public static MR.OutlierTypeMask Compl(MR.OutlierTypeMask a)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compl_MR_OutlierTypeMask", ExactSpelling = true)]
        extern static MR.OutlierTypeMask __MR_compl_MR_OutlierTypeMask(MR.OutlierTypeMask a);
        return __MR_compl_MR_OutlierTypeMask(a);
    }

    /// Generated from function `MR::operator&=`.
    public static unsafe ref MR.OutlierTypeMask BitandAssign(ref MR.OutlierTypeMask a, MR.OutlierTypeMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_assign_MR_OutlierTypeMask", ExactSpelling = true)]
        extern static MR.OutlierTypeMask *__MR_bitand_assign_MR_OutlierTypeMask(MR.OutlierTypeMask *a, MR.OutlierTypeMask b);
        fixed (MR.OutlierTypeMask *__ptr_a = &a)
        {
            return ref *__MR_bitand_assign_MR_OutlierTypeMask(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator|=`.
    public static unsafe ref MR.OutlierTypeMask BitorAssign(ref MR.OutlierTypeMask a, MR.OutlierTypeMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_assign_MR_OutlierTypeMask", ExactSpelling = true)]
        extern static MR.OutlierTypeMask *__MR_bitor_assign_MR_OutlierTypeMask(MR.OutlierTypeMask *a, MR.OutlierTypeMask b);
        fixed (MR.OutlierTypeMask *__ptr_a = &a)
        {
            return ref *__MR_bitor_assign_MR_OutlierTypeMask(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator*`.
    public static MR.OutlierTypeMask Mul(MR.OutlierTypeMask a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_OutlierTypeMask_bool", ExactSpelling = true)]
        extern static MR.OutlierTypeMask __MR_mul_MR_OutlierTypeMask_bool(MR.OutlierTypeMask a, byte b);
        return __MR_mul_MR_OutlierTypeMask_bool(a, b ? (byte)1 : (byte)0);
    }

    /// Generated from function `MR::operator*`.
    public static MR.OutlierTypeMask Mul(bool a, MR.OutlierTypeMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_OutlierTypeMask", ExactSpelling = true)]
        extern static MR.OutlierTypeMask __MR_mul_bool_MR_OutlierTypeMask(byte a, MR.OutlierTypeMask b);
        return __MR_mul_bool_MR_OutlierTypeMask(a ? (byte)1 : (byte)0, b);
    }

    /// Generated from function `MR::operator*=`.
    public static unsafe ref MR.OutlierTypeMask MulAssign(ref MR.OutlierTypeMask a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_OutlierTypeMask_bool", ExactSpelling = true)]
        extern static MR.OutlierTypeMask *__MR_mul_assign_MR_OutlierTypeMask_bool(MR.OutlierTypeMask *a, byte b);
        fixed (MR.OutlierTypeMask *__ptr_a = &a)
        {
            return ref *__MR_mul_assign_MR_OutlierTypeMask_bool(__ptr_a, b ? (byte)1 : (byte)0);
        }
    }

    /// Finding outlier points
    /// Generated from function `MR::findOutliers`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> FindOutliers(MR.Const_PointCloud pc, MR.Const_FindOutliersParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findOutliers", ExactSpelling = true)]
        extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_findOutliers(MR.Const_PointCloud._Underlying *pc, MR.Const_FindOutliersParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_findOutliers(pc._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
