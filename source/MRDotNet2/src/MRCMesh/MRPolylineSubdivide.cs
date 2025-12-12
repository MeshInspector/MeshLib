public static partial class MR
{
    /// Generated from class `MR::PolylineSubdivideSettings`.
    /// This is the const half of the class.
    public class Const_PolylineSubdivideSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineSubdivideSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineSubdivideSettings_Destroy(_Underlying *_this);
            __MR_PolylineSubdivideSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineSubdivideSettings() {Dispose(false);}

        /// Subdivision is stopped when all edges are not longer than this value
        public unsafe float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_PolylineSubdivideSettings_Get_maxEdgeLen(_Underlying *_this);
                return *__MR_PolylineSubdivideSettings_Get_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed
        public unsafe int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_PolylineSubdivideSettings_Get_maxEdgeSplits(_Underlying *_this);
                return *__MR_PolylineSubdivideSettings_Get_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Region on polyline to be subdivided: both edge vertices must be there to allow spitting,
        /// it is updated during the operation
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_PolylineSubdivideSettings_Get_region(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// New vertices appeared during subdivision will be added here
        public unsafe ref void * NewVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_newVerts", ExactSpelling = true)]
                extern static void **__MR_PolylineSubdivideSettings_Get_newVerts(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_Get_newVerts(_UnderlyingPtr);
            }
        }

        /// This option works best for natural lines, where all segments have similar size,
        /// and no sharp angles in between
        public unsafe bool UseCurvature
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_useCurvature", ExactSpelling = true)]
                extern static bool *__MR_PolylineSubdivideSettings_Get_useCurvature(_Underlying *_this);
                return *__MR_PolylineSubdivideSettings_Get_useCurvature(_UnderlyingPtr);
            }
        }

        /// this function is called each time a new vertex has been created
        public unsafe MR.Std.Const_Function_VoidFuncFromMRVertId OnVertCreated
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_onVertCreated", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRVertId._Underlying *__MR_PolylineSubdivideSettings_Get_onVertCreated(_Underlying *_this);
                return new(__MR_PolylineSubdivideSettings_Get_onVertCreated(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time edge (e) is split into (e1->e)
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_PolylineSubdivideSettings_Get_onEdgeSplit(_Underlying *_this);
                return new(__MR_PolylineSubdivideSettings_Get_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_Get_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PolylineSubdivideSettings_Get_progressCallback(_Underlying *_this);
                return new(__MR_PolylineSubdivideSettings_Get_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineSubdivideSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineSubdivideSettings_DefaultConstruct();
        }

        /// Constructs `MR::PolylineSubdivideSettings` elementwise.
        public unsafe Const_PolylineSubdivideSettings(float maxEdgeLen, int maxEdgeSplits, MR.VertBitSet? region, MR.VertBitSet? newVerts, bool useCurvature, MR.Std._ByValue_Function_VoidFuncFromMRVertId onVertCreated, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_ConstructFrom(float maxEdgeLen, int maxEdgeSplits, MR.VertBitSet._Underlying *region, MR.VertBitSet._Underlying *newVerts, byte useCurvature, MR.Misc._PassBy onVertCreated_pass_by, MR.Std.Function_VoidFuncFromMRVertId._Underlying *onVertCreated, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
            _UnderlyingPtr = __MR_PolylineSubdivideSettings_ConstructFrom(maxEdgeLen, maxEdgeSplits, region is not null ? region._UnderlyingPtr : null, newVerts is not null ? newVerts._UnderlyingPtr : null, useCurvature ? (byte)1 : (byte)0, onVertCreated.PassByMode, onVertCreated.Value is not null ? onVertCreated.Value._UnderlyingPtr : null, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PolylineSubdivideSettings::PolylineSubdivideSettings`.
        public unsafe Const_PolylineSubdivideSettings(MR._ByValue_PolylineSubdivideSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineSubdivideSettings._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineSubdivideSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::PolylineSubdivideSettings`.
    /// This is the non-const half of the class.
    public class PolylineSubdivideSettings : Const_PolylineSubdivideSettings
    {
        internal unsafe PolylineSubdivideSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Subdivision is stopped when all edges are not longer than this value
        public new unsafe ref float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_PolylineSubdivideSettings_GetMutable_maxEdgeLen(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_GetMutable_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed
        public new unsafe ref int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_PolylineSubdivideSettings_GetMutable_maxEdgeSplits(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_GetMutable_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Region on polyline to be subdivided: both edge vertices must be there to allow spitting,
        /// it is updated during the operation
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_PolylineSubdivideSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// New vertices appeared during subdivision will be added here
        public new unsafe ref void * NewVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_newVerts", ExactSpelling = true)]
                extern static void **__MR_PolylineSubdivideSettings_GetMutable_newVerts(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_GetMutable_newVerts(_UnderlyingPtr);
            }
        }

        /// This option works best for natural lines, where all segments have similar size,
        /// and no sharp angles in between
        public new unsafe ref bool UseCurvature
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_useCurvature", ExactSpelling = true)]
                extern static bool *__MR_PolylineSubdivideSettings_GetMutable_useCurvature(_Underlying *_this);
                return ref *__MR_PolylineSubdivideSettings_GetMutable_useCurvature(_UnderlyingPtr);
            }
        }

        /// this function is called each time a new vertex has been created
        public new unsafe MR.Std.Function_VoidFuncFromMRVertId OnVertCreated
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_onVertCreated", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRVertId._Underlying *__MR_PolylineSubdivideSettings_GetMutable_onVertCreated(_Underlying *_this);
                return new(__MR_PolylineSubdivideSettings_GetMutable_onVertCreated(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time edge (e) is split into (e1->e)
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_PolylineSubdivideSettings_GetMutable_onEdgeSplit(_Underlying *_this);
                return new(__MR_PolylineSubdivideSettings_GetMutable_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_GetMutable_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PolylineSubdivideSettings_GetMutable_progressCallback(_Underlying *_this);
                return new(__MR_PolylineSubdivideSettings_GetMutable_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineSubdivideSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineSubdivideSettings_DefaultConstruct();
        }

        /// Constructs `MR::PolylineSubdivideSettings` elementwise.
        public unsafe PolylineSubdivideSettings(float maxEdgeLen, int maxEdgeSplits, MR.VertBitSet? region, MR.VertBitSet? newVerts, bool useCurvature, MR.Std._ByValue_Function_VoidFuncFromMRVertId onVertCreated, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_ConstructFrom(float maxEdgeLen, int maxEdgeSplits, MR.VertBitSet._Underlying *region, MR.VertBitSet._Underlying *newVerts, byte useCurvature, MR.Misc._PassBy onVertCreated_pass_by, MR.Std.Function_VoidFuncFromMRVertId._Underlying *onVertCreated, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
            _UnderlyingPtr = __MR_PolylineSubdivideSettings_ConstructFrom(maxEdgeLen, maxEdgeSplits, region is not null ? region._UnderlyingPtr : null, newVerts is not null ? newVerts._UnderlyingPtr : null, useCurvature ? (byte)1 : (byte)0, onVertCreated.PassByMode, onVertCreated.Value is not null ? onVertCreated.Value._UnderlyingPtr : null, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PolylineSubdivideSettings::PolylineSubdivideSettings`.
        public unsafe PolylineSubdivideSettings(MR._ByValue_PolylineSubdivideSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineSubdivideSettings._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineSubdivideSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolylineSubdivideSettings::operator=`.
        public unsafe MR.PolylineSubdivideSettings Assign(MR._ByValue_PolylineSubdivideSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineSubdivideSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineSubdivideSettings._Underlying *__MR_PolylineSubdivideSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PolylineSubdivideSettings._Underlying *_other);
            return new(__MR_PolylineSubdivideSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PolylineSubdivideSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PolylineSubdivideSettings`/`Const_PolylineSubdivideSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PolylineSubdivideSettings
    {
        internal readonly Const_PolylineSubdivideSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PolylineSubdivideSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PolylineSubdivideSettings(Const_PolylineSubdivideSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PolylineSubdivideSettings(Const_PolylineSubdivideSettings arg) {return new(arg);}
        public _ByValue_PolylineSubdivideSettings(MR.Misc._Moved<PolylineSubdivideSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PolylineSubdivideSettings(MR.Misc._Moved<PolylineSubdivideSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PolylineSubdivideSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineSubdivideSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineSubdivideSettings`/`Const_PolylineSubdivideSettings` directly.
    public class _InOptMut_PolylineSubdivideSettings
    {
        public PolylineSubdivideSettings? Opt;

        public _InOptMut_PolylineSubdivideSettings() {}
        public _InOptMut_PolylineSubdivideSettings(PolylineSubdivideSettings value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineSubdivideSettings(PolylineSubdivideSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineSubdivideSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineSubdivideSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineSubdivideSettings`/`Const_PolylineSubdivideSettings` to pass it to the function.
    public class _InOptConst_PolylineSubdivideSettings
    {
        public Const_PolylineSubdivideSettings? Opt;

        public _InOptConst_PolylineSubdivideSettings() {}
        public _InOptConst_PolylineSubdivideSettings(Const_PolylineSubdivideSettings value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineSubdivideSettings(Const_PolylineSubdivideSettings value) {return new(value);}
    }

    /// Split edges in polyline according to the settings;\n
    /// \return The total number of edge splits performed
    /// Generated from function `MR::subdividePolyline`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe int SubdividePolyline(MR.Polyline2 polyline, MR.Const_PolylineSubdivideSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdividePolyline_MR_Polyline2", ExactSpelling = true)]
        extern static int __MR_subdividePolyline_MR_Polyline2(MR.Polyline2._Underlying *polyline, MR.Const_PolylineSubdivideSettings._Underlying *settings);
        return __MR_subdividePolyline_MR_Polyline2(polyline._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null);
    }

    /// Generated from function `MR::subdividePolyline`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe int SubdividePolyline(MR.Polyline3 polyline, MR.Const_PolylineSubdivideSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdividePolyline_MR_Polyline3", ExactSpelling = true)]
        extern static int __MR_subdividePolyline_MR_Polyline3(MR.Polyline3._Underlying *polyline, MR.Const_PolylineSubdivideSettings._Underlying *settings);
        return __MR_subdividePolyline_MR_Polyline3(polyline._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null);
    }
}
