public static partial class MR
{
    /// Generated from class `MR::UniformSamplingSettings`.
    /// This is the const half of the class.
    public class Const_UniformSamplingSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UniformSamplingSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_UniformSamplingSettings_Destroy(_Underlying *_this);
            __MR_UniformSamplingSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UniformSamplingSettings() {Dispose(false);}

        /// minimal distance between samples
        public unsafe float Distance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_Get_distance", ExactSpelling = true)]
                extern static float *__MR_UniformSamplingSettings_Get_distance(_Underlying *_this);
                return *__MR_UniformSamplingSettings_Get_distance(_UnderlyingPtr);
            }
        }

        /// if point cloud has normals then automatically decreases local distance to make sure that all points inside have absolute normal dot product not less than this value;
        /// this is to make sampling denser in the regions of high curvature;
        /// value <=0 means ignore normals;
        /// value >=1 means select all points (practically useless)
        public unsafe float MinNormalDot
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_Get_minNormalDot", ExactSpelling = true)]
                extern static float *__MR_UniformSamplingSettings_Get_minNormalDot(_Underlying *_this);
                return *__MR_UniformSamplingSettings_Get_minNormalDot(_UnderlyingPtr);
            }
        }

        /// if true process the points in lexicographical order, which gives tighter and more uniform samples;
        /// if false process the points according to their ids, which is faster
        public unsafe bool LexicographicalOrder
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_Get_lexicographicalOrder", ExactSpelling = true)]
                extern static bool *__MR_UniformSamplingSettings_Get_lexicographicalOrder(_Underlying *_this);
                return *__MR_UniformSamplingSettings_Get_lexicographicalOrder(_UnderlyingPtr);
            }
        }

        /// if not nullptr then these normals will be used during sampling instead of normals in the cloud itself
        public unsafe ref readonly void * PNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_Get_pNormals", ExactSpelling = true)]
                extern static void **__MR_UniformSamplingSettings_Get_pNormals(_Underlying *_this);
                return ref *__MR_UniformSamplingSettings_Get_pNormals(_UnderlyingPtr);
            }
        }

        /// to report progress and cancel processing
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_UniformSamplingSettings_Get_progress(_Underlying *_this);
                return new(__MR_UniformSamplingSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UniformSamplingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_UniformSamplingSettings_DefaultConstruct();
        }

        /// Constructs `MR::UniformSamplingSettings` elementwise.
        public unsafe Const_UniformSamplingSettings(float distance, float minNormalDot, bool lexicographicalOrder, MR.Const_VertCoords? pNormals, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_ConstructFrom(float distance, float minNormalDot, byte lexicographicalOrder, MR.Const_VertCoords._Underlying *pNormals, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_UniformSamplingSettings_ConstructFrom(distance, minNormalDot, lexicographicalOrder ? (byte)1 : (byte)0, pNormals is not null ? pNormals._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::UniformSamplingSettings::UniformSamplingSettings`.
        public unsafe Const_UniformSamplingSettings(MR._ByValue_UniformSamplingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniformSamplingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_UniformSamplingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::UniformSamplingSettings`.
    /// This is the non-const half of the class.
    public class UniformSamplingSettings : Const_UniformSamplingSettings
    {
        internal unsafe UniformSamplingSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// minimal distance between samples
        public new unsafe ref float Distance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_GetMutable_distance", ExactSpelling = true)]
                extern static float *__MR_UniformSamplingSettings_GetMutable_distance(_Underlying *_this);
                return ref *__MR_UniformSamplingSettings_GetMutable_distance(_UnderlyingPtr);
            }
        }

        /// if point cloud has normals then automatically decreases local distance to make sure that all points inside have absolute normal dot product not less than this value;
        /// this is to make sampling denser in the regions of high curvature;
        /// value <=0 means ignore normals;
        /// value >=1 means select all points (practically useless)
        public new unsafe ref float MinNormalDot
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_GetMutable_minNormalDot", ExactSpelling = true)]
                extern static float *__MR_UniformSamplingSettings_GetMutable_minNormalDot(_Underlying *_this);
                return ref *__MR_UniformSamplingSettings_GetMutable_minNormalDot(_UnderlyingPtr);
            }
        }

        /// if true process the points in lexicographical order, which gives tighter and more uniform samples;
        /// if false process the points according to their ids, which is faster
        public new unsafe ref bool LexicographicalOrder
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_GetMutable_lexicographicalOrder", ExactSpelling = true)]
                extern static bool *__MR_UniformSamplingSettings_GetMutable_lexicographicalOrder(_Underlying *_this);
                return ref *__MR_UniformSamplingSettings_GetMutable_lexicographicalOrder(_UnderlyingPtr);
            }
        }

        /// if not nullptr then these normals will be used during sampling instead of normals in the cloud itself
        public new unsafe ref readonly void * PNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_GetMutable_pNormals", ExactSpelling = true)]
                extern static void **__MR_UniformSamplingSettings_GetMutable_pNormals(_Underlying *_this);
                return ref *__MR_UniformSamplingSettings_GetMutable_pNormals(_UnderlyingPtr);
            }
        }

        /// to report progress and cancel processing
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_UniformSamplingSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_UniformSamplingSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe UniformSamplingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_UniformSamplingSettings_DefaultConstruct();
        }

        /// Constructs `MR::UniformSamplingSettings` elementwise.
        public unsafe UniformSamplingSettings(float distance, float minNormalDot, bool lexicographicalOrder, MR.Const_VertCoords? pNormals, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_ConstructFrom(float distance, float minNormalDot, byte lexicographicalOrder, MR.Const_VertCoords._Underlying *pNormals, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_UniformSamplingSettings_ConstructFrom(distance, minNormalDot, lexicographicalOrder ? (byte)1 : (byte)0, pNormals is not null ? pNormals._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::UniformSamplingSettings::UniformSamplingSettings`.
        public unsafe UniformSamplingSettings(MR._ByValue_UniformSamplingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniformSamplingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_UniformSamplingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::UniformSamplingSettings::operator=`.
        public unsafe MR.UniformSamplingSettings Assign(MR._ByValue_UniformSamplingSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniformSamplingSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UniformSamplingSettings._Underlying *__MR_UniformSamplingSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UniformSamplingSettings._Underlying *_other);
            return new(__MR_UniformSamplingSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UniformSamplingSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UniformSamplingSettings`/`Const_UniformSamplingSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UniformSamplingSettings
    {
        internal readonly Const_UniformSamplingSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UniformSamplingSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UniformSamplingSettings(Const_UniformSamplingSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UniformSamplingSettings(Const_UniformSamplingSettings arg) {return new(arg);}
        public _ByValue_UniformSamplingSettings(MR.Misc._Moved<UniformSamplingSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UniformSamplingSettings(MR.Misc._Moved<UniformSamplingSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UniformSamplingSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UniformSamplingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniformSamplingSettings`/`Const_UniformSamplingSettings` directly.
    public class _InOptMut_UniformSamplingSettings
    {
        public UniformSamplingSettings? Opt;

        public _InOptMut_UniformSamplingSettings() {}
        public _InOptMut_UniformSamplingSettings(UniformSamplingSettings value) {Opt = value;}
        public static implicit operator _InOptMut_UniformSamplingSettings(UniformSamplingSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `UniformSamplingSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UniformSamplingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniformSamplingSettings`/`Const_UniformSamplingSettings` to pass it to the function.
    public class _InOptConst_UniformSamplingSettings
    {
        public Const_UniformSamplingSettings? Opt;

        public _InOptConst_UniformSamplingSettings() {}
        public _InOptConst_UniformSamplingSettings(Const_UniformSamplingSettings value) {Opt = value;}
        public static implicit operator _InOptConst_UniformSamplingSettings(Const_UniformSamplingSettings value) {return new(value);}
    }

    /// Sample vertices, removing ones that are too close;
    /// returns std::nullopt if it was terminated by the callback
    /// Generated from function `MR::pointUniformSampling`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> PointUniformSampling(MR.Const_PointCloud pointCloud, MR.Const_UniformSamplingSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointUniformSampling", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_pointUniformSampling(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_UniformSamplingSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_pointUniformSampling(pointCloud._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }

    /// Composes new point cloud consisting of uniform samples of original point cloud;
    /// returns std::nullopt if it was terminated by the callback
    /// Generated from function `MR::makeUniformSampledCloud`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRPointCloud> MakeUniformSampledCloud(MR.Const_PointCloud pointCloud, MR.Const_UniformSamplingSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeUniformSampledCloud", ExactSpelling = true)]
        extern static MR.Std.Optional_MRPointCloud._Underlying *__MR_makeUniformSampledCloud(MR.Const_PointCloud._Underlying *pointCloud, MR.Const_UniformSamplingSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Std.Optional_MRPointCloud(__MR_makeUniformSampledCloud(pointCloud._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
