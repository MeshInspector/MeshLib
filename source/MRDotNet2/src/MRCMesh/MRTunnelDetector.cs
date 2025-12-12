public static partial class MR
{
    /// Generated from class `MR::DetectTunnelSettings`.
    /// This is the const half of the class.
    public class Const_DetectTunnelSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DetectTunnelSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DetectTunnelSettings_Destroy(_Underlying *_this);
            __MR_DetectTunnelSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DetectTunnelSettings() {Dispose(false);}

        /// maximal length of tunnel loops to consider
        public unsafe float MaxTunnelLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_Get_maxTunnelLength", ExactSpelling = true)]
                extern static float *__MR_DetectTunnelSettings_Get_maxTunnelLength(_Underlying *_this);
                return *__MR_DetectTunnelSettings_Get_maxTunnelLength(_UnderlyingPtr);
            }
        }

        /// maximal number of iterations to detect all tunnels;
        /// on a big mesh with many tunnels even one iteration can take a while
        public unsafe int MaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_Get_maxIters", ExactSpelling = true)]
                extern static int *__MR_DetectTunnelSettings_Get_maxIters(_Underlying *_this);
                return *__MR_DetectTunnelSettings_Get_maxIters(_UnderlyingPtr);
            }
        }

        /// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
        public unsafe MR.Std.Const_Function_FloatFuncFromMREdgeId Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_Get_metric", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *__MR_DetectTunnelSettings_Get_metric(_Underlying *_this);
                return new(__MR_DetectTunnelSettings_Get_metric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// to report algorithm progress and cancel from outside
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DetectTunnelSettings_Get_progress(_Underlying *_this);
                return new(__MR_DetectTunnelSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DetectTunnelSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DetectTunnelSettings_DefaultConstruct();
        }

        /// Constructs `MR::DetectTunnelSettings` elementwise.
        public unsafe Const_DetectTunnelSettings(float maxTunnelLength, int maxIters, MR.Std._ByValue_Function_FloatFuncFromMREdgeId metric, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_ConstructFrom(float maxTunnelLength, int maxIters, MR.Misc._PassBy metric_pass_by, MR.Std.Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DetectTunnelSettings_ConstructFrom(maxTunnelLength, maxIters, metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DetectTunnelSettings::DetectTunnelSettings`.
        public unsafe Const_DetectTunnelSettings(MR._ByValue_DetectTunnelSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DetectTunnelSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DetectTunnelSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DetectTunnelSettings`.
    /// This is the non-const half of the class.
    public class DetectTunnelSettings : Const_DetectTunnelSettings
    {
        internal unsafe DetectTunnelSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// maximal length of tunnel loops to consider
        public new unsafe ref float MaxTunnelLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_GetMutable_maxTunnelLength", ExactSpelling = true)]
                extern static float *__MR_DetectTunnelSettings_GetMutable_maxTunnelLength(_Underlying *_this);
                return ref *__MR_DetectTunnelSettings_GetMutable_maxTunnelLength(_UnderlyingPtr);
            }
        }

        /// maximal number of iterations to detect all tunnels;
        /// on a big mesh with many tunnels even one iteration can take a while
        public new unsafe ref int MaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_GetMutable_maxIters", ExactSpelling = true)]
                extern static int *__MR_DetectTunnelSettings_GetMutable_maxIters(_Underlying *_this);
                return ref *__MR_DetectTunnelSettings_GetMutable_maxIters(_UnderlyingPtr);
            }
        }

        /// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
        public new unsafe MR.Std.Function_FloatFuncFromMREdgeId Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_GetMutable_metric", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromMREdgeId._Underlying *__MR_DetectTunnelSettings_GetMutable_metric(_Underlying *_this);
                return new(__MR_DetectTunnelSettings_GetMutable_metric(_UnderlyingPtr), is_owning: false);
            }
        }

        /// to report algorithm progress and cancel from outside
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DetectTunnelSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_DetectTunnelSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DetectTunnelSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DetectTunnelSettings_DefaultConstruct();
        }

        /// Constructs `MR::DetectTunnelSettings` elementwise.
        public unsafe DetectTunnelSettings(float maxTunnelLength, int maxIters, MR.Std._ByValue_Function_FloatFuncFromMREdgeId metric, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_ConstructFrom(float maxTunnelLength, int maxIters, MR.Misc._PassBy metric_pass_by, MR.Std.Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DetectTunnelSettings_ConstructFrom(maxTunnelLength, maxIters, metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DetectTunnelSettings::DetectTunnelSettings`.
        public unsafe DetectTunnelSettings(MR._ByValue_DetectTunnelSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DetectTunnelSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DetectTunnelSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DetectTunnelSettings::operator=`.
        public unsafe MR.DetectTunnelSettings Assign(MR._ByValue_DetectTunnelSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DetectTunnelSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DetectTunnelSettings._Underlying *__MR_DetectTunnelSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DetectTunnelSettings._Underlying *_other);
            return new(__MR_DetectTunnelSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DetectTunnelSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DetectTunnelSettings`/`Const_DetectTunnelSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DetectTunnelSettings
    {
        internal readonly Const_DetectTunnelSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DetectTunnelSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DetectTunnelSettings(Const_DetectTunnelSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DetectTunnelSettings(Const_DetectTunnelSettings arg) {return new(arg);}
        public _ByValue_DetectTunnelSettings(MR.Misc._Moved<DetectTunnelSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DetectTunnelSettings(MR.Misc._Moved<DetectTunnelSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DetectTunnelSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DetectTunnelSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DetectTunnelSettings`/`Const_DetectTunnelSettings` directly.
    public class _InOptMut_DetectTunnelSettings
    {
        public DetectTunnelSettings? Opt;

        public _InOptMut_DetectTunnelSettings() {}
        public _InOptMut_DetectTunnelSettings(DetectTunnelSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DetectTunnelSettings(DetectTunnelSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DetectTunnelSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DetectTunnelSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DetectTunnelSettings`/`Const_DetectTunnelSettings` to pass it to the function.
    public class _InOptConst_DetectTunnelSettings
    {
        public Const_DetectTunnelSettings? Opt;

        public _InOptConst_DetectTunnelSettings() {}
        public _InOptConst_DetectTunnelSettings(Const_DetectTunnelSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DetectTunnelSettings(Const_DetectTunnelSettings value) {return new(value);}
    }

    /// detects all not-contractible-in-point and not-equivalent tunnel loops on the mesh;
    /// trying to include in the loops the edges with the smallest metric;
    /// if no metric is given then discreteMinusAbsMeanCurvatureMetric will be used
    /// Generated from function `MR::detectBasisTunnels`.
    /// Parameter `metric` defaults to `{}`.
    /// Parameter `progressCallback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdVectorMREdgeId_StdString> DetectBasisTunnels(MR.Const_MeshPart mp, MR.Std._ByValue_Function_FloatFuncFromMREdgeId? metric = null, MR.Std._ByValue_Function_BoolFuncFromFloat? progressCallback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_detectBasisTunnels", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdVectorMREdgeId_StdString._Underlying *__MR_detectBasisTunnels(MR.Const_MeshPart._Underlying *mp, MR.Misc._PassBy metric_pass_by, MR.Std.Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
        return MR.Misc.Move(new MR.Expected_StdVectorStdVectorMREdgeId_StdString(__MR_detectBasisTunnels(mp._UnderlyingPtr, metric is not null ? metric.PassByMode : MR.Misc._PassBy.default_arg, metric is not null && metric.Value is not null ? metric.Value._UnderlyingPtr : null, progressCallback is not null ? progressCallback.PassByMode : MR.Misc._PassBy.default_arg, progressCallback is not null && progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// returns tunnels as a number of faces;
    /// if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere
    /// Generated from function `MR::detectTunnelFaces`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> DetectTunnelFaces(MR.Const_MeshPart mp, MR.Const_DetectTunnelSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_detectTunnelFaces", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_detectTunnelFaces(MR.Const_MeshPart._Underlying *mp, MR.Const_DetectTunnelSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_detectTunnelFaces(mp._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }
}
