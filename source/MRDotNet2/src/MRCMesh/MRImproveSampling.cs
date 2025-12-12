public static partial class MR
{
    /// Generated from class `MR::ImproveSamplingSettings`.
    /// This is the const half of the class.
    public class Const_ImproveSamplingSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ImproveSamplingSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_ImproveSamplingSettings_Destroy(_Underlying *_this);
            __MR_ImproveSamplingSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ImproveSamplingSettings() {Dispose(false);}

        /// the number of algorithm iterations to perform
        public unsafe int NumIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_numIters", ExactSpelling = true)]
                extern static int *__MR_ImproveSamplingSettings_Get_numIters(_Underlying *_this);
                return *__MR_ImproveSamplingSettings_Get_numIters(_UnderlyingPtr);
            }
        }

        /// if a sample represents less than this number of input points then such sample will be discarded;
        /// it can be used to remove outliers
        public unsafe int MinPointsInSample
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_minPointsInSample", ExactSpelling = true)]
                extern static int *__MR_ImproveSamplingSettings_Get_minPointsInSample(_Underlying *_this);
                return *__MR_ImproveSamplingSettings_Get_minPointsInSample(_UnderlyingPtr);
            }
        }

        /// optional output: mapping from input point id to sample id
        public unsafe ref void * Pt2sm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_pt2sm", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_Get_pt2sm(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_Get_pt2sm(_UnderlyingPtr);
            }
        }

        /// optional output: new cloud containing averaged points and normals for each sample
        public unsafe ref void * CloudOfSamples
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_cloudOfSamples", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_Get_cloudOfSamples(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_Get_cloudOfSamples(_UnderlyingPtr);
            }
        }

        /// optional output: the number of points in each sample
        public unsafe ref void * PtsInSm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_ptsInSm", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_Get_ptsInSm(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_Get_ptsInSm(_UnderlyingPtr);
            }
        }

        /// optional input: colors of input points
        public unsafe ref readonly void * PtColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_ptColors", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_Get_ptColors(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_Get_ptColors(_UnderlyingPtr);
            }
        }

        /// optional output: averaged colors of samples
        public unsafe ref void * SmColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_smColors", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_Get_smColors(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_Get_smColors(_UnderlyingPtr);
            }
        }

        /// output progress status and receive cancel signal
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ImproveSamplingSettings_Get_progress(_Underlying *_this);
                return new(__MR_ImproveSamplingSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ImproveSamplingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_ImproveSamplingSettings_DefaultConstruct();
        }

        /// Constructs `MR::ImproveSamplingSettings` elementwise.
        public unsafe Const_ImproveSamplingSettings(int numIters, int minPointsInSample, MR.VertMap? pt2sm, MR.PointCloud? cloudOfSamples, MR.Vector_Int_MRVertId? ptsInSm, MR.Const_VertColors? ptColors, MR.VertColors? smColors, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_ConstructFrom(int numIters, int minPointsInSample, MR.VertMap._Underlying *pt2sm, MR.PointCloud._Underlying *cloudOfSamples, MR.Vector_Int_MRVertId._Underlying *ptsInSm, MR.Const_VertColors._Underlying *ptColors, MR.VertColors._Underlying *smColors, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_ImproveSamplingSettings_ConstructFrom(numIters, minPointsInSample, pt2sm is not null ? pt2sm._UnderlyingPtr : null, cloudOfSamples is not null ? cloudOfSamples._UnderlyingPtr : null, ptsInSm is not null ? ptsInSm._UnderlyingPtr : null, ptColors is not null ? ptColors._UnderlyingPtr : null, smColors is not null ? smColors._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ImproveSamplingSettings::ImproveSamplingSettings`.
        public unsafe Const_ImproveSamplingSettings(MR._ByValue_ImproveSamplingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ImproveSamplingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_ImproveSamplingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ImproveSamplingSettings`.
    /// This is the non-const half of the class.
    public class ImproveSamplingSettings : Const_ImproveSamplingSettings
    {
        internal unsafe ImproveSamplingSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the number of algorithm iterations to perform
        public new unsafe ref int NumIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_numIters", ExactSpelling = true)]
                extern static int *__MR_ImproveSamplingSettings_GetMutable_numIters(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_numIters(_UnderlyingPtr);
            }
        }

        /// if a sample represents less than this number of input points then such sample will be discarded;
        /// it can be used to remove outliers
        public new unsafe ref int MinPointsInSample
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_minPointsInSample", ExactSpelling = true)]
                extern static int *__MR_ImproveSamplingSettings_GetMutable_minPointsInSample(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_minPointsInSample(_UnderlyingPtr);
            }
        }

        /// optional output: mapping from input point id to sample id
        public new unsafe ref void * Pt2sm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_pt2sm", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_GetMutable_pt2sm(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_pt2sm(_UnderlyingPtr);
            }
        }

        /// optional output: new cloud containing averaged points and normals for each sample
        public new unsafe ref void * CloudOfSamples
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_cloudOfSamples", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_GetMutable_cloudOfSamples(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_cloudOfSamples(_UnderlyingPtr);
            }
        }

        /// optional output: the number of points in each sample
        public new unsafe ref void * PtsInSm
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_ptsInSm", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_GetMutable_ptsInSm(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_ptsInSm(_UnderlyingPtr);
            }
        }

        /// optional input: colors of input points
        public new unsafe ref readonly void * PtColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_ptColors", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_GetMutable_ptColors(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_ptColors(_UnderlyingPtr);
            }
        }

        /// optional output: averaged colors of samples
        public new unsafe ref void * SmColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_smColors", ExactSpelling = true)]
                extern static void **__MR_ImproveSamplingSettings_GetMutable_smColors(_Underlying *_this);
                return ref *__MR_ImproveSamplingSettings_GetMutable_smColors(_UnderlyingPtr);
            }
        }

        /// output progress status and receive cancel signal
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ImproveSamplingSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_ImproveSamplingSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ImproveSamplingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_ImproveSamplingSettings_DefaultConstruct();
        }

        /// Constructs `MR::ImproveSamplingSettings` elementwise.
        public unsafe ImproveSamplingSettings(int numIters, int minPointsInSample, MR.VertMap? pt2sm, MR.PointCloud? cloudOfSamples, MR.Vector_Int_MRVertId? ptsInSm, MR.Const_VertColors? ptColors, MR.VertColors? smColors, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_ConstructFrom(int numIters, int minPointsInSample, MR.VertMap._Underlying *pt2sm, MR.PointCloud._Underlying *cloudOfSamples, MR.Vector_Int_MRVertId._Underlying *ptsInSm, MR.Const_VertColors._Underlying *ptColors, MR.VertColors._Underlying *smColors, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_ImproveSamplingSettings_ConstructFrom(numIters, minPointsInSample, pt2sm is not null ? pt2sm._UnderlyingPtr : null, cloudOfSamples is not null ? cloudOfSamples._UnderlyingPtr : null, ptsInSm is not null ? ptsInSm._UnderlyingPtr : null, ptColors is not null ? ptColors._UnderlyingPtr : null, smColors is not null ? smColors._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ImproveSamplingSettings::ImproveSamplingSettings`.
        public unsafe ImproveSamplingSettings(MR._ByValue_ImproveSamplingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ImproveSamplingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_ImproveSamplingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ImproveSamplingSettings::operator=`.
        public unsafe MR.ImproveSamplingSettings Assign(MR._ByValue_ImproveSamplingSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ImproveSamplingSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ImproveSamplingSettings._Underlying *__MR_ImproveSamplingSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ImproveSamplingSettings._Underlying *_other);
            return new(__MR_ImproveSamplingSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ImproveSamplingSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ImproveSamplingSettings`/`Const_ImproveSamplingSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ImproveSamplingSettings
    {
        internal readonly Const_ImproveSamplingSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ImproveSamplingSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ImproveSamplingSettings(Const_ImproveSamplingSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ImproveSamplingSettings(Const_ImproveSamplingSettings arg) {return new(arg);}
        public _ByValue_ImproveSamplingSettings(MR.Misc._Moved<ImproveSamplingSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ImproveSamplingSettings(MR.Misc._Moved<ImproveSamplingSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ImproveSamplingSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ImproveSamplingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ImproveSamplingSettings`/`Const_ImproveSamplingSettings` directly.
    public class _InOptMut_ImproveSamplingSettings
    {
        public ImproveSamplingSettings? Opt;

        public _InOptMut_ImproveSamplingSettings() {}
        public _InOptMut_ImproveSamplingSettings(ImproveSamplingSettings value) {Opt = value;}
        public static implicit operator _InOptMut_ImproveSamplingSettings(ImproveSamplingSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `ImproveSamplingSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ImproveSamplingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ImproveSamplingSettings`/`Const_ImproveSamplingSettings` to pass it to the function.
    public class _InOptConst_ImproveSamplingSettings
    {
        public Const_ImproveSamplingSettings? Opt;

        public _InOptConst_ImproveSamplingSettings() {}
        public _InOptConst_ImproveSamplingSettings(Const_ImproveSamplingSettings value) {Opt = value;}
        public static implicit operator _InOptConst_ImproveSamplingSettings(Const_ImproveSamplingSettings value) {return new(value);}
    }

    /// Finds more representative sampling starting from a given one following k-means method;
    /// \param samples input and output selected sample points from \param cloud;
    /// \return false if it was terminated by the callback
    /// Generated from function `MR::improveSampling`.
    public static unsafe bool ImproveSampling(MR.Const_PointCloud cloud, MR.VertBitSet samples, MR.Const_ImproveSamplingSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_improveSampling", ExactSpelling = true)]
        extern static byte __MR_improveSampling(MR.Const_PointCloud._Underlying *cloud, MR.VertBitSet._Underlying *samples, MR.Const_ImproveSamplingSettings._Underlying *settings);
        return __MR_improveSampling(cloud._UnderlyingPtr, samples._UnderlyingPtr, settings._UnderlyingPtr) != 0;
    }
}
