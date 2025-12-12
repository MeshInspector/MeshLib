public static partial class MR
{
    /// Generated from class `MR::FaceDistancesSettings`.
    /// This is the const half of the class.
    public class Const_FaceDistancesSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FaceDistancesSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceDistancesSettings_Destroy(_Underlying *_this);
            __MR_FaceDistancesSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceDistancesSettings() {Dispose(false);}

        public unsafe MR.FaceDistancesSettings.OutputFaceValues Out
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_Get_out", ExactSpelling = true)]
                extern static MR.FaceDistancesSettings.OutputFaceValues *__MR_FaceDistancesSettings_Get_out(_Underlying *_this);
                return *__MR_FaceDistancesSettings_Get_out(_UnderlyingPtr);
            }
        }

        /// optional output of the maximal distance to the most distant face
        public unsafe ref float * MaxDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_Get_maxDist", ExactSpelling = true)]
                extern static float **__MR_FaceDistancesSettings_Get_maxDist(_Underlying *_this);
                return ref *__MR_FaceDistancesSettings_Get_maxDist(_UnderlyingPtr);
            }
        }

        /// for progress reporting and cancellation
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FaceDistancesSettings_Get_progress(_Underlying *_this);
                return new(__MR_FaceDistancesSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceDistancesSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceDistancesSettings_DefaultConstruct();
        }

        /// Constructs `MR::FaceDistancesSettings` elementwise.
        public unsafe Const_FaceDistancesSettings(MR.FaceDistancesSettings.OutputFaceValues out_, MR.Misc.InOut<float>? maxDist, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_ConstructFrom(MR.FaceDistancesSettings.OutputFaceValues out_, float *maxDist, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            float __value_maxDist = maxDist is not null ? maxDist.Value : default(float);
            _UnderlyingPtr = __MR_FaceDistancesSettings_ConstructFrom(out_, maxDist is not null ? &__value_maxDist : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            if (maxDist is not null) maxDist.Value = __value_maxDist;
        }

        /// Generated from constructor `MR::FaceDistancesSettings::FaceDistancesSettings`.
        public unsafe Const_FaceDistancesSettings(MR._ByValue_FaceDistancesSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceDistancesSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FaceDistancesSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        public enum OutputFaceValues : int
        {
            ///< each face will get its distance from start in the result
            Distances = 0,
            ///< each face will get its sequential order (1,2,...) from start in the result
            SeqOrder = 1,
        }
    }

    /// Generated from class `MR::FaceDistancesSettings`.
    /// This is the non-const half of the class.
    public class FaceDistancesSettings : Const_FaceDistancesSettings
    {
        internal unsafe FaceDistancesSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.FaceDistancesSettings.OutputFaceValues Out
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_GetMutable_out", ExactSpelling = true)]
                extern static MR.FaceDistancesSettings.OutputFaceValues *__MR_FaceDistancesSettings_GetMutable_out(_Underlying *_this);
                return ref *__MR_FaceDistancesSettings_GetMutable_out(_UnderlyingPtr);
            }
        }

        /// optional output of the maximal distance to the most distant face
        public new unsafe ref float * MaxDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_GetMutable_maxDist", ExactSpelling = true)]
                extern static float **__MR_FaceDistancesSettings_GetMutable_maxDist(_Underlying *_this);
                return ref *__MR_FaceDistancesSettings_GetMutable_maxDist(_UnderlyingPtr);
            }
        }

        /// for progress reporting and cancellation
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FaceDistancesSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_FaceDistancesSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceDistancesSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceDistancesSettings_DefaultConstruct();
        }

        /// Constructs `MR::FaceDistancesSettings` elementwise.
        public unsafe FaceDistancesSettings(MR.FaceDistancesSettings.OutputFaceValues out_, MR.Misc.InOut<float>? maxDist, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_ConstructFrom(MR.FaceDistancesSettings.OutputFaceValues out_, float *maxDist, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            float __value_maxDist = maxDist is not null ? maxDist.Value : default(float);
            _UnderlyingPtr = __MR_FaceDistancesSettings_ConstructFrom(out_, maxDist is not null ? &__value_maxDist : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            if (maxDist is not null) maxDist.Value = __value_maxDist;
        }

        /// Generated from constructor `MR::FaceDistancesSettings::FaceDistancesSettings`.
        public unsafe FaceDistancesSettings(MR._ByValue_FaceDistancesSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceDistancesSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FaceDistancesSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FaceDistancesSettings::operator=`.
        public unsafe MR.FaceDistancesSettings Assign(MR._ByValue_FaceDistancesSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceDistancesSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FaceDistancesSettings._Underlying *__MR_FaceDistancesSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FaceDistancesSettings._Underlying *_other);
            return new(__MR_FaceDistancesSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FaceDistancesSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FaceDistancesSettings`/`Const_FaceDistancesSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FaceDistancesSettings
    {
        internal readonly Const_FaceDistancesSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FaceDistancesSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FaceDistancesSettings(Const_FaceDistancesSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FaceDistancesSettings(Const_FaceDistancesSettings arg) {return new(arg);}
        public _ByValue_FaceDistancesSettings(MR.Misc._Moved<FaceDistancesSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FaceDistancesSettings(MR.Misc._Moved<FaceDistancesSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FaceDistancesSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceDistancesSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceDistancesSettings`/`Const_FaceDistancesSettings` directly.
    public class _InOptMut_FaceDistancesSettings
    {
        public FaceDistancesSettings? Opt;

        public _InOptMut_FaceDistancesSettings() {}
        public _InOptMut_FaceDistancesSettings(FaceDistancesSettings value) {Opt = value;}
        public static implicit operator _InOptMut_FaceDistancesSettings(FaceDistancesSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `FaceDistancesSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceDistancesSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceDistancesSettings`/`Const_FaceDistancesSettings` to pass it to the function.
    public class _InOptConst_FaceDistancesSettings
    {
        public Const_FaceDistancesSettings? Opt;

        public _InOptConst_FaceDistancesSettings() {}
        public _InOptConst_FaceDistancesSettings(Const_FaceDistancesSettings value) {Opt = value;}
        public static implicit operator _InOptConst_FaceDistancesSettings(Const_FaceDistancesSettings value) {return new(value);}
    }

    /// computes and returns the distance of traveling from one of start faces to all other reachable faces on the mesh;
    /// all unreachable faces will get FLT_MAX value;
    /// \param starts all start faces will get value 0 in the result;
    /// \param metric metric(e) says the distance of traveling from left(e) to right(e)
    /// Generated from function `MR::calcFaceDistances`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRFaceScalars> CalcFaceDistances(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.Const_FaceBitSet starts, MR.Const_FaceDistancesSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcFaceDistances", ExactSpelling = true)]
        extern static MR.Std.Optional_MRFaceScalars._Underlying *__MR_calcFaceDistances(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Const_FaceBitSet._Underlying *starts, MR.Const_FaceDistancesSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Std.Optional_MRFaceScalars(__MR_calcFaceDistances(topology._UnderlyingPtr, metric._UnderlyingPtr, starts._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }
}
