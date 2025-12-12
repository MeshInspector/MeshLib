public static partial class MR
{
    /// Generated from class `MR::DenoiseViaNormalsSettings`.
    /// This is the const half of the class.
    public class Const_DenoiseViaNormalsSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DenoiseViaNormalsSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DenoiseViaNormalsSettings_Destroy(_Underlying *_this);
            __MR_DenoiseViaNormalsSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DenoiseViaNormalsSettings() {Dispose(false);}

        /// use approximated computation, which is much faster than precise solution
        public unsafe bool FastIndicatorComputation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_fastIndicatorComputation", ExactSpelling = true)]
                extern static bool *__MR_DenoiseViaNormalsSettings_Get_fastIndicatorComputation(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_fastIndicatorComputation(_UnderlyingPtr);
            }
        }

        /// 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
        public unsafe float Beta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_beta", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_Get_beta(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_beta(_UnderlyingPtr);
            }
        }

        /// the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
        public unsafe float Gamma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_gamma", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_Get_gamma(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_gamma(_UnderlyingPtr);
            }
        }

        /// the number of iterations to smooth normals and find creases; the more the better quality, but longer computation
        public unsafe int NormalIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_normalIters", ExactSpelling = true)]
                extern static int *__MR_DenoiseViaNormalsSettings_Get_normalIters(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_normalIters(_UnderlyingPtr);
            }
        }

        /// the number of iterations to update vertex coordinates from found normals; the more the better quality, but longer computation
        public unsafe int PointIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_pointIters", ExactSpelling = true)]
                extern static int *__MR_DenoiseViaNormalsSettings_Get_pointIters(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_pointIters(_UnderlyingPtr);
            }
        }

        /// how much resulting points must be attracted to initial points (e.g. to avoid general shrinkage), must be > 0
        public unsafe float GuideWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_guideWeight", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_Get_guideWeight(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_guideWeight(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_DenoiseViaNormalsSettings_Get_limitNearInitial(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_Get_maxInitialDist(_Underlying *_this);
                return *__MR_DenoiseViaNormalsSettings_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// optionally returns creases found during smoothing
        public unsafe ref void * OutCreases
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_outCreases", ExactSpelling = true)]
                extern static void **__MR_DenoiseViaNormalsSettings_Get_outCreases(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_Get_outCreases(_UnderlyingPtr);
            }
        }

        /// to get the progress and optionally cancel
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DenoiseViaNormalsSettings_Get_cb(_Underlying *_this);
                return new(__MR_DenoiseViaNormalsSettings_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DenoiseViaNormalsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DenoiseViaNormalsSettings_DefaultConstruct();
        }

        /// Constructs `MR::DenoiseViaNormalsSettings` elementwise.
        public unsafe Const_DenoiseViaNormalsSettings(bool fastIndicatorComputation, float beta, float gamma, int normalIters, int pointIters, float guideWeight, bool limitNearInitial, float maxInitialDist, MR.UndirectedEdgeBitSet? outCreases, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_ConstructFrom(byte fastIndicatorComputation, float beta, float gamma, int normalIters, int pointIters, float guideWeight, byte limitNearInitial, float maxInitialDist, MR.UndirectedEdgeBitSet._Underlying *outCreases, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_DenoiseViaNormalsSettings_ConstructFrom(fastIndicatorComputation ? (byte)1 : (byte)0, beta, gamma, normalIters, pointIters, guideWeight, limitNearInitial ? (byte)1 : (byte)0, maxInitialDist, outCreases is not null ? outCreases._UnderlyingPtr : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DenoiseViaNormalsSettings::DenoiseViaNormalsSettings`.
        public unsafe Const_DenoiseViaNormalsSettings(MR._ByValue_DenoiseViaNormalsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DenoiseViaNormalsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DenoiseViaNormalsSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DenoiseViaNormalsSettings`.
    /// This is the non-const half of the class.
    public class DenoiseViaNormalsSettings : Const_DenoiseViaNormalsSettings
    {
        internal unsafe DenoiseViaNormalsSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// use approximated computation, which is much faster than precise solution
        public new unsafe ref bool FastIndicatorComputation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_fastIndicatorComputation", ExactSpelling = true)]
                extern static bool *__MR_DenoiseViaNormalsSettings_GetMutable_fastIndicatorComputation(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_fastIndicatorComputation(_UnderlyingPtr);
            }
        }

        /// 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
        public new unsafe ref float Beta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_beta", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_GetMutable_beta(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_beta(_UnderlyingPtr);
            }
        }

        /// the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
        public new unsafe ref float Gamma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_gamma", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_GetMutable_gamma(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_gamma(_UnderlyingPtr);
            }
        }

        /// the number of iterations to smooth normals and find creases; the more the better quality, but longer computation
        public new unsafe ref int NormalIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_normalIters", ExactSpelling = true)]
                extern static int *__MR_DenoiseViaNormalsSettings_GetMutable_normalIters(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_normalIters(_UnderlyingPtr);
            }
        }

        /// the number of iterations to update vertex coordinates from found normals; the more the better quality, but longer computation
        public new unsafe ref int PointIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_pointIters", ExactSpelling = true)]
                extern static int *__MR_DenoiseViaNormalsSettings_GetMutable_pointIters(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_pointIters(_UnderlyingPtr);
            }
        }

        /// how much resulting points must be attracted to initial points (e.g. to avoid general shrinkage), must be > 0
        public new unsafe ref float GuideWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_guideWeight", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_GetMutable_guideWeight(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_guideWeight(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_DenoiseViaNormalsSettings_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_DenoiseViaNormalsSettings_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// optionally returns creases found during smoothing
        public new unsafe ref void * OutCreases
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_outCreases", ExactSpelling = true)]
                extern static void **__MR_DenoiseViaNormalsSettings_GetMutable_outCreases(_Underlying *_this);
                return ref *__MR_DenoiseViaNormalsSettings_GetMutable_outCreases(_UnderlyingPtr);
            }
        }

        /// to get the progress and optionally cancel
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DenoiseViaNormalsSettings_GetMutable_cb(_Underlying *_this);
                return new(__MR_DenoiseViaNormalsSettings_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DenoiseViaNormalsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DenoiseViaNormalsSettings_DefaultConstruct();
        }

        /// Constructs `MR::DenoiseViaNormalsSettings` elementwise.
        public unsafe DenoiseViaNormalsSettings(bool fastIndicatorComputation, float beta, float gamma, int normalIters, int pointIters, float guideWeight, bool limitNearInitial, float maxInitialDist, MR.UndirectedEdgeBitSet? outCreases, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_ConstructFrom(byte fastIndicatorComputation, float beta, float gamma, int normalIters, int pointIters, float guideWeight, byte limitNearInitial, float maxInitialDist, MR.UndirectedEdgeBitSet._Underlying *outCreases, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_DenoiseViaNormalsSettings_ConstructFrom(fastIndicatorComputation ? (byte)1 : (byte)0, beta, gamma, normalIters, pointIters, guideWeight, limitNearInitial ? (byte)1 : (byte)0, maxInitialDist, outCreases is not null ? outCreases._UnderlyingPtr : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DenoiseViaNormalsSettings::DenoiseViaNormalsSettings`.
        public unsafe DenoiseViaNormalsSettings(MR._ByValue_DenoiseViaNormalsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DenoiseViaNormalsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DenoiseViaNormalsSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DenoiseViaNormalsSettings::operator=`.
        public unsafe MR.DenoiseViaNormalsSettings Assign(MR._ByValue_DenoiseViaNormalsSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenoiseViaNormalsSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DenoiseViaNormalsSettings._Underlying *__MR_DenoiseViaNormalsSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DenoiseViaNormalsSettings._Underlying *_other);
            return new(__MR_DenoiseViaNormalsSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DenoiseViaNormalsSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DenoiseViaNormalsSettings`/`Const_DenoiseViaNormalsSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DenoiseViaNormalsSettings
    {
        internal readonly Const_DenoiseViaNormalsSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DenoiseViaNormalsSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DenoiseViaNormalsSettings(Const_DenoiseViaNormalsSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DenoiseViaNormalsSettings(Const_DenoiseViaNormalsSettings arg) {return new(arg);}
        public _ByValue_DenoiseViaNormalsSettings(MR.Misc._Moved<DenoiseViaNormalsSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DenoiseViaNormalsSettings(MR.Misc._Moved<DenoiseViaNormalsSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DenoiseViaNormalsSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DenoiseViaNormalsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DenoiseViaNormalsSettings`/`Const_DenoiseViaNormalsSettings` directly.
    public class _InOptMut_DenoiseViaNormalsSettings
    {
        public DenoiseViaNormalsSettings? Opt;

        public _InOptMut_DenoiseViaNormalsSettings() {}
        public _InOptMut_DenoiseViaNormalsSettings(DenoiseViaNormalsSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DenoiseViaNormalsSettings(DenoiseViaNormalsSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DenoiseViaNormalsSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DenoiseViaNormalsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DenoiseViaNormalsSettings`/`Const_DenoiseViaNormalsSettings` to pass it to the function.
    public class _InOptConst_DenoiseViaNormalsSettings
    {
        public Const_DenoiseViaNormalsSettings? Opt;

        public _InOptConst_DenoiseViaNormalsSettings() {}
        public _InOptConst_DenoiseViaNormalsSettings(Const_DenoiseViaNormalsSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DenoiseViaNormalsSettings(Const_DenoiseViaNormalsSettings value) {return new(value);}
    }

    /// Smooth face normals, given
    /// \param mesh contains topology information and coordinates for equation weights
    /// \param normals input noisy normals and output smooth normals
    /// \param v edge indicator function (1 - smooth edge, 0 - crease edge)
    /// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
    /// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (19)
    /// Generated from function `MR::denoiseNormals`.
    public static unsafe void DenoiseNormals(MR.Const_Mesh mesh, MR.FaceNormals normals, MR.Const_UndirectedEdgeScalars v, float gamma)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_denoiseNormals", ExactSpelling = true)]
        extern static void __MR_denoiseNormals(MR.Const_Mesh._Underlying *mesh, MR.FaceNormals._Underlying *normals, MR.Const_UndirectedEdgeScalars._Underlying *v, float gamma);
        __MR_denoiseNormals(mesh._UnderlyingPtr, normals._UnderlyingPtr, v._UnderlyingPtr, gamma);
    }

    /// Compute edge indicator function (1 - smooth edge, 0 - crease edge) by solving large system of linear equations
    /// \param mesh contains topology information and coordinates for equation weights
    /// \param normals per-face normals
    /// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
    /// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
    /// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (20)
    /// Generated from function `MR::updateIndicator`.
    public static unsafe void UpdateIndicator(MR.Const_Mesh mesh, MR.UndirectedEdgeScalars v, MR.Const_FaceNormals normals, float beta, float gamma)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_updateIndicator", ExactSpelling = true)]
        extern static void __MR_updateIndicator(MR.Const_Mesh._Underlying *mesh, MR.UndirectedEdgeScalars._Underlying *v, MR.Const_FaceNormals._Underlying *normals, float beta, float gamma);
        __MR_updateIndicator(mesh._UnderlyingPtr, v._UnderlyingPtr, normals._UnderlyingPtr, beta, gamma);
    }

    /// Compute edge indicator function (1 - smooth edge, 0 - crease edge) by approximation without solving the system of linear equations
    /// \param normals per-face normals
    /// \param beta 0.001 - sharp edges, 0.01 - moderate edges, 0.1 - smooth edges
    /// \param gamma the amount of smoothing: 0 - no smoothing, 1 - average smoothing, ...
    /// see the article "Mesh Denoising via a Novel Mumford-Shah Framework", equation (20)
    /// Generated from function `MR::updateIndicatorFast`.
    public static unsafe void UpdateIndicatorFast(MR.Const_MeshTopology topology, MR.UndirectedEdgeScalars v, MR.Const_FaceNormals normals, float beta, float gamma)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_updateIndicatorFast", ExactSpelling = true)]
        extern static void __MR_updateIndicatorFast(MR.Const_MeshTopology._Underlying *topology, MR.UndirectedEdgeScalars._Underlying *v, MR.Const_FaceNormals._Underlying *normals, float beta, float gamma);
        __MR_updateIndicatorFast(topology._UnderlyingPtr, v._UnderlyingPtr, normals._UnderlyingPtr, beta, gamma);
    }

    /// Reduces noise in given mesh,
    /// see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
    /// Generated from function `MR::meshDenoiseViaNormals`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> MeshDenoiseViaNormals(MR.Mesh mesh, MR.Const_DenoiseViaNormalsSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshDenoiseViaNormals", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_meshDenoiseViaNormals(MR.Mesh._Underlying *mesh, MR.Const_DenoiseViaNormalsSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_meshDenoiseViaNormals(mesh._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }
}
