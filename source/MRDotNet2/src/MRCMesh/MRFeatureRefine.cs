public static partial class MR
{
    /// Optional parameters for \ref refineFeatureObject
    /// Generated from class `MR::RefineParameters`.
    /// This is the const half of the class.
    public class Const_RefineParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RefineParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_RefineParameters_Destroy(_Underlying *_this);
            __MR_RefineParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RefineParameters() {Dispose(false);}

        /// Maximum distance from the source model to the feature
        public unsafe float DistanceLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Get_distanceLimit", ExactSpelling = true)]
                extern static float *__MR_RefineParameters_Get_distanceLimit(_Underlying *_this);
                return *__MR_RefineParameters_Get_distanceLimit(_UnderlyingPtr);
            }
        }

        /// Maximum angle between the source model's normal and the feature's normal
        public unsafe float NormalTolerance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Get_normalTolerance", ExactSpelling = true)]
                extern static float *__MR_RefineParameters_Get_normalTolerance(_Underlying *_this);
                return *__MR_RefineParameters_Get_normalTolerance(_UnderlyingPtr);
            }
        }

        /// (for meshes only) Reference faces used for filtering intermediate results that are too far from it
        public unsafe ref readonly void * FaceRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Get_faceRegion", ExactSpelling = true)]
                extern static void **__MR_RefineParameters_Get_faceRegion(_Underlying *_this);
                return ref *__MR_RefineParameters_Get_faceRegion(_UnderlyingPtr);
            }
        }

        /// (for meshes only) Reference vertices used for filtering intermediate results that are too far from it
        public unsafe ref readonly void * VertRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Get_vertRegion", ExactSpelling = true)]
                extern static void **__MR_RefineParameters_Get_vertRegion(_Underlying *_this);
                return ref *__MR_RefineParameters_Get_vertRegion(_UnderlyingPtr);
            }
        }

        /// Maximum amount of iterations performed until a stable set of points is found
        public unsafe int MaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Get_maxIterations", ExactSpelling = true)]
                extern static int *__MR_RefineParameters_Get_maxIterations(_Underlying *_this);
                return *__MR_RefineParameters_Get_maxIterations(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_RefineParameters_Get_callback(_Underlying *_this);
                return new(__MR_RefineParameters_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RefineParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_RefineParameters_DefaultConstruct();
        }

        /// Constructs `MR::RefineParameters` elementwise.
        public unsafe Const_RefineParameters(float distanceLimit, float normalTolerance, MR.Const_FaceBitSet? faceRegion, MR.Const_VertBitSet? vertRegion, int maxIterations, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_ConstructFrom(float distanceLimit, float normalTolerance, MR.Const_FaceBitSet._Underlying *faceRegion, MR.Const_VertBitSet._Underlying *vertRegion, int maxIterations, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_RefineParameters_ConstructFrom(distanceLimit, normalTolerance, faceRegion is not null ? faceRegion._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null, maxIterations, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::RefineParameters::RefineParameters`.
        public unsafe Const_RefineParameters(MR._ByValue_RefineParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RefineParameters._Underlying *_other);
            _UnderlyingPtr = __MR_RefineParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Optional parameters for \ref refineFeatureObject
    /// Generated from class `MR::RefineParameters`.
    /// This is the non-const half of the class.
    public class RefineParameters : Const_RefineParameters
    {
        internal unsafe RefineParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Maximum distance from the source model to the feature
        public new unsafe ref float DistanceLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_GetMutable_distanceLimit", ExactSpelling = true)]
                extern static float *__MR_RefineParameters_GetMutable_distanceLimit(_Underlying *_this);
                return ref *__MR_RefineParameters_GetMutable_distanceLimit(_UnderlyingPtr);
            }
        }

        /// Maximum angle between the source model's normal and the feature's normal
        public new unsafe ref float NormalTolerance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_GetMutable_normalTolerance", ExactSpelling = true)]
                extern static float *__MR_RefineParameters_GetMutable_normalTolerance(_Underlying *_this);
                return ref *__MR_RefineParameters_GetMutable_normalTolerance(_UnderlyingPtr);
            }
        }

        /// (for meshes only) Reference faces used for filtering intermediate results that are too far from it
        public new unsafe ref readonly void * FaceRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_GetMutable_faceRegion", ExactSpelling = true)]
                extern static void **__MR_RefineParameters_GetMutable_faceRegion(_Underlying *_this);
                return ref *__MR_RefineParameters_GetMutable_faceRegion(_UnderlyingPtr);
            }
        }

        /// (for meshes only) Reference vertices used for filtering intermediate results that are too far from it
        public new unsafe ref readonly void * VertRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_GetMutable_vertRegion", ExactSpelling = true)]
                extern static void **__MR_RefineParameters_GetMutable_vertRegion(_Underlying *_this);
                return ref *__MR_RefineParameters_GetMutable_vertRegion(_UnderlyingPtr);
            }
        }

        /// Maximum amount of iterations performed until a stable set of points is found
        public new unsafe ref int MaxIterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_GetMutable_maxIterations", ExactSpelling = true)]
                extern static int *__MR_RefineParameters_GetMutable_maxIterations(_Underlying *_this);
                return ref *__MR_RefineParameters_GetMutable_maxIterations(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_RefineParameters_GetMutable_callback(_Underlying *_this);
                return new(__MR_RefineParameters_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RefineParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_RefineParameters_DefaultConstruct();
        }

        /// Constructs `MR::RefineParameters` elementwise.
        public unsafe RefineParameters(float distanceLimit, float normalTolerance, MR.Const_FaceBitSet? faceRegion, MR.Const_VertBitSet? vertRegion, int maxIterations, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_ConstructFrom(float distanceLimit, float normalTolerance, MR.Const_FaceBitSet._Underlying *faceRegion, MR.Const_VertBitSet._Underlying *vertRegion, int maxIterations, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_RefineParameters_ConstructFrom(distanceLimit, normalTolerance, faceRegion is not null ? faceRegion._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null, maxIterations, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::RefineParameters::RefineParameters`.
        public unsafe RefineParameters(MR._ByValue_RefineParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RefineParameters._Underlying *_other);
            _UnderlyingPtr = __MR_RefineParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::RefineParameters::operator=`.
        public unsafe MR.RefineParameters Assign(MR._ByValue_RefineParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RefineParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RefineParameters._Underlying *__MR_RefineParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RefineParameters._Underlying *_other);
            return new(__MR_RefineParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `RefineParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RefineParameters`/`Const_RefineParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RefineParameters
    {
        internal readonly Const_RefineParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RefineParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RefineParameters(Const_RefineParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_RefineParameters(Const_RefineParameters arg) {return new(arg);}
        public _ByValue_RefineParameters(MR.Misc._Moved<RefineParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RefineParameters(MR.Misc._Moved<RefineParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `RefineParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RefineParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RefineParameters`/`Const_RefineParameters` directly.
    public class _InOptMut_RefineParameters
    {
        public RefineParameters? Opt;

        public _InOptMut_RefineParameters() {}
        public _InOptMut_RefineParameters(RefineParameters value) {Opt = value;}
        public static implicit operator _InOptMut_RefineParameters(RefineParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `RefineParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RefineParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RefineParameters`/`Const_RefineParameters` to pass it to the function.
    public class _InOptConst_RefineParameters
    {
        public Const_RefineParameters? Opt;

        public _InOptConst_RefineParameters() {}
        public _InOptConst_RefineParameters(Const_RefineParameters value) {Opt = value;}
        public static implicit operator _InOptConst_RefineParameters(Const_RefineParameters value) {return new(value);}
    }

    /// Recalculate the feature object's position so it would better fit with the given mesh
    /// Generated from function `MR::refineFeatureObject`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRAffineXf3f_StdString> RefineFeatureObject(MR.Const_FeatureObject featObj, MR.Const_Mesh mesh, MR.Const_RefineParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_refineFeatureObject_MR_Mesh", ExactSpelling = true)]
        extern static MR.Expected_MRAffineXf3f_StdString._Underlying *__MR_refineFeatureObject_MR_Mesh(MR.Const_FeatureObject._Underlying *featObj, MR.Const_Mesh._Underlying *mesh, MR.Const_RefineParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRAffineXf3f_StdString(__MR_refineFeatureObject_MR_Mesh(featObj._UnderlyingPtr, mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Recalculate the feature object's position so it would better fit with the given point cloud
    /// Generated from function `MR::refineFeatureObject`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRAffineXf3f_StdString> RefineFeatureObject(MR.Const_FeatureObject featObj, MR.Const_PointCloud pointCloud, MR.Const_RefineParameters? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_refineFeatureObject_MR_PointCloud", ExactSpelling = true)]
        extern static MR.Expected_MRAffineXf3f_StdString._Underlying *__MR_refineFeatureObject_MR_PointCloud(MR.Const_FeatureObject._Underlying *featObj, MR.Const_PointCloud._Underlying *pointCloud, MR.Const_RefineParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRAffineXf3f_StdString(__MR_refineFeatureObject_MR_PointCloud(featObj._UnderlyingPtr, pointCloud._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
