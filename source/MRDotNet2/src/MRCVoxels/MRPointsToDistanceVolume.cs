public static partial class MR
{
    /// Generated from class `MR::PointsToDistanceVolumeParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceVolumeParams`
    /// This is the const half of the class.
    public class Const_PointsToDistanceVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointsToDistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PointsToDistanceVolumeParams_Destroy(_Underlying *_this);
            __MR_PointsToDistanceVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointsToDistanceVolumeParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_DistanceVolumeParams(Const_PointsToDistanceVolumeParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_UpcastTo_MR_DistanceVolumeParams", ExactSpelling = true)]
            extern static MR.Const_DistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_UpcastTo_MR_DistanceVolumeParams(_Underlying *_this);
            MR.Const_DistanceVolumeParams ret = new(__MR_PointsToDistanceVolumeParams_UpcastTo_MR_DistanceVolumeParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// it the distance of highest influence of a point;
        /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
        public unsafe float Sigma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_sigma", ExactSpelling = true)]
                extern static float *__MR_PointsToDistanceVolumeParams_Get_sigma(_Underlying *_this);
                return *__MR_PointsToDistanceVolumeParams_Get_sigma(_UnderlyingPtr);
            }
        }

        /// minimum sum of influence weights from surrounding points for a voxel to get a value, meaning that there shall be at least this number of points in close proximity
        public unsafe float MinWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_minWeight", ExactSpelling = true)]
                extern static float *__MR_PointsToDistanceVolumeParams_Get_minWeight(_Underlying *_this);
                return *__MR_PointsToDistanceVolumeParams_Get_minWeight(_UnderlyingPtr);
            }
        }

        /// optional input: if this pointer is set then function will use these normals instead of ones present in cloud
        public unsafe ref readonly void * PtNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_ptNormals", ExactSpelling = true)]
                extern static void **__MR_PointsToDistanceVolumeParams_Get_ptNormals(_Underlying *_this);
                return ref *__MR_PointsToDistanceVolumeParams_Get_ptNormals(_UnderlyingPtr);
            }
        }

        /// origin point of voxels box
        public unsafe MR.Const_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_origin", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointsToDistanceVolumeParams_Get_origin(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_Get_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PointsToDistanceVolumeParams_Get_cb(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// size of voxel on each axis
        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_PointsToDistanceVolumeParams_Get_voxelSize(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// num voxels along each axis
        public unsafe MR.Const_Vector3i Dimensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_Get_dimensions", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_PointsToDistanceVolumeParams_Get_dimensions(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_Get_dimensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointsToDistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsToDistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsToDistanceVolumeParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointsToDistanceVolumeParams::PointsToDistanceVolumeParams`.
        public unsafe Const_PointsToDistanceVolumeParams(MR._ByValue_PointsToDistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsToDistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsToDistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_PointsToDistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::PointsToDistanceVolumeParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceVolumeParams`
    /// This is the non-const half of the class.
    public class PointsToDistanceVolumeParams : Const_PointsToDistanceVolumeParams
    {
        internal unsafe PointsToDistanceVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.DistanceVolumeParams(PointsToDistanceVolumeParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_UpcastTo_MR_DistanceVolumeParams", ExactSpelling = true)]
            extern static MR.DistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_UpcastTo_MR_DistanceVolumeParams(_Underlying *_this);
            MR.DistanceVolumeParams ret = new(__MR_PointsToDistanceVolumeParams_UpcastTo_MR_DistanceVolumeParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// it the distance of highest influence of a point;
        /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
        public new unsafe ref float Sigma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_sigma", ExactSpelling = true)]
                extern static float *__MR_PointsToDistanceVolumeParams_GetMutable_sigma(_Underlying *_this);
                return ref *__MR_PointsToDistanceVolumeParams_GetMutable_sigma(_UnderlyingPtr);
            }
        }

        /// minimum sum of influence weights from surrounding points for a voxel to get a value, meaning that there shall be at least this number of points in close proximity
        public new unsafe ref float MinWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_minWeight", ExactSpelling = true)]
                extern static float *__MR_PointsToDistanceVolumeParams_GetMutable_minWeight(_Underlying *_this);
                return ref *__MR_PointsToDistanceVolumeParams_GetMutable_minWeight(_UnderlyingPtr);
            }
        }

        /// optional input: if this pointer is set then function will use these normals instead of ones present in cloud
        public new unsafe ref readonly void * PtNormals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_ptNormals", ExactSpelling = true)]
                extern static void **__MR_PointsToDistanceVolumeParams_GetMutable_ptNormals(_Underlying *_this);
                return ref *__MR_PointsToDistanceVolumeParams_GetMutable_ptNormals(_UnderlyingPtr);
            }
        }

        /// origin point of voxels box
        public new unsafe MR.Mut_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_origin", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointsToDistanceVolumeParams_GetMutable_origin(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_GetMutable_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PointsToDistanceVolumeParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// size of voxel on each axis
        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_PointsToDistanceVolumeParams_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// num voxels along each axis
        public new unsafe MR.Mut_Vector3i Dimensions
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_GetMutable_dimensions", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_PointsToDistanceVolumeParams_GetMutable_dimensions(_Underlying *_this);
                return new(__MR_PointsToDistanceVolumeParams_GetMutable_dimensions(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointsToDistanceVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsToDistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsToDistanceVolumeParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointsToDistanceVolumeParams::PointsToDistanceVolumeParams`.
        public unsafe PointsToDistanceVolumeParams(MR._ByValue_PointsToDistanceVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsToDistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsToDistanceVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_PointsToDistanceVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointsToDistanceVolumeParams::operator=`.
        public unsafe MR.PointsToDistanceVolumeParams Assign(MR._ByValue_PointsToDistanceVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToDistanceVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointsToDistanceVolumeParams._Underlying *__MR_PointsToDistanceVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsToDistanceVolumeParams._Underlying *_other);
            return new(__MR_PointsToDistanceVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointsToDistanceVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointsToDistanceVolumeParams`/`Const_PointsToDistanceVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointsToDistanceVolumeParams
    {
        internal readonly Const_PointsToDistanceVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointsToDistanceVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointsToDistanceVolumeParams(Const_PointsToDistanceVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointsToDistanceVolumeParams(Const_PointsToDistanceVolumeParams arg) {return new(arg);}
        public _ByValue_PointsToDistanceVolumeParams(MR.Misc._Moved<PointsToDistanceVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointsToDistanceVolumeParams(MR.Misc._Moved<PointsToDistanceVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointsToDistanceVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointsToDistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsToDistanceVolumeParams`/`Const_PointsToDistanceVolumeParams` directly.
    public class _InOptMut_PointsToDistanceVolumeParams
    {
        public PointsToDistanceVolumeParams? Opt;

        public _InOptMut_PointsToDistanceVolumeParams() {}
        public _InOptMut_PointsToDistanceVolumeParams(PointsToDistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_PointsToDistanceVolumeParams(PointsToDistanceVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointsToDistanceVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointsToDistanceVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsToDistanceVolumeParams`/`Const_PointsToDistanceVolumeParams` to pass it to the function.
    public class _InOptConst_PointsToDistanceVolumeParams
    {
        public Const_PointsToDistanceVolumeParams? Opt;

        public _InOptConst_PointsToDistanceVolumeParams() {}
        public _InOptConst_PointsToDistanceVolumeParams(Const_PointsToDistanceVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_PointsToDistanceVolumeParams(Const_PointsToDistanceVolumeParams value) {return new(value);}
    }

    /// makes SimpleVolume filled with signed distances to points with normals
    /// Generated from function `MR::pointsToDistanceVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolume_StdString> PointsToDistanceVolume(MR.Const_PointCloud cloud, MR.Const_PointsToDistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointsToDistanceVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolume_StdString._Underlying *__MR_pointsToDistanceVolume(MR.Const_PointCloud._Underlying *cloud, MR.Const_PointsToDistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolume_StdString(__MR_pointsToDistanceVolume(cloud._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// makes FunctionVolume representing signed distances to points with normals
    /// Generated from function `MR::pointsToDistanceFunctionVolume`.
    public static unsafe MR.Misc._Moved<MR.FunctionVolume> PointsToDistanceFunctionVolume(MR.Const_PointCloud cloud, MR.Const_PointsToDistanceVolumeParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointsToDistanceFunctionVolume", ExactSpelling = true)]
        extern static MR.FunctionVolume._Underlying *__MR_pointsToDistanceFunctionVolume(MR.Const_PointCloud._Underlying *cloud, MR.Const_PointsToDistanceVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.FunctionVolume(__MR_pointsToDistanceFunctionVolume(cloud._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// given
    /// \param cloud      a point cloud
    /// \param colors     colors of each point in the cloud
    /// \param tgtPoints  some target points
    /// \param tgtVerts   mask of valid target points
    /// \param sigma      the distance of highest influence of a point
    /// \param cb         progress callback
    /// computes the colors in valid target points by averaging the colors from the point cloud
    /// Generated from function `MR::calcAvgColors`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertColors_StdString> CalcAvgColors(MR.Const_PointCloud cloud, MR.Const_VertColors colors, MR.Const_VertCoords tgtPoints, MR.Const_VertBitSet tgtVerts, float sigma, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcAvgColors", ExactSpelling = true)]
        extern static MR.Expected_MRVertColors_StdString._Underlying *__MR_calcAvgColors(MR.Const_PointCloud._Underlying *cloud, MR.Const_VertColors._Underlying *colors, MR.Const_VertCoords._Underlying *tgtPoints, MR.Const_VertBitSet._Underlying *tgtVerts, float sigma, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRVertColors_StdString(__MR_calcAvgColors(cloud._UnderlyingPtr, colors._UnderlyingPtr, tgtPoints._UnderlyingPtr, tgtVerts._UnderlyingPtr, sigma, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }
}
