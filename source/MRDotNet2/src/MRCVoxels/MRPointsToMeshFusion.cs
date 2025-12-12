public static partial class MR
{
    /// Generated from class `MR::PointsToMeshParameters`.
    /// This is the const half of the class.
    public class Const_PointsToMeshParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointsToMeshParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_PointsToMeshParameters_Destroy(_Underlying *_this);
            __MR_PointsToMeshParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointsToMeshParameters() {Dispose(false);}

        /// it the distance of highest influence of a point;
        /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
        public unsafe float Sigma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Get_sigma", ExactSpelling = true)]
                extern static float *__MR_PointsToMeshParameters_Get_sigma(_Underlying *_this);
                return *__MR_PointsToMeshParameters_Get_sigma(_UnderlyingPtr);
            }
        }

        /// minimum sum of influence weights from surrounding points for a triangle to appear, meaning that there shall be at least this number of points in close proximity
        public unsafe float MinWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Get_minWeight", ExactSpelling = true)]
                extern static float *__MR_PointsToMeshParameters_Get_minWeight(_Underlying *_this);
                return *__MR_PointsToMeshParameters_Get_minWeight(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_PointsToMeshParameters_Get_voxelSize(_Underlying *_this);
                return *__MR_PointsToMeshParameters_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// optional input: colors of input points
        public unsafe ref readonly void * PtColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Get_ptColors", ExactSpelling = true)]
                extern static void **__MR_PointsToMeshParameters_Get_ptColors(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_Get_ptColors(_UnderlyingPtr);
            }
        }

        /// optional output: averaged colors of mesh vertices
        public unsafe ref void * VColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Get_vColors", ExactSpelling = true)]
                extern static void **__MR_PointsToMeshParameters_Get_vColors(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_Get_vColors(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PointsToMeshParameters_Get_progress(_Underlying *_this);
                return new(__MR_PointsToMeshParameters_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointsToMeshParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsToMeshParameters._Underlying *__MR_PointsToMeshParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsToMeshParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointsToMeshParameters::PointsToMeshParameters`.
        public unsafe Const_PointsToMeshParameters(MR._ByValue_PointsToMeshParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsToMeshParameters._Underlying *__MR_PointsToMeshParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsToMeshParameters._Underlying *_other);
            _UnderlyingPtr = __MR_PointsToMeshParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::PointsToMeshParameters`.
    /// This is the non-const half of the class.
    public class PointsToMeshParameters : Const_PointsToMeshParameters
    {
        internal unsafe PointsToMeshParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// it the distance of highest influence of a point;
        /// the maximal influence distance is 3*sigma; beyond that distance the influence is strictly zero
        public new unsafe ref float Sigma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_GetMutable_sigma", ExactSpelling = true)]
                extern static float *__MR_PointsToMeshParameters_GetMutable_sigma(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_GetMutable_sigma(_UnderlyingPtr);
            }
        }

        /// minimum sum of influence weights from surrounding points for a triangle to appear, meaning that there shall be at least this number of points in close proximity
        public new unsafe ref float MinWeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_GetMutable_minWeight", ExactSpelling = true)]
                extern static float *__MR_PointsToMeshParameters_GetMutable_minWeight(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_GetMutable_minWeight(_UnderlyingPtr);
            }
        }

        /// Size of voxel in grid conversions;
        /// The user is responsible for setting some positive value here
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_PointsToMeshParameters_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// optional input: colors of input points
        public new unsafe ref readonly void * PtColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_GetMutable_ptColors", ExactSpelling = true)]
                extern static void **__MR_PointsToMeshParameters_GetMutable_ptColors(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_GetMutable_ptColors(_UnderlyingPtr);
            }
        }

        /// optional output: averaged colors of mesh vertices
        public new unsafe ref void * VColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_GetMutable_vColors", ExactSpelling = true)]
                extern static void **__MR_PointsToMeshParameters_GetMutable_vColors(_Underlying *_this);
                return ref *__MR_PointsToMeshParameters_GetMutable_vColors(_UnderlyingPtr);
            }
        }

        /// Progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PointsToMeshParameters_GetMutable_progress(_Underlying *_this);
                return new(__MR_PointsToMeshParameters_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointsToMeshParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsToMeshParameters._Underlying *__MR_PointsToMeshParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsToMeshParameters_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointsToMeshParameters::PointsToMeshParameters`.
        public unsafe PointsToMeshParameters(MR._ByValue_PointsToMeshParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsToMeshParameters._Underlying *__MR_PointsToMeshParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsToMeshParameters._Underlying *_other);
            _UnderlyingPtr = __MR_PointsToMeshParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointsToMeshParameters::operator=`.
        public unsafe MR.PointsToMeshParameters Assign(MR._ByValue_PointsToMeshParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointsToMeshParameters._Underlying *__MR_PointsToMeshParameters_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsToMeshParameters._Underlying *_other);
            return new(__MR_PointsToMeshParameters_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointsToMeshParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointsToMeshParameters`/`Const_PointsToMeshParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointsToMeshParameters
    {
        internal readonly Const_PointsToMeshParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointsToMeshParameters() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointsToMeshParameters(Const_PointsToMeshParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointsToMeshParameters(Const_PointsToMeshParameters arg) {return new(arg);}
        public _ByValue_PointsToMeshParameters(MR.Misc._Moved<PointsToMeshParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointsToMeshParameters(MR.Misc._Moved<PointsToMeshParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointsToMeshParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointsToMeshParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsToMeshParameters`/`Const_PointsToMeshParameters` directly.
    public class _InOptMut_PointsToMeshParameters
    {
        public PointsToMeshParameters? Opt;

        public _InOptMut_PointsToMeshParameters() {}
        public _InOptMut_PointsToMeshParameters(PointsToMeshParameters value) {Opt = value;}
        public static implicit operator _InOptMut_PointsToMeshParameters(PointsToMeshParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointsToMeshParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointsToMeshParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsToMeshParameters`/`Const_PointsToMeshParameters` to pass it to the function.
    public class _InOptConst_PointsToMeshParameters
    {
        public Const_PointsToMeshParameters? Opt;

        public _InOptConst_PointsToMeshParameters() {}
        public _InOptConst_PointsToMeshParameters(Const_PointsToMeshParameters value) {Opt = value;}
        public static implicit operator _InOptConst_PointsToMeshParameters(Const_PointsToMeshParameters value) {return new(value);}
    }

    /// makes mesh from points with normals by constructing intermediate volume with signed distances
    /// and then using marching cubes algorithm to extract the surface from there
    /// Generated from function `MR::pointsToMeshFusion`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> PointsToMeshFusion(MR.Const_PointCloud cloud, MR.Const_PointsToMeshParameters params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_pointsToMeshFusion", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_pointsToMeshFusion(MR.Const_PointCloud._Underlying *cloud, MR.Const_PointsToMeshParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_pointsToMeshFusion(cloud._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
