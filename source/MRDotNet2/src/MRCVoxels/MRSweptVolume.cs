public static partial class MR
{
    /// Parameters for computeSweptVolume* functions
    /// Generated from class `MR::ComputeSweptVolumeParameters`.
    /// This is the const half of the class.
    public class Const_ComputeSweptVolumeParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ComputeSweptVolumeParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_ComputeSweptVolumeParameters_Destroy(_Underlying *_this);
            __MR_ComputeSweptVolumeParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ComputeSweptVolumeParameters() {Dispose(false);}

        /// toolpath
        public unsafe MR.Const_Polyline3 Path
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Get_path", ExactSpelling = true)]
                extern static MR.Const_Polyline3._Underlying *__MR_ComputeSweptVolumeParameters_Get_path(_Underlying *_this);
                return new(__MR_ComputeSweptVolumeParameters_Get_path(_UnderlyingPtr), is_owning: false);
            }
        }

        /// tool mesh
        public unsafe MR.Const_MeshPart ToolMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Get_toolMesh", ExactSpelling = true)]
                extern static MR.Const_MeshPart._Underlying *__MR_ComputeSweptVolumeParameters_Get_toolMesh(_Underlying *_this);
                return new(__MR_ComputeSweptVolumeParameters_Get_toolMesh(_UnderlyingPtr), is_owning: false);
            }
        }

        /// tool specifications, can be used for more precise computations
        /// the tool spec and the tool mesh are expected to relate to the same tool
        /// if omitted, tool mesh is used
        public unsafe ref readonly void * ToolSpec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Get_toolSpec", ExactSpelling = true)]
                extern static void **__MR_ComputeSweptVolumeParameters_Get_toolSpec(_Underlying *_this);
                return ref *__MR_ComputeSweptVolumeParameters_Get_toolSpec(_UnderlyingPtr);
            }
        }

        /// voxel size for internal voxel volumes
        // TODO: replace with tolerance and make the voxel size implementation-specific
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_ComputeSweptVolumeParameters_Get_voxelSize(_Underlying *_this);
                return *__MR_ComputeSweptVolumeParameters_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// (distance volume) max memory amount used for the distance volume, zero for no limits
        public unsafe ulong MemoryLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Get_memoryLimit", ExactSpelling = true)]
                extern static ulong *__MR_ComputeSweptVolumeParameters_Get_memoryLimit(_Underlying *_this);
                return *__MR_ComputeSweptVolumeParameters_Get_memoryLimit(_UnderlyingPtr);
            }
        }

        /// progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ComputeSweptVolumeParameters_Get_cb(_Underlying *_this);
                return new(__MR_ComputeSweptVolumeParameters_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::ComputeSweptVolumeParameters::ComputeSweptVolumeParameters`.
        public unsafe Const_ComputeSweptVolumeParameters(MR._ByValue_ComputeSweptVolumeParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ComputeSweptVolumeParameters._Underlying *__MR_ComputeSweptVolumeParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ComputeSweptVolumeParameters._Underlying *_other);
            _UnderlyingPtr = __MR_ComputeSweptVolumeParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Constructs `MR::ComputeSweptVolumeParameters` elementwise.
        public unsafe Const_ComputeSweptVolumeParameters(MR.Const_Polyline3 path, MR.Const_MeshPart toolMesh, MR.Const_EndMillTool? toolSpec, float voxelSize, ulong memoryLimit, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.ComputeSweptVolumeParameters._Underlying *__MR_ComputeSweptVolumeParameters_ConstructFrom(MR.Const_Polyline3._Underlying *path, MR.MeshPart._Underlying *toolMesh, MR.Const_EndMillTool._Underlying *toolSpec, float voxelSize, ulong memoryLimit, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_ComputeSweptVolumeParameters_ConstructFrom(path._UnderlyingPtr, toolMesh._UnderlyingPtr, toolSpec is not null ? toolSpec._UnderlyingPtr : null, voxelSize, memoryLimit, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }
    }

    /// Parameters for computeSweptVolume* functions
    /// Generated from class `MR::ComputeSweptVolumeParameters`.
    /// This is the non-const half of the class.
    public class ComputeSweptVolumeParameters : Const_ComputeSweptVolumeParameters
    {
        internal unsafe ComputeSweptVolumeParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// tool mesh
        public new unsafe MR.MeshPart ToolMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_GetMutable_toolMesh", ExactSpelling = true)]
                extern static MR.MeshPart._Underlying *__MR_ComputeSweptVolumeParameters_GetMutable_toolMesh(_Underlying *_this);
                return new(__MR_ComputeSweptVolumeParameters_GetMutable_toolMesh(_UnderlyingPtr), is_owning: false);
            }
        }

        /// tool specifications, can be used for more precise computations
        /// the tool spec and the tool mesh are expected to relate to the same tool
        /// if omitted, tool mesh is used
        public new unsafe ref readonly void * ToolSpec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_GetMutable_toolSpec", ExactSpelling = true)]
                extern static void **__MR_ComputeSweptVolumeParameters_GetMutable_toolSpec(_Underlying *_this);
                return ref *__MR_ComputeSweptVolumeParameters_GetMutable_toolSpec(_UnderlyingPtr);
            }
        }

        /// voxel size for internal voxel volumes
        // TODO: replace with tolerance and make the voxel size implementation-specific
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_ComputeSweptVolumeParameters_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_ComputeSweptVolumeParameters_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// (distance volume) max memory amount used for the distance volume, zero for no limits
        public new unsafe ref ulong MemoryLimit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_GetMutable_memoryLimit", ExactSpelling = true)]
                extern static ulong *__MR_ComputeSweptVolumeParameters_GetMutable_memoryLimit(_Underlying *_this);
                return ref *__MR_ComputeSweptVolumeParameters_GetMutable_memoryLimit(_UnderlyingPtr);
            }
        }

        /// progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ComputeSweptVolumeParameters_GetMutable_cb(_Underlying *_this);
                return new(__MR_ComputeSweptVolumeParameters_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::ComputeSweptVolumeParameters::ComputeSweptVolumeParameters`.
        public unsafe ComputeSweptVolumeParameters(MR._ByValue_ComputeSweptVolumeParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ComputeSweptVolumeParameters._Underlying *__MR_ComputeSweptVolumeParameters_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ComputeSweptVolumeParameters._Underlying *_other);
            _UnderlyingPtr = __MR_ComputeSweptVolumeParameters_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Constructs `MR::ComputeSweptVolumeParameters` elementwise.
        public unsafe ComputeSweptVolumeParameters(MR.Const_Polyline3 path, MR.Const_MeshPart toolMesh, MR.Const_EndMillTool? toolSpec, float voxelSize, ulong memoryLimit, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ComputeSweptVolumeParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.ComputeSweptVolumeParameters._Underlying *__MR_ComputeSweptVolumeParameters_ConstructFrom(MR.Const_Polyline3._Underlying *path, MR.MeshPart._Underlying *toolMesh, MR.Const_EndMillTool._Underlying *toolSpec, float voxelSize, ulong memoryLimit, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_ComputeSweptVolumeParameters_ConstructFrom(path._UnderlyingPtr, toolMesh._UnderlyingPtr, toolSpec is not null ? toolSpec._UnderlyingPtr : null, voxelSize, memoryLimit, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ComputeSweptVolumeParameters` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ComputeSweptVolumeParameters`/`Const_ComputeSweptVolumeParameters` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ComputeSweptVolumeParameters
    {
        internal readonly Const_ComputeSweptVolumeParameters? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ComputeSweptVolumeParameters(Const_ComputeSweptVolumeParameters new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ComputeSweptVolumeParameters(Const_ComputeSweptVolumeParameters arg) {return new(arg);}
        public _ByValue_ComputeSweptVolumeParameters(MR.Misc._Moved<ComputeSweptVolumeParameters> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ComputeSweptVolumeParameters(MR.Misc._Moved<ComputeSweptVolumeParameters> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ComputeSweptVolumeParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ComputeSweptVolumeParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ComputeSweptVolumeParameters`/`Const_ComputeSweptVolumeParameters` directly.
    public class _InOptMut_ComputeSweptVolumeParameters
    {
        public ComputeSweptVolumeParameters? Opt;

        public _InOptMut_ComputeSweptVolumeParameters() {}
        public _InOptMut_ComputeSweptVolumeParameters(ComputeSweptVolumeParameters value) {Opt = value;}
        public static implicit operator _InOptMut_ComputeSweptVolumeParameters(ComputeSweptVolumeParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `ComputeSweptVolumeParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ComputeSweptVolumeParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ComputeSweptVolumeParameters`/`Const_ComputeSweptVolumeParameters` to pass it to the function.
    public class _InOptConst_ComputeSweptVolumeParameters
    {
        public Const_ComputeSweptVolumeParameters? Opt;

        public _InOptConst_ComputeSweptVolumeParameters() {}
        public _InOptConst_ComputeSweptVolumeParameters(Const_ComputeSweptVolumeParameters value) {Opt = value;}
        public static implicit operator _InOptConst_ComputeSweptVolumeParameters(Const_ComputeSweptVolumeParameters value) {return new(value);}
    }

    /// Interface for custom tool distance computation implementations
    /// Generated from class `MR::IComputeToolDistance`.
    /// This is the const half of the class.
    public class Const_IComputeToolDistance : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IComputeToolDistance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IComputeToolDistance_Destroy", ExactSpelling = true)]
            extern static void __MR_IComputeToolDistance_Destroy(_Underlying *_this);
            __MR_IComputeToolDistance_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IComputeToolDistance() {Dispose(false);}

        /// Compute tool distance
        /// Generated from method `MR::IComputeToolDistance::computeToolDistance`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> ComputeToolDistance(MR.Std.Vector_Float output, MR.Const_Vector3i dims, float voxelSize, MR.Const_Vector3f origin, float padding)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IComputeToolDistance_computeToolDistance", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IComputeToolDistance_computeToolDistance(_Underlying *_this, MR.Std.Vector_Float._Underlying *output, MR.Const_Vector3i._Underlying *dims, float voxelSize, MR.Const_Vector3f._Underlying *origin, float padding);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IComputeToolDistance_computeToolDistance(_UnderlyingPtr, output._UnderlyingPtr, dims._UnderlyingPtr, voxelSize, origin._UnderlyingPtr, padding), is_owning: true));
        }
    }

    /// Interface for custom tool distance computation implementations
    /// Generated from class `MR::IComputeToolDistance`.
    /// This is the non-const half of the class.
    public class IComputeToolDistance : Const_IComputeToolDistance
    {
        internal unsafe IComputeToolDistance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Prepare for a voxel grid of given dims and copy tool path and tool spec data
        /// \return Maximum dimensions that can be processed at once (e.g. due to memory limits)
        /// Generated from method `MR::IComputeToolDistance::prepare`.
        public unsafe MR.Misc._Moved<MR.Expected_MRVector3i_StdString> Prepare(MR.Const_Vector3i dims, MR.Const_Polyline3 toolpath, MR.Const_EndMillTool toolSpec)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IComputeToolDistance_prepare_MR_EndMillTool", ExactSpelling = true)]
            extern static MR.Expected_MRVector3i_StdString._Underlying *__MR_IComputeToolDistance_prepare_MR_EndMillTool(_Underlying *_this, MR.Const_Vector3i._Underlying *dims, MR.Const_Polyline3._Underlying *toolpath, MR.Const_EndMillTool._Underlying *toolSpec);
            return MR.Misc.Move(new MR.Expected_MRVector3i_StdString(__MR_IComputeToolDistance_prepare_MR_EndMillTool(_UnderlyingPtr, dims._UnderlyingPtr, toolpath._UnderlyingPtr, toolSpec._UnderlyingPtr), is_owning: true));
        }

        /// Prepare for a voxel grid of given dims and copy tool path and tool spec data
        /// \return Maximum dimensions that can be processed at once (e.g. due to memory limits)
        /// Generated from method `MR::IComputeToolDistance::prepare`.
        public unsafe MR.Misc._Moved<MR.Expected_MRVector3i_StdString> Prepare(MR.Const_Vector3i dims, MR.Const_Polyline3 toolpath, MR.Const_Polyline2 toolProfile)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IComputeToolDistance_prepare_MR_Polyline2", ExactSpelling = true)]
            extern static MR.Expected_MRVector3i_StdString._Underlying *__MR_IComputeToolDistance_prepare_MR_Polyline2(_Underlying *_this, MR.Const_Vector3i._Underlying *dims, MR.Const_Polyline3._Underlying *toolpath, MR.Const_Polyline2._Underlying *toolProfile);
            return MR.Misc.Move(new MR.Expected_MRVector3i_StdString(__MR_IComputeToolDistance_prepare_MR_Polyline2(_UnderlyingPtr, dims._UnderlyingPtr, toolpath._UnderlyingPtr, toolProfile._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used for optional parameters of class `IComputeToolDistance` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IComputeToolDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IComputeToolDistance`/`Const_IComputeToolDistance` directly.
    public class _InOptMut_IComputeToolDistance
    {
        public IComputeToolDistance? Opt;

        public _InOptMut_IComputeToolDistance() {}
        public _InOptMut_IComputeToolDistance(IComputeToolDistance value) {Opt = value;}
        public static implicit operator _InOptMut_IComputeToolDistance(IComputeToolDistance value) {return new(value);}
    }

    /// This is used for optional parameters of class `IComputeToolDistance` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IComputeToolDistance`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IComputeToolDistance`/`Const_IComputeToolDistance` to pass it to the function.
    public class _InOptConst_IComputeToolDistance
    {
        public Const_IComputeToolDistance? Opt;

        public _InOptConst_IComputeToolDistance() {}
        public _InOptConst_IComputeToolDistance(Const_IComputeToolDistance value) {Opt = value;}
        public static implicit operator _InOptConst_IComputeToolDistance(Const_IComputeToolDistance value) {return new(value);}
    }

    /// Compute bounding box for swept volume for given tool and toolpath
    /// Generated from function `MR::computeWorkArea`.
    public static unsafe MR.Box3f ComputeWorkArea(MR.Const_Polyline3 toolpath, MR.Const_MeshPart tool)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeWorkArea", ExactSpelling = true)]
        extern static MR.Box3f __MR_computeWorkArea(MR.Const_Polyline3._Underlying *toolpath, MR.Const_MeshPart._Underlying *tool);
        return __MR_computeWorkArea(toolpath._UnderlyingPtr, tool._UnderlyingPtr);
    }

    /// Compute required voxel volume's dimensions for given work area
    /// Generated from function `MR::computeGridBox`.
    public static unsafe MR.Box3i ComputeGridBox(MR.Const_Box3f workArea, float voxelSize)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeGridBox", ExactSpelling = true)]
        extern static MR.Box3i __MR_computeGridBox(MR.Const_Box3f._Underlying *workArea, float voxelSize);
        return __MR_computeGridBox(workArea._UnderlyingPtr, voxelSize);
    }

    /// Compute swept volume for given toolpath and tool
    /// Builds mesh for each tool movement and joins them using voxel boolean
    /// Generated from function `MR::computeSweptVolumeWithMeshMovement`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> ComputeSweptVolumeWithMeshMovement(MR.Const_ComputeSweptVolumeParameters params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSweptVolumeWithMeshMovement", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_computeSweptVolumeWithMeshMovement(MR.Const_ComputeSweptVolumeParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_computeSweptVolumeWithMeshMovement(params_._UnderlyingPtr), is_owning: true));
    }

    /// Compute swept volume for given toolpath and tool
    /// Creates a distance-to-tool volume and converts it to mesh using the marching cubes algorithm
    /// Generated from function `MR::computeSweptVolumeWithDistanceVolume`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> ComputeSweptVolumeWithDistanceVolume(MR.Const_ComputeSweptVolumeParameters params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSweptVolumeWithDistanceVolume", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_computeSweptVolumeWithDistanceVolume(MR.Const_ComputeSweptVolumeParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_computeSweptVolumeWithDistanceVolume(params_._UnderlyingPtr), is_owning: true));
    }

    /// Compute swept volume for given toolpath and tool
    /// Creates a distance-to-tool volume using custom tool distance computation object and converts it to mesh using
    /// the marching cubes algorithm
    /// Generated from function `MR::computeSweptVolumeWithCustomToolDistance`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> ComputeSweptVolumeWithCustomToolDistance(MR.IComputeToolDistance comp, MR.Const_ComputeSweptVolumeParameters params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSweptVolumeWithCustomToolDistance", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_computeSweptVolumeWithCustomToolDistance(MR.IComputeToolDistance._Underlying *comp, MR.Const_ComputeSweptVolumeParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_computeSweptVolumeWithCustomToolDistance(comp._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
