public static partial class MR
{
    // Parameters structure for meshToVolume function
    /// Generated from class `MR::MeshToVolumeParams`.
    /// This is the const half of the class.
    public class Const_MeshToVolumeParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshToVolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshToVolumeParams_Destroy(_Underlying *_this);
            __MR_MeshToVolumeParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshToVolumeParams() {Dispose(false);}

        public unsafe MR.MeshToVolumeParams.Type Type_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Get_type", ExactSpelling = true)]
                extern static MR.MeshToVolumeParams.Type *__MR_MeshToVolumeParams_Get_type(_Underlying *_this);
                return *__MR_MeshToVolumeParams_Get_type(_UnderlyingPtr);
            }
        }

        // the number of voxels around surface to calculate distance in (should be positive)
        public unsafe float SurfaceOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Get_surfaceOffset", ExactSpelling = true)]
                extern static float *__MR_MeshToVolumeParams_Get_surfaceOffset(_Underlying *_this);
                return *__MR_MeshToVolumeParams_Get_surfaceOffset(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshToVolumeParams_Get_voxelSize(_Underlying *_this);
                return new(__MR_MeshToVolumeParams_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        // mesh initial transform
        public unsafe MR.Const_AffineXf3f WorldXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Get_worldXf", ExactSpelling = true)]
                extern static MR.Const_AffineXf3f._Underlying *__MR_MeshToVolumeParams_Get_worldXf(_Underlying *_this);
                return new(__MR_MeshToVolumeParams_Get_worldXf(_UnderlyingPtr), is_owning: false);
            }
        }

        // optional output: xf to original mesh (respecting worldXf)
        public unsafe ref MR.AffineXf3f * OutXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Get_outXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshToVolumeParams_Get_outXf(_Underlying *_this);
                return ref *__MR_MeshToVolumeParams_Get_outXf(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MeshToVolumeParams_Get_cb(_Underlying *_this);
                return new(__MR_MeshToVolumeParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshToVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::MeshToVolumeParams` elementwise.
        public unsafe Const_MeshToVolumeParams(MR.MeshToVolumeParams.Type type, float surfaceOffset, MR.Vector3f voxelSize, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f? outXf, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_ConstructFrom(MR.MeshToVolumeParams.Type type, float surfaceOffset, MR.Vector3f voxelSize, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f._Underlying *outXf, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_MeshToVolumeParams_ConstructFrom(type, surfaceOffset, voxelSize, worldXf, outXf is not null ? outXf._UnderlyingPtr : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshToVolumeParams::MeshToVolumeParams`.
        public unsafe Const_MeshToVolumeParams(MR._ByValue_MeshToVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshToVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        // Conversion type
        public enum Type : int
        {
            // only closed meshes can be converted with signed type
            Signed = 0,
            // this type leads to shell like iso-surfaces
            Unsigned = 1,
        }
    }

    // Parameters structure for meshToVolume function
    /// Generated from class `MR::MeshToVolumeParams`.
    /// This is the non-const half of the class.
    public class MeshToVolumeParams : Const_MeshToVolumeParams
    {
        internal unsafe MeshToVolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.MeshToVolumeParams.Type Type_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_GetMutable_type", ExactSpelling = true)]
                extern static MR.MeshToVolumeParams.Type *__MR_MeshToVolumeParams_GetMutable_type(_Underlying *_this);
                return ref *__MR_MeshToVolumeParams_GetMutable_type(_UnderlyingPtr);
            }
        }

        // the number of voxels around surface to calculate distance in (should be positive)
        public new unsafe ref float SurfaceOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_GetMutable_surfaceOffset", ExactSpelling = true)]
                extern static float *__MR_MeshToVolumeParams_GetMutable_surfaceOffset(_Underlying *_this);
                return ref *__MR_MeshToVolumeParams_GetMutable_surfaceOffset(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshToVolumeParams_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_MeshToVolumeParams_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        // mesh initial transform
        public new unsafe MR.Mut_AffineXf3f WorldXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_GetMutable_worldXf", ExactSpelling = true)]
                extern static MR.Mut_AffineXf3f._Underlying *__MR_MeshToVolumeParams_GetMutable_worldXf(_Underlying *_this);
                return new(__MR_MeshToVolumeParams_GetMutable_worldXf(_UnderlyingPtr), is_owning: false);
            }
        }

        // optional output: xf to original mesh (respecting worldXf)
        public new unsafe ref MR.AffineXf3f * OutXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_GetMutable_outXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshToVolumeParams_GetMutable_outXf(_Underlying *_this);
                return ref *__MR_MeshToVolumeParams_GetMutable_outXf(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MeshToVolumeParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_MeshToVolumeParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshToVolumeParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToVolumeParams_DefaultConstruct();
        }

        /// Constructs `MR::MeshToVolumeParams` elementwise.
        public unsafe MeshToVolumeParams(MR.MeshToVolumeParams.Type type, float surfaceOffset, MR.Vector3f voxelSize, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f? outXf, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_ConstructFrom(MR.MeshToVolumeParams.Type type, float surfaceOffset, MR.Vector3f voxelSize, MR.AffineXf3f worldXf, MR.Mut_AffineXf3f._Underlying *outXf, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_MeshToVolumeParams_ConstructFrom(type, surfaceOffset, voxelSize, worldXf, outXf is not null ? outXf._UnderlyingPtr : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshToVolumeParams::MeshToVolumeParams`.
        public unsafe MeshToVolumeParams(MR._ByValue_MeshToVolumeParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshToVolumeParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToVolumeParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshToVolumeParams::operator=`.
        public unsafe MR.MeshToVolumeParams Assign(MR._ByValue_MeshToVolumeParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToVolumeParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshToVolumeParams._Underlying *__MR_MeshToVolumeParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshToVolumeParams._Underlying *_other);
            return new(__MR_MeshToVolumeParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshToVolumeParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshToVolumeParams`/`Const_MeshToVolumeParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshToVolumeParams
    {
        internal readonly Const_MeshToVolumeParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshToVolumeParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshToVolumeParams(Const_MeshToVolumeParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshToVolumeParams(Const_MeshToVolumeParams arg) {return new(arg);}
        public _ByValue_MeshToVolumeParams(MR.Misc._Moved<MeshToVolumeParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshToVolumeParams(MR.Misc._Moved<MeshToVolumeParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshToVolumeParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshToVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToVolumeParams`/`Const_MeshToVolumeParams` directly.
    public class _InOptMut_MeshToVolumeParams
    {
        public MeshToVolumeParams? Opt;

        public _InOptMut_MeshToVolumeParams() {}
        public _InOptMut_MeshToVolumeParams(MeshToVolumeParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshToVolumeParams(MeshToVolumeParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshToVolumeParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshToVolumeParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToVolumeParams`/`Const_MeshToVolumeParams` to pass it to the function.
    public class _InOptConst_MeshToVolumeParams
    {
        public Const_MeshToVolumeParams? Opt;

        public _InOptConst_MeshToVolumeParams() {}
        public _InOptConst_MeshToVolumeParams(Const_MeshToVolumeParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshToVolumeParams(Const_MeshToVolumeParams value) {return new(value);}
    }

    /// parameters of OpenVDB Grid to Mesh conversion using Dual Marching Cubes algorithm
    /// Generated from class `MR::GridToMeshSettings`.
    /// This is the const half of the class.
    public class Const_GridToMeshSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GridToMeshSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_GridToMeshSettings_Destroy(_Underlying *_this);
            __MR_GridToMeshSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GridToMeshSettings() {Dispose(false);}

        /// the size of each voxel in the grid
        public unsafe MR.Const_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_voxelSize", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_GridToMeshSettings_Get_voxelSize(_Underlying *_this);
                return new(__MR_GridToMeshSettings_Get_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// layer of grid with this value would be converted in mesh; isoValue can be negative only in level set grids
        public unsafe float IsoValue
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_isoValue", ExactSpelling = true)]
                extern static float *__MR_GridToMeshSettings_Get_isoValue(_Underlying *_this);
                return *__MR_GridToMeshSettings_Get_isoValue(_UnderlyingPtr);
            }
        }

        /// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones (curvature can be lost on high values)
        public unsafe float Adaptivity
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_adaptivity", ExactSpelling = true)]
                extern static float *__MR_GridToMeshSettings_Get_adaptivity(_Underlying *_this);
                return *__MR_GridToMeshSettings_Get_adaptivity(_UnderlyingPtr);
            }
        }

        /// if the mesh exceeds this number of faces, an error returns
        public unsafe int MaxFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_maxFaces", ExactSpelling = true)]
                extern static int *__MR_GridToMeshSettings_Get_maxFaces(_Underlying *_this);
                return *__MR_GridToMeshSettings_Get_maxFaces(_UnderlyingPtr);
            }
        }

        /// if the mesh exceeds this number of vertices, an error returns
        public unsafe int MaxVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_maxVertices", ExactSpelling = true)]
                extern static int *__MR_GridToMeshSettings_Get_maxVertices(_Underlying *_this);
                return *__MR_GridToMeshSettings_Get_maxVertices(_UnderlyingPtr);
            }
        }

        public unsafe bool RelaxDisorientedTriangles
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_relaxDisorientedTriangles", ExactSpelling = true)]
                extern static bool *__MR_GridToMeshSettings_Get_relaxDisorientedTriangles(_Underlying *_this);
                return *__MR_GridToMeshSettings_Get_relaxDisorientedTriangles(_UnderlyingPtr);
            }
        }

        /// to receive progress and request cancellation
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_GridToMeshSettings_Get_cb(_Underlying *_this);
                return new(__MR_GridToMeshSettings_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GridToMeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_GridToMeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::GridToMeshSettings` elementwise.
        public unsafe Const_GridToMeshSettings(MR.Vector3f voxelSize, float isoValue, float adaptivity, int maxFaces, int maxVertices, bool relaxDisorientedTriangles, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_ConstructFrom(MR.Vector3f voxelSize, float isoValue, float adaptivity, int maxFaces, int maxVertices, byte relaxDisorientedTriangles, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_GridToMeshSettings_ConstructFrom(voxelSize, isoValue, adaptivity, maxFaces, maxVertices, relaxDisorientedTriangles ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::GridToMeshSettings::GridToMeshSettings`.
        public unsafe Const_GridToMeshSettings(MR._ByValue_GridToMeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GridToMeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_GridToMeshSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// parameters of OpenVDB Grid to Mesh conversion using Dual Marching Cubes algorithm
    /// Generated from class `MR::GridToMeshSettings`.
    /// This is the non-const half of the class.
    public class GridToMeshSettings : Const_GridToMeshSettings
    {
        internal unsafe GridToMeshSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the size of each voxel in the grid
        public new unsafe MR.Mut_Vector3f VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_voxelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_GridToMeshSettings_GetMutable_voxelSize(_Underlying *_this);
                return new(__MR_GridToMeshSettings_GetMutable_voxelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        /// layer of grid with this value would be converted in mesh; isoValue can be negative only in level set grids
        public new unsafe ref float IsoValue
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_isoValue", ExactSpelling = true)]
                extern static float *__MR_GridToMeshSettings_GetMutable_isoValue(_Underlying *_this);
                return ref *__MR_GridToMeshSettings_GetMutable_isoValue(_UnderlyingPtr);
            }
        }

        /// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones (curvature can be lost on high values)
        public new unsafe ref float Adaptivity
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_adaptivity", ExactSpelling = true)]
                extern static float *__MR_GridToMeshSettings_GetMutable_adaptivity(_Underlying *_this);
                return ref *__MR_GridToMeshSettings_GetMutable_adaptivity(_UnderlyingPtr);
            }
        }

        /// if the mesh exceeds this number of faces, an error returns
        public new unsafe ref int MaxFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_maxFaces", ExactSpelling = true)]
                extern static int *__MR_GridToMeshSettings_GetMutable_maxFaces(_Underlying *_this);
                return ref *__MR_GridToMeshSettings_GetMutable_maxFaces(_UnderlyingPtr);
            }
        }

        /// if the mesh exceeds this number of vertices, an error returns
        public new unsafe ref int MaxVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_maxVertices", ExactSpelling = true)]
                extern static int *__MR_GridToMeshSettings_GetMutable_maxVertices(_Underlying *_this);
                return ref *__MR_GridToMeshSettings_GetMutable_maxVertices(_UnderlyingPtr);
            }
        }

        public new unsafe ref bool RelaxDisorientedTriangles
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_relaxDisorientedTriangles", ExactSpelling = true)]
                extern static bool *__MR_GridToMeshSettings_GetMutable_relaxDisorientedTriangles(_Underlying *_this);
                return ref *__MR_GridToMeshSettings_GetMutable_relaxDisorientedTriangles(_UnderlyingPtr);
            }
        }

        /// to receive progress and request cancellation
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_GridToMeshSettings_GetMutable_cb(_Underlying *_this);
                return new(__MR_GridToMeshSettings_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe GridToMeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_GridToMeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::GridToMeshSettings` elementwise.
        public unsafe GridToMeshSettings(MR.Vector3f voxelSize, float isoValue, float adaptivity, int maxFaces, int maxVertices, bool relaxDisorientedTriangles, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_ConstructFrom(MR.Vector3f voxelSize, float isoValue, float adaptivity, int maxFaces, int maxVertices, byte relaxDisorientedTriangles, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_GridToMeshSettings_ConstructFrom(voxelSize, isoValue, adaptivity, maxFaces, maxVertices, relaxDisorientedTriangles ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::GridToMeshSettings::GridToMeshSettings`.
        public unsafe GridToMeshSettings(MR._ByValue_GridToMeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GridToMeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_GridToMeshSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::GridToMeshSettings::operator=`.
        public unsafe MR.GridToMeshSettings Assign(MR._ByValue_GridToMeshSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GridToMeshSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GridToMeshSettings._Underlying *__MR_GridToMeshSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GridToMeshSettings._Underlying *_other);
            return new(__MR_GridToMeshSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `GridToMeshSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `GridToMeshSettings`/`Const_GridToMeshSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_GridToMeshSettings
    {
        internal readonly Const_GridToMeshSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_GridToMeshSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_GridToMeshSettings(Const_GridToMeshSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_GridToMeshSettings(Const_GridToMeshSettings arg) {return new(arg);}
        public _ByValue_GridToMeshSettings(MR.Misc._Moved<GridToMeshSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_GridToMeshSettings(MR.Misc._Moved<GridToMeshSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `GridToMeshSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GridToMeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GridToMeshSettings`/`Const_GridToMeshSettings` directly.
    public class _InOptMut_GridToMeshSettings
    {
        public GridToMeshSettings? Opt;

        public _InOptMut_GridToMeshSettings() {}
        public _InOptMut_GridToMeshSettings(GridToMeshSettings value) {Opt = value;}
        public static implicit operator _InOptMut_GridToMeshSettings(GridToMeshSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `GridToMeshSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GridToMeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GridToMeshSettings`/`Const_GridToMeshSettings` to pass it to the function.
    public class _InOptConst_GridToMeshSettings
    {
        public Const_GridToMeshSettings? Opt;

        public _InOptConst_GridToMeshSettings() {}
        public _InOptConst_GridToMeshSettings(Const_GridToMeshSettings value) {Opt = value;}
        public static implicit operator _InOptConst_GridToMeshSettings(Const_GridToMeshSettings value) {return new(value);}
    }

    /// Generated from class `MR::MakeSignedByWindingNumberSettings`.
    /// This is the const half of the class.
    public class Const_MakeSignedByWindingNumberSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MakeSignedByWindingNumberSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_MakeSignedByWindingNumberSettings_Destroy(_Underlying *_this);
            __MR_MakeSignedByWindingNumberSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MakeSignedByWindingNumberSettings() {Dispose(false);}

        /// defines the mapping from mesh reference from to grid reference frame
        public unsafe MR.Const_AffineXf3f MeshToGridXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_Get_meshToGridXf", ExactSpelling = true)]
                extern static MR.Const_AffineXf3f._Underlying *__MR_MakeSignedByWindingNumberSettings_Get_meshToGridXf(_Underlying *_this);
                return new(__MR_MakeSignedByWindingNumberSettings_Get_meshToGridXf(_UnderlyingPtr), is_owning: false);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_MakeSignedByWindingNumberSettings_Get_fwn(_Underlying *_this);
                return new(__MR_MakeSignedByWindingNumberSettings_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_MakeSignedByWindingNumberSettings_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_MakeSignedByWindingNumberSettings_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_MakeSignedByWindingNumberSettings_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_MakeSignedByWindingNumberSettings_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// to report algorithm's progress and to cancel it
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MakeSignedByWindingNumberSettings_Get_progress(_Underlying *_this);
                return new(__MR_MakeSignedByWindingNumberSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MakeSignedByWindingNumberSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MakeSignedByWindingNumberSettings_DefaultConstruct();
        }

        /// Constructs `MR::MakeSignedByWindingNumberSettings` elementwise.
        public unsafe Const_MakeSignedByWindingNumberSettings(MR.AffineXf3f meshToGridXf, MR._ByValue_IFastWindingNumber fwn, float windingNumberThreshold, float windingNumberBeta, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_ConstructFrom(MR.AffineXf3f meshToGridXf, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn, float windingNumberThreshold, float windingNumberBeta, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_MakeSignedByWindingNumberSettings_ConstructFrom(meshToGridXf, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null, windingNumberThreshold, windingNumberBeta, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MakeSignedByWindingNumberSettings::MakeSignedByWindingNumberSettings`.
        public unsafe Const_MakeSignedByWindingNumberSettings(MR._ByValue_MakeSignedByWindingNumberSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MakeSignedByWindingNumberSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MakeSignedByWindingNumberSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::MakeSignedByWindingNumberSettings`.
    /// This is the non-const half of the class.
    public class MakeSignedByWindingNumberSettings : Const_MakeSignedByWindingNumberSettings
    {
        internal unsafe MakeSignedByWindingNumberSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// defines the mapping from mesh reference from to grid reference frame
        public new unsafe MR.Mut_AffineXf3f MeshToGridXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_GetMutable_meshToGridXf", ExactSpelling = true)]
                extern static MR.Mut_AffineXf3f._Underlying *__MR_MakeSignedByWindingNumberSettings_GetMutable_meshToGridXf(_Underlying *_this);
                return new(__MR_MakeSignedByWindingNumberSettings_GetMutable_meshToGridXf(_UnderlyingPtr), is_owning: false);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_MakeSignedByWindingNumberSettings_GetMutable_fwn(_Underlying *_this);
                return new(__MR_MakeSignedByWindingNumberSettings_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_MakeSignedByWindingNumberSettings_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_MakeSignedByWindingNumberSettings_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_MakeSignedByWindingNumberSettings_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_MakeSignedByWindingNumberSettings_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// to report algorithm's progress and to cancel it
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MakeSignedByWindingNumberSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_MakeSignedByWindingNumberSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MakeSignedByWindingNumberSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MakeSignedByWindingNumberSettings_DefaultConstruct();
        }

        /// Constructs `MR::MakeSignedByWindingNumberSettings` elementwise.
        public unsafe MakeSignedByWindingNumberSettings(MR.AffineXf3f meshToGridXf, MR._ByValue_IFastWindingNumber fwn, float windingNumberThreshold, float windingNumberBeta, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_ConstructFrom(MR.AffineXf3f meshToGridXf, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn, float windingNumberThreshold, float windingNumberBeta, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_MakeSignedByWindingNumberSettings_ConstructFrom(meshToGridXf, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null, windingNumberThreshold, windingNumberBeta, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MakeSignedByWindingNumberSettings::MakeSignedByWindingNumberSettings`.
        public unsafe MakeSignedByWindingNumberSettings(MR._ByValue_MakeSignedByWindingNumberSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MakeSignedByWindingNumberSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MakeSignedByWindingNumberSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MakeSignedByWindingNumberSettings::operator=`.
        public unsafe MR.MakeSignedByWindingNumberSettings Assign(MR._ByValue_MakeSignedByWindingNumberSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeSignedByWindingNumberSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MakeSignedByWindingNumberSettings._Underlying *__MR_MakeSignedByWindingNumberSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MakeSignedByWindingNumberSettings._Underlying *_other);
            return new(__MR_MakeSignedByWindingNumberSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MakeSignedByWindingNumberSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MakeSignedByWindingNumberSettings`/`Const_MakeSignedByWindingNumberSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MakeSignedByWindingNumberSettings
    {
        internal readonly Const_MakeSignedByWindingNumberSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MakeSignedByWindingNumberSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MakeSignedByWindingNumberSettings(Const_MakeSignedByWindingNumberSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MakeSignedByWindingNumberSettings(Const_MakeSignedByWindingNumberSettings arg) {return new(arg);}
        public _ByValue_MakeSignedByWindingNumberSettings(MR.Misc._Moved<MakeSignedByWindingNumberSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MakeSignedByWindingNumberSettings(MR.Misc._Moved<MakeSignedByWindingNumberSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MakeSignedByWindingNumberSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MakeSignedByWindingNumberSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MakeSignedByWindingNumberSettings`/`Const_MakeSignedByWindingNumberSettings` directly.
    public class _InOptMut_MakeSignedByWindingNumberSettings
    {
        public MakeSignedByWindingNumberSettings? Opt;

        public _InOptMut_MakeSignedByWindingNumberSettings() {}
        public _InOptMut_MakeSignedByWindingNumberSettings(MakeSignedByWindingNumberSettings value) {Opt = value;}
        public static implicit operator _InOptMut_MakeSignedByWindingNumberSettings(MakeSignedByWindingNumberSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `MakeSignedByWindingNumberSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MakeSignedByWindingNumberSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MakeSignedByWindingNumberSettings`/`Const_MakeSignedByWindingNumberSettings` to pass it to the function.
    public class _InOptConst_MakeSignedByWindingNumberSettings
    {
        public Const_MakeSignedByWindingNumberSettings? Opt;

        public _InOptConst_MakeSignedByWindingNumberSettings() {}
        public _InOptConst_MakeSignedByWindingNumberSettings(Const_MakeSignedByWindingNumberSettings value) {Opt = value;}
        public static implicit operator _InOptConst_MakeSignedByWindingNumberSettings(Const_MakeSignedByWindingNumberSettings value) {return new(value);}
    }

    /// Generated from class `MR::DoubleOffsetSettings`.
    /// This is the const half of the class.
    public class Const_DoubleOffsetSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DoubleOffsetSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DoubleOffsetSettings_Destroy(_Underlying *_this);
            __MR_DoubleOffsetSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DoubleOffsetSettings() {Dispose(false);}

        /// the size of voxel in intermediate voxel grid representation
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_Get_voxelSize(_Underlying *_this);
                return *__MR_DoubleOffsetSettings_Get_voxelSize(_UnderlyingPtr);
            }
        }

        /// the amount of first offset
        public unsafe float OffsetA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_offsetA", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_Get_offsetA(_Underlying *_this);
                return *__MR_DoubleOffsetSettings_Get_offsetA(_UnderlyingPtr);
            }
        }

        /// the amount of second offset
        public unsafe float OffsetB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_offsetB", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_Get_offsetB(_Underlying *_this);
                return *__MR_DoubleOffsetSettings_Get_offsetB(_UnderlyingPtr);
            }
        }

        /// in [0; 1] - ratio of combining small triangles into bigger ones (curvature can be lost on high values)
        public unsafe float Adaptivity
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_adaptivity", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_Get_adaptivity(_Underlying *_this);
                return *__MR_DoubleOffsetSettings_Get_adaptivity(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        public unsafe MR.Const_IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_fwn", ExactSpelling = true)]
                extern static MR.Const_IFastWindingNumber._UnderlyingShared *__MR_DoubleOffsetSettings_Get_fwn(_Underlying *_this);
                return new(__MR_DoubleOffsetSettings_Get_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_DoubleOffsetSettings_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_DoubleOffsetSettings_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// to report algorithm's progress and to cancel it
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DoubleOffsetSettings_Get_progress(_Underlying *_this);
                return new(__MR_DoubleOffsetSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DoubleOffsetSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DoubleOffsetSettings_DefaultConstruct();
        }

        /// Constructs `MR::DoubleOffsetSettings` elementwise.
        public unsafe Const_DoubleOffsetSettings(float voxelSize, float offsetA, float offsetB, float adaptivity, MR._ByValue_IFastWindingNumber fwn, float windingNumberThreshold, float windingNumberBeta, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_ConstructFrom(float voxelSize, float offsetA, float offsetB, float adaptivity, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn, float windingNumberThreshold, float windingNumberBeta, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DoubleOffsetSettings_ConstructFrom(voxelSize, offsetA, offsetB, adaptivity, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null, windingNumberThreshold, windingNumberBeta, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DoubleOffsetSettings::DoubleOffsetSettings`.
        public unsafe Const_DoubleOffsetSettings(MR._ByValue_DoubleOffsetSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DoubleOffsetSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DoubleOffsetSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::DoubleOffsetSettings`.
    /// This is the non-const half of the class.
    public class DoubleOffsetSettings : Const_DoubleOffsetSettings
    {
        internal unsafe DoubleOffsetSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the size of voxel in intermediate voxel grid representation
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_DoubleOffsetSettings_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        /// the amount of first offset
        public new unsafe ref float OffsetA
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_offsetA", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_GetMutable_offsetA(_Underlying *_this);
                return ref *__MR_DoubleOffsetSettings_GetMutable_offsetA(_UnderlyingPtr);
            }
        }

        /// the amount of second offset
        public new unsafe ref float OffsetB
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_offsetB", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_GetMutable_offsetB(_Underlying *_this);
                return ref *__MR_DoubleOffsetSettings_GetMutable_offsetB(_UnderlyingPtr);
            }
        }

        /// in [0; 1] - ratio of combining small triangles into bigger ones (curvature can be lost on high values)
        public new unsafe ref float Adaptivity
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_adaptivity", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_GetMutable_adaptivity(_Underlying *_this);
                return ref *__MR_DoubleOffsetSettings_GetMutable_adaptivity(_UnderlyingPtr);
            }
        }

        /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
        public new unsafe MR.IFastWindingNumber Fwn
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_fwn", ExactSpelling = true)]
                extern static MR.IFastWindingNumber._UnderlyingShared *__MR_DoubleOffsetSettings_GetMutable_fwn(_Underlying *_this);
                return new(__MR_DoubleOffsetSettings_GetMutable_fwn(_UnderlyingPtr), is_owning: false);
            }
        }

        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_DoubleOffsetSettings_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_DoubleOffsetSettings_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_DoubleOffsetSettings_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// to report algorithm's progress and to cancel it
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DoubleOffsetSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_DoubleOffsetSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DoubleOffsetSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DoubleOffsetSettings_DefaultConstruct();
        }

        /// Constructs `MR::DoubleOffsetSettings` elementwise.
        public unsafe DoubleOffsetSettings(float voxelSize, float offsetA, float offsetB, float adaptivity, MR._ByValue_IFastWindingNumber fwn, float windingNumberThreshold, float windingNumberBeta, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_ConstructFrom(float voxelSize, float offsetA, float offsetB, float adaptivity, MR.Misc._PassBy fwn_pass_by, MR.IFastWindingNumber._UnderlyingShared *fwn, float windingNumberThreshold, float windingNumberBeta, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DoubleOffsetSettings_ConstructFrom(voxelSize, offsetA, offsetB, adaptivity, fwn.PassByMode, fwn.Value is not null ? fwn.Value._UnderlyingSharedPtr : null, windingNumberThreshold, windingNumberBeta, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DoubleOffsetSettings::DoubleOffsetSettings`.
        public unsafe DoubleOffsetSettings(MR._ByValue_DoubleOffsetSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DoubleOffsetSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DoubleOffsetSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DoubleOffsetSettings::operator=`.
        public unsafe MR.DoubleOffsetSettings Assign(MR._ByValue_DoubleOffsetSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DoubleOffsetSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DoubleOffsetSettings._Underlying *__MR_DoubleOffsetSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DoubleOffsetSettings._Underlying *_other);
            return new(__MR_DoubleOffsetSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DoubleOffsetSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DoubleOffsetSettings`/`Const_DoubleOffsetSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DoubleOffsetSettings
    {
        internal readonly Const_DoubleOffsetSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DoubleOffsetSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DoubleOffsetSettings(Const_DoubleOffsetSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DoubleOffsetSettings(Const_DoubleOffsetSettings arg) {return new(arg);}
        public _ByValue_DoubleOffsetSettings(MR.Misc._Moved<DoubleOffsetSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DoubleOffsetSettings(MR.Misc._Moved<DoubleOffsetSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DoubleOffsetSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DoubleOffsetSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DoubleOffsetSettings`/`Const_DoubleOffsetSettings` directly.
    public class _InOptMut_DoubleOffsetSettings
    {
        public DoubleOffsetSettings? Opt;

        public _InOptMut_DoubleOffsetSettings() {}
        public _InOptMut_DoubleOffsetSettings(DoubleOffsetSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DoubleOffsetSettings(DoubleOffsetSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DoubleOffsetSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DoubleOffsetSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DoubleOffsetSettings`/`Const_DoubleOffsetSettings` to pass it to the function.
    public class _InOptConst_DoubleOffsetSettings
    {
        public Const_DoubleOffsetSettings? Opt;

        public _InOptConst_DoubleOffsetSettings() {}
        public _InOptConst_DoubleOffsetSettings(Const_DoubleOffsetSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DoubleOffsetSettings(Const_DoubleOffsetSettings value) {return new(value);}
    }

    // closed surface is required
    // surfaceOffset - number voxels around surface to calculate distance in (should be positive)
    // returns null if was canceled by progress callback
    /// Generated from function `MR::meshToLevelSet`.
    /// Parameter `surfaceOffset` defaults to `3`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> MeshToLevelSet(MR.Const_MeshPart mp, MR.Const_AffineXf3f xf, MR.Const_Vector3f voxelSize, float? surfaceOffset = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToLevelSet", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_meshToLevelSet(MR.Const_MeshPart._Underlying *mp, MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector3f._Underlying *voxelSize, float *surfaceOffset, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        float __deref_surfaceOffset = surfaceOffset.GetValueOrDefault();
        return MR.Misc.Move(new MR.FloatGrid(__MR_meshToLevelSet(mp._UnderlyingPtr, xf._UnderlyingPtr, voxelSize._UnderlyingPtr, surfaceOffset.HasValue ? &__deref_surfaceOffset : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    // does not require closed surface, resulting grid cannot be used for boolean operations,
    // surfaceOffset - the number of voxels around surface to calculate distance in (should be positive)
    // returns null if was canceled by progress callback
    /// Generated from function `MR::meshToDistanceField`.
    /// Parameter `surfaceOffset` defaults to `3`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> MeshToDistanceField(MR.Const_MeshPart mp, MR.Const_AffineXf3f xf, MR.Const_Vector3f voxelSize, float? surfaceOffset = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToDistanceField", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_meshToDistanceField(MR.Const_MeshPart._Underlying *mp, MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector3f._Underlying *voxelSize, float *surfaceOffset, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        float __deref_surfaceOffset = surfaceOffset.GetValueOrDefault();
        return MR.Misc.Move(new MR.FloatGrid(__MR_meshToDistanceField(mp._UnderlyingPtr, xf._UnderlyingPtr, voxelSize._UnderlyingPtr, surfaceOffset.HasValue ? &__deref_surfaceOffset : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    // eval min max value from FloatGrid
    /// Generated from function `MR::evalGridMinMax`.
    public static unsafe void EvalGridMinMax(MR.Const_FloatGrid grid, ref float min, ref float max)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_evalGridMinMax", ExactSpelling = true)]
        extern static void __MR_evalGridMinMax(MR.Const_FloatGrid._Underlying *grid, float *min, float *max);
        fixed (float *__ptr_min = &min)
        {
            fixed (float *__ptr_max = &max)
            {
                __MR_evalGridMinMax(grid._UnderlyingPtr, __ptr_min, __ptr_max);
            }
        }
    }

    /// converts mesh (or its part) into a volume filled with signed or unsigned distances to mesh using OpenVDB library;
    /// for signed distances the mesh must be closed;
    /// *params.outXf is untouched
    /// Generated from function `MR::meshToDistanceVdbVolume`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> MeshToDistanceVdbVolume(MR.Const_MeshPart mp, MR.Const_MeshToVolumeParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToDistanceVdbVolume", ExactSpelling = true)]
        extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_meshToDistanceVdbVolume(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshToVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_meshToDistanceVdbVolume(mp._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// converts mesh (or its part) into a volume filled with signed or unsigned distances to mesh using OpenVDB library;
    /// for signed distances the mesh must be closed;
    /// prior to conversion, world space is shifted to ensure that the bounding box of offset mesh is in positive quarter-space,
    /// and the shift is written in *params.outXf
    /// Generated from function `MR::meshToVolume`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVdbVolume_StdString> MeshToVolume(MR.Const_MeshPart mp, MR.Const_MeshToVolumeParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_meshToVolume", ExactSpelling = true)]
        extern static MR.Expected_MRVdbVolume_StdString._Underlying *__MR_meshToVolume(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshToVolumeParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRVdbVolume_StdString(__MR_meshToVolume(mp._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    // fills VdbVolume data from FloatGrid (does not fill voxels size, cause we expect it outside)
    /// Generated from function `MR::floatGridToVdbVolume`.
    public static unsafe MR.Misc._Moved<MR.VdbVolume> FloatGridToVdbVolume(MR._ByValue_FloatGrid grid)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_floatGridToVdbVolume", ExactSpelling = true)]
        extern static MR.VdbVolume._Underlying *__MR_floatGridToVdbVolume(MR.Misc._PassBy grid_pass_by, MR.FloatGrid._Underlying *grid);
        return MR.Misc.Move(new MR.VdbVolume(__MR_floatGridToVdbVolume(grid.PassByMode, grid.Value is not null ? grid.Value._UnderlyingPtr : null), is_owning: true));
    }

    // make FloatGrid from SimpleVolume
    // make copy of data
    // background - the new background value for FloatGrid
    // grid can be used to make iso-surface later with gridToMesh function
    /// Generated from function `MR::simpleVolumeToDenseGrid`.
    /// Parameter `background` defaults to `0.0f`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> SimpleVolumeToDenseGrid(MR.Const_SimpleVolume simpleVolume, float? background = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_simpleVolumeToDenseGrid", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_simpleVolumeToDenseGrid(MR.Const_SimpleVolume._Underlying *simpleVolume, float *background, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        float __deref_background = background.GetValueOrDefault();
        return MR.Misc.Move(new MR.FloatGrid(__MR_simpleVolumeToDenseGrid(simpleVolume._UnderlyingPtr, background.HasValue ? &__deref_background : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    // set the simpleVolume.min as the background value
    /// Generated from function `MR::simpleVolumeToVdbVolume`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.VdbVolume> SimpleVolumeToVdbVolume(MR.Const_SimpleVolumeMinMax simpleVolume, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_simpleVolumeToVdbVolume", ExactSpelling = true)]
        extern static MR.VdbVolume._Underlying *__MR_simpleVolumeToVdbVolume(MR.Const_SimpleVolumeMinMax._Underlying *simpleVolume, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.VdbVolume(__MR_simpleVolumeToVdbVolume(simpleVolume._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    // make VdbVolume from FunctionVolume
    // make copy of data
    // set minimum value as the background value
    /// Generated from function `MR::functionVolumeToVdbVolume`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.VdbVolume> FunctionVolumeToVdbVolume(MR.Const_FunctionVolume functoinVolume, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_functionVolumeToVdbVolume", ExactSpelling = true)]
        extern static MR.VdbVolume._Underlying *__MR_functionVolumeToVdbVolume(MR.Const_FunctionVolume._Underlying *functoinVolume, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.VdbVolume(__MR_functionVolumeToVdbVolume(functoinVolume._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    // make SimpleVolume from VdbVolume
    // make copy of data
    /// Generated from function `MR::vdbVolumeToSimpleVolume`.
    /// Parameter `activeBox` defaults to `MR::Box3i()`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolumeMinMax_StdString> VdbVolumeToSimpleVolume(MR.Const_VdbVolume vdbVolume, MR.Const_Box3i? activeBox = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vdbVolumeToSimpleVolume", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolumeMinMax_StdString._Underlying *__MR_vdbVolumeToSimpleVolume(MR.Const_VdbVolume._Underlying *vdbVolume, MR.Const_Box3i._Underlying *activeBox, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolumeMinMax_StdString(__MR_vdbVolumeToSimpleVolume(vdbVolume._UnderlyingPtr, activeBox is not null ? activeBox._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Makes normalized SimpleVolume from VdbVolume
    /// Normalisation consist of scaling values linearly from the source scale to the interval [0;1]
    /// @note Makes copy of data
    /// @param sourceScale if specified, defines the initial scale of voxels.
    ///     If not specified, it is estimated as min. and max. values from the voxels
    /// Generated from function `MR::vdbVolumeToSimpleVolumeNorm`.
    /// Parameter `activeBox` defaults to `MR::Box3i()`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolumeMinMax_StdString> VdbVolumeToSimpleVolumeNorm(MR.Const_VdbVolume vdbVolume, MR.Const_Box3i? activeBox = null, MR._InOpt_Box1f sourceScale = default, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vdbVolumeToSimpleVolumeNorm", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolumeMinMax_StdString._Underlying *__MR_vdbVolumeToSimpleVolumeNorm(MR.Const_VdbVolume._Underlying *vdbVolume, MR.Const_Box3i._Underlying *activeBox, MR.Box1f *sourceScale, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolumeMinMax_StdString(__MR_vdbVolumeToSimpleVolumeNorm(vdbVolume._UnderlyingPtr, activeBox is not null ? activeBox._UnderlyingPtr : null, sourceScale.HasValue ? &sourceScale.Object : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Makes SimpleVolumeU16 from VdbVolume
    /// Values are linearly scaled from the source scale to the range corresponding to uint16_t
    /// @note Makes copy of data
    /// @param sourceScale if specified, defines the initial scale of voxels.
    ///     If not specified, it is estimated as min. and max. values from the voxels
    /// Generated from function `MR::vdbVolumeToSimpleVolumeU16`.
    /// Parameter `activeBox` defaults to `MR::Box3i()`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRSimpleVolumeMinMaxU16_StdString> VdbVolumeToSimpleVolumeU16(MR.Const_VdbVolume vdbVolume, MR.Const_Box3i? activeBox = null, MR._InOpt_Box1f sourceScale = default, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vdbVolumeToSimpleVolumeU16", ExactSpelling = true)]
        extern static MR.Expected_MRSimpleVolumeMinMaxU16_StdString._Underlying *__MR_vdbVolumeToSimpleVolumeU16(MR.Const_VdbVolume._Underlying *vdbVolume, MR.Const_Box3i._Underlying *activeBox, MR.Box1f *sourceScale, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRSimpleVolumeMinMaxU16_StdString(__MR_vdbVolumeToSimpleVolumeU16(vdbVolume._UnderlyingPtr, activeBox is not null ? activeBox._UnderlyingPtr : null, sourceScale.HasValue ? &sourceScale.Object : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm
    /// Generated from function `MR::gridToMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> GridToMesh(MR.Const_FloatGrid grid, MR.Const_GridToMeshSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_gridToMesh_const_MR_FloatGrid_ref", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_gridToMesh_const_MR_FloatGrid_ref(MR.Const_FloatGrid._Underlying *grid, MR.Const_GridToMeshSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_gridToMesh_const_MR_FloatGrid_ref(grid._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }

    /// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm;
    /// deletes grid in the middle to reduce peak memory consumption
    /// Generated from function `MR::gridToMesh`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> GridToMesh(MR.Misc._Moved<MR.FloatGrid> grid, MR.Const_GridToMeshSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_gridToMesh_MR_FloatGrid_rvalue_ref", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_gridToMesh_MR_FloatGrid_rvalue_ref(MR.FloatGrid._Underlying *grid, MR.Const_GridToMeshSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_gridToMesh_MR_FloatGrid_rvalue_ref(grid.Value._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }

    /// set signs for unsigned distance field grid using generalized winding number computed at voxel grid point from refMesh
    /// Generated from function `MR::makeSignedByWindingNumber`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> MakeSignedByWindingNumber(MR.FloatGrid grid, MR.Const_Vector3f voxelSize, MR.Const_Mesh refMesh, MR.Const_MakeSignedByWindingNumberSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSignedByWindingNumber", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_makeSignedByWindingNumber(MR.FloatGrid._Underlying *grid, MR.Const_Vector3f._Underlying *voxelSize, MR.Const_Mesh._Underlying *refMesh, MR.Const_MakeSignedByWindingNumberSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_makeSignedByWindingNumber(grid._UnderlyingPtr, voxelSize._UnderlyingPtr, refMesh._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }

    /// performs convention from mesh to voxel grid and back with offsetA, and than same with offsetB;
    /// if input mesh is not closed then the sign of distance field will be obtained using generalized winding number computation
    /// Generated from function `MR::doubleOffsetVdb`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> DoubleOffsetVdb(MR.Const_MeshPart mp, MR.Const_DoubleOffsetSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_doubleOffsetVdb", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_doubleOffsetVdb(MR.Const_MeshPart._Underlying *mp, MR.Const_DoubleOffsetSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_doubleOffsetVdb(mp._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
