public static partial class MR
{
    /// Generated from class `MR::MoveMeshToVoxelMaxDerivSettings`.
    /// This is the const half of the class.
    public class Const_MoveMeshToVoxelMaxDerivSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MoveMeshToVoxelMaxDerivSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_MoveMeshToVoxelMaxDerivSettings_Destroy(_Underlying *_this);
            __MR_MoveMeshToVoxelMaxDerivSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MoveMeshToVoxelMaxDerivSettings() {Dispose(false);}

        /// number of iterations. Each iteration moves vertex only slightly and smooths the vector field of shifts.
        public unsafe int Iters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Get_iters", ExactSpelling = true)]
                extern static int *__MR_MoveMeshToVoxelMaxDerivSettings_Get_iters(_Underlying *_this);
                return *__MR_MoveMeshToVoxelMaxDerivSettings_Get_iters(_UnderlyingPtr);
            }
        }

        /// number of points to sample for each vertex. Samples are used to get the picewice-linear function of density and
        /// estimate the derivative based on it
        public unsafe int SamplePoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Get_samplePoints", ExactSpelling = true)]
                extern static int *__MR_MoveMeshToVoxelMaxDerivSettings_Get_samplePoints(_Underlying *_this);
                return *__MR_MoveMeshToVoxelMaxDerivSettings_Get_samplePoints(_UnderlyingPtr);
            }
        }

        /// degree of the polynomial used to fit sampled points. Must be in range [3; 6]
        public unsafe int Degree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Get_degree", ExactSpelling = true)]
                extern static int *__MR_MoveMeshToVoxelMaxDerivSettings_Get_degree(_Underlying *_this);
                return *__MR_MoveMeshToVoxelMaxDerivSettings_Get_degree(_UnderlyingPtr);
            }
        }

        /// for each iteration, if target position of the vertex is greater than this threshold, it is disregarded.
        /// For small degrees, this value should be small, for large degrees it may be larger.
        /// Measured in number of voxels.
        public unsafe float OutlierThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Get_outlierThreshold", ExactSpelling = true)]
                extern static float *__MR_MoveMeshToVoxelMaxDerivSettings_Get_outlierThreshold(_Underlying *_this);
                return *__MR_MoveMeshToVoxelMaxDerivSettings_Get_outlierThreshold(_UnderlyingPtr);
            }
        }

        /// force of the smoothing (relaxation) of vector field of shifts on each iteration
        public unsafe float IntermediateSmoothForce
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Get_intermediateSmoothForce", ExactSpelling = true)]
                extern static float *__MR_MoveMeshToVoxelMaxDerivSettings_Get_intermediateSmoothForce(_Underlying *_this);
                return *__MR_MoveMeshToVoxelMaxDerivSettings_Get_intermediateSmoothForce(_UnderlyingPtr);
            }
        }

        /// force of initial smoothing of vertices, before applying the algorithm
        public unsafe float PreparationSmoothForce
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_Get_preparationSmoothForce", ExactSpelling = true)]
                extern static float *__MR_MoveMeshToVoxelMaxDerivSettings_Get_preparationSmoothForce(_Underlying *_this);
                return *__MR_MoveMeshToVoxelMaxDerivSettings_Get_preparationSmoothForce(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MoveMeshToVoxelMaxDerivSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MoveMeshToVoxelMaxDerivSettings_DefaultConstruct();
        }

        /// Constructs `MR::MoveMeshToVoxelMaxDerivSettings` elementwise.
        public unsafe Const_MoveMeshToVoxelMaxDerivSettings(int iters, int samplePoints, int degree, float outlierThreshold, float intermediateSmoothForce, float preparationSmoothForce) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_ConstructFrom(int iters, int samplePoints, int degree, float outlierThreshold, float intermediateSmoothForce, float preparationSmoothForce);
            _UnderlyingPtr = __MR_MoveMeshToVoxelMaxDerivSettings_ConstructFrom(iters, samplePoints, degree, outlierThreshold, intermediateSmoothForce, preparationSmoothForce);
        }

        /// Generated from constructor `MR::MoveMeshToVoxelMaxDerivSettings::MoveMeshToVoxelMaxDerivSettings`.
        public unsafe Const_MoveMeshToVoxelMaxDerivSettings(MR.Const_MoveMeshToVoxelMaxDerivSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_ConstructFromAnother(MR.MoveMeshToVoxelMaxDerivSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MoveMeshToVoxelMaxDerivSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MoveMeshToVoxelMaxDerivSettings`.
    /// This is the non-const half of the class.
    public class MoveMeshToVoxelMaxDerivSettings : Const_MoveMeshToVoxelMaxDerivSettings
    {
        internal unsafe MoveMeshToVoxelMaxDerivSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// number of iterations. Each iteration moves vertex only slightly and smooths the vector field of shifts.
        public new unsafe ref int Iters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_iters", ExactSpelling = true)]
                extern static int *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_iters(_Underlying *_this);
                return ref *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_iters(_UnderlyingPtr);
            }
        }

        /// number of points to sample for each vertex. Samples are used to get the picewice-linear function of density and
        /// estimate the derivative based on it
        public new unsafe ref int SamplePoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_samplePoints", ExactSpelling = true)]
                extern static int *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_samplePoints(_Underlying *_this);
                return ref *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_samplePoints(_UnderlyingPtr);
            }
        }

        /// degree of the polynomial used to fit sampled points. Must be in range [3; 6]
        public new unsafe ref int Degree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_degree", ExactSpelling = true)]
                extern static int *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_degree(_Underlying *_this);
                return ref *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_degree(_UnderlyingPtr);
            }
        }

        /// for each iteration, if target position of the vertex is greater than this threshold, it is disregarded.
        /// For small degrees, this value should be small, for large degrees it may be larger.
        /// Measured in number of voxels.
        public new unsafe ref float OutlierThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_outlierThreshold", ExactSpelling = true)]
                extern static float *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_outlierThreshold(_Underlying *_this);
                return ref *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_outlierThreshold(_UnderlyingPtr);
            }
        }

        /// force of the smoothing (relaxation) of vector field of shifts on each iteration
        public new unsafe ref float IntermediateSmoothForce
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_intermediateSmoothForce", ExactSpelling = true)]
                extern static float *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_intermediateSmoothForce(_Underlying *_this);
                return ref *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_intermediateSmoothForce(_UnderlyingPtr);
            }
        }

        /// force of initial smoothing of vertices, before applying the algorithm
        public new unsafe ref float PreparationSmoothForce
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_preparationSmoothForce", ExactSpelling = true)]
                extern static float *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_preparationSmoothForce(_Underlying *_this);
                return ref *__MR_MoveMeshToVoxelMaxDerivSettings_GetMutable_preparationSmoothForce(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MoveMeshToVoxelMaxDerivSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_MoveMeshToVoxelMaxDerivSettings_DefaultConstruct();
        }

        /// Constructs `MR::MoveMeshToVoxelMaxDerivSettings` elementwise.
        public unsafe MoveMeshToVoxelMaxDerivSettings(int iters, int samplePoints, int degree, float outlierThreshold, float intermediateSmoothForce, float preparationSmoothForce) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_ConstructFrom(int iters, int samplePoints, int degree, float outlierThreshold, float intermediateSmoothForce, float preparationSmoothForce);
            _UnderlyingPtr = __MR_MoveMeshToVoxelMaxDerivSettings_ConstructFrom(iters, samplePoints, degree, outlierThreshold, intermediateSmoothForce, preparationSmoothForce);
        }

        /// Generated from constructor `MR::MoveMeshToVoxelMaxDerivSettings::MoveMeshToVoxelMaxDerivSettings`.
        public unsafe MoveMeshToVoxelMaxDerivSettings(MR.Const_MoveMeshToVoxelMaxDerivSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_ConstructFromAnother(MR.MoveMeshToVoxelMaxDerivSettings._Underlying *_other);
            _UnderlyingPtr = __MR_MoveMeshToVoxelMaxDerivSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MoveMeshToVoxelMaxDerivSettings::operator=`.
        public unsafe MR.MoveMeshToVoxelMaxDerivSettings Assign(MR.Const_MoveMeshToVoxelMaxDerivSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MoveMeshToVoxelMaxDerivSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MoveMeshToVoxelMaxDerivSettings._Underlying *__MR_MoveMeshToVoxelMaxDerivSettings_AssignFromAnother(_Underlying *_this, MR.MoveMeshToVoxelMaxDerivSettings._Underlying *_other);
            return new(__MR_MoveMeshToVoxelMaxDerivSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MoveMeshToVoxelMaxDerivSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MoveMeshToVoxelMaxDerivSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MoveMeshToVoxelMaxDerivSettings`/`Const_MoveMeshToVoxelMaxDerivSettings` directly.
    public class _InOptMut_MoveMeshToVoxelMaxDerivSettings
    {
        public MoveMeshToVoxelMaxDerivSettings? Opt;

        public _InOptMut_MoveMeshToVoxelMaxDerivSettings() {}
        public _InOptMut_MoveMeshToVoxelMaxDerivSettings(MoveMeshToVoxelMaxDerivSettings value) {Opt = value;}
        public static implicit operator _InOptMut_MoveMeshToVoxelMaxDerivSettings(MoveMeshToVoxelMaxDerivSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `MoveMeshToVoxelMaxDerivSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MoveMeshToVoxelMaxDerivSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MoveMeshToVoxelMaxDerivSettings`/`Const_MoveMeshToVoxelMaxDerivSettings` to pass it to the function.
    public class _InOptConst_MoveMeshToVoxelMaxDerivSettings
    {
        public Const_MoveMeshToVoxelMaxDerivSettings? Opt;

        public _InOptConst_MoveMeshToVoxelMaxDerivSettings() {}
        public _InOptConst_MoveMeshToVoxelMaxDerivSettings(Const_MoveMeshToVoxelMaxDerivSettings value) {Opt = value;}
        public static implicit operator _InOptConst_MoveMeshToVoxelMaxDerivSettings(Const_MoveMeshToVoxelMaxDerivSettings value) {return new(value);}
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>`.
    /// This is the const half of the class.
    public class Const_MeshOnVoxelsT_MRMesh_MRVdbVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOnVoxelsT_MRMesh_MRVdbVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Destroy(_Underlying *_this);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOnVoxelsT_MRMesh_MRVdbVolume() {Dispose(false);}

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_MRMesh_MRVdbVolume(MR._ByValue_MeshOnVoxelsT_MRMesh_MRVdbVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_MRMesh_MRVdbVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_MRMesh_MRVdbVolume(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_VdbVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Construct(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_VdbVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }

        // Access to base data
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::mesh`.
        public unsafe MR.Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_mesh", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_mesh(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::volume`.
        public unsafe MR.Const_VdbVolume Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_volume", ExactSpelling = true)]
            extern static MR.Const_VdbVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_volume(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_volume(_UnderlyingPtr), is_owning: false);
        }

        // Cached number of valid vertices
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::numVerts`.
        public unsafe int NumVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_numVerts", ExactSpelling = true)]
            extern static int __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_numVerts(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_numVerts(_UnderlyingPtr);
        }

        // Voxel size as scalar
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::voxelSize`.
        public unsafe float VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_voxelSize", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_voxelSize(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_voxelSize(_UnderlyingPtr);
        }

        // Transformation mesh to volume
        // All points are in voxels volume space, unless otherwise is implied
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xf_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xf_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xf_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::xf`.
        public unsafe MR.Vector3f Xf(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xf_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xf_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xf_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::xfInv`.
        public unsafe MR.AffineXf3f XfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xfInv_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xfInv_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xfInv_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::xfInv`.
        public unsafe MR.Vector3f XfInv(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xfInv_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xfInv_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_xfInv_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        // Vertex position
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::point`.
        public unsafe MR.Vector3f Point(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_point", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_point(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_point(_UnderlyingPtr, v);
        }

        // Volume value
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::getValue`.
        public unsafe float GetValue(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getValue", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getValue(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getValue(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        // Get offset vector (mesh normal for a vertex with `voxelSize` length)
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::getOffsetVector`.
        public unsafe MR.Vector3f GetOffsetVector(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getOffsetVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getOffsetVector(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getOffsetVector(_UnderlyingPtr, v);
        }

        // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
        // Pseudo-index is a signed number; for whole index, is is whole or half-whole
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::pseudoIndex`.
        public static float PseudoIndex(float index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_pseudoIndex_float", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_pseudoIndex_float(float index, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_pseudoIndex_float(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::pseudoIndex`.
        public static float PseudoIndex(int index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_pseudoIndex_int", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_pseudoIndex_int(int index, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_pseudoIndex_int(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::indexFromPseudoIndex`.
        public static float IndexFromPseudoIndex(float pseudoIndex, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_indexFromPseudoIndex", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_indexFromPseudoIndex(float pseudoIndex, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_indexFromPseudoIndex(pseudoIndex, count);
        }

        // Get row of points with `offset` stride
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::getPoints`.
        public unsafe void GetPoints(MR.Std.Vector_MRVector3f result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getPoints", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getPoints(_Underlying *_this, MR.Std.Vector_MRVector3f._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getPoints(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get volume values for a row of points
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::getValues`.
        public unsafe void GetValues(MR.Std.Vector_Float result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getValues", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getValues(_Underlying *_this, MR.Std.Vector_Float._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getValues(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get derivatives from result of `getValues`
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::getDerivatives`.
        public static unsafe void GetDerivatives(MR.Std.Vector_Float result, MR.Std.Const_Vector_Float values)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getDerivatives", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getDerivatives(MR.Std.Vector_Float._Underlying *result, MR.Std.Const_Vector_Float._Underlying *values);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getDerivatives(result._UnderlyingPtr, values._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::getBestPolynomial`.
        public static unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> GetBestPolynomial(MR.Std.Const_Vector_Float values, ulong degree)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getBestPolynomial", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getBestPolynomial(MR.Std.Const_Vector_Float._Underlying *values, ulong degree);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_getBestPolynomial(values._UnderlyingPtr, degree), is_owning: true));
        }
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>`.
    /// This is the non-const half of the class.
    public class MeshOnVoxelsT_MRMesh_MRVdbVolume : Const_MeshOnVoxelsT_MRMesh_MRVdbVolume
    {
        internal unsafe MeshOnVoxelsT_MRMesh_MRVdbVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_MRMesh_MRVdbVolume(MR._ByValue_MeshOnVoxelsT_MRMesh_MRVdbVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_MRMesh_MRVdbVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_MRMesh_MRVdbVolume(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_VdbVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Construct(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_VdbVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_VdbVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOnVoxelsT_MRMesh_MRVdbVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRVdbVolume`/`Const_MeshOnVoxelsT_MRMesh_MRVdbVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOnVoxelsT_MRMesh_MRVdbVolume
    {
        internal readonly Const_MeshOnVoxelsT_MRMesh_MRVdbVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOnVoxelsT_MRMesh_MRVdbVolume(Const_MeshOnVoxelsT_MRMesh_MRVdbVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOnVoxelsT_MRMesh_MRVdbVolume(Const_MeshOnVoxelsT_MRMesh_MRVdbVolume arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_MRMesh_MRVdbVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOnVoxelsT_MRMesh_MRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRVdbVolume`/`Const_MeshOnVoxelsT_MRMesh_MRVdbVolume` directly.
    public class _InOptMut_MeshOnVoxelsT_MRMesh_MRVdbVolume
    {
        public MeshOnVoxelsT_MRMesh_MRVdbVolume? Opt;

        public _InOptMut_MeshOnVoxelsT_MRMesh_MRVdbVolume() {}
        public _InOptMut_MeshOnVoxelsT_MRMesh_MRVdbVolume(MeshOnVoxelsT_MRMesh_MRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOnVoxelsT_MRMesh_MRVdbVolume(MeshOnVoxelsT_MRMesh_MRVdbVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_MRMesh_MRVdbVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOnVoxelsT_MRMesh_MRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRVdbVolume`/`Const_MeshOnVoxelsT_MRMesh_MRVdbVolume` to pass it to the function.
    public class _InOptConst_MeshOnVoxelsT_MRMesh_MRVdbVolume
    {
        public Const_MeshOnVoxelsT_MRMesh_MRVdbVolume? Opt;

        public _InOptConst_MeshOnVoxelsT_MRMesh_MRVdbVolume() {}
        public _InOptConst_MeshOnVoxelsT_MRMesh_MRVdbVolume(Const_MeshOnVoxelsT_MRMesh_MRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOnVoxelsT_MRMesh_MRVdbVolume(Const_MeshOnVoxelsT_MRMesh_MRVdbVolume value) {return new(value);}
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>`.
    /// This is the const half of the class.
    public class Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Destroy(_Underlying *_this);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume() {Dispose(false);}

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(MR._ByValue_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_ConstMRMesh_MRVdbVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(MR.Const_Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_VdbVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_VdbVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }

        // Access to base data
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::mesh`.
        public unsafe MR.Const_Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_mesh", ExactSpelling = true)]
            extern static MR.Const_Mesh._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_mesh(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::volume`.
        public unsafe MR.Const_VdbVolume Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_volume", ExactSpelling = true)]
            extern static MR.Const_VdbVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_volume(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_volume(_UnderlyingPtr), is_owning: false);
        }

        // Cached number of valid vertices
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::numVerts`.
        public unsafe int NumVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_numVerts", ExactSpelling = true)]
            extern static int __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_numVerts(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_numVerts(_UnderlyingPtr);
        }

        // Voxel size as scalar
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::voxelSize`.
        public unsafe float VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_voxelSize", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_voxelSize(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_voxelSize(_UnderlyingPtr);
        }

        // Transformation mesh to volume
        // All points are in voxels volume space, unless otherwise is implied
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xf_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xf_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xf_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::xf`.
        public unsafe MR.Vector3f Xf(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xf_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xf_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xf_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::xfInv`.
        public unsafe MR.AffineXf3f XfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xfInv_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xfInv_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xfInv_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::xfInv`.
        public unsafe MR.Vector3f XfInv(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xfInv_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xfInv_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_xfInv_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        // Vertex position
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::point`.
        public unsafe MR.Vector3f Point(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_point", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_point(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_point(_UnderlyingPtr, v);
        }

        // Volume value
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::getValue`.
        public unsafe float GetValue(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getValue", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getValue(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getValue(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        // Get offset vector (mesh normal for a vertex with `voxelSize` length)
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::getOffsetVector`.
        public unsafe MR.Vector3f GetOffsetVector(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getOffsetVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getOffsetVector(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getOffsetVector(_UnderlyingPtr, v);
        }

        // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
        // Pseudo-index is a signed number; for whole index, is is whole or half-whole
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::pseudoIndex`.
        public static float PseudoIndex(float index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_pseudoIndex_float", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_pseudoIndex_float(float index, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_pseudoIndex_float(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::pseudoIndex`.
        public static float PseudoIndex(int index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_pseudoIndex_int", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_pseudoIndex_int(int index, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_pseudoIndex_int(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::indexFromPseudoIndex`.
        public static float IndexFromPseudoIndex(float pseudoIndex, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_indexFromPseudoIndex", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_indexFromPseudoIndex(float pseudoIndex, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_indexFromPseudoIndex(pseudoIndex, count);
        }

        // Get row of points with `offset` stride
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::getPoints`.
        public unsafe void GetPoints(MR.Std.Vector_MRVector3f result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getPoints", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getPoints(_Underlying *_this, MR.Std.Vector_MRVector3f._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getPoints(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get volume values for a row of points
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::getValues`.
        public unsafe void GetValues(MR.Std.Vector_Float result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getValues", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getValues(_Underlying *_this, MR.Std.Vector_Float._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getValues(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get derivatives from result of `getValues`
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::getDerivatives`.
        public static unsafe void GetDerivatives(MR.Std.Vector_Float result, MR.Std.Const_Vector_Float values)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getDerivatives", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getDerivatives(MR.Std.Vector_Float._Underlying *result, MR.Std.Const_Vector_Float._Underlying *values);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getDerivatives(result._UnderlyingPtr, values._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::getBestPolynomial`.
        public static unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> GetBestPolynomial(MR.Std.Const_Vector_Float values, ulong degree)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getBestPolynomial", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getBestPolynomial(MR.Std.Const_Vector_Float._Underlying *values, ulong degree);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_getBestPolynomial(values._UnderlyingPtr, degree), is_owning: true));
        }
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>`.
    /// This is the non-const half of the class.
    public class MeshOnVoxelsT_ConstMRMesh_MRVdbVolume : Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume
    {
        internal unsafe MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(MR._ByValue_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_ConstMRMesh_MRVdbVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::VdbVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(MR.Const_Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_VdbVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRVdbVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_VdbVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_VdbVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOnVoxelsT_ConstMRMesh_MRVdbVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRVdbVolume`/`Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume
    {
        internal readonly Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_ConstMRMesh_MRVdbVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRVdbVolume`/`Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume` directly.
    public class _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume
    {
        public MeshOnVoxelsT_ConstMRMesh_MRVdbVolume? Opt;

        public _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume() {}
        public _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(MeshOnVoxelsT_ConstMRMesh_MRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(MeshOnVoxelsT_ConstMRMesh_MRVdbVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_ConstMRMesh_MRVdbVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRVdbVolume`/`Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume` to pass it to the function.
    public class _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume
    {
        public Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume? Opt;

        public _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume() {}
        public _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRVdbVolume value) {return new(value);}
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>`.
    /// This is the const half of the class.
    public class Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Destroy(_Underlying *_this);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax() {Dispose(false);}

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(MR._ByValue_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_SimpleVolumeMinMax volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Construct(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }

        // Access to base data
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::mesh`.
        public unsafe MR.Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_mesh", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_mesh(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::volume`.
        public unsafe MR.Const_SimpleVolumeMinMax Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_volume", ExactSpelling = true)]
            extern static MR.Const_SimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_volume(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_volume(_UnderlyingPtr), is_owning: false);
        }

        // Cached number of valid vertices
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::numVerts`.
        public unsafe int NumVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_numVerts", ExactSpelling = true)]
            extern static int __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_numVerts(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_numVerts(_UnderlyingPtr);
        }

        // Voxel size as scalar
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::voxelSize`.
        public unsafe float VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_voxelSize", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_voxelSize(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_voxelSize(_UnderlyingPtr);
        }

        // Transformation mesh to volume
        // All points are in voxels volume space, unless otherwise is implied
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xf_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xf_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xf_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::xf`.
        public unsafe MR.Vector3f Xf(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xf_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xf_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xf_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::xfInv`.
        public unsafe MR.AffineXf3f XfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::xfInv`.
        public unsafe MR.Vector3f XfInv(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        // Vertex position
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::point`.
        public unsafe MR.Vector3f Point(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_point", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_point(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_point(_UnderlyingPtr, v);
        }

        // Volume value
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::getValue`.
        public unsafe float GetValue(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getValue", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getValue(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getValue(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        // Get offset vector (mesh normal for a vertex with `voxelSize` length)
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::getOffsetVector`.
        public unsafe MR.Vector3f GetOffsetVector(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getOffsetVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getOffsetVector(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getOffsetVector(_UnderlyingPtr, v);
        }

        // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
        // Pseudo-index is a signed number; for whole index, is is whole or half-whole
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::pseudoIndex`.
        public static float PseudoIndex(float index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_float", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_float(float index, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_float(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::pseudoIndex`.
        public static float PseudoIndex(int index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_int", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_int(int index, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_int(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::indexFromPseudoIndex`.
        public static float IndexFromPseudoIndex(float pseudoIndex, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_indexFromPseudoIndex", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_indexFromPseudoIndex(float pseudoIndex, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_indexFromPseudoIndex(pseudoIndex, count);
        }

        // Get row of points with `offset` stride
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::getPoints`.
        public unsafe void GetPoints(MR.Std.Vector_MRVector3f result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getPoints", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getPoints(_Underlying *_this, MR.Std.Vector_MRVector3f._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getPoints(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get volume values for a row of points
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::getValues`.
        public unsafe void GetValues(MR.Std.Vector_Float result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getValues", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getValues(_Underlying *_this, MR.Std.Vector_Float._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getValues(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get derivatives from result of `getValues`
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::getDerivatives`.
        public static unsafe void GetDerivatives(MR.Std.Vector_Float result, MR.Std.Const_Vector_Float values)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getDerivatives", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getDerivatives(MR.Std.Vector_Float._Underlying *result, MR.Std.Const_Vector_Float._Underlying *values);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getDerivatives(result._UnderlyingPtr, values._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::getBestPolynomial`.
        public static unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> GetBestPolynomial(MR.Std.Const_Vector_Float values, ulong degree)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getBestPolynomial", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getBestPolynomial(MR.Std.Const_Vector_Float._Underlying *values, ulong degree);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_getBestPolynomial(values._UnderlyingPtr, degree), is_owning: true));
        }
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>`.
    /// This is the non-const half of the class.
    public class MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax : Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax
    {
        internal unsafe MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(MR._ByValue_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_SimpleVolumeMinMax volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Construct(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_SimpleVolumeMinMax_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax`/`Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax
    {
        internal readonly Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax`/`Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax` directly.
    public class _InOptMut_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax
    {
        public MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax? Opt;

        public _InOptMut_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax() {}
        public _InOptMut_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax`/`Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax` to pass it to the function.
    public class _InOptConst_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax
    {
        public Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax? Opt;

        public _InOptConst_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax() {}
        public _InOptConst_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_MRMesh_MRSimpleVolumeMinMax value) {return new(value);}
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>`.
    /// This is the const half of the class.
    public class Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Destroy(_Underlying *_this);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax() {Dispose(false);}

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(MR._ByValue_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(MR.Const_Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_SimpleVolumeMinMax volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }

        // Access to base data
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::mesh`.
        public unsafe MR.Const_Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_mesh", ExactSpelling = true)]
            extern static MR.Const_Mesh._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_mesh(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::volume`.
        public unsafe MR.Const_SimpleVolumeMinMax Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_volume", ExactSpelling = true)]
            extern static MR.Const_SimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_volume(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_volume(_UnderlyingPtr), is_owning: false);
        }

        // Cached number of valid vertices
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::numVerts`.
        public unsafe int NumVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_numVerts", ExactSpelling = true)]
            extern static int __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_numVerts(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_numVerts(_UnderlyingPtr);
        }

        // Voxel size as scalar
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::voxelSize`.
        public unsafe float VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_voxelSize", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_voxelSize(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_voxelSize(_UnderlyingPtr);
        }

        // Transformation mesh to volume
        // All points are in voxels volume space, unless otherwise is implied
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xf_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xf_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xf_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::xf`.
        public unsafe MR.Vector3f Xf(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xf_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xf_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xf_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::xfInv`.
        public unsafe MR.AffineXf3f XfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::xfInv`.
        public unsafe MR.Vector3f XfInv(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_xfInv_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        // Vertex position
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::point`.
        public unsafe MR.Vector3f Point(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_point", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_point(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_point(_UnderlyingPtr, v);
        }

        // Volume value
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::getValue`.
        public unsafe float GetValue(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getValue", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getValue(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getValue(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        // Get offset vector (mesh normal for a vertex with `voxelSize` length)
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::getOffsetVector`.
        public unsafe MR.Vector3f GetOffsetVector(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getOffsetVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getOffsetVector(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getOffsetVector(_UnderlyingPtr, v);
        }

        // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
        // Pseudo-index is a signed number; for whole index, is is whole or half-whole
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::pseudoIndex`.
        public static float PseudoIndex(float index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_float", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_float(float index, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_float(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::pseudoIndex`.
        public static float PseudoIndex(int index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_int", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_int(int index, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_pseudoIndex_int(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::indexFromPseudoIndex`.
        public static float IndexFromPseudoIndex(float pseudoIndex, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_indexFromPseudoIndex", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_indexFromPseudoIndex(float pseudoIndex, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_indexFromPseudoIndex(pseudoIndex, count);
        }

        // Get row of points with `offset` stride
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::getPoints`.
        public unsafe void GetPoints(MR.Std.Vector_MRVector3f result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getPoints", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getPoints(_Underlying *_this, MR.Std.Vector_MRVector3f._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getPoints(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get volume values for a row of points
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::getValues`.
        public unsafe void GetValues(MR.Std.Vector_Float result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getValues", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getValues(_Underlying *_this, MR.Std.Vector_Float._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getValues(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get derivatives from result of `getValues`
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::getDerivatives`.
        public static unsafe void GetDerivatives(MR.Std.Vector_Float result, MR.Std.Const_Vector_Float values)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getDerivatives", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getDerivatives(MR.Std.Vector_Float._Underlying *result, MR.Std.Const_Vector_Float._Underlying *values);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getDerivatives(result._UnderlyingPtr, values._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::getBestPolynomial`.
        public static unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> GetBestPolynomial(MR.Std.Const_Vector_Float values, ulong degree)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getBestPolynomial", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getBestPolynomial(MR.Std.Const_Vector_Float._Underlying *values, ulong degree);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_getBestPolynomial(values._UnderlyingPtr, degree), is_owning: true));
        }
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>`.
    /// This is the non-const half of the class.
    public class MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax : Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax
    {
        internal unsafe MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(MR._ByValue_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::SimpleVolumeMinMax>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(MR.Const_Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_SimpleVolumeMinMax volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_SimpleVolumeMinMax_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax`/`Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax
    {
        internal readonly Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax`/`Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax` directly.
    public class _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax
    {
        public MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax? Opt;

        public _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax() {}
        public _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax`/`Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax` to pass it to the function.
    public class _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax
    {
        public Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax? Opt;

        public _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax() {}
        public _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax(Const_MeshOnVoxelsT_ConstMRMesh_MRSimpleVolumeMinMax value) {return new(value);}
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>`.
    /// This is the const half of the class.
    public class Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Destroy(_Underlying *_this);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume() {Dispose(false);}

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume(MR._ByValue_MeshOnVoxelsT_MRMesh_MRFunctionVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_MRMesh_MRFunctionVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_FunctionVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Construct(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_FunctionVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }

        // Access to base data
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::mesh`.
        public unsafe MR.Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_mesh", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_mesh(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::volume`.
        public unsafe MR.Const_FunctionVolume Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_volume", ExactSpelling = true)]
            extern static MR.Const_FunctionVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_volume(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_volume(_UnderlyingPtr), is_owning: false);
        }

        // Cached number of valid vertices
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::numVerts`.
        public unsafe int NumVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_numVerts", ExactSpelling = true)]
            extern static int __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_numVerts(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_numVerts(_UnderlyingPtr);
        }

        // Voxel size as scalar
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::voxelSize`.
        public unsafe float VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_voxelSize", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_voxelSize(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_voxelSize(_UnderlyingPtr);
        }

        // Transformation mesh to volume
        // All points are in voxels volume space, unless otherwise is implied
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xf_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xf_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xf_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::xf`.
        public unsafe MR.Vector3f Xf(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xf_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xf_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xf_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::xfInv`.
        public unsafe MR.AffineXf3f XfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xfInv_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xfInv_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xfInv_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::xfInv`.
        public unsafe MR.Vector3f XfInv(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xfInv_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xfInv_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_xfInv_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        // Vertex position
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::point`.
        public unsafe MR.Vector3f Point(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_point", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_point(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_point(_UnderlyingPtr, v);
        }

        // Volume value
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::getValue`.
        public unsafe float GetValue(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getValue", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getValue(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getValue(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        // Get offset vector (mesh normal for a vertex with `voxelSize` length)
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::getOffsetVector`.
        public unsafe MR.Vector3f GetOffsetVector(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getOffsetVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getOffsetVector(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getOffsetVector(_UnderlyingPtr, v);
        }

        // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
        // Pseudo-index is a signed number; for whole index, is is whole or half-whole
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::pseudoIndex`.
        public static float PseudoIndex(float index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_pseudoIndex_float", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_pseudoIndex_float(float index, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_pseudoIndex_float(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::pseudoIndex`.
        public static float PseudoIndex(int index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_pseudoIndex_int", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_pseudoIndex_int(int index, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_pseudoIndex_int(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::indexFromPseudoIndex`.
        public static float IndexFromPseudoIndex(float pseudoIndex, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_indexFromPseudoIndex", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_indexFromPseudoIndex(float pseudoIndex, int count);
            return __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_indexFromPseudoIndex(pseudoIndex, count);
        }

        // Get row of points with `offset` stride
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::getPoints`.
        public unsafe void GetPoints(MR.Std.Vector_MRVector3f result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getPoints", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getPoints(_Underlying *_this, MR.Std.Vector_MRVector3f._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getPoints(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get volume values for a row of points
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::getValues`.
        public unsafe void GetValues(MR.Std.Vector_Float result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getValues", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getValues(_Underlying *_this, MR.Std.Vector_Float._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getValues(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get derivatives from result of `getValues`
        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::getDerivatives`.
        public static unsafe void GetDerivatives(MR.Std.Vector_Float result, MR.Std.Const_Vector_Float values)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getDerivatives", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getDerivatives(MR.Std.Vector_Float._Underlying *result, MR.Std.Const_Vector_Float._Underlying *values);
            __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getDerivatives(result._UnderlyingPtr, values._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::getBestPolynomial`.
        public static unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> GetBestPolynomial(MR.Std.Const_Vector_Float values, ulong degree)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getBestPolynomial", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getBestPolynomial(MR.Std.Const_Vector_Float._Underlying *values, ulong degree);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_getBestPolynomial(values._UnderlyingPtr, degree), is_owning: true));
        }
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>`.
    /// This is the non-const half of the class.
    public class MeshOnVoxelsT_MRMesh_MRFunctionVolume : Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume
    {
        internal unsafe MeshOnVoxelsT_MRMesh_MRFunctionVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_MRMesh_MRFunctionVolume(MR._ByValue_MeshOnVoxelsT_MRMesh_MRFunctionVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_MRMesh_MRFunctionVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_MRMesh_MRFunctionVolume(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_FunctionVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_MRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Construct(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_FunctionVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_MR_Mesh_MR_FunctionVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOnVoxelsT_MRMesh_MRFunctionVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRFunctionVolume`/`Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOnVoxelsT_MRMesh_MRFunctionVolume
    {
        internal readonly Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOnVoxelsT_MRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOnVoxelsT_MRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_MRMesh_MRFunctionVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOnVoxelsT_MRMesh_MRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRFunctionVolume`/`Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume` directly.
    public class _InOptMut_MeshOnVoxelsT_MRMesh_MRFunctionVolume
    {
        public MeshOnVoxelsT_MRMesh_MRFunctionVolume? Opt;

        public _InOptMut_MeshOnVoxelsT_MRMesh_MRFunctionVolume() {}
        public _InOptMut_MeshOnVoxelsT_MRMesh_MRFunctionVolume(MeshOnVoxelsT_MRMesh_MRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOnVoxelsT_MRMesh_MRFunctionVolume(MeshOnVoxelsT_MRMesh_MRFunctionVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_MRMesh_MRFunctionVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOnVoxelsT_MRMesh_MRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_MRMesh_MRFunctionVolume`/`Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume` to pass it to the function.
    public class _InOptConst_MeshOnVoxelsT_MRMesh_MRFunctionVolume
    {
        public Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume? Opt;

        public _InOptConst_MeshOnVoxelsT_MRMesh_MRFunctionVolume() {}
        public _InOptConst_MeshOnVoxelsT_MRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOnVoxelsT_MRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_MRMesh_MRFunctionVolume value) {return new(value);}
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>`.
    /// This is the const half of the class.
    public class Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Destroy(_Underlying *_this);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume() {Dispose(false);}

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(MR._ByValue_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(MR.Const_Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_FunctionVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_FunctionVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }

        // Access to base data
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::mesh`.
        public unsafe MR.Const_Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_mesh", ExactSpelling = true)]
            extern static MR.Const_Mesh._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_mesh(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::volume`.
        public unsafe MR.Const_FunctionVolume Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_volume", ExactSpelling = true)]
            extern static MR.Const_FunctionVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_volume(_Underlying *_this);
            return new(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_volume(_UnderlyingPtr), is_owning: false);
        }

        // Cached number of valid vertices
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::numVerts`.
        public unsafe int NumVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_numVerts", ExactSpelling = true)]
            extern static int __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_numVerts(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_numVerts(_UnderlyingPtr);
        }

        // Voxel size as scalar
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::voxelSize`.
        public unsafe float VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_voxelSize", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_voxelSize(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_voxelSize(_UnderlyingPtr);
        }

        // Transformation mesh to volume
        // All points are in voxels volume space, unless otherwise is implied
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xf_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xf_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xf_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::xf`.
        public unsafe MR.Vector3f Xf(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xf_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xf_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xf_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::xfInv`.
        public unsafe MR.AffineXf3f XfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xfInv_0", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xfInv_0(_Underlying *_this);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xfInv_0(_UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::xfInv`.
        public unsafe MR.Vector3f XfInv(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xfInv_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xfInv_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_xfInv_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        // Vertex position
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::point`.
        public unsafe MR.Vector3f Point(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_point", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_point(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_point(_UnderlyingPtr, v);
        }

        // Volume value
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::getValue`.
        public unsafe float GetValue(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getValue", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getValue(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getValue(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        // Get offset vector (mesh normal for a vertex with `voxelSize` length)
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::getOffsetVector`.
        public unsafe MR.Vector3f GetOffsetVector(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getOffsetVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getOffsetVector(_Underlying *_this, MR.VertId v);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getOffsetVector(_UnderlyingPtr, v);
        }

        // Get a pseudo-index for a zero-based point index in a zero-centered row of `count` points
        // Pseudo-index is a signed number; for whole index, is is whole or half-whole
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::pseudoIndex`.
        public static float PseudoIndex(float index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_pseudoIndex_float", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_pseudoIndex_float(float index, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_pseudoIndex_float(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::pseudoIndex`.
        public static float PseudoIndex(int index, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_pseudoIndex_int", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_pseudoIndex_int(int index, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_pseudoIndex_int(index, count);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::indexFromPseudoIndex`.
        public static float IndexFromPseudoIndex(float pseudoIndex, int count)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_indexFromPseudoIndex", ExactSpelling = true)]
            extern static float __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_indexFromPseudoIndex(float pseudoIndex, int count);
            return __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_indexFromPseudoIndex(pseudoIndex, count);
        }

        // Get row of points with `offset` stride
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::getPoints`.
        public unsafe void GetPoints(MR.Std.Vector_MRVector3f result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getPoints", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getPoints(_Underlying *_this, MR.Std.Vector_MRVector3f._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getPoints(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get volume values for a row of points
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::getValues`.
        public unsafe void GetValues(MR.Std.Vector_Float result, MR.Const_Vector3f pos, MR.Const_Vector3f offset)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getValues", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getValues(_Underlying *_this, MR.Std.Vector_Float._Underlying *result, MR.Const_Vector3f._Underlying *pos, MR.Const_Vector3f._Underlying *offset);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getValues(_UnderlyingPtr, result._UnderlyingPtr, pos._UnderlyingPtr, offset._UnderlyingPtr);
        }

        // Get derivatives from result of `getValues`
        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::getDerivatives`.
        public static unsafe void GetDerivatives(MR.Std.Vector_Float result, MR.Std.Const_Vector_Float values)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getDerivatives", ExactSpelling = true)]
            extern static void __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getDerivatives(MR.Std.Vector_Float._Underlying *result, MR.Std.Const_Vector_Float._Underlying *values);
            __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getDerivatives(result._UnderlyingPtr, values._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::getBestPolynomial`.
        public static unsafe MR.Misc._Moved<MR.PolynomialWrapper_Float> GetBestPolynomial(MR.Std.Const_Vector_Float values, ulong degree)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getBestPolynomial", ExactSpelling = true)]
            extern static MR.PolynomialWrapper_Float._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getBestPolynomial(MR.Std.Const_Vector_Float._Underlying *values, ulong degree);
            return MR.Misc.Move(new MR.PolynomialWrapper_Float(__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_getBestPolynomial(values._UnderlyingPtr, degree), is_owning: true));
        }
    }

    /// Helper class to organize mesh and voxels volume access and build point sequences
    /// \note this class is not thread-safe but accessing same volume from different instances is ok
    /// Generated from class `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>`.
    /// This is the non-const half of the class.
    public class MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume : Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume
    {
        internal unsafe MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(MR._ByValue_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(MR.Misc._PassBy other_pass_by, MR.MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume._Underlying *other);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_ConstructFromAnother(other.PassByMode, other.Value is not null ? other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshOnVoxelsT<const MR::Mesh, MR::FunctionVolume>::MeshOnVoxelsT`.
        public unsafe MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(MR.Const_Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_FunctionVolume volume, MR.Const_AffineXf3f volumeXf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Construct", ExactSpelling = true)]
            extern static MR.MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume._Underlying *__MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_FunctionVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf);
            _UnderlyingPtr = __MR_MeshOnVoxelsT_const_MR_Mesh_MR_FunctionVolume_Construct(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume`/`Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume
    {
        internal readonly Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume`/`Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume` directly.
    public class _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume
    {
        public MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume? Opt;

        public _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume() {}
        public _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptMut_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume`/`Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume` to pass it to the function.
    public class _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume
    {
        public Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume? Opt;

        public _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume() {}
        public _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptConst_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume(Const_MeshOnVoxelsT_ConstMRMesh_MRFunctionVolume value) {return new(value);}
    }

    /// Moves each vertex along its normal to the minimize (with sign, i.e. maximize the absolute value with negative sign) the derivative
    /// of voxels.
    /// @return Vertices that were moved by the algorithm
    /// Generated from function `MR::moveMeshToVoxelMaxDeriv<MR::VdbVolume>`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> MoveMeshToVoxelMaxDeriv(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_VdbVolume volume, MR.Const_AffineXf3f volumeXf, MR.Const_MoveMeshToVoxelMaxDerivSettings settings, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_moveMeshToVoxelMaxDeriv_MR_VdbVolume", ExactSpelling = true)]
        extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_moveMeshToVoxelMaxDeriv_MR_VdbVolume(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_VdbVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf, MR.Const_MoveMeshToVoxelMaxDerivSettings._Underlying *settings, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_moveMeshToVoxelMaxDeriv_MR_VdbVolume(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr, settings._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Moves each vertex along its normal to the minimize (with sign, i.e. maximize the absolute value with negative sign) the derivative
    /// of voxels.
    /// @return Vertices that were moved by the algorithm
    /// Generated from function `MR::moveMeshToVoxelMaxDeriv<MR::SimpleVolumeMinMax>`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> MoveMeshToVoxelMaxDeriv(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_SimpleVolumeMinMax volume, MR.Const_AffineXf3f volumeXf, MR.Const_MoveMeshToVoxelMaxDerivSettings settings, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_moveMeshToVoxelMaxDeriv_MR_SimpleVolumeMinMax", ExactSpelling = true)]
        extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_moveMeshToVoxelMaxDeriv_MR_SimpleVolumeMinMax(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf, MR.Const_MoveMeshToVoxelMaxDerivSettings._Underlying *settings, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_moveMeshToVoxelMaxDeriv_MR_SimpleVolumeMinMax(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr, settings._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Moves each vertex along its normal to the minimize (with sign, i.e. maximize the absolute value with negative sign) the derivative
    /// of voxels.
    /// @return Vertices that were moved by the algorithm
    /// Generated from function `MR::moveMeshToVoxelMaxDeriv<MR::FunctionVolume>`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> MoveMeshToVoxelMaxDeriv(MR.Mesh mesh, MR.Const_AffineXf3f meshXf, MR.Const_FunctionVolume volume, MR.Const_AffineXf3f volumeXf, MR.Const_MoveMeshToVoxelMaxDerivSettings settings, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_moveMeshToVoxelMaxDeriv_MR_FunctionVolume", ExactSpelling = true)]
        extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_moveMeshToVoxelMaxDeriv_MR_FunctionVolume(MR.Mesh._Underlying *mesh, MR.Const_AffineXf3f._Underlying *meshXf, MR.Const_FunctionVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *volumeXf, MR.Const_MoveMeshToVoxelMaxDerivSettings._Underlying *settings, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_moveMeshToVoxelMaxDeriv_MR_FunctionVolume(mesh._UnderlyingPtr, meshXf._UnderlyingPtr, volume._UnderlyingPtr, volumeXf._UnderlyingPtr, settings._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
    }
}
