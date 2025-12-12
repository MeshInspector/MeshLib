public static partial class MR
{
    /// Generated from class `MR::SharpenMarchingCubesMeshSettings`.
    /// This is the const half of the class.
    public class Const_SharpenMarchingCubesMeshSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SharpenMarchingCubesMeshSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_SharpenMarchingCubesMeshSettings_Destroy(_Underlying *_this);
            __MR_SharpenMarchingCubesMeshSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SharpenMarchingCubesMeshSettings() {Dispose(false);}

        /// minimal surface deviation to introduce new vertex in a voxel;
        /// recommended set equal to ( voxel size / 25 )
        public unsafe float MinNewVertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_minNewVertDev", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_Get_minNewVertDev(_Underlying *_this);
                return *__MR_SharpenMarchingCubesMeshSettings_Get_minNewVertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes);
        /// recommended set equal to ( 5 * voxel size )
        public unsafe float MaxNewRank2VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_maxNewRank2VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_Get_maxNewRank2VertDev(_Underlying *_this);
                return *__MR_SharpenMarchingCubesMeshSettings_Get_maxNewRank2VertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes);
        /// recommended set equal to ( 2 * voxel size )
        public unsafe float MaxNewRank3VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_maxNewRank3VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_Get_maxNewRank3VertDev(_Underlying *_this);
                return *__MR_SharpenMarchingCubesMeshSettings_Get_maxNewRank3VertDev(_UnderlyingPtr);
            }
        }

        /// relative to reference mesh
        public unsafe float Offset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_offset", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_Get_offset(_Underlying *_this);
                return *__MR_SharpenMarchingCubesMeshSettings_Get_offset(_UnderlyingPtr);
            }
        }

        /// correct positions of the input vertices using reference mesh by not more than this distance;
        /// big correction can be wrong and result from self-intersections in the reference mesh
        /// recommended set equal to ( voxel size / 2 )
        public unsafe float MaxOldVertPosCorrection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_maxOldVertPosCorrection", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_Get_maxOldVertPosCorrection(_Underlying *_this);
                return *__MR_SharpenMarchingCubesMeshSettings_Get_maxOldVertPosCorrection(_UnderlyingPtr);
            }
        }

        /// the number of iterations to best select positions for new vertices,
        /// the probability of self-intersections and spikes are higher if posSelIters = 0
        public unsafe int PosSelIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_posSelIters", ExactSpelling = true)]
                extern static int *__MR_SharpenMarchingCubesMeshSettings_Get_posSelIters(_Underlying *_this);
                return *__MR_SharpenMarchingCubesMeshSettings_Get_posSelIters(_UnderlyingPtr);
            }
        }

        /// if non-null then created sharp edges will be saved here
        public unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_Get_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_SharpenMarchingCubesMeshSettings_Get_outSharpEdges(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_Get_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SharpenMarchingCubesMeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SharpenMarchingCubesMeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::SharpenMarchingCubesMeshSettings` elementwise.
        public unsafe Const_SharpenMarchingCubesMeshSettings(float minNewVertDev, float maxNewRank2VertDev, float maxNewRank3VertDev, float offset, float maxOldVertPosCorrection, int posSelIters, MR.UndirectedEdgeBitSet? outSharpEdges) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_ConstructFrom(float minNewVertDev, float maxNewRank2VertDev, float maxNewRank3VertDev, float offset, float maxOldVertPosCorrection, int posSelIters, MR.UndirectedEdgeBitSet._Underlying *outSharpEdges);
            _UnderlyingPtr = __MR_SharpenMarchingCubesMeshSettings_ConstructFrom(minNewVertDev, maxNewRank2VertDev, maxNewRank3VertDev, offset, maxOldVertPosCorrection, posSelIters, outSharpEdges is not null ? outSharpEdges._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SharpenMarchingCubesMeshSettings::SharpenMarchingCubesMeshSettings`.
        public unsafe Const_SharpenMarchingCubesMeshSettings(MR.Const_SharpenMarchingCubesMeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_ConstructFromAnother(MR.SharpenMarchingCubesMeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SharpenMarchingCubesMeshSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::SharpenMarchingCubesMeshSettings`.
    /// This is the non-const half of the class.
    public class SharpenMarchingCubesMeshSettings : Const_SharpenMarchingCubesMeshSettings
    {
        internal unsafe SharpenMarchingCubesMeshSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// minimal surface deviation to introduce new vertex in a voxel;
        /// recommended set equal to ( voxel size / 25 )
        public new unsafe ref float MinNewVertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_minNewVertDev", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_GetMutable_minNewVertDev(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_minNewVertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes);
        /// recommended set equal to ( 5 * voxel size )
        public new unsafe ref float MaxNewRank2VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_maxNewRank2VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_GetMutable_maxNewRank2VertDev(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_maxNewRank2VertDev(_UnderlyingPtr);
            }
        }

        /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes);
        /// recommended set equal to ( 2 * voxel size )
        public new unsafe ref float MaxNewRank3VertDev
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_maxNewRank3VertDev", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_GetMutable_maxNewRank3VertDev(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_maxNewRank3VertDev(_UnderlyingPtr);
            }
        }

        /// relative to reference mesh
        public new unsafe ref float Offset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_offset", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_GetMutable_offset(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_offset(_UnderlyingPtr);
            }
        }

        /// correct positions of the input vertices using reference mesh by not more than this distance;
        /// big correction can be wrong and result from self-intersections in the reference mesh
        /// recommended set equal to ( voxel size / 2 )
        public new unsafe ref float MaxOldVertPosCorrection
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_maxOldVertPosCorrection", ExactSpelling = true)]
                extern static float *__MR_SharpenMarchingCubesMeshSettings_GetMutable_maxOldVertPosCorrection(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_maxOldVertPosCorrection(_UnderlyingPtr);
            }
        }

        /// the number of iterations to best select positions for new vertices,
        /// the probability of self-intersections and spikes are higher if posSelIters = 0
        public new unsafe ref int PosSelIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_posSelIters", ExactSpelling = true)]
                extern static int *__MR_SharpenMarchingCubesMeshSettings_GetMutable_posSelIters(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_posSelIters(_UnderlyingPtr);
            }
        }

        /// if non-null then created sharp edges will be saved here
        public new unsafe ref void * OutSharpEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_GetMutable_outSharpEdges", ExactSpelling = true)]
                extern static void **__MR_SharpenMarchingCubesMeshSettings_GetMutable_outSharpEdges(_Underlying *_this);
                return ref *__MR_SharpenMarchingCubesMeshSettings_GetMutable_outSharpEdges(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SharpenMarchingCubesMeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SharpenMarchingCubesMeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::SharpenMarchingCubesMeshSettings` elementwise.
        public unsafe SharpenMarchingCubesMeshSettings(float minNewVertDev, float maxNewRank2VertDev, float maxNewRank3VertDev, float offset, float maxOldVertPosCorrection, int posSelIters, MR.UndirectedEdgeBitSet? outSharpEdges) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_ConstructFrom(float minNewVertDev, float maxNewRank2VertDev, float maxNewRank3VertDev, float offset, float maxOldVertPosCorrection, int posSelIters, MR.UndirectedEdgeBitSet._Underlying *outSharpEdges);
            _UnderlyingPtr = __MR_SharpenMarchingCubesMeshSettings_ConstructFrom(minNewVertDev, maxNewRank2VertDev, maxNewRank3VertDev, offset, maxOldVertPosCorrection, posSelIters, outSharpEdges is not null ? outSharpEdges._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SharpenMarchingCubesMeshSettings::SharpenMarchingCubesMeshSettings`.
        public unsafe SharpenMarchingCubesMeshSettings(MR.Const_SharpenMarchingCubesMeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_ConstructFromAnother(MR.SharpenMarchingCubesMeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SharpenMarchingCubesMeshSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SharpenMarchingCubesMeshSettings::operator=`.
        public unsafe MR.SharpenMarchingCubesMeshSettings Assign(MR.Const_SharpenMarchingCubesMeshSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SharpenMarchingCubesMeshSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SharpenMarchingCubesMeshSettings._Underlying *__MR_SharpenMarchingCubesMeshSettings_AssignFromAnother(_Underlying *_this, MR.SharpenMarchingCubesMeshSettings._Underlying *_other);
            return new(__MR_SharpenMarchingCubesMeshSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SharpenMarchingCubesMeshSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SharpenMarchingCubesMeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SharpenMarchingCubesMeshSettings`/`Const_SharpenMarchingCubesMeshSettings` directly.
    public class _InOptMut_SharpenMarchingCubesMeshSettings
    {
        public SharpenMarchingCubesMeshSettings? Opt;

        public _InOptMut_SharpenMarchingCubesMeshSettings() {}
        public _InOptMut_SharpenMarchingCubesMeshSettings(SharpenMarchingCubesMeshSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SharpenMarchingCubesMeshSettings(SharpenMarchingCubesMeshSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SharpenMarchingCubesMeshSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SharpenMarchingCubesMeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SharpenMarchingCubesMeshSettings`/`Const_SharpenMarchingCubesMeshSettings` to pass it to the function.
    public class _InOptConst_SharpenMarchingCubesMeshSettings
    {
        public Const_SharpenMarchingCubesMeshSettings? Opt;

        public _InOptConst_SharpenMarchingCubesMeshSettings() {}
        public _InOptConst_SharpenMarchingCubesMeshSettings(Const_SharpenMarchingCubesMeshSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SharpenMarchingCubesMeshSettings(Const_SharpenMarchingCubesMeshSettings value) {return new(value);}
    }

    /// adjust the mesh \param vox produced by marching cubes method (NOT dual marching cubes) by
    /// 1) correcting positions of all vertices to given offset relative to \param ref mesh (if correctOldVertPos == true);
    /// 2) introducing new vertices in the voxels where the normals change abruptly.
    /// \param face2voxel mapping from Face Id to Voxel Id where it is located
    /// Generated from function `MR::sharpenMarchingCubesMesh`.
    public static unsafe void SharpenMarchingCubesMesh(MR.Const_MeshPart ref_, MR.Mesh vox, MR.Vector_MRVoxelId_MRFaceId face2voxel, MR.Const_SharpenMarchingCubesMeshSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sharpenMarchingCubesMesh", ExactSpelling = true)]
        extern static void __MR_sharpenMarchingCubesMesh(MR.Const_MeshPart._Underlying *ref_, MR.Mesh._Underlying *vox, MR.Vector_MRVoxelId_MRFaceId._Underlying *face2voxel, MR.Const_SharpenMarchingCubesMeshSettings._Underlying *settings);
        __MR_sharpenMarchingCubesMesh(ref_._UnderlyingPtr, vox._UnderlyingPtr, face2voxel._UnderlyingPtr, settings._UnderlyingPtr);
    }
}
