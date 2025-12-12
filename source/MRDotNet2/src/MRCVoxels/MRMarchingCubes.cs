public static partial class MR
{
    /// Generated from class `MR::MarchingCubesParams`.
    /// This is the const half of the class.
    public class Const_MarchingCubesParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MarchingCubesParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MarchingCubesParams_Destroy(_Underlying *_this);
            __MR_MarchingCubesParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MarchingCubesParams() {Dispose(false);}

        /// origin point of voxels box in 3D space with output mesh
        public unsafe MR.Const_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_origin", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MarchingCubesParams_Get_origin(_Underlying *_this);
                return new(__MR_MarchingCubesParams_Get_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MarchingCubesParams_Get_cb(_Underlying *_this);
                return new(__MR_MarchingCubesParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// target iso-value of the surface to be extracted from volume
        public unsafe float Iso
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_iso", ExactSpelling = true)]
                extern static float *__MR_MarchingCubesParams_Get_iso(_Underlying *_this);
                return *__MR_MarchingCubesParams_Get_iso(_UnderlyingPtr);
            }
        }

        /// should be false for dense volumes, and true for distance volume
        public unsafe bool LessInside
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_lessInside", ExactSpelling = true)]
                extern static bool *__MR_MarchingCubesParams_Get_lessInside(_Underlying *_this);
                return *__MR_MarchingCubesParams_Get_lessInside(_UnderlyingPtr);
            }
        }

        /// optional output map FaceId->VoxelId
        public unsafe ref void * OutVoxelPerFaceMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_outVoxelPerFaceMap", ExactSpelling = true)]
                extern static void **__MR_MarchingCubesParams_Get_outVoxelPerFaceMap(_Underlying *_this);
                return ref *__MR_MarchingCubesParams_Get_outVoxelPerFaceMap(_UnderlyingPtr);
            }
        }

        /// function to calculate position of result mesh points
        /// if the function isn't set, a linear positioner will be used
        /// note: this function is called in parallel from different threads
        public unsafe MR.Std.Const_Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat Positioner
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_positioner", ExactSpelling = true)]
                extern static MR.Std.Const_Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat._Underlying *__MR_MarchingCubesParams_Get_positioner(_Underlying *_this);
                return new(__MR_MarchingCubesParams_Get_positioner(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if the mesh exceeds this number of vertices, an error returns
        public unsafe int MaxVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_maxVertices", ExactSpelling = true)]
                extern static int *__MR_MarchingCubesParams_Get_maxVertices(_Underlying *_this);
                return *__MR_MarchingCubesParams_Get_maxVertices(_UnderlyingPtr);
            }
        }

        public unsafe MR.MarchingCubesParams.CachingMode CachingMode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_cachingMode", ExactSpelling = true)]
                extern static MR.MarchingCubesParams.CachingMode *__MR_MarchingCubesParams_Get_cachingMode(_Underlying *_this);
                return *__MR_MarchingCubesParams_Get_cachingMode(_UnderlyingPtr);
            }
        }

        /// this optional function is called when volume is no longer needed to deallocate it and reduce peak memory consumption
        public unsafe MR.Std.Const_Function_VoidFunc FreeVolume
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_Get_freeVolume", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFunc._Underlying *__MR_MarchingCubesParams_Get_freeVolume(_Underlying *_this);
                return new(__MR_MarchingCubesParams_Get_freeVolume(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MarchingCubesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MarchingCubesParams_DefaultConstruct();
        }

        /// Constructs `MR::MarchingCubesParams` elementwise.
        public unsafe Const_MarchingCubesParams(MR.Vector3f origin, MR.Std._ByValue_Function_BoolFuncFromFloat cb, float iso, bool lessInside, MR.Vector_MRVoxelId_MRFaceId? outVoxelPerFaceMap, MR.Std._ByValue_Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat positioner, int maxVertices, MR.MarchingCubesParams.CachingMode cachingMode, MR.Std._ByValue_Function_VoidFunc freeVolume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_ConstructFrom(MR.Vector3f origin, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float iso, byte lessInside, MR.Vector_MRVoxelId_MRFaceId._Underlying *outVoxelPerFaceMap, MR.Misc._PassBy positioner_pass_by, MR.Std.Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat._Underlying *positioner, int maxVertices, MR.MarchingCubesParams.CachingMode cachingMode, MR.Misc._PassBy freeVolume_pass_by, MR.Std.Function_VoidFunc._Underlying *freeVolume);
            _UnderlyingPtr = __MR_MarchingCubesParams_ConstructFrom(origin, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, iso, lessInside ? (byte)1 : (byte)0, outVoxelPerFaceMap is not null ? outVoxelPerFaceMap._UnderlyingPtr : null, positioner.PassByMode, positioner.Value is not null ? positioner.Value._UnderlyingPtr : null, maxVertices, cachingMode, freeVolume.PassByMode, freeVolume.Value is not null ? freeVolume.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MarchingCubesParams::MarchingCubesParams`.
        public unsafe Const_MarchingCubesParams(MR._ByValue_MarchingCubesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MarchingCubesParams._Underlying *_other);
            _UnderlyingPtr = __MR_MarchingCubesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// caching mode to reduce the number of accesses to voxel volume data on the first pass of the algorithm by consuming more memory on cache;
        /// note: the cache for the second pass of the algorithm (bit sets of invalid and lower-than-iso voxels are always allocated)
        public enum CachingMode : int
        {
            /// choose caching mode automatically depending on volume type
            /// (current defaults: Normal for FunctionVolume and VdbVolume, None for others)
            Automatic = 0,
            /// don't cache any data
            None = 1,
            /// allocates 2 full slices per parallel thread
            Normal = 2,
        }
    }

    /// Generated from class `MR::MarchingCubesParams`.
    /// This is the non-const half of the class.
    public class MarchingCubesParams : Const_MarchingCubesParams
    {
        internal unsafe MarchingCubesParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// origin point of voxels box in 3D space with output mesh
        public new unsafe MR.Mut_Vector3f Origin
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_origin", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MarchingCubesParams_GetMutable_origin(_Underlying *_this);
                return new(__MR_MarchingCubesParams_GetMutable_origin(_UnderlyingPtr), is_owning: false);
            }
        }

        /// progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MarchingCubesParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_MarchingCubesParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// target iso-value of the surface to be extracted from volume
        public new unsafe ref float Iso
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_iso", ExactSpelling = true)]
                extern static float *__MR_MarchingCubesParams_GetMutable_iso(_Underlying *_this);
                return ref *__MR_MarchingCubesParams_GetMutable_iso(_UnderlyingPtr);
            }
        }

        /// should be false for dense volumes, and true for distance volume
        public new unsafe ref bool LessInside
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_lessInside", ExactSpelling = true)]
                extern static bool *__MR_MarchingCubesParams_GetMutable_lessInside(_Underlying *_this);
                return ref *__MR_MarchingCubesParams_GetMutable_lessInside(_UnderlyingPtr);
            }
        }

        /// optional output map FaceId->VoxelId
        public new unsafe ref void * OutVoxelPerFaceMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_outVoxelPerFaceMap", ExactSpelling = true)]
                extern static void **__MR_MarchingCubesParams_GetMutable_outVoxelPerFaceMap(_Underlying *_this);
                return ref *__MR_MarchingCubesParams_GetMutable_outVoxelPerFaceMap(_UnderlyingPtr);
            }
        }

        /// function to calculate position of result mesh points
        /// if the function isn't set, a linear positioner will be used
        /// note: this function is called in parallel from different threads
        public new unsafe MR.Std.Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat Positioner
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_positioner", ExactSpelling = true)]
                extern static MR.Std.Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat._Underlying *__MR_MarchingCubesParams_GetMutable_positioner(_Underlying *_this);
                return new(__MR_MarchingCubesParams_GetMutable_positioner(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if the mesh exceeds this number of vertices, an error returns
        public new unsafe ref int MaxVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_maxVertices", ExactSpelling = true)]
                extern static int *__MR_MarchingCubesParams_GetMutable_maxVertices(_Underlying *_this);
                return ref *__MR_MarchingCubesParams_GetMutable_maxVertices(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.MarchingCubesParams.CachingMode CachingMode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_cachingMode", ExactSpelling = true)]
                extern static MR.MarchingCubesParams.CachingMode *__MR_MarchingCubesParams_GetMutable_cachingMode(_Underlying *_this);
                return ref *__MR_MarchingCubesParams_GetMutable_cachingMode(_UnderlyingPtr);
            }
        }

        /// this optional function is called when volume is no longer needed to deallocate it and reduce peak memory consumption
        public new unsafe MR.Std.Function_VoidFunc FreeVolume
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_GetMutable_freeVolume", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFunc._Underlying *__MR_MarchingCubesParams_GetMutable_freeVolume(_Underlying *_this);
                return new(__MR_MarchingCubesParams_GetMutable_freeVolume(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MarchingCubesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MarchingCubesParams_DefaultConstruct();
        }

        /// Constructs `MR::MarchingCubesParams` elementwise.
        public unsafe MarchingCubesParams(MR.Vector3f origin, MR.Std._ByValue_Function_BoolFuncFromFloat cb, float iso, bool lessInside, MR.Vector_MRVoxelId_MRFaceId? outVoxelPerFaceMap, MR.Std._ByValue_Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat positioner, int maxVertices, MR.MarchingCubesParams.CachingMode cachingMode, MR.Std._ByValue_Function_VoidFunc freeVolume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_ConstructFrom(MR.Vector3f origin, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float iso, byte lessInside, MR.Vector_MRVoxelId_MRFaceId._Underlying *outVoxelPerFaceMap, MR.Misc._PassBy positioner_pass_by, MR.Std.Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat._Underlying *positioner, int maxVertices, MR.MarchingCubesParams.CachingMode cachingMode, MR.Misc._PassBy freeVolume_pass_by, MR.Std.Function_VoidFunc._Underlying *freeVolume);
            _UnderlyingPtr = __MR_MarchingCubesParams_ConstructFrom(origin, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, iso, lessInside ? (byte)1 : (byte)0, outVoxelPerFaceMap is not null ? outVoxelPerFaceMap._UnderlyingPtr : null, positioner.PassByMode, positioner.Value is not null ? positioner.Value._UnderlyingPtr : null, maxVertices, cachingMode, freeVolume.PassByMode, freeVolume.Value is not null ? freeVolume.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MarchingCubesParams::MarchingCubesParams`.
        public unsafe MarchingCubesParams(MR._ByValue_MarchingCubesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MarchingCubesParams._Underlying *_other);
            _UnderlyingPtr = __MR_MarchingCubesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MarchingCubesParams::operator=`.
        public unsafe MR.MarchingCubesParams Assign(MR._ByValue_MarchingCubesParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MarchingCubesParams._Underlying *__MR_MarchingCubesParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MarchingCubesParams._Underlying *_other);
            return new(__MR_MarchingCubesParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MarchingCubesParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MarchingCubesParams`/`Const_MarchingCubesParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MarchingCubesParams
    {
        internal readonly Const_MarchingCubesParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MarchingCubesParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MarchingCubesParams(Const_MarchingCubesParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MarchingCubesParams(Const_MarchingCubesParams arg) {return new(arg);}
        public _ByValue_MarchingCubesParams(MR.Misc._Moved<MarchingCubesParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MarchingCubesParams(MR.Misc._Moved<MarchingCubesParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MarchingCubesParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MarchingCubesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MarchingCubesParams`/`Const_MarchingCubesParams` directly.
    public class _InOptMut_MarchingCubesParams
    {
        public MarchingCubesParams? Opt;

        public _InOptMut_MarchingCubesParams() {}
        public _InOptMut_MarchingCubesParams(MarchingCubesParams value) {Opt = value;}
        public static implicit operator _InOptMut_MarchingCubesParams(MarchingCubesParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MarchingCubesParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MarchingCubesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MarchingCubesParams`/`Const_MarchingCubesParams` to pass it to the function.
    public class _InOptConst_MarchingCubesParams
    {
        public Const_MarchingCubesParams? Opt;

        public _InOptConst_MarchingCubesParams() {}
        public _InOptConst_MarchingCubesParams(Const_MarchingCubesParams value) {Opt = value;}
        public static implicit operator _InOptConst_MarchingCubesParams(Const_MarchingCubesParams value) {return new(value);}
    }

    /// converts volume split on parts by planes z=const into mesh,
    /// last z-layer of previous part must be repeated as first z-layer of next part
    /// usage:
    /// MarchingCubesByParts x( dims, params);
    /// x.addPart( part1 );
    /// ...
    /// x.addPart( partN );
    /// Mesh mesh = Mesh::fromTriMesh( *x.finalize() );
    /// Generated from class `MR::MarchingCubesByParts`.
    /// This is the const half of the class.
    public class Const_MarchingCubesByParts : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MarchingCubesByParts(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_Destroy", ExactSpelling = true)]
            extern static void __MR_MarchingCubesByParts_Destroy(_Underlying *_this);
            __MR_MarchingCubesByParts_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MarchingCubesByParts() {Dispose(false);}

        /// Generated from constructor `MR::MarchingCubesByParts::MarchingCubesByParts`.
        public unsafe Const_MarchingCubesByParts(MR._ByValue_MarchingCubesByParts s) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MarchingCubesByParts._Underlying *__MR_MarchingCubesByParts_ConstructFromAnother(MR.Misc._PassBy s_pass_by, MR.MarchingCubesByParts._Underlying *s);
            _UnderlyingPtr = __MR_MarchingCubesByParts_ConstructFromAnother(s.PassByMode, s.Value is not null ? s.Value._UnderlyingPtr : null);
        }

        /// prepares convention for given volume dimensions and given parameters
        /// \param layersPerBlock all z-slices of the volume will be partitioned on blocks of given size to process blocks in parallel (0 means auto-select layersPerBlock)
        /// Generated from constructor `MR::MarchingCubesByParts::MarchingCubesByParts`.
        /// Parameter `layersPerBlock` defaults to `0`.
        public unsafe Const_MarchingCubesByParts(MR.Const_Vector3i dims, MR.Const_MarchingCubesParams params_, int? layersPerBlock = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_Construct", ExactSpelling = true)]
            extern static MR.MarchingCubesByParts._Underlying *__MR_MarchingCubesByParts_Construct(MR.Const_Vector3i._Underlying *dims, MR.Const_MarchingCubesParams._Underlying *params_, int *layersPerBlock);
            int __deref_layersPerBlock = layersPerBlock.GetValueOrDefault();
            _UnderlyingPtr = __MR_MarchingCubesByParts_Construct(dims._UnderlyingPtr, params_._UnderlyingPtr, layersPerBlock.HasValue ? &__deref_layersPerBlock : null);
        }

        /// the number of z-slices of the volume in the blocks
        /// Generated from method `MR::MarchingCubesByParts::layersPerBlock`.
        public unsafe int LayersPerBlock()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_layersPerBlock", ExactSpelling = true)]
            extern static int __MR_MarchingCubesByParts_layersPerBlock(_Underlying *_this);
            return __MR_MarchingCubesByParts_layersPerBlock(_UnderlyingPtr);
        }

        /// the last z-layer of the previous part and the first z-layer of the next part
        /// Generated from method `MR::MarchingCubesByParts::nextZ`.
        public unsafe int NextZ()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_nextZ", ExactSpelling = true)]
            extern static int __MR_MarchingCubesByParts_nextZ(_Underlying *_this);
            return __MR_MarchingCubesByParts_nextZ(_UnderlyingPtr);
        }
    }

    /// converts volume split on parts by planes z=const into mesh,
    /// last z-layer of previous part must be repeated as first z-layer of next part
    /// usage:
    /// MarchingCubesByParts x( dims, params);
    /// x.addPart( part1 );
    /// ...
    /// x.addPart( partN );
    /// Mesh mesh = Mesh::fromTriMesh( *x.finalize() );
    /// Generated from class `MR::MarchingCubesByParts`.
    /// This is the non-const half of the class.
    public class MarchingCubesByParts : Const_MarchingCubesByParts
    {
        internal unsafe MarchingCubesByParts(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::MarchingCubesByParts::MarchingCubesByParts`.
        public unsafe MarchingCubesByParts(MR._ByValue_MarchingCubesByParts s) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MarchingCubesByParts._Underlying *__MR_MarchingCubesByParts_ConstructFromAnother(MR.Misc._PassBy s_pass_by, MR.MarchingCubesByParts._Underlying *s);
            _UnderlyingPtr = __MR_MarchingCubesByParts_ConstructFromAnother(s.PassByMode, s.Value is not null ? s.Value._UnderlyingPtr : null);
        }

        /// prepares convention for given volume dimensions and given parameters
        /// \param layersPerBlock all z-slices of the volume will be partitioned on blocks of given size to process blocks in parallel (0 means auto-select layersPerBlock)
        /// Generated from constructor `MR::MarchingCubesByParts::MarchingCubesByParts`.
        /// Parameter `layersPerBlock` defaults to `0`.
        public unsafe MarchingCubesByParts(MR.Const_Vector3i dims, MR.Const_MarchingCubesParams params_, int? layersPerBlock = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_Construct", ExactSpelling = true)]
            extern static MR.MarchingCubesByParts._Underlying *__MR_MarchingCubesByParts_Construct(MR.Const_Vector3i._Underlying *dims, MR.Const_MarchingCubesParams._Underlying *params_, int *layersPerBlock);
            int __deref_layersPerBlock = layersPerBlock.GetValueOrDefault();
            _UnderlyingPtr = __MR_MarchingCubesByParts_Construct(dims._UnderlyingPtr, params_._UnderlyingPtr, layersPerBlock.HasValue ? &__deref_layersPerBlock : null);
        }

        /// Generated from method `MR::MarchingCubesByParts::operator=`.
        public unsafe MR.MarchingCubesByParts Assign(MR._ByValue_MarchingCubesByParts s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MarchingCubesByParts._Underlying *__MR_MarchingCubesByParts_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy s_pass_by, MR.MarchingCubesByParts._Underlying *s);
            return new(__MR_MarchingCubesByParts_AssignFromAnother(_UnderlyingPtr, s.PassByMode, s.Value is not null ? s.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// adds one more part of volume into consideration, with first z=nextZ()
        /// Generated from method `MR::MarchingCubesByParts::addPart`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> AddPart(MR.Const_SimpleVolume part)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_addPart", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MarchingCubesByParts_addPart(_Underlying *_this, MR.Const_SimpleVolume._Underlying *part);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MarchingCubesByParts_addPart(_UnderlyingPtr, part._UnderlyingPtr), is_owning: true));
        }

        /// finishes processing and outputs produced trimesh
        /// Generated from method `MR::MarchingCubesByParts::finalize`.
        public unsafe MR.Misc._Moved<MR.Expected_MRTriMesh_StdString> Finalize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MarchingCubesByParts_finalize", ExactSpelling = true)]
            extern static MR.Expected_MRTriMesh_StdString._Underlying *__MR_MarchingCubesByParts_finalize(_Underlying *_this);
            return MR.Misc.Move(new MR.Expected_MRTriMesh_StdString(__MR_MarchingCubesByParts_finalize(_UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `MarchingCubesByParts` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MarchingCubesByParts`/`Const_MarchingCubesByParts` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MarchingCubesByParts
    {
        internal readonly Const_MarchingCubesByParts? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MarchingCubesByParts(MR.Misc._Moved<MarchingCubesByParts> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MarchingCubesByParts(MR.Misc._Moved<MarchingCubesByParts> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MarchingCubesByParts` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MarchingCubesByParts`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MarchingCubesByParts`/`Const_MarchingCubesByParts` directly.
    public class _InOptMut_MarchingCubesByParts
    {
        public MarchingCubesByParts? Opt;

        public _InOptMut_MarchingCubesByParts() {}
        public _InOptMut_MarchingCubesByParts(MarchingCubesByParts value) {Opt = value;}
        public static implicit operator _InOptMut_MarchingCubesByParts(MarchingCubesByParts value) {return new(value);}
    }

    /// This is used for optional parameters of class `MarchingCubesByParts` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MarchingCubesByParts`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MarchingCubesByParts`/`Const_MarchingCubesByParts` to pass it to the function.
    public class _InOptConst_MarchingCubesByParts
    {
        public Const_MarchingCubesByParts? Opt;

        public _InOptConst_MarchingCubesByParts() {}
        public _InOptConst_MarchingCubesByParts(Const_MarchingCubesByParts value) {Opt = value;}
        public static implicit operator _InOptConst_MarchingCubesByParts(Const_MarchingCubesByParts value) {return new(value);}
    }

    // makes Mesh from SimpleVolume with given settings using Marching Cubes algorithm
    /// Generated from function `MR::marchingCubes`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MarchingCubes(MR.Const_SimpleVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubes_MR_SimpleVolume", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_marchingCubes_MR_SimpleVolume(MR.Const_SimpleVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_marchingCubes_MR_SimpleVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::marchingCubesAsTriMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRTriMesh_StdString> MarchingCubesAsTriMesh(MR.Const_SimpleVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubesAsTriMesh_MR_SimpleVolume", ExactSpelling = true)]
        extern static MR.Expected_MRTriMesh_StdString._Underlying *__MR_marchingCubesAsTriMesh_MR_SimpleVolume(MR.Const_SimpleVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRTriMesh_StdString(__MR_marchingCubesAsTriMesh_MR_SimpleVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    // makes Mesh from SimpleVolumeMinMax with given settings using Marching Cubes algorithm
    /// Generated from function `MR::marchingCubes`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MarchingCubes(MR.Const_SimpleVolumeMinMax volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubes_MR_SimpleVolumeMinMax", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_marchingCubes_MR_SimpleVolumeMinMax(MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_marchingCubes_MR_SimpleVolumeMinMax(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::marchingCubesAsTriMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRTriMesh_StdString> MarchingCubesAsTriMesh(MR.Const_SimpleVolumeMinMax volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubesAsTriMesh_MR_SimpleVolumeMinMax", ExactSpelling = true)]
        extern static MR.Expected_MRTriMesh_StdString._Underlying *__MR_marchingCubesAsTriMesh_MR_SimpleVolumeMinMax(MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRTriMesh_StdString(__MR_marchingCubesAsTriMesh_MR_SimpleVolumeMinMax(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    // makes Mesh from VdbVolume with given settings using Marching Cubes algorithm
    /// Generated from function `MR::marchingCubes`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MarchingCubes(MR.Const_VdbVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubes_MR_VdbVolume", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_marchingCubes_MR_VdbVolume(MR.Const_VdbVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_marchingCubes_MR_VdbVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::marchingCubesAsTriMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRTriMesh_StdString> MarchingCubesAsTriMesh(MR.Const_VdbVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubesAsTriMesh_MR_VdbVolume", ExactSpelling = true)]
        extern static MR.Expected_MRTriMesh_StdString._Underlying *__MR_marchingCubesAsTriMesh_MR_VdbVolume(MR.Const_VdbVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRTriMesh_StdString(__MR_marchingCubesAsTriMesh_MR_VdbVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    // makes Mesh from FunctionVolume with given settings using Marching Cubes algorithm
    /// Generated from function `MR::marchingCubes`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MarchingCubes(MR.Const_FunctionVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubes_MR_FunctionVolume", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_marchingCubes_MR_FunctionVolume(MR.Const_FunctionVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_marchingCubes_MR_FunctionVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::marchingCubesAsTriMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRTriMesh_StdString> MarchingCubesAsTriMesh(MR.Const_FunctionVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubesAsTriMesh_MR_FunctionVolume", ExactSpelling = true)]
        extern static MR.Expected_MRTriMesh_StdString._Underlying *__MR_marchingCubesAsTriMesh_MR_FunctionVolume(MR.Const_FunctionVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRTriMesh_StdString(__MR_marchingCubesAsTriMesh_MR_FunctionVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    // makes Mesh from SimpleBinaryVolume with given settings using Marching Cubes algorithm
    /// Generated from function `MR::marchingCubes`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> MarchingCubes(MR.Const_SimpleBinaryVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubes_MR_SimpleBinaryVolume", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_marchingCubes_MR_SimpleBinaryVolume(MR.Const_SimpleBinaryVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_marchingCubes_MR_SimpleBinaryVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::marchingCubesAsTriMesh`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRTriMesh_StdString> MarchingCubesAsTriMesh(MR.Const_SimpleBinaryVolume volume, MR.Const_MarchingCubesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_marchingCubesAsTriMesh_MR_SimpleBinaryVolume", ExactSpelling = true)]
        extern static MR.Expected_MRTriMesh_StdString._Underlying *__MR_marchingCubesAsTriMesh_MR_SimpleBinaryVolume(MR.Const_SimpleBinaryVolume._Underlying *volume, MR.Const_MarchingCubesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRTriMesh_StdString(__MR_marchingCubesAsTriMesh_MR_SimpleBinaryVolume(volume._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
