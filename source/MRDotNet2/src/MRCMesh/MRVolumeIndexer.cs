public static partial class MR
{
    public enum OutEdge : sbyte
    {
        Invalid = -1,
        PlusZ = 0,
        MinusZ = 1,
        PlusY = 2,
        MinusY = 3,
        PlusX = 4,
        MinusX = 5,
        Count = 6,
    }

    /// contains both linear Id and 3D coordinates of the same voxel
    /// Generated from class `MR::VoxelLocation`.
    /// This is the const half of the class.
    public class Const_VoxelLocation : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelLocation(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelLocation_Destroy(_Underlying *_this);
            __MR_VoxelLocation_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelLocation() {Dispose(false);}

        public unsafe MR.Const_VoxelId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_Get_id", ExactSpelling = true)]
                extern static MR.Const_VoxelId._Underlying *__MR_VoxelLocation_Get_id(_Underlying *_this);
                return new(__MR_VoxelLocation_Get_id(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3i Pos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_Get_pos", ExactSpelling = true)]
                extern static MR.Const_Vector3i._Underlying *__MR_VoxelLocation_Get_pos(_Underlying *_this);
                return new(__MR_VoxelLocation_Get_pos(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VoxelLocation() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelLocation_DefaultConstruct();
        }

        /// Constructs `MR::VoxelLocation` elementwise.
        public unsafe Const_VoxelLocation(MR.VoxelId id, MR.Vector3i pos) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_ConstructFrom", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_ConstructFrom(MR.VoxelId id, MR.Vector3i pos);
            _UnderlyingPtr = __MR_VoxelLocation_ConstructFrom(id, pos);
        }

        /// Generated from constructor `MR::VoxelLocation::VoxelLocation`.
        public unsafe Const_VoxelLocation(MR.Const_VoxelLocation _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_ConstructFromAnother(MR.VoxelLocation._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelLocation_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// check for validity
        /// Generated from conversion operator `MR::VoxelLocation::operator bool`.
        public static unsafe explicit operator bool(MR.Const_VoxelLocation _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_VoxelLocation_ConvertTo_bool(MR.Const_VoxelLocation._Underlying *_this);
            return __MR_VoxelLocation_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }
    }

    /// contains both linear Id and 3D coordinates of the same voxel
    /// Generated from class `MR::VoxelLocation`.
    /// This is the non-const half of the class.
    public class VoxelLocation : Const_VoxelLocation
    {
        internal unsafe VoxelLocation(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_VoxelId Id
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_GetMutable_id", ExactSpelling = true)]
                extern static MR.Mut_VoxelId._Underlying *__MR_VoxelLocation_GetMutable_id(_Underlying *_this);
                return new(__MR_VoxelLocation_GetMutable_id(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3i Pos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_GetMutable_pos", ExactSpelling = true)]
                extern static MR.Mut_Vector3i._Underlying *__MR_VoxelLocation_GetMutable_pos(_Underlying *_this);
                return new(__MR_VoxelLocation_GetMutable_pos(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VoxelLocation() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_DefaultConstruct();
            _UnderlyingPtr = __MR_VoxelLocation_DefaultConstruct();
        }

        /// Constructs `MR::VoxelLocation` elementwise.
        public unsafe VoxelLocation(MR.VoxelId id, MR.Vector3i pos) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_ConstructFrom", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_ConstructFrom(MR.VoxelId id, MR.Vector3i pos);
            _UnderlyingPtr = __MR_VoxelLocation_ConstructFrom(id, pos);
        }

        /// Generated from constructor `MR::VoxelLocation::VoxelLocation`.
        public unsafe VoxelLocation(MR.Const_VoxelLocation _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_ConstructFromAnother(MR.VoxelLocation._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelLocation_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelLocation::operator=`.
        public unsafe MR.VoxelLocation Assign(MR.Const_VoxelLocation _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelLocation_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VoxelLocation_AssignFromAnother(_Underlying *_this, MR.VoxelLocation._Underlying *_other);
            return new(__MR_VoxelLocation_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VoxelLocation` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelLocation`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelLocation`/`Const_VoxelLocation` directly.
    public class _InOptMut_VoxelLocation
    {
        public VoxelLocation? Opt;

        public _InOptMut_VoxelLocation() {}
        public _InOptMut_VoxelLocation(VoxelLocation value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelLocation(VoxelLocation value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelLocation` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelLocation`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelLocation`/`Const_VoxelLocation` to pass it to the function.
    public class _InOptConst_VoxelLocation
    {
        public Const_VoxelLocation? Opt;

        public _InOptConst_VoxelLocation() {}
        public _InOptConst_VoxelLocation(Const_VoxelLocation value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelLocation(Const_VoxelLocation value) {return new(value);}
    }

    /// Generated from class `MR::VolumeIndexer`.
    /// This is the const half of the class.
    public class Const_VolumeIndexer : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VolumeIndexer(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_Destroy", ExactSpelling = true)]
            extern static void __MR_VolumeIndexer_Destroy(_Underlying *_this);
            __MR_VolumeIndexer_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VolumeIndexer() {Dispose(false);}

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public unsafe Const_VolumeIndexer(MR.Const_VolumeIndexer _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeIndexer._Underlying *__MR_VolumeIndexer_ConstructFromAnother(MR.VolumeIndexer._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeIndexer_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public unsafe Const_VolumeIndexer(MR.Const_Vector3i dims) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_Construct", ExactSpelling = true)]
            extern static MR.VolumeIndexer._Underlying *__MR_VolumeIndexer_Construct(MR.Const_Vector3i._Underlying *dims);
            _UnderlyingPtr = __MR_VolumeIndexer_Construct(dims._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public static unsafe implicit operator Const_VolumeIndexer(MR.Const_Vector3i dims) {return new(dims);}

        /// Generated from method `MR::VolumeIndexer::dims`.
        public unsafe MR.Const_Vector3i Dims()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_dims", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_VolumeIndexer_dims(_Underlying *_this);
            return new(__MR_VolumeIndexer_dims(_UnderlyingPtr), is_owning: false);
        }

        /// returns the total number of voxels
        /// Generated from method `MR::VolumeIndexer::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_size", ExactSpelling = true)]
            extern static ulong __MR_VolumeIndexer_size(_Underlying *_this);
            return __MR_VolumeIndexer_size(_UnderlyingPtr);
        }

        /// returns the last plus one voxel Id for defining iteration range
        /// Generated from method `MR::VolumeIndexer::endId`.
        public unsafe MR.VoxelId EndId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_endId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VolumeIndexer_endId(_Underlying *_this);
            return __MR_VolumeIndexer_endId(_UnderlyingPtr);
        }

        /// Generated from method `MR::VolumeIndexer::sizeXY`.
        public unsafe ulong SizeXY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_sizeXY", ExactSpelling = true)]
            extern static ulong __MR_VolumeIndexer_sizeXY(_Underlying *_this);
            return __MR_VolumeIndexer_sizeXY(_UnderlyingPtr);
        }

        /// Generated from method `MR::VolumeIndexer::toPos`.
        public unsafe MR.Vector3i ToPos(MR.VoxelId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_toPos", ExactSpelling = true)]
            extern static MR.Vector3i __MR_VolumeIndexer_toPos(_Underlying *_this, MR.VoxelId id);
            return __MR_VolumeIndexer_toPos(_UnderlyingPtr, id);
        }

        /// Generated from method `MR::VolumeIndexer::toVoxelId`.
        public unsafe MR.VoxelId ToVoxelId(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_toVoxelId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VolumeIndexer_toVoxelId(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return __MR_VolumeIndexer_toVoxelId(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::VolumeIndexer::toLoc`.
        public unsafe MR.VoxelLocation ToLoc(MR.VoxelId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_toLoc_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VolumeIndexer_toLoc_MR_VoxelId(_Underlying *_this, MR.VoxelId id);
            return new(__MR_VolumeIndexer_toLoc_MR_VoxelId(_UnderlyingPtr, id), is_owning: true);
        }

        /// Generated from method `MR::VolumeIndexer::toLoc`.
        public unsafe MR.VoxelLocation ToLoc(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_toLoc_MR_Vector3i", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VolumeIndexer_toLoc_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return new(__MR_VolumeIndexer_toLoc_MR_Vector3i(_UnderlyingPtr, pos._UnderlyingPtr), is_owning: true);
        }

        /// returns true if this voxel is within dimensions
        /// Generated from method `MR::VolumeIndexer::isInDims`.
        public unsafe bool IsInDims(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_isInDims", ExactSpelling = true)]
            extern static byte __MR_VolumeIndexer_isInDims(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return __MR_VolumeIndexer_isInDims(_UnderlyingPtr, pos._UnderlyingPtr) != 0;
        }

        /// returns true if this voxel is on the boundary of the volume
        /// Generated from method `MR::VolumeIndexer::isBdVoxel`.
        public unsafe bool IsBdVoxel(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_isBdVoxel", ExactSpelling = true)]
            extern static byte __MR_VolumeIndexer_isBdVoxel(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return __MR_VolumeIndexer_isBdVoxel(_UnderlyingPtr, pos._UnderlyingPtr) != 0;
        }

        /// returns true if v1 is within at most 6 neighbors of v0
        /// Generated from method `MR::VolumeIndexer::areNeigbors`.
        public unsafe bool AreNeigbors(MR.VoxelId v0, MR.VoxelId v1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_areNeigbors_MR_VoxelId", ExactSpelling = true)]
            extern static byte __MR_VolumeIndexer_areNeigbors_MR_VoxelId(_Underlying *_this, MR.VoxelId v0, MR.VoxelId v1);
            return __MR_VolumeIndexer_areNeigbors_MR_VoxelId(_UnderlyingPtr, v0, v1) != 0;
        }

        /// Generated from method `MR::VolumeIndexer::areNeigbors`.
        public unsafe bool AreNeigbors(MR.Const_Vector3i pos0, MR.Const_Vector3i pos1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_areNeigbors_MR_Vector3i", ExactSpelling = true)]
            extern static byte __MR_VolumeIndexer_areNeigbors_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pos0, MR.Const_Vector3i._Underlying *pos1);
            return __MR_VolumeIndexer_areNeigbors_MR_Vector3i(_UnderlyingPtr, pos0._UnderlyingPtr, pos1._UnderlyingPtr) != 0;
        }

        /// given existing voxel at (pos), returns whether it has valid neighbor specified by the edge (toNei)
        /// Generated from method `MR::VolumeIndexer::hasNeighbour`.
        public unsafe bool HasNeighbour(MR.Const_Vector3i pos, MR.OutEdge toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_hasNeighbour", ExactSpelling = true)]
            extern static byte __MR_VolumeIndexer_hasNeighbour(_Underlying *_this, MR.Const_Vector3i._Underlying *pos, MR.OutEdge toNei);
            return __MR_VolumeIndexer_hasNeighbour(_UnderlyingPtr, pos._UnderlyingPtr, toNei) != 0;
        }

        /// returns id of v's neighbor specified by the edge
        /// Generated from method `MR::VolumeIndexer::getNeighbor`.
        public unsafe MR.VoxelId GetNeighbor(MR.VoxelId v, MR.OutEdge toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_getNeighbor_2_MR_VoxelId", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VolumeIndexer_getNeighbor_2_MR_VoxelId(_Underlying *_this, MR.VoxelId v, MR.OutEdge toNei);
            return __MR_VolumeIndexer_getNeighbor_2_MR_VoxelId(_UnderlyingPtr, v, toNei);
        }

        /// Generated from method `MR::VolumeIndexer::getNeighbor`.
        public unsafe MR.VoxelId GetNeighbor(MR.VoxelId v, MR.Const_Vector3i pos, MR.OutEdge toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_getNeighbor_3", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VolumeIndexer_getNeighbor_3(_Underlying *_this, MR.VoxelId v, MR.Const_Vector3i._Underlying *pos, MR.OutEdge toNei);
            return __MR_VolumeIndexer_getNeighbor_3(_UnderlyingPtr, v, pos._UnderlyingPtr, toNei);
        }

        /// given existing voxel at (loc), returns its neighbor specified by the edge (toNei);
        /// if the neighbour does not exist (loc is on boundary), returns invalid VoxelLocation
        /// Generated from method `MR::VolumeIndexer::getNeighbor`.
        public unsafe MR.VoxelLocation GetNeighbor(MR.Const_VoxelLocation loc, MR.OutEdge toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_getNeighbor_2_MR_VoxelLocation", ExactSpelling = true)]
            extern static MR.VoxelLocation._Underlying *__MR_VolumeIndexer_getNeighbor_2_MR_VoxelLocation(_Underlying *_this, MR.Const_VoxelLocation._Underlying *loc, MR.OutEdge toNei);
            return new(__MR_VolumeIndexer_getNeighbor_2_MR_VoxelLocation(_UnderlyingPtr, loc._UnderlyingPtr, toNei), is_owning: true);
        }

        /// returns id of v's neighbor specified by the edge, which is known to exist (so skipping a lot of checks)
        /// Generated from method `MR::VolumeIndexer::getExistingNeighbor`.
        public unsafe MR.VoxelId GetExistingNeighbor(MR.VoxelId v, MR.OutEdge toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_getExistingNeighbor", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VolumeIndexer_getExistingNeighbor(_Underlying *_this, MR.VoxelId v, MR.OutEdge toNei);
            return __MR_VolumeIndexer_getExistingNeighbor(_UnderlyingPtr, v, toNei);
        }

        /// Generated from method `MR::VolumeIndexer::getNeighbor`.
        public unsafe MR.VoxelId GetNeighbor(MR.VoxelId v, MR.Const_Vector3i pos, bool bdPos, MR.OutEdge toNei)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_getNeighbor_4", ExactSpelling = true)]
            extern static MR.VoxelId __MR_VolumeIndexer_getNeighbor_4(_Underlying *_this, MR.VoxelId v, MR.Const_Vector3i._Underlying *pos, byte bdPos, MR.OutEdge toNei);
            return __MR_VolumeIndexer_getNeighbor_4(_UnderlyingPtr, v, pos._UnderlyingPtr, bdPos ? (byte)1 : (byte)0, toNei);
        }
    }

    /// Generated from class `MR::VolumeIndexer`.
    /// This is the non-const half of the class.
    public class VolumeIndexer : Const_VolumeIndexer
    {
        internal unsafe VolumeIndexer(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public unsafe VolumeIndexer(MR.Const_VolumeIndexer _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VolumeIndexer._Underlying *__MR_VolumeIndexer_ConstructFromAnother(MR.VolumeIndexer._Underlying *_other);
            _UnderlyingPtr = __MR_VolumeIndexer_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public unsafe VolumeIndexer(MR.Const_Vector3i dims) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_Construct", ExactSpelling = true)]
            extern static MR.VolumeIndexer._Underlying *__MR_VolumeIndexer_Construct(MR.Const_Vector3i._Underlying *dims);
            _UnderlyingPtr = __MR_VolumeIndexer_Construct(dims._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public static unsafe implicit operator VolumeIndexer(MR.Const_Vector3i dims) {return new(dims);}

        /// Generated from method `MR::VolumeIndexer::operator=`.
        public unsafe MR.VolumeIndexer Assign(MR.Const_VolumeIndexer _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VolumeIndexer_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VolumeIndexer._Underlying *__MR_VolumeIndexer_AssignFromAnother(_Underlying *_this, MR.VolumeIndexer._Underlying *_other);
            return new(__MR_VolumeIndexer_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VolumeIndexer` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VolumeIndexer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeIndexer`/`Const_VolumeIndexer` directly.
    public class _InOptMut_VolumeIndexer
    {
        public VolumeIndexer? Opt;

        public _InOptMut_VolumeIndexer() {}
        public _InOptMut_VolumeIndexer(VolumeIndexer value) {Opt = value;}
        public static implicit operator _InOptMut_VolumeIndexer(VolumeIndexer value) {return new(value);}
    }

    /// This is used for optional parameters of class `VolumeIndexer` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VolumeIndexer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VolumeIndexer`/`Const_VolumeIndexer` to pass it to the function.
    public class _InOptConst_VolumeIndexer
    {
        public Const_VolumeIndexer? Opt;

        public _InOptConst_VolumeIndexer() {}
        public _InOptConst_VolumeIndexer(Const_VolumeIndexer value) {Opt = value;}
        public static implicit operator _InOptConst_VolumeIndexer(Const_VolumeIndexer value) {return new(value);}

        /// Generated from constructor `MR::VolumeIndexer::VolumeIndexer`.
        public static unsafe implicit operator _InOptConst_VolumeIndexer(MR.Const_Vector3i dims) {return new MR.VolumeIndexer(dims);}
    }

    /// Generated from function `MR::opposite`.
    public static MR.OutEdge Opposite(MR.OutEdge e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_opposite_MR_OutEdge", ExactSpelling = true)]
        extern static MR.OutEdge __MR_opposite_MR_OutEdge(MR.OutEdge e);
        return __MR_opposite_MR_OutEdge(e);
    }

    /// expands VoxelBitSet with given number of steps
    /// Generated from function `MR::expandVoxelsMask`.
    /// Parameter `expansion` defaults to `1`.
    public static unsafe void ExpandVoxelsMask(MR.VoxelBitSet mask, MR.Const_VolumeIndexer indexer, int? expansion = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_expandVoxelsMask", ExactSpelling = true)]
        extern static void __MR_expandVoxelsMask(MR.VoxelBitSet._Underlying *mask, MR.Const_VolumeIndexer._Underlying *indexer, int *expansion);
        int __deref_expansion = expansion.GetValueOrDefault();
        __MR_expandVoxelsMask(mask._UnderlyingPtr, indexer._UnderlyingPtr, expansion.HasValue ? &__deref_expansion : null);
    }

    /// shrinks VoxelBitSet with given number of steps
    /// Generated from function `MR::shrinkVoxelsMask`.
    /// Parameter `shrinkage` defaults to `1`.
    public static unsafe void ShrinkVoxelsMask(MR.VoxelBitSet mask, MR.Const_VolumeIndexer indexer, int? shrinkage = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_shrinkVoxelsMask", ExactSpelling = true)]
        extern static void __MR_shrinkVoxelsMask(MR.VoxelBitSet._Underlying *mask, MR.Const_VolumeIndexer._Underlying *indexer, int *shrinkage);
        int __deref_shrinkage = shrinkage.GetValueOrDefault();
        __MR_shrinkVoxelsMask(mask._UnderlyingPtr, indexer._UnderlyingPtr, shrinkage.HasValue ? &__deref_shrinkage : null);
    }
}
