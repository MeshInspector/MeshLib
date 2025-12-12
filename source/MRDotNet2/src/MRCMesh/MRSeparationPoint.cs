public static partial class MR
{
    public enum NeighborDir : int
    {
        X = 0,
        Y = 1,
        Z = 2,
        Count = 3,
    }

    /// storage for points on voxel edges used in Marching Cubes algorithms
    /// Generated from class `MR::SeparationPointStorage`.
    /// This is the const half of the class.
    public class Const_SeparationPointStorage : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SeparationPointStorage(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Destroy", ExactSpelling = true)]
            extern static void __MR_SeparationPointStorage_Destroy(_Underlying *_this);
            __MR_SeparationPointStorage_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SeparationPointStorage() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SeparationPointStorage() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SeparationPointStorage._Underlying *__MR_SeparationPointStorage_DefaultConstruct();
            _UnderlyingPtr = __MR_SeparationPointStorage_DefaultConstruct();
        }

        /// Generated from constructor `MR::SeparationPointStorage::SeparationPointStorage`.
        public unsafe Const_SeparationPointStorage(MR._ByValue_SeparationPointStorage _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SeparationPointStorage._Underlying *__MR_SeparationPointStorage_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SeparationPointStorage._Underlying *_other);
            _UnderlyingPtr = __MR_SeparationPointStorage_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// finds the set (locating the block) by voxel id
        /// Generated from method `MR::SeparationPointStorage::findSeparationPointSet`.
        public unsafe MR.Std.Const_Array_MRVertId_3? FindSeparationPointSet(ulong voxelId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_findSeparationPointSet", ExactSpelling = true)]
            extern static MR.Std.Const_Array_MRVertId_3._Underlying *__MR_SeparationPointStorage_findSeparationPointSet(_Underlying *_this, ulong voxelId);
            var __ret = __MR_SeparationPointStorage_findSeparationPointSet(_UnderlyingPtr, voxelId);
            return __ret is not null ? new MR.Std.Const_Array_MRVertId_3(__ret, is_owning: false) : null;
        }

        /// combines triangulations from every block into one and returns it
        /// Generated from method `MR::SeparationPointStorage::getTriangulation`.
        public unsafe MR.Misc._Moved<MR.Triangulation> GetTriangulation(MR.Vector_MRVoxelId_MRFaceId? outVoxelPerFaceMap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_getTriangulation", ExactSpelling = true)]
            extern static MR.Triangulation._Underlying *__MR_SeparationPointStorage_getTriangulation(_Underlying *_this, MR.Vector_MRVoxelId_MRFaceId._Underlying *outVoxelPerFaceMap);
            return MR.Misc.Move(new MR.Triangulation(__MR_SeparationPointStorage_getTriangulation(_UnderlyingPtr, outVoxelPerFaceMap is not null ? outVoxelPerFaceMap._UnderlyingPtr : null), is_owning: true));
        }

        /// obtains coordinates of all stored points
        /// Generated from method `MR::SeparationPointStorage::getPoints`.
        public unsafe void GetPoints(MR.VertCoords points)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_getPoints", ExactSpelling = true)]
            extern static void __MR_SeparationPointStorage_getPoints(_Underlying *_this, MR.VertCoords._Underlying *points);
            __MR_SeparationPointStorage_getPoints(_UnderlyingPtr, points._UnderlyingPtr);
        }

        /// Generated from class `MR::SeparationPointStorage::Block`.
        /// This is the const half of the class.
        public class Const_Block : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Block(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_Destroy", ExactSpelling = true)]
                extern static void __MR_SeparationPointStorage_Block_Destroy(_Underlying *_this);
                __MR_SeparationPointStorage_Block_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Block() {Dispose(false);}

            public unsafe MR.Phmap.Const_FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T Smap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_Get_smap", ExactSpelling = true)]
                    extern static MR.Phmap.Const_FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T._Underlying *__MR_SeparationPointStorage_Block_Get_smap(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_Get_smap(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Std.Const_Vector_MRVector3f Coords
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_Get_coords", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRVector3f._Underlying *__MR_SeparationPointStorage_Block_Get_coords(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_Get_coords(_UnderlyingPtr), is_owning: false);
                }
            }

            /// after makeUniqueVids(), it is the unique id of first point in coords
            public unsafe MR.Const_VertId Shift
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_Get_shift", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_SeparationPointStorage_Block_Get_shift(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_Get_shift(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Triangulation Tris
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_Get_tris", ExactSpelling = true)]
                    extern static MR.Const_Triangulation._Underlying *__MR_SeparationPointStorage_Block_Get_tris(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_Get_tris(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_Vector_MRVoxelId_MRFaceId FaceMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_Get_faceMap", ExactSpelling = true)]
                    extern static MR.Const_Vector_MRVoxelId_MRFaceId._Underlying *__MR_SeparationPointStorage_Block_Get_faceMap(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_Get_faceMap(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Block() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_DefaultConstruct();
                _UnderlyingPtr = __MR_SeparationPointStorage_Block_DefaultConstruct();
            }

            /// Constructs `MR::SeparationPointStorage::Block` elementwise.
            public unsafe Const_Block(MR.Phmap._ByValue_FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T smap, MR.Std._ByValue_Vector_MRVector3f coords, MR.VertId shift, MR._ByValue_Triangulation tris, MR._ByValue_Vector_MRVoxelId_MRFaceId faceMap) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_ConstructFrom", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_ConstructFrom(MR.Misc._PassBy smap_pass_by, MR.Phmap.FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T._Underlying *smap, MR.Misc._PassBy coords_pass_by, MR.Std.Vector_MRVector3f._Underlying *coords, MR.VertId shift, MR.Misc._PassBy tris_pass_by, MR.Triangulation._Underlying *tris, MR.Misc._PassBy faceMap_pass_by, MR.Vector_MRVoxelId_MRFaceId._Underlying *faceMap);
                _UnderlyingPtr = __MR_SeparationPointStorage_Block_ConstructFrom(smap.PassByMode, smap.Value is not null ? smap.Value._UnderlyingPtr : null, coords.PassByMode, coords.Value is not null ? coords.Value._UnderlyingPtr : null, shift, tris.PassByMode, tris.Value is not null ? tris.Value._UnderlyingPtr : null, faceMap.PassByMode, faceMap.Value is not null ? faceMap.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::SeparationPointStorage::Block::Block`.
            public unsafe Const_Block(MR.SeparationPointStorage._ByValue_Block _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SeparationPointStorage.Block._Underlying *_other);
                _UnderlyingPtr = __MR_SeparationPointStorage_Block_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// during filling, it is the id of next valid point;
            /// Generated from method `MR::SeparationPointStorage::Block::nextVid`.
            public unsafe MR.VertId NextVid()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_nextVid", ExactSpelling = true)]
                extern static MR.VertId __MR_SeparationPointStorage_Block_nextVid(_Underlying *_this);
                return __MR_SeparationPointStorage_Block_nextVid(_UnderlyingPtr);
            }
        }

        /// Generated from class `MR::SeparationPointStorage::Block`.
        /// This is the non-const half of the class.
        public class Block : Const_Block
        {
            internal unsafe Block(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Phmap.FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T Smap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_GetMutable_smap", ExactSpelling = true)]
                    extern static MR.Phmap.FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T._Underlying *__MR_SeparationPointStorage_Block_GetMutable_smap(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_GetMutable_smap(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Std.Vector_MRVector3f Coords
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_GetMutable_coords", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3f._Underlying *__MR_SeparationPointStorage_Block_GetMutable_coords(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_GetMutable_coords(_UnderlyingPtr), is_owning: false);
                }
            }

            /// after makeUniqueVids(), it is the unique id of first point in coords
            public new unsafe MR.Mut_VertId Shift
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_GetMutable_shift", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_SeparationPointStorage_Block_GetMutable_shift(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_GetMutable_shift(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Triangulation Tris
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_GetMutable_tris", ExactSpelling = true)]
                    extern static MR.Triangulation._Underlying *__MR_SeparationPointStorage_Block_GetMutable_tris(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_GetMutable_tris(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Vector_MRVoxelId_MRFaceId FaceMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_GetMutable_faceMap", ExactSpelling = true)]
                    extern static MR.Vector_MRVoxelId_MRFaceId._Underlying *__MR_SeparationPointStorage_Block_GetMutable_faceMap(_Underlying *_this);
                    return new(__MR_SeparationPointStorage_Block_GetMutable_faceMap(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Block() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_DefaultConstruct();
                _UnderlyingPtr = __MR_SeparationPointStorage_Block_DefaultConstruct();
            }

            /// Constructs `MR::SeparationPointStorage::Block` elementwise.
            public unsafe Block(MR.Phmap._ByValue_FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T smap, MR.Std._ByValue_Vector_MRVector3f coords, MR.VertId shift, MR._ByValue_Triangulation tris, MR._ByValue_Vector_MRVoxelId_MRFaceId faceMap) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_ConstructFrom", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_ConstructFrom(MR.Misc._PassBy smap_pass_by, MR.Phmap.FlatHashMap_MRUint64T_StdArrayMRVertId3_PhmapHashMRUint64T._Underlying *smap, MR.Misc._PassBy coords_pass_by, MR.Std.Vector_MRVector3f._Underlying *coords, MR.VertId shift, MR.Misc._PassBy tris_pass_by, MR.Triangulation._Underlying *tris, MR.Misc._PassBy faceMap_pass_by, MR.Vector_MRVoxelId_MRFaceId._Underlying *faceMap);
                _UnderlyingPtr = __MR_SeparationPointStorage_Block_ConstructFrom(smap.PassByMode, smap.Value is not null ? smap.Value._UnderlyingPtr : null, coords.PassByMode, coords.Value is not null ? coords.Value._UnderlyingPtr : null, shift, tris.PassByMode, tris.Value is not null ? tris.Value._UnderlyingPtr : null, faceMap.PassByMode, faceMap.Value is not null ? faceMap.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::SeparationPointStorage::Block::Block`.
            public unsafe Block(MR.SeparationPointStorage._ByValue_Block _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SeparationPointStorage.Block._Underlying *_other);
                _UnderlyingPtr = __MR_SeparationPointStorage_Block_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::SeparationPointStorage::Block::operator=`.
            public unsafe MR.SeparationPointStorage.Block Assign(MR.SeparationPointStorage._ByValue_Block _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_Block_AssignFromAnother", ExactSpelling = true)]
                extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_Block_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SeparationPointStorage.Block._Underlying *_other);
                return new(__MR_SeparationPointStorage_Block_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Block` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Block`/`Const_Block` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Block
        {
            internal readonly Const_Block? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Block() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Block(Const_Block new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Block(Const_Block arg) {return new(arg);}
            public _ByValue_Block(MR.Misc._Moved<Block> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Block(MR.Misc._Moved<Block> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Block` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Block`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Block`/`Const_Block` directly.
        public class _InOptMut_Block
        {
            public Block? Opt;

            public _InOptMut_Block() {}
            public _InOptMut_Block(Block value) {Opt = value;}
            public static implicit operator _InOptMut_Block(Block value) {return new(value);}
        }

        /// This is used for optional parameters of class `Block` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Block`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Block`/`Const_Block` to pass it to the function.
        public class _InOptConst_Block
        {
            public Const_Block? Opt;

            public _InOptConst_Block() {}
            public _InOptConst_Block(Const_Block value) {Opt = value;}
            public static implicit operator _InOptConst_Block(Const_Block value) {return new(value);}
        }
    }

    /// storage for points on voxel edges used in Marching Cubes algorithms
    /// Generated from class `MR::SeparationPointStorage`.
    /// This is the non-const half of the class.
    public class SeparationPointStorage : Const_SeparationPointStorage
    {
        internal unsafe SeparationPointStorage(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe SeparationPointStorage() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SeparationPointStorage._Underlying *__MR_SeparationPointStorage_DefaultConstruct();
            _UnderlyingPtr = __MR_SeparationPointStorage_DefaultConstruct();
        }

        /// Generated from constructor `MR::SeparationPointStorage::SeparationPointStorage`.
        public unsafe SeparationPointStorage(MR._ByValue_SeparationPointStorage _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SeparationPointStorage._Underlying *__MR_SeparationPointStorage_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SeparationPointStorage._Underlying *_other);
            _UnderlyingPtr = __MR_SeparationPointStorage_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SeparationPointStorage::operator=`.
        public unsafe MR.SeparationPointStorage Assign(MR._ByValue_SeparationPointStorage _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SeparationPointStorage._Underlying *__MR_SeparationPointStorage_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SeparationPointStorage._Underlying *_other);
            return new(__MR_SeparationPointStorage_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// prepares storage for given number of blocks, each containing given size of voxels
        /// Generated from method `MR::SeparationPointStorage::resize`.
        public unsafe void Resize(ulong blockCount, ulong blockSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_resize", ExactSpelling = true)]
            extern static void __MR_SeparationPointStorage_resize(_Underlying *_this, ulong blockCount, ulong blockSize);
            __MR_SeparationPointStorage_resize(_UnderlyingPtr, blockCount, blockSize);
        }

        /// get block for filling in the thread responsible for it
        /// Generated from method `MR::SeparationPointStorage::getBlock`.
        public unsafe MR.SeparationPointStorage.Block GetBlock(ulong blockIndex)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_getBlock", ExactSpelling = true)]
            extern static MR.SeparationPointStorage.Block._Underlying *__MR_SeparationPointStorage_getBlock(_Underlying *_this, ulong blockIndex);
            return new(__MR_SeparationPointStorage_getBlock(_UnderlyingPtr, blockIndex), is_owning: false);
        }

        /// shifts vertex ids in each block (after they are filled) to make them unique;
        /// returns the total number of valid points in the storage
        /// Generated from method `MR::SeparationPointStorage::makeUniqueVids`.
        public unsafe int MakeUniqueVids()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SeparationPointStorage_makeUniqueVids", ExactSpelling = true)]
            extern static int __MR_SeparationPointStorage_makeUniqueVids(_Underlying *_this);
            return __MR_SeparationPointStorage_makeUniqueVids(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SeparationPointStorage` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SeparationPointStorage`/`Const_SeparationPointStorage` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SeparationPointStorage
    {
        internal readonly Const_SeparationPointStorage? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SeparationPointStorage() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SeparationPointStorage(Const_SeparationPointStorage new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SeparationPointStorage(Const_SeparationPointStorage arg) {return new(arg);}
        public _ByValue_SeparationPointStorage(MR.Misc._Moved<SeparationPointStorage> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SeparationPointStorage(MR.Misc._Moved<SeparationPointStorage> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SeparationPointStorage` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SeparationPointStorage`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SeparationPointStorage`/`Const_SeparationPointStorage` directly.
    public class _InOptMut_SeparationPointStorage
    {
        public SeparationPointStorage? Opt;

        public _InOptMut_SeparationPointStorage() {}
        public _InOptMut_SeparationPointStorage(SeparationPointStorage value) {Opt = value;}
        public static implicit operator _InOptMut_SeparationPointStorage(SeparationPointStorage value) {return new(value);}
    }

    /// This is used for optional parameters of class `SeparationPointStorage` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SeparationPointStorage`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SeparationPointStorage`/`Const_SeparationPointStorage` to pass it to the function.
    public class _InOptConst_SeparationPointStorage
    {
        public Const_SeparationPointStorage? Opt;

        public _InOptConst_SeparationPointStorage() {}
        public _InOptConst_SeparationPointStorage(Const_SeparationPointStorage value) {Opt = value;}
        public static implicit operator _InOptConst_SeparationPointStorage(Const_SeparationPointStorage value) {return new(value);}
    }
}
