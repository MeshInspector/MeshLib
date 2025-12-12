public static partial class MR
{
    /// Generated from class `MR::PointCloud`.
    /// This is the const half of the class.
    public class Const_PointCloud : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PointCloud_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PointCloud_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PointCloud_UseCount();
                return __MR_std_shared_ptr_MR_PointCloud_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointCloud_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PointCloud_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PointCloud(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointCloud_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointCloud_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointCloud_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointCloud_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PointCloud(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PointCloud _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointCloud_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PointCloud_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointCloud_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointCloud_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointCloud_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PointCloud_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PointCloud_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointCloud() {Dispose(false);}

        /// coordinates of points
        public unsafe MR.Const_VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_Get_points", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_PointCloud_Get_points(_Underlying *_this);
                return new(__MR_PointCloud_Get_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// unit normal directions of points (can be empty if no normals are known)
        public unsafe MR.Const_VertCoords Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_Get_normals", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_PointCloud_Get_normals(_Underlying *_this);
                return new(__MR_PointCloud_Get_normals(_UnderlyingPtr), is_owning: false);
            }
        }

        /// only points and normals corresponding to set bits here are valid
        public unsafe MR.Const_VertBitSet ValidPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_Get_validPoints", ExactSpelling = true)]
                extern static MR.Const_VertBitSet._Underlying *__MR_PointCloud_Get_validPoints(_Underlying *_this);
                return new(__MR_PointCloud_Get_validPoints(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointCloud() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointCloud._Underlying *__MR_PointCloud_DefaultConstruct();
            _LateMakeShared(__MR_PointCloud_DefaultConstruct());
        }

        /// Generated from constructor `MR::PointCloud::PointCloud`.
        public unsafe Const_PointCloud(MR._ByValue_PointCloud _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloud._Underlying *__MR_PointCloud_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointCloud._Underlying *_other);
            _LateMakeShared(__MR_PointCloud_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// computes the total number of valid points in the cloud
        /// Generated from method `MR::PointCloud::calcNumValidPoints`.
        public unsafe ulong CalcNumValidPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_calcNumValidPoints", ExactSpelling = true)]
            extern static ulong __MR_PointCloud_calcNumValidPoints(_Underlying *_this);
            return __MR_PointCloud_calcNumValidPoints(_UnderlyingPtr);
        }

        /// returns true if there is a normal for each point
        /// Generated from method `MR::PointCloud::hasNormals`.
        public unsafe bool HasNormals()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_hasNormals", ExactSpelling = true)]
            extern static byte __MR_PointCloud_hasNormals(_Underlying *_this);
            return __MR_PointCloud_hasNormals(_UnderlyingPtr) != 0;
        }

        /// if region pointer is not null then converts it in reference, otherwise returns all valid points in the cloud
        /// Generated from method `MR::PointCloud::getVertIds`.
        public unsafe MR.Const_VertBitSet GetVertIds(MR.Const_VertBitSet? region)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_getVertIds", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_PointCloud_getVertIds(_Underlying *_this, MR.Const_VertBitSet._Underlying *region);
            return new(__MR_PointCloud_getVertIds(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: false);
        }

        /// returns cached aabb-tree for this point cloud, creating it if it did not exist in a thread-safe manner
        /// Generated from method `MR::PointCloud::getAABBTree`.
        public unsafe MR.Const_AABBTreePoints GetAABBTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_getAABBTree", ExactSpelling = true)]
            extern static MR.Const_AABBTreePoints._Underlying *__MR_PointCloud_getAABBTree(_Underlying *_this);
            return new(__MR_PointCloud_getAABBTree(_UnderlyingPtr), is_owning: false);
        }

        /// returns cached aabb-tree for this point cloud, but does not create it if it did not exist
        /// Generated from method `MR::PointCloud::getAABBTreeNotCreate`.
        public unsafe MR.Const_AABBTreePoints? GetAABBTreeNotCreate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_getAABBTreeNotCreate", ExactSpelling = true)]
            extern static MR.Const_AABBTreePoints._Underlying *__MR_PointCloud_getAABBTreeNotCreate(_Underlying *_this);
            var __ret = __MR_PointCloud_getAABBTreeNotCreate(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_AABBTreePoints(__ret, is_owning: false) : null;
        }

        /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
        /// Generated from method `MR::PointCloud::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_PointCloud_getBoundingBox(_Underlying *_this);
            return __MR_PointCloud_getBoundingBox(_UnderlyingPtr);
        }

        /// passes through all valid points and finds the minimal bounding box containing all of them;
        /// if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::PointCloud::computeBoundingBox`.
        public unsafe MR.Box3f ComputeBoundingBox(MR.Const_AffineXf3f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_computeBoundingBox_1", ExactSpelling = true)]
            extern static MR.Box3f __MR_PointCloud_computeBoundingBox_1(_Underlying *_this, MR.Const_AffineXf3f._Underlying *toWorld);
            return __MR_PointCloud_computeBoundingBox_1(_UnderlyingPtr, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        /// passes through all given vertices (or all valid vertices if region == null) and finds the minimal bounding box containing all of them
        /// if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::PointCloud::computeBoundingBox`.
        public unsafe MR.Box3f ComputeBoundingBox(MR.Const_VertBitSet? region, MR.Const_AffineXf3f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_computeBoundingBox_2", ExactSpelling = true)]
            extern static MR.Box3f __MR_PointCloud_computeBoundingBox_2(_Underlying *_this, MR.Const_VertBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *toWorld);
            return __MR_PointCloud_computeBoundingBox_2(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        /// computes average position of all valid points
        /// Generated from method `MR::PointCloud::findCenterFromPoints`.
        public unsafe MR.Vector3f FindCenterFromPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_findCenterFromPoints", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PointCloud_findCenterFromPoints(_Underlying *_this);
            return __MR_PointCloud_findCenterFromPoints(_UnderlyingPtr);
        }

        /// computes bounding box and returns its center
        /// Generated from method `MR::PointCloud::findCenterFromBBox`.
        public unsafe MR.Vector3f FindCenterFromBBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_findCenterFromBBox", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PointCloud_findCenterFromBBox(_Underlying *_this);
            return __MR_PointCloud_findCenterFromBBox(_UnderlyingPtr);
        }

        /// returns all valid point ids sorted lexicographically by their coordinates (optimal for uniform sampling)
        /// Generated from method `MR::PointCloud::getLexicographicalOrder`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRVertId> GetLexicographicalOrder()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_getLexicographicalOrder", ExactSpelling = true)]
            extern static MR.Std.Vector_MRVertId._Underlying *__MR_PointCloud_getLexicographicalOrder(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRVertId(__MR_PointCloud_getLexicographicalOrder(_UnderlyingPtr), is_owning: true));
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::PointCloud::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PointCloud_heapBytes(_Underlying *_this);
            return __MR_PointCloud_heapBytes(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PointCloud`.
    /// This is the non-const half of the class.
    public class PointCloud : Const_PointCloud
    {
        internal unsafe PointCloud(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PointCloud(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// coordinates of points
        public new unsafe MR.VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_GetMutable_points", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_PointCloud_GetMutable_points(_Underlying *_this);
                return new(__MR_PointCloud_GetMutable_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// unit normal directions of points (can be empty if no normals are known)
        public new unsafe MR.VertCoords Normals
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_GetMutable_normals", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_PointCloud_GetMutable_normals(_Underlying *_this);
                return new(__MR_PointCloud_GetMutable_normals(_UnderlyingPtr), is_owning: false);
            }
        }

        /// only points and normals corresponding to set bits here are valid
        public new unsafe MR.VertBitSet ValidPoints
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_GetMutable_validPoints", ExactSpelling = true)]
                extern static MR.VertBitSet._Underlying *__MR_PointCloud_GetMutable_validPoints(_Underlying *_this);
                return new(__MR_PointCloud_GetMutable_validPoints(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointCloud() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointCloud._Underlying *__MR_PointCloud_DefaultConstruct();
            _LateMakeShared(__MR_PointCloud_DefaultConstruct());
        }

        /// Generated from constructor `MR::PointCloud::PointCloud`.
        public unsafe PointCloud(MR._ByValue_PointCloud _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloud._Underlying *__MR_PointCloud_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointCloud._Underlying *_other);
            _LateMakeShared(__MR_PointCloud_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::PointCloud::operator=`.
        public unsafe MR.PointCloud Assign(MR._ByValue_PointCloud _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointCloud._Underlying *__MR_PointCloud_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointCloud._Underlying *_other);
            return new(__MR_PointCloud_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// applies given transformation to specified points and corresponding transformation to their normals if present;
        /// if region is nullptr, all valid points are modified
        /// Generated from method `MR::PointCloud::transform`.
        public unsafe void Transform(MR.Const_AffineXf3f xf, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_transform", ExactSpelling = true)]
            extern static void __MR_PointCloud_transform(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.Const_VertBitSet._Underlying *region);
            __MR_PointCloud_transform(_UnderlyingPtr, xf._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// appends points (and normals if it possible) (from) in addition to this points
        /// if this obj have normals and from obj has not it then don't do anything
        /// \param extNormals if given then they will be copied instead of from.normals
        /// Generated from method `MR::PointCloud::addPartByMask`.
        /// Parameter `outMap` defaults to `{}`.
        public unsafe void AddPartByMask(MR.Const_PointCloud from, MR.Const_VertBitSet fromVerts, MR.Const_CloudPartMapping? outMap = null, MR.Const_VertCoords? extNormals = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_addPartByMask", ExactSpelling = true)]
            extern static void __MR_PointCloud_addPartByMask(_Underlying *_this, MR.Const_PointCloud._Underlying *from, MR.Const_VertBitSet._Underlying *fromVerts, MR.Const_CloudPartMapping._Underlying *outMap, MR.Const_VertCoords._Underlying *extNormals);
            __MR_PointCloud_addPartByMask(_UnderlyingPtr, from._UnderlyingPtr, fromVerts._UnderlyingPtr, outMap is not null ? outMap._UnderlyingPtr : null, extNormals is not null ? extNormals._UnderlyingPtr : null);
        }

        /// appends a point and returns its VertId
        /// Generated from method `MR::PointCloud::addPoint`.
        public unsafe MR.VertId AddPoint(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_addPoint_1", ExactSpelling = true)]
            extern static MR.VertId __MR_PointCloud_addPoint_1(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return __MR_PointCloud_addPoint_1(_UnderlyingPtr, point._UnderlyingPtr);
        }

        /// appends a point with normal and returns its VertId
        /// Generated from method `MR::PointCloud::addPoint`.
        public unsafe MR.VertId AddPoint(MR.Const_Vector3f point, MR.Const_Vector3f normal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_addPoint_2", ExactSpelling = true)]
            extern static MR.VertId __MR_PointCloud_addPoint_2(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.Const_Vector3f._Underlying *normal);
            return __MR_PointCloud_addPoint_2(_UnderlyingPtr, point._UnderlyingPtr, normal._UnderlyingPtr);
        }

        /// reflects the points from a given plane
        /// Generated from method `MR::PointCloud::mirror`.
        public unsafe void Mirror(MR.Const_Plane3f plane)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_mirror", ExactSpelling = true)]
            extern static void __MR_PointCloud_mirror(_Underlying *_this, MR.Const_Plane3f._Underlying *plane);
            __MR_PointCloud_mirror(_UnderlyingPtr, plane._UnderlyingPtr);
        }

        /// flip orientation (normals) of given points (or all valid points is nullptr)
        /// Generated from method `MR::PointCloud::flipOrientation`.
        public unsafe void FlipOrientation(MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_flipOrientation", ExactSpelling = true)]
            extern static void __MR_PointCloud_flipOrientation(_Underlying *_this, MR.Const_VertBitSet._Underlying *region);
            __MR_PointCloud_flipOrientation(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// tightly packs all arrays eliminating invalid points, but relative order of valid points is preserved;
        /// returns false if the cloud was packed before the call and nothing has been changed;
        /// if pack is done optionally returns mappings: new.id -> old.id
        /// Generated from method `MR::PointCloud::pack`.
        public unsafe bool Pack(MR.VertMap? outNew2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_pack_MR_VertMap_ptr", ExactSpelling = true)]
            extern static byte __MR_PointCloud_pack_MR_VertMap_ptr(_Underlying *_this, MR.VertMap._Underlying *outNew2Old);
            return __MR_PointCloud_pack_MR_VertMap_ptr(_UnderlyingPtr, outNew2Old is not null ? outNew2Old._UnderlyingPtr : null) != 0;
        }

        /// tightly packs all arrays eliminating invalid points, reorders valid points according to given strategy;
        /// \return points mapping: old -> new
        /// Generated from method `MR::PointCloud::pack`.
        public unsafe MR.Misc._Moved<MR.VertBMap> Pack(MR.Reorder reoder)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_pack_MR_Reorder", ExactSpelling = true)]
            extern static MR.VertBMap._Underlying *__MR_PointCloud_pack_MR_Reorder(_Underlying *_this, MR.Reorder reoder);
            return MR.Misc.Move(new MR.VertBMap(__MR_PointCloud_pack_MR_Reorder(_UnderlyingPtr, reoder), is_owning: true));
        }

        /// Invalidates caches (e.g. aabb-tree) after a change in point cloud
        /// Generated from method `MR::PointCloud::invalidateCaches`.
        public unsafe void InvalidateCaches()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloud_invalidateCaches", ExactSpelling = true)]
            extern static void __MR_PointCloud_invalidateCaches(_Underlying *_this);
            __MR_PointCloud_invalidateCaches(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointCloud` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointCloud`/`Const_PointCloud` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointCloud
    {
        internal readonly Const_PointCloud? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointCloud() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointCloud(Const_PointCloud new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointCloud(Const_PointCloud arg) {return new(arg);}
        public _ByValue_PointCloud(MR.Misc._Moved<PointCloud> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointCloud(MR.Misc._Moved<PointCloud> arg) {return new(arg);}
    }

    /// This is used as a function parameter when the underlying function receives an optional `PointCloud` by value,
    ///   and also has a default argument, meaning it has two different null states.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointCloud`/`Const_PointCloud` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument.
    /// * Pass `MR.Misc.NullOptType` to pass no object.
    public class _ByValueOptOpt_PointCloud
    {
        internal readonly Const_PointCloud? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValueOptOpt_PointCloud() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValueOptOpt_PointCloud(Const_PointCloud new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValueOptOpt_PointCloud(Const_PointCloud arg) {return new(arg);}
        public _ByValueOptOpt_PointCloud(MR.Misc._Moved<PointCloud> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValueOptOpt_PointCloud(MR.Misc._Moved<PointCloud> arg) {return new(arg);}
        public _ByValueOptOpt_PointCloud(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
        public static implicit operator _ByValueOptOpt_PointCloud(MR.Misc.NullOptType nullopt) {return new(nullopt);}
    }

    /// This is used for optional parameters of class `PointCloud` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointCloud`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloud`/`Const_PointCloud` directly.
    public class _InOptMut_PointCloud
    {
        public PointCloud? Opt;

        public _InOptMut_PointCloud() {}
        public _InOptMut_PointCloud(PointCloud value) {Opt = value;}
        public static implicit operator _InOptMut_PointCloud(PointCloud value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointCloud` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointCloud`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloud`/`Const_PointCloud` to pass it to the function.
    public class _InOptConst_PointCloud
    {
        public Const_PointCloud? Opt;

        public _InOptConst_PointCloud() {}
        public _InOptConst_PointCloud(Const_PointCloud value) {Opt = value;}
        public static implicit operator _InOptConst_PointCloud(Const_PointCloud value) {return new(value);}
    }
}
