public static partial class MR
{
    /// polyline that stores points of type V
    /// Generated from class `MR::Polyline2`.
    /// This is the const half of the class.
    public class Const_Polyline2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Polyline2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Destroy", ExactSpelling = true)]
            extern static void __MR_Polyline2_Destroy(_Underlying *_this);
            __MR_Polyline2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polyline2() {Dispose(false);}

        public unsafe MR.Const_PolylineTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Get_topology", ExactSpelling = true)]
                extern static MR.Const_PolylineTopology._Underlying *__MR_Polyline2_Get_topology(_Underlying *_this);
                return new(__MR_Polyline2_Get_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertCoords2 Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Get_points", ExactSpelling = true)]
                extern static MR.Const_VertCoords2._Underlying *__MR_Polyline2_Get_points(_Underlying *_this);
                return new(__MR_Polyline2_Get_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polyline2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_DefaultConstruct();
            _UnderlyingPtr = __MR_Polyline2_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Const_Polyline2(MR._ByValue_Polyline2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polyline2._Underlying *_other);
            _UnderlyingPtr = __MR_Polyline2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Const_Polyline2(MR.Std.Const_Vector_MRVector2f contour) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Construct_1_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_Construct_1_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *contour);
            _UnderlyingPtr = __MR_Polyline2_Construct_1_std_vector_MR_Vector2f(contour._UnderlyingPtr);
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator Const_Polyline2(MR.Std.Const_Vector_MRVector2f contour) {return new(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Const_Polyline2(MR.Std.Const_Vector_StdVectorMRVector2f contours) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Construct_1_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_Construct_1_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours);
            _UnderlyingPtr = __MR_Polyline2_Construct_1_std_vector_std_vector_MR_Vector2f(contours._UnderlyingPtr);
        }

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator Const_Polyline2(MR.Std.Const_Vector_StdVectorMRVector2f contours) {return new(contours);}

        /// creates comp2firstVert.size()-1 not-closed polylines
        /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
        /// \param ps point coordinates
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Const_Polyline2(MR.Std.Const_Vector_MRVertId comp2firstVert, MR._ByValue_VertCoords2 ps) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Construct_2", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_Construct_2(MR.Std.Const_Vector_MRVertId._Underlying *comp2firstVert, MR.Misc._PassBy ps_pass_by, MR.VertCoords2._Underlying *ps);
            _UnderlyingPtr = __MR_Polyline2_Construct_2(comp2firstVert._UnderlyingPtr, ps.PassByMode, ps.Value is not null ? ps.Value._UnderlyingPtr : null);
        }

        /// returns coordinates of the edge origin
        /// Generated from method `MR::Polyline2::orgPnt`.
        public unsafe MR.Vector2f OrgPnt(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_orgPnt", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_orgPnt(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_orgPnt(_UnderlyingPtr, e);
        }

        /// returns coordinates of the edge destination
        /// Generated from method `MR::Polyline2::destPnt`.
        public unsafe MR.Vector2f DestPnt(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_destPnt", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_destPnt(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_destPnt(_UnderlyingPtr, e);
        }

        /// returns a point on the edge: origin point for f=0 and destination point for f=1
        /// Generated from method `MR::Polyline2::edgePoint`.
        public unsafe MR.Vector2f EdgePoint(MR.EdgeId e, float f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgePoint_2", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_edgePoint_2(_Underlying *_this, MR.EdgeId e, float f);
            return __MR_Polyline2_edgePoint_2(_UnderlyingPtr, e, f);
        }

        /// computes coordinates of point given as edge and relative position on it
        /// Generated from method `MR::Polyline2::edgePoint`.
        public unsafe MR.Vector2f EdgePoint(MR.Const_EdgePoint ep)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgePoint_1", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_edgePoint_1(_Underlying *_this, MR.Const_EdgePoint._Underlying *ep);
            return __MR_Polyline2_edgePoint_1(_UnderlyingPtr, ep._UnderlyingPtr);
        }

        /// returns edge's centroid
        /// Generated from method `MR::Polyline2::edgeCenter`.
        public unsafe MR.Vector2f EdgeCenter(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgeCenter", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_edgeCenter(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_edgeCenter(_UnderlyingPtr, e);
        }

        /// returns vector equal to edge destination point minus edge origin point
        /// Generated from method `MR::Polyline2::edgeVector`.
        public unsafe MR.Vector2f EdgeVector(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgeVector", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_edgeVector(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_edgeVector(_UnderlyingPtr, e);
        }

        /// returns line segment of given edge
        /// Generated from method `MR::Polyline2::edgeSegment`.
        public unsafe MR.LineSegm2f EdgeSegment(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgeSegment", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_Polyline2_edgeSegment(_Underlying *_this, MR.EdgeId e);
            return new(__MR_Polyline2_edgeSegment(_UnderlyingPtr, e), is_owning: true);
        }

        /// converts vertex into edge-point representation
        /// Generated from method `MR::Polyline2::toEdgePoint`.
        public unsafe MR.EdgePoint ToEdgePoint(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_toEdgePoint_1", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_Polyline2_toEdgePoint_1(_Underlying *_this, MR.VertId v);
            return new(__MR_Polyline2_toEdgePoint_1(_UnderlyingPtr, v), is_owning: true);
        }

        /// converts edge and point's coordinates into edge-point representation
        /// Generated from method `MR::Polyline2::toEdgePoint`.
        public unsafe MR.EdgePoint ToEdgePoint(MR.EdgeId e, MR.Const_Vector2f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_toEdgePoint_2", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_Polyline2_toEdgePoint_2(_Underlying *_this, MR.EdgeId e, MR.Const_Vector2f._Underlying *p);
            return new(__MR_Polyline2_toEdgePoint_2(_UnderlyingPtr, e, p._UnderlyingPtr), is_owning: true);
        }

        /// returns Euclidean length of the edge
        /// Generated from method `MR::Polyline2::edgeLength`.
        public unsafe float EdgeLength(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgeLength", ExactSpelling = true)]
            extern static float __MR_Polyline2_edgeLength(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_edgeLength(_UnderlyingPtr, e);
        }

        /// returns squared Euclidean length of the edge (faster to compute than length)
        /// Generated from method `MR::Polyline2::edgeLengthSq`.
        public unsafe float EdgeLengthSq(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_edgeLengthSq", ExactSpelling = true)]
            extern static float __MR_Polyline2_edgeLengthSq(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_edgeLengthSq(_UnderlyingPtr, e);
        }

        /// calculates directed loop area if iterating in `e` direction
        /// .z = FLT_MAX if `e` does not represent a loop
        /// Generated from method `MR::Polyline2::loopDirArea`.
        public unsafe MR.Vector3f LoopDirArea(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_loopDirArea", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline2_loopDirArea(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_loopDirArea(_UnderlyingPtr, e);
        }

        /// returns total length of the polyline
        /// Generated from method `MR::Polyline2::totalLength`.
        public unsafe float TotalLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_totalLength", ExactSpelling = true)]
            extern static float __MR_Polyline2_totalLength(_Underlying *_this);
            return __MR_Polyline2_totalLength(_UnderlyingPtr);
        }

        /// returns average edge length in the polyline
        /// Generated from method `MR::Polyline2::averageEdgeLength`.
        public unsafe float AverageEdgeLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_averageEdgeLength", ExactSpelling = true)]
            extern static float __MR_Polyline2_averageEdgeLength(_Underlying *_this);
            return __MR_Polyline2_averageEdgeLength(_UnderlyingPtr);
        }

        /// returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner
        /// Generated from method `MR::Polyline2::getAABBTree`.
        public unsafe MR.Const_AABBTreePolyline2 GetAABBTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_getAABBTree", ExactSpelling = true)]
            extern static MR.Const_AABBTreePolyline2._Underlying *__MR_Polyline2_getAABBTree(_Underlying *_this);
            return new(__MR_Polyline2_getAABBTree(_UnderlyingPtr), is_owning: false);
        }

        /// returns cached aabb-tree for this polyline, but does not create it if it did not exist
        /// Generated from method `MR::Polyline2::getAABBTreeNotCreate`.
        public unsafe MR.Const_AABBTreePolyline2? GetAABBTreeNotCreate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_getAABBTreeNotCreate", ExactSpelling = true)]
            extern static MR.Const_AABBTreePolyline2._Underlying *__MR_Polyline2_getAABBTreeNotCreate(_Underlying *_this);
            var __ret = __MR_Polyline2_getAABBTreeNotCreate(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_AABBTreePolyline2(__ret, is_owning: false) : null;
        }

        /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
        /// Generated from method `MR::Polyline2::getBoundingBox`.
        public unsafe MR.Box2f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box2f __MR_Polyline2_getBoundingBox(_Underlying *_this);
            return __MR_Polyline2_getBoundingBox(_UnderlyingPtr);
        }

        /// passes through all valid points and finds the minimal bounding box containing all of them
        /// \details if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::Polyline2::computeBoundingBox`.
        public unsafe MR.Box2f ComputeBoundingBox(MR.Const_AffineXf2f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_computeBoundingBox", ExactSpelling = true)]
            extern static MR.Box2f __MR_Polyline2_computeBoundingBox(_Underlying *_this, MR.Const_AffineXf2f._Underlying *toWorld);
            return __MR_Polyline2_computeBoundingBox(_UnderlyingPtr, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        // computes average position of all valid polyline vertices
        /// Generated from method `MR::Polyline2::findCenterFromPoints`.
        public unsafe MR.Vector2f FindCenterFromPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_findCenterFromPoints", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Polyline2_findCenterFromPoints(_Underlying *_this);
            return __MR_Polyline2_findCenterFromPoints(_UnderlyingPtr);
        }

        /// convert Polyline to simple contour structures with vector of points inside
        /// \details if all even edges are consistently oriented, then the output contours will be oriented the same
        /// \param vertMap optional output map for for each contour point to corresponding VertId
        /// Generated from method `MR::Polyline2::contours`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> Contours(MR.Std.Vector_StdVectorMRVertId? vertMap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_contours", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_Polyline2_contours(_Underlying *_this, MR.Std.Vector_StdVectorMRVertId._Underlying *vertMap);
            return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_Polyline2_contours(_UnderlyingPtr, vertMap is not null ? vertMap._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Polyline2::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Polyline2_heapBytes(_Underlying *_this);
            return __MR_Polyline2_heapBytes(_UnderlyingPtr);
        }
    }

    /// polyline that stores points of type V
    /// Generated from class `MR::Polyline2`.
    /// This is the non-const half of the class.
    public class Polyline2 : Const_Polyline2
    {
        internal unsafe Polyline2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.PolylineTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_GetMutable_topology", ExactSpelling = true)]
                extern static MR.PolylineTopology._Underlying *__MR_Polyline2_GetMutable_topology(_Underlying *_this);
                return new(__MR_Polyline2_GetMutable_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertCoords2 Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_GetMutable_points", ExactSpelling = true)]
                extern static MR.VertCoords2._Underlying *__MR_Polyline2_GetMutable_points(_Underlying *_this);
                return new(__MR_Polyline2_GetMutable_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polyline2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_DefaultConstruct();
            _UnderlyingPtr = __MR_Polyline2_DefaultConstruct();
        }

        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Polyline2(MR._ByValue_Polyline2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polyline2._Underlying *_other);
            _UnderlyingPtr = __MR_Polyline2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Polyline2(MR.Std.Const_Vector_MRVector2f contour) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Construct_1_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_Construct_1_std_vector_MR_Vector2f(MR.Std.Const_Vector_MRVector2f._Underlying *contour);
            _UnderlyingPtr = __MR_Polyline2_Construct_1_std_vector_MR_Vector2f(contour._UnderlyingPtr);
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator Polyline2(MR.Std.Const_Vector_MRVector2f contour) {return new(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Polyline2(MR.Std.Const_Vector_StdVectorMRVector2f contours) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Construct_1_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_Construct_1_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours);
            _UnderlyingPtr = __MR_Polyline2_Construct_1_std_vector_std_vector_MR_Vector2f(contours._UnderlyingPtr);
        }

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator Polyline2(MR.Std.Const_Vector_StdVectorMRVector2f contours) {return new(contours);}

        /// creates comp2firstVert.size()-1 not-closed polylines
        /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
        /// \param ps point coordinates
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public unsafe Polyline2(MR.Std.Const_Vector_MRVertId comp2firstVert, MR._ByValue_VertCoords2 ps) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_Construct_2", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_Construct_2(MR.Std.Const_Vector_MRVertId._Underlying *comp2firstVert, MR.Misc._PassBy ps_pass_by, MR.VertCoords2._Underlying *ps);
            _UnderlyingPtr = __MR_Polyline2_Construct_2(comp2firstVert._UnderlyingPtr, ps.PassByMode, ps.Value is not null ? ps.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Polyline2::operator=`.
        public unsafe MR.Polyline2 Assign(MR._ByValue_Polyline2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polyline2._Underlying *__MR_Polyline2_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polyline2._Underlying *_other);
            return new(__MR_Polyline2_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// adds connected line in this, passing progressively via points *[vs, vs+num)
        /// \details if closed argument is true then the last and the first points will be additionally connected
        /// \return the edge from first new to second new vertex    
        /// Generated from method `MR::Polyline2::addFromPoints`.
        public unsafe MR.EdgeId AddFromPoints(MR.Const_Vector2f? vs, ulong num, bool closed)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addFromPoints_3", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_addFromPoints_3(_Underlying *_this, MR.Const_Vector2f._Underlying *vs, ulong num, byte closed);
            return __MR_Polyline2_addFromPoints_3(_UnderlyingPtr, vs is not null ? vs._UnderlyingPtr : null, num, closed ? (byte)1 : (byte)0);
        }

        /// adds connected line in this, passing progressively via points *[vs, vs+num)
        /// \details if num > 2 && vs[0] == vs[num-1] then a closed line is created
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline2::addFromPoints`.
        public unsafe MR.EdgeId AddFromPoints(MR.Const_Vector2f? vs, ulong num)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addFromPoints_2", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_addFromPoints_2(_Underlying *_this, MR.Const_Vector2f._Underlying *vs, ulong num);
            return __MR_Polyline2_addFromPoints_2(_UnderlyingPtr, vs is not null ? vs._UnderlyingPtr : null, num);
        }

        /// appends polyline (from) in addition to this polyline: creates new edges, verts and points;
        /// \param outVmap,outEmap (optionally) returns mappings: from.id -> this.id
        /// Generated from method `MR::Polyline2::addPart`.
        public unsafe void AddPart(MR.Const_Polyline2 from, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addPart", ExactSpelling = true)]
            extern static void __MR_Polyline2_addPart(_Underlying *_this, MR.Const_Polyline2._Underlying *from, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            __MR_Polyline2_addPart(_UnderlyingPtr, from._UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// appends polyline (from) in addition to this polyline: creates new edges, verts and points
        /// Generated from method `MR::Polyline2::addPartByMask`.
        public unsafe void AddPartByMask(MR.Const_Polyline2 from, MR.Const_UndirectedEdgeBitSet mask, MR.VertMap? outVmap = null, MR.EdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addPartByMask", ExactSpelling = true)]
            extern static void __MR_Polyline2_addPartByMask(_Underlying *_this, MR.Const_Polyline2._Underlying *from, MR.Const_UndirectedEdgeBitSet._Underlying *mask, MR.VertMap._Underlying *outVmap, MR.EdgeMap._Underlying *outEmap);
            __MR_Polyline2_addPartByMask(_UnderlyingPtr, from._UnderlyingPtr, mask._UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// tightly packs all arrays eliminating lone edges and invalid verts and points,
        /// optionally returns mappings: old.id -> new.id
        /// Generated from method `MR::Polyline2::pack`.
        public unsafe void Pack(MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_pack", ExactSpelling = true)]
            extern static void __MR_Polyline2_pack(_Underlying *_this, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            __MR_Polyline2_pack(_UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// applies given transformation to all valid polyline vertices
        /// Generated from method `MR::Polyline2::transform`.
        public unsafe void Transform(MR.Const_AffineXf2f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_transform", ExactSpelling = true)]
            extern static void __MR_Polyline2_transform(_Underlying *_this, MR.Const_AffineXf2f._Underlying *xf);
            __MR_Polyline2_transform(_UnderlyingPtr, xf._UnderlyingPtr);
        }

        /// split given edge on two parts:
        /// dest(returned-edge) = org(e) - newly created vertex,
        /// org(returned-edge) = org(e-before-split),
        /// dest(e) = dest(e-before-split)
        /// Generated from method `MR::Polyline2::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e, MR.Const_Vector2f newVertPos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_splitEdge_2", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_splitEdge_2(_Underlying *_this, MR.EdgeId e, MR.Const_Vector2f._Underlying *newVertPos);
            return __MR_Polyline2_splitEdge_2(_UnderlyingPtr, e, newVertPos._UnderlyingPtr);
        }

        // same, but split given edge on two equal parts
        /// Generated from method `MR::Polyline2::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_splitEdge_1", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_splitEdge_1(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline2_splitEdge_1(_UnderlyingPtr, e);
        }

        /// Invalidates caches (e.g. aabb-tree) after a change in polyline
        /// Generated from method `MR::Polyline2::invalidateCaches`.
        public unsafe void InvalidateCaches()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_invalidateCaches", ExactSpelling = true)]
            extern static void __MR_Polyline2_invalidateCaches(_Underlying *_this);
            __MR_Polyline2_invalidateCaches(_UnderlyingPtr);
        }

        /// adds path to this polyline
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline2::addFromEdgePath`.
        public unsafe MR.EdgeId AddFromEdgePath(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addFromEdgePath", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_addFromEdgePath(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *path);
            return __MR_Polyline2_addFromEdgePath(_UnderlyingPtr, mesh._UnderlyingPtr, path._UnderlyingPtr);
        }

        /// adds path to this polyline
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline2::addFromSurfacePath`.
        public unsafe MR.EdgeId AddFromSurfacePath(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addFromSurfacePath", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_addFromSurfacePath(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *path);
            return __MR_Polyline2_addFromSurfacePath(_UnderlyingPtr, mesh._UnderlyingPtr, path._UnderlyingPtr);
        }

        /// adds general path = start-path-end (where both start and end are optional) to this polyline
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline2::addFromGeneralSurfacePath`.
        public unsafe MR.EdgeId AddFromGeneralSurfacePath(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Std.Const_Vector_MREdgePoint path, MR.Const_MeshTriPoint end)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline2_addFromGeneralSurfacePath", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline2_addFromGeneralSurfacePath(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Std.Const_Vector_MREdgePoint._Underlying *path, MR.Const_MeshTriPoint._Underlying *end);
            return __MR_Polyline2_addFromGeneralSurfacePath(_UnderlyingPtr, mesh._UnderlyingPtr, start._UnderlyingPtr, path._UnderlyingPtr, end._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polyline2` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polyline2`/`Const_Polyline2` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polyline2
    {
        internal readonly Const_Polyline2? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polyline2() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polyline2(Const_Polyline2 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polyline2(Const_Polyline2 arg) {return new(arg);}
        public _ByValue_Polyline2(MR.Misc._Moved<Polyline2> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polyline2(MR.Misc._Moved<Polyline2> arg) {return new(arg);}

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator _ByValue_Polyline2(MR.Std.Const_Vector_MRVector2f contour) {return new MR.Polyline2(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator _ByValue_Polyline2(MR.Std.Const_Vector_StdVectorMRVector2f contours) {return new MR.Polyline2(contours);}
    }

    /// This is used for optional parameters of class `Polyline2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polyline2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polyline2`/`Const_Polyline2` directly.
    public class _InOptMut_Polyline2
    {
        public Polyline2? Opt;

        public _InOptMut_Polyline2() {}
        public _InOptMut_Polyline2(Polyline2 value) {Opt = value;}
        public static implicit operator _InOptMut_Polyline2(Polyline2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polyline2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polyline2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polyline2`/`Const_Polyline2` to pass it to the function.
    public class _InOptConst_Polyline2
    {
        public Const_Polyline2? Opt;

        public _InOptConst_Polyline2() {}
        public _InOptConst_Polyline2(Const_Polyline2 value) {Opt = value;}
        public static implicit operator _InOptConst_Polyline2(Const_Polyline2 value) {return new(value);}

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator _InOptConst_Polyline2(MR.Std.Const_Vector_MRVector2f contour) {return new MR.Polyline2(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline2::Polyline2`.
        public static unsafe implicit operator _InOptConst_Polyline2(MR.Std.Const_Vector_StdVectorMRVector2f contours) {return new MR.Polyline2(contours);}
    }

    /// polyline that stores points of type V
    /// Generated from class `MR::Polyline3`.
    /// This is the const half of the class.
    public class Const_Polyline3 : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_Polyline3_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_Polyline3_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_Polyline3_UseCount();
                return __MR_std_shared_ptr_MR_Polyline3_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Polyline3_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_Polyline3_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_Polyline3(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Polyline3_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Polyline3_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Polyline3_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Polyline3_ConstructNonOwning(ptr);
        }

        internal unsafe Const_Polyline3(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe Polyline3 _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Polyline3_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_Polyline3_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Polyline3_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Polyline3_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Polyline3_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_Polyline3_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_Polyline3_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Polyline3() {Dispose(false);}

        public unsafe MR.Const_PolylineTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Get_topology", ExactSpelling = true)]
                extern static MR.Const_PolylineTopology._Underlying *__MR_Polyline3_Get_topology(_Underlying *_this);
                return new(__MR_Polyline3_Get_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Get_points", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_Polyline3_Get_points(_Underlying *_this);
                return new(__MR_Polyline3_Get_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Polyline3() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_DefaultConstruct();
            _LateMakeShared(__MR_Polyline3_DefaultConstruct());
        }

        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Const_Polyline3(MR._ByValue_Polyline3 _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polyline3._Underlying *_other);
            _LateMakeShared(__MR_Polyline3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Const_Polyline3(MR.Std.Const_Vector_MRVector3f contour) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Construct_1_std_vector_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_Construct_1_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *contour);
            _LateMakeShared(__MR_Polyline3_Construct_1_std_vector_MR_Vector3f(contour._UnderlyingPtr));
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator Const_Polyline3(MR.Std.Const_Vector_MRVector3f contour) {return new(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Const_Polyline3(MR.Std.Const_Vector_StdVectorMRVector3f contours) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Construct_1_std_vector_std_vector_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_Construct_1_std_vector_std_vector_MR_Vector3f(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours);
            _LateMakeShared(__MR_Polyline3_Construct_1_std_vector_std_vector_MR_Vector3f(contours._UnderlyingPtr));
        }

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator Const_Polyline3(MR.Std.Const_Vector_StdVectorMRVector3f contours) {return new(contours);}

        /// creates comp2firstVert.size()-1 not-closed polylines
        /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
        /// \param ps point coordinates
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Const_Polyline3(MR.Std.Const_Vector_MRVertId comp2firstVert, MR._ByValue_VertCoords ps) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Construct_2", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_Construct_2(MR.Std.Const_Vector_MRVertId._Underlying *comp2firstVert, MR.Misc._PassBy ps_pass_by, MR.VertCoords._Underlying *ps);
            _LateMakeShared(__MR_Polyline3_Construct_2(comp2firstVert._UnderlyingPtr, ps.PassByMode, ps.Value is not null ? ps.Value._UnderlyingPtr : null));
        }

        /// returns coordinates of the edge origin
        /// Generated from method `MR::Polyline3::orgPnt`.
        public unsafe MR.Vector3f OrgPnt(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_orgPnt", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_orgPnt(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_orgPnt(_UnderlyingPtr, e);
        }

        /// returns coordinates of the edge destination
        /// Generated from method `MR::Polyline3::destPnt`.
        public unsafe MR.Vector3f DestPnt(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_destPnt", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_destPnt(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_destPnt(_UnderlyingPtr, e);
        }

        /// returns a point on the edge: origin point for f=0 and destination point for f=1
        /// Generated from method `MR::Polyline3::edgePoint`.
        public unsafe MR.Vector3f EdgePoint(MR.EdgeId e, float f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgePoint_2", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_edgePoint_2(_Underlying *_this, MR.EdgeId e, float f);
            return __MR_Polyline3_edgePoint_2(_UnderlyingPtr, e, f);
        }

        /// computes coordinates of point given as edge and relative position on it
        /// Generated from method `MR::Polyline3::edgePoint`.
        public unsafe MR.Vector3f EdgePoint(MR.Const_EdgePoint ep)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgePoint_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_edgePoint_1(_Underlying *_this, MR.Const_EdgePoint._Underlying *ep);
            return __MR_Polyline3_edgePoint_1(_UnderlyingPtr, ep._UnderlyingPtr);
        }

        /// returns edge's centroid
        /// Generated from method `MR::Polyline3::edgeCenter`.
        public unsafe MR.Vector3f EdgeCenter(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgeCenter", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_edgeCenter(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_edgeCenter(_UnderlyingPtr, e);
        }

        /// returns vector equal to edge destination point minus edge origin point
        /// Generated from method `MR::Polyline3::edgeVector`.
        public unsafe MR.Vector3f EdgeVector(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgeVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_edgeVector(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_edgeVector(_UnderlyingPtr, e);
        }

        /// returns line segment of given edge
        /// Generated from method `MR::Polyline3::edgeSegment`.
        public unsafe MR.LineSegm3f EdgeSegment(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgeSegment", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_Polyline3_edgeSegment(_Underlying *_this, MR.EdgeId e);
            return new(__MR_Polyline3_edgeSegment(_UnderlyingPtr, e), is_owning: true);
        }

        /// converts vertex into edge-point representation
        /// Generated from method `MR::Polyline3::toEdgePoint`.
        public unsafe MR.EdgePoint ToEdgePoint(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_toEdgePoint_1", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_Polyline3_toEdgePoint_1(_Underlying *_this, MR.VertId v);
            return new(__MR_Polyline3_toEdgePoint_1(_UnderlyingPtr, v), is_owning: true);
        }

        /// converts edge and point's coordinates into edge-point representation
        /// Generated from method `MR::Polyline3::toEdgePoint`.
        public unsafe MR.EdgePoint ToEdgePoint(MR.EdgeId e, MR.Const_Vector3f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_toEdgePoint_2", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_Polyline3_toEdgePoint_2(_Underlying *_this, MR.EdgeId e, MR.Const_Vector3f._Underlying *p);
            return new(__MR_Polyline3_toEdgePoint_2(_UnderlyingPtr, e, p._UnderlyingPtr), is_owning: true);
        }

        /// returns Euclidean length of the edge
        /// Generated from method `MR::Polyline3::edgeLength`.
        public unsafe float EdgeLength(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgeLength", ExactSpelling = true)]
            extern static float __MR_Polyline3_edgeLength(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_edgeLength(_UnderlyingPtr, e);
        }

        /// returns squared Euclidean length of the edge (faster to compute than length)
        /// Generated from method `MR::Polyline3::edgeLengthSq`.
        public unsafe float EdgeLengthSq(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_edgeLengthSq", ExactSpelling = true)]
            extern static float __MR_Polyline3_edgeLengthSq(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_edgeLengthSq(_UnderlyingPtr, e);
        }

        /// calculates directed loop area if iterating in `e` direction
        /// .z = FLT_MAX if `e` does not represent a loop
        /// Generated from method `MR::Polyline3::loopDirArea`.
        public unsafe MR.Vector3f LoopDirArea(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_loopDirArea", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_loopDirArea(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_loopDirArea(_UnderlyingPtr, e);
        }

        /// returns total length of the polyline
        /// Generated from method `MR::Polyline3::totalLength`.
        public unsafe float TotalLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_totalLength", ExactSpelling = true)]
            extern static float __MR_Polyline3_totalLength(_Underlying *_this);
            return __MR_Polyline3_totalLength(_UnderlyingPtr);
        }

        /// returns average edge length in the polyline
        /// Generated from method `MR::Polyline3::averageEdgeLength`.
        public unsafe float AverageEdgeLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_averageEdgeLength", ExactSpelling = true)]
            extern static float __MR_Polyline3_averageEdgeLength(_Underlying *_this);
            return __MR_Polyline3_averageEdgeLength(_UnderlyingPtr);
        }

        /// returns cached aabb-tree for this polyline, creating it if it did not exist in a thread-safe manner
        /// Generated from method `MR::Polyline3::getAABBTree`.
        public unsafe MR.Const_AABBTreePolyline3 GetAABBTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_getAABBTree", ExactSpelling = true)]
            extern static MR.Const_AABBTreePolyline3._Underlying *__MR_Polyline3_getAABBTree(_Underlying *_this);
            return new(__MR_Polyline3_getAABBTree(_UnderlyingPtr), is_owning: false);
        }

        /// returns cached aabb-tree for this polyline, but does not create it if it did not exist
        /// Generated from method `MR::Polyline3::getAABBTreeNotCreate`.
        public unsafe MR.Const_AABBTreePolyline3? GetAABBTreeNotCreate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_getAABBTreeNotCreate", ExactSpelling = true)]
            extern static MR.Const_AABBTreePolyline3._Underlying *__MR_Polyline3_getAABBTreeNotCreate(_Underlying *_this);
            var __ret = __MR_Polyline3_getAABBTreeNotCreate(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_AABBTreePolyline3(__ret, is_owning: false) : null;
        }

        /// returns the minimal bounding box containing all valid vertices (implemented via getAABBTree())
        /// Generated from method `MR::Polyline3::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_Polyline3_getBoundingBox(_Underlying *_this);
            return __MR_Polyline3_getBoundingBox(_UnderlyingPtr);
        }

        /// passes through all valid points and finds the minimal bounding box containing all of them
        /// \details if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::Polyline3::computeBoundingBox`.
        public unsafe MR.Box3f ComputeBoundingBox(MR.Const_AffineXf3f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_computeBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_Polyline3_computeBoundingBox(_Underlying *_this, MR.Const_AffineXf3f._Underlying *toWorld);
            return __MR_Polyline3_computeBoundingBox(_UnderlyingPtr, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        // computes average position of all valid polyline vertices
        /// Generated from method `MR::Polyline3::findCenterFromPoints`.
        public unsafe MR.Vector3f FindCenterFromPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_findCenterFromPoints", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Polyline3_findCenterFromPoints(_Underlying *_this);
            return __MR_Polyline3_findCenterFromPoints(_UnderlyingPtr);
        }

        /// convert Polyline to simple contour structures with vector of points inside
        /// \details if all even edges are consistently oriented, then the output contours will be oriented the same
        /// \param vertMap optional output map for for each contour point to corresponding VertId
        /// Generated from method `MR::Polyline3::contours`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector3f> Contours(MR.Std.Vector_StdVectorMRVertId? vertMap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_contours", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMRVector3f._Underlying *__MR_Polyline3_contours(_Underlying *_this, MR.Std.Vector_StdVectorMRVertId._Underlying *vertMap);
            return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector3f(__MR_Polyline3_contours(_UnderlyingPtr, vertMap is not null ? vertMap._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Polyline3::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Polyline3_heapBytes(_Underlying *_this);
            return __MR_Polyline3_heapBytes(_UnderlyingPtr);
        }
    }

    /// polyline that stores points of type V
    /// Generated from class `MR::Polyline3`.
    /// This is the non-const half of the class.
    public class Polyline3 : Const_Polyline3
    {
        internal unsafe Polyline3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe Polyline3(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        public new unsafe MR.PolylineTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_GetMutable_topology", ExactSpelling = true)]
                extern static MR.PolylineTopology._Underlying *__MR_Polyline3_GetMutable_topology(_Underlying *_this);
                return new(__MR_Polyline3_GetMutable_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_GetMutable_points", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_Polyline3_GetMutable_points(_Underlying *_this);
                return new(__MR_Polyline3_GetMutable_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Polyline3() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_DefaultConstruct();
            _LateMakeShared(__MR_Polyline3_DefaultConstruct());
        }

        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Polyline3(MR._ByValue_Polyline3 _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Polyline3._Underlying *_other);
            _LateMakeShared(__MR_Polyline3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Polyline3(MR.Std.Const_Vector_MRVector3f contour) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Construct_1_std_vector_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_Construct_1_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *contour);
            _LateMakeShared(__MR_Polyline3_Construct_1_std_vector_MR_Vector3f(contour._UnderlyingPtr));
        }

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator Polyline3(MR.Std.Const_Vector_MRVector3f contour) {return new(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Polyline3(MR.Std.Const_Vector_StdVectorMRVector3f contours) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Construct_1_std_vector_std_vector_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_Construct_1_std_vector_std_vector_MR_Vector3f(MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours);
            _LateMakeShared(__MR_Polyline3_Construct_1_std_vector_std_vector_MR_Vector3f(contours._UnderlyingPtr));
        }

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator Polyline3(MR.Std.Const_Vector_StdVectorMRVector3f contours) {return new(contours);}

        /// creates comp2firstVert.size()-1 not-closed polylines
        /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
        /// \param ps point coordinates
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public unsafe Polyline3(MR.Std.Const_Vector_MRVertId comp2firstVert, MR._ByValue_VertCoords ps) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_Construct_2", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_Construct_2(MR.Std.Const_Vector_MRVertId._Underlying *comp2firstVert, MR.Misc._PassBy ps_pass_by, MR.VertCoords._Underlying *ps);
            _LateMakeShared(__MR_Polyline3_Construct_2(comp2firstVert._UnderlyingPtr, ps.PassByMode, ps.Value is not null ? ps.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::Polyline3::operator=`.
        public unsafe MR.Polyline3 Assign(MR._ByValue_Polyline3 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Polyline3._Underlying *__MR_Polyline3_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Polyline3._Underlying *_other);
            return new(__MR_Polyline3_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// adds connected line in this, passing progressively via points *[vs, vs+num)
        /// \details if closed argument is true then the last and the first points will be additionally connected
        /// \return the edge from first new to second new vertex    
        /// Generated from method `MR::Polyline3::addFromPoints`.
        public unsafe MR.EdgeId AddFromPoints(MR.Const_Vector3f? vs, ulong num, bool closed)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addFromPoints_3", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_addFromPoints_3(_Underlying *_this, MR.Const_Vector3f._Underlying *vs, ulong num, byte closed);
            return __MR_Polyline3_addFromPoints_3(_UnderlyingPtr, vs is not null ? vs._UnderlyingPtr : null, num, closed ? (byte)1 : (byte)0);
        }

        /// adds connected line in this, passing progressively via points *[vs, vs+num)
        /// \details if num > 2 && vs[0] == vs[num-1] then a closed line is created
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline3::addFromPoints`.
        public unsafe MR.EdgeId AddFromPoints(MR.Const_Vector3f? vs, ulong num)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addFromPoints_2", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_addFromPoints_2(_Underlying *_this, MR.Const_Vector3f._Underlying *vs, ulong num);
            return __MR_Polyline3_addFromPoints_2(_UnderlyingPtr, vs is not null ? vs._UnderlyingPtr : null, num);
        }

        /// appends polyline (from) in addition to this polyline: creates new edges, verts and points;
        /// \param outVmap,outEmap (optionally) returns mappings: from.id -> this.id
        /// Generated from method `MR::Polyline3::addPart`.
        public unsafe void AddPart(MR.Const_Polyline3 from, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addPart", ExactSpelling = true)]
            extern static void __MR_Polyline3_addPart(_Underlying *_this, MR.Const_Polyline3._Underlying *from, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            __MR_Polyline3_addPart(_UnderlyingPtr, from._UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// appends polyline (from) in addition to this polyline: creates new edges, verts and points
        /// Generated from method `MR::Polyline3::addPartByMask`.
        public unsafe void AddPartByMask(MR.Const_Polyline3 from, MR.Const_UndirectedEdgeBitSet mask, MR.VertMap? outVmap = null, MR.EdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addPartByMask", ExactSpelling = true)]
            extern static void __MR_Polyline3_addPartByMask(_Underlying *_this, MR.Const_Polyline3._Underlying *from, MR.Const_UndirectedEdgeBitSet._Underlying *mask, MR.VertMap._Underlying *outVmap, MR.EdgeMap._Underlying *outEmap);
            __MR_Polyline3_addPartByMask(_UnderlyingPtr, from._UnderlyingPtr, mask._UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// tightly packs all arrays eliminating lone edges and invalid verts and points,
        /// optionally returns mappings: old.id -> new.id
        /// Generated from method `MR::Polyline3::pack`.
        public unsafe void Pack(MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_pack", ExactSpelling = true)]
            extern static void __MR_Polyline3_pack(_Underlying *_this, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            __MR_Polyline3_pack(_UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// applies given transformation to all valid polyline vertices
        /// Generated from method `MR::Polyline3::transform`.
        public unsafe void Transform(MR.Const_AffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_transform", ExactSpelling = true)]
            extern static void __MR_Polyline3_transform(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf);
            __MR_Polyline3_transform(_UnderlyingPtr, xf._UnderlyingPtr);
        }

        /// split given edge on two parts:
        /// dest(returned-edge) = org(e) - newly created vertex,
        /// org(returned-edge) = org(e-before-split),
        /// dest(e) = dest(e-before-split)
        /// Generated from method `MR::Polyline3::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e, MR.Const_Vector3f newVertPos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_splitEdge_2", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_splitEdge_2(_Underlying *_this, MR.EdgeId e, MR.Const_Vector3f._Underlying *newVertPos);
            return __MR_Polyline3_splitEdge_2(_UnderlyingPtr, e, newVertPos._UnderlyingPtr);
        }

        // same, but split given edge on two equal parts
        /// Generated from method `MR::Polyline3::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_splitEdge_1", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_splitEdge_1(_Underlying *_this, MR.EdgeId e);
            return __MR_Polyline3_splitEdge_1(_UnderlyingPtr, e);
        }

        /// Invalidates caches (e.g. aabb-tree) after a change in polyline
        /// Generated from method `MR::Polyline3::invalidateCaches`.
        public unsafe void InvalidateCaches()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_invalidateCaches", ExactSpelling = true)]
            extern static void __MR_Polyline3_invalidateCaches(_Underlying *_this);
            __MR_Polyline3_invalidateCaches(_UnderlyingPtr);
        }

        /// adds path to this polyline
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline3::addFromEdgePath`.
        public unsafe MR.EdgeId AddFromEdgePath(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addFromEdgePath", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_addFromEdgePath(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *path);
            return __MR_Polyline3_addFromEdgePath(_UnderlyingPtr, mesh._UnderlyingPtr, path._UnderlyingPtr);
        }

        /// adds path to this polyline
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline3::addFromSurfacePath`.
        public unsafe MR.EdgeId AddFromSurfacePath(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addFromSurfacePath", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_addFromSurfacePath(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *path);
            return __MR_Polyline3_addFromSurfacePath(_UnderlyingPtr, mesh._UnderlyingPtr, path._UnderlyingPtr);
        }

        /// adds general path = start-path-end (where both start and end are optional) to this polyline
        /// \return the edge from first new to second new vertex
        /// Generated from method `MR::Polyline3::addFromGeneralSurfacePath`.
        public unsafe MR.EdgeId AddFromGeneralSurfacePath(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Std.Const_Vector_MREdgePoint path, MR.Const_MeshTriPoint end)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_addFromGeneralSurfacePath", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Polyline3_addFromGeneralSurfacePath(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Std.Const_Vector_MREdgePoint._Underlying *path, MR.Const_MeshTriPoint._Underlying *end);
            return __MR_Polyline3_addFromGeneralSurfacePath(_UnderlyingPtr, mesh._UnderlyingPtr, start._UnderlyingPtr, path._UnderlyingPtr, end._UnderlyingPtr);
        }

        /// reflects the polyline from a given plane. Enabled only for Polyline3f
        /// Generated from method `MR::Polyline3::mirror<MR::Vector3f>`.
        public unsafe void Mirror(MR.Const_Plane3f plane)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Polyline3_mirror", ExactSpelling = true)]
            extern static void __MR_Polyline3_mirror(_Underlying *_this, MR.Const_Plane3f._Underlying *plane);
            __MR_Polyline3_mirror(_UnderlyingPtr, plane._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Polyline3` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Polyline3`/`Const_Polyline3` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Polyline3
    {
        internal readonly Const_Polyline3? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Polyline3() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Polyline3(Const_Polyline3 new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Polyline3(Const_Polyline3 arg) {return new(arg);}
        public _ByValue_Polyline3(MR.Misc._Moved<Polyline3> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Polyline3(MR.Misc._Moved<Polyline3> arg) {return new(arg);}

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator _ByValue_Polyline3(MR.Std.Const_Vector_MRVector3f contour) {return new MR.Polyline3(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator _ByValue_Polyline3(MR.Std.Const_Vector_StdVectorMRVector3f contours) {return new MR.Polyline3(contours);}
    }

    /// This is used for optional parameters of class `Polyline3` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Polyline3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polyline3`/`Const_Polyline3` directly.
    public class _InOptMut_Polyline3
    {
        public Polyline3? Opt;

        public _InOptMut_Polyline3() {}
        public _InOptMut_Polyline3(Polyline3 value) {Opt = value;}
        public static implicit operator _InOptMut_Polyline3(Polyline3 value) {return new(value);}
    }

    /// This is used for optional parameters of class `Polyline3` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Polyline3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Polyline3`/`Const_Polyline3` to pass it to the function.
    public class _InOptConst_Polyline3
    {
        public Const_Polyline3? Opt;

        public _InOptConst_Polyline3() {}
        public _InOptConst_Polyline3(Const_Polyline3 value) {Opt = value;}
        public static implicit operator _InOptConst_Polyline3(Const_Polyline3 value) {return new(value);}

        /// creates polyline from one contour (open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator _InOptConst_Polyline3(MR.Std.Const_Vector_MRVector3f contour) {return new MR.Polyline3(contour);}

        /// creates polyline from several contours (each can be open or closed)
        /// Generated from constructor `MR::Polyline3::Polyline3`.
        public static unsafe implicit operator _InOptConst_Polyline3(MR.Std.Const_Vector_StdVectorMRVector3f contours) {return new MR.Polyline3(contours);}
    }
}
