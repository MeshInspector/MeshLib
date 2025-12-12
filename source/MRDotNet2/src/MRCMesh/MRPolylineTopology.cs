public static partial class MR
{
    /// topology of one or several polylines (how line segments are connected in lines) common for 2D and 3D polylines
    /// Generated from class `MR::PolylineTopology`.
    /// This is the const half of the class.
    public class Const_PolylineTopology : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_PolylineTopology>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineTopology(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_Destroy(_Underlying *_this);
            __MR_PolylineTopology_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineTopology() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineTopology() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineTopology._Underlying *__MR_PolylineTopology_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineTopology_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineTopology::PolylineTopology`.
        public unsafe Const_PolylineTopology(MR._ByValue_PolylineTopology _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTopology._Underlying *__MR_PolylineTopology_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineTopology._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineTopology_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// checks whether the edge is disconnected from all other edges and disassociated from all vertices (as if after makeEdge)
        /// Generated from method `MR::PolylineTopology::isLoneEdge`.
        public unsafe bool IsLoneEdge(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_isLoneEdge", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_isLoneEdge(_Underlying *_this, MR.EdgeId a);
            return __MR_PolylineTopology_isLoneEdge(_UnderlyingPtr, a) != 0;
        }

        /// returns last not lone undirected edge id, or invalid id if no such edge exists
        /// Generated from method `MR::PolylineTopology::lastNotLoneUndirectedEdge`.
        public unsafe MR.UndirectedEdgeId LastNotLoneUndirectedEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_lastNotLoneUndirectedEdge", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_PolylineTopology_lastNotLoneUndirectedEdge(_Underlying *_this);
            return __MR_PolylineTopology_lastNotLoneUndirectedEdge(_UnderlyingPtr);
        }

        /// returns last not lone edge id, or invalid id if no such edge exists
        /// Generated from method `MR::PolylineTopology::lastNotLoneEdge`.
        public unsafe MR.EdgeId LastNotLoneEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_lastNotLoneEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_lastNotLoneEdge(_Underlying *_this);
            return __MR_PolylineTopology_lastNotLoneEdge(_UnderlyingPtr);
        }

        /// returns the number of half-edge records including lone ones
        /// Generated from method `MR::PolylineTopology::edgeSize`.
        public unsafe ulong EdgeSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_edgeSize", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_edgeSize(_Underlying *_this);
            return __MR_PolylineTopology_edgeSize(_UnderlyingPtr);
        }

        /// returns the number of allocated edge records
        /// Generated from method `MR::PolylineTopology::edgeCapacity`.
        public unsafe ulong EdgeCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_edgeCapacity", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_edgeCapacity(_Underlying *_this);
            return __MR_PolylineTopology_edgeCapacity(_UnderlyingPtr);
        }

        /// returns the number of undirected edges (pairs of half-edges) including lone ones
        /// Generated from method `MR::PolylineTopology::undirectedEdgeSize`.
        public unsafe ulong UndirectedEdgeSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_undirectedEdgeSize", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_undirectedEdgeSize(_Underlying *_this);
            return __MR_PolylineTopology_undirectedEdgeSize(_UnderlyingPtr);
        }

        /// returns the number of allocated undirected edges (pairs of half-edges)
        /// Generated from method `MR::PolylineTopology::undirectedEdgeCapacity`.
        public unsafe ulong UndirectedEdgeCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_undirectedEdgeCapacity", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_undirectedEdgeCapacity(_Underlying *_this);
            return __MR_PolylineTopology_undirectedEdgeCapacity(_UnderlyingPtr);
        }

        /// computes the number of not-lone (valid) undirected edges
        /// Generated from method `MR::PolylineTopology::computeNotLoneUndirectedEdges`.
        public unsafe ulong ComputeNotLoneUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_computeNotLoneUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_computeNotLoneUndirectedEdges(_Underlying *_this);
            return __MR_PolylineTopology_computeNotLoneUndirectedEdges(_UnderlyingPtr);
        }

        /// returns true if given edge is within valid range and not-lone
        /// Generated from method `MR::PolylineTopology::hasEdge`.
        public unsafe bool HasEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_hasEdge", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_hasEdge(_Underlying *_this, MR.EdgeId e);
            return __MR_PolylineTopology_hasEdge(_UnderlyingPtr, e) != 0;
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::PolylineTopology::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_heapBytes(_Underlying *_this);
            return __MR_PolylineTopology_heapBytes(_UnderlyingPtr);
        }

        /// next (counter clock wise) half-edge in the origin ring
        /// Generated from method `MR::PolylineTopology::next`.
        public unsafe MR.EdgeId Next(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_next(_Underlying *_this, MR.EdgeId he);
            return __MR_PolylineTopology_next(_UnderlyingPtr, he);
        }

        /// returns origin vertex of half-edge
        /// Generated from method `MR::PolylineTopology::org`.
        public unsafe MR.VertId Org(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_org", ExactSpelling = true)]
            extern static MR.VertId __MR_PolylineTopology_org(_Underlying *_this, MR.EdgeId he);
            return __MR_PolylineTopology_org(_UnderlyingPtr, he);
        }

        /// returns destination vertex of half-edge
        /// Generated from method `MR::PolylineTopology::dest`.
        public unsafe MR.VertId Dest(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_dest", ExactSpelling = true)]
            extern static MR.VertId __MR_PolylineTopology_dest(_Underlying *_this, MR.EdgeId he);
            return __MR_PolylineTopology_dest(_UnderlyingPtr, he);
        }

        /// for all valid vertices this vector contains an edge with the origin there
        /// Generated from method `MR::PolylineTopology::edgePerVertex`.
        public unsafe MR.Const_Vector_MREdgeId_MRVertId EdgePerVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_edgePerVertex", ExactSpelling = true)]
            extern static MR.Const_Vector_MREdgeId_MRVertId._Underlying *__MR_PolylineTopology_edgePerVertex(_Underlying *_this);
            return new(__MR_PolylineTopology_edgePerVertex(_UnderlyingPtr), is_owning: false);
        }

        /// for all edges this vector contains its origin
        /// Generated from method `MR::PolylineTopology::getOrgs`.
        public unsafe MR.Misc._Moved<MR.Vector_MRVertId_MREdgeId> GetOrgs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_getOrgs", ExactSpelling = true)]
            extern static MR.Vector_MRVertId_MREdgeId._Underlying *__MR_PolylineTopology_getOrgs(_Underlying *_this);
            return MR.Misc.Move(new MR.Vector_MRVertId_MREdgeId(__MR_PolylineTopology_getOrgs(_UnderlyingPtr), is_owning: true));
        }

        /// returns valid edge if given vertex is present in the mesh
        /// Generated from method `MR::PolylineTopology::edgeWithOrg`.
        public unsafe MR.EdgeId EdgeWithOrg(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_edgeWithOrg", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_edgeWithOrg(_Underlying *_this, MR.VertId a);
            return __MR_PolylineTopology_edgeWithOrg(_UnderlyingPtr, a);
        }

        /// returns true if given vertex is present in the mesh
        /// Generated from method `MR::PolylineTopology::hasVert`.
        public unsafe bool HasVert(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_hasVert", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_hasVert(_Underlying *_this, MR.VertId a);
            return __MR_PolylineTopology_hasVert(_UnderlyingPtr, a) != 0;
        }

        /// returns 0 if given vertex does not exist, 1 if it has one incident edge, and 2 if it has two incident edges
        /// Generated from method `MR::PolylineTopology::getVertDegree`.
        public unsafe int GetVertDegree(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_getVertDegree", ExactSpelling = true)]
            extern static int __MR_PolylineTopology_getVertDegree(_Underlying *_this, MR.VertId a);
            return __MR_PolylineTopology_getVertDegree(_UnderlyingPtr, a);
        }

        /// returns the number of valid vertices
        /// Generated from method `MR::PolylineTopology::numValidVerts`.
        public unsafe int NumValidVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_numValidVerts", ExactSpelling = true)]
            extern static int __MR_PolylineTopology_numValidVerts(_Underlying *_this);
            return __MR_PolylineTopology_numValidVerts(_UnderlyingPtr);
        }

        /// returns last valid vertex id, or invalid id if no single valid vertex exists
        /// Generated from method `MR::PolylineTopology::lastValidVert`.
        public unsafe MR.VertId LastValidVert()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_lastValidVert", ExactSpelling = true)]
            extern static MR.VertId __MR_PolylineTopology_lastValidVert(_Underlying *_this);
            return __MR_PolylineTopology_lastValidVert(_UnderlyingPtr);
        }

        /// returns the number of vertex records including invalid ones
        /// Generated from method `MR::PolylineTopology::vertSize`.
        public unsafe ulong VertSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_vertSize", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_vertSize(_Underlying *_this);
            return __MR_PolylineTopology_vertSize(_UnderlyingPtr);
        }

        /// returns the number of allocated vert records
        /// Generated from method `MR::PolylineTopology::vertCapacity`.
        public unsafe ulong VertCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_vertCapacity", ExactSpelling = true)]
            extern static ulong __MR_PolylineTopology_vertCapacity(_Underlying *_this);
            return __MR_PolylineTopology_vertCapacity(_UnderlyingPtr);
        }

        /// returns cached set of all valid vertices
        /// Generated from method `MR::PolylineTopology::getValidVerts`.
        public unsafe MR.Const_VertBitSet GetValidVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_getValidVerts", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_PolylineTopology_getValidVerts(_Underlying *_this);
            return new(__MR_PolylineTopology_getValidVerts(_UnderlyingPtr), is_owning: false);
        }

        /// if region pointer is not null then converts it in reference, otherwise returns all valid vertices in the mesh
        /// Generated from method `MR::PolylineTopology::getVertIds`.
        public unsafe MR.Const_VertBitSet GetVertIds(MR.Const_VertBitSet? region)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_getVertIds", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_PolylineTopology_getVertIds(_Underlying *_this, MR.Const_VertBitSet._Underlying *region);
            return new(__MR_PolylineTopology_getVertIds(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: false);
        }

        /// finds and returns edge from o to d in the mesh; returns invalid edge otherwise
        /// Generated from method `MR::PolylineTopology::findEdge`.
        public unsafe MR.EdgeId FindEdge(MR.VertId o, MR.VertId d)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_findEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_findEdge(_Underlying *_this, MR.VertId o, MR.VertId d);
            return __MR_PolylineTopology_findEdge(_UnderlyingPtr, o, d);
        }

        /// returns all vertices incident to path edges
        /// Generated from method `MR::PolylineTopology::getPathVertices`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetPathVertices(MR.Std.Const_Vector_MREdgeId path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_getPathVertices", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_PolylineTopology_getPathVertices(_Underlying *_this, MR.Std.Const_Vector_MREdgeId._Underlying *path);
            return MR.Misc.Move(new MR.VertBitSet(__MR_PolylineTopology_getPathVertices(_UnderlyingPtr, path._UnderlyingPtr), is_owning: true));
        }

        /// saves this in binary stream
        /// Generated from method `MR::PolylineTopology::write`.
        public unsafe void Write(MR.Std.Ostream s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_write", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_write(_Underlying *_this, MR.Std.Ostream._Underlying *s);
            __MR_PolylineTopology_write(_UnderlyingPtr, s._UnderlyingPtr);
        }

        /// comparison via edges (all other members are considered as not important caches)
        /// Generated from method `MR::PolylineTopology::operator==`.
        public static unsafe bool operator==(MR.Const_PolylineTopology _this, MR.Const_PolylineTopology b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_PolylineTopology", ExactSpelling = true)]
            extern static byte __MR_equal_MR_PolylineTopology(MR.Const_PolylineTopology._Underlying *_this, MR.Const_PolylineTopology._Underlying *b);
            return __MR_equal_MR_PolylineTopology(_this._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_PolylineTopology _this, MR.Const_PolylineTopology b)
        {
            return !(_this == b);
        }

        /// returns true if for each edge e: e == e.next() || e.odd() == next( e ).sym().odd()
        /// Generated from method `MR::PolylineTopology::isConsistentlyOriented`.
        public unsafe bool IsConsistentlyOriented()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_isConsistentlyOriented", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_isConsistentlyOriented(_Underlying *_this);
            return __MR_PolylineTopology_isConsistentlyOriented(_UnderlyingPtr) != 0;
        }

        /// verifies that all internal data structures are valid
        /// Generated from method `MR::PolylineTopology::checkValidity`.
        public unsafe bool CheckValidity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_checkValidity", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_checkValidity(_Underlying *_this);
            return __MR_PolylineTopology_checkValidity(_UnderlyingPtr) != 0;
        }

        /// returns true if the polyline does not have any holes
        /// Generated from method `MR::PolylineTopology::isClosed`.
        public unsafe bool IsClosed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_isClosed", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_isClosed(_Underlying *_this);
            return __MR_PolylineTopology_isClosed(_UnderlyingPtr) != 0;
        }

        // IEquatable:

        public bool Equals(MR.Const_PolylineTopology? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_PolylineTopology)
                return this == (MR.Const_PolylineTopology)other;
            return false;
        }
    }

    /// topology of one or several polylines (how line segments are connected in lines) common for 2D and 3D polylines
    /// Generated from class `MR::PolylineTopology`.
    /// This is the non-const half of the class.
    public class PolylineTopology : Const_PolylineTopology
    {
        internal unsafe PolylineTopology(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineTopology() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineTopology._Underlying *__MR_PolylineTopology_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineTopology_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineTopology::PolylineTopology`.
        public unsafe PolylineTopology(MR._ByValue_PolylineTopology _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTopology._Underlying *__MR_PolylineTopology_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PolylineTopology._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineTopology_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PolylineTopology::operator=`.
        public unsafe MR.PolylineTopology Assign(MR._ByValue_PolylineTopology _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTopology._Underlying *__MR_PolylineTopology_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PolylineTopology._Underlying *_other);
            return new(__MR_PolylineTopology_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// build topology of comp2firstVert.size()-1 not-closed polylines
        /// each pair (a,b) of indices in \param comp2firstVert defines vertex range of a polyline [a,b)
        /// Generated from method `MR::PolylineTopology::buildOpenLines`.
        public unsafe void BuildOpenLines(MR.Std.Const_Vector_MRVertId comp2firstVert)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_buildOpenLines", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_buildOpenLines(_Underlying *_this, MR.Std.Const_Vector_MRVertId._Underlying *comp2firstVert);
            __MR_PolylineTopology_buildOpenLines(_UnderlyingPtr, comp2firstVert._UnderlyingPtr);
        }

        /// creates an edge not associated with any vertex
        /// Generated from method `MR::PolylineTopology::makeEdge`.
        public unsafe MR.EdgeId MakeEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_makeEdge_0", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_makeEdge_0(_Underlying *_this);
            return __MR_PolylineTopology_makeEdge_0(_UnderlyingPtr);
        }

        /// makes an edge connecting vertices a and b
        /// \param e if valid then the function does not make new edge, but connects the vertices using given one (and returns it)
        /// \details if
        ///   1) any of the vertices is invalid or not within the vertSize(),
        ///   2) or a==b,
        ///   3) or either a or b already has 2 incident edges,
        ///   4) given edge e is not lone or not within the edgeSize()
        /// then makeEdge(a,b) does nothing and returns invalid edge
        /// Generated from method `MR::PolylineTopology::makeEdge`.
        /// Parameter `e` defaults to `{}`.
        public unsafe MR.EdgeId MakeEdge(MR.VertId a, MR.VertId b, MR._InOpt_EdgeId e = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_makeEdge_3", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_makeEdge_3(_Underlying *_this, MR.VertId a, MR.VertId b, MR.EdgeId *e);
            return __MR_PolylineTopology_makeEdge_3(_UnderlyingPtr, a, b, e.HasValue ? &e.Object : null);
        }

        /// for every given edge[ue] calls makeEdge( edge[ue][0], edge[ue][1], ue ), making new edges so that edges.size() <= undirectedEdgeSize() at the end
        /// \return the total number of edges created
        /// Generated from method `MR::PolylineTopology::makeEdges`.
        public unsafe int MakeEdges(MR.Const_Edges edges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_makeEdges", ExactSpelling = true)]
            extern static int __MR_PolylineTopology_makeEdges(_Underlying *_this, MR.Const_Edges._Underlying *edges);
            return __MR_PolylineTopology_makeEdges(_UnderlyingPtr, edges._UnderlyingPtr);
        }

        /// sets the capacity of half-edges vector
        /// Generated from method `MR::PolylineTopology::edgeReserve`.
        public unsafe void EdgeReserve(ulong newCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_edgeReserve", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_edgeReserve(_Underlying *_this, ulong newCapacity);
            __MR_PolylineTopology_edgeReserve(_UnderlyingPtr, newCapacity);
        }

        /// given edge becomes lone after the call, so it is un-spliced from connected edges, and if it was not connected at origin or destination, then that vertex is deleted as well
        /// Generated from method `MR::PolylineTopology::deleteEdge`.
        public unsafe void DeleteEdge(MR.UndirectedEdgeId ue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_deleteEdge", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_deleteEdge(_Underlying *_this, MR.UndirectedEdgeId ue);
            __MR_PolylineTopology_deleteEdge(_UnderlyingPtr, ue);
        }

        /// calls deleteEdge for every set bit
        /// Generated from method `MR::PolylineTopology::deleteEdges`.
        public unsafe void DeleteEdges(MR.Const_UndirectedEdgeBitSet es)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_deleteEdges", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_deleteEdges(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *es);
            __MR_PolylineTopology_deleteEdges(_UnderlyingPtr, es._UnderlyingPtr);
        }

        /// given two half edges do either of two: \n
        /// 1) if a and b were from distinct rings, puts them in one ring; \n
        /// 2) if a and b were from the same ring, puts them in separate rings;
        /// \details the cut in rings in both cases is made after a and b
        /// Generated from method `MR::PolylineTopology::splice`.
        public unsafe void Splice(MR.EdgeId a, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_splice", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_splice(_Underlying *_this, MR.EdgeId a, MR.EdgeId b);
            __MR_PolylineTopology_splice(_UnderlyingPtr, a, b);
        }

        /// sets new origin to the full origin ring including this edge
        /// \details edgePerVertex_ table is updated accordingly
        /// Generated from method `MR::PolylineTopology::setOrg`.
        public unsafe void SetOrg(MR.EdgeId a, MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_setOrg", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_setOrg(_Underlying *_this, MR.EdgeId a, MR.VertId v);
            __MR_PolylineTopology_setOrg(_UnderlyingPtr, a, v);
        }

        /// creates new vert-id not associated with any edge yet
        /// Generated from method `MR::PolylineTopology::addVertId`.
        public unsafe MR.VertId AddVertId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_addVertId", ExactSpelling = true)]
            extern static MR.VertId __MR_PolylineTopology_addVertId(_Underlying *_this);
            return __MR_PolylineTopology_addVertId(_UnderlyingPtr);
        }

        /// explicitly increases the size of vertices vector
        /// Generated from method `MR::PolylineTopology::vertResize`.
        public unsafe void VertResize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_vertResize", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_vertResize(_Underlying *_this, ulong newSize);
            __MR_PolylineTopology_vertResize(_UnderlyingPtr, newSize);
        }

        /// explicitly increases the size of vertices vector, doubling the current capacity if it was not enough
        /// Generated from method `MR::PolylineTopology::vertResizeWithReserve`.
        public unsafe void VertResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_vertResizeWithReserve", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_vertResizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_PolylineTopology_vertResizeWithReserve(_UnderlyingPtr, newSize);
        }

        /// sets the capacity of vertices vector
        /// Generated from method `MR::PolylineTopology::vertReserve`.
        public unsafe void VertReserve(ulong newCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_vertReserve", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_vertReserve(_Underlying *_this, ulong newCapacity);
            __MR_PolylineTopology_vertReserve(_UnderlyingPtr, newCapacity);
        }

        /// split given edge on two parts:
        /// dest(returned-edge) = org(e) - newly created vertex,
        /// org(returned-edge) = org(e-before-split),
        /// dest(e) = dest(e-before-split)
        /// Generated from method `MR::PolylineTopology::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_splitEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_splitEdge(_Underlying *_this, MR.EdgeId e);
            return __MR_PolylineTopology_splitEdge(_UnderlyingPtr, e);
        }

        /// adds polyline in this topology passing progressively via vertices *[vs, vs+num);
        /// if vs[0] == vs[num-1] then a closed polyline is created;
        /// return the edge from first to second vertex
        /// Generated from method `MR::PolylineTopology::makePolyline`.
        public unsafe MR.EdgeId MakePolyline(MR.Const_VertId? vs, ulong num)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_makePolyline", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineTopology_makePolyline(_Underlying *_this, MR.Const_VertId._Underlying *vs, ulong num);
            return __MR_PolylineTopology_makePolyline(_UnderlyingPtr, vs is not null ? vs._UnderlyingPtr : null, num);
        }

        /// appends polyline topology (from) in addition to the current topology: creates new edges, verts;
        /// \param outVmap,outEmap (optionally) returns mappings: from.id -> this.id
        /// Generated from method `MR::PolylineTopology::addPart`.
        public unsafe void AddPart(MR.Const_PolylineTopology from, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_addPart", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_addPart(_Underlying *_this, MR.Const_PolylineTopology._Underlying *from, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            __MR_PolylineTopology_addPart(_UnderlyingPtr, from._UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// appends polyline topology (from) in addition to the current topology: creates new edges, verts;
        /// Generated from method `MR::PolylineTopology::addPartByMask`.
        public unsafe void AddPartByMask(MR.Const_PolylineTopology from, MR.Const_UndirectedEdgeBitSet mask, MR.VertMap? outVmap = null, MR.EdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_addPartByMask", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_addPartByMask(_Underlying *_this, MR.Const_PolylineTopology._Underlying *from, MR.Const_UndirectedEdgeBitSet._Underlying *mask, MR.VertMap._Underlying *outVmap, MR.EdgeMap._Underlying *outEmap);
            __MR_PolylineTopology_addPartByMask(_UnderlyingPtr, from._UnderlyingPtr, mask._UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// tightly packs all arrays eliminating lone edges and invalid vertices
        /// \param outVmap,outEmap if given returns mappings: old.id -> new.id;
        /// Generated from method `MR::PolylineTopology::pack`.
        public unsafe void Pack(MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_pack", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_pack(_Underlying *_this, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap);
            __MR_PolylineTopology_pack(_UnderlyingPtr, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null);
        }

        /// loads this from binary stream
        /// Generated from method `MR::PolylineTopology::read`.
        public unsafe bool Read(MR.Std.Istream s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_read", ExactSpelling = true)]
            extern static byte __MR_PolylineTopology_read(_Underlying *_this, MR.Std.Istream._Underlying *s);
            return __MR_PolylineTopology_read(_UnderlyingPtr, s._UnderlyingPtr) != 0;
        }

        /// changes the orientation of all edges: every edge e is replaced with e.sym()
        /// Generated from method `MR::PolylineTopology::flip`.
        public unsafe void Flip()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_flip", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_flip(_Underlying *_this);
            __MR_PolylineTopology_flip(_UnderlyingPtr);
        }

        /// computes numValidVerts_ and validVerts_ from edgePerVertex_
        /// Generated from method `MR::PolylineTopology::computeValidsFromEdges`.
        public unsafe void ComputeValidsFromEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTopology_computeValidsFromEdges", ExactSpelling = true)]
            extern static void __MR_PolylineTopology_computeValidsFromEdges(_Underlying *_this);
            __MR_PolylineTopology_computeValidsFromEdges(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PolylineTopology` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PolylineTopology`/`Const_PolylineTopology` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PolylineTopology
    {
        internal readonly Const_PolylineTopology? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PolylineTopology() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PolylineTopology(Const_PolylineTopology new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PolylineTopology(Const_PolylineTopology arg) {return new(arg);}
        public _ByValue_PolylineTopology(MR.Misc._Moved<PolylineTopology> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PolylineTopology(MR.Misc._Moved<PolylineTopology> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PolylineTopology` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineTopology`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineTopology`/`Const_PolylineTopology` directly.
    public class _InOptMut_PolylineTopology
    {
        public PolylineTopology? Opt;

        public _InOptMut_PolylineTopology() {}
        public _InOptMut_PolylineTopology(PolylineTopology value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineTopology(PolylineTopology value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineTopology` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineTopology`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineTopology`/`Const_PolylineTopology` to pass it to the function.
    public class _InOptConst_PolylineTopology
    {
        public Const_PolylineTopology? Opt;

        public _InOptConst_PolylineTopology() {}
        public _InOptConst_PolylineTopology(Const_PolylineTopology value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineTopology(Const_PolylineTopology value) {return new(value);}
    }

    /// simplifies construction of connected polyline in the topology
    /// Generated from class `MR::PolylineMaker`.
    /// This is the const half of the class.
    public class Const_PolylineMaker : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineMaker(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineMaker_Destroy(_Underlying *_this);
            __MR_PolylineMaker_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineMaker() {Dispose(false);}

        public unsafe MR.PolylineTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_Get_topology", ExactSpelling = true)]
                extern static MR.PolylineTopology._Underlying *__MR_PolylineMaker_Get_topology(_Underlying *_this);
                return new(__MR_PolylineMaker_Get_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public unsafe Const_PolylineMaker(MR.Const_PolylineMaker _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineMaker._Underlying *__MR_PolylineMaker_ConstructFromAnother(MR.PolylineMaker._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineMaker_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public unsafe Const_PolylineMaker(MR.PolylineTopology t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_Construct", ExactSpelling = true)]
            extern static MR.PolylineMaker._Underlying *__MR_PolylineMaker_Construct(MR.PolylineTopology._Underlying *t);
            _UnderlyingPtr = __MR_PolylineMaker_Construct(t._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public static unsafe implicit operator Const_PolylineMaker(MR.PolylineTopology t) {return new(t);}
    }

    /// simplifies construction of connected polyline in the topology
    /// Generated from class `MR::PolylineMaker`.
    /// This is the non-const half of the class.
    public class PolylineMaker : Const_PolylineMaker
    {
        internal unsafe PolylineMaker(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public unsafe PolylineMaker(MR.Const_PolylineMaker _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineMaker._Underlying *__MR_PolylineMaker_ConstructFromAnother(MR.PolylineMaker._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineMaker_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public unsafe PolylineMaker(MR.PolylineTopology t) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_Construct", ExactSpelling = true)]
            extern static MR.PolylineMaker._Underlying *__MR_PolylineMaker_Construct(MR.PolylineTopology._Underlying *t);
            _UnderlyingPtr = __MR_PolylineMaker_Construct(t._UnderlyingPtr);
        }

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public static unsafe implicit operator PolylineMaker(MR.PolylineTopology t) {return new(t);}

        /// creates first edge of polyline
        /// \param v first vertex of the polyline
        /// Generated from method `MR::PolylineMaker::start`.
        public unsafe MR.EdgeId Start(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_start", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineMaker_start(_Underlying *_this, MR.VertId v);
            return __MR_PolylineMaker_start(_UnderlyingPtr, v);
        }

        /// makes next edge of polyline
        /// \param v next vertex of the polyline
        /// Generated from method `MR::PolylineMaker::proceed`.
        public unsafe MR.EdgeId Proceed(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_proceed", ExactSpelling = true)]
            extern static MR.EdgeId __MR_PolylineMaker_proceed(_Underlying *_this, MR.VertId v);
            return __MR_PolylineMaker_proceed(_UnderlyingPtr, v);
        }

        /// closes the polyline
        /// Generated from method `MR::PolylineMaker::close`.
        public unsafe void Close()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_close", ExactSpelling = true)]
            extern static void __MR_PolylineMaker_close(_Underlying *_this);
            __MR_PolylineMaker_close(_UnderlyingPtr);
        }

        /// finishes the polyline adding final vertex in it
        /// Generated from method `MR::PolylineMaker::finishOpen`.
        public unsafe void FinishOpen(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineMaker_finishOpen", ExactSpelling = true)]
            extern static void __MR_PolylineMaker_finishOpen(_Underlying *_this, MR.VertId v);
            __MR_PolylineMaker_finishOpen(_UnderlyingPtr, v);
        }
    }

    /// This is used for optional parameters of class `PolylineMaker` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineMaker`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineMaker`/`Const_PolylineMaker` directly.
    public class _InOptMut_PolylineMaker
    {
        public PolylineMaker? Opt;

        public _InOptMut_PolylineMaker() {}
        public _InOptMut_PolylineMaker(PolylineMaker value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineMaker(PolylineMaker value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineMaker` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineMaker`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineMaker`/`Const_PolylineMaker` to pass it to the function.
    public class _InOptConst_PolylineMaker
    {
        public Const_PolylineMaker? Opt;

        public _InOptConst_PolylineMaker() {}
        public _InOptConst_PolylineMaker(Const_PolylineMaker value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineMaker(Const_PolylineMaker value) {return new(value);}

        /// Generated from constructor `MR::PolylineMaker::PolylineMaker`.
        public static unsafe implicit operator _InOptConst_PolylineMaker(MR.PolylineTopology t) {return new MR.PolylineMaker(t);}
    }
}
