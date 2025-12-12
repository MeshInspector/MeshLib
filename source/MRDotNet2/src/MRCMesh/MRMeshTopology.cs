public static partial class MR
{
    /// Mesh Topology
    /// Generated from class `MR::MeshTopology`.
    /// This is the const half of the class.
    public class Const_MeshTopology : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_MeshTopology>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshTopology(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshTopology_Destroy(_Underlying *_this);
            __MR_MeshTopology_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshTopology() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshTopology() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshTopology_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTopology_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTopology::MeshTopology`.
        public unsafe Const_MeshTopology(MR._ByValue_MeshTopology _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshTopology_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshTopology._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTopology_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// checks whether the edge is disconnected from all other edges and disassociated from all vertices and faces (as if after makeEdge)
        /// Generated from method `MR::MeshTopology::isLoneEdge`.
        public unsafe bool IsLoneEdge(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isLoneEdge", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isLoneEdge(_Underlying *_this, MR.EdgeId a);
            return __MR_MeshTopology_isLoneEdge(_UnderlyingPtr, a) != 0;
        }

        /// returns last not lone undirected edge id, or invalid id if no such edge exists
        /// Generated from method `MR::MeshTopology::lastNotLoneUndirectedEdge`.
        public unsafe MR.UndirectedEdgeId LastNotLoneUndirectedEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_lastNotLoneUndirectedEdge", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_MeshTopology_lastNotLoneUndirectedEdge(_Underlying *_this);
            return __MR_MeshTopology_lastNotLoneUndirectedEdge(_UnderlyingPtr);
        }

        /// returns last not lone edge id, or invalid id if no such edge exists
        /// Generated from method `MR::MeshTopology::lastNotLoneEdge`.
        public unsafe MR.EdgeId LastNotLoneEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_lastNotLoneEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_lastNotLoneEdge(_Underlying *_this);
            return __MR_MeshTopology_lastNotLoneEdge(_UnderlyingPtr);
        }

        /// remove all lone edges from given set
        /// Generated from method `MR::MeshTopology::excludeLoneEdges`.
        public unsafe void ExcludeLoneEdges(MR.UndirectedEdgeBitSet edges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_excludeLoneEdges", ExactSpelling = true)]
            extern static void __MR_MeshTopology_excludeLoneEdges(_Underlying *_this, MR.UndirectedEdgeBitSet._Underlying *edges);
            __MR_MeshTopology_excludeLoneEdges(_UnderlyingPtr, edges._UnderlyingPtr);
        }

        /// returns the number of half-edge records including lone ones
        /// Generated from method `MR::MeshTopology::edgeSize`.
        public unsafe ulong EdgeSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgeSize", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_edgeSize(_Underlying *_this);
            return __MR_MeshTopology_edgeSize(_UnderlyingPtr);
        }

        /// returns the number of allocated edge records
        /// Generated from method `MR::MeshTopology::edgeCapacity`.
        public unsafe ulong EdgeCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgeCapacity", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_edgeCapacity(_Underlying *_this);
            return __MR_MeshTopology_edgeCapacity(_UnderlyingPtr);
        }

        /// returns the number of undirected edges (pairs of half-edges) including lone ones
        /// Generated from method `MR::MeshTopology::undirectedEdgeSize`.
        public unsafe ulong UndirectedEdgeSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_undirectedEdgeSize", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_undirectedEdgeSize(_Underlying *_this);
            return __MR_MeshTopology_undirectedEdgeSize(_UnderlyingPtr);
        }

        /// returns the number of allocated undirected edges (pairs of half-edges)
        /// Generated from method `MR::MeshTopology::undirectedEdgeCapacity`.
        public unsafe ulong UndirectedEdgeCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_undirectedEdgeCapacity", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_undirectedEdgeCapacity(_Underlying *_this);
            return __MR_MeshTopology_undirectedEdgeCapacity(_UnderlyingPtr);
        }

        /// computes the number of not-lone (valid) undirected edges
        /// Generated from method `MR::MeshTopology::computeNotLoneUndirectedEdges`.
        public unsafe ulong ComputeNotLoneUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_computeNotLoneUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_computeNotLoneUndirectedEdges(_Underlying *_this);
            return __MR_MeshTopology_computeNotLoneUndirectedEdges(_UnderlyingPtr);
        }

        /// finds and returns all not-lone (valid) undirected edges
        /// Generated from method `MR::MeshTopology::findNotLoneUndirectedEdges`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindNotLoneUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findNotLoneUndirectedEdges", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_MeshTopology_findNotLoneUndirectedEdges(_Underlying *_this);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_MeshTopology_findNotLoneUndirectedEdges(_UnderlyingPtr), is_owning: true));
        }

        /// returns true if given edge is within valid range and not-lone
        /// Generated from method `MR::MeshTopology::hasEdge`.
        public unsafe bool HasEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_hasEdge", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_hasEdge(_Underlying *_this, MR.EdgeId e);
            return __MR_MeshTopology_hasEdge(_UnderlyingPtr, e) != 0;
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::MeshTopology::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_heapBytes(_Underlying *_this);
            return __MR_MeshTopology_heapBytes(_UnderlyingPtr);
        }

        /// next (counter clock wise) half-edge in the origin ring
        /// Generated from method `MR::MeshTopology::next`.
        public unsafe MR.EdgeId Next(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_next(_Underlying *_this, MR.EdgeId he);
            return __MR_MeshTopology_next(_UnderlyingPtr, he);
        }

        /// previous (clock wise) half-edge in the origin ring
        /// Generated from method `MR::MeshTopology::prev`.
        public unsafe MR.EdgeId Prev(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_prev", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_prev(_Underlying *_this, MR.EdgeId he);
            return __MR_MeshTopology_prev(_UnderlyingPtr, he);
        }

        /// returns origin vertex of half-edge
        /// Generated from method `MR::MeshTopology::org`.
        public unsafe MR.VertId Org(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_org", ExactSpelling = true)]
            extern static MR.VertId __MR_MeshTopology_org(_Underlying *_this, MR.EdgeId he);
            return __MR_MeshTopology_org(_UnderlyingPtr, he);
        }

        /// returns destination vertex of half-edge
        /// Generated from method `MR::MeshTopology::dest`.
        public unsafe MR.VertId Dest(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_dest", ExactSpelling = true)]
            extern static MR.VertId __MR_MeshTopology_dest(_Underlying *_this, MR.EdgeId he);
            return __MR_MeshTopology_dest(_UnderlyingPtr, he);
        }

        /// returns left face of half-edge
        /// Generated from method `MR::MeshTopology::left`.
        public unsafe MR.FaceId Left(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_left", ExactSpelling = true)]
            extern static MR.FaceId __MR_MeshTopology_left(_Underlying *_this, MR.EdgeId he);
            return __MR_MeshTopology_left(_UnderlyingPtr, he);
        }

        /// returns right face of half-edge
        /// Generated from method `MR::MeshTopology::right`.
        public unsafe MR.FaceId Right(MR.EdgeId he)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_right", ExactSpelling = true)]
            extern static MR.FaceId __MR_MeshTopology_right(_Underlying *_this, MR.EdgeId he);
            return __MR_MeshTopology_right(_UnderlyingPtr, he);
        }

        /// returns true if a and b are both from the same origin ring
        /// Generated from method `MR::MeshTopology::fromSameOriginRing`.
        public unsafe bool FromSameOriginRing(MR.EdgeId a, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_fromSameOriginRing", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_fromSameOriginRing(_Underlying *_this, MR.EdgeId a, MR.EdgeId b);
            return __MR_MeshTopology_fromSameOriginRing(_UnderlyingPtr, a, b) != 0;
        }

        /// returns true if a and b are both from the same left face ring
        /// Generated from method `MR::MeshTopology::fromSameLeftRing`.
        public unsafe bool FromSameLeftRing(MR.EdgeId a, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_fromSameLeftRing", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_fromSameLeftRing(_Underlying *_this, MR.EdgeId a, MR.EdgeId b);
            return __MR_MeshTopology_fromSameLeftRing(_UnderlyingPtr, a, b) != 0;
        }

        /// returns the number of edges around the origin vertex, returns 1 for lone edges
        /// Generated from method `MR::MeshTopology::getOrgDegree`.
        public unsafe int GetOrgDegree(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getOrgDegree", ExactSpelling = true)]
            extern static int __MR_MeshTopology_getOrgDegree(_Underlying *_this, MR.EdgeId a);
            return __MR_MeshTopology_getOrgDegree(_UnderlyingPtr, a);
        }

        /// returns the number of edges around the given vertex
        /// Generated from method `MR::MeshTopology::getVertDegree`.
        public unsafe int GetVertDegree(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getVertDegree", ExactSpelling = true)]
            extern static int __MR_MeshTopology_getVertDegree(_Underlying *_this, MR.VertId v);
            return __MR_MeshTopology_getVertDegree(_UnderlyingPtr, v);
        }

        /// returns the number of edges around the left face: 3 for triangular faces, ...
        /// Generated from method `MR::MeshTopology::getLeftDegree`.
        public unsafe int GetLeftDegree(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftDegree", ExactSpelling = true)]
            extern static int __MR_MeshTopology_getLeftDegree(_Underlying *_this, MR.EdgeId a);
            return __MR_MeshTopology_getLeftDegree(_UnderlyingPtr, a);
        }

        /// returns the number of edges around the given face: 3 for triangular faces, ...
        /// Generated from method `MR::MeshTopology::getFaceDegree`.
        public unsafe int GetFaceDegree(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getFaceDegree", ExactSpelling = true)]
            extern static int __MR_MeshTopology_getFaceDegree(_Underlying *_this, MR.FaceId f);
            return __MR_MeshTopology_getFaceDegree(_UnderlyingPtr, f);
        }

        /// returns true if the cell to the left of a is triangular
        /// Generated from method `MR::MeshTopology::isLeftTri`.
        public unsafe bool IsLeftTri(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isLeftTri", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isLeftTri(_Underlying *_this, MR.EdgeId a);
            return __MR_MeshTopology_isLeftTri(_UnderlyingPtr, a) != 0;
        }

        /// gets 3 vertices of given triangular face;
        /// the vertices are returned in counter-clockwise order if look from mesh outside
        /// Generated from method `MR::MeshTopology::getTriVerts`.
        public unsafe void GetTriVerts(MR.FaceId f, MR.Mut_VertId v0, MR.Mut_VertId v1, MR.Mut_VertId v2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getTriVerts_4", ExactSpelling = true)]
            extern static void __MR_MeshTopology_getTriVerts_4(_Underlying *_this, MR.FaceId f, MR.Mut_VertId._Underlying *v0, MR.Mut_VertId._Underlying *v1, MR.Mut_VertId._Underlying *v2);
            __MR_MeshTopology_getTriVerts_4(_UnderlyingPtr, f, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshTopology::getTriVerts`.
        public unsafe void GetTriVerts(MR.FaceId f, MR.Std.Mut_Array_MRVertId_3 v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getTriVerts_2", ExactSpelling = true)]
            extern static void __MR_MeshTopology_getTriVerts_2(_Underlying *_this, MR.FaceId f, MR.Std.Mut_Array_MRVertId_3._Underlying *v);
            __MR_MeshTopology_getTriVerts_2(_UnderlyingPtr, f, v._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshTopology::getTriVerts`.
        public unsafe MR.Std.Array_MRVertId_3 GetTriVerts(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getTriVerts_1", ExactSpelling = true)]
            extern static MR.Std.Array_MRVertId_3 __MR_MeshTopology_getTriVerts_1(_Underlying *_this, MR.FaceId f);
            return __MR_MeshTopology_getTriVerts_1(_UnderlyingPtr, f);
        }

        /// return true if triangular face (f) has (v) as one of its vertices
        /// Generated from method `MR::MeshTopology::isTriVert`.
        public unsafe bool IsTriVert(MR.FaceId f, MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isTriVert", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isTriVert(_Underlying *_this, MR.FaceId f, MR.VertId v);
            return __MR_MeshTopology_isTriVert(_UnderlyingPtr, f, v) != 0;
        }

        /// returns three vertex ids for valid triangles, invalid triangles are skipped
        /// Generated from method `MR::MeshTopology::getAllTriVerts`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdArrayMRVertId3> GetAllTriVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getAllTriVerts", ExactSpelling = true)]
            extern static MR.Std.Vector_StdArrayMRVertId3._Underlying *__MR_MeshTopology_getAllTriVerts(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdArrayMRVertId3(__MR_MeshTopology_getAllTriVerts(_UnderlyingPtr), is_owning: true));
        }

        /// returns three vertex ids for valid triangles (which can be accessed by FaceId),
        /// vertex ids for invalid triangles are undefined, and shall not be read
        /// Generated from method `MR::MeshTopology::getTriangulation`.
        public unsafe MR.Misc._Moved<MR.Triangulation> GetTriangulation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getTriangulation", ExactSpelling = true)]
            extern static MR.Triangulation._Underlying *__MR_MeshTopology_getTriangulation(_Underlying *_this);
            return MR.Misc.Move(new MR.Triangulation(__MR_MeshTopology_getTriangulation(_UnderlyingPtr), is_owning: true));
        }

        /// gets 3 vertices of the left face ( face-id may not exist, but the shape must be triangular)
        /// the vertices are returned in counter-clockwise order if look from mesh outside: v0 = org( a ), v1 = dest( a )
        /// Generated from method `MR::MeshTopology::getLeftTriVerts`.
        public unsafe void GetLeftTriVerts(MR.EdgeId a, MR.Mut_VertId v0, MR.Mut_VertId v1, MR.Mut_VertId v2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftTriVerts_4", ExactSpelling = true)]
            extern static void __MR_MeshTopology_getLeftTriVerts_4(_Underlying *_this, MR.EdgeId a, MR.Mut_VertId._Underlying *v0, MR.Mut_VertId._Underlying *v1, MR.Mut_VertId._Underlying *v2);
            __MR_MeshTopology_getLeftTriVerts_4(_UnderlyingPtr, a, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshTopology::getLeftTriVerts`.
        public unsafe void GetLeftTriVerts(MR.EdgeId a, MR.Std.Mut_Array_MRVertId_3 v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftTriVerts_2", ExactSpelling = true)]
            extern static void __MR_MeshTopology_getLeftTriVerts_2(_Underlying *_this, MR.EdgeId a, MR.Std.Mut_Array_MRVertId_3._Underlying *v);
            __MR_MeshTopology_getLeftTriVerts_2(_UnderlyingPtr, a, v._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshTopology::getLeftTriVerts`.
        public unsafe MR.Std.Array_MRVertId_3 GetLeftTriVerts(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftTriVerts_1", ExactSpelling = true)]
            extern static MR.Std.Array_MRVertId_3 __MR_MeshTopology_getLeftTriVerts_1(_Underlying *_this, MR.EdgeId a);
            return __MR_MeshTopology_getLeftTriVerts_1(_UnderlyingPtr, a);
        }

        /// given one edge with triangular face on the left;
        /// returns two other edges of the same face, oriented to have this face on the left;
        /// the edges are returned in counter-clockwise order if look from mesh outside
        /// Generated from method `MR::MeshTopology::getLeftTriEdges`.
        public unsafe void GetLeftTriEdges(MR.EdgeId e0, MR.Mut_EdgeId e1, MR.Mut_EdgeId e2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftTriEdges", ExactSpelling = true)]
            extern static void __MR_MeshTopology_getLeftTriEdges(_Underlying *_this, MR.EdgeId e0, MR.Mut_EdgeId._Underlying *e1, MR.Mut_EdgeId._Underlying *e2);
            __MR_MeshTopology_getLeftTriEdges(_UnderlyingPtr, e0, e1._UnderlyingPtr, e2._UnderlyingPtr);
        }

        /// gets 3 edges of given triangular face, oriented to have it on the left;
        /// the edges are returned in counter-clockwise order if look from mesh outside
        /// Generated from method `MR::MeshTopology::getTriEdges`.
        public unsafe void GetTriEdges(MR.FaceId f, MR.Mut_EdgeId e0, MR.Mut_EdgeId e1, MR.Mut_EdgeId e2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getTriEdges", ExactSpelling = true)]
            extern static void __MR_MeshTopology_getTriEdges(_Underlying *_this, MR.FaceId f, MR.Mut_EdgeId._Underlying *e0, MR.Mut_EdgeId._Underlying *e1, MR.Mut_EdgeId._Underlying *e2);
            __MR_MeshTopology_getTriEdges(_UnderlyingPtr, f, e0._UnderlyingPtr, e1._UnderlyingPtr, e2._UnderlyingPtr);
        }

        /// returns true if the cell to the left of a is quadrangular
        /// Generated from method `MR::MeshTopology::isLeftQuad`.
        public unsafe bool IsLeftQuad(MR.EdgeId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isLeftQuad", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isLeftQuad(_Underlying *_this, MR.EdgeId a);
            return __MR_MeshTopology_isLeftQuad(_UnderlyingPtr, a) != 0;
        }

        /// for all valid vertices this vector contains an edge with the origin there
        /// Generated from method `MR::MeshTopology::edgePerVertex`.
        public unsafe MR.Const_Vector_MREdgeId_MRVertId EdgePerVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgePerVertex", ExactSpelling = true)]
            extern static MR.Const_Vector_MREdgeId_MRVertId._Underlying *__MR_MeshTopology_edgePerVertex(_Underlying *_this);
            return new(__MR_MeshTopology_edgePerVertex(_UnderlyingPtr), is_owning: false);
        }

        /// returns valid edge if given vertex is present in the mesh
        /// Generated from method `MR::MeshTopology::edgeWithOrg`.
        public unsafe MR.EdgeId EdgeWithOrg(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgeWithOrg", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_edgeWithOrg(_Underlying *_this, MR.VertId a);
            return __MR_MeshTopology_edgeWithOrg(_UnderlyingPtr, a);
        }

        /// returns true if given vertex is present in the mesh
        /// Generated from method `MR::MeshTopology::hasVert`.
        public unsafe bool HasVert(MR.VertId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_hasVert", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_hasVert(_Underlying *_this, MR.VertId a);
            return __MR_MeshTopology_hasVert(_UnderlyingPtr, a) != 0;
        }

        /// returns the number of valid vertices
        /// Generated from method `MR::MeshTopology::numValidVerts`.
        public unsafe int NumValidVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_numValidVerts", ExactSpelling = true)]
            extern static int __MR_MeshTopology_numValidVerts(_Underlying *_this);
            return __MR_MeshTopology_numValidVerts(_UnderlyingPtr);
        }

        /// returns last valid vertex id, or invalid id if no single valid vertex exists
        /// Generated from method `MR::MeshTopology::lastValidVert`.
        public unsafe MR.VertId LastValidVert()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_lastValidVert", ExactSpelling = true)]
            extern static MR.VertId __MR_MeshTopology_lastValidVert(_Underlying *_this);
            return __MR_MeshTopology_lastValidVert(_UnderlyingPtr);
        }

        /// returns the number of vertex records including invalid ones
        /// Generated from method `MR::MeshTopology::vertSize`.
        public unsafe ulong VertSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_vertSize", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_vertSize(_Underlying *_this);
            return __MR_MeshTopology_vertSize(_UnderlyingPtr);
        }

        /// returns the number of allocated vert records
        /// Generated from method `MR::MeshTopology::vertCapacity`.
        public unsafe ulong VertCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_vertCapacity", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_vertCapacity(_Underlying *_this);
            return __MR_MeshTopology_vertCapacity(_UnderlyingPtr);
        }

        /// returns cached set of all valid vertices
        /// Generated from method `MR::MeshTopology::getValidVerts`.
        public unsafe MR.Const_VertBitSet GetValidVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getValidVerts", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_MeshTopology_getValidVerts(_Underlying *_this);
            return new(__MR_MeshTopology_getValidVerts(_UnderlyingPtr), is_owning: false);
        }

        /// sets in (vs) all valid vertices that were not selected before the call, and resets other bits
        /// Generated from method `MR::MeshTopology::flip`.
        public unsafe void Flip(MR.VertBitSet vs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_flip_MR_VertBitSet", ExactSpelling = true)]
            extern static void __MR_MeshTopology_flip_MR_VertBitSet(_Underlying *_this, MR.VertBitSet._Underlying *vs);
            __MR_MeshTopology_flip_MR_VertBitSet(_UnderlyingPtr, vs._UnderlyingPtr);
        }

        /// if region pointer is not null then converts it in reference, otherwise returns all valid vertices in the mesh
        /// Generated from method `MR::MeshTopology::getVertIds`.
        public unsafe MR.Const_VertBitSet GetVertIds(MR.Const_VertBitSet? region)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getVertIds", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_MeshTopology_getVertIds(_Underlying *_this, MR.Const_VertBitSet._Underlying *region);
            return new(__MR_MeshTopology_getVertIds(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: false);
        }

        /// for all valid faces this vector contains an edge with that face at left
        /// Generated from method `MR::MeshTopology::edgePerFace`.
        public unsafe MR.Const_Vector_MREdgeId_MRFaceId EdgePerFace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgePerFace", ExactSpelling = true)]
            extern static MR.Const_Vector_MREdgeId_MRFaceId._Underlying *__MR_MeshTopology_edgePerFace(_Underlying *_this);
            return new(__MR_MeshTopology_edgePerFace(_UnderlyingPtr), is_owning: false);
        }

        /// returns valid edge if given vertex is present in the mesh
        /// Generated from method `MR::MeshTopology::edgeWithLeft`.
        public unsafe MR.EdgeId EdgeWithLeft(MR.FaceId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgeWithLeft", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_edgeWithLeft(_Underlying *_this, MR.FaceId a);
            return __MR_MeshTopology_edgeWithLeft(_UnderlyingPtr, a);
        }

        /// returns true if given face is present in the mesh
        /// Generated from method `MR::MeshTopology::hasFace`.
        public unsafe bool HasFace(MR.FaceId a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_hasFace", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_hasFace(_Underlying *_this, MR.FaceId a);
            return __MR_MeshTopology_hasFace(_UnderlyingPtr, a) != 0;
        }

        /// if two valid faces share the same edge then it is found and returned
        /// Generated from method `MR::MeshTopology::sharedEdge`.
        public unsafe MR.EdgeId SharedEdge(MR.FaceId l, MR.FaceId r)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_sharedEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_sharedEdge(_Underlying *_this, MR.FaceId l, MR.FaceId r);
            return __MR_MeshTopology_sharedEdge(_UnderlyingPtr, l, r);
        }

        /// if two valid edges share the same vertex then it is found and returned as Edge with this vertex in origin
        /// Generated from method `MR::MeshTopology::sharedVertInOrg`.
        public unsafe MR.EdgeId SharedVertInOrg(MR.EdgeId a, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_sharedVertInOrg_MR_EdgeId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_sharedVertInOrg_MR_EdgeId(_Underlying *_this, MR.EdgeId a, MR.EdgeId b);
            return __MR_MeshTopology_sharedVertInOrg_MR_EdgeId(_UnderlyingPtr, a, b);
        }

        /// if two valid faces share the same vertex then it is found and returned as Edge with this vertex in origin
        /// Generated from method `MR::MeshTopology::sharedVertInOrg`.
        public unsafe MR.EdgeId SharedVertInOrg(MR.FaceId l, MR.FaceId r)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_sharedVertInOrg_MR_FaceId", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_sharedVertInOrg_MR_FaceId(_Underlying *_this, MR.FaceId l, MR.FaceId r);
            return __MR_MeshTopology_sharedVertInOrg_MR_FaceId(_UnderlyingPtr, l, r);
        }

        /// if two valid edges belong to same valid face then it is found and returned
        /// Generated from method `MR::MeshTopology::sharedFace`.
        public unsafe MR.FaceId SharedFace(MR.EdgeId a, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_sharedFace", ExactSpelling = true)]
            extern static MR.FaceId __MR_MeshTopology_sharedFace(_Underlying *_this, MR.EdgeId a, MR.EdgeId b);
            return __MR_MeshTopology_sharedFace(_UnderlyingPtr, a, b);
        }

        /// returns the number of valid faces
        /// Generated from method `MR::MeshTopology::numValidFaces`.
        public unsafe int NumValidFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_numValidFaces", ExactSpelling = true)]
            extern static int __MR_MeshTopology_numValidFaces(_Underlying *_this);
            return __MR_MeshTopology_numValidFaces(_UnderlyingPtr);
        }

        /// returns last valid face id, or invalid id if no single valid face exists
        /// Generated from method `MR::MeshTopology::lastValidFace`.
        public unsafe MR.FaceId LastValidFace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_lastValidFace", ExactSpelling = true)]
            extern static MR.FaceId __MR_MeshTopology_lastValidFace(_Underlying *_this);
            return __MR_MeshTopology_lastValidFace(_UnderlyingPtr);
        }

        /// returns the number of face records including invalid ones
        /// Generated from method `MR::MeshTopology::faceSize`.
        public unsafe ulong FaceSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_faceSize", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_faceSize(_Underlying *_this);
            return __MR_MeshTopology_faceSize(_UnderlyingPtr);
        }

        /// returns the number of allocated face records
        /// Generated from method `MR::MeshTopology::faceCapacity`.
        public unsafe ulong FaceCapacity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_faceCapacity", ExactSpelling = true)]
            extern static ulong __MR_MeshTopology_faceCapacity(_Underlying *_this);
            return __MR_MeshTopology_faceCapacity(_UnderlyingPtr);
        }

        /// returns cached set of all valid faces
        /// Generated from method `MR::MeshTopology::getValidFaces`.
        public unsafe MR.Const_FaceBitSet GetValidFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getValidFaces", ExactSpelling = true)]
            extern static MR.Const_FaceBitSet._Underlying *__MR_MeshTopology_getValidFaces(_Underlying *_this);
            return new(__MR_MeshTopology_getValidFaces(_UnderlyingPtr), is_owning: false);
        }

        /// sets in (fs) all valid faces that were not selected before the call, and resets other bits
        /// Generated from method `MR::MeshTopology::flip`.
        public unsafe void Flip(MR.FaceBitSet fs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_flip_MR_FaceBitSet", ExactSpelling = true)]
            extern static void __MR_MeshTopology_flip_MR_FaceBitSet(_Underlying *_this, MR.FaceBitSet._Underlying *fs);
            __MR_MeshTopology_flip_MR_FaceBitSet(_UnderlyingPtr, fs._UnderlyingPtr);
        }

        /// if region pointer is not null then converts it in reference, otherwise returns all valid faces in the mesh
        /// Generated from method `MR::MeshTopology::getFaceIds`.
        public unsafe MR.Const_FaceBitSet GetFaceIds(MR.Const_FaceBitSet? region)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getFaceIds", ExactSpelling = true)]
            extern static MR.Const_FaceBitSet._Underlying *__MR_MeshTopology_getFaceIds(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region);
            return new(__MR_MeshTopology_getFaceIds(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: false);
        }

        /// returns the first boundary edge (for given region or for whole mesh if region is nullptr) in counter-clockwise order starting from given edge with the same left face or hole;
        /// returns invalid edge if no boundary edge is found
        /// Generated from method `MR::MeshTopology::bdEdgeSameLeft`.
        public unsafe MR.EdgeId BdEdgeSameLeft(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_bdEdgeSameLeft", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_bdEdgeSameLeft(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_bdEdgeSameLeft(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
        }

        /// returns true if left(e) is a valid (region) face,
        /// and it has a boundary edge (isBdEdge(e,region) == true)
        /// Generated from method `MR::MeshTopology::isLeftBdFace`.
        public unsafe bool IsLeftBdFace(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isLeftBdFace", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isLeftBdFace(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isLeftBdFace(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns a boundary edge with given left face considering boundary of given region (or for whole mesh if region is nullptr);
        /// returns invalid edge if no boundary edge is found
        /// Generated from method `MR::MeshTopology::bdEdgeWithLeft`.
        public unsafe MR.EdgeId BdEdgeWithLeft(MR.FaceId f, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_bdEdgeWithLeft", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_bdEdgeWithLeft(_Underlying *_this, MR.FaceId f, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_bdEdgeWithLeft(_UnderlyingPtr, f, region is not null ? region._UnderlyingPtr : null);
        }

        /// returns true if given face belongs to the region and it has a boundary edge (isBdEdge(e,region) == true)
        /// Generated from method `MR::MeshTopology::isBdFace`.
        public unsafe bool IsBdFace(MR.FaceId f, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isBdFace", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isBdFace(_Underlying *_this, MR.FaceId f, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isBdFace(_UnderlyingPtr, f, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns all faces for which isBdFace(f, region) is true
        /// Generated from method `MR::MeshTopology::findBdFaces`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> FindBdFaces(MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findBdFaces", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshTopology_findBdFaces(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshTopology_findBdFaces(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// return true if left face of given edge belongs to region (or just have valid id if region is nullptr)
        /// Generated from method `MR::MeshTopology::isLeftInRegion`.
        public unsafe bool IsLeftInRegion(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isLeftInRegion", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isLeftInRegion(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isLeftInRegion(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// return true if given edge is inner for given region (or for whole mesh if region is nullptr)
        /// Generated from method `MR::MeshTopology::isInnerEdge`.
        public unsafe bool IsInnerEdge(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isInnerEdge", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isInnerEdge(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isInnerEdge(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// isBdEdge(e) returns true, if the edge (e) is a boundary edge of the mesh:
        ///     (e) has a hole from one or both sides.
        /// isBdEdge(e, region) returns true, if the edge (e) is a boundary edge of the given region:
        ///     (e) has a region's face from one side (region.test(f0)==true) and a hole or not-region face from the other side (!f1 || region.test(f1)==false).
        /// If the region contains all faces of the mesh then isBdEdge(e) is the union of isBdEdge(e, region) and not-lone edges without both left and right faces.
        /// Generated from method `MR::MeshTopology::isBdEdge`.
        public unsafe bool IsBdEdge(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isBdEdge", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isBdEdge(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isBdEdge(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns all (test) edges for which left(e) does not belong to the region and isBdEdge(e, region) is true
        /// Generated from method `MR::MeshTopology::findLeftBdEdges`.
        public unsafe MR.Misc._Moved<MR.EdgeBitSet> FindLeftBdEdges(MR.Const_FaceBitSet? region = null, MR.Const_EdgeBitSet? test = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findLeftBdEdges", ExactSpelling = true)]
            extern static MR.EdgeBitSet._Underlying *__MR_MeshTopology_findLeftBdEdges(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region, MR.Const_EdgeBitSet._Underlying *test);
            return MR.Misc.Move(new MR.EdgeBitSet(__MR_MeshTopology_findLeftBdEdges(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, test is not null ? test._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the first boundary edge (for given region or for whole mesh if region is nullptr) in counter-clockwise order starting from given edge with the same origin;
        /// returns invalid edge if no boundary edge is found
        /// Generated from method `MR::MeshTopology::bdEdgeSameOrigin`.
        public unsafe MR.EdgeId BdEdgeSameOrigin(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_bdEdgeSameOrigin", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_bdEdgeSameOrigin(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_bdEdgeSameOrigin(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
        }

        /// returns true if edge's origin is on (region) boundary
        /// Generated from method `MR::MeshTopology::isBdVertexInOrg`.
        public unsafe bool IsBdVertexInOrg(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isBdVertexInOrg", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isBdVertexInOrg(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isBdVertexInOrg(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns a boundary edge with given vertex in origin considering boundary of given region (or for whole mesh if region is nullptr);
        /// returns invalid edge if no boundary edge is found
        /// Generated from method `MR::MeshTopology::bdEdgeWithOrigin`.
        public unsafe MR.EdgeId BdEdgeWithOrigin(MR.VertId v, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_bdEdgeWithOrigin", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_bdEdgeWithOrigin(_Underlying *_this, MR.VertId v, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_bdEdgeWithOrigin(_UnderlyingPtr, v, region is not null ? region._UnderlyingPtr : null);
        }

        /// returns true if given vertex is on (region) boundary
        /// Generated from method `MR::MeshTopology::isBdVertex`.
        public unsafe bool IsBdVertex(MR.VertId v, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isBdVertex", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isBdVertex(_Underlying *_this, MR.VertId v, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isBdVertex(_UnderlyingPtr, v, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns all (test) vertices for which isBdVertex(v, region) is true
        /// Generated from method `MR::MeshTopology::findBdVerts`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> FindBdVerts(MR.Const_FaceBitSet? region = null, MR.Const_VertBitSet? test = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findBdVerts", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_MeshTopology_findBdVerts(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region, MR.Const_VertBitSet._Underlying *test);
            return MR.Misc.Move(new MR.VertBitSet(__MR_MeshTopology_findBdVerts(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, test is not null ? test._UnderlyingPtr : null), is_owning: true));
        }

        /// returns true if one of incident faces of given vertex pertain to given region (or any such face exists if region is nullptr)
        /// Generated from method `MR::MeshTopology::isInnerOrBdVertex`.
        public unsafe bool IsInnerOrBdVertex(MR.VertId v, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isInnerOrBdVertex", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isInnerOrBdVertex(_Underlying *_this, MR.VertId v, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isInnerOrBdVertex(_UnderlyingPtr, v, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns true if left face of given edge belongs to given region (if provided) and right face either does not exist or does not belong
        /// Generated from method `MR::MeshTopology::isLeftBdEdge`.
        public unsafe bool IsLeftBdEdge(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isLeftBdEdge", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isLeftBdEdge(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isLeftBdEdge(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// return true if given edge is inner or boundary for given region (or for whole mesh if region is nullptr), returns false for lone edges
        /// Generated from method `MR::MeshTopology::isInnerOrBdEdge`.
        public unsafe bool IsInnerOrBdEdge(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isInnerOrBdEdge", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isInnerOrBdEdge(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isInnerOrBdEdge(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// given a (region) boundary edge with no right face in given region, returns next boundary edge for the same region: dest(e)==org(res)
        /// Generated from method `MR::MeshTopology::nextLeftBd`.
        public unsafe MR.EdgeId NextLeftBd(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_nextLeftBd", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_nextLeftBd(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_nextLeftBd(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
        }

        /// given a (region) boundary edge with no right face in given region, returns previous boundary edge for the same region; dest(res)==org(e)
        /// Generated from method `MR::MeshTopology::prevLeftBd`.
        public unsafe MR.EdgeId PrevLeftBd(MR.EdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_prevLeftBd", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_prevLeftBd(_Underlying *_this, MR.EdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_prevLeftBd(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
        }

        /// finds and returns edge from o to d in the mesh; returns invalid edge otherwise
        /// Generated from method `MR::MeshTopology::findEdge`.
        public unsafe MR.EdgeId FindEdge(MR.VertId o, MR.VertId d)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_findEdge(_Underlying *_this, MR.VertId o, MR.VertId d);
            return __MR_MeshTopology_findEdge(_UnderlyingPtr, o, d);
        }

        /// returns true if the mesh (region) does not have any neighboring holes
        /// Generated from method `MR::MeshTopology::isClosed`.
        public unsafe bool IsClosed(MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_isClosed", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_isClosed(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTopology_isClosed(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns one edge with no valid left face for every boundary in the mesh;
        /// if region is given, then returned edges must have valid right faces from the region
        /// Generated from method `MR::MeshTopology::findHoleRepresentiveEdges`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> FindHoleRepresentiveEdges(MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findHoleRepresentiveEdges", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeId._Underlying *__MR_MeshTopology_findHoleRepresentiveEdges(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_MeshTopology_findHoleRepresentiveEdges(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the number of hole loops in the mesh;
        /// \param holeRepresentativeEdges optional output of the smallest edge id with no valid left face in every hole
        /// Generated from method `MR::MeshTopology::findNumHoles`.
        public unsafe int FindNumHoles(MR.EdgeBitSet? holeRepresentativeEdges = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_findNumHoles", ExactSpelling = true)]
            extern static int __MR_MeshTopology_findNumHoles(_Underlying *_this, MR.EdgeBitSet._Underlying *holeRepresentativeEdges);
            return __MR_MeshTopology_findNumHoles(_UnderlyingPtr, holeRepresentativeEdges is not null ? holeRepresentativeEdges._UnderlyingPtr : null);
        }

        /// returns full edge-loop of left face from (e) starting from (e) itself
        /// Generated from method `MR::MeshTopology::getLeftRing`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> GetLeftRing(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftRing", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeId._Underlying *__MR_MeshTopology_getLeftRing(_Underlying *_this, MR.EdgeId e);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_MeshTopology_getLeftRing(_UnderlyingPtr, e), is_owning: true));
        }

        /// returns full edge-loops of left faces from every edge in (es);
        /// each edge-loop will be returned only once even if some faces are represented by more than one edge in (es)
        /// Generated from method `MR::MeshTopology::getLeftRings`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> GetLeftRings(MR.Std.Const_Vector_MREdgeId es)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getLeftRings", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_MeshTopology_getLeftRings(_Underlying *_this, MR.Std.Const_Vector_MREdgeId._Underlying *es);
            return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_MeshTopology_getLeftRings(_UnderlyingPtr, es._UnderlyingPtr), is_owning: true));
        }

        /// returns all vertices incident to path edges
        /// Generated from method `MR::MeshTopology::getPathVertices`.
        public unsafe MR.Misc._Moved<MR.VertBitSet> GetPathVertices(MR.Std.Const_Vector_MREdgeId path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getPathVertices", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_MeshTopology_getPathVertices(_Underlying *_this, MR.Std.Const_Vector_MREdgeId._Underlying *path);
            return MR.Misc.Move(new MR.VertBitSet(__MR_MeshTopology_getPathVertices(_UnderlyingPtr, path._UnderlyingPtr), is_owning: true));
        }

        /// returns all valid left faces of path edges
        /// Generated from method `MR::MeshTopology::getPathLeftFaces`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetPathLeftFaces(MR.Std.Const_Vector_MREdgeId path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getPathLeftFaces", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshTopology_getPathLeftFaces(_Underlying *_this, MR.Std.Const_Vector_MREdgeId._Underlying *path);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshTopology_getPathLeftFaces(_UnderlyingPtr, path._UnderlyingPtr), is_owning: true));
        }

        /// returns all valid right faces of path edges
        /// Generated from method `MR::MeshTopology::getPathRightFaces`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetPathRightFaces(MR.Std.Const_Vector_MREdgeId path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_getPathRightFaces", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshTopology_getPathRightFaces(_Underlying *_this, MR.Std.Const_Vector_MREdgeId._Underlying *path);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshTopology_getPathRightFaces(_UnderlyingPtr, path._UnderlyingPtr), is_owning: true));
        }

        /// saves in binary stream
        /// Generated from method `MR::MeshTopology::write`.
        public unsafe void Write(MR.Std.Ostream s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_write", ExactSpelling = true)]
            extern static void __MR_MeshTopology_write(_Underlying *_this, MR.Std.Ostream._Underlying *s);
            __MR_MeshTopology_write(_UnderlyingPtr, s._UnderlyingPtr);
        }

        /// compare that two topologies are exactly the same
        /// Generated from method `MR::MeshTopology::operator==`.
        public static unsafe bool operator==(MR.Const_MeshTopology _this, MR.Const_MeshTopology b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_MeshTopology", ExactSpelling = true)]
            extern static byte __MR_equal_MR_MeshTopology(MR.Const_MeshTopology._Underlying *_this, MR.Const_MeshTopology._Underlying *b);
            return __MR_equal_MR_MeshTopology(_this._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_MeshTopology _this, MR.Const_MeshTopology b)
        {
            return !(_this == b);
        }

        /// returns whether the methods validVerts(), validFaces(), numValidVerts(), numValidFaces() can be called
        /// Generated from method `MR::MeshTopology::updatingValids`.
        public unsafe bool UpdatingValids()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_updatingValids", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_updatingValids(_Underlying *_this);
            return __MR_MeshTopology_updatingValids(_UnderlyingPtr) != 0;
        }

        /// verifies that all internal data structures are valid;
        /// if allVerts=true then checks in addition that all not-lone edges have valid vertices on both ends
        /// Generated from method `MR::MeshTopology::checkValidity`.
        /// Parameter `cb` defaults to `{}`.
        /// Parameter `allVerts` defaults to `true`.
        public unsafe bool CheckValidity(MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, bool? allVerts = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_checkValidity", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_checkValidity(_Underlying *_this, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, byte *allVerts);
            byte __deref_allVerts = allVerts.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_MeshTopology_checkValidity(_UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, allVerts.HasValue ? &__deref_allVerts : null) != 0;
        }

        // IEquatable:

        public bool Equals(MR.Const_MeshTopology? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_MeshTopology)
                return this == (MR.Const_MeshTopology)other;
            return false;
        }
    }

    /// Mesh Topology
    /// Generated from class `MR::MeshTopology`.
    /// This is the non-const half of the class.
    public class MeshTopology : Const_MeshTopology
    {
        internal unsafe MeshTopology(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshTopology() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshTopology_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTopology_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTopology::MeshTopology`.
        public unsafe MeshTopology(MR._ByValue_MeshTopology _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshTopology_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshTopology._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTopology_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshTopology::operator=`.
        public unsafe MR.MeshTopology Assign(MR._ByValue_MeshTopology _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshTopology._Underlying *__MR_MeshTopology_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshTopology._Underlying *_other);
            return new(__MR_MeshTopology_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// creates an edge not associated with any vertex or face
        /// Generated from method `MR::MeshTopology::makeEdge`.
        public unsafe MR.EdgeId MakeEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_makeEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_makeEdge(_Underlying *_this);
            return __MR_MeshTopology_makeEdge(_UnderlyingPtr);
        }

        /// sets the capacity of half-edges vector
        /// Generated from method `MR::MeshTopology::edgeReserve`.
        public unsafe void EdgeReserve(ulong newCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_edgeReserve", ExactSpelling = true)]
            extern static void __MR_MeshTopology_edgeReserve(_Underlying *_this, ulong newCapacity);
            __MR_MeshTopology_edgeReserve(_UnderlyingPtr, newCapacity);
        }

        /// requests the removal of unused capacity
        /// Generated from method `MR::MeshTopology::shrinkToFit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_shrinkToFit", ExactSpelling = true)]
            extern static void __MR_MeshTopology_shrinkToFit(_Underlying *_this);
            __MR_MeshTopology_shrinkToFit(_UnderlyingPtr);
        }

        /// given two half edges do either of two:
        /// 1) if a and b were from distinct rings, puts them in one ring;
        /// 2) if a and b were from the same ring, puts them in separate rings;
        /// the cut in rings in both cases is made after a and b
        /// Generated from method `MR::MeshTopology::splice`.
        public unsafe void Splice(MR.EdgeId a, MR.EdgeId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_splice", ExactSpelling = true)]
            extern static void __MR_MeshTopology_splice(_Underlying *_this, MR.EdgeId a, MR.EdgeId b);
            __MR_MeshTopology_splice(_UnderlyingPtr, a, b);
        }

        /// collapses given edge in a vertex and deletes
        /// 1) faces: left( e ) and right( e );
        /// 2) edges: e, next( e.sym() ), prev( e.sym() ), and optionally next( e ), prev( e ) if their left and right triangles are deleted;
        /// 3) all vertices that lost their last edge;
        /// calls onEdgeDel for every deleted edge (del) including given (e);
        /// if valid (rem) is given then dest( del ) = dest( rem ) and their origins are in different ends of collapsing edge, (rem) shall take the place of (del)
        /// \return prev( e ) if it is still valid
        /// Generated from method `MR::MeshTopology::collapseEdge`.
        public unsafe MR.EdgeId CollapseEdge(MR.EdgeId e, MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeDel)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_collapseEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_collapseEdge(_Underlying *_this, MR.EdgeId e, MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeDel);
            return __MR_MeshTopology_collapseEdge(_UnderlyingPtr, e, onEdgeDel._UnderlyingPtr);
        }

        /// sets new origin to the full origin ring including this edge;
        /// edgePerVertex_ table is updated accordingly
        /// Generated from method `MR::MeshTopology::setOrg`.
        public unsafe void SetOrg(MR.EdgeId a, MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_setOrg", ExactSpelling = true)]
            extern static void __MR_MeshTopology_setOrg(_Underlying *_this, MR.EdgeId a, MR.VertId v);
            __MR_MeshTopology_setOrg(_UnderlyingPtr, a, v);
        }

        /// sets new left face to the full left ring including this edge;
        /// edgePerFace_ table is updated accordingly
        /// Generated from method `MR::MeshTopology::setLeft`.
        public unsafe void SetLeft(MR.EdgeId a, MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_setLeft", ExactSpelling = true)]
            extern static void __MR_MeshTopology_setLeft(_Underlying *_this, MR.EdgeId a, MR.FaceId f);
            __MR_MeshTopology_setLeft(_UnderlyingPtr, a, f);
        }

        /// creates new vert-id not associated with any edge yet
        /// Generated from method `MR::MeshTopology::addVertId`.
        public unsafe MR.VertId AddVertId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addVertId", ExactSpelling = true)]
            extern static MR.VertId __MR_MeshTopology_addVertId(_Underlying *_this);
            return __MR_MeshTopology_addVertId(_UnderlyingPtr);
        }

        /// explicitly increases the size of vertices vector
        /// Generated from method `MR::MeshTopology::vertResize`.
        public unsafe void VertResize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_vertResize", ExactSpelling = true)]
            extern static void __MR_MeshTopology_vertResize(_Underlying *_this, ulong newSize);
            __MR_MeshTopology_vertResize(_UnderlyingPtr, newSize);
        }

        /// explicitly increases the size of vertices vector, doubling the current capacity if it was not enough
        /// Generated from method `MR::MeshTopology::vertResizeWithReserve`.
        public unsafe void VertResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_vertResizeWithReserve", ExactSpelling = true)]
            extern static void __MR_MeshTopology_vertResizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_MeshTopology_vertResizeWithReserve(_UnderlyingPtr, newSize);
        }

        /// sets the capacity of vertices vector
        /// Generated from method `MR::MeshTopology::vertReserve`.
        public unsafe void VertReserve(ulong newCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_vertReserve", ExactSpelling = true)]
            extern static void __MR_MeshTopology_vertReserve(_Underlying *_this, ulong newCapacity);
            __MR_MeshTopology_vertReserve(_UnderlyingPtr, newCapacity);
        }

        /// creates new face-id not associated with any edge yet
        /// Generated from method `MR::MeshTopology::addFaceId`.
        public unsafe MR.FaceId AddFaceId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addFaceId", ExactSpelling = true)]
            extern static MR.FaceId __MR_MeshTopology_addFaceId(_Underlying *_this);
            return __MR_MeshTopology_addFaceId(_UnderlyingPtr);
        }

        /// deletes the face, also deletes its edges and vertices if they were not shared by other faces and not in \param keepFaces
        /// Generated from method `MR::MeshTopology::deleteFace`.
        public unsafe void DeleteFace(MR.FaceId f, MR.Const_UndirectedEdgeBitSet? keepEdges = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_deleteFace", ExactSpelling = true)]
            extern static void __MR_MeshTopology_deleteFace(_Underlying *_this, MR.FaceId f, MR.Const_UndirectedEdgeBitSet._Underlying *keepEdges);
            __MR_MeshTopology_deleteFace(_UnderlyingPtr, f, keepEdges is not null ? keepEdges._UnderlyingPtr : null);
        }

        /// deletes multiple given faces by calling \ref deleteFace for each
        /// Generated from method `MR::MeshTopology::deleteFaces`.
        public unsafe void DeleteFaces(MR.Const_FaceBitSet fs, MR.Const_UndirectedEdgeBitSet? keepEdges = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_deleteFaces", ExactSpelling = true)]
            extern static void __MR_MeshTopology_deleteFaces(_Underlying *_this, MR.Const_FaceBitSet._Underlying *fs, MR.Const_UndirectedEdgeBitSet._Underlying *keepEdges);
            __MR_MeshTopology_deleteFaces(_UnderlyingPtr, fs._UnderlyingPtr, keepEdges is not null ? keepEdges._UnderlyingPtr : null);
        }

        /// explicitly increases the size of faces vector
        /// Generated from method `MR::MeshTopology::faceResize`.
        public unsafe void FaceResize(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_faceResize", ExactSpelling = true)]
            extern static void __MR_MeshTopology_faceResize(_Underlying *_this, ulong newSize);
            __MR_MeshTopology_faceResize(_UnderlyingPtr, newSize);
        }

        /// explicitly increases the size of faces vector, doubling the current capacity if it was not enough
        /// Generated from method `MR::MeshTopology::faceResizeWithReserve`.
        public unsafe void FaceResizeWithReserve(ulong newSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_faceResizeWithReserve", ExactSpelling = true)]
            extern static void __MR_MeshTopology_faceResizeWithReserve(_Underlying *_this, ulong newSize);
            __MR_MeshTopology_faceResizeWithReserve(_UnderlyingPtr, newSize);
        }

        /// sets the capacity of faces vector
        /// Generated from method `MR::MeshTopology::faceReserve`.
        public unsafe void FaceReserve(ulong newCapacity)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_faceReserve", ExactSpelling = true)]
            extern static void __MR_MeshTopology_faceReserve(_Underlying *_this, ulong newCapacity);
            __MR_MeshTopology_faceReserve(_UnderlyingPtr, newCapacity);
        }

        /// given the edge with left and right triangular faces, which form together a quadrangle,
        /// rotates the edge counter-clockwise inside the quadrangle
        /// Generated from method `MR::MeshTopology::flipEdge`.
        public unsafe void FlipEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_flipEdge", ExactSpelling = true)]
            extern static void __MR_MeshTopology_flipEdge(_Underlying *_this, MR.EdgeId e);
            __MR_MeshTopology_flipEdge(_UnderlyingPtr, e);
        }

        /// split given edge on two parts:
        /// dest(returned-edge) = org(e) - newly created vertex,
        /// org(returned-edge) = org(e-before-split),
        /// dest(e) = dest(e-before-split)
        /// \details left and right faces of given edge if valid are also subdivided on two parts each;
        /// the split edge will keep both face IDs and their degrees, and the new edge will have new face IDs and new faces are triangular;
        /// if left or right faces of the original edge were in the region, then include new parts of these faces in the region
        /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
        /// Generated from method `MR::MeshTopology::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e, MR.FaceBitSet? region = null, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_splitEdge", ExactSpelling = true)]
            extern static MR.EdgeId __MR_MeshTopology_splitEdge(_Underlying *_this, MR.EdgeId e, MR.FaceBitSet._Underlying *region, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old);
            return __MR_MeshTopology_splitEdge(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null);
        }

        /// split given triangle on three triangles, introducing new vertex (which is returned) inside original triangle and connecting it to its vertices
        /// \details if region is given, then it must include (f) and new faces will be added there as well
        /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
        /// Generated from method `MR::MeshTopology::splitFace`.
        public unsafe MR.VertId SplitFace(MR.FaceId f, MR.FaceBitSet? region = null, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_splitFace", ExactSpelling = true)]
            extern static MR.VertId __MR_MeshTopology_splitFace(_Underlying *_this, MR.FaceId f, MR.FaceBitSet._Underlying *region, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old);
            return __MR_MeshTopology_splitFace(_UnderlyingPtr, f, region is not null ? region._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null);
        }

        /// flip orientation (normals) of
        /// * all mesh elements if \param fullComponents is nullptr, or
        /// * given mesh components in \param fullComponents.
        /// The behavior is undefined if fullComponents is given but there are connected components with some edges included and not-included there.
        /// Generated from method `MR::MeshTopology::flipOrientation`.
        public unsafe void FlipOrientation(MR.Const_UndirectedEdgeBitSet? fullComponents = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_flipOrientation", ExactSpelling = true)]
            extern static void __MR_MeshTopology_flipOrientation(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *fullComponents);
            __MR_MeshTopology_flipOrientation(_UnderlyingPtr, fullComponents is not null ? fullComponents._UnderlyingPtr : null);
        }

        /// appends mesh topology (from) in addition to the current topology: creates new edges, faces, verts;
        /// \param rearrangeTriangles if true then the order of triangles is selected according to the order of their vertices,
        /// please call rotateTriangles() first
        /// Generated from method `MR::MeshTopology::addPart`.
        /// Parameter `map` defaults to `{}`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void AddPart(MR.Const_MeshTopology from, MR.Const_PartMapping? map = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addPart_3", ExactSpelling = true)]
            extern static void __MR_MeshTopology_addPart_3(_Underlying *_this, MR.Const_MeshTopology._Underlying *from, MR.Const_PartMapping._Underlying *map, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_MeshTopology_addPart_3(_UnderlyingPtr, from._UnderlyingPtr, map is not null ? map._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// Generated from method `MR::MeshTopology::addPart`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void AddPart(MR.Const_MeshTopology from, MR.FaceMap? outFmap = null, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addPart_5", ExactSpelling = true)]
            extern static void __MR_MeshTopology_addPart_5(_Underlying *_this, MR.Const_MeshTopology._Underlying *from, MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_MeshTopology_addPart_5(_UnderlyingPtr, from._UnderlyingPtr, outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// the same but copies only portion of (from) specified by fromFaces,
        /// Generated from method `MR::MeshTopology::addPartByMask`.
        /// Parameter `map` defaults to `{}`.
        public unsafe void AddPartByMask(MR.Const_MeshTopology from, MR.Const_FaceBitSet? fromFaces, MR.Const_PartMapping? map = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addPartByMask_3", ExactSpelling = true)]
            extern static void __MR_MeshTopology_addPartByMask_3(_Underlying *_this, MR.Const_MeshTopology._Underlying *from, MR.Const_FaceBitSet._Underlying *fromFaces, MR.Const_PartMapping._Underlying *map);
            __MR_MeshTopology_addPartByMask_3(_UnderlyingPtr, from._UnderlyingPtr, fromFaces is not null ? fromFaces._UnderlyingPtr : null, map is not null ? map._UnderlyingPtr : null);
        }

        /// this version has more parameters
        /// \param flipOrientation if true then every from triangle is inverted before adding
        /// \param thisContours contours on this mesh (no left face) that have to be stitched with
        /// \param fromContours contours on from mesh during addition (no left face if flipOrientation otherwise no right face)
        /// Generated from method `MR::MeshTopology::addPartByMask`.
        /// Parameter `flipOrientation` defaults to `false`.
        /// Parameter `thisContours` defaults to `{}`.
        /// Parameter `fromContours` defaults to `{}`.
        /// Parameter `map` defaults to `{}`.
        public unsafe void AddPartByMask(MR.Const_MeshTopology from, MR.Const_FaceBitSet? fromFaces, bool? flipOrientation = null, MR.Std.Const_Vector_StdVectorMREdgeId? thisContours = null, MR.Std.Const_Vector_StdVectorMREdgeId? fromContours = null, MR.Const_PartMapping? map = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addPartByMask_6", ExactSpelling = true)]
            extern static void __MR_MeshTopology_addPartByMask_6(_Underlying *_this, MR.Const_MeshTopology._Underlying *from, MR.Const_FaceBitSet._Underlying *fromFaces, byte *flipOrientation, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *thisContours, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *fromContours, MR.Const_PartMapping._Underlying *map);
            byte __deref_flipOrientation = flipOrientation.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_MeshTopology_addPartByMask_6(_UnderlyingPtr, from._UnderlyingPtr, fromFaces is not null ? fromFaces._UnderlyingPtr : null, flipOrientation.HasValue ? &__deref_flipOrientation : null, thisContours is not null ? thisContours._UnderlyingPtr : null, fromContours is not null ? fromContours._UnderlyingPtr : null, map is not null ? map._UnderlyingPtr : null);
        }

        /// for each triangle selects edgeWithLeft with minimal origin vertex
        /// Generated from method `MR::MeshTopology::rotateTriangles`.
        public unsafe void RotateTriangles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_rotateTriangles", ExactSpelling = true)]
            extern static void __MR_MeshTopology_rotateTriangles(_Underlying *_this);
            __MR_MeshTopology_rotateTriangles(_UnderlyingPtr);
        }

        /// tightly packs all arrays eliminating lone edges and invalid faces and vertices
        /// \param outFmap,outVmap,outEmap if given returns mappings: old.id -> new.id;
        /// \param rearrangeTriangles if true then calls rotateTriangles()
        /// and selects the order of triangles according to the order of their vertices
        /// Generated from method `MR::MeshTopology::pack`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void Pack(MR.FaceMap? outFmap = null, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_pack_4", ExactSpelling = true)]
            extern static void __MR_MeshTopology_pack_4(_Underlying *_this, MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_MeshTopology_pack_4(_UnderlyingPtr, outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// tightly packs all arrays eliminating lone edges and invalid faces and vertices;
        /// reorder all faces, vertices and edges according to given maps, each containing old id -> new id mapping
        /// Generated from method `MR::MeshTopology::pack`.
        public unsafe void Pack(MR.Const_PackMapping map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_pack_1", ExactSpelling = true)]
            extern static void __MR_MeshTopology_pack_1(_Underlying *_this, MR.Const_PackMapping._Underlying *map);
            __MR_MeshTopology_pack_1(_UnderlyingPtr, map._UnderlyingPtr);
        }

        /// tightly packs all arrays eliminating lone edges and invalid faces and vertices;
        /// reorder all faces, vertices and edges according to given maps, each containing old id -> new id mapping;
        /// unlike \ref pack method, this method allocates minimal amount of memory for its operation but works much slower
        /// Generated from method `MR::MeshTopology::packMinMem`.
        public unsafe void PackMinMem(MR.Const_PackMapping map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_packMinMem", ExactSpelling = true)]
            extern static void __MR_MeshTopology_packMinMem(_Underlying *_this, MR.Const_PackMapping._Underlying *map);
            __MR_MeshTopology_packMinMem(_UnderlyingPtr, map._UnderlyingPtr);
        }

        /// loads from binary stream
        /// \return text of error if any
        /// Generated from method `MR::MeshTopology::read`.
        /// Parameter `callback` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> Read(MR.Std.Istream s, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_read", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_MeshTopology_read(_Underlying *_this, MR.Std.Istream._Underlying *s, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_MeshTopology_read(_UnderlyingPtr, s._UnderlyingPtr, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// These function are for parallel mesh creation from different threads. If you are not sure, do not use them.
        /// \details resizes all internal vectors and sets the numbers of valid elements in preparation for addPackedPart;
        /// edges are resized without initialization (so the user must initialize them using addPackedPart)
        /// Generated from method `MR::MeshTopology::resizeBeforeParallelAdd`.
        public unsafe void ResizeBeforeParallelAdd(ulong edgeSize, ulong vertSize, ulong faceSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_resizeBeforeParallelAdd", ExactSpelling = true)]
            extern static void __MR_MeshTopology_resizeBeforeParallelAdd(_Underlying *_this, ulong edgeSize, ulong vertSize, ulong faceSize);
            __MR_MeshTopology_resizeBeforeParallelAdd(_UnderlyingPtr, edgeSize, vertSize, faceSize);
        }

        /// copies topology (from) into this;
        /// \param from edges must be tightly packes without any lone edges, and they are mapped [0, from.edges.size()) -> [toEdgeId, toEdgeId + from.edges.size());
        /// \param fmap,vmap mapping of vertices and faces if it is given ( from.id -> this.id )
        /// Generated from method `MR::MeshTopology::addPackedPart`.
        public unsafe void AddPackedPart(MR.Const_MeshTopology from, MR.EdgeId toEdgeId, MR.Const_FaceMap fmap, MR.Const_VertMap vmap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_addPackedPart", ExactSpelling = true)]
            extern static void __MR_MeshTopology_addPackedPart(_Underlying *_this, MR.Const_MeshTopology._Underlying *from, MR.EdgeId toEdgeId, MR.Const_FaceMap._Underlying *fmap, MR.Const_VertMap._Underlying *vmap);
            __MR_MeshTopology_addPackedPart(_UnderlyingPtr, from._UnderlyingPtr, toEdgeId, fmap._UnderlyingPtr, vmap._UnderlyingPtr);
        }

        /// compute
        /// 1) numValidVerts_ and validVerts_ from edgePerVertex_
        /// 2) numValidFaces_ and validFaces_ from edgePerFace_
        /// and activates their auto-update
        /// Generated from method `MR::MeshTopology::computeValidsFromEdges`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe bool ComputeValidsFromEdges(MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_computeValidsFromEdges", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_computeValidsFromEdges(_Underlying *_this, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return __MR_MeshTopology_computeValidsFromEdges(_UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
        }

        /// stops updating validVerts(), validFaces(), numValidVerts(), numValidFaces() for parallel processing of mesh parts
        /// Generated from method `MR::MeshTopology::stopUpdatingValids`.
        public unsafe void StopUpdatingValids()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_stopUpdatingValids", ExactSpelling = true)]
            extern static void __MR_MeshTopology_stopUpdatingValids(_Underlying *_this);
            __MR_MeshTopology_stopUpdatingValids(_UnderlyingPtr);
        }

        /// for incident vertices and faces of given edges, remember one of them as edgeWithOrg and edgeWithLeft;
        /// this is important in parallel algorithms where other edges may change but stable ones will survive
        /// Generated from method `MR::MeshTopology::preferEdges`.
        public unsafe void PreferEdges(MR.Const_UndirectedEdgeBitSet stableEdges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_preferEdges", ExactSpelling = true)]
            extern static void __MR_MeshTopology_preferEdges(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *stableEdges);
            __MR_MeshTopology_preferEdges(_UnderlyingPtr, stableEdges._UnderlyingPtr);
        }

        // constructs triangular grid mesh topology in parallel
        /// Generated from method `MR::MeshTopology::buildGridMesh`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe bool BuildGridMesh(MR.Const_GridSettings settings, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopology_buildGridMesh", ExactSpelling = true)]
            extern static byte __MR_MeshTopology_buildGridMesh(_Underlying *_this, MR.Const_GridSettings._Underlying *settings, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return __MR_MeshTopology_buildGridMesh(_UnderlyingPtr, settings._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshTopology` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshTopology`/`Const_MeshTopology` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshTopology
    {
        internal readonly Const_MeshTopology? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshTopology() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshTopology(Const_MeshTopology new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshTopology(Const_MeshTopology arg) {return new(arg);}
        public _ByValue_MeshTopology(MR.Misc._Moved<MeshTopology> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshTopology(MR.Misc._Moved<MeshTopology> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshTopology` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshTopology`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTopology`/`Const_MeshTopology` directly.
    public class _InOptMut_MeshTopology
    {
        public MeshTopology? Opt;

        public _InOptMut_MeshTopology() {}
        public _InOptMut_MeshTopology(MeshTopology value) {Opt = value;}
        public static implicit operator _InOptMut_MeshTopology(MeshTopology value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshTopology` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshTopology`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTopology`/`Const_MeshTopology` to pass it to the function.
    public class _InOptConst_MeshTopology
    {
        public Const_MeshTopology? Opt;

        public _InOptConst_MeshTopology() {}
        public _InOptConst_MeshTopology(Const_MeshTopology value) {Opt = value;}
        public static implicit operator _InOptConst_MeshTopology(Const_MeshTopology value) {return new(value);}
    }
}
