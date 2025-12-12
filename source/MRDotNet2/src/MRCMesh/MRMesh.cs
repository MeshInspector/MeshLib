public static partial class MR
{
    /// This class represents a mesh, including topology (connectivity) information and point coordinates,
    /// as well as some caches to accelerate search algorithms
    /// Generated from class `MR::Mesh`.
    /// This is the const half of the class.
    public class Const_Mesh : MR.Misc.SharedObject, System.IDisposable, System.IEquatable<MR.Const_Mesh>
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_Mesh_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_Mesh_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_Mesh_UseCount();
                return __MR_std_shared_ptr_MR_Mesh_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Mesh_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_Mesh_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_Mesh(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Mesh_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Mesh_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Mesh_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Mesh_ConstructNonOwning(ptr);
        }

        internal unsafe Const_Mesh(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe Mesh _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Mesh_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_Mesh_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Mesh_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Mesh_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Mesh_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_Mesh_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_Mesh_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Mesh() {Dispose(false);}

        public unsafe MR.Const_MeshTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_Get_topology", ExactSpelling = true)]
                extern static MR.Const_MeshTopology._Underlying *__MR_Mesh_Get_topology(_Underlying *_this);
                return new(__MR_Mesh_Get_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_Get_points", ExactSpelling = true)]
                extern static MR.Const_VertCoords._Underlying *__MR_Mesh_Get_points(_Underlying *_this);
                return new(__MR_Mesh_Get_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Mesh() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_DefaultConstruct();
            _LateMakeShared(__MR_Mesh_DefaultConstruct());
        }

        /// Generated from constructor `MR::Mesh::Mesh`.
        public unsafe Const_Mesh(MR._ByValue_Mesh _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Mesh._Underlying *_other);
            _LateMakeShared(__MR_Mesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// construct mesh from vertex coordinates and a set of triangles with given ids
        /// Generated from method `MR::Mesh::fromTriangles`.
        /// Parameter `settings` defaults to `{}`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Mesh> FromTriangles(MR._ByValue_VertCoords vertexCoordinates, MR.Const_Triangulation t, MR.MeshBuilder.Const_BuildSettings? settings = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_fromTriangles", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_fromTriangles(MR.Misc._PassBy vertexCoordinates_pass_by, MR.VertCoords._Underlying *vertexCoordinates, MR.Const_Triangulation._Underlying *t, MR.MeshBuilder.Const_BuildSettings._Underlying *settings, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Mesh(__MR_Mesh_fromTriangles(vertexCoordinates.PassByMode, vertexCoordinates.Value is not null ? vertexCoordinates.Value._UnderlyingPtr : null, t._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// construct mesh from TriMesh representation
        /// Generated from method `MR::Mesh::fromTriMesh`.
        /// Parameter `settings` defaults to `{}`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Mesh> FromTriMesh(MR.Misc._Moved<MR.TriMesh> triMesh, MR.MeshBuilder.Const_BuildSettings? settings = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_fromTriMesh", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_fromTriMesh(MR.TriMesh._Underlying *triMesh, MR.MeshBuilder.Const_BuildSettings._Underlying *settings, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Mesh(__MR_Mesh_fromTriMesh(triMesh.Value._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// construct mesh from vertex coordinates and a set of triangles with given ids;
        /// unlike simple fromTriangles() it tries to resolve non-manifold vertices by creating duplicate vertices
        /// Generated from method `MR::Mesh::fromTrianglesDuplicatingNonManifoldVertices`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Mesh> FromTrianglesDuplicatingNonManifoldVertices(MR._ByValue_VertCoords vertexCoordinates, MR.Triangulation t, MR.Std.Vector_MRMeshBuilderVertDuplication? dups = null, MR.MeshBuilder.Const_BuildSettings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_fromTrianglesDuplicatingNonManifoldVertices", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_fromTrianglesDuplicatingNonManifoldVertices(MR.Misc._PassBy vertexCoordinates_pass_by, MR.VertCoords._Underlying *vertexCoordinates, MR.Triangulation._Underlying *t, MR.Std.Vector_MRMeshBuilderVertDuplication._Underlying *dups, MR.MeshBuilder.Const_BuildSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Mesh(__MR_Mesh_fromTrianglesDuplicatingNonManifoldVertices(vertexCoordinates.PassByMode, vertexCoordinates.Value is not null ? vertexCoordinates.Value._UnderlyingPtr : null, t._UnderlyingPtr, dups is not null ? dups._UnderlyingPtr : null, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// construct mesh from vertex coordinates and construct mesh topology from face soup,
        /// where each face can have arbitrary degree (not only triangles);
        /// all non-triangular faces will be automatically subdivided on triangles
        /// Generated from method `MR::Mesh::fromFaceSoup`.
        /// Parameter `settings` defaults to `{}`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Mesh> FromFaceSoup(MR._ByValue_VertCoords vertexCoordinates, MR.Std.Const_Vector_MRVertId verts, MR.Const_Vector_MRMeshBuilderVertSpan_MRFaceId faces, MR.MeshBuilder.Const_BuildSettings? settings = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_fromFaceSoup", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_fromFaceSoup(MR.Misc._PassBy vertexCoordinates_pass_by, MR.VertCoords._Underlying *vertexCoordinates, MR.Std.Const_Vector_MRVertId._Underlying *verts, MR.Const_Vector_MRMeshBuilderVertSpan_MRFaceId._Underlying *faces, MR.MeshBuilder.Const_BuildSettings._Underlying *settings, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Mesh(__MR_Mesh_fromFaceSoup(vertexCoordinates.PassByMode, vertexCoordinates.Value is not null ? vertexCoordinates.Value._UnderlyingPtr : null, verts._UnderlyingPtr, faces._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// construct mesh from point triples;
        /// \param duplicateNonManifoldVertices = false, all coinciding points are given the same VertId in the result;
        /// \param duplicateNonManifoldVertices = true, it tries to avoid non-manifold vertices by creating duplicate vertices with same coordinates
        /// Generated from method `MR::Mesh::fromPointTriples`.
        public static unsafe MR.Misc._Moved<MR.Mesh> FromPointTriples(MR.Std.Const_Vector_StdArrayMRVector3f3 posTriples, bool duplicateNonManifoldVertices)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_fromPointTriples", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_fromPointTriples(MR.Std.Const_Vector_StdArrayMRVector3f3._Underlying *posTriples, byte duplicateNonManifoldVertices);
            return MR.Misc.Move(new MR.Mesh(__MR_Mesh_fromPointTriples(posTriples._UnderlyingPtr, duplicateNonManifoldVertices ? (byte)1 : (byte)0), is_owning: true));
        }

        /// compare that two meshes are exactly the same
        /// Generated from method `MR::Mesh::operator==`.
        public static unsafe bool operator==(MR.Const_Mesh _this, MR.Const_Mesh b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Mesh", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Mesh(MR.Const_Mesh._Underlying *_this, MR.Const_Mesh._Underlying *b);
            return __MR_equal_MR_Mesh(_this._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Mesh _this, MR.Const_Mesh b)
        {
            return !(_this == b);
        }

        /// returns coordinates of the edge origin
        /// Generated from method `MR::Mesh::orgPnt`.
        public unsafe MR.Vector3f OrgPnt(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_orgPnt", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_orgPnt(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_orgPnt(_UnderlyingPtr, e);
        }

        /// returns coordinates of the edge destination
        /// Generated from method `MR::Mesh::destPnt`.
        public unsafe MR.Vector3f DestPnt(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_destPnt", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_destPnt(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_destPnt(_UnderlyingPtr, e);
        }

        /// returns vector equal to edge destination point minus edge origin point
        /// Generated from method `MR::Mesh::edgeVector`.
        public unsafe MR.Vector3f EdgeVector(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgeVector", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_edgeVector(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_edgeVector(_UnderlyingPtr, e);
        }

        /// returns line segment of given edge
        /// Generated from method `MR::Mesh::edgeSegment`.
        public unsafe MR.LineSegm3f EdgeSegment(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgeSegment", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_Mesh_edgeSegment(_Underlying *_this, MR.EdgeId e);
            return new(__MR_Mesh_edgeSegment(_UnderlyingPtr, e), is_owning: true);
        }

        /// returns a point on the edge: origin point for f=0 and destination point for f=1
        /// Generated from method `MR::Mesh::edgePoint`.
        public unsafe MR.Vector3f EdgePoint(MR.EdgeId e, float f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgePoint_2", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_edgePoint_2(_Underlying *_this, MR.EdgeId e, float f);
            return __MR_Mesh_edgePoint_2(_UnderlyingPtr, e, f);
        }

        /// computes coordinates of point given as edge and relative position on it
        /// Generated from method `MR::Mesh::edgePoint`.
        public unsafe MR.Vector3f EdgePoint(MR.Const_EdgePoint ep)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgePoint_1", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_edgePoint_1(_Underlying *_this, MR.Const_EdgePoint._Underlying *ep);
            return __MR_Mesh_edgePoint_1(_UnderlyingPtr, ep._UnderlyingPtr);
        }

        /// computes the center of given edge
        /// Generated from method `MR::Mesh::edgeCenter`.
        public unsafe MR.Vector3f EdgeCenter(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgeCenter", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_edgeCenter(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_edgeCenter(_UnderlyingPtr, e);
        }

        /// returns three points of left face of e: v0 = orgPnt( e ), v1 = destPnt( e )
        /// Generated from method `MR::Mesh::getLeftTriPoints`.
        public unsafe void GetLeftTriPoints(MR.EdgeId e, MR.Mut_Vector3f v0, MR.Mut_Vector3f v1, MR.Mut_Vector3f v2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getLeftTriPoints_4", ExactSpelling = true)]
            extern static void __MR_Mesh_getLeftTriPoints_4(_Underlying *_this, MR.EdgeId e, MR.Mut_Vector3f._Underlying *v0, MR.Mut_Vector3f._Underlying *v1, MR.Mut_Vector3f._Underlying *v2);
            __MR_Mesh_getLeftTriPoints_4(_UnderlyingPtr, e, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// returns three points of left face of e: res[0] = orgPnt( e ), res[1] = destPnt( e )
        /// Generated from method `MR::Mesh::getLeftTriPoints`.
        public unsafe MR.Std.Array_MRVector3f_3 GetLeftTriPoints(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getLeftTriPoints_1", ExactSpelling = true)]
            extern static MR.Std.Array_MRVector3f_3 __MR_Mesh_getLeftTriPoints_1(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_getLeftTriPoints_1(_UnderlyingPtr, e);
        }

        /// returns three points of given face
        /// Generated from method `MR::Mesh::getTriPoints`.
        public unsafe void GetTriPoints(MR.FaceId f, MR.Mut_Vector3f v0, MR.Mut_Vector3f v1, MR.Mut_Vector3f v2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getTriPoints_4", ExactSpelling = true)]
            extern static void __MR_Mesh_getTriPoints_4(_Underlying *_this, MR.FaceId f, MR.Mut_Vector3f._Underlying *v0, MR.Mut_Vector3f._Underlying *v1, MR.Mut_Vector3f._Underlying *v2);
            __MR_Mesh_getTriPoints_4(_UnderlyingPtr, f, v0._UnderlyingPtr, v1._UnderlyingPtr, v2._UnderlyingPtr);
        }

        /// returns three points of given face
        /// Generated from method `MR::Mesh::getTriPoints`.
        public unsafe MR.Std.Array_MRVector3f_3 GetTriPoints(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getTriPoints_1", ExactSpelling = true)]
            extern static MR.Std.Array_MRVector3f_3 __MR_Mesh_getTriPoints_1(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_getTriPoints_1(_UnderlyingPtr, f);
        }

        /// computes coordinates of point given as face and barycentric representation
        /// Generated from method `MR::Mesh::triPoint`.
        public unsafe MR.Vector3f TriPoint(MR.Const_MeshTriPoint p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_triPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_triPoint(_Underlying *_this, MR.Const_MeshTriPoint._Underlying *p);
            return __MR_Mesh_triPoint(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// returns the centroid of given triangle
        /// Generated from method `MR::Mesh::triCenter`.
        public unsafe MR.Vector3f TriCenter(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_triCenter", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_triCenter(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_triCenter(_UnderlyingPtr, f);
        }

        /// returns aspect ratio of given mesh triangle equal to the ratio of the circum-radius to twice its in-radius
        /// Generated from method `MR::Mesh::triangleAspectRatio`.
        public unsafe float TriangleAspectRatio(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_triangleAspectRatio", ExactSpelling = true)]
            extern static float __MR_Mesh_triangleAspectRatio(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_triangleAspectRatio(_UnderlyingPtr, f);
        }

        /// returns squared circumcircle diameter of given mesh triangle
        /// Generated from method `MR::Mesh::circumcircleDiameterSq`.
        public unsafe float CircumcircleDiameterSq(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_circumcircleDiameterSq", ExactSpelling = true)]
            extern static float __MR_Mesh_circumcircleDiameterSq(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_circumcircleDiameterSq(_UnderlyingPtr, f);
        }

        /// returns circumcircle diameter of given mesh triangle
        /// Generated from method `MR::Mesh::circumcircleDiameter`.
        public unsafe float CircumcircleDiameter(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_circumcircleDiameter", ExactSpelling = true)]
            extern static float __MR_Mesh_circumcircleDiameter(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_circumcircleDiameter(_UnderlyingPtr, f);
        }

        /// converts vertex into barycentric representation
        /// Generated from method `MR::Mesh::toTriPoint`.
        public unsafe MR.MeshTriPoint ToTriPoint(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_toTriPoint_1_MR_VertId", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_Mesh_toTriPoint_1_MR_VertId(_Underlying *_this, MR.VertId v);
            return new(__MR_Mesh_toTriPoint_1_MR_VertId(_UnderlyingPtr, v), is_owning: true);
        }

        /// converts face id and 3d point into barycentric representation
        /// Generated from method `MR::Mesh::toTriPoint`.
        public unsafe MR.MeshTriPoint ToTriPoint(MR.FaceId f, MR.Const_Vector3f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_toTriPoint_2", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_Mesh_toTriPoint_2(_Underlying *_this, MR.FaceId f, MR.Const_Vector3f._Underlying *p);
            return new(__MR_Mesh_toTriPoint_2(_UnderlyingPtr, f, p._UnderlyingPtr), is_owning: true);
        }

        /// converts face id and 3d point into barycentric representation
        /// Generated from method `MR::Mesh::toTriPoint`.
        public unsafe MR.MeshTriPoint ToTriPoint(MR.Const_PointOnFace p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_toTriPoint_1_MR_PointOnFace", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_Mesh_toTriPoint_1_MR_PointOnFace(_Underlying *_this, MR.Const_PointOnFace._Underlying *p);
            return new(__MR_Mesh_toTriPoint_1_MR_PointOnFace(_UnderlyingPtr, p._UnderlyingPtr), is_owning: true);
        }

        /// converts vertex into edge-point representation
        /// Generated from method `MR::Mesh::toEdgePoint`.
        public unsafe MR.EdgePoint ToEdgePoint(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_toEdgePoint_1", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_Mesh_toEdgePoint_1(_Underlying *_this, MR.VertId v);
            return new(__MR_Mesh_toEdgePoint_1(_UnderlyingPtr, v), is_owning: true);
        }

        /// converts edge and 3d point into edge-point representation
        /// Generated from method `MR::Mesh::toEdgePoint`.
        public unsafe MR.EdgePoint ToEdgePoint(MR.EdgeId e, MR.Const_Vector3f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_toEdgePoint_2", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_Mesh_toEdgePoint_2(_Underlying *_this, MR.EdgeId e, MR.Const_Vector3f._Underlying *p);
            return new(__MR_Mesh_toEdgePoint_2(_UnderlyingPtr, e, p._UnderlyingPtr), is_owning: true);
        }

        /// returns one of three face vertices, closest to given point
        /// Generated from method `MR::Mesh::getClosestVertex`.
        public unsafe MR.VertId GetClosestVertex(MR.Const_PointOnFace p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getClosestVertex_MR_PointOnFace", ExactSpelling = true)]
            extern static MR.VertId __MR_Mesh_getClosestVertex_MR_PointOnFace(_Underlying *_this, MR.Const_PointOnFace._Underlying *p);
            return __MR_Mesh_getClosestVertex_MR_PointOnFace(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// returns one of three face vertices, closest to given point
        /// Generated from method `MR::Mesh::getClosestVertex`.
        public unsafe MR.VertId GetClosestVertex(MR.Const_MeshTriPoint p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getClosestVertex_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.VertId __MR_Mesh_getClosestVertex_MR_MeshTriPoint(_Underlying *_this, MR.Const_MeshTriPoint._Underlying *p);
            return __MR_Mesh_getClosestVertex_MR_MeshTriPoint(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// returns one of three face edges, closest to given point
        /// Generated from method `MR::Mesh::getClosestEdge`.
        public unsafe MR.UndirectedEdgeId GetClosestEdge(MR.Const_PointOnFace p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getClosestEdge_MR_PointOnFace", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Mesh_getClosestEdge_MR_PointOnFace(_Underlying *_this, MR.Const_PointOnFace._Underlying *p);
            return __MR_Mesh_getClosestEdge_MR_PointOnFace(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// returns one of three face edges, closest to given point
        /// Generated from method `MR::Mesh::getClosestEdge`.
        public unsafe MR.UndirectedEdgeId GetClosestEdge(MR.Const_MeshTriPoint p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getClosestEdge_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_Mesh_getClosestEdge_MR_MeshTriPoint(_Underlying *_this, MR.Const_MeshTriPoint._Underlying *p);
            return __MR_Mesh_getClosestEdge_MR_MeshTriPoint(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// returns Euclidean length of the edge
        /// Generated from method `MR::Mesh::edgeLength`.
        public unsafe float EdgeLength(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgeLength", ExactSpelling = true)]
            extern static float __MR_Mesh_edgeLength(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_edgeLength(_UnderlyingPtr, e);
        }

        /// computes and returns the lengths of all edges in the mesh
        /// Generated from method `MR::Mesh::edgeLengths`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeScalars> EdgeLengths()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgeLengths", ExactSpelling = true)]
            extern static MR.UndirectedEdgeScalars._Underlying *__MR_Mesh_edgeLengths(_Underlying *_this);
            return MR.Misc.Move(new MR.UndirectedEdgeScalars(__MR_Mesh_edgeLengths(_UnderlyingPtr), is_owning: true));
        }

        /// returns squared Euclidean length of the edge (faster to compute than length)
        /// Generated from method `MR::Mesh::edgeLengthSq`.
        public unsafe float EdgeLengthSq(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_edgeLengthSq", ExactSpelling = true)]
            extern static float __MR_Mesh_edgeLengthSq(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_edgeLengthSq(_UnderlyingPtr, e);
        }

        /// computes directed double area of left triangular face of given edge
        /// Generated from method `MR::Mesh::leftDirDblArea`.
        public unsafe MR.Vector3f LeftDirDblArea(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_leftDirDblArea", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_leftDirDblArea(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_leftDirDblArea(_UnderlyingPtr, e);
        }

        /// computes directed double area for a triangular face from its vertices
        /// Generated from method `MR::Mesh::dirDblArea`.
        public unsafe MR.Vector3f DirDblArea(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dirDblArea_MR_FaceId", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_dirDblArea_MR_FaceId(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_dirDblArea_MR_FaceId(_UnderlyingPtr, f);
        }

        /// returns twice the area of given face
        /// Generated from method `MR::Mesh::dblArea`.
        public unsafe float DblArea(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dblArea_MR_FaceId", ExactSpelling = true)]
            extern static float __MR_Mesh_dblArea_MR_FaceId(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_dblArea_MR_FaceId(_UnderlyingPtr, f);
        }

        /// returns the area of given face
        /// Generated from method `MR::Mesh::area`.
        public unsafe float Area(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_area_MR_FaceId", ExactSpelling = true)]
            extern static float __MR_Mesh_area_MR_FaceId(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_area_MR_FaceId(_UnderlyingPtr, f);
        }

        /// computes the area of given face-region (or whole mesh)
        /// Generated from method `MR::Mesh::area`.
        public unsafe double Area(MR.Const_FaceBitSet? fs = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_area_const_MR_FaceBitSet_ptr", ExactSpelling = true)]
            extern static double __MR_Mesh_area_const_MR_FaceBitSet_ptr(_Underlying *_this, MR.Const_FaceBitSet._Underlying *fs);
            return __MR_Mesh_area_const_MR_FaceBitSet_ptr(_UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
        }

        /// computes the sum of directed areas for faces from given region (or whole mesh)
        /// Generated from method `MR::Mesh::dirArea`.
        public unsafe MR.Vector3d DirArea(MR.Const_FaceBitSet? fs = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dirArea", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Mesh_dirArea(_Underlying *_this, MR.Const_FaceBitSet._Underlying *fs);
            return __MR_Mesh_dirArea(_UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
        }

        /// computes the sum of absolute projected area of faces from given region (or whole mesh) as visible if look from given direction
        /// Generated from method `MR::Mesh::projArea`.
        public unsafe double ProjArea(MR.Const_Vector3f dir, MR.Const_FaceBitSet? fs = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_projArea", ExactSpelling = true)]
            extern static double __MR_Mesh_projArea(_Underlying *_this, MR.Const_Vector3f._Underlying *dir, MR.Const_FaceBitSet._Underlying *fs);
            return __MR_Mesh_projArea(_UnderlyingPtr, dir._UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
        }

        /// returns volume of the object surrounded by given region (or whole mesh if (region) is nullptr);
        /// if the region has holes then each hole will be virtually filled by adding triangles for each edge and the hole's geometrical center
        /// Generated from method `MR::Mesh::volume`.
        public unsafe double Volume(MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_volume", ExactSpelling = true)]
            extern static double __MR_Mesh_volume(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region);
            return __MR_Mesh_volume(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// computes the perimeter of the hole specified by one of its edges with no valid left face (left is hole)
        /// Generated from method `MR::Mesh::holePerimiter`.
        public unsafe double HolePerimiter(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_holePerimiter", ExactSpelling = true)]
            extern static double __MR_Mesh_holePerimiter(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_holePerimiter(_UnderlyingPtr, e);
        }

        /// computes directed area of the hole specified by one of its edges with no valid left face (left is hole);
        /// if the hole is planar then returned vector is orthogonal to the plane pointing outside and its magnitude is equal to hole area
        /// Generated from method `MR::Mesh::holeDirArea`.
        public unsafe MR.Vector3d HoleDirArea(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_holeDirArea", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Mesh_holeDirArea(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_holeDirArea(_UnderlyingPtr, e);
        }

        /// computes unit vector that is both orthogonal to given edge and to the normal of its left triangle, the vector is directed inside left triangle
        /// Generated from method `MR::Mesh::leftTangent`.
        public unsafe MR.Vector3f LeftTangent(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_leftTangent", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_leftTangent(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_leftTangent(_UnderlyingPtr, e);
        }

        /// computes triangular face normal from its vertices
        /// Generated from method `MR::Mesh::leftNormal`.
        public unsafe MR.Vector3f LeftNormal(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_leftNormal", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_leftNormal(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_leftNormal(_UnderlyingPtr, e);
        }

        /// computes triangular face normal from its vertices
        /// Generated from method `MR::Mesh::normal`.
        public unsafe MR.Vector3f Normal(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_normal_MR_FaceId", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_normal_MR_FaceId(_Underlying *_this, MR.FaceId f);
            return __MR_Mesh_normal_MR_FaceId(_UnderlyingPtr, f);
        }

        /// returns the plane containing given triangular face with normal looking outwards
        /// Generated from method `MR::Mesh::getPlane3f`.
        public unsafe MR.Plane3f GetPlane3f(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getPlane3f", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_Mesh_getPlane3f(_Underlying *_this, MR.FaceId f);
            return new(__MR_Mesh_getPlane3f(_UnderlyingPtr, f), is_owning: true);
        }

        /// Generated from method `MR::Mesh::getPlane3d`.
        public unsafe MR.Plane3d GetPlane3d(MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getPlane3d", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_Mesh_getPlane3d(_Underlying *_this, MR.FaceId f);
            return new(__MR_Mesh_getPlane3d(_UnderlyingPtr, f), is_owning: true);
        }

        /// computes sum of directed double areas of all triangles around given vertex
        /// Generated from method `MR::Mesh::dirDblArea`.
        public unsafe MR.Vector3f DirDblArea(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dirDblArea_MR_VertId", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_dirDblArea_MR_VertId(_Underlying *_this, MR.VertId v);
            return __MR_Mesh_dirDblArea_MR_VertId(_UnderlyingPtr, v);
        }

        /// computes the length of summed directed double areas of all triangles around given vertex
        /// Generated from method `MR::Mesh::dblArea`.
        public unsafe float DblArea(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dblArea_MR_VertId", ExactSpelling = true)]
            extern static float __MR_Mesh_dblArea_MR_VertId(_Underlying *_this, MR.VertId v);
            return __MR_Mesh_dblArea_MR_VertId(_UnderlyingPtr, v);
        }

        /// computes normal in a vertex using sum of directed areas of neighboring triangles
        /// Generated from method `MR::Mesh::normal`.
        public unsafe MR.Vector3f Normal(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_normal_MR_VertId", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_normal_MR_VertId(_Underlying *_this, MR.VertId v);
            return __MR_Mesh_normal_MR_VertId(_UnderlyingPtr, v);
        }

        /// computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates and normalizes again;
        /// this is the same normal as in rendering with smooth shading
        /// Generated from method `MR::Mesh::normal`.
        public unsafe MR.Vector3f Normal(MR.Const_MeshTriPoint p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_normal_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_normal_MR_MeshTriPoint(_Underlying *_this, MR.Const_MeshTriPoint._Underlying *p);
            return __MR_Mesh_normal_MR_MeshTriPoint(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// computes angle-weighted sum of normals of incident faces of given vertex (only (region) faces will be considered);
        /// the sum is normalized before returning
        /// Generated from method `MR::Mesh::pseudonormal`.
        public unsafe MR.Vector3f Pseudonormal(MR.VertId v, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_pseudonormal_MR_VertId", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_pseudonormal_MR_VertId(_Underlying *_this, MR.VertId v, MR.Const_FaceBitSet._Underlying *region);
            return __MR_Mesh_pseudonormal_MR_VertId(_UnderlyingPtr, v, region is not null ? region._UnderlyingPtr : null);
        }

        /// computes normalized half sum of face normals sharing given edge (only (region) faces will be considered);
        /// Generated from method `MR::Mesh::pseudonormal`.
        public unsafe MR.Vector3f Pseudonormal(MR.UndirectedEdgeId e, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_pseudonormal_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_pseudonormal_MR_UndirectedEdgeId(_Underlying *_this, MR.UndirectedEdgeId e, MR.Const_FaceBitSet._Underlying *region);
            return __MR_Mesh_pseudonormal_MR_UndirectedEdgeId(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
        }

        /// returns pseudonormal in corresponding face/edge/vertex for signed distance calculation
        /// as suggested in the article "Signed Distance Computation Using the Angle Weighted Pseudonormal" by J. Andreas Baerentzen and Henrik Aanaes,
        /// https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B_rentzen.pdf
        /// unlike normal( const MeshTriPoint & p ), this is not a smooth function
        /// Generated from method `MR::Mesh::pseudonormal`.
        public unsafe MR.Vector3f Pseudonormal(MR.Const_MeshTriPoint p, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_pseudonormal_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_pseudonormal_MR_MeshTriPoint(_Underlying *_this, MR.Const_MeshTriPoint._Underlying *p, MR.Const_FaceBitSet._Underlying *region);
            return __MR_Mesh_pseudonormal_MR_MeshTriPoint(_UnderlyingPtr, p._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// given a point (pt) in 3D and the closest point to in on mesh (proj),
        /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
        /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
        /// Generated from method `MR::Mesh::signedDistance`.
        public unsafe float SignedDistance(MR.Const_Vector3f pt, MR.Const_MeshProjectionResult proj, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_signedDistance_3_MR_MeshProjectionResult", ExactSpelling = true)]
            extern static float __MR_Mesh_signedDistance_3_MR_MeshProjectionResult(_Underlying *_this, MR.Const_Vector3f._Underlying *pt, MR.Const_MeshProjectionResult._Underlying *proj, MR.Const_FaceBitSet._Underlying *region);
            return __MR_Mesh_signedDistance_3_MR_MeshProjectionResult(_UnderlyingPtr, pt._UnderlyingPtr, proj._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// given a point (pt) in 3D, computes the closest point on mesh, and
        /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
        /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
        /// Generated from method `MR::Mesh::signedDistance`.
        public unsafe float SignedDistance(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_signedDistance_1", ExactSpelling = true)]
            extern static float __MR_Mesh_signedDistance_1(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Mesh_signedDistance_1(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// given a point (pt) in 3D, computes the closest point on mesh, and
        /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
        ///   or std::nullopt if the projection point is not within maxDist;
        /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
        /// Generated from method `MR::Mesh::signedDistance`.
        public unsafe MR.Std.Optional_Float SignedDistance(MR.Const_Vector3f pt, float maxDistSq, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_signedDistance_3_float", ExactSpelling = true)]
            extern static MR.Std.Optional_Float._Underlying *__MR_Mesh_signedDistance_3_float(_Underlying *_this, MR.Const_Vector3f._Underlying *pt, float maxDistSq, MR.Const_FaceBitSet._Underlying *region);
            return new(__MR_Mesh_signedDistance_3_float(_UnderlyingPtr, pt._UnderlyingPtr, maxDistSq, region is not null ? region._UnderlyingPtr : null), is_owning: true);
        }

        /// computes generalized winding number in a point (pt), which is
        /// * for closed mesh with normals outside: 1 inside, 0 outside;
        /// * for planar mesh: 0.5 inside, -0.5 outside;
        /// and in general is equal to (portion of solid angle where inside part of mesh is observable) minus (portion of solid angle where outside part of mesh is observable)
        /// \param beta determines the precision of fast approximation: the more the better, recommended value 2 or more
        /// Generated from method `MR::Mesh::calcFastWindingNumber`.
        /// Parameter `beta` defaults to `2`.
        public unsafe float CalcFastWindingNumber(MR.Const_Vector3f pt, float? beta = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_calcFastWindingNumber", ExactSpelling = true)]
            extern static float __MR_Mesh_calcFastWindingNumber(_Underlying *_this, MR.Const_Vector3f._Underlying *pt, float *beta);
            float __deref_beta = beta.GetValueOrDefault();
            return __MR_Mesh_calcFastWindingNumber(_UnderlyingPtr, pt._UnderlyingPtr, beta.HasValue ? &__deref_beta : null);
        }

        /// computes whether a point (pt) is located outside the object surrounded by this mesh using generalized winding number
        /// \param beta determines the precision of winding number computation: the more the better, recommended value 2 or more
        /// Generated from method `MR::Mesh::isOutside`.
        /// Parameter `windingNumberThreshold` defaults to `0.5f`.
        /// Parameter `beta` defaults to `2`.
        public unsafe bool IsOutside(MR.Const_Vector3f pt, float? windingNumberThreshold = null, float? beta = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_isOutside", ExactSpelling = true)]
            extern static byte __MR_Mesh_isOutside(_Underlying *_this, MR.Const_Vector3f._Underlying *pt, float *windingNumberThreshold, float *beta);
            float __deref_windingNumberThreshold = windingNumberThreshold.GetValueOrDefault();
            float __deref_beta = beta.GetValueOrDefault();
            return __MR_Mesh_isOutside(_UnderlyingPtr, pt._UnderlyingPtr, windingNumberThreshold.HasValue ? &__deref_windingNumberThreshold : null, beta.HasValue ? &__deref_beta : null) != 0;
        }

        /// computes whether a point (pt) is located outside the object surrounded by this mesh
        /// using pseudonormal at the closest point to in on mesh (proj);
        /// this method works much faster than \ref isOutside but can return wrong sign if the closest point is located on self-intersecting part of the mesh
        /// Generated from method `MR::Mesh::isOutsideByProjNorm`.
        public unsafe bool IsOutsideByProjNorm(MR.Const_Vector3f pt, MR.Const_MeshProjectionResult proj, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_isOutsideByProjNorm", ExactSpelling = true)]
            extern static byte __MR_Mesh_isOutsideByProjNorm(_Underlying *_this, MR.Const_Vector3f._Underlying *pt, MR.Const_MeshProjectionResult._Underlying *proj, MR.Const_FaceBitSet._Underlying *region);
            return __MR_Mesh_isOutsideByProjNorm(_UnderlyingPtr, pt._UnderlyingPtr, proj._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// computes the sum of triangle angles at given vertex; optionally returns whether the vertex is on boundary
        /// Generated from method `MR::Mesh::sumAngles`.
        public unsafe float SumAngles(MR.VertId v, MR.Misc.InOut<bool>? outBoundaryVert = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_sumAngles", ExactSpelling = true)]
            extern static float __MR_Mesh_sumAngles(_Underlying *_this, MR.VertId v, bool *outBoundaryVert);
            bool __value_outBoundaryVert = outBoundaryVert is not null ? outBoundaryVert.Value : default(bool);
            var __ret = __MR_Mesh_sumAngles(_UnderlyingPtr, v, outBoundaryVert is not null ? &__value_outBoundaryVert : null);
            if (outBoundaryVert is not null) outBoundaryVert.Value = __value_outBoundaryVert;
            return __ret;
        }

        /// returns vertices where the sum of triangle angles is below given threshold
        /// Generated from method `MR::Mesh::findSpikeVertices`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_MRVertBitSet_StdString> FindSpikeVertices(float minSumAngle, MR.Const_VertBitSet? region = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findSpikeVertices", ExactSpelling = true)]
            extern static MR.Expected_MRVertBitSet_StdString._Underlying *__MR_Mesh_findSpikeVertices(_Underlying *_this, float minSumAngle, MR.Const_VertBitSet._Underlying *region, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_MRVertBitSet_StdString(__MR_Mesh_findSpikeVertices(_UnderlyingPtr, minSumAngle, region is not null ? region._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// given an edge between two triangular faces, computes sine of dihedral angle between them:
        /// 0 if both faces are in the same plane,
        /// positive if the faces form convex surface,
        /// negative if the faces form concave surface
        /// Generated from method `MR::Mesh::dihedralAngleSin`.
        public unsafe float DihedralAngleSin(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dihedralAngleSin", ExactSpelling = true)]
            extern static float __MR_Mesh_dihedralAngleSin(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_dihedralAngleSin(_UnderlyingPtr, e);
        }

        /// given an edge between two triangular faces, computes cosine of dihedral angle between them:
        /// 1 if both faces are in the same plane,
        /// 0 if the surface makes right angle turn at the edge,
        /// -1 if the faces overlap one another
        /// Generated from method `MR::Mesh::dihedralAngleCos`.
        public unsafe float DihedralAngleCos(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dihedralAngleCos", ExactSpelling = true)]
            extern static float __MR_Mesh_dihedralAngleCos(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_dihedralAngleCos(_UnderlyingPtr, e);
        }

        /// given an edge between two triangular faces, computes the dihedral angle between them:
        /// 0 if both faces are in the same plane,
        /// positive if the faces form convex surface,
        /// negative if the faces form concave surface;
        /// please consider the usage of faster dihedralAngleSin(e) and dihedralAngleCos(e)
        /// Generated from method `MR::Mesh::dihedralAngle`.
        public unsafe float DihedralAngle(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_dihedralAngle", ExactSpelling = true)]
            extern static float __MR_Mesh_dihedralAngle(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_dihedralAngle(_UnderlyingPtr, e);
        }

        /// computes discrete mean curvature in given vertex, measures in length^-1;
        /// 0 for planar regions, positive for convex surface, negative for concave surface
        /// Generated from method `MR::Mesh::discreteMeanCurvature`.
        public unsafe float DiscreteMeanCurvature(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_discreteMeanCurvature_MR_VertId", ExactSpelling = true)]
            extern static float __MR_Mesh_discreteMeanCurvature_MR_VertId(_Underlying *_this, MR.VertId v);
            return __MR_Mesh_discreteMeanCurvature_MR_VertId(_UnderlyingPtr, v);
        }

        /// computes discrete mean curvature in given edge, measures in length^-1;
        /// 0 for planar regions, positive for convex surface, negative for concave surface
        /// Generated from method `MR::Mesh::discreteMeanCurvature`.
        public unsafe float DiscreteMeanCurvature(MR.UndirectedEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_discreteMeanCurvature_MR_UndirectedEdgeId", ExactSpelling = true)]
            extern static float __MR_Mesh_discreteMeanCurvature_MR_UndirectedEdgeId(_Underlying *_this, MR.UndirectedEdgeId e);
            return __MR_Mesh_discreteMeanCurvature_MR_UndirectedEdgeId(_UnderlyingPtr, e);
        }

        /// computes discrete Gaussian curvature (or angle defect) at given vertex,
        /// which 0 in inner vertices on planar mesh parts and reaches 2*pi on needle's tip, see http://math.uchicago.edu/~may/REU2015/REUPapers/Upadhyay.pdf
        /// optionally returns whether the vertex is on boundary
        /// Generated from method `MR::Mesh::discreteGaussianCurvature`.
        public unsafe float DiscreteGaussianCurvature(MR.VertId v, MR.Misc.InOut<bool>? outBoundaryVert = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_discreteGaussianCurvature", ExactSpelling = true)]
            extern static float __MR_Mesh_discreteGaussianCurvature(_Underlying *_this, MR.VertId v, bool *outBoundaryVert);
            bool __value_outBoundaryVert = outBoundaryVert is not null ? outBoundaryVert.Value : default(bool);
            var __ret = __MR_Mesh_discreteGaussianCurvature(_UnderlyingPtr, v, outBoundaryVert is not null ? &__value_outBoundaryVert : null);
            if (outBoundaryVert is not null) outBoundaryVert.Value = __value_outBoundaryVert;
            return __ret;
        }

        /// finds all mesh edges where dihedral angle is distinct from planar PI angle on at least given value
        /// Generated from method `MR::Mesh::findCreaseEdges`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> FindCreaseEdges(float angleFromPlanar)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findCreaseEdges", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_Mesh_findCreaseEdges(_Underlying *_this, float angleFromPlanar);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_Mesh_findCreaseEdges(_UnderlyingPtr, angleFromPlanar), is_owning: true));
        }

        /// computes cotangent of the angle in the left( e ) triangle opposite to e,
        /// and returns 0 if left face does not exist
        /// Generated from method `MR::Mesh::leftCotan`.
        public unsafe float LeftCotan(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_leftCotan", ExactSpelling = true)]
            extern static float __MR_Mesh_leftCotan(_Underlying *_this, MR.EdgeId e);
            return __MR_Mesh_leftCotan(_UnderlyingPtr, e);
        }

        /// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
        /// consider cotangents zero for not existing triangles
        /// Generated from method `MR::Mesh::cotan`.
        public unsafe float Cotan(MR.UndirectedEdgeId ue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_cotan", ExactSpelling = true)]
            extern static float __MR_Mesh_cotan(_Underlying *_this, MR.UndirectedEdgeId ue);
            return __MR_Mesh_cotan(_UnderlyingPtr, ue);
        }

        /// computes quadratic form in the vertex as the sum of squared distances from
        /// 1) planes of adjacent triangles, with the weight equal to the angle of adjacent triangle at this vertex divided on PI in case of angleWeigted=true;
        /// 2) lines of adjacent boundary and crease edges
        /// Generated from method `MR::Mesh::quadraticForm`.
        public unsafe MR.QuadraticForm3f QuadraticForm(MR.VertId v, bool angleWeigted, MR.Const_FaceBitSet? region = null, MR.Const_UndirectedEdgeBitSet? creases = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_quadraticForm", ExactSpelling = true)]
            extern static MR.QuadraticForm3f._Underlying *__MR_Mesh_quadraticForm(_Underlying *_this, MR.VertId v, byte angleWeigted, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *creases);
            return new(__MR_Mesh_quadraticForm(_UnderlyingPtr, v, angleWeigted ? (byte)1 : (byte)0, region is not null ? region._UnderlyingPtr : null, creases is not null ? creases._UnderlyingPtr : null), is_owning: true);
        }

        /// returns the bounding box containing all valid vertices (implemented via getAABBTree())
        /// this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision
        /// Generated from method `MR::Mesh::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_Mesh_getBoundingBox(_Underlying *_this);
            return __MR_Mesh_getBoundingBox(_UnderlyingPtr);
        }

        /// passes through all valid vertices and finds the minimal bounding box containing all of them;
        /// if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::Mesh::computeBoundingBox`.
        public unsafe MR.Box3f ComputeBoundingBox(MR.Const_AffineXf3f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_computeBoundingBox_1", ExactSpelling = true)]
            extern static MR.Box3f __MR_Mesh_computeBoundingBox_1(_Underlying *_this, MR.Const_AffineXf3f._Underlying *toWorld);
            return __MR_Mesh_computeBoundingBox_1(_UnderlyingPtr, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        /// passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them
        /// if toWorld transformation is given then returns minimal bounding box in world space
        /// Generated from method `MR::Mesh::computeBoundingBox`.
        public unsafe MR.Box3f ComputeBoundingBox(MR.Const_FaceBitSet? region, MR.Const_AffineXf3f? toWorld = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_computeBoundingBox_2", ExactSpelling = true)]
            extern static MR.Box3f __MR_Mesh_computeBoundingBox_2(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *toWorld);
            return __MR_Mesh_computeBoundingBox_2(_UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, toWorld is not null ? toWorld._UnderlyingPtr : null);
        }

        /// computes average length of an edge in this mesh
        /// Generated from method `MR::Mesh::averageEdgeLength`.
        public unsafe float AverageEdgeLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_averageEdgeLength", ExactSpelling = true)]
            extern static float __MR_Mesh_averageEdgeLength(_Underlying *_this);
            return __MR_Mesh_averageEdgeLength(_UnderlyingPtr);
        }

        /// computes average position of all valid mesh vertices
        /// Generated from method `MR::Mesh::findCenterFromPoints`.
        public unsafe MR.Vector3f FindCenterFromPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findCenterFromPoints", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_findCenterFromPoints(_Underlying *_this);
            return __MR_Mesh_findCenterFromPoints(_UnderlyingPtr);
        }

        /// computes center of mass considering that density of all triangles is the same
        /// Generated from method `MR::Mesh::findCenterFromFaces`.
        public unsafe MR.Vector3f FindCenterFromFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findCenterFromFaces", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_findCenterFromFaces(_Underlying *_this);
            return __MR_Mesh_findCenterFromFaces(_UnderlyingPtr);
        }

        /// computes bounding box and returns its center
        /// Generated from method `MR::Mesh::findCenterFromBBox`.
        public unsafe MR.Vector3f FindCenterFromBBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findCenterFromBBox", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Mesh_findCenterFromBBox(_Underlying *_this);
            return __MR_Mesh_findCenterFromBBox(_UnderlyingPtr);
        }

        /// creates new mesh from given triangles of this mesh
        /// Generated from method `MR::Mesh::cloneRegion`.
        /// Parameter `flipOrientation` defaults to `false`.
        /// Parameter `map` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Mesh> CloneRegion(MR.Const_FaceBitSet region, bool? flipOrientation = null, MR.Const_PartMapping? map = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_cloneRegion", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_cloneRegion(_Underlying *_this, MR.Const_FaceBitSet._Underlying *region, byte *flipOrientation, MR.Const_PartMapping._Underlying *map);
            byte __deref_flipOrientation = flipOrientation.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.Mesh(__MR_Mesh_cloneRegion(_UnderlyingPtr, region._UnderlyingPtr, flipOrientation.HasValue ? &__deref_flipOrientation : null, map is not null ? map._UnderlyingPtr : null), is_owning: true));
        }

        /// finds the closest mesh point on this mesh (or its region) to given point;
        /// \param point source location to look the closest to
        /// \param res found closest point including Euclidean coordinates and FaceId
        /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
        /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
        /// \return false if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
        /// Generated from method `MR::Mesh::projectPoint`.
        /// Parameter `maxDistSq` defaults to `3.40282347e38f`.
        public unsafe bool ProjectPoint(MR.Const_Vector3f point, MR.PointOnFace res, float? maxDistSq = null, MR.Const_FaceBitSet? region = null, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_projectPoint_5_MR_PointOnFace", ExactSpelling = true)]
            extern static byte __MR_Mesh_projectPoint_5_MR_PointOnFace(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.PointOnFace._Underlying *res, float *maxDistSq, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *xf);
            float __deref_maxDistSq = maxDistSq.GetValueOrDefault();
            return __MR_Mesh_projectPoint_5_MR_PointOnFace(_UnderlyingPtr, point._UnderlyingPtr, res._UnderlyingPtr, maxDistSq.HasValue ? &__deref_maxDistSq : null, region is not null ? region._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null) != 0;
        }

        /// finds the closest mesh point on this mesh (or its region) to given point;
        /// \param point source location to look the closest to
        /// \param res found closest point including Euclidean coordinates, barycentric coordinates, FaceId and squared distance to point
        /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
        /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
        /// \return false if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
        /// Generated from method `MR::Mesh::projectPoint`.
        /// Parameter `maxDistSq` defaults to `3.40282347e38f`.
        public unsafe bool ProjectPoint(MR.Const_Vector3f point, MR.MeshProjectionResult res, float? maxDistSq = null, MR.Const_FaceBitSet? region = null, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_projectPoint_5_MR_MeshProjectionResult", ExactSpelling = true)]
            extern static byte __MR_Mesh_projectPoint_5_MR_MeshProjectionResult(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.MeshProjectionResult._Underlying *res, float *maxDistSq, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *xf);
            float __deref_maxDistSq = maxDistSq.GetValueOrDefault();
            return __MR_Mesh_projectPoint_5_MR_MeshProjectionResult(_UnderlyingPtr, point._UnderlyingPtr, res._UnderlyingPtr, maxDistSq.HasValue ? &__deref_maxDistSq : null, region is not null ? region._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::Mesh::findClosestPoint`.
        /// Parameter `maxDistSq` defaults to `3.40282347e38f`.
        public unsafe bool FindClosestPoint(MR.Const_Vector3f point, MR.MeshProjectionResult res, float? maxDistSq = null, MR.Const_FaceBitSet? region = null, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findClosestPoint_5", ExactSpelling = true)]
            extern static byte __MR_Mesh_findClosestPoint_5(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.MeshProjectionResult._Underlying *res, float *maxDistSq, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *xf);
            float __deref_maxDistSq = maxDistSq.GetValueOrDefault();
            return __MR_Mesh_findClosestPoint_5(_UnderlyingPtr, point._UnderlyingPtr, res._UnderlyingPtr, maxDistSq.HasValue ? &__deref_maxDistSq : null, region is not null ? region._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null) != 0;
        }

        /// finds the closest mesh point on this mesh (or its region) to given point;
        /// \param point source location to look the closest to
        /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
        /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
        /// \return found closest point including Euclidean coordinates, barycentric coordinates, FaceId and squared distance to point
        ///         or std::nullopt if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
        /// Generated from method `MR::Mesh::projectPoint`.
        /// Parameter `maxDistSq` defaults to `3.40282347e38f`.
        public unsafe MR.MeshProjectionResult ProjectPoint(MR.Const_Vector3f point, float? maxDistSq = null, MR.Const_FaceBitSet? region = null, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_projectPoint_4", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_Mesh_projectPoint_4(_Underlying *_this, MR.Const_Vector3f._Underlying *point, float *maxDistSq, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *xf);
            float __deref_maxDistSq = maxDistSq.GetValueOrDefault();
            return new(__MR_Mesh_projectPoint_4(_UnderlyingPtr, point._UnderlyingPtr, maxDistSq.HasValue ? &__deref_maxDistSq : null, region is not null ? region._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null), is_owning: true);
        }

        /// Generated from method `MR::Mesh::findClosestPoint`.
        /// Parameter `maxDistSq` defaults to `3.40282347e38f`.
        public unsafe MR.MeshProjectionResult FindClosestPoint(MR.Const_Vector3f point, float? maxDistSq = null, MR.Const_FaceBitSet? region = null, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_findClosestPoint_4", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_Mesh_findClosestPoint_4(_Underlying *_this, MR.Const_Vector3f._Underlying *point, float *maxDistSq, MR.Const_FaceBitSet._Underlying *region, MR.Const_AffineXf3f._Underlying *xf);
            float __deref_maxDistSq = maxDistSq.GetValueOrDefault();
            return new(__MR_Mesh_findClosestPoint_4(_UnderlyingPtr, point._UnderlyingPtr, maxDistSq.HasValue ? &__deref_maxDistSq : null, region is not null ? region._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null), is_owning: true);
        }

        /// returns cached aabb-tree for this mesh, creating it if it did not exist in a thread-safe manner
        /// Generated from method `MR::Mesh::getAABBTree`.
        public unsafe MR.Const_AABBTree GetAABBTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getAABBTree", ExactSpelling = true)]
            extern static MR.Const_AABBTree._Underlying *__MR_Mesh_getAABBTree(_Underlying *_this);
            return new(__MR_Mesh_getAABBTree(_UnderlyingPtr), is_owning: false);
        }

        /// returns cached aabb-tree for this mesh, but does not create it if it did not exist
        /// Generated from method `MR::Mesh::getAABBTreeNotCreate`.
        public unsafe MR.Const_AABBTree? GetAABBTreeNotCreate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getAABBTreeNotCreate", ExactSpelling = true)]
            extern static MR.Const_AABBTree._Underlying *__MR_Mesh_getAABBTreeNotCreate(_Underlying *_this);
            var __ret = __MR_Mesh_getAABBTreeNotCreate(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_AABBTree(__ret, is_owning: false) : null;
        }

        /// returns cached aabb-tree for points of this mesh, creating it if it did not exist in a thread-safe manner
        /// Generated from method `MR::Mesh::getAABBTreePoints`.
        public unsafe MR.Const_AABBTreePoints GetAABBTreePoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getAABBTreePoints", ExactSpelling = true)]
            extern static MR.Const_AABBTreePoints._Underlying *__MR_Mesh_getAABBTreePoints(_Underlying *_this);
            return new(__MR_Mesh_getAABBTreePoints(_UnderlyingPtr), is_owning: false);
        }

        /// returns cached aabb-tree for points of this mesh, but does not create it if it did not exist
        /// Generated from method `MR::Mesh::getAABBTreePointsNotCreate`.
        public unsafe MR.Const_AABBTreePoints? GetAABBTreePointsNotCreate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getAABBTreePointsNotCreate", ExactSpelling = true)]
            extern static MR.Const_AABBTreePoints._Underlying *__MR_Mesh_getAABBTreePointsNotCreate(_Underlying *_this);
            var __ret = __MR_Mesh_getAABBTreePointsNotCreate(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_AABBTreePoints(__ret, is_owning: false) : null;
        }

        /// returns cached dipoles of aabb-tree nodes for this mesh, creating it if it did not exist in a thread-safe manner
        /// Generated from method `MR::Mesh::getDipoles`.
        public unsafe MR.Const_Dipoles GetDipoles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getDipoles", ExactSpelling = true)]
            extern static MR.Const_Dipoles._Underlying *__MR_Mesh_getDipoles(_Underlying *_this);
            return new(__MR_Mesh_getDipoles(_UnderlyingPtr), is_owning: false);
        }

        /// returns cached dipoles of aabb-tree nodes for this mesh, but does not create it if it did not exist
        /// Generated from method `MR::Mesh::getDipolesNotCreate`.
        public unsafe MR.Const_Dipoles? GetDipolesNotCreate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_getDipolesNotCreate", ExactSpelling = true)]
            extern static MR.Const_Dipoles._Underlying *__MR_Mesh_getDipolesNotCreate(_Underlying *_this);
            var __ret = __MR_Mesh_getDipolesNotCreate(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_Dipoles(__ret, is_owning: false) : null;
        }

        // returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Mesh::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Mesh_heapBytes(_Underlying *_this);
            return __MR_Mesh_heapBytes(_UnderlyingPtr);
        }

        // IEquatable:

        public bool Equals(MR.Const_Mesh? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Mesh)
                return this == (MR.Const_Mesh)other;
            return false;
        }
    }

    /// This class represents a mesh, including topology (connectivity) information and point coordinates,
    /// as well as some caches to accelerate search algorithms
    /// Generated from class `MR::Mesh`.
    /// This is the non-const half of the class.
    public class Mesh : Const_Mesh
    {
        internal unsafe Mesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe Mesh(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        public new unsafe MR.MeshTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_GetMutable_topology", ExactSpelling = true)]
                extern static MR.MeshTopology._Underlying *__MR_Mesh_GetMutable_topology(_Underlying *_this);
                return new(__MR_Mesh_GetMutable_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.VertCoords Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_GetMutable_points", ExactSpelling = true)]
                extern static MR.VertCoords._Underlying *__MR_Mesh_GetMutable_points(_Underlying *_this);
                return new(__MR_Mesh_GetMutable_points(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mesh() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_DefaultConstruct();
            _LateMakeShared(__MR_Mesh_DefaultConstruct());
        }

        /// Generated from constructor `MR::Mesh::Mesh`.
        public unsafe Mesh(MR._ByValue_Mesh _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Mesh._Underlying *_other);
            _LateMakeShared(__MR_Mesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::Mesh::operator=`.
        public unsafe MR.Mesh Assign(MR._ByValue_Mesh _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_Mesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Mesh._Underlying *_other);
            return new(__MR_Mesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// for all points not in topology.getValidVerts() sets coordinates to (0,0,0)
        /// Generated from method `MR::Mesh::zeroUnusedPoints`.
        public unsafe void ZeroUnusedPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_zeroUnusedPoints", ExactSpelling = true)]
            extern static void __MR_Mesh_zeroUnusedPoints(_Underlying *_this);
            __MR_Mesh_zeroUnusedPoints(_UnderlyingPtr);
        }

        /// applies given transformation to specified vertices
        /// if region is nullptr, all valid mesh vertices are used
        /// \snippet cpp-examples/MeshModification.dox.cpp MeshTransform
        /// Generated from method `MR::Mesh::transform`.
        public unsafe void Transform(MR.Const_AffineXf3f xf, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_transform", ExactSpelling = true)]
            extern static void __MR_Mesh_transform(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.Const_VertBitSet._Underlying *region);
            __MR_Mesh_transform(_UnderlyingPtr, xf._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        /// creates new point and assigns given position to it
        /// Generated from method `MR::Mesh::addPoint`.
        public unsafe MR.VertId AddPoint(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addPoint", ExactSpelling = true)]
            extern static MR.VertId __MR_Mesh_addPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_Mesh_addPoint(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// append points to mesh and connect them as closed edge loop
        /// returns first EdgeId of new edges
        /// Generated from method `MR::Mesh::addSeparateEdgeLoop`.
        public unsafe MR.EdgeId AddSeparateEdgeLoop(MR.Std.Const_Vector_MRVector3f contourPoints)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addSeparateEdgeLoop", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Mesh_addSeparateEdgeLoop(_Underlying *_this, MR.Std.Const_Vector_MRVector3f._Underlying *contourPoints);
            return __MR_Mesh_addSeparateEdgeLoop(_UnderlyingPtr, contourPoints._UnderlyingPtr);
        }

        /// append points to mesh and connect them
        /// returns first EdgeId of new edges
        /// Generated from method `MR::Mesh::addSeparateContours`.
        public unsafe MR.EdgeId AddSeparateContours(MR.Std.Const_Vector_StdVectorMRVector3f contours, MR.Const_AffineXf3f? xf = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addSeparateContours", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Mesh_addSeparateContours(_Underlying *_this, MR.Std.Const_Vector_StdVectorMRVector3f._Underlying *contours, MR.Const_AffineXf3f._Underlying *xf);
            return __MR_Mesh_addSeparateContours(_UnderlyingPtr, contours._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// append points to mesh and connect them to given edges making edge loop
        /// first point connects with first edge dest
        /// last point connects with last edge org
        /// note that first and last edge should have no left face
        /// Generated from method `MR::Mesh::attachEdgeLoopPart`.
        public unsafe void AttachEdgeLoopPart(MR.EdgeId first, MR.EdgeId last, MR.Std.Const_Vector_MRVector3f contourPoints)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_attachEdgeLoopPart", ExactSpelling = true)]
            extern static void __MR_Mesh_attachEdgeLoopPart(_Underlying *_this, MR.EdgeId first, MR.EdgeId last, MR.Std.Const_Vector_MRVector3f._Underlying *contourPoints);
            __MR_Mesh_attachEdgeLoopPart(_UnderlyingPtr, first, last, contourPoints._UnderlyingPtr);
        }

        /// split given edge on two parts:
        /// dest(returned-edge) = org(e) - newly created vertex,
        /// org(returned-edge) = org(e-before-split),
        /// dest(e) = dest(e-before-split)
        /// \details left and right faces of given edge if valid are also subdivided on two parts each;
        /// the split edge will keep both face IDs and their degrees, and the new edge will have new face IDs and new faces are triangular;
        /// if left or right faces of the original edge were in the region, then include new parts of these faces in the region
        /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
        /// Generated from method `MR::Mesh::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e, MR.Const_Vector3f newVertPos, MR.FaceBitSet? region = null, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_splitEdge_4", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Mesh_splitEdge_4(_Underlying *_this, MR.EdgeId e, MR.Const_Vector3f._Underlying *newVertPos, MR.FaceBitSet._Underlying *region, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old);
            return __MR_Mesh_splitEdge_4(_UnderlyingPtr, e, newVertPos._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null);
        }

        // same, but split given edge on two equal parts
        /// Generated from method `MR::Mesh::splitEdge`.
        public unsafe MR.EdgeId SplitEdge(MR.EdgeId e, MR.FaceBitSet? region = null, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_splitEdge_3", ExactSpelling = true)]
            extern static MR.EdgeId __MR_Mesh_splitEdge_3(_Underlying *_this, MR.EdgeId e, MR.FaceBitSet._Underlying *region, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old);
            return __MR_Mesh_splitEdge_3(_UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null);
        }

        /// split given triangle on three triangles, introducing new vertex with given coordinates and connecting it to original triangle vertices;
        /// if region is given, then it must include (f) and new faces will be added there as well
        /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
        /// Generated from method `MR::Mesh::splitFace`.
        public unsafe MR.VertId SplitFace(MR.FaceId f, MR.Const_Vector3f newVertPos, MR.FaceBitSet? region = null, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_splitFace_4", ExactSpelling = true)]
            extern static MR.VertId __MR_Mesh_splitFace_4(_Underlying *_this, MR.FaceId f, MR.Const_Vector3f._Underlying *newVertPos, MR.FaceBitSet._Underlying *region, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old);
            return __MR_Mesh_splitFace_4(_UnderlyingPtr, f, newVertPos._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null);
        }

        // same, putting new vertex in the centroid of original triangle
        /// Generated from method `MR::Mesh::splitFace`.
        public unsafe MR.VertId SplitFace(MR.FaceId f, MR.FaceBitSet? region = null, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId? new2Old = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_splitFace_3", ExactSpelling = true)]
            extern static MR.VertId __MR_Mesh_splitFace_3(_Underlying *_this, MR.FaceId f, MR.FaceBitSet._Underlying *region, MR.Phmap.FlatHashMap_MRFaceId_MRFaceId._Underlying *new2Old);
            return __MR_Mesh_splitFace_3(_UnderlyingPtr, f, region is not null ? region._UnderlyingPtr : null, new2Old is not null ? new2Old._UnderlyingPtr : null);
        }

        /// appends another mesh as separate connected component(s) to this
        /// Generated from method `MR::Mesh::addMesh`.
        /// Parameter `map` defaults to `{}`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void AddMesh(MR.Const_Mesh from, MR.Const_PartMapping? map = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addMesh_3", ExactSpelling = true)]
            extern static void __MR_Mesh_addMesh_3(_Underlying *_this, MR.Const_Mesh._Underlying *from, MR.PartMapping._Underlying *map, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Mesh_addMesh_3(_UnderlyingPtr, from._UnderlyingPtr, map is not null ? map._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// Generated from method `MR::Mesh::addMesh`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void AddMesh(MR.Const_Mesh from, MR.FaceMap? outFmap, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addMesh_5", ExactSpelling = true)]
            extern static void __MR_Mesh_addMesh_5(_Underlying *_this, MR.Const_Mesh._Underlying *from, MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Mesh_addMesh_5(_UnderlyingPtr, from._UnderlyingPtr, outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// appends whole or part of another mesh as separate connected component(s) to this
        /// Generated from method `MR::Mesh::addMeshPart`.
        public unsafe void AddMeshPart(MR.Const_MeshPart from, MR.Const_PartMapping map)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addMeshPart_2", ExactSpelling = true)]
            extern static void __MR_Mesh_addMeshPart_2(_Underlying *_this, MR.Const_MeshPart._Underlying *from, MR.Const_PartMapping._Underlying *map);
            __MR_Mesh_addMeshPart_2(_UnderlyingPtr, from._UnderlyingPtr, map._UnderlyingPtr);
        }

        /// appends whole or part of another mesh to this joining added faces with existed ones along given contours
        /// \param flipOrientation true means that every (from) triangle is inverted before adding
        /// Generated from method `MR::Mesh::addMeshPart`.
        /// Parameter `flipOrientation` defaults to `false`.
        /// Parameter `thisContours` defaults to `{}`.
        /// Parameter `fromContours` defaults to `{}`.
        /// Parameter `map` defaults to `{}`.
        public unsafe void AddMeshPart(MR.Const_MeshPart from, bool? flipOrientation = null, MR.Std.Const_Vector_StdVectorMREdgeId? thisContours = null, MR.Std.Const_Vector_StdVectorMREdgeId? fromContours = null, MR.Const_PartMapping? map = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_addMeshPart_5", ExactSpelling = true)]
            extern static void __MR_Mesh_addMeshPart_5(_Underlying *_this, MR.Const_MeshPart._Underlying *from, byte *flipOrientation, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *thisContours, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *fromContours, MR.PartMapping._Underlying *map);
            byte __deref_flipOrientation = flipOrientation.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Mesh_addMeshPart_5(_UnderlyingPtr, from._UnderlyingPtr, flipOrientation.HasValue ? &__deref_flipOrientation : null, thisContours is not null ? thisContours._UnderlyingPtr : null, fromContours is not null ? fromContours._UnderlyingPtr : null, map is not null ? map._UnderlyingPtr : null);
        }

        /// tightly packs all arrays eliminating lone edges and invalid faces, vertices and points
        /// Generated from method `MR::Mesh::pack`.
        /// Parameter `map` defaults to `{}`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void Pack(MR.Const_PartMapping? map = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_pack_2_MR_PartMapping", ExactSpelling = true)]
            extern static void __MR_Mesh_pack_2_MR_PartMapping(_Underlying *_this, MR.Const_PartMapping._Underlying *map, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Mesh_pack_2_MR_PartMapping(_UnderlyingPtr, map is not null ? map._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// Generated from method `MR::Mesh::pack`.
        /// Parameter `rearrangeTriangles` defaults to `false`.
        public unsafe void Pack(MR.FaceMap? outFmap, MR.VertMap? outVmap = null, MR.WholeEdgeMap? outEmap = null, bool? rearrangeTriangles = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_pack_4", ExactSpelling = true)]
            extern static void __MR_Mesh_pack_4(_Underlying *_this, MR.FaceMap._Underlying *outFmap, MR.VertMap._Underlying *outVmap, MR.WholeEdgeMap._Underlying *outEmap, byte *rearrangeTriangles);
            byte __deref_rearrangeTriangles = rearrangeTriangles.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Mesh_pack_4(_UnderlyingPtr, outFmap is not null ? outFmap._UnderlyingPtr : null, outVmap is not null ? outVmap._UnderlyingPtr : null, outEmap is not null ? outEmap._UnderlyingPtr : null, rearrangeTriangles.HasValue ? &__deref_rearrangeTriangles : null);
        }

        /// tightly packs all arrays eliminating lone edges and invalid faces, vertices and points,
        /// reorder all faces, vertices and edges according to given maps, each containing old id -> new id mapping
        /// Generated from method `MR::Mesh::pack`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> Pack(MR.Const_PackMapping map, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_pack_2_MR_PackMapping", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_Mesh_pack_2_MR_PackMapping(_Underlying *_this, MR.Const_PackMapping._Underlying *map, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_Mesh_pack_2_MR_PackMapping(_UnderlyingPtr, map._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// packs tightly and rearranges vertices, triangles and edges to put close in space elements in close indices
        /// \param preserveAABBTree whether to keep valid mesh's AABB tree after return (it will take longer to compute and it will occupy more memory)
        /// Generated from method `MR::Mesh::packOptimally`.
        /// Parameter `preserveAABBTree` defaults to `true`.
        public unsafe MR.Misc._Moved<MR.PackMapping> PackOptimally(bool? preserveAABBTree = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_packOptimally_1", ExactSpelling = true)]
            extern static MR.PackMapping._Underlying *__MR_Mesh_packOptimally_1(_Underlying *_this, byte *preserveAABBTree);
            byte __deref_preserveAABBTree = preserveAABBTree.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.PackMapping(__MR_Mesh_packOptimally_1(_UnderlyingPtr, preserveAABBTree.HasValue ? &__deref_preserveAABBTree : null), is_owning: true));
        }

        /// Generated from method `MR::Mesh::packOptimally`.
        public unsafe MR.Misc._Moved<MR.Expected_MRPackMapping_StdString> PackOptimally(bool preserveAABBTree, MR.Std._ByValue_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_packOptimally_2", ExactSpelling = true)]
            extern static MR.Expected_MRPackMapping_StdString._Underlying *__MR_Mesh_packOptimally_2(_Underlying *_this, byte preserveAABBTree, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_MRPackMapping_StdString(__MR_Mesh_packOptimally_2(_UnderlyingPtr, preserveAABBTree ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces and not in \param keepFaces
        /// Generated from method `MR::Mesh::deleteFaces`.
        public unsafe void DeleteFaces(MR.Const_FaceBitSet fs, MR.Const_UndirectedEdgeBitSet? keepEdges = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_deleteFaces", ExactSpelling = true)]
            extern static void __MR_Mesh_deleteFaces(_Underlying *_this, MR.Const_FaceBitSet._Underlying *fs, MR.Const_UndirectedEdgeBitSet._Underlying *keepEdges);
            __MR_Mesh_deleteFaces(_UnderlyingPtr, fs._UnderlyingPtr, keepEdges is not null ? keepEdges._UnderlyingPtr : null);
        }

        /// invalidates caches (aabb-trees) after any change in mesh geometry or topology
        /// \param pointsChanged specifies whether points have changed (otherwise only topology has changed)
        /// Generated from method `MR::Mesh::invalidateCaches`.
        /// Parameter `pointsChanged` defaults to `true`.
        public unsafe void InvalidateCaches(bool? pointsChanged = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_invalidateCaches", ExactSpelling = true)]
            extern static void __MR_Mesh_invalidateCaches(_Underlying *_this, byte *pointsChanged);
            byte __deref_pointsChanged = pointsChanged.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Mesh_invalidateCaches(_UnderlyingPtr, pointsChanged.HasValue ? &__deref_pointsChanged : null);
        }

        /// updates existing caches in case of few vertices were changed insignificantly,
        /// and topology remained unchanged;
        /// it shall be considered as a faster alternative to invalidateCaches() and following rebuild of trees
        /// Generated from method `MR::Mesh::updateCaches`.
        public unsafe void UpdateCaches(MR.Const_VertBitSet changedVerts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_updateCaches", ExactSpelling = true)]
            extern static void __MR_Mesh_updateCaches(_Underlying *_this, MR.Const_VertBitSet._Underlying *changedVerts);
            __MR_Mesh_updateCaches(_UnderlyingPtr, changedVerts._UnderlyingPtr);
        }

        /// requests the removal of unused capacity
        /// Generated from method `MR::Mesh::shrinkToFit`.
        public unsafe void ShrinkToFit()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_shrinkToFit", ExactSpelling = true)]
            extern static void __MR_Mesh_shrinkToFit(_Underlying *_this);
            __MR_Mesh_shrinkToFit(_UnderlyingPtr);
        }

        /// reflects the mesh from a given plane
        /// Generated from method `MR::Mesh::mirror`.
        public unsafe void Mirror(MR.Const_Plane3f plane)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Mesh_mirror", ExactSpelling = true)]
            extern static void __MR_Mesh_mirror(_Underlying *_this, MR.Const_Plane3f._Underlying *plane);
            __MR_Mesh_mirror(_UnderlyingPtr, plane._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Mesh` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Mesh`/`Const_Mesh` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Mesh
    {
        internal readonly Const_Mesh? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Mesh() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Mesh(Const_Mesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Mesh(Const_Mesh arg) {return new(arg);}
        public _ByValue_Mesh(MR.Misc._Moved<Mesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Mesh(MR.Misc._Moved<Mesh> arg) {return new(arg);}
    }

    /// This is used as a function parameter when the underlying function receives an optional `Mesh` by value,
    ///   and also has a default argument, meaning it has two different null states.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Mesh`/`Const_Mesh` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument.
    /// * Pass `MR.Misc.NullOptType` to pass no object.
    public class _ByValueOptOpt_Mesh
    {
        internal readonly Const_Mesh? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValueOptOpt_Mesh() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValueOptOpt_Mesh(Const_Mesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValueOptOpt_Mesh(Const_Mesh arg) {return new(arg);}
        public _ByValueOptOpt_Mesh(MR.Misc._Moved<Mesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValueOptOpt_Mesh(MR.Misc._Moved<Mesh> arg) {return new(arg);}
        public _ByValueOptOpt_Mesh(MR.Misc.NullOptType nullopt) {PassByMode = MR.Misc._PassBy.no_object;}
        public static implicit operator _ByValueOptOpt_Mesh(MR.Misc.NullOptType nullopt) {return new(nullopt);}
    }

    /// This is used for optional parameters of class `Mesh` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Mesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mesh`/`Const_Mesh` directly.
    public class _InOptMut_Mesh
    {
        public Mesh? Opt;

        public _InOptMut_Mesh() {}
        public _InOptMut_Mesh(Mesh value) {Opt = value;}
        public static implicit operator _InOptMut_Mesh(Mesh value) {return new(value);}
    }

    /// This is used for optional parameters of class `Mesh` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Mesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mesh`/`Const_Mesh` to pass it to the function.
    public class _InOptConst_Mesh
    {
        public Const_Mesh? Opt;

        public _InOptConst_Mesh() {}
        public _InOptConst_Mesh(Const_Mesh value) {Opt = value;}
        public static implicit operator _InOptConst_Mesh(Const_Mesh value) {return new(value);}
    }
}
