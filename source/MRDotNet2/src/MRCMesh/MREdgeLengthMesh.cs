public static partial class MR
{
    /// Unlike the classic mesh that stores coordinates of its vertices, this class
    /// stores the lengths of all edges. It can be used for construction of intrinsic Intrinsic Delaunay Triangulations.
    /// Generated from class `MR::EdgeLengthMesh`.
    /// This is the const half of the class.
    public class Const_EdgeLengthMesh : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgeLengthMesh(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgeLengthMesh_Destroy(_Underlying *_this);
            __MR_EdgeLengthMesh_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgeLengthMesh() {Dispose(false);}

        public unsafe MR.Const_MeshTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_Get_topology", ExactSpelling = true)]
                extern static MR.Const_MeshTopology._Underlying *__MR_EdgeLengthMesh_Get_topology(_Underlying *_this);
                return new(__MR_EdgeLengthMesh_Get_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_UndirectedEdgeScalars EdgeLengths
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_Get_edgeLengths", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeScalars._Underlying *__MR_EdgeLengthMesh_Get_edgeLengths(_Underlying *_this);
                return new(__MR_EdgeLengthMesh_Get_edgeLengths(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgeLengthMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeLengthMesh_DefaultConstruct();
        }

        /// Constructs `MR::EdgeLengthMesh` elementwise.
        public unsafe Const_EdgeLengthMesh(MR._ByValue_MeshTopology topology, MR._ByValue_UndirectedEdgeScalars edgeLengths) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_ConstructFrom", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_ConstructFrom(MR.Misc._PassBy topology_pass_by, MR.MeshTopology._Underlying *topology, MR.Misc._PassBy edgeLengths_pass_by, MR.UndirectedEdgeScalars._Underlying *edgeLengths);
            _UnderlyingPtr = __MR_EdgeLengthMesh_ConstructFrom(topology.PassByMode, topology.Value is not null ? topology.Value._UnderlyingPtr : null, edgeLengths.PassByMode, edgeLengths.Value is not null ? edgeLengths.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgeLengthMesh::EdgeLengthMesh`.
        public unsafe Const_EdgeLengthMesh(MR._ByValue_EdgeLengthMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgeLengthMesh._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeLengthMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// construct EdgeLengthMesh from an ordinary mesh
        /// Generated from method `MR::EdgeLengthMesh::fromMesh`.
        public static unsafe MR.Misc._Moved<MR.EdgeLengthMesh> FromMesh(MR.Const_Mesh mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_fromMesh", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_fromMesh(MR.Const_Mesh._Underlying *mesh);
            return MR.Misc.Move(new MR.EdgeLengthMesh(__MR_EdgeLengthMesh_fromMesh(mesh._UnderlyingPtr), is_owning: true));
        }

        /// computes cotangent of the angle in the left( e ) triangle opposite to e,
        /// and returns 0 if left face does not exist
        /// Generated from method `MR::EdgeLengthMesh::leftCotan`.
        public unsafe float LeftCotan(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_leftCotan", ExactSpelling = true)]
            extern static float __MR_EdgeLengthMesh_leftCotan(_Underlying *_this, MR.EdgeId e);
            return __MR_EdgeLengthMesh_leftCotan(_UnderlyingPtr, e);
        }

        /// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
        /// consider cotangents zero for not existing triangles
        /// Generated from method `MR::EdgeLengthMesh::cotan`.
        public unsafe float Cotan(MR.UndirectedEdgeId ue)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_cotan", ExactSpelling = true)]
            extern static float __MR_EdgeLengthMesh_cotan(_Underlying *_this, MR.UndirectedEdgeId ue);
            return __MR_EdgeLengthMesh_cotan(_UnderlyingPtr, ue);
        }

        /// returns true if given edge satisfies Delaunay conditions,
        /// returns false if the edge needs to be flipped to satisfy Delaunay conditions,
        /// passing negative threshold makes more edges satisfy Delaunay conditions
        /// Generated from method `MR::EdgeLengthMesh::isDelone`.
        /// Parameter `threshold` defaults to `0`.
        public unsafe bool IsDelone(MR.UndirectedEdgeId ue, float? threshold = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_isDelone", ExactSpelling = true)]
            extern static byte __MR_EdgeLengthMesh_isDelone(_Underlying *_this, MR.UndirectedEdgeId ue, float *threshold);
            float __deref_threshold = threshold.GetValueOrDefault();
            return __MR_EdgeLengthMesh_isDelone(_UnderlyingPtr, ue, threshold.HasValue ? &__deref_threshold : null) != 0;
        }

        /// given the edge with left and right triangular faces, which form together a quadrangle,
        /// returns the length of geodesic line on original mesh between the vertices of the quadrangle opposite to given edge;
        /// returns std::nullopt if the geodesic line does not go fully inside the quadrangle
        /// Generated from method `MR::EdgeLengthMesh::edgeLengthAfterFlip`.
        public unsafe MR.Std.Optional_Float EdgeLengthAfterFlip(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_edgeLengthAfterFlip", ExactSpelling = true)]
            extern static MR.Std.Optional_Float._Underlying *__MR_EdgeLengthMesh_edgeLengthAfterFlip(_Underlying *_this, MR.EdgeId e);
            return new(__MR_EdgeLengthMesh_edgeLengthAfterFlip(_UnderlyingPtr, e), is_owning: true);
        }
    }

    /// Unlike the classic mesh that stores coordinates of its vertices, this class
    /// stores the lengths of all edges. It can be used for construction of intrinsic Intrinsic Delaunay Triangulations.
    /// Generated from class `MR::EdgeLengthMesh`.
    /// This is the non-const half of the class.
    public class EdgeLengthMesh : Const_EdgeLengthMesh
    {
        internal unsafe EdgeLengthMesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.MeshTopology Topology
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_GetMutable_topology", ExactSpelling = true)]
                extern static MR.MeshTopology._Underlying *__MR_EdgeLengthMesh_GetMutable_topology(_Underlying *_this);
                return new(__MR_EdgeLengthMesh_GetMutable_topology(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.UndirectedEdgeScalars EdgeLengths
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_GetMutable_edgeLengths", ExactSpelling = true)]
                extern static MR.UndirectedEdgeScalars._Underlying *__MR_EdgeLengthMesh_GetMutable_edgeLengths(_Underlying *_this);
                return new(__MR_EdgeLengthMesh_GetMutable_edgeLengths(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgeLengthMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeLengthMesh_DefaultConstruct();
        }

        /// Constructs `MR::EdgeLengthMesh` elementwise.
        public unsafe EdgeLengthMesh(MR._ByValue_MeshTopology topology, MR._ByValue_UndirectedEdgeScalars edgeLengths) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_ConstructFrom", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_ConstructFrom(MR.Misc._PassBy topology_pass_by, MR.MeshTopology._Underlying *topology, MR.Misc._PassBy edgeLengths_pass_by, MR.UndirectedEdgeScalars._Underlying *edgeLengths);
            _UnderlyingPtr = __MR_EdgeLengthMesh_ConstructFrom(topology.PassByMode, topology.Value is not null ? topology.Value._UnderlyingPtr : null, edgeLengths.PassByMode, edgeLengths.Value is not null ? edgeLengths.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgeLengthMesh::EdgeLengthMesh`.
        public unsafe EdgeLengthMesh(MR._ByValue_EdgeLengthMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgeLengthMesh._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeLengthMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::EdgeLengthMesh::operator=`.
        public unsafe MR.EdgeLengthMesh Assign(MR._ByValue_EdgeLengthMesh _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgeLengthMesh._Underlying *__MR_EdgeLengthMesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.EdgeLengthMesh._Underlying *_other);
            return new(__MR_EdgeLengthMesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// given the edge with left and right triangular faces, which form together a quadrangle,
        /// rotates the edge counter-clockwise inside the quadrangle;
        /// the length of e becomes equal to the length of geodesic line between its new ends on original mesh;
        /// does not flip and returns false if the geodesic line does not go fully inside the quadrangle
        /// Generated from method `MR::EdgeLengthMesh::flipEdge`.
        public unsafe bool FlipEdge(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeLengthMesh_flipEdge", ExactSpelling = true)]
            extern static byte __MR_EdgeLengthMesh_flipEdge(_Underlying *_this, MR.EdgeId e);
            return __MR_EdgeLengthMesh_flipEdge(_UnderlyingPtr, e) != 0;
        }
    }

    /// This is used as a function parameter when the underlying function receives `EdgeLengthMesh` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EdgeLengthMesh`/`Const_EdgeLengthMesh` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EdgeLengthMesh
    {
        internal readonly Const_EdgeLengthMesh? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EdgeLengthMesh() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_EdgeLengthMesh(Const_EdgeLengthMesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EdgeLengthMesh(Const_EdgeLengthMesh arg) {return new(arg);}
        public _ByValue_EdgeLengthMesh(MR.Misc._Moved<EdgeLengthMesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EdgeLengthMesh(MR.Misc._Moved<EdgeLengthMesh> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EdgeLengthMesh` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgeLengthMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeLengthMesh`/`Const_EdgeLengthMesh` directly.
    public class _InOptMut_EdgeLengthMesh
    {
        public EdgeLengthMesh? Opt;

        public _InOptMut_EdgeLengthMesh() {}
        public _InOptMut_EdgeLengthMesh(EdgeLengthMesh value) {Opt = value;}
        public static implicit operator _InOptMut_EdgeLengthMesh(EdgeLengthMesh value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgeLengthMesh` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgeLengthMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeLengthMesh`/`Const_EdgeLengthMesh` to pass it to the function.
    public class _InOptConst_EdgeLengthMesh
    {
        public Const_EdgeLengthMesh? Opt;

        public _InOptConst_EdgeLengthMesh() {}
        public _InOptConst_EdgeLengthMesh(Const_EdgeLengthMesh value) {Opt = value;}
        public static implicit operator _InOptConst_EdgeLengthMesh(Const_EdgeLengthMesh value) {return new(value);}
    }
}
