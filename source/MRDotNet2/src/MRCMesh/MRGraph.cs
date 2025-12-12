public static partial class MR
{
    /// mathematical graph consisting from vertices and undirected edges
    /// Generated from class `MR::Graph`.
    /// This is the const half of the class.
    public class Const_Graph : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Graph(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_Destroy", ExactSpelling = true)]
            extern static void __MR_Graph_Destroy(_Underlying *_this);
            __MR_Graph_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Graph() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Graph() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Graph._Underlying *__MR_Graph_DefaultConstruct();
            _UnderlyingPtr = __MR_Graph_DefaultConstruct();
        }

        /// Generated from constructor `MR::Graph::Graph`.
        public unsafe Const_Graph(MR._ByValue_Graph _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Graph._Underlying *__MR_Graph_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Graph._Underlying *_other);
            _UnderlyingPtr = __MR_Graph_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// returns the number of vertex records, including invalid ones
        /// Generated from method `MR::Graph::vertSize`.
        public unsafe ulong VertSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_vertSize", ExactSpelling = true)]
            extern static ulong __MR_Graph_vertSize(_Underlying *_this);
            return __MR_Graph_vertSize(_UnderlyingPtr);
        }

        /// returns all valid vertices in the graph
        /// Generated from method `MR::Graph::validVerts`.
        public unsafe MR.Const_GraphVertBitSet ValidVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_validVerts", ExactSpelling = true)]
            extern static MR.Const_GraphVertBitSet._Underlying *__MR_Graph_validVerts(_Underlying *_this);
            return new(__MR_Graph_validVerts(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if given vertex is valid
        /// Generated from method `MR::Graph::valid`.
        public unsafe bool Valid(MR.GraphVertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_valid_MR_GraphVertId", ExactSpelling = true)]
            extern static byte __MR_Graph_valid_MR_GraphVertId(_Underlying *_this, MR.GraphVertId v);
            return __MR_Graph_valid_MR_GraphVertId(_UnderlyingPtr, v) != 0;
        }

        /// returns the number of edge records, including invalid ones
        /// Generated from method `MR::Graph::edgeSize`.
        public unsafe ulong EdgeSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_edgeSize", ExactSpelling = true)]
            extern static ulong __MR_Graph_edgeSize(_Underlying *_this);
            return __MR_Graph_edgeSize(_UnderlyingPtr);
        }

        /// returns all valid edges in the graph
        /// Generated from method `MR::Graph::validEdges`.
        public unsafe MR.Const_GraphEdgeBitSet ValidEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_validEdges", ExactSpelling = true)]
            extern static MR.Const_GraphEdgeBitSet._Underlying *__MR_Graph_validEdges(_Underlying *_this);
            return new(__MR_Graph_validEdges(_UnderlyingPtr), is_owning: false);
        }

        /// returns true if given edge is valid
        /// Generated from method `MR::Graph::valid`.
        public unsafe bool Valid(MR.GraphEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_valid_MR_GraphEdgeId", ExactSpelling = true)]
            extern static byte __MR_Graph_valid_MR_GraphEdgeId(_Underlying *_this, MR.GraphEdgeId e);
            return __MR_Graph_valid_MR_GraphEdgeId(_UnderlyingPtr, e) != 0;
        }

        /// returns all edges adjacent to given vertex
        /// Generated from method `MR::Graph::neighbours`.
        public unsafe MR.Std.Const_Vector_MRGraphEdgeId Neighbours(MR.GraphVertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_neighbours", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRGraphEdgeId._Underlying *__MR_Graph_neighbours(_Underlying *_this, MR.GraphVertId v);
            return new(__MR_Graph_neighbours(_UnderlyingPtr, v), is_owning: false);
        }

        /// returns the ends of given edge
        /// Generated from method `MR::Graph::ends`.
        public unsafe MR.Graph.Const_EndVertices Ends(MR.GraphEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_ends", ExactSpelling = true)]
            extern static MR.Graph.Const_EndVertices._Underlying *__MR_Graph_ends(_Underlying *_this, MR.GraphEdgeId e);
            return new(__MR_Graph_ends(_UnderlyingPtr, e), is_owning: false);
        }

        /// finds and returns edge between vertices a and b; returns invalid edge otherwise
        /// Generated from method `MR::Graph::findEdge`.
        public unsafe MR.GraphEdgeId FindEdge(MR.GraphVertId a, MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_findEdge", ExactSpelling = true)]
            extern static MR.GraphEdgeId __MR_Graph_findEdge(_Underlying *_this, MR.GraphVertId a, MR.GraphVertId b);
            return __MR_Graph_findEdge(_UnderlyingPtr, a, b);
        }

        /// returns true if the vertices a and b are neighbors
        /// Generated from method `MR::Graph::areNeighbors`.
        public unsafe bool AreNeighbors(MR.GraphVertId a, MR.GraphVertId b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_areNeighbors", ExactSpelling = true)]
            extern static byte __MR_Graph_areNeighbors(_Underlying *_this, MR.GraphVertId a, MR.GraphVertId b);
            return __MR_Graph_areNeighbors(_UnderlyingPtr, a, b) != 0;
        }

        /// verifies that all internal data structures are valid
        /// Generated from method `MR::Graph::checkValidity`.
        public unsafe bool CheckValidity()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_checkValidity", ExactSpelling = true)]
            extern static byte __MR_Graph_checkValidity(_Underlying *_this);
            return __MR_Graph_checkValidity(_UnderlyingPtr) != 0;
        }

        /// Generated from class `MR::Graph::EndVertices`.
        /// This is the const half of the class.
        public class Const_EndVertices : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_EndVertices(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_Destroy", ExactSpelling = true)]
                extern static void __MR_Graph_EndVertices_Destroy(_Underlying *_this);
                __MR_Graph_EndVertices_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_EndVertices() {Dispose(false);}

            // v0 < v1
            public unsafe MR.Const_GraphVertId V0
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_Get_v0", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_Graph_EndVertices_Get_v0(_Underlying *_this);
                    return new(__MR_Graph_EndVertices_Get_v0(_UnderlyingPtr), is_owning: false);
                }
            }

            // v0 < v1
            public unsafe MR.Const_GraphVertId V1
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_Get_v1", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_Graph_EndVertices_Get_v1(_Underlying *_this);
                    return new(__MR_Graph_EndVertices_Get_v1(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_EndVertices() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_DefaultConstruct();
                _UnderlyingPtr = __MR_Graph_EndVertices_DefaultConstruct();
            }

            /// Constructs `MR::Graph::EndVertices` elementwise.
            public unsafe Const_EndVertices(MR.GraphVertId v0, MR.GraphVertId v1) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_ConstructFrom", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_ConstructFrom(MR.GraphVertId v0, MR.GraphVertId v1);
                _UnderlyingPtr = __MR_Graph_EndVertices_ConstructFrom(v0, v1);
            }

            /// Generated from constructor `MR::Graph::EndVertices::EndVertices`.
            public unsafe Const_EndVertices(MR.Graph.Const_EndVertices _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_ConstructFromAnother(MR.Graph.EndVertices._Underlying *_other);
                _UnderlyingPtr = __MR_Graph_EndVertices_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Graph::EndVertices::otherEnd`.
            public unsafe MR.GraphVertId OtherEnd(MR.GraphVertId a)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_otherEnd", ExactSpelling = true)]
                extern static MR.GraphVertId __MR_Graph_EndVertices_otherEnd(_Underlying *_this, MR.GraphVertId a);
                return __MR_Graph_EndVertices_otherEnd(_UnderlyingPtr, a);
            }
        }

        /// Generated from class `MR::Graph::EndVertices`.
        /// This is the non-const half of the class.
        public class EndVertices : Const_EndVertices
        {
            internal unsafe EndVertices(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // v0 < v1
            public new unsafe MR.Mut_GraphVertId V0
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_GetMutable_v0", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_Graph_EndVertices_GetMutable_v0(_Underlying *_this);
                    return new(__MR_Graph_EndVertices_GetMutable_v0(_UnderlyingPtr), is_owning: false);
                }
            }

            // v0 < v1
            public new unsafe MR.Mut_GraphVertId V1
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_GetMutable_v1", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_Graph_EndVertices_GetMutable_v1(_Underlying *_this);
                    return new(__MR_Graph_EndVertices_GetMutable_v1(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe EndVertices() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_DefaultConstruct();
                _UnderlyingPtr = __MR_Graph_EndVertices_DefaultConstruct();
            }

            /// Constructs `MR::Graph::EndVertices` elementwise.
            public unsafe EndVertices(MR.GraphVertId v0, MR.GraphVertId v1) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_ConstructFrom", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_ConstructFrom(MR.GraphVertId v0, MR.GraphVertId v1);
                _UnderlyingPtr = __MR_Graph_EndVertices_ConstructFrom(v0, v1);
            }

            /// Generated from constructor `MR::Graph::EndVertices::EndVertices`.
            public unsafe EndVertices(MR.Graph.Const_EndVertices _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_ConstructFromAnother(MR.Graph.EndVertices._Underlying *_other);
                _UnderlyingPtr = __MR_Graph_EndVertices_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Graph::EndVertices::operator=`.
            public unsafe MR.Graph.EndVertices Assign(MR.Graph.Const_EndVertices _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Graph.EndVertices._Underlying *__MR_Graph_EndVertices_AssignFromAnother(_Underlying *_this, MR.Graph.EndVertices._Underlying *_other);
                return new(__MR_Graph_EndVertices_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }

            /// Generated from method `MR::Graph::EndVertices::replaceEnd`.
            public unsafe void ReplaceEnd(MR.GraphVertId what, MR.GraphVertId with)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_EndVertices_replaceEnd", ExactSpelling = true)]
                extern static void __MR_Graph_EndVertices_replaceEnd(_Underlying *_this, MR.GraphVertId what, MR.GraphVertId with);
                __MR_Graph_EndVertices_replaceEnd(_UnderlyingPtr, what, with);
            }
        }

        /// This is used for optional parameters of class `EndVertices` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_EndVertices`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `EndVertices`/`Const_EndVertices` directly.
        public class _InOptMut_EndVertices
        {
            public EndVertices? Opt;

            public _InOptMut_EndVertices() {}
            public _InOptMut_EndVertices(EndVertices value) {Opt = value;}
            public static implicit operator _InOptMut_EndVertices(EndVertices value) {return new(value);}
        }

        /// This is used for optional parameters of class `EndVertices` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_EndVertices`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `EndVertices`/`Const_EndVertices` to pass it to the function.
        public class _InOptConst_EndVertices
        {
            public Const_EndVertices? Opt;

            public _InOptConst_EndVertices() {}
            public _InOptConst_EndVertices(Const_EndVertices value) {Opt = value;}
            public static implicit operator _InOptConst_EndVertices(Const_EndVertices value) {return new(value);}
        }
    }

    /// mathematical graph consisting from vertices and undirected edges
    /// Generated from class `MR::Graph`.
    /// This is the non-const half of the class.
    public class Graph : Const_Graph
    {
        internal unsafe Graph(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Graph() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Graph._Underlying *__MR_Graph_DefaultConstruct();
            _UnderlyingPtr = __MR_Graph_DefaultConstruct();
        }

        /// Generated from constructor `MR::Graph::Graph`.
        public unsafe Graph(MR._ByValue_Graph _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Graph._Underlying *__MR_Graph_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Graph._Underlying *_other);
            _UnderlyingPtr = __MR_Graph_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Graph::operator=`.
        public unsafe MR.Graph Assign(MR._ByValue_Graph _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Graph._Underlying *__MR_Graph_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Graph._Underlying *_other);
            return new(__MR_Graph_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// constructs the graph from all valid vertices and edges
        /// Generated from method `MR::Graph::construct`.
        public unsafe void Construct(MR._ByValue_Vector_StdVectorMRGraphEdgeId_MRGraphVertId neighboursPerVertex, MR._ByValue_Vector_MRGraphEndVertices_MRGraphEdgeId endsPerEdge)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_construct", ExactSpelling = true)]
            extern static void __MR_Graph_construct(_Underlying *_this, MR.Misc._PassBy neighboursPerVertex_pass_by, MR.Vector_StdVectorMRGraphEdgeId_MRGraphVertId._Underlying *neighboursPerVertex, MR.Misc._PassBy endsPerEdge_pass_by, MR.Vector_MRGraphEndVertices_MRGraphEdgeId._Underlying *endsPerEdge);
            __MR_Graph_construct(_UnderlyingPtr, neighboursPerVertex.PassByMode, neighboursPerVertex.Value is not null ? neighboursPerVertex.Value._UnderlyingPtr : null, endsPerEdge.PassByMode, endsPerEdge.Value is not null ? endsPerEdge.Value._UnderlyingPtr : null);
        }

        /// unites two vertices into one (deleting the second one),
        /// which leads to deletion and merge of some edges, for which callback is called
        /// Generated from method `MR::Graph::merge`.
        public unsafe void Merge(MR.GraphVertId remnant, MR.GraphVertId dead, MR.Std._ByValue_Function_VoidFuncFromMRGraphEdgeIdMRGraphEdgeId onMergeEdges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Graph_merge", ExactSpelling = true)]
            extern static void __MR_Graph_merge(_Underlying *_this, MR.GraphVertId remnant, MR.GraphVertId dead, MR.Misc._PassBy onMergeEdges_pass_by, MR.Std.Function_VoidFuncFromMRGraphEdgeIdMRGraphEdgeId._Underlying *onMergeEdges);
            __MR_Graph_merge(_UnderlyingPtr, remnant, dead, onMergeEdges.PassByMode, onMergeEdges.Value is not null ? onMergeEdges.Value._UnderlyingPtr : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Graph` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Graph`/`Const_Graph` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Graph
    {
        internal readonly Const_Graph? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Graph() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Graph(Const_Graph new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Graph(Const_Graph arg) {return new(arg);}
        public _ByValue_Graph(MR.Misc._Moved<Graph> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Graph(MR.Misc._Moved<Graph> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Graph` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Graph`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Graph`/`Const_Graph` directly.
    public class _InOptMut_Graph
    {
        public Graph? Opt;

        public _InOptMut_Graph() {}
        public _InOptMut_Graph(Graph value) {Opt = value;}
        public static implicit operator _InOptMut_Graph(Graph value) {return new(value);}
    }

    /// This is used for optional parameters of class `Graph` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Graph`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Graph`/`Const_Graph` to pass it to the function.
    public class _InOptConst_Graph
    {
        public Const_Graph? Opt;

        public _InOptConst_Graph() {}
        public _InOptConst_Graph(Const_Graph value) {Opt = value;}
        public static implicit operator _InOptConst_Graph(Const_Graph value) {return new(value);}
    }
}
