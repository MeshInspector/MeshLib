public static partial class MR
{
    /// a vertex with associated penalty metric
    /// to designate a possible start or end of edge path
    /// Generated from class `MR::TerminalVertex`.
    /// This is the const half of the class.
    public class Const_TerminalVertex : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TerminalVertex(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_Destroy", ExactSpelling = true)]
            extern static void __MR_TerminalVertex_Destroy(_Underlying *_this);
            __MR_TerminalVertex_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TerminalVertex() {Dispose(false);}

        public unsafe MR.Const_VertId V
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_Get_v", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_TerminalVertex_Get_v(_Underlying *_this);
                return new(__MR_TerminalVertex_Get_v(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_Get_metric", ExactSpelling = true)]
                extern static float *__MR_TerminalVertex_Get_metric(_Underlying *_this);
                return *__MR_TerminalVertex_Get_metric(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TerminalVertex() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_DefaultConstruct();
            _UnderlyingPtr = __MR_TerminalVertex_DefaultConstruct();
        }

        /// Constructs `MR::TerminalVertex` elementwise.
        public unsafe Const_TerminalVertex(MR.VertId v, float metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_ConstructFrom", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_ConstructFrom(MR.VertId v, float metric);
            _UnderlyingPtr = __MR_TerminalVertex_ConstructFrom(v, metric);
        }

        /// Generated from constructor `MR::TerminalVertex::TerminalVertex`.
        public unsafe Const_TerminalVertex(MR.Const_TerminalVertex _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_ConstructFromAnother(MR.TerminalVertex._Underlying *_other);
            _UnderlyingPtr = __MR_TerminalVertex_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// a vertex with associated penalty metric
    /// to designate a possible start or end of edge path
    /// Generated from class `MR::TerminalVertex`.
    /// This is the non-const half of the class.
    public class TerminalVertex : Const_TerminalVertex
    {
        internal unsafe TerminalVertex(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_VertId V
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_GetMutable_v", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_TerminalVertex_GetMutable_v(_Underlying *_this);
                return new(__MR_TerminalVertex_GetMutable_v(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_GetMutable_metric", ExactSpelling = true)]
                extern static float *__MR_TerminalVertex_GetMutable_metric(_Underlying *_this);
                return ref *__MR_TerminalVertex_GetMutable_metric(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TerminalVertex() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_DefaultConstruct();
            _UnderlyingPtr = __MR_TerminalVertex_DefaultConstruct();
        }

        /// Constructs `MR::TerminalVertex` elementwise.
        public unsafe TerminalVertex(MR.VertId v, float metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_ConstructFrom", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_ConstructFrom(MR.VertId v, float metric);
            _UnderlyingPtr = __MR_TerminalVertex_ConstructFrom(v, metric);
        }

        /// Generated from constructor `MR::TerminalVertex::TerminalVertex`.
        public unsafe TerminalVertex(MR.Const_TerminalVertex _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_ConstructFromAnother(MR.TerminalVertex._Underlying *_other);
            _UnderlyingPtr = __MR_TerminalVertex_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TerminalVertex::operator=`.
        public unsafe MR.TerminalVertex Assign(MR.Const_TerminalVertex _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TerminalVertex_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TerminalVertex._Underlying *__MR_TerminalVertex_AssignFromAnother(_Underlying *_this, MR.TerminalVertex._Underlying *_other);
            return new(__MR_TerminalVertex_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TerminalVertex` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TerminalVertex`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TerminalVertex`/`Const_TerminalVertex` directly.
    public class _InOptMut_TerminalVertex
    {
        public TerminalVertex? Opt;

        public _InOptMut_TerminalVertex() {}
        public _InOptMut_TerminalVertex(TerminalVertex value) {Opt = value;}
        public static implicit operator _InOptMut_TerminalVertex(TerminalVertex value) {return new(value);}
    }

    /// This is used for optional parameters of class `TerminalVertex` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TerminalVertex`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TerminalVertex`/`Const_TerminalVertex` to pass it to the function.
    public class _InOptConst_TerminalVertex
    {
        public Const_TerminalVertex? Opt;

        public _InOptConst_TerminalVertex() {}
        public _InOptConst_TerminalVertex(Const_TerminalVertex value) {Opt = value;}
        public static implicit operator _InOptConst_TerminalVertex(Const_TerminalVertex value) {return new(value);}
    }

    /// returns true if every next edge starts where previous edge ends
    /// Generated from function `MR::isEdgePath`.
    public static unsafe bool IsEdgePath(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isEdgePath", ExactSpelling = true)]
        extern static byte __MR_isEdgePath(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *edges);
        return __MR_isEdgePath(topology._UnderlyingPtr, edges._UnderlyingPtr) != 0;
    }

    /// returns true if every next edge starts where previous edge ends, and start vertex coincides with finish vertex
    /// Generated from function `MR::isEdgeLoop`.
    public static unsafe bool IsEdgeLoop(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isEdgeLoop", ExactSpelling = true)]
        extern static byte __MR_isEdgeLoop(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *edges);
        return __MR_isEdgeLoop(topology._UnderlyingPtr, edges._UnderlyingPtr) != 0;
    }

    /// given a number of edge loops, splits every loop that passes via a vertex more than once on smaller loops without self-intersections
    /// Generated from function `MR::splitOnSimpleLoops`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> SplitOnSimpleLoops(MR.Const_MeshTopology topology, MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> loops)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_splitOnSimpleLoops", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_splitOnSimpleLoops(MR.Const_MeshTopology._Underlying *topology, MR.Std.Vector_StdVectorMREdgeId._Underlying *loops);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_splitOnSimpleLoops(topology._UnderlyingPtr, loops.Value._UnderlyingPtr), is_owning: true));
    }

    /// reverses the order of edges and flips each edge orientation, thus
    /// making the opposite directed edge path
    /// Generated from function `MR::reverse`.
    public static unsafe void Reverse(MR.Std.Vector_MREdgeId path)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_reverse_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static void __MR_reverse_std_vector_MR_EdgeId(MR.Std.Vector_MREdgeId._Underlying *path);
        __MR_reverse_std_vector_MR_EdgeId(path._UnderlyingPtr);
    }

    /// reverse every path in the vector
    /// Generated from function `MR::reverse`.
    public static unsafe void Reverse(MR.Std.Vector_StdVectorMREdgeId paths)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_reverse_std_vector_std_vector_MR_EdgeId", ExactSpelling = true)]
        extern static void __MR_reverse_std_vector_std_vector_MR_EdgeId(MR.Std.Vector_StdVectorMREdgeId._Underlying *paths);
        __MR_reverse_std_vector_std_vector_MR_EdgeId(paths._UnderlyingPtr);
    }

    /// computes summed metric of all edges in the path
    /// Generated from function `MR::calcPathMetric`.
    public static unsafe double CalcPathMetric(MR.Std.Const_Vector_MREdgeId path, MR.Std._ByValue_Function_FloatFuncFromMREdgeId metric)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcPathMetric", ExactSpelling = true)]
        extern static double __MR_calcPathMetric(MR.Std.Const_Vector_MREdgeId._Underlying *path, MR.Misc._PassBy metric_pass_by, MR.Std.Function_FloatFuncFromMREdgeId._Underlying *metric);
        return __MR_calcPathMetric(path._UnderlyingPtr, metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null);
    }

    /// Generated from function `MR::calcPathLength`.
    public static unsafe double CalcPathLength(MR.Std.Const_Vector_MREdgeId path, MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcPathLength", ExactSpelling = true)]
        extern static double __MR_calcPathLength(MR.Std.Const_Vector_MREdgeId._Underlying *path, MR.Const_Mesh._Underlying *mesh);
        return __MR_calcPathLength(path._UnderlyingPtr, mesh._UnderlyingPtr);
    }

    /// returns the vector with the magnitude equal to the area surrounded by the loop (if the loop is planar),
    /// and directed to see the loop in ccw order from the vector tip
    /// Generated from function `MR::calcOrientedArea`.
    public static unsafe MR.Vector3d CalcOrientedArea(MR.Std.Const_Vector_MREdgeId loop, MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcOrientedArea", ExactSpelling = true)]
        extern static MR.Vector3d __MR_calcOrientedArea(MR.Std.Const_Vector_MREdgeId._Underlying *loop, MR.Const_Mesh._Underlying *mesh);
        return __MR_calcOrientedArea(loop._UnderlyingPtr, mesh._UnderlyingPtr);
    }

    /// sorts given paths in ascending order of their metrics
    /// Generated from function `MR::sortPathsByMetric`.
    public static unsafe void SortPathsByMetric(MR.Std.Vector_StdVectorMREdgeId paths, MR.Std._ByValue_Function_FloatFuncFromMREdgeId metric)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sortPathsByMetric", ExactSpelling = true)]
        extern static void __MR_sortPathsByMetric(MR.Std.Vector_StdVectorMREdgeId._Underlying *paths, MR.Misc._PassBy metric_pass_by, MR.Std.Function_FloatFuncFromMREdgeId._Underlying *metric);
        __MR_sortPathsByMetric(paths._UnderlyingPtr, metric.PassByMode, metric.Value is not null ? metric.Value._UnderlyingPtr : null);
    }

    /// Generated from function `MR::sortPathsByLength`.
    public static unsafe void SortPathsByLength(MR.Std.Vector_StdVectorMREdgeId paths, MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sortPathsByLength", ExactSpelling = true)]
        extern static void __MR_sortPathsByLength(MR.Std.Vector_StdVectorMREdgeId._Underlying *paths, MR.Const_Mesh._Underlying *mesh);
        __MR_sortPathsByLength(paths._UnderlyingPtr, mesh._UnderlyingPtr);
    }

    /// adds all faces incident to loop vertices and located to the left from the loop to given FaceBitSet
    /// Generated from function `MR::addLeftBand`.
    public static unsafe void AddLeftBand(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId loop, MR.FaceBitSet addHere)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_addLeftBand", ExactSpelling = true)]
        extern static void __MR_addLeftBand(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *loop, MR.FaceBitSet._Underlying *addHere);
        __MR_addLeftBand(topology._UnderlyingPtr, loop._UnderlyingPtr, addHere._UnderlyingPtr);
    }

    /// finds the shortest path in euclidean metric from start to finish vertices using Dijkstra algorithm;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildShortestPath`.
    /// Parameter `maxPathLen` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildShortestPath(MR.Const_Mesh mesh, MR.VertId start, MR.VertId finish, float? maxPathLen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildShortestPath_MR_VertId", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildShortestPath_MR_VertId(MR.Const_Mesh._Underlying *mesh, MR.VertId start, MR.VertId finish, float *maxPathLen);
        float __deref_maxPathLen = maxPathLen.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildShortestPath_MR_VertId(mesh._UnderlyingPtr, start, finish, maxPathLen.HasValue ? &__deref_maxPathLen : null), is_owning: true));
    }

    /// finds the shortest path in euclidean metric from start to finish vertices using bidirectional modification of Dijkstra algorithm,
    /// constructing the path simultaneously from both start and finish, which is faster for long paths;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildShortestPathBiDir`.
    /// Parameter `maxPathLen` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildShortestPathBiDir(MR.Const_Mesh mesh, MR.VertId start, MR.VertId finish, float? maxPathLen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildShortestPathBiDir_4", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildShortestPathBiDir_4(MR.Const_Mesh._Underlying *mesh, MR.VertId start, MR.VertId finish, float *maxPathLen);
        float __deref_maxPathLen = maxPathLen.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildShortestPathBiDir_4(mesh._UnderlyingPtr, start, finish, maxPathLen.HasValue ? &__deref_maxPathLen : null), is_owning: true));
    }

    /// finds the path from a vertex in start-triangle to a vertex in finish-triangle,
    /// so that the length start-first_vertex-...-last_vertex-finish is shortest in euclidean metric;
    /// using bidirectional modification of Dijkstra algorithm, constructing the path simultaneously from both start and finish, which is faster for long paths;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildShortestPathBiDir`.
    /// Parameter `maxPathLen` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildShortestPathBiDir(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint finish, MR.Mut_VertId? outPathStart = null, MR.Mut_VertId? outPathFinish = null, float? maxPathLen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildShortestPathBiDir_6", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildShortestPathBiDir_6(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *finish, MR.Mut_VertId._Underlying *outPathStart, MR.Mut_VertId._Underlying *outPathFinish, float *maxPathLen);
        float __deref_maxPathLen = maxPathLen.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildShortestPathBiDir_6(mesh._UnderlyingPtr, start._UnderlyingPtr, finish._UnderlyingPtr, outPathStart is not null ? outPathStart._UnderlyingPtr : null, outPathFinish is not null ? outPathFinish._UnderlyingPtr : null, maxPathLen.HasValue ? &__deref_maxPathLen : null), is_owning: true));
    }

    /// finds the shortest path in euclidean metric from start to finish vertices using A* modification of Dijkstra algorithm,
    /// which is faster for near linear path;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildShortestPathAStar`.
    /// Parameter `maxPathLen` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildShortestPathAStar(MR.Const_Mesh mesh, MR.VertId start, MR.VertId finish, float? maxPathLen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildShortestPathAStar_4", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildShortestPathAStar_4(MR.Const_Mesh._Underlying *mesh, MR.VertId start, MR.VertId finish, float *maxPathLen);
        float __deref_maxPathLen = maxPathLen.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildShortestPathAStar_4(mesh._UnderlyingPtr, start, finish, maxPathLen.HasValue ? &__deref_maxPathLen : null), is_owning: true));
    }

    /// finds the path from a vertex in start-triangle to a vertex in finish-triangle,
    /// so that the length start-first_vertex-...-last_vertex-finish is shortest in euclidean metric;
    /// using A* modification of Dijkstra algorithm, which is faster for near linear path;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildShortestPathAStar`.
    /// Parameter `maxPathLen` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildShortestPathAStar(MR.Const_Mesh mesh, MR.Const_MeshTriPoint start, MR.Const_MeshTriPoint finish, MR.Mut_VertId? outPathStart = null, MR.Mut_VertId? outPathFinish = null, float? maxPathLen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildShortestPathAStar_6", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildShortestPathAStar_6(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *start, MR.Const_MeshTriPoint._Underlying *finish, MR.Mut_VertId._Underlying *outPathStart, MR.Mut_VertId._Underlying *outPathFinish, float *maxPathLen);
        float __deref_maxPathLen = maxPathLen.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildShortestPathAStar_6(mesh._UnderlyingPtr, start._UnderlyingPtr, finish._UnderlyingPtr, outPathStart is not null ? outPathStart._UnderlyingPtr : null, outPathFinish is not null ? outPathFinish._UnderlyingPtr : null, maxPathLen.HasValue ? &__deref_maxPathLen : null), is_owning: true));
    }

    /// builds shortest path in euclidean metric from start to finish vertices; if no path can be found then empty path is returned
    /// Generated from function `MR::buildShortestPath`.
    /// Parameter `maxPathLen` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildShortestPath(MR.Const_Mesh mesh, MR.VertId start, MR.Const_VertBitSet finish, float? maxPathLen = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildShortestPath_MR_VertBitSet", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildShortestPath_MR_VertBitSet(MR.Const_Mesh._Underlying *mesh, MR.VertId start, MR.Const_VertBitSet._Underlying *finish, float *maxPathLen);
        float __deref_maxPathLen = maxPathLen.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildShortestPath_MR_VertBitSet(mesh._UnderlyingPtr, start, finish._UnderlyingPtr, maxPathLen.HasValue ? &__deref_maxPathLen : null), is_owning: true));
    }

    /// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
    /// Generated from function `MR::buildSmallestMetricPath`.
    /// Parameter `maxPathMetric` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildSmallestMetricPath(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.VertId start, MR.VertId finish, float? maxPathMetric = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildSmallestMetricPath_const_MR_MeshTopology_ref_MR_VertId", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildSmallestMetricPath_const_MR_MeshTopology_ref_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.VertId start, MR.VertId finish, float *maxPathMetric);
        float __deref_maxPathMetric = maxPathMetric.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildSmallestMetricPath_const_MR_MeshTopology_ref_MR_VertId(topology._UnderlyingPtr, metric._UnderlyingPtr, start, finish, maxPathMetric.HasValue ? &__deref_maxPathMetric : null), is_owning: true));
    }

    /// finds the smallest metric path from start vertex to finish vertex,
    /// using bidirectional modification of Dijkstra algorithm, constructing the path simultaneously from both start and finish, which is faster for long paths;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildSmallestMetricPathBiDir`.
    /// Parameter `maxPathMetric` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildSmallestMetricPathBiDir(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.VertId start, MR.VertId finish, float? maxPathMetric = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildSmallestMetricPathBiDir_5", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildSmallestMetricPathBiDir_5(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.VertId start, MR.VertId finish, float *maxPathMetric);
        float __deref_maxPathMetric = maxPathMetric.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildSmallestMetricPathBiDir_5(topology._UnderlyingPtr, metric._UnderlyingPtr, start, finish, maxPathMetric.HasValue ? &__deref_maxPathMetric : null), is_owning: true));
    }

    /// finds the smallest metric path from one of start vertices to one of the finish vertices,
    /// using bidirectional modification of Dijkstra algorithm, constructing the path simultaneously from both starts and finishes, which is faster for long paths;
    /// if no path can be found then empty path is returned
    /// Generated from function `MR::buildSmallestMetricPathBiDir`.
    /// Parameter `maxPathMetric` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildSmallestMetricPathBiDir(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.Const_TerminalVertex? starts, int numStarts, MR.Const_TerminalVertex? finishes, int numFinishes, MR.Mut_VertId? outPathStart = null, MR.Mut_VertId? outPathFinish = null, float? maxPathMetric = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildSmallestMetricPathBiDir_9", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildSmallestMetricPathBiDir_9(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.Const_TerminalVertex._Underlying *starts, int numStarts, MR.Const_TerminalVertex._Underlying *finishes, int numFinishes, MR.Mut_VertId._Underlying *outPathStart, MR.Mut_VertId._Underlying *outPathFinish, float *maxPathMetric);
        float __deref_maxPathMetric = maxPathMetric.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildSmallestMetricPathBiDir_9(topology._UnderlyingPtr, metric._UnderlyingPtr, starts is not null ? starts._UnderlyingPtr : null, numStarts, finishes is not null ? finishes._UnderlyingPtr : null, numFinishes, outPathStart is not null ? outPathStart._UnderlyingPtr : null, outPathFinish is not null ? outPathFinish._UnderlyingPtr : null, maxPathMetric.HasValue ? &__deref_maxPathMetric : null), is_owning: true));
    }

    /// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
    /// Generated from function `MR::buildSmallestMetricPath`.
    /// Parameter `maxPathMetric` defaults to `3.40282347e38f`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> BuildSmallestMetricPath(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.VertId start, MR.Const_VertBitSet finish, float? maxPathMetric = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_buildSmallestMetricPath_const_MR_MeshTopology_ref_MR_VertBitSet", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_buildSmallestMetricPath_const_MR_MeshTopology_ref_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.VertId start, MR.Const_VertBitSet._Underlying *finish, float *maxPathMetric);
        float __deref_maxPathMetric = maxPathMetric.GetValueOrDefault();
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_buildSmallestMetricPath_const_MR_MeshTopology_ref_MR_VertBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, start, finish._UnderlyingPtr, maxPathMetric.HasValue ? &__deref_maxPathMetric : null), is_owning: true));
    }

    /// returns all vertices from given region ordered in each connected component in breadth-first way
    /// Generated from function `MR::getVertexOrdering`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVertId> GetVertexOrdering(MR.Const_MeshTopology topology, MR._ByValue_VertBitSet region)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVertexOrdering_MR_MeshTopology", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVertId._Underlying *__MR_getVertexOrdering_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology, MR.Misc._PassBy region_pass_by, MR.VertBitSet._Underlying *region);
        return MR.Misc.Move(new MR.Std.Vector_MRVertId(__MR_getVertexOrdering_MR_MeshTopology(topology._UnderlyingPtr, region.PassByMode, region.Value is not null ? region.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// finds all closed loops from given edges and removes them from edges
    /// Generated from function `MR::extractClosedLoops`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> ExtractClosedLoops(MR.Const_MeshTopology topology, MR.EdgeBitSet edges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractClosedLoops_2", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_extractClosedLoops_2(MR.Const_MeshTopology._Underlying *topology, MR.EdgeBitSet._Underlying *edges);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_extractClosedLoops_2(topology._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
    }

    /// Generated from function `MR::extractClosedLoops`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMREdgeId> ExtractClosedLoops(MR.Const_MeshTopology topology, MR.Std.Const_Vector_MREdgeId inEdges, MR.EdgeBitSet? outNotLoopEdges = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractClosedLoops_3", ExactSpelling = true)]
        extern static MR.Std.Vector_StdVectorMREdgeId._Underlying *__MR_extractClosedLoops_3(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Vector_MREdgeId._Underlying *inEdges, MR.EdgeBitSet._Underlying *outNotLoopEdges);
        return MR.Misc.Move(new MR.Std.Vector_StdVectorMREdgeId(__MR_extractClosedLoops_3(topology._UnderlyingPtr, inEdges._UnderlyingPtr, outNotLoopEdges is not null ? outNotLoopEdges._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::extractLongestClosedLoop`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> ExtractLongestClosedLoop(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId inEdges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_extractLongestClosedLoop", ExactSpelling = true)]
        extern static MR.Std.Vector_MREdgeId._Underlying *__MR_extractLongestClosedLoop(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *inEdges);
        return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_extractLongestClosedLoop(mesh._UnderlyingPtr, inEdges._UnderlyingPtr), is_owning: true));
    }

    /// expands the region (of faces or vertices) on given metric value. returns false if callback also returns false
    /// Generated from function `MR::dilateRegionByMetric`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegionByMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.FaceBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegionByMetric_MR_FaceBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegionByMetric_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.FaceBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegionByMetric_MR_FaceBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegionByMetric`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegionByMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegionByMetric_MR_VertBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegionByMetric_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegionByMetric_MR_VertBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegionByMetric`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegionByMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.UndirectedEdgeBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegionByMetric_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegionByMetric_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.UndirectedEdgeBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegionByMetric_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// shrinks the region (of faces or vertices) on given metric value. returns false if callback also returns false
    /// Generated from function `MR::erodeRegionByMetric`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegionByMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.FaceBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegionByMetric_MR_FaceBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegionByMetric_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.FaceBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegionByMetric_MR_FaceBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegionByMetric`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegionByMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegionByMetric_MR_VertBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegionByMetric_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegionByMetric_MR_VertBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegionByMetric`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegionByMetric(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric, MR.UndirectedEdgeBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegionByMetric_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegionByMetric_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric, MR.UndirectedEdgeBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegionByMetric_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, metric._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// expands the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
    /// Generated from function `MR::dilateRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_Mesh mesh, MR.FaceBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_4_MR_FaceBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_4_MR_FaceBitSet(MR.Const_Mesh._Underlying *mesh, MR.FaceBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegion_4_MR_FaceBitSet(mesh._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_Mesh mesh, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_4_MR_VertBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_4_MR_VertBitSet(MR.Const_Mesh._Underlying *mesh, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegion_4_MR_VertBitSet(mesh._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_Mesh mesh, MR.UndirectedEdgeBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_4_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_4_MR_UndirectedEdgeBitSet(MR.Const_Mesh._Underlying *mesh, MR.UndirectedEdgeBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegion_4_MR_UndirectedEdgeBitSet(mesh._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_FaceBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_FaceBitSet(topology._UnderlyingPtr, points._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_VertBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_VertBitSet(topology._UnderlyingPtr, points._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::dilateRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool DilateRegion(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static byte __MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_dilateRegion_5_const_MR_MeshTopology_ref_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, points._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// shrinks the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
    /// Generated from function `MR::erodeRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_Mesh mesh, MR.FaceBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_4_MR_FaceBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_4_MR_FaceBitSet(MR.Const_Mesh._Underlying *mesh, MR.FaceBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegion_4_MR_FaceBitSet(mesh._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_Mesh mesh, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_4_MR_VertBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_4_MR_VertBitSet(MR.Const_Mesh._Underlying *mesh, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegion_4_MR_VertBitSet(mesh._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_Mesh mesh, MR.UndirectedEdgeBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_4_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_4_MR_UndirectedEdgeBitSet(MR.Const_Mesh._Underlying *mesh, MR.UndirectedEdgeBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegion_4_MR_UndirectedEdgeBitSet(mesh._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.FaceBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_FaceBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_FaceBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.FaceBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_FaceBitSet(topology._UnderlyingPtr, points._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_VertBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_VertBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_VertBitSet(topology._UnderlyingPtr, points._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::erodeRegion`.
    /// Parameter `callback` defaults to `{}`.
    public static unsafe bool ErodeRegion(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.UndirectedEdgeBitSet region, float dilation, MR.Std._ByValue_Function_BoolFuncFromFloat? callback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
        extern static byte __MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_UndirectedEdgeBitSet(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.UndirectedEdgeBitSet._Underlying *region, float dilation, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
        return __MR_erodeRegion_5_const_MR_MeshTopology_ref_MR_UndirectedEdgeBitSet(topology._UnderlyingPtr, points._UnderlyingPtr, region._UnderlyingPtr, dilation, callback is not null ? callback.PassByMode : MR.Misc._PassBy.default_arg, callback is not null && callback.Value is not null ? callback.Value._UnderlyingPtr : null) != 0;
    }

    /// finds all intersection points between given path and plane, adds them in outIntersections and returns their number
    /// Generated from function `MR::getPathPlaneIntersections`.
    public static unsafe int GetPathPlaneIntersections(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId path, MR.Const_Plane3f plane, MR.Std.Vector_MREdgePoint? outIntersections = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPathPlaneIntersections", ExactSpelling = true)]
        extern static int __MR_getPathPlaneIntersections(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *path, MR.Const_Plane3f._Underlying *plane, MR.Std.Vector_MREdgePoint._Underlying *outIntersections);
        return __MR_getPathPlaneIntersections(mesh._UnderlyingPtr, path._UnderlyingPtr, plane._UnderlyingPtr, outIntersections is not null ? outIntersections._UnderlyingPtr : null);
    }

    /// finds all intersection points between given contour and plane, adds them in outIntersections and returns their number
    /// Generated from function `MR::getContourPlaneIntersections`.
    public static unsafe int GetContourPlaneIntersections(MR.Std.Const_Vector_MRVector3f path, MR.Const_Plane3f plane, MR.Std.Vector_MRVector3f? outIntersections = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getContourPlaneIntersections", ExactSpelling = true)]
        extern static int __MR_getContourPlaneIntersections(MR.Std.Const_Vector_MRVector3f._Underlying *path, MR.Const_Plane3f._Underlying *plane, MR.Std.Vector_MRVector3f._Underlying *outIntersections);
        return __MR_getContourPlaneIntersections(path._UnderlyingPtr, plane._UnderlyingPtr, outIntersections is not null ? outIntersections._UnderlyingPtr : null);
    }

    /// finds all path edges located in given plane with given tolerance, adds them in outInPlaneEdges and returns their number
    /// Generated from function `MR::getPathEdgesInPlane`.
    /// Parameter `tolerance` defaults to `0.0f`.
    public static unsafe int GetPathEdgesInPlane(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgeId path, MR.Const_Plane3f plane, float? tolerance = null, MR.Std.Vector_MREdgeId? outInPlaneEdges = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getPathEdgesInPlane", ExactSpelling = true)]
        extern static int __MR_getPathEdgesInPlane(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgeId._Underlying *path, MR.Const_Plane3f._Underlying *plane, float *tolerance, MR.Std.Vector_MREdgeId._Underlying *outInPlaneEdges);
        float __deref_tolerance = tolerance.GetValueOrDefault();
        return __MR_getPathEdgesInPlane(mesh._UnderlyingPtr, path._UnderlyingPtr, plane._UnderlyingPtr, tolerance.HasValue ? &__deref_tolerance : null, outInPlaneEdges is not null ? outInPlaneEdges._UnderlyingPtr : null);
    }
}
