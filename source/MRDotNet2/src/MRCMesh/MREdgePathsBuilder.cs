public static partial class MR
{
    /// information associated with each vertex by the paths builder
    /// Generated from class `MR::VertPathInfo`.
    /// This is the const half of the class.
    public class Const_VertPathInfo : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertPathInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_Destroy", ExactSpelling = true)]
            extern static void __MR_VertPathInfo_Destroy(_Underlying *_this);
            __MR_VertPathInfo_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertPathInfo() {Dispose(false);}

        // edge from this vertex to its predecessor in the forest
        public unsafe MR.Const_EdgeId Back
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_Get_back", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_VertPathInfo_Get_back(_Underlying *_this);
                return new(__MR_VertPathInfo_Get_back(_UnderlyingPtr), is_owning: false);
            }
        }

        // best summed metric to reach this vertex
        public unsafe float Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_Get_metric", ExactSpelling = true)]
                extern static float *__MR_VertPathInfo_Get_metric(_Underlying *_this);
                return *__MR_VertPathInfo_Get_metric(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertPathInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_VertPathInfo_DefaultConstruct();
        }

        /// Constructs `MR::VertPathInfo` elementwise.
        public unsafe Const_VertPathInfo(MR.EdgeId back, float metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_ConstructFrom", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_ConstructFrom(MR.EdgeId back, float metric);
            _UnderlyingPtr = __MR_VertPathInfo_ConstructFrom(back, metric);
        }

        /// Generated from constructor `MR::VertPathInfo::VertPathInfo`.
        public unsafe Const_VertPathInfo(MR.Const_VertPathInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_ConstructFromAnother(MR.VertPathInfo._Underlying *_other);
            _UnderlyingPtr = __MR_VertPathInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VertPathInfo::isStart`.
        public unsafe bool IsStart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_isStart", ExactSpelling = true)]
            extern static byte __MR_VertPathInfo_isStart(_Underlying *_this);
            return __MR_VertPathInfo_isStart(_UnderlyingPtr) != 0;
        }
    }

    /// information associated with each vertex by the paths builder
    /// Generated from class `MR::VertPathInfo`.
    /// This is the non-const half of the class.
    public class VertPathInfo : Const_VertPathInfo
    {
        internal unsafe VertPathInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // edge from this vertex to its predecessor in the forest
        public new unsafe MR.Mut_EdgeId Back
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_GetMutable_back", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_VertPathInfo_GetMutable_back(_Underlying *_this);
                return new(__MR_VertPathInfo_GetMutable_back(_UnderlyingPtr), is_owning: false);
            }
        }

        // best summed metric to reach this vertex
        public new unsafe ref float Metric
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_GetMutable_metric", ExactSpelling = true)]
                extern static float *__MR_VertPathInfo_GetMutable_metric(_Underlying *_this);
                return ref *__MR_VertPathInfo_GetMutable_metric(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertPathInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_VertPathInfo_DefaultConstruct();
        }

        /// Constructs `MR::VertPathInfo` elementwise.
        public unsafe VertPathInfo(MR.EdgeId back, float metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_ConstructFrom", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_ConstructFrom(MR.EdgeId back, float metric);
            _UnderlyingPtr = __MR_VertPathInfo_ConstructFrom(back, metric);
        }

        /// Generated from constructor `MR::VertPathInfo::VertPathInfo`.
        public unsafe VertPathInfo(MR.Const_VertPathInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_ConstructFromAnother(MR.VertPathInfo._Underlying *_other);
            _UnderlyingPtr = __MR_VertPathInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::VertPathInfo::operator=`.
        public unsafe MR.VertPathInfo Assign(MR.Const_VertPathInfo _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertPathInfo_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertPathInfo._Underlying *__MR_VertPathInfo_AssignFromAnother(_Underlying *_this, MR.VertPathInfo._Underlying *_other);
            return new(__MR_VertPathInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `VertPathInfo` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertPathInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertPathInfo`/`Const_VertPathInfo` directly.
    public class _InOptMut_VertPathInfo
    {
        public VertPathInfo? Opt;

        public _InOptMut_VertPathInfo() {}
        public _InOptMut_VertPathInfo(VertPathInfo value) {Opt = value;}
        public static implicit operator _InOptMut_VertPathInfo(VertPathInfo value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertPathInfo` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertPathInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertPathInfo`/`Const_VertPathInfo` to pass it to the function.
    public class _InOptConst_VertPathInfo
    {
        public Const_VertPathInfo? Opt;

        public _InOptConst_VertPathInfo() {}
        public _InOptConst_VertPathInfo(Const_VertPathInfo value) {Opt = value;}
        public static implicit operator _InOptConst_VertPathInfo(Const_VertPathInfo value) {return new(value);}
    }

    /// the class is responsible for finding smallest metric edge paths on a mesh
    /// Generated from class `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>`.
    /// This is the const half of the class.
    public class Const_EdgePathsBuilderT_MRTrivialMetricToPenalty : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgePathsBuilderT_MRTrivialMetricToPenalty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Destroy(_Underlying *_this);
            __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgePathsBuilderT_MRTrivialMetricToPenalty() {Dispose(false);}

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::EdgePathsBuilderT`.
        public unsafe Const_EdgePathsBuilderT_MRTrivialMetricToPenalty(MR._ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgePathsBuilderT_MRTrivialMetricToPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::EdgePathsBuilderT`.
        public unsafe Const_EdgePathsBuilderT_MRTrivialMetricToPenalty(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Construct", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Construct(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Construct(topology._UnderlyingPtr, metric._UnderlyingPtr);
        }

        /// returns true if further edge forest growth is impossible
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::done`.
        public unsafe bool Done()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_done", ExactSpelling = true)]
            extern static byte __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_done(_Underlying *_this);
            return __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_done(_UnderlyingPtr) != 0;
        }

        /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::doneDistance`.
        public unsafe float DoneDistance()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_doneDistance", ExactSpelling = true)]
            extern static float __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_doneDistance(_Underlying *_this);
            return __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_doneDistance(_UnderlyingPtr);
        }

        /// gives read access to the map from vertex to path to it
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::vertPathInfoMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRVertId_MRVertPathInfo VertPathInfoMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_vertPathInfoMap", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRVertId_MRVertPathInfo._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_vertPathInfoMap(_Underlying *_this);
            return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_vertPathInfoMap(_UnderlyingPtr), is_owning: false);
        }

        /// returns one element from the map (or nullptr if the element is missing)
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::getVertInfo`.
        public unsafe MR.Const_VertPathInfo? GetVertInfo(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_getVertInfo", ExactSpelling = true)]
            extern static MR.Const_VertPathInfo._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_getVertInfo(_Underlying *_this, MR.VertId v);
            var __ret = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_getVertInfo(_UnderlyingPtr, v);
            return __ret is not null ? new MR.Const_VertPathInfo(__ret, is_owning: false) : null;
        }

        /// returns the path in the forest from given vertex to one of start vertices
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::getPathBack`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> GetPathBack(MR.VertId backpathStart)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_getPathBack", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeId._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_getPathBack(_Underlying *_this, MR.VertId backpathStart);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_getPathBack(_UnderlyingPtr, backpathStart), is_owning: true));
        }

        /// information about just reached vertex (with final metric value)
        /// Generated from class `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert`.
        /// This is the const half of the class.
        public class Const_ReachedVert : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ReachedVert(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Destroy", ExactSpelling = true)]
                extern static void __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Destroy(_Underlying *_this);
                __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ReachedVert() {Dispose(false);}

            public unsafe MR.Const_VertId V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_v", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_v(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_v(_UnderlyingPtr), is_owning: false);
                }
            }

            /// edge from this vertex to its predecessor in the forest (if this vertex is not start)
            public unsafe MR.Const_EdgeId Backward
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_backward", ExactSpelling = true)]
                    extern static MR.Const_EdgeId._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_backward(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_backward(_UnderlyingPtr), is_owning: false);
                }
            }

            /// not reached vertices are ordered in priority queue by their penalty (with the smallest value on top);
            /// penalty is equal to metric in ordinary Dijkstra, or equal to (metric + target distance lower bound) in A*
            public unsafe float Penalty
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_penalty", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_penalty(_Underlying *_this);
                    return *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_penalty(_UnderlyingPtr);
                }
            }

            /// summed metric to reach this vertex
            public unsafe float Metric
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_metric", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_metric(_Underlying *_this);
                    return *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_Get_metric(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ReachedVert() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_DefaultConstruct", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_DefaultConstruct();
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_DefaultConstruct();
            }

            /// Constructs `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert` elementwise.
            public unsafe Const_ReachedVert(MR.VertId v, MR.EdgeId backward, float penalty, float metric) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFrom", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFrom(MR.VertId v, MR.EdgeId backward, float penalty, float metric);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFrom(v, backward, penalty, metric);
            }

            /// Generated from constructor `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert::ReachedVert`.
            public unsafe Const_ReachedVert(MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.Const_ReachedVert _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFromAnother(MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *_other);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// information about just reached vertex (with final metric value)
        /// Generated from class `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert`.
        /// This is the non-const half of the class.
        public class ReachedVert : Const_ReachedVert
        {
            internal unsafe ReachedVert(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_VertId V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_v", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_v(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_v(_UnderlyingPtr), is_owning: false);
                }
            }

            /// edge from this vertex to its predecessor in the forest (if this vertex is not start)
            public new unsafe MR.Mut_EdgeId Backward
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_backward", ExactSpelling = true)]
                    extern static MR.Mut_EdgeId._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_backward(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_backward(_UnderlyingPtr), is_owning: false);
                }
            }

            /// not reached vertices are ordered in priority queue by their penalty (with the smallest value on top);
            /// penalty is equal to metric in ordinary Dijkstra, or equal to (metric + target distance lower bound) in A*
            public new unsafe ref float Penalty
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_penalty", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_penalty(_Underlying *_this);
                    return ref *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_penalty(_UnderlyingPtr);
                }
            }

            /// summed metric to reach this vertex
            public new unsafe ref float Metric
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_metric", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_metric(_Underlying *_this);
                    return ref *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_GetMutable_metric(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ReachedVert() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_DefaultConstruct", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_DefaultConstruct();
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_DefaultConstruct();
            }

            /// Constructs `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert` elementwise.
            public unsafe ReachedVert(MR.VertId v, MR.EdgeId backward, float penalty, float metric) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFrom", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFrom(MR.VertId v, MR.EdgeId backward, float penalty, float metric);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFrom(v, backward, penalty, metric);
            }

            /// Generated from constructor `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert::ReachedVert`.
            public unsafe ReachedVert(MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.Const_ReachedVert _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFromAnother(MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *_other);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::ReachedVert::operator=`.
            public unsafe MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert Assign(MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.Const_ReachedVert _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_AssignFromAnother", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_AssignFromAnother(_Underlying *_this, MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *_other);
                return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ReachedVert_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ReachedVert` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ReachedVert`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ReachedVert`/`Const_ReachedVert` directly.
        public class _InOptMut_ReachedVert
        {
            public ReachedVert? Opt;

            public _InOptMut_ReachedVert() {}
            public _InOptMut_ReachedVert(ReachedVert value) {Opt = value;}
            public static implicit operator _InOptMut_ReachedVert(ReachedVert value) {return new(value);}
        }

        /// This is used for optional parameters of class `ReachedVert` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ReachedVert`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ReachedVert`/`Const_ReachedVert` to pass it to the function.
        public class _InOptConst_ReachedVert
        {
            public Const_ReachedVert? Opt;

            public _InOptConst_ReachedVert() {}
            public _InOptConst_ReachedVert(Const_ReachedVert value) {Opt = value;}
            public static implicit operator _InOptConst_ReachedVert(Const_ReachedVert value) {return new(value);}
        }
    }

    /// the class is responsible for finding smallest metric edge paths on a mesh
    /// Generated from class `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>`.
    /// This is the non-const half of the class.
    public class EdgePathsBuilderT_MRTrivialMetricToPenalty : Const_EdgePathsBuilderT_MRTrivialMetricToPenalty
    {
        internal unsafe EdgePathsBuilderT_MRTrivialMetricToPenalty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::EdgePathsBuilderT`.
        public unsafe EdgePathsBuilderT_MRTrivialMetricToPenalty(MR._ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgePathsBuilderT_MRTrivialMetricToPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::EdgePathsBuilderT`.
        public unsafe EdgePathsBuilderT_MRTrivialMetricToPenalty(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Construct", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Construct(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_Construct(topology._UnderlyingPtr, metric._UnderlyingPtr);
        }

        /// compares proposed metric with best value known for startVert;
        /// if proposed metric is smaller then adds it in the queue and returns true
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::addStart`.
        public unsafe bool AddStart(MR.VertId startVert, float startMetric)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_addStart", ExactSpelling = true)]
            extern static byte __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_addStart(_Underlying *_this, MR.VertId startVert, float startMetric);
            return __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_addStart(_UnderlyingPtr, startVert, startMetric) != 0;
        }

        /// include one more vertex in the final forest, returning vertex-info for the newly reached vertex;
        /// returns invalid VertId in v-field if no more vertices left
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::reachNext`.
        public unsafe MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert ReachNext()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_reachNext", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_reachNext(_Underlying *_this);
            return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_reachNext(_UnderlyingPtr), is_owning: true);
        }

        /// adds steps for all origin ring edges of the reached vertex;
        /// returns true if at least one step was added
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::addOrgRingSteps`.
        public unsafe bool AddOrgRingSteps(MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.Const_ReachedVert rv)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_addOrgRingSteps", ExactSpelling = true)]
            extern static byte __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_addOrgRingSteps(_Underlying *_this, MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.Const_ReachedVert._Underlying *rv);
            return __MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_addOrgRingSteps(_UnderlyingPtr, rv._UnderlyingPtr) != 0;
        }

        /// the same as reachNext() + addOrgRingSteps()
        /// Generated from method `MR::EdgePathsBuilderT<MR::TrivialMetricToPenalty>::growOneEdge`.
        public unsafe MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert GrowOneEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_growOneEdge", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRTrivialMetricToPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_growOneEdge(_Underlying *_this);
            return new(__MR_EdgePathsBuilderT_MR_TrivialMetricToPenalty_growOneEdge(_UnderlyingPtr), is_owning: true);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EdgePathsBuilderT_MRTrivialMetricToPenalty` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EdgePathsBuilderT_MRTrivialMetricToPenalty`/`Const_EdgePathsBuilderT_MRTrivialMetricToPenalty` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty
    {
        internal readonly Const_EdgePathsBuilderT_MRTrivialMetricToPenalty? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty(Const_EdgePathsBuilderT_MRTrivialMetricToPenalty new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty(Const_EdgePathsBuilderT_MRTrivialMetricToPenalty arg) {return new(arg);}
        public _ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty(MR.Misc._Moved<EdgePathsBuilderT_MRTrivialMetricToPenalty> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EdgePathsBuilderT_MRTrivialMetricToPenalty(MR.Misc._Moved<EdgePathsBuilderT_MRTrivialMetricToPenalty> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EdgePathsBuilderT_MRTrivialMetricToPenalty` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgePathsBuilderT_MRTrivialMetricToPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePathsBuilderT_MRTrivialMetricToPenalty`/`Const_EdgePathsBuilderT_MRTrivialMetricToPenalty` directly.
    public class _InOptMut_EdgePathsBuilderT_MRTrivialMetricToPenalty
    {
        public EdgePathsBuilderT_MRTrivialMetricToPenalty? Opt;

        public _InOptMut_EdgePathsBuilderT_MRTrivialMetricToPenalty() {}
        public _InOptMut_EdgePathsBuilderT_MRTrivialMetricToPenalty(EdgePathsBuilderT_MRTrivialMetricToPenalty value) {Opt = value;}
        public static implicit operator _InOptMut_EdgePathsBuilderT_MRTrivialMetricToPenalty(EdgePathsBuilderT_MRTrivialMetricToPenalty value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgePathsBuilderT_MRTrivialMetricToPenalty` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgePathsBuilderT_MRTrivialMetricToPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePathsBuilderT_MRTrivialMetricToPenalty`/`Const_EdgePathsBuilderT_MRTrivialMetricToPenalty` to pass it to the function.
    public class _InOptConst_EdgePathsBuilderT_MRTrivialMetricToPenalty
    {
        public Const_EdgePathsBuilderT_MRTrivialMetricToPenalty? Opt;

        public _InOptConst_EdgePathsBuilderT_MRTrivialMetricToPenalty() {}
        public _InOptConst_EdgePathsBuilderT_MRTrivialMetricToPenalty(Const_EdgePathsBuilderT_MRTrivialMetricToPenalty value) {Opt = value;}
        public static implicit operator _InOptConst_EdgePathsBuilderT_MRTrivialMetricToPenalty(Const_EdgePathsBuilderT_MRTrivialMetricToPenalty value) {return new(value);}
    }

    /// the class is responsible for finding smallest metric edge paths on a mesh
    /// Generated from class `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgePathsAStarBuilder`
    /// This is the const half of the class.
    public class Const_EdgePathsBuilderT_MRMetricToAStarPenalty : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgePathsBuilderT_MRMetricToAStarPenalty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Destroy(_Underlying *_this);
            __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgePathsBuilderT_MRMetricToAStarPenalty() {Dispose(false);}

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::EdgePathsBuilderT`.
        public unsafe Const_EdgePathsBuilderT_MRMetricToAStarPenalty(MR._ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::EdgePathsBuilderT`.
        public unsafe Const_EdgePathsBuilderT_MRMetricToAStarPenalty(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Construct", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Construct(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Construct(topology._UnderlyingPtr, metric._UnderlyingPtr);
        }

        /// returns true if further edge forest growth is impossible
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::done`.
        public unsafe bool Done()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_done", ExactSpelling = true)]
            extern static byte __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_done(_Underlying *_this);
            return __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_done(_UnderlyingPtr) != 0;
        }

        /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::doneDistance`.
        public unsafe float DoneDistance()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_doneDistance", ExactSpelling = true)]
            extern static float __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_doneDistance(_Underlying *_this);
            return __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_doneDistance(_UnderlyingPtr);
        }

        /// gives read access to the map from vertex to path to it
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::vertPathInfoMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRVertId_MRVertPathInfo VertPathInfoMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_vertPathInfoMap", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRVertId_MRVertPathInfo._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_vertPathInfoMap(_Underlying *_this);
            return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_vertPathInfoMap(_UnderlyingPtr), is_owning: false);
        }

        /// returns one element from the map (or nullptr if the element is missing)
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::getVertInfo`.
        public unsafe MR.Const_VertPathInfo? GetVertInfo(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_getVertInfo", ExactSpelling = true)]
            extern static MR.Const_VertPathInfo._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_getVertInfo(_Underlying *_this, MR.VertId v);
            var __ret = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_getVertInfo(_UnderlyingPtr, v);
            return __ret is not null ? new MR.Const_VertPathInfo(__ret, is_owning: false) : null;
        }

        /// returns the path in the forest from given vertex to one of start vertices
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::getPathBack`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> GetPathBack(MR.VertId backpathStart)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_getPathBack", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeId._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_getPathBack(_Underlying *_this, MR.VertId backpathStart);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_getPathBack(_UnderlyingPtr, backpathStart), is_owning: true));
        }

        /// information about just reached vertex (with final metric value)
        /// Generated from class `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert`.
        /// This is the const half of the class.
        public class Const_ReachedVert : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ReachedVert(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Destroy", ExactSpelling = true)]
                extern static void __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Destroy(_Underlying *_this);
                __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ReachedVert() {Dispose(false);}

            public unsafe MR.Const_VertId V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_v", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_v(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_v(_UnderlyingPtr), is_owning: false);
                }
            }

            /// edge from this vertex to its predecessor in the forest (if this vertex is not start)
            public unsafe MR.Const_EdgeId Backward
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_backward", ExactSpelling = true)]
                    extern static MR.Const_EdgeId._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_backward(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_backward(_UnderlyingPtr), is_owning: false);
                }
            }

            /// not reached vertices are ordered in priority queue by their penalty (with the smallest value on top);
            /// penalty is equal to metric in ordinary Dijkstra, or equal to (metric + target distance lower bound) in A*
            public unsafe float Penalty
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_penalty", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_penalty(_Underlying *_this);
                    return *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_penalty(_UnderlyingPtr);
                }
            }

            /// summed metric to reach this vertex
            public unsafe float Metric
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_metric", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_metric(_Underlying *_this);
                    return *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_Get_metric(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ReachedVert() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_DefaultConstruct", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_DefaultConstruct();
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_DefaultConstruct();
            }

            /// Constructs `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert` elementwise.
            public unsafe Const_ReachedVert(MR.VertId v, MR.EdgeId backward, float penalty, float metric) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFrom", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFrom(MR.VertId v, MR.EdgeId backward, float penalty, float metric);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFrom(v, backward, penalty, metric);
            }

            /// Generated from constructor `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert::ReachedVert`.
            public unsafe Const_ReachedVert(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFromAnother(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *_other);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// information about just reached vertex (with final metric value)
        /// Generated from class `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert`.
        /// This is the non-const half of the class.
        public class ReachedVert : Const_ReachedVert
        {
            internal unsafe ReachedVert(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_VertId V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_v", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_v(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_v(_UnderlyingPtr), is_owning: false);
                }
            }

            /// edge from this vertex to its predecessor in the forest (if this vertex is not start)
            public new unsafe MR.Mut_EdgeId Backward
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_backward", ExactSpelling = true)]
                    extern static MR.Mut_EdgeId._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_backward(_Underlying *_this);
                    return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_backward(_UnderlyingPtr), is_owning: false);
                }
            }

            /// not reached vertices are ordered in priority queue by their penalty (with the smallest value on top);
            /// penalty is equal to metric in ordinary Dijkstra, or equal to (metric + target distance lower bound) in A*
            public new unsafe ref float Penalty
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_penalty", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_penalty(_Underlying *_this);
                    return ref *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_penalty(_UnderlyingPtr);
                }
            }

            /// summed metric to reach this vertex
            public new unsafe ref float Metric
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_metric", ExactSpelling = true)]
                    extern static float *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_metric(_Underlying *_this);
                    return ref *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_GetMutable_metric(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ReachedVert() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_DefaultConstruct", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_DefaultConstruct();
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_DefaultConstruct();
            }

            /// Constructs `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert` elementwise.
            public unsafe ReachedVert(MR.VertId v, MR.EdgeId backward, float penalty, float metric) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFrom", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFrom(MR.VertId v, MR.EdgeId backward, float penalty, float metric);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFrom(v, backward, penalty, metric);
            }

            /// Generated from constructor `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert::ReachedVert`.
            public unsafe ReachedVert(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFromAnother(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *_other);
                _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::ReachedVert::operator=`.
            public unsafe MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert Assign(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_AssignFromAnother", ExactSpelling = true)]
                extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_AssignFromAnother(_Underlying *_this, MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *_other);
                return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ReachedVert_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ReachedVert` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ReachedVert`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ReachedVert`/`Const_ReachedVert` directly.
        public class _InOptMut_ReachedVert
        {
            public ReachedVert? Opt;

            public _InOptMut_ReachedVert() {}
            public _InOptMut_ReachedVert(ReachedVert value) {Opt = value;}
            public static implicit operator _InOptMut_ReachedVert(ReachedVert value) {return new(value);}
        }

        /// This is used for optional parameters of class `ReachedVert` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ReachedVert`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ReachedVert`/`Const_ReachedVert` to pass it to the function.
        public class _InOptConst_ReachedVert
        {
            public Const_ReachedVert? Opt;

            public _InOptConst_ReachedVert() {}
            public _InOptConst_ReachedVert(Const_ReachedVert value) {Opt = value;}
            public static implicit operator _InOptConst_ReachedVert(Const_ReachedVert value) {return new(value);}
        }
    }

    /// the class is responsible for finding smallest metric edge paths on a mesh
    /// Generated from class `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgePathsAStarBuilder`
    /// This is the non-const half of the class.
    public class EdgePathsBuilderT_MRMetricToAStarPenalty : Const_EdgePathsBuilderT_MRMetricToAStarPenalty
    {
        internal unsafe EdgePathsBuilderT_MRMetricToAStarPenalty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::EdgePathsBuilderT`.
        public unsafe EdgePathsBuilderT_MRMetricToAStarPenalty(MR._ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::EdgePathsBuilderT`.
        public unsafe EdgePathsBuilderT_MRMetricToAStarPenalty(MR.Const_MeshTopology topology, MR.Std.Const_Function_FloatFuncFromMREdgeId metric) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Construct", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Construct(MR.Const_MeshTopology._Underlying *topology, MR.Std.Const_Function_FloatFuncFromMREdgeId._Underlying *metric);
            _UnderlyingPtr = __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_Construct(topology._UnderlyingPtr, metric._UnderlyingPtr);
        }

        /// compares proposed metric with best value known for startVert;
        /// if proposed metric is smaller then adds it in the queue and returns true
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::addStart`.
        public unsafe bool AddStart(MR.VertId startVert, float startMetric)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_addStart", ExactSpelling = true)]
            extern static byte __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_addStart(_Underlying *_this, MR.VertId startVert, float startMetric);
            return __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_addStart(_UnderlyingPtr, startVert, startMetric) != 0;
        }

        /// include one more vertex in the final forest, returning vertex-info for the newly reached vertex;
        /// returns invalid VertId in v-field if no more vertices left
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::reachNext`.
        public unsafe MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert ReachNext()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_reachNext", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_reachNext(_Underlying *_this);
            return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_reachNext(_UnderlyingPtr), is_owning: true);
        }

        /// adds steps for all origin ring edges of the reached vertex;
        /// returns true if at least one step was added
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::addOrgRingSteps`.
        public unsafe bool AddOrgRingSteps(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert rv)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_addOrgRingSteps", ExactSpelling = true)]
            extern static byte __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_addOrgRingSteps(_Underlying *_this, MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert._Underlying *rv);
            return __MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_addOrgRingSteps(_UnderlyingPtr, rv._UnderlyingPtr) != 0;
        }

        /// the same as reachNext() + addOrgRingSteps()
        /// Generated from method `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>::growOneEdge`.
        public unsafe MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert GrowOneEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_growOneEdge", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_growOneEdge(_Underlying *_this);
            return new(__MR_EdgePathsBuilderT_MR_MetricToAStarPenalty_growOneEdge(_UnderlyingPtr), is_owning: true);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EdgePathsBuilderT_MRMetricToAStarPenalty` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EdgePathsBuilderT_MRMetricToAStarPenalty`/`Const_EdgePathsBuilderT_MRMetricToAStarPenalty` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty
    {
        internal readonly Const_EdgePathsBuilderT_MRMetricToAStarPenalty? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty(Const_EdgePathsBuilderT_MRMetricToAStarPenalty new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty(Const_EdgePathsBuilderT_MRMetricToAStarPenalty arg) {return new(arg);}
        public _ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty(MR.Misc._Moved<EdgePathsBuilderT_MRMetricToAStarPenalty> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EdgePathsBuilderT_MRMetricToAStarPenalty(MR.Misc._Moved<EdgePathsBuilderT_MRMetricToAStarPenalty> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EdgePathsBuilderT_MRMetricToAStarPenalty` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgePathsBuilderT_MRMetricToAStarPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePathsBuilderT_MRMetricToAStarPenalty`/`Const_EdgePathsBuilderT_MRMetricToAStarPenalty` directly.
    public class _InOptMut_EdgePathsBuilderT_MRMetricToAStarPenalty
    {
        public EdgePathsBuilderT_MRMetricToAStarPenalty? Opt;

        public _InOptMut_EdgePathsBuilderT_MRMetricToAStarPenalty() {}
        public _InOptMut_EdgePathsBuilderT_MRMetricToAStarPenalty(EdgePathsBuilderT_MRMetricToAStarPenalty value) {Opt = value;}
        public static implicit operator _InOptMut_EdgePathsBuilderT_MRMetricToAStarPenalty(EdgePathsBuilderT_MRMetricToAStarPenalty value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgePathsBuilderT_MRMetricToAStarPenalty` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgePathsBuilderT_MRMetricToAStarPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePathsBuilderT_MRMetricToAStarPenalty`/`Const_EdgePathsBuilderT_MRMetricToAStarPenalty` to pass it to the function.
    public class _InOptConst_EdgePathsBuilderT_MRMetricToAStarPenalty
    {
        public Const_EdgePathsBuilderT_MRMetricToAStarPenalty? Opt;

        public _InOptConst_EdgePathsBuilderT_MRMetricToAStarPenalty() {}
        public _InOptConst_EdgePathsBuilderT_MRMetricToAStarPenalty(Const_EdgePathsBuilderT_MRMetricToAStarPenalty value) {Opt = value;}
        public static implicit operator _InOptConst_EdgePathsBuilderT_MRMetricToAStarPenalty(Const_EdgePathsBuilderT_MRMetricToAStarPenalty value) {return new(value);}
    }

    /// the vertices in the queue are ordered by their metric from a start location
    /// Generated from class `MR::TrivialMetricToPenalty`.
    /// This is the const half of the class.
    public class Const_TrivialMetricToPenalty : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TrivialMetricToPenalty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_Destroy", ExactSpelling = true)]
            extern static void __MR_TrivialMetricToPenalty_Destroy(_Underlying *_this);
            __MR_TrivialMetricToPenalty_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TrivialMetricToPenalty() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TrivialMetricToPenalty() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TrivialMetricToPenalty._Underlying *__MR_TrivialMetricToPenalty_DefaultConstruct();
            _UnderlyingPtr = __MR_TrivialMetricToPenalty_DefaultConstruct();
        }

        /// Generated from constructor `MR::TrivialMetricToPenalty::TrivialMetricToPenalty`.
        public unsafe Const_TrivialMetricToPenalty(MR.Const_TrivialMetricToPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TrivialMetricToPenalty._Underlying *__MR_TrivialMetricToPenalty_ConstructFromAnother(MR.TrivialMetricToPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_TrivialMetricToPenalty_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TrivialMetricToPenalty::operator()`.
        public unsafe float Call(float metric, MR.VertId _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_call", ExactSpelling = true)]
            extern static float __MR_TrivialMetricToPenalty_call(_Underlying *_this, float metric, MR.VertId _2);
            return __MR_TrivialMetricToPenalty_call(_UnderlyingPtr, metric, _2);
        }
    }

    /// the vertices in the queue are ordered by their metric from a start location
    /// Generated from class `MR::TrivialMetricToPenalty`.
    /// This is the non-const half of the class.
    public class TrivialMetricToPenalty : Const_TrivialMetricToPenalty
    {
        internal unsafe TrivialMetricToPenalty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe TrivialMetricToPenalty() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TrivialMetricToPenalty._Underlying *__MR_TrivialMetricToPenalty_DefaultConstruct();
            _UnderlyingPtr = __MR_TrivialMetricToPenalty_DefaultConstruct();
        }

        /// Generated from constructor `MR::TrivialMetricToPenalty::TrivialMetricToPenalty`.
        public unsafe TrivialMetricToPenalty(MR.Const_TrivialMetricToPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TrivialMetricToPenalty._Underlying *__MR_TrivialMetricToPenalty_ConstructFromAnother(MR.TrivialMetricToPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_TrivialMetricToPenalty_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::TrivialMetricToPenalty::operator=`.
        public unsafe MR.TrivialMetricToPenalty Assign(MR.Const_TrivialMetricToPenalty _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TrivialMetricToPenalty_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TrivialMetricToPenalty._Underlying *__MR_TrivialMetricToPenalty_AssignFromAnother(_Underlying *_this, MR.TrivialMetricToPenalty._Underlying *_other);
            return new(__MR_TrivialMetricToPenalty_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TrivialMetricToPenalty` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TrivialMetricToPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TrivialMetricToPenalty`/`Const_TrivialMetricToPenalty` directly.
    public class _InOptMut_TrivialMetricToPenalty
    {
        public TrivialMetricToPenalty? Opt;

        public _InOptMut_TrivialMetricToPenalty() {}
        public _InOptMut_TrivialMetricToPenalty(TrivialMetricToPenalty value) {Opt = value;}
        public static implicit operator _InOptMut_TrivialMetricToPenalty(TrivialMetricToPenalty value) {return new(value);}
    }

    /// This is used for optional parameters of class `TrivialMetricToPenalty` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TrivialMetricToPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TrivialMetricToPenalty`/`Const_TrivialMetricToPenalty` to pass it to the function.
    public class _InOptConst_TrivialMetricToPenalty
    {
        public Const_TrivialMetricToPenalty? Opt;

        public _InOptConst_TrivialMetricToPenalty() {}
        public _InOptConst_TrivialMetricToPenalty(Const_TrivialMetricToPenalty value) {Opt = value;}
        public static implicit operator _InOptConst_TrivialMetricToPenalty(Const_TrivialMetricToPenalty value) {return new(value);}
    }

    /// the vertices in the queue are ordered by the sum of their metric from a start location and the
    /// lower bound of a path till target point (A* heuristic)
    /// Generated from class `MR::MetricToAStarPenalty`.
    /// This is the const half of the class.
    public class Const_MetricToAStarPenalty : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MetricToAStarPenalty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_Destroy", ExactSpelling = true)]
            extern static void __MR_MetricToAStarPenalty_Destroy(_Underlying *_this);
            __MR_MetricToAStarPenalty_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MetricToAStarPenalty() {Dispose(false);}

        public unsafe ref readonly void * Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_Get_points", ExactSpelling = true)]
                extern static void **__MR_MetricToAStarPenalty_Get_points(_Underlying *_this);
                return ref *__MR_MetricToAStarPenalty_Get_points(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Vector3f Target
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_Get_target", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MetricToAStarPenalty_Get_target(_Underlying *_this);
                return new(__MR_MetricToAStarPenalty_Get_target(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MetricToAStarPenalty() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_DefaultConstruct();
            _UnderlyingPtr = __MR_MetricToAStarPenalty_DefaultConstruct();
        }

        /// Constructs `MR::MetricToAStarPenalty` elementwise.
        public unsafe Const_MetricToAStarPenalty(MR.Const_VertCoords? points, MR.Vector3f target) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_ConstructFrom", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_ConstructFrom(MR.Const_VertCoords._Underlying *points, MR.Vector3f target);
            _UnderlyingPtr = __MR_MetricToAStarPenalty_ConstructFrom(points is not null ? points._UnderlyingPtr : null, target);
        }

        /// Generated from constructor `MR::MetricToAStarPenalty::MetricToAStarPenalty`.
        public unsafe Const_MetricToAStarPenalty(MR.Const_MetricToAStarPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_ConstructFromAnother(MR.MetricToAStarPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_MetricToAStarPenalty_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MetricToAStarPenalty::operator()`.
        public unsafe float Call(float metric, MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_call", ExactSpelling = true)]
            extern static float __MR_MetricToAStarPenalty_call(_Underlying *_this, float metric, MR.VertId v);
            return __MR_MetricToAStarPenalty_call(_UnderlyingPtr, metric, v);
        }
    }

    /// the vertices in the queue are ordered by the sum of their metric from a start location and the
    /// lower bound of a path till target point (A* heuristic)
    /// Generated from class `MR::MetricToAStarPenalty`.
    /// This is the non-const half of the class.
    public class MetricToAStarPenalty : Const_MetricToAStarPenalty
    {
        internal unsafe MetricToAStarPenalty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref readonly void * Points
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_GetMutable_points", ExactSpelling = true)]
                extern static void **__MR_MetricToAStarPenalty_GetMutable_points(_Underlying *_this);
                return ref *__MR_MetricToAStarPenalty_GetMutable_points(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Mut_Vector3f Target
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_GetMutable_target", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MetricToAStarPenalty_GetMutable_target(_Underlying *_this);
                return new(__MR_MetricToAStarPenalty_GetMutable_target(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MetricToAStarPenalty() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_DefaultConstruct();
            _UnderlyingPtr = __MR_MetricToAStarPenalty_DefaultConstruct();
        }

        /// Constructs `MR::MetricToAStarPenalty` elementwise.
        public unsafe MetricToAStarPenalty(MR.Const_VertCoords? points, MR.Vector3f target) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_ConstructFrom", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_ConstructFrom(MR.Const_VertCoords._Underlying *points, MR.Vector3f target);
            _UnderlyingPtr = __MR_MetricToAStarPenalty_ConstructFrom(points is not null ? points._UnderlyingPtr : null, target);
        }

        /// Generated from constructor `MR::MetricToAStarPenalty::MetricToAStarPenalty`.
        public unsafe MetricToAStarPenalty(MR.Const_MetricToAStarPenalty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_ConstructFromAnother(MR.MetricToAStarPenalty._Underlying *_other);
            _UnderlyingPtr = __MR_MetricToAStarPenalty_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MetricToAStarPenalty::operator=`.
        public unsafe MR.MetricToAStarPenalty Assign(MR.Const_MetricToAStarPenalty _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MetricToAStarPenalty_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MetricToAStarPenalty._Underlying *__MR_MetricToAStarPenalty_AssignFromAnother(_Underlying *_this, MR.MetricToAStarPenalty._Underlying *_other);
            return new(__MR_MetricToAStarPenalty_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MetricToAStarPenalty` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MetricToAStarPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MetricToAStarPenalty`/`Const_MetricToAStarPenalty` directly.
    public class _InOptMut_MetricToAStarPenalty
    {
        public MetricToAStarPenalty? Opt;

        public _InOptMut_MetricToAStarPenalty() {}
        public _InOptMut_MetricToAStarPenalty(MetricToAStarPenalty value) {Opt = value;}
        public static implicit operator _InOptMut_MetricToAStarPenalty(MetricToAStarPenalty value) {return new(value);}
    }

    /// This is used for optional parameters of class `MetricToAStarPenalty` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MetricToAStarPenalty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MetricToAStarPenalty`/`Const_MetricToAStarPenalty` to pass it to the function.
    public class _InOptConst_MetricToAStarPenalty
    {
        public Const_MetricToAStarPenalty? Opt;

        public _InOptConst_MetricToAStarPenalty() {}
        public _InOptConst_MetricToAStarPenalty(Const_MetricToAStarPenalty value) {Opt = value;}
        public static implicit operator _InOptConst_MetricToAStarPenalty(Const_MetricToAStarPenalty value) {return new(value);}
    }

    /// the class is responsible for finding shortest edge paths on a mesh in Euclidean metric
    /// using A* heuristics
    /// Generated from class `MR::EdgePathsAStarBuilder`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>`
    /// This is the const half of the class.
    public class Const_EdgePathsAStarBuilder : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgePathsAStarBuilder(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgePathsAStarBuilder_Destroy(_Underlying *_this);
            __MR_EdgePathsAStarBuilder_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgePathsAStarBuilder() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_EdgePathsBuilderT_MRMetricToAStarPenalty(Const_EdgePathsAStarBuilder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_UpcastTo_MR_EdgePathsBuilderT_MR_MetricToAStarPenalty", ExactSpelling = true)]
            extern static MR.Const_EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *__MR_EdgePathsAStarBuilder_UpcastTo_MR_EdgePathsBuilderT_MR_MetricToAStarPenalty(_Underlying *_this);
            MR.Const_EdgePathsBuilderT_MRMetricToAStarPenalty ret = new(__MR_EdgePathsAStarBuilder_UpcastTo_MR_EdgePathsBuilderT_MR_MetricToAStarPenalty(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::EdgePathsAStarBuilder::EdgePathsAStarBuilder`.
        public unsafe Const_EdgePathsAStarBuilder(MR._ByValue_EdgePathsAStarBuilder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePathsAStarBuilder._Underlying *__MR_EdgePathsAStarBuilder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgePathsAStarBuilder._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePathsAStarBuilder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgePathsAStarBuilder::EdgePathsAStarBuilder`.
        public unsafe Const_EdgePathsAStarBuilder(MR.Const_Mesh mesh, MR.VertId target, MR.VertId start) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_Construct_MR_VertId", ExactSpelling = true)]
            extern static MR.EdgePathsAStarBuilder._Underlying *__MR_EdgePathsAStarBuilder_Construct_MR_VertId(MR.Const_Mesh._Underlying *mesh, MR.VertId target, MR.VertId start);
            _UnderlyingPtr = __MR_EdgePathsAStarBuilder_Construct_MR_VertId(mesh._UnderlyingPtr, target, start);
        }

        /// Generated from constructor `MR::EdgePathsAStarBuilder::EdgePathsAStarBuilder`.
        public unsafe Const_EdgePathsAStarBuilder(MR.Const_Mesh mesh, MR.Const_MeshTriPoint target, MR.Const_MeshTriPoint start) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_Construct_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.EdgePathsAStarBuilder._Underlying *__MR_EdgePathsAStarBuilder_Construct_MR_MeshTriPoint(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *target, MR.Const_MeshTriPoint._Underlying *start);
            _UnderlyingPtr = __MR_EdgePathsAStarBuilder_Construct_MR_MeshTriPoint(mesh._UnderlyingPtr, target._UnderlyingPtr, start._UnderlyingPtr);
        }

        /// returns true if further edge forest growth is impossible
        /// Generated from method `MR::EdgePathsAStarBuilder::done`.
        public unsafe bool Done()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_done", ExactSpelling = true)]
            extern static byte __MR_EdgePathsAStarBuilder_done(_Underlying *_this);
            return __MR_EdgePathsAStarBuilder_done(_UnderlyingPtr) != 0;
        }

        /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
        /// Generated from method `MR::EdgePathsAStarBuilder::doneDistance`.
        public unsafe float DoneDistance()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_doneDistance", ExactSpelling = true)]
            extern static float __MR_EdgePathsAStarBuilder_doneDistance(_Underlying *_this);
            return __MR_EdgePathsAStarBuilder_doneDistance(_UnderlyingPtr);
        }

        /// gives read access to the map from vertex to path to it
        /// Generated from method `MR::EdgePathsAStarBuilder::vertPathInfoMap`.
        public unsafe MR.Phmap.Const_FlatHashMap_MRVertId_MRVertPathInfo VertPathInfoMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_vertPathInfoMap", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_MRVertId_MRVertPathInfo._Underlying *__MR_EdgePathsAStarBuilder_vertPathInfoMap(_Underlying *_this);
            return new(__MR_EdgePathsAStarBuilder_vertPathInfoMap(_UnderlyingPtr), is_owning: false);
        }

        /// returns one element from the map (or nullptr if the element is missing)
        /// Generated from method `MR::EdgePathsAStarBuilder::getVertInfo`.
        public unsafe MR.Const_VertPathInfo? GetVertInfo(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_getVertInfo", ExactSpelling = true)]
            extern static MR.Const_VertPathInfo._Underlying *__MR_EdgePathsAStarBuilder_getVertInfo(_Underlying *_this, MR.VertId v);
            var __ret = __MR_EdgePathsAStarBuilder_getVertInfo(_UnderlyingPtr, v);
            return __ret is not null ? new MR.Const_VertPathInfo(__ret, is_owning: false) : null;
        }

        /// returns the path in the forest from given vertex to one of start vertices
        /// Generated from method `MR::EdgePathsAStarBuilder::getPathBack`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeId> GetPathBack(MR.VertId backpathStart)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_getPathBack", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeId._Underlying *__MR_EdgePathsAStarBuilder_getPathBack(_Underlying *_this, MR.VertId backpathStart);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeId(__MR_EdgePathsAStarBuilder_getPathBack(_UnderlyingPtr, backpathStart), is_owning: true));
        }
    }

    /// the class is responsible for finding shortest edge paths on a mesh in Euclidean metric
    /// using A* heuristics
    /// Generated from class `MR::EdgePathsAStarBuilder`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::EdgePathsBuilderT<MR::MetricToAStarPenalty>`
    /// This is the non-const half of the class.
    public class EdgePathsAStarBuilder : Const_EdgePathsAStarBuilder
    {
        internal unsafe EdgePathsAStarBuilder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.EdgePathsBuilderT_MRMetricToAStarPenalty(EdgePathsAStarBuilder self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_UpcastTo_MR_EdgePathsBuilderT_MR_MetricToAStarPenalty", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty._Underlying *__MR_EdgePathsAStarBuilder_UpcastTo_MR_EdgePathsBuilderT_MR_MetricToAStarPenalty(_Underlying *_this);
            MR.EdgePathsBuilderT_MRMetricToAStarPenalty ret = new(__MR_EdgePathsAStarBuilder_UpcastTo_MR_EdgePathsBuilderT_MR_MetricToAStarPenalty(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::EdgePathsAStarBuilder::EdgePathsAStarBuilder`.
        public unsafe EdgePathsAStarBuilder(MR._ByValue_EdgePathsAStarBuilder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePathsAStarBuilder._Underlying *__MR_EdgePathsAStarBuilder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.EdgePathsAStarBuilder._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePathsAStarBuilder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EdgePathsAStarBuilder::EdgePathsAStarBuilder`.
        public unsafe EdgePathsAStarBuilder(MR.Const_Mesh mesh, MR.VertId target, MR.VertId start) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_Construct_MR_VertId", ExactSpelling = true)]
            extern static MR.EdgePathsAStarBuilder._Underlying *__MR_EdgePathsAStarBuilder_Construct_MR_VertId(MR.Const_Mesh._Underlying *mesh, MR.VertId target, MR.VertId start);
            _UnderlyingPtr = __MR_EdgePathsAStarBuilder_Construct_MR_VertId(mesh._UnderlyingPtr, target, start);
        }

        /// Generated from constructor `MR::EdgePathsAStarBuilder::EdgePathsAStarBuilder`.
        public unsafe EdgePathsAStarBuilder(MR.Const_Mesh mesh, MR.Const_MeshTriPoint target, MR.Const_MeshTriPoint start) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_Construct_MR_MeshTriPoint", ExactSpelling = true)]
            extern static MR.EdgePathsAStarBuilder._Underlying *__MR_EdgePathsAStarBuilder_Construct_MR_MeshTriPoint(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *target, MR.Const_MeshTriPoint._Underlying *start);
            _UnderlyingPtr = __MR_EdgePathsAStarBuilder_Construct_MR_MeshTriPoint(mesh._UnderlyingPtr, target._UnderlyingPtr, start._UnderlyingPtr);
        }

        /// compares proposed metric with best value known for startVert;
        /// if proposed metric is smaller then adds it in the queue and returns true
        /// Generated from method `MR::EdgePathsAStarBuilder::addStart`.
        public unsafe bool AddStart(MR.VertId startVert, float startMetric)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_addStart", ExactSpelling = true)]
            extern static byte __MR_EdgePathsAStarBuilder_addStart(_Underlying *_this, MR.VertId startVert, float startMetric);
            return __MR_EdgePathsAStarBuilder_addStart(_UnderlyingPtr, startVert, startMetric) != 0;
        }

        /// include one more vertex in the final forest, returning vertex-info for the newly reached vertex;
        /// returns invalid VertId in v-field if no more vertices left
        /// Generated from method `MR::EdgePathsAStarBuilder::reachNext`.
        public unsafe MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert ReachNext()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_reachNext", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsAStarBuilder_reachNext(_Underlying *_this);
            return new(__MR_EdgePathsAStarBuilder_reachNext(_UnderlyingPtr), is_owning: true);
        }

        /// adds steps for all origin ring edges of the reached vertex;
        /// returns true if at least one step was added
        /// Generated from method `MR::EdgePathsAStarBuilder::addOrgRingSteps`.
        public unsafe bool AddOrgRingSteps(MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert rv)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_addOrgRingSteps", ExactSpelling = true)]
            extern static byte __MR_EdgePathsAStarBuilder_addOrgRingSteps(_Underlying *_this, MR.EdgePathsBuilderT_MRMetricToAStarPenalty.Const_ReachedVert._Underlying *rv);
            return __MR_EdgePathsAStarBuilder_addOrgRingSteps(_UnderlyingPtr, rv._UnderlyingPtr) != 0;
        }

        /// the same as reachNext() + addOrgRingSteps()
        /// Generated from method `MR::EdgePathsAStarBuilder::growOneEdge`.
        public unsafe MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert GrowOneEdge()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePathsAStarBuilder_growOneEdge", ExactSpelling = true)]
            extern static MR.EdgePathsBuilderT_MRMetricToAStarPenalty.ReachedVert._Underlying *__MR_EdgePathsAStarBuilder_growOneEdge(_Underlying *_this);
            return new(__MR_EdgePathsAStarBuilder_growOneEdge(_UnderlyingPtr), is_owning: true);
        }
    }

    /// This is used as a function parameter when the underlying function receives `EdgePathsAStarBuilder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `EdgePathsAStarBuilder`/`Const_EdgePathsAStarBuilder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_EdgePathsAStarBuilder
    {
        internal readonly Const_EdgePathsAStarBuilder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_EdgePathsAStarBuilder(Const_EdgePathsAStarBuilder new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_EdgePathsAStarBuilder(Const_EdgePathsAStarBuilder arg) {return new(arg);}
        public _ByValue_EdgePathsAStarBuilder(MR.Misc._Moved<EdgePathsAStarBuilder> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_EdgePathsAStarBuilder(MR.Misc._Moved<EdgePathsAStarBuilder> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `EdgePathsAStarBuilder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgePathsAStarBuilder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePathsAStarBuilder`/`Const_EdgePathsAStarBuilder` directly.
    public class _InOptMut_EdgePathsAStarBuilder
    {
        public EdgePathsAStarBuilder? Opt;

        public _InOptMut_EdgePathsAStarBuilder() {}
        public _InOptMut_EdgePathsAStarBuilder(EdgePathsAStarBuilder value) {Opt = value;}
        public static implicit operator _InOptMut_EdgePathsAStarBuilder(EdgePathsAStarBuilder value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgePathsAStarBuilder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgePathsAStarBuilder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePathsAStarBuilder`/`Const_EdgePathsAStarBuilder` to pass it to the function.
    public class _InOptConst_EdgePathsAStarBuilder
    {
        public Const_EdgePathsAStarBuilder? Opt;

        public _InOptConst_EdgePathsAStarBuilder() {}
        public _InOptConst_EdgePathsAStarBuilder(Const_EdgePathsAStarBuilder value) {Opt = value;}
        public static implicit operator _InOptConst_EdgePathsAStarBuilder(Const_EdgePathsAStarBuilder value) {return new(value);}
    }
}
