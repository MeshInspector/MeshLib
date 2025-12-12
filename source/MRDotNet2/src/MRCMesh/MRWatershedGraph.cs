public static partial class MR
{
    /// graphs representing rain basins on the mesh
    /// Generated from class `MR::WatershedGraph`.
    /// This is the const half of the class.
    public class Const_WatershedGraph : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_WatershedGraph(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_Destroy", ExactSpelling = true)]
            extern static void __MR_WatershedGraph_Destroy(_Underlying *_this);
            __MR_WatershedGraph_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_WatershedGraph() {Dispose(false);}

        /// Generated from constructor `MR::WatershedGraph::WatershedGraph`.
        public unsafe Const_WatershedGraph(MR._ByValue_WatershedGraph _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WatershedGraph._Underlying *__MR_WatershedGraph_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WatershedGraph._Underlying *_other);
            _UnderlyingPtr = __MR_WatershedGraph_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// constructs the graph from given mesh, heights in z-coordinate, and initial subdivision on basins
        /// Generated from constructor `MR::WatershedGraph::WatershedGraph`.
        public unsafe Const_WatershedGraph(MR.Const_Mesh mesh, MR.Const_Vector_Int_MRFaceId face2basin, int numBasins) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_Construct", ExactSpelling = true)]
            extern static MR.WatershedGraph._Underlying *__MR_WatershedGraph_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector_Int_MRFaceId._Underlying *face2basin, int numBasins);
            _UnderlyingPtr = __MR_WatershedGraph_Construct(mesh._UnderlyingPtr, face2basin._UnderlyingPtr, numBasins);
        }

        /// returns height at given vertex or FLT_MAX if the vertex is invalid
        /// Generated from method `MR::WatershedGraph::getHeightAt`.
        public unsafe float GetHeightAt(MR.VertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getHeightAt", ExactSpelling = true)]
            extern static float __MR_WatershedGraph_getHeightAt(_Underlying *_this, MR.VertId v);
            return __MR_WatershedGraph_getHeightAt(_UnderlyingPtr, v);
        }

        /// returns underlying graph where each basin is a vertex
        /// Generated from method `MR::WatershedGraph::graph`.
        public unsafe MR.Const_Graph Graph()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_graph", ExactSpelling = true)]
            extern static MR.Const_Graph._Underlying *__MR_WatershedGraph_graph(_Underlying *_this);
            return new(__MR_WatershedGraph_graph(_UnderlyingPtr), is_owning: false);
        }

        /// returns total precipitation area
        /// Generated from method `MR::WatershedGraph::totalArea`.
        public unsafe float TotalArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_totalArea", ExactSpelling = true)]
            extern static float __MR_WatershedGraph_totalArea(_Underlying *_this);
            return __MR_WatershedGraph_totalArea(_UnderlyingPtr);
        }

        /// returns the current number of basins (excluding special "outside" basin)
        /// Generated from method `MR::WatershedGraph::numBasins`.
        public unsafe int NumBasins()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_numBasins", ExactSpelling = true)]
            extern static int __MR_WatershedGraph_numBasins(_Underlying *_this);
            return __MR_WatershedGraph_numBasins(_UnderlyingPtr);
        }

        /// returns data associated with given basin
        /// Generated from method `MR::WatershedGraph::basinInfo`.
        public unsafe MR.WatershedGraph.Const_BasinInfo BasinInfo_(MR.GraphVertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_basinInfo_const", ExactSpelling = true)]
            extern static MR.WatershedGraph.Const_BasinInfo._Underlying *__MR_WatershedGraph_basinInfo_const(_Underlying *_this, MR.GraphVertId v);
            return new(__MR_WatershedGraph_basinInfo_const(_UnderlyingPtr, v), is_owning: false);
        }

        /// returns data associated with given boundary between basins
        /// Generated from method `MR::WatershedGraph::bdInfo`.
        public unsafe MR.WatershedGraph.Const_BdInfo BdInfo_(MR.GraphEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_bdInfo_const", ExactSpelling = true)]
            extern static MR.WatershedGraph.Const_BdInfo._Underlying *__MR_WatershedGraph_bdInfo_const(_Underlying *_this, MR.GraphEdgeId e);
            return new(__MR_WatershedGraph_bdInfo_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// returns special "basin" representing outside areas of the mesh
        /// Generated from method `MR::WatershedGraph::outsideId`.
        public unsafe MR.GraphVertId OutsideId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_outsideId", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_WatershedGraph_outsideId(_Underlying *_this);
            return __MR_WatershedGraph_outsideId(_UnderlyingPtr);
        }

        /// for valid basin returns self id; for invalid basin returns the id of basin it was merged in
        /// Generated from method `MR::WatershedGraph::getRootBasin`.
        public unsafe MR.GraphVertId GetRootBasin(MR.GraphVertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getRootBasin", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_WatershedGraph_getRootBasin(_Underlying *_this, MR.GraphVertId v);
            return __MR_WatershedGraph_getRootBasin(_UnderlyingPtr, v);
        }

        /// returns the basin where the flow from this basin goes next (it can be self id if the basin is not full yet)
        /// Generated from method `MR::WatershedGraph::flowsTo`.
        public unsafe MR.GraphVertId FlowsTo(MR.GraphVertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_flowsTo", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_WatershedGraph_flowsTo(_Underlying *_this, MR.GraphVertId v);
            return __MR_WatershedGraph_flowsTo(_UnderlyingPtr, v);
        }

        /// returns the basin where the flow from this basin finally goes (it can be self id if the basin is not full yet);
        /// \param exceptOutside if true then the method returns the basin that receives water flow from (v) just before outside
        /// Generated from method `MR::WatershedGraph::flowsFinallyTo`.
        /// Parameter `exceptOutside` defaults to `false`.
        public unsafe MR.GraphVertId FlowsFinallyTo(MR.GraphVertId v, bool? exceptOutside = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_flowsFinallyTo", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_WatershedGraph_flowsFinallyTo(_Underlying *_this, MR.GraphVertId v, byte *exceptOutside);
            byte __deref_exceptOutside = exceptOutside.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_WatershedGraph_flowsFinallyTo(_UnderlyingPtr, v, exceptOutside.HasValue ? &__deref_exceptOutside : null);
        }

        /// finds the lowest boundary between basins and its height, which is defined
        /// as the minimal different between lowest boundary point and lowest point in a basin
        /// Generated from method `MR::WatershedGraph::findLowestBd`.
        public unsafe MR.Std.Pair_MRGraphEdgeId_Float FindLowestBd()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_findLowestBd", ExactSpelling = true)]
            extern static MR.Std.Pair_MRGraphEdgeId_Float._Underlying *__MR_WatershedGraph_findLowestBd(_Underlying *_this);
            return new(__MR_WatershedGraph_findLowestBd(_UnderlyingPtr), is_owning: true);
        }

        /// returns the mesh faces of given basin
        /// Generated from method `MR::WatershedGraph::getBasinFaces`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetBasinFaces(MR.GraphVertId basin)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getBasinFaces", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_WatershedGraph_getBasinFaces(_Underlying *_this, MR.GraphVertId basin);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_WatershedGraph_getBasinFaces(_UnderlyingPtr, basin), is_owning: true));
        }

        /// returns the mesh faces of each valid basin;
        /// \param joinOverflowBasins if true then overflowing basins will be merged in the target basins (except for overflow in outside)
        /// Generated from method `MR::WatershedGraph::getAllBasinFaces`.
        /// Parameter `joinOverflowBasins` defaults to `false`.
        public unsafe MR.Misc._Moved<MR.Vector_MRFaceBitSet_MRGraphVertId> GetAllBasinFaces(bool? joinOverflowBasins = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getAllBasinFaces", ExactSpelling = true)]
            extern static MR.Vector_MRFaceBitSet_MRGraphVertId._Underlying *__MR_WatershedGraph_getAllBasinFaces(_Underlying *_this, byte *joinOverflowBasins);
            byte __deref_joinOverflowBasins = joinOverflowBasins.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.Vector_MRFaceBitSet_MRGraphVertId(__MR_WatershedGraph_getAllBasinFaces(_UnderlyingPtr, joinOverflowBasins.HasValue ? &__deref_joinOverflowBasins : null), is_owning: true));
        }

        /// returns the mesh faces of given basin with at least one vertex below given level
        /// Generated from method `MR::WatershedGraph::getBasinFacesBelowLevel`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetBasinFacesBelowLevel(MR.GraphVertId basin, float waterLevel)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getBasinFacesBelowLevel", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_WatershedGraph_getBasinFacesBelowLevel(_Underlying *_this, MR.GraphVertId basin, float waterLevel);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_WatershedGraph_getBasinFacesBelowLevel(_UnderlyingPtr, basin, waterLevel), is_owning: true));
        }

        /// returns water volume in basin when its surface reaches given level, which must be in between
        /// the lowest basin level and the lowest level on basin's boundary
        /// Generated from method `MR::WatershedGraph::computeBasinVolume`.
        public unsafe double ComputeBasinVolume(MR.GraphVertId basin, float waterLevel)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_computeBasinVolume", ExactSpelling = true)]
            extern static double __MR_WatershedGraph_computeBasinVolume(_Underlying *_this, MR.GraphVertId basin, float waterLevel);
            return __MR_WatershedGraph_computeBasinVolume(_UnderlyingPtr, basin, waterLevel);
        }

        /// returns the mesh edges between current basins
        /// \param joinOverflowBasins if true then overflowing basins will be merged in the target basins (except for overflow in outside)
        /// Generated from method `MR::WatershedGraph::getInterBasinEdges`.
        /// Parameter `joinOverflowBasins` defaults to `false`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetInterBasinEdges(bool? joinOverflowBasins = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getInterBasinEdges", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_WatershedGraph_getInterBasinEdges(_Underlying *_this, byte *joinOverflowBasins);
            byte __deref_joinOverflowBasins = joinOverflowBasins.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_WatershedGraph_getInterBasinEdges(_UnderlyingPtr, joinOverflowBasins.HasValue ? &__deref_joinOverflowBasins : null), is_owning: true));
        }

        /// returns all overflow points in the graph
        /// Generated from method `MR::WatershedGraph::getOverflowPoints`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRWatershedGraphOverflowPoint> GetOverflowPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_getOverflowPoints", ExactSpelling = true)]
            extern static MR.Std.Vector_MRWatershedGraphOverflowPoint._Underlying *__MR_WatershedGraph_getOverflowPoints(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRWatershedGraphOverflowPoint(__MR_WatershedGraph_getOverflowPoints(_UnderlyingPtr), is_owning: true));
        }

        /// computes a map from initial basin id to a valid basin in which it was merged
        /// \param joinOverflowBasins if true then overflowing basins will be merged in the target basins (except for overflow in outside)
        /// Generated from method `MR::WatershedGraph::iniBasin2Tgt`.
        /// Parameter `joinOverflowBasins` defaults to `false`.
        public unsafe MR.Misc._Moved<MR.Vector_MRGraphVertId_MRGraphVertId> IniBasin2Tgt(bool? joinOverflowBasins = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_iniBasin2Tgt", ExactSpelling = true)]
            extern static MR.Vector_MRGraphVertId_MRGraphVertId._Underlying *__MR_WatershedGraph_iniBasin2Tgt(_Underlying *_this, byte *joinOverflowBasins);
            byte __deref_joinOverflowBasins = joinOverflowBasins.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.Vector_MRGraphVertId_MRGraphVertId(__MR_WatershedGraph_iniBasin2Tgt(_UnderlyingPtr, joinOverflowBasins.HasValue ? &__deref_joinOverflowBasins : null), is_owning: true));
        }

        /// associated with each vertex in graph
        /// Generated from class `MR::WatershedGraph::BasinInfo`.
        /// This is the const half of the class.
        public class Const_BasinInfo : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BasinInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Destroy", ExactSpelling = true)]
                extern static void __MR_WatershedGraph_BasinInfo_Destroy(_Underlying *_this);
                __MR_WatershedGraph_BasinInfo_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BasinInfo() {Dispose(false);}

            ///< in the whole basin
            public unsafe MR.Const_VertId LowestVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_lowestVert", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_WatershedGraph_BasinInfo_Get_lowestVert(_Underlying *_this);
                    return new(__MR_WatershedGraph_BasinInfo_Get_lowestVert(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< lowest level (z-coordinate of lowestVert) in the basin
            public unsafe float LowestLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_lowestLevel", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_lowestLevel(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_lowestLevel(_UnderlyingPtr);
                }
            }

            ///< precipitation area that flows in this basin (and if it is full, continue flowing next)
            public unsafe float Area
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_area", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_area(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_area(_UnderlyingPtr);
                }
            }

            ///< lowest position on the boundary of the basin
            public unsafe float LowestBdLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_lowestBdLevel", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_lowestBdLevel(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_lowestBdLevel(_UnderlyingPtr);
                }
            }

            ///< full water volume to be accumulated in the basin till water reaches the lowest height on the boundary
            public unsafe float MaxVolume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_maxVolume", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_maxVolume(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_maxVolume(_UnderlyingPtr);
                }
            }

            ///< accumulated water volume in the basin so far
            public unsafe float AccVolume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_accVolume", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_accVolume(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_accVolume(_UnderlyingPtr);
                }
            }

            ///< the amount when accVolume was last updated
            public unsafe float LastUpdateAmount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_lastUpdateAmount", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_lastUpdateAmount(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_lastUpdateAmount(_UnderlyingPtr);
                }
            }

            ///< water level in the basin when it was formed (by merge or creation)
            public unsafe float LastMergeLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_lastMergeLevel", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_lastMergeLevel(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_lastMergeLevel(_UnderlyingPtr);
                }
            }

            ///< water volume in the basin when it was formed (by merge or creation)
            public unsafe float LastMergeVolume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_lastMergeVolume", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_Get_lastMergeVolume(_Underlying *_this);
                    return *__MR_WatershedGraph_BasinInfo_Get_lastMergeVolume(_UnderlyingPtr);
                }
            }

            ///< when level=lowestBdLevel, volume=0, all water from this basin overflows via this boundary
            public unsafe MR.Const_GraphEdgeId OverflowVia
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_Get_overflowVia", ExactSpelling = true)]
                    extern static MR.Const_GraphEdgeId._Underlying *__MR_WatershedGraph_BasinInfo_Get_overflowVia(_Underlying *_this);
                    return new(__MR_WatershedGraph_BasinInfo_Get_overflowVia(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BasinInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WatershedGraph.BasinInfo._Underlying *__MR_WatershedGraph_BasinInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_WatershedGraph_BasinInfo_DefaultConstruct();
            }

            /// Generated from constructor `MR::WatershedGraph::BasinInfo::BasinInfo`.
            public unsafe Const_BasinInfo(MR.WatershedGraph.Const_BasinInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.BasinInfo._Underlying *__MR_WatershedGraph_BasinInfo_ConstructFromAnother(MR.WatershedGraph.BasinInfo._Underlying *_other);
                _UnderlyingPtr = __MR_WatershedGraph_BasinInfo_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// amount of precipitation (in same units as mesh coordinates and water level),
            /// which can be added before overflowing the basin
            /// Generated from method `MR::WatershedGraph::BasinInfo::amountTillOverflow`.
            public unsafe float AmountTillOverflow()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_amountTillOverflow", ExactSpelling = true)]
                extern static float __MR_WatershedGraph_BasinInfo_amountTillOverflow(_Underlying *_this);
                return __MR_WatershedGraph_BasinInfo_amountTillOverflow(_UnderlyingPtr);
            }

            /// approximate current level of water (z-coordinate) in the basin
            /// Generated from method `MR::WatershedGraph::BasinInfo::approxLevel`.
            public unsafe float ApproxLevel()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_approxLevel", ExactSpelling = true)]
                extern static float __MR_WatershedGraph_BasinInfo_approxLevel(_Underlying *_this);
                return __MR_WatershedGraph_BasinInfo_approxLevel(_UnderlyingPtr);
            }
        }

        /// associated with each vertex in graph
        /// Generated from class `MR::WatershedGraph::BasinInfo`.
        /// This is the non-const half of the class.
        public class BasinInfo : Const_BasinInfo
        {
            internal unsafe BasinInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            ///< in the whole basin
            public new unsafe MR.Mut_VertId LowestVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_lowestVert", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_WatershedGraph_BasinInfo_GetMutable_lowestVert(_Underlying *_this);
                    return new(__MR_WatershedGraph_BasinInfo_GetMutable_lowestVert(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< lowest level (z-coordinate of lowestVert) in the basin
            public new unsafe ref float LowestLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_lowestLevel", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_lowestLevel(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_lowestLevel(_UnderlyingPtr);
                }
            }

            ///< precipitation area that flows in this basin (and if it is full, continue flowing next)
            public new unsafe ref float Area
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_area", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_area(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_area(_UnderlyingPtr);
                }
            }

            ///< lowest position on the boundary of the basin
            public new unsafe ref float LowestBdLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_lowestBdLevel", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_lowestBdLevel(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_lowestBdLevel(_UnderlyingPtr);
                }
            }

            ///< full water volume to be accumulated in the basin till water reaches the lowest height on the boundary
            public new unsafe ref float MaxVolume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_maxVolume", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_maxVolume(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_maxVolume(_UnderlyingPtr);
                }
            }

            ///< accumulated water volume in the basin so far
            public new unsafe ref float AccVolume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_accVolume", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_accVolume(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_accVolume(_UnderlyingPtr);
                }
            }

            ///< the amount when accVolume was last updated
            public new unsafe ref float LastUpdateAmount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_lastUpdateAmount", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_lastUpdateAmount(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_lastUpdateAmount(_UnderlyingPtr);
                }
            }

            ///< water level in the basin when it was formed (by merge or creation)
            public new unsafe ref float LastMergeLevel
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_lastMergeLevel", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_lastMergeLevel(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_lastMergeLevel(_UnderlyingPtr);
                }
            }

            ///< water volume in the basin when it was formed (by merge or creation)
            public new unsafe ref float LastMergeVolume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_lastMergeVolume", ExactSpelling = true)]
                    extern static float *__MR_WatershedGraph_BasinInfo_GetMutable_lastMergeVolume(_Underlying *_this);
                    return ref *__MR_WatershedGraph_BasinInfo_GetMutable_lastMergeVolume(_UnderlyingPtr);
                }
            }

            ///< when level=lowestBdLevel, volume=0, all water from this basin overflows via this boundary
            public new unsafe MR.Mut_GraphEdgeId OverflowVia
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_GetMutable_overflowVia", ExactSpelling = true)]
                    extern static MR.Mut_GraphEdgeId._Underlying *__MR_WatershedGraph_BasinInfo_GetMutable_overflowVia(_Underlying *_this);
                    return new(__MR_WatershedGraph_BasinInfo_GetMutable_overflowVia(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BasinInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WatershedGraph.BasinInfo._Underlying *__MR_WatershedGraph_BasinInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_WatershedGraph_BasinInfo_DefaultConstruct();
            }

            /// Generated from constructor `MR::WatershedGraph::BasinInfo::BasinInfo`.
            public unsafe BasinInfo(MR.WatershedGraph.Const_BasinInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.BasinInfo._Underlying *__MR_WatershedGraph_BasinInfo_ConstructFromAnother(MR.WatershedGraph.BasinInfo._Underlying *_other);
                _UnderlyingPtr = __MR_WatershedGraph_BasinInfo_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::WatershedGraph::BasinInfo::operator=`.
            public unsafe MR.WatershedGraph.BasinInfo Assign(MR.WatershedGraph.Const_BasinInfo _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.BasinInfo._Underlying *__MR_WatershedGraph_BasinInfo_AssignFromAnother(_Underlying *_this, MR.WatershedGraph.BasinInfo._Underlying *_other);
                return new(__MR_WatershedGraph_BasinInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }

            /// updates accumulated volume in the basin to the moment of given precipitation amount
            /// Generated from method `MR::WatershedGraph::BasinInfo::updateAccVolume`.
            public unsafe void UpdateAccVolume(float amount)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BasinInfo_updateAccVolume", ExactSpelling = true)]
                extern static void __MR_WatershedGraph_BasinInfo_updateAccVolume(_Underlying *_this, float amount);
                __MR_WatershedGraph_BasinInfo_updateAccVolume(_UnderlyingPtr, amount);
            }
        }

        /// This is used for optional parameters of class `BasinInfo` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BasinInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BasinInfo`/`Const_BasinInfo` directly.
        public class _InOptMut_BasinInfo
        {
            public BasinInfo? Opt;

            public _InOptMut_BasinInfo() {}
            public _InOptMut_BasinInfo(BasinInfo value) {Opt = value;}
            public static implicit operator _InOptMut_BasinInfo(BasinInfo value) {return new(value);}
        }

        /// This is used for optional parameters of class `BasinInfo` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BasinInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BasinInfo`/`Const_BasinInfo` to pass it to the function.
        public class _InOptConst_BasinInfo
        {
            public Const_BasinInfo? Opt;

            public _InOptConst_BasinInfo() {}
            public _InOptConst_BasinInfo(Const_BasinInfo value) {Opt = value;}
            public static implicit operator _InOptConst_BasinInfo(Const_BasinInfo value) {return new(value);}
        }

        /// associated with each edge in graph
        /// Generated from class `MR::WatershedGraph::BdInfo`.
        /// This is the const half of the class.
        public class Const_BdInfo : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BdInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_Destroy", ExactSpelling = true)]
                extern static void __MR_WatershedGraph_BdInfo_Destroy(_Underlying *_this);
                __MR_WatershedGraph_BdInfo_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BdInfo() {Dispose(false);}

            ///< on this boundary
            public unsafe MR.Const_VertId LowestVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_Get_lowestVert", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_WatershedGraph_BdInfo_Get_lowestVert(_Underlying *_this);
                    return new(__MR_WatershedGraph_BdInfo_Get_lowestVert(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BdInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_WatershedGraph_BdInfo_DefaultConstruct();
            }

            /// Constructs `MR::WatershedGraph::BdInfo` elementwise.
            public unsafe Const_BdInfo(MR.VertId lowestVert) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_ConstructFrom", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_ConstructFrom(MR.VertId lowestVert);
                _UnderlyingPtr = __MR_WatershedGraph_BdInfo_ConstructFrom(lowestVert);
            }

            /// Generated from constructor `MR::WatershedGraph::BdInfo::BdInfo`.
            public unsafe Const_BdInfo(MR.WatershedGraph.Const_BdInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_ConstructFromAnother(MR.WatershedGraph.BdInfo._Underlying *_other);
                _UnderlyingPtr = __MR_WatershedGraph_BdInfo_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// associated with each edge in graph
        /// Generated from class `MR::WatershedGraph::BdInfo`.
        /// This is the non-const half of the class.
        public class BdInfo : Const_BdInfo
        {
            internal unsafe BdInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            ///< on this boundary
            public new unsafe MR.Mut_VertId LowestVert
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_GetMutable_lowestVert", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_WatershedGraph_BdInfo_GetMutable_lowestVert(_Underlying *_this);
                    return new(__MR_WatershedGraph_BdInfo_GetMutable_lowestVert(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BdInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_WatershedGraph_BdInfo_DefaultConstruct();
            }

            /// Constructs `MR::WatershedGraph::BdInfo` elementwise.
            public unsafe BdInfo(MR.VertId lowestVert) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_ConstructFrom", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_ConstructFrom(MR.VertId lowestVert);
                _UnderlyingPtr = __MR_WatershedGraph_BdInfo_ConstructFrom(lowestVert);
            }

            /// Generated from constructor `MR::WatershedGraph::BdInfo::BdInfo`.
            public unsafe BdInfo(MR.WatershedGraph.Const_BdInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_ConstructFromAnother(MR.WatershedGraph.BdInfo._Underlying *_other);
                _UnderlyingPtr = __MR_WatershedGraph_BdInfo_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::WatershedGraph::BdInfo::operator=`.
            public unsafe MR.WatershedGraph.BdInfo Assign(MR.WatershedGraph.Const_BdInfo _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_BdInfo_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_BdInfo_AssignFromAnother(_Underlying *_this, MR.WatershedGraph.BdInfo._Underlying *_other);
                return new(__MR_WatershedGraph_BdInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `BdInfo` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BdInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BdInfo`/`Const_BdInfo` directly.
        public class _InOptMut_BdInfo
        {
            public BdInfo? Opt;

            public _InOptMut_BdInfo() {}
            public _InOptMut_BdInfo(BdInfo value) {Opt = value;}
            public static implicit operator _InOptMut_BdInfo(BdInfo value) {return new(value);}
        }

        /// This is used for optional parameters of class `BdInfo` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BdInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BdInfo`/`Const_BdInfo` to pass it to the function.
        public class _InOptConst_BdInfo
        {
            public Const_BdInfo? Opt;

            public _InOptConst_BdInfo() {}
            public _InOptConst_BdInfo(Const_BdInfo value) {Opt = value;}
            public static implicit operator _InOptConst_BdInfo(Const_BdInfo value) {return new(value);}
        }

        /// describes a point where a flow from one basin overflows into another basin
        /// Generated from class `MR::WatershedGraph::OverflowPoint`.
        /// This is the const half of the class.
        public class Const_OverflowPoint : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_OverflowPoint(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_Destroy", ExactSpelling = true)]
                extern static void __MR_WatershedGraph_OverflowPoint_Destroy(_Underlying *_this);
                __MR_WatershedGraph_OverflowPoint_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_OverflowPoint() {Dispose(false);}

            // mesh vertex on the boundary of full basin and the other where it overflows
            public unsafe MR.Const_VertId V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_Get_v", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_WatershedGraph_OverflowPoint_Get_v(_Underlying *_this);
                    return new(__MR_WatershedGraph_OverflowPoint_Get_v(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_GraphVertId FullBasin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_Get_fullBasin", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_WatershedGraph_OverflowPoint_Get_fullBasin(_Underlying *_this);
                    return new(__MR_WatershedGraph_OverflowPoint_Get_fullBasin(_UnderlyingPtr), is_owning: false);
                }
            }

            // basin where the flow from v goes
            public unsafe MR.Const_GraphVertId OverflowTo
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_Get_overflowTo", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_WatershedGraph_OverflowPoint_Get_overflowTo(_Underlying *_this);
                    return new(__MR_WatershedGraph_OverflowPoint_Get_overflowTo(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_OverflowPoint() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_DefaultConstruct();
                _UnderlyingPtr = __MR_WatershedGraph_OverflowPoint_DefaultConstruct();
            }

            /// Constructs `MR::WatershedGraph::OverflowPoint` elementwise.
            public unsafe Const_OverflowPoint(MR.VertId v, MR.GraphVertId fullBasin, MR.GraphVertId overflowTo) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_ConstructFrom", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_ConstructFrom(MR.VertId v, MR.GraphVertId fullBasin, MR.GraphVertId overflowTo);
                _UnderlyingPtr = __MR_WatershedGraph_OverflowPoint_ConstructFrom(v, fullBasin, overflowTo);
            }

            /// Generated from constructor `MR::WatershedGraph::OverflowPoint::OverflowPoint`.
            public unsafe Const_OverflowPoint(MR.WatershedGraph.Const_OverflowPoint _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_ConstructFromAnother(MR.WatershedGraph.OverflowPoint._Underlying *_other);
                _UnderlyingPtr = __MR_WatershedGraph_OverflowPoint_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// describes a point where a flow from one basin overflows into another basin
        /// Generated from class `MR::WatershedGraph::OverflowPoint`.
        /// This is the non-const half of the class.
        public class OverflowPoint : Const_OverflowPoint
        {
            internal unsafe OverflowPoint(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // mesh vertex on the boundary of full basin and the other where it overflows
            public new unsafe MR.Mut_VertId V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_GetMutable_v", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_WatershedGraph_OverflowPoint_GetMutable_v(_Underlying *_this);
                    return new(__MR_WatershedGraph_OverflowPoint_GetMutable_v(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_GraphVertId FullBasin
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_GetMutable_fullBasin", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_WatershedGraph_OverflowPoint_GetMutable_fullBasin(_Underlying *_this);
                    return new(__MR_WatershedGraph_OverflowPoint_GetMutable_fullBasin(_UnderlyingPtr), is_owning: false);
                }
            }

            // basin where the flow from v goes
            public new unsafe MR.Mut_GraphVertId OverflowTo
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_GetMutable_overflowTo", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_WatershedGraph_OverflowPoint_GetMutable_overflowTo(_Underlying *_this);
                    return new(__MR_WatershedGraph_OverflowPoint_GetMutable_overflowTo(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe OverflowPoint() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_DefaultConstruct", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_DefaultConstruct();
                _UnderlyingPtr = __MR_WatershedGraph_OverflowPoint_DefaultConstruct();
            }

            /// Constructs `MR::WatershedGraph::OverflowPoint` elementwise.
            public unsafe OverflowPoint(MR.VertId v, MR.GraphVertId fullBasin, MR.GraphVertId overflowTo) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_ConstructFrom", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_ConstructFrom(MR.VertId v, MR.GraphVertId fullBasin, MR.GraphVertId overflowTo);
                _UnderlyingPtr = __MR_WatershedGraph_OverflowPoint_ConstructFrom(v, fullBasin, overflowTo);
            }

            /// Generated from constructor `MR::WatershedGraph::OverflowPoint::OverflowPoint`.
            public unsafe OverflowPoint(MR.WatershedGraph.Const_OverflowPoint _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_ConstructFromAnother(MR.WatershedGraph.OverflowPoint._Underlying *_other);
                _UnderlyingPtr = __MR_WatershedGraph_OverflowPoint_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::WatershedGraph::OverflowPoint::operator=`.
            public unsafe MR.WatershedGraph.OverflowPoint Assign(MR.WatershedGraph.Const_OverflowPoint _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_OverflowPoint_AssignFromAnother", ExactSpelling = true)]
                extern static MR.WatershedGraph.OverflowPoint._Underlying *__MR_WatershedGraph_OverflowPoint_AssignFromAnother(_Underlying *_this, MR.WatershedGraph.OverflowPoint._Underlying *_other);
                return new(__MR_WatershedGraph_OverflowPoint_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `OverflowPoint` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_OverflowPoint`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `OverflowPoint`/`Const_OverflowPoint` directly.
        public class _InOptMut_OverflowPoint
        {
            public OverflowPoint? Opt;

            public _InOptMut_OverflowPoint() {}
            public _InOptMut_OverflowPoint(OverflowPoint value) {Opt = value;}
            public static implicit operator _InOptMut_OverflowPoint(OverflowPoint value) {return new(value);}
        }

        /// This is used for optional parameters of class `OverflowPoint` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_OverflowPoint`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `OverflowPoint`/`Const_OverflowPoint` to pass it to the function.
        public class _InOptConst_OverflowPoint
        {
            public Const_OverflowPoint? Opt;

            public _InOptConst_OverflowPoint() {}
            public _InOptConst_OverflowPoint(Const_OverflowPoint value) {Opt = value;}
            public static implicit operator _InOptConst_OverflowPoint(Const_OverflowPoint value) {return new(value);}
        }
    }

    /// graphs representing rain basins on the mesh
    /// Generated from class `MR::WatershedGraph`.
    /// This is the non-const half of the class.
    public class WatershedGraph : Const_WatershedGraph
    {
        internal unsafe WatershedGraph(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::WatershedGraph::WatershedGraph`.
        public unsafe WatershedGraph(MR._ByValue_WatershedGraph _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WatershedGraph._Underlying *__MR_WatershedGraph_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.WatershedGraph._Underlying *_other);
            _UnderlyingPtr = __MR_WatershedGraph_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// constructs the graph from given mesh, heights in z-coordinate, and initial subdivision on basins
        /// Generated from constructor `MR::WatershedGraph::WatershedGraph`.
        public unsafe WatershedGraph(MR.Const_Mesh mesh, MR.Const_Vector_Int_MRFaceId face2basin, int numBasins) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_Construct", ExactSpelling = true)]
            extern static MR.WatershedGraph._Underlying *__MR_WatershedGraph_Construct(MR.Const_Mesh._Underlying *mesh, MR.Const_Vector_Int_MRFaceId._Underlying *face2basin, int numBasins);
            _UnderlyingPtr = __MR_WatershedGraph_Construct(mesh._UnderlyingPtr, face2basin._UnderlyingPtr, numBasins);
        }

        /// Generated from method `MR::WatershedGraph::basinInfo`.
        public unsafe new MR.WatershedGraph.BasinInfo BasinInfo_(MR.GraphVertId v)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_basinInfo", ExactSpelling = true)]
            extern static MR.WatershedGraph.BasinInfo._Underlying *__MR_WatershedGraph_basinInfo(_Underlying *_this, MR.GraphVertId v);
            return new(__MR_WatershedGraph_basinInfo(_UnderlyingPtr, v), is_owning: false);
        }

        /// Generated from method `MR::WatershedGraph::bdInfo`.
        public unsafe new MR.WatershedGraph.BdInfo BdInfo_(MR.GraphEdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_bdInfo", ExactSpelling = true)]
            extern static MR.WatershedGraph.BdInfo._Underlying *__MR_WatershedGraph_bdInfo(_Underlying *_this, MR.GraphEdgeId e);
            return new(__MR_WatershedGraph_bdInfo(_UnderlyingPtr, e), is_owning: false);
        }

        /// replaces parent of each basin with its computed root;
        /// this speeds up following calls to getRootBasin()
        /// Generated from method `MR::WatershedGraph::setParentsToRoots`.
        public unsafe void SetParentsToRoots()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_setParentsToRoots", ExactSpelling = true)]
            extern static void __MR_WatershedGraph_setParentsToRoots(_Underlying *_this);
            __MR_WatershedGraph_setParentsToRoots(_UnderlyingPtr);
        }

        /// merges basin v1 into basin v0, v1 is deleted after that, returns v0
        /// Generated from method `MR::WatershedGraph::merge`.
        public unsafe MR.GraphVertId Merge(MR.GraphVertId v0, MR.GraphVertId v1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_merge", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_WatershedGraph_merge(_Underlying *_this, MR.GraphVertId v0, MR.GraphVertId v1);
            return __MR_WatershedGraph_merge(_UnderlyingPtr, v0, v1);
        }

        /// merges two basins sharing given boundary, returns remaining basin
        /// Generated from method `MR::WatershedGraph::mergeViaBd`.
        public unsafe MR.GraphVertId MergeViaBd(MR.GraphEdgeId bd)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WatershedGraph_mergeViaBd", ExactSpelling = true)]
            extern static MR.GraphVertId __MR_WatershedGraph_mergeViaBd(_Underlying *_this, MR.GraphEdgeId bd);
            return __MR_WatershedGraph_mergeViaBd(_UnderlyingPtr, bd);
        }
    }

    /// This is used as a function parameter when the underlying function receives `WatershedGraph` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `WatershedGraph`/`Const_WatershedGraph` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_WatershedGraph
    {
        internal readonly Const_WatershedGraph? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_WatershedGraph(Const_WatershedGraph new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_WatershedGraph(Const_WatershedGraph arg) {return new(arg);}
        public _ByValue_WatershedGraph(MR.Misc._Moved<WatershedGraph> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_WatershedGraph(MR.Misc._Moved<WatershedGraph> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `WatershedGraph` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_WatershedGraph`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WatershedGraph`/`Const_WatershedGraph` directly.
    public class _InOptMut_WatershedGraph
    {
        public WatershedGraph? Opt;

        public _InOptMut_WatershedGraph() {}
        public _InOptMut_WatershedGraph(WatershedGraph value) {Opt = value;}
        public static implicit operator _InOptMut_WatershedGraph(WatershedGraph value) {return new(value);}
    }

    /// This is used for optional parameters of class `WatershedGraph` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_WatershedGraph`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WatershedGraph`/`Const_WatershedGraph` to pass it to the function.
    public class _InOptConst_WatershedGraph
    {
        public Const_WatershedGraph? Opt;

        public _InOptConst_WatershedGraph() {}
        public _InOptConst_WatershedGraph(Const_WatershedGraph value) {Opt = value;}
        public static implicit operator _InOptConst_WatershedGraph(Const_WatershedGraph value) {return new(value);}
    }
}
