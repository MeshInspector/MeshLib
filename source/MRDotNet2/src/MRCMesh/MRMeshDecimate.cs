public static partial class MR
{
    /// Defines the order of edge collapses inside Decimate algorithm
    public enum DecimateStrategy : int
    {
        // the next edge to collapse will be the one that introduced minimal error to the surface
        MinimizeError = 0,
        // the next edge to collapse will be the shortest one
        ShortestEdgeFirst = 1,
    }

    /**
    * \struct MR::DecimateSettings
    * \brief Parameters structure for MR::decimateMesh
    *
    *
    * \sa \ref decimateMesh
    */
    /// Generated from class `MR::DecimateSettings`.
    /// This is the const half of the class.
    public class Const_DecimateSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DecimateSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DecimateSettings_Destroy(_Underlying *_this);
            __MR_DecimateSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DecimateSettings() {Dispose(false);}

        public unsafe MR.DecimateStrategy Strategy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_strategy", ExactSpelling = true)]
                extern static MR.DecimateStrategy *__MR_DecimateSettings_Get_strategy(_Underlying *_this);
                return *__MR_DecimateSettings_Get_strategy(_UnderlyingPtr);
            }
        }

        /// for DecimateStrategy::MinimizeError:
        ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
        /// for DecimateStrategy::ShortestEdgeFirst only:
        ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
        public unsafe float MaxError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxError", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_maxError(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxError(_UnderlyingPtr);
            }
        }

        /// Maximal possible edge length created during decimation
        public unsafe float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_maxEdgeLen(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximal shift of a boundary during one edge collapse
        public unsafe float MaxBdShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxBdShift", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_maxBdShift(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxBdShift(_UnderlyingPtr);
            }
        }

        /// Maximal possible aspect ratio of a triangle introduced during decimation
        public unsafe float MaxTriangleAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxTriangleAspectRatio", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_maxTriangleAspectRatio(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxTriangleAspectRatio(_UnderlyingPtr);
            }
        }

        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public unsafe float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_criticalTriAspectRatio(_Underlying *_this);
                return *__MR_DecimateSettings_Get_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
        public unsafe float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_tinyEdgeLength(_Underlying *_this);
                return *__MR_DecimateSettings_Get_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_stabilizer(_Underlying *_this);
                return *__MR_DecimateSettings_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// if false, then quadratic error metric is equal to the sum of distances to the planes of original mesh triangles;
        /// if true, then the sum is weighted, and the weight is equal to the angle of adjacent triangle at the vertex divided on PI (to get one after summing all 3 vertices of the triangle)
        public unsafe bool AngleWeightedDistToPlane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_angleWeightedDistToPlane", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_angleWeightedDistToPlane(_Underlying *_this);
                return *__MR_DecimateSettings_Get_angleWeightedDistToPlane(_UnderlyingPtr);
            }
        }

        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public unsafe bool OptimizeVertexPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_optimizeVertexPos", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_optimizeVertexPos(_Underlying *_this);
                return *__MR_DecimateSettings_Get_optimizeVertexPos(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted vertices
        public unsafe int MaxDeletedVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxDeletedVertices", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_Get_maxDeletedVertices(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxDeletedVertices(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted faces
        public unsafe int MaxDeletedFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxDeletedFaces", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_Get_maxDeletedFaces(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxDeletedFaces(_UnderlyingPtr);
            }
        }

        /// Region on mesh to be decimated, it is updated during the operation
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_region(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped, but they can be collapsed or replaced during collapse of nearby edges so it is updated during the operation
        public unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_notFlippable", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_notFlippable(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_notFlippable(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapse of edges incident to notFlippable edges,
        /// which can move vertices of notFlippable edges unless they are fixed
        public unsafe bool CollapseNearNotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_collapseNearNotFlippable", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_collapseNearNotFlippable(_Underlying *_this);
                return *__MR_DecimateSettings_Get_collapseNearNotFlippable(_UnderlyingPtr);
            }
        }

        /// If pointer is not null, then only edges from here can be collapsed (and some nearby edges can disappear);
        /// the algorithm updates this map during collapses, removing or replacing elements
        public unsafe ref void * EdgesToCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_edgesToCollapse", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_edgesToCollapse(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_edgesToCollapse(_UnderlyingPtr);
            }
        }

        /// if an edge present as a key in this map is flipped or collapsed, then same happens to the value-edge (with same collapse position);
        /// the algorithm updates this map during collapses, removing or replacing elements
        public unsafe ref void * TwinMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_twinMap", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_twinMap(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_twinMap(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
        public unsafe bool TouchNearBdEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_touchNearBdEdges", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_touchNearBdEdges(_Underlying *_this);
                return *__MR_DecimateSettings_Get_touchNearBdEdges(_UnderlyingPtr);
            }
        }

        /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
        /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
        /// this setting is ignored if touchNearBdEdges=false
        public unsafe bool TouchBdVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_touchBdVerts", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_touchBdVerts(_Underlying *_this);
                return *__MR_DecimateSettings_Get_touchBdVerts(_UnderlyingPtr);
            }
        }

        /// if touchNearBdEdges=false or touchBdVerts=false then the algorithm needs to know about all boundary vertices;
        /// if the pointer is not null then boundary vertices detection is replaced with testing values in this bit-set;
        /// the algorithm updates this set if it packs the mesh
        public unsafe ref void * BdVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_bdVerts", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_bdVerts(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_bdVerts(_UnderlyingPtr);
            }
        }

        /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
        /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
        public unsafe float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_Get_maxAngleChange(_Underlying *_this);
                return *__MR_DecimateSettings_Get_maxAngleChange(_UnderlyingPtr);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public unsafe MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector3fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_DecimateSettings_Get_preCollapse(_Underlying *_this);
                return new(__MR_DecimateSettings_Get_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback for adjusting error introduced by this
        * edge collapse and the collapse position.
        * \details On input the callback gets the squared error and position computed by standard means,
        * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
        * This callback can be called from many threads in parallel and must be thread-safe.
        * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
        */
        public unsafe MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef AdjustCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_adjustCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *__MR_DecimateSettings_Get_adjustCollapse(_Underlying *_this);
                return new(__MR_DecimateSettings_Get_adjustCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time edge (del) is deleted;
        /// if valid (rem) is given then dest(del) = dest(rem) and their origins are in different ends of collapsing edge, (rem) shall take the place of (del)
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeDel
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_onEdgeDel", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_DecimateSettings_Get_onEdgeDel(_Underlying *_this);
                return new(__MR_DecimateSettings_Get_onEdgeDel(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief  If not null, then vertex quadratic forms are stored there;
        * if on input the vector is not empty then initialization is skipped in favor of values from there;
        * on output: quadratic form for each remaining vertex is returned there
        */
        public unsafe ref void * VertForms
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_vertForms", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_vertForms(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_vertForms(_UnderlyingPtr);
            }
        }

        ///  whether to pack mesh at the end
        public unsafe bool PackMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_packMesh", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_packMesh(_Underlying *_this);
                return *__MR_DecimateSettings_Get_packMesh(_UnderlyingPtr);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DecimateSettings_Get_progressCallback(_Underlying *_this);
                return new(__MR_DecimateSettings_Get_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
        /// IMPORTANT: please call mesh.packOptimally() before calling decimating with subdivideParts > 1, otherwise performance will be bad
        public unsafe int SubdivideParts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_subdivideParts", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_Get_subdivideParts(_Underlying *_this);
                return *__MR_DecimateSettings_Get_subdivideParts(_UnderlyingPtr);
            }
        }

        /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
        /// to eliminate small edges near the border of individual parts
        public unsafe bool DecimateBetweenParts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_decimateBetweenParts", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_Get_decimateBetweenParts(_Underlying *_this);
                return *__MR_DecimateSettings_Get_decimateBetweenParts(_UnderlyingPtr);
            }
        }

        /// if not null, then it contains the faces of each subdivision part on input, which must not overlap,
        /// and after decimation of all parts, the region inside each part is put here;
        /// decimateBetweenParts=true or packMesh=true are not compatible with this option
        public unsafe ref void * PartFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_partFaces", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_Get_partFaces(_Underlying *_this);
                return ref *__MR_DecimateSettings_Get_partFaces(_UnderlyingPtr);
            }
        }

        /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
        public unsafe int MinFacesInPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_Get_minFacesInPart", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_Get_minFacesInPart(_Underlying *_this);
                return *__MR_DecimateSettings_Get_minFacesInPart(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DecimateSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimateSettings_DefaultConstruct();
        }

        /// Constructs `MR::DecimateSettings` elementwise.
        public unsafe Const_DecimateSettings(MR.DecimateStrategy strategy, float maxError, float maxEdgeLen, float maxBdShift, float maxTriangleAspectRatio, float criticalTriAspectRatio, float tinyEdgeLength, float stabilizer, bool angleWeightedDistToPlane, bool optimizeVertexPos, int maxDeletedVertices, int maxDeletedFaces, MR.FaceBitSet? region, MR.UndirectedEdgeBitSet? notFlippable, bool collapseNearNotFlippable, MR.UndirectedEdgeBitSet? edgesToCollapse, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? twinMap, bool touchNearBdEdges, bool touchBdVerts, MR.VertBitSet? bdVerts, float maxAngleChange, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector3fRef preCollapse, MR.Std._ByValue_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef adjustCollapse, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeDel, MR.Vector_MRQuadraticForm3f_MRVertId? vertForms, bool packMesh, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback, int subdivideParts, bool decimateBetweenParts, MR.Std.Vector_MRFaceBitSet? partFaces, int minFacesInPart) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_ConstructFrom(MR.DecimateStrategy strategy, float maxError, float maxEdgeLen, float maxBdShift, float maxTriangleAspectRatio, float criticalTriAspectRatio, float tinyEdgeLength, float stabilizer, byte angleWeightedDistToPlane, byte optimizeVertexPos, int maxDeletedVertices, int maxDeletedFaces, MR.FaceBitSet._Underlying *region, MR.UndirectedEdgeBitSet._Underlying *notFlippable, byte collapseNearNotFlippable, MR.UndirectedEdgeBitSet._Underlying *edgesToCollapse, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *twinMap, byte touchNearBdEdges, byte touchBdVerts, MR.VertBitSet._Underlying *bdVerts, float maxAngleChange, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *preCollapse, MR.Misc._PassBy adjustCollapse_pass_by, MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *adjustCollapse, MR.Misc._PassBy onEdgeDel_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeDel, MR.Vector_MRQuadraticForm3f_MRVertId._Underlying *vertForms, byte packMesh, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback, int subdivideParts, byte decimateBetweenParts, MR.Std.Vector_MRFaceBitSet._Underlying *partFaces, int minFacesInPart);
            _UnderlyingPtr = __MR_DecimateSettings_ConstructFrom(strategy, maxError, maxEdgeLen, maxBdShift, maxTriangleAspectRatio, criticalTriAspectRatio, tinyEdgeLength, stabilizer, angleWeightedDistToPlane ? (byte)1 : (byte)0, optimizeVertexPos ? (byte)1 : (byte)0, maxDeletedVertices, maxDeletedFaces, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, collapseNearNotFlippable ? (byte)1 : (byte)0, edgesToCollapse is not null ? edgesToCollapse._UnderlyingPtr : null, twinMap is not null ? twinMap._UnderlyingPtr : null, touchNearBdEdges ? (byte)1 : (byte)0, touchBdVerts ? (byte)1 : (byte)0, bdVerts is not null ? bdVerts._UnderlyingPtr : null, maxAngleChange, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, adjustCollapse.PassByMode, adjustCollapse.Value is not null ? adjustCollapse.Value._UnderlyingPtr : null, onEdgeDel.PassByMode, onEdgeDel.Value is not null ? onEdgeDel.Value._UnderlyingPtr : null, vertForms is not null ? vertForms._UnderlyingPtr : null, packMesh ? (byte)1 : (byte)0, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null, subdivideParts, decimateBetweenParts ? (byte)1 : (byte)0, partFaces is not null ? partFaces._UnderlyingPtr : null, minFacesInPart);
        }

        /// Generated from constructor `MR::DecimateSettings::DecimateSettings`.
        public unsafe Const_DecimateSettings(MR._ByValue_DecimateSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DecimateSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DecimateSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * \struct MR::DecimateSettings
    * \brief Parameters structure for MR::decimateMesh
    *
    *
    * \sa \ref decimateMesh
    */
    /// Generated from class `MR::DecimateSettings`.
    /// This is the non-const half of the class.
    public class DecimateSettings : Const_DecimateSettings
    {
        internal unsafe DecimateSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.DecimateStrategy Strategy
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_strategy", ExactSpelling = true)]
                extern static MR.DecimateStrategy *__MR_DecimateSettings_GetMutable_strategy(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_strategy(_UnderlyingPtr);
            }
        }

        /// for DecimateStrategy::MinimizeError:
        ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
        /// for DecimateStrategy::ShortestEdgeFirst only:
        ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
        public new unsafe ref float MaxError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxError", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_maxError(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxError(_UnderlyingPtr);
            }
        }

        /// Maximal possible edge length created during decimation
        public new unsafe ref float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_maxEdgeLen(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximal shift of a boundary during one edge collapse
        public new unsafe ref float MaxBdShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxBdShift", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_maxBdShift(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxBdShift(_UnderlyingPtr);
            }
        }

        /// Maximal possible aspect ratio of a triangle introduced during decimation
        public new unsafe ref float MaxTriangleAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxTriangleAspectRatio", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_maxTriangleAspectRatio(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxTriangleAspectRatio(_UnderlyingPtr);
            }
        }

        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public new unsafe ref float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_criticalTriAspectRatio(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
        public new unsafe ref float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_tinyEdgeLength(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// if false, then quadratic error metric is equal to the sum of distances to the planes of original mesh triangles;
        /// if true, then the sum is weighted, and the weight is equal to the angle of adjacent triangle at the vertex divided on PI (to get one after summing all 3 vertices of the triangle)
        public new unsafe ref bool AngleWeightedDistToPlane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_angleWeightedDistToPlane", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_angleWeightedDistToPlane(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_angleWeightedDistToPlane(_UnderlyingPtr);
            }
        }

        /// if true then after each edge collapse the position of remaining vertex is optimized to
        /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
        public new unsafe ref bool OptimizeVertexPos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_optimizeVertexPos", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_optimizeVertexPos(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_optimizeVertexPos(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted vertices
        public new unsafe ref int MaxDeletedVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxDeletedVertices", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_GetMutable_maxDeletedVertices(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxDeletedVertices(_UnderlyingPtr);
            }
        }

        /// Limit on the number of deleted faces
        public new unsafe ref int MaxDeletedFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxDeletedFaces", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_GetMutable_maxDeletedFaces(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxDeletedFaces(_UnderlyingPtr);
            }
        }

        /// Region on mesh to be decimated, it is updated during the operation
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped, but they can be collapsed or replaced during collapse of nearby edges so it is updated during the operation
        public new unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_notFlippable", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_notFlippable(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_notFlippable(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapse of edges incident to notFlippable edges,
        /// which can move vertices of notFlippable edges unless they are fixed
        public new unsafe ref bool CollapseNearNotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_collapseNearNotFlippable", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_collapseNearNotFlippable(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_collapseNearNotFlippable(_UnderlyingPtr);
            }
        }

        /// If pointer is not null, then only edges from here can be collapsed (and some nearby edges can disappear);
        /// the algorithm updates this map during collapses, removing or replacing elements
        public new unsafe ref void * EdgesToCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_edgesToCollapse", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_edgesToCollapse(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_edgesToCollapse(_UnderlyingPtr);
            }
        }

        /// if an edge present as a key in this map is flipped or collapsed, then same happens to the value-edge (with same collapse position);
        /// the algorithm updates this map during collapses, removing or replacing elements
        public new unsafe ref void * TwinMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_twinMap", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_twinMap(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_twinMap(_UnderlyingPtr);
            }
        }

        /// Whether to allow collapsing or flipping edges having at least one vertex on (region) boundary
        public new unsafe ref bool TouchNearBdEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_touchNearBdEdges", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_touchNearBdEdges(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_touchNearBdEdges(_UnderlyingPtr);
            }
        }

        /// touchBdVerts=true: allow moving and eliminating boundary vertices during edge collapses;
        /// touchBdVerts=false: allow only collapsing an edge having only one boundary vertex in that vertex, so position and count of boundary vertices do not change;
        /// this setting is ignored if touchNearBdEdges=false
        public new unsafe ref bool TouchBdVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_touchBdVerts", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_touchBdVerts(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_touchBdVerts(_UnderlyingPtr);
            }
        }

        /// if touchNearBdEdges=false or touchBdVerts=false then the algorithm needs to know about all boundary vertices;
        /// if the pointer is not null then boundary vertices detection is replaced with testing values in this bit-set;
        /// the algorithm updates this set if it packs the mesh
        public new unsafe ref void * BdVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_bdVerts", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_bdVerts(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_bdVerts(_UnderlyingPtr);
            }
        }

        /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
        /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
        public new unsafe ref float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_DecimateSettings_GetMutable_maxAngleChange(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_maxAngleChange(_UnderlyingPtr);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public new unsafe MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_DecimateSettings_GetMutable_preCollapse(_Underlying *_this);
                return new(__MR_DecimateSettings_GetMutable_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback for adjusting error introduced by this
        * edge collapse and the collapse position.
        * \details On input the callback gets the squared error and position computed by standard means,
        * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
        * This callback can be called from many threads in parallel and must be thread-safe.
        * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
        */
        public new unsafe MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef AdjustCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_adjustCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *__MR_DecimateSettings_GetMutable_adjustCollapse(_Underlying *_this);
                return new(__MR_DecimateSettings_GetMutable_adjustCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time edge (del) is deleted;
        /// if valid (rem) is given then dest(del) = dest(rem) and their origins are in different ends of collapsing edge, (rem) shall take the place of (del)
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeDel
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_onEdgeDel", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_DecimateSettings_GetMutable_onEdgeDel(_Underlying *_this);
                return new(__MR_DecimateSettings_GetMutable_onEdgeDel(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief  If not null, then vertex quadratic forms are stored there;
        * if on input the vector is not empty then initialization is skipped in favor of values from there;
        * on output: quadratic form for each remaining vertex is returned there
        */
        public new unsafe ref void * VertForms
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_vertForms", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_vertForms(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_vertForms(_UnderlyingPtr);
            }
        }

        ///  whether to pack mesh at the end
        public new unsafe ref bool PackMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_packMesh", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_packMesh(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_packMesh(_UnderlyingPtr);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DecimateSettings_GetMutable_progressCallback(_Underlying *_this);
                return new(__MR_DecimateSettings_GetMutable_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
        /// IMPORTANT: please call mesh.packOptimally() before calling decimating with subdivideParts > 1, otherwise performance will be bad
        public new unsafe ref int SubdivideParts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_subdivideParts", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_GetMutable_subdivideParts(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_subdivideParts(_UnderlyingPtr);
            }
        }

        /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
        /// to eliminate small edges near the border of individual parts
        public new unsafe ref bool DecimateBetweenParts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_decimateBetweenParts", ExactSpelling = true)]
                extern static bool *__MR_DecimateSettings_GetMutable_decimateBetweenParts(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_decimateBetweenParts(_UnderlyingPtr);
            }
        }

        /// if not null, then it contains the faces of each subdivision part on input, which must not overlap,
        /// and after decimation of all parts, the region inside each part is put here;
        /// decimateBetweenParts=true or packMesh=true are not compatible with this option
        public new unsafe ref void * PartFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_partFaces", ExactSpelling = true)]
                extern static void **__MR_DecimateSettings_GetMutable_partFaces(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_partFaces(_UnderlyingPtr);
            }
        }

        /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
        public new unsafe ref int MinFacesInPart
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_GetMutable_minFacesInPart", ExactSpelling = true)]
                extern static int *__MR_DecimateSettings_GetMutable_minFacesInPart(_Underlying *_this);
                return ref *__MR_DecimateSettings_GetMutable_minFacesInPart(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DecimateSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimateSettings_DefaultConstruct();
        }

        /// Constructs `MR::DecimateSettings` elementwise.
        public unsafe DecimateSettings(MR.DecimateStrategy strategy, float maxError, float maxEdgeLen, float maxBdShift, float maxTriangleAspectRatio, float criticalTriAspectRatio, float tinyEdgeLength, float stabilizer, bool angleWeightedDistToPlane, bool optimizeVertexPos, int maxDeletedVertices, int maxDeletedFaces, MR.FaceBitSet? region, MR.UndirectedEdgeBitSet? notFlippable, bool collapseNearNotFlippable, MR.UndirectedEdgeBitSet? edgesToCollapse, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId? twinMap, bool touchNearBdEdges, bool touchBdVerts, MR.VertBitSet? bdVerts, float maxAngleChange, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector3fRef preCollapse, MR.Std._ByValue_Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef adjustCollapse, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeDel, MR.Vector_MRQuadraticForm3f_MRVertId? vertForms, bool packMesh, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback, int subdivideParts, bool decimateBetweenParts, MR.Std.Vector_MRFaceBitSet? partFaces, int minFacesInPart) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_ConstructFrom(MR.DecimateStrategy strategy, float maxError, float maxEdgeLen, float maxBdShift, float maxTriangleAspectRatio, float criticalTriAspectRatio, float tinyEdgeLength, float stabilizer, byte angleWeightedDistToPlane, byte optimizeVertexPos, int maxDeletedVertices, int maxDeletedFaces, MR.FaceBitSet._Underlying *region, MR.UndirectedEdgeBitSet._Underlying *notFlippable, byte collapseNearNotFlippable, MR.UndirectedEdgeBitSet._Underlying *edgesToCollapse, MR.Phmap.FlatHashMap_MRUndirectedEdgeId_MRUndirectedEdgeId._Underlying *twinMap, byte touchNearBdEdges, byte touchBdVerts, MR.VertBitSet._Underlying *bdVerts, float maxAngleChange, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *preCollapse, MR.Misc._PassBy adjustCollapse_pass_by, MR.Std.Function_VoidFuncFromMRUndirectedEdgeIdFloatRefMRVector3fRef._Underlying *adjustCollapse, MR.Misc._PassBy onEdgeDel_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeDel, MR.Vector_MRQuadraticForm3f_MRVertId._Underlying *vertForms, byte packMesh, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback, int subdivideParts, byte decimateBetweenParts, MR.Std.Vector_MRFaceBitSet._Underlying *partFaces, int minFacesInPart);
            _UnderlyingPtr = __MR_DecimateSettings_ConstructFrom(strategy, maxError, maxEdgeLen, maxBdShift, maxTriangleAspectRatio, criticalTriAspectRatio, tinyEdgeLength, stabilizer, angleWeightedDistToPlane ? (byte)1 : (byte)0, optimizeVertexPos ? (byte)1 : (byte)0, maxDeletedVertices, maxDeletedFaces, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, collapseNearNotFlippable ? (byte)1 : (byte)0, edgesToCollapse is not null ? edgesToCollapse._UnderlyingPtr : null, twinMap is not null ? twinMap._UnderlyingPtr : null, touchNearBdEdges ? (byte)1 : (byte)0, touchBdVerts ? (byte)1 : (byte)0, bdVerts is not null ? bdVerts._UnderlyingPtr : null, maxAngleChange, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, adjustCollapse.PassByMode, adjustCollapse.Value is not null ? adjustCollapse.Value._UnderlyingPtr : null, onEdgeDel.PassByMode, onEdgeDel.Value is not null ? onEdgeDel.Value._UnderlyingPtr : null, vertForms is not null ? vertForms._UnderlyingPtr : null, packMesh ? (byte)1 : (byte)0, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null, subdivideParts, decimateBetweenParts ? (byte)1 : (byte)0, partFaces is not null ? partFaces._UnderlyingPtr : null, minFacesInPart);
        }

        /// Generated from constructor `MR::DecimateSettings::DecimateSettings`.
        public unsafe DecimateSettings(MR._ByValue_DecimateSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DecimateSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DecimateSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DecimateSettings::operator=`.
        public unsafe MR.DecimateSettings Assign(MR._ByValue_DecimateSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DecimateSettings._Underlying *__MR_DecimateSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DecimateSettings._Underlying *_other);
            return new(__MR_DecimateSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DecimateSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DecimateSettings`/`Const_DecimateSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DecimateSettings
    {
        internal readonly Const_DecimateSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DecimateSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DecimateSettings(Const_DecimateSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DecimateSettings(Const_DecimateSettings arg) {return new(arg);}
        public _ByValue_DecimateSettings(MR.Misc._Moved<DecimateSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DecimateSettings(MR.Misc._Moved<DecimateSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DecimateSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DecimateSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimateSettings`/`Const_DecimateSettings` directly.
    public class _InOptMut_DecimateSettings
    {
        public DecimateSettings? Opt;

        public _InOptMut_DecimateSettings() {}
        public _InOptMut_DecimateSettings(DecimateSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DecimateSettings(DecimateSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DecimateSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DecimateSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimateSettings`/`Const_DecimateSettings` to pass it to the function.
    public class _InOptConst_DecimateSettings
    {
        public Const_DecimateSettings? Opt;

        public _InOptConst_DecimateSettings() {}
        public _InOptConst_DecimateSettings(Const_DecimateSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DecimateSettings(Const_DecimateSettings value) {return new(value);}
    }

    /**
    * \struct MR::DecimateResult
    * \brief Results of MR::decimateMesh
    *
    *
    * \sa \ref decimateMesh
    * \sa \ref decimateParallelMesh
    * \sa \ref resolveMeshDegenerations
    */
    /// Generated from class `MR::DecimateResult`.
    /// This is the const half of the class.
    public class Const_DecimateResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DecimateResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_Destroy", ExactSpelling = true)]
            extern static void __MR_DecimateResult_Destroy(_Underlying *_this);
            __MR_DecimateResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DecimateResult() {Dispose(false);}

        ///< Number deleted verts. Same as the number of performed collapses
        public unsafe int VertsDeleted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_Get_vertsDeleted", ExactSpelling = true)]
                extern static int *__MR_DecimateResult_Get_vertsDeleted(_Underlying *_this);
                return *__MR_DecimateResult_Get_vertsDeleted(_UnderlyingPtr);
            }
        }

        ///< Number deleted faces
        public unsafe int FacesDeleted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_Get_facesDeleted", ExactSpelling = true)]
                extern static int *__MR_DecimateResult_Get_facesDeleted(_Underlying *_this);
                return *__MR_DecimateResult_Get_facesDeleted(_UnderlyingPtr);
            }
        }

        /// for DecimateStrategy::MinimizeError:
        ///    estimated distance deviation of decimated mesh from the original mesh
        /// for DecimateStrategy::ShortestEdgeFirst:
        ///    the shortest remaining edge in the mesh
        public unsafe float ErrorIntroduced
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_Get_errorIntroduced", ExactSpelling = true)]
                extern static float *__MR_DecimateResult_Get_errorIntroduced(_Underlying *_this);
                return *__MR_DecimateResult_Get_errorIntroduced(_UnderlyingPtr);
            }
        }

        /// whether the algorithm was cancelled by the callback
        public unsafe bool Cancelled
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_Get_cancelled", ExactSpelling = true)]
                extern static bool *__MR_DecimateResult_Get_cancelled(_Underlying *_this);
                return *__MR_DecimateResult_Get_cancelled(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DecimateResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimateResult_DefaultConstruct();
        }

        /// Constructs `MR::DecimateResult` elementwise.
        public unsafe Const_DecimateResult(int vertsDeleted, int facesDeleted, float errorIntroduced, bool cancelled) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_ConstructFrom(int vertsDeleted, int facesDeleted, float errorIntroduced, byte cancelled);
            _UnderlyingPtr = __MR_DecimateResult_ConstructFrom(vertsDeleted, facesDeleted, errorIntroduced, cancelled ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::DecimateResult::DecimateResult`.
        public unsafe Const_DecimateResult(MR.Const_DecimateResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_ConstructFromAnother(MR.DecimateResult._Underlying *_other);
            _UnderlyingPtr = __MR_DecimateResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /**
    * \struct MR::DecimateResult
    * \brief Results of MR::decimateMesh
    *
    *
    * \sa \ref decimateMesh
    * \sa \ref decimateParallelMesh
    * \sa \ref resolveMeshDegenerations
    */
    /// Generated from class `MR::DecimateResult`.
    /// This is the non-const half of the class.
    public class DecimateResult : Const_DecimateResult
    {
        internal unsafe DecimateResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< Number deleted verts. Same as the number of performed collapses
        public new unsafe ref int VertsDeleted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_GetMutable_vertsDeleted", ExactSpelling = true)]
                extern static int *__MR_DecimateResult_GetMutable_vertsDeleted(_Underlying *_this);
                return ref *__MR_DecimateResult_GetMutable_vertsDeleted(_UnderlyingPtr);
            }
        }

        ///< Number deleted faces
        public new unsafe ref int FacesDeleted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_GetMutable_facesDeleted", ExactSpelling = true)]
                extern static int *__MR_DecimateResult_GetMutable_facesDeleted(_Underlying *_this);
                return ref *__MR_DecimateResult_GetMutable_facesDeleted(_UnderlyingPtr);
            }
        }

        /// for DecimateStrategy::MinimizeError:
        ///    estimated distance deviation of decimated mesh from the original mesh
        /// for DecimateStrategy::ShortestEdgeFirst:
        ///    the shortest remaining edge in the mesh
        public new unsafe ref float ErrorIntroduced
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_GetMutable_errorIntroduced", ExactSpelling = true)]
                extern static float *__MR_DecimateResult_GetMutable_errorIntroduced(_Underlying *_this);
                return ref *__MR_DecimateResult_GetMutable_errorIntroduced(_UnderlyingPtr);
            }
        }

        /// whether the algorithm was cancelled by the callback
        public new unsafe ref bool Cancelled
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_GetMutable_cancelled", ExactSpelling = true)]
                extern static bool *__MR_DecimateResult_GetMutable_cancelled(_Underlying *_this);
                return ref *__MR_DecimateResult_GetMutable_cancelled(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DecimateResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_DefaultConstruct();
            _UnderlyingPtr = __MR_DecimateResult_DefaultConstruct();
        }

        /// Constructs `MR::DecimateResult` elementwise.
        public unsafe DecimateResult(int vertsDeleted, int facesDeleted, float errorIntroduced, bool cancelled) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_ConstructFrom(int vertsDeleted, int facesDeleted, float errorIntroduced, byte cancelled);
            _UnderlyingPtr = __MR_DecimateResult_ConstructFrom(vertsDeleted, facesDeleted, errorIntroduced, cancelled ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::DecimateResult::DecimateResult`.
        public unsafe DecimateResult(MR.Const_DecimateResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_ConstructFromAnother(MR.DecimateResult._Underlying *_other);
            _UnderlyingPtr = __MR_DecimateResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::DecimateResult::operator=`.
        public unsafe MR.DecimateResult Assign(MR.Const_DecimateResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DecimateResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DecimateResult._Underlying *__MR_DecimateResult_AssignFromAnother(_Underlying *_this, MR.DecimateResult._Underlying *_other);
            return new(__MR_DecimateResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DecimateResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DecimateResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimateResult`/`Const_DecimateResult` directly.
    public class _InOptMut_DecimateResult
    {
        public DecimateResult? Opt;

        public _InOptMut_DecimateResult() {}
        public _InOptMut_DecimateResult(DecimateResult value) {Opt = value;}
        public static implicit operator _InOptMut_DecimateResult(DecimateResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `DecimateResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DecimateResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DecimateResult`/`Const_DecimateResult` to pass it to the function.
    public class _InOptConst_DecimateResult
    {
        public Const_DecimateResult? Opt;

        public _InOptConst_DecimateResult() {}
        public _InOptConst_DecimateResult(Const_DecimateResult value) {Opt = value;}
        public static implicit operator _InOptConst_DecimateResult(Const_DecimateResult value) {return new(value);}
    }

    /// Generated from class `MR::ResolveMeshDegenSettings`.
    /// This is the const half of the class.
    public class Const_ResolveMeshDegenSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ResolveMeshDegenSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_ResolveMeshDegenSettings_Destroy(_Underlying *_this);
            __MR_ResolveMeshDegenSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ResolveMeshDegenSettings() {Dispose(false);}

        /// maximum permitted deviation from the original surface
        public unsafe float MaxDeviation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Get_maxDeviation", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_Get_maxDeviation(_Underlying *_this);
                return *__MR_ResolveMeshDegenSettings_Get_maxDeviation(_UnderlyingPtr);
            }
        }

        /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
        public unsafe float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Get_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_Get_tinyEdgeLength(_Underlying *_this);
                return *__MR_ResolveMeshDegenSettings_Get_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// Permit edge flips if it does not change dihedral angle more than on this value
        public unsafe float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Get_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_Get_maxAngleChange(_Underlying *_this);
                return *__MR_ResolveMeshDegenSettings_Get_maxAngleChange(_UnderlyingPtr);
            }
        }

        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public unsafe float CriticalAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Get_criticalAspectRatio", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_Get_criticalAspectRatio(_Underlying *_this);
                return *__MR_ResolveMeshDegenSettings_Get_criticalAspectRatio(_UnderlyingPtr);
            }
        }

        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_Get_stabilizer(_Underlying *_this);
                return *__MR_ResolveMeshDegenSettings_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// degenerations will be fixed only in given region, which is updated during the processing
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_ResolveMeshDegenSettings_Get_region(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ResolveMeshDegenSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_ResolveMeshDegenSettings_DefaultConstruct();
        }

        /// Constructs `MR::ResolveMeshDegenSettings` elementwise.
        public unsafe Const_ResolveMeshDegenSettings(float maxDeviation, float tinyEdgeLength, float maxAngleChange, float criticalAspectRatio, float stabilizer, MR.FaceBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_ConstructFrom(float maxDeviation, float tinyEdgeLength, float maxAngleChange, float criticalAspectRatio, float stabilizer, MR.FaceBitSet._Underlying *region);
            _UnderlyingPtr = __MR_ResolveMeshDegenSettings_ConstructFrom(maxDeviation, tinyEdgeLength, maxAngleChange, criticalAspectRatio, stabilizer, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ResolveMeshDegenSettings::ResolveMeshDegenSettings`.
        public unsafe Const_ResolveMeshDegenSettings(MR.Const_ResolveMeshDegenSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_ConstructFromAnother(MR.ResolveMeshDegenSettings._Underlying *_other);
            _UnderlyingPtr = __MR_ResolveMeshDegenSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ResolveMeshDegenSettings`.
    /// This is the non-const half of the class.
    public class ResolveMeshDegenSettings : Const_ResolveMeshDegenSettings
    {
        internal unsafe ResolveMeshDegenSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// maximum permitted deviation from the original surface
        public new unsafe ref float MaxDeviation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_GetMutable_maxDeviation", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_GetMutable_maxDeviation(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_GetMutable_maxDeviation(_UnderlyingPtr);
            }
        }

        /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
        public new unsafe ref float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_GetMutable_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_GetMutable_tinyEdgeLength(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_GetMutable_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// Permit edge flips if it does not change dihedral angle more than on this value
        public new unsafe ref float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_GetMutable_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_GetMutable_maxAngleChange(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_GetMutable_maxAngleChange(_UnderlyingPtr);
            }
        }

        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public new unsafe ref float CriticalAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_GetMutable_criticalAspectRatio", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_GetMutable_criticalAspectRatio(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_GetMutable_criticalAspectRatio(_UnderlyingPtr);
            }
        }

        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_ResolveMeshDegenSettings_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// degenerations will be fixed only in given region, which is updated during the processing
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_ResolveMeshDegenSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_ResolveMeshDegenSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ResolveMeshDegenSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_ResolveMeshDegenSettings_DefaultConstruct();
        }

        /// Constructs `MR::ResolveMeshDegenSettings` elementwise.
        public unsafe ResolveMeshDegenSettings(float maxDeviation, float tinyEdgeLength, float maxAngleChange, float criticalAspectRatio, float stabilizer, MR.FaceBitSet? region) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_ConstructFrom(float maxDeviation, float tinyEdgeLength, float maxAngleChange, float criticalAspectRatio, float stabilizer, MR.FaceBitSet._Underlying *region);
            _UnderlyingPtr = __MR_ResolveMeshDegenSettings_ConstructFrom(maxDeviation, tinyEdgeLength, maxAngleChange, criticalAspectRatio, stabilizer, region is not null ? region._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ResolveMeshDegenSettings::ResolveMeshDegenSettings`.
        public unsafe ResolveMeshDegenSettings(MR.Const_ResolveMeshDegenSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_ConstructFromAnother(MR.ResolveMeshDegenSettings._Underlying *_other);
            _UnderlyingPtr = __MR_ResolveMeshDegenSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ResolveMeshDegenSettings::operator=`.
        public unsafe MR.ResolveMeshDegenSettings Assign(MR.Const_ResolveMeshDegenSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ResolveMeshDegenSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ResolveMeshDegenSettings._Underlying *__MR_ResolveMeshDegenSettings_AssignFromAnother(_Underlying *_this, MR.ResolveMeshDegenSettings._Underlying *_other);
            return new(__MR_ResolveMeshDegenSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ResolveMeshDegenSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ResolveMeshDegenSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ResolveMeshDegenSettings`/`Const_ResolveMeshDegenSettings` directly.
    public class _InOptMut_ResolveMeshDegenSettings
    {
        public ResolveMeshDegenSettings? Opt;

        public _InOptMut_ResolveMeshDegenSettings() {}
        public _InOptMut_ResolveMeshDegenSettings(ResolveMeshDegenSettings value) {Opt = value;}
        public static implicit operator _InOptMut_ResolveMeshDegenSettings(ResolveMeshDegenSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `ResolveMeshDegenSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ResolveMeshDegenSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ResolveMeshDegenSettings`/`Const_ResolveMeshDegenSettings` to pass it to the function.
    public class _InOptConst_ResolveMeshDegenSettings
    {
        public Const_ResolveMeshDegenSettings? Opt;

        public _InOptConst_ResolveMeshDegenSettings() {}
        public _InOptConst_ResolveMeshDegenSettings(Const_ResolveMeshDegenSettings value) {Opt = value;}
        public static implicit operator _InOptConst_ResolveMeshDegenSettings(Const_ResolveMeshDegenSettings value) {return new(value);}
    }

    /// Generated from class `MR::RemeshSettings`.
    /// This is the const half of the class.
    public class Const_RemeshSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RemeshSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_RemeshSettings_Destroy(_Underlying *_this);
            __MR_RemeshSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RemeshSettings() {Dispose(false);}

        /// the algorithm will try to keep the length of all edges close to this value,
        /// splitting the edges longer than targetEdgeLen, and then eliminating the edges shorter than targetEdgeLen
        public unsafe float TargetEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_targetEdgeLen", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_Get_targetEdgeLen(_Underlying *_this);
                return *__MR_RemeshSettings_Get_targetEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed during subdivision
        public unsafe int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_RemeshSettings_Get_maxEdgeSplits(_Underlying *_this);
                return *__MR_RemeshSettings_Get_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value
        public unsafe float MaxAngleChangeAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_maxAngleChangeAfterFlip", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_Get_maxAngleChangeAfterFlip(_Underlying *_this);
                return *__MR_RemeshSettings_Get_maxAngleChangeAfterFlip(_UnderlyingPtr);
            }
        }

        /// Allows or prohibits splitting and/or collapse boundary edges
        /// it recommended to keep default value here for better quality
        public unsafe bool FrozenBoundary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_frozenBoundary", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_Get_frozenBoundary(_Underlying *_this);
                return *__MR_RemeshSettings_Get_frozenBoundary(_UnderlyingPtr);
            }
        }

        /// Maximal shift of a boundary during one edge collapse
        /// only makes sense if `frozenBoundary=false`
        public unsafe float MaxBdShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_maxBdShift", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_Get_maxBdShift(_Underlying *_this);
                return *__MR_RemeshSettings_Get_maxBdShift(_UnderlyingPtr);
            }
        }

        /// This option in subdivision works best for natural surfaces, where all triangles are close to equilateral and have similar area,
        /// and no sharp edges in between
        public unsafe bool UseCurvature
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_useCurvature", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_Get_useCurvature(_Underlying *_this);
                return *__MR_RemeshSettings_Get_useCurvature(_UnderlyingPtr);
            }
        }

        /// An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value.
        /// So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization.
        /// Please set it to a smaller value only if frozenBoundary==true, otherwise many narrow triangles can appear near border
        public unsafe float MaxSplittableTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_maxSplittableTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_Get_maxSplittableTriAspectRatio(_Underlying *_this);
                return *__MR_RemeshSettings_Get_maxSplittableTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// the number of iterations of final relaxation of mesh vertices;
        /// few iterations can give almost perfect uniformity of the vertices and edge lengths but deviate from the original surface
        public unsafe int FinalRelaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_finalRelaxIters", ExactSpelling = true)]
                extern static int *__MR_RemeshSettings_Get_finalRelaxIters(_Underlying *_this);
                return *__MR_RemeshSettings_Get_finalRelaxIters(_UnderlyingPtr);
            }
        }

        /// if true prevents the surface from shrinkage after many iterations
        public unsafe bool FinalRelaxNoShrinkage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_finalRelaxNoShrinkage", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_Get_finalRelaxNoShrinkage(_Underlying *_this);
                return *__MR_RemeshSettings_Get_finalRelaxNoShrinkage(_UnderlyingPtr);
            }
        }

        /// Region on mesh to be changed, it is updated during the operation
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_RemeshSettings_Get_region(_Underlying *_this);
                return ref *__MR_RemeshSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped or collapsed, but they can be replaced during collapse of nearby edges so it is updated during the operation;
        /// also the vertices incident to these edges are excluded from relaxation
        public unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_notFlippable", ExactSpelling = true)]
                extern static void **__MR_RemeshSettings_Get_notFlippable(_Underlying *_this);
                return ref *__MR_RemeshSettings_Get_notFlippable(_UnderlyingPtr);
            }
        }

        ///  whether to pack mesh at the end
        public unsafe bool PackMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_packMesh", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_Get_packMesh(_Underlying *_this);
                return *__MR_RemeshSettings_Get_packMesh(_UnderlyingPtr);
            }
        }

        /// if true, then every new vertex after subdivision will be projected on the original mesh (before smoothing);
        /// this does not affect the vertices moved on other stages of the processing
        public unsafe bool ProjectOnOriginalMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_projectOnOriginalMesh", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_Get_projectOnOriginalMesh(_Underlying *_this);
                return *__MR_RemeshSettings_Get_projectOnOriginalMesh(_UnderlyingPtr);
            }
        }

        /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_RemeshSettings_Get_onEdgeSplit(_Underlying *_this);
                return new(__MR_RemeshSettings_Get_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if valid (e1) is given then dest(e) = dest(e1) and their origins are in different ends of collapsing edge, e1 shall take the place of e
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeDel
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_onEdgeDel", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_RemeshSettings_Get_onEdgeDel(_Underlying *_this);
                return new(__MR_RemeshSettings_Get_onEdgeDel(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public unsafe MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector3fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_RemeshSettings_Get_preCollapse(_Underlying *_this);
                return new(__MR_RemeshSettings_Get_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_Get_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_RemeshSettings_Get_progressCallback(_Underlying *_this);
                return new(__MR_RemeshSettings_Get_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RemeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_RemeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::RemeshSettings` elementwise.
        public unsafe Const_RemeshSettings(float targetEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, bool frozenBoundary, float maxBdShift, bool useCurvature, float maxSplittableTriAspectRatio, int finalRelaxIters, bool finalRelaxNoShrinkage, MR.FaceBitSet? region, MR.UndirectedEdgeBitSet? notFlippable, bool packMesh, bool projectOnOriginalMesh, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeDel, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector3fRef preCollapse, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_ConstructFrom(float targetEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, byte frozenBoundary, float maxBdShift, byte useCurvature, float maxSplittableTriAspectRatio, int finalRelaxIters, byte finalRelaxNoShrinkage, MR.FaceBitSet._Underlying *region, MR.UndirectedEdgeBitSet._Underlying *notFlippable, byte packMesh, byte projectOnOriginalMesh, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.Misc._PassBy onEdgeDel_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeDel, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *preCollapse, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
            _UnderlyingPtr = __MR_RemeshSettings_ConstructFrom(targetEdgeLen, maxEdgeSplits, maxAngleChangeAfterFlip, frozenBoundary ? (byte)1 : (byte)0, maxBdShift, useCurvature ? (byte)1 : (byte)0, maxSplittableTriAspectRatio, finalRelaxIters, finalRelaxNoShrinkage ? (byte)1 : (byte)0, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, packMesh ? (byte)1 : (byte)0, projectOnOriginalMesh ? (byte)1 : (byte)0, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, onEdgeDel.PassByMode, onEdgeDel.Value is not null ? onEdgeDel.Value._UnderlyingPtr : null, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::RemeshSettings::RemeshSettings`.
        public unsafe Const_RemeshSettings(MR._ByValue_RemeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RemeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_RemeshSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::RemeshSettings`.
    /// This is the non-const half of the class.
    public class RemeshSettings : Const_RemeshSettings
    {
        internal unsafe RemeshSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the algorithm will try to keep the length of all edges close to this value,
        /// splitting the edges longer than targetEdgeLen, and then eliminating the edges shorter than targetEdgeLen
        public new unsafe ref float TargetEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_targetEdgeLen", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_GetMutable_targetEdgeLen(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_targetEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed during subdivision
        public new unsafe ref int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_RemeshSettings_GetMutable_maxEdgeSplits(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value
        public new unsafe ref float MaxAngleChangeAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_maxAngleChangeAfterFlip", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_GetMutable_maxAngleChangeAfterFlip(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_maxAngleChangeAfterFlip(_UnderlyingPtr);
            }
        }

        /// Allows or prohibits splitting and/or collapse boundary edges
        /// it recommended to keep default value here for better quality
        public new unsafe ref bool FrozenBoundary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_frozenBoundary", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_GetMutable_frozenBoundary(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_frozenBoundary(_UnderlyingPtr);
            }
        }

        /// Maximal shift of a boundary during one edge collapse
        /// only makes sense if `frozenBoundary=false`
        public new unsafe ref float MaxBdShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_maxBdShift", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_GetMutable_maxBdShift(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_maxBdShift(_UnderlyingPtr);
            }
        }

        /// This option in subdivision works best for natural surfaces, where all triangles are close to equilateral and have similar area,
        /// and no sharp edges in between
        public new unsafe ref bool UseCurvature
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_useCurvature", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_GetMutable_useCurvature(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_useCurvature(_UnderlyingPtr);
            }
        }

        /// An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value.
        /// So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization.
        /// Please set it to a smaller value only if frozenBoundary==true, otherwise many narrow triangles can appear near border
        public new unsafe ref float MaxSplittableTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_maxSplittableTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_RemeshSettings_GetMutable_maxSplittableTriAspectRatio(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_maxSplittableTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// the number of iterations of final relaxation of mesh vertices;
        /// few iterations can give almost perfect uniformity of the vertices and edge lengths but deviate from the original surface
        public new unsafe ref int FinalRelaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_finalRelaxIters", ExactSpelling = true)]
                extern static int *__MR_RemeshSettings_GetMutable_finalRelaxIters(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_finalRelaxIters(_UnderlyingPtr);
            }
        }

        /// if true prevents the surface from shrinkage after many iterations
        public new unsafe ref bool FinalRelaxNoShrinkage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_finalRelaxNoShrinkage", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_GetMutable_finalRelaxNoShrinkage(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_finalRelaxNoShrinkage(_UnderlyingPtr);
            }
        }

        /// Region on mesh to be changed, it is updated during the operation
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_RemeshSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped or collapsed, but they can be replaced during collapse of nearby edges so it is updated during the operation;
        /// also the vertices incident to these edges are excluded from relaxation
        public new unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_notFlippable", ExactSpelling = true)]
                extern static void **__MR_RemeshSettings_GetMutable_notFlippable(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_notFlippable(_UnderlyingPtr);
            }
        }

        ///  whether to pack mesh at the end
        public new unsafe ref bool PackMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_packMesh", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_GetMutable_packMesh(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_packMesh(_UnderlyingPtr);
            }
        }

        /// if true, then every new vertex after subdivision will be projected on the original mesh (before smoothing);
        /// this does not affect the vertices moved on other stages of the processing
        public new unsafe ref bool ProjectOnOriginalMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_projectOnOriginalMesh", ExactSpelling = true)]
                extern static bool *__MR_RemeshSettings_GetMutable_projectOnOriginalMesh(_Underlying *_this);
                return ref *__MR_RemeshSettings_GetMutable_projectOnOriginalMesh(_UnderlyingPtr);
            }
        }

        /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_RemeshSettings_GetMutable_onEdgeSplit(_Underlying *_this);
                return new(__MR_RemeshSettings_GetMutable_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if valid (e1) is given then dest(e) = dest(e1) and their origins are in different ends of collapsing edge, e1 shall take the place of e
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeDel
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_onEdgeDel", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_RemeshSettings_GetMutable_onEdgeDel(_Underlying *_this);
                return new(__MR_RemeshSettings_GetMutable_onEdgeDel(_UnderlyingPtr), is_owning: false);
            }
        }

        /**
        * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
        * \details It receives the edge being collapsed: its destination vertex will disappear,
        * and its origin vertex will get new position (provided as the second argument) after collapse;
        * If the callback returns false, then the collapse is prohibited
        */
        public new unsafe MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef PreCollapse
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_preCollapse", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *__MR_RemeshSettings_GetMutable_preCollapse(_Underlying *_this);
                return new(__MR_RemeshSettings_GetMutable_preCollapse(_UnderlyingPtr), is_owning: false);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_GetMutable_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_RemeshSettings_GetMutable_progressCallback(_Underlying *_this);
                return new(__MR_RemeshSettings_GetMutable_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RemeshSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_RemeshSettings_DefaultConstruct();
        }

        /// Constructs `MR::RemeshSettings` elementwise.
        public unsafe RemeshSettings(float targetEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, bool frozenBoundary, float maxBdShift, bool useCurvature, float maxSplittableTriAspectRatio, int finalRelaxIters, bool finalRelaxNoShrinkage, MR.FaceBitSet? region, MR.UndirectedEdgeBitSet? notFlippable, bool packMesh, bool projectOnOriginalMesh, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeDel, MR.Std._ByValue_Function_BoolFuncFromMREdgeIdConstMRVector3fRef preCollapse, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_ConstructFrom(float targetEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, byte frozenBoundary, float maxBdShift, byte useCurvature, float maxSplittableTriAspectRatio, int finalRelaxIters, byte finalRelaxNoShrinkage, MR.FaceBitSet._Underlying *region, MR.UndirectedEdgeBitSet._Underlying *notFlippable, byte packMesh, byte projectOnOriginalMesh, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.Misc._PassBy onEdgeDel_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeDel, MR.Misc._PassBy preCollapse_pass_by, MR.Std.Function_BoolFuncFromMREdgeIdConstMRVector3fRef._Underlying *preCollapse, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
            _UnderlyingPtr = __MR_RemeshSettings_ConstructFrom(targetEdgeLen, maxEdgeSplits, maxAngleChangeAfterFlip, frozenBoundary ? (byte)1 : (byte)0, maxBdShift, useCurvature ? (byte)1 : (byte)0, maxSplittableTriAspectRatio, finalRelaxIters, finalRelaxNoShrinkage ? (byte)1 : (byte)0, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, packMesh ? (byte)1 : (byte)0, projectOnOriginalMesh ? (byte)1 : (byte)0, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, onEdgeDel.PassByMode, onEdgeDel.Value is not null ? onEdgeDel.Value._UnderlyingPtr : null, preCollapse.PassByMode, preCollapse.Value is not null ? preCollapse.Value._UnderlyingPtr : null, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::RemeshSettings::RemeshSettings`.
        public unsafe RemeshSettings(MR._ByValue_RemeshSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.RemeshSettings._Underlying *_other);
            _UnderlyingPtr = __MR_RemeshSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::RemeshSettings::operator=`.
        public unsafe MR.RemeshSettings Assign(MR._ByValue_RemeshSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RemeshSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RemeshSettings._Underlying *__MR_RemeshSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.RemeshSettings._Underlying *_other);
            return new(__MR_RemeshSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `RemeshSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `RemeshSettings`/`Const_RemeshSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_RemeshSettings
    {
        internal readonly Const_RemeshSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_RemeshSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_RemeshSettings(Const_RemeshSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_RemeshSettings(Const_RemeshSettings arg) {return new(arg);}
        public _ByValue_RemeshSettings(MR.Misc._Moved<RemeshSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_RemeshSettings(MR.Misc._Moved<RemeshSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `RemeshSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RemeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RemeshSettings`/`Const_RemeshSettings` directly.
    public class _InOptMut_RemeshSettings
    {
        public RemeshSettings? Opt;

        public _InOptMut_RemeshSettings() {}
        public _InOptMut_RemeshSettings(RemeshSettings value) {Opt = value;}
        public static implicit operator _InOptMut_RemeshSettings(RemeshSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `RemeshSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RemeshSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RemeshSettings`/`Const_RemeshSettings` to pass it to the function.
    public class _InOptConst_RemeshSettings
    {
        public Const_RemeshSettings? Opt;

        public _InOptConst_RemeshSettings() {}
        public _InOptConst_RemeshSettings(Const_RemeshSettings value) {Opt = value;}
        public static implicit operator _InOptConst_RemeshSettings(Const_RemeshSettings value) {return new(value);}
    }

    /**
    * \brief Performs mesh simplification in mesh region according to the settings
    *
    * \snippet cpp-examples/MeshDecimate.dox.cpp 0
    *
    * \image html decimate/decimate_before.png "Before" width = 350cm
    * \image html decimate/decimate_after.png "After" width = 350cm
    */
    /// Generated from function `MR::decimateMesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.DecimateResult DecimateMesh(MR.Mesh mesh, MR.Const_DecimateSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decimateMesh", ExactSpelling = true)]
        extern static MR.DecimateResult._Underlying *__MR_decimateMesh(MR.Mesh._Underlying *mesh, MR.Const_DecimateSettings._Underlying *settings);
        return new(__MR_decimateMesh(mesh._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true);
    }

    /// Performs mesh simplification with per-element attributes according to given settings;
    /// \detail settings.region must be null, and real simplification region will be data face selection (or whole mesh if no face selection)
    /// Generated from function `MR::decimateObjectMeshData`.
    public static unsafe MR.DecimateResult DecimateObjectMeshData(MR.ObjectMeshData data, MR.Const_DecimateSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_decimateObjectMeshData", ExactSpelling = true)]
        extern static MR.DecimateResult._Underlying *__MR_decimateObjectMeshData(MR.ObjectMeshData._Underlying *data, MR.Const_DecimateSettings._Underlying *settings);
        return new(__MR_decimateObjectMeshData(data._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true);
    }

    /// returns the data of decimated mesh given ObjectMesh (which remains unchanged) and decimation parameters
    /// Generated from function `MR::makeDecimatedObjectMeshData`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRObjectMeshData> MakeDecimatedObjectMeshData(MR.Const_ObjectMesh obj, MR.Const_DecimateSettings settings, MR.DecimateResult? outRes = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDecimatedObjectMeshData", ExactSpelling = true)]
        extern static MR.Std.Optional_MRObjectMeshData._Underlying *__MR_makeDecimatedObjectMeshData(MR.Const_ObjectMesh._Underlying *obj, MR.Const_DecimateSettings._Underlying *settings, MR.DecimateResult._Underlying *outRes);
        return MR.Misc.Move(new MR.Std.Optional_MRObjectMeshData(__MR_makeDecimatedObjectMeshData(obj._UnderlyingPtr, settings._UnderlyingPtr, outRes is not null ? outRes._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Computes quadratic form at given vertex of the initial surface before decimation
    *
    */
    /// Generated from function `MR::computeFormAtVertex`.
    public static unsafe MR.QuadraticForm3f ComputeFormAtVertex(MR.Const_MeshPart mp, MR.VertId v, float stabilizer, bool angleWeigted, MR.Const_UndirectedEdgeBitSet? creases = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeFormAtVertex", ExactSpelling = true)]
        extern static MR.QuadraticForm3f._Underlying *__MR_computeFormAtVertex(MR.Const_MeshPart._Underlying *mp, MR.VertId v, float stabilizer, byte angleWeigted, MR.Const_UndirectedEdgeBitSet._Underlying *creases);
        return new(__MR_computeFormAtVertex(mp._UnderlyingPtr, v, stabilizer, angleWeigted ? (byte)1 : (byte)0, creases is not null ? creases._UnderlyingPtr : null), is_owning: true);
    }

    /**
    * \brief Computes quadratic forms at every vertex of mesh part before decimation
    *
    */
    /// Generated from function `MR::computeFormsAtVertices`.
    public static unsafe MR.Misc._Moved<MR.Vector_MRQuadraticForm3f_MRVertId> ComputeFormsAtVertices(MR.Const_MeshPart mp, float stabilizer, bool angleWeigted, MR.Const_UndirectedEdgeBitSet? creases = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeFormsAtVertices", ExactSpelling = true)]
        extern static MR.Vector_MRQuadraticForm3f_MRVertId._Underlying *__MR_computeFormsAtVertices(MR.Const_MeshPart._Underlying *mp, float stabilizer, byte angleWeigted, MR.Const_UndirectedEdgeBitSet._Underlying *creases);
        return MR.Misc.Move(new MR.Vector_MRQuadraticForm3f_MRVertId(__MR_computeFormsAtVertices(mp._UnderlyingPtr, stabilizer, angleWeigted ? (byte)1 : (byte)0, creases is not null ? creases._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief returns given subdivision part of all valid faces;
    * parallel threads shall be able to safely modify these bits because they do not share any block with other parts
    *
    */
    /// Generated from function `MR::getSubdividePart`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetSubdividePart(MR.Const_FaceBitSet valids, ulong subdivideParts, ulong myPart)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getSubdividePart", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_getSubdividePart(MR.Const_FaceBitSet._Underlying *valids, ulong subdivideParts, ulong myPart);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_getSubdividePart(valids._UnderlyingPtr, subdivideParts, myPart), is_owning: true));
    }

    /**
    * \brief Removes degenerate triangles in a mesh by calling decimateMesh function with appropriate settings
    * \details consider using \ref fixMeshDegeneracies for more complex cases
    *
    * \return true if the mesh has been changed
    *
    * \sa \ref decimateMesh
    */
    /// Generated from function `MR::resolveMeshDegenerations`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe bool ResolveMeshDegenerations(MR.Mesh mesh, MR.Const_ResolveMeshDegenSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_resolveMeshDegenerations", ExactSpelling = true)]
        extern static byte __MR_resolveMeshDegenerations(MR.Mesh._Underlying *mesh, MR.Const_ResolveMeshDegenSettings._Underlying *settings);
        return __MR_resolveMeshDegenerations(mesh._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null) != 0;
    }

    // Splits too long and eliminates too short edges from the mesh
    /// Generated from function `MR::remesh`.
    public static unsafe bool Remesh(MR.Mesh mesh, MR.Const_RemeshSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_remesh", ExactSpelling = true)]
        extern static byte __MR_remesh(MR.Mesh._Underlying *mesh, MR.Const_RemeshSettings._Underlying *settings);
        return __MR_remesh(mesh._UnderlyingPtr, settings._UnderlyingPtr) != 0;
    }
}
