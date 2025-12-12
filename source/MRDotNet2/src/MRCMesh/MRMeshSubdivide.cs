public static partial class MR
{
    /// Generated from class `MR::SubdivideSettings`.
    /// This is the const half of the class.
    public class Const_SubdivideSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SubdivideSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_SubdivideSettings_Destroy(_Underlying *_this);
            __MR_SubdivideSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SubdivideSettings() {Dispose(false);}

        /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
        public unsafe float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_maxEdgeLen(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed
        public unsafe int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_SubdivideSettings_Get_maxEdgeSplits(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation
        public unsafe float MaxDeviationAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maxDeviationAfterFlip", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_maxDeviationAfterFlip(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_maxDeviationAfterFlip(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
        public unsafe float MaxAngleChangeAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maxAngleChangeAfterFlip", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_maxAngleChangeAfterFlip(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_maxAngleChangeAfterFlip(_UnderlyingPtr);
            }
        }

        /// If this value is less than FLT_MAX then edge flips will
        /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
        /// Unit: rad
        public unsafe float CriticalAspectRatioFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_criticalAspectRatioFlip", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_criticalAspectRatioFlip(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_criticalAspectRatioFlip(_UnderlyingPtr);
            }
        }

        /// Region on mesh to be subdivided, it is updated during the operation
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_Get_region(_Underlying *_this);
                return ref *__MR_SubdivideSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// Additional region to update during subdivision: if a face from here is split, it is replaced with new sub-faces;
        /// note that Subdivide can split faces even outside of main \p region, so it might be necessary to update another region
        public unsafe ref void * MaintainRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maintainRegion", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_Get_maintainRegion(_Underlying *_this);
                return ref *__MR_SubdivideSettings_Get_maintainRegion(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
        public unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_notFlippable", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_Get_notFlippable(_Underlying *_this);
                return ref *__MR_SubdivideSettings_Get_notFlippable(_UnderlyingPtr);
            }
        }

        /// New vertices appeared during subdivision will be added here
        public unsafe ref void * NewVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_newVerts", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_Get_newVerts(_Underlying *_this);
                return ref *__MR_SubdivideSettings_Get_newVerts(_UnderlyingPtr);
            }
        }

        /// If false do not touch border edges (cannot subdivide lone faces)\n
        /// use \ref MR::findRegionOuterFaces to find boundary faces
        public unsafe bool SubdivideBorder
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_subdivideBorder", ExactSpelling = true)]
                extern static bool *__MR_SubdivideSettings_Get_subdivideBorder(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_subdivideBorder(_UnderlyingPtr);
            }
        }

        /// The subdivision stops as soon as all triangles (in the region) have aspect ratio below or equal to this value
        public unsafe float MaxTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maxTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_maxTriAspectRatio(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_maxTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value.
        /// So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization.
        /// Please set it to a smaller value only if subdivideBorder==false, otherwise many narrow triangles can appear near border
        public unsafe float MaxSplittableTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_maxSplittableTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_maxSplittableTriAspectRatio(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_maxSplittableTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// Puts new vertices so that they form a smooth surface together with existing vertices.
        /// This option works best for natural surfaces without sharp edges in between triangles
        public unsafe bool SmoothMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_smoothMode", ExactSpelling = true)]
                extern static bool *__MR_SubdivideSettings_Get_smoothMode(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_smoothMode(_UnderlyingPtr);
            }
        }

        // 30 degrees
        public unsafe float MinSharpDihedralAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_minSharpDihedralAngle", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_Get_minSharpDihedralAngle(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_minSharpDihedralAngle(_UnderlyingPtr);
            }
        }

        /// if true, then every new vertex will be projected on the original mesh (before smoothing)
        public unsafe bool ProjectOnOriginalMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_projectOnOriginalMesh", ExactSpelling = true)]
                extern static bool *__MR_SubdivideSettings_Get_projectOnOriginalMesh(_Underlying *_this);
                return *__MR_SubdivideSettings_Get_projectOnOriginalMesh(_UnderlyingPtr);
            }
        }

        /// this function is called each time edge (e) is going to split, if it returns false then this split will be skipped
        public unsafe MR.Std.Const_Function_BoolFuncFromMREdgeId BeforeEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_beforeEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMREdgeId._Underlying *__MR_SubdivideSettings_Get_beforeEdgeSplit(_Underlying *_this);
                return new(__MR_SubdivideSettings_Get_beforeEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time a new vertex has been created, but before the ring is made Delone
        public unsafe MR.Std.Const_Function_VoidFuncFromMRVertId OnVertCreated
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_onVertCreated", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMRVertId._Underlying *__MR_SubdivideSettings_Get_onVertCreated(_Underlying *_this);
                return new(__MR_SubdivideSettings_Get_onVertCreated(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_SubdivideSettings_Get_onEdgeSplit(_Underlying *_this);
                return new(__MR_SubdivideSettings_Get_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_Get_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_SubdivideSettings_Get_progressCallback(_Underlying *_this);
                return new(__MR_SubdivideSettings_Get_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SubdivideSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SubdivideSettings_DefaultConstruct();
        }

        /// Constructs `MR::SubdivideSettings` elementwise.
        public unsafe Const_SubdivideSettings(float maxEdgeLen, int maxEdgeSplits, float maxDeviationAfterFlip, float maxAngleChangeAfterFlip, float criticalAspectRatioFlip, MR.FaceBitSet? region, MR.FaceBitSet? maintainRegion, MR.UndirectedEdgeBitSet? notFlippable, MR.VertBitSet? newVerts, bool subdivideBorder, float maxTriAspectRatio, float maxSplittableTriAspectRatio, bool smoothMode, float minSharpDihedralAngle, bool projectOnOriginalMesh, MR.Std._ByValue_Function_BoolFuncFromMREdgeId beforeEdgeSplit, MR.Std._ByValue_Function_VoidFuncFromMRVertId onVertCreated, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_ConstructFrom(float maxEdgeLen, int maxEdgeSplits, float maxDeviationAfterFlip, float maxAngleChangeAfterFlip, float criticalAspectRatioFlip, MR.FaceBitSet._Underlying *region, MR.FaceBitSet._Underlying *maintainRegion, MR.UndirectedEdgeBitSet._Underlying *notFlippable, MR.VertBitSet._Underlying *newVerts, byte subdivideBorder, float maxTriAspectRatio, float maxSplittableTriAspectRatio, byte smoothMode, float minSharpDihedralAngle, byte projectOnOriginalMesh, MR.Misc._PassBy beforeEdgeSplit_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *beforeEdgeSplit, MR.Misc._PassBy onVertCreated_pass_by, MR.Std.Function_VoidFuncFromMRVertId._Underlying *onVertCreated, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
            _UnderlyingPtr = __MR_SubdivideSettings_ConstructFrom(maxEdgeLen, maxEdgeSplits, maxDeviationAfterFlip, maxAngleChangeAfterFlip, criticalAspectRatioFlip, region is not null ? region._UnderlyingPtr : null, maintainRegion is not null ? maintainRegion._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, newVerts is not null ? newVerts._UnderlyingPtr : null, subdivideBorder ? (byte)1 : (byte)0, maxTriAspectRatio, maxSplittableTriAspectRatio, smoothMode ? (byte)1 : (byte)0, minSharpDihedralAngle, projectOnOriginalMesh ? (byte)1 : (byte)0, beforeEdgeSplit.PassByMode, beforeEdgeSplit.Value is not null ? beforeEdgeSplit.Value._UnderlyingPtr : null, onVertCreated.PassByMode, onVertCreated.Value is not null ? onVertCreated.Value._UnderlyingPtr : null, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SubdivideSettings::SubdivideSettings`.
        public unsafe Const_SubdivideSettings(MR._ByValue_SubdivideSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SubdivideSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SubdivideSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::SubdivideSettings`.
    /// This is the non-const half of the class.
    public class SubdivideSettings : Const_SubdivideSettings
    {
        internal unsafe SubdivideSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
        public new unsafe ref float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_maxEdgeLen(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed
        public new unsafe ref int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_SubdivideSettings_GetMutable_maxEdgeSplits(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation
        public new unsafe ref float MaxDeviationAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maxDeviationAfterFlip", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_maxDeviationAfterFlip(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maxDeviationAfterFlip(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
        public new unsafe ref float MaxAngleChangeAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maxAngleChangeAfterFlip", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_maxAngleChangeAfterFlip(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maxAngleChangeAfterFlip(_UnderlyingPtr);
            }
        }

        /// If this value is less than FLT_MAX then edge flips will
        /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
        /// Unit: rad
        public new unsafe ref float CriticalAspectRatioFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_criticalAspectRatioFlip", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_criticalAspectRatioFlip(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_criticalAspectRatioFlip(_UnderlyingPtr);
            }
        }

        /// Region on mesh to be subdivided, it is updated during the operation
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Additional region to update during subdivision: if a face from here is split, it is replaced with new sub-faces;
        /// note that Subdivide can split faces even outside of main \p region, so it might be necessary to update another region
        public new unsafe ref void * MaintainRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maintainRegion", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_GetMutable_maintainRegion(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maintainRegion(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
        public new unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_notFlippable", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_GetMutable_notFlippable(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_notFlippable(_UnderlyingPtr);
            }
        }

        /// New vertices appeared during subdivision will be added here
        public new unsafe ref void * NewVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_newVerts", ExactSpelling = true)]
                extern static void **__MR_SubdivideSettings_GetMutable_newVerts(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_newVerts(_UnderlyingPtr);
            }
        }

        /// If false do not touch border edges (cannot subdivide lone faces)\n
        /// use \ref MR::findRegionOuterFaces to find boundary faces
        public new unsafe ref bool SubdivideBorder
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_subdivideBorder", ExactSpelling = true)]
                extern static bool *__MR_SubdivideSettings_GetMutable_subdivideBorder(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_subdivideBorder(_UnderlyingPtr);
            }
        }

        /// The subdivision stops as soon as all triangles (in the region) have aspect ratio below or equal to this value
        public new unsafe ref float MaxTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maxTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_maxTriAspectRatio(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maxTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value.
        /// So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization.
        /// Please set it to a smaller value only if subdivideBorder==false, otherwise many narrow triangles can appear near border
        public new unsafe ref float MaxSplittableTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_maxSplittableTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_maxSplittableTriAspectRatio(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_maxSplittableTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// Puts new vertices so that they form a smooth surface together with existing vertices.
        /// This option works best for natural surfaces without sharp edges in between triangles
        public new unsafe ref bool SmoothMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_smoothMode", ExactSpelling = true)]
                extern static bool *__MR_SubdivideSettings_GetMutable_smoothMode(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_smoothMode(_UnderlyingPtr);
            }
        }

        // 30 degrees
        public new unsafe ref float MinSharpDihedralAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_minSharpDihedralAngle", ExactSpelling = true)]
                extern static float *__MR_SubdivideSettings_GetMutable_minSharpDihedralAngle(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_minSharpDihedralAngle(_UnderlyingPtr);
            }
        }

        /// if true, then every new vertex will be projected on the original mesh (before smoothing)
        public new unsafe ref bool ProjectOnOriginalMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_projectOnOriginalMesh", ExactSpelling = true)]
                extern static bool *__MR_SubdivideSettings_GetMutable_projectOnOriginalMesh(_Underlying *_this);
                return ref *__MR_SubdivideSettings_GetMutable_projectOnOriginalMesh(_UnderlyingPtr);
            }
        }

        /// this function is called each time edge (e) is going to split, if it returns false then this split will be skipped
        public new unsafe MR.Std.Function_BoolFuncFromMREdgeId BeforeEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_beforeEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeId._Underlying *__MR_SubdivideSettings_GetMutable_beforeEdgeSplit(_Underlying *_this);
                return new(__MR_SubdivideSettings_GetMutable_beforeEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time a new vertex has been created, but before the ring is made Delone
        public new unsafe MR.Std.Function_VoidFuncFromMRVertId OnVertCreated
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_onVertCreated", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMRVertId._Underlying *__MR_SubdivideSettings_GetMutable_onVertCreated(_Underlying *_this);
                return new(__MR_SubdivideSettings_GetMutable_onVertCreated(_UnderlyingPtr), is_owning: false);
            }
        }

        /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_SubdivideSettings_GetMutable_onEdgeSplit(_Underlying *_this);
                return new(__MR_SubdivideSettings_GetMutable_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// callback to report algorithm progress and cancel it by user request
        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCallback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_GetMutable_progressCallback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_SubdivideSettings_GetMutable_progressCallback(_Underlying *_this);
                return new(__MR_SubdivideSettings_GetMutable_progressCallback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SubdivideSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SubdivideSettings_DefaultConstruct();
        }

        /// Constructs `MR::SubdivideSettings` elementwise.
        public unsafe SubdivideSettings(float maxEdgeLen, int maxEdgeSplits, float maxDeviationAfterFlip, float maxAngleChangeAfterFlip, float criticalAspectRatioFlip, MR.FaceBitSet? region, MR.FaceBitSet? maintainRegion, MR.UndirectedEdgeBitSet? notFlippable, MR.VertBitSet? newVerts, bool subdivideBorder, float maxTriAspectRatio, float maxSplittableTriAspectRatio, bool smoothMode, float minSharpDihedralAngle, bool projectOnOriginalMesh, MR.Std._ByValue_Function_BoolFuncFromMREdgeId beforeEdgeSplit, MR.Std._ByValue_Function_VoidFuncFromMRVertId onVertCreated, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.Std._ByValue_Function_BoolFuncFromFloat progressCallback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_ConstructFrom(float maxEdgeLen, int maxEdgeSplits, float maxDeviationAfterFlip, float maxAngleChangeAfterFlip, float criticalAspectRatioFlip, MR.FaceBitSet._Underlying *region, MR.FaceBitSet._Underlying *maintainRegion, MR.UndirectedEdgeBitSet._Underlying *notFlippable, MR.VertBitSet._Underlying *newVerts, byte subdivideBorder, float maxTriAspectRatio, float maxSplittableTriAspectRatio, byte smoothMode, float minSharpDihedralAngle, byte projectOnOriginalMesh, MR.Misc._PassBy beforeEdgeSplit_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *beforeEdgeSplit, MR.Misc._PassBy onVertCreated_pass_by, MR.Std.Function_VoidFuncFromMRVertId._Underlying *onVertCreated, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.Misc._PassBy progressCallback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCallback);
            _UnderlyingPtr = __MR_SubdivideSettings_ConstructFrom(maxEdgeLen, maxEdgeSplits, maxDeviationAfterFlip, maxAngleChangeAfterFlip, criticalAspectRatioFlip, region is not null ? region._UnderlyingPtr : null, maintainRegion is not null ? maintainRegion._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, newVerts is not null ? newVerts._UnderlyingPtr : null, subdivideBorder ? (byte)1 : (byte)0, maxTriAspectRatio, maxSplittableTriAspectRatio, smoothMode ? (byte)1 : (byte)0, minSharpDihedralAngle, projectOnOriginalMesh ? (byte)1 : (byte)0, beforeEdgeSplit.PassByMode, beforeEdgeSplit.Value is not null ? beforeEdgeSplit.Value._UnderlyingPtr : null, onVertCreated.PassByMode, onVertCreated.Value is not null ? onVertCreated.Value._UnderlyingPtr : null, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, progressCallback.PassByMode, progressCallback.Value is not null ? progressCallback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SubdivideSettings::SubdivideSettings`.
        public unsafe SubdivideSettings(MR._ByValue_SubdivideSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SubdivideSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SubdivideSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SubdivideSettings::operator=`.
        public unsafe MR.SubdivideSettings Assign(MR._ByValue_SubdivideSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SubdivideSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SubdivideSettings._Underlying *__MR_SubdivideSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SubdivideSettings._Underlying *_other);
            return new(__MR_SubdivideSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SubdivideSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SubdivideSettings`/`Const_SubdivideSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SubdivideSettings
    {
        internal readonly Const_SubdivideSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SubdivideSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SubdivideSettings(Const_SubdivideSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SubdivideSettings(Const_SubdivideSettings arg) {return new(arg);}
        public _ByValue_SubdivideSettings(MR.Misc._Moved<SubdivideSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SubdivideSettings(MR.Misc._Moved<SubdivideSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SubdivideSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SubdivideSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SubdivideSettings`/`Const_SubdivideSettings` directly.
    public class _InOptMut_SubdivideSettings
    {
        public SubdivideSettings? Opt;

        public _InOptMut_SubdivideSettings() {}
        public _InOptMut_SubdivideSettings(SubdivideSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SubdivideSettings(SubdivideSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SubdivideSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SubdivideSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SubdivideSettings`/`Const_SubdivideSettings` to pass it to the function.
    public class _InOptConst_SubdivideSettings
    {
        public Const_SubdivideSettings? Opt;

        public _InOptConst_SubdivideSettings() {}
        public _InOptConst_SubdivideSettings(Const_SubdivideSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SubdivideSettings(Const_SubdivideSettings value) {return new(value);}
    }

    /// splits edges in mesh region according to the settings;\n
    /// \return The total number of edge splits performed
    /// Generated from function `MR::subdivideMesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe int SubdivideMesh(MR.Mesh mesh, MR.Const_SubdivideSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdivideMesh_MR_Mesh", ExactSpelling = true)]
        extern static int __MR_subdivideMesh_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_SubdivideSettings._Underlying *settings);
        return __MR_subdivideMesh_MR_Mesh(mesh._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null);
    }

    /// subdivides mesh with per-element attributes according to given settings;
    /// \detail if settings.region is not null, then given region must be a subset of current face selection or face selection must absent
    /// \return The total number of edge splits performed
    /// Generated from function `MR::subdivideMesh`.
    public static unsafe int SubdivideMesh(MR.ObjectMeshData data, MR.Const_SubdivideSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_subdivideMesh_MR_ObjectMeshData", ExactSpelling = true)]
        extern static int __MR_subdivideMesh_MR_ObjectMeshData(MR.ObjectMeshData._Underlying *data, MR.Const_SubdivideSettings._Underlying *settings);
        return __MR_subdivideMesh_MR_ObjectMeshData(data._UnderlyingPtr, settings._UnderlyingPtr);
    }

    /// creates a copy of given mesh part, subdivides it to get rid of too long edges compared with voxelSize, then packs resulting mesh,
    /// this is called typically in preparation for 3D space sampling with voxelSize step, and subdivision is important for making leaves of AABB tree not too big compared with voxelSize
    /// Generated from function `MR::copySubdividePackMesh`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> CopySubdividePackMesh(MR.Const_MeshPart mp, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_copySubdividePackMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_copySubdividePackMesh(MR.Const_MeshPart._Underlying *mp, float voxelSize, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_copySubdividePackMesh(mp._UnderlyingPtr, voxelSize, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
    }

    /// returns the data of subdivided mesh given ObjectMesh (which remains unchanged) and subdivision parameters
    /// Generated from function `MR::makeSubdividedObjectMeshData`.
    public static unsafe MR.Misc._Moved<MR.ObjectMeshData> MakeSubdividedObjectMeshData(MR.Const_ObjectMesh obj, MR.Const_SubdivideSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSubdividedObjectMeshData", ExactSpelling = true)]
        extern static MR.ObjectMeshData._Underlying *__MR_makeSubdividedObjectMeshData(MR.Const_ObjectMesh._Underlying *obj, MR.Const_SubdivideSettings._Underlying *settings);
        return MR.Misc.Move(new MR.ObjectMeshData(__MR_makeSubdividedObjectMeshData(obj._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
