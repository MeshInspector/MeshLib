public static partial class MR
{
    /// Generated from class `MR::FillHoleNicelySettings`.
    /// This is the const half of the class.
    public class Const_FillHoleNicelySettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FillHoleNicelySettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Destroy", ExactSpelling = true)]
            extern static void __MR_FillHoleNicelySettings_Destroy(_Underlying *_this);
            __MR_FillHoleNicelySettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FillHoleNicelySettings() {Dispose(false);}

        /// how to triangulate the hole, must be specified by the user
        public unsafe MR.Const_FillHoleParams TriangulateParams
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_triangulateParams", ExactSpelling = true)]
                extern static MR.Const_FillHoleParams._Underlying *__MR_FillHoleNicelySettings_Get_triangulateParams(_Underlying *_this);
                return new(__MR_FillHoleNicelySettings_Get_triangulateParams(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If false then additional vertices are created inside the patch for best mesh quality
        public unsafe bool TriangulateOnly
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_triangulateOnly", ExactSpelling = true)]
                extern static bool *__MR_FillHoleNicelySettings_Get_triangulateOnly(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_triangulateOnly(_UnderlyingPtr);
            }
        }

        /// in triangulateOnly = false mode, edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
        public unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_notFlippable", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_Get_notFlippable(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_Get_notFlippable(_UnderlyingPtr);
            }
        }

        /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
        public unsafe float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_FillHoleNicelySettings_Get_maxEdgeLen(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed during subdivision
        public unsafe int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_FillHoleNicelySettings_Get_maxEdgeSplits(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
        public unsafe float MaxAngleChangeAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_maxAngleChangeAfterFlip", ExactSpelling = true)]
                extern static float *__MR_FillHoleNicelySettings_Get_maxAngleChangeAfterFlip(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_maxAngleChangeAfterFlip(_UnderlyingPtr);
            }
        }

        /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
        public unsafe bool SmoothCurvature
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_smoothCurvature", ExactSpelling = true)]
                extern static bool *__MR_FillHoleNicelySettings_Get_smoothCurvature(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_smoothCurvature(_UnderlyingPtr);
            }
        }

        /// Additionally smooth 3 layers of vertices near hole boundary both inside and outside of the hole
        public unsafe bool NaturalSmooth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_naturalSmooth", ExactSpelling = true)]
                extern static bool *__MR_FillHoleNicelySettings_Get_naturalSmooth(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_naturalSmooth(_UnderlyingPtr);
            }
        }

        /// (If this is set) this function is called in subdivision each time edge (e) is going to split, if it returns false then this split will be skipped
        public unsafe MR.Std.Const_Function_BoolFuncFromMREdgeId BeforeEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_beforeEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMREdgeId._Underlying *__MR_FillHoleNicelySettings_Get_beforeEdgeSplit(_Underlying *_this);
                return new(__MR_FillHoleNicelySettings_Get_beforeEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// (If this is set) this function is called in subdivision each time edge (e) is split into (e1->e), but before the ring is made Delone
        public unsafe MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_FillHoleNicelySettings_Get_onEdgeSplit(_Underlying *_this);
                return new(__MR_FillHoleNicelySettings_Get_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// edge weighting scheme for smoothCurvature mode
        public unsafe MR.EdgeWeights EdgeWeights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_edgeWeights", ExactSpelling = true)]
                extern static MR.EdgeWeights *__MR_FillHoleNicelySettings_Get_edgeWeights(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_edgeWeights(_UnderlyingPtr);
            }
        }

        /// vertex mass scheme for smoothCurvature mode
        public unsafe MR.VertexMass Vmass
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_vmass", ExactSpelling = true)]
                extern static MR.VertexMass *__MR_FillHoleNicelySettings_Get_vmass(_Underlying *_this);
                return *__MR_FillHoleNicelySettings_Get_vmass(_UnderlyingPtr);
            }
        }

        /// optional uv-coordinates of vertices; if provided then elements corresponding to new vertices will be added there
        public unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_uvCoords", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_Get_uvCoords(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_Get_uvCoords(_UnderlyingPtr);
            }
        }

        /// optional colors of vertices; if provided then elements corresponding to new vertices will be added there
        public unsafe ref void * ColorMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_colorMap", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_Get_colorMap(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_Get_colorMap(_UnderlyingPtr);
            }
        }

        /// optional colors of faces; if provided then elements corresponding to new faces will be added there
        public unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_Get_faceColors", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_Get_faceColors(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_Get_faceColors(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FillHoleNicelySettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleNicelySettings_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleNicelySettings` elementwise.
        public unsafe Const_FillHoleNicelySettings(MR._ByValue_FillHoleParams triangulateParams, bool triangulateOnly, MR.UndirectedEdgeBitSet? notFlippable, float maxEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, bool smoothCurvature, bool naturalSmooth, MR.Std._ByValue_Function_BoolFuncFromMREdgeId beforeEdgeSplit, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.EdgeWeights edgeWeights, MR.VertexMass vmass, MR.VertCoords2? uvCoords, MR.VertColors? colorMap, MR.FaceColors? faceColors) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_ConstructFrom(MR.Misc._PassBy triangulateParams_pass_by, MR.FillHoleParams._Underlying *triangulateParams, byte triangulateOnly, MR.UndirectedEdgeBitSet._Underlying *notFlippable, float maxEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, byte smoothCurvature, byte naturalSmooth, MR.Misc._PassBy beforeEdgeSplit_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *beforeEdgeSplit, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.EdgeWeights edgeWeights, MR.VertexMass vmass, MR.VertCoords2._Underlying *uvCoords, MR.VertColors._Underlying *colorMap, MR.FaceColors._Underlying *faceColors);
            _UnderlyingPtr = __MR_FillHoleNicelySettings_ConstructFrom(triangulateParams.PassByMode, triangulateParams.Value is not null ? triangulateParams.Value._UnderlyingPtr : null, triangulateOnly ? (byte)1 : (byte)0, notFlippable is not null ? notFlippable._UnderlyingPtr : null, maxEdgeLen, maxEdgeSplits, maxAngleChangeAfterFlip, smoothCurvature ? (byte)1 : (byte)0, naturalSmooth ? (byte)1 : (byte)0, beforeEdgeSplit.PassByMode, beforeEdgeSplit.Value is not null ? beforeEdgeSplit.Value._UnderlyingPtr : null, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, edgeWeights, vmass, uvCoords is not null ? uvCoords._UnderlyingPtr : null, colorMap is not null ? colorMap._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FillHoleNicelySettings::FillHoleNicelySettings`.
        public unsafe Const_FillHoleNicelySettings(MR._ByValue_FillHoleNicelySettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FillHoleNicelySettings._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleNicelySettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::FillHoleNicelySettings`.
    /// This is the non-const half of the class.
    public class FillHoleNicelySettings : Const_FillHoleNicelySettings
    {
        internal unsafe FillHoleNicelySettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// how to triangulate the hole, must be specified by the user
        public new unsafe MR.FillHoleParams TriangulateParams
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_triangulateParams", ExactSpelling = true)]
                extern static MR.FillHoleParams._Underlying *__MR_FillHoleNicelySettings_GetMutable_triangulateParams(_Underlying *_this);
                return new(__MR_FillHoleNicelySettings_GetMutable_triangulateParams(_UnderlyingPtr), is_owning: false);
            }
        }

        /// If false then additional vertices are created inside the patch for best mesh quality
        public new unsafe ref bool TriangulateOnly
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_triangulateOnly", ExactSpelling = true)]
                extern static bool *__MR_FillHoleNicelySettings_GetMutable_triangulateOnly(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_triangulateOnly(_UnderlyingPtr);
            }
        }

        /// in triangulateOnly = false mode, edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
        public new unsafe ref void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_notFlippable", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_GetMutable_notFlippable(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_notFlippable(_UnderlyingPtr);
            }
        }

        /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
        public new unsafe ref float MaxEdgeLen
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_maxEdgeLen", ExactSpelling = true)]
                extern static float *__MR_FillHoleNicelySettings_GetMutable_maxEdgeLen(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_maxEdgeLen(_UnderlyingPtr);
            }
        }

        /// Maximum number of edge splits allowed during subdivision
        public new unsafe ref int MaxEdgeSplits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_maxEdgeSplits", ExactSpelling = true)]
                extern static int *__MR_FillHoleNicelySettings_GetMutable_maxEdgeSplits(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_maxEdgeSplits(_UnderlyingPtr);
            }
        }

        /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
        public new unsafe ref float MaxAngleChangeAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_maxAngleChangeAfterFlip", ExactSpelling = true)]
                extern static float *__MR_FillHoleNicelySettings_GetMutable_maxAngleChangeAfterFlip(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_maxAngleChangeAfterFlip(_UnderlyingPtr);
            }
        }

        /// Whether to make patch over the hole smooth both inside and on its boundary with existed surface
        public new unsafe ref bool SmoothCurvature
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_smoothCurvature", ExactSpelling = true)]
                extern static bool *__MR_FillHoleNicelySettings_GetMutable_smoothCurvature(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_smoothCurvature(_UnderlyingPtr);
            }
        }

        /// Additionally smooth 3 layers of vertices near hole boundary both inside and outside of the hole
        public new unsafe ref bool NaturalSmooth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_naturalSmooth", ExactSpelling = true)]
                extern static bool *__MR_FillHoleNicelySettings_GetMutable_naturalSmooth(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_naturalSmooth(_UnderlyingPtr);
            }
        }

        /// (If this is set) this function is called in subdivision each time edge (e) is going to split, if it returns false then this split will be skipped
        public new unsafe MR.Std.Function_BoolFuncFromMREdgeId BeforeEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_beforeEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMREdgeId._Underlying *__MR_FillHoleNicelySettings_GetMutable_beforeEdgeSplit(_Underlying *_this);
                return new(__MR_FillHoleNicelySettings_GetMutable_beforeEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// (If this is set) this function is called in subdivision each time edge (e) is split into (e1->e), but before the ring is made Delone
        public new unsafe MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId OnEdgeSplit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_onEdgeSplit", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *__MR_FillHoleNicelySettings_GetMutable_onEdgeSplit(_Underlying *_this);
                return new(__MR_FillHoleNicelySettings_GetMutable_onEdgeSplit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// edge weighting scheme for smoothCurvature mode
        public new unsafe ref MR.EdgeWeights EdgeWeights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_edgeWeights", ExactSpelling = true)]
                extern static MR.EdgeWeights *__MR_FillHoleNicelySettings_GetMutable_edgeWeights(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_edgeWeights(_UnderlyingPtr);
            }
        }

        /// vertex mass scheme for smoothCurvature mode
        public new unsafe ref MR.VertexMass Vmass
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_vmass", ExactSpelling = true)]
                extern static MR.VertexMass *__MR_FillHoleNicelySettings_GetMutable_vmass(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_vmass(_UnderlyingPtr);
            }
        }

        /// optional uv-coordinates of vertices; if provided then elements corresponding to new vertices will be added there
        public new unsafe ref void * UvCoords
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_uvCoords", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_GetMutable_uvCoords(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_uvCoords(_UnderlyingPtr);
            }
        }

        /// optional colors of vertices; if provided then elements corresponding to new vertices will be added there
        public new unsafe ref void * ColorMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_colorMap", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_GetMutable_colorMap(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_colorMap(_UnderlyingPtr);
            }
        }

        /// optional colors of faces; if provided then elements corresponding to new faces will be added there
        public new unsafe ref void * FaceColors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_GetMutable_faceColors", ExactSpelling = true)]
                extern static void **__MR_FillHoleNicelySettings_GetMutable_faceColors(_Underlying *_this);
                return ref *__MR_FillHoleNicelySettings_GetMutable_faceColors(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FillHoleNicelySettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FillHoleNicelySettings_DefaultConstruct();
        }

        /// Constructs `MR::FillHoleNicelySettings` elementwise.
        public unsafe FillHoleNicelySettings(MR._ByValue_FillHoleParams triangulateParams, bool triangulateOnly, MR.UndirectedEdgeBitSet? notFlippable, float maxEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, bool smoothCurvature, bool naturalSmooth, MR.Std._ByValue_Function_BoolFuncFromMREdgeId beforeEdgeSplit, MR.Std._ByValue_Function_VoidFuncFromMREdgeIdMREdgeId onEdgeSplit, MR.EdgeWeights edgeWeights, MR.VertexMass vmass, MR.VertCoords2? uvCoords, MR.VertColors? colorMap, MR.FaceColors? faceColors) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_ConstructFrom(MR.Misc._PassBy triangulateParams_pass_by, MR.FillHoleParams._Underlying *triangulateParams, byte triangulateOnly, MR.UndirectedEdgeBitSet._Underlying *notFlippable, float maxEdgeLen, int maxEdgeSplits, float maxAngleChangeAfterFlip, byte smoothCurvature, byte naturalSmooth, MR.Misc._PassBy beforeEdgeSplit_pass_by, MR.Std.Function_BoolFuncFromMREdgeId._Underlying *beforeEdgeSplit, MR.Misc._PassBy onEdgeSplit_pass_by, MR.Std.Function_VoidFuncFromMREdgeIdMREdgeId._Underlying *onEdgeSplit, MR.EdgeWeights edgeWeights, MR.VertexMass vmass, MR.VertCoords2._Underlying *uvCoords, MR.VertColors._Underlying *colorMap, MR.FaceColors._Underlying *faceColors);
            _UnderlyingPtr = __MR_FillHoleNicelySettings_ConstructFrom(triangulateParams.PassByMode, triangulateParams.Value is not null ? triangulateParams.Value._UnderlyingPtr : null, triangulateOnly ? (byte)1 : (byte)0, notFlippable is not null ? notFlippable._UnderlyingPtr : null, maxEdgeLen, maxEdgeSplits, maxAngleChangeAfterFlip, smoothCurvature ? (byte)1 : (byte)0, naturalSmooth ? (byte)1 : (byte)0, beforeEdgeSplit.PassByMode, beforeEdgeSplit.Value is not null ? beforeEdgeSplit.Value._UnderlyingPtr : null, onEdgeSplit.PassByMode, onEdgeSplit.Value is not null ? onEdgeSplit.Value._UnderlyingPtr : null, edgeWeights, vmass, uvCoords is not null ? uvCoords._UnderlyingPtr : null, colorMap is not null ? colorMap._UnderlyingPtr : null, faceColors is not null ? faceColors._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FillHoleNicelySettings::FillHoleNicelySettings`.
        public unsafe FillHoleNicelySettings(MR._ByValue_FillHoleNicelySettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FillHoleNicelySettings._Underlying *_other);
            _UnderlyingPtr = __MR_FillHoleNicelySettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FillHoleNicelySettings::operator=`.
        public unsafe MR.FillHoleNicelySettings Assign(MR._ByValue_FillHoleNicelySettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillHoleNicelySettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FillHoleNicelySettings._Underlying *__MR_FillHoleNicelySettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FillHoleNicelySettings._Underlying *_other);
            return new(__MR_FillHoleNicelySettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FillHoleNicelySettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FillHoleNicelySettings`/`Const_FillHoleNicelySettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FillHoleNicelySettings
    {
        internal readonly Const_FillHoleNicelySettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FillHoleNicelySettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FillHoleNicelySettings(Const_FillHoleNicelySettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FillHoleNicelySettings(Const_FillHoleNicelySettings arg) {return new(arg);}
        public _ByValue_FillHoleNicelySettings(MR.Misc._Moved<FillHoleNicelySettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FillHoleNicelySettings(MR.Misc._Moved<FillHoleNicelySettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FillHoleNicelySettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FillHoleNicelySettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleNicelySettings`/`Const_FillHoleNicelySettings` directly.
    public class _InOptMut_FillHoleNicelySettings
    {
        public FillHoleNicelySettings? Opt;

        public _InOptMut_FillHoleNicelySettings() {}
        public _InOptMut_FillHoleNicelySettings(FillHoleNicelySettings value) {Opt = value;}
        public static implicit operator _InOptMut_FillHoleNicelySettings(FillHoleNicelySettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `FillHoleNicelySettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FillHoleNicelySettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FillHoleNicelySettings`/`Const_FillHoleNicelySettings` to pass it to the function.
    public class _InOptConst_FillHoleNicelySettings
    {
        public Const_FillHoleNicelySettings? Opt;

        public _InOptConst_FillHoleNicelySettings() {}
        public _InOptConst_FillHoleNicelySettings(Const_FillHoleNicelySettings value) {Opt = value;}
        public static implicit operator _InOptConst_FillHoleNicelySettings(Const_FillHoleNicelySettings value) {return new(value);}
    }

    /// fills a hole in mesh specified by one of its edge,
    /// optionally subdivides new patch on smaller triangles,
    /// optionally make smooth connection with existing triangles outside the hole
    /// \return triangles of the patch
    /// Generated from function `MR::fillHoleNicely`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FillHoleNicely(MR.Mesh mesh, MR.EdgeId holeEdge, MR.Const_FillHoleNicelySettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fillHoleNicely", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_fillHoleNicely(MR.Mesh._Underlying *mesh, MR.EdgeId holeEdge, MR.Const_FillHoleNicelySettings._Underlying *settings);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_fillHoleNicely(mesh._UnderlyingPtr, holeEdge, settings._UnderlyingPtr), is_owning: true));
    }
}
