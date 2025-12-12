public static partial class MR
{
    public enum BypassDirection : int
    {
        Clockwise = 0,
        CounterClockwise = 1,
    }

    /// Generated from class `MR::ToolPathParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ConstantCuspParams`
    /// This is the const half of the class.
    public class Const_ToolPathParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ToolPathParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ToolPathParams_Destroy(_Underlying *_this);
            __MR_ToolPathParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ToolPathParams() {Dispose(false);}

        // radius of the milling tool
        public unsafe float MillRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_millRadius", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_millRadius(_Underlying *_this);
                return *__MR_ToolPathParams_Get_millRadius(_UnderlyingPtr);
            }
        }

        // size of voxel needed to offset mesh
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_voxelSize(_Underlying *_this);
                return *__MR_ToolPathParams_Get_voxelSize(_UnderlyingPtr);
            }
        }

        // distance between sections built along Z axis
        // in Constant Cusp mode sectionStep should be bigger than voxelSize (x1.2 or more is recomended)
        public unsafe float SectionStep
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_sectionStep", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_sectionStep(_Underlying *_this);
                return *__MR_ToolPathParams_Get_sectionStep(_UnderlyingPtr);
            }
        }

        // if distance to the next section is smaller than it, transition will be performed along the surface
        // otherwise transition will be through the safe plane
        public unsafe float CritTransitionLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_critTransitionLength", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_critTransitionLength(_Underlying *_this);
                return *__MR_ToolPathParams_Get_critTransitionLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving down, it will be slowed down in this distance from mesh
        public unsafe float PlungeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_plungeLength", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_plungeLength(_Underlying *_this);
                return *__MR_ToolPathParams_Get_plungeLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving up, it will be slowed down in this distance from mesh
        public unsafe float RetractLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_retractLength", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_retractLength(_Underlying *_this);
                return *__MR_ToolPathParams_Get_retractLength(_UnderlyingPtr);
            }
        }

        // speed of slow movement down
        public unsafe float PlungeFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_plungeFeed", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_plungeFeed(_Underlying *_this);
                return *__MR_ToolPathParams_Get_plungeFeed(_UnderlyingPtr);
            }
        }

        // speed of slow movement up
        public unsafe float RetractFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_retractFeed", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_retractFeed(_Underlying *_this);
                return *__MR_ToolPathParams_Get_retractFeed(_UnderlyingPtr);
            }
        }

        // speed of regular milling
        public unsafe float BaseFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_baseFeed", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_baseFeed(_Underlying *_this);
                return *__MR_ToolPathParams_Get_baseFeed(_UnderlyingPtr);
            }
        }

        // z-coordinate of plane where tool can move in any direction without touching the object
        public unsafe float SafeZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_safeZ", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_safeZ(_Underlying *_this);
                return *__MR_ToolPathParams_Get_safeZ(_UnderlyingPtr);
            }
        }

        // which direction isolines or sections should be passed in
        public unsafe MR.BypassDirection BypassDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_bypassDir", ExactSpelling = true)]
                extern static MR.BypassDirection *__MR_ToolPathParams_Get_bypassDir(_Underlying *_this);
                return *__MR_ToolPathParams_Get_bypassDir(_UnderlyingPtr);
            }
        }

        // mesh can be transformed using xf parameter
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_ToolPathParams_Get_xf(_Underlying *_this);
                return ref *__MR_ToolPathParams_Get_xf(_UnderlyingPtr);
            }
        }

        // if true then a tool path for a flat milling tool will be generated
        public unsafe bool FlatTool
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_flatTool", ExactSpelling = true)]
                extern static bool *__MR_ToolPathParams_Get_flatTool(_Underlying *_this);
                return *__MR_ToolPathParams_Get_flatTool(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ToolPathParams_Get_cb(_Underlying *_this);
                return new(__MR_ToolPathParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        // if > 0 - expand the trajectory creation area and create toolpath to mill excess material to make empty areas.
        // The area has the shape of a box.
        // Lacing specific only.
        public unsafe float ToolpathExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_toolpathExpansion", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_Get_toolpathExpansion(_Underlying *_this);
                return *__MR_ToolPathParams_Get_toolpathExpansion(_UnderlyingPtr);
            }
        }

        // optional output, stores isolines without transits
        public unsafe ref void * Isolines
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_isolines", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_Get_isolines(_Underlying *_this);
                return ref *__MR_ToolPathParams_Get_isolines(_UnderlyingPtr);
            }
        }

        // optional output, polyline containing start vertices for isolines
        public unsafe ref void * StartContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_startContours", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_Get_startContours(_Underlying *_this);
                return ref *__MR_ToolPathParams_Get_startContours(_UnderlyingPtr);
            }
        }

        // start vertices on the offset mesh used for calcutating isolines
        public unsafe ref void * StartVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_startVertices", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_Get_startVertices(_Underlying *_this);
                return ref *__MR_ToolPathParams_Get_startVertices(_UnderlyingPtr);
            }
        }

        public unsafe ref void * OffsetMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_Get_offsetMesh", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_Get_offsetMesh(_Underlying *_this);
                return ref *__MR_ToolPathParams_Get_offsetMesh(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ToolPathParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ToolPathParams_DefaultConstruct();
        }

        /// Constructs `MR::ToolPathParams` elementwise.
        public unsafe Const_ToolPathParams(float millRadius, float voxelSize, float sectionStep, float critTransitionLength, float plungeLength, float retractLength, float plungeFeed, float retractFeed, float baseFeed, float safeZ, MR.BypassDirection bypassDir, MR.Const_AffineXf3f? xf, bool flatTool, MR.Std._ByValue_Function_BoolFuncFromFloat cb, float toolpathExpansion, MR.Std.Vector_StdVectorMRVector3f? isolines, MR.Std.Vector_StdVectorMRVector3f? startContours, MR.Std.Vector_MRVector3f? startVertices, MR.MeshPart? offsetMesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_ConstructFrom(float millRadius, float voxelSize, float sectionStep, float critTransitionLength, float plungeLength, float retractLength, float plungeFeed, float retractFeed, float baseFeed, float safeZ, MR.BypassDirection bypassDir, MR.Const_AffineXf3f._Underlying *xf, byte flatTool, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float toolpathExpansion, MR.Std.Vector_StdVectorMRVector3f._Underlying *isolines, MR.Std.Vector_StdVectorMRVector3f._Underlying *startContours, MR.Std.Vector_MRVector3f._Underlying *startVertices, MR.MeshPart._Underlying *offsetMesh);
            _UnderlyingPtr = __MR_ToolPathParams_ConstructFrom(millRadius, voxelSize, sectionStep, critTransitionLength, plungeLength, retractLength, plungeFeed, retractFeed, baseFeed, safeZ, bypassDir, xf is not null ? xf._UnderlyingPtr : null, flatTool ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, toolpathExpansion, isolines is not null ? isolines._UnderlyingPtr : null, startContours is not null ? startContours._UnderlyingPtr : null, startVertices is not null ? startVertices._UnderlyingPtr : null, offsetMesh is not null ? offsetMesh._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ToolPathParams::ToolPathParams`.
        public unsafe Const_ToolPathParams(MR._ByValue_ToolPathParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ToolPathParams._Underlying *_other);
            _UnderlyingPtr = __MR_ToolPathParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ToolPathParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::ConstantCuspParams`
    /// This is the non-const half of the class.
    public class ToolPathParams : Const_ToolPathParams
    {
        internal unsafe ToolPathParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // radius of the milling tool
        public new unsafe ref float MillRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_millRadius", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_millRadius(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_millRadius(_UnderlyingPtr);
            }
        }

        // size of voxel needed to offset mesh
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        // distance between sections built along Z axis
        // in Constant Cusp mode sectionStep should be bigger than voxelSize (x1.2 or more is recomended)
        public new unsafe ref float SectionStep
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_sectionStep", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_sectionStep(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_sectionStep(_UnderlyingPtr);
            }
        }

        // if distance to the next section is smaller than it, transition will be performed along the surface
        // otherwise transition will be through the safe plane
        public new unsafe ref float CritTransitionLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_critTransitionLength", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_critTransitionLength(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_critTransitionLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving down, it will be slowed down in this distance from mesh
        public new unsafe ref float PlungeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_plungeLength", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_plungeLength(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_plungeLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving up, it will be slowed down in this distance from mesh
        public new unsafe ref float RetractLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_retractLength", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_retractLength(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_retractLength(_UnderlyingPtr);
            }
        }

        // speed of slow movement down
        public new unsafe ref float PlungeFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_plungeFeed", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_plungeFeed(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_plungeFeed(_UnderlyingPtr);
            }
        }

        // speed of slow movement up
        public new unsafe ref float RetractFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_retractFeed", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_retractFeed(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_retractFeed(_UnderlyingPtr);
            }
        }

        // speed of regular milling
        public new unsafe ref float BaseFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_baseFeed", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_baseFeed(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_baseFeed(_UnderlyingPtr);
            }
        }

        // z-coordinate of plane where tool can move in any direction without touching the object
        public new unsafe ref float SafeZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_safeZ", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_safeZ(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_safeZ(_UnderlyingPtr);
            }
        }

        // which direction isolines or sections should be passed in
        public new unsafe ref MR.BypassDirection BypassDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_bypassDir", ExactSpelling = true)]
                extern static MR.BypassDirection *__MR_ToolPathParams_GetMutable_bypassDir(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_bypassDir(_UnderlyingPtr);
            }
        }

        // mesh can be transformed using xf parameter
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_ToolPathParams_GetMutable_xf(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_xf(_UnderlyingPtr);
            }
        }

        // if true then a tool path for a flat milling tool will be generated
        public new unsafe ref bool FlatTool
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_flatTool", ExactSpelling = true)]
                extern static bool *__MR_ToolPathParams_GetMutable_flatTool(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_flatTool(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ToolPathParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_ToolPathParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        // if > 0 - expand the trajectory creation area and create toolpath to mill excess material to make empty areas.
        // The area has the shape of a box.
        // Lacing specific only.
        public new unsafe ref float ToolpathExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_toolpathExpansion", ExactSpelling = true)]
                extern static float *__MR_ToolPathParams_GetMutable_toolpathExpansion(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_toolpathExpansion(_UnderlyingPtr);
            }
        }

        // optional output, stores isolines without transits
        public new unsafe ref void * Isolines
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_isolines", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_GetMutable_isolines(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_isolines(_UnderlyingPtr);
            }
        }

        // optional output, polyline containing start vertices for isolines
        public new unsafe ref void * StartContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_startContours", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_GetMutable_startContours(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_startContours(_UnderlyingPtr);
            }
        }

        // start vertices on the offset mesh used for calcutating isolines
        public new unsafe ref void * StartVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_startVertices", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_GetMutable_startVertices(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_startVertices(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * OffsetMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_GetMutable_offsetMesh", ExactSpelling = true)]
                extern static void **__MR_ToolPathParams_GetMutable_offsetMesh(_Underlying *_this);
                return ref *__MR_ToolPathParams_GetMutable_offsetMesh(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ToolPathParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ToolPathParams_DefaultConstruct();
        }

        /// Constructs `MR::ToolPathParams` elementwise.
        public unsafe ToolPathParams(float millRadius, float voxelSize, float sectionStep, float critTransitionLength, float plungeLength, float retractLength, float plungeFeed, float retractFeed, float baseFeed, float safeZ, MR.BypassDirection bypassDir, MR.Const_AffineXf3f? xf, bool flatTool, MR.Std._ByValue_Function_BoolFuncFromFloat cb, float toolpathExpansion, MR.Std.Vector_StdVectorMRVector3f? isolines, MR.Std.Vector_StdVectorMRVector3f? startContours, MR.Std.Vector_MRVector3f? startVertices, MR.MeshPart? offsetMesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_ConstructFrom(float millRadius, float voxelSize, float sectionStep, float critTransitionLength, float plungeLength, float retractLength, float plungeFeed, float retractFeed, float baseFeed, float safeZ, MR.BypassDirection bypassDir, MR.Const_AffineXf3f._Underlying *xf, byte flatTool, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, float toolpathExpansion, MR.Std.Vector_StdVectorMRVector3f._Underlying *isolines, MR.Std.Vector_StdVectorMRVector3f._Underlying *startContours, MR.Std.Vector_MRVector3f._Underlying *startVertices, MR.MeshPart._Underlying *offsetMesh);
            _UnderlyingPtr = __MR_ToolPathParams_ConstructFrom(millRadius, voxelSize, sectionStep, critTransitionLength, plungeLength, retractLength, plungeFeed, retractFeed, baseFeed, safeZ, bypassDir, xf is not null ? xf._UnderlyingPtr : null, flatTool ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null, toolpathExpansion, isolines is not null ? isolines._UnderlyingPtr : null, startContours is not null ? startContours._UnderlyingPtr : null, startVertices is not null ? startVertices._UnderlyingPtr : null, offsetMesh is not null ? offsetMesh._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ToolPathParams::ToolPathParams`.
        public unsafe ToolPathParams(MR._ByValue_ToolPathParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ToolPathParams._Underlying *_other);
            _UnderlyingPtr = __MR_ToolPathParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ToolPathParams::operator=`.
        public unsafe MR.ToolPathParams Assign(MR._ByValue_ToolPathParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ToolPathParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ToolPathParams._Underlying *_other);
            return new(__MR_ToolPathParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ToolPathParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ToolPathParams`/`Const_ToolPathParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ToolPathParams
    {
        internal readonly Const_ToolPathParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ToolPathParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ToolPathParams(Const_ToolPathParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ToolPathParams(Const_ToolPathParams arg) {return new(arg);}
        public _ByValue_ToolPathParams(MR.Misc._Moved<ToolPathParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ToolPathParams(MR.Misc._Moved<ToolPathParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ToolPathParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ToolPathParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ToolPathParams`/`Const_ToolPathParams` directly.
    public class _InOptMut_ToolPathParams
    {
        public ToolPathParams? Opt;

        public _InOptMut_ToolPathParams() {}
        public _InOptMut_ToolPathParams(ToolPathParams value) {Opt = value;}
        public static implicit operator _InOptMut_ToolPathParams(ToolPathParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ToolPathParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ToolPathParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ToolPathParams`/`Const_ToolPathParams` to pass it to the function.
    public class _InOptConst_ToolPathParams
    {
        public Const_ToolPathParams? Opt;

        public _InOptConst_ToolPathParams() {}
        public _InOptConst_ToolPathParams(Const_ToolPathParams value) {Opt = value;}
        public static implicit operator _InOptConst_ToolPathParams(Const_ToolPathParams value) {return new(value);}
    }

    /// Generated from class `MR::ConstantCuspParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ToolPathParams`
    /// This is the const half of the class.
    public class Const_ConstantCuspParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ConstantCuspParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ConstantCuspParams_Destroy(_Underlying *_this);
            __MR_ConstantCuspParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ConstantCuspParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ToolPathParams(Const_ConstantCuspParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_UpcastTo_MR_ToolPathParams", ExactSpelling = true)]
            extern static MR.Const_ToolPathParams._Underlying *__MR_ConstantCuspParams_UpcastTo_MR_ToolPathParams(_Underlying *_this);
            MR.Const_ToolPathParams ret = new(__MR_ConstantCuspParams_UpcastTo_MR_ToolPathParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // if true isolines will be processed from center point to the boundary (usually it means from up to down)
        public unsafe bool FromCenterToBoundary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_fromCenterToBoundary", ExactSpelling = true)]
                extern static bool *__MR_ConstantCuspParams_Get_fromCenterToBoundary(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_fromCenterToBoundary(_UnderlyingPtr);
            }
        }

        // radius of the milling tool
        public unsafe float MillRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_millRadius", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_millRadius(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_millRadius(_UnderlyingPtr);
            }
        }

        // size of voxel needed to offset mesh
        public unsafe float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_voxelSize", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_voxelSize(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_voxelSize(_UnderlyingPtr);
            }
        }

        // distance between sections built along Z axis
        // in Constant Cusp mode sectionStep should be bigger than voxelSize (x1.2 or more is recomended)
        public unsafe float SectionStep
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_sectionStep", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_sectionStep(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_sectionStep(_UnderlyingPtr);
            }
        }

        // if distance to the next section is smaller than it, transition will be performed along the surface
        // otherwise transition will be through the safe plane
        public unsafe float CritTransitionLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_critTransitionLength", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_critTransitionLength(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_critTransitionLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving down, it will be slowed down in this distance from mesh
        public unsafe float PlungeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_plungeLength", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_plungeLength(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_plungeLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving up, it will be slowed down in this distance from mesh
        public unsafe float RetractLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_retractLength", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_retractLength(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_retractLength(_UnderlyingPtr);
            }
        }

        // speed of slow movement down
        public unsafe float PlungeFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_plungeFeed", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_plungeFeed(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_plungeFeed(_UnderlyingPtr);
            }
        }

        // speed of slow movement up
        public unsafe float RetractFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_retractFeed", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_retractFeed(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_retractFeed(_UnderlyingPtr);
            }
        }

        // speed of regular milling
        public unsafe float BaseFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_baseFeed", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_baseFeed(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_baseFeed(_UnderlyingPtr);
            }
        }

        // z-coordinate of plane where tool can move in any direction without touching the object
        public unsafe float SafeZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_safeZ", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_safeZ(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_safeZ(_UnderlyingPtr);
            }
        }

        // which direction isolines or sections should be passed in
        public unsafe MR.BypassDirection BypassDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_bypassDir", ExactSpelling = true)]
                extern static MR.BypassDirection *__MR_ConstantCuspParams_Get_bypassDir(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_bypassDir(_UnderlyingPtr);
            }
        }

        // mesh can be transformed using xf parameter
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_ConstantCuspParams_Get_xf(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_Get_xf(_UnderlyingPtr);
            }
        }

        // if true then a tool path for a flat milling tool will be generated
        public unsafe bool FlatTool
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_flatTool", ExactSpelling = true)]
                extern static bool *__MR_ConstantCuspParams_Get_flatTool(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_flatTool(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ConstantCuspParams_Get_cb(_Underlying *_this);
                return new(__MR_ConstantCuspParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        // if > 0 - expand the trajectory creation area and create toolpath to mill excess material to make empty areas.
        // The area has the shape of a box.
        // Lacing specific only.
        public unsafe float ToolpathExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_toolpathExpansion", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_Get_toolpathExpansion(_Underlying *_this);
                return *__MR_ConstantCuspParams_Get_toolpathExpansion(_UnderlyingPtr);
            }
        }

        // optional output, stores isolines without transits
        public unsafe ref void * Isolines
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_isolines", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_Get_isolines(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_Get_isolines(_UnderlyingPtr);
            }
        }

        // optional output, polyline containing start vertices for isolines
        public unsafe ref void * StartContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_startContours", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_Get_startContours(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_Get_startContours(_UnderlyingPtr);
            }
        }

        // start vertices on the offset mesh used for calcutating isolines
        public unsafe ref void * StartVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_startVertices", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_Get_startVertices(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_Get_startVertices(_UnderlyingPtr);
            }
        }

        public unsafe ref void * OffsetMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_Get_offsetMesh", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_Get_offsetMesh(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_Get_offsetMesh(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ConstantCuspParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ConstantCuspParams._Underlying *__MR_ConstantCuspParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ConstantCuspParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::ConstantCuspParams::ConstantCuspParams`.
        public unsafe Const_ConstantCuspParams(MR._ByValue_ConstantCuspParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ConstantCuspParams._Underlying *__MR_ConstantCuspParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ConstantCuspParams._Underlying *_other);
            _UnderlyingPtr = __MR_ConstantCuspParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ConstantCuspParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ToolPathParams`
    /// This is the non-const half of the class.
    public class ConstantCuspParams : Const_ConstantCuspParams
    {
        internal unsafe ConstantCuspParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ToolPathParams(ConstantCuspParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_UpcastTo_MR_ToolPathParams", ExactSpelling = true)]
            extern static MR.ToolPathParams._Underlying *__MR_ConstantCuspParams_UpcastTo_MR_ToolPathParams(_Underlying *_this);
            MR.ToolPathParams ret = new(__MR_ConstantCuspParams_UpcastTo_MR_ToolPathParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // if true isolines will be processed from center point to the boundary (usually it means from up to down)
        public new unsafe ref bool FromCenterToBoundary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_fromCenterToBoundary", ExactSpelling = true)]
                extern static bool *__MR_ConstantCuspParams_GetMutable_fromCenterToBoundary(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_fromCenterToBoundary(_UnderlyingPtr);
            }
        }

        // radius of the milling tool
        public new unsafe ref float MillRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_millRadius", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_millRadius(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_millRadius(_UnderlyingPtr);
            }
        }

        // size of voxel needed to offset mesh
        public new unsafe ref float VoxelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_voxelSize", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_voxelSize(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_voxelSize(_UnderlyingPtr);
            }
        }

        // distance between sections built along Z axis
        // in Constant Cusp mode sectionStep should be bigger than voxelSize (x1.2 or more is recomended)
        public new unsafe ref float SectionStep
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_sectionStep", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_sectionStep(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_sectionStep(_UnderlyingPtr);
            }
        }

        // if distance to the next section is smaller than it, transition will be performed along the surface
        // otherwise transition will be through the safe plane
        public new unsafe ref float CritTransitionLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_critTransitionLength", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_critTransitionLength(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_critTransitionLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving down, it will be slowed down in this distance from mesh
        public new unsafe ref float PlungeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_plungeLength", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_plungeLength(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_plungeLength(_UnderlyingPtr);
            }
        }

        // when the mill is moving up, it will be slowed down in this distance from mesh
        public new unsafe ref float RetractLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_retractLength", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_retractLength(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_retractLength(_UnderlyingPtr);
            }
        }

        // speed of slow movement down
        public new unsafe ref float PlungeFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_plungeFeed", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_plungeFeed(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_plungeFeed(_UnderlyingPtr);
            }
        }

        // speed of slow movement up
        public new unsafe ref float RetractFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_retractFeed", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_retractFeed(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_retractFeed(_UnderlyingPtr);
            }
        }

        // speed of regular milling
        public new unsafe ref float BaseFeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_baseFeed", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_baseFeed(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_baseFeed(_UnderlyingPtr);
            }
        }

        // z-coordinate of plane where tool can move in any direction without touching the object
        public new unsafe ref float SafeZ
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_safeZ", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_safeZ(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_safeZ(_UnderlyingPtr);
            }
        }

        // which direction isolines or sections should be passed in
        public new unsafe ref MR.BypassDirection BypassDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_bypassDir", ExactSpelling = true)]
                extern static MR.BypassDirection *__MR_ConstantCuspParams_GetMutable_bypassDir(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_bypassDir(_UnderlyingPtr);
            }
        }

        // mesh can be transformed using xf parameter
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_ConstantCuspParams_GetMutable_xf(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_xf(_UnderlyingPtr);
            }
        }

        // if true then a tool path for a flat milling tool will be generated
        public new unsafe ref bool FlatTool
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_flatTool", ExactSpelling = true)]
                extern static bool *__MR_ConstantCuspParams_GetMutable_flatTool(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_flatTool(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ConstantCuspParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_ConstantCuspParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        // if > 0 - expand the trajectory creation area and create toolpath to mill excess material to make empty areas.
        // The area has the shape of a box.
        // Lacing specific only.
        public new unsafe ref float ToolpathExpansion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_toolpathExpansion", ExactSpelling = true)]
                extern static float *__MR_ConstantCuspParams_GetMutable_toolpathExpansion(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_toolpathExpansion(_UnderlyingPtr);
            }
        }

        // optional output, stores isolines without transits
        public new unsafe ref void * Isolines
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_isolines", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_GetMutable_isolines(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_isolines(_UnderlyingPtr);
            }
        }

        // optional output, polyline containing start vertices for isolines
        public new unsafe ref void * StartContours
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_startContours", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_GetMutable_startContours(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_startContours(_UnderlyingPtr);
            }
        }

        // start vertices on the offset mesh used for calcutating isolines
        public new unsafe ref void * StartVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_startVertices", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_GetMutable_startVertices(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_startVertices(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * OffsetMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_GetMutable_offsetMesh", ExactSpelling = true)]
                extern static void **__MR_ConstantCuspParams_GetMutable_offsetMesh(_Underlying *_this);
                return ref *__MR_ConstantCuspParams_GetMutable_offsetMesh(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ConstantCuspParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ConstantCuspParams._Underlying *__MR_ConstantCuspParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ConstantCuspParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::ConstantCuspParams::ConstantCuspParams`.
        public unsafe ConstantCuspParams(MR._ByValue_ConstantCuspParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ConstantCuspParams._Underlying *__MR_ConstantCuspParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ConstantCuspParams._Underlying *_other);
            _UnderlyingPtr = __MR_ConstantCuspParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ConstantCuspParams::operator=`.
        public unsafe MR.ConstantCuspParams Assign(MR._ByValue_ConstantCuspParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ConstantCuspParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ConstantCuspParams._Underlying *__MR_ConstantCuspParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ConstantCuspParams._Underlying *_other);
            return new(__MR_ConstantCuspParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ConstantCuspParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ConstantCuspParams`/`Const_ConstantCuspParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ConstantCuspParams
    {
        internal readonly Const_ConstantCuspParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ConstantCuspParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ConstantCuspParams(Const_ConstantCuspParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ConstantCuspParams(Const_ConstantCuspParams arg) {return new(arg);}
        public _ByValue_ConstantCuspParams(MR.Misc._Moved<ConstantCuspParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ConstantCuspParams(MR.Misc._Moved<ConstantCuspParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ConstantCuspParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ConstantCuspParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ConstantCuspParams`/`Const_ConstantCuspParams` directly.
    public class _InOptMut_ConstantCuspParams
    {
        public ConstantCuspParams? Opt;

        public _InOptMut_ConstantCuspParams() {}
        public _InOptMut_ConstantCuspParams(ConstantCuspParams value) {Opt = value;}
        public static implicit operator _InOptMut_ConstantCuspParams(ConstantCuspParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ConstantCuspParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ConstantCuspParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ConstantCuspParams`/`Const_ConstantCuspParams` to pass it to the function.
    public class _InOptConst_ConstantCuspParams
    {
        public Const_ConstantCuspParams? Opt;

        public _InOptConst_ConstantCuspParams() {}
        public _InOptConst_ConstantCuspParams(Const_ConstantCuspParams value) {Opt = value;}
        public static implicit operator _InOptConst_ConstantCuspParams(Const_ConstantCuspParams value) {return new(value);}
    }

    /// Generated from class `MR::LineInterpolationParams`.
    /// This is the const half of the class.
    public class Const_LineInterpolationParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LineInterpolationParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_Destroy", ExactSpelling = true)]
            extern static void __MR_LineInterpolationParams_Destroy(_Underlying *_this);
            __MR_LineInterpolationParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LineInterpolationParams() {Dispose(false);}

        // maximal deviation from given line
        public unsafe float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_Get_eps", ExactSpelling = true)]
                extern static float *__MR_LineInterpolationParams_Get_eps(_Underlying *_this);
                return *__MR_LineInterpolationParams_Get_eps(_UnderlyingPtr);
            }
        }

        // maximal length of the line
        public unsafe float MaxLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_Get_maxLength", ExactSpelling = true)]
                extern static float *__MR_LineInterpolationParams_Get_maxLength(_Underlying *_this);
                return *__MR_LineInterpolationParams_Get_maxLength(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_LineInterpolationParams_Get_cb(_Underlying *_this);
                return new(__MR_LineInterpolationParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LineInterpolationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_LineInterpolationParams_DefaultConstruct();
        }

        /// Constructs `MR::LineInterpolationParams` elementwise.
        public unsafe Const_LineInterpolationParams(float eps, float maxLength, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_ConstructFrom(float eps, float maxLength, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_LineInterpolationParams_ConstructFrom(eps, maxLength, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::LineInterpolationParams::LineInterpolationParams`.
        public unsafe Const_LineInterpolationParams(MR._ByValue_LineInterpolationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LineInterpolationParams._Underlying *_other);
            _UnderlyingPtr = __MR_LineInterpolationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::LineInterpolationParams`.
    /// This is the non-const half of the class.
    public class LineInterpolationParams : Const_LineInterpolationParams
    {
        internal unsafe LineInterpolationParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // maximal deviation from given line
        public new unsafe ref float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_GetMutable_eps", ExactSpelling = true)]
                extern static float *__MR_LineInterpolationParams_GetMutable_eps(_Underlying *_this);
                return ref *__MR_LineInterpolationParams_GetMutable_eps(_UnderlyingPtr);
            }
        }

        // maximal length of the line
        public new unsafe ref float MaxLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_GetMutable_maxLength", ExactSpelling = true)]
                extern static float *__MR_LineInterpolationParams_GetMutable_maxLength(_Underlying *_this);
                return ref *__MR_LineInterpolationParams_GetMutable_maxLength(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_LineInterpolationParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_LineInterpolationParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LineInterpolationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_LineInterpolationParams_DefaultConstruct();
        }

        /// Constructs `MR::LineInterpolationParams` elementwise.
        public unsafe LineInterpolationParams(float eps, float maxLength, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_ConstructFrom(float eps, float maxLength, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_LineInterpolationParams_ConstructFrom(eps, maxLength, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::LineInterpolationParams::LineInterpolationParams`.
        public unsafe LineInterpolationParams(MR._ByValue_LineInterpolationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LineInterpolationParams._Underlying *_other);
            _UnderlyingPtr = __MR_LineInterpolationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LineInterpolationParams::operator=`.
        public unsafe MR.LineInterpolationParams Assign(MR._ByValue_LineInterpolationParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineInterpolationParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LineInterpolationParams._Underlying *__MR_LineInterpolationParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LineInterpolationParams._Underlying *_other);
            return new(__MR_LineInterpolationParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LineInterpolationParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LineInterpolationParams`/`Const_LineInterpolationParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LineInterpolationParams
    {
        internal readonly Const_LineInterpolationParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LineInterpolationParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LineInterpolationParams(Const_LineInterpolationParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LineInterpolationParams(Const_LineInterpolationParams arg) {return new(arg);}
        public _ByValue_LineInterpolationParams(MR.Misc._Moved<LineInterpolationParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LineInterpolationParams(MR.Misc._Moved<LineInterpolationParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LineInterpolationParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LineInterpolationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineInterpolationParams`/`Const_LineInterpolationParams` directly.
    public class _InOptMut_LineInterpolationParams
    {
        public LineInterpolationParams? Opt;

        public _InOptMut_LineInterpolationParams() {}
        public _InOptMut_LineInterpolationParams(LineInterpolationParams value) {Opt = value;}
        public static implicit operator _InOptMut_LineInterpolationParams(LineInterpolationParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `LineInterpolationParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LineInterpolationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineInterpolationParams`/`Const_LineInterpolationParams` to pass it to the function.
    public class _InOptConst_LineInterpolationParams
    {
        public Const_LineInterpolationParams? Opt;

        public _InOptConst_LineInterpolationParams() {}
        public _InOptConst_LineInterpolationParams(Const_LineInterpolationParams value) {Opt = value;}
        public static implicit operator _InOptConst_LineInterpolationParams(Const_LineInterpolationParams value) {return new(value);}
    }

    /// Generated from class `MR::ArcInterpolationParams`.
    /// This is the const half of the class.
    public class Const_ArcInterpolationParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ArcInterpolationParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ArcInterpolationParams_Destroy(_Underlying *_this);
            __MR_ArcInterpolationParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ArcInterpolationParams() {Dispose(false);}

        // maximal deviation of arc from given path
        public unsafe float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_Get_eps", ExactSpelling = true)]
                extern static float *__MR_ArcInterpolationParams_Get_eps(_Underlying *_this);
                return *__MR_ArcInterpolationParams_Get_eps(_UnderlyingPtr);
            }
        }

        // maximal radius of the arc
        public unsafe float MaxRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_Get_maxRadius", ExactSpelling = true)]
                extern static float *__MR_ArcInterpolationParams_Get_maxRadius(_Underlying *_this);
                return *__MR_ArcInterpolationParams_Get_maxRadius(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ArcInterpolationParams_Get_cb(_Underlying *_this);
                return new(__MR_ArcInterpolationParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ArcInterpolationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ArcInterpolationParams_DefaultConstruct();
        }

        /// Constructs `MR::ArcInterpolationParams` elementwise.
        public unsafe Const_ArcInterpolationParams(float eps, float maxRadius, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_ConstructFrom(float eps, float maxRadius, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_ArcInterpolationParams_ConstructFrom(eps, maxRadius, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ArcInterpolationParams::ArcInterpolationParams`.
        public unsafe Const_ArcInterpolationParams(MR._ByValue_ArcInterpolationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ArcInterpolationParams._Underlying *_other);
            _UnderlyingPtr = __MR_ArcInterpolationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ArcInterpolationParams`.
    /// This is the non-const half of the class.
    public class ArcInterpolationParams : Const_ArcInterpolationParams
    {
        internal unsafe ArcInterpolationParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // maximal deviation of arc from given path
        public new unsafe ref float Eps
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_GetMutable_eps", ExactSpelling = true)]
                extern static float *__MR_ArcInterpolationParams_GetMutable_eps(_Underlying *_this);
                return ref *__MR_ArcInterpolationParams_GetMutable_eps(_UnderlyingPtr);
            }
        }

        // maximal radius of the arc
        public new unsafe ref float MaxRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_GetMutable_maxRadius", ExactSpelling = true)]
                extern static float *__MR_ArcInterpolationParams_GetMutable_maxRadius(_Underlying *_this);
                return ref *__MR_ArcInterpolationParams_GetMutable_maxRadius(_UnderlyingPtr);
            }
        }

        // callback for reporting on progress
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ArcInterpolationParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_ArcInterpolationParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ArcInterpolationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ArcInterpolationParams_DefaultConstruct();
        }

        /// Constructs `MR::ArcInterpolationParams` elementwise.
        public unsafe ArcInterpolationParams(float eps, float maxRadius, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_ConstructFrom(float eps, float maxRadius, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_ArcInterpolationParams_ConstructFrom(eps, maxRadius, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ArcInterpolationParams::ArcInterpolationParams`.
        public unsafe ArcInterpolationParams(MR._ByValue_ArcInterpolationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ArcInterpolationParams._Underlying *_other);
            _UnderlyingPtr = __MR_ArcInterpolationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ArcInterpolationParams::operator=`.
        public unsafe MR.ArcInterpolationParams Assign(MR._ByValue_ArcInterpolationParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ArcInterpolationParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ArcInterpolationParams._Underlying *__MR_ArcInterpolationParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ArcInterpolationParams._Underlying *_other);
            return new(__MR_ArcInterpolationParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ArcInterpolationParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ArcInterpolationParams`/`Const_ArcInterpolationParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ArcInterpolationParams
    {
        internal readonly Const_ArcInterpolationParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ArcInterpolationParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ArcInterpolationParams(Const_ArcInterpolationParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ArcInterpolationParams(Const_ArcInterpolationParams arg) {return new(arg);}
        public _ByValue_ArcInterpolationParams(MR.Misc._Moved<ArcInterpolationParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ArcInterpolationParams(MR.Misc._Moved<ArcInterpolationParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ArcInterpolationParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ArcInterpolationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ArcInterpolationParams`/`Const_ArcInterpolationParams` directly.
    public class _InOptMut_ArcInterpolationParams
    {
        public ArcInterpolationParams? Opt;

        public _InOptMut_ArcInterpolationParams() {}
        public _InOptMut_ArcInterpolationParams(ArcInterpolationParams value) {Opt = value;}
        public static implicit operator _InOptMut_ArcInterpolationParams(ArcInterpolationParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ArcInterpolationParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ArcInterpolationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ArcInterpolationParams`/`Const_ArcInterpolationParams` to pass it to the function.
    public class _InOptConst_ArcInterpolationParams
    {
        public Const_ArcInterpolationParams? Opt;

        public _InOptConst_ArcInterpolationParams() {}
        public _InOptConst_ArcInterpolationParams(Const_ArcInterpolationParams value) {Opt = value;}
        public static implicit operator _InOptConst_ArcInterpolationParams(Const_ArcInterpolationParams value) {return new(value);}
    }

    public enum MoveType : int
    {
        None = -1,
        FastLinear = 0,
        Linear = 1,
        ArcCW = 2,
        ArcCCW = 3,
    }

    public enum ArcPlane : int
    {
        None = -1,
        XY = 17,
        XZ = 18,
        YZ = 19,
    }

    /// Generated from class `MR::GCommand`.
    /// This is the const half of the class.
    public class Const_GCommand : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GCommand(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Destroy", ExactSpelling = true)]
            extern static void __MR_GCommand_Destroy(_Underlying *_this);
            __MR_GCommand_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GCommand() {Dispose(false);}

        // type of command GX (G0, G1, etc). By default - G1
        public unsafe MR.MoveType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_type", ExactSpelling = true)]
                extern static MR.MoveType *__MR_GCommand_Get_type(_Underlying *_this);
                return *__MR_GCommand_Get_type(_UnderlyingPtr);
            }
        }

        // Place for comment
        public unsafe MR.ArcPlane ArcPlane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_arcPlane", ExactSpelling = true)]
                extern static MR.ArcPlane *__MR_GCommand_Get_arcPlane(_Underlying *_this);
                return *__MR_GCommand_Get_arcPlane(_UnderlyingPtr);
            }
        }

        // feedrate for move
        public unsafe float Feed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_feed", ExactSpelling = true)]
                extern static float *__MR_GCommand_Get_feed(_Underlying *_this);
                return *__MR_GCommand_Get_feed(_UnderlyingPtr);
            }
        }

        // coordinates of destination point
        public unsafe float X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_x", ExactSpelling = true)]
                extern static float *__MR_GCommand_Get_x(_Underlying *_this);
                return *__MR_GCommand_Get_x(_UnderlyingPtr);
            }
        }

        public unsafe float Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_y", ExactSpelling = true)]
                extern static float *__MR_GCommand_Get_y(_Underlying *_this);
                return *__MR_GCommand_Get_y(_UnderlyingPtr);
            }
        }

        public unsafe float Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_z", ExactSpelling = true)]
                extern static float *__MR_GCommand_Get_z(_Underlying *_this);
                return *__MR_GCommand_Get_z(_UnderlyingPtr);
            }
        }

        // if moveType is ArcCW or ArcCCW center of the arc shoult be specified
        public unsafe MR.Const_Vector3f ArcCenter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_Get_arcCenter", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_GCommand_Get_arcCenter(_Underlying *_this);
                return new(__MR_GCommand_Get_arcCenter(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GCommand() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_DefaultConstruct();
            _UnderlyingPtr = __MR_GCommand_DefaultConstruct();
        }

        /// Constructs `MR::GCommand` elementwise.
        public unsafe Const_GCommand(MR.MoveType type, MR.ArcPlane arcPlane, float feed, float x, float y, float z, MR.Vector3f arcCenter) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_ConstructFrom", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_ConstructFrom(MR.MoveType type, MR.ArcPlane arcPlane, float feed, float x, float y, float z, MR.Vector3f arcCenter);
            _UnderlyingPtr = __MR_GCommand_ConstructFrom(type, arcPlane, feed, x, y, z, arcCenter);
        }

        /// Generated from constructor `MR::GCommand::GCommand`.
        public unsafe Const_GCommand(MR.Const_GCommand _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_ConstructFromAnother(MR.GCommand._Underlying *_other);
            _UnderlyingPtr = __MR_GCommand_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::GCommand`.
    /// This is the non-const half of the class.
    public class GCommand : Const_GCommand
    {
        internal unsafe GCommand(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // type of command GX (G0, G1, etc). By default - G1
        public new unsafe ref MR.MoveType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_type", ExactSpelling = true)]
                extern static MR.MoveType *__MR_GCommand_GetMutable_type(_Underlying *_this);
                return ref *__MR_GCommand_GetMutable_type(_UnderlyingPtr);
            }
        }

        // Place for comment
        public new unsafe ref MR.ArcPlane ArcPlane
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_arcPlane", ExactSpelling = true)]
                extern static MR.ArcPlane *__MR_GCommand_GetMutable_arcPlane(_Underlying *_this);
                return ref *__MR_GCommand_GetMutable_arcPlane(_UnderlyingPtr);
            }
        }

        // feedrate for move
        public new unsafe ref float Feed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_feed", ExactSpelling = true)]
                extern static float *__MR_GCommand_GetMutable_feed(_Underlying *_this);
                return ref *__MR_GCommand_GetMutable_feed(_UnderlyingPtr);
            }
        }

        // coordinates of destination point
        public new unsafe ref float X
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_x", ExactSpelling = true)]
                extern static float *__MR_GCommand_GetMutable_x(_Underlying *_this);
                return ref *__MR_GCommand_GetMutable_x(_UnderlyingPtr);
            }
        }

        public new unsafe ref float Y
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_y", ExactSpelling = true)]
                extern static float *__MR_GCommand_GetMutable_y(_Underlying *_this);
                return ref *__MR_GCommand_GetMutable_y(_UnderlyingPtr);
            }
        }

        public new unsafe ref float Z
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_z", ExactSpelling = true)]
                extern static float *__MR_GCommand_GetMutable_z(_Underlying *_this);
                return ref *__MR_GCommand_GetMutable_z(_UnderlyingPtr);
            }
        }

        // if moveType is ArcCW or ArcCCW center of the arc shoult be specified
        public new unsafe MR.Mut_Vector3f ArcCenter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_GetMutable_arcCenter", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_GCommand_GetMutable_arcCenter(_Underlying *_this);
                return new(__MR_GCommand_GetMutable_arcCenter(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe GCommand() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_DefaultConstruct();
            _UnderlyingPtr = __MR_GCommand_DefaultConstruct();
        }

        /// Constructs `MR::GCommand` elementwise.
        public unsafe GCommand(MR.MoveType type, MR.ArcPlane arcPlane, float feed, float x, float y, float z, MR.Vector3f arcCenter) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_ConstructFrom", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_ConstructFrom(MR.MoveType type, MR.ArcPlane arcPlane, float feed, float x, float y, float z, MR.Vector3f arcCenter);
            _UnderlyingPtr = __MR_GCommand_ConstructFrom(type, arcPlane, feed, x, y, z, arcCenter);
        }

        /// Generated from constructor `MR::GCommand::GCommand`.
        public unsafe GCommand(MR.Const_GCommand _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_ConstructFromAnother(MR.GCommand._Underlying *_other);
            _UnderlyingPtr = __MR_GCommand_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::GCommand::operator=`.
        public unsafe MR.GCommand Assign(MR.Const_GCommand _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GCommand_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GCommand._Underlying *__MR_GCommand_AssignFromAnother(_Underlying *_this, MR.GCommand._Underlying *_other);
            return new(__MR_GCommand_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `GCommand` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GCommand`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GCommand`/`Const_GCommand` directly.
    public class _InOptMut_GCommand
    {
        public GCommand? Opt;

        public _InOptMut_GCommand() {}
        public _InOptMut_GCommand(GCommand value) {Opt = value;}
        public static implicit operator _InOptMut_GCommand(GCommand value) {return new(value);}
    }

    /// This is used for optional parameters of class `GCommand` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GCommand`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GCommand`/`Const_GCommand` to pass it to the function.
    public class _InOptConst_GCommand
    {
        public Const_GCommand? Opt;

        public _InOptConst_GCommand() {}
        public _InOptConst_GCommand(Const_GCommand value) {Opt = value;}
        public static implicit operator _InOptConst_GCommand(Const_GCommand value) {return new(value);}
    }

    /// Generated from class `MR::ToolPathResult`.
    /// This is the const half of the class.
    public class Const_ToolPathResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ToolPathResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_Destroy", ExactSpelling = true)]
            extern static void __MR_ToolPathResult_Destroy(_Underlying *_this);
            __MR_ToolPathResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ToolPathResult() {Dispose(false);}

        // mesh after fixing undercuts and offset
        public unsafe MR.Const_Mesh ModifiedMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_Get_modifiedMesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_ToolPathResult_Get_modifiedMesh(_Underlying *_this);
                return new(__MR_ToolPathResult_Get_modifiedMesh(_UnderlyingPtr), is_owning: false);
            }
        }

        // selected region projected from the original mesh to the offset
        public unsafe MR.Const_FaceBitSet ModifiedRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_Get_modifiedRegion", ExactSpelling = true)]
                extern static MR.Const_FaceBitSet._Underlying *__MR_ToolPathResult_Get_modifiedRegion(_Underlying *_this);
                return new(__MR_ToolPathResult_Get_modifiedRegion(_UnderlyingPtr), is_owning: false);
            }
        }

        // constains type of movement and its feed
        public unsafe MR.Std.Const_Vector_MRGCommand Commands
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_Get_commands", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRGCommand._Underlying *__MR_ToolPathResult_Get_commands(_Underlying *_this);
                return new(__MR_ToolPathResult_Get_commands(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ToolPathResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_DefaultConstruct();
            _UnderlyingPtr = __MR_ToolPathResult_DefaultConstruct();
        }

        /// Constructs `MR::ToolPathResult` elementwise.
        public unsafe Const_ToolPathResult(MR._ByValue_Mesh modifiedMesh, MR._ByValue_FaceBitSet modifiedRegion, MR.Std._ByValue_Vector_MRGCommand commands) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_ConstructFrom(MR.Misc._PassBy modifiedMesh_pass_by, MR.Mesh._Underlying *modifiedMesh, MR.Misc._PassBy modifiedRegion_pass_by, MR.FaceBitSet._Underlying *modifiedRegion, MR.Misc._PassBy commands_pass_by, MR.Std.Vector_MRGCommand._Underlying *commands);
            _UnderlyingPtr = __MR_ToolPathResult_ConstructFrom(modifiedMesh.PassByMode, modifiedMesh.Value is not null ? modifiedMesh.Value._UnderlyingPtr : null, modifiedRegion.PassByMode, modifiedRegion.Value is not null ? modifiedRegion.Value._UnderlyingPtr : null, commands.PassByMode, commands.Value is not null ? commands.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ToolPathResult::ToolPathResult`.
        public unsafe Const_ToolPathResult(MR._ByValue_ToolPathResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ToolPathResult._Underlying *_other);
            _UnderlyingPtr = __MR_ToolPathResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ToolPathResult`.
    /// This is the non-const half of the class.
    public class ToolPathResult : Const_ToolPathResult
    {
        internal unsafe ToolPathResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // mesh after fixing undercuts and offset
        public new unsafe MR.Mesh ModifiedMesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_GetMutable_modifiedMesh", ExactSpelling = true)]
                extern static MR.Mesh._Underlying *__MR_ToolPathResult_GetMutable_modifiedMesh(_Underlying *_this);
                return new(__MR_ToolPathResult_GetMutable_modifiedMesh(_UnderlyingPtr), is_owning: false);
            }
        }

        // selected region projected from the original mesh to the offset
        public new unsafe MR.FaceBitSet ModifiedRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_GetMutable_modifiedRegion", ExactSpelling = true)]
                extern static MR.FaceBitSet._Underlying *__MR_ToolPathResult_GetMutable_modifiedRegion(_Underlying *_this);
                return new(__MR_ToolPathResult_GetMutable_modifiedRegion(_UnderlyingPtr), is_owning: false);
            }
        }

        // constains type of movement and its feed
        public new unsafe MR.Std.Vector_MRGCommand Commands
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_GetMutable_commands", ExactSpelling = true)]
                extern static MR.Std.Vector_MRGCommand._Underlying *__MR_ToolPathResult_GetMutable_commands(_Underlying *_this);
                return new(__MR_ToolPathResult_GetMutable_commands(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ToolPathResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_DefaultConstruct();
            _UnderlyingPtr = __MR_ToolPathResult_DefaultConstruct();
        }

        /// Constructs `MR::ToolPathResult` elementwise.
        public unsafe ToolPathResult(MR._ByValue_Mesh modifiedMesh, MR._ByValue_FaceBitSet modifiedRegion, MR.Std._ByValue_Vector_MRGCommand commands) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_ConstructFrom(MR.Misc._PassBy modifiedMesh_pass_by, MR.Mesh._Underlying *modifiedMesh, MR.Misc._PassBy modifiedRegion_pass_by, MR.FaceBitSet._Underlying *modifiedRegion, MR.Misc._PassBy commands_pass_by, MR.Std.Vector_MRGCommand._Underlying *commands);
            _UnderlyingPtr = __MR_ToolPathResult_ConstructFrom(modifiedMesh.PassByMode, modifiedMesh.Value is not null ? modifiedMesh.Value._UnderlyingPtr : null, modifiedRegion.PassByMode, modifiedRegion.Value is not null ? modifiedRegion.Value._UnderlyingPtr : null, commands.PassByMode, commands.Value is not null ? commands.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ToolPathResult::ToolPathResult`.
        public unsafe ToolPathResult(MR._ByValue_ToolPathResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ToolPathResult._Underlying *_other);
            _UnderlyingPtr = __MR_ToolPathResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ToolPathResult::operator=`.
        public unsafe MR.ToolPathResult Assign(MR._ByValue_ToolPathResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ToolPathResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ToolPathResult._Underlying *__MR_ToolPathResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ToolPathResult._Underlying *_other);
            return new(__MR_ToolPathResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ToolPathResult` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ToolPathResult`/`Const_ToolPathResult` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ToolPathResult
    {
        internal readonly Const_ToolPathResult? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ToolPathResult() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ToolPathResult(Const_ToolPathResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ToolPathResult(Const_ToolPathResult arg) {return new(arg);}
        public _ByValue_ToolPathResult(MR.Misc._Moved<ToolPathResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ToolPathResult(MR.Misc._Moved<ToolPathResult> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ToolPathResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ToolPathResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ToolPathResult`/`Const_ToolPathResult` directly.
    public class _InOptMut_ToolPathResult
    {
        public ToolPathResult? Opt;

        public _InOptMut_ToolPathResult() {}
        public _InOptMut_ToolPathResult(ToolPathResult value) {Opt = value;}
        public static implicit operator _InOptMut_ToolPathResult(ToolPathResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `ToolPathResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ToolPathResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ToolPathResult`/`Const_ToolPathResult` to pass it to the function.
    public class _InOptConst_ToolPathResult
    {
        public Const_ToolPathResult? Opt;

        public _InOptConst_ToolPathResult() {}
        public _InOptConst_ToolPathResult(Const_ToolPathResult value) {Opt = value;}
        public static implicit operator _InOptConst_ToolPathResult(Const_ToolPathResult value) {return new(value);}
    }

    // compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
    // this toolpath is built from the parallel sections along Z-axis
    // mesh can be transformed using xf parameter
    /// Generated from function `MR::constantZToolPath`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRToolPathResult_StdString> ConstantZToolPath(MR.Const_MeshPart mp, MR.Const_ToolPathParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_constantZToolPath", ExactSpelling = true)]
        extern static MR.Expected_MRToolPathResult_StdString._Underlying *__MR_constantZToolPath(MR.Const_MeshPart._Underlying *mp, MR.Const_ToolPathParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRToolPathResult_StdString(__MR_constantZToolPath(mp._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    // Slices are built along the axis defined by cutDirection argument (can be Axis::X or Axis::Y)
    /// Generated from function `MR::lacingToolPath`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRToolPathResult_StdString> LacingToolPath(MR.Const_MeshPart mp, MR.Const_ToolPathParams params_, MR.Axis cutDirection)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_lacingToolPath", ExactSpelling = true)]
        extern static MR.Expected_MRToolPathResult_StdString._Underlying *__MR_lacingToolPath(MR.Const_MeshPart._Underlying *mp, MR.Const_ToolPathParams._Underlying *params_, MR.Axis cutDirection);
        return MR.Misc.Move(new MR.Expected_MRToolPathResult_StdString(__MR_lacingToolPath(mp._UnderlyingPtr, params_._UnderlyingPtr, cutDirection), is_owning: true));
    }

    // compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
    // this toolpath is built from geodesic parallels divercing from the given start point or from the bounaries of selected areas
    // if neither is specified, the lowest section by XY plane will be used as a start contour
    // mesh can be transformed using xf parameter
    /// Generated from function `MR::constantCuspToolPath`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRToolPathResult_StdString> ConstantCuspToolPath(MR.Const_MeshPart mp, MR.Const_ConstantCuspParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_constantCuspToolPath", ExactSpelling = true)]
        extern static MR.Expected_MRToolPathResult_StdString._Underlying *__MR_constantCuspToolPath(MR.Const_MeshPart._Underlying *mp, MR.Const_ConstantCuspParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRToolPathResult_StdString(__MR_constantCuspToolPath(mp._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    // generates G-Code for milling tool
    /// Generated from function `MR::exportToolPathToGCode`.
    public static unsafe MR.Misc._Moved<MR.ObjectGcode> ExportToolPathToGCode(MR.Std.Const_Vector_MRGCommand commands)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_exportToolPathToGCode", ExactSpelling = true)]
        extern static MR.ObjectGcode._UnderlyingShared *__MR_exportToolPathToGCode(MR.Std.Const_Vector_MRGCommand._Underlying *commands);
        return MR.Misc.Move(new MR.ObjectGcode(__MR_exportToolPathToGCode(commands._UnderlyingPtr), is_owning: true));
    }

    // interpolates several points lying on the same straight line with one move
    /// Generated from function `MR::interpolateLines`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> InterpolateLines(MR.Std.Vector_MRGCommand commands, MR.Const_LineInterpolationParams params_, MR.Axis axis)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_interpolateLines", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_interpolateLines(MR.Std.Vector_MRGCommand._Underlying *commands, MR.Const_LineInterpolationParams._Underlying *params_, MR.Axis axis);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_interpolateLines(commands._UnderlyingPtr, params_._UnderlyingPtr, axis), is_owning: true));
    }

    // interpolates given path with arcs
    /// Generated from function `MR::interpolateArcs`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> InterpolateArcs(MR.Std.Vector_MRGCommand commands, MR.Const_ArcInterpolationParams params_, MR.Axis axis)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_interpolateArcs", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_interpolateArcs(MR.Std.Vector_MRGCommand._Underlying *commands, MR.Const_ArcInterpolationParams._Underlying *params_, MR.Axis axis);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_interpolateArcs(commands._UnderlyingPtr, params_._UnderlyingPtr, axis), is_owning: true));
    }

    // makes the given selection more smooth with shifthing a boundary of the selection outside and back. Input mesh is changed because we have to cut new edges along the new boundaries
    // \param expandOffset defines how much the boundary is expanded
    // \param expandOffset defines how much the boundary is shrinked after that
    /// Generated from function `MR::smoothSelection`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> SmoothSelection(MR.Mesh mesh, MR.Const_FaceBitSet region, float expandOffset, float shrinkOffset)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_smoothSelection", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_smoothSelection(MR.Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *region, float expandOffset, float shrinkOffset);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_smoothSelection(mesh._UnderlyingPtr, region._UnderlyingPtr, expandOffset, shrinkOffset), is_owning: true));
    }
}
