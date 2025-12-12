public static partial class MR
{
    public enum Side : int
    {
        Negative = 0,
        Positive = 1,
    }

    /// Generated from class `MR::FindInnerShellSettings`.
    /// This is the const half of the class.
    public class Const_FindInnerShellSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FindInnerShellSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_FindInnerShellSettings_Destroy(_Underlying *_this);
            __MR_FindInnerShellSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FindInnerShellSettings() {Dispose(false);}

        /// specifies which side of shell is of interest: negative or positive relative to mesh normals
        public unsafe MR.Side Side
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_Get_side", ExactSpelling = true)]
                extern static MR.Side *__MR_FindInnerShellSettings_Get_side(_Underlying *_this);
                return *__MR_FindInnerShellSettings_Get_side(_UnderlyingPtr);
            }
        }

        /// specifies maximum squared distance from shell parts of interest to source mesh
        public unsafe float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_Get_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_FindInnerShellSettings_Get_maxDistSq(_Underlying *_this);
                return *__MR_FindInnerShellSettings_Get_maxDistSq(_UnderlyingPtr);
            }
        }

        /// if true, a slower algorithm is activated that is more robust in the presence of self-intersections on mesh
        public unsafe bool UseWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_Get_useWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_FindInnerShellSettings_Get_useWindingNumber(_Underlying *_this);
                return *__MR_FindInnerShellSettings_Get_useWindingNumber(_UnderlyingPtr);
            }
        }

        /// positive side if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_FindInnerShellSettings_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_FindInnerShellSettings_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// components of proper side with smaller number of vertices than this value will be removed from the result;
        /// components of wrong side with smaller number of vertices than this value will be added to the result
        public unsafe int MinVertsInComp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_Get_minVertsInComp", ExactSpelling = true)]
                extern static int *__MR_FindInnerShellSettings_Get_minVertsInComp(_Underlying *_this);
                return *__MR_FindInnerShellSettings_Get_minVertsInComp(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FindInnerShellSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindInnerShellSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindInnerShellSettings` elementwise.
        public unsafe Const_FindInnerShellSettings(MR.Side side, float maxDistSq, bool useWindingNumber, float windingNumberThreshold, int minVertsInComp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_ConstructFrom(MR.Side side, float maxDistSq, byte useWindingNumber, float windingNumberThreshold, int minVertsInComp);
            _UnderlyingPtr = __MR_FindInnerShellSettings_ConstructFrom(side, maxDistSq, useWindingNumber ? (byte)1 : (byte)0, windingNumberThreshold, minVertsInComp);
        }

        /// Generated from constructor `MR::FindInnerShellSettings::FindInnerShellSettings`.
        public unsafe Const_FindInnerShellSettings(MR.Const_FindInnerShellSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_ConstructFromAnother(MR.FindInnerShellSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindInnerShellSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::FindInnerShellSettings`.
    /// This is the non-const half of the class.
    public class FindInnerShellSettings : Const_FindInnerShellSettings
    {
        internal unsafe FindInnerShellSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// specifies which side of shell is of interest: negative or positive relative to mesh normals
        public new unsafe ref MR.Side Side
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_GetMutable_side", ExactSpelling = true)]
                extern static MR.Side *__MR_FindInnerShellSettings_GetMutable_side(_Underlying *_this);
                return ref *__MR_FindInnerShellSettings_GetMutable_side(_UnderlyingPtr);
            }
        }

        /// specifies maximum squared distance from shell parts of interest to source mesh
        public new unsafe ref float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_GetMutable_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_FindInnerShellSettings_GetMutable_maxDistSq(_Underlying *_this);
                return ref *__MR_FindInnerShellSettings_GetMutable_maxDistSq(_UnderlyingPtr);
            }
        }

        /// if true, a slower algorithm is activated that is more robust in the presence of self-intersections on mesh
        public new unsafe ref bool UseWindingNumber
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_GetMutable_useWindingNumber", ExactSpelling = true)]
                extern static bool *__MR_FindInnerShellSettings_GetMutable_useWindingNumber(_Underlying *_this);
                return ref *__MR_FindInnerShellSettings_GetMutable_useWindingNumber(_UnderlyingPtr);
            }
        }

        /// positive side if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_FindInnerShellSettings_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_FindInnerShellSettings_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// components of proper side with smaller number of vertices than this value will be removed from the result;
        /// components of wrong side with smaller number of vertices than this value will be added to the result
        public new unsafe ref int MinVertsInComp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_GetMutable_minVertsInComp", ExactSpelling = true)]
                extern static int *__MR_FindInnerShellSettings_GetMutable_minVertsInComp(_Underlying *_this);
                return ref *__MR_FindInnerShellSettings_GetMutable_minVertsInComp(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FindInnerShellSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindInnerShellSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindInnerShellSettings` elementwise.
        public unsafe FindInnerShellSettings(MR.Side side, float maxDistSq, bool useWindingNumber, float windingNumberThreshold, int minVertsInComp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_ConstructFrom(MR.Side side, float maxDistSq, byte useWindingNumber, float windingNumberThreshold, int minVertsInComp);
            _UnderlyingPtr = __MR_FindInnerShellSettings_ConstructFrom(side, maxDistSq, useWindingNumber ? (byte)1 : (byte)0, windingNumberThreshold, minVertsInComp);
        }

        /// Generated from constructor `MR::FindInnerShellSettings::FindInnerShellSettings`.
        public unsafe FindInnerShellSettings(MR.Const_FindInnerShellSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_ConstructFromAnother(MR.FindInnerShellSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindInnerShellSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::FindInnerShellSettings::operator=`.
        public unsafe MR.FindInnerShellSettings Assign(MR.Const_FindInnerShellSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindInnerShellSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FindInnerShellSettings._Underlying *__MR_FindInnerShellSettings_AssignFromAnother(_Underlying *_this, MR.FindInnerShellSettings._Underlying *_other);
            return new(__MR_FindInnerShellSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FindInnerShellSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FindInnerShellSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindInnerShellSettings`/`Const_FindInnerShellSettings` directly.
    public class _InOptMut_FindInnerShellSettings
    {
        public FindInnerShellSettings? Opt;

        public _InOptMut_FindInnerShellSettings() {}
        public _InOptMut_FindInnerShellSettings(FindInnerShellSettings value) {Opt = value;}
        public static implicit operator _InOptMut_FindInnerShellSettings(FindInnerShellSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `FindInnerShellSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FindInnerShellSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindInnerShellSettings`/`Const_FindInnerShellSettings` to pass it to the function.
    public class _InOptConst_FindInnerShellSettings
    {
        public Const_FindInnerShellSettings? Opt;

        public _InOptConst_FindInnerShellSettings() {}
        public _InOptConst_FindInnerShellSettings(Const_FindInnerShellSettings value) {Opt = value;}
        public static implicit operator _InOptConst_FindInnerShellSettings(Const_FindInnerShellSettings value) {return new(value);}
    }

    /// information about shell vertex
    /// Generated from class `MR::ShellVertexInfo`.
    /// This is the const half of the class.
    public class Const_ShellVertexInfo : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ShellVertexInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_Destroy", ExactSpelling = true)]
            extern static void __MR_ShellVertexInfo_Destroy(_Underlying *_this);
            __MR_ShellVertexInfo_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ShellVertexInfo() {Dispose(false);}

        /// true when shell vertex is within settings.maxDist from source mesh
        public unsafe bool InRange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_Get_inRange", ExactSpelling = true)]
                extern static bool *__MR_ShellVertexInfo_Get_inRange(_Underlying *_this);
                return *__MR_ShellVertexInfo_Get_inRange(_UnderlyingPtr);
            }
        }

        /// shell vertex projects on source mesh boundary (never true for winding rule mode)
        public unsafe bool ProjOnBd
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_Get_projOnBd", ExactSpelling = true)]
                extern static bool *__MR_ShellVertexInfo_Get_projOnBd(_Underlying *_this);
                return *__MR_ShellVertexInfo_Get_projOnBd(_UnderlyingPtr);
            }
        }

        /// whether shell vertex is on requested side of source mesh
        public unsafe bool RightSide
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_Get_rightSide", ExactSpelling = true)]
                extern static bool *__MR_ShellVertexInfo_Get_rightSide(_Underlying *_this);
                return *__MR_ShellVertexInfo_Get_rightSide(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ShellVertexInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_ShellVertexInfo_DefaultConstruct();
        }

        /// Constructs `MR::ShellVertexInfo` elementwise.
        public unsafe Const_ShellVertexInfo(bool inRange, bool projOnBd, bool rightSide) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_ConstructFrom", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_ConstructFrom(byte inRange, byte projOnBd, byte rightSide);
            _UnderlyingPtr = __MR_ShellVertexInfo_ConstructFrom(inRange ? (byte)1 : (byte)0, projOnBd ? (byte)1 : (byte)0, rightSide ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::ShellVertexInfo::ShellVertexInfo`.
        public unsafe Const_ShellVertexInfo(MR.Const_ShellVertexInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_ConstructFromAnother(MR.ShellVertexInfo._Underlying *_other);
            _UnderlyingPtr = __MR_ShellVertexInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if shell vertex is in range, does not project on boundary and located on proper side
        /// Generated from method `MR::ShellVertexInfo::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_valid", ExactSpelling = true)]
            extern static byte __MR_ShellVertexInfo_valid(_Underlying *_this);
            return __MR_ShellVertexInfo_valid(_UnderlyingPtr) != 0;
        }
    }

    /// information about shell vertex
    /// Generated from class `MR::ShellVertexInfo`.
    /// This is the non-const half of the class.
    public class ShellVertexInfo : Const_ShellVertexInfo
    {
        internal unsafe ShellVertexInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// true when shell vertex is within settings.maxDist from source mesh
        public new unsafe ref bool InRange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_GetMutable_inRange", ExactSpelling = true)]
                extern static bool *__MR_ShellVertexInfo_GetMutable_inRange(_Underlying *_this);
                return ref *__MR_ShellVertexInfo_GetMutable_inRange(_UnderlyingPtr);
            }
        }

        /// shell vertex projects on source mesh boundary (never true for winding rule mode)
        public new unsafe ref bool ProjOnBd
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_GetMutable_projOnBd", ExactSpelling = true)]
                extern static bool *__MR_ShellVertexInfo_GetMutable_projOnBd(_Underlying *_this);
                return ref *__MR_ShellVertexInfo_GetMutable_projOnBd(_UnderlyingPtr);
            }
        }

        /// whether shell vertex is on requested side of source mesh
        public new unsafe ref bool RightSide
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_GetMutable_rightSide", ExactSpelling = true)]
                extern static bool *__MR_ShellVertexInfo_GetMutable_rightSide(_Underlying *_this);
                return ref *__MR_ShellVertexInfo_GetMutable_rightSide(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ShellVertexInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_ShellVertexInfo_DefaultConstruct();
        }

        /// Constructs `MR::ShellVertexInfo` elementwise.
        public unsafe ShellVertexInfo(bool inRange, bool projOnBd, bool rightSide) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_ConstructFrom", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_ConstructFrom(byte inRange, byte projOnBd, byte rightSide);
            _UnderlyingPtr = __MR_ShellVertexInfo_ConstructFrom(inRange ? (byte)1 : (byte)0, projOnBd ? (byte)1 : (byte)0, rightSide ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::ShellVertexInfo::ShellVertexInfo`.
        public unsafe ShellVertexInfo(MR.Const_ShellVertexInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_ConstructFromAnother(MR.ShellVertexInfo._Underlying *_other);
            _UnderlyingPtr = __MR_ShellVertexInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ShellVertexInfo::operator=`.
        public unsafe MR.ShellVertexInfo Assign(MR.Const_ShellVertexInfo _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ShellVertexInfo_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ShellVertexInfo._Underlying *__MR_ShellVertexInfo_AssignFromAnother(_Underlying *_this, MR.ShellVertexInfo._Underlying *_other);
            return new(__MR_ShellVertexInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ShellVertexInfo` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ShellVertexInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ShellVertexInfo`/`Const_ShellVertexInfo` directly.
    public class _InOptMut_ShellVertexInfo
    {
        public ShellVertexInfo? Opt;

        public _InOptMut_ShellVertexInfo() {}
        public _InOptMut_ShellVertexInfo(ShellVertexInfo value) {Opt = value;}
        public static implicit operator _InOptMut_ShellVertexInfo(ShellVertexInfo value) {return new(value);}
    }

    /// This is used for optional parameters of class `ShellVertexInfo` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ShellVertexInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ShellVertexInfo`/`Const_ShellVertexInfo` to pass it to the function.
    public class _InOptConst_ShellVertexInfo
    {
        public Const_ShellVertexInfo? Opt;

        public _InOptConst_ShellVertexInfo() {}
        public _InOptConst_ShellVertexInfo(Const_ShellVertexInfo value) {Opt = value;}
        public static implicit operator _InOptConst_ShellVertexInfo(Const_ShellVertexInfo value) {return new(value);}
    }

    /// Tests \param shellPoint from bidirectional shell constructed for an open \param mp;
    /// \return whether the distance from given point to given mesh part is of same sign as settings.side,
    /// if useWindingNumber = false, returns false for all points projecting on mesh boundary
    /// Generated from function `MR::classifyShellVert`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.ShellVertexInfo ClassifyShellVert(MR.Const_MeshPart mp, MR.Const_Vector3f shellPoint, MR.Const_FindInnerShellSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_classifyShellVert", ExactSpelling = true)]
        extern static MR.ShellVertexInfo._Underlying *__MR_classifyShellVert(MR.Const_MeshPart._Underlying *mp, MR.Const_Vector3f._Underlying *shellPoint, MR.Const_FindInnerShellSettings._Underlying *settings);
        return new(__MR_classifyShellVert(mp._UnderlyingPtr, shellPoint._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true);
    }

    /// Finds inner-shell vertices on bidirectional \param shell constructed for an open \param mp;
    /// The function will return all shell vertices that have distance to mesh of same sign as settings.side
    /// Generated from function `MR::findInnerShellVerts`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> FindInnerShellVerts(MR.Const_MeshPart mp, MR.Const_Mesh shell, MR.Const_FindInnerShellSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findInnerShellVerts", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_findInnerShellVerts(MR.Const_MeshPart._Underlying *mp, MR.Const_Mesh._Underlying *shell, MR.Const_FindInnerShellSettings._Underlying *settings);
        return MR.Misc.Move(new MR.VertBitSet(__MR_findInnerShellVerts(mp._UnderlyingPtr, shell._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }

    /// Finds inner-shell faces on bidirectional \param shell constructed for an open \param mp;
    /// The function will return all shell faces (after some subdivision) that have distance to mesh of same sign as settings.side
    /// Generated from function `MR::findInnerShellFacesWithSplits`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FindInnerShellFacesWithSplits(MR.Const_MeshPart mp, MR.Mesh shell, MR.Const_FindInnerShellSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findInnerShellFacesWithSplits", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_findInnerShellFacesWithSplits(MR.Const_MeshPart._Underlying *mp, MR.Mesh._Underlying *shell, MR.Const_FindInnerShellSettings._Underlying *settings);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_findInnerShellFacesWithSplits(mp._UnderlyingPtr, shell._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }
}
