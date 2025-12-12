public static partial class MR
{
    /// Generated from class `MR::DeloneSettings`.
    /// This is the const half of the class.
    public class Const_DeloneSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DeloneSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DeloneSettings_Destroy(_Underlying *_this);
            __MR_DeloneSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DeloneSettings() {Dispose(false);}

        /// Maximal allowed surface deviation during every individual flip
        public unsafe float MaxDeviationAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Get_maxDeviationAfterFlip", ExactSpelling = true)]
                extern static float *__MR_DeloneSettings_Get_maxDeviationAfterFlip(_Underlying *_this);
                return *__MR_DeloneSettings_Get_maxDeviationAfterFlip(_UnderlyingPtr);
            }
        }

        /// Maximal allowed dihedral angle change (in radians) over the flipped edge
        public unsafe float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Get_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_DeloneSettings_Get_maxAngleChange(_Underlying *_this);
                return *__MR_DeloneSettings_Get_maxAngleChange(_UnderlyingPtr);
            }
        }

        /// if this value is less than FLT_MAX then the algorithm will
        /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
        public unsafe float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Get_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_DeloneSettings_Get_criticalTriAspectRatio(_Underlying *_this);
                return *__MR_DeloneSettings_Get_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// Only edges with left and right faces in this set can be flipped
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_DeloneSettings_Get_region(_Underlying *_this);
                return ref *__MR_DeloneSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped
        public unsafe ref readonly void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Get_notFlippable", ExactSpelling = true)]
                extern static void **__MR_DeloneSettings_Get_notFlippable(_Underlying *_this);
                return ref *__MR_DeloneSettings_Get_notFlippable(_UnderlyingPtr);
            }
        }

        /// Only edges with origin or destination in this set before or after flip can be flipped
        public unsafe ref readonly void * VertRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_Get_vertRegion", ExactSpelling = true)]
                extern static void **__MR_DeloneSettings_Get_vertRegion(_Underlying *_this);
                return ref *__MR_DeloneSettings_Get_vertRegion(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DeloneSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DeloneSettings_DefaultConstruct();
        }

        /// Constructs `MR::DeloneSettings` elementwise.
        public unsafe Const_DeloneSettings(float maxDeviationAfterFlip, float maxAngleChange, float criticalTriAspectRatio, MR.Const_FaceBitSet? region, MR.Const_UndirectedEdgeBitSet? notFlippable, MR.Const_VertBitSet? vertRegion) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_ConstructFrom(float maxDeviationAfterFlip, float maxAngleChange, float criticalTriAspectRatio, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *notFlippable, MR.Const_VertBitSet._Underlying *vertRegion);
            _UnderlyingPtr = __MR_DeloneSettings_ConstructFrom(maxDeviationAfterFlip, maxAngleChange, criticalTriAspectRatio, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DeloneSettings::DeloneSettings`.
        public unsafe Const_DeloneSettings(MR.Const_DeloneSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_ConstructFromAnother(MR.DeloneSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DeloneSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::DeloneSettings`.
    /// This is the non-const half of the class.
    public class DeloneSettings : Const_DeloneSettings
    {
        internal unsafe DeloneSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Maximal allowed surface deviation during every individual flip
        public new unsafe ref float MaxDeviationAfterFlip
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_GetMutable_maxDeviationAfterFlip", ExactSpelling = true)]
                extern static float *__MR_DeloneSettings_GetMutable_maxDeviationAfterFlip(_Underlying *_this);
                return ref *__MR_DeloneSettings_GetMutable_maxDeviationAfterFlip(_UnderlyingPtr);
            }
        }

        /// Maximal allowed dihedral angle change (in radians) over the flipped edge
        public new unsafe ref float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_GetMutable_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_DeloneSettings_GetMutable_maxAngleChange(_Underlying *_this);
                return ref *__MR_DeloneSettings_GetMutable_maxAngleChange(_UnderlyingPtr);
            }
        }

        /// if this value is less than FLT_MAX then the algorithm will
        /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
        public new unsafe ref float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_GetMutable_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_DeloneSettings_GetMutable_criticalTriAspectRatio(_Underlying *_this);
                return ref *__MR_DeloneSettings_GetMutable_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// Only edges with left and right faces in this set can be flipped
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_DeloneSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_DeloneSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped
        public new unsafe ref readonly void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_GetMutable_notFlippable", ExactSpelling = true)]
                extern static void **__MR_DeloneSettings_GetMutable_notFlippable(_Underlying *_this);
                return ref *__MR_DeloneSettings_GetMutable_notFlippable(_UnderlyingPtr);
            }
        }

        /// Only edges with origin or destination in this set before or after flip can be flipped
        public new unsafe ref readonly void * VertRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_GetMutable_vertRegion", ExactSpelling = true)]
                extern static void **__MR_DeloneSettings_GetMutable_vertRegion(_Underlying *_this);
                return ref *__MR_DeloneSettings_GetMutable_vertRegion(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DeloneSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DeloneSettings_DefaultConstruct();
        }

        /// Constructs `MR::DeloneSettings` elementwise.
        public unsafe DeloneSettings(float maxDeviationAfterFlip, float maxAngleChange, float criticalTriAspectRatio, MR.Const_FaceBitSet? region, MR.Const_UndirectedEdgeBitSet? notFlippable, MR.Const_VertBitSet? vertRegion) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_ConstructFrom(float maxDeviationAfterFlip, float maxAngleChange, float criticalTriAspectRatio, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *notFlippable, MR.Const_VertBitSet._Underlying *vertRegion);
            _UnderlyingPtr = __MR_DeloneSettings_ConstructFrom(maxDeviationAfterFlip, maxAngleChange, criticalTriAspectRatio, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DeloneSettings::DeloneSettings`.
        public unsafe DeloneSettings(MR.Const_DeloneSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_ConstructFromAnother(MR.DeloneSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DeloneSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::DeloneSettings::operator=`.
        public unsafe MR.DeloneSettings Assign(MR.Const_DeloneSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DeloneSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DeloneSettings._Underlying *__MR_DeloneSettings_AssignFromAnother(_Underlying *_this, MR.DeloneSettings._Underlying *_other);
            return new(__MR_DeloneSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DeloneSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DeloneSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DeloneSettings`/`Const_DeloneSettings` directly.
    public class _InOptMut_DeloneSettings
    {
        public DeloneSettings? Opt;

        public _InOptMut_DeloneSettings() {}
        public _InOptMut_DeloneSettings(DeloneSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DeloneSettings(DeloneSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DeloneSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DeloneSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DeloneSettings`/`Const_DeloneSettings` to pass it to the function.
    public class _InOptConst_DeloneSettings
    {
        public Const_DeloneSettings? Opt;

        public _InOptConst_DeloneSettings() {}
        public _InOptConst_DeloneSettings(Const_DeloneSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DeloneSettings(Const_DeloneSettings value) {return new(value);}
    }

    public enum FlipEdge : int
    {
        ///< edge flipping is possible
        Can = 0,
        ///< edge flipping is prohibited by topology or by constraints
        Cannot = 1,
        ///< edge flipping is required to solve some topology issue
        Must = 2,
    }

    /// Generated from class `MR::IntrinsicDeloneSettings`.
    /// This is the const half of the class.
    public class Const_IntrinsicDeloneSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IntrinsicDeloneSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_IntrinsicDeloneSettings_Destroy(_Underlying *_this);
            __MR_IntrinsicDeloneSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IntrinsicDeloneSettings() {Dispose(false);}

        /// the edge is considered Delaunay, if cotan(a1) + cotan(a2) >= threshold;
        /// passing positive(negative) threshold makes less(more) edges satisfy Delaunay conditions
        public unsafe float Threshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_Get_threshold", ExactSpelling = true)]
                extern static float *__MR_IntrinsicDeloneSettings_Get_threshold(_Underlying *_this);
                return *__MR_IntrinsicDeloneSettings_Get_threshold(_UnderlyingPtr);
            }
        }

        /// Only edges with left and right faces in this set can be flipped
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_IntrinsicDeloneSettings_Get_region(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_Get_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped
        public unsafe ref readonly void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_Get_notFlippable", ExactSpelling = true)]
                extern static void **__MR_IntrinsicDeloneSettings_Get_notFlippable(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_Get_notFlippable(_UnderlyingPtr);
            }
        }

        /// Only edges with origin or destination in this set before or after flip can be flipped
        public unsafe ref readonly void * VertRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_Get_vertRegion", ExactSpelling = true)]
                extern static void **__MR_IntrinsicDeloneSettings_Get_vertRegion(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_Get_vertRegion(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IntrinsicDeloneSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_IntrinsicDeloneSettings_DefaultConstruct();
        }

        /// Constructs `MR::IntrinsicDeloneSettings` elementwise.
        public unsafe Const_IntrinsicDeloneSettings(float threshold, MR.Const_FaceBitSet? region, MR.Const_UndirectedEdgeBitSet? notFlippable, MR.Const_VertBitSet? vertRegion) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_ConstructFrom(float threshold, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *notFlippable, MR.Const_VertBitSet._Underlying *vertRegion);
            _UnderlyingPtr = __MR_IntrinsicDeloneSettings_ConstructFrom(threshold, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::IntrinsicDeloneSettings::IntrinsicDeloneSettings`.
        public unsafe Const_IntrinsicDeloneSettings(MR.Const_IntrinsicDeloneSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_ConstructFromAnother(MR.IntrinsicDeloneSettings._Underlying *_other);
            _UnderlyingPtr = __MR_IntrinsicDeloneSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IntrinsicDeloneSettings`.
    /// This is the non-const half of the class.
    public class IntrinsicDeloneSettings : Const_IntrinsicDeloneSettings
    {
        internal unsafe IntrinsicDeloneSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the edge is considered Delaunay, if cotan(a1) + cotan(a2) >= threshold;
        /// passing positive(negative) threshold makes less(more) edges satisfy Delaunay conditions
        public new unsafe ref float Threshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_GetMutable_threshold", ExactSpelling = true)]
                extern static float *__MR_IntrinsicDeloneSettings_GetMutable_threshold(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_GetMutable_threshold(_UnderlyingPtr);
            }
        }

        /// Only edges with left and right faces in this set can be flipped
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_IntrinsicDeloneSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// Edges specified by this bit-set will never be flipped
        public new unsafe ref readonly void * NotFlippable
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_GetMutable_notFlippable", ExactSpelling = true)]
                extern static void **__MR_IntrinsicDeloneSettings_GetMutable_notFlippable(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_GetMutable_notFlippable(_UnderlyingPtr);
            }
        }

        /// Only edges with origin or destination in this set before or after flip can be flipped
        public new unsafe ref readonly void * VertRegion
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_GetMutable_vertRegion", ExactSpelling = true)]
                extern static void **__MR_IntrinsicDeloneSettings_GetMutable_vertRegion(_Underlying *_this);
                return ref *__MR_IntrinsicDeloneSettings_GetMutable_vertRegion(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe IntrinsicDeloneSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_IntrinsicDeloneSettings_DefaultConstruct();
        }

        /// Constructs `MR::IntrinsicDeloneSettings` elementwise.
        public unsafe IntrinsicDeloneSettings(float threshold, MR.Const_FaceBitSet? region, MR.Const_UndirectedEdgeBitSet? notFlippable, MR.Const_VertBitSet? vertRegion) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_ConstructFrom(float threshold, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *notFlippable, MR.Const_VertBitSet._Underlying *vertRegion);
            _UnderlyingPtr = __MR_IntrinsicDeloneSettings_ConstructFrom(threshold, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::IntrinsicDeloneSettings::IntrinsicDeloneSettings`.
        public unsafe IntrinsicDeloneSettings(MR.Const_IntrinsicDeloneSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_ConstructFromAnother(MR.IntrinsicDeloneSettings._Underlying *_other);
            _UnderlyingPtr = __MR_IntrinsicDeloneSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IntrinsicDeloneSettings::operator=`.
        public unsafe MR.IntrinsicDeloneSettings Assign(MR.Const_IntrinsicDeloneSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IntrinsicDeloneSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IntrinsicDeloneSettings._Underlying *__MR_IntrinsicDeloneSettings_AssignFromAnother(_Underlying *_this, MR.IntrinsicDeloneSettings._Underlying *_other);
            return new(__MR_IntrinsicDeloneSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IntrinsicDeloneSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IntrinsicDeloneSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntrinsicDeloneSettings`/`Const_IntrinsicDeloneSettings` directly.
    public class _InOptMut_IntrinsicDeloneSettings
    {
        public IntrinsicDeloneSettings? Opt;

        public _InOptMut_IntrinsicDeloneSettings() {}
        public _InOptMut_IntrinsicDeloneSettings(IntrinsicDeloneSettings value) {Opt = value;}
        public static implicit operator _InOptMut_IntrinsicDeloneSettings(IntrinsicDeloneSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `IntrinsicDeloneSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IntrinsicDeloneSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IntrinsicDeloneSettings`/`Const_IntrinsicDeloneSettings` to pass it to the function.
    public class _InOptConst_IntrinsicDeloneSettings
    {
        public Const_IntrinsicDeloneSettings? Opt;

        public _InOptConst_IntrinsicDeloneSettings() {}
        public _InOptConst_IntrinsicDeloneSettings(Const_IntrinsicDeloneSettings value) {Opt = value;}
        public static implicit operator _InOptConst_IntrinsicDeloneSettings(Const_IntrinsicDeloneSettings value) {return new(value);}
    }

    /// given quadrangle ABCD, checks whether its edge AC satisfies Delone's condition;
    /// if dihedral angles
    ///   1) between triangles ABD and DBC and
    ///   2) between triangles ABC and ACD
    /// differ more than on maxAngleChange then also returns true to prevent flipping from 1) to 2)
    /// Generated from function `MR::checkDeloneQuadrangle`.
    /// Parameter `maxAngleChange` defaults to `1.7976931348623157e308`.
    public static unsafe bool CheckDeloneQuadrangle(MR.Const_Vector3d a, MR.Const_Vector3d b, MR.Const_Vector3d c, MR.Const_Vector3d d, double? maxAngleChange = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_checkDeloneQuadrangle_MR_Vector3d", ExactSpelling = true)]
        extern static byte __MR_checkDeloneQuadrangle_MR_Vector3d(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b, MR.Const_Vector3d._Underlying *c, MR.Const_Vector3d._Underlying *d, double *maxAngleChange);
        double __deref_maxAngleChange = maxAngleChange.GetValueOrDefault();
        return __MR_checkDeloneQuadrangle_MR_Vector3d(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr, maxAngleChange.HasValue ? &__deref_maxAngleChange : null) != 0;
    }

    /// converts arguments in double and calls above function
    /// Generated from function `MR::checkDeloneQuadrangle`.
    /// Parameter `maxAngleChange` defaults to `3.40282347e38f`.
    public static unsafe bool CheckDeloneQuadrangle(MR.Const_Vector3f a, MR.Const_Vector3f b, MR.Const_Vector3f c, MR.Const_Vector3f d, float? maxAngleChange = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_checkDeloneQuadrangle_MR_Vector3f", ExactSpelling = true)]
        extern static byte __MR_checkDeloneQuadrangle_MR_Vector3f(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, MR.Const_Vector3f._Underlying *c, MR.Const_Vector3f._Underlying *d, float *maxAngleChange);
        float __deref_maxAngleChange = maxAngleChange.GetValueOrDefault();
        return __MR_checkDeloneQuadrangle_MR_Vector3f(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr, maxAngleChange.HasValue ? &__deref_maxAngleChange : null) != 0;
    }

    /// consider topology and constraints to decide about flip possibility
    /// Generated from function `MR::canFlipEdge`.
    public static unsafe MR.FlipEdge CanFlipEdge(MR.Const_MeshTopology topology, MR.EdgeId edge, MR.Const_FaceBitSet? region = null, MR.Const_UndirectedEdgeBitSet? notFlippable = null, MR.Const_VertBitSet? vertRegion = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_canFlipEdge", ExactSpelling = true)]
        extern static MR.FlipEdge __MR_canFlipEdge(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge, MR.Const_FaceBitSet._Underlying *region, MR.Const_UndirectedEdgeBitSet._Underlying *notFlippable, MR.Const_VertBitSet._Underlying *vertRegion);
        return __MR_canFlipEdge(topology._UnderlyingPtr, edge, region is not null ? region._UnderlyingPtr : null, notFlippable is not null ? notFlippable._UnderlyingPtr : null, vertRegion is not null ? vertRegion._UnderlyingPtr : null);
    }

    /// consider quadrangle formed by left and right triangles of given edge, and
    /// checks whether this edge satisfies Delone's condition in the quadrangle;
    /// \return false otherwise if flipping the edge does not introduce too large surface deviation (can be returned only for inner edge of the region)
    /// Generated from function `MR::checkDeloneQuadrangleInMesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe bool CheckDeloneQuadrangleInMesh(MR.Const_Mesh mesh, MR.EdgeId edge, MR.Const_DeloneSettings? settings = null, MR.Misc.InOut<float>? deviationSqAfterFlip = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_checkDeloneQuadrangleInMesh_4", ExactSpelling = true)]
        extern static byte __MR_checkDeloneQuadrangleInMesh_4(MR.Const_Mesh._Underlying *mesh, MR.EdgeId edge, MR.Const_DeloneSettings._Underlying *settings, float *deviationSqAfterFlip);
        float __value_deviationSqAfterFlip = deviationSqAfterFlip is not null ? deviationSqAfterFlip.Value : default(float);
        var __ret = __MR_checkDeloneQuadrangleInMesh_4(mesh._UnderlyingPtr, edge, settings is not null ? settings._UnderlyingPtr : null, deviationSqAfterFlip is not null ? &__value_deviationSqAfterFlip : null);
        if (deviationSqAfterFlip is not null) deviationSqAfterFlip.Value = __value_deviationSqAfterFlip;
        return __ret != 0;
    }

    /// Generated from function `MR::checkDeloneQuadrangleInMesh`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe bool CheckDeloneQuadrangleInMesh(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId edge, MR.Const_DeloneSettings? settings = null, MR.Misc.InOut<float>? deviationSqAfterFlip = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_checkDeloneQuadrangleInMesh_5", ExactSpelling = true)]
        extern static byte __MR_checkDeloneQuadrangleInMesh_5(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId edge, MR.Const_DeloneSettings._Underlying *settings, float *deviationSqAfterFlip);
        float __value_deviationSqAfterFlip = deviationSqAfterFlip is not null ? deviationSqAfterFlip.Value : default(float);
        var __ret = __MR_checkDeloneQuadrangleInMesh_5(topology._UnderlyingPtr, points._UnderlyingPtr, edge, settings is not null ? settings._UnderlyingPtr : null, deviationSqAfterFlip is not null ? &__value_deviationSqAfterFlip : null);
        if (deviationSqAfterFlip is not null) deviationSqAfterFlip.Value = __value_deviationSqAfterFlip;
        return __ret != 0;
    }

    /// given quadrangle ABCD, selects how to best triangulate it:
    ///   false = by introducing BD diagonal and splitting ABCD on triangles ABD and DBC,
    ///   true  = by introducing AC diagonal and splitting ABCD on triangles ABC and ACD
    /// Generated from function `MR::bestQuadrangleDiagonal`.
    public static unsafe bool BestQuadrangleDiagonal(MR.Const_Vector3f a, MR.Const_Vector3f b, MR.Const_Vector3f c, MR.Const_Vector3f d)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bestQuadrangleDiagonal", ExactSpelling = true)]
        extern static byte __MR_bestQuadrangleDiagonal(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, MR.Const_Vector3f._Underlying *c, MR.Const_Vector3f._Underlying *d);
        return __MR_bestQuadrangleDiagonal(a._UnderlyingPtr, b._UnderlyingPtr, c._UnderlyingPtr, d._UnderlyingPtr) != 0;
    }

    /// improves mesh triangulation in a ring of vertices with common origin and represented by edge e
    /// Generated from function `MR::makeDeloneOriginRing`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe void MakeDeloneOriginRing(MR.Mesh mesh, MR.EdgeId e, MR.Const_DeloneSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDeloneOriginRing_3", ExactSpelling = true)]
        extern static void __MR_makeDeloneOriginRing_3(MR.Mesh._Underlying *mesh, MR.EdgeId e, MR.Const_DeloneSettings._Underlying *settings);
        __MR_makeDeloneOriginRing_3(mesh._UnderlyingPtr, e, settings is not null ? settings._UnderlyingPtr : null);
    }

    /// Generated from function `MR::makeDeloneOriginRing`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe void MakeDeloneOriginRing(MR.MeshTopology topology, MR.Const_VertCoords points, MR.EdgeId e, MR.Const_DeloneSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDeloneOriginRing_4", ExactSpelling = true)]
        extern static void __MR_makeDeloneOriginRing_4(MR.MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.EdgeId e, MR.Const_DeloneSettings._Underlying *settings);
        __MR_makeDeloneOriginRing_4(topology._UnderlyingPtr, points._UnderlyingPtr, e, settings is not null ? settings._UnderlyingPtr : null);
    }

    /// improves mesh triangulation by performing flipping of edges to satisfy Delone local property,
    /// consider every edge at most numIters times, and allow surface deviation at most on given value during every individual flip,
    /// \return the number of flips done
    /// \param numIters Maximal iteration count
    /// \param progressCallback Callback to report algorithm progress and cancel it by user request
    /// Generated from function `MR::makeDeloneEdgeFlips`.
    /// Parameter `settings` defaults to `{}`.
    /// Parameter `numIters` defaults to `1`.
    /// Parameter `progressCallback` defaults to `{}`.
    public static unsafe int MakeDeloneEdgeFlips(MR.Mesh mesh, MR.Const_DeloneSettings? settings = null, int? numIters = null, MR.Std.Const_Function_BoolFuncFromFloat? progressCallback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDeloneEdgeFlips_4_MR_Mesh", ExactSpelling = true)]
        extern static int __MR_makeDeloneEdgeFlips_4_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_DeloneSettings._Underlying *settings, int *numIters, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCallback);
        int __deref_numIters = numIters.GetValueOrDefault();
        return __MR_makeDeloneEdgeFlips_4_MR_Mesh(mesh._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, numIters.HasValue ? &__deref_numIters : null, progressCallback is not null ? progressCallback._UnderlyingPtr : null);
    }

    /// Generated from function `MR::makeDeloneEdgeFlips`.
    /// Parameter `settings` defaults to `{}`.
    /// Parameter `numIters` defaults to `1`.
    /// Parameter `progressCallback` defaults to `{}`.
    public static unsafe int MakeDeloneEdgeFlips(MR.MeshTopology topology, MR.Const_VertCoords points, MR.Const_DeloneSettings? settings = null, int? numIters = null, MR.Std.Const_Function_BoolFuncFromFloat? progressCallback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDeloneEdgeFlips_5", ExactSpelling = true)]
        extern static int __MR_makeDeloneEdgeFlips_5(MR.MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.Const_DeloneSettings._Underlying *settings, int *numIters, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCallback);
        int __deref_numIters = numIters.GetValueOrDefault();
        return __MR_makeDeloneEdgeFlips_5(topology._UnderlyingPtr, points._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, numIters.HasValue ? &__deref_numIters : null, progressCallback is not null ? progressCallback._UnderlyingPtr : null);
    }

    /// improves mesh triangulation by performing flipping of edges to satisfy Intrinsic Delaunay local property,
    /// consider every edge at most numIters times,
    /// \return the number of flips done
    /// \param numIters Maximal iteration count
    /// \param progressCallback Callback to report algorithm progress and cancel it by user request
    /// see "An Algorithm for the Construction of Intrinsic Delaunay Triangulations with Applications to Digital Geometry Processing". https://page.math.tu-berlin.de/~bobenko/papers/InDel.pdf
    /// Generated from function `MR::makeDeloneEdgeFlips`.
    /// Parameter `settings` defaults to `{}`.
    /// Parameter `numIters` defaults to `1`.
    /// Parameter `progressCallback` defaults to `{}`.
    public static unsafe int MakeDeloneEdgeFlips(MR.EdgeLengthMesh mesh, MR.Const_IntrinsicDeloneSettings? settings = null, int? numIters = null, MR.Std.Const_Function_BoolFuncFromFloat? progressCallback = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDeloneEdgeFlips_4_MR_EdgeLengthMesh", ExactSpelling = true)]
        extern static int __MR_makeDeloneEdgeFlips_4_MR_EdgeLengthMesh(MR.EdgeLengthMesh._Underlying *mesh, MR.Const_IntrinsicDeloneSettings._Underlying *settings, int *numIters, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progressCallback);
        int __deref_numIters = numIters.GetValueOrDefault();
        return __MR_makeDeloneEdgeFlips_4_MR_EdgeLengthMesh(mesh._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null, numIters.HasValue ? &__deref_numIters : null, progressCallback is not null ? progressCallback._UnderlyingPtr : null);
    }
}
