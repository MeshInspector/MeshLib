public static partial class MR
{
    /// encodes a point on an edge of mesh or of polyline
    /// Generated from class `MR::EdgePoint`.
    /// This is the const half of the class.
    public class Const_EdgePoint : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_EdgePoint>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgePoint(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgePoint_Destroy(_Underlying *_this);
            __MR_EdgePoint_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgePoint() {Dispose(false);}

        public unsafe MR.Const_EdgeId E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Get_e", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_EdgePoint_Get_e(_Underlying *_this);
                return new(__MR_EdgePoint_Get_e(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )
        public unsafe MR.Const_SegmPointf A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Get_a", ExactSpelling = true)]
                extern static MR.Const_SegmPointf._Underlying *__MR_EdgePoint_Get_a(_Underlying *_this);
                return new(__MR_EdgePoint_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgePoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgePoint_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe Const_EdgePoint(MR.Const_EdgePoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_ConstructFromAnother(MR.EdgePoint._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePoint_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe Const_EdgePoint(MR.EdgeId e, float a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Construct_MR_EdgeId", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_Construct_MR_EdgeId(MR.EdgeId e, float a);
            _UnderlyingPtr = __MR_EdgePoint_Construct_MR_EdgeId(e, a);
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe Const_EdgePoint(MR.Const_MeshTopology topology, MR.VertId v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Construct_MR_MeshTopology", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_Construct_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology, MR.VertId v);
            _UnderlyingPtr = __MR_EdgePoint_Construct_MR_MeshTopology(topology._UnderlyingPtr, v);
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe Const_EdgePoint(MR.Const_PolylineTopology topology, MR.VertId v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Construct_MR_PolylineTopology", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_Construct_MR_PolylineTopology(MR.Const_PolylineTopology._Underlying *topology, MR.VertId v);
            _UnderlyingPtr = __MR_EdgePoint_Construct_MR_PolylineTopology(topology._UnderlyingPtr, v);
        }

        /// Generated from conversion operator `MR::EdgePoint::operator bool`.
        public static unsafe explicit operator bool(MR.Const_EdgePoint _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_EdgePoint_ConvertTo_bool(MR.Const_EdgePoint._Underlying *_this);
            return __MR_EdgePoint_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
        /// Generated from method `MR::EdgePoint::inVertex`.
        public unsafe MR.VertId InVertex(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_inVertex_1_MR_MeshTopology", ExactSpelling = true)]
            extern static MR.VertId __MR_EdgePoint_inVertex_1_MR_MeshTopology(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return __MR_EdgePoint_inVertex_1_MR_MeshTopology(_UnderlyingPtr, topology._UnderlyingPtr);
        }

        /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
        /// Generated from method `MR::EdgePoint::inVertex`.
        public unsafe MR.VertId InVertex(MR.Const_PolylineTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_inVertex_1_MR_PolylineTopology", ExactSpelling = true)]
            extern static MR.VertId __MR_EdgePoint_inVertex_1_MR_PolylineTopology(_Underlying *_this, MR.Const_PolylineTopology._Underlying *topology);
            return __MR_EdgePoint_inVertex_1_MR_PolylineTopology(_UnderlyingPtr, topology._UnderlyingPtr);
        }

        /// returns one of two edge vertices, closest to this point
        /// Generated from method `MR::EdgePoint::getClosestVertex`.
        public unsafe MR.VertId GetClosestVertex(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_getClosestVertex_MR_MeshTopology", ExactSpelling = true)]
            extern static MR.VertId __MR_EdgePoint_getClosestVertex_MR_MeshTopology(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return __MR_EdgePoint_getClosestVertex_MR_MeshTopology(_UnderlyingPtr, topology._UnderlyingPtr);
        }

        /// returns one of two edge vertices, closest to this point
        /// Generated from method `MR::EdgePoint::getClosestVertex`.
        public unsafe MR.VertId GetClosestVertex(MR.Const_PolylineTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_getClosestVertex_MR_PolylineTopology", ExactSpelling = true)]
            extern static MR.VertId __MR_EdgePoint_getClosestVertex_MR_PolylineTopology(_Underlying *_this, MR.Const_PolylineTopology._Underlying *topology);
            return __MR_EdgePoint_getClosestVertex_MR_PolylineTopology(_UnderlyingPtr, topology._UnderlyingPtr);
        }

        /// returns true if the point is in a vertex
        /// Generated from method `MR::EdgePoint::inVertex`.
        public unsafe bool InVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_inVertex_0", ExactSpelling = true)]
            extern static byte __MR_EdgePoint_inVertex_0(_Underlying *_this);
            return __MR_EdgePoint_inVertex_0(_UnderlyingPtr) != 0;
        }

        /// returns true if the point is on the boundary of the region (or for whole mesh if region is nullptr)
        /// Generated from method `MR::EdgePoint::isBd`.
        public unsafe bool IsBd(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_isBd", ExactSpelling = true)]
            extern static byte __MR_EdgePoint_isBd(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
            return __MR_EdgePoint_isBd(_UnderlyingPtr, topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// consider this valid if the edge ID is valid
        /// Generated from method `MR::EdgePoint::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_valid", ExactSpelling = true)]
            extern static byte __MR_EdgePoint_valid(_Underlying *_this);
            return __MR_EdgePoint_valid(_UnderlyingPtr) != 0;
        }

        /// represents the same point relative to sym edge in
        /// Generated from method `MR::EdgePoint::sym`.
        public unsafe MR.EdgePoint Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_sym", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_sym(_Underlying *_this);
            return new(__MR_EdgePoint_sym(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if two edge-points are equal including equal not-unique representation
        /// Generated from method `MR::EdgePoint::operator==`.
        public static unsafe bool operator==(MR.Const_EdgePoint _this, MR.Const_EdgePoint rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_EdgePoint", ExactSpelling = true)]
            extern static byte __MR_equal_MR_EdgePoint(MR.Const_EdgePoint._Underlying *_this, MR.Const_EdgePoint._Underlying *rhs);
            return __MR_equal_MR_EdgePoint(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_EdgePoint _this, MR.Const_EdgePoint rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_EdgePoint? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_EdgePoint)
                return this == (MR.Const_EdgePoint)other;
            return false;
        }
    }

    /// encodes a point on an edge of mesh or of polyline
    /// Generated from class `MR::EdgePoint`.
    /// This is the non-const half of the class.
    public class EdgePoint : Const_EdgePoint
    {
        internal unsafe EdgePoint(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_EdgeId E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_GetMutable_e", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_EdgePoint_GetMutable_e(_Underlying *_this);
                return new(__MR_EdgePoint_GetMutable_e(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< a in [0,1], a=0 => point is in org( e ), a=1 => point is in dest( e )
        public new unsafe MR.SegmPointf A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_GetMutable_a", ExactSpelling = true)]
                extern static MR.SegmPointf._Underlying *__MR_EdgePoint_GetMutable_a(_Underlying *_this);
                return new(__MR_EdgePoint_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgePoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgePoint_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe EdgePoint(MR.Const_EdgePoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_ConstructFromAnother(MR.EdgePoint._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePoint_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe EdgePoint(MR.EdgeId e, float a) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Construct_MR_EdgeId", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_Construct_MR_EdgeId(MR.EdgeId e, float a);
            _UnderlyingPtr = __MR_EdgePoint_Construct_MR_EdgeId(e, a);
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe EdgePoint(MR.Const_MeshTopology topology, MR.VertId v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Construct_MR_MeshTopology", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_Construct_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology, MR.VertId v);
            _UnderlyingPtr = __MR_EdgePoint_Construct_MR_MeshTopology(topology._UnderlyingPtr, v);
        }

        /// Generated from constructor `MR::EdgePoint::EdgePoint`.
        public unsafe EdgePoint(MR.Const_PolylineTopology topology, MR.VertId v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_Construct_MR_PolylineTopology", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_Construct_MR_PolylineTopology(MR.Const_PolylineTopology._Underlying *topology, MR.VertId v);
            _UnderlyingPtr = __MR_EdgePoint_Construct_MR_PolylineTopology(topology._UnderlyingPtr, v);
        }

        /// Generated from method `MR::EdgePoint::operator=`.
        public unsafe MR.EdgePoint Assign(MR.Const_EdgePoint _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgePoint_AssignFromAnother(_Underlying *_this, MR.EdgePoint._Underlying *_other);
            return new(__MR_EdgePoint_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// sets this to the closest end of the edge
        /// Generated from method `MR::EdgePoint::moveToClosestVertex`.
        public unsafe void MoveToClosestVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePoint_moveToClosestVertex", ExactSpelling = true)]
            extern static void __MR_EdgePoint_moveToClosestVertex(_Underlying *_this);
            __MR_EdgePoint_moveToClosestVertex(_UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `EdgePoint` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgePoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePoint`/`Const_EdgePoint` directly.
    public class _InOptMut_EdgePoint
    {
        public EdgePoint? Opt;

        public _InOptMut_EdgePoint() {}
        public _InOptMut_EdgePoint(EdgePoint value) {Opt = value;}
        public static implicit operator _InOptMut_EdgePoint(EdgePoint value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgePoint` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgePoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePoint`/`Const_EdgePoint` to pass it to the function.
    public class _InOptConst_EdgePoint
    {
        public Const_EdgePoint? Opt;

        public _InOptConst_EdgePoint() {}
        public _InOptConst_EdgePoint(Const_EdgePoint value) {Opt = value;}
        public static implicit operator _InOptConst_EdgePoint(Const_EdgePoint value) {return new(value);}
    }

    /// two edge-points (e.g. representing collision point of two edges)
    /// Generated from class `MR::EdgePointPair`.
    /// This is the const half of the class.
    public class Const_EdgePointPair : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_EdgePointPair>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgePointPair(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgePointPair_Destroy(_Underlying *_this);
            __MR_EdgePointPair_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgePointPair() {Dispose(false);}

        public unsafe MR.Const_EdgePoint A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_Get_a", ExactSpelling = true)]
                extern static MR.Const_EdgePoint._Underlying *__MR_EdgePointPair_Get_a(_Underlying *_this);
                return new(__MR_EdgePointPair_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_EdgePoint B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_Get_b", ExactSpelling = true)]
                extern static MR.Const_EdgePoint._Underlying *__MR_EdgePointPair_Get_b(_Underlying *_this);
                return new(__MR_EdgePointPair_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgePointPair() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgePointPair_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgePointPair::EdgePointPair`.
        public unsafe Const_EdgePointPair(MR.Const_EdgePointPair _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_ConstructFromAnother(MR.EdgePointPair._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePointPair_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgePointPair::EdgePointPair`.
        public unsafe Const_EdgePointPair(MR.Const_EdgePoint ia, MR.Const_EdgePoint ib) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_Construct", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_Construct(MR.EdgePoint._Underlying *ia, MR.EdgePoint._Underlying *ib);
            _UnderlyingPtr = __MR_EdgePointPair_Construct(ia._UnderlyingPtr, ib._UnderlyingPtr);
        }

        /// returns true if two edge-point pairs are equal including equal not-unique representation
        /// Generated from method `MR::EdgePointPair::operator==`.
        public static unsafe bool operator==(MR.Const_EdgePointPair _this, MR.Const_EdgePointPair rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_EdgePointPair", ExactSpelling = true)]
            extern static byte __MR_equal_MR_EdgePointPair(MR.Const_EdgePointPair._Underlying *_this, MR.Const_EdgePointPair._Underlying *rhs);
            return __MR_equal_MR_EdgePointPair(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_EdgePointPair _this, MR.Const_EdgePointPair rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_EdgePointPair? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_EdgePointPair)
                return this == (MR.Const_EdgePointPair)other;
            return false;
        }
    }

    /// two edge-points (e.g. representing collision point of two edges)
    /// Generated from class `MR::EdgePointPair`.
    /// This is the non-const half of the class.
    public class EdgePointPair : Const_EdgePointPair
    {
        internal unsafe EdgePointPair(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.EdgePoint A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_GetMutable_a", ExactSpelling = true)]
                extern static MR.EdgePoint._Underlying *__MR_EdgePointPair_GetMutable_a(_Underlying *_this);
                return new(__MR_EdgePointPair_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.EdgePoint B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_GetMutable_b", ExactSpelling = true)]
                extern static MR.EdgePoint._Underlying *__MR_EdgePointPair_GetMutable_b(_Underlying *_this);
                return new(__MR_EdgePointPair_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgePointPair() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgePointPair_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgePointPair::EdgePointPair`.
        public unsafe EdgePointPair(MR.Const_EdgePointPair _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_ConstructFromAnother(MR.EdgePointPair._Underlying *_other);
            _UnderlyingPtr = __MR_EdgePointPair_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgePointPair::EdgePointPair`.
        public unsafe EdgePointPair(MR.Const_EdgePoint ia, MR.Const_EdgePoint ib) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_Construct", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_Construct(MR.EdgePoint._Underlying *ia, MR.EdgePoint._Underlying *ib);
            _UnderlyingPtr = __MR_EdgePointPair_Construct(ia._UnderlyingPtr, ib._UnderlyingPtr);
        }

        /// Generated from method `MR::EdgePointPair::operator=`.
        public unsafe MR.EdgePointPair Assign(MR.Const_EdgePointPair _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgePointPair_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgePointPair._Underlying *__MR_EdgePointPair_AssignFromAnother(_Underlying *_this, MR.EdgePointPair._Underlying *_other);
            return new(__MR_EdgePointPair_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `EdgePointPair` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgePointPair`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePointPair`/`Const_EdgePointPair` directly.
    public class _InOptMut_EdgePointPair
    {
        public EdgePointPair? Opt;

        public _InOptMut_EdgePointPair() {}
        public _InOptMut_EdgePointPair(EdgePointPair value) {Opt = value;}
        public static implicit operator _InOptMut_EdgePointPair(EdgePointPair value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgePointPair` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgePointPair`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgePointPair`/`Const_EdgePointPair` to pass it to the function.
    public class _InOptConst_EdgePointPair
    {
        public Const_EdgePointPair? Opt;

        public _InOptConst_EdgePointPair() {}
        public _InOptConst_EdgePointPair(Const_EdgePointPair value) {Opt = value;}
        public static implicit operator _InOptConst_EdgePointPair(Const_EdgePointPair value) {return new(value);}
    }

    /// Represents a segment on one edge
    /// Generated from class `MR::EdgeSegment`.
    /// This is the const half of the class.
    public class Const_EdgeSegment : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_EdgeSegment>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EdgeSegment(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_Destroy", ExactSpelling = true)]
            extern static void __MR_EdgeSegment_Destroy(_Underlying *_this);
            __MR_EdgeSegment_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EdgeSegment() {Dispose(false);}

        /// id of the edge
        public unsafe MR.Const_EdgeId E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_Get_e", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_EdgeSegment_Get_e(_Underlying *_this);
                return new(__MR_EdgeSegment_Get_e(_UnderlyingPtr), is_owning: false);
            }
        }

        /// start of the segment
        public unsafe MR.Const_SegmPointf A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_Get_a", ExactSpelling = true)]
                extern static MR.Const_SegmPointf._Underlying *__MR_EdgeSegment_Get_a(_Underlying *_this);
                return new(__MR_EdgeSegment_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// end of the segment
        public unsafe MR.Const_SegmPointf B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_Get_b", ExactSpelling = true)]
                extern static MR.Const_SegmPointf._Underlying *__MR_EdgeSegment_Get_b(_Underlying *_this);
                return new(__MR_EdgeSegment_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EdgeSegment() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeSegment_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeSegment::EdgeSegment`.
        public unsafe Const_EdgeSegment(MR.Const_EdgeSegment _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_ConstructFromAnother(MR.EdgeSegment._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeSegment_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgeSegment::EdgeSegment`.
        /// Parameter `a` defaults to `0.0f`.
        /// Parameter `b` defaults to `1.0f`.
        public unsafe Const_EdgeSegment(MR.EdgeId e, float? a = null, float? b = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_Construct", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_Construct(MR.EdgeId e, float *a, float *b);
            float __deref_a = a.GetValueOrDefault();
            float __deref_b = b.GetValueOrDefault();
            _UnderlyingPtr = __MR_EdgeSegment_Construct(e, a.HasValue ? &__deref_a : null, b.HasValue ? &__deref_b : null);
        }

        /// returns starting EdgePoint
        /// Generated from method `MR::EdgeSegment::edgePointA`.
        public unsafe MR.EdgePoint EdgePointA()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_edgePointA", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgeSegment_edgePointA(_Underlying *_this);
            return new(__MR_EdgeSegment_edgePointA(_UnderlyingPtr), is_owning: true);
        }

        /// returns ending EdgePoint
        /// Generated from method `MR::EdgeSegment::edgePointB`.
        public unsafe MR.EdgePoint EdgePointB()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_edgePointB", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_EdgeSegment_edgePointB(_Underlying *_this);
            return new(__MR_EdgeSegment_edgePointB(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if the edge is valid and start point is less than end point
        /// Generated from method `MR::EdgeSegment::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_valid", ExactSpelling = true)]
            extern static byte __MR_EdgeSegment_valid(_Underlying *_this);
            return __MR_EdgeSegment_valid(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::EdgeSegment::operator==`.
        public static unsafe bool operator==(MR.Const_EdgeSegment _this, MR.Const_EdgeSegment rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_EdgeSegment", ExactSpelling = true)]
            extern static byte __MR_equal_MR_EdgeSegment(MR.Const_EdgeSegment._Underlying *_this, MR.Const_EdgeSegment._Underlying *rhs);
            return __MR_equal_MR_EdgeSegment(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_EdgeSegment _this, MR.Const_EdgeSegment rhs)
        {
            return !(_this == rhs);
        }

        /// represents the same segment relative to sym edge in
        /// Generated from method `MR::EdgeSegment::sym`.
        public unsafe MR.EdgeSegment Sym()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_sym", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_sym(_Underlying *_this);
            return new(__MR_EdgeSegment_sym(_UnderlyingPtr), is_owning: true);
        }

        // IEquatable:

        public bool Equals(MR.Const_EdgeSegment? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_EdgeSegment)
                return this == (MR.Const_EdgeSegment)other;
            return false;
        }
    }

    /// Represents a segment on one edge
    /// Generated from class `MR::EdgeSegment`.
    /// This is the non-const half of the class.
    public class EdgeSegment : Const_EdgeSegment
    {
        internal unsafe EdgeSegment(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// id of the edge
        public new unsafe MR.Mut_EdgeId E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_GetMutable_e", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_EdgeSegment_GetMutable_e(_Underlying *_this);
                return new(__MR_EdgeSegment_GetMutable_e(_UnderlyingPtr), is_owning: false);
            }
        }

        /// start of the segment
        public new unsafe MR.SegmPointf A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_GetMutable_a", ExactSpelling = true)]
                extern static MR.SegmPointf._Underlying *__MR_EdgeSegment_GetMutable_a(_Underlying *_this);
                return new(__MR_EdgeSegment_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        /// end of the segment
        public new unsafe MR.SegmPointf B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_GetMutable_b", ExactSpelling = true)]
                extern static MR.SegmPointf._Underlying *__MR_EdgeSegment_GetMutable_b(_Underlying *_this);
                return new(__MR_EdgeSegment_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EdgeSegment() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_DefaultConstruct();
            _UnderlyingPtr = __MR_EdgeSegment_DefaultConstruct();
        }

        /// Generated from constructor `MR::EdgeSegment::EdgeSegment`.
        public unsafe EdgeSegment(MR.Const_EdgeSegment _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_ConstructFromAnother(MR.EdgeSegment._Underlying *_other);
            _UnderlyingPtr = __MR_EdgeSegment_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EdgeSegment::EdgeSegment`.
        /// Parameter `a` defaults to `0.0f`.
        /// Parameter `b` defaults to `1.0f`.
        public unsafe EdgeSegment(MR.EdgeId e, float? a = null, float? b = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_Construct", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_Construct(MR.EdgeId e, float *a, float *b);
            float __deref_a = a.GetValueOrDefault();
            float __deref_b = b.GetValueOrDefault();
            _UnderlyingPtr = __MR_EdgeSegment_Construct(e, a.HasValue ? &__deref_a : null, b.HasValue ? &__deref_b : null);
        }

        /// Generated from method `MR::EdgeSegment::operator=`.
        public unsafe MR.EdgeSegment Assign(MR.Const_EdgeSegment _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EdgeSegment_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EdgeSegment._Underlying *__MR_EdgeSegment_AssignFromAnother(_Underlying *_this, MR.EdgeSegment._Underlying *_other);
            return new(__MR_EdgeSegment_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `EdgeSegment` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EdgeSegment`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeSegment`/`Const_EdgeSegment` directly.
    public class _InOptMut_EdgeSegment
    {
        public EdgeSegment? Opt;

        public _InOptMut_EdgeSegment() {}
        public _InOptMut_EdgeSegment(EdgeSegment value) {Opt = value;}
        public static implicit operator _InOptMut_EdgeSegment(EdgeSegment value) {return new(value);}
    }

    /// This is used for optional parameters of class `EdgeSegment` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EdgeSegment`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EdgeSegment`/`Const_EdgeSegment` to pass it to the function.
    public class _InOptConst_EdgeSegment
    {
        public Const_EdgeSegment? Opt;

        public _InOptConst_EdgeSegment() {}
        public _InOptConst_EdgeSegment(Const_EdgeSegment value) {Opt = value;}
        public static implicit operator _InOptConst_EdgeSegment(Const_EdgeSegment value) {return new(value);}
    }

    /// returns true if two edge-points are equal considering different representations
    /// Generated from function `MR::same`.
    public static unsafe bool Same(MR.Const_MeshTopology topology, MR.Const_EdgePoint lhs, MR.Const_EdgePoint rhs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_same_MR_EdgePoint", ExactSpelling = true)]
        extern static byte __MR_same_MR_EdgePoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_EdgePoint._Underlying *lhs, MR.Const_EdgePoint._Underlying *rhs);
        return __MR_same_MR_EdgePoint(topology._UnderlyingPtr, lhs._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
    }

    /// returns true if points a and b are located on a boundary of the same triangle;
    /// \details if true a.e and b.e are updated to have that triangle on the left
    /// \related EdgePoint
    /// Generated from function `MR::fromSameTriangle`.
    public static unsafe bool FromSameTriangle(MR.Const_MeshTopology topology, MR.EdgePoint a, MR.EdgePoint b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fromSameTriangle_MR_EdgePoint_ref", ExactSpelling = true)]
        extern static byte __MR_fromSameTriangle_MR_EdgePoint_ref(MR.Const_MeshTopology._Underlying *topology, MR.EdgePoint._Underlying *a, MR.EdgePoint._Underlying *b);
        return __MR_fromSameTriangle_MR_EdgePoint_ref(topology._UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr) != 0;
    }

    /// returns true if points a and b are located on a boundary of the same triangle;
    /// \details if true a.e and b.e are updated to have that triangle on the left
    /// \related EdgePoint
    /// Generated from function `MR::fromSameTriangle`.
    public static unsafe bool FromSameTriangle(MR.Const_MeshTopology topology, MR.Misc._Moved<MR.EdgePoint> a, MR.Misc._Moved<MR.EdgePoint> b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fromSameTriangle_MR_EdgePoint_rvalue_ref", ExactSpelling = true)]
        extern static byte __MR_fromSameTriangle_MR_EdgePoint_rvalue_ref(MR.Const_MeshTopology._Underlying *topology, MR.EdgePoint._Underlying *a, MR.EdgePoint._Underlying *b);
        return __MR_fromSameTriangle_MR_EdgePoint_rvalue_ref(topology._UnderlyingPtr, a.Value._UnderlyingPtr, b.Value._UnderlyingPtr) != 0;
    }
}
