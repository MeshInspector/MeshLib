public static partial class MR
{
    /// Generated from class `MR::WeightedVertex`.
    /// This is the const half of the class.
    public class Const_WeightedVertex : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_WeightedVertex(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_Destroy", ExactSpelling = true)]
            extern static void __MR_WeightedVertex_Destroy(_Underlying *_this);
            __MR_WeightedVertex_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_WeightedVertex() {Dispose(false);}

        public unsafe MR.Const_VertId V
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_Get_v", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_WeightedVertex_Get_v(_Underlying *_this);
                return new(__MR_WeightedVertex_Get_v(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_Get_weight", ExactSpelling = true)]
                extern static float *__MR_WeightedVertex_Get_weight(_Underlying *_this);
                return *__MR_WeightedVertex_Get_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_WeightedVertex() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_DefaultConstruct", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_DefaultConstruct();
            _UnderlyingPtr = __MR_WeightedVertex_DefaultConstruct();
        }

        /// Constructs `MR::WeightedVertex` elementwise.
        public unsafe Const_WeightedVertex(MR.VertId v, float weight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_ConstructFrom", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_ConstructFrom(MR.VertId v, float weight);
            _UnderlyingPtr = __MR_WeightedVertex_ConstructFrom(v, weight);
        }

        /// Generated from constructor `MR::WeightedVertex::WeightedVertex`.
        public unsafe Const_WeightedVertex(MR.Const_WeightedVertex _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_ConstructFromAnother(MR.WeightedVertex._Underlying *_other);
            _UnderlyingPtr = __MR_WeightedVertex_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::WeightedVertex`.
    /// This is the non-const half of the class.
    public class WeightedVertex : Const_WeightedVertex
    {
        internal unsafe WeightedVertex(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_VertId V
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_GetMutable_v", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_WeightedVertex_GetMutable_v(_Underlying *_this);
                return new(__MR_WeightedVertex_GetMutable_v(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Weight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_GetMutable_weight", ExactSpelling = true)]
                extern static float *__MR_WeightedVertex_GetMutable_weight(_Underlying *_this);
                return ref *__MR_WeightedVertex_GetMutable_weight(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe WeightedVertex() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_DefaultConstruct", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_DefaultConstruct();
            _UnderlyingPtr = __MR_WeightedVertex_DefaultConstruct();
        }

        /// Constructs `MR::WeightedVertex` elementwise.
        public unsafe WeightedVertex(MR.VertId v, float weight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_ConstructFrom", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_ConstructFrom(MR.VertId v, float weight);
            _UnderlyingPtr = __MR_WeightedVertex_ConstructFrom(v, weight);
        }

        /// Generated from constructor `MR::WeightedVertex::WeightedVertex`.
        public unsafe WeightedVertex(MR.Const_WeightedVertex _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_ConstructFromAnother(MR.WeightedVertex._Underlying *_other);
            _UnderlyingPtr = __MR_WeightedVertex_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::WeightedVertex::operator=`.
        public unsafe MR.WeightedVertex Assign(MR.Const_WeightedVertex _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_WeightedVertex_AssignFromAnother", ExactSpelling = true)]
            extern static MR.WeightedVertex._Underlying *__MR_WeightedVertex_AssignFromAnother(_Underlying *_this, MR.WeightedVertex._Underlying *_other);
            return new(__MR_WeightedVertex_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `WeightedVertex` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_WeightedVertex`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WeightedVertex`/`Const_WeightedVertex` directly.
    public class _InOptMut_WeightedVertex
    {
        public WeightedVertex? Opt;

        public _InOptMut_WeightedVertex() {}
        public _InOptMut_WeightedVertex(WeightedVertex value) {Opt = value;}
        public static implicit operator _InOptMut_WeightedVertex(WeightedVertex value) {return new(value);}
    }

    /// This is used for optional parameters of class `WeightedVertex` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_WeightedVertex`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `WeightedVertex`/`Const_WeightedVertex` to pass it to the function.
    public class _InOptConst_WeightedVertex
    {
        public Const_WeightedVertex? Opt;

        public _InOptConst_WeightedVertex() {}
        public _InOptConst_WeightedVertex(Const_WeightedVertex value) {Opt = value;}
        public static implicit operator _InOptConst_WeightedVertex(Const_WeightedVertex value) {return new(value);}
    }

    /// encodes a point inside a triangular mesh face using barycentric coordinates
    /// \details Notations used below: \n
    ///   v0 - the value in org( e ) \n
    ///   v1 - the value in dest( e ) \n
    ///   v2 - the value in dest( next( e ) )
    /// Generated from class `MR::MeshTriPoint`.
    /// This is the const half of the class.
    public class Const_MeshTriPoint : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_MeshTriPoint>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshTriPoint(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshTriPoint_Destroy(_Underlying *_this);
            __MR_MeshTriPoint_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshTriPoint() {Dispose(false);}

        ///< left face of this edge is considered
        public unsafe MR.Const_EdgeId E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Get_e", ExactSpelling = true)]
                extern static MR.Const_EdgeId._Underlying *__MR_MeshTriPoint_Get_e(_Underlying *_this);
                return new(__MR_MeshTriPoint_Get_e(_UnderlyingPtr), is_owning: false);
            }
        }

        /// barycentric coordinates
        /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
        /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
        /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
        public unsafe MR.Const_TriPointf Bary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Get_bary", ExactSpelling = true)]
                extern static MR.Const_TriPointf._Underlying *__MR_MeshTriPoint_Get_bary(_Underlying *_this);
                return new(__MR_MeshTriPoint_Get_bary(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshTriPoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTriPoint_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe Const_MeshTriPoint(MR.Const_MeshTriPoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_ConstructFromAnother(MR.MeshTriPoint._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTriPoint_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe Const_MeshTriPoint(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_1_MR_NoInit", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_1_MR_NoInit(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_1_MR_NoInit(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public static unsafe implicit operator Const_MeshTriPoint(MR.Const_NoInit _1) {return new(_1);}

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe Const_MeshTriPoint(MR.EdgeId e, MR.Const_TriPointf bary) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_2_MR_EdgeId", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_2_MR_EdgeId(MR.EdgeId e, MR.TriPointf._Underlying *bary);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_2_MR_EdgeId(e, bary._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe Const_MeshTriPoint(MR.Const_EdgePoint ep) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_1_MR_EdgePoint", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_1_MR_EdgePoint(MR.Const_EdgePoint._Underlying *ep);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_1_MR_EdgePoint(ep._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public static unsafe implicit operator Const_MeshTriPoint(MR.Const_EdgePoint ep) {return new(ep);}

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe Const_MeshTriPoint(MR.Const_MeshTopology topology, MR.VertId v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_2_MR_MeshTopology", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_2_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology, MR.VertId v);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_2_MR_MeshTopology(topology._UnderlyingPtr, v);
        }

        /// Generated from conversion operator `MR::MeshTriPoint::operator bool`.
        public static unsafe explicit operator bool(MR.Const_MeshTriPoint _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_MeshTriPoint_ConvertTo_bool(MR.Const_MeshTriPoint._Underlying *_this);
            return __MR_MeshTriPoint_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// returns valid vertex id if the point is in vertex, otherwise returns invalid id
        /// Generated from method `MR::MeshTriPoint::inVertex`.
        public unsafe MR.VertId InVertex(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_inVertex_1", ExactSpelling = true)]
            extern static MR.VertId __MR_MeshTriPoint_inVertex_1(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return __MR_MeshTriPoint_inVertex_1(_UnderlyingPtr, topology._UnderlyingPtr);
        }

        /// returns true if the point is in a vertex
        /// Generated from method `MR::MeshTriPoint::inVertex`.
        public unsafe bool InVertex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_inVertex_0", ExactSpelling = true)]
            extern static byte __MR_MeshTriPoint_inVertex_0(_Underlying *_this);
            return __MR_MeshTriPoint_inVertex_0(_UnderlyingPtr) != 0;
        }

        /// returns valid value if the point is on edge and topology.left(result.e) == topology.left(this->e),
        /// otherwise returns invalid MeshEdgePoint
        /// Generated from method `MR::MeshTriPoint::onEdge`.
        public unsafe MR.EdgePoint OnEdge(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_onEdge", ExactSpelling = true)]
            extern static MR.EdgePoint._Underlying *__MR_MeshTriPoint_onEdge(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return new(__MR_MeshTriPoint_onEdge(_UnderlyingPtr, topology._UnderlyingPtr), is_owning: true);
        }

        /// returns true if the point is in vertex or on edge, and that location is on the boundary of the region
        /// Generated from method `MR::MeshTriPoint::isBd`.
        public unsafe bool IsBd(MR.Const_MeshTopology topology, MR.Const_FaceBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_isBd", ExactSpelling = true)]
            extern static byte __MR_MeshTriPoint_isBd(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, MR.Const_FaceBitSet._Underlying *region);
            return __MR_MeshTriPoint_isBd(_UnderlyingPtr, topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null) != 0;
        }

        /// returns true if the point is inside or on the boundary of given triangular face
        /// Generated from method `MR::MeshTriPoint::fromTriangle`.
        public unsafe bool FromTriangle(MR.Const_MeshTopology topology, MR.FaceId f)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_fromTriangle", ExactSpelling = true)]
            extern static byte __MR_MeshTriPoint_fromTriangle(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, MR.FaceId f);
            return __MR_MeshTriPoint_fromTriangle(_UnderlyingPtr, topology._UnderlyingPtr, f) != 0;
        }

        /// consider this valid if the edge ID is valid
        /// Generated from method `MR::MeshTriPoint::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_valid", ExactSpelling = true)]
            extern static byte __MR_MeshTriPoint_valid(_Underlying *_this);
            return __MR_MeshTriPoint_valid(_UnderlyingPtr) != 0;
        }

        /// represents the same point relative to next edge in the same triangle
        /// Generated from method `MR::MeshTriPoint::lnext`.
        public unsafe MR.MeshTriPoint Lnext(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_lnext", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_lnext(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return new(__MR_MeshTriPoint_lnext(_UnderlyingPtr, topology._UnderlyingPtr), is_owning: true);
        }

        /// represents the same point relative to the topology.edgeWithLeft( topology.left( e ) )
        /// Generated from method `MR::MeshTriPoint::canonical`.
        public unsafe MR.MeshTriPoint Canonical(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_canonical", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_canonical(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return new(__MR_MeshTriPoint_canonical(_UnderlyingPtr, topology._UnderlyingPtr), is_owning: true);
        }

        /// returns three weighted triangle's vertices with the sum of not-negative weights equal to 1, and the largest weight in the closest vertex
        /// Generated from method `MR::MeshTriPoint::getWeightedVerts`.
        public unsafe MR.Std.Array_MRWeightedVertex_3 GetWeightedVerts(MR.Const_MeshTopology topology)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_getWeightedVerts", ExactSpelling = true)]
            extern static MR.Std.Array_MRWeightedVertex_3._Underlying *__MR_MeshTriPoint_getWeightedVerts(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology);
            return new(__MR_MeshTriPoint_getWeightedVerts(_UnderlyingPtr, topology._UnderlyingPtr), is_owning: true);
        }

        /// returns true if two points are equal including equal not-unique representation
        /// Generated from method `MR::MeshTriPoint::operator==`.
        public static unsafe bool operator==(MR.Const_MeshTriPoint _this, MR.Const_MeshTriPoint rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_MeshTriPoint", ExactSpelling = true)]
            extern static byte __MR_equal_MR_MeshTriPoint(MR.Const_MeshTriPoint._Underlying *_this, MR.Const_MeshTriPoint._Underlying *rhs);
            return __MR_equal_MR_MeshTriPoint(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_MeshTriPoint _this, MR.Const_MeshTriPoint rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_MeshTriPoint? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_MeshTriPoint)
                return this == (MR.Const_MeshTriPoint)other;
            return false;
        }
    }

    /// encodes a point inside a triangular mesh face using barycentric coordinates
    /// \details Notations used below: \n
    ///   v0 - the value in org( e ) \n
    ///   v1 - the value in dest( e ) \n
    ///   v2 - the value in dest( next( e ) )
    /// Generated from class `MR::MeshTriPoint`.
    /// This is the non-const half of the class.
    public class MeshTriPoint : Const_MeshTriPoint
    {
        internal unsafe MeshTriPoint(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< left face of this edge is considered
        public new unsafe MR.Mut_EdgeId E
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_GetMutable_e", ExactSpelling = true)]
                extern static MR.Mut_EdgeId._Underlying *__MR_MeshTriPoint_GetMutable_e(_Underlying *_this);
                return new(__MR_MeshTriPoint_GetMutable_e(_UnderlyingPtr), is_owning: false);
            }
        }

        /// barycentric coordinates
        /// \details a in [0,1], a=0 => point is on next( e ) edge, a=1 => point is in dest( e )
        /// b in [0,1], b=0 => point is on e edge, b=1 => point is in dest( next( e ) )
        /// a+b in [0,1], a+b=0 => point is in org( e ), a+b=1 => point is on prev( e.sym() ) edge
        public new unsafe MR.TriPointf Bary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_GetMutable_bary", ExactSpelling = true)]
                extern static MR.TriPointf._Underlying *__MR_MeshTriPoint_GetMutable_bary(_Underlying *_this);
                return new(__MR_MeshTriPoint_GetMutable_bary(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshTriPoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTriPoint_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe MeshTriPoint(MR.Const_MeshTriPoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_ConstructFromAnother(MR.MeshTriPoint._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTriPoint_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe MeshTriPoint(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_1_MR_NoInit", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_1_MR_NoInit(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_1_MR_NoInit(_1._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public static unsafe implicit operator MeshTriPoint(MR.Const_NoInit _1) {return new(_1);}

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe MeshTriPoint(MR.EdgeId e, MR.Const_TriPointf bary) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_2_MR_EdgeId", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_2_MR_EdgeId(MR.EdgeId e, MR.TriPointf._Underlying *bary);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_2_MR_EdgeId(e, bary._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe MeshTriPoint(MR.Const_EdgePoint ep) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_1_MR_EdgePoint", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_1_MR_EdgePoint(MR.Const_EdgePoint._Underlying *ep);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_1_MR_EdgePoint(ep._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public static unsafe implicit operator MeshTriPoint(MR.Const_EdgePoint ep) {return new(ep);}

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public unsafe MeshTriPoint(MR.Const_MeshTopology topology, MR.VertId v) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_Construct_2_MR_MeshTopology", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_Construct_2_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology, MR.VertId v);
            _UnderlyingPtr = __MR_MeshTriPoint_Construct_2_MR_MeshTopology(topology._UnderlyingPtr, v);
        }

        /// Generated from method `MR::MeshTriPoint::operator=`.
        public unsafe MR.MeshTriPoint Assign(MR.Const_MeshTriPoint _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTriPoint_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshTriPoint._Underlying *__MR_MeshTriPoint_AssignFromAnother(_Underlying *_this, MR.MeshTriPoint._Underlying *_other);
            return new(__MR_MeshTriPoint_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshTriPoint` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshTriPoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTriPoint`/`Const_MeshTriPoint` directly.
    public class _InOptMut_MeshTriPoint
    {
        public MeshTriPoint? Opt;

        public _InOptMut_MeshTriPoint() {}
        public _InOptMut_MeshTriPoint(MeshTriPoint value) {Opt = value;}
        public static implicit operator _InOptMut_MeshTriPoint(MeshTriPoint value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshTriPoint` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshTriPoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTriPoint`/`Const_MeshTriPoint` to pass it to the function.
    public class _InOptConst_MeshTriPoint
    {
        public Const_MeshTriPoint? Opt;

        public _InOptConst_MeshTriPoint() {}
        public _InOptConst_MeshTriPoint(Const_MeshTriPoint value) {Opt = value;}
        public static implicit operator _InOptConst_MeshTriPoint(Const_MeshTriPoint value) {return new(value);}

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public static unsafe implicit operator _InOptConst_MeshTriPoint(MR.Const_NoInit _1) {return new MR.MeshTriPoint(_1);}

        /// Generated from constructor `MR::MeshTriPoint::MeshTriPoint`.
        public static unsafe implicit operator _InOptConst_MeshTriPoint(MR.Const_EdgePoint ep) {return new MR.MeshTriPoint(ep);}
    }

    /// returns true if two points are equal considering different representations
    /// Generated from function `MR::same`.
    public static unsafe bool Same(MR.Const_MeshTopology topology, MR.Const_MeshTriPoint lhs, MR.Const_MeshTriPoint rhs)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_same_MR_MeshTriPoint", ExactSpelling = true)]
        extern static byte __MR_same_MR_MeshTriPoint(MR.Const_MeshTopology._Underlying *topology, MR.Const_MeshTriPoint._Underlying *lhs, MR.Const_MeshTriPoint._Underlying *rhs);
        return __MR_same_MR_MeshTriPoint(topology._UnderlyingPtr, lhs._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
    }

    /// returns true if points a and b are located insides or on a boundary of the same triangle;
    /// if true a.e and b.e are updated to have that triangle on the left
    /// Generated from function `MR::fromSameTriangle`.
    public static unsafe bool FromSameTriangle(MR.Const_MeshTopology topology, MR.MeshTriPoint a, MR.MeshTriPoint b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fromSameTriangle_MR_MeshTriPoint_ref", ExactSpelling = true)]
        extern static byte __MR_fromSameTriangle_MR_MeshTriPoint_ref(MR.Const_MeshTopology._Underlying *topology, MR.MeshTriPoint._Underlying *a, MR.MeshTriPoint._Underlying *b);
        return __MR_fromSameTriangle_MR_MeshTriPoint_ref(topology._UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr) != 0;
    }

    /// returns true if points a and b are located insides or on a boundary of the same triangle;
    /// if true a.e and b.e are updated to have that triangle on the left
    /// Generated from function `MR::fromSameTriangle`.
    public static unsafe bool FromSameTriangle(MR.Const_MeshTopology topology, MR.Misc._Moved<MR.MeshTriPoint> a, MR.Misc._Moved<MR.MeshTriPoint> b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fromSameTriangle_MR_MeshTriPoint_rvalue_ref", ExactSpelling = true)]
        extern static byte __MR_fromSameTriangle_MR_MeshTriPoint_rvalue_ref(MR.Const_MeshTopology._Underlying *topology, MR.MeshTriPoint._Underlying *a, MR.MeshTriPoint._Underlying *b);
        return __MR_fromSameTriangle_MR_MeshTriPoint_rvalue_ref(topology._UnderlyingPtr, a.Value._UnderlyingPtr, b.Value._UnderlyingPtr) != 0;
    }

    /// returns MeshTriPoint representation of given vertex with given edge field; or invalid MeshTriPoint if it is not possible
    /// Generated from function `MR::getVertexAsMeshTriPoint`.
    public static unsafe MR.MeshTriPoint GetVertexAsMeshTriPoint(MR.Const_MeshTopology topology, MR.EdgeId e, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getVertexAsMeshTriPoint", ExactSpelling = true)]
        extern static MR.MeshTriPoint._Underlying *__MR_getVertexAsMeshTriPoint(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId e, MR.VertId v);
        return new(__MR_getVertexAsMeshTriPoint(topology._UnderlyingPtr, e, v), is_owning: true);
    }
}
