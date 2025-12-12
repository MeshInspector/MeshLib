public static partial class MR
{
    /// bounding volume hierarchy for line segments
    /// Generated from class `MR::AABBTreePolyline2`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>`
    /// This is the const half of the class.
    public class Const_AABBTreePolyline2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreePolyline2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreePolyline2_Destroy(_Underlying *_this);
            __MR_AABBTreePolyline2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreePolyline2() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreePolyline2 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f", ExactSpelling = true)]
            extern static MR.Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreePolyline2_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f(_Underlying *_this);
            MR.Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f ret = new(__MR_AABBTreePolyline2_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreePolyline2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreePolyline2_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreePolyline2::AABBTreePolyline2`.
        public unsafe Const_AABBTreePolyline2(MR._ByValue_AABBTreePolyline2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreePolyline2._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreePolyline2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given polyline
        /// Generated from constructor `MR::AABBTreePolyline2::AABBTreePolyline2`.
        public unsafe Const_AABBTreePolyline2(MR.Const_Polyline2 polyline) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_Construct", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_Construct(MR.Const_Polyline2._Underlying *polyline);
            _UnderlyingPtr = __MR_AABBTreePolyline2_Construct(polyline._UnderlyingPtr);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreePolyline2::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f_MRNodeId._Underlying *__MR_AABBTreePolyline2_nodes(_Underlying *_this);
            return new(__MR_AABBTreePolyline2_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreePolyline2::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreePolyline2_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreePolyline2_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreePolyline2::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreePolyline2_rootNodeId();
            return __MR_AABBTreePolyline2_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreePolyline2::getBoundingBox`.
        public unsafe MR.Box2f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box2f __MR_AABBTreePolyline2_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreePolyline2_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreePolyline2::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreePolyline2_heapBytes(_Underlying *_this);
            return __MR_AABBTreePolyline2_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreePolyline2::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreePolyline2_numLeaves(_Underlying *_this);
            return __MR_AABBTreePolyline2_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreePolyline2::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreePolyline2_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreePolyline2_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreePolyline2::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_AABBTreePolyline2_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_AABBTreePolyline2_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreePolyline2::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_UndirectedEdgeBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreePolyline2_getNodesFromLeaves(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreePolyline2_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreePolyline2::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreePolyline2_getLeafOrder(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreePolyline2_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// bounding volume hierarchy for line segments
    /// Generated from class `MR::AABBTreePolyline2`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>`
    /// This is the non-const half of the class.
    public class AABBTreePolyline2 : Const_AABBTreePolyline2
    {
        internal unsafe AABBTreePolyline2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(AABBTreePolyline2 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreePolyline2_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f(_Underlying *_this);
            MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f ret = new(__MR_AABBTreePolyline2_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreePolyline2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreePolyline2_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreePolyline2::AABBTreePolyline2`.
        public unsafe AABBTreePolyline2(MR._ByValue_AABBTreePolyline2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreePolyline2._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreePolyline2_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given polyline
        /// Generated from constructor `MR::AABBTreePolyline2::AABBTreePolyline2`.
        public unsafe AABBTreePolyline2(MR.Const_Polyline2 polyline) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_Construct", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_Construct(MR.Const_Polyline2._Underlying *polyline);
            _UnderlyingPtr = __MR_AABBTreePolyline2_Construct(polyline._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreePolyline2::operator=`.
        public unsafe MR.AABBTreePolyline2 Assign(MR._ByValue_AABBTreePolyline2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePolyline2._Underlying *__MR_AABBTreePolyline2_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreePolyline2._Underlying *_other);
            return new(__MR_AABBTreePolyline2_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreePolyline2::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline2_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreePolyline2_getLeafOrderAndReset(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreePolyline2_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreePolyline2` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreePolyline2`/`Const_AABBTreePolyline2` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreePolyline2
    {
        internal readonly Const_AABBTreePolyline2? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreePolyline2() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreePolyline2(MR.Misc._Moved<AABBTreePolyline2> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreePolyline2(MR.Misc._Moved<AABBTreePolyline2> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreePolyline2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreePolyline2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreePolyline2`/`Const_AABBTreePolyline2` directly.
    public class _InOptMut_AABBTreePolyline2
    {
        public AABBTreePolyline2? Opt;

        public _InOptMut_AABBTreePolyline2() {}
        public _InOptMut_AABBTreePolyline2(AABBTreePolyline2 value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreePolyline2(AABBTreePolyline2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreePolyline2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreePolyline2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreePolyline2`/`Const_AABBTreePolyline2` to pass it to the function.
    public class _InOptConst_AABBTreePolyline2
    {
        public Const_AABBTreePolyline2? Opt;

        public _InOptConst_AABBTreePolyline2() {}
        public _InOptConst_AABBTreePolyline2(Const_AABBTreePolyline2 value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreePolyline2(Const_AABBTreePolyline2 value) {return new(value);}
    }

    /// bounding volume hierarchy for line segments
    /// Generated from class `MR::AABBTreePolyline3`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>`
    /// This is the const half of the class.
    public class Const_AABBTreePolyline3 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreePolyline3(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreePolyline3_Destroy(_Underlying *_this);
            __MR_AABBTreePolyline3_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreePolyline3() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreePolyline3 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f", ExactSpelling = true)]
            extern static MR.Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreePolyline3_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f(_Underlying *_this);
            MR.Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f ret = new(__MR_AABBTreePolyline3_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreePolyline3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreePolyline3_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreePolyline3::AABBTreePolyline3`.
        public unsafe Const_AABBTreePolyline3(MR._ByValue_AABBTreePolyline3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreePolyline3._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreePolyline3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given polyline
        /// Generated from constructor `MR::AABBTreePolyline3::AABBTreePolyline3`.
        public unsafe Const_AABBTreePolyline3(MR.Const_Polyline3 polyline) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_Construct_1", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_Construct_1(MR.Const_Polyline3._Underlying *polyline);
            _UnderlyingPtr = __MR_AABBTreePolyline3_Construct_1(polyline._UnderlyingPtr);
        }

        /// creates tree for selected edges on the mesh (only for 3d tree)
        /// Generated from constructor `MR::AABBTreePolyline3::AABBTreePolyline3`.
        public unsafe Const_AABBTreePolyline3(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet edgeSet) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_Construct_2", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_Construct_2(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *edgeSet);
            _UnderlyingPtr = __MR_AABBTreePolyline3_Construct_2(mesh._UnderlyingPtr, edgeSet._UnderlyingPtr);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreePolyline3::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f_MRNodeId._Underlying *__MR_AABBTreePolyline3_nodes(_Underlying *_this);
            return new(__MR_AABBTreePolyline3_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreePolyline3::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreePolyline3_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreePolyline3_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreePolyline3::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreePolyline3_rootNodeId();
            return __MR_AABBTreePolyline3_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreePolyline3::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTreePolyline3_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreePolyline3_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreePolyline3::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreePolyline3_heapBytes(_Underlying *_this);
            return __MR_AABBTreePolyline3_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreePolyline3::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreePolyline3_numLeaves(_Underlying *_this);
            return __MR_AABBTreePolyline3_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreePolyline3::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreePolyline3_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreePolyline3_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreePolyline3::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_AABBTreePolyline3_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_AABBTreePolyline3_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreePolyline3::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_UndirectedEdgeBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreePolyline3_getNodesFromLeaves(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreePolyline3_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreePolyline3::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreePolyline3_getLeafOrder(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreePolyline3_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// bounding volume hierarchy for line segments
    /// Generated from class `MR::AABBTreePolyline3`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>`
    /// This is the non-const half of the class.
    public class AABBTreePolyline3 : Const_AABBTreePolyline3
    {
        internal unsafe AABBTreePolyline3(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(AABBTreePolyline3 self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreePolyline3_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f(_Underlying *_this);
            MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f ret = new(__MR_AABBTreePolyline3_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreePolyline3() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreePolyline3_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreePolyline3::AABBTreePolyline3`.
        public unsafe AABBTreePolyline3(MR._ByValue_AABBTreePolyline3 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreePolyline3._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreePolyline3_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given polyline
        /// Generated from constructor `MR::AABBTreePolyline3::AABBTreePolyline3`.
        public unsafe AABBTreePolyline3(MR.Const_Polyline3 polyline) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_Construct_1", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_Construct_1(MR.Const_Polyline3._Underlying *polyline);
            _UnderlyingPtr = __MR_AABBTreePolyline3_Construct_1(polyline._UnderlyingPtr);
        }

        /// creates tree for selected edges on the mesh (only for 3d tree)
        /// Generated from constructor `MR::AABBTreePolyline3::AABBTreePolyline3`.
        public unsafe AABBTreePolyline3(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet edgeSet) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_Construct_2", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_Construct_2(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *edgeSet);
            _UnderlyingPtr = __MR_AABBTreePolyline3_Construct_2(mesh._UnderlyingPtr, edgeSet._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreePolyline3::operator=`.
        public unsafe MR.AABBTreePolyline3 Assign(MR._ByValue_AABBTreePolyline3 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePolyline3._Underlying *__MR_AABBTreePolyline3_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreePolyline3._Underlying *_other);
            return new(__MR_AABBTreePolyline3_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreePolyline3::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePolyline3_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreePolyline3_getLeafOrderAndReset(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreePolyline3_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreePolyline3` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreePolyline3`/`Const_AABBTreePolyline3` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreePolyline3
    {
        internal readonly Const_AABBTreePolyline3? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreePolyline3() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreePolyline3(MR.Misc._Moved<AABBTreePolyline3> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreePolyline3(MR.Misc._Moved<AABBTreePolyline3> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreePolyline3` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreePolyline3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreePolyline3`/`Const_AABBTreePolyline3` directly.
    public class _InOptMut_AABBTreePolyline3
    {
        public AABBTreePolyline3? Opt;

        public _InOptMut_AABBTreePolyline3() {}
        public _InOptMut_AABBTreePolyline3(AABBTreePolyline3 value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreePolyline3(AABBTreePolyline3 value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreePolyline3` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreePolyline3`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreePolyline3`/`Const_AABBTreePolyline3` to pass it to the function.
    public class _InOptConst_AABBTreePolyline3
    {
        public Const_AABBTreePolyline3? Opt;

        public _InOptConst_AABBTreePolyline3() {}
        public _InOptConst_AABBTreePolyline3(Const_AABBTreePolyline3 value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreePolyline3(Const_AABBTreePolyline3 value) {return new(value);}
    }

    /// Generated from class `MR::PolylineTraits<MR::Vector2f>`.
    /// This is the const half of the class.
    public class Const_PolylineTraits_MRVector2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineTraits_MRVector2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector2f_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineTraits_MR_Vector2f_Destroy(_Underlying *_this);
            __MR_PolylineTraits_MR_Vector2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineTraits_MRVector2f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineTraits_MRVector2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector2f._Underlying *__MR_PolylineTraits_MR_Vector2f_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineTraits<MR::Vector2f>::PolylineTraits`.
        public unsafe Const_PolylineTraits_MRVector2f(MR.Const_PolylineTraits_MRVector2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector2f._Underlying *__MR_PolylineTraits_MR_Vector2f_ConstructFromAnother(MR.PolylineTraits_MRVector2f._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector2f_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PolylineTraits<MR::Vector2f>`.
    /// This is the non-const half of the class.
    public class PolylineTraits_MRVector2f : Const_PolylineTraits_MRVector2f
    {
        internal unsafe PolylineTraits_MRVector2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineTraits_MRVector2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector2f._Underlying *__MR_PolylineTraits_MR_Vector2f_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineTraits<MR::Vector2f>::PolylineTraits`.
        public unsafe PolylineTraits_MRVector2f(MR.Const_PolylineTraits_MRVector2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector2f._Underlying *__MR_PolylineTraits_MR_Vector2f_ConstructFromAnother(MR.PolylineTraits_MRVector2f._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineTraits<MR::Vector2f>::operator=`.
        public unsafe MR.PolylineTraits_MRVector2f Assign(MR.Const_PolylineTraits_MRVector2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector2f._Underlying *__MR_PolylineTraits_MR_Vector2f_AssignFromAnother(_Underlying *_this, MR.PolylineTraits_MRVector2f._Underlying *_other);
            return new(__MR_PolylineTraits_MR_Vector2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineTraits_MRVector2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineTraits_MRVector2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineTraits_MRVector2f`/`Const_PolylineTraits_MRVector2f` directly.
    public class _InOptMut_PolylineTraits_MRVector2f
    {
        public PolylineTraits_MRVector2f? Opt;

        public _InOptMut_PolylineTraits_MRVector2f() {}
        public _InOptMut_PolylineTraits_MRVector2f(PolylineTraits_MRVector2f value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineTraits_MRVector2f(PolylineTraits_MRVector2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineTraits_MRVector2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineTraits_MRVector2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineTraits_MRVector2f`/`Const_PolylineTraits_MRVector2f` to pass it to the function.
    public class _InOptConst_PolylineTraits_MRVector2f
    {
        public Const_PolylineTraits_MRVector2f? Opt;

        public _InOptConst_PolylineTraits_MRVector2f() {}
        public _InOptConst_PolylineTraits_MRVector2f(Const_PolylineTraits_MRVector2f value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineTraits_MRVector2f(Const_PolylineTraits_MRVector2f value) {return new(value);}
    }

    /// Generated from class `MR::PolylineTraits<MR::Vector3f>`.
    /// This is the const half of the class.
    public class Const_PolylineTraits_MRVector3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineTraits_MRVector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector3f_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineTraits_MR_Vector3f_Destroy(_Underlying *_this);
            __MR_PolylineTraits_MR_Vector3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineTraits_MRVector3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineTraits_MRVector3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector3f._Underlying *__MR_PolylineTraits_MR_Vector3f_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineTraits<MR::Vector3f>::PolylineTraits`.
        public unsafe Const_PolylineTraits_MRVector3f(MR.Const_PolylineTraits_MRVector3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector3f._Underlying *__MR_PolylineTraits_MR_Vector3f_ConstructFromAnother(MR.PolylineTraits_MRVector3f._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector3f_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PolylineTraits<MR::Vector3f>`.
    /// This is the non-const half of the class.
    public class PolylineTraits_MRVector3f : Const_PolylineTraits_MRVector3f
    {
        internal unsafe PolylineTraits_MRVector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineTraits_MRVector3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector3f._Underlying *__MR_PolylineTraits_MR_Vector3f_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineTraits<MR::Vector3f>::PolylineTraits`.
        public unsafe PolylineTraits_MRVector3f(MR.Const_PolylineTraits_MRVector3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector3f._Underlying *__MR_PolylineTraits_MR_Vector3f_ConstructFromAnother(MR.PolylineTraits_MRVector3f._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineTraits_MR_Vector3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineTraits<MR::Vector3f>::operator=`.
        public unsafe MR.PolylineTraits_MRVector3f Assign(MR.Const_PolylineTraits_MRVector3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineTraits_MR_Vector3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineTraits_MRVector3f._Underlying *__MR_PolylineTraits_MR_Vector3f_AssignFromAnother(_Underlying *_this, MR.PolylineTraits_MRVector3f._Underlying *_other);
            return new(__MR_PolylineTraits_MR_Vector3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineTraits_MRVector3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineTraits_MRVector3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineTraits_MRVector3f`/`Const_PolylineTraits_MRVector3f` directly.
    public class _InOptMut_PolylineTraits_MRVector3f
    {
        public PolylineTraits_MRVector3f? Opt;

        public _InOptMut_PolylineTraits_MRVector3f() {}
        public _InOptMut_PolylineTraits_MRVector3f(PolylineTraits_MRVector3f value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineTraits_MRVector3f(PolylineTraits_MRVector3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineTraits_MRVector3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineTraits_MRVector3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineTraits_MRVector3f`/`Const_PolylineTraits_MRVector3f` to pass it to the function.
    public class _InOptConst_PolylineTraits_MRVector3f
    {
        public Const_PolylineTraits_MRVector3f? Opt;

        public _InOptConst_PolylineTraits_MRVector3f() {}
        public _InOptConst_PolylineTraits_MRVector3f(Const_PolylineTraits_MRVector3f value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineTraits_MRVector3f(Const_PolylineTraits_MRVector3f value) {return new(value);}
    }
}
