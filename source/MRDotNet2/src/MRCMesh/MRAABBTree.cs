public static partial class MR
{
    /// bounding volume hierarchy
    /// Generated from class `MR::AABBTree`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>`
    /// This is the const half of the class.
    public class Const_AABBTree : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTree(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTree_Destroy(_Underlying *_this);
            __MR_AABBTree_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTree() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTree self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f", ExactSpelling = true)]
            extern static MR.Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTree_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f(_Underlying *_this);
            MR.Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f ret = new(__MR_AABBTree_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTree() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTree_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTree::AABBTree`.
        public unsafe Const_AABBTree(MR._ByValue_AABBTree _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTree._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTree_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given mesh or its part
        /// Generated from constructor `MR::AABBTree::AABBTree`.
        public unsafe Const_AABBTree(MR.Const_MeshPart mp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_Construct", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_Construct(MR.Const_MeshPart._Underlying *mp);
            _UnderlyingPtr = __MR_AABBTree_Construct(mp._UnderlyingPtr);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTree::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRFaceTagMRBox3f_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRFaceTagMRBox3f_MRNodeId._Underlying *__MR_AABBTree_nodes(_Underlying *_this);
            return new(__MR_AABBTree_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTree::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTree_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTree_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTree::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTree_rootNodeId();
            return __MR_AABBTree_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTree::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTree_getBoundingBox(_Underlying *_this);
            return __MR_AABBTree_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTree::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTree_heapBytes(_Underlying *_this);
            return __MR_AABBTree_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTree::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTree_numLeaves(_Underlying *_this);
            return __MR_AABBTree_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTree::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTree_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTree_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTree::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_AABBTree_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_AABBTree_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTree::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_FaceBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTree_getNodesFromLeaves(_Underlying *_this, MR.Const_FaceBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTree_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTree::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.FaceBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTree_getLeafOrder(_Underlying *_this, MR.FaceBMap._Underlying *leafMap);
            __MR_AABBTree_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// bounding volume hierarchy
    /// Generated from class `MR::AABBTree`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>`
    /// This is the non-const half of the class.
    public class AABBTree : Const_AABBTree
    {
        internal unsafe AABBTree(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(AABBTree self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTree_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f(_Underlying *_this);
            MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f ret = new(__MR_AABBTree_UpcastTo_MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTree() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTree_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTree::AABBTree`.
        public unsafe AABBTree(MR._ByValue_AABBTree _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTree._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTree_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given mesh or its part
        /// Generated from constructor `MR::AABBTree::AABBTree`.
        public unsafe AABBTree(MR.Const_MeshPart mp) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_Construct", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_Construct(MR.Const_MeshPart._Underlying *mp);
            _UnderlyingPtr = __MR_AABBTree_Construct(mp._UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTree::operator=`.
        public unsafe MR.AABBTree Assign(MR._ByValue_AABBTree _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTree._Underlying *__MR_AABBTree_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTree._Underlying *_other);
            return new(__MR_AABBTree_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// updates bounding boxes of the nodes containing changed vertices;
        /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
        /// \param mesh same mesh for which this tree was constructed but with updated coordinates;
        /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit)
        /// Generated from method `MR::AABBTree::refit`.
        public unsafe void Refit(MR.Const_Mesh mesh, MR.Const_VertBitSet changedVerts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_refit", ExactSpelling = true)]
            extern static void __MR_AABBTree_refit(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *changedVerts);
            __MR_AABBTree_refit(_UnderlyingPtr, mesh._UnderlyingPtr, changedVerts._UnderlyingPtr);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTree::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.FaceBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTree_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTree_getLeafOrderAndReset(_Underlying *_this, MR.FaceBMap._Underlying *leafMap);
            __MR_AABBTree_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTree` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTree`/`Const_AABBTree` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTree
    {
        internal readonly Const_AABBTree? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTree() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTree(MR.Misc._Moved<AABBTree> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTree(MR.Misc._Moved<AABBTree> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTree` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTree`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTree`/`Const_AABBTree` directly.
    public class _InOptMut_AABBTree
    {
        public AABBTree? Opt;

        public _InOptMut_AABBTree() {}
        public _InOptMut_AABBTree(AABBTree value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTree(AABBTree value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTree` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTree`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTree`/`Const_AABBTree` to pass it to the function.
    public class _InOptConst_AABBTree
    {
        public Const_AABBTree? Opt;

        public _InOptConst_AABBTree() {}
        public _InOptConst_AABBTree(Const_AABBTree value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTree(Const_AABBTree value) {return new(value);}
    }
}
