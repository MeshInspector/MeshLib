public static partial class MR
{
    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTree`
    /// This is the const half of the class.
    public class Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy(_Underlying *_this);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::AABBTreeBase`.
        public unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRFaceTagMRBox3f_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRFaceTagMRBox3f_MRNodeId._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_nodes(_Underlying *_this);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_rootNodeId();
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_heapBytes(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_numLeaves(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.FaceBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_FaceBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getNodesFromLeaves(_Underlying *_this, MR.Const_FaceBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.FaceBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getLeafOrder(_Underlying *_this, MR.FaceBMap._Underlying *leafMap);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTree`
    /// This is the non-const half of the class.
    public class AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f : Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        internal unsafe AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::AABBTreeBase`.
        public unsafe AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::operator=`.
        public unsafe MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f Assign(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f._Underlying *_other);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::FaceTag, MR::Box3f>>::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.FaceBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getLeafOrderAndReset(_Underlying *_this, MR.FaceBMap._Underlying *leafMap);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_FaceTag_MR_Box3f_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        internal readonly Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f arg) {return new(arg);}
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(MR.Misc._Moved<AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(MR.Misc._Moved<AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f` directly.
    public class _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        public AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f? Opt;

        public _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f() {}
        public _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f` to pass it to the function.
    public class _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f
    {
        public Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f? Opt;

        public _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f() {}
        public _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRFaceTagMRBox3f value) {return new(value);}
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::ObjTreeTraits>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeObjects`
    /// This is the const half of the class.
    public class Const_AABBTreeBase_MRObjTreeTraits : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeBase_MRObjTreeTraits(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_ObjTreeTraits_Destroy(_Underlying *_this);
            __MR_AABBTreeBase_MR_ObjTreeTraits_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeBase_MRObjTreeTraits() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeBase_MRObjTreeTraits() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_ObjTreeTraits_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::ObjTreeTraits>::AABBTreeBase`.
        public unsafe Const_AABBTreeBase_MRObjTreeTraits(MR._ByValue_AABBTreeBase_MRObjTreeTraits _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRObjTreeTraits._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_ObjTreeTraits_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRObjTreeTraits_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRObjTreeTraits_MRNodeId._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_nodes(_Underlying *_this);
            return new(__MR_AABBTreeBase_MR_ObjTreeTraits_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRObjTreeTraits Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreeBase_MR_ObjTreeTraits_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreeBase_MR_ObjTreeTraits_rootNodeId();
            return __MR_AABBTreeBase_MR_ObjTreeTraits_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTreeBase_MR_ObjTreeTraits_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreeBase_MR_ObjTreeTraits_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_ObjTreeTraits_heapBytes(_Underlying *_this);
            return __MR_AABBTreeBase_MR_ObjTreeTraits_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_ObjTreeTraits_numLeaves(_Underlying *_this);
            return __MR_AABBTreeBase_MR_ObjTreeTraits_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreeBase_MR_ObjTreeTraits_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_AABBTreeBase_MR_ObjTreeTraits_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_ObjBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_getNodesFromLeaves(_Underlying *_this, MR.Const_ObjBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreeBase_MR_ObjTreeTraits_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.BMap_MRObjId_MRObjId leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_ObjTreeTraits_getLeafOrder(_Underlying *_this, MR.BMap_MRObjId_MRObjId._Underlying *leafMap);
            __MR_AABBTreeBase_MR_ObjTreeTraits_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::ObjTreeTraits>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeObjects`
    /// This is the non-const half of the class.
    public class AABBTreeBase_MRObjTreeTraits : Const_AABBTreeBase_MRObjTreeTraits
    {
        internal unsafe AABBTreeBase_MRObjTreeTraits(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeBase_MRObjTreeTraits() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_ObjTreeTraits_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::ObjTreeTraits>::AABBTreeBase`.
        public unsafe AABBTreeBase_MRObjTreeTraits(MR._ByValue_AABBTreeBase_MRObjTreeTraits _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRObjTreeTraits._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_ObjTreeTraits_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::operator=`.
        public unsafe MR.AABBTreeBase_MRObjTreeTraits Assign(MR._ByValue_AABBTreeBase_MRObjTreeTraits _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeBase_MR_ObjTreeTraits_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRObjTreeTraits._Underlying *_other);
            return new(__MR_AABBTreeBase_MR_ObjTreeTraits_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::ObjTreeTraits>::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.BMap_MRObjId_MRObjId leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_ObjTreeTraits_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_ObjTreeTraits_getLeafOrderAndReset(_Underlying *_this, MR.BMap_MRObjId_MRObjId._Underlying *leafMap);
            __MR_AABBTreeBase_MR_ObjTreeTraits_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreeBase_MRObjTreeTraits` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreeBase_MRObjTreeTraits`/`Const_AABBTreeBase_MRObjTreeTraits` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreeBase_MRObjTreeTraits
    {
        internal readonly Const_AABBTreeBase_MRObjTreeTraits? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreeBase_MRObjTreeTraits() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreeBase_MRObjTreeTraits(Const_AABBTreeBase_MRObjTreeTraits new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_AABBTreeBase_MRObjTreeTraits(Const_AABBTreeBase_MRObjTreeTraits arg) {return new(arg);}
        public _ByValue_AABBTreeBase_MRObjTreeTraits(MR.Misc._Moved<AABBTreeBase_MRObjTreeTraits> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreeBase_MRObjTreeTraits(MR.Misc._Moved<AABBTreeBase_MRObjTreeTraits> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRObjTreeTraits` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeBase_MRObjTreeTraits`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRObjTreeTraits`/`Const_AABBTreeBase_MRObjTreeTraits` directly.
    public class _InOptMut_AABBTreeBase_MRObjTreeTraits
    {
        public AABBTreeBase_MRObjTreeTraits? Opt;

        public _InOptMut_AABBTreeBase_MRObjTreeTraits() {}
        public _InOptMut_AABBTreeBase_MRObjTreeTraits(AABBTreeBase_MRObjTreeTraits value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeBase_MRObjTreeTraits(AABBTreeBase_MRObjTreeTraits value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRObjTreeTraits` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeBase_MRObjTreeTraits`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRObjTreeTraits`/`Const_AABBTreeBase_MRObjTreeTraits` to pass it to the function.
    public class _InOptConst_AABBTreeBase_MRObjTreeTraits
    {
        public Const_AABBTreeBase_MRObjTreeTraits? Opt;

        public _InOptConst_AABBTreeBase_MRObjTreeTraits() {}
        public _InOptConst_AABBTreeBase_MRObjTreeTraits(Const_AABBTreeBase_MRObjTreeTraits value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeBase_MRObjTreeTraits(Const_AABBTreeBase_MRObjTreeTraits value) {return new(value);}
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreePolyline2`
    /// This is the const half of the class.
    public class Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy(_Underlying *_this);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::AABBTreeBase`.
        public unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f_MRNodeId._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_nodes(_Underlying *_this);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_rootNodeId();
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::getBoundingBox`.
        public unsafe MR.Box2f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box2f __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_heapBytes(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_numLeaves(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_UndirectedEdgeBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getNodesFromLeaves(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getLeafOrder(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreePolyline2`
    /// This is the non-const half of the class.
    public class AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f : Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        internal unsafe AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::AABBTreeBase`.
        public unsafe AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::operator=`.
        public unsafe MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f Assign(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f._Underlying *_other);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box2f>>::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getLeafOrderAndReset(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box2f_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        internal readonly Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f arg) {return new(arg);}
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR.Misc._Moved<AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(MR.Misc._Moved<AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` directly.
    public class _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        public AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f? Opt;

        public _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {}
        public _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f` to pass it to the function.
    public class _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f
    {
        public Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f? Opt;

        public _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f() {}
        public _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox2f value) {return new(value);}
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreePolyline3`
    /// This is the const half of the class.
    public class Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy(_Underlying *_this);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::AABBTreeBase`.
        public unsafe Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f_MRNodeId._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_nodes(_Underlying *_this);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_rootNodeId();
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_heapBytes(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_numLeaves(_Underlying *_this);
            return __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_UndirectedEdgeBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getNodesFromLeaves(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getLeafOrder(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// base class for most AABB-trees (except for AABBTreePoints)
    /// Generated from class `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreePolyline3`
    /// This is the non-const half of the class.
    public class AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f : Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        internal unsafe AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::AABBTreeBase`.
        public unsafe AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::operator=`.
        public unsafe MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f Assign(MR._ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f._Underlying *_other);
            return new(__MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeBase<MR::AABBTreeTraits<MR::UndirectedEdgeTag, MR::Box3f>>::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.UndirectedEdgeBMap leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getLeafOrderAndReset(_Underlying *_this, MR.UndirectedEdgeBMap._Underlying *leafMap);
            __MR_AABBTreeBase_MR_AABBTreeTraits_MR_UndirectedEdgeTag_MR_Box3f_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        internal readonly Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f arg) {return new(arg);}
        public _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR.Misc._Moved<AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(MR.Misc._Moved<AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` directly.
    public class _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        public AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f? Opt;

        public _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {}
        public _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f`/`Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f` to pass it to the function.
    public class _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f
    {
        public Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f? Opt;

        public _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f() {}
        public _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f(Const_AABBTreeBase_MRAABBTreeTraitsMRUndirectedEdgeTagMRBox3f value) {return new(value);}
    }
}
