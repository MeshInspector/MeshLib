public static partial class MR
{
    /// Generated from class `MR::ObjTreeTraits`.
    /// This is the const half of the class.
    public class Const_ObjTreeTraits : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjTreeTraits(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjTreeTraits_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjTreeTraits_Destroy(_Underlying *_this);
            __MR_ObjTreeTraits_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjTreeTraits() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjTreeTraits() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjTreeTraits_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjTreeTraits._Underlying *__MR_ObjTreeTraits_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjTreeTraits_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjTreeTraits::ObjTreeTraits`.
        public unsafe Const_ObjTreeTraits(MR.Const_ObjTreeTraits _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjTreeTraits_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjTreeTraits._Underlying *__MR_ObjTreeTraits_ConstructFromAnother(MR.ObjTreeTraits._Underlying *_other);
            _UnderlyingPtr = __MR_ObjTreeTraits_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjTreeTraits`.
    /// This is the non-const half of the class.
    public class ObjTreeTraits : Const_ObjTreeTraits
    {
        internal unsafe ObjTreeTraits(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjTreeTraits() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjTreeTraits_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjTreeTraits._Underlying *__MR_ObjTreeTraits_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjTreeTraits_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjTreeTraits::ObjTreeTraits`.
        public unsafe ObjTreeTraits(MR.Const_ObjTreeTraits _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjTreeTraits_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjTreeTraits._Underlying *__MR_ObjTreeTraits_ConstructFromAnother(MR.ObjTreeTraits._Underlying *_other);
            _UnderlyingPtr = __MR_ObjTreeTraits_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjTreeTraits::operator=`.
        public unsafe MR.ObjTreeTraits Assign(MR.Const_ObjTreeTraits _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjTreeTraits_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjTreeTraits._Underlying *__MR_ObjTreeTraits_AssignFromAnother(_Underlying *_this, MR.ObjTreeTraits._Underlying *_other);
            return new(__MR_ObjTreeTraits_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjTreeTraits` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjTreeTraits`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjTreeTraits`/`Const_ObjTreeTraits` directly.
    public class _InOptMut_ObjTreeTraits
    {
        public ObjTreeTraits? Opt;

        public _InOptMut_ObjTreeTraits() {}
        public _InOptMut_ObjTreeTraits(ObjTreeTraits value) {Opt = value;}
        public static implicit operator _InOptMut_ObjTreeTraits(ObjTreeTraits value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjTreeTraits` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjTreeTraits`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjTreeTraits`/`Const_ObjTreeTraits` to pass it to the function.
    public class _InOptConst_ObjTreeTraits
    {
        public Const_ObjTreeTraits? Opt;

        public _InOptConst_ObjTreeTraits() {}
        public _InOptConst_ObjTreeTraits(Const_ObjTreeTraits value) {Opt = value;}
        public static implicit operator _InOptConst_ObjTreeTraits(Const_ObjTreeTraits value) {return new(value);}
    }

    /// tree containing world bounding boxes of individual objects having individual local-to-world transformations
    /// Generated from class `MR::AABBTreeObjects`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::ObjTreeTraits>`
    /// This is the const half of the class.
    public class Const_AABBTreeObjects : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreeObjects(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreeObjects_Destroy(_Underlying *_this);
            __MR_AABBTreeObjects_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreeObjects() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_AABBTreeBase_MRObjTreeTraits(Const_AABBTreeObjects self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_UpcastTo_MR_AABBTreeBase_MR_ObjTreeTraits", ExactSpelling = true)]
            extern static MR.Const_AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeObjects_UpcastTo_MR_AABBTreeBase_MR_ObjTreeTraits(_Underlying *_this);
            MR.Const_AABBTreeBase_MRObjTreeTraits ret = new(__MR_AABBTreeObjects_UpcastTo_MR_AABBTreeBase_MR_ObjTreeTraits(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_AABBTreeObjects() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeObjects_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeObjects::AABBTreeObjects`.
        public unsafe Const_AABBTreeObjects(MR._ByValue_AABBTreeObjects _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeObjects._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeObjects_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given set of objects each with its own transformation
        /// Generated from constructor `MR::AABBTreeObjects::AABBTreeObjects`.
        public unsafe Const_AABBTreeObjects(MR._ByValue_Vector_MRMeshOrPointsXf_MRObjId objs) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_Construct", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_Construct(MR.Misc._PassBy objs_pass_by, MR.Vector_MRMeshOrPointsXf_MRObjId._Underlying *objs);
            _UnderlyingPtr = __MR_AABBTreeObjects_Construct(objs.PassByMode, objs.Value is not null ? objs.Value._UnderlyingPtr : null);
        }

        /// gets object by its id
        /// Generated from method `MR::AABBTreeObjects::obj`.
        public unsafe MR.Const_MeshOrPoints Obj(MR.ObjId oi)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_obj", ExactSpelling = true)]
            extern static MR.Const_MeshOrPoints._Underlying *__MR_AABBTreeObjects_obj(_Underlying *_this, MR.ObjId oi);
            return new(__MR_AABBTreeObjects_obj(_UnderlyingPtr, oi), is_owning: false);
        }

        /// gets transformation from local space of given object to world space
        /// Generated from method `MR::AABBTreeObjects::toWorld`.
        public unsafe MR.Const_AffineXf3f ToWorld(MR.ObjId oi)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_toWorld", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_AABBTreeObjects_toWorld(_Underlying *_this, MR.ObjId oi);
            return new(__MR_AABBTreeObjects_toWorld(_UnderlyingPtr, oi), is_owning: false);
        }

        /// gets transformation from world space to local space of given object
        /// Generated from method `MR::AABBTreeObjects::toLocal`.
        public unsafe MR.Const_AffineXf3f ToLocal(MR.ObjId oi)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_toLocal_1", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_AABBTreeObjects_toLocal_1(_Underlying *_this, MR.ObjId oi);
            return new(__MR_AABBTreeObjects_toLocal_1(_UnderlyingPtr, oi), is_owning: false);
        }

        /// gets mapping: objId -> its transformation from world space to local space
        /// Generated from method `MR::AABBTreeObjects::toLocal`.
        public unsafe MR.Const_Vector_MRAffineXf3f_MRObjId ToLocal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_toLocal_0", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAffineXf3f_MRObjId._Underlying *__MR_AABBTreeObjects_toLocal_0(_Underlying *_this);
            return new(__MR_AABBTreeObjects_toLocal_0(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to all nodes
        /// Generated from method `MR::AABBTreeObjects::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreeNodeMRObjTreeTraits_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreeNodeMRObjTreeTraits_MRNodeId._Underlying *__MR_AABBTreeObjects_nodes(_Underlying *_this);
            return new(__MR_AABBTreeObjects_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// const-access to any node
        /// Generated from method `MR::AABBTreeObjects::operator[]`.
        public unsafe MR.Const_AABBTreeNode_MRObjTreeTraits Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_index", ExactSpelling = true)]
            extern static MR.Const_AABBTreeNode_MRObjTreeTraits._Underlying *__MR_AABBTreeObjects_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreeObjects_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// returns root node id
        /// Generated from method `MR::AABBTreeObjects::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreeObjects_rootNodeId();
            return __MR_AABBTreeObjects_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreeObjects::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTreeObjects_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreeObjects_getBoundingBox(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreeObjects::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeObjects_heapBytes(_Underlying *_this);
            return __MR_AABBTreeObjects_heapBytes(_UnderlyingPtr);
        }

        /// returns the number of leaves in whole tree
        /// Generated from method `MR::AABBTreeObjects::numLeaves`.
        public unsafe ulong NumLeaves()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_numLeaves", ExactSpelling = true)]
            extern static ulong __MR_AABBTreeObjects_numLeaves(_Underlying *_this);
            return __MR_AABBTreeObjects_numLeaves(_UnderlyingPtr);
        }

        /// returns at least given number of top-level not-intersecting subtrees, union of which contain all tree leaves
        /// Generated from method `MR::AABBTreeObjects::getSubtrees`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRNodeId> GetSubtrees(int minNum)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_getSubtrees", ExactSpelling = true)]
            extern static MR.Std.Vector_MRNodeId._Underlying *__MR_AABBTreeObjects_getSubtrees(_Underlying *_this, int minNum);
            return MR.Misc.Move(new MR.Std.Vector_MRNodeId(__MR_AABBTreeObjects_getSubtrees(_UnderlyingPtr, minNum), is_owning: true));
        }

        /// returns all leaves in the subtree with given root
        /// Generated from method `MR::AABBTreeObjects::getSubtreeLeaves`.
        public unsafe MR.Misc._Moved<MR.ObjBitSet> GetSubtreeLeaves(MR.NodeId subtreeRoot)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_getSubtreeLeaves", ExactSpelling = true)]
            extern static MR.ObjBitSet._Underlying *__MR_AABBTreeObjects_getSubtreeLeaves(_Underlying *_this, MR.NodeId subtreeRoot);
            return MR.Misc.Move(new MR.ObjBitSet(__MR_AABBTreeObjects_getSubtreeLeaves(_UnderlyingPtr, subtreeRoot), is_owning: true));
        }

        /// returns set of nodes containing among direct or indirect children given leaves
        /// Generated from method `MR::AABBTreeObjects::getNodesFromLeaves`.
        public unsafe MR.Misc._Moved<MR.NodeBitSet> GetNodesFromLeaves(MR.Const_ObjBitSet leaves)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_getNodesFromLeaves", ExactSpelling = true)]
            extern static MR.NodeBitSet._Underlying *__MR_AABBTreeObjects_getNodesFromLeaves(_Underlying *_this, MR.Const_ObjBitSet._Underlying *leaves);
            return MR.Misc.Move(new MR.NodeBitSet(__MR_AABBTreeObjects_getNodesFromLeaves(_UnderlyingPtr, leaves._UnderlyingPtr), is_owning: true));
        }

        /// fills map: LeafId -> leaf#;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeObjects::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.BMap_MRObjId_MRObjId leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreeObjects_getLeafOrder(_Underlying *_this, MR.BMap_MRObjId_MRObjId._Underlying *leafMap);
            __MR_AABBTreeObjects_getLeafOrder(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// tree containing world bounding boxes of individual objects having individual local-to-world transformations
    /// Generated from class `MR::AABBTreeObjects`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::AABBTreeBase<MR::ObjTreeTraits>`
    /// This is the non-const half of the class.
    public class AABBTreeObjects : Const_AABBTreeObjects
    {
        internal unsafe AABBTreeObjects(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.AABBTreeBase_MRObjTreeTraits(AABBTreeObjects self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_UpcastTo_MR_AABBTreeBase_MR_ObjTreeTraits", ExactSpelling = true)]
            extern static MR.AABBTreeBase_MRObjTreeTraits._Underlying *__MR_AABBTreeObjects_UpcastTo_MR_AABBTreeBase_MR_ObjTreeTraits(_Underlying *_this);
            MR.AABBTreeBase_MRObjTreeTraits ret = new(__MR_AABBTreeObjects_UpcastTo_MR_AABBTreeBase_MR_ObjTreeTraits(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe AABBTreeObjects() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_DefaultConstruct", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_DefaultConstruct();
            _UnderlyingPtr = __MR_AABBTreeObjects_DefaultConstruct();
        }

        /// Generated from constructor `MR::AABBTreeObjects::AABBTreeObjects`.
        public unsafe AABBTreeObjects(MR._ByValue_AABBTreeObjects _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreeObjects._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreeObjects_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given set of objects each with its own transformation
        /// Generated from constructor `MR::AABBTreeObjects::AABBTreeObjects`.
        public unsafe AABBTreeObjects(MR._ByValue_Vector_MRMeshOrPointsXf_MRObjId objs) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_Construct", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_Construct(MR.Misc._PassBy objs_pass_by, MR.Vector_MRMeshOrPointsXf_MRObjId._Underlying *objs);
            _UnderlyingPtr = __MR_AABBTreeObjects_Construct(objs.PassByMode, objs.Value is not null ? objs.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreeObjects::operator=`.
        public unsafe MR.AABBTreeObjects Assign(MR._ByValue_AABBTreeObjects _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreeObjects._Underlying *__MR_AABBTreeObjects_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreeObjects._Underlying *_other);
            return new(__MR_AABBTreeObjects_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fills map: LeafId -> leaf#, then resets leaf order to 0,1,2,...;
        /// buffer in leafMap must be resized before the call, and caller is responsible for filling missing leaf elements
        /// Generated from method `MR::AABBTreeObjects::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.BMap_MRObjId_MRObjId leafMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreeObjects_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreeObjects_getLeafOrderAndReset(_Underlying *_this, MR.BMap_MRObjId_MRObjId._Underlying *leafMap);
            __MR_AABBTreeObjects_getLeafOrderAndReset(_UnderlyingPtr, leafMap._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreeObjects` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreeObjects`/`Const_AABBTreeObjects` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreeObjects
    {
        internal readonly Const_AABBTreeObjects? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreeObjects() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_AABBTreeObjects(Const_AABBTreeObjects new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_AABBTreeObjects(Const_AABBTreeObjects arg) {return new(arg);}
        public _ByValue_AABBTreeObjects(MR.Misc._Moved<AABBTreeObjects> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreeObjects(MR.Misc._Moved<AABBTreeObjects> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `AABBTreeObjects` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreeObjects`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeObjects`/`Const_AABBTreeObjects` directly.
    public class _InOptMut_AABBTreeObjects
    {
        public AABBTreeObjects? Opt;

        public _InOptMut_AABBTreeObjects() {}
        public _InOptMut_AABBTreeObjects(AABBTreeObjects value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreeObjects(AABBTreeObjects value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreeObjects` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreeObjects`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreeObjects`/`Const_AABBTreeObjects` to pass it to the function.
    public class _InOptConst_AABBTreeObjects
    {
        public Const_AABBTreeObjects? Opt;

        public _InOptConst_AABBTreeObjects() {}
        public _InOptConst_AABBTreeObjects(Const_AABBTreeObjects value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreeObjects(Const_AABBTreeObjects value) {return new(value);}
    }
}
