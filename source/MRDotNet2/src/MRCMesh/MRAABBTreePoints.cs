public static partial class MR
{
    /// bounding volume hierarchy for point cloud structure
    /// Generated from class `MR::AABBTreePoints`.
    /// This is the const half of the class.
    public class Const_AABBTreePoints : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AABBTreePoints(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Destroy", ExactSpelling = true)]
            extern static void __MR_AABBTreePoints_Destroy(_Underlying *_this);
            __MR_AABBTreePoints_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AABBTreePoints() {Dispose(false);}

        /// maximum number of points in leaf node of tree (all of leafs should have this number of points except last one)
        public static unsafe int MaxNumPointsInLeaf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Get_MaxNumPointsInLeaf", ExactSpelling = true)]
                extern static int *__MR_AABBTreePoints_Get_MaxNumPointsInLeaf();
                return *__MR_AABBTreePoints_Get_MaxNumPointsInLeaf();
            }
        }

        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe Const_AABBTreePoints(MR._ByValue_AABBTreePoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreePoints._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreePoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given point cloud
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe Const_AABBTreePoints(MR.Const_PointCloud pointCloud) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Construct_1_MR_PointCloud", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_Construct_1_MR_PointCloud(MR.Const_PointCloud._Underlying *pointCloud);
            _UnderlyingPtr = __MR_AABBTreePoints_Construct_1_MR_PointCloud(pointCloud._UnderlyingPtr);
        }

        /// creates tree for given point cloud
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator Const_AABBTreePoints(MR.Const_PointCloud pointCloud) {return new(pointCloud);}

        /// creates tree for vertices of given mesh
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe Const_AABBTreePoints(MR.Const_Mesh mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Construct_1_MR_Mesh", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_Construct_1_MR_Mesh(MR.Const_Mesh._Underlying *mesh);
            _UnderlyingPtr = __MR_AABBTreePoints_Construct_1_MR_Mesh(mesh._UnderlyingPtr);
        }

        /// creates tree for vertices of given mesh
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator Const_AABBTreePoints(MR.Const_Mesh mesh) {return new(mesh);}

        /// creates tree from given valid points
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe Const_AABBTreePoints(MR.Const_VertCoords points, MR.Const_VertBitSet? validPoints = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Construct_2", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_Construct_2(MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *validPoints);
            _UnderlyingPtr = __MR_AABBTreePoints_Construct_2(points._UnderlyingPtr, validPoints is not null ? validPoints._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreePoints::nodes`.
        public unsafe MR.Const_Vector_MRAABBTreePointsNode_MRNodeId Nodes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_nodes", ExactSpelling = true)]
            extern static MR.Const_Vector_MRAABBTreePointsNode_MRNodeId._Underlying *__MR_AABBTreePoints_nodes(_Underlying *_this);
            return new(__MR_AABBTreePoints_nodes(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::AABBTreePoints::operator[]`.
        public unsafe MR.AABBTreePoints.Const_Node Index(MR.NodeId nid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_index", ExactSpelling = true)]
            extern static MR.AABBTreePoints.Const_Node._Underlying *__MR_AABBTreePoints_index(_Underlying *_this, MR.NodeId nid);
            return new(__MR_AABBTreePoints_index(_UnderlyingPtr, nid), is_owning: false);
        }

        /// Generated from method `MR::AABBTreePoints::rootNodeId`.
        public static MR.NodeId RootNodeId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_rootNodeId", ExactSpelling = true)]
            extern static MR.NodeId __MR_AABBTreePoints_rootNodeId();
            return __MR_AABBTreePoints_rootNodeId();
        }

        /// returns the root node bounding box
        /// Generated from method `MR::AABBTreePoints::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_AABBTreePoints_getBoundingBox(_Underlying *_this);
            return __MR_AABBTreePoints_getBoundingBox(_UnderlyingPtr);
        }

        /// Generated from method `MR::AABBTreePoints::orderedPoints`.
        public unsafe MR.Std.Const_Vector_MRAABBTreePointsPoint OrderedPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_orderedPoints", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRAABBTreePointsPoint._Underlying *__MR_AABBTreePoints_orderedPoints(_Underlying *_this);
            return new(__MR_AABBTreePoints_orderedPoints(_UnderlyingPtr), is_owning: false);
        }

        /// returns the mapping original VertId to new id following the points order in the tree;
        /// buffer in vertMap must be resized before the call, and caller is responsible for filling missing vertex elements
        /// Generated from method `MR::AABBTreePoints::getLeafOrder`.
        public unsafe void GetLeafOrder(MR.VertBMap vertMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_getLeafOrder", ExactSpelling = true)]
            extern static void __MR_AABBTreePoints_getLeafOrder(_Underlying *_this, MR.VertBMap._Underlying *vertMap);
            __MR_AABBTreePoints_getLeafOrder(_UnderlyingPtr, vertMap._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::AABBTreePoints::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_AABBTreePoints_heapBytes(_Underlying *_this);
            return __MR_AABBTreePoints_heapBytes(_UnderlyingPtr);
        }

        /// Generated from class `MR::AABBTreePoints::Node`.
        /// This is the const half of the class.
        public class Const_Node : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Node(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_Destroy", ExactSpelling = true)]
                extern static void __MR_AABBTreePoints_Node_Destroy(_Underlying *_this);
                __MR_AABBTreePoints_Node_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Node() {Dispose(false);}

            ///< bounding box of whole subtree
            public unsafe MR.Const_Box3f Box
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_Get_box", ExactSpelling = true)]
                    extern static MR.Const_Box3f._Underlying *__MR_AABBTreePoints_Node_Get_box(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Node_Get_box(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< left child node for an inner node, or -(l+1) is the index of the first point in a leaf node
            public unsafe MR.Const_NodeId L
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_Get_l", ExactSpelling = true)]
                    extern static MR.Const_NodeId._Underlying *__MR_AABBTreePoints_Node_Get_l(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Node_Get_l(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< right child node for an inner node, or -(r+1) is the index of the last point in a leaf node
            public unsafe MR.Const_NodeId R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_Get_r", ExactSpelling = true)]
                    extern static MR.Const_NodeId._Underlying *__MR_AABBTreePoints_Node_Get_r(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Node_Get_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Node() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_DefaultConstruct", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_DefaultConstruct();
                _UnderlyingPtr = __MR_AABBTreePoints_Node_DefaultConstruct();
            }

            /// Constructs `MR::AABBTreePoints::Node` elementwise.
            public unsafe Const_Node(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_ConstructFrom", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
                _UnderlyingPtr = __MR_AABBTreePoints_Node_ConstructFrom(box, l, r);
            }

            /// Generated from constructor `MR::AABBTreePoints::Node::Node`.
            public unsafe Const_Node(MR.AABBTreePoints.Const_Node _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_ConstructFromAnother(MR.AABBTreePoints.Node._Underlying *_other);
                _UnderlyingPtr = __MR_AABBTreePoints_Node_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// returns true if node represent real points, false if it has child nodes
            /// Generated from method `MR::AABBTreePoints::Node::leaf`.
            public unsafe bool Leaf()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_leaf", ExactSpelling = true)]
                extern static byte __MR_AABBTreePoints_Node_leaf(_Underlying *_this);
                return __MR_AABBTreePoints_Node_leaf(_UnderlyingPtr) != 0;
            }

            /// returns [first,last) indices of leaf points
            /// Generated from method `MR::AABBTreePoints::Node::getLeafPointRange`.
            public unsafe MR.Std.Pair_Int_Int GetLeafPointRange()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_getLeafPointRange", ExactSpelling = true)]
                extern static MR.Std.Pair_Int_Int._Underlying *__MR_AABBTreePoints_Node_getLeafPointRange(_Underlying *_this);
                return new(__MR_AABBTreePoints_Node_getLeafPointRange(_UnderlyingPtr), is_owning: true);
            }
        }

        /// Generated from class `MR::AABBTreePoints::Node`.
        /// This is the non-const half of the class.
        public class Node : Const_Node
        {
            internal unsafe Node(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            ///< bounding box of whole subtree
            public new unsafe MR.Mut_Box3f Box
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_GetMutable_box", ExactSpelling = true)]
                    extern static MR.Mut_Box3f._Underlying *__MR_AABBTreePoints_Node_GetMutable_box(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Node_GetMutable_box(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< left child node for an inner node, or -(l+1) is the index of the first point in a leaf node
            public new unsafe MR.Mut_NodeId L
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_GetMutable_l", ExactSpelling = true)]
                    extern static MR.Mut_NodeId._Underlying *__MR_AABBTreePoints_Node_GetMutable_l(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Node_GetMutable_l(_UnderlyingPtr), is_owning: false);
                }
            }

            ///< right child node for an inner node, or -(r+1) is the index of the last point in a leaf node
            public new unsafe MR.Mut_NodeId R
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_GetMutable_r", ExactSpelling = true)]
                    extern static MR.Mut_NodeId._Underlying *__MR_AABBTreePoints_Node_GetMutable_r(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Node_GetMutable_r(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Node() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_DefaultConstruct", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_DefaultConstruct();
                _UnderlyingPtr = __MR_AABBTreePoints_Node_DefaultConstruct();
            }

            /// Constructs `MR::AABBTreePoints::Node` elementwise.
            public unsafe Node(MR.Box3f box, MR.NodeId l, MR.NodeId r) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_ConstructFrom", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_ConstructFrom(MR.Box3f box, MR.NodeId l, MR.NodeId r);
                _UnderlyingPtr = __MR_AABBTreePoints_Node_ConstructFrom(box, l, r);
            }

            /// Generated from constructor `MR::AABBTreePoints::Node::Node`.
            public unsafe Node(MR.AABBTreePoints.Const_Node _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_ConstructFromAnother(MR.AABBTreePoints.Node._Underlying *_other);
                _UnderlyingPtr = __MR_AABBTreePoints_Node_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::AABBTreePoints::Node::operator=`.
            public unsafe MR.AABBTreePoints.Node Assign(MR.AABBTreePoints.Const_Node _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_AssignFromAnother", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Node._Underlying *__MR_AABBTreePoints_Node_AssignFromAnother(_Underlying *_this, MR.AABBTreePoints.Node._Underlying *_other);
                return new(__MR_AABBTreePoints_Node_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }

            /// sets [first,last) to this node (leaf)
            /// Generated from method `MR::AABBTreePoints::Node::setLeafPointRange`.
            public unsafe void SetLeafPointRange(int first, int last)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Node_setLeafPointRange", ExactSpelling = true)]
                extern static void __MR_AABBTreePoints_Node_setLeafPointRange(_Underlying *_this, int first, int last);
                __MR_AABBTreePoints_Node_setLeafPointRange(_UnderlyingPtr, first, last);
            }
        }

        /// This is used for optional parameters of class `Node` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Node`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Node`/`Const_Node` directly.
        public class _InOptMut_Node
        {
            public Node? Opt;

            public _InOptMut_Node() {}
            public _InOptMut_Node(Node value) {Opt = value;}
            public static implicit operator _InOptMut_Node(Node value) {return new(value);}
        }

        /// This is used for optional parameters of class `Node` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Node`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Node`/`Const_Node` to pass it to the function.
        public class _InOptConst_Node
        {
            public Const_Node? Opt;

            public _InOptConst_Node() {}
            public _InOptConst_Node(Const_Node value) {Opt = value;}
            public static implicit operator _InOptConst_Node(Const_Node value) {return new(value);}
        }

        /// Generated from class `MR::AABBTreePoints::Point`.
        /// This is the const half of the class.
        public class Const_Point : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Point(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_Destroy", ExactSpelling = true)]
                extern static void __MR_AABBTreePoints_Point_Destroy(_Underlying *_this);
                __MR_AABBTreePoints_Point_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Point() {Dispose(false);}

            public unsafe MR.Const_Vector3f Coord
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_Get_coord", ExactSpelling = true)]
                    extern static MR.Const_Vector3f._Underlying *__MR_AABBTreePoints_Point_Get_coord(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Point_Get_coord(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_VertId Id
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_Get_id", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_AABBTreePoints_Point_Get_id(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Point_Get_id(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Point() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_DefaultConstruct", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_DefaultConstruct();
                _UnderlyingPtr = __MR_AABBTreePoints_Point_DefaultConstruct();
            }

            /// Constructs `MR::AABBTreePoints::Point` elementwise.
            public unsafe Const_Point(MR.Vector3f coord, MR.VertId id) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_ConstructFrom", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_ConstructFrom(MR.Vector3f coord, MR.VertId id);
                _UnderlyingPtr = __MR_AABBTreePoints_Point_ConstructFrom(coord, id);
            }

            /// Generated from constructor `MR::AABBTreePoints::Point::Point`.
            public unsafe Const_Point(MR.AABBTreePoints.Const_Point _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_ConstructFromAnother(MR.AABBTreePoints.Point._Underlying *_other);
                _UnderlyingPtr = __MR_AABBTreePoints_Point_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::AABBTreePoints::Point`.
        /// This is the non-const half of the class.
        public class Point : Const_Point
        {
            internal unsafe Point(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_Vector3f Coord
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_GetMutable_coord", ExactSpelling = true)]
                    extern static MR.Mut_Vector3f._Underlying *__MR_AABBTreePoints_Point_GetMutable_coord(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Point_GetMutable_coord(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_VertId Id
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_GetMutable_id", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_AABBTreePoints_Point_GetMutable_id(_Underlying *_this);
                    return new(__MR_AABBTreePoints_Point_GetMutable_id(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Point() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_DefaultConstruct", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_DefaultConstruct();
                _UnderlyingPtr = __MR_AABBTreePoints_Point_DefaultConstruct();
            }

            /// Constructs `MR::AABBTreePoints::Point` elementwise.
            public unsafe Point(MR.Vector3f coord, MR.VertId id) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_ConstructFrom", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_ConstructFrom(MR.Vector3f coord, MR.VertId id);
                _UnderlyingPtr = __MR_AABBTreePoints_Point_ConstructFrom(coord, id);
            }

            /// Generated from constructor `MR::AABBTreePoints::Point::Point`.
            public unsafe Point(MR.AABBTreePoints.Const_Point _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_ConstructFromAnother(MR.AABBTreePoints.Point._Underlying *_other);
                _UnderlyingPtr = __MR_AABBTreePoints_Point_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::AABBTreePoints::Point::operator=`.
            public unsafe MR.AABBTreePoints.Point Assign(MR.AABBTreePoints.Const_Point _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Point_AssignFromAnother", ExactSpelling = true)]
                extern static MR.AABBTreePoints.Point._Underlying *__MR_AABBTreePoints_Point_AssignFromAnother(_Underlying *_this, MR.AABBTreePoints.Point._Underlying *_other);
                return new(__MR_AABBTreePoints_Point_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Point` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Point`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Point`/`Const_Point` directly.
        public class _InOptMut_Point
        {
            public Point? Opt;

            public _InOptMut_Point() {}
            public _InOptMut_Point(Point value) {Opt = value;}
            public static implicit operator _InOptMut_Point(Point value) {return new(value);}
        }

        /// This is used for optional parameters of class `Point` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Point`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Point`/`Const_Point` to pass it to the function.
        public class _InOptConst_Point
        {
            public Const_Point? Opt;

            public _InOptConst_Point() {}
            public _InOptConst_Point(Const_Point value) {Opt = value;}
            public static implicit operator _InOptConst_Point(Const_Point value) {return new(value);}
        }
    }

    /// bounding volume hierarchy for point cloud structure
    /// Generated from class `MR::AABBTreePoints`.
    /// This is the non-const half of the class.
    public class AABBTreePoints : Const_AABBTreePoints
    {
        internal unsafe AABBTreePoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe AABBTreePoints(MR._ByValue_AABBTreePoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.AABBTreePoints._Underlying *_other);
            _UnderlyingPtr = __MR_AABBTreePoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// creates tree for given point cloud
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe AABBTreePoints(MR.Const_PointCloud pointCloud) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Construct_1_MR_PointCloud", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_Construct_1_MR_PointCloud(MR.Const_PointCloud._Underlying *pointCloud);
            _UnderlyingPtr = __MR_AABBTreePoints_Construct_1_MR_PointCloud(pointCloud._UnderlyingPtr);
        }

        /// creates tree for given point cloud
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator AABBTreePoints(MR.Const_PointCloud pointCloud) {return new(pointCloud);}

        /// creates tree for vertices of given mesh
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe AABBTreePoints(MR.Const_Mesh mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Construct_1_MR_Mesh", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_Construct_1_MR_Mesh(MR.Const_Mesh._Underlying *mesh);
            _UnderlyingPtr = __MR_AABBTreePoints_Construct_1_MR_Mesh(mesh._UnderlyingPtr);
        }

        /// creates tree for vertices of given mesh
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator AABBTreePoints(MR.Const_Mesh mesh) {return new(mesh);}

        /// creates tree from given valid points
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public unsafe AABBTreePoints(MR.Const_VertCoords points, MR.Const_VertBitSet? validPoints = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_Construct_2", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_Construct_2(MR.Const_VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *validPoints);
            _UnderlyingPtr = __MR_AABBTreePoints_Construct_2(points._UnderlyingPtr, validPoints is not null ? validPoints._UnderlyingPtr : null);
        }

        /// Generated from method `MR::AABBTreePoints::operator=`.
        public unsafe MR.AABBTreePoints Assign(MR._ByValue_AABBTreePoints _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AABBTreePoints._Underlying *__MR_AABBTreePoints_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.AABBTreePoints._Underlying *_other);
            return new(__MR_AABBTreePoints_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// returns the mapping original VertId to new id following the points order in the tree;
        /// then resets leaf order as if the points were renumberd following the mapping;
        /// buffer in vertMap must be resized before the call, and caller is responsible for filling missing vertex elements
        /// Generated from method `MR::AABBTreePoints::getLeafOrderAndReset`.
        public unsafe void GetLeafOrderAndReset(MR.VertBMap vertMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_getLeafOrderAndReset", ExactSpelling = true)]
            extern static void __MR_AABBTreePoints_getLeafOrderAndReset(_Underlying *_this, MR.VertBMap._Underlying *vertMap);
            __MR_AABBTreePoints_getLeafOrderAndReset(_UnderlyingPtr, vertMap._UnderlyingPtr);
        }

        /// updates bounding boxes of the nodes containing changed vertices;
        /// this is a faster alternative to full tree rebuild (but the tree after refit might be less efficient)
        /// \param newCoords coordinates of all vertices including changed ones;
        /// \param changedVerts vertex ids with modified coordinates (since tree construction or last refit)
        /// Generated from method `MR::AABBTreePoints::refit`.
        public unsafe void Refit(MR.Const_VertCoords newCoords, MR.Const_VertBitSet changedVerts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AABBTreePoints_refit", ExactSpelling = true)]
            extern static void __MR_AABBTreePoints_refit(_Underlying *_this, MR.Const_VertCoords._Underlying *newCoords, MR.Const_VertBitSet._Underlying *changedVerts);
            __MR_AABBTreePoints_refit(_UnderlyingPtr, newCoords._UnderlyingPtr, changedVerts._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `AABBTreePoints` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `AABBTreePoints`/`Const_AABBTreePoints` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_AABBTreePoints
    {
        internal readonly Const_AABBTreePoints? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_AABBTreePoints(MR.Misc._Moved<AABBTreePoints> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_AABBTreePoints(MR.Misc._Moved<AABBTreePoints> arg) {return new(arg);}

        /// creates tree for given point cloud
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator _ByValue_AABBTreePoints(MR.Const_PointCloud pointCloud) {return new MR.AABBTreePoints(pointCloud);}

        /// creates tree for vertices of given mesh
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator _ByValue_AABBTreePoints(MR.Const_Mesh mesh) {return new MR.AABBTreePoints(mesh);}
    }

    /// This is used for optional parameters of class `AABBTreePoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AABBTreePoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreePoints`/`Const_AABBTreePoints` directly.
    public class _InOptMut_AABBTreePoints
    {
        public AABBTreePoints? Opt;

        public _InOptMut_AABBTreePoints() {}
        public _InOptMut_AABBTreePoints(AABBTreePoints value) {Opt = value;}
        public static implicit operator _InOptMut_AABBTreePoints(AABBTreePoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `AABBTreePoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AABBTreePoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AABBTreePoints`/`Const_AABBTreePoints` to pass it to the function.
    public class _InOptConst_AABBTreePoints
    {
        public Const_AABBTreePoints? Opt;

        public _InOptConst_AABBTreePoints() {}
        public _InOptConst_AABBTreePoints(Const_AABBTreePoints value) {Opt = value;}
        public static implicit operator _InOptConst_AABBTreePoints(Const_AABBTreePoints value) {return new(value);}

        /// creates tree for given point cloud
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator _InOptConst_AABBTreePoints(MR.Const_PointCloud pointCloud) {return new MR.AABBTreePoints(pointCloud);}

        /// creates tree for vertices of given mesh
        /// Generated from constructor `MR::AABBTreePoints::AABBTreePoints`.
        public static unsafe implicit operator _InOptConst_AABBTreePoints(MR.Const_Mesh mesh) {return new MR.AABBTreePoints(mesh);}
    }

    // returns the number of nodes in the binary tree with given number of points
    /// Generated from function `MR::getNumNodesPoints`.
    public static int GetNumNodesPoints(int numPoints)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getNumNodesPoints", ExactSpelling = true)]
        extern static int __MR_getNumNodesPoints(int numPoints);
        return __MR_getNumNodesPoints(numPoints);
    }
}
