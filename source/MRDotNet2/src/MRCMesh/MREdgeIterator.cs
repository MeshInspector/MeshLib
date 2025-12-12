public static partial class MR
{
    // The iterator to find all not-lone undirected edges in the mesh
    /// Generated from class `MR::UndirectedEdgeIterator`.
    /// This is the const half of the class.
    public class Const_UndirectedEdgeIterator : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_UndirectedEdgeIterator>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_UndirectedEdgeIterator_Destroy(_Underlying *_this);
            __MR_UndirectedEdgeIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UndirectedEdgeIterator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UndirectedEdgeIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeIterator_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public unsafe Const_UndirectedEdgeIterator(MR.Const_UndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_ConstructFromAnother(MR.UndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public unsafe Const_UndirectedEdgeIterator(MR.Const_MeshTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_Construct(MR.Const_MeshTopology._Underlying *topology);
            _UnderlyingPtr = __MR_UndirectedEdgeIterator_Construct(topology._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public static unsafe implicit operator Const_UndirectedEdgeIterator(MR.Const_MeshTopology topology) {return new(topology);}

        /// Generated from method `MR::UndirectedEdgeIterator::operator++`.
        public static unsafe UndirectedEdgeIterator operator++(MR.Const_UndirectedEdgeIterator _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_UndirectedEdgeIterator", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_incr_MR_UndirectedEdgeIterator(MR.Const_UndirectedEdgeIterator._Underlying *_this);
            UndirectedEdgeIterator _this_copy = new(_this);
            MR.UndirectedEdgeIterator _unused_ret = new(__MR_incr_MR_UndirectedEdgeIterator(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::UndirectedEdgeIterator::operator*`.
        public unsafe MR.UndirectedEdgeId Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_UndirectedEdgeIterator", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_deref_MR_UndirectedEdgeIterator(_Underlying *_this);
            return __MR_deref_MR_UndirectedEdgeIterator(_UnderlyingPtr);
        }

        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_UndirectedEdgeIterator a, MR.Const_UndirectedEdgeIterator b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_UndirectedEdgeIterator", ExactSpelling = true)]
            extern static byte __MR_equal_MR_UndirectedEdgeIterator(MR.Const_UndirectedEdgeIterator._Underlying *a, MR.Const_UndirectedEdgeIterator._Underlying *b);
            return __MR_equal_MR_UndirectedEdgeIterator(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_UndirectedEdgeIterator a, MR.Const_UndirectedEdgeIterator b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_UndirectedEdgeIterator? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_UndirectedEdgeIterator)
                return this == (MR.Const_UndirectedEdgeIterator)other;
            return false;
        }
    }

    // The iterator to find all not-lone undirected edges in the mesh
    /// Generated from class `MR::UndirectedEdgeIterator`.
    /// This is the non-const half of the class.
    public class UndirectedEdgeIterator : Const_UndirectedEdgeIterator
    {
        internal unsafe UndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UndirectedEdgeIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirectedEdgeIterator_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public unsafe UndirectedEdgeIterator(MR.Const_UndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_ConstructFromAnother(MR.UndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_UndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public unsafe UndirectedEdgeIterator(MR.Const_MeshTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_Construct(MR.Const_MeshTopology._Underlying *topology);
            _UnderlyingPtr = __MR_UndirectedEdgeIterator_Construct(topology._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public static unsafe implicit operator UndirectedEdgeIterator(MR.Const_MeshTopology topology) {return new(topology);}

        /// Generated from method `MR::UndirectedEdgeIterator::operator=`.
        public unsafe MR.UndirectedEdgeIterator Assign(MR.Const_UndirectedEdgeIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirectedEdgeIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_UndirectedEdgeIterator_AssignFromAnother(_Underlying *_this, MR.UndirectedEdgeIterator._Underlying *_other);
            return new(__MR_UndirectedEdgeIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::UndirectedEdgeIterator::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_UndirectedEdgeIterator", ExactSpelling = true)]
            extern static MR.UndirectedEdgeIterator._Underlying *__MR_incr_MR_UndirectedEdgeIterator(_Underlying *_this);
            MR.UndirectedEdgeIterator _unused_ret = new(__MR_incr_MR_UndirectedEdgeIterator(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `UndirectedEdgeIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeIterator`/`Const_UndirectedEdgeIterator` directly.
    public class _InOptMut_UndirectedEdgeIterator
    {
        public UndirectedEdgeIterator? Opt;

        public _InOptMut_UndirectedEdgeIterator() {}
        public _InOptMut_UndirectedEdgeIterator(UndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptMut_UndirectedEdgeIterator(UndirectedEdgeIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `UndirectedEdgeIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirectedEdgeIterator`/`Const_UndirectedEdgeIterator` to pass it to the function.
    public class _InOptConst_UndirectedEdgeIterator
    {
        public Const_UndirectedEdgeIterator? Opt;

        public _InOptConst_UndirectedEdgeIterator() {}
        public _InOptConst_UndirectedEdgeIterator(Const_UndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptConst_UndirectedEdgeIterator(Const_UndirectedEdgeIterator value) {return new(value);}

        // creates begin iterator
        /// Generated from constructor `MR::UndirectedEdgeIterator::UndirectedEdgeIterator`.
        public static unsafe implicit operator _InOptConst_UndirectedEdgeIterator(MR.Const_MeshTopology topology) {return new MR.UndirectedEdgeIterator(topology);}
    }

    /// Generated from function `MR::undirectedEdges`.
    public static unsafe MR.IteratorRange_MRUndirectedEdgeIterator UndirectedEdges(MR.Const_MeshTopology topology)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_undirectedEdges_MR_MeshTopology", ExactSpelling = true)]
        extern static MR.IteratorRange_MRUndirectedEdgeIterator._Underlying *__MR_undirectedEdges_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology);
        return new(__MR_undirectedEdges_MR_MeshTopology(topology._UnderlyingPtr), is_owning: true);
    }
}
