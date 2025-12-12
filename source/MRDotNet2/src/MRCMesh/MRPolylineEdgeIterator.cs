public static partial class MR
{
    /// The iterator to find all not-lone undirected edges in the polyline topology
    /// Generated from class `MR::PolylineUndirectedEdgeIterator`.
    /// This is the const half of the class.
    public class Const_PolylineUndirectedEdgeIterator : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_PolylineUndirectedEdgeIterator>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineUndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineUndirectedEdgeIterator_Destroy(_Underlying *_this);
            __MR_PolylineUndirectedEdgeIterator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineUndirectedEdgeIterator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineUndirectedEdgeIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineUndirectedEdgeIterator_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public unsafe Const_PolylineUndirectedEdgeIterator(MR.Const_PolylineUndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(MR.PolylineUndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public unsafe Const_PolylineUndirectedEdgeIterator(MR.Const_PolylineTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_Construct(MR.Const_PolylineTopology._Underlying *topology);
            _UnderlyingPtr = __MR_PolylineUndirectedEdgeIterator_Construct(topology._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public static unsafe implicit operator Const_PolylineUndirectedEdgeIterator(MR.Const_PolylineTopology topology) {return new(topology);}

        /// Generated from method `MR::PolylineUndirectedEdgeIterator::operator++`.
        public static unsafe PolylineUndirectedEdgeIterator operator++(MR.Const_PolylineUndirectedEdgeIterator _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_PolylineUndirectedEdgeIterator", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_incr_MR_PolylineUndirectedEdgeIterator(MR.Const_PolylineUndirectedEdgeIterator._Underlying *_this);
            PolylineUndirectedEdgeIterator _this_copy = new(_this);
            MR.PolylineUndirectedEdgeIterator _unused_ret = new(__MR_incr_MR_PolylineUndirectedEdgeIterator(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::PolylineUndirectedEdgeIterator::operator*`.
        public unsafe MR.UndirectedEdgeId Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_PolylineUndirectedEdgeIterator", ExactSpelling = true)]
            extern static MR.UndirectedEdgeId __MR_deref_MR_PolylineUndirectedEdgeIterator(_Underlying *_this);
            return __MR_deref_MR_PolylineUndirectedEdgeIterator(_UnderlyingPtr);
        }

        /// \related PolylineUndirectedEdgeIterator
        /// Generated from function `MR::operator==`.
        public static unsafe bool operator==(MR.Const_PolylineUndirectedEdgeIterator a, MR.Const_PolylineUndirectedEdgeIterator b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_PolylineUndirectedEdgeIterator", ExactSpelling = true)]
            extern static byte __MR_equal_MR_PolylineUndirectedEdgeIterator(MR.Const_PolylineUndirectedEdgeIterator._Underlying *a, MR.Const_PolylineUndirectedEdgeIterator._Underlying *b);
            return __MR_equal_MR_PolylineUndirectedEdgeIterator(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_PolylineUndirectedEdgeIterator a, MR.Const_PolylineUndirectedEdgeIterator b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_PolylineUndirectedEdgeIterator? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_PolylineUndirectedEdgeIterator)
                return this == (MR.Const_PolylineUndirectedEdgeIterator)other;
            return false;
        }
    }

    /// The iterator to find all not-lone undirected edges in the polyline topology
    /// Generated from class `MR::PolylineUndirectedEdgeIterator`.
    /// This is the non-const half of the class.
    public class PolylineUndirectedEdgeIterator : Const_PolylineUndirectedEdgeIterator
    {
        internal unsafe PolylineUndirectedEdgeIterator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineUndirectedEdgeIterator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineUndirectedEdgeIterator_DefaultConstruct();
        }

        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public unsafe PolylineUndirectedEdgeIterator(MR.Const_PolylineUndirectedEdgeIterator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(MR.PolylineUndirectedEdgeIterator._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineUndirectedEdgeIterator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public unsafe PolylineUndirectedEdgeIterator(MR.Const_PolylineTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_Construct", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_Construct(MR.Const_PolylineTopology._Underlying *topology);
            _UnderlyingPtr = __MR_PolylineUndirectedEdgeIterator_Construct(topology._UnderlyingPtr);
        }

        // creates begin iterator
        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public static unsafe implicit operator PolylineUndirectedEdgeIterator(MR.Const_PolylineTopology topology) {return new(topology);}

        /// Generated from method `MR::PolylineUndirectedEdgeIterator::operator=`.
        public unsafe MR.PolylineUndirectedEdgeIterator Assign(MR.Const_PolylineUndirectedEdgeIterator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineUndirectedEdgeIterator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_PolylineUndirectedEdgeIterator_AssignFromAnother(_Underlying *_this, MR.PolylineUndirectedEdgeIterator._Underlying *_other);
            return new(__MR_PolylineUndirectedEdgeIterator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PolylineUndirectedEdgeIterator::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_PolylineUndirectedEdgeIterator", ExactSpelling = true)]
            extern static MR.PolylineUndirectedEdgeIterator._Underlying *__MR_incr_MR_PolylineUndirectedEdgeIterator(_Underlying *_this);
            MR.PolylineUndirectedEdgeIterator _unused_ret = new(__MR_incr_MR_PolylineUndirectedEdgeIterator(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineUndirectedEdgeIterator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineUndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineUndirectedEdgeIterator`/`Const_PolylineUndirectedEdgeIterator` directly.
    public class _InOptMut_PolylineUndirectedEdgeIterator
    {
        public PolylineUndirectedEdgeIterator? Opt;

        public _InOptMut_PolylineUndirectedEdgeIterator() {}
        public _InOptMut_PolylineUndirectedEdgeIterator(PolylineUndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineUndirectedEdgeIterator(PolylineUndirectedEdgeIterator value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineUndirectedEdgeIterator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineUndirectedEdgeIterator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineUndirectedEdgeIterator`/`Const_PolylineUndirectedEdgeIterator` to pass it to the function.
    public class _InOptConst_PolylineUndirectedEdgeIterator
    {
        public Const_PolylineUndirectedEdgeIterator? Opt;

        public _InOptConst_PolylineUndirectedEdgeIterator() {}
        public _InOptConst_PolylineUndirectedEdgeIterator(Const_PolylineUndirectedEdgeIterator value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineUndirectedEdgeIterator(Const_PolylineUndirectedEdgeIterator value) {return new(value);}

        // creates begin iterator
        /// Generated from constructor `MR::PolylineUndirectedEdgeIterator::PolylineUndirectedEdgeIterator`.
        public static unsafe implicit operator _InOptConst_PolylineUndirectedEdgeIterator(MR.Const_PolylineTopology topology) {return new MR.PolylineUndirectedEdgeIterator(topology);}
    }

    /// Generated from function `MR::undirectedEdges`.
    public static unsafe MR.IteratorRange_MRPolylineUndirectedEdgeIterator UndirectedEdges(MR.Const_PolylineTopology topology)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_undirectedEdges_MR_PolylineTopology", ExactSpelling = true)]
        extern static MR.IteratorRange_MRPolylineUndirectedEdgeIterator._Underlying *__MR_undirectedEdges_MR_PolylineTopology(MR.Const_PolylineTopology._Underlying *topology);
        return new(__MR_undirectedEdges_MR_PolylineTopology(topology._UnderlyingPtr), is_owning: true);
    }
}
