public static partial class MR
{
    // The iterator to find all edges in a ring of edges (e.g. all edges with same origin or all edges with same left face)
    /// Generated from class `MR::RingIterator<MR::NextEdgeSameOrigin>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NextEdgeSameOrigin`
    /// This is the const half of the class.
    public class Const_RingIterator_MRNextEdgeSameOrigin : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_RingIterator_MRNextEdgeSameOrigin>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RingIterator_MRNextEdgeSameOrigin(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_Destroy", ExactSpelling = true)]
            extern static void __MR_RingIterator_MR_NextEdgeSameOrigin_Destroy(_Underlying *_this);
            __MR_RingIterator_MR_NextEdgeSameOrigin_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RingIterator_MRNextEdgeSameOrigin() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_NextEdgeSameOrigin(Const_RingIterator_MRNextEdgeSameOrigin self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_UpcastTo_MR_NextEdgeSameOrigin", ExactSpelling = true)]
            extern static MR.Const_NextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_UpcastTo_MR_NextEdgeSameOrigin(_Underlying *_this);
            MR.Const_NextEdgeSameOrigin ret = new(__MR_RingIterator_MR_NextEdgeSameOrigin_UpcastTo_MR_NextEdgeSameOrigin(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameOrigin>::RingIterator`.
        public unsafe Const_RingIterator_MRNextEdgeSameOrigin(MR.Const_RingIterator_MRNextEdgeSameOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(MR.RingIterator_MRNextEdgeSameOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameOrigin>::RingIterator`.
        public unsafe Const_RingIterator_MRNextEdgeSameOrigin(MR.Const_MeshTopology topology, MR.EdgeId edge, bool first) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_Construct", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_Construct(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge, byte first);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameOrigin_Construct(topology._UnderlyingPtr, edge, first ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameOrigin>::operator++`.
        public static unsafe RingIterator_MRNextEdgeSameOrigin operator++(MR.Const_RingIterator_MRNextEdgeSameOrigin _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RingIterator_MR_NextEdgeSameOrigin", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_incr_MR_RingIterator_MR_NextEdgeSameOrigin(MR.Const_RingIterator_MRNextEdgeSameOrigin._Underlying *_this);
            RingIterator_MRNextEdgeSameOrigin _this_copy = new(_this);
            MR.RingIterator_MRNextEdgeSameOrigin _unused_ret = new(__MR_incr_MR_RingIterator_MR_NextEdgeSameOrigin(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameOrigin>::operator*`.
        public unsafe MR.EdgeId Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_RingIterator_MR_NextEdgeSameOrigin", ExactSpelling = true)]
            extern static MR.EdgeId __MR_deref_MR_RingIterator_MR_NextEdgeSameOrigin(_Underlying *_this);
            return __MR_deref_MR_RingIterator_MR_NextEdgeSameOrigin(_UnderlyingPtr);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameOrigin>::first`.
        public unsafe bool First()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_first", ExactSpelling = true)]
            extern static byte __MR_RingIterator_MR_NextEdgeSameOrigin_first(_Underlying *_this);
            return __MR_RingIterator_MR_NextEdgeSameOrigin_first(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameOrigin>::next`.
        public unsafe MR.EdgeId Next(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_RingIterator_MR_NextEdgeSameOrigin_next(_Underlying *_this, MR.EdgeId e);
            return __MR_RingIterator_MR_NextEdgeSameOrigin_next(_UnderlyingPtr, e);
        }

        /// Generated from function `MR::operator==<MR::NextEdgeSameOrigin>`.
        public static unsafe bool operator==(MR.Const_RingIterator_MRNextEdgeSameOrigin a, MR.Const_RingIterator_MRNextEdgeSameOrigin b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_RingIterator_MR_NextEdgeSameOrigin", ExactSpelling = true)]
            extern static byte __MR_equal_MR_RingIterator_MR_NextEdgeSameOrigin(MR.Const_RingIterator_MRNextEdgeSameOrigin._Underlying *a, MR.Const_RingIterator_MRNextEdgeSameOrigin._Underlying *b);
            return __MR_equal_MR_RingIterator_MR_NextEdgeSameOrigin(a._UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_RingIterator_MRNextEdgeSameOrigin a, MR.Const_RingIterator_MRNextEdgeSameOrigin b)
        {
            return !(a == b);
        }

        // IEquatable:

        public bool Equals(MR.Const_RingIterator_MRNextEdgeSameOrigin? b)
        {
            if (b is null)
                return false;
            return this == b;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_RingIterator_MRNextEdgeSameOrigin)
                return this == (MR.Const_RingIterator_MRNextEdgeSameOrigin)other;
            return false;
        }
    }

    // The iterator to find all edges in a ring of edges (e.g. all edges with same origin or all edges with same left face)
    /// Generated from class `MR::RingIterator<MR::NextEdgeSameOrigin>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NextEdgeSameOrigin`
    /// This is the non-const half of the class.
    public class RingIterator_MRNextEdgeSameOrigin : Const_RingIterator_MRNextEdgeSameOrigin
    {
        internal unsafe RingIterator_MRNextEdgeSameOrigin(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.NextEdgeSameOrigin(RingIterator_MRNextEdgeSameOrigin self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_UpcastTo_MR_NextEdgeSameOrigin", ExactSpelling = true)]
            extern static MR.NextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_UpcastTo_MR_NextEdgeSameOrigin(_Underlying *_this);
            MR.NextEdgeSameOrigin ret = new(__MR_RingIterator_MR_NextEdgeSameOrigin_UpcastTo_MR_NextEdgeSameOrigin(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameOrigin>::RingIterator`.
        public unsafe RingIterator_MRNextEdgeSameOrigin(MR.Const_RingIterator_MRNextEdgeSameOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(MR.RingIterator_MRNextEdgeSameOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameOrigin>::RingIterator`.
        public unsafe RingIterator_MRNextEdgeSameOrigin(MR.Const_MeshTopology topology, MR.EdgeId edge, bool first) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_Construct", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_Construct(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge, byte first);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameOrigin_Construct(topology._UnderlyingPtr, edge, first ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameOrigin>::operator=`.
        public unsafe MR.RingIterator_MRNextEdgeSameOrigin Assign(MR.Const_RingIterator_MRNextEdgeSameOrigin _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameOrigin_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_RingIterator_MR_NextEdgeSameOrigin_AssignFromAnother(_Underlying *_this, MR.RingIterator_MRNextEdgeSameOrigin._Underlying *_other);
            return new(__MR_RingIterator_MR_NextEdgeSameOrigin_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameOrigin>::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RingIterator_MR_NextEdgeSameOrigin", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameOrigin._Underlying *__MR_incr_MR_RingIterator_MR_NextEdgeSameOrigin(_Underlying *_this);
            MR.RingIterator_MRNextEdgeSameOrigin _unused_ret = new(__MR_incr_MR_RingIterator_MR_NextEdgeSameOrigin(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RingIterator_MRNextEdgeSameOrigin` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RingIterator_MRNextEdgeSameOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RingIterator_MRNextEdgeSameOrigin`/`Const_RingIterator_MRNextEdgeSameOrigin` directly.
    public class _InOptMut_RingIterator_MRNextEdgeSameOrigin
    {
        public RingIterator_MRNextEdgeSameOrigin? Opt;

        public _InOptMut_RingIterator_MRNextEdgeSameOrigin() {}
        public _InOptMut_RingIterator_MRNextEdgeSameOrigin(RingIterator_MRNextEdgeSameOrigin value) {Opt = value;}
        public static implicit operator _InOptMut_RingIterator_MRNextEdgeSameOrigin(RingIterator_MRNextEdgeSameOrigin value) {return new(value);}
    }

    /// This is used for optional parameters of class `RingIterator_MRNextEdgeSameOrigin` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RingIterator_MRNextEdgeSameOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RingIterator_MRNextEdgeSameOrigin`/`Const_RingIterator_MRNextEdgeSameOrigin` to pass it to the function.
    public class _InOptConst_RingIterator_MRNextEdgeSameOrigin
    {
        public Const_RingIterator_MRNextEdgeSameOrigin? Opt;

        public _InOptConst_RingIterator_MRNextEdgeSameOrigin() {}
        public _InOptConst_RingIterator_MRNextEdgeSameOrigin(Const_RingIterator_MRNextEdgeSameOrigin value) {Opt = value;}
        public static implicit operator _InOptConst_RingIterator_MRNextEdgeSameOrigin(Const_RingIterator_MRNextEdgeSameOrigin value) {return new(value);}
    }

    // The iterator to find all edges in a ring of edges (e.g. all edges with same origin or all edges with same left face)
    /// Generated from class `MR::RingIterator<MR::NextEdgeSameLeft>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NextEdgeSameLeft`
    /// This is the const half of the class.
    public class Const_RingIterator_MRNextEdgeSameLeft : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RingIterator_MRNextEdgeSameLeft(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_Destroy", ExactSpelling = true)]
            extern static void __MR_RingIterator_MR_NextEdgeSameLeft_Destroy(_Underlying *_this);
            __MR_RingIterator_MR_NextEdgeSameLeft_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RingIterator_MRNextEdgeSameLeft() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_NextEdgeSameLeft(Const_RingIterator_MRNextEdgeSameLeft self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_UpcastTo_MR_NextEdgeSameLeft", ExactSpelling = true)]
            extern static MR.Const_NextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_UpcastTo_MR_NextEdgeSameLeft(_Underlying *_this);
            MR.Const_NextEdgeSameLeft ret = new(__MR_RingIterator_MR_NextEdgeSameLeft_UpcastTo_MR_NextEdgeSameLeft(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameLeft>::RingIterator`.
        public unsafe Const_RingIterator_MRNextEdgeSameLeft(MR.Const_RingIterator_MRNextEdgeSameLeft _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(MR.RingIterator_MRNextEdgeSameLeft._Underlying *_other);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameLeft>::RingIterator`.
        public unsafe Const_RingIterator_MRNextEdgeSameLeft(MR.Const_MeshTopology topology, MR.EdgeId edge, bool first) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_Construct", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_Construct(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge, byte first);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameLeft_Construct(topology._UnderlyingPtr, edge, first ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameLeft>::operator++`.
        public static unsafe RingIterator_MRNextEdgeSameLeft operator++(MR.Const_RingIterator_MRNextEdgeSameLeft _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RingIterator_MR_NextEdgeSameLeft", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_incr_MR_RingIterator_MR_NextEdgeSameLeft(MR.Const_RingIterator_MRNextEdgeSameLeft._Underlying *_this);
            RingIterator_MRNextEdgeSameLeft _this_copy = new(_this);
            MR.RingIterator_MRNextEdgeSameLeft _unused_ret = new(__MR_incr_MR_RingIterator_MR_NextEdgeSameLeft(_this_copy._UnderlyingPtr), is_owning: false);
            return _this_copy;
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameLeft>::operator*`.
        public unsafe MR.EdgeId Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_RingIterator_MR_NextEdgeSameLeft", ExactSpelling = true)]
            extern static MR.EdgeId __MR_deref_MR_RingIterator_MR_NextEdgeSameLeft(_Underlying *_this);
            return __MR_deref_MR_RingIterator_MR_NextEdgeSameLeft(_UnderlyingPtr);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameLeft>::first`.
        public unsafe bool First()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_first", ExactSpelling = true)]
            extern static byte __MR_RingIterator_MR_NextEdgeSameLeft_first(_Underlying *_this);
            return __MR_RingIterator_MR_NextEdgeSameLeft_first(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameLeft>::next`.
        public unsafe MR.EdgeId Next(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_RingIterator_MR_NextEdgeSameLeft_next(_Underlying *_this, MR.EdgeId e);
            return __MR_RingIterator_MR_NextEdgeSameLeft_next(_UnderlyingPtr, e);
        }
    }

    // The iterator to find all edges in a ring of edges (e.g. all edges with same origin or all edges with same left face)
    /// Generated from class `MR::RingIterator<MR::NextEdgeSameLeft>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::NextEdgeSameLeft`
    /// This is the non-const half of the class.
    public class RingIterator_MRNextEdgeSameLeft : Const_RingIterator_MRNextEdgeSameLeft
    {
        internal unsafe RingIterator_MRNextEdgeSameLeft(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.NextEdgeSameLeft(RingIterator_MRNextEdgeSameLeft self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_UpcastTo_MR_NextEdgeSameLeft", ExactSpelling = true)]
            extern static MR.NextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_UpcastTo_MR_NextEdgeSameLeft(_Underlying *_this);
            MR.NextEdgeSameLeft ret = new(__MR_RingIterator_MR_NextEdgeSameLeft_UpcastTo_MR_NextEdgeSameLeft(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameLeft>::RingIterator`.
        public unsafe RingIterator_MRNextEdgeSameLeft(MR.Const_RingIterator_MRNextEdgeSameLeft _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(MR.RingIterator_MRNextEdgeSameLeft._Underlying *_other);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameLeft_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RingIterator<MR::NextEdgeSameLeft>::RingIterator`.
        public unsafe RingIterator_MRNextEdgeSameLeft(MR.Const_MeshTopology topology, MR.EdgeId edge, bool first) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_Construct", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_Construct(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge, byte first);
            _UnderlyingPtr = __MR_RingIterator_MR_NextEdgeSameLeft_Construct(topology._UnderlyingPtr, edge, first ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameLeft>::operator=`.
        public unsafe MR.RingIterator_MRNextEdgeSameLeft Assign(MR.Const_RingIterator_MRNextEdgeSameLeft _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RingIterator_MR_NextEdgeSameLeft_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_RingIterator_MR_NextEdgeSameLeft_AssignFromAnother(_Underlying *_this, MR.RingIterator_MRNextEdgeSameLeft._Underlying *_other);
            return new(__MR_RingIterator_MR_NextEdgeSameLeft_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::RingIterator<MR::NextEdgeSameLeft>::operator++`.
        public unsafe void Incr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_incr_MR_RingIterator_MR_NextEdgeSameLeft", ExactSpelling = true)]
            extern static MR.RingIterator_MRNextEdgeSameLeft._Underlying *__MR_incr_MR_RingIterator_MR_NextEdgeSameLeft(_Underlying *_this);
            MR.RingIterator_MRNextEdgeSameLeft _unused_ret = new(__MR_incr_MR_RingIterator_MR_NextEdgeSameLeft(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RingIterator_MRNextEdgeSameLeft` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RingIterator_MRNextEdgeSameLeft`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RingIterator_MRNextEdgeSameLeft`/`Const_RingIterator_MRNextEdgeSameLeft` directly.
    public class _InOptMut_RingIterator_MRNextEdgeSameLeft
    {
        public RingIterator_MRNextEdgeSameLeft? Opt;

        public _InOptMut_RingIterator_MRNextEdgeSameLeft() {}
        public _InOptMut_RingIterator_MRNextEdgeSameLeft(RingIterator_MRNextEdgeSameLeft value) {Opt = value;}
        public static implicit operator _InOptMut_RingIterator_MRNextEdgeSameLeft(RingIterator_MRNextEdgeSameLeft value) {return new(value);}
    }

    /// This is used for optional parameters of class `RingIterator_MRNextEdgeSameLeft` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RingIterator_MRNextEdgeSameLeft`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RingIterator_MRNextEdgeSameLeft`/`Const_RingIterator_MRNextEdgeSameLeft` to pass it to the function.
    public class _InOptConst_RingIterator_MRNextEdgeSameLeft
    {
        public Const_RingIterator_MRNextEdgeSameLeft? Opt;

        public _InOptConst_RingIterator_MRNextEdgeSameLeft() {}
        public _InOptConst_RingIterator_MRNextEdgeSameLeft(Const_RingIterator_MRNextEdgeSameLeft value) {Opt = value;}
        public static implicit operator _InOptConst_RingIterator_MRNextEdgeSameLeft(Const_RingIterator_MRNextEdgeSameLeft value) {return new(value);}
    }

    /// Generated from class `MR::NextEdgeSameOrigin`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::RingIterator<MR::NextEdgeSameOrigin>`
    /// This is the const half of the class.
    public class Const_NextEdgeSameOrigin : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NextEdgeSameOrigin(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_Destroy", ExactSpelling = true)]
            extern static void __MR_NextEdgeSameOrigin_Destroy(_Underlying *_this);
            __MR_NextEdgeSameOrigin_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NextEdgeSameOrigin() {Dispose(false);}

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public unsafe Const_NextEdgeSameOrigin(MR.Const_NextEdgeSameOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NextEdgeSameOrigin._Underlying *__MR_NextEdgeSameOrigin_ConstructFromAnother(MR.NextEdgeSameOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_NextEdgeSameOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public unsafe Const_NextEdgeSameOrigin(MR.Const_MeshTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_Construct", ExactSpelling = true)]
            extern static MR.NextEdgeSameOrigin._Underlying *__MR_NextEdgeSameOrigin_Construct(MR.Const_MeshTopology._Underlying *topology);
            _UnderlyingPtr = __MR_NextEdgeSameOrigin_Construct(topology._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public static unsafe implicit operator Const_NextEdgeSameOrigin(MR.Const_MeshTopology topology) {return new(topology);}

        /// Generated from method `MR::NextEdgeSameOrigin::next`.
        public unsafe MR.EdgeId Next(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_NextEdgeSameOrigin_next(_Underlying *_this, MR.EdgeId e);
            return __MR_NextEdgeSameOrigin_next(_UnderlyingPtr, e);
        }
    }

    /// Generated from class `MR::NextEdgeSameOrigin`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::RingIterator<MR::NextEdgeSameOrigin>`
    /// This is the non-const half of the class.
    public class NextEdgeSameOrigin : Const_NextEdgeSameOrigin
    {
        internal unsafe NextEdgeSameOrigin(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public unsafe NextEdgeSameOrigin(MR.Const_NextEdgeSameOrigin _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NextEdgeSameOrigin._Underlying *__MR_NextEdgeSameOrigin_ConstructFromAnother(MR.NextEdgeSameOrigin._Underlying *_other);
            _UnderlyingPtr = __MR_NextEdgeSameOrigin_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public unsafe NextEdgeSameOrigin(MR.Const_MeshTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_Construct", ExactSpelling = true)]
            extern static MR.NextEdgeSameOrigin._Underlying *__MR_NextEdgeSameOrigin_Construct(MR.Const_MeshTopology._Underlying *topology);
            _UnderlyingPtr = __MR_NextEdgeSameOrigin_Construct(topology._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public static unsafe implicit operator NextEdgeSameOrigin(MR.Const_MeshTopology topology) {return new(topology);}

        /// Generated from method `MR::NextEdgeSameOrigin::operator=`.
        public unsafe MR.NextEdgeSameOrigin Assign(MR.Const_NextEdgeSameOrigin _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameOrigin_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NextEdgeSameOrigin._Underlying *__MR_NextEdgeSameOrigin_AssignFromAnother(_Underlying *_this, MR.NextEdgeSameOrigin._Underlying *_other);
            return new(__MR_NextEdgeSameOrigin_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NextEdgeSameOrigin` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NextEdgeSameOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NextEdgeSameOrigin`/`Const_NextEdgeSameOrigin` directly.
    public class _InOptMut_NextEdgeSameOrigin
    {
        public NextEdgeSameOrigin? Opt;

        public _InOptMut_NextEdgeSameOrigin() {}
        public _InOptMut_NextEdgeSameOrigin(NextEdgeSameOrigin value) {Opt = value;}
        public static implicit operator _InOptMut_NextEdgeSameOrigin(NextEdgeSameOrigin value) {return new(value);}
    }

    /// This is used for optional parameters of class `NextEdgeSameOrigin` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NextEdgeSameOrigin`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NextEdgeSameOrigin`/`Const_NextEdgeSameOrigin` to pass it to the function.
    public class _InOptConst_NextEdgeSameOrigin
    {
        public Const_NextEdgeSameOrigin? Opt;

        public _InOptConst_NextEdgeSameOrigin() {}
        public _InOptConst_NextEdgeSameOrigin(Const_NextEdgeSameOrigin value) {Opt = value;}
        public static implicit operator _InOptConst_NextEdgeSameOrigin(Const_NextEdgeSameOrigin value) {return new(value);}

        /// Generated from constructor `MR::NextEdgeSameOrigin::NextEdgeSameOrigin`.
        public static unsafe implicit operator _InOptConst_NextEdgeSameOrigin(MR.Const_MeshTopology topology) {return new MR.NextEdgeSameOrigin(topology);}
    }

    /// Generated from class `MR::NextEdgeSameLeft`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::RingIterator<MR::NextEdgeSameLeft>`
    /// This is the const half of the class.
    public class Const_NextEdgeSameLeft : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NextEdgeSameLeft(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_Destroy", ExactSpelling = true)]
            extern static void __MR_NextEdgeSameLeft_Destroy(_Underlying *_this);
            __MR_NextEdgeSameLeft_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NextEdgeSameLeft() {Dispose(false);}

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public unsafe Const_NextEdgeSameLeft(MR.Const_NextEdgeSameLeft _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NextEdgeSameLeft._Underlying *__MR_NextEdgeSameLeft_ConstructFromAnother(MR.NextEdgeSameLeft._Underlying *_other);
            _UnderlyingPtr = __MR_NextEdgeSameLeft_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public unsafe Const_NextEdgeSameLeft(MR.Const_MeshTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_Construct", ExactSpelling = true)]
            extern static MR.NextEdgeSameLeft._Underlying *__MR_NextEdgeSameLeft_Construct(MR.Const_MeshTopology._Underlying *topology);
            _UnderlyingPtr = __MR_NextEdgeSameLeft_Construct(topology._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public static unsafe implicit operator Const_NextEdgeSameLeft(MR.Const_MeshTopology topology) {return new(topology);}

        /// Generated from method `MR::NextEdgeSameLeft::next`.
        public unsafe MR.EdgeId Next(MR.EdgeId e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_next", ExactSpelling = true)]
            extern static MR.EdgeId __MR_NextEdgeSameLeft_next(_Underlying *_this, MR.EdgeId e);
            return __MR_NextEdgeSameLeft_next(_UnderlyingPtr, e);
        }
    }

    /// Generated from class `MR::NextEdgeSameLeft`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::RingIterator<MR::NextEdgeSameLeft>`
    /// This is the non-const half of the class.
    public class NextEdgeSameLeft : Const_NextEdgeSameLeft
    {
        internal unsafe NextEdgeSameLeft(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public unsafe NextEdgeSameLeft(MR.Const_NextEdgeSameLeft _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NextEdgeSameLeft._Underlying *__MR_NextEdgeSameLeft_ConstructFromAnother(MR.NextEdgeSameLeft._Underlying *_other);
            _UnderlyingPtr = __MR_NextEdgeSameLeft_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public unsafe NextEdgeSameLeft(MR.Const_MeshTopology topology) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_Construct", ExactSpelling = true)]
            extern static MR.NextEdgeSameLeft._Underlying *__MR_NextEdgeSameLeft_Construct(MR.Const_MeshTopology._Underlying *topology);
            _UnderlyingPtr = __MR_NextEdgeSameLeft_Construct(topology._UnderlyingPtr);
        }

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public static unsafe implicit operator NextEdgeSameLeft(MR.Const_MeshTopology topology) {return new(topology);}

        /// Generated from method `MR::NextEdgeSameLeft::operator=`.
        public unsafe MR.NextEdgeSameLeft Assign(MR.Const_NextEdgeSameLeft _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NextEdgeSameLeft_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NextEdgeSameLeft._Underlying *__MR_NextEdgeSameLeft_AssignFromAnother(_Underlying *_this, MR.NextEdgeSameLeft._Underlying *_other);
            return new(__MR_NextEdgeSameLeft_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `NextEdgeSameLeft` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NextEdgeSameLeft`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NextEdgeSameLeft`/`Const_NextEdgeSameLeft` directly.
    public class _InOptMut_NextEdgeSameLeft
    {
        public NextEdgeSameLeft? Opt;

        public _InOptMut_NextEdgeSameLeft() {}
        public _InOptMut_NextEdgeSameLeft(NextEdgeSameLeft value) {Opt = value;}
        public static implicit operator _InOptMut_NextEdgeSameLeft(NextEdgeSameLeft value) {return new(value);}
    }

    /// This is used for optional parameters of class `NextEdgeSameLeft` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NextEdgeSameLeft`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NextEdgeSameLeft`/`Const_NextEdgeSameLeft` to pass it to the function.
    public class _InOptConst_NextEdgeSameLeft
    {
        public Const_NextEdgeSameLeft? Opt;

        public _InOptConst_NextEdgeSameLeft() {}
        public _InOptConst_NextEdgeSameLeft(Const_NextEdgeSameLeft value) {Opt = value;}
        public static implicit operator _InOptConst_NextEdgeSameLeft(Const_NextEdgeSameLeft value) {return new(value);}

        /// Generated from constructor `MR::NextEdgeSameLeft::NextEdgeSameLeft`.
        public static unsafe implicit operator _InOptConst_NextEdgeSameLeft(MR.Const_MeshTopology topology) {return new MR.NextEdgeSameLeft(topology);}
    }

    // to iterate over all edges with same origin vertex  as firstEdge (INCLUDING firstEdge)
    // for ( Edge e : orgRing( topology, firstEdge ) ) ...
    /// Generated from function `MR::orgRing`.
    public static unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin OrgRing(MR.Const_MeshTopology topology, MR.EdgeId edge)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orgRing_MR_EdgeId", ExactSpelling = true)]
        extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_orgRing_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge);
        return new(__MR_orgRing_MR_EdgeId(topology._UnderlyingPtr, edge), is_owning: true);
    }

    /// Generated from function `MR::orgRing`.
    public static unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin OrgRing(MR.Const_MeshTopology topology, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orgRing_MR_VertId", ExactSpelling = true)]
        extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_orgRing_MR_VertId(MR.Const_MeshTopology._Underlying *topology, MR.VertId v);
        return new(__MR_orgRing_MR_VertId(topology._UnderlyingPtr, v), is_owning: true);
    }

    // to iterate over all edges with same origin vertex as firstEdge (EXCLUDING firstEdge)
    // for ( Edge e : orgRing0( topology, firstEdge ) ) ...
    /// Generated from function `MR::orgRing0`.
    public static unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin OrgRing0(MR.Const_MeshTopology topology, MR.EdgeId edge)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_orgRing0", ExactSpelling = true)]
        extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameOrigin._Underlying *__MR_orgRing0(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge);
        return new(__MR_orgRing0(topology._UnderlyingPtr, edge), is_owning: true);
    }

    // to iterate over all edges with same left face as firstEdge (INCLUDING firstEdge)
    // for ( Edge e : leftRing( topology, firstEdge ) ) ...
    /// Generated from function `MR::leftRing`.
    public static unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft LeftRing(MR.Const_MeshTopology topology, MR.EdgeId edge)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftRing_MR_EdgeId", ExactSpelling = true)]
        extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_leftRing_MR_EdgeId(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge);
        return new(__MR_leftRing_MR_EdgeId(topology._UnderlyingPtr, edge), is_owning: true);
    }

    /// Generated from function `MR::leftRing`.
    public static unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft LeftRing(MR.Const_MeshTopology topology, MR.FaceId f)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftRing_MR_FaceId", ExactSpelling = true)]
        extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_leftRing_MR_FaceId(MR.Const_MeshTopology._Underlying *topology, MR.FaceId f);
        return new(__MR_leftRing_MR_FaceId(topology._UnderlyingPtr, f), is_owning: true);
    }

    // to iterate over all edges with same left face as firstEdge (EXCLUDING firstEdge)
    // for ( Edge e : leftRing0( topology, firstEdge ) ) ...
    /// Generated from function `MR::leftRing0`.
    public static unsafe MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft LeftRing0(MR.Const_MeshTopology topology, MR.EdgeId edge)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_leftRing0", ExactSpelling = true)]
        extern static MR.IteratorRange_MRRingIteratorMRNextEdgeSameLeft._Underlying *__MR_leftRing0(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId edge);
        return new(__MR_leftRing0(topology._UnderlyingPtr, edge), is_owning: true);
    }
}
