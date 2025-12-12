public static partial class MR
{
    /// this object stores a difference between two topologies: both in coordinates and in topology
    /// \details if the topologies are similar then this object is small, if the topologies are very distinct then this object will be even larger than one topology itself
    /// Generated from class `MR::MeshTopologyDiff`.
    /// This is the const half of the class.
    public class Const_MeshTopologyDiff : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshTopologyDiff(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshTopologyDiff_Destroy(_Underlying *_this);
            __MR_MeshTopologyDiff_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshTopologyDiff() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshTopologyDiff() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTopologyDiff_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTopologyDiff::MeshTopologyDiff`.
        public unsafe Const_MeshTopologyDiff(MR._ByValue_MeshTopologyDiff _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshTopologyDiff._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTopologyDiff_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the difference, that can be applied to topology-from in order to get topology-to
        /// Generated from constructor `MR::MeshTopologyDiff::MeshTopologyDiff`.
        public unsafe Const_MeshTopologyDiff(MR.Const_MeshTopology from, MR.Const_MeshTopology to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_Construct", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_Construct(MR.Const_MeshTopology._Underlying *from, MR.Const_MeshTopology._Underlying *to);
            _UnderlyingPtr = __MR_MeshTopologyDiff_Construct(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// returns true if this object does contain some difference in topology;
        /// if (from) has more topology elements than (to) and the common elements are the same,
        /// then the method will return false since nothing is stored here
        /// Generated from method `MR::MeshTopologyDiff::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_any", ExactSpelling = true)]
            extern static byte __MR_MeshTopologyDiff_any(_Underlying *_this);
            return __MR_MeshTopologyDiff_any(_UnderlyingPtr) != 0;
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::MeshTopologyDiff::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_MeshTopologyDiff_heapBytes(_Underlying *_this);
            return __MR_MeshTopologyDiff_heapBytes(_UnderlyingPtr);
        }
    }

    /// this object stores a difference between two topologies: both in coordinates and in topology
    /// \details if the topologies are similar then this object is small, if the topologies are very distinct then this object will be even larger than one topology itself
    /// Generated from class `MR::MeshTopologyDiff`.
    /// This is the non-const half of the class.
    public class MeshTopologyDiff : Const_MeshTopologyDiff
    {
        internal unsafe MeshTopologyDiff(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshTopologyDiff() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTopologyDiff_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTopologyDiff::MeshTopologyDiff`.
        public unsafe MeshTopologyDiff(MR._ByValue_MeshTopologyDiff _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshTopologyDiff._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTopologyDiff_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the difference, that can be applied to topology-from in order to get topology-to
        /// Generated from constructor `MR::MeshTopologyDiff::MeshTopologyDiff`.
        public unsafe MeshTopologyDiff(MR.Const_MeshTopology from, MR.Const_MeshTopology to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_Construct", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_Construct(MR.Const_MeshTopology._Underlying *from, MR.Const_MeshTopology._Underlying *to);
            _UnderlyingPtr = __MR_MeshTopologyDiff_Construct(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshTopologyDiff::operator=`.
        public unsafe MR.MeshTopologyDiff Assign(MR._ByValue_MeshTopologyDiff _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshTopologyDiff._Underlying *__MR_MeshTopologyDiff_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshTopologyDiff._Underlying *_other);
            return new(__MR_MeshTopologyDiff_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// given topology-from on input converts it in topology-to,
        /// this object is updated to become the reverse difference from original topology-to to original topology-from
        /// Generated from method `MR::MeshTopologyDiff::applyAndSwap`.
        public unsafe void ApplyAndSwap(MR.MeshTopology t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTopologyDiff_applyAndSwap", ExactSpelling = true)]
            extern static void __MR_MeshTopologyDiff_applyAndSwap(_Underlying *_this, MR.MeshTopology._Underlying *t);
            __MR_MeshTopologyDiff_applyAndSwap(_UnderlyingPtr, t._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshTopologyDiff` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshTopologyDiff`/`Const_MeshTopologyDiff` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshTopologyDiff
    {
        internal readonly Const_MeshTopologyDiff? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshTopologyDiff() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshTopologyDiff(Const_MeshTopologyDiff new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshTopologyDiff(Const_MeshTopologyDiff arg) {return new(arg);}
        public _ByValue_MeshTopologyDiff(MR.Misc._Moved<MeshTopologyDiff> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshTopologyDiff(MR.Misc._Moved<MeshTopologyDiff> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshTopologyDiff` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshTopologyDiff`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTopologyDiff`/`Const_MeshTopologyDiff` directly.
    public class _InOptMut_MeshTopologyDiff
    {
        public MeshTopologyDiff? Opt;

        public _InOptMut_MeshTopologyDiff() {}
        public _InOptMut_MeshTopologyDiff(MeshTopologyDiff value) {Opt = value;}
        public static implicit operator _InOptMut_MeshTopologyDiff(MeshTopologyDiff value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshTopologyDiff` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshTopologyDiff`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTopologyDiff`/`Const_MeshTopologyDiff` to pass it to the function.
    public class _InOptConst_MeshTopologyDiff
    {
        public Const_MeshTopologyDiff? Opt;

        public _InOptConst_MeshTopologyDiff() {}
        public _InOptConst_MeshTopologyDiff(Const_MeshTopologyDiff value) {Opt = value;}
        public static implicit operator _InOptConst_MeshTopologyDiff(Const_MeshTopologyDiff value) {return new(value);}
    }
}
