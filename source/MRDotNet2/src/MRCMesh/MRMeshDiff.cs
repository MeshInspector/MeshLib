public static partial class MR
{
    /// this object stores a difference between two meshes: both in coordinates and in topology
    /// \details if the meshes are similar then this object is small, if the meshes are very distinct then this object will be even larger than one mesh itself
    /// Generated from class `MR::MeshDiff`.
    /// This is the const half of the class.
    public class Const_MeshDiff : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshDiff(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshDiff_Destroy(_Underlying *_this);
            __MR_MeshDiff_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshDiff() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshDiff() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshDiff_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshDiff::MeshDiff`.
        public unsafe Const_MeshDiff(MR._ByValue_MeshDiff _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshDiff._Underlying *_other);
            _UnderlyingPtr = __MR_MeshDiff_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the difference, that can be applied to mesh-from in order to get mesh-to
        /// Generated from constructor `MR::MeshDiff::MeshDiff`.
        public unsafe Const_MeshDiff(MR.Const_Mesh from, MR.Const_Mesh to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_Construct", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_Construct(MR.Const_Mesh._Underlying *from, MR.Const_Mesh._Underlying *to);
            _UnderlyingPtr = __MR_MeshDiff_Construct(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// returns true if this object does contain some difference in point coordinates or in topology;
        /// if (from) mesh has just more points or more topology elements than (to) and the common elements are the same,
        /// then the method will return false since nothing is stored here
        /// Generated from method `MR::MeshDiff::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_any", ExactSpelling = true)]
            extern static byte __MR_MeshDiff_any(_Underlying *_this);
            return __MR_MeshDiff_any(_UnderlyingPtr) != 0;
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::MeshDiff::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_MeshDiff_heapBytes(_Underlying *_this);
            return __MR_MeshDiff_heapBytes(_UnderlyingPtr);
        }
    }

    /// this object stores a difference between two meshes: both in coordinates and in topology
    /// \details if the meshes are similar then this object is small, if the meshes are very distinct then this object will be even larger than one mesh itself
    /// Generated from class `MR::MeshDiff`.
    /// This is the non-const half of the class.
    public class MeshDiff : Const_MeshDiff
    {
        internal unsafe MeshDiff(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshDiff() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshDiff_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshDiff::MeshDiff`.
        public unsafe MeshDiff(MR._ByValue_MeshDiff _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshDiff._Underlying *_other);
            _UnderlyingPtr = __MR_MeshDiff_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the difference, that can be applied to mesh-from in order to get mesh-to
        /// Generated from constructor `MR::MeshDiff::MeshDiff`.
        public unsafe MeshDiff(MR.Const_Mesh from, MR.Const_Mesh to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_Construct", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_Construct(MR.Const_Mesh._Underlying *from, MR.Const_Mesh._Underlying *to);
            _UnderlyingPtr = __MR_MeshDiff_Construct(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshDiff::operator=`.
        public unsafe MR.MeshDiff Assign(MR._ByValue_MeshDiff _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshDiff._Underlying *__MR_MeshDiff_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshDiff._Underlying *_other);
            return new(__MR_MeshDiff_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// given mesh-from on input converts it in mesh-to,
        /// this object is updated to become the reverse difference from original mesh-to to original mesh-from
        /// Generated from method `MR::MeshDiff::applyAndSwap`.
        public unsafe void ApplyAndSwap(MR.Mesh m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshDiff_applyAndSwap", ExactSpelling = true)]
            extern static void __MR_MeshDiff_applyAndSwap(_Underlying *_this, MR.Mesh._Underlying *m);
            __MR_MeshDiff_applyAndSwap(_UnderlyingPtr, m._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshDiff` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshDiff`/`Const_MeshDiff` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshDiff
    {
        internal readonly Const_MeshDiff? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshDiff() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshDiff(Const_MeshDiff new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshDiff(Const_MeshDiff arg) {return new(arg);}
        public _ByValue_MeshDiff(MR.Misc._Moved<MeshDiff> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshDiff(MR.Misc._Moved<MeshDiff> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshDiff` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshDiff`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshDiff`/`Const_MeshDiff` directly.
    public class _InOptMut_MeshDiff
    {
        public MeshDiff? Opt;

        public _InOptMut_MeshDiff() {}
        public _InOptMut_MeshDiff(MeshDiff value) {Opt = value;}
        public static implicit operator _InOptMut_MeshDiff(MeshDiff value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshDiff` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshDiff`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshDiff`/`Const_MeshDiff` to pass it to the function.
    public class _InOptConst_MeshDiff
    {
        public Const_MeshDiff? Opt;

        public _InOptConst_MeshDiff() {}
        public _InOptConst_MeshDiff(Const_MeshDiff value) {Opt = value;}
        public static implicit operator _InOptConst_MeshDiff(Const_MeshDiff value) {return new(value);}
    }
}
