public static partial class MR
{
    /// this object stores a difference between two vectors with 3D coordinates
    /// \details if the vectors are similar then this object is small, if the vectors are very distinct then this object will be even larger than one vector itself
    /// Generated from class `MR::VertCoordsDiff`.
    /// This is the const half of the class.
    public class Const_VertCoordsDiff : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertCoordsDiff(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_Destroy", ExactSpelling = true)]
            extern static void __MR_VertCoordsDiff_Destroy(_Underlying *_this);
            __MR_VertCoordsDiff_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertCoordsDiff() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertCoordsDiff() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_DefaultConstruct();
            _UnderlyingPtr = __MR_VertCoordsDiff_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertCoordsDiff::VertCoordsDiff`.
        public unsafe Const_VertCoordsDiff(MR._ByValue_VertCoordsDiff _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertCoordsDiff._Underlying *_other);
            _UnderlyingPtr = __MR_VertCoordsDiff_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the difference, that can be applied to vector-from in order to get vector-to
        /// Generated from constructor `MR::VertCoordsDiff::VertCoordsDiff`.
        public unsafe Const_VertCoordsDiff(MR.Const_VertCoords from, MR.Const_VertCoords to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_Construct", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_Construct(MR.Const_VertCoords._Underlying *from, MR.Const_VertCoords._Underlying *to);
            _UnderlyingPtr = __MR_VertCoordsDiff_Construct(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// returns true if this object does contain some difference in point coordinates;
        /// if (from) vector has just more points and the common elements are the same,
        /// then the method will return false since nothing is stored here
        /// Generated from method `MR::VertCoordsDiff::any`.
        public unsafe bool Any()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_any", ExactSpelling = true)]
            extern static byte __MR_VertCoordsDiff_any(_Underlying *_this);
            return __MR_VertCoordsDiff_any(_UnderlyingPtr) != 0;
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::VertCoordsDiff::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_VertCoordsDiff_heapBytes(_Underlying *_this);
            return __MR_VertCoordsDiff_heapBytes(_UnderlyingPtr);
        }
    }

    /// this object stores a difference between two vectors with 3D coordinates
    /// \details if the vectors are similar then this object is small, if the vectors are very distinct then this object will be even larger than one vector itself
    /// Generated from class `MR::VertCoordsDiff`.
    /// This is the non-const half of the class.
    public class VertCoordsDiff : Const_VertCoordsDiff
    {
        internal unsafe VertCoordsDiff(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertCoordsDiff() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_DefaultConstruct();
            _UnderlyingPtr = __MR_VertCoordsDiff_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertCoordsDiff::VertCoordsDiff`.
        public unsafe VertCoordsDiff(MR._ByValue_VertCoordsDiff _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertCoordsDiff._Underlying *_other);
            _UnderlyingPtr = __MR_VertCoordsDiff_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the difference, that can be applied to vector-from in order to get vector-to
        /// Generated from constructor `MR::VertCoordsDiff::VertCoordsDiff`.
        public unsafe VertCoordsDiff(MR.Const_VertCoords from, MR.Const_VertCoords to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_Construct", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_Construct(MR.Const_VertCoords._Underlying *from, MR.Const_VertCoords._Underlying *to);
            _UnderlyingPtr = __MR_VertCoordsDiff_Construct(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// Generated from method `MR::VertCoordsDiff::operator=`.
        public unsafe MR.VertCoordsDiff Assign(MR._ByValue_VertCoordsDiff _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertCoordsDiff._Underlying *__MR_VertCoordsDiff_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VertCoordsDiff._Underlying *_other);
            return new(__MR_VertCoordsDiff_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// given vector-from on input converts it in vector-to,
        /// this object is updated to become the reverse difference from original vector-to to original vector-from
        /// Generated from method `MR::VertCoordsDiff::applyAndSwap`.
        public unsafe void ApplyAndSwap(MR.VertCoords m)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertCoordsDiff_applyAndSwap", ExactSpelling = true)]
            extern static void __MR_VertCoordsDiff_applyAndSwap(_Underlying *_this, MR.VertCoords._Underlying *m);
            __MR_VertCoordsDiff_applyAndSwap(_UnderlyingPtr, m._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VertCoordsDiff` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VertCoordsDiff`/`Const_VertCoordsDiff` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VertCoordsDiff
    {
        internal readonly Const_VertCoordsDiff? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VertCoordsDiff() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VertCoordsDiff(Const_VertCoordsDiff new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VertCoordsDiff(Const_VertCoordsDiff arg) {return new(arg);}
        public _ByValue_VertCoordsDiff(MR.Misc._Moved<VertCoordsDiff> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VertCoordsDiff(MR.Misc._Moved<VertCoordsDiff> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VertCoordsDiff` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertCoordsDiff`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertCoordsDiff`/`Const_VertCoordsDiff` directly.
    public class _InOptMut_VertCoordsDiff
    {
        public VertCoordsDiff? Opt;

        public _InOptMut_VertCoordsDiff() {}
        public _InOptMut_VertCoordsDiff(VertCoordsDiff value) {Opt = value;}
        public static implicit operator _InOptMut_VertCoordsDiff(VertCoordsDiff value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertCoordsDiff` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertCoordsDiff`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertCoordsDiff`/`Const_VertCoordsDiff` to pass it to the function.
    public class _InOptConst_VertCoordsDiff
    {
        public Const_VertCoordsDiff? Opt;

        public _InOptConst_VertCoordsDiff() {}
        public _InOptConst_VertCoordsDiff(Const_VertCoordsDiff value) {Opt = value;}
        public static implicit operator _InOptConst_VertCoordsDiff(Const_VertCoordsDiff value) {return new(value);}
    }
}
