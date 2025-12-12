public static partial class MR
{
    /// Simple class for calculating histogram
    /// Generated from class `MR::Histogram`.
    /// This is the const half of the class.
    public class Const_Histogram : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Histogram(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_Destroy", ExactSpelling = true)]
            extern static void __MR_Histogram_Destroy(_Underlying *_this);
            __MR_Histogram_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Histogram() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Histogram() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_DefaultConstruct();
            _UnderlyingPtr = __MR_Histogram_DefaultConstruct();
        }

        /// Generated from constructor `MR::Histogram::Histogram`.
        public unsafe Const_Histogram(MR._ByValue_Histogram _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Histogram._Underlying *_other);
            _UnderlyingPtr = __MR_Histogram_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Initialize histogram with minimum and maximum values, and number of bins
        /// Generated from constructor `MR::Histogram::Histogram`.
        public unsafe Const_Histogram(float min, float max, ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_Construct", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_Construct(float min, float max, ulong size);
            _UnderlyingPtr = __MR_Histogram_Construct(min, max, size);
        }

        /// Gets bins
        /// Generated from method `MR::Histogram::getBins`.
        public unsafe MR.Std.Const_Vector_MRUint64T GetBins()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_getBins", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRUint64T._Underlying *__MR_Histogram_getBins(_Underlying *_this);
            return new(__MR_Histogram_getBins(_UnderlyingPtr), is_owning: false);
        }

        /// Gets minimum value of histogram
        /// Generated from method `MR::Histogram::getMin`.
        public unsafe float GetMin()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_getMin", ExactSpelling = true)]
            extern static float __MR_Histogram_getMin(_Underlying *_this);
            return __MR_Histogram_getMin(_UnderlyingPtr);
        }

        /// Gets maximum value of histogram
        /// Generated from method `MR::Histogram::getMax`.
        public unsafe float GetMax()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_getMax", ExactSpelling = true)]
            extern static float __MR_Histogram_getMax(_Underlying *_this);
            return __MR_Histogram_getMax(_UnderlyingPtr);
        }

        /// Gets id of bin that inherits sample
        /// Generated from method `MR::Histogram::getBinId`.
        public unsafe ulong GetBinId(float sample)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_getBinId", ExactSpelling = true)]
            extern static ulong __MR_Histogram_getBinId(_Underlying *_this, float sample);
            return __MR_Histogram_getBinId(_UnderlyingPtr, sample);
        }

        /// Gets minimum and maximum of diapason inherited by bin
        /// Generated from method `MR::Histogram::getBinMinMax`.
        public unsafe MR.Std.Pair_Float_Float GetBinMinMax(ulong binId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_getBinMinMax", ExactSpelling = true)]
            extern static MR.Std.Pair_Float_Float._Underlying *__MR_Histogram_getBinMinMax(_Underlying *_this, ulong binId);
            return new(__MR_Histogram_getBinMinMax(_UnderlyingPtr, binId), is_owning: true);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Histogram::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Histogram_heapBytes(_Underlying *_this);
            return __MR_Histogram_heapBytes(_UnderlyingPtr);
        }
    }

    /// Simple class for calculating histogram
    /// Generated from class `MR::Histogram`.
    /// This is the non-const half of the class.
    public class Histogram : Const_Histogram
    {
        internal unsafe Histogram(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Histogram() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_DefaultConstruct();
            _UnderlyingPtr = __MR_Histogram_DefaultConstruct();
        }

        /// Generated from constructor `MR::Histogram::Histogram`.
        public unsafe Histogram(MR._ByValue_Histogram _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Histogram._Underlying *_other);
            _UnderlyingPtr = __MR_Histogram_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Initialize histogram with minimum and maximum values, and number of bins
        /// Generated from constructor `MR::Histogram::Histogram`.
        public unsafe Histogram(float min, float max, ulong size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_Construct", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_Construct(float min, float max, ulong size);
            _UnderlyingPtr = __MR_Histogram_Construct(min, max, size);
        }

        /// Generated from method `MR::Histogram::operator=`.
        public unsafe MR.Histogram Assign(MR._ByValue_Histogram _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_Histogram_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Histogram._Underlying *_other);
            return new(__MR_Histogram_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Adds sample to corresponding bin
        /// Generated from method `MR::Histogram::addSample`.
        /// Parameter `count` defaults to `1`.
        public unsafe void AddSample(float sample, ulong? count = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_addSample", ExactSpelling = true)]
            extern static void __MR_Histogram_addSample(_Underlying *_this, float sample, ulong *count);
            ulong __deref_count = count.GetValueOrDefault();
            __MR_Histogram_addSample(_UnderlyingPtr, sample, count.HasValue ? &__deref_count : null);
        }

        /// Adds bins of input hist to this
        /// Generated from method `MR::Histogram::addHistogram`.
        public unsafe void AddHistogram(MR.Const_Histogram hist)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Histogram_addHistogram", ExactSpelling = true)]
            extern static void __MR_Histogram_addHistogram(_Underlying *_this, MR.Const_Histogram._Underlying *hist);
            __MR_Histogram_addHistogram(_UnderlyingPtr, hist._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Histogram` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Histogram`/`Const_Histogram` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Histogram
    {
        internal readonly Const_Histogram? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Histogram() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Histogram(Const_Histogram new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Histogram(Const_Histogram arg) {return new(arg);}
        public _ByValue_Histogram(MR.Misc._Moved<Histogram> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Histogram(MR.Misc._Moved<Histogram> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Histogram` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Histogram`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Histogram`/`Const_Histogram` directly.
    public class _InOptMut_Histogram
    {
        public Histogram? Opt;

        public _InOptMut_Histogram() {}
        public _InOptMut_Histogram(Histogram value) {Opt = value;}
        public static implicit operator _InOptMut_Histogram(Histogram value) {return new(value);}
    }

    /// This is used for optional parameters of class `Histogram` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Histogram`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Histogram`/`Const_Histogram` to pass it to the function.
    public class _InOptConst_Histogram
    {
        public Const_Histogram? Opt;

        public _InOptConst_Histogram() {}
        public _InOptConst_Histogram(Const_Histogram value) {Opt = value;}
        public static implicit operator _InOptConst_Histogram(Const_Histogram value) {return new(value);}
    }
}
