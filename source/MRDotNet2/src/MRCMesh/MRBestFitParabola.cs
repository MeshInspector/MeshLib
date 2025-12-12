public static partial class MR
{
    /// accumulates a number of (x,y) points to find the best-least-squares parabola approximating them
    /// Generated from class `MR::BestFitParabolaf`.
    /// This is the const half of the class.
    public class Const_BestFitParabolaf : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BestFitParabolaf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_Destroy", ExactSpelling = true)]
            extern static void __MR_BestFitParabolaf_Destroy(_Underlying *_this);
            __MR_BestFitParabolaf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BestFitParabolaf() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BestFitParabolaf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BestFitParabolaf._Underlying *__MR_BestFitParabolaf_DefaultConstruct();
            _UnderlyingPtr = __MR_BestFitParabolaf_DefaultConstruct();
        }

        /// Generated from constructor `MR::BestFitParabolaf::BestFitParabolaf`.
        public unsafe Const_BestFitParabolaf(MR.Const_BestFitParabolaf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BestFitParabolaf._Underlying *__MR_BestFitParabolaf_ConstructFromAnother(MR.BestFitParabolaf._Underlying *_other);
            _UnderlyingPtr = __MR_BestFitParabolaf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes the best approximating parabola from the accumulated points;
        /// Generated from method `MR::BestFitParabolaf::getBestParabola`.
        /// Parameter `tol` defaults to `std::numeric_limits<float>::epsilon()`.
        public unsafe MR.Parabolaf GetBestParabola(float? tol = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_getBestParabola", ExactSpelling = true)]
            extern static MR.Parabolaf._Underlying *__MR_BestFitParabolaf_getBestParabola(_Underlying *_this, float *tol);
            float __deref_tol = tol.GetValueOrDefault();
            return new(__MR_BestFitParabolaf_getBestParabola(_UnderlyingPtr, tol.HasValue ? &__deref_tol : null), is_owning: true);
        }
    }

    /// accumulates a number of (x,y) points to find the best-least-squares parabola approximating them
    /// Generated from class `MR::BestFitParabolaf`.
    /// This is the non-const half of the class.
    public class BestFitParabolaf : Const_BestFitParabolaf
    {
        internal unsafe BestFitParabolaf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe BestFitParabolaf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BestFitParabolaf._Underlying *__MR_BestFitParabolaf_DefaultConstruct();
            _UnderlyingPtr = __MR_BestFitParabolaf_DefaultConstruct();
        }

        /// Generated from constructor `MR::BestFitParabolaf::BestFitParabolaf`.
        public unsafe BestFitParabolaf(MR.Const_BestFitParabolaf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BestFitParabolaf._Underlying *__MR_BestFitParabolaf_ConstructFromAnother(MR.BestFitParabolaf._Underlying *_other);
            _UnderlyingPtr = __MR_BestFitParabolaf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::BestFitParabolaf::operator=`.
        public unsafe MR.BestFitParabolaf Assign(MR.Const_BestFitParabolaf _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BestFitParabolaf._Underlying *__MR_BestFitParabolaf_AssignFromAnother(_Underlying *_this, MR.BestFitParabolaf._Underlying *_other);
            return new(__MR_BestFitParabolaf_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// accumulates one more point for parabola fitting
        /// Generated from method `MR::BestFitParabolaf::addPoint`.
        public unsafe void AddPoint(float x, float y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_addPoint_2", ExactSpelling = true)]
            extern static void __MR_BestFitParabolaf_addPoint_2(_Underlying *_this, float x, float y);
            __MR_BestFitParabolaf_addPoint_2(_UnderlyingPtr, x, y);
        }

        /// accumulates one more point with given weight for parabola fitting
        /// Generated from method `MR::BestFitParabolaf::addPoint`.
        public unsafe void AddPoint(float x, float y, float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolaf_addPoint_3", ExactSpelling = true)]
            extern static void __MR_BestFitParabolaf_addPoint_3(_Underlying *_this, float x, float y, float weight);
            __MR_BestFitParabolaf_addPoint_3(_UnderlyingPtr, x, y, weight);
        }
    }

    /// This is used for optional parameters of class `BestFitParabolaf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BestFitParabolaf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BestFitParabolaf`/`Const_BestFitParabolaf` directly.
    public class _InOptMut_BestFitParabolaf
    {
        public BestFitParabolaf? Opt;

        public _InOptMut_BestFitParabolaf() {}
        public _InOptMut_BestFitParabolaf(BestFitParabolaf value) {Opt = value;}
        public static implicit operator _InOptMut_BestFitParabolaf(BestFitParabolaf value) {return new(value);}
    }

    /// This is used for optional parameters of class `BestFitParabolaf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BestFitParabolaf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BestFitParabolaf`/`Const_BestFitParabolaf` to pass it to the function.
    public class _InOptConst_BestFitParabolaf
    {
        public Const_BestFitParabolaf? Opt;

        public _InOptConst_BestFitParabolaf() {}
        public _InOptConst_BestFitParabolaf(Const_BestFitParabolaf value) {Opt = value;}
        public static implicit operator _InOptConst_BestFitParabolaf(Const_BestFitParabolaf value) {return new(value);}
    }

    /// accumulates a number of (x,y) points to find the best-least-squares parabola approximating them
    /// Generated from class `MR::BestFitParabolad`.
    /// This is the const half of the class.
    public class Const_BestFitParabolad : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BestFitParabolad(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_Destroy", ExactSpelling = true)]
            extern static void __MR_BestFitParabolad_Destroy(_Underlying *_this);
            __MR_BestFitParabolad_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BestFitParabolad() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BestFitParabolad() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BestFitParabolad._Underlying *__MR_BestFitParabolad_DefaultConstruct();
            _UnderlyingPtr = __MR_BestFitParabolad_DefaultConstruct();
        }

        /// Generated from constructor `MR::BestFitParabolad::BestFitParabolad`.
        public unsafe Const_BestFitParabolad(MR.Const_BestFitParabolad _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BestFitParabolad._Underlying *__MR_BestFitParabolad_ConstructFromAnother(MR.BestFitParabolad._Underlying *_other);
            _UnderlyingPtr = __MR_BestFitParabolad_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes the best approximating parabola from the accumulated points;
        /// Generated from method `MR::BestFitParabolad::getBestParabola`.
        /// Parameter `tol` defaults to `std::numeric_limits<double>::epsilon()`.
        public unsafe MR.Parabolad GetBestParabola(double? tol = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_getBestParabola", ExactSpelling = true)]
            extern static MR.Parabolad._Underlying *__MR_BestFitParabolad_getBestParabola(_Underlying *_this, double *tol);
            double __deref_tol = tol.GetValueOrDefault();
            return new(__MR_BestFitParabolad_getBestParabola(_UnderlyingPtr, tol.HasValue ? &__deref_tol : null), is_owning: true);
        }
    }

    /// accumulates a number of (x,y) points to find the best-least-squares parabola approximating them
    /// Generated from class `MR::BestFitParabolad`.
    /// This is the non-const half of the class.
    public class BestFitParabolad : Const_BestFitParabolad
    {
        internal unsafe BestFitParabolad(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe BestFitParabolad() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BestFitParabolad._Underlying *__MR_BestFitParabolad_DefaultConstruct();
            _UnderlyingPtr = __MR_BestFitParabolad_DefaultConstruct();
        }

        /// Generated from constructor `MR::BestFitParabolad::BestFitParabolad`.
        public unsafe BestFitParabolad(MR.Const_BestFitParabolad _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BestFitParabolad._Underlying *__MR_BestFitParabolad_ConstructFromAnother(MR.BestFitParabolad._Underlying *_other);
            _UnderlyingPtr = __MR_BestFitParabolad_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::BestFitParabolad::operator=`.
        public unsafe MR.BestFitParabolad Assign(MR.Const_BestFitParabolad _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BestFitParabolad._Underlying *__MR_BestFitParabolad_AssignFromAnother(_Underlying *_this, MR.BestFitParabolad._Underlying *_other);
            return new(__MR_BestFitParabolad_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// accumulates one more point for parabola fitting
        /// Generated from method `MR::BestFitParabolad::addPoint`.
        public unsafe void AddPoint(double x, double y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_addPoint_2", ExactSpelling = true)]
            extern static void __MR_BestFitParabolad_addPoint_2(_Underlying *_this, double x, double y);
            __MR_BestFitParabolad_addPoint_2(_UnderlyingPtr, x, y);
        }

        /// accumulates one more point with given weight for parabola fitting
        /// Generated from method `MR::BestFitParabolad::addPoint`.
        public unsafe void AddPoint(double x, double y, double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BestFitParabolad_addPoint_3", ExactSpelling = true)]
            extern static void __MR_BestFitParabolad_addPoint_3(_Underlying *_this, double x, double y, double weight);
            __MR_BestFitParabolad_addPoint_3(_UnderlyingPtr, x, y, weight);
        }
    }

    /// This is used for optional parameters of class `BestFitParabolad` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BestFitParabolad`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BestFitParabolad`/`Const_BestFitParabolad` directly.
    public class _InOptMut_BestFitParabolad
    {
        public BestFitParabolad? Opt;

        public _InOptMut_BestFitParabolad() {}
        public _InOptMut_BestFitParabolad(BestFitParabolad value) {Opt = value;}
        public static implicit operator _InOptMut_BestFitParabolad(BestFitParabolad value) {return new(value);}
    }

    /// This is used for optional parameters of class `BestFitParabolad` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BestFitParabolad`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BestFitParabolad`/`Const_BestFitParabolad` to pass it to the function.
    public class _InOptConst_BestFitParabolad
    {
        public Const_BestFitParabolad? Opt;

        public _InOptConst_BestFitParabolad() {}
        public _InOptConst_BestFitParabolad(Const_BestFitParabolad value) {Opt = value;}
        public static implicit operator _InOptConst_BestFitParabolad(Const_BestFitParabolad value) {return new(value);}
    }
}
