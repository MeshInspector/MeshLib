public static partial class MR
{
    /// Class to accumulate points and make best line / plane approximation
    /// Generated from class `MR::PointAccumulator`.
    /// This is the const half of the class.
    public class Const_PointAccumulator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointAccumulator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_Destroy", ExactSpelling = true)]
            extern static void __MR_PointAccumulator_Destroy(_Underlying *_this);
            __MR_PointAccumulator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointAccumulator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointAccumulator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointAccumulator._Underlying *__MR_PointAccumulator_DefaultConstruct();
            _UnderlyingPtr = __MR_PointAccumulator_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointAccumulator::PointAccumulator`.
        public unsafe Const_PointAccumulator(MR.Const_PointAccumulator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointAccumulator._Underlying *__MR_PointAccumulator_ConstructFromAnother(MR.PointAccumulator._Underlying *_other);
            _UnderlyingPtr = __MR_PointAccumulator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes the best approximating plane from the accumulated points
        /// Generated from method `MR::PointAccumulator::getBestPlane`.
        public unsafe MR.Plane3d GetBestPlane()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getBestPlane", ExactSpelling = true)]
            extern static MR.Plane3d._Underlying *__MR_PointAccumulator_getBestPlane(_Underlying *_this);
            return new(__MR_PointAccumulator_getBestPlane(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::PointAccumulator::getBestPlanef`.
        public unsafe MR.Plane3f GetBestPlanef()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getBestPlanef", ExactSpelling = true)]
            extern static MR.Plane3f._Underlying *__MR_PointAccumulator_getBestPlanef(_Underlying *_this);
            return new(__MR_PointAccumulator_getBestPlanef(_UnderlyingPtr), is_owning: true);
        }

        /// computes the best approximating line from the accumulated points
        /// Generated from method `MR::PointAccumulator::getBestLine`.
        public unsafe MR.Line3d GetBestLine()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getBestLine", ExactSpelling = true)]
            extern static MR.Line3d._Underlying *__MR_PointAccumulator_getBestLine(_Underlying *_this);
            return new(__MR_PointAccumulator_getBestLine(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::PointAccumulator::getBestLinef`.
        public unsafe MR.Line3f GetBestLinef()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getBestLinef", ExactSpelling = true)]
            extern static MR.Line3f._Underlying *__MR_PointAccumulator_getBestLinef(_Underlying *_this);
            return new(__MR_PointAccumulator_getBestLinef(_UnderlyingPtr), is_owning: true);
        }

        /// computes centroid and eigenvectors/eigenvalues of the covariance matrix of the accumulated points
        /// Generated from method `MR::PointAccumulator::getCenteredCovarianceEigen`.
        public unsafe bool GetCenteredCovarianceEigen(MR.Mut_Vector3d centroid, MR.Mut_Matrix3d eigenvectors, MR.Mut_Vector3d eigenvalues)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getCenteredCovarianceEigen_MR_Vector3d", ExactSpelling = true)]
            extern static byte __MR_PointAccumulator_getCenteredCovarianceEigen_MR_Vector3d(_Underlying *_this, MR.Mut_Vector3d._Underlying *centroid, MR.Mut_Matrix3d._Underlying *eigenvectors, MR.Mut_Vector3d._Underlying *eigenvalues);
            return __MR_PointAccumulator_getCenteredCovarianceEigen_MR_Vector3d(_UnderlyingPtr, centroid._UnderlyingPtr, eigenvectors._UnderlyingPtr, eigenvalues._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::PointAccumulator::getCenteredCovarianceEigen`.
        public unsafe bool GetCenteredCovarianceEigen(MR.Mut_Vector3f centroid, MR.Mut_Matrix3f eigenvectors, MR.Mut_Vector3f eigenvalues)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getCenteredCovarianceEigen_MR_Vector3f", ExactSpelling = true)]
            extern static byte __MR_PointAccumulator_getCenteredCovarianceEigen_MR_Vector3f(_Underlying *_this, MR.Mut_Vector3f._Underlying *centroid, MR.Mut_Matrix3f._Underlying *eigenvectors, MR.Mut_Vector3f._Underlying *eigenvalues);
            return __MR_PointAccumulator_getCenteredCovarianceEigen_MR_Vector3f(_UnderlyingPtr, centroid._UnderlyingPtr, eigenvectors._UnderlyingPtr, eigenvalues._UnderlyingPtr) != 0;
        }

        /// returns the transformation that maps (0,0,0) into point centroid,
        /// and maps vectors (1,0,0), (0,1,0), (0,0,1) into first, second, third eigenvectors corresponding to ascending eigenvalues
        /// Generated from method `MR::PointAccumulator::getBasicXf`.
        public unsafe MR.AffineXf3d GetBasicXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getBasicXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointAccumulator_getBasicXf(_Underlying *_this);
            return __MR_PointAccumulator_getBasicXf(_UnderlyingPtr);
        }

        /// Generated from method `MR::PointAccumulator::getBasicXf3f`.
        public unsafe MR.AffineXf3f GetBasicXf3f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_getBasicXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_PointAccumulator_getBasicXf3f(_Underlying *_this);
            return __MR_PointAccumulator_getBasicXf3f(_UnderlyingPtr);
        }

        /// returns 4 transformations, each maps (0,0,0) into point centroid,
        /// and maps vectors (1,0,0), (0,1,0), (0,0,1) into +/- first, +/- second, +/- third eigenvectors (forming positive reference frame) corresponding to ascending eigenvalues
        /// Generated from method `MR::PointAccumulator::get4BasicXfs`.
        public unsafe MR.Std.Array_MRAffineXf3d_4 Get4BasicXfs()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_get4BasicXfs", ExactSpelling = true)]
            extern static MR.Std.Array_MRAffineXf3d_4 __MR_PointAccumulator_get4BasicXfs(_Underlying *_this);
            return __MR_PointAccumulator_get4BasicXfs(_UnderlyingPtr);
        }

        /// Generated from method `MR::PointAccumulator::get4BasicXfs3f`.
        public unsafe MR.Std.Array_MRAffineXf3f_4 Get4BasicXfs3f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_get4BasicXfs3f", ExactSpelling = true)]
            extern static MR.Std.Array_MRAffineXf3f_4 __MR_PointAccumulator_get4BasicXfs3f(_Underlying *_this);
            return __MR_PointAccumulator_get4BasicXfs3f(_UnderlyingPtr);
        }

        /// Generated from method `MR::PointAccumulator::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_valid", ExactSpelling = true)]
            extern static byte __MR_PointAccumulator_valid(_Underlying *_this);
            return __MR_PointAccumulator_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Class to accumulate points and make best line / plane approximation
    /// Generated from class `MR::PointAccumulator`.
    /// This is the non-const half of the class.
    public class PointAccumulator : Const_PointAccumulator
    {
        internal unsafe PointAccumulator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointAccumulator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointAccumulator._Underlying *__MR_PointAccumulator_DefaultConstruct();
            _UnderlyingPtr = __MR_PointAccumulator_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointAccumulator::PointAccumulator`.
        public unsafe PointAccumulator(MR.Const_PointAccumulator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointAccumulator._Underlying *__MR_PointAccumulator_ConstructFromAnother(MR.PointAccumulator._Underlying *_other);
            _UnderlyingPtr = __MR_PointAccumulator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointAccumulator::operator=`.
        public unsafe MR.PointAccumulator Assign(MR.Const_PointAccumulator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointAccumulator._Underlying *__MR_PointAccumulator_AssignFromAnother(_Underlying *_this, MR.PointAccumulator._Underlying *_other);
            return new(__MR_PointAccumulator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PointAccumulator::addPoint`.
        public unsafe void AddPoint(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_addPoint_1_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_PointAccumulator_addPoint_1_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            __MR_PointAccumulator_addPoint_1_MR_Vector3d(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::PointAccumulator::addPoint`.
        public unsafe void AddPoint(MR.Const_Vector3d pt, double weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_addPoint_2_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_PointAccumulator_addPoint_2_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *pt, double weight);
            __MR_PointAccumulator_addPoint_2_MR_Vector3d(_UnderlyingPtr, pt._UnderlyingPtr, weight);
        }

        /// Generated from method `MR::PointAccumulator::addPoint`.
        public unsafe void AddPoint(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_addPoint_1_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_PointAccumulator_addPoint_1_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            __MR_PointAccumulator_addPoint_1_MR_Vector3f(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// Generated from method `MR::PointAccumulator::addPoint`.
        public unsafe void AddPoint(MR.Const_Vector3f pt, float weight)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointAccumulator_addPoint_2_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_PointAccumulator_addPoint_2_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *pt, float weight);
            __MR_PointAccumulator_addPoint_2_MR_Vector3f(_UnderlyingPtr, pt._UnderlyingPtr, weight);
        }
    }

    /// This is used for optional parameters of class `PointAccumulator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointAccumulator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointAccumulator`/`Const_PointAccumulator` directly.
    public class _InOptMut_PointAccumulator
    {
        public PointAccumulator? Opt;

        public _InOptMut_PointAccumulator() {}
        public _InOptMut_PointAccumulator(PointAccumulator value) {Opt = value;}
        public static implicit operator _InOptMut_PointAccumulator(PointAccumulator value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointAccumulator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointAccumulator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointAccumulator`/`Const_PointAccumulator` to pass it to the function.
    public class _InOptConst_PointAccumulator
    {
        public Const_PointAccumulator? Opt;

        public _InOptConst_PointAccumulator() {}
        public _InOptConst_PointAccumulator(Const_PointAccumulator value) {Opt = value;}
        public static implicit operator _InOptConst_PointAccumulator(Const_PointAccumulator value) {return new(value);}
    }

    /// Class to accumulate planes to find then their crossing point
    /// Generated from class `MR::PlaneAccumulator`.
    /// This is the const half of the class.
    public class Const_PlaneAccumulator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PlaneAccumulator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_Destroy", ExactSpelling = true)]
            extern static void __MR_PlaneAccumulator_Destroy(_Underlying *_this);
            __MR_PlaneAccumulator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PlaneAccumulator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PlaneAccumulator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PlaneAccumulator._Underlying *__MR_PlaneAccumulator_DefaultConstruct();
            _UnderlyingPtr = __MR_PlaneAccumulator_DefaultConstruct();
        }

        /// Generated from constructor `MR::PlaneAccumulator::PlaneAccumulator`.
        public unsafe Const_PlaneAccumulator(MR.Const_PlaneAccumulator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PlaneAccumulator._Underlying *__MR_PlaneAccumulator_ConstructFromAnother(MR.PlaneAccumulator._Underlying *_other);
            _UnderlyingPtr = __MR_PlaneAccumulator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// computes the point that minimizes the sum of squared distances to accumulated planes;
        /// if such point is not unique then returns the one closest to p0
        /// \param tol relative epsilon-tolerance for too small number detection
        /// \param rank optional output for solution matrix rank according to given tolerance
        /// \param space rank=1: unit normal to solution plane, rank=2: unit direction of solution line, rank=3: zero vector
        /// Generated from method `MR::PlaneAccumulator::findBestCrossPoint`.
        public unsafe MR.Vector3d FindBestCrossPoint(MR.Const_Vector3d p0, double tol, MR.Misc.InOut<int>? rank = null, MR.Mut_Vector3d? space = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_findBestCrossPoint_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Vector3d __MR_PlaneAccumulator_findBestCrossPoint_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *p0, double tol, int *rank, MR.Mut_Vector3d._Underlying *space);
            int __value_rank = rank is not null ? rank.Value : default(int);
            var __ret = __MR_PlaneAccumulator_findBestCrossPoint_MR_Vector3d(_UnderlyingPtr, p0._UnderlyingPtr, tol, rank is not null ? &__value_rank : null, space is not null ? space._UnderlyingPtr : null);
            if (rank is not null) rank.Value = __value_rank;
            return __ret;
        }

        /// Generated from method `MR::PlaneAccumulator::findBestCrossPoint`.
        public unsafe MR.Vector3f FindBestCrossPoint(MR.Const_Vector3f p0, float tol, MR.Misc.InOut<int>? rank = null, MR.Mut_Vector3f? space = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_findBestCrossPoint_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Vector3f __MR_PlaneAccumulator_findBestCrossPoint_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *p0, float tol, int *rank, MR.Mut_Vector3f._Underlying *space);
            int __value_rank = rank is not null ? rank.Value : default(int);
            var __ret = __MR_PlaneAccumulator_findBestCrossPoint_MR_Vector3f(_UnderlyingPtr, p0._UnderlyingPtr, tol, rank is not null ? &__value_rank : null, space is not null ? space._UnderlyingPtr : null);
            if (rank is not null) rank.Value = __value_rank;
            return __ret;
        }
    }

    /// Class to accumulate planes to find then their crossing point
    /// Generated from class `MR::PlaneAccumulator`.
    /// This is the non-const half of the class.
    public class PlaneAccumulator : Const_PlaneAccumulator
    {
        internal unsafe PlaneAccumulator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PlaneAccumulator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PlaneAccumulator._Underlying *__MR_PlaneAccumulator_DefaultConstruct();
            _UnderlyingPtr = __MR_PlaneAccumulator_DefaultConstruct();
        }

        /// Generated from constructor `MR::PlaneAccumulator::PlaneAccumulator`.
        public unsafe PlaneAccumulator(MR.Const_PlaneAccumulator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PlaneAccumulator._Underlying *__MR_PlaneAccumulator_ConstructFromAnother(MR.PlaneAccumulator._Underlying *_other);
            _UnderlyingPtr = __MR_PlaneAccumulator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneAccumulator::operator=`.
        public unsafe MR.PlaneAccumulator Assign(MR.Const_PlaneAccumulator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PlaneAccumulator._Underlying *__MR_PlaneAccumulator_AssignFromAnother(_Underlying *_this, MR.PlaneAccumulator._Underlying *_other);
            return new(__MR_PlaneAccumulator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::PlaneAccumulator::addPlane`.
        public unsafe void AddPlane(MR.Const_Plane3d pl)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_addPlane_MR_Plane3d", ExactSpelling = true)]
            extern static void __MR_PlaneAccumulator_addPlane_MR_Plane3d(_Underlying *_this, MR.Const_Plane3d._Underlying *pl);
            __MR_PlaneAccumulator_addPlane_MR_Plane3d(_UnderlyingPtr, pl._UnderlyingPtr);
        }

        /// Generated from method `MR::PlaneAccumulator::addPlane`.
        public unsafe void AddPlane(MR.Const_Plane3f pl)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlaneAccumulator_addPlane_MR_Plane3f", ExactSpelling = true)]
            extern static void __MR_PlaneAccumulator_addPlane_MR_Plane3f(_Underlying *_this, MR.Const_Plane3f._Underlying *pl);
            __MR_PlaneAccumulator_addPlane_MR_Plane3f(_UnderlyingPtr, pl._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `PlaneAccumulator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PlaneAccumulator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PlaneAccumulator`/`Const_PlaneAccumulator` directly.
    public class _InOptMut_PlaneAccumulator
    {
        public PlaneAccumulator? Opt;

        public _InOptMut_PlaneAccumulator() {}
        public _InOptMut_PlaneAccumulator(PlaneAccumulator value) {Opt = value;}
        public static implicit operator _InOptMut_PlaneAccumulator(PlaneAccumulator value) {return new(value);}
    }

    /// This is used for optional parameters of class `PlaneAccumulator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PlaneAccumulator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PlaneAccumulator`/`Const_PlaneAccumulator` to pass it to the function.
    public class _InOptConst_PlaneAccumulator
    {
        public Const_PlaneAccumulator? Opt;

        public _InOptConst_PlaneAccumulator() {}
        public _InOptConst_PlaneAccumulator(Const_PlaneAccumulator value) {Opt = value;}
        public static implicit operator _InOptConst_PlaneAccumulator(Const_PlaneAccumulator value) {return new(value);}
    }

    /// Adds in existing PointAccumulator all given points
    /// Generated from function `MR::accumulatePoints`.
    public static unsafe void AccumulatePoints(MR.PointAccumulator accum, MR.Std.Const_Vector_MRVector3f points, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_accumulatePoints_std_vector_MR_Vector3f", ExactSpelling = true)]
        extern static void __MR_accumulatePoints_std_vector_MR_Vector3f(MR.PointAccumulator._Underlying *accum, MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_AffineXf3f._Underlying *xf);
        __MR_accumulatePoints_std_vector_MR_Vector3f(accum._UnderlyingPtr, points._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Adds in existing PointAccumulator all given weighed points
    /// Generated from function `MR::accumulateWeighedPoints`.
    public static unsafe void AccumulateWeighedPoints(MR.PointAccumulator accum, MR.Std.Const_Vector_MRVector3f points, MR.Std.Const_Vector_Float weights, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_accumulateWeighedPoints", ExactSpelling = true)]
        extern static void __MR_accumulateWeighedPoints(MR.PointAccumulator._Underlying *accum, MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Std.Const_Vector_Float._Underlying *weights, MR.Const_AffineXf3f._Underlying *xf);
        __MR_accumulateWeighedPoints(accum._UnderlyingPtr, points._UnderlyingPtr, weights._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Adds in existing PointAccumulator all mesh face centers with the weight equal to face area
    /// Generated from function `MR::accumulateFaceCenters`.
    public static unsafe void AccumulateFaceCenters(MR.PointAccumulator accum, MR.Const_MeshPart mp, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_accumulateFaceCenters", ExactSpelling = true)]
        extern static void __MR_accumulateFaceCenters(MR.PointAccumulator._Underlying *accum, MR.Const_MeshPart._Underlying *mp, MR.Const_AffineXf3f._Underlying *xf);
        __MR_accumulateFaceCenters(accum._UnderlyingPtr, mp._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Adds in existing PointAccumulator all line centers with the weight equal to the length line
    /// Generated from function `MR::accumulateLineCenters`.
    public static unsafe void AccumulateLineCenters(MR.PointAccumulator accum, MR.Const_Polyline3 pl, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_accumulateLineCenters", ExactSpelling = true)]
        extern static void __MR_accumulateLineCenters(MR.PointAccumulator._Underlying *accum, MR.Const_Polyline3._Underlying *pl, MR.Const_AffineXf3f._Underlying *xf);
        __MR_accumulateLineCenters(accum._UnderlyingPtr, pl._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }

    /// Adds in existing PointAccumulator all points from the cloud (region) with weight 1
    /// Generated from function `MR::accumulatePoints`.
    public static unsafe void AccumulatePoints(MR.PointAccumulator accum, MR.Const_PointCloudPart pcp, MR.Const_AffineXf3f? xf = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_accumulatePoints_MR_PointCloudPart", ExactSpelling = true)]
        extern static void __MR_accumulatePoints_MR_PointCloudPart(MR.PointAccumulator._Underlying *accum, MR.Const_PointCloudPart._Underlying *pcp, MR.Const_AffineXf3f._Underlying *xf);
        __MR_accumulatePoints_MR_PointCloudPart(accum._UnderlyingPtr, pcp._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
    }
}
