public static partial class MR
{
    /// \brief Class for deforming mesh using Bernstein interpolation
    /// \snippet cpp-examples/FreeFormDeformation.dox.cpp 0
    /// Generated from class `MR::FreeFormDeformer`.
    /// This is the const half of the class.
    public class Const_FreeFormDeformer : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FreeFormDeformer(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_Destroy", ExactSpelling = true)]
            extern static void __MR_FreeFormDeformer_Destroy(_Underlying *_this);
            __MR_FreeFormDeformer_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FreeFormDeformer() {Dispose(false);}

        /// Generated from constructor `MR::FreeFormDeformer::FreeFormDeformer`.
        public unsafe Const_FreeFormDeformer(MR._ByValue_FreeFormDeformer _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FreeFormDeformer._Underlying *__MR_FreeFormDeformer_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FreeFormDeformer._Underlying *_other);
            _UnderlyingPtr = __MR_FreeFormDeformer_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        // Only set mesh ref
        /// Generated from constructor `MR::FreeFormDeformer::FreeFormDeformer`.
        public unsafe Const_FreeFormDeformer(MR.VertCoords coords, MR.Const_VertBitSet valid) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_Construct_MR_VertCoords", ExactSpelling = true)]
            extern static MR.FreeFormDeformer._Underlying *__MR_FreeFormDeformer_Construct_MR_VertCoords(MR.VertCoords._Underlying *coords, MR.Const_VertBitSet._Underlying *valid);
            _UnderlyingPtr = __MR_FreeFormDeformer_Construct_MR_VertCoords(coords._UnderlyingPtr, valid._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FreeFormDeformer::FreeFormDeformer`.
        public unsafe Const_FreeFormDeformer(MR.Mesh mesh, MR.Const_VertBitSet? region = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_Construct_MR_Mesh", ExactSpelling = true)]
            extern static MR.FreeFormDeformer._Underlying *__MR_FreeFormDeformer_Construct_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            _UnderlyingPtr = __MR_FreeFormDeformer_Construct_MR_Mesh(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        // Gets ref grid point position
        /// Generated from method `MR::FreeFormDeformer::getRefGridPointPosition`.
        public unsafe MR.Const_Vector3f GetRefGridPointPosition(MR.Const_Vector3i coordOfPointInGrid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_getRefGridPointPosition", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_FreeFormDeformer_getRefGridPointPosition(_Underlying *_this, MR.Const_Vector3i._Underlying *coordOfPointInGrid);
            return new(__MR_FreeFormDeformer_getRefGridPointPosition(_UnderlyingPtr, coordOfPointInGrid._UnderlyingPtr), is_owning: false);
        }

        // Apply updated grid to all mesh points in parallel
        // ensure updating render object after using it
        /// Generated from method `MR::FreeFormDeformer::apply`.
        public unsafe void Apply()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_apply", ExactSpelling = true)]
            extern static void __MR_FreeFormDeformer_apply(_Underlying *_this);
            __MR_FreeFormDeformer_apply(_UnderlyingPtr);
        }

        // Apply updated grid to given point
        /// Generated from method `MR::FreeFormDeformer::applySinglePoint`.
        public unsafe MR.Vector3f ApplySinglePoint(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_applySinglePoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_FreeFormDeformer_applySinglePoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return __MR_FreeFormDeformer_applySinglePoint(_UnderlyingPtr, point._UnderlyingPtr);
        }

        // Get one dimension index by grid coord
        /// Generated from method `MR::FreeFormDeformer::getIndex`.
        public unsafe int GetIndex(MR.Const_Vector3i coordOfPointInGrid)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_getIndex", ExactSpelling = true)]
            extern static int __MR_FreeFormDeformer_getIndex(_Underlying *_this, MR.Const_Vector3i._Underlying *coordOfPointInGrid);
            return __MR_FreeFormDeformer_getIndex(_UnderlyingPtr, coordOfPointInGrid._UnderlyingPtr);
        }

        // Get grid coord by index
        /// Generated from method `MR::FreeFormDeformer::getCoord`.
        public unsafe MR.Vector3i GetCoord(int index)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_getCoord", ExactSpelling = true)]
            extern static MR.Vector3i __MR_FreeFormDeformer_getCoord(_Underlying *_this, int index);
            return __MR_FreeFormDeformer_getCoord(_UnderlyingPtr, index);
        }

        /// Generated from method `MR::FreeFormDeformer::getAllRefGridPositions`.
        public unsafe MR.Std.Const_Vector_MRVector3f GetAllRefGridPositions()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_getAllRefGridPositions", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRVector3f._Underlying *__MR_FreeFormDeformer_getAllRefGridPositions(_Underlying *_this);
            return new(__MR_FreeFormDeformer_getAllRefGridPositions(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FreeFormDeformer::getResolution`.
        public unsafe MR.Const_Vector3i GetResolution()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_getResolution", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_FreeFormDeformer_getResolution(_Underlying *_this);
            return new(__MR_FreeFormDeformer_getResolution(_UnderlyingPtr), is_owning: false);
        }
    }

    /// \brief Class for deforming mesh using Bernstein interpolation
    /// \snippet cpp-examples/FreeFormDeformation.dox.cpp 0
    /// Generated from class `MR::FreeFormDeformer`.
    /// This is the non-const half of the class.
    public class FreeFormDeformer : Const_FreeFormDeformer
    {
        internal unsafe FreeFormDeformer(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::FreeFormDeformer::FreeFormDeformer`.
        public unsafe FreeFormDeformer(MR._ByValue_FreeFormDeformer _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FreeFormDeformer._Underlying *__MR_FreeFormDeformer_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FreeFormDeformer._Underlying *_other);
            _UnderlyingPtr = __MR_FreeFormDeformer_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        // Only set mesh ref
        /// Generated from constructor `MR::FreeFormDeformer::FreeFormDeformer`.
        public unsafe FreeFormDeformer(MR.VertCoords coords, MR.Const_VertBitSet valid) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_Construct_MR_VertCoords", ExactSpelling = true)]
            extern static MR.FreeFormDeformer._Underlying *__MR_FreeFormDeformer_Construct_MR_VertCoords(MR.VertCoords._Underlying *coords, MR.Const_VertBitSet._Underlying *valid);
            _UnderlyingPtr = __MR_FreeFormDeformer_Construct_MR_VertCoords(coords._UnderlyingPtr, valid._UnderlyingPtr);
        }

        /// Generated from constructor `MR::FreeFormDeformer::FreeFormDeformer`.
        public unsafe FreeFormDeformer(MR.Mesh mesh, MR.Const_VertBitSet? region = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_Construct_MR_Mesh", ExactSpelling = true)]
            extern static MR.FreeFormDeformer._Underlying *__MR_FreeFormDeformer_Construct_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            _UnderlyingPtr = __MR_FreeFormDeformer_Construct_MR_Mesh(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
        }

        // Calculates all points' normalized positions in parallel
        // sets ref grid by initialBox, if initialBox is invalid uses mesh bounding box instead
        /// Generated from method `MR::FreeFormDeformer::init`.
        /// Parameter `resolution` defaults to `Vector3i::diagonal(2)`.
        /// Parameter `initialBox` defaults to `MR::Box3f()`.
        public unsafe void Init(MR.Const_Vector3i? resolution = null, MR.Const_Box3f? initialBox = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_init", ExactSpelling = true)]
            extern static void __MR_FreeFormDeformer_init(_Underlying *_this, MR.Const_Vector3i._Underlying *resolution, MR.Const_Box3f._Underlying *initialBox);
            __MR_FreeFormDeformer_init(_UnderlyingPtr, resolution is not null ? resolution._UnderlyingPtr : null, initialBox is not null ? initialBox._UnderlyingPtr : null);
        }

        // Updates ref grid point position
        /// Generated from method `MR::FreeFormDeformer::setRefGridPointPosition`.
        public unsafe void SetRefGridPointPosition(MR.Const_Vector3i coordOfPointInGrid, MR.Const_Vector3f newPos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_setRefGridPointPosition", ExactSpelling = true)]
            extern static void __MR_FreeFormDeformer_setRefGridPointPosition(_Underlying *_this, MR.Const_Vector3i._Underlying *coordOfPointInGrid, MR.Const_Vector3f._Underlying *newPos);
            __MR_FreeFormDeformer_setRefGridPointPosition(_UnderlyingPtr, coordOfPointInGrid._UnderlyingPtr, newPos._UnderlyingPtr);
        }

        /// Generated from method `MR::FreeFormDeformer::setAllRefGridPositions`.
        public unsafe void SetAllRefGridPositions(MR.Std.Const_Vector_MRVector3f refPoints)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormDeformer_setAllRefGridPositions", ExactSpelling = true)]
            extern static void __MR_FreeFormDeformer_setAllRefGridPositions(_Underlying *_this, MR.Std.Const_Vector_MRVector3f._Underlying *refPoints);
            __MR_FreeFormDeformer_setAllRefGridPositions(_UnderlyingPtr, refPoints._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FreeFormDeformer` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FreeFormDeformer`/`Const_FreeFormDeformer` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FreeFormDeformer
    {
        internal readonly Const_FreeFormDeformer? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FreeFormDeformer(Const_FreeFormDeformer new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FreeFormDeformer(Const_FreeFormDeformer arg) {return new(arg);}
        public _ByValue_FreeFormDeformer(MR.Misc._Moved<FreeFormDeformer> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FreeFormDeformer(MR.Misc._Moved<FreeFormDeformer> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FreeFormDeformer` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FreeFormDeformer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FreeFormDeformer`/`Const_FreeFormDeformer` directly.
    public class _InOptMut_FreeFormDeformer
    {
        public FreeFormDeformer? Opt;

        public _InOptMut_FreeFormDeformer() {}
        public _InOptMut_FreeFormDeformer(FreeFormDeformer value) {Opt = value;}
        public static implicit operator _InOptMut_FreeFormDeformer(FreeFormDeformer value) {return new(value);}
    }

    /// This is used for optional parameters of class `FreeFormDeformer` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FreeFormDeformer`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FreeFormDeformer`/`Const_FreeFormDeformer` to pass it to the function.
    public class _InOptConst_FreeFormDeformer
    {
        public Const_FreeFormDeformer? Opt;

        public _InOptConst_FreeFormDeformer() {}
        public _InOptConst_FreeFormDeformer(Const_FreeFormDeformer value) {Opt = value;}
        public static implicit operator _InOptConst_FreeFormDeformer(Const_FreeFormDeformer value) {return new(value);}
    }

    /// Class to accumulate source and target points for free form alignment
    /// Calculates best Free Form transform to fit given source->target deformation
    /// origin ref grid as box corners ( resolution parameter specifies how to divide box )
    /// Generated from class `MR::FreeFormBestFit`.
    /// This is the const half of the class.
    public class Const_FreeFormBestFit : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FreeFormBestFit(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_Destroy", ExactSpelling = true)]
            extern static void __MR_FreeFormBestFit_Destroy(_Underlying *_this);
            __MR_FreeFormBestFit_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FreeFormBestFit() {Dispose(false);}

        /// Generated from constructor `MR::FreeFormBestFit::FreeFormBestFit`.
        public unsafe Const_FreeFormBestFit(MR._ByValue_FreeFormBestFit _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FreeFormBestFit._Underlying *__MR_FreeFormBestFit_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FreeFormBestFit._Underlying *_other);
            _UnderlyingPtr = __MR_FreeFormBestFit_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// initialize the class, compute cached values and reserve space for matrices
        /// Generated from constructor `MR::FreeFormBestFit::FreeFormBestFit`.
        /// Parameter `resolution` defaults to `Vector3i::diagonal(2)`.
        public unsafe Const_FreeFormBestFit(MR.Const_Box3d box, MR.Const_Vector3i? resolution = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_Construct", ExactSpelling = true)]
            extern static MR.FreeFormBestFit._Underlying *__MR_FreeFormBestFit_Construct(MR.Const_Box3d._Underlying *box, MR.Const_Vector3i._Underlying *resolution);
            _UnderlyingPtr = __MR_FreeFormBestFit_Construct(box._UnderlyingPtr, resolution is not null ? resolution._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FreeFormBestFit::getStabilizer`.
        public unsafe double GetStabilizer()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_getStabilizer", ExactSpelling = true)]
            extern static double __MR_FreeFormBestFit_getStabilizer(_Underlying *_this);
            return __MR_FreeFormBestFit_getStabilizer(_UnderlyingPtr);
        }
    }

    /// Class to accumulate source and target points for free form alignment
    /// Calculates best Free Form transform to fit given source->target deformation
    /// origin ref grid as box corners ( resolution parameter specifies how to divide box )
    /// Generated from class `MR::FreeFormBestFit`.
    /// This is the non-const half of the class.
    public class FreeFormBestFit : Const_FreeFormBestFit
    {
        internal unsafe FreeFormBestFit(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::FreeFormBestFit::FreeFormBestFit`.
        public unsafe FreeFormBestFit(MR._ByValue_FreeFormBestFit _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FreeFormBestFit._Underlying *__MR_FreeFormBestFit_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FreeFormBestFit._Underlying *_other);
            _UnderlyingPtr = __MR_FreeFormBestFit_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// initialize the class, compute cached values and reserve space for matrices
        /// Generated from constructor `MR::FreeFormBestFit::FreeFormBestFit`.
        /// Parameter `resolution` defaults to `Vector3i::diagonal(2)`.
        public unsafe FreeFormBestFit(MR.Const_Box3d box, MR.Const_Vector3i? resolution = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_Construct", ExactSpelling = true)]
            extern static MR.FreeFormBestFit._Underlying *__MR_FreeFormBestFit_Construct(MR.Const_Box3d._Underlying *box, MR.Const_Vector3i._Underlying *resolution);
            _UnderlyingPtr = __MR_FreeFormBestFit_Construct(box._UnderlyingPtr, resolution is not null ? resolution._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FreeFormBestFit::operator=`.
        public unsafe MR.FreeFormBestFit Assign(MR._ByValue_FreeFormBestFit _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FreeFormBestFit._Underlying *__MR_FreeFormBestFit_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FreeFormBestFit._Underlying *_other);
            return new(__MR_FreeFormBestFit_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// add pair of source and target point to accumulator
        /// Generated from method `MR::FreeFormBestFit::addPair`.
        /// Parameter `w` defaults to `1.0`.
        public unsafe void AddPair(MR.Const_Vector3d src, MR.Const_Vector3d tgt, double? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_addPair_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_FreeFormBestFit_addPair_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *src, MR.Const_Vector3d._Underlying *tgt, double *w);
            double __deref_w = w.GetValueOrDefault();
            __MR_FreeFormBestFit_addPair_MR_Vector3d(_UnderlyingPtr, src._UnderlyingPtr, tgt._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// Generated from method `MR::FreeFormBestFit::addPair`.
        /// Parameter `w` defaults to `1.0f`.
        public unsafe void AddPair(MR.Const_Vector3f src, MR.Const_Vector3f tgt, float? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_addPair_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_FreeFormBestFit_addPair_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *src, MR.Const_Vector3f._Underlying *tgt, float *w);
            float __deref_w = w.GetValueOrDefault();
            __MR_FreeFormBestFit_addPair_MR_Vector3f(_UnderlyingPtr, src._UnderlyingPtr, tgt._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// adds other instance of FreeFormBestFit if it has same ref grid
        /// Generated from method `MR::FreeFormBestFit::addOther`.
        public unsafe void AddOther(MR.Const_FreeFormBestFit other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_addOther", ExactSpelling = true)]
            extern static void __MR_FreeFormBestFit_addOther(_Underlying *_this, MR.Const_FreeFormBestFit._Underlying *other);
            __MR_FreeFormBestFit_addOther(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// stabilizer adds additional weights to keep result grid closer to origins
        /// recommended values (0;1], but it can be higher
        /// Generated from method `MR::FreeFormBestFit::setStabilizer`.
        public unsafe void SetStabilizer(double stabilizer)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_setStabilizer", ExactSpelling = true)]
            extern static void __MR_FreeFormBestFit_setStabilizer(_Underlying *_this, double stabilizer);
            __MR_FreeFormBestFit_setStabilizer(_UnderlyingPtr, stabilizer);
        }

        /// finds best grid points positions to align source points to target points
        /// Generated from method `MR::FreeFormBestFit::findBestDeformationReferenceGrid`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> FindBestDeformationReferenceGrid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FreeFormBestFit_findBestDeformationReferenceGrid", ExactSpelling = true)]
            extern static MR.Std.Vector_MRVector3f._Underlying *__MR_FreeFormBestFit_findBestDeformationReferenceGrid(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_FreeFormBestFit_findBestDeformationReferenceGrid(_UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `FreeFormBestFit` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FreeFormBestFit`/`Const_FreeFormBestFit` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FreeFormBestFit
    {
        internal readonly Const_FreeFormBestFit? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FreeFormBestFit(Const_FreeFormBestFit new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FreeFormBestFit(Const_FreeFormBestFit arg) {return new(arg);}
        public _ByValue_FreeFormBestFit(MR.Misc._Moved<FreeFormBestFit> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FreeFormBestFit(MR.Misc._Moved<FreeFormBestFit> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FreeFormBestFit` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FreeFormBestFit`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FreeFormBestFit`/`Const_FreeFormBestFit` directly.
    public class _InOptMut_FreeFormBestFit
    {
        public FreeFormBestFit? Opt;

        public _InOptMut_FreeFormBestFit() {}
        public _InOptMut_FreeFormBestFit(FreeFormBestFit value) {Opt = value;}
        public static implicit operator _InOptMut_FreeFormBestFit(FreeFormBestFit value) {return new(value);}
    }

    /// This is used for optional parameters of class `FreeFormBestFit` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FreeFormBestFit`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FreeFormBestFit`/`Const_FreeFormBestFit` to pass it to the function.
    public class _InOptConst_FreeFormBestFit
    {
        public Const_FreeFormBestFit? Opt;

        public _InOptConst_FreeFormBestFit() {}
        public _InOptConst_FreeFormBestFit(Const_FreeFormBestFit value) {Opt = value;}
        public static implicit operator _InOptConst_FreeFormBestFit(Const_FreeFormBestFit value) {return new(value);}
    }

    /// Returns positions of grid points in given box with given resolution 
    /// Generated from function `MR::makeFreeFormOriginGrid`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> MakeFreeFormOriginGrid(MR.Const_Box3f box, MR.Const_Vector3i resolution)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeFreeFormOriginGrid", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_makeFreeFormOriginGrid(MR.Const_Box3f._Underlying *box, MR.Const_Vector3i._Underlying *resolution);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_makeFreeFormOriginGrid(box._UnderlyingPtr, resolution._UnderlyingPtr), is_owning: true));
    }

    // Calculates best Free Form transform to fit given source->target deformation
    // origin ref grid as box corners ( resolution parameter specifies how to divide box )
    // samplesToBox - if set used to transform source and target points to box space
    // returns new positions of ref grid
    /// Generated from function `MR::findBestFreeformDeformation`.
    /// Parameter `resolution` defaults to `Vector3i::diagonal(2)`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> FindBestFreeformDeformation(MR.Const_Box3f box, MR.Std.Const_Vector_MRVector3f source, MR.Std.Const_Vector_MRVector3f target, MR.Const_Vector3i? resolution = null, MR.Const_AffineXf3f? samplesToBox = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findBestFreeformDeformation", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_findBestFreeformDeformation(MR.Const_Box3f._Underlying *box, MR.Std.Const_Vector_MRVector3f._Underlying *source, MR.Std.Const_Vector_MRVector3f._Underlying *target, MR.Const_Vector3i._Underlying *resolution, MR.Const_AffineXf3f._Underlying *samplesToBox);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_findBestFreeformDeformation(box._UnderlyingPtr, source._UnderlyingPtr, target._UnderlyingPtr, resolution is not null ? resolution._UnderlyingPtr : null, samplesToBox is not null ? samplesToBox._UnderlyingPtr : null), is_owning: true));
    }
}
