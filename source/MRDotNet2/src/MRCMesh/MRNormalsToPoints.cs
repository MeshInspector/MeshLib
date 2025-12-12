public static partial class MR
{
    /// The purpose of this class is to update vertex positions given target triangle normals;
    /// see the article "Static/Dynamic Filtering for Mesh Geometry"
    /// Generated from class `MR::NormalsToPoints`.
    /// This is the const half of the class.
    public class Const_NormalsToPoints : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NormalsToPoints(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_Destroy", ExactSpelling = true)]
            extern static void __MR_NormalsToPoints_Destroy(_Underlying *_this);
            __MR_NormalsToPoints_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NormalsToPoints() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NormalsToPoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NormalsToPoints._Underlying *__MR_NormalsToPoints_DefaultConstruct();
            _UnderlyingPtr = __MR_NormalsToPoints_DefaultConstruct();
        }

        /// Generated from constructor `MR::NormalsToPoints::NormalsToPoints`.
        public unsafe Const_NormalsToPoints(MR._ByValue_NormalsToPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NormalsToPoints._Underlying *__MR_NormalsToPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NormalsToPoints._Underlying *_other);
            _UnderlyingPtr = __MR_NormalsToPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        // pImpl
        /// Generated from class `MR::NormalsToPoints::ISolver`.
        /// This is the const half of the class.
        public class Const_ISolver : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ISolver(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_ISolver_Destroy", ExactSpelling = true)]
                extern static void __MR_NormalsToPoints_ISolver_Destroy(_Underlying *_this);
                __MR_NormalsToPoints_ISolver_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ISolver() {Dispose(false);}
        }

        // pImpl
        /// Generated from class `MR::NormalsToPoints::ISolver`.
        /// This is the non-const half of the class.
        public class ISolver : Const_ISolver
        {
            internal unsafe ISolver(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// Generated from method `MR::NormalsToPoints::ISolver::prepare`.
            public unsafe void Prepare(MR.Const_MeshTopology topology, float guideWeight)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_ISolver_prepare", ExactSpelling = true)]
                extern static void __MR_NormalsToPoints_ISolver_prepare(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, float guideWeight);
                __MR_NormalsToPoints_ISolver_prepare(_UnderlyingPtr, topology._UnderlyingPtr, guideWeight);
            }

            /// Generated from method `MR::NormalsToPoints::ISolver::run`.
            public unsafe void Run(MR.Const_VertCoords guide, MR.Const_FaceNormals normals, MR.VertCoords points, float maxInitialDistSq)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_ISolver_run", ExactSpelling = true)]
                extern static void __MR_NormalsToPoints_ISolver_run(_Underlying *_this, MR.Const_VertCoords._Underlying *guide, MR.Const_FaceNormals._Underlying *normals, MR.VertCoords._Underlying *points, float maxInitialDistSq);
                __MR_NormalsToPoints_ISolver_run(_UnderlyingPtr, guide._UnderlyingPtr, normals._UnderlyingPtr, points._UnderlyingPtr, maxInitialDistSq);
            }
        }

        /// This is used for optional parameters of class `ISolver` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ISolver`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ISolver`/`Const_ISolver` directly.
        public class _InOptMut_ISolver
        {
            public ISolver? Opt;

            public _InOptMut_ISolver() {}
            public _InOptMut_ISolver(ISolver value) {Opt = value;}
            public static implicit operator _InOptMut_ISolver(ISolver value) {return new(value);}
        }

        /// This is used for optional parameters of class `ISolver` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ISolver`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ISolver`/`Const_ISolver` to pass it to the function.
        public class _InOptConst_ISolver
        {
            public Const_ISolver? Opt;

            public _InOptConst_ISolver() {}
            public _InOptConst_ISolver(Const_ISolver value) {Opt = value;}
            public static implicit operator _InOptConst_ISolver(Const_ISolver value) {return new(value);}
        }
    }

    /// The purpose of this class is to update vertex positions given target triangle normals;
    /// see the article "Static/Dynamic Filtering for Mesh Geometry"
    /// Generated from class `MR::NormalsToPoints`.
    /// This is the non-const half of the class.
    public class NormalsToPoints : Const_NormalsToPoints
    {
        internal unsafe NormalsToPoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe NormalsToPoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NormalsToPoints._Underlying *__MR_NormalsToPoints_DefaultConstruct();
            _UnderlyingPtr = __MR_NormalsToPoints_DefaultConstruct();
        }

        /// Generated from constructor `MR::NormalsToPoints::NormalsToPoints`.
        public unsafe NormalsToPoints(MR._ByValue_NormalsToPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NormalsToPoints._Underlying *__MR_NormalsToPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NormalsToPoints._Underlying *_other);
            _UnderlyingPtr = __MR_NormalsToPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::NormalsToPoints::operator=`.
        public unsafe MR.NormalsToPoints Assign(MR._ByValue_NormalsToPoints _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NormalsToPoints._Underlying *__MR_NormalsToPoints_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.NormalsToPoints._Underlying *_other);
            return new(__MR_NormalsToPoints_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// builds linear system and prepares a solver for it;
        /// please call it only once for mesh, and then run as many times as you like
        /// \param guideWeight how much resulting points must be attracted to initial points, must be > 0
        /// Generated from method `MR::NormalsToPoints::prepare`.
        /// Parameter `guideWeight` defaults to `1`.
        public unsafe void Prepare(MR.Const_MeshTopology topology, float? guideWeight = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_prepare", ExactSpelling = true)]
            extern static void __MR_NormalsToPoints_prepare(_Underlying *_this, MR.Const_MeshTopology._Underlying *topology, float *guideWeight);
            float __deref_guideWeight = guideWeight.GetValueOrDefault();
            __MR_NormalsToPoints_prepare(_UnderlyingPtr, topology._UnderlyingPtr, guideWeight.HasValue ? &__deref_guideWeight : null);
        }

        /// performs one iteration consisting of projection of all triangles on planes with given normals and finding best points from them
        /// \param guide target vertex positions to avoid under-determined system
        /// \param normals target face normals
        /// \param points initial approximation on input, updated approximation on output
        /// \param maxInitialDistSq the maximum squared distance between a point and its position in (guide)
        /// Generated from method `MR::NormalsToPoints::run`.
        public unsafe void Run(MR.Const_VertCoords guide, MR.Const_FaceNormals normals, MR.VertCoords points)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_run_3", ExactSpelling = true)]
            extern static void __MR_NormalsToPoints_run_3(_Underlying *_this, MR.Const_VertCoords._Underlying *guide, MR.Const_FaceNormals._Underlying *normals, MR.VertCoords._Underlying *points);
            __MR_NormalsToPoints_run_3(_UnderlyingPtr, guide._UnderlyingPtr, normals._UnderlyingPtr, points._UnderlyingPtr);
        }

        /// Generated from method `MR::NormalsToPoints::run`.
        public unsafe void Run(MR.Const_VertCoords guide, MR.Const_FaceNormals normals, MR.VertCoords points, float maxInitialDistSq)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NormalsToPoints_run_4", ExactSpelling = true)]
            extern static void __MR_NormalsToPoints_run_4(_Underlying *_this, MR.Const_VertCoords._Underlying *guide, MR.Const_FaceNormals._Underlying *normals, MR.VertCoords._Underlying *points, float maxInitialDistSq);
            __MR_NormalsToPoints_run_4(_UnderlyingPtr, guide._UnderlyingPtr, normals._UnderlyingPtr, points._UnderlyingPtr, maxInitialDistSq);
        }
    }

    /// This is used as a function parameter when the underlying function receives `NormalsToPoints` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `NormalsToPoints`/`Const_NormalsToPoints` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_NormalsToPoints
    {
        internal readonly Const_NormalsToPoints? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_NormalsToPoints() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_NormalsToPoints(MR.Misc._Moved<NormalsToPoints> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_NormalsToPoints(MR.Misc._Moved<NormalsToPoints> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `NormalsToPoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NormalsToPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NormalsToPoints`/`Const_NormalsToPoints` directly.
    public class _InOptMut_NormalsToPoints
    {
        public NormalsToPoints? Opt;

        public _InOptMut_NormalsToPoints() {}
        public _InOptMut_NormalsToPoints(NormalsToPoints value) {Opt = value;}
        public static implicit operator _InOptMut_NormalsToPoints(NormalsToPoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `NormalsToPoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NormalsToPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NormalsToPoints`/`Const_NormalsToPoints` to pass it to the function.
    public class _InOptConst_NormalsToPoints
    {
        public Const_NormalsToPoints? Opt;

        public _InOptConst_NormalsToPoints() {}
        public _InOptConst_NormalsToPoints(Const_NormalsToPoints value) {Opt = value;}
        public static implicit operator _InOptConst_NormalsToPoints(Const_NormalsToPoints value) {return new(value);}
    }
}
