public static partial class MR
{
    /// Generated from class `MR::MeshRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RelaxParams`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshApproxRelaxParams`
    ///     `MR::MeshEqualizeTriAreasParams`
    /// This is the const half of the class.
    public class Const_MeshRelaxParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshRelaxParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshRelaxParams_Destroy(_Underlying *_this);
            __MR_MeshRelaxParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshRelaxParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RelaxParams(Const_MeshRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.Const_RelaxParams._Underlying *__MR_MeshRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.Const_RelaxParams ret = new(__MR_MeshRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
        public unsafe bool HardSmoothTetrahedrons
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_hardSmoothTetrahedrons", ExactSpelling = true)]
                extern static bool *__MR_MeshRelaxParams_Get_hardSmoothTetrahedrons(_Underlying *_this);
                return *__MR_MeshRelaxParams_Get_hardSmoothTetrahedrons(_UnderlyingPtr);
            }
        }

        /// weight for each vertex. By default, all the vertices have equal weights.
        public unsafe ref readonly void * Weights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_weights", ExactSpelling = true)]
                extern static void **__MR_MeshRelaxParams_Get_weights(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_Get_weights(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_MeshRelaxParams_Get_iterations(_Underlying *_this);
                return *__MR_MeshRelaxParams_Get_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_MeshRelaxParams_Get_region(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_Get_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public unsafe float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_force", ExactSpelling = true)]
                extern static float *__MR_MeshRelaxParams_Get_force(_Underlying *_this);
                return *__MR_MeshRelaxParams_Get_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_MeshRelaxParams_Get_limitNearInitial(_Underlying *_this);
                return *__MR_MeshRelaxParams_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_MeshRelaxParams_Get_maxInitialDist(_Underlying *_this);
                return *__MR_MeshRelaxParams_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshRelaxParams::MeshRelaxParams`.
        public unsafe Const_MeshRelaxParams(MR.Const_MeshRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshRelaxParams_ConstructFromAnother(MR.MeshRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RelaxParams`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshApproxRelaxParams`
    ///     `MR::MeshEqualizeTriAreasParams`
    /// This is the non-const half of the class.
    public class MeshRelaxParams : Const_MeshRelaxParams
    {
        internal unsafe MeshRelaxParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RelaxParams(MeshRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_MeshRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.RelaxParams ret = new(__MR_MeshRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
        public new unsafe ref bool HardSmoothTetrahedrons
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_hardSmoothTetrahedrons", ExactSpelling = true)]
                extern static bool *__MR_MeshRelaxParams_GetMutable_hardSmoothTetrahedrons(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_hardSmoothTetrahedrons(_UnderlyingPtr);
            }
        }

        /// weight for each vertex. By default, all the vertices have equal weights.
        public new unsafe ref readonly void * Weights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_weights", ExactSpelling = true)]
                extern static void **__MR_MeshRelaxParams_GetMutable_weights(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_weights(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_MeshRelaxParams_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_MeshRelaxParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public new unsafe ref float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_force", ExactSpelling = true)]
                extern static float *__MR_MeshRelaxParams_GetMutable_force(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_MeshRelaxParams_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_MeshRelaxParams_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_MeshRelaxParams_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshRelaxParams::MeshRelaxParams`.
        public unsafe MeshRelaxParams(MR.Const_MeshRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshRelaxParams_ConstructFromAnother(MR.MeshRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshRelaxParams::operator=`.
        public unsafe MR.MeshRelaxParams Assign(MR.Const_MeshRelaxParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshRelaxParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshRelaxParams_AssignFromAnother(_Underlying *_this, MR.MeshRelaxParams._Underlying *_other);
            return new(__MR_MeshRelaxParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshRelaxParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshRelaxParams`/`Const_MeshRelaxParams` directly.
    public class _InOptMut_MeshRelaxParams
    {
        public MeshRelaxParams? Opt;

        public _InOptMut_MeshRelaxParams() {}
        public _InOptMut_MeshRelaxParams(MeshRelaxParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshRelaxParams(MeshRelaxParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshRelaxParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshRelaxParams`/`Const_MeshRelaxParams` to pass it to the function.
    public class _InOptConst_MeshRelaxParams
    {
        public Const_MeshRelaxParams? Opt;

        public _InOptConst_MeshRelaxParams() {}
        public _InOptConst_MeshRelaxParams(Const_MeshRelaxParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshRelaxParams(Const_MeshRelaxParams value) {return new(value);}
    }

    /// Generated from class `MR::MeshEqualizeTriAreasParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::RelaxParams`
    /// This is the const half of the class.
    public class Const_MeshEqualizeTriAreasParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshEqualizeTriAreasParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshEqualizeTriAreasParams_Destroy(_Underlying *_this);
            __MR_MeshEqualizeTriAreasParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshEqualizeTriAreasParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RelaxParams(Const_MeshEqualizeTriAreasParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.Const_RelaxParams._Underlying *__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.Const_RelaxParams ret = new(__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_MeshRelaxParams(Const_MeshEqualizeTriAreasParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_UpcastTo_MR_MeshRelaxParams", ExactSpelling = true)]
            extern static MR.Const_MeshRelaxParams._Underlying *__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_MeshRelaxParams(_Underlying *_this);
            MR.Const_MeshRelaxParams ret = new(__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_MeshRelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// if true prevents the surface from shrinkage after many iterations;
        /// technically it is done by solving the same task in the plane orthogonal to normal direction
        public unsafe bool NoShrinkage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_noShrinkage", ExactSpelling = true)]
                extern static bool *__MR_MeshEqualizeTriAreasParams_Get_noShrinkage(_Underlying *_this);
                return *__MR_MeshEqualizeTriAreasParams_Get_noShrinkage(_UnderlyingPtr);
            }
        }

        /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
        public unsafe bool HardSmoothTetrahedrons
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_hardSmoothTetrahedrons", ExactSpelling = true)]
                extern static bool *__MR_MeshEqualizeTriAreasParams_Get_hardSmoothTetrahedrons(_Underlying *_this);
                return *__MR_MeshEqualizeTriAreasParams_Get_hardSmoothTetrahedrons(_UnderlyingPtr);
            }
        }

        /// weight for each vertex. By default, all the vertices have equal weights.
        public unsafe ref readonly void * Weights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_weights", ExactSpelling = true)]
                extern static void **__MR_MeshEqualizeTriAreasParams_Get_weights(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_Get_weights(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_MeshEqualizeTriAreasParams_Get_iterations(_Underlying *_this);
                return *__MR_MeshEqualizeTriAreasParams_Get_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_MeshEqualizeTriAreasParams_Get_region(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_Get_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public unsafe float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_force", ExactSpelling = true)]
                extern static float *__MR_MeshEqualizeTriAreasParams_Get_force(_Underlying *_this);
                return *__MR_MeshEqualizeTriAreasParams_Get_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_MeshEqualizeTriAreasParams_Get_limitNearInitial(_Underlying *_this);
                return *__MR_MeshEqualizeTriAreasParams_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_MeshEqualizeTriAreasParams_Get_maxInitialDist(_Underlying *_this);
                return *__MR_MeshEqualizeTriAreasParams_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshEqualizeTriAreasParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshEqualizeTriAreasParams._Underlying *__MR_MeshEqualizeTriAreasParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshEqualizeTriAreasParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshEqualizeTriAreasParams::MeshEqualizeTriAreasParams`.
        public unsafe Const_MeshEqualizeTriAreasParams(MR.Const_MeshEqualizeTriAreasParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshEqualizeTriAreasParams._Underlying *__MR_MeshEqualizeTriAreasParams_ConstructFromAnother(MR.MeshEqualizeTriAreasParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshEqualizeTriAreasParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshEqualizeTriAreasParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::RelaxParams`
    /// This is the non-const half of the class.
    public class MeshEqualizeTriAreasParams : Const_MeshEqualizeTriAreasParams
    {
        internal unsafe MeshEqualizeTriAreasParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RelaxParams(MeshEqualizeTriAreasParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.RelaxParams ret = new(__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.MeshRelaxParams(MeshEqualizeTriAreasParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_UpcastTo_MR_MeshRelaxParams", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_MeshRelaxParams(_Underlying *_this);
            MR.MeshRelaxParams ret = new(__MR_MeshEqualizeTriAreasParams_UpcastTo_MR_MeshRelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// if true prevents the surface from shrinkage after many iterations;
        /// technically it is done by solving the same task in the plane orthogonal to normal direction
        public new unsafe ref bool NoShrinkage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_noShrinkage", ExactSpelling = true)]
                extern static bool *__MR_MeshEqualizeTriAreasParams_GetMutable_noShrinkage(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_noShrinkage(_UnderlyingPtr);
            }
        }

        /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
        public new unsafe ref bool HardSmoothTetrahedrons
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_hardSmoothTetrahedrons", ExactSpelling = true)]
                extern static bool *__MR_MeshEqualizeTriAreasParams_GetMutable_hardSmoothTetrahedrons(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_hardSmoothTetrahedrons(_UnderlyingPtr);
            }
        }

        /// weight for each vertex. By default, all the vertices have equal weights.
        public new unsafe ref readonly void * Weights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_weights", ExactSpelling = true)]
                extern static void **__MR_MeshEqualizeTriAreasParams_GetMutable_weights(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_weights(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_MeshEqualizeTriAreasParams_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_MeshEqualizeTriAreasParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public new unsafe ref float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_force", ExactSpelling = true)]
                extern static float *__MR_MeshEqualizeTriAreasParams_GetMutable_force(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_MeshEqualizeTriAreasParams_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_MeshEqualizeTriAreasParams_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_MeshEqualizeTriAreasParams_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshEqualizeTriAreasParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshEqualizeTriAreasParams._Underlying *__MR_MeshEqualizeTriAreasParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshEqualizeTriAreasParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshEqualizeTriAreasParams::MeshEqualizeTriAreasParams`.
        public unsafe MeshEqualizeTriAreasParams(MR.Const_MeshEqualizeTriAreasParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshEqualizeTriAreasParams._Underlying *__MR_MeshEqualizeTriAreasParams_ConstructFromAnother(MR.MeshEqualizeTriAreasParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshEqualizeTriAreasParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshEqualizeTriAreasParams::operator=`.
        public unsafe MR.MeshEqualizeTriAreasParams Assign(MR.Const_MeshEqualizeTriAreasParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshEqualizeTriAreasParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshEqualizeTriAreasParams._Underlying *__MR_MeshEqualizeTriAreasParams_AssignFromAnother(_Underlying *_this, MR.MeshEqualizeTriAreasParams._Underlying *_other);
            return new(__MR_MeshEqualizeTriAreasParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshEqualizeTriAreasParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshEqualizeTriAreasParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshEqualizeTriAreasParams`/`Const_MeshEqualizeTriAreasParams` directly.
    public class _InOptMut_MeshEqualizeTriAreasParams
    {
        public MeshEqualizeTriAreasParams? Opt;

        public _InOptMut_MeshEqualizeTriAreasParams() {}
        public _InOptMut_MeshEqualizeTriAreasParams(MeshEqualizeTriAreasParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshEqualizeTriAreasParams(MeshEqualizeTriAreasParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshEqualizeTriAreasParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshEqualizeTriAreasParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshEqualizeTriAreasParams`/`Const_MeshEqualizeTriAreasParams` to pass it to the function.
    public class _InOptConst_MeshEqualizeTriAreasParams
    {
        public Const_MeshEqualizeTriAreasParams? Opt;

        public _InOptConst_MeshEqualizeTriAreasParams() {}
        public _InOptConst_MeshEqualizeTriAreasParams(Const_MeshEqualizeTriAreasParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshEqualizeTriAreasParams(Const_MeshEqualizeTriAreasParams value) {return new(value);}
    }

    /// Generated from class `MR::MeshApproxRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::RelaxParams`
    /// This is the const half of the class.
    public class Const_MeshApproxRelaxParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshApproxRelaxParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshApproxRelaxParams_Destroy(_Underlying *_this);
            __MR_MeshApproxRelaxParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshApproxRelaxParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RelaxParams(Const_MeshApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.Const_RelaxParams._Underlying *__MR_MeshApproxRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.Const_RelaxParams ret = new(__MR_MeshApproxRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_MeshRelaxParams(Const_MeshApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_UpcastTo_MR_MeshRelaxParams", ExactSpelling = true)]
            extern static MR.Const_MeshRelaxParams._Underlying *__MR_MeshApproxRelaxParams_UpcastTo_MR_MeshRelaxParams(_Underlying *_this);
            MR.Const_MeshRelaxParams ret = new(__MR_MeshApproxRelaxParams_UpcastTo_MR_MeshRelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// radius to find neighbors by surface
        /// 0.0f - default = 1e-3 * sqrt(surface area)
        public unsafe float SurfaceDilateRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_surfaceDilateRadius", ExactSpelling = true)]
                extern static float *__MR_MeshApproxRelaxParams_Get_surfaceDilateRadius(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_surfaceDilateRadius(_UnderlyingPtr);
            }
        }

        public unsafe MR.RelaxApproxType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_type", ExactSpelling = true)]
                extern static MR.RelaxApproxType *__MR_MeshApproxRelaxParams_Get_type(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_type(_UnderlyingPtr);
            }
        }

        /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
        public unsafe bool HardSmoothTetrahedrons
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_hardSmoothTetrahedrons", ExactSpelling = true)]
                extern static bool *__MR_MeshApproxRelaxParams_Get_hardSmoothTetrahedrons(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_hardSmoothTetrahedrons(_UnderlyingPtr);
            }
        }

        /// weight for each vertex. By default, all the vertices have equal weights.
        public unsafe ref readonly void * Weights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_weights", ExactSpelling = true)]
                extern static void **__MR_MeshApproxRelaxParams_Get_weights(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_Get_weights(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_MeshApproxRelaxParams_Get_iterations(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_MeshApproxRelaxParams_Get_region(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_Get_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public unsafe float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_force", ExactSpelling = true)]
                extern static float *__MR_MeshApproxRelaxParams_Get_force(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_MeshApproxRelaxParams_Get_limitNearInitial(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_MeshApproxRelaxParams_Get_maxInitialDist(_Underlying *_this);
                return *__MR_MeshApproxRelaxParams_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshApproxRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshApproxRelaxParams._Underlying *__MR_MeshApproxRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshApproxRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshApproxRelaxParams::MeshApproxRelaxParams`.
        public unsafe Const_MeshApproxRelaxParams(MR.Const_MeshApproxRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshApproxRelaxParams._Underlying *__MR_MeshApproxRelaxParams_ConstructFromAnother(MR.MeshApproxRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshApproxRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshApproxRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::RelaxParams`
    /// This is the non-const half of the class.
    public class MeshApproxRelaxParams : Const_MeshApproxRelaxParams
    {
        internal unsafe MeshApproxRelaxParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RelaxParams(MeshApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_MeshApproxRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.RelaxParams ret = new(__MR_MeshApproxRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.MeshRelaxParams(MeshApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_UpcastTo_MR_MeshRelaxParams", ExactSpelling = true)]
            extern static MR.MeshRelaxParams._Underlying *__MR_MeshApproxRelaxParams_UpcastTo_MR_MeshRelaxParams(_Underlying *_this);
            MR.MeshRelaxParams ret = new(__MR_MeshApproxRelaxParams_UpcastTo_MR_MeshRelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// radius to find neighbors by surface
        /// 0.0f - default = 1e-3 * sqrt(surface area)
        public new unsafe ref float SurfaceDilateRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_surfaceDilateRadius", ExactSpelling = true)]
                extern static float *__MR_MeshApproxRelaxParams_GetMutable_surfaceDilateRadius(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_surfaceDilateRadius(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.RelaxApproxType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_type", ExactSpelling = true)]
                extern static MR.RelaxApproxType *__MR_MeshApproxRelaxParams_GetMutable_type(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_type(_UnderlyingPtr);
            }
        }

        /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
        public new unsafe ref bool HardSmoothTetrahedrons
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_hardSmoothTetrahedrons", ExactSpelling = true)]
                extern static bool *__MR_MeshApproxRelaxParams_GetMutable_hardSmoothTetrahedrons(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_hardSmoothTetrahedrons(_UnderlyingPtr);
            }
        }

        /// weight for each vertex. By default, all the vertices have equal weights.
        public new unsafe ref readonly void * Weights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_weights", ExactSpelling = true)]
                extern static void **__MR_MeshApproxRelaxParams_GetMutable_weights(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_weights(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_MeshApproxRelaxParams_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_MeshApproxRelaxParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public new unsafe ref float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_force", ExactSpelling = true)]
                extern static float *__MR_MeshApproxRelaxParams_GetMutable_force(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_MeshApproxRelaxParams_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_MeshApproxRelaxParams_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_MeshApproxRelaxParams_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshApproxRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshApproxRelaxParams._Underlying *__MR_MeshApproxRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshApproxRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshApproxRelaxParams::MeshApproxRelaxParams`.
        public unsafe MeshApproxRelaxParams(MR.Const_MeshApproxRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshApproxRelaxParams._Underlying *__MR_MeshApproxRelaxParams_ConstructFromAnother(MR.MeshApproxRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshApproxRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshApproxRelaxParams::operator=`.
        public unsafe MR.MeshApproxRelaxParams Assign(MR.Const_MeshApproxRelaxParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshApproxRelaxParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshApproxRelaxParams._Underlying *__MR_MeshApproxRelaxParams_AssignFromAnother(_Underlying *_this, MR.MeshApproxRelaxParams._Underlying *_other);
            return new(__MR_MeshApproxRelaxParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshApproxRelaxParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshApproxRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshApproxRelaxParams`/`Const_MeshApproxRelaxParams` directly.
    public class _InOptMut_MeshApproxRelaxParams
    {
        public MeshApproxRelaxParams? Opt;

        public _InOptMut_MeshApproxRelaxParams() {}
        public _InOptMut_MeshApproxRelaxParams(MeshApproxRelaxParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshApproxRelaxParams(MeshApproxRelaxParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshApproxRelaxParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshApproxRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshApproxRelaxParams`/`Const_MeshApproxRelaxParams` to pass it to the function.
    public class _InOptConst_MeshApproxRelaxParams
    {
        public Const_MeshApproxRelaxParams? Opt;

        public _InOptConst_MeshApproxRelaxParams() {}
        public _InOptConst_MeshApproxRelaxParams(Const_MeshApproxRelaxParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshApproxRelaxParams(Const_MeshApproxRelaxParams value) {return new(value);}
    }

    /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
    /// \return true if was finished successfully, false if was interrupted by progress callback
    /// Generated from function `MR::relax`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool Relax(MR.Mesh mesh, MR.Const_MeshRelaxParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relax_3_MR_Mesh", ExactSpelling = true)]
        extern static byte __MR_relax_3_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_MeshRelaxParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relax_3_MR_Mesh(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::relax`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool Relax(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_MeshRelaxParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relax_4", ExactSpelling = true)]
        extern static byte __MR_relax_4(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_MeshRelaxParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relax_4(topology._UnderlyingPtr, points._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// computes position of a vertex, when all neighbor triangles have almost equal areas,
    /// more precisely it minimizes sum_i (area_i)^2 by adjusting the position of this vertex only
    /// Generated from function `MR::vertexPosEqualNeiAreas`.
    public static unsafe MR.Vector3f VertexPosEqualNeiAreas(MR.Const_Mesh mesh, MR.VertId v, bool noShrinkage)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vertexPosEqualNeiAreas_3", ExactSpelling = true)]
        extern static MR.Vector3f __MR_vertexPosEqualNeiAreas_3(MR.Const_Mesh._Underlying *mesh, MR.VertId v, byte noShrinkage);
        return __MR_vertexPosEqualNeiAreas_3(mesh._UnderlyingPtr, v, noShrinkage ? (byte)1 : (byte)0);
    }

    /// Generated from function `MR::vertexPosEqualNeiAreas`.
    public static unsafe MR.Vector3f VertexPosEqualNeiAreas(MR.Const_MeshTopology topology, MR.Const_VertCoords points, MR.VertId v, bool noShrinkage)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_vertexPosEqualNeiAreas_4", ExactSpelling = true)]
        extern static MR.Vector3f __MR_vertexPosEqualNeiAreas_4(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertCoords._Underlying *points, MR.VertId v, byte noShrinkage);
        return __MR_vertexPosEqualNeiAreas_4(topology._UnderlyingPtr, points._UnderlyingPtr, v, noShrinkage ? (byte)1 : (byte)0);
    }

    /// applies given number of iterations with movement toward vertexPosEqualNeiAreas() to the whole mesh ( or some region if it is specified )
    /// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
    /// Generated from function `MR::equalizeTriAreas`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool EqualizeTriAreas(MR.Mesh mesh, MR.Const_MeshEqualizeTriAreasParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equalizeTriAreas_3", ExactSpelling = true)]
        extern static byte __MR_equalizeTriAreas_3(MR.Mesh._Underlying *mesh, MR.Const_MeshEqualizeTriAreasParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_equalizeTriAreas_3(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::equalizeTriAreas`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool EqualizeTriAreas(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_MeshEqualizeTriAreasParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equalizeTriAreas_4", ExactSpelling = true)]
        extern static byte __MR_equalizeTriAreas_4(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_MeshEqualizeTriAreasParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_equalizeTriAreas_4(topology._UnderlyingPtr, points._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified ) \n
    /// do not really keeps volume but tries hard
    /// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
    /// Generated from function `MR::relaxKeepVolume`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool RelaxKeepVolume(MR.Mesh mesh, MR.Const_MeshRelaxParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relaxKeepVolume_3_MR_Mesh", ExactSpelling = true)]
        extern static byte __MR_relaxKeepVolume_3_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_MeshRelaxParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relaxKeepVolume_3_MR_Mesh(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::relaxKeepVolume`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool RelaxKeepVolume(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_MeshRelaxParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relaxKeepVolume_4", ExactSpelling = true)]
        extern static byte __MR_relaxKeepVolume_4(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_MeshRelaxParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relaxKeepVolume_4(topology._UnderlyingPtr, points._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
    /// approx neighborhoods
    /// \return true if the operation completed successfully, and false if it was interrupted by the progress callback.
    /// Generated from function `MR::relaxApprox`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool RelaxApprox(MR.Mesh mesh, MR.Const_MeshApproxRelaxParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relaxApprox_3_MR_Mesh", ExactSpelling = true)]
        extern static byte __MR_relaxApprox_3_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_MeshApproxRelaxParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relaxApprox_3_MR_Mesh(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// Generated from function `MR::relaxApprox`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool RelaxApprox(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_MeshApproxRelaxParams? params_ = null, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relaxApprox_4", ExactSpelling = true)]
        extern static byte __MR_relaxApprox_4(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_MeshApproxRelaxParams._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relaxApprox_4(topology._UnderlyingPtr, points._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// applies at most given number of relaxation iterations the spikes detected by given threshold
    /// Generated from function `MR::removeSpikes`.
    public static unsafe void RemoveSpikes(MR.Mesh mesh, int maxIterations, float minSumAngle, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_removeSpikes_4", ExactSpelling = true)]
        extern static void __MR_removeSpikes_4(MR.Mesh._Underlying *mesh, int maxIterations, float minSumAngle, MR.Const_VertBitSet._Underlying *region);
        __MR_removeSpikes_4(mesh._UnderlyingPtr, maxIterations, minSumAngle, region is not null ? region._UnderlyingPtr : null);
    }

    /// Generated from function `MR::removeSpikes`.
    public static unsafe void RemoveSpikes(MR.Const_MeshTopology topology, MR.VertCoords points, int maxIterations, float minSumAngle, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_removeSpikes_5", ExactSpelling = true)]
        extern static void __MR_removeSpikes_5(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, int maxIterations, float minSumAngle, MR.Const_VertBitSet._Underlying *region);
        __MR_removeSpikes_5(topology._UnderlyingPtr, points._UnderlyingPtr, maxIterations, minSumAngle, region is not null ? region._UnderlyingPtr : null);
    }

    /// given a region of faces on the mesh, moves boundary vertices of the region
    /// to make the region contour much smoother with minor optimization of mesh topology near region boundary;
    /// \param numIters >= 1 how many times to run the algorithm to achieve a better quality,
    /// solution is typically oscillates back and forth so even number of iterations is recommended
    /// Generated from function `MR::smoothRegionBoundary`.
    /// Parameter `numIters` defaults to `4`.
    public static unsafe void SmoothRegionBoundary(MR.Mesh mesh, MR.Const_FaceBitSet regionFaces, int? numIters = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_smoothRegionBoundary", ExactSpelling = true)]
        extern static void __MR_smoothRegionBoundary(MR.Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *regionFaces, int *numIters);
        int __deref_numIters = numIters.GetValueOrDefault();
        __MR_smoothRegionBoundary(mesh._UnderlyingPtr, regionFaces._UnderlyingPtr, numIters.HasValue ? &__deref_numIters : null);
    }

    /// move all region vertices with exactly three neighbor vertices in the center of the neighbors
    /// Generated from function `MR::hardSmoothTetrahedrons`.
    public static unsafe void HardSmoothTetrahedrons(MR.Mesh mesh, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hardSmoothTetrahedrons_2", ExactSpelling = true)]
        extern static void __MR_hardSmoothTetrahedrons_2(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
        __MR_hardSmoothTetrahedrons_2(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }

    /// Generated from function `MR::hardSmoothTetrahedrons`.
    public static unsafe void HardSmoothTetrahedrons(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hardSmoothTetrahedrons_3", ExactSpelling = true)]
        extern static void __MR_hardSmoothTetrahedrons_3(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *region);
        __MR_hardSmoothTetrahedrons_3(topology._UnderlyingPtr, points._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null);
    }
}
