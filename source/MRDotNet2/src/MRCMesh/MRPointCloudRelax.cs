public static partial class MR
{
    /// Generated from class `MR::PointCloudRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RelaxParams`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointCloudApproxRelaxParams`
    /// This is the const half of the class.
    public class Const_PointCloudRelaxParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointCloudRelaxParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PointCloudRelaxParams_Destroy(_Underlying *_this);
            __MR_PointCloudRelaxParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointCloudRelaxParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RelaxParams(Const_PointCloudRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.Const_RelaxParams._Underlying *__MR_PointCloudRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.Const_RelaxParams ret = new(__MR_PointCloudRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// radius to find neighbors in,
        /// 0.0 - default, 0.1*boundibg box diagonal
        public unsafe float NeighborhoodRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Get_neighborhoodRadius", ExactSpelling = true)]
                extern static float *__MR_PointCloudRelaxParams_Get_neighborhoodRadius(_Underlying *_this);
                return *__MR_PointCloudRelaxParams_Get_neighborhoodRadius(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_PointCloudRelaxParams_Get_iterations(_Underlying *_this);
                return *__MR_PointCloudRelaxParams_Get_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_PointCloudRelaxParams_Get_region(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_Get_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public unsafe float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Get_force", ExactSpelling = true)]
                extern static float *__MR_PointCloudRelaxParams_Get_force(_Underlying *_this);
                return *__MR_PointCloudRelaxParams_Get_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_PointCloudRelaxParams_Get_limitNearInitial(_Underlying *_this);
                return *__MR_PointCloudRelaxParams_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_PointCloudRelaxParams_Get_maxInitialDist(_Underlying *_this);
                return *__MR_PointCloudRelaxParams_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointCloudRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointCloudRelaxParams._Underlying *__MR_PointCloudRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PointCloudRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointCloudRelaxParams::PointCloudRelaxParams`.
        public unsafe Const_PointCloudRelaxParams(MR.Const_PointCloudRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudRelaxParams._Underlying *__MR_PointCloudRelaxParams_ConstructFromAnother(MR.PointCloudRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_PointCloudRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PointCloudRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RelaxParams`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointCloudApproxRelaxParams`
    /// This is the non-const half of the class.
    public class PointCloudRelaxParams : Const_PointCloudRelaxParams
    {
        internal unsafe PointCloudRelaxParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RelaxParams(PointCloudRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_PointCloudRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.RelaxParams ret = new(__MR_PointCloudRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// radius to find neighbors in,
        /// 0.0 - default, 0.1*boundibg box diagonal
        public new unsafe ref float NeighborhoodRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_GetMutable_neighborhoodRadius", ExactSpelling = true)]
                extern static float *__MR_PointCloudRelaxParams_GetMutable_neighborhoodRadius(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_GetMutable_neighborhoodRadius(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_PointCloudRelaxParams_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_PointCloudRelaxParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public new unsafe ref float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_GetMutable_force", ExactSpelling = true)]
                extern static float *__MR_PointCloudRelaxParams_GetMutable_force(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_GetMutable_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_PointCloudRelaxParams_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_PointCloudRelaxParams_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_PointCloudRelaxParams_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointCloudRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointCloudRelaxParams._Underlying *__MR_PointCloudRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PointCloudRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointCloudRelaxParams::PointCloudRelaxParams`.
        public unsafe PointCloudRelaxParams(MR.Const_PointCloudRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudRelaxParams._Underlying *__MR_PointCloudRelaxParams_ConstructFromAnother(MR.PointCloudRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_PointCloudRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointCloudRelaxParams::operator=`.
        public unsafe MR.PointCloudRelaxParams Assign(MR.Const_PointCloudRelaxParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudRelaxParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudRelaxParams._Underlying *__MR_PointCloudRelaxParams_AssignFromAnother(_Underlying *_this, MR.PointCloudRelaxParams._Underlying *_other);
            return new(__MR_PointCloudRelaxParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointCloudRelaxParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointCloudRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloudRelaxParams`/`Const_PointCloudRelaxParams` directly.
    public class _InOptMut_PointCloudRelaxParams
    {
        public PointCloudRelaxParams? Opt;

        public _InOptMut_PointCloudRelaxParams() {}
        public _InOptMut_PointCloudRelaxParams(PointCloudRelaxParams value) {Opt = value;}
        public static implicit operator _InOptMut_PointCloudRelaxParams(PointCloudRelaxParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointCloudRelaxParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointCloudRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloudRelaxParams`/`Const_PointCloudRelaxParams` to pass it to the function.
    public class _InOptConst_PointCloudRelaxParams
    {
        public Const_PointCloudRelaxParams? Opt;

        public _InOptConst_PointCloudRelaxParams() {}
        public _InOptConst_PointCloudRelaxParams(Const_PointCloudRelaxParams value) {Opt = value;}
        public static implicit operator _InOptConst_PointCloudRelaxParams(Const_PointCloudRelaxParams value) {return new(value);}
    }

    /// Generated from class `MR::PointCloudApproxRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointCloudRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::RelaxParams`
    /// This is the const half of the class.
    public class Const_PointCloudApproxRelaxParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointCloudApproxRelaxParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PointCloudApproxRelaxParams_Destroy(_Underlying *_this);
            __MR_PointCloudApproxRelaxParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointCloudApproxRelaxParams() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RelaxParams(Const_PointCloudApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.Const_RelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.Const_RelaxParams ret = new(__MR_PointCloudApproxRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.Const_PointCloudRelaxParams(Const_PointCloudApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_UpcastTo_MR_PointCloudRelaxParams", ExactSpelling = true)]
            extern static MR.Const_PointCloudRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_UpcastTo_MR_PointCloudRelaxParams(_Underlying *_this);
            MR.Const_PointCloudRelaxParams ret = new(__MR_PointCloudApproxRelaxParams_UpcastTo_MR_PointCloudRelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.RelaxApproxType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_type", ExactSpelling = true)]
                extern static MR.RelaxApproxType *__MR_PointCloudApproxRelaxParams_Get_type(_Underlying *_this);
                return *__MR_PointCloudApproxRelaxParams_Get_type(_UnderlyingPtr);
            }
        }

        /// radius to find neighbors in,
        /// 0.0 - default, 0.1*boundibg box diagonal
        public unsafe float NeighborhoodRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_neighborhoodRadius", ExactSpelling = true)]
                extern static float *__MR_PointCloudApproxRelaxParams_Get_neighborhoodRadius(_Underlying *_this);
                return *__MR_PointCloudApproxRelaxParams_Get_neighborhoodRadius(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_PointCloudApproxRelaxParams_Get_iterations(_Underlying *_this);
                return *__MR_PointCloudApproxRelaxParams_Get_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_PointCloudApproxRelaxParams_Get_region(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_Get_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public unsafe float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_force", ExactSpelling = true)]
                extern static float *__MR_PointCloudApproxRelaxParams_Get_force(_Underlying *_this);
                return *__MR_PointCloudApproxRelaxParams_Get_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_PointCloudApproxRelaxParams_Get_limitNearInitial(_Underlying *_this);
                return *__MR_PointCloudApproxRelaxParams_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_PointCloudApproxRelaxParams_Get_maxInitialDist(_Underlying *_this);
                return *__MR_PointCloudApproxRelaxParams_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointCloudApproxRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointCloudApproxRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PointCloudApproxRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointCloudApproxRelaxParams::PointCloudApproxRelaxParams`.
        public unsafe Const_PointCloudApproxRelaxParams(MR.Const_PointCloudApproxRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudApproxRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_ConstructFromAnother(MR.PointCloudApproxRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_PointCloudApproxRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PointCloudApproxRelaxParams`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointCloudRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::RelaxParams`
    /// This is the non-const half of the class.
    public class PointCloudApproxRelaxParams : Const_PointCloudApproxRelaxParams
    {
        internal unsafe PointCloudApproxRelaxParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RelaxParams(PointCloudApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_UpcastTo_MR_RelaxParams", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_UpcastTo_MR_RelaxParams(_Underlying *_this);
            MR.RelaxParams ret = new(__MR_PointCloudApproxRelaxParams_UpcastTo_MR_RelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }
        public static unsafe implicit operator MR.PointCloudRelaxParams(PointCloudApproxRelaxParams self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_UpcastTo_MR_PointCloudRelaxParams", ExactSpelling = true)]
            extern static MR.PointCloudRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_UpcastTo_MR_PointCloudRelaxParams(_Underlying *_this);
            MR.PointCloudRelaxParams ret = new(__MR_PointCloudApproxRelaxParams_UpcastTo_MR_PointCloudRelaxParams(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref MR.RelaxApproxType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_type", ExactSpelling = true)]
                extern static MR.RelaxApproxType *__MR_PointCloudApproxRelaxParams_GetMutable_type(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_type(_UnderlyingPtr);
            }
        }

        /// radius to find neighbors in,
        /// 0.0 - default, 0.1*boundibg box diagonal
        public new unsafe ref float NeighborhoodRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_neighborhoodRadius", ExactSpelling = true)]
                extern static float *__MR_PointCloudApproxRelaxParams_GetMutable_neighborhoodRadius(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_neighborhoodRadius(_UnderlyingPtr);
            }
        }

        /// number of iterations
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_PointCloudApproxRelaxParams_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_PointCloudApproxRelaxParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public new unsafe ref float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_force", ExactSpelling = true)]
                extern static float *__MR_PointCloudApproxRelaxParams_GetMutable_force(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_PointCloudApproxRelaxParams_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_PointCloudApproxRelaxParams_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_PointCloudApproxRelaxParams_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointCloudApproxRelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointCloudApproxRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PointCloudApproxRelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointCloudApproxRelaxParams::PointCloudApproxRelaxParams`.
        public unsafe PointCloudApproxRelaxParams(MR.Const_PointCloudApproxRelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudApproxRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_ConstructFromAnother(MR.PointCloudApproxRelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_PointCloudApproxRelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointCloudApproxRelaxParams::operator=`.
        public unsafe MR.PointCloudApproxRelaxParams Assign(MR.Const_PointCloudApproxRelaxParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointCloudApproxRelaxParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointCloudApproxRelaxParams._Underlying *__MR_PointCloudApproxRelaxParams_AssignFromAnother(_Underlying *_this, MR.PointCloudApproxRelaxParams._Underlying *_other);
            return new(__MR_PointCloudApproxRelaxParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointCloudApproxRelaxParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointCloudApproxRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloudApproxRelaxParams`/`Const_PointCloudApproxRelaxParams` directly.
    public class _InOptMut_PointCloudApproxRelaxParams
    {
        public PointCloudApproxRelaxParams? Opt;

        public _InOptMut_PointCloudApproxRelaxParams() {}
        public _InOptMut_PointCloudApproxRelaxParams(PointCloudApproxRelaxParams value) {Opt = value;}
        public static implicit operator _InOptMut_PointCloudApproxRelaxParams(PointCloudApproxRelaxParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointCloudApproxRelaxParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointCloudApproxRelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointCloudApproxRelaxParams`/`Const_PointCloudApproxRelaxParams` to pass it to the function.
    public class _InOptConst_PointCloudApproxRelaxParams
    {
        public Const_PointCloudApproxRelaxParams? Opt;

        public _InOptConst_PointCloudApproxRelaxParams() {}
        public _InOptConst_PointCloudApproxRelaxParams(Const_PointCloudApproxRelaxParams value) {Opt = value;}
        public static implicit operator _InOptConst_PointCloudApproxRelaxParams(Const_PointCloudApproxRelaxParams value) {return new(value);}
    }

    /// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
    /// \return true if was finished successfully, false if was interrupted by progress callback
    /// Generated from function `MR::relax`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool Relax(MR.PointCloud pointCloud, MR.Const_PointCloudRelaxParams? params_ = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relax_3_MR_PointCloud", ExactSpelling = true)]
        extern static byte __MR_relax_3_MR_PointCloud(MR.PointCloud._Underlying *pointCloud, MR.Const_PointCloudRelaxParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relax_3_MR_PointCloud(pointCloud._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
    }

    /// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
    /// do not really keeps volume but tries hard
    /// \return true if was finished successfully, false if was interrupted by progress callback
    /// Generated from function `MR::relaxKeepVolume`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool RelaxKeepVolume(MR.PointCloud pointCloud, MR.Const_PointCloudRelaxParams? params_ = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relaxKeepVolume_3_MR_PointCloud", ExactSpelling = true)]
        extern static byte __MR_relaxKeepVolume_3_MR_PointCloud(MR.PointCloud._Underlying *pointCloud, MR.Const_PointCloudRelaxParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relaxKeepVolume_3_MR_PointCloud(pointCloud._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
    }

    /// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
    /// approx neighborhoods
    /// \return true if was finished successfully, false if was interrupted by progress callback
    /// Generated from function `MR::relaxApprox`.
    /// Parameter `params_` defaults to `{}`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool RelaxApprox(MR.PointCloud pointCloud, MR.Const_PointCloudApproxRelaxParams? params_ = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_relaxApprox_3_MR_PointCloud", ExactSpelling = true)]
        extern static byte __MR_relaxApprox_3_MR_PointCloud(MR.PointCloud._Underlying *pointCloud, MR.Const_PointCloudApproxRelaxParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_relaxApprox_3_MR_PointCloud(pointCloud._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
    }
}
