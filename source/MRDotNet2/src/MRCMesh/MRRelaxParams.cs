public static partial class MR
{
    /// Generated from class `MR::RelaxParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshRelaxParams`
    ///     `MR::PointCloudRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::MeshApproxRelaxParams`
    ///     `MR::MeshEqualizeTriAreasParams`
    ///     `MR::PointCloudApproxRelaxParams`
    /// This is the const half of the class.
    public class Const_RelaxParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RelaxParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_Destroy", ExactSpelling = true)]
            extern static void __MR_RelaxParams_Destroy(_Underlying *_this);
            __MR_RelaxParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RelaxParams() {Dispose(false);}

        /// number of iterations
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_RelaxParams_Get_iterations(_Underlying *_this);
                return *__MR_RelaxParams_Get_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_RelaxParams_Get_region(_Underlying *_this);
                return ref *__MR_RelaxParams_Get_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public unsafe float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_Get_force", ExactSpelling = true)]
                extern static float *__MR_RelaxParams_Get_force(_Underlying *_this);
                return *__MR_RelaxParams_Get_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public unsafe bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_Get_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_RelaxParams_Get_limitNearInitial(_Underlying *_this);
                return *__MR_RelaxParams_Get_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public unsafe float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_Get_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_RelaxParams_Get_maxInitialDist(_Underlying *_this);
                return *__MR_RelaxParams_Get_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_RelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_RelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::RelaxParams::RelaxParams`.
        public unsafe Const_RelaxParams(MR.Const_RelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_RelaxParams_ConstructFromAnother(MR.RelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_RelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::RelaxParams`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshRelaxParams`
    ///     `MR::PointCloudRelaxParams`
    ///   Indirect: (non-virtual)
    ///     `MR::MeshApproxRelaxParams`
    ///     `MR::MeshEqualizeTriAreasParams`
    ///     `MR::PointCloudApproxRelaxParams`
    /// This is the non-const half of the class.
    public class RelaxParams : Const_RelaxParams
    {
        internal unsafe RelaxParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// number of iterations
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_RelaxParams_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_RelaxParams_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// region to relax
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_RelaxParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_RelaxParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// speed of relaxing, typical values (0.0, 0.5]
        public new unsafe ref float Force
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_GetMutable_force", ExactSpelling = true)]
                extern static float *__MR_RelaxParams_GetMutable_force(_Underlying *_this);
                return ref *__MR_RelaxParams_GetMutable_force(_UnderlyingPtr);
            }
        }

        /// if true then maximal displacement of each point during denoising will be limited
        public new unsafe ref bool LimitNearInitial
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_GetMutable_limitNearInitial", ExactSpelling = true)]
                extern static bool *__MR_RelaxParams_GetMutable_limitNearInitial(_Underlying *_this);
                return ref *__MR_RelaxParams_GetMutable_limitNearInitial(_UnderlyingPtr);
            }
        }

        /// maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false
        public new unsafe ref float MaxInitialDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_GetMutable_maxInitialDist", ExactSpelling = true)]
                extern static float *__MR_RelaxParams_GetMutable_maxInitialDist(_Underlying *_this);
                return ref *__MR_RelaxParams_GetMutable_maxInitialDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RelaxParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_RelaxParams_DefaultConstruct();
            _UnderlyingPtr = __MR_RelaxParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::RelaxParams::RelaxParams`.
        public unsafe RelaxParams(MR.Const_RelaxParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_RelaxParams_ConstructFromAnother(MR.RelaxParams._Underlying *_other);
            _UnderlyingPtr = __MR_RelaxParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::RelaxParams::operator=`.
        public unsafe MR.RelaxParams Assign(MR.Const_RelaxParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RelaxParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RelaxParams._Underlying *__MR_RelaxParams_AssignFromAnother(_Underlying *_this, MR.RelaxParams._Underlying *_other);
            return new(__MR_RelaxParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RelaxParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RelaxParams`/`Const_RelaxParams` directly.
    public class _InOptMut_RelaxParams
    {
        public RelaxParams? Opt;

        public _InOptMut_RelaxParams() {}
        public _InOptMut_RelaxParams(RelaxParams value) {Opt = value;}
        public static implicit operator _InOptMut_RelaxParams(RelaxParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `RelaxParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RelaxParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RelaxParams`/`Const_RelaxParams` to pass it to the function.
    public class _InOptConst_RelaxParams
    {
        public Const_RelaxParams? Opt;

        public _InOptConst_RelaxParams() {}
        public _InOptConst_RelaxParams(Const_RelaxParams value) {Opt = value;}
        public static implicit operator _InOptConst_RelaxParams(Const_RelaxParams value) {return new(value);}
    }

    public enum RelaxApproxType : int
    {
        Planar = 0,
        Quadric = 1,
    }
}
