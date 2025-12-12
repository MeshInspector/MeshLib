public static partial class MR
{
    public enum ConeFitterType : int
    {
        // approximation of cone axis by principal component method
        ApproximationPCM = 0,
        HemisphereSearchFit = 1,
        SpecificAxisFit = 2,
    }

    /// Generated from class `MR::Cone3ApproximationParams`.
    /// This is the const half of the class.
    public class Const_Cone3ApproximationParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Cone3ApproximationParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_Destroy", ExactSpelling = true)]
            extern static void __MR_Cone3ApproximationParams_Destroy(_Underlying *_this);
            __MR_Cone3ApproximationParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Cone3ApproximationParams() {Dispose(false);}

        public unsafe int LevenbergMarquardtMaxIteration
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_Get_levenbergMarquardtMaxIteration", ExactSpelling = true)]
                extern static int *__MR_Cone3ApproximationParams_Get_levenbergMarquardtMaxIteration(_Underlying *_this);
                return *__MR_Cone3ApproximationParams_Get_levenbergMarquardtMaxIteration(_UnderlyingPtr);
            }
        }

        public unsafe MR.ConeFitterType ConeFitterType
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_Get_coneFitterType", ExactSpelling = true)]
                extern static MR.ConeFitterType *__MR_Cone3ApproximationParams_Get_coneFitterType(_Underlying *_this);
                return *__MR_Cone3ApproximationParams_Get_coneFitterType(_UnderlyingPtr);
            }
        }

        public unsafe int HemisphereSearchPhiResolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_Get_hemisphereSearchPhiResolution", ExactSpelling = true)]
                extern static int *__MR_Cone3ApproximationParams_Get_hemisphereSearchPhiResolution(_Underlying *_this);
                return *__MR_Cone3ApproximationParams_Get_hemisphereSearchPhiResolution(_UnderlyingPtr);
            }
        }

        public unsafe int HemisphereSearchThetaResolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_Get_hemisphereSearchThetaResolution", ExactSpelling = true)]
                extern static int *__MR_Cone3ApproximationParams_Get_hemisphereSearchThetaResolution(_Underlying *_this);
                return *__MR_Cone3ApproximationParams_Get_hemisphereSearchThetaResolution(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Cone3ApproximationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_Cone3ApproximationParams_DefaultConstruct();
        }

        /// Constructs `MR::Cone3ApproximationParams` elementwise.
        public unsafe Const_Cone3ApproximationParams(int levenbergMarquardtMaxIteration, MR.ConeFitterType coneFitterType, int hemisphereSearchPhiResolution, int hemisphereSearchThetaResolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_ConstructFrom(int levenbergMarquardtMaxIteration, MR.ConeFitterType coneFitterType, int hemisphereSearchPhiResolution, int hemisphereSearchThetaResolution);
            _UnderlyingPtr = __MR_Cone3ApproximationParams_ConstructFrom(levenbergMarquardtMaxIteration, coneFitterType, hemisphereSearchPhiResolution, hemisphereSearchThetaResolution);
        }

        /// Generated from constructor `MR::Cone3ApproximationParams::Cone3ApproximationParams`.
        public unsafe Const_Cone3ApproximationParams(MR.Const_Cone3ApproximationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_ConstructFromAnother(MR.Cone3ApproximationParams._Underlying *_other);
            _UnderlyingPtr = __MR_Cone3ApproximationParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::Cone3ApproximationParams`.
    /// This is the non-const half of the class.
    public class Cone3ApproximationParams : Const_Cone3ApproximationParams
    {
        internal unsafe Cone3ApproximationParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref int LevenbergMarquardtMaxIteration
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_GetMutable_levenbergMarquardtMaxIteration", ExactSpelling = true)]
                extern static int *__MR_Cone3ApproximationParams_GetMutable_levenbergMarquardtMaxIteration(_Underlying *_this);
                return ref *__MR_Cone3ApproximationParams_GetMutable_levenbergMarquardtMaxIteration(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.ConeFitterType ConeFitterType
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_GetMutable_coneFitterType", ExactSpelling = true)]
                extern static MR.ConeFitterType *__MR_Cone3ApproximationParams_GetMutable_coneFitterType(_Underlying *_this);
                return ref *__MR_Cone3ApproximationParams_GetMutable_coneFitterType(_UnderlyingPtr);
            }
        }

        public new unsafe ref int HemisphereSearchPhiResolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_GetMutable_hemisphereSearchPhiResolution", ExactSpelling = true)]
                extern static int *__MR_Cone3ApproximationParams_GetMutable_hemisphereSearchPhiResolution(_Underlying *_this);
                return ref *__MR_Cone3ApproximationParams_GetMutable_hemisphereSearchPhiResolution(_UnderlyingPtr);
            }
        }

        public new unsafe ref int HemisphereSearchThetaResolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_GetMutable_hemisphereSearchThetaResolution", ExactSpelling = true)]
                extern static int *__MR_Cone3ApproximationParams_GetMutable_hemisphereSearchThetaResolution(_Underlying *_this);
                return ref *__MR_Cone3ApproximationParams_GetMutable_hemisphereSearchThetaResolution(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Cone3ApproximationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_Cone3ApproximationParams_DefaultConstruct();
        }

        /// Constructs `MR::Cone3ApproximationParams` elementwise.
        public unsafe Cone3ApproximationParams(int levenbergMarquardtMaxIteration, MR.ConeFitterType coneFitterType, int hemisphereSearchPhiResolution, int hemisphereSearchThetaResolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_ConstructFrom(int levenbergMarquardtMaxIteration, MR.ConeFitterType coneFitterType, int hemisphereSearchPhiResolution, int hemisphereSearchThetaResolution);
            _UnderlyingPtr = __MR_Cone3ApproximationParams_ConstructFrom(levenbergMarquardtMaxIteration, coneFitterType, hemisphereSearchPhiResolution, hemisphereSearchThetaResolution);
        }

        /// Generated from constructor `MR::Cone3ApproximationParams::Cone3ApproximationParams`.
        public unsafe Cone3ApproximationParams(MR.Const_Cone3ApproximationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_ConstructFromAnother(MR.Cone3ApproximationParams._Underlying *_other);
            _UnderlyingPtr = __MR_Cone3ApproximationParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Cone3ApproximationParams::operator=`.
        public unsafe MR.Cone3ApproximationParams Assign(MR.Const_Cone3ApproximationParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3ApproximationParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Cone3ApproximationParams._Underlying *__MR_Cone3ApproximationParams_AssignFromAnother(_Underlying *_this, MR.Cone3ApproximationParams._Underlying *_other);
            return new(__MR_Cone3ApproximationParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Cone3ApproximationParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Cone3ApproximationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cone3ApproximationParams`/`Const_Cone3ApproximationParams` directly.
    public class _InOptMut_Cone3ApproximationParams
    {
        public Cone3ApproximationParams? Opt;

        public _InOptMut_Cone3ApproximationParams() {}
        public _InOptMut_Cone3ApproximationParams(Cone3ApproximationParams value) {Opt = value;}
        public static implicit operator _InOptMut_Cone3ApproximationParams(Cone3ApproximationParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `Cone3ApproximationParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Cone3ApproximationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cone3ApproximationParams`/`Const_Cone3ApproximationParams` to pass it to the function.
    public class _InOptConst_Cone3ApproximationParams
    {
        public Const_Cone3ApproximationParams? Opt;

        public _InOptConst_Cone3ApproximationParams() {}
        public _InOptConst_Cone3ApproximationParams(Const_Cone3ApproximationParams value) {Opt = value;}
        public static implicit operator _InOptConst_Cone3ApproximationParams(Const_Cone3ApproximationParams value) {return new(value);}
    }
}
