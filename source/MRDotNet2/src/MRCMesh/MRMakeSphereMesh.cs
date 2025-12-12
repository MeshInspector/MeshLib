public static partial class MR
{
    /// Generated from class `MR::SphereParams`.
    /// This is the const half of the class.
    public class Const_SphereParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SphereParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_Destroy", ExactSpelling = true)]
            extern static void __MR_SphereParams_Destroy(_Underlying *_this);
            __MR_SphereParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SphereParams() {Dispose(false);}

        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_Get_radius", ExactSpelling = true)]
                extern static float *__MR_SphereParams_Get_radius(_Underlying *_this);
                return *__MR_SphereParams_Get_radius(_UnderlyingPtr);
            }
        }

        public unsafe int NumMeshVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_Get_numMeshVertices", ExactSpelling = true)]
                extern static int *__MR_SphereParams_Get_numMeshVertices(_Underlying *_this);
                return *__MR_SphereParams_Get_numMeshVertices(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SphereParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_DefaultConstruct();
            _UnderlyingPtr = __MR_SphereParams_DefaultConstruct();
        }

        /// Constructs `MR::SphereParams` elementwise.
        public unsafe Const_SphereParams(float radius, int numMeshVertices) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_ConstructFrom(float radius, int numMeshVertices);
            _UnderlyingPtr = __MR_SphereParams_ConstructFrom(radius, numMeshVertices);
        }

        /// Generated from constructor `MR::SphereParams::SphereParams`.
        public unsafe Const_SphereParams(MR.Const_SphereParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_ConstructFromAnother(MR.SphereParams._Underlying *_other);
            _UnderlyingPtr = __MR_SphereParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::SphereParams`.
    /// This is the non-const half of the class.
    public class SphereParams : Const_SphereParams
    {
        internal unsafe SphereParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_SphereParams_GetMutable_radius(_Underlying *_this);
                return ref *__MR_SphereParams_GetMutable_radius(_UnderlyingPtr);
            }
        }

        public new unsafe ref int NumMeshVertices
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_GetMutable_numMeshVertices", ExactSpelling = true)]
                extern static int *__MR_SphereParams_GetMutable_numMeshVertices(_Underlying *_this);
                return ref *__MR_SphereParams_GetMutable_numMeshVertices(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SphereParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_DefaultConstruct();
            _UnderlyingPtr = __MR_SphereParams_DefaultConstruct();
        }

        /// Constructs `MR::SphereParams` elementwise.
        public unsafe SphereParams(float radius, int numMeshVertices) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_ConstructFrom(float radius, int numMeshVertices);
            _UnderlyingPtr = __MR_SphereParams_ConstructFrom(radius, numMeshVertices);
        }

        /// Generated from constructor `MR::SphereParams::SphereParams`.
        public unsafe SphereParams(MR.Const_SphereParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_ConstructFromAnother(MR.SphereParams._Underlying *_other);
            _UnderlyingPtr = __MR_SphereParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SphereParams::operator=`.
        public unsafe MR.SphereParams Assign(MR.Const_SphereParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SphereParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SphereParams._Underlying *__MR_SphereParams_AssignFromAnother(_Underlying *_this, MR.SphereParams._Underlying *_other);
            return new(__MR_SphereParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SphereParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SphereParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SphereParams`/`Const_SphereParams` directly.
    public class _InOptMut_SphereParams
    {
        public SphereParams? Opt;

        public _InOptMut_SphereParams() {}
        public _InOptMut_SphereParams(SphereParams value) {Opt = value;}
        public static implicit operator _InOptMut_SphereParams(SphereParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `SphereParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SphereParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SphereParams`/`Const_SphereParams` to pass it to the function.
    public class _InOptConst_SphereParams
    {
        public Const_SphereParams? Opt;

        public _InOptConst_SphereParams() {}
        public _InOptConst_SphereParams(Const_SphereParams value) {Opt = value;}
        public static implicit operator _InOptConst_SphereParams(Const_SphereParams value) {return new(value);}
    }

    /// creates a mesh of sphere with irregular triangulation
    /// Generated from function `MR::makeSphere`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeSphere(MR.Const_SphereParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeSphere", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeSphere(MR.Const_SphereParams._Underlying *params_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeSphere(params_._UnderlyingPtr), is_owning: true));
    }

    /// creates a mesh of sphere with regular triangulation (parallels and meridians)
    /// Generated from function `MR::makeUVSphere`.
    /// Parameter `radius` defaults to `1.0`.
    /// Parameter `horisontalResolution` defaults to `16`.
    /// Parameter `verticalResolution` defaults to `16`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeUVSphere(float? radius = null, int? horisontalResolution = null, int? verticalResolution = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeUVSphere", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeUVSphere(float *radius, int *horisontalResolution, int *verticalResolution);
        float __deref_radius = radius.GetValueOrDefault();
        int __deref_horisontalResolution = horisontalResolution.GetValueOrDefault();
        int __deref_verticalResolution = verticalResolution.GetValueOrDefault();
        return MR.Misc.Move(new MR.Mesh(__MR_makeUVSphere(radius.HasValue ? &__deref_radius : null, horisontalResolution.HasValue ? &__deref_horisontalResolution : null, verticalResolution.HasValue ? &__deref_verticalResolution : null), is_owning: true));
    }
}
