public static partial class MR
{
    /// Generated from class `MR::TriIntersectResult`.
    /// This is the const half of the class.
    public class Const_TriIntersectResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TriIntersectResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_Destroy", ExactSpelling = true)]
            extern static void __MR_TriIntersectResult_Destroy(_Underlying *_this);
            __MR_TriIntersectResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TriIntersectResult() {Dispose(false);}

        // barycentric representation
        public unsafe MR.Const_TriPointf Bary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_Get_bary", ExactSpelling = true)]
                extern static MR.Const_TriPointf._Underlying *__MR_TriIntersectResult_Get_bary(_Underlying *_this);
                return new(__MR_TriIntersectResult_Get_bary(_UnderlyingPtr), is_owning: false);
            }
        }

        // distance from ray origin to p in dir length units
        public unsafe float T
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_Get_t", ExactSpelling = true)]
                extern static float *__MR_TriIntersectResult_Get_t(_Underlying *_this);
                return *__MR_TriIntersectResult_Get_t(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::TriIntersectResult::TriIntersectResult`.
        public unsafe Const_TriIntersectResult(MR.Const_TriIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriIntersectResult._Underlying *__MR_TriIntersectResult_ConstructFromAnother(MR.TriIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_TriIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriIntersectResult::TriIntersectResult`.
        public unsafe Const_TriIntersectResult(float U, float V, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_Construct", ExactSpelling = true)]
            extern static MR.TriIntersectResult._Underlying *__MR_TriIntersectResult_Construct(float U, float V, float dist);
            _UnderlyingPtr = __MR_TriIntersectResult_Construct(U, V, dist);
        }
    }

    /// Generated from class `MR::TriIntersectResult`.
    /// This is the non-const half of the class.
    public class TriIntersectResult : Const_TriIntersectResult
    {
        internal unsafe TriIntersectResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // barycentric representation
        public new unsafe MR.TriPointf Bary
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_GetMutable_bary", ExactSpelling = true)]
                extern static MR.TriPointf._Underlying *__MR_TriIntersectResult_GetMutable_bary(_Underlying *_this);
                return new(__MR_TriIntersectResult_GetMutable_bary(_UnderlyingPtr), is_owning: false);
            }
        }

        // distance from ray origin to p in dir length units
        public new unsafe ref float T
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_GetMutable_t", ExactSpelling = true)]
                extern static float *__MR_TriIntersectResult_GetMutable_t(_Underlying *_this);
                return ref *__MR_TriIntersectResult_GetMutable_t(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::TriIntersectResult::TriIntersectResult`.
        public unsafe TriIntersectResult(MR.Const_TriIntersectResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TriIntersectResult._Underlying *__MR_TriIntersectResult_ConstructFromAnother(MR.TriIntersectResult._Underlying *_other);
            _UnderlyingPtr = __MR_TriIntersectResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TriIntersectResult::TriIntersectResult`.
        public unsafe TriIntersectResult(float U, float V, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_Construct", ExactSpelling = true)]
            extern static MR.TriIntersectResult._Underlying *__MR_TriIntersectResult_Construct(float U, float V, float dist);
            _UnderlyingPtr = __MR_TriIntersectResult_Construct(U, V, dist);
        }

        /// Generated from method `MR::TriIntersectResult::operator=`.
        public unsafe MR.TriIntersectResult Assign(MR.Const_TriIntersectResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriIntersectResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TriIntersectResult._Underlying *__MR_TriIntersectResult_AssignFromAnother(_Underlying *_this, MR.TriIntersectResult._Underlying *_other);
            return new(__MR_TriIntersectResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TriIntersectResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TriIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriIntersectResult`/`Const_TriIntersectResult` directly.
    public class _InOptMut_TriIntersectResult
    {
        public TriIntersectResult? Opt;

        public _InOptMut_TriIntersectResult() {}
        public _InOptMut_TriIntersectResult(TriIntersectResult value) {Opt = value;}
        public static implicit operator _InOptMut_TriIntersectResult(TriIntersectResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `TriIntersectResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TriIntersectResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TriIntersectResult`/`Const_TriIntersectResult` to pass it to the function.
    public class _InOptConst_TriIntersectResult
    {
        public Const_TriIntersectResult? Opt;

        public _InOptConst_TriIntersectResult() {}
        public _InOptConst_TriIntersectResult(Const_TriIntersectResult value) {Opt = value;}
        public static implicit operator _InOptConst_TriIntersectResult(Const_TriIntersectResult value) {return new(value);}
    }

    /// Generated from function `MR::rayTriangleIntersect<float>`.
    public static unsafe MR.Std.Optional_MRTriIntersectResult RayTriangleIntersect(MR.Const_Vector3f oriA, MR.Const_Vector3f oriB, MR.Const_Vector3f oriC, MR.Const_IntersectionPrecomputes_Float prec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayTriangleIntersect_float_MR_IntersectionPrecomputes_float", ExactSpelling = true)]
        extern static MR.Std.Optional_MRTriIntersectResult._Underlying *__MR_rayTriangleIntersect_float_MR_IntersectionPrecomputes_float(MR.Const_Vector3f._Underlying *oriA, MR.Const_Vector3f._Underlying *oriB, MR.Const_Vector3f._Underlying *oriC, MR.Const_IntersectionPrecomputes_Float._Underlying *prec);
        return new(__MR_rayTriangleIntersect_float_MR_IntersectionPrecomputes_float(oriA._UnderlyingPtr, oriB._UnderlyingPtr, oriC._UnderlyingPtr, prec._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::rayTriangleIntersect<double>`.
    public static unsafe MR.Std.Optional_MRTriIntersectResult RayTriangleIntersect(MR.Const_Vector3d oriA, MR.Const_Vector3d oriB, MR.Const_Vector3d oriC, MR.Const_IntersectionPrecomputes_Double prec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayTriangleIntersect_double_MR_IntersectionPrecomputes_double", ExactSpelling = true)]
        extern static MR.Std.Optional_MRTriIntersectResult._Underlying *__MR_rayTriangleIntersect_double_MR_IntersectionPrecomputes_double(MR.Const_Vector3d._Underlying *oriA, MR.Const_Vector3d._Underlying *oriB, MR.Const_Vector3d._Underlying *oriC, MR.Const_IntersectionPrecomputes_Double._Underlying *prec);
        return new(__MR_rayTriangleIntersect_double_MR_IntersectionPrecomputes_double(oriA._UnderlyingPtr, oriB._UnderlyingPtr, oriC._UnderlyingPtr, prec._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::rayTriangleIntersect<float>`.
    public static unsafe MR.Std.Optional_MRTriIntersectResult RayTriangleIntersect(MR.Const_Vector3f oriA, MR.Const_Vector3f oriB, MR.Const_Vector3f oriC, MR.Const_Vector3f dir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayTriangleIntersect_float_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Optional_MRTriIntersectResult._Underlying *__MR_rayTriangleIntersect_float_MR_Vector3f(MR.Const_Vector3f._Underlying *oriA, MR.Const_Vector3f._Underlying *oriB, MR.Const_Vector3f._Underlying *oriC, MR.Const_Vector3f._Underlying *dir);
        return new(__MR_rayTriangleIntersect_float_MR_Vector3f(oriA._UnderlyingPtr, oriB._UnderlyingPtr, oriC._UnderlyingPtr, dir._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::rayTriangleIntersect<double>`.
    public static unsafe MR.Std.Optional_MRTriIntersectResult RayTriangleIntersect(MR.Const_Vector3d oriA, MR.Const_Vector3d oriB, MR.Const_Vector3d oriC, MR.Const_Vector3d dir)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayTriangleIntersect_double_MR_Vector3d", ExactSpelling = true)]
        extern static MR.Std.Optional_MRTriIntersectResult._Underlying *__MR_rayTriangleIntersect_double_MR_Vector3d(MR.Const_Vector3d._Underlying *oriA, MR.Const_Vector3d._Underlying *oriB, MR.Const_Vector3d._Underlying *oriC, MR.Const_Vector3d._Underlying *dir);
        return new(__MR_rayTriangleIntersect_double_MR_Vector3d(oriA._UnderlyingPtr, oriB._UnderlyingPtr, oriC._UnderlyingPtr, dir._UnderlyingPtr), is_owning: true);
    }
}
