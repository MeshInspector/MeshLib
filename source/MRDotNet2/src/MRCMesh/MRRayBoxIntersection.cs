public static partial class MR
{
    /// Generated from class `MR::RayOrigin<float>`.
    /// This is the const half of the class.
    public class Const_RayOrigin_Float : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RayOrigin_Float(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RayOrigin_float_Destroy", ExactSpelling = true)]
            extern static void __MR_RayOrigin_float_Destroy(_Underlying *_this);
            __MR_RayOrigin_float_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RayOrigin_Float() {Dispose(false);}

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public unsafe Const_RayOrigin_Float(MR.Const_RayOrigin_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RayOrigin_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RayOrigin_Float._Underlying *__MR_RayOrigin_float_ConstructFromAnother(MR.RayOrigin_Float._Underlying *_other);
            _UnderlyingPtr = __MR_RayOrigin_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public unsafe Const_RayOrigin_Float(MR.Const_Vector3f ro) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RayOrigin_float_Construct", ExactSpelling = true)]
            extern static MR.RayOrigin_Float._Underlying *__MR_RayOrigin_float_Construct(MR.Const_Vector3f._Underlying *ro);
            _UnderlyingPtr = __MR_RayOrigin_float_Construct(ro._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public static unsafe implicit operator Const_RayOrigin_Float(MR.Const_Vector3f ro) {return new(ro);}
    }

    /// Generated from class `MR::RayOrigin<float>`.
    /// This is the non-const half of the class.
    public class RayOrigin_Float : Const_RayOrigin_Float
    {
        internal unsafe RayOrigin_Float(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public unsafe RayOrigin_Float(MR.Const_RayOrigin_Float _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RayOrigin_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RayOrigin_Float._Underlying *__MR_RayOrigin_float_ConstructFromAnother(MR.RayOrigin_Float._Underlying *_other);
            _UnderlyingPtr = __MR_RayOrigin_float_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public unsafe RayOrigin_Float(MR.Const_Vector3f ro) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RayOrigin_float_Construct", ExactSpelling = true)]
            extern static MR.RayOrigin_Float._Underlying *__MR_RayOrigin_float_Construct(MR.Const_Vector3f._Underlying *ro);
            _UnderlyingPtr = __MR_RayOrigin_float_Construct(ro._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public static unsafe implicit operator RayOrigin_Float(MR.Const_Vector3f ro) {return new(ro);}

        /// Generated from method `MR::RayOrigin<float>::operator=`.
        public unsafe MR.RayOrigin_Float Assign(MR.Const_RayOrigin_Float _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RayOrigin_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RayOrigin_Float._Underlying *__MR_RayOrigin_float_AssignFromAnother(_Underlying *_this, MR.RayOrigin_Float._Underlying *_other);
            return new(__MR_RayOrigin_float_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RayOrigin_Float` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RayOrigin_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RayOrigin_Float`/`Const_RayOrigin_Float` directly.
    public class _InOptMut_RayOrigin_Float
    {
        public RayOrigin_Float? Opt;

        public _InOptMut_RayOrigin_Float() {}
        public _InOptMut_RayOrigin_Float(RayOrigin_Float value) {Opt = value;}
        public static implicit operator _InOptMut_RayOrigin_Float(RayOrigin_Float value) {return new(value);}
    }

    /// This is used for optional parameters of class `RayOrigin_Float` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RayOrigin_Float`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RayOrigin_Float`/`Const_RayOrigin_Float` to pass it to the function.
    public class _InOptConst_RayOrigin_Float
    {
        public Const_RayOrigin_Float? Opt;

        public _InOptConst_RayOrigin_Float() {}
        public _InOptConst_RayOrigin_Float(Const_RayOrigin_Float value) {Opt = value;}
        public static implicit operator _InOptConst_RayOrigin_Float(Const_RayOrigin_Float value) {return new(value);}

        /// Generated from constructor `MR::RayOrigin<float>::RayOrigin`.
        public static unsafe implicit operator _InOptConst_RayOrigin_Float(MR.Const_Vector3f ro) {return new MR.RayOrigin_Float(ro);}
    }

    /// finds intersection between the Ray and the Box.
    /// Precomputed values could be useful for several calls with the same direction,
    /// see "An Efficient and Robust Ray-Box Intersection Algorithm" at https://people.csail.mit.edu/amy/papers/box-jgt.pdf
    /// Generated from function `MR::rayBoxIntersect<float>`.
    public static unsafe bool RayBoxIntersect(MR.Const_Box3f box, MR.Const_RayOrigin_Float rayOrigin, ref float t0, ref float t1, MR.Const_IntersectionPrecomputes_Float prec)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayBoxIntersect_5", ExactSpelling = true)]
        extern static byte __MR_rayBoxIntersect_5(MR.Const_Box3f._Underlying *box, MR.Const_RayOrigin_Float._Underlying *rayOrigin, float *t0, float *t1, MR.Const_IntersectionPrecomputes_Float._Underlying *prec);
        fixed (float *__ptr_t0 = &t0)
        {
            fixed (float *__ptr_t1 = &t1)
            {
                return __MR_rayBoxIntersect_5(box._UnderlyingPtr, rayOrigin._UnderlyingPtr, __ptr_t0, __ptr_t1, prec._UnderlyingPtr) != 0;
            }
        }
    }

    /// Generated from function `MR::rayBoxIntersect<float>`.
    public static unsafe bool RayBoxIntersect(MR.Const_Box3f box, MR.Const_Line3f line, float t0, float t1)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayBoxIntersect_4_MR_Box3f", ExactSpelling = true)]
        extern static byte __MR_rayBoxIntersect_4_MR_Box3f(MR.Const_Box3f._Underlying *box, MR.Const_Line3f._Underlying *line, float t0, float t1);
        return __MR_rayBoxIntersect_4_MR_Box3f(box._UnderlyingPtr, line._UnderlyingPtr, t0, t1) != 0;
    }
}
