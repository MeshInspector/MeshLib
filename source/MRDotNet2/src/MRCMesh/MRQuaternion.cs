public static partial class MR
{
    /// Represents a quaternion following the notations from
    /// https://en.wikipedia.org/wiki/Quaternion
    /// Generated from class `MR::Quaternionf`.
    /// This is the const half of the class.
    public class Const_Quaternionf : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Quaternionf(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Destroy", ExactSpelling = true)]
            extern static void __MR_Quaternionf_Destroy(_Underlying *_this);
            __MR_Quaternionf_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Quaternionf() {Dispose(false);}

        ///< real part of the quaternion
        public unsafe float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Get_a", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_Get_a(_Underlying *_this);
                return *__MR_Quaternionf_Get_a(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public unsafe float B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Get_b", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_Get_b(_Underlying *_this);
                return *__MR_Quaternionf_Get_b(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public unsafe float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Get_c", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_Get_c(_Underlying *_this);
                return *__MR_Quaternionf_Get_c(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public unsafe float D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Get_d", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_Get_d(_Underlying *_this);
                return *__MR_Quaternionf_Get_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Quaternionf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_DefaultConstruct();
            _UnderlyingPtr = __MR_Quaternionf_DefaultConstruct();
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Const_Quaternionf(MR.Const_Quaternionf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_ConstructFromAnother(MR.Quaternionf._Underlying *_other);
            _UnderlyingPtr = __MR_Quaternionf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Const_Quaternionf(float a, float b, float c, float d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_4", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_4(float a, float b, float c, float d);
            _UnderlyingPtr = __MR_Quaternionf_Construct_4(a, b, c, d);
        }

        /// \related Quaternion
        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Const_Quaternionf(MR.Const_Vector3f axis, float angle) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_float", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_float(MR.Const_Vector3f._Underlying *axis, float angle);
            _UnderlyingPtr = __MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_float(axis._UnderlyingPtr, angle);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Const_Quaternionf(float real, MR.Const_Vector3f im) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_2_float", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_2_float(float real, MR.Const_Vector3f._Underlying *im);
            _UnderlyingPtr = __MR_Quaternionf_Construct_2_float(real, im._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Const_Quaternionf(MR.Const_Matrix3f m) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_1", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_1(MR.Const_Matrix3f._Underlying *m);
            _UnderlyingPtr = __MR_Quaternionf_Construct_1(m._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public static unsafe implicit operator Const_Quaternionf(MR.Const_Matrix3f m) {return new(m);}

        /// finds shorter arc rotation quaternion from one vector to another
        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Const_Quaternionf(MR.Const_Vector3f from, MR.Const_Vector3f to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_MR_Vector3f(MR.Const_Vector3f._Underlying *from, MR.Const_Vector3f._Underlying *to);
            _UnderlyingPtr = __MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_MR_Vector3f(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// converts this into 3x3 rotation matrix
        /// Generated from conversion operator `MR::Quaternionf::operator MR::Matrix3f`.
        public static unsafe implicit operator MR.Matrix3f(MR.Const_Quaternionf _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_ConvertTo_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Quaternionf_ConvertTo_MR_Matrix3f(MR.Const_Quaternionf._Underlying *_this);
            return __MR_Quaternionf_ConvertTo_MR_Matrix3f(_this._UnderlyingPtr);
        }

        /// returns imaginary part of the quaternion as a vector
        /// Generated from method `MR::Quaternionf::im`.
        public unsafe MR.Vector3f Im()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_im", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Quaternionf_im(_Underlying *_this);
            return __MR_Quaternionf_im(_UnderlyingPtr);
        }

        /// returns angle of rotation encoded in this quaternion
        /// Generated from method `MR::Quaternionf::angle`.
        public unsafe float Angle()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_angle", ExactSpelling = true)]
            extern static float __MR_Quaternionf_angle(_Underlying *_this);
            return __MR_Quaternionf_angle(_UnderlyingPtr);
        }

        /// returns axis of rotation encoded in this quaternion
        /// Generated from method `MR::Quaternionf::axis`.
        public unsafe MR.Vector3f Axis()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_axis", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Quaternionf_axis(_Underlying *_this);
            return __MR_Quaternionf_axis(_UnderlyingPtr);
        }

        /// Generated from method `MR::Quaternionf::normSq`.
        public unsafe float NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_normSq", ExactSpelling = true)]
            extern static float __MR_Quaternionf_normSq(_Underlying *_this);
            return __MR_Quaternionf_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Quaternionf::norm`.
        public unsafe float Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_norm", ExactSpelling = true)]
            extern static float __MR_Quaternionf_norm(_Underlying *_this);
            return __MR_Quaternionf_norm(_UnderlyingPtr);
        }

        /// returns quaternion representing the same rotation, using the opposite rotation direction and opposite angle
        /// Generated from method `MR::Quaternionf::operator-`.
        public static unsafe MR.Quaternionf operator-(MR.Const_Quaternionf _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Quaternionf", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_neg_MR_Quaternionf(MR.Const_Quaternionf._Underlying *_this);
            return new(__MR_neg_MR_Quaternionf(_this._UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::Quaternionf::normalized`.
        public unsafe MR.Quaternionf Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_normalized", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_normalized(_Underlying *_this);
            return new(__MR_Quaternionf_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// computes conjugate quaternion, which for unit quaternions encodes the opposite rotation
        /// Generated from method `MR::Quaternionf::conjugate`.
        public unsafe MR.Quaternionf Conjugate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_conjugate", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_conjugate(_Underlying *_this);
            return new(__MR_Quaternionf_conjugate(_UnderlyingPtr), is_owning: true);
        }

        /// computes reciprocal quaternion
        /// Generated from method `MR::Quaternionf::inverse`.
        public unsafe MR.Quaternionf Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_inverse", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_inverse(_Underlying *_this);
            return new(__MR_Quaternionf_inverse(_UnderlyingPtr), is_owning: true);
        }

        /// for unit quaternion returns the rotation of point p, which is faster to compute for single point;
        /// for multiple points it is faster to create matrix representation and apply it to the points
        /// Generated from method `MR::Quaternionf::operator()`.
        public unsafe MR.Vector3f Call(MR.Const_Vector3f p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_call", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Quaternionf_call(_Underlying *_this, MR.Const_Vector3f._Underlying *p);
            return __MR_Quaternionf_call(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// given t in [0,1], interpolates linearly two quaternions giving in general not-unit quaternion
        /// Generated from method `MR::Quaternionf::lerp`.
        public static unsafe MR.Quaternionf Lerp(MR.Const_Quaternionf q0, MR.Const_Quaternionf q1, float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_lerp", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_lerp(MR.Const_Quaternionf._Underlying *q0, MR.Const_Quaternionf._Underlying *q1, float t);
            return new(__MR_Quaternionf_lerp(q0._UnderlyingPtr, q1._UnderlyingPtr, t), is_owning: true);
        }

        /// given t in [0,1] and two unit quaternions, interpolates them spherically and produces another unit quaternion
        /// Generated from method `MR::Quaternionf::slerp`.
        public static unsafe MR.Quaternionf Slerp(MR.Const_Quaternionf q0, MR.Const_Quaternionf q1, float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_slerp_3_MR_Quaternionf", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_slerp_3_MR_Quaternionf(MR.Quaternionf._Underlying *q0, MR.Quaternionf._Underlying *q1, float t);
            return new(__MR_Quaternionf_slerp_3_MR_Quaternionf(q0._UnderlyingPtr, q1._UnderlyingPtr, t), is_owning: true);
        }

        /// given t in [0,1] and two rotation matrices, interpolates them spherically and produces another rotation matrix
        /// Generated from method `MR::Quaternionf::slerp`.
        public static unsafe MR.Matrix3f Slerp(MR.Const_Matrix3f m0, MR.Const_Matrix3f m1, float t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_slerp_3_MR_Matrix3f", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_Quaternionf_slerp_3_MR_Matrix3f(MR.Const_Matrix3f._Underlying *m0, MR.Const_Matrix3f._Underlying *m1, float t);
            return __MR_Quaternionf_slerp_3_MR_Matrix3f(m0._UnderlyingPtr, m1._UnderlyingPtr, t);
        }

        /// given t in [0,1] and rigid transformations, interpolates them spherically and produces another rigid transformation;
        /// p is the only point that will have straight line movement during interpolation
        /// Generated from method `MR::Quaternionf::slerp`.
        /// Parameter `p` defaults to `{}`.
        public static unsafe MR.AffineXf3f Slerp(MR.Const_AffineXf3f xf0, MR.Const_AffineXf3f xf1, float t, MR.Const_Vector3f? p = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_slerp_4", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_Quaternionf_slerp_4(MR.Const_AffineXf3f._Underlying *xf0, MR.Const_AffineXf3f._Underlying *xf1, float t, MR.Const_Vector3f._Underlying *p);
            return __MR_Quaternionf_slerp_4(xf0._UnderlyingPtr, xf1._UnderlyingPtr, t, p is not null ? p._UnderlyingPtr : null);
        }
    }

    /// Represents a quaternion following the notations from
    /// https://en.wikipedia.org/wiki/Quaternion
    /// Generated from class `MR::Quaternionf`.
    /// This is the non-const half of the class.
    public class Quaternionf : Const_Quaternionf
    {
        internal unsafe Quaternionf(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< real part of the quaternion
        public new unsafe ref float A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_GetMutable_a", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_GetMutable_a(_Underlying *_this);
                return ref *__MR_Quaternionf_GetMutable_a(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public new unsafe ref float B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_GetMutable_b", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_GetMutable_b(_Underlying *_this);
                return ref *__MR_Quaternionf_GetMutable_b(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public new unsafe ref float C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_GetMutable_c", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_GetMutable_c(_Underlying *_this);
                return ref *__MR_Quaternionf_GetMutable_c(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public new unsafe ref float D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_GetMutable_d", ExactSpelling = true)]
                extern static float *__MR_Quaternionf_GetMutable_d(_Underlying *_this);
                return ref *__MR_Quaternionf_GetMutable_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Quaternionf() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_DefaultConstruct();
            _UnderlyingPtr = __MR_Quaternionf_DefaultConstruct();
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Quaternionf(MR.Const_Quaternionf _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_ConstructFromAnother(MR.Quaternionf._Underlying *_other);
            _UnderlyingPtr = __MR_Quaternionf_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Quaternionf(float a, float b, float c, float d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_4", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_4(float a, float b, float c, float d);
            _UnderlyingPtr = __MR_Quaternionf_Construct_4(a, b, c, d);
        }

        /// \related Quaternion
        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Quaternionf(MR.Const_Vector3f axis, float angle) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_float", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_float(MR.Const_Vector3f._Underlying *axis, float angle);
            _UnderlyingPtr = __MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_float(axis._UnderlyingPtr, angle);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Quaternionf(float real, MR.Const_Vector3f im) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_2_float", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_2_float(float real, MR.Const_Vector3f._Underlying *im);
            _UnderlyingPtr = __MR_Quaternionf_Construct_2_float(real, im._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Quaternionf(MR.Const_Matrix3f m) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_1", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_1(MR.Const_Matrix3f._Underlying *m);
            _UnderlyingPtr = __MR_Quaternionf_Construct_1(m._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public static unsafe implicit operator Quaternionf(MR.Const_Matrix3f m) {return new(m);}

        /// finds shorter arc rotation quaternion from one vector to another
        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public unsafe Quaternionf(MR.Const_Vector3f from, MR.Const_Vector3f to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_MR_Vector3f(MR.Const_Vector3f._Underlying *from, MR.Const_Vector3f._Underlying *to);
            _UnderlyingPtr = __MR_Quaternionf_Construct_2_const_MR_Vector3f_ref_MR_Vector3f(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// Generated from method `MR::Quaternionf::operator=`.
        public unsafe MR.Quaternionf Assign(MR.Const_Quaternionf _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_AssignFromAnother(_Underlying *_this, MR.Quaternionf._Underlying *_other);
            return new(__MR_Quaternionf_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// scales this quaternion to make its norm unit
        /// Generated from method `MR::Quaternionf::normalize`.
        public unsafe void Normalize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_normalize", ExactSpelling = true)]
            extern static void __MR_Quaternionf_normalize(_Underlying *_this);
            __MR_Quaternionf_normalize(_UnderlyingPtr);
        }

        /// Generated from method `MR::Quaternionf::operator*=`.
        public unsafe MR.Quaternionf MulAssign(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_mul_assign", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_mul_assign(_Underlying *_this, float s);
            return new(__MR_Quaternionf_mul_assign(_UnderlyingPtr, s), is_owning: false);
        }

        /// Generated from method `MR::Quaternionf::operator/=`.
        public unsafe MR.Quaternionf DivAssign(float s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaternionf_div_assign", ExactSpelling = true)]
            extern static MR.Quaternionf._Underlying *__MR_Quaternionf_div_assign(_Underlying *_this, float s);
            return new(__MR_Quaternionf_div_assign(_UnderlyingPtr, s), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Quaternionf` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Quaternionf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Quaternionf`/`Const_Quaternionf` directly.
    public class _InOptMut_Quaternionf
    {
        public Quaternionf? Opt;

        public _InOptMut_Quaternionf() {}
        public _InOptMut_Quaternionf(Quaternionf value) {Opt = value;}
        public static implicit operator _InOptMut_Quaternionf(Quaternionf value) {return new(value);}
    }

    /// This is used for optional parameters of class `Quaternionf` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Quaternionf`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Quaternionf`/`Const_Quaternionf` to pass it to the function.
    public class _InOptConst_Quaternionf
    {
        public Const_Quaternionf? Opt;

        public _InOptConst_Quaternionf() {}
        public _InOptConst_Quaternionf(Const_Quaternionf value) {Opt = value;}
        public static implicit operator _InOptConst_Quaternionf(Const_Quaternionf value) {return new(value);}

        /// Generated from constructor `MR::Quaternionf::Quaternionf`.
        public static unsafe implicit operator _InOptConst_Quaternionf(MR.Const_Matrix3f m) {return new MR.Quaternionf(m);}
    }

    /// Represents a quaternion following the notations from
    /// https://en.wikipedia.org/wiki/Quaternion
    /// Generated from class `MR::Quaterniond`.
    /// This is the const half of the class.
    public class Const_Quaterniond : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Quaterniond(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Destroy", ExactSpelling = true)]
            extern static void __MR_Quaterniond_Destroy(_Underlying *_this);
            __MR_Quaterniond_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Quaterniond() {Dispose(false);}

        ///< real part of the quaternion
        public unsafe double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Get_a", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_Get_a(_Underlying *_this);
                return *__MR_Quaterniond_Get_a(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public unsafe double B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Get_b", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_Get_b(_Underlying *_this);
                return *__MR_Quaterniond_Get_b(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public unsafe double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Get_c", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_Get_c(_Underlying *_this);
                return *__MR_Quaterniond_Get_c(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public unsafe double D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Get_d", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_Get_d(_Underlying *_this);
                return *__MR_Quaterniond_Get_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Quaterniond() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_DefaultConstruct();
            _UnderlyingPtr = __MR_Quaterniond_DefaultConstruct();
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Const_Quaterniond(MR.Const_Quaterniond _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_ConstructFromAnother(MR.Quaterniond._Underlying *_other);
            _UnderlyingPtr = __MR_Quaterniond_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Const_Quaterniond(double a, double b, double c, double d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_4", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_4(double a, double b, double c, double d);
            _UnderlyingPtr = __MR_Quaterniond_Construct_4(a, b, c, d);
        }

        /// \related Quaternion
        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Const_Quaterniond(MR.Const_Vector3d axis, double angle) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_double", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_double(MR.Const_Vector3d._Underlying *axis, double angle);
            _UnderlyingPtr = __MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_double(axis._UnderlyingPtr, angle);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Const_Quaterniond(double real, MR.Const_Vector3d im) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_2_double", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_2_double(double real, MR.Const_Vector3d._Underlying *im);
            _UnderlyingPtr = __MR_Quaterniond_Construct_2_double(real, im._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Const_Quaterniond(MR.Const_Matrix3d m) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_1", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_1(MR.Const_Matrix3d._Underlying *m);
            _UnderlyingPtr = __MR_Quaterniond_Construct_1(m._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public static unsafe implicit operator Const_Quaterniond(MR.Const_Matrix3d m) {return new(m);}

        /// finds shorter arc rotation quaternion from one vector to another
        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Const_Quaterniond(MR.Const_Vector3d from, MR.Const_Vector3d to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_MR_Vector3d(MR.Const_Vector3d._Underlying *from, MR.Const_Vector3d._Underlying *to);
            _UnderlyingPtr = __MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_MR_Vector3d(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// converts this into 3x3 rotation matrix
        /// Generated from conversion operator `MR::Quaterniond::operator MR::Matrix3d`.
        public static unsafe implicit operator MR.Matrix3d(MR.Const_Quaterniond _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_ConvertTo_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Quaterniond_ConvertTo_MR_Matrix3d(MR.Const_Quaterniond._Underlying *_this);
            return __MR_Quaterniond_ConvertTo_MR_Matrix3d(_this._UnderlyingPtr);
        }

        /// returns imaginary part of the quaternion as a vector
        /// Generated from method `MR::Quaterniond::im`.
        public unsafe MR.Vector3d Im()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_im", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Quaterniond_im(_Underlying *_this);
            return __MR_Quaterniond_im(_UnderlyingPtr);
        }

        /// returns angle of rotation encoded in this quaternion
        /// Generated from method `MR::Quaterniond::angle`.
        public unsafe double Angle()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_angle", ExactSpelling = true)]
            extern static double __MR_Quaterniond_angle(_Underlying *_this);
            return __MR_Quaterniond_angle(_UnderlyingPtr);
        }

        /// returns axis of rotation encoded in this quaternion
        /// Generated from method `MR::Quaterniond::axis`.
        public unsafe MR.Vector3d Axis()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_axis", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Quaterniond_axis(_Underlying *_this);
            return __MR_Quaterniond_axis(_UnderlyingPtr);
        }

        /// Generated from method `MR::Quaterniond::normSq`.
        public unsafe double NormSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_normSq", ExactSpelling = true)]
            extern static double __MR_Quaterniond_normSq(_Underlying *_this);
            return __MR_Quaterniond_normSq(_UnderlyingPtr);
        }

        /// Generated from method `MR::Quaterniond::norm`.
        public unsafe double Norm()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_norm", ExactSpelling = true)]
            extern static double __MR_Quaterniond_norm(_Underlying *_this);
            return __MR_Quaterniond_norm(_UnderlyingPtr);
        }

        /// returns quaternion representing the same rotation, using the opposite rotation direction and opposite angle
        /// Generated from method `MR::Quaterniond::operator-`.
        public static unsafe MR.Quaterniond operator-(MR.Const_Quaterniond _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_neg_MR_Quaterniond", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_neg_MR_Quaterniond(MR.Const_Quaterniond._Underlying *_this);
            return new(__MR_neg_MR_Quaterniond(_this._UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::Quaterniond::normalized`.
        public unsafe MR.Quaterniond Normalized()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_normalized", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_normalized(_Underlying *_this);
            return new(__MR_Quaterniond_normalized(_UnderlyingPtr), is_owning: true);
        }

        /// computes conjugate quaternion, which for unit quaternions encodes the opposite rotation
        /// Generated from method `MR::Quaterniond::conjugate`.
        public unsafe MR.Quaterniond Conjugate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_conjugate", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_conjugate(_Underlying *_this);
            return new(__MR_Quaterniond_conjugate(_UnderlyingPtr), is_owning: true);
        }

        /// computes reciprocal quaternion
        /// Generated from method `MR::Quaterniond::inverse`.
        public unsafe MR.Quaterniond Inverse()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_inverse", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_inverse(_Underlying *_this);
            return new(__MR_Quaterniond_inverse(_UnderlyingPtr), is_owning: true);
        }

        /// for unit quaternion returns the rotation of point p, which is faster to compute for single point;
        /// for multiple points it is faster to create matrix representation and apply it to the points
        /// Generated from method `MR::Quaterniond::operator()`.
        public unsafe MR.Vector3d Call(MR.Const_Vector3d p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_call", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Quaterniond_call(_Underlying *_this, MR.Const_Vector3d._Underlying *p);
            return __MR_Quaterniond_call(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// given t in [0,1], interpolates linearly two quaternions giving in general not-unit quaternion
        /// Generated from method `MR::Quaterniond::lerp`.
        public static unsafe MR.Quaterniond Lerp(MR.Const_Quaterniond q0, MR.Const_Quaterniond q1, double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_lerp", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_lerp(MR.Const_Quaterniond._Underlying *q0, MR.Const_Quaterniond._Underlying *q1, double t);
            return new(__MR_Quaterniond_lerp(q0._UnderlyingPtr, q1._UnderlyingPtr, t), is_owning: true);
        }

        /// given t in [0,1] and two unit quaternions, interpolates them spherically and produces another unit quaternion
        /// Generated from method `MR::Quaterniond::slerp`.
        public static unsafe MR.Quaterniond Slerp(MR.Const_Quaterniond q0, MR.Const_Quaterniond q1, double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_slerp_3_MR_Quaterniond", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_slerp_3_MR_Quaterniond(MR.Quaterniond._Underlying *q0, MR.Quaterniond._Underlying *q1, double t);
            return new(__MR_Quaterniond_slerp_3_MR_Quaterniond(q0._UnderlyingPtr, q1._UnderlyingPtr, t), is_owning: true);
        }

        /// given t in [0,1] and two rotation matrices, interpolates them spherically and produces another rotation matrix
        /// Generated from method `MR::Quaterniond::slerp`.
        public static unsafe MR.Matrix3d Slerp(MR.Const_Matrix3d m0, MR.Const_Matrix3d m1, double t)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_slerp_3_MR_Matrix3d", ExactSpelling = true)]
            extern static MR.Matrix3d __MR_Quaterniond_slerp_3_MR_Matrix3d(MR.Const_Matrix3d._Underlying *m0, MR.Const_Matrix3d._Underlying *m1, double t);
            return __MR_Quaterniond_slerp_3_MR_Matrix3d(m0._UnderlyingPtr, m1._UnderlyingPtr, t);
        }

        /// given t in [0,1] and rigid transformations, interpolates them spherically and produces another rigid transformation;
        /// p is the only point that will have straight line movement during interpolation
        /// Generated from method `MR::Quaterniond::slerp`.
        /// Parameter `p` defaults to `{}`.
        public static unsafe MR.AffineXf3d Slerp(MR.Const_AffineXf3d xf0, MR.Const_AffineXf3d xf1, double t, MR.Const_Vector3d? p = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_slerp_4", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_Quaterniond_slerp_4(MR.Const_AffineXf3d._Underlying *xf0, MR.Const_AffineXf3d._Underlying *xf1, double t, MR.Const_Vector3d._Underlying *p);
            return __MR_Quaterniond_slerp_4(xf0._UnderlyingPtr, xf1._UnderlyingPtr, t, p is not null ? p._UnderlyingPtr : null);
        }
    }

    /// Represents a quaternion following the notations from
    /// https://en.wikipedia.org/wiki/Quaternion
    /// Generated from class `MR::Quaterniond`.
    /// This is the non-const half of the class.
    public class Quaterniond : Const_Quaterniond
    {
        internal unsafe Quaterniond(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< real part of the quaternion
        public new unsafe ref double A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_GetMutable_a", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_GetMutable_a(_Underlying *_this);
                return ref *__MR_Quaterniond_GetMutable_a(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public new unsafe ref double B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_GetMutable_b", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_GetMutable_b(_Underlying *_this);
                return ref *__MR_Quaterniond_GetMutable_b(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public new unsafe ref double C
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_GetMutable_c", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_GetMutable_c(_Underlying *_this);
                return ref *__MR_Quaterniond_GetMutable_c(_UnderlyingPtr);
            }
        }

        ///< imaginary part: b*i + c*j + d*k
        public new unsafe ref double D
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_GetMutable_d", ExactSpelling = true)]
                extern static double *__MR_Quaterniond_GetMutable_d(_Underlying *_this);
                return ref *__MR_Quaterniond_GetMutable_d(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Quaterniond() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_DefaultConstruct();
            _UnderlyingPtr = __MR_Quaterniond_DefaultConstruct();
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Quaterniond(MR.Const_Quaterniond _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_ConstructFromAnother(MR.Quaterniond._Underlying *_other);
            _UnderlyingPtr = __MR_Quaterniond_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Quaterniond(double a, double b, double c, double d) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_4", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_4(double a, double b, double c, double d);
            _UnderlyingPtr = __MR_Quaterniond_Construct_4(a, b, c, d);
        }

        /// \related Quaternion
        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Quaterniond(MR.Const_Vector3d axis, double angle) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_double", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_double(MR.Const_Vector3d._Underlying *axis, double angle);
            _UnderlyingPtr = __MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_double(axis._UnderlyingPtr, angle);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Quaterniond(double real, MR.Const_Vector3d im) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_2_double", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_2_double(double real, MR.Const_Vector3d._Underlying *im);
            _UnderlyingPtr = __MR_Quaterniond_Construct_2_double(real, im._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Quaterniond(MR.Const_Matrix3d m) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_1", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_1(MR.Const_Matrix3d._Underlying *m);
            _UnderlyingPtr = __MR_Quaterniond_Construct_1(m._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public static unsafe implicit operator Quaterniond(MR.Const_Matrix3d m) {return new(m);}

        /// finds shorter arc rotation quaternion from one vector to another
        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public unsafe Quaterniond(MR.Const_Vector3d from, MR.Const_Vector3d to) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_MR_Vector3d", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_MR_Vector3d(MR.Const_Vector3d._Underlying *from, MR.Const_Vector3d._Underlying *to);
            _UnderlyingPtr = __MR_Quaterniond_Construct_2_const_MR_Vector3d_ref_MR_Vector3d(from._UnderlyingPtr, to._UnderlyingPtr);
        }

        /// Generated from method `MR::Quaterniond::operator=`.
        public unsafe MR.Quaterniond Assign(MR.Const_Quaterniond _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_AssignFromAnother(_Underlying *_this, MR.Quaterniond._Underlying *_other);
            return new(__MR_Quaterniond_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// scales this quaternion to make its norm unit
        /// Generated from method `MR::Quaterniond::normalize`.
        public unsafe void Normalize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_normalize", ExactSpelling = true)]
            extern static void __MR_Quaterniond_normalize(_Underlying *_this);
            __MR_Quaterniond_normalize(_UnderlyingPtr);
        }

        /// Generated from method `MR::Quaterniond::operator*=`.
        public unsafe MR.Quaterniond MulAssign(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_mul_assign", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_mul_assign(_Underlying *_this, double s);
            return new(__MR_Quaterniond_mul_assign(_UnderlyingPtr, s), is_owning: false);
        }

        /// Generated from method `MR::Quaterniond::operator/=`.
        public unsafe MR.Quaterniond DivAssign(double s)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Quaterniond_div_assign", ExactSpelling = true)]
            extern static MR.Quaterniond._Underlying *__MR_Quaterniond_div_assign(_Underlying *_this, double s);
            return new(__MR_Quaterniond_div_assign(_UnderlyingPtr, s), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Quaterniond` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Quaterniond`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Quaterniond`/`Const_Quaterniond` directly.
    public class _InOptMut_Quaterniond
    {
        public Quaterniond? Opt;

        public _InOptMut_Quaterniond() {}
        public _InOptMut_Quaterniond(Quaterniond value) {Opt = value;}
        public static implicit operator _InOptMut_Quaterniond(Quaterniond value) {return new(value);}
    }

    /// This is used for optional parameters of class `Quaterniond` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Quaterniond`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Quaterniond`/`Const_Quaterniond` to pass it to the function.
    public class _InOptConst_Quaterniond
    {
        public Const_Quaterniond? Opt;

        public _InOptConst_Quaterniond() {}
        public _InOptConst_Quaterniond(Const_Quaterniond value) {Opt = value;}
        public static implicit operator _InOptConst_Quaterniond(Const_Quaterniond value) {return new(value);}

        /// Generated from constructor `MR::Quaterniond::Quaterniond`.
        public static unsafe implicit operator _InOptConst_Quaterniond(MR.Const_Matrix3d m) {return new MR.Quaterniond(m);}
    }
}
