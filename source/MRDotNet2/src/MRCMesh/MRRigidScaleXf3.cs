public static partial class MR
{
    /// rigid (with scale) transformation that multiplies all distances on same scale: y = s*A*x + b,
    /// where s is a scalar, A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidScaleXf3f`.
    /// This is the const half of the class.
    public class Const_RigidScaleXf3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RigidScaleXf3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_Destroy", ExactSpelling = true)]
            extern static void __MR_RigidScaleXf3f_Destroy(_Underlying *_this);
            __MR_RigidScaleXf3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RigidScaleXf3f() {Dispose(false);}

        ///< rotation angles relative to x,y,z axes
        public unsafe MR.Const_Vector3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_RigidScaleXf3f_Get_a(_Underlying *_this);
                return new(__MR_RigidScaleXf3f_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public unsafe MR.Const_Vector3f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_RigidScaleXf3f_Get_b(_Underlying *_this);
                return new(__MR_RigidScaleXf3f_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< scaling
        public unsafe float S
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_Get_s", ExactSpelling = true)]
                extern static float *__MR_RigidScaleXf3f_Get_s(_Underlying *_this);
                return *__MR_RigidScaleXf3f_Get_s(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RigidScaleXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidScaleXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidScaleXf3f::RigidScaleXf3f`.
        public unsafe Const_RigidScaleXf3f(MR.Const_RigidScaleXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_ConstructFromAnother(MR.RigidScaleXf3f._Underlying *_other);
            _UnderlyingPtr = __MR_RigidScaleXf3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidScaleXf3f::RigidScaleXf3f`.
        public unsafe Const_RigidScaleXf3f(MR.Const_Vector3f a, MR.Const_Vector3f b, float s) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_Construct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_Construct(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, float s);
            _UnderlyingPtr = __MR_RigidScaleXf3f_Construct(a._UnderlyingPtr, b._UnderlyingPtr, s);
        }

        /// converts this into rigid (with scale) transformation, which non-linearly depends on angles
        /// Generated from method `MR::RigidScaleXf3f::rigidScaleXf`.
        public unsafe MR.AffineXf3f RigidScaleXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_rigidScaleXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_RigidScaleXf3f_rigidScaleXf(_Underlying *_this);
            return __MR_RigidScaleXf3f_rigidScaleXf(_UnderlyingPtr);
        }

        /// converts this into not-rigid transformation but with matrix, which linearly depends on angles
        /// Generated from method `MR::RigidScaleXf3f::linearXf`.
        public unsafe MR.AffineXf3f LinearXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_linearXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_RigidScaleXf3f_linearXf(_Underlying *_this);
            return __MR_RigidScaleXf3f_linearXf(_UnderlyingPtr);
        }
    }

    /// rigid (with scale) transformation that multiplies all distances on same scale: y = s*A*x + b,
    /// where s is a scalar, A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidScaleXf3f`.
    /// This is the non-const half of the class.
    public class RigidScaleXf3f : Const_RigidScaleXf3f
    {
        internal unsafe RigidScaleXf3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< rotation angles relative to x,y,z axes
        public new unsafe MR.Mut_Vector3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_RigidScaleXf3f_GetMutable_a(_Underlying *_this);
                return new(__MR_RigidScaleXf3f_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public new unsafe MR.Mut_Vector3f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_RigidScaleXf3f_GetMutable_b(_Underlying *_this);
                return new(__MR_RigidScaleXf3f_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< scaling
        public new unsafe ref float S
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_GetMutable_s", ExactSpelling = true)]
                extern static float *__MR_RigidScaleXf3f_GetMutable_s(_Underlying *_this);
                return ref *__MR_RigidScaleXf3f_GetMutable_s(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RigidScaleXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidScaleXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidScaleXf3f::RigidScaleXf3f`.
        public unsafe RigidScaleXf3f(MR.Const_RigidScaleXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_ConstructFromAnother(MR.RigidScaleXf3f._Underlying *_other);
            _UnderlyingPtr = __MR_RigidScaleXf3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidScaleXf3f::RigidScaleXf3f`.
        public unsafe RigidScaleXf3f(MR.Const_Vector3f a, MR.Const_Vector3f b, float s) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_Construct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_Construct(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, float s);
            _UnderlyingPtr = __MR_RigidScaleXf3f_Construct(a._UnderlyingPtr, b._UnderlyingPtr, s);
        }

        /// Generated from method `MR::RigidScaleXf3f::operator=`.
        public unsafe MR.RigidScaleXf3f Assign(MR.Const_RigidScaleXf3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RigidScaleXf3f._Underlying *__MR_RigidScaleXf3f_AssignFromAnother(_Underlying *_this, MR.RigidScaleXf3f._Underlying *_other);
            return new(__MR_RigidScaleXf3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RigidScaleXf3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RigidScaleXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidScaleXf3f`/`Const_RigidScaleXf3f` directly.
    public class _InOptMut_RigidScaleXf3f
    {
        public RigidScaleXf3f? Opt;

        public _InOptMut_RigidScaleXf3f() {}
        public _InOptMut_RigidScaleXf3f(RigidScaleXf3f value) {Opt = value;}
        public static implicit operator _InOptMut_RigidScaleXf3f(RigidScaleXf3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `RigidScaleXf3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RigidScaleXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidScaleXf3f`/`Const_RigidScaleXf3f` to pass it to the function.
    public class _InOptConst_RigidScaleXf3f
    {
        public Const_RigidScaleXf3f? Opt;

        public _InOptConst_RigidScaleXf3f() {}
        public _InOptConst_RigidScaleXf3f(Const_RigidScaleXf3f value) {Opt = value;}
        public static implicit operator _InOptConst_RigidScaleXf3f(Const_RigidScaleXf3f value) {return new(value);}
    }

    /// rigid (with scale) transformation that multiplies all distances on same scale: y = s*A*x + b,
    /// where s is a scalar, A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidScaleXf3d`.
    /// This is the const half of the class.
    public class Const_RigidScaleXf3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RigidScaleXf3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_Destroy", ExactSpelling = true)]
            extern static void __MR_RigidScaleXf3d_Destroy(_Underlying *_this);
            __MR_RigidScaleXf3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RigidScaleXf3d() {Dispose(false);}

        ///< rotation angles relative to x,y,z axes
        public unsafe MR.Const_Vector3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_RigidScaleXf3d_Get_a(_Underlying *_this);
                return new(__MR_RigidScaleXf3d_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public unsafe MR.Const_Vector3d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_RigidScaleXf3d_Get_b(_Underlying *_this);
                return new(__MR_RigidScaleXf3d_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< scaling
        public unsafe double S
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_Get_s", ExactSpelling = true)]
                extern static double *__MR_RigidScaleXf3d_Get_s(_Underlying *_this);
                return *__MR_RigidScaleXf3d_Get_s(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RigidScaleXf3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidScaleXf3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidScaleXf3d::RigidScaleXf3d`.
        public unsafe Const_RigidScaleXf3d(MR.Const_RigidScaleXf3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_ConstructFromAnother(MR.RigidScaleXf3d._Underlying *_other);
            _UnderlyingPtr = __MR_RigidScaleXf3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidScaleXf3d::RigidScaleXf3d`.
        public unsafe Const_RigidScaleXf3d(MR.Const_Vector3d a, MR.Const_Vector3d b, double s) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_Construct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_Construct(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b, double s);
            _UnderlyingPtr = __MR_RigidScaleXf3d_Construct(a._UnderlyingPtr, b._UnderlyingPtr, s);
        }

        /// converts this into rigid (with scale) transformation, which non-linearly depends on angles
        /// Generated from method `MR::RigidScaleXf3d::rigidScaleXf`.
        public unsafe MR.AffineXf3d RigidScaleXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_rigidScaleXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_RigidScaleXf3d_rigidScaleXf(_Underlying *_this);
            return __MR_RigidScaleXf3d_rigidScaleXf(_UnderlyingPtr);
        }

        /// converts this into not-rigid transformation but with matrix, which linearly depends on angles
        /// Generated from method `MR::RigidScaleXf3d::linearXf`.
        public unsafe MR.AffineXf3d LinearXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_linearXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_RigidScaleXf3d_linearXf(_Underlying *_this);
            return __MR_RigidScaleXf3d_linearXf(_UnderlyingPtr);
        }
    }

    /// rigid (with scale) transformation that multiplies all distances on same scale: y = s*A*x + b,
    /// where s is a scalar, A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidScaleXf3d`.
    /// This is the non-const half of the class.
    public class RigidScaleXf3d : Const_RigidScaleXf3d
    {
        internal unsafe RigidScaleXf3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< rotation angles relative to x,y,z axes
        public new unsafe MR.Mut_Vector3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_RigidScaleXf3d_GetMutable_a(_Underlying *_this);
                return new(__MR_RigidScaleXf3d_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public new unsafe MR.Mut_Vector3d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_RigidScaleXf3d_GetMutable_b(_Underlying *_this);
                return new(__MR_RigidScaleXf3d_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< scaling
        public new unsafe ref double S
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_GetMutable_s", ExactSpelling = true)]
                extern static double *__MR_RigidScaleXf3d_GetMutable_s(_Underlying *_this);
                return ref *__MR_RigidScaleXf3d_GetMutable_s(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RigidScaleXf3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidScaleXf3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidScaleXf3d::RigidScaleXf3d`.
        public unsafe RigidScaleXf3d(MR.Const_RigidScaleXf3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_ConstructFromAnother(MR.RigidScaleXf3d._Underlying *_other);
            _UnderlyingPtr = __MR_RigidScaleXf3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidScaleXf3d::RigidScaleXf3d`.
        public unsafe RigidScaleXf3d(MR.Const_Vector3d a, MR.Const_Vector3d b, double s) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_Construct", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_Construct(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b, double s);
            _UnderlyingPtr = __MR_RigidScaleXf3d_Construct(a._UnderlyingPtr, b._UnderlyingPtr, s);
        }

        /// Generated from method `MR::RigidScaleXf3d::operator=`.
        public unsafe MR.RigidScaleXf3d Assign(MR.Const_RigidScaleXf3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidScaleXf3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_RigidScaleXf3d_AssignFromAnother(_Underlying *_this, MR.RigidScaleXf3d._Underlying *_other);
            return new(__MR_RigidScaleXf3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RigidScaleXf3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RigidScaleXf3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidScaleXf3d`/`Const_RigidScaleXf3d` directly.
    public class _InOptMut_RigidScaleXf3d
    {
        public RigidScaleXf3d? Opt;

        public _InOptMut_RigidScaleXf3d() {}
        public _InOptMut_RigidScaleXf3d(RigidScaleXf3d value) {Opt = value;}
        public static implicit operator _InOptMut_RigidScaleXf3d(RigidScaleXf3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `RigidScaleXf3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RigidScaleXf3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidScaleXf3d`/`Const_RigidScaleXf3d` to pass it to the function.
    public class _InOptConst_RigidScaleXf3d
    {
        public Const_RigidScaleXf3d? Opt;

        public _InOptConst_RigidScaleXf3d() {}
        public _InOptConst_RigidScaleXf3d(Const_RigidScaleXf3d value) {Opt = value;}
        public static implicit operator _InOptConst_RigidScaleXf3d(Const_RigidScaleXf3d value) {return new(value);}
    }
}
