public static partial class MR
{
    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball1f`.
    /// This is the const half of the class.
    public class Const_Ball1f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Ball1f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_Destroy", ExactSpelling = true)]
            extern static void __MR_Ball1f_Destroy(_Underlying *_this);
            __MR_Ball1f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Ball1f() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Ball1f_Get_elements();
                return *__MR_Ball1f_Get_elements();
            }
        }

        ///< ball's center
        public unsafe float Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_Get_center", ExactSpelling = true)]
                extern static float *__MR_Ball1f_Get_center(_Underlying *_this);
                return *__MR_Ball1f_Get_center(_UnderlyingPtr);
            }
        }

        ///< ball's squared radius
        public unsafe float RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_Get_radiusSq", ExactSpelling = true)]
                extern static float *__MR_Ball1f_Get_radiusSq(_Underlying *_this);
                return *__MR_Ball1f_Get_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Ball1f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball1f_DefaultConstruct();
        }

        /// Constructs `MR::Ball1f` elementwise.
        public unsafe Const_Ball1f(float center, float radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_ConstructFrom(float center, float radiusSq);
            _UnderlyingPtr = __MR_Ball1f_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball1f::Ball1f`.
        public unsafe Const_Ball1f(MR.Const_Ball1f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_ConstructFromAnother(MR.Ball1f._Underlying *_other);
            _UnderlyingPtr = __MR_Ball1f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if given point is strictly inside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball1f::inside`.
        public unsafe bool Inside(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_inside", ExactSpelling = true)]
            extern static byte __MR_Ball1f_inside(_Underlying *_this, float *pt);
            return __MR_Ball1f_inside(_UnderlyingPtr, &pt) != 0;
        }

        /// returns true if given point is strictly outside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball1f::outside`.
        public unsafe bool Outside(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_outside", ExactSpelling = true)]
            extern static byte __MR_Ball1f_outside(_Underlying *_this, float *pt);
            return __MR_Ball1f_outside(_UnderlyingPtr, &pt) != 0;
        }
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball1f`.
    /// This is the non-const half of the class.
    public class Ball1f : Const_Ball1f
    {
        internal unsafe Ball1f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< ball's center
        public new unsafe ref float Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_GetMutable_center", ExactSpelling = true)]
                extern static float *__MR_Ball1f_GetMutable_center(_Underlying *_this);
                return ref *__MR_Ball1f_GetMutable_center(_UnderlyingPtr);
            }
        }

        ///< ball's squared radius
        public new unsafe ref float RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_GetMutable_radiusSq", ExactSpelling = true)]
                extern static float *__MR_Ball1f_GetMutable_radiusSq(_Underlying *_this);
                return ref *__MR_Ball1f_GetMutable_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Ball1f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball1f_DefaultConstruct();
        }

        /// Constructs `MR::Ball1f` elementwise.
        public unsafe Ball1f(float center, float radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_ConstructFrom(float center, float radiusSq);
            _UnderlyingPtr = __MR_Ball1f_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball1f::Ball1f`.
        public unsafe Ball1f(MR.Const_Ball1f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_ConstructFromAnother(MR.Ball1f._Underlying *_other);
            _UnderlyingPtr = __MR_Ball1f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Ball1f::operator=`.
        public unsafe MR.Ball1f Assign(MR.Const_Ball1f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Ball1f._Underlying *__MR_Ball1f_AssignFromAnother(_Underlying *_this, MR.Ball1f._Underlying *_other);
            return new(__MR_Ball1f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Ball1f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Ball1f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball1f`/`Const_Ball1f` directly.
    public class _InOptMut_Ball1f
    {
        public Ball1f? Opt;

        public _InOptMut_Ball1f() {}
        public _InOptMut_Ball1f(Ball1f value) {Opt = value;}
        public static implicit operator _InOptMut_Ball1f(Ball1f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Ball1f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Ball1f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball1f`/`Const_Ball1f` to pass it to the function.
    public class _InOptConst_Ball1f
    {
        public Const_Ball1f? Opt;

        public _InOptConst_Ball1f() {}
        public _InOptConst_Ball1f(Const_Ball1f value) {Opt = value;}
        public static implicit operator _InOptConst_Ball1f(Const_Ball1f value) {return new(value);}
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball1d`.
    /// This is the const half of the class.
    public class Const_Ball1d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Ball1d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_Destroy", ExactSpelling = true)]
            extern static void __MR_Ball1d_Destroy(_Underlying *_this);
            __MR_Ball1d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Ball1d() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Ball1d_Get_elements();
                return *__MR_Ball1d_Get_elements();
            }
        }

        ///< ball's center
        public unsafe double Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_Get_center", ExactSpelling = true)]
                extern static double *__MR_Ball1d_Get_center(_Underlying *_this);
                return *__MR_Ball1d_Get_center(_UnderlyingPtr);
            }
        }

        ///< ball's squared radius
        public unsafe double RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_Get_radiusSq", ExactSpelling = true)]
                extern static double *__MR_Ball1d_Get_radiusSq(_Underlying *_this);
                return *__MR_Ball1d_Get_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Ball1d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball1d_DefaultConstruct();
        }

        /// Constructs `MR::Ball1d` elementwise.
        public unsafe Const_Ball1d(double center, double radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_ConstructFrom(double center, double radiusSq);
            _UnderlyingPtr = __MR_Ball1d_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball1d::Ball1d`.
        public unsafe Const_Ball1d(MR.Const_Ball1d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_ConstructFromAnother(MR.Ball1d._Underlying *_other);
            _UnderlyingPtr = __MR_Ball1d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if given point is strictly inside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball1d::inside`.
        public unsafe bool Inside(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_inside", ExactSpelling = true)]
            extern static byte __MR_Ball1d_inside(_Underlying *_this, double *pt);
            return __MR_Ball1d_inside(_UnderlyingPtr, &pt) != 0;
        }

        /// returns true if given point is strictly outside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball1d::outside`.
        public unsafe bool Outside(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_outside", ExactSpelling = true)]
            extern static byte __MR_Ball1d_outside(_Underlying *_this, double *pt);
            return __MR_Ball1d_outside(_UnderlyingPtr, &pt) != 0;
        }
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball1d`.
    /// This is the non-const half of the class.
    public class Ball1d : Const_Ball1d
    {
        internal unsafe Ball1d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< ball's center
        public new unsafe ref double Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_GetMutable_center", ExactSpelling = true)]
                extern static double *__MR_Ball1d_GetMutable_center(_Underlying *_this);
                return ref *__MR_Ball1d_GetMutable_center(_UnderlyingPtr);
            }
        }

        ///< ball's squared radius
        public new unsafe ref double RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_GetMutable_radiusSq", ExactSpelling = true)]
                extern static double *__MR_Ball1d_GetMutable_radiusSq(_Underlying *_this);
                return ref *__MR_Ball1d_GetMutable_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Ball1d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball1d_DefaultConstruct();
        }

        /// Constructs `MR::Ball1d` elementwise.
        public unsafe Ball1d(double center, double radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_ConstructFrom(double center, double radiusSq);
            _UnderlyingPtr = __MR_Ball1d_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball1d::Ball1d`.
        public unsafe Ball1d(MR.Const_Ball1d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_ConstructFromAnother(MR.Ball1d._Underlying *_other);
            _UnderlyingPtr = __MR_Ball1d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Ball1d::operator=`.
        public unsafe MR.Ball1d Assign(MR.Const_Ball1d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball1d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Ball1d._Underlying *__MR_Ball1d_AssignFromAnother(_Underlying *_this, MR.Ball1d._Underlying *_other);
            return new(__MR_Ball1d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Ball1d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Ball1d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball1d`/`Const_Ball1d` directly.
    public class _InOptMut_Ball1d
    {
        public Ball1d? Opt;

        public _InOptMut_Ball1d() {}
        public _InOptMut_Ball1d(Ball1d value) {Opt = value;}
        public static implicit operator _InOptMut_Ball1d(Ball1d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Ball1d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Ball1d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball1d`/`Const_Ball1d` to pass it to the function.
    public class _InOptConst_Ball1d
    {
        public Const_Ball1d? Opt;

        public _InOptConst_Ball1d() {}
        public _InOptConst_Ball1d(Const_Ball1d value) {Opt = value;}
        public static implicit operator _InOptConst_Ball1d(Const_Ball1d value) {return new(value);}
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball2f`.
    /// This is the const half of the class.
    public class Const_Ball2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Ball2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_Destroy", ExactSpelling = true)]
            extern static void __MR_Ball2f_Destroy(_Underlying *_this);
            __MR_Ball2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Ball2f() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Ball2f_Get_elements();
                return *__MR_Ball2f_Get_elements();
            }
        }

        ///< ball's center
        public unsafe MR.Const_Vector2f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_Ball2f_Get_center(_Underlying *_this);
                return new(__MR_Ball2f_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public unsafe float RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_Get_radiusSq", ExactSpelling = true)]
                extern static float *__MR_Ball2f_Get_radiusSq(_Underlying *_this);
                return *__MR_Ball2f_Get_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Ball2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball2f_DefaultConstruct();
        }

        /// Constructs `MR::Ball2f` elementwise.
        public unsafe Const_Ball2f(MR.Vector2f center, float radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_ConstructFrom(MR.Vector2f center, float radiusSq);
            _UnderlyingPtr = __MR_Ball2f_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball2f::Ball2f`.
        public unsafe Const_Ball2f(MR.Const_Ball2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_ConstructFromAnother(MR.Ball2f._Underlying *_other);
            _UnderlyingPtr = __MR_Ball2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if given point is strictly inside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball2f::inside`.
        public unsafe bool Inside(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_inside", ExactSpelling = true)]
            extern static byte __MR_Ball2f_inside(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            return __MR_Ball2f_inside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// returns true if given point is strictly outside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball2f::outside`.
        public unsafe bool Outside(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_outside", ExactSpelling = true)]
            extern static byte __MR_Ball2f_outside(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            return __MR_Ball2f_outside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball2f`.
    /// This is the non-const half of the class.
    public class Ball2f : Const_Ball2f
    {
        internal unsafe Ball2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< ball's center
        public new unsafe MR.Mut_Vector2f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_Ball2f_GetMutable_center(_Underlying *_this);
                return new(__MR_Ball2f_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public new unsafe ref float RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_GetMutable_radiusSq", ExactSpelling = true)]
                extern static float *__MR_Ball2f_GetMutable_radiusSq(_Underlying *_this);
                return ref *__MR_Ball2f_GetMutable_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Ball2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball2f_DefaultConstruct();
        }

        /// Constructs `MR::Ball2f` elementwise.
        public unsafe Ball2f(MR.Vector2f center, float radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_ConstructFrom(MR.Vector2f center, float radiusSq);
            _UnderlyingPtr = __MR_Ball2f_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball2f::Ball2f`.
        public unsafe Ball2f(MR.Const_Ball2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_ConstructFromAnother(MR.Ball2f._Underlying *_other);
            _UnderlyingPtr = __MR_Ball2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Ball2f::operator=`.
        public unsafe MR.Ball2f Assign(MR.Const_Ball2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Ball2f._Underlying *__MR_Ball2f_AssignFromAnother(_Underlying *_this, MR.Ball2f._Underlying *_other);
            return new(__MR_Ball2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Ball2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Ball2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball2f`/`Const_Ball2f` directly.
    public class _InOptMut_Ball2f
    {
        public Ball2f? Opt;

        public _InOptMut_Ball2f() {}
        public _InOptMut_Ball2f(Ball2f value) {Opt = value;}
        public static implicit operator _InOptMut_Ball2f(Ball2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Ball2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Ball2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball2f`/`Const_Ball2f` to pass it to the function.
    public class _InOptConst_Ball2f
    {
        public Const_Ball2f? Opt;

        public _InOptConst_Ball2f() {}
        public _InOptConst_Ball2f(Const_Ball2f value) {Opt = value;}
        public static implicit operator _InOptConst_Ball2f(Const_Ball2f value) {return new(value);}
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball2d`.
    /// This is the const half of the class.
    public class Const_Ball2d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Ball2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_Destroy", ExactSpelling = true)]
            extern static void __MR_Ball2d_Destroy(_Underlying *_this);
            __MR_Ball2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Ball2d() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Ball2d_Get_elements();
                return *__MR_Ball2d_Get_elements();
            }
        }

        ///< ball's center
        public unsafe MR.Const_Vector2d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_Ball2d_Get_center(_Underlying *_this);
                return new(__MR_Ball2d_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public unsafe double RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_Get_radiusSq", ExactSpelling = true)]
                extern static double *__MR_Ball2d_Get_radiusSq(_Underlying *_this);
                return *__MR_Ball2d_Get_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Ball2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball2d_DefaultConstruct();
        }

        /// Constructs `MR::Ball2d` elementwise.
        public unsafe Const_Ball2d(MR.Vector2d center, double radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_ConstructFrom(MR.Vector2d center, double radiusSq);
            _UnderlyingPtr = __MR_Ball2d_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball2d::Ball2d`.
        public unsafe Const_Ball2d(MR.Const_Ball2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_ConstructFromAnother(MR.Ball2d._Underlying *_other);
            _UnderlyingPtr = __MR_Ball2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if given point is strictly inside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball2d::inside`.
        public unsafe bool Inside(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_inside", ExactSpelling = true)]
            extern static byte __MR_Ball2d_inside(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            return __MR_Ball2d_inside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// returns true if given point is strictly outside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball2d::outside`.
        public unsafe bool Outside(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_outside", ExactSpelling = true)]
            extern static byte __MR_Ball2d_outside(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            return __MR_Ball2d_outside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball2d`.
    /// This is the non-const half of the class.
    public class Ball2d : Const_Ball2d
    {
        internal unsafe Ball2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< ball's center
        public new unsafe MR.Mut_Vector2d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_Ball2d_GetMutable_center(_Underlying *_this);
                return new(__MR_Ball2d_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public new unsafe ref double RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_GetMutable_radiusSq", ExactSpelling = true)]
                extern static double *__MR_Ball2d_GetMutable_radiusSq(_Underlying *_this);
                return ref *__MR_Ball2d_GetMutable_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Ball2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball2d_DefaultConstruct();
        }

        /// Constructs `MR::Ball2d` elementwise.
        public unsafe Ball2d(MR.Vector2d center, double radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_ConstructFrom(MR.Vector2d center, double radiusSq);
            _UnderlyingPtr = __MR_Ball2d_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball2d::Ball2d`.
        public unsafe Ball2d(MR.Const_Ball2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_ConstructFromAnother(MR.Ball2d._Underlying *_other);
            _UnderlyingPtr = __MR_Ball2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Ball2d::operator=`.
        public unsafe MR.Ball2d Assign(MR.Const_Ball2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Ball2d._Underlying *__MR_Ball2d_AssignFromAnother(_Underlying *_this, MR.Ball2d._Underlying *_other);
            return new(__MR_Ball2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Ball2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Ball2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball2d`/`Const_Ball2d` directly.
    public class _InOptMut_Ball2d
    {
        public Ball2d? Opt;

        public _InOptMut_Ball2d() {}
        public _InOptMut_Ball2d(Ball2d value) {Opt = value;}
        public static implicit operator _InOptMut_Ball2d(Ball2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Ball2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Ball2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball2d`/`Const_Ball2d` to pass it to the function.
    public class _InOptConst_Ball2d
    {
        public Const_Ball2d? Opt;

        public _InOptConst_Ball2d() {}
        public _InOptConst_Ball2d(Const_Ball2d value) {Opt = value;}
        public static implicit operator _InOptConst_Ball2d(Const_Ball2d value) {return new(value);}
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball3f`.
    /// This is the const half of the class.
    public class Const_Ball3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Ball3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Ball3f_Destroy(_Underlying *_this);
            __MR_Ball3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Ball3f() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Ball3f_Get_elements();
                return *__MR_Ball3f_Get_elements();
            }
        }

        ///< ball's center
        public unsafe MR.Const_Vector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Ball3f_Get_center(_Underlying *_this);
                return new(__MR_Ball3f_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public unsafe float RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_Get_radiusSq", ExactSpelling = true)]
                extern static float *__MR_Ball3f_Get_radiusSq(_Underlying *_this);
                return *__MR_Ball3f_Get_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Ball3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball3f_DefaultConstruct();
        }

        /// Constructs `MR::Ball3f` elementwise.
        public unsafe Const_Ball3f(MR.Vector3f center, float radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_ConstructFrom(MR.Vector3f center, float radiusSq);
            _UnderlyingPtr = __MR_Ball3f_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball3f::Ball3f`.
        public unsafe Const_Ball3f(MR.Const_Ball3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_ConstructFromAnother(MR.Ball3f._Underlying *_other);
            _UnderlyingPtr = __MR_Ball3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if given point is strictly inside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball3f::inside`.
        public unsafe bool Inside(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_inside", ExactSpelling = true)]
            extern static byte __MR_Ball3f_inside(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Ball3f_inside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// returns true if given point is strictly outside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball3f::outside`.
        public unsafe bool Outside(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_outside", ExactSpelling = true)]
            extern static byte __MR_Ball3f_outside(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Ball3f_outside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball3f`.
    /// This is the non-const half of the class.
    public class Ball3f : Const_Ball3f
    {
        internal unsafe Ball3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< ball's center
        public new unsafe MR.Mut_Vector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Ball3f_GetMutable_center(_Underlying *_this);
                return new(__MR_Ball3f_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public new unsafe ref float RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_GetMutable_radiusSq", ExactSpelling = true)]
                extern static float *__MR_Ball3f_GetMutable_radiusSq(_Underlying *_this);
                return ref *__MR_Ball3f_GetMutable_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Ball3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball3f_DefaultConstruct();
        }

        /// Constructs `MR::Ball3f` elementwise.
        public unsafe Ball3f(MR.Vector3f center, float radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_ConstructFrom(MR.Vector3f center, float radiusSq);
            _UnderlyingPtr = __MR_Ball3f_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball3f::Ball3f`.
        public unsafe Ball3f(MR.Const_Ball3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_ConstructFromAnother(MR.Ball3f._Underlying *_other);
            _UnderlyingPtr = __MR_Ball3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Ball3f::operator=`.
        public unsafe MR.Ball3f Assign(MR.Const_Ball3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Ball3f._Underlying *__MR_Ball3f_AssignFromAnother(_Underlying *_this, MR.Ball3f._Underlying *_other);
            return new(__MR_Ball3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Ball3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Ball3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball3f`/`Const_Ball3f` directly.
    public class _InOptMut_Ball3f
    {
        public Ball3f? Opt;

        public _InOptMut_Ball3f() {}
        public _InOptMut_Ball3f(Ball3f value) {Opt = value;}
        public static implicit operator _InOptMut_Ball3f(Ball3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Ball3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Ball3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball3f`/`Const_Ball3f` to pass it to the function.
    public class _InOptConst_Ball3f
    {
        public Const_Ball3f? Opt;

        public _InOptConst_Ball3f() {}
        public _InOptConst_Ball3f(Const_Ball3f value) {Opt = value;}
        public static implicit operator _InOptConst_Ball3f(Const_Ball3f value) {return new(value);}
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball3d`.
    /// This is the const half of the class.
    public class Const_Ball3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Ball3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Ball3d_Destroy(_Underlying *_this);
            __MR_Ball3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Ball3d() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Ball3d_Get_elements();
                return *__MR_Ball3d_Get_elements();
            }
        }

        ///< ball's center
        public unsafe MR.Const_Vector3d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_Ball3d_Get_center(_Underlying *_this);
                return new(__MR_Ball3d_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public unsafe double RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_Get_radiusSq", ExactSpelling = true)]
                extern static double *__MR_Ball3d_Get_radiusSq(_Underlying *_this);
                return *__MR_Ball3d_Get_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Ball3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball3d_DefaultConstruct();
        }

        /// Constructs `MR::Ball3d` elementwise.
        public unsafe Const_Ball3d(MR.Vector3d center, double radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_ConstructFrom(MR.Vector3d center, double radiusSq);
            _UnderlyingPtr = __MR_Ball3d_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball3d::Ball3d`.
        public unsafe Const_Ball3d(MR.Const_Ball3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_ConstructFromAnother(MR.Ball3d._Underlying *_other);
            _UnderlyingPtr = __MR_Ball3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if given point is strictly inside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball3d::inside`.
        public unsafe bool Inside(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_inside", ExactSpelling = true)]
            extern static byte __MR_Ball3d_inside(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            return __MR_Ball3d_inside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// returns true if given point is strictly outside the ball (not on its spherical surface)
        /// Generated from method `MR::Ball3d::outside`.
        public unsafe bool Outside(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_outside", ExactSpelling = true)]
            extern static byte __MR_Ball3d_outside(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            return __MR_Ball3d_outside(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }
    }

    /// a ball = points surrounded by a sphere in arbitrary space with vector type V
    /// Generated from class `MR::Ball3d`.
    /// This is the non-const half of the class.
    public class Ball3d : Const_Ball3d
    {
        internal unsafe Ball3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< ball's center
        public new unsafe MR.Mut_Vector3d Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_Ball3d_GetMutable_center(_Underlying *_this);
                return new(__MR_Ball3d_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< ball's squared radius
        public new unsafe ref double RadiusSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_GetMutable_radiusSq", ExactSpelling = true)]
                extern static double *__MR_Ball3d_GetMutable_radiusSq(_Underlying *_this);
                return ref *__MR_Ball3d_GetMutable_radiusSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Ball3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Ball3d_DefaultConstruct();
        }

        /// Constructs `MR::Ball3d` elementwise.
        public unsafe Ball3d(MR.Vector3d center, double radiusSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_ConstructFrom", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_ConstructFrom(MR.Vector3d center, double radiusSq);
            _UnderlyingPtr = __MR_Ball3d_ConstructFrom(center, radiusSq);
        }

        /// Generated from constructor `MR::Ball3d::Ball3d`.
        public unsafe Ball3d(MR.Const_Ball3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_ConstructFromAnother(MR.Ball3d._Underlying *_other);
            _UnderlyingPtr = __MR_Ball3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Ball3d::operator=`.
        public unsafe MR.Ball3d Assign(MR.Const_Ball3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Ball3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Ball3d._Underlying *__MR_Ball3d_AssignFromAnother(_Underlying *_this, MR.Ball3d._Underlying *_other);
            return new(__MR_Ball3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Ball3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Ball3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball3d`/`Const_Ball3d` directly.
    public class _InOptMut_Ball3d
    {
        public Ball3d? Opt;

        public _InOptMut_Ball3d() {}
        public _InOptMut_Ball3d(Ball3d value) {Opt = value;}
        public static implicit operator _InOptMut_Ball3d(Ball3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Ball3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Ball3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Ball3d`/`Const_Ball3d` to pass it to the function.
    public class _InOptConst_Ball3d
    {
        public Const_Ball3d? Opt;

        public _InOptConst_Ball3d() {}
        public _InOptConst_Ball3d(Const_Ball3d value) {Opt = value;}
        public static implicit operator _InOptConst_Ball3d(Const_Ball3d value) {return new(value);}
    }
}
