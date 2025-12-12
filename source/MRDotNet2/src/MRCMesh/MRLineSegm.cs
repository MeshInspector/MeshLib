public static partial class MR
{
    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm2f`.
    /// This is the const half of the class.
    public class Const_LineSegm2f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LineSegm2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_Destroy", ExactSpelling = true)]
            extern static void __MR_LineSegm2f_Destroy(_Underlying *_this);
            __MR_LineSegm2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LineSegm2f() {Dispose(false);}

        public unsafe MR.Const_Vector2f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_LineSegm2f_Get_a(_Underlying *_this);
                return new(__MR_LineSegm2f_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector2f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_LineSegm2f_Get_b(_Underlying *_this);
                return new(__MR_LineSegm2f_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LineSegm2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm2f::LineSegm2f`.
        public unsafe Const_LineSegm2f(MR.Const_LineSegm2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_ConstructFromAnother(MR.LineSegm2f._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm2f::LineSegm2f`.
        public unsafe Const_LineSegm2f(MR.Const_Vector2f a, MR.Const_Vector2f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_Construct", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_Construct(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm2f_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns directional vector of the line
        /// Generated from method `MR::LineSegm2f::dir`.
        public unsafe MR.Vector2f Dir()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_dir", ExactSpelling = true)]
            extern static MR.Vector2f __MR_LineSegm2f_dir(_Underlying *_this);
            return __MR_LineSegm2f_dir(_UnderlyingPtr);
        }

        /// returns squared length of this line segment
        /// Generated from method `MR::LineSegm2f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_lengthSq", ExactSpelling = true)]
            extern static float __MR_LineSegm2f_lengthSq(_Underlying *_this);
            return __MR_LineSegm2f_lengthSq(_UnderlyingPtr);
        }

        /// returns the length of this line segment
        /// Generated from method `MR::LineSegm2f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_length", ExactSpelling = true)]
            extern static float __MR_LineSegm2f_length(_Underlying *_this);
            return __MR_LineSegm2f_length(_UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns a and param=1 returns b
        /// Generated from method `MR::LineSegm2f::operator()`.
        public unsafe MR.Vector2f Call(float param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_call", ExactSpelling = true)]
            extern static MR.Vector2f __MR_LineSegm2f_call(_Underlying *_this, float param);
            return __MR_LineSegm2f_call(_UnderlyingPtr, param);
        }
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm2f`.
    /// This is the non-const half of the class.
    public class LineSegm2f : Const_LineSegm2f
    {
        internal unsafe LineSegm2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector2f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_LineSegm2f_GetMutable_a(_Underlying *_this);
                return new(__MR_LineSegm2f_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector2f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_LineSegm2f_GetMutable_b(_Underlying *_this);
                return new(__MR_LineSegm2f_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LineSegm2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm2f::LineSegm2f`.
        public unsafe LineSegm2f(MR.Const_LineSegm2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_ConstructFromAnother(MR.LineSegm2f._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm2f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm2f::LineSegm2f`.
        public unsafe LineSegm2f(MR.Const_Vector2f a, MR.Const_Vector2f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_Construct", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_Construct(MR.Const_Vector2f._Underlying *a, MR.Const_Vector2f._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm2f_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::LineSegm2f::operator=`.
        public unsafe MR.LineSegm2f Assign(MR.Const_LineSegm2f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm2f._Underlying *__MR_LineSegm2f_AssignFromAnother(_Underlying *_this, MR.LineSegm2f._Underlying *_other);
            return new(__MR_LineSegm2f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `LineSegm2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LineSegm2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm2f`/`Const_LineSegm2f` directly.
    public class _InOptMut_LineSegm2f
    {
        public LineSegm2f? Opt;

        public _InOptMut_LineSegm2f() {}
        public _InOptMut_LineSegm2f(LineSegm2f value) {Opt = value;}
        public static implicit operator _InOptMut_LineSegm2f(LineSegm2f value) {return new(value);}
    }

    /// This is used for optional parameters of class `LineSegm2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LineSegm2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm2f`/`Const_LineSegm2f` to pass it to the function.
    public class _InOptConst_LineSegm2f
    {
        public Const_LineSegm2f? Opt;

        public _InOptConst_LineSegm2f() {}
        public _InOptConst_LineSegm2f(Const_LineSegm2f value) {Opt = value;}
        public static implicit operator _InOptConst_LineSegm2f(Const_LineSegm2f value) {return new(value);}
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm2d`.
    /// This is the const half of the class.
    public class Const_LineSegm2d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LineSegm2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_Destroy", ExactSpelling = true)]
            extern static void __MR_LineSegm2d_Destroy(_Underlying *_this);
            __MR_LineSegm2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LineSegm2d() {Dispose(false);}

        public unsafe MR.Const_Vector2d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_LineSegm2d_Get_a(_Underlying *_this);
                return new(__MR_LineSegm2d_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector2d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector2d._Underlying *__MR_LineSegm2d_Get_b(_Underlying *_this);
                return new(__MR_LineSegm2d_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LineSegm2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm2d::LineSegm2d`.
        public unsafe Const_LineSegm2d(MR.Const_LineSegm2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_ConstructFromAnother(MR.LineSegm2d._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm2d::LineSegm2d`.
        public unsafe Const_LineSegm2d(MR.Const_Vector2d a, MR.Const_Vector2d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_Construct", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_Construct(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm2d_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns directional vector of the line
        /// Generated from method `MR::LineSegm2d::dir`.
        public unsafe MR.Vector2d Dir()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_dir", ExactSpelling = true)]
            extern static MR.Vector2d __MR_LineSegm2d_dir(_Underlying *_this);
            return __MR_LineSegm2d_dir(_UnderlyingPtr);
        }

        /// returns squared length of this line segment
        /// Generated from method `MR::LineSegm2d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_lengthSq", ExactSpelling = true)]
            extern static double __MR_LineSegm2d_lengthSq(_Underlying *_this);
            return __MR_LineSegm2d_lengthSq(_UnderlyingPtr);
        }

        /// returns the length of this line segment
        /// Generated from method `MR::LineSegm2d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_length", ExactSpelling = true)]
            extern static double __MR_LineSegm2d_length(_Underlying *_this);
            return __MR_LineSegm2d_length(_UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns a and param=1 returns b
        /// Generated from method `MR::LineSegm2d::operator()`.
        public unsafe MR.Vector2d Call(double param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_call", ExactSpelling = true)]
            extern static MR.Vector2d __MR_LineSegm2d_call(_Underlying *_this, double param);
            return __MR_LineSegm2d_call(_UnderlyingPtr, param);
        }
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm2d`.
    /// This is the non-const half of the class.
    public class LineSegm2d : Const_LineSegm2d
    {
        internal unsafe LineSegm2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector2d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_LineSegm2d_GetMutable_a(_Underlying *_this);
                return new(__MR_LineSegm2d_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector2d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector2d._Underlying *__MR_LineSegm2d_GetMutable_b(_Underlying *_this);
                return new(__MR_LineSegm2d_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LineSegm2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm2d::LineSegm2d`.
        public unsafe LineSegm2d(MR.Const_LineSegm2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_ConstructFromAnother(MR.LineSegm2d._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm2d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm2d::LineSegm2d`.
        public unsafe LineSegm2d(MR.Const_Vector2d a, MR.Const_Vector2d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_Construct", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_Construct(MR.Const_Vector2d._Underlying *a, MR.Const_Vector2d._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm2d_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::LineSegm2d::operator=`.
        public unsafe MR.LineSegm2d Assign(MR.Const_LineSegm2d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm2d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm2d._Underlying *__MR_LineSegm2d_AssignFromAnother(_Underlying *_this, MR.LineSegm2d._Underlying *_other);
            return new(__MR_LineSegm2d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `LineSegm2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LineSegm2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm2d`/`Const_LineSegm2d` directly.
    public class _InOptMut_LineSegm2d
    {
        public LineSegm2d? Opt;

        public _InOptMut_LineSegm2d() {}
        public _InOptMut_LineSegm2d(LineSegm2d value) {Opt = value;}
        public static implicit operator _InOptMut_LineSegm2d(LineSegm2d value) {return new(value);}
    }

    /// This is used for optional parameters of class `LineSegm2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LineSegm2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm2d`/`Const_LineSegm2d` to pass it to the function.
    public class _InOptConst_LineSegm2d
    {
        public Const_LineSegm2d? Opt;

        public _InOptConst_LineSegm2d() {}
        public _InOptConst_LineSegm2d(Const_LineSegm2d value) {Opt = value;}
        public static implicit operator _InOptConst_LineSegm2d(Const_LineSegm2d value) {return new(value);}
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm3f`.
    /// This is the const half of the class.
    public class Const_LineSegm3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LineSegm3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_Destroy", ExactSpelling = true)]
            extern static void __MR_LineSegm3f_Destroy(_Underlying *_this);
            __MR_LineSegm3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LineSegm3f() {Dispose(false);}

        public unsafe MR.Const_Vector3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_LineSegm3f_Get_a(_Underlying *_this);
                return new(__MR_LineSegm3f_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_LineSegm3f_Get_b(_Underlying *_this);
                return new(__MR_LineSegm3f_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LineSegm3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm3f::LineSegm3f`.
        public unsafe Const_LineSegm3f(MR.Const_LineSegm3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_ConstructFromAnother(MR.LineSegm3f._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm3f::LineSegm3f`.
        public unsafe Const_LineSegm3f(MR.Const_Vector3f a, MR.Const_Vector3f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_Construct", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_Construct(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm3f_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns directional vector of the line
        /// Generated from method `MR::LineSegm3f::dir`.
        public unsafe MR.Vector3f Dir()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_dir", ExactSpelling = true)]
            extern static MR.Vector3f __MR_LineSegm3f_dir(_Underlying *_this);
            return __MR_LineSegm3f_dir(_UnderlyingPtr);
        }

        /// returns squared length of this line segment
        /// Generated from method `MR::LineSegm3f::lengthSq`.
        public unsafe float LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_lengthSq", ExactSpelling = true)]
            extern static float __MR_LineSegm3f_lengthSq(_Underlying *_this);
            return __MR_LineSegm3f_lengthSq(_UnderlyingPtr);
        }

        /// returns the length of this line segment
        /// Generated from method `MR::LineSegm3f::length`.
        public unsafe float Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_length", ExactSpelling = true)]
            extern static float __MR_LineSegm3f_length(_Underlying *_this);
            return __MR_LineSegm3f_length(_UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns a and param=1 returns b
        /// Generated from method `MR::LineSegm3f::operator()`.
        public unsafe MR.Vector3f Call(float param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_call", ExactSpelling = true)]
            extern static MR.Vector3f __MR_LineSegm3f_call(_Underlying *_this, float param);
            return __MR_LineSegm3f_call(_UnderlyingPtr, param);
        }
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm3f`.
    /// This is the non-const half of the class.
    public class LineSegm3f : Const_LineSegm3f
    {
        internal unsafe LineSegm3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_LineSegm3f_GetMutable_a(_Underlying *_this);
                return new(__MR_LineSegm3f_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_LineSegm3f_GetMutable_b(_Underlying *_this);
                return new(__MR_LineSegm3f_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LineSegm3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm3f::LineSegm3f`.
        public unsafe LineSegm3f(MR.Const_LineSegm3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_ConstructFromAnother(MR.LineSegm3f._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm3f::LineSegm3f`.
        public unsafe LineSegm3f(MR.Const_Vector3f a, MR.Const_Vector3f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_Construct", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_Construct(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm3f_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::LineSegm3f::operator=`.
        public unsafe MR.LineSegm3f Assign(MR.Const_LineSegm3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm3f._Underlying *__MR_LineSegm3f_AssignFromAnother(_Underlying *_this, MR.LineSegm3f._Underlying *_other);
            return new(__MR_LineSegm3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `LineSegm3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LineSegm3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm3f`/`Const_LineSegm3f` directly.
    public class _InOptMut_LineSegm3f
    {
        public LineSegm3f? Opt;

        public _InOptMut_LineSegm3f() {}
        public _InOptMut_LineSegm3f(LineSegm3f value) {Opt = value;}
        public static implicit operator _InOptMut_LineSegm3f(LineSegm3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `LineSegm3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LineSegm3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm3f`/`Const_LineSegm3f` to pass it to the function.
    public class _InOptConst_LineSegm3f
    {
        public Const_LineSegm3f? Opt;

        public _InOptConst_LineSegm3f() {}
        public _InOptConst_LineSegm3f(Const_LineSegm3f value) {Opt = value;}
        public static implicit operator _InOptConst_LineSegm3f(Const_LineSegm3f value) {return new(value);}
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm3d`.
    /// This is the const half of the class.
    public class Const_LineSegm3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LineSegm3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_Destroy", ExactSpelling = true)]
            extern static void __MR_LineSegm3d_Destroy(_Underlying *_this);
            __MR_LineSegm3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LineSegm3d() {Dispose(false);}

        public unsafe MR.Const_Vector3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_LineSegm3d_Get_a(_Underlying *_this);
                return new(__MR_LineSegm3d_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector3d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_LineSegm3d_Get_b(_Underlying *_this);
                return new(__MR_LineSegm3d_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LineSegm3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm3d::LineSegm3d`.
        public unsafe Const_LineSegm3d(MR.Const_LineSegm3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_ConstructFromAnother(MR.LineSegm3d._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm3d::LineSegm3d`.
        public unsafe Const_LineSegm3d(MR.Const_Vector3d a, MR.Const_Vector3d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_Construct", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_Construct(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm3d_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns directional vector of the line
        /// Generated from method `MR::LineSegm3d::dir`.
        public unsafe MR.Vector3d Dir()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_dir", ExactSpelling = true)]
            extern static MR.Vector3d __MR_LineSegm3d_dir(_Underlying *_this);
            return __MR_LineSegm3d_dir(_UnderlyingPtr);
        }

        /// returns squared length of this line segment
        /// Generated from method `MR::LineSegm3d::lengthSq`.
        public unsafe double LengthSq()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_lengthSq", ExactSpelling = true)]
            extern static double __MR_LineSegm3d_lengthSq(_Underlying *_this);
            return __MR_LineSegm3d_lengthSq(_UnderlyingPtr);
        }

        /// returns the length of this line segment
        /// Generated from method `MR::LineSegm3d::length`.
        public unsafe double Length()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_length", ExactSpelling = true)]
            extern static double __MR_LineSegm3d_length(_Underlying *_this);
            return __MR_LineSegm3d_length(_UnderlyingPtr);
        }

        /// returns point on the line, where param=0 returns a and param=1 returns b
        /// Generated from method `MR::LineSegm3d::operator()`.
        public unsafe MR.Vector3d Call(double param)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_call", ExactSpelling = true)]
            extern static MR.Vector3d __MR_LineSegm3d_call(_Underlying *_this, double param);
            return __MR_LineSegm3d_call(_UnderlyingPtr, param);
        }
    }

    // a segment of straight dimensional line
    /// Generated from class `MR::LineSegm3d`.
    /// This is the non-const half of the class.
    public class LineSegm3d : Const_LineSegm3d
    {
        internal unsafe LineSegm3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_LineSegm3d_GetMutable_a(_Underlying *_this);
                return new(__MR_LineSegm3d_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector3d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_LineSegm3d_GetMutable_b(_Underlying *_this);
                return new(__MR_LineSegm3d_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LineSegm3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_DefaultConstruct();
            _UnderlyingPtr = __MR_LineSegm3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::LineSegm3d::LineSegm3d`.
        public unsafe LineSegm3d(MR.Const_LineSegm3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_ConstructFromAnother(MR.LineSegm3d._Underlying *_other);
            _UnderlyingPtr = __MR_LineSegm3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::LineSegm3d::LineSegm3d`.
        public unsafe LineSegm3d(MR.Const_Vector3d a, MR.Const_Vector3d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_Construct", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_Construct(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            _UnderlyingPtr = __MR_LineSegm3d_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::LineSegm3d::operator=`.
        public unsafe MR.LineSegm3d Assign(MR.Const_LineSegm3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LineSegm3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LineSegm3d._Underlying *__MR_LineSegm3d_AssignFromAnother(_Underlying *_this, MR.LineSegm3d._Underlying *_other);
            return new(__MR_LineSegm3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `LineSegm3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LineSegm3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm3d`/`Const_LineSegm3d` directly.
    public class _InOptMut_LineSegm3d
    {
        public LineSegm3d? Opt;

        public _InOptMut_LineSegm3d() {}
        public _InOptMut_LineSegm3d(LineSegm3d value) {Opt = value;}
        public static implicit operator _InOptMut_LineSegm3d(LineSegm3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `LineSegm3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LineSegm3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LineSegm3d`/`Const_LineSegm3d` to pass it to the function.
    public class _InOptConst_LineSegm3d
    {
        public Const_LineSegm3d? Opt;

        public _InOptConst_LineSegm3d() {}
        public _InOptConst_LineSegm3d(Const_LineSegm3d value) {Opt = value;}
        public static implicit operator _InOptConst_LineSegm3d(Const_LineSegm3d value) {return new(value);}
    }
}
