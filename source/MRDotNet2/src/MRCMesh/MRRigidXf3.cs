public static partial class MR
{
    /// rigid transformation preserving all distances: y = A*x + b,
    /// where A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidXf3f`.
    /// This is the const half of the class.
    public class Const_RigidXf3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RigidXf3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_Destroy", ExactSpelling = true)]
            extern static void __MR_RigidXf3f_Destroy(_Underlying *_this);
            __MR_RigidXf3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RigidXf3f() {Dispose(false);}

        ///< rotation angles relative to x,y,z axes
        public unsafe MR.Const_Vector3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_RigidXf3f_Get_a(_Underlying *_this);
                return new(__MR_RigidXf3f_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public unsafe MR.Const_Vector3f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_RigidXf3f_Get_b(_Underlying *_this);
                return new(__MR_RigidXf3f_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RigidXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidXf3f::RigidXf3f`.
        public unsafe Const_RigidXf3f(MR.Const_RigidXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_ConstructFromAnother(MR.RigidXf3f._Underlying *_other);
            _UnderlyingPtr = __MR_RigidXf3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidXf3f::RigidXf3f`.
        public unsafe Const_RigidXf3f(MR.Const_Vector3f a, MR.Const_Vector3f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_Construct", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_Construct(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            _UnderlyingPtr = __MR_RigidXf3f_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// converts this into rigid transformation, which non-linearly depends on angles
        /// Generated from method `MR::RigidXf3f::rigidXf`.
        public unsafe MR.AffineXf3f RigidXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_rigidXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_RigidXf3f_rigidXf(_Underlying *_this);
            return __MR_RigidXf3f_rigidXf(_UnderlyingPtr);
        }

        /// converts this into not-rigid transformation but with matrix, which linearly depends on angles
        /// Generated from method `MR::RigidXf3f::linearXf`.
        public unsafe MR.AffineXf3f LinearXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_linearXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_RigidXf3f_linearXf(_Underlying *_this);
            return __MR_RigidXf3f_linearXf(_UnderlyingPtr);
        }
    }

    /// rigid transformation preserving all distances: y = A*x + b,
    /// where A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidXf3f`.
    /// This is the non-const half of the class.
    public class RigidXf3f : Const_RigidXf3f
    {
        internal unsafe RigidXf3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< rotation angles relative to x,y,z axes
        public new unsafe MR.Mut_Vector3f A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_RigidXf3f_GetMutable_a(_Underlying *_this);
                return new(__MR_RigidXf3f_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public new unsafe MR.Mut_Vector3f B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_RigidXf3f_GetMutable_b(_Underlying *_this);
                return new(__MR_RigidXf3f_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RigidXf3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidXf3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidXf3f::RigidXf3f`.
        public unsafe RigidXf3f(MR.Const_RigidXf3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_ConstructFromAnother(MR.RigidXf3f._Underlying *_other);
            _UnderlyingPtr = __MR_RigidXf3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidXf3f::RigidXf3f`.
        public unsafe RigidXf3f(MR.Const_Vector3f a, MR.Const_Vector3f b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_Construct", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_Construct(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b);
            _UnderlyingPtr = __MR_RigidXf3f_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::RigidXf3f::operator=`.
        public unsafe MR.RigidXf3f Assign(MR.Const_RigidXf3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RigidXf3f._Underlying *__MR_RigidXf3f_AssignFromAnother(_Underlying *_this, MR.RigidXf3f._Underlying *_other);
            return new(__MR_RigidXf3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RigidXf3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RigidXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidXf3f`/`Const_RigidXf3f` directly.
    public class _InOptMut_RigidXf3f
    {
        public RigidXf3f? Opt;

        public _InOptMut_RigidXf3f() {}
        public _InOptMut_RigidXf3f(RigidXf3f value) {Opt = value;}
        public static implicit operator _InOptMut_RigidXf3f(RigidXf3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `RigidXf3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RigidXf3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidXf3f`/`Const_RigidXf3f` to pass it to the function.
    public class _InOptConst_RigidXf3f
    {
        public Const_RigidXf3f? Opt;

        public _InOptConst_RigidXf3f() {}
        public _InOptConst_RigidXf3f(Const_RigidXf3f value) {Opt = value;}
        public static implicit operator _InOptConst_RigidXf3f(Const_RigidXf3f value) {return new(value);}
    }

    /// rigid transformation preserving all distances: y = A*x + b,
    /// where A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidXf3d`.
    /// This is the const half of the class.
    public class Const_RigidXf3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_RigidXf3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_Destroy", ExactSpelling = true)]
            extern static void __MR_RigidXf3d_Destroy(_Underlying *_this);
            __MR_RigidXf3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_RigidXf3d() {Dispose(false);}

        ///< rotation angles relative to x,y,z axes
        public unsafe MR.Const_Vector3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_Get_a", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_RigidXf3d_Get_a(_Underlying *_this);
                return new(__MR_RigidXf3d_Get_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public unsafe MR.Const_Vector3d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_Get_b", ExactSpelling = true)]
                extern static MR.Const_Vector3d._Underlying *__MR_RigidXf3d_Get_b(_Underlying *_this);
                return new(__MR_RigidXf3d_Get_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_RigidXf3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidXf3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidXf3d::RigidXf3d`.
        public unsafe Const_RigidXf3d(MR.Const_RigidXf3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_ConstructFromAnother(MR.RigidXf3d._Underlying *_other);
            _UnderlyingPtr = __MR_RigidXf3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidXf3d::RigidXf3d`.
        public unsafe Const_RigidXf3d(MR.Const_Vector3d a, MR.Const_Vector3d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_Construct", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_Construct(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            _UnderlyingPtr = __MR_RigidXf3d_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// converts this into rigid transformation, which non-linearly depends on angles
        /// Generated from method `MR::RigidXf3d::rigidXf`.
        public unsafe MR.AffineXf3d RigidXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_rigidXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_RigidXf3d_rigidXf(_Underlying *_this);
            return __MR_RigidXf3d_rigidXf(_UnderlyingPtr);
        }

        /// converts this into not-rigid transformation but with matrix, which linearly depends on angles
        /// Generated from method `MR::RigidXf3d::linearXf`.
        public unsafe MR.AffineXf3d LinearXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_linearXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_RigidXf3d_linearXf(_Underlying *_this);
            return __MR_RigidXf3d_linearXf(_UnderlyingPtr);
        }
    }

    /// rigid transformation preserving all distances: y = A*x + b,
    /// where A is rotation matrix 3x3 stored as 3 angles, and b is shift vector
    /// Generated from class `MR::RigidXf3d`.
    /// This is the non-const half of the class.
    public class RigidXf3d : Const_RigidXf3d
    {
        internal unsafe RigidXf3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< rotation angles relative to x,y,z axes
        public new unsafe MR.Mut_Vector3d A
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_GetMutable_a", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_RigidXf3d_GetMutable_a(_Underlying *_this);
                return new(__MR_RigidXf3d_GetMutable_a(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< shift
        public new unsafe MR.Mut_Vector3d B
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_GetMutable_b", ExactSpelling = true)]
                extern static MR.Mut_Vector3d._Underlying *__MR_RigidXf3d_GetMutable_b(_Underlying *_this);
                return new(__MR_RigidXf3d_GetMutable_b(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe RigidXf3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_DefaultConstruct();
            _UnderlyingPtr = __MR_RigidXf3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::RigidXf3d::RigidXf3d`.
        public unsafe RigidXf3d(MR.Const_RigidXf3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_ConstructFromAnother(MR.RigidXf3d._Underlying *_other);
            _UnderlyingPtr = __MR_RigidXf3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::RigidXf3d::RigidXf3d`.
        public unsafe RigidXf3d(MR.Const_Vector3d a, MR.Const_Vector3d b) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_Construct", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_Construct(MR.Const_Vector3d._Underlying *a, MR.Const_Vector3d._Underlying *b);
            _UnderlyingPtr = __MR_RigidXf3d_Construct(a._UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::RigidXf3d::operator=`.
        public unsafe MR.RigidXf3d Assign(MR.Const_RigidXf3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_RigidXf3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.RigidXf3d._Underlying *__MR_RigidXf3d_AssignFromAnother(_Underlying *_this, MR.RigidXf3d._Underlying *_other);
            return new(__MR_RigidXf3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `RigidXf3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_RigidXf3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidXf3d`/`Const_RigidXf3d` directly.
    public class _InOptMut_RigidXf3d
    {
        public RigidXf3d? Opt;

        public _InOptMut_RigidXf3d() {}
        public _InOptMut_RigidXf3d(RigidXf3d value) {Opt = value;}
        public static implicit operator _InOptMut_RigidXf3d(RigidXf3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `RigidXf3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_RigidXf3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `RigidXf3d`/`Const_RigidXf3d` to pass it to the function.
    public class _InOptConst_RigidXf3d
    {
        public Const_RigidXf3d? Opt;

        public _InOptConst_RigidXf3d() {}
        public _InOptConst_RigidXf3d(Const_RigidXf3d value) {Opt = value;}
        public static implicit operator _InOptConst_RigidXf3d(Const_RigidXf3d value) {return new(value);}
    }
}
