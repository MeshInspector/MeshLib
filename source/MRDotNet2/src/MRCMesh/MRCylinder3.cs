public static partial class MR
{
    // A class describing a cylinder as a mathematical object.A cylinder is represented by a centerline, a radius, and a length.template <typename T>
    // TODO: Cylinder3 could be infinite. For example for infinite Cylinder3 we could use negative length or length = -1
    /// Generated from class `MR::Cylinder3f`.
    /// This is the const half of the class.
    public class Const_Cylinder3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Cylinder3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Cylinder3f_Destroy(_Underlying *_this);
            __MR_Cylinder3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Cylinder3f() {Dispose(false);}

        public unsafe MR.Const_Line3f MainAxis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Get_mainAxis", ExactSpelling = true)]
                extern static MR.Const_Line3f._Underlying *__MR_Cylinder3f_Get_mainAxis(_Underlying *_this);
                return new(__MR_Cylinder3f_Get_mainAxis(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Get_radius", ExactSpelling = true)]
                extern static float *__MR_Cylinder3f_Get_radius(_Underlying *_this);
                return *__MR_Cylinder3f_Get_radius(_UnderlyingPtr);
            }
        }

        public unsafe float Length
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Get_length", ExactSpelling = true)]
                extern static float *__MR_Cylinder3f_Get_length(_Underlying *_this);
                return *__MR_Cylinder3f_Get_length(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Cylinder3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Cylinder3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cylinder3f::Cylinder3f`.
        public unsafe Const_Cylinder3f(MR.Const_Cylinder3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_ConstructFromAnother(MR.Cylinder3f._Underlying *_other);
            _UnderlyingPtr = __MR_Cylinder3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Cylinder3f::Cylinder3f`.
        public unsafe Const_Cylinder3f(MR.Const_Vector3f inCenter, MR.Const_Vector3f inDirectoin, float inRadius, float inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Construct_4", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_Construct_4(MR.Const_Vector3f._Underlying *inCenter, MR.Const_Vector3f._Underlying *inDirectoin, float inRadius, float inLength);
            _UnderlyingPtr = __MR_Cylinder3f_Construct_4(inCenter._UnderlyingPtr, inDirectoin._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from constructor `MR::Cylinder3f::Cylinder3f`.
        public unsafe Const_Cylinder3f(MR.Const_Line3f inAxis, float inRadius, float inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Construct_3", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_Construct_3(MR.Const_Line3f._Underlying *inAxis, float inRadius, float inLength);
            _UnderlyingPtr = __MR_Cylinder3f_Construct_3(inAxis._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from method `MR::Cylinder3f::center`.
        public unsafe MR.Const_Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_center_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Cylinder3f_center_const(_Underlying *_this);
            return new(__MR_Cylinder3f_center_const(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cylinder3f::direction`.
        public unsafe MR.Const_Vector3f Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_direction_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Cylinder3f_direction_const(_Underlying *_this);
            return new(__MR_Cylinder3f_direction_const(_UnderlyingPtr), is_owning: false);
        }
    }

    // A class describing a cylinder as a mathematical object.A cylinder is represented by a centerline, a radius, and a length.template <typename T>
    // TODO: Cylinder3 could be infinite. For example for infinite Cylinder3 we could use negative length or length = -1
    /// Generated from class `MR::Cylinder3f`.
    /// This is the non-const half of the class.
    public class Cylinder3f : Const_Cylinder3f
    {
        internal unsafe Cylinder3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Line3f MainAxis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_GetMutable_mainAxis", ExactSpelling = true)]
                extern static MR.Line3f._Underlying *__MR_Cylinder3f_GetMutable_mainAxis(_Underlying *_this);
                return new(__MR_Cylinder3f_GetMutable_mainAxis(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_Cylinder3f_GetMutable_radius(_Underlying *_this);
                return ref *__MR_Cylinder3f_GetMutable_radius(_UnderlyingPtr);
            }
        }

        public new unsafe ref float Length
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_GetMutable_length", ExactSpelling = true)]
                extern static float *__MR_Cylinder3f_GetMutable_length(_Underlying *_this);
                return ref *__MR_Cylinder3f_GetMutable_length(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Cylinder3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Cylinder3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cylinder3f::Cylinder3f`.
        public unsafe Cylinder3f(MR.Const_Cylinder3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_ConstructFromAnother(MR.Cylinder3f._Underlying *_other);
            _UnderlyingPtr = __MR_Cylinder3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Cylinder3f::Cylinder3f`.
        public unsafe Cylinder3f(MR.Const_Vector3f inCenter, MR.Const_Vector3f inDirectoin, float inRadius, float inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Construct_4", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_Construct_4(MR.Const_Vector3f._Underlying *inCenter, MR.Const_Vector3f._Underlying *inDirectoin, float inRadius, float inLength);
            _UnderlyingPtr = __MR_Cylinder3f_Construct_4(inCenter._UnderlyingPtr, inDirectoin._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from constructor `MR::Cylinder3f::Cylinder3f`.
        public unsafe Cylinder3f(MR.Const_Line3f inAxis, float inRadius, float inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_Construct_3", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_Construct_3(MR.Const_Line3f._Underlying *inAxis, float inRadius, float inLength);
            _UnderlyingPtr = __MR_Cylinder3f_Construct_3(inAxis._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from method `MR::Cylinder3f::operator=`.
        public unsafe MR.Cylinder3f Assign(MR.Const_Cylinder3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Cylinder3f._Underlying *__MR_Cylinder3f_AssignFromAnother(_Underlying *_this, MR.Cylinder3f._Underlying *_other);
            return new(__MR_Cylinder3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cylinder3f::center`.
        public unsafe new MR.Mut_Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_center", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Cylinder3f_center(_Underlying *_this);
            return new(__MR_Cylinder3f_center(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cylinder3f::direction`.
        public unsafe new MR.Mut_Vector3f Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3f_direction", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Cylinder3f_direction(_Underlying *_this);
            return new(__MR_Cylinder3f_direction(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Cylinder3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Cylinder3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cylinder3f`/`Const_Cylinder3f` directly.
    public class _InOptMut_Cylinder3f
    {
        public Cylinder3f? Opt;

        public _InOptMut_Cylinder3f() {}
        public _InOptMut_Cylinder3f(Cylinder3f value) {Opt = value;}
        public static implicit operator _InOptMut_Cylinder3f(Cylinder3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Cylinder3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Cylinder3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cylinder3f`/`Const_Cylinder3f` to pass it to the function.
    public class _InOptConst_Cylinder3f
    {
        public Const_Cylinder3f? Opt;

        public _InOptConst_Cylinder3f() {}
        public _InOptConst_Cylinder3f(Const_Cylinder3f value) {Opt = value;}
        public static implicit operator _InOptConst_Cylinder3f(Const_Cylinder3f value) {return new(value);}
    }

    // A class describing a cylinder as a mathematical object.A cylinder is represented by a centerline, a radius, and a length.template <typename T>
    // TODO: Cylinder3 could be infinite. For example for infinite Cylinder3 we could use negative length or length = -1
    /// Generated from class `MR::Cylinder3d`.
    /// This is the const half of the class.
    public class Const_Cylinder3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Cylinder3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Cylinder3d_Destroy(_Underlying *_this);
            __MR_Cylinder3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Cylinder3d() {Dispose(false);}

        public unsafe MR.Const_Line3d MainAxis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Get_mainAxis", ExactSpelling = true)]
                extern static MR.Const_Line3d._Underlying *__MR_Cylinder3d_Get_mainAxis(_Underlying *_this);
                return new(__MR_Cylinder3d_Get_mainAxis(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe double Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Get_radius", ExactSpelling = true)]
                extern static double *__MR_Cylinder3d_Get_radius(_Underlying *_this);
                return *__MR_Cylinder3d_Get_radius(_UnderlyingPtr);
            }
        }

        public unsafe double Length
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Get_length", ExactSpelling = true)]
                extern static double *__MR_Cylinder3d_Get_length(_Underlying *_this);
                return *__MR_Cylinder3d_Get_length(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Cylinder3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Cylinder3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cylinder3d::Cylinder3d`.
        public unsafe Const_Cylinder3d(MR.Const_Cylinder3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_ConstructFromAnother(MR.Cylinder3d._Underlying *_other);
            _UnderlyingPtr = __MR_Cylinder3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Cylinder3d::Cylinder3d`.
        public unsafe Const_Cylinder3d(MR.Const_Vector3d inCenter, MR.Const_Vector3d inDirectoin, double inRadius, double inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Construct_4", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_Construct_4(MR.Const_Vector3d._Underlying *inCenter, MR.Const_Vector3d._Underlying *inDirectoin, double inRadius, double inLength);
            _UnderlyingPtr = __MR_Cylinder3d_Construct_4(inCenter._UnderlyingPtr, inDirectoin._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from constructor `MR::Cylinder3d::Cylinder3d`.
        public unsafe Const_Cylinder3d(MR.Const_Line3d inAxis, double inRadius, double inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Construct_3", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_Construct_3(MR.Const_Line3d._Underlying *inAxis, double inRadius, double inLength);
            _UnderlyingPtr = __MR_Cylinder3d_Construct_3(inAxis._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from method `MR::Cylinder3d::center`.
        public unsafe MR.Const_Vector3d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_center_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Cylinder3d_center_const(_Underlying *_this);
            return new(__MR_Cylinder3d_center_const(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cylinder3d::direction`.
        public unsafe MR.Const_Vector3d Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_direction_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Cylinder3d_direction_const(_Underlying *_this);
            return new(__MR_Cylinder3d_direction_const(_UnderlyingPtr), is_owning: false);
        }
    }

    // A class describing a cylinder as a mathematical object.A cylinder is represented by a centerline, a radius, and a length.template <typename T>
    // TODO: Cylinder3 could be infinite. For example for infinite Cylinder3 we could use negative length or length = -1
    /// Generated from class `MR::Cylinder3d`.
    /// This is the non-const half of the class.
    public class Cylinder3d : Const_Cylinder3d
    {
        internal unsafe Cylinder3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Line3d MainAxis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_GetMutable_mainAxis", ExactSpelling = true)]
                extern static MR.Line3d._Underlying *__MR_Cylinder3d_GetMutable_mainAxis(_Underlying *_this);
                return new(__MR_Cylinder3d_GetMutable_mainAxis(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref double Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_GetMutable_radius", ExactSpelling = true)]
                extern static double *__MR_Cylinder3d_GetMutable_radius(_Underlying *_this);
                return ref *__MR_Cylinder3d_GetMutable_radius(_UnderlyingPtr);
            }
        }

        public new unsafe ref double Length
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_GetMutable_length", ExactSpelling = true)]
                extern static double *__MR_Cylinder3d_GetMutable_length(_Underlying *_this);
                return ref *__MR_Cylinder3d_GetMutable_length(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Cylinder3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Cylinder3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cylinder3d::Cylinder3d`.
        public unsafe Cylinder3d(MR.Const_Cylinder3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_ConstructFromAnother(MR.Cylinder3d._Underlying *_other);
            _UnderlyingPtr = __MR_Cylinder3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Cylinder3d::Cylinder3d`.
        public unsafe Cylinder3d(MR.Const_Vector3d inCenter, MR.Const_Vector3d inDirectoin, double inRadius, double inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Construct_4", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_Construct_4(MR.Const_Vector3d._Underlying *inCenter, MR.Const_Vector3d._Underlying *inDirectoin, double inRadius, double inLength);
            _UnderlyingPtr = __MR_Cylinder3d_Construct_4(inCenter._UnderlyingPtr, inDirectoin._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from constructor `MR::Cylinder3d::Cylinder3d`.
        public unsafe Cylinder3d(MR.Const_Line3d inAxis, double inRadius, double inLength) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_Construct_3", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_Construct_3(MR.Const_Line3d._Underlying *inAxis, double inRadius, double inLength);
            _UnderlyingPtr = __MR_Cylinder3d_Construct_3(inAxis._UnderlyingPtr, inRadius, inLength);
        }

        /// Generated from method `MR::Cylinder3d::operator=`.
        public unsafe MR.Cylinder3d Assign(MR.Const_Cylinder3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Cylinder3d._Underlying *__MR_Cylinder3d_AssignFromAnother(_Underlying *_this, MR.Cylinder3d._Underlying *_other);
            return new(__MR_Cylinder3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cylinder3d::center`.
        public unsafe new MR.Mut_Vector3d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_center", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Cylinder3d_center(_Underlying *_this);
            return new(__MR_Cylinder3d_center(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cylinder3d::direction`.
        public unsafe new MR.Mut_Vector3d Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cylinder3d_direction", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Cylinder3d_direction(_Underlying *_this);
            return new(__MR_Cylinder3d_direction(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Cylinder3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Cylinder3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cylinder3d`/`Const_Cylinder3d` directly.
    public class _InOptMut_Cylinder3d
    {
        public Cylinder3d? Opt;

        public _InOptMut_Cylinder3d() {}
        public _InOptMut_Cylinder3d(Cylinder3d value) {Opt = value;}
        public static implicit operator _InOptMut_Cylinder3d(Cylinder3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Cylinder3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Cylinder3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cylinder3d`/`Const_Cylinder3d` to pass it to the function.
    public class _InOptConst_Cylinder3d
    {
        public Const_Cylinder3d? Opt;

        public _InOptConst_Cylinder3d() {}
        public _InOptConst_Cylinder3d(Const_Cylinder3d value) {Opt = value;}
        public static implicit operator _InOptConst_Cylinder3d(Const_Cylinder3d value) {return new(value);}
    }
}
