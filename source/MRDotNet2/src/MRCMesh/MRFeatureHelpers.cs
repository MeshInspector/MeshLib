public static partial class MR
{
    //! Which object type we're adding.
    //! Update `ObjKindTraits` if you change this enum.
    public enum FeaturesObjectKind : int
    {
        Point = 0,
        Line = 1,
        Plane = 2,
        Circle = 3,
        Sphere = 4,
        Cylinder = 5,
        Cone = 6,
        Count = 7,
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Point>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindPoint : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindPoint(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Point_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Point_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindPoint() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindPoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Point_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Point>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindPoint(MR.Const_ObjKindTraits_MRFeaturesObjectKindPoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Point_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Point>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindPoint : Const_ObjKindTraits_MRFeaturesObjectKindPoint
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindPoint(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindPoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Point_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Point>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindPoint(MR.Const_ObjKindTraits_MRFeaturesObjectKindPoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Point_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Point>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindPoint Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindPoint _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Point_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindPoint._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Point_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindPoint` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindPoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindPoint`/`Const_ObjKindTraits_MRFeaturesObjectKindPoint` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindPoint
    {
        public ObjKindTraits_MRFeaturesObjectKindPoint? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindPoint() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindPoint(ObjKindTraits_MRFeaturesObjectKindPoint value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindPoint(ObjKindTraits_MRFeaturesObjectKindPoint value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindPoint` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindPoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindPoint`/`Const_ObjKindTraits_MRFeaturesObjectKindPoint` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindPoint
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindPoint? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindPoint() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindPoint(Const_ObjKindTraits_MRFeaturesObjectKindPoint value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindPoint(Const_ObjKindTraits_MRFeaturesObjectKindPoint value) {return new(value);}
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Line>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindLine : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindLine(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Line_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Line_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindLine() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindLine() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Line_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Line>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindLine(MR.Const_ObjKindTraits_MRFeaturesObjectKindLine _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Line_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Line>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindLine : Const_ObjKindTraits_MRFeaturesObjectKindLine
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindLine(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindLine() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Line_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Line>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindLine(MR.Const_ObjKindTraits_MRFeaturesObjectKindLine _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Line_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Line>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindLine Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindLine _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Line_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindLine._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Line_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindLine` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindLine`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindLine`/`Const_ObjKindTraits_MRFeaturesObjectKindLine` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindLine
    {
        public ObjKindTraits_MRFeaturesObjectKindLine? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindLine() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindLine(ObjKindTraits_MRFeaturesObjectKindLine value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindLine(ObjKindTraits_MRFeaturesObjectKindLine value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindLine` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindLine`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindLine`/`Const_ObjKindTraits_MRFeaturesObjectKindLine` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindLine
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindLine? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindLine() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindLine(Const_ObjKindTraits_MRFeaturesObjectKindLine value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindLine(Const_ObjKindTraits_MRFeaturesObjectKindLine value) {return new(value);}
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Plane>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindPlane : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindPlane(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindPlane() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindPlane() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Plane>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindPlane(MR.Const_ObjKindTraits_MRFeaturesObjectKindPlane _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Plane>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindPlane : Const_ObjKindTraits_MRFeaturesObjectKindPlane
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindPlane(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindPlane() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Plane>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindPlane(MR.Const_ObjKindTraits_MRFeaturesObjectKindPlane _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Plane>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindPlane Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindPlane _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindPlane._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Plane_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindPlane` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindPlane`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindPlane`/`Const_ObjKindTraits_MRFeaturesObjectKindPlane` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindPlane
    {
        public ObjKindTraits_MRFeaturesObjectKindPlane? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindPlane() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindPlane(ObjKindTraits_MRFeaturesObjectKindPlane value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindPlane(ObjKindTraits_MRFeaturesObjectKindPlane value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindPlane` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindPlane`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindPlane`/`Const_ObjKindTraits_MRFeaturesObjectKindPlane` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindPlane
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindPlane? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindPlane() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindPlane(Const_ObjKindTraits_MRFeaturesObjectKindPlane value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindPlane(Const_ObjKindTraits_MRFeaturesObjectKindPlane value) {return new(value);}
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Circle>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindCircle : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindCircle(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindCircle() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindCircle() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Circle>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindCircle(MR.Const_ObjKindTraits_MRFeaturesObjectKindCircle _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Circle>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindCircle : Const_ObjKindTraits_MRFeaturesObjectKindCircle
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindCircle(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindCircle() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Circle>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindCircle(MR.Const_ObjKindTraits_MRFeaturesObjectKindCircle _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Circle>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindCircle Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindCircle _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindCircle._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Circle_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindCircle` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindCircle`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindCircle`/`Const_ObjKindTraits_MRFeaturesObjectKindCircle` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindCircle
    {
        public ObjKindTraits_MRFeaturesObjectKindCircle? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindCircle() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindCircle(ObjKindTraits_MRFeaturesObjectKindCircle value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindCircle(ObjKindTraits_MRFeaturesObjectKindCircle value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindCircle` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindCircle`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindCircle`/`Const_ObjKindTraits_MRFeaturesObjectKindCircle` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindCircle
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindCircle? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindCircle() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindCircle(Const_ObjKindTraits_MRFeaturesObjectKindCircle value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindCircle(Const_ObjKindTraits_MRFeaturesObjectKindCircle value) {return new(value);}
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Sphere>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindSphere : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindSphere(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindSphere() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindSphere() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Sphere>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindSphere(MR.Const_ObjKindTraits_MRFeaturesObjectKindSphere _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Sphere>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindSphere : Const_ObjKindTraits_MRFeaturesObjectKindSphere
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindSphere(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindSphere() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Sphere>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindSphere(MR.Const_ObjKindTraits_MRFeaturesObjectKindSphere _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Sphere>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindSphere Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindSphere _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindSphere._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Sphere_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindSphere` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindSphere`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindSphere`/`Const_ObjKindTraits_MRFeaturesObjectKindSphere` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindSphere
    {
        public ObjKindTraits_MRFeaturesObjectKindSphere? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindSphere() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindSphere(ObjKindTraits_MRFeaturesObjectKindSphere value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindSphere(ObjKindTraits_MRFeaturesObjectKindSphere value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindSphere` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindSphere`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindSphere`/`Const_ObjKindTraits_MRFeaturesObjectKindSphere` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindSphere
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindSphere? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindSphere() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindSphere(Const_ObjKindTraits_MRFeaturesObjectKindSphere value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindSphere(Const_ObjKindTraits_MRFeaturesObjectKindSphere value) {return new(value);}
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Cylinder>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindCylinder : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindCylinder(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindCylinder() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindCylinder() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Cylinder>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindCylinder(MR.Const_ObjKindTraits_MRFeaturesObjectKindCylinder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Cylinder>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindCylinder : Const_ObjKindTraits_MRFeaturesObjectKindCylinder
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindCylinder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindCylinder() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Cylinder>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindCylinder(MR.Const_ObjKindTraits_MRFeaturesObjectKindCylinder _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Cylinder>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindCylinder Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindCylinder _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindCylinder._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Cylinder_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindCylinder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindCylinder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindCylinder`/`Const_ObjKindTraits_MRFeaturesObjectKindCylinder` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindCylinder
    {
        public ObjKindTraits_MRFeaturesObjectKindCylinder? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindCylinder() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindCylinder(ObjKindTraits_MRFeaturesObjectKindCylinder value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindCylinder(ObjKindTraits_MRFeaturesObjectKindCylinder value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindCylinder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindCylinder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindCylinder`/`Const_ObjKindTraits_MRFeaturesObjectKindCylinder` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindCylinder
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindCylinder? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindCylinder() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindCylinder(Const_ObjKindTraits_MRFeaturesObjectKindCylinder value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindCylinder(Const_ObjKindTraits_MRFeaturesObjectKindCylinder value) {return new(value);}
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Cone>`.
    /// This is the const half of the class.
    public class Const_ObjKindTraits_MRFeaturesObjectKindCone : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ObjKindTraits_MRFeaturesObjectKindCone(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_Destroy", ExactSpelling = true)]
            extern static void __MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_Destroy(_Underlying *_this);
            __MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjKindTraits_MRFeaturesObjectKindCone() {Dispose(false);}

        public static unsafe MR.Std.Const_StringView Name
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_Get_name", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_Get_name();
                return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_Get_name(), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindCone() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Cone>::ObjKindTraits`.
        public unsafe Const_ObjKindTraits_MRFeaturesObjectKindCone(MR.Const_ObjKindTraits_MRFeaturesObjectKindCone _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ObjKindTraits<MR::FeaturesObjectKind::Cone>`.
    /// This is the non-const half of the class.
    public class ObjKindTraits_MRFeaturesObjectKindCone : Const_ObjKindTraits_MRFeaturesObjectKindCone
    {
        internal unsafe ObjKindTraits_MRFeaturesObjectKindCone(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjKindTraits_MRFeaturesObjectKindCone() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_DefaultConstruct();
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_DefaultConstruct();
        }

        /// Generated from constructor `MR::ObjKindTraits<MR::FeaturesObjectKind::Cone>::ObjKindTraits`.
        public unsafe ObjKindTraits_MRFeaturesObjectKindCone(MR.Const_ObjKindTraits_MRFeaturesObjectKindCone _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_ConstructFromAnother(MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *_other);
            _UnderlyingPtr = __MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjKindTraits<MR::FeaturesObjectKind::Cone>::operator=`.
        public unsafe MR.ObjKindTraits_MRFeaturesObjectKindCone Assign(MR.Const_ObjKindTraits_MRFeaturesObjectKindCone _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_AssignFromAnother(_Underlying *_this, MR.ObjKindTraits_MRFeaturesObjectKindCone._Underlying *_other);
            return new(__MR_ObjKindTraits_MR_FeaturesObjectKind_Cone_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindCone` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjKindTraits_MRFeaturesObjectKindCone`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindCone`/`Const_ObjKindTraits_MRFeaturesObjectKindCone` directly.
    public class _InOptMut_ObjKindTraits_MRFeaturesObjectKindCone
    {
        public ObjKindTraits_MRFeaturesObjectKindCone? Opt;

        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindCone() {}
        public _InOptMut_ObjKindTraits_MRFeaturesObjectKindCone(ObjKindTraits_MRFeaturesObjectKindCone value) {Opt = value;}
        public static implicit operator _InOptMut_ObjKindTraits_MRFeaturesObjectKindCone(ObjKindTraits_MRFeaturesObjectKindCone value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjKindTraits_MRFeaturesObjectKindCone` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjKindTraits_MRFeaturesObjectKindCone`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjKindTraits_MRFeaturesObjectKindCone`/`Const_ObjKindTraits_MRFeaturesObjectKindCone` to pass it to the function.
    public class _InOptConst_ObjKindTraits_MRFeaturesObjectKindCone
    {
        public Const_ObjKindTraits_MRFeaturesObjectKindCone? Opt;

        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindCone() {}
        public _InOptConst_ObjKindTraits_MRFeaturesObjectKindCone(Const_ObjKindTraits_MRFeaturesObjectKindCone value) {Opt = value;}
        public static implicit operator _InOptConst_ObjKindTraits_MRFeaturesObjectKindCone(Const_ObjKindTraits_MRFeaturesObjectKindCone value) {return new(value);}
    }

    //! Allocates an object of type `kind`, passing `params...` to its constructor.
    /// Generated from function `MR::makeObjectFromEnum<>`.
    public static unsafe MR.Misc._Moved<MR.VisualObject> MakeObjectFromEnum(MR.FeaturesObjectKind kind)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectFromEnum", ExactSpelling = true)]
        extern static MR.VisualObject._UnderlyingShared *__MR_makeObjectFromEnum(MR.FeaturesObjectKind kind);
        return MR.Misc.Move(new MR.VisualObject(__MR_makeObjectFromEnum(kind), is_owning: true));
    }

    //! Allocates an object of type `kind`, passing `params...` to its constructor.
    /// Generated from function `MR::makeObjectFromClassName<>`.
    public static unsafe MR.Misc._Moved<MR.VisualObject> MakeObjectFromClassName(ReadOnlySpan<char> className)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeObjectFromClassName", ExactSpelling = true)]
        extern static MR.VisualObject._UnderlyingShared *__MR_makeObjectFromClassName(byte *className, byte *className_end);
        byte[] __bytes_className = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(className.Length)];
        int __len_className = System.Text.Encoding.UTF8.GetBytes(className, __bytes_className);
        fixed (byte *__ptr_className = __bytes_className)
        {
            return MR.Misc.Move(new MR.VisualObject(__MR_makeObjectFromClassName(__ptr_className, __ptr_className + __len_className), is_owning: true));
        }
    }

    // Using forEachObjectKind the template collects a list of features for which the method ...->getNormal() is available
    /// Generated from function `MR::getFeatureNormal`.
    public static unsafe MR.Std.Optional_MRVector3f GetFeatureNormal(MR.FeatureObject? feature)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getFeatureNormal", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVector3f._Underlying *__MR_getFeatureNormal(MR.FeatureObject._Underlying *feature);
        return new(__MR_getFeatureNormal(feature is not null ? feature._UnderlyingPtr : null), is_owning: true);
    }

    // Using forEachObjectKind the template collects a list of features for which the method ...->getDirection() is available
    /// Generated from function `MR::getFeatureDirection`.
    public static unsafe MR.Std.Optional_MRVector3f GetFeatureDirection(MR.FeatureObject? feature)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getFeatureDirection", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVector3f._Underlying *__MR_getFeatureDirection(MR.FeatureObject._Underlying *feature);
        return new(__MR_getFeatureDirection(feature is not null ? feature._UnderlyingPtr : null), is_owning: true);
    }

    // Try to getNormal from specific feature using forEachObjectKind template. Returns nullopt is ...->getNormal() is not available for given feature type.
    /// Generated from function `MR::getFeaturesTypeWithNormals`.
    public static unsafe MR.Misc._Moved<MR.Std.UnorderedSet_StdString> GetFeaturesTypeWithNormals()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getFeaturesTypeWithNormals", ExactSpelling = true)]
        extern static MR.Std.UnorderedSet_StdString._Underlying *__MR_getFeaturesTypeWithNormals();
        return MR.Misc.Move(new MR.Std.UnorderedSet_StdString(__MR_getFeaturesTypeWithNormals(), is_owning: true));
    }

    // Try to getDirection from specific feature using forEachObjectKind template. Returns nullopt is ...->getDirection() is not available for given feature type.
    /// Generated from function `MR::getFeaturesTypeWithDirections`.
    public static unsafe MR.Misc._Moved<MR.Std.UnorderedSet_StdString> GetFeaturesTypeWithDirections()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getFeaturesTypeWithDirections", ExactSpelling = true)]
        extern static MR.Std.UnorderedSet_StdString._Underlying *__MR_getFeaturesTypeWithDirections();
        return MR.Misc.Move(new MR.Std.UnorderedSet_StdString(__MR_getFeaturesTypeWithDirections(), is_owning: true));
    }
}
