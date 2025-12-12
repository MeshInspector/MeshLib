public static partial class MR
{
    // Base class for cone parameterization
    /// Generated from class `MR::Cone3f`.
    /// This is the const half of the class.
    public class Const_Cone3f : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Cone3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Cone3f_Destroy(_Underlying *_this);
            __MR_Cone3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Cone3f() {Dispose(false);}

        // the combination of the apex of the cone and the direction of its main axis in space. 
        public unsafe MR.Const_Line3f Axis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_Get_axis", ExactSpelling = true)]
                extern static MR.Const_Line3f._Underlying *__MR_Cone3f_Get_axis(_Underlying *_this);
                return new(__MR_Cone3f_Get_axis(_UnderlyingPtr), is_owning: false);
            }
        }

        // cone angle, main axis vs side
        public unsafe float Angle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_Get_angle", ExactSpelling = true)]
                extern static float *__MR_Cone3f_Get_angle(_Underlying *_this);
                return *__MR_Cone3f_Get_angle(_UnderlyingPtr);
            }
        }

        // cone height
        public unsafe float Height
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_Get_height", ExactSpelling = true)]
                extern static float *__MR_Cone3f_Get_height(_Underlying *_this);
                return *__MR_Cone3f_Get_height(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Cone3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Cone3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cone3f::Cone3f`.
        public unsafe Const_Cone3f(MR.Const_Cone3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_ConstructFromAnother(MR.Cone3f._Underlying *_other);
            _UnderlyingPtr = __MR_Cone3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // inAxis -- apex position and main axis direction.  For any cone point dot( point , direction ) >=0
        // inAngle -- cone angle, main axis vs side
        // inHeight -- cone inHeight
        // main axis direction could be non normalized.
        /// Generated from constructor `MR::Cone3f::Cone3f`.
        public unsafe Const_Cone3f(MR.Const_Line3f inAxis, float inAngle, float inHeight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_Construct", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_Construct(MR.Const_Line3f._Underlying *inAxis, float inAngle, float inHeight);
            _UnderlyingPtr = __MR_Cone3f_Construct(inAxis._UnderlyingPtr, inAngle, inHeight);
        }

        // now we use an apex as center of the cone. 
        /// Generated from method `MR::Cone3f::center`.
        public unsafe MR.Const_Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_center_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Cone3f_center_const(_Underlying *_this);
            return new(__MR_Cone3f_center_const(_UnderlyingPtr), is_owning: false);
        }

        // main axis direction. It could be non normalized. For any cone point dot( point , direction ) >=0
        /// Generated from method `MR::Cone3f::direction`.
        public unsafe MR.Const_Vector3f Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_direction_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Cone3f_direction_const(_Underlying *_this);
            return new(__MR_Cone3f_direction_const(_UnderlyingPtr), is_owning: false);
        }

        // return cone apex position
        /// Generated from method `MR::Cone3f::apex`.
        public unsafe MR.Const_Vector3f Apex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_apex_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Cone3f_apex_const(_Underlying *_this);
            return new(__MR_Cone3f_apex_const(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cone3f::projectPoint`.
        public unsafe MR.Vector3f ProjectPoint(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_projectPoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Cone3f_projectPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return __MR_Cone3f_projectPoint(_UnderlyingPtr, point._UnderlyingPtr);
        }
    }

    // Base class for cone parameterization
    /// Generated from class `MR::Cone3f`.
    /// This is the non-const half of the class.
    public class Cone3f : Const_Cone3f
    {
        internal unsafe Cone3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // the combination of the apex of the cone and the direction of its main axis in space. 
        public new unsafe MR.Line3f Axis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_GetMutable_axis", ExactSpelling = true)]
                extern static MR.Line3f._Underlying *__MR_Cone3f_GetMutable_axis(_Underlying *_this);
                return new(__MR_Cone3f_GetMutable_axis(_UnderlyingPtr), is_owning: false);
            }
        }

        // cone angle, main axis vs side
        public new unsafe ref float Angle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_GetMutable_angle", ExactSpelling = true)]
                extern static float *__MR_Cone3f_GetMutable_angle(_Underlying *_this);
                return ref *__MR_Cone3f_GetMutable_angle(_UnderlyingPtr);
            }
        }

        // cone height
        public new unsafe ref float Height
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_GetMutable_height", ExactSpelling = true)]
                extern static float *__MR_Cone3f_GetMutable_height(_Underlying *_this);
                return ref *__MR_Cone3f_GetMutable_height(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Cone3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_DefaultConstruct();
            _UnderlyingPtr = __MR_Cone3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cone3f::Cone3f`.
        public unsafe Cone3f(MR.Const_Cone3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_ConstructFromAnother(MR.Cone3f._Underlying *_other);
            _UnderlyingPtr = __MR_Cone3f_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // inAxis -- apex position and main axis direction.  For any cone point dot( point , direction ) >=0
        // inAngle -- cone angle, main axis vs side
        // inHeight -- cone inHeight
        // main axis direction could be non normalized.
        /// Generated from constructor `MR::Cone3f::Cone3f`.
        public unsafe Cone3f(MR.Const_Line3f inAxis, float inAngle, float inHeight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_Construct", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_Construct(MR.Const_Line3f._Underlying *inAxis, float inAngle, float inHeight);
            _UnderlyingPtr = __MR_Cone3f_Construct(inAxis._UnderlyingPtr, inAngle, inHeight);
        }

        /// Generated from method `MR::Cone3f::operator=`.
        public unsafe MR.Cone3f Assign(MR.Const_Cone3f _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Cone3f._Underlying *__MR_Cone3f_AssignFromAnother(_Underlying *_this, MR.Cone3f._Underlying *_other);
            return new(__MR_Cone3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        // now we use an apex as center of the cone. 
        /// Generated from method `MR::Cone3f::center`.
        public unsafe new MR.Mut_Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_center", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Cone3f_center(_Underlying *_this);
            return new(__MR_Cone3f_center(_UnderlyingPtr), is_owning: false);
        }

        // main axis direction. It could be non normalized. For any cone point dot( point , direction ) >=0
        /// Generated from method `MR::Cone3f::direction`.
        public unsafe new MR.Mut_Vector3f Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_direction", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Cone3f_direction(_Underlying *_this);
            return new(__MR_Cone3f_direction(_UnderlyingPtr), is_owning: false);
        }

        // return cone apex position 
        /// Generated from method `MR::Cone3f::apex`.
        public unsafe new MR.Mut_Vector3f Apex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3f_apex", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Cone3f_apex(_Underlying *_this);
            return new(__MR_Cone3f_apex(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Cone3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Cone3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cone3f`/`Const_Cone3f` directly.
    public class _InOptMut_Cone3f
    {
        public Cone3f? Opt;

        public _InOptMut_Cone3f() {}
        public _InOptMut_Cone3f(Cone3f value) {Opt = value;}
        public static implicit operator _InOptMut_Cone3f(Cone3f value) {return new(value);}
    }

    /// This is used for optional parameters of class `Cone3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Cone3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cone3f`/`Const_Cone3f` to pass it to the function.
    public class _InOptConst_Cone3f
    {
        public Const_Cone3f? Opt;

        public _InOptConst_Cone3f() {}
        public _InOptConst_Cone3f(Const_Cone3f value) {Opt = value;}
        public static implicit operator _InOptConst_Cone3f(Const_Cone3f value) {return new(value);}
    }

    // Base class for cone parameterization
    /// Generated from class `MR::Cone3d`.
    /// This is the const half of the class.
    public class Const_Cone3d : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Cone3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Cone3d_Destroy(_Underlying *_this);
            __MR_Cone3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Cone3d() {Dispose(false);}

        // the combination of the apex of the cone and the direction of its main axis in space. 
        public unsafe MR.Const_Line3d Axis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_Get_axis", ExactSpelling = true)]
                extern static MR.Const_Line3d._Underlying *__MR_Cone3d_Get_axis(_Underlying *_this);
                return new(__MR_Cone3d_Get_axis(_UnderlyingPtr), is_owning: false);
            }
        }

        // cone angle, main axis vs side
        public unsafe double Angle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_Get_angle", ExactSpelling = true)]
                extern static double *__MR_Cone3d_Get_angle(_Underlying *_this);
                return *__MR_Cone3d_Get_angle(_UnderlyingPtr);
            }
        }

        // cone height
        public unsafe double Height
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_Get_height", ExactSpelling = true)]
                extern static double *__MR_Cone3d_Get_height(_Underlying *_this);
                return *__MR_Cone3d_Get_height(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Cone3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Cone3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cone3d::Cone3d`.
        public unsafe Const_Cone3d(MR.Const_Cone3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_ConstructFromAnother(MR.Cone3d._Underlying *_other);
            _UnderlyingPtr = __MR_Cone3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // inAxis -- apex position and main axis direction.  For any cone point dot( point , direction ) >=0
        // inAngle -- cone angle, main axis vs side
        // inHeight -- cone inHeight
        // main axis direction could be non normalized.
        /// Generated from constructor `MR::Cone3d::Cone3d`.
        public unsafe Const_Cone3d(MR.Const_Line3d inAxis, double inAngle, double inHeight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_Construct", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_Construct(MR.Const_Line3d._Underlying *inAxis, double inAngle, double inHeight);
            _UnderlyingPtr = __MR_Cone3d_Construct(inAxis._UnderlyingPtr, inAngle, inHeight);
        }

        // now we use an apex as center of the cone. 
        /// Generated from method `MR::Cone3d::center`.
        public unsafe MR.Const_Vector3d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_center_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Cone3d_center_const(_Underlying *_this);
            return new(__MR_Cone3d_center_const(_UnderlyingPtr), is_owning: false);
        }

        // main axis direction. It could be non normalized. For any cone point dot( point , direction ) >=0
        /// Generated from method `MR::Cone3d::direction`.
        public unsafe MR.Const_Vector3d Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_direction_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Cone3d_direction_const(_Underlying *_this);
            return new(__MR_Cone3d_direction_const(_UnderlyingPtr), is_owning: false);
        }

        // return cone apex position
        /// Generated from method `MR::Cone3d::apex`.
        public unsafe MR.Const_Vector3d Apex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_apex_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Cone3d_apex_const(_Underlying *_this);
            return new(__MR_Cone3d_apex_const(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Cone3d::projectPoint`.
        public unsafe MR.Vector3d ProjectPoint(MR.Const_Vector3d point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_projectPoint", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Cone3d_projectPoint(_Underlying *_this, MR.Const_Vector3d._Underlying *point);
            return __MR_Cone3d_projectPoint(_UnderlyingPtr, point._UnderlyingPtr);
        }
    }

    // Base class for cone parameterization
    /// Generated from class `MR::Cone3d`.
    /// This is the non-const half of the class.
    public class Cone3d : Const_Cone3d
    {
        internal unsafe Cone3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // the combination of the apex of the cone and the direction of its main axis in space. 
        public new unsafe MR.Line3d Axis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_GetMutable_axis", ExactSpelling = true)]
                extern static MR.Line3d._Underlying *__MR_Cone3d_GetMutable_axis(_Underlying *_this);
                return new(__MR_Cone3d_GetMutable_axis(_UnderlyingPtr), is_owning: false);
            }
        }

        // cone angle, main axis vs side
        public new unsafe ref double Angle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_GetMutable_angle", ExactSpelling = true)]
                extern static double *__MR_Cone3d_GetMutable_angle(_Underlying *_this);
                return ref *__MR_Cone3d_GetMutable_angle(_UnderlyingPtr);
            }
        }

        // cone height
        public new unsafe ref double Height
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_GetMutable_height", ExactSpelling = true)]
                extern static double *__MR_Cone3d_GetMutable_height(_Underlying *_this);
                return ref *__MR_Cone3d_GetMutable_height(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Cone3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_DefaultConstruct();
            _UnderlyingPtr = __MR_Cone3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Cone3d::Cone3d`.
        public unsafe Cone3d(MR.Const_Cone3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_ConstructFromAnother(MR.Cone3d._Underlying *_other);
            _UnderlyingPtr = __MR_Cone3d_ConstructFromAnother(_other._UnderlyingPtr);
        }

        // inAxis -- apex position and main axis direction.  For any cone point dot( point , direction ) >=0
        // inAngle -- cone angle, main axis vs side
        // inHeight -- cone inHeight
        // main axis direction could be non normalized.
        /// Generated from constructor `MR::Cone3d::Cone3d`.
        public unsafe Cone3d(MR.Const_Line3d inAxis, double inAngle, double inHeight) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_Construct", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_Construct(MR.Const_Line3d._Underlying *inAxis, double inAngle, double inHeight);
            _UnderlyingPtr = __MR_Cone3d_Construct(inAxis._UnderlyingPtr, inAngle, inHeight);
        }

        /// Generated from method `MR::Cone3d::operator=`.
        public unsafe MR.Cone3d Assign(MR.Const_Cone3d _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Cone3d._Underlying *__MR_Cone3d_AssignFromAnother(_Underlying *_this, MR.Cone3d._Underlying *_other);
            return new(__MR_Cone3d_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        // now we use an apex as center of the cone. 
        /// Generated from method `MR::Cone3d::center`.
        public unsafe new MR.Mut_Vector3d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_center", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Cone3d_center(_Underlying *_this);
            return new(__MR_Cone3d_center(_UnderlyingPtr), is_owning: false);
        }

        // main axis direction. It could be non normalized. For any cone point dot( point , direction ) >=0
        /// Generated from method `MR::Cone3d::direction`.
        public unsafe new MR.Mut_Vector3d Direction()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_direction", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Cone3d_direction(_Underlying *_this);
            return new(__MR_Cone3d_direction(_UnderlyingPtr), is_owning: false);
        }

        // return cone apex position 
        /// Generated from method `MR::Cone3d::apex`.
        public unsafe new MR.Mut_Vector3d Apex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Cone3d_apex", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Cone3d_apex(_Underlying *_this);
            return new(__MR_Cone3d_apex(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Cone3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Cone3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cone3d`/`Const_Cone3d` directly.
    public class _InOptMut_Cone3d
    {
        public Cone3d? Opt;

        public _InOptMut_Cone3d() {}
        public _InOptMut_Cone3d(Cone3d value) {Opt = value;}
        public static implicit operator _InOptMut_Cone3d(Cone3d value) {return new(value);}
    }

    /// This is used for optional parameters of class `Cone3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Cone3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Cone3d`/`Const_Cone3d` to pass it to the function.
    public class _InOptConst_Cone3d
    {
        public Const_Cone3d? Opt;

        public _InOptConst_Cone3d() {}
        public _InOptConst_Cone3d(Const_Cone3d value) {Opt = value;}
        public static implicit operator _InOptConst_Cone3d(Const_Cone3d value) {return new(value);}
    }
}
