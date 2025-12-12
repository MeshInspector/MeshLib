public static partial class MR
{
    // Classifies `FeatureObjectSharedProperty`, mostly for informational purposes.
    public enum FeaturePropertyKind : int
    {
        // Position, normally Vector3f.
        Position = 0,
        // Length or size.
        LinearDimension = 1,
        // Direction, normally Vector3f.
        Direction = 2,
        // Angle, normally float. Measure in radians.
        Angle = 3,
        Other = 4,
    }

    // FeatureObjectSharedProperty struct is designed to represent a shared property of a feature object, enabling the use of generalized getter and setter methods for property manipulation.
    // propertyName: A string representing the name of the property.
    // getter : A std::function encapsulating a method with no parameters that returns a FeaturesPropertyTypesVariant.This allows for a generic way to retrieve the value of the property.
    // setter : A std::function encapsulating a method that takes a FeaturesPropertyTypesVariant as a parameter and returns void.This function sets the value of the property.
    // The templated constructor of this struct takes the property name, pointers to the getter and setter member functions, and a pointer to the object( obj ).
    // The constructor initializes the propertyName and uses lambdas to adapt the member function pointers into std::function objects that conform to the expected
    // getter and setter signatures.The getter lambda invokes the getter method on the object, and the setter lambda ensures the correct variant type is passed before
    // invoking the setter method.
    /// Generated from class `MR::FeatureObjectSharedProperty`.
    /// This is the const half of the class.
    public class Const_FeatureObjectSharedProperty : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FeatureObjectSharedProperty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_Destroy", ExactSpelling = true)]
            extern static void __MR_FeatureObjectSharedProperty_Destroy(_Underlying *_this);
            __MR_FeatureObjectSharedProperty_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FeatureObjectSharedProperty() {Dispose(false);}

        public unsafe MR.Std.Const_String PropertyName
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_Get_propertyName", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_FeatureObjectSharedProperty_Get_propertyName(_Underlying *_this);
                return new(__MR_FeatureObjectSharedProperty_Get_propertyName(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.FeaturePropertyKind Kind
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_Get_kind", ExactSpelling = true)]
                extern static MR.FeaturePropertyKind *__MR_FeatureObjectSharedProperty_Get_kind(_Underlying *_this);
                return *__MR_FeatureObjectSharedProperty_Get_kind(_UnderlyingPtr);
            }
        }

        // due to getAllSharedProperties in FeatureObject returns static vector, we need externaly setup object to invoke setter ad getter.
        public unsafe MR.Std.Const_Function_StdVariantFloatMRVector3fFuncFromConstMRFeatureObjectPtrMRViewportId Getter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_Get_getter", ExactSpelling = true)]
                extern static MR.Std.Const_Function_StdVariantFloatMRVector3fFuncFromConstMRFeatureObjectPtrMRViewportId._Underlying *__MR_FeatureObjectSharedProperty_Get_getter(_Underlying *_this);
                return new(__MR_FeatureObjectSharedProperty_Get_getter(_UnderlyingPtr), is_owning: false);
            }
        }

        // NOTE: `id` should usually be `{}`, not the current viewport ID, to set the property for all viewports.
        // Passing a non-zero ID would only modify the active viewport, and per-viewport properties aren't usually used.
        public unsafe MR.Std.Const_Function_VoidFuncFromConstStdVariantFloatMRVector3fRefMRFeatureObjectPtrMRViewportId Setter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_Get_setter", ExactSpelling = true)]
                extern static MR.Std.Const_Function_VoidFuncFromConstStdVariantFloatMRVector3fRefMRFeatureObjectPtrMRViewportId._Underlying *__MR_FeatureObjectSharedProperty_Get_setter(_Underlying *_this);
                return new(__MR_FeatureObjectSharedProperty_Get_setter(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::FeatureObjectSharedProperty::FeatureObjectSharedProperty`.
        public unsafe Const_FeatureObjectSharedProperty(MR._ByValue_FeatureObjectSharedProperty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FeatureObjectSharedProperty._Underlying *__MR_FeatureObjectSharedProperty_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FeatureObjectSharedProperty._Underlying *_other);
            _UnderlyingPtr = __MR_FeatureObjectSharedProperty_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    // FeatureObjectSharedProperty struct is designed to represent a shared property of a feature object, enabling the use of generalized getter and setter methods for property manipulation.
    // propertyName: A string representing the name of the property.
    // getter : A std::function encapsulating a method with no parameters that returns a FeaturesPropertyTypesVariant.This allows for a generic way to retrieve the value of the property.
    // setter : A std::function encapsulating a method that takes a FeaturesPropertyTypesVariant as a parameter and returns void.This function sets the value of the property.
    // The templated constructor of this struct takes the property name, pointers to the getter and setter member functions, and a pointer to the object( obj ).
    // The constructor initializes the propertyName and uses lambdas to adapt the member function pointers into std::function objects that conform to the expected
    // getter and setter signatures.The getter lambda invokes the getter method on the object, and the setter lambda ensures the correct variant type is passed before
    // invoking the setter method.
    /// Generated from class `MR::FeatureObjectSharedProperty`.
    /// This is the non-const half of the class.
    public class FeatureObjectSharedProperty : Const_FeatureObjectSharedProperty
    {
        internal unsafe FeatureObjectSharedProperty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.String PropertyName
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_GetMutable_propertyName", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_FeatureObjectSharedProperty_GetMutable_propertyName(_Underlying *_this);
                return new(__MR_FeatureObjectSharedProperty_GetMutable_propertyName(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref MR.FeaturePropertyKind Kind
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_GetMutable_kind", ExactSpelling = true)]
                extern static MR.FeaturePropertyKind *__MR_FeatureObjectSharedProperty_GetMutable_kind(_Underlying *_this);
                return ref *__MR_FeatureObjectSharedProperty_GetMutable_kind(_UnderlyingPtr);
            }
        }

        // due to getAllSharedProperties in FeatureObject returns static vector, we need externaly setup object to invoke setter ad getter.
        public new unsafe MR.Std.Function_StdVariantFloatMRVector3fFuncFromConstMRFeatureObjectPtrMRViewportId Getter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_GetMutable_getter", ExactSpelling = true)]
                extern static MR.Std.Function_StdVariantFloatMRVector3fFuncFromConstMRFeatureObjectPtrMRViewportId._Underlying *__MR_FeatureObjectSharedProperty_GetMutable_getter(_Underlying *_this);
                return new(__MR_FeatureObjectSharedProperty_GetMutable_getter(_UnderlyingPtr), is_owning: false);
            }
        }

        // NOTE: `id` should usually be `{}`, not the current viewport ID, to set the property for all viewports.
        // Passing a non-zero ID would only modify the active viewport, and per-viewport properties aren't usually used.
        public new unsafe MR.Std.Function_VoidFuncFromConstStdVariantFloatMRVector3fRefMRFeatureObjectPtrMRViewportId Setter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_GetMutable_setter", ExactSpelling = true)]
                extern static MR.Std.Function_VoidFuncFromConstStdVariantFloatMRVector3fRefMRFeatureObjectPtrMRViewportId._Underlying *__MR_FeatureObjectSharedProperty_GetMutable_setter(_Underlying *_this);
                return new(__MR_FeatureObjectSharedProperty_GetMutable_setter(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Generated from constructor `MR::FeatureObjectSharedProperty::FeatureObjectSharedProperty`.
        public unsafe FeatureObjectSharedProperty(MR._ByValue_FeatureObjectSharedProperty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FeatureObjectSharedProperty._Underlying *__MR_FeatureObjectSharedProperty_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FeatureObjectSharedProperty._Underlying *_other);
            _UnderlyingPtr = __MR_FeatureObjectSharedProperty_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FeatureObjectSharedProperty::operator=`.
        public unsafe MR.FeatureObjectSharedProperty Assign(MR._ByValue_FeatureObjectSharedProperty _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectSharedProperty_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FeatureObjectSharedProperty._Underlying *__MR_FeatureObjectSharedProperty_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FeatureObjectSharedProperty._Underlying *_other);
            return new(__MR_FeatureObjectSharedProperty_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FeatureObjectSharedProperty` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FeatureObjectSharedProperty`/`Const_FeatureObjectSharedProperty` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FeatureObjectSharedProperty
    {
        internal readonly Const_FeatureObjectSharedProperty? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FeatureObjectSharedProperty(Const_FeatureObjectSharedProperty new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FeatureObjectSharedProperty(Const_FeatureObjectSharedProperty arg) {return new(arg);}
        public _ByValue_FeatureObjectSharedProperty(MR.Misc._Moved<FeatureObjectSharedProperty> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FeatureObjectSharedProperty(MR.Misc._Moved<FeatureObjectSharedProperty> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FeatureObjectSharedProperty` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FeatureObjectSharedProperty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FeatureObjectSharedProperty`/`Const_FeatureObjectSharedProperty` directly.
    public class _InOptMut_FeatureObjectSharedProperty
    {
        public FeatureObjectSharedProperty? Opt;

        public _InOptMut_FeatureObjectSharedProperty() {}
        public _InOptMut_FeatureObjectSharedProperty(FeatureObjectSharedProperty value) {Opt = value;}
        public static implicit operator _InOptMut_FeatureObjectSharedProperty(FeatureObjectSharedProperty value) {return new(value);}
    }

    /// This is used for optional parameters of class `FeatureObjectSharedProperty` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FeatureObjectSharedProperty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FeatureObjectSharedProperty`/`Const_FeatureObjectSharedProperty` to pass it to the function.
    public class _InOptConst_FeatureObjectSharedProperty
    {
        public Const_FeatureObjectSharedProperty? Opt;

        public _InOptConst_FeatureObjectSharedProperty() {}
        public _InOptConst_FeatureObjectSharedProperty(Const_FeatureObjectSharedProperty value) {Opt = value;}
        public static implicit operator _InOptConst_FeatureObjectSharedProperty(Const_FeatureObjectSharedProperty value) {return new(value);}
    }

    /// Generated from class `MR::FeatureObjectProjectPointResult`.
    /// This is the const half of the class.
    public class Const_FeatureObjectProjectPointResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FeatureObjectProjectPointResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_Destroy", ExactSpelling = true)]
            extern static void __MR_FeatureObjectProjectPointResult_Destroy(_Underlying *_this);
            __MR_FeatureObjectProjectPointResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FeatureObjectProjectPointResult() {Dispose(false);}

        public unsafe MR.Const_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_Get_point", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_FeatureObjectProjectPointResult_Get_point(_Underlying *_this);
                return new(__MR_FeatureObjectProjectPointResult_Get_point(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Std.Const_Optional_MRVector3f Normal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_Get_normal", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRVector3f._Underlying *__MR_FeatureObjectProjectPointResult_Get_normal(_Underlying *_this);
                return new(__MR_FeatureObjectProjectPointResult_Get_normal(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FeatureObjectProjectPointResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_DefaultConstruct();
            _UnderlyingPtr = __MR_FeatureObjectProjectPointResult_DefaultConstruct();
        }

        /// Constructs `MR::FeatureObjectProjectPointResult` elementwise.
        public unsafe Const_FeatureObjectProjectPointResult(MR.Vector3f point, MR._InOpt_Vector3f normal) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_ConstructFrom(MR.Vector3f point, MR.Vector3f *normal);
            _UnderlyingPtr = __MR_FeatureObjectProjectPointResult_ConstructFrom(point, normal.HasValue ? &normal.Object : null);
        }

        /// Generated from constructor `MR::FeatureObjectProjectPointResult::FeatureObjectProjectPointResult`.
        public unsafe Const_FeatureObjectProjectPointResult(MR.Const_FeatureObjectProjectPointResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_ConstructFromAnother(MR.FeatureObjectProjectPointResult._Underlying *_other);
            _UnderlyingPtr = __MR_FeatureObjectProjectPointResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::FeatureObjectProjectPointResult`.
    /// This is the non-const half of the class.
    public class FeatureObjectProjectPointResult : Const_FeatureObjectProjectPointResult
    {
        internal unsafe FeatureObjectProjectPointResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f Point
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_GetMutable_point", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_FeatureObjectProjectPointResult_GetMutable_point(_Underlying *_this);
                return new(__MR_FeatureObjectProjectPointResult_GetMutable_point(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Std.Optional_MRVector3f Normal
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_GetMutable_normal", ExactSpelling = true)]
                extern static MR.Std.Optional_MRVector3f._Underlying *__MR_FeatureObjectProjectPointResult_GetMutable_normal(_Underlying *_this);
                return new(__MR_FeatureObjectProjectPointResult_GetMutable_normal(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FeatureObjectProjectPointResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_DefaultConstruct();
            _UnderlyingPtr = __MR_FeatureObjectProjectPointResult_DefaultConstruct();
        }

        /// Constructs `MR::FeatureObjectProjectPointResult` elementwise.
        public unsafe FeatureObjectProjectPointResult(MR.Vector3f point, MR._InOpt_Vector3f normal) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_ConstructFrom(MR.Vector3f point, MR.Vector3f *normal);
            _UnderlyingPtr = __MR_FeatureObjectProjectPointResult_ConstructFrom(point, normal.HasValue ? &normal.Object : null);
        }

        /// Generated from constructor `MR::FeatureObjectProjectPointResult::FeatureObjectProjectPointResult`.
        public unsafe FeatureObjectProjectPointResult(MR.Const_FeatureObjectProjectPointResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_ConstructFromAnother(MR.FeatureObjectProjectPointResult._Underlying *_other);
            _UnderlyingPtr = __MR_FeatureObjectProjectPointResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObjectProjectPointResult::operator=`.
        public unsafe MR.FeatureObjectProjectPointResult Assign(MR.Const_FeatureObjectProjectPointResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObjectProjectPointResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObjectProjectPointResult_AssignFromAnother(_Underlying *_this, MR.FeatureObjectProjectPointResult._Underlying *_other);
            return new(__MR_FeatureObjectProjectPointResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FeatureObjectProjectPointResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FeatureObjectProjectPointResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FeatureObjectProjectPointResult`/`Const_FeatureObjectProjectPointResult` directly.
    public class _InOptMut_FeatureObjectProjectPointResult
    {
        public FeatureObjectProjectPointResult? Opt;

        public _InOptMut_FeatureObjectProjectPointResult() {}
        public _InOptMut_FeatureObjectProjectPointResult(FeatureObjectProjectPointResult value) {Opt = value;}
        public static implicit operator _InOptMut_FeatureObjectProjectPointResult(FeatureObjectProjectPointResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `FeatureObjectProjectPointResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FeatureObjectProjectPointResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FeatureObjectProjectPointResult`/`Const_FeatureObjectProjectPointResult` to pass it to the function.
    public class _InOptConst_FeatureObjectProjectPointResult
    {
        public Const_FeatureObjectProjectPointResult? Opt;

        public _InOptConst_FeatureObjectProjectPointResult() {}
        public _InOptConst_FeatureObjectProjectPointResult(Const_FeatureObjectProjectPointResult value) {Opt = value;}
        public static implicit operator _InOptConst_FeatureObjectProjectPointResult(Const_FeatureObjectProjectPointResult value) {return new(value);}
    }

    public enum FeatureVisualizePropertyType : int
    {
        Subfeatures = 0,
        // If true, show additional details on the name tag, such as point coordinates. Not all features use this.
        DetailsOnNameTag = 1,
        Count = 2,
    }

    /// An interface class which allows feature objects to share setters and getters on their main properties, for convenient presentation in the UI
    /// Generated from class `MR::FeatureObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::LineObject`
    ///     `MR::PlaneObject`
    ///     `MR::PointObject`
    ///   Indirect: (non-virtual)
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::SphereObject`
    /// This is the const half of the class.
    public class Const_FeatureObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_FeatureObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_FeatureObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_FeatureObject_UseCount();
                return __MR_std_shared_ptr_MR_FeatureObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FeatureObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_FeatureObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_FeatureObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FeatureObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FeatureObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_FeatureObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_FeatureObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_FeatureObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe FeatureObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FeatureObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_FeatureObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FeatureObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_FeatureObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FeatureObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_FeatureObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_FeatureObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FeatureObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_FeatureObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_FeatureObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FeatureObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_FeatureObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_FeatureObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FeatureObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_FeatureObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_FeatureObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FeatureObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_FeatureObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_FeatureObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_FeatureObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_FeatureObject?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_FeatureObject(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_FeatureObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::FeatureObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_FeatureObject_StaticTypeName();
            var __ret = __MR_FeatureObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::FeatureObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_FeatureObject_typeName(_Underlying *_this);
            var __ret = __MR_FeatureObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::FeatureObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_FeatureObject_StaticClassName();
            var __ret = __MR_FeatureObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::FeatureObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_FeatureObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_FeatureObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FeatureObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_FeatureObject_StaticClassNameInPlural();
            var __ret = __MR_FeatureObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::FeatureObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_FeatureObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_FeatureObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Create and generate list of bounded getters and setters for the main properties of feature object, together with prop. name for display and edit into UI.
        /// Generated from method `MR::FeatureObject::getAllSharedProperties`.
        public unsafe MR.Std.Const_Vector_MRFeatureObjectSharedProperty GetAllSharedProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getAllSharedProperties", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRFeatureObjectSharedProperty._Underlying *__MR_FeatureObject_getAllSharedProperties(_Underlying *_this);
            return new(__MR_FeatureObject_getAllSharedProperties(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FeatureObject::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_FeatureObject_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FeatureObject::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_FeatureObject_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_FeatureObject_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FeatureObject::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_FeatureObject_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_FeatureObject_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        // Since a point on an abstract feature is difficult to uniquely parameterize,
        // the projection function simultaneously returns the normal to the surface at the projection point.
        /// Generated from method `MR::FeatureObject::projectPoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.FeatureObjectProjectPointResult ProjectPoint(MR.Const_Vector3f point, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_projectPoint", ExactSpelling = true)]
            extern static MR.FeatureObjectProjectPointResult._Underlying *__MR_FeatureObject_projectPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point, MR.ViewportId *id);
            return new(__MR_FeatureObject_projectPoint(_UnderlyingPtr, point._UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: true);
        }

        /// Generated from method `MR::FeatureObject::getNormal`.
        public unsafe MR.Std.Optional_MRVector3f GetNormal(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getNormal", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVector3f._Underlying *__MR_FeatureObject_getNormal(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return new(__MR_FeatureObject_getNormal(_UnderlyingPtr, point._UnderlyingPtr), is_owning: true);
        }

        // Returns point considered as base for the feature
        /// Generated from method `MR::FeatureObject::getBasePoint`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Vector3f GetBasePoint(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getBasePoint", ExactSpelling = true)]
            extern static MR.Vector3f __MR_FeatureObject_getBasePoint(_Underlying *_this, MR.ViewportId *id);
            return __MR_FeatureObject_getBasePoint(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        // The cached orthonormalized rotation matrix.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::FeatureObject::getRotationMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetRotationMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getRotationMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_FeatureObject_getRotationMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_FeatureObject_getRotationMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // The cached scale and shear matrix. The main diagnoal stores the scale, and some other elements store the shearing.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::FeatureObject::getScaleShearMatrix`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Matrix3f GetScaleShearMatrix(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getScaleShearMatrix", ExactSpelling = true)]
            extern static MR.Matrix3f __MR_FeatureObject_getScaleShearMatrix(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_FeatureObject_getScaleShearMatrix(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        // This color is used for subfeatures.
        // `isDef` receives false if matrix is overridden for this specific viewport.
        /// Generated from method `MR::FeatureObject::getDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetDecorationsColor(bool selected, MR._InOpt_ViewportId viewportId = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getDecorationsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_FeatureObject_getDecorationsColor(_Underlying *_this, byte selected, MR.ViewportId *viewportId, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_FeatureObject_getDecorationsColor(_UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// Generated from method `MR::FeatureObject::getDecorationsColorForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetDecorationsColorForAllViewports(bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_FeatureObject_getDecorationsColorForAllViewports(_Underlying *_this, byte selected);
            return new(__MR_FeatureObject_getDecorationsColorForAllViewports(_UnderlyingPtr, selected ? (byte)1 : (byte)0), is_owning: false);
        }

        // Point size and line width, for primary rendering rather than subfeatures.
        /// Generated from method `MR::FeatureObject::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getPointSize", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getPointSize(_Underlying *_this);
            return __MR_FeatureObject_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObject::getLineWidth`.
        public unsafe float GetLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getLineWidth", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getLineWidth(_Underlying *_this);
            return __MR_FeatureObject_getLineWidth(_UnderlyingPtr);
        }

        // Point size and line width, for subfeatures rather than primary rendering.
        /// Generated from method `MR::FeatureObject::getSubfeaturePointSize`.
        public unsafe float GetSubfeaturePointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSubfeaturePointSize", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getSubfeaturePointSize(_Underlying *_this);
            return __MR_FeatureObject_getSubfeaturePointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObject::getSubfeatureLineWidth`.
        public unsafe float GetSubfeatureLineWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSubfeatureLineWidth", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getSubfeatureLineWidth(_Underlying *_this);
            return __MR_FeatureObject_getSubfeatureLineWidth(_UnderlyingPtr);
        }

        // Per-component alpha multipliers. The global alpha is multiplied by thise.
        /// Generated from method `MR::FeatureObject::getMainFeatureAlpha`.
        public unsafe float GetMainFeatureAlpha()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getMainFeatureAlpha", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getMainFeatureAlpha(_Underlying *_this);
            return __MR_FeatureObject_getMainFeatureAlpha(_UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObject::getSubfeatureAlphaPoints`.
        public unsafe float GetSubfeatureAlphaPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getSubfeatureAlphaPoints(_Underlying *_this);
            return __MR_FeatureObject_getSubfeatureAlphaPoints(_UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObject::getSubfeatureAlphaLines`.
        public unsafe float GetSubfeatureAlphaLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSubfeatureAlphaLines", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getSubfeatureAlphaLines(_Underlying *_this);
            return __MR_FeatureObject_getSubfeatureAlphaLines(_UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObject::getSubfeatureAlphaMesh`.
        public unsafe float GetSubfeatureAlphaMesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getSubfeatureAlphaMesh(_Underlying *_this);
            return __MR_FeatureObject_getSubfeatureAlphaMesh(_UnderlyingPtr);
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::FeatureObject::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_FeatureObject_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::FeatureObject::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_FeatureObject_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_FeatureObject_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::FeatureObject::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_FeatureObject_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::FeatureObject::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_FeatureObject_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_FeatureObject_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::FeatureObject::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_FeatureObject_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_FeatureObject_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::FeatureObject::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_FeatureObject_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_FeatureObject_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::FeatureObject::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_FeatureObject_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_FeatureObject_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::FeatureObject::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_FeatureObject_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_FeatureObject_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::FeatureObject::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_FeatureObject_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_FeatureObject_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::FeatureObject::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_FeatureObject_getDirtyFlags(_Underlying *_this);
            return __MR_FeatureObject_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::FeatureObject::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_resetDirty", ExactSpelling = true)]
            extern static void __MR_FeatureObject_resetDirty(_Underlying *_this);
            __MR_FeatureObject_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::FeatureObject::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_FeatureObject_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_FeatureObject_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::FeatureObject::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_FeatureObject_getBoundingBox(_Underlying *_this);
            return __MR_FeatureObject_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::FeatureObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_FeatureObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_FeatureObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::FeatureObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_FeatureObject_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::FeatureObject::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isPickable", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_FeatureObject_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::FeatureObject::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_FeatureObject_getColoringType(_Underlying *_this);
            return __MR_FeatureObject_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::FeatureObject::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getShininess", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getShininess(_Underlying *_this);
            return __MR_FeatureObject_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::FeatureObject::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getSpecularStrength(_Underlying *_this);
            return __MR_FeatureObject_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::FeatureObject::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_FeatureObject_getAmbientStrength(_Underlying *_this);
            return __MR_FeatureObject_getAmbientStrength(_UnderlyingPtr);
        }

        /// clones this object only, without its children,
        /// making new object the owner of all copied resources
        /// Generated from method `MR::FeatureObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_FeatureObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_FeatureObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones this object only, without its children,
        /// making new object to share resources with this object
        /// Generated from method `MR::FeatureObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_FeatureObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_FeatureObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::FeatureObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_render", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_FeatureObject_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::FeatureObject::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_renderForPicker", ExactSpelling = true)]
            extern static void __MR_FeatureObject_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_FeatureObject_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::FeatureObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_renderUi", ExactSpelling = true)]
            extern static void __MR_FeatureObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_FeatureObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::FeatureObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_FeatureObject_heapBytes(_Underlying *_this);
            return __MR_FeatureObject_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::FeatureObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_FeatureObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_FeatureObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::FeatureObject::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_FeatureObject_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FeatureObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_FeatureObject_name(_Underlying *_this);
            return new(__MR_FeatureObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::FeatureObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_FeatureObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_FeatureObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::FeatureObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_FeatureObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_FeatureObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::FeatureObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_FeatureObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_FeatureObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::FeatureObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_FeatureObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_FeatureObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::FeatureObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_FeatureObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::FeatureObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isLocked(_Underlying *_this);
            return __MR_FeatureObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::FeatureObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isParentLocked(_Underlying *_this);
            return __MR_FeatureObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::FeatureObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_FeatureObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::FeatureObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isSelected(_Underlying *_this);
            return __MR_FeatureObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FeatureObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isAncillary(_Underlying *_this);
            return __MR_FeatureObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::FeatureObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isGlobalAncillary(_Underlying *_this);
            return __MR_FeatureObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::FeatureObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_FeatureObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::FeatureObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_FeatureObject_visibilityMask(_Underlying *_this);
            return new(__MR_FeatureObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::FeatureObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_FeatureObject_resetRedrawFlag(_Underlying *_this);
            __MR_FeatureObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::FeatureObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_FeatureObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_FeatureObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::FeatureObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_FeatureObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_FeatureObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::FeatureObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_FeatureObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_FeatureObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::FeatureObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_FeatureObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::FeatureObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_hasModel(_Underlying *_this);
            return __MR_FeatureObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::FeatureObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_FeatureObject_tags(_Underlying *_this);
            return new(__MR_FeatureObject_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::FeatureObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_FeatureObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::FeatureObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_FeatureObject_getModelHash(_Underlying *_this);
            return __MR_FeatureObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::FeatureObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_FeatureObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_FeatureObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// An interface class which allows feature objects to share setters and getters on their main properties, for convenient presentation in the UI
    /// Generated from class `MR::FeatureObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::LineObject`
    ///     `MR::PlaneObject`
    ///     `MR::PointObject`
    ///   Indirect: (non-virtual)
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::SphereObject`
    /// This is the non-const half of the class.
    public class FeatureObject : Const_FeatureObject
    {
        internal unsafe FeatureObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe FeatureObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(FeatureObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_FeatureObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FeatureObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(FeatureObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_FeatureObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FeatureObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(FeatureObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_FeatureObject_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FeatureObject_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator FeatureObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_FeatureObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_FeatureObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator FeatureObject?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_FeatureObject", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_FeatureObject(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_FeatureObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from method `MR::FeatureObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setXf", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_FeatureObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::FeatureObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_resetXf", ExactSpelling = true)]
            extern static void __MR_FeatureObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_FeatureObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::FeatureObject::setDecorationsColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetDecorationsColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setDecorationsColor", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setDecorationsColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_FeatureObject_setDecorationsColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// Generated from method `MR::FeatureObject::setDecorationsColorForAllViewports`.
        public unsafe void SetDecorationsColorForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool selected)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setDecorationsColorForAllViewports", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setDecorationsColorForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte selected);
            __MR_FeatureObject_setDecorationsColorForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::FeatureObject::setPointSize`.
        public unsafe void SetPointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setPointSize", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setPointSize(_Underlying *_this, float pointSize);
            __MR_FeatureObject_setPointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::FeatureObject::setLineWidth`.
        public unsafe void SetLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setLineWidth", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setLineWidth(_Underlying *_this, float lineWidth);
            __MR_FeatureObject_setLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::FeatureObject::setSubfeaturePointSize`.
        public unsafe void SetSubfeaturePointSize(float pointSize)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setSubfeaturePointSize", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setSubfeaturePointSize(_Underlying *_this, float pointSize);
            __MR_FeatureObject_setSubfeaturePointSize(_UnderlyingPtr, pointSize);
        }

        /// Generated from method `MR::FeatureObject::setSubfeatureLineWidth`.
        public unsafe void SetSubfeatureLineWidth(float lineWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setSubfeatureLineWidth", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setSubfeatureLineWidth(_Underlying *_this, float lineWidth);
            __MR_FeatureObject_setSubfeatureLineWidth(_UnderlyingPtr, lineWidth);
        }

        /// Generated from method `MR::FeatureObject::setMainFeatureAlpha`.
        public unsafe void SetMainFeatureAlpha(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setMainFeatureAlpha", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setMainFeatureAlpha(_Underlying *_this, float alpha);
            __MR_FeatureObject_setMainFeatureAlpha(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::FeatureObject::setSubfeatureAlphaPoints`.
        public unsafe void SetSubfeatureAlphaPoints(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setSubfeatureAlphaPoints", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setSubfeatureAlphaPoints(_Underlying *_this, float alpha);
            __MR_FeatureObject_setSubfeatureAlphaPoints(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::FeatureObject::setSubfeatureAlphaLines`.
        public unsafe void SetSubfeatureAlphaLines(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setSubfeatureAlphaLines", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setSubfeatureAlphaLines(_Underlying *_this, float alpha);
            __MR_FeatureObject_setSubfeatureAlphaLines(_UnderlyingPtr, alpha);
        }

        /// Generated from method `MR::FeatureObject::setSubfeatureAlphaMesh`.
        public unsafe void SetSubfeatureAlphaMesh(float alpha)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setSubfeatureAlphaMesh", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setSubfeatureAlphaMesh(_Underlying *_this, float alpha);
            __MR_FeatureObject_setSubfeatureAlphaMesh(_UnderlyingPtr, alpha);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::FeatureObject::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::FeatureObject::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::FeatureObject::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_FeatureObject_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::FeatureObject::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_FeatureObject_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::FeatureObject::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_FeatureObject_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_FeatureObject_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::FeatureObject::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::FeatureObject::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setFrontColor", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_FeatureObject_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::FeatureObject::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_FeatureObject_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::FeatureObject::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_FeatureObject_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::FeatureObject::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setBackColor", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_FeatureObject_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::FeatureObject::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_FeatureObject_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::FeatureObject::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_FeatureObject_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::FeatureObject::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_FeatureObject_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::FeatureObject::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setPickable", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::FeatureObject::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setColoringType", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_FeatureObject_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::FeatureObject::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setShininess", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setShininess(_Underlying *_this, float shininess);
            __MR_FeatureObject_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::FeatureObject::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_FeatureObject_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::FeatureObject::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_FeatureObject_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::FeatureObject::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_FeatureObject_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::FeatureObject::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_FeatureObject_resetFrontColor(_Underlying *_this);
            __MR_FeatureObject_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::FeatureObject::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_resetColors", ExactSpelling = true)]
            extern static void __MR_FeatureObject_resetColors(_Underlying *_this);
            __MR_FeatureObject_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::FeatureObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setName", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_FeatureObject_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::FeatureObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_FeatureObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FeatureObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_FeatureObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::FeatureObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_applyScale", ExactSpelling = true)]
            extern static void __MR_FeatureObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_FeatureObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::FeatureObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FeatureObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setLocked", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setLocked(_Underlying *_this, byte on);
            __MR_FeatureObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::FeatureObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_FeatureObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::FeatureObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_detachFromParent(_Underlying *_this);
            return __MR_FeatureObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::FeatureObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_addChild", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_FeatureObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::FeatureObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_FeatureObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::FeatureObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_FeatureObject_removeAllChildren(_Underlying *_this);
            __MR_FeatureObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::FeatureObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_FeatureObject_sortChildren(_Underlying *_this);
            __MR_FeatureObject_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::FeatureObject::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_select", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_select(_Underlying *_this, byte on);
            return __MR_FeatureObject_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::FeatureObject::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setAncillary(_Underlying *_this, byte ancillary);
            __MR_FeatureObject_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::FeatureObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setVisible", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::FeatureObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_FeatureObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_FeatureObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::FeatureObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_swap", ExactSpelling = true)]
            extern static void __MR_FeatureObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_FeatureObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::FeatureObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_addTag", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_FeatureObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::FeatureObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FeatureObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_FeatureObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_FeatureObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `FeatureObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FeatureObject`/`Const_FeatureObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FeatureObject
    {
        internal readonly Const_FeatureObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `FeatureObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FeatureObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FeatureObject`/`Const_FeatureObject` directly.
    public class _InOptMut_FeatureObject
    {
        public FeatureObject? Opt;

        public _InOptMut_FeatureObject() {}
        public _InOptMut_FeatureObject(FeatureObject value) {Opt = value;}
        public static implicit operator _InOptMut_FeatureObject(FeatureObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `FeatureObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FeatureObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FeatureObject`/`Const_FeatureObject` to pass it to the function.
    public class _InOptConst_FeatureObject
    {
        public Const_FeatureObject? Opt;

        public _InOptConst_FeatureObject() {}
        public _InOptConst_FeatureObject(Const_FeatureObject value) {Opt = value;}
        public static implicit operator _InOptConst_FeatureObject(Const_FeatureObject value) {return new(value);}
    }
}
