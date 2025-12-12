public static partial class MR
{
    /// the main purpose of this class is to avoid copy and move constructor and assignment operator
    /// implementation in Object class, which has too many fields for that;
    /// since every object stores a pointer on its parent,
    /// copying of this object does not copy the children and moving is taken with care
    /// Generated from class `MR::ObjectChildrenHolder`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::Object`
    ///   Indirect: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::AngleMeasurementObject`
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::FeatureObject`
    ///     `MR::LineObject`
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLabel`
    ///     `MR::ObjectLines`
    ///     `MR::ObjectLinesHolder`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectMeshHolder`
    ///     `MR::ObjectPoints`
    ///     `MR::ObjectPointsHolder`
    ///     `MR::ObjectVoxels`
    ///     `MR::PlaneObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::PointObject`
    ///     `MR::RadiusMeasurementObject`
    ///     `MR::SceneRootObject`
    ///     `MR::SphereObject`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_ObjectChildrenHolder : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectChildrenHolder_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectChildrenHolder_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectChildrenHolder_UseCount();
                return __MR_std_shared_ptr_MR_ObjectChildrenHolder_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectChildrenHolder(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectChildrenHolder_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectChildrenHolder_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectChildrenHolder(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectChildrenHolder _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectChildrenHolder_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectChildrenHolder_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectChildrenHolder_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectChildrenHolder_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectChildrenHolder_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectChildrenHolder_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectChildrenHolder() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectChildrenHolder() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectChildrenHolder_DefaultConstruct();
            _LateMakeShared(__MR_ObjectChildrenHolder_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectChildrenHolder::ObjectChildrenHolder`.
        public unsafe Const_ObjectChildrenHolder(MR._ByValue_ObjectChildrenHolder _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectChildrenHolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectChildrenHolder._Underlying *_other);
            _LateMakeShared(__MR_ObjectChildrenHolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectChildrenHolder::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectChildrenHolder_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectChildrenHolder_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }

        /// returns the amount of memory this object occupies on heap,
        /// including the memory of all recognized children
        /// Generated from method `MR::ObjectChildrenHolder::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectChildrenHolder_heapBytes(_Underlying *_this);
            return __MR_ObjectChildrenHolder_heapBytes(_UnderlyingPtr);
        }
    }

    /// the main purpose of this class is to avoid copy and move constructor and assignment operator
    /// implementation in Object class, which has too many fields for that;
    /// since every object stores a pointer on its parent,
    /// copying of this object does not copy the children and moving is taken with care
    /// Generated from class `MR::ObjectChildrenHolder`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::Object`
    ///   Indirect: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::AngleMeasurementObject`
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::FeatureObject`
    ///     `MR::LineObject`
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLabel`
    ///     `MR::ObjectLines`
    ///     `MR::ObjectLinesHolder`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectMeshHolder`
    ///     `MR::ObjectPoints`
    ///     `MR::ObjectPointsHolder`
    ///     `MR::ObjectVoxels`
    ///     `MR::PlaneObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::PointObject`
    ///     `MR::RadiusMeasurementObject`
    ///     `MR::SceneRootObject`
    ///     `MR::SphereObject`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class ObjectChildrenHolder : Const_ObjectChildrenHolder
    {
        internal unsafe ObjectChildrenHolder(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectChildrenHolder(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectChildrenHolder() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectChildrenHolder_DefaultConstruct();
            _LateMakeShared(__MR_ObjectChildrenHolder_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectChildrenHolder::ObjectChildrenHolder`.
        public unsafe ObjectChildrenHolder(MR._ByValue_ObjectChildrenHolder _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectChildrenHolder_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectChildrenHolder._Underlying *_other);
            _LateMakeShared(__MR_ObjectChildrenHolder_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectChildrenHolder::operator=`.
        public unsafe MR.ObjectChildrenHolder Assign(MR._ByValue_ObjectChildrenHolder _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectChildrenHolder_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectChildrenHolder_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectChildrenHolder._Underlying *_other);
            return new(__MR_ObjectChildrenHolder_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectChildrenHolder` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectChildrenHolder`/`Const_ObjectChildrenHolder` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectChildrenHolder
    {
        internal readonly Const_ObjectChildrenHolder? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectChildrenHolder() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectChildrenHolder(Const_ObjectChildrenHolder new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ObjectChildrenHolder(Const_ObjectChildrenHolder arg) {return new(arg);}
        public _ByValue_ObjectChildrenHolder(MR.Misc._Moved<ObjectChildrenHolder> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectChildrenHolder(MR.Misc._Moved<ObjectChildrenHolder> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectChildrenHolder` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectChildrenHolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectChildrenHolder`/`Const_ObjectChildrenHolder` directly.
    public class _InOptMut_ObjectChildrenHolder
    {
        public ObjectChildrenHolder? Opt;

        public _InOptMut_ObjectChildrenHolder() {}
        public _InOptMut_ObjectChildrenHolder(ObjectChildrenHolder value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectChildrenHolder(ObjectChildrenHolder value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectChildrenHolder` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectChildrenHolder`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectChildrenHolder`/`Const_ObjectChildrenHolder` to pass it to the function.
    public class _InOptConst_ObjectChildrenHolder
    {
        public Const_ObjectChildrenHolder? Opt;

        public _InOptConst_ObjectChildrenHolder() {}
        public _InOptConst_ObjectChildrenHolder(Const_ObjectChildrenHolder value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectChildrenHolder(Const_ObjectChildrenHolder value) {return new(value);}
    }

    /// named object in the data model
    /// Generated from class `MR::Object`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SceneRootObject`
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::AngleMeasurementObject`
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::FeatureObject`
    ///     `MR::LineObject`
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLabel`
    ///     `MR::ObjectLines`
    ///     `MR::ObjectLinesHolder`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectMeshHolder`
    ///     `MR::ObjectPoints`
    ///     `MR::ObjectPointsHolder`
    ///     `MR::ObjectVoxels`
    ///     `MR::PlaneObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::PointObject`
    ///     `MR::RadiusMeasurementObject`
    ///     `MR::SphereObject`
    /// This is the const half of the class.
    public class Const_Object : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_Object_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_Object_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_Object_UseCount();
                return __MR_std_shared_ptr_MR_Object_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Object_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_Object_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_Object(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Object_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Object_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Object_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Object_ConstructNonOwning(ptr);
        }

        internal unsafe Const_Object(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe Object _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Object_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_Object_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_Object_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_Object_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_Object_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_Object_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_Object_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Object() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_Object self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_Object_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_Object_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Object() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_DefaultConstruct();
            _LateMakeShared(__MR_Object_DefaultConstruct());
        }

        /// Generated from constructor `MR::Object::Object`.
        public unsafe Const_Object(MR._ByValue_Object _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Object._Underlying *_other);
            _LateMakeShared(__MR_Object_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        // return name of subtype for serialization purposes
        /// Generated from method `MR::Object::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_Object_StaticTypeName();
            var __ret = __MR_Object_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::Object::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_typeName", ExactSpelling = true)]
            extern static byte *__MR_Object_typeName(_Underlying *_this);
            var __ret = __MR_Object_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// return human readable name of subclass
        /// Generated from method `MR::Object::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_Object_StaticClassName();
            var __ret = __MR_Object_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::Object::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_Object_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_Object_className(_UnderlyingPtr), is_owning: true));
        }

        /// return human readable name of subclass in plural form
        /// Generated from method `MR::Object::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_Object_StaticClassNameInPlural();
            var __ret = __MR_Object_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::Object::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_Object_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_Object_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::Object::asType<MR::VisualObject>`.
        public unsafe MR.Const_VisualObject? AsType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_asType_const", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_Object_asType_const(_Underlying *_this);
            var __ret = __MR_Object_asType_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_VisualObject(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::Object::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_Object_name(_Underlying *_this);
            return new(__MR_Object_name(_UnderlyingPtr), is_owning: false);
        }

        /// finds a direct child by name
        /// Generated from method `MR::Object::find`.
        public unsafe MR.Misc._Moved<MR.Object> Find(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_find_const", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_find_const(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                return MR.Misc.Move(new MR.Object(__MR_Object_find_const(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name), is_owning: true));
            }
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::Object::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_Object_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_Object_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::Object::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_Object_xfsForAllViewports(_Underlying *_this);
            return new(__MR_Object_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::Object::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_Object_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_Object_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::Object::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_Object_globalVisibilityMask(_Underlying *_this);
            return new(__MR_Object_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::Object::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_Object_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_Object_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::Object::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isLocked", ExactSpelling = true)]
            extern static byte __MR_Object_isLocked(_Underlying *_this);
            return __MR_Object_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::Object::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_Object_isParentLocked(_Underlying *_this);
            return __MR_Object_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// returns parent object in the tree
        /// Generated from method `MR::Object::parent`.
        public unsafe MR.Const_Object? Parent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_parent_const", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_Object_parent_const(_Underlying *_this);
            var __ret = __MR_Object_parent_const(_UnderlyingPtr);
            return __ret is not null ? new MR.Const_Object(__ret, is_owning: false) : null;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::Object::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isAncestor", ExactSpelling = true)]
            extern static byte __MR_Object_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_Object_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::Object::findCommonAncestor`.
        public unsafe MR.Const_Object? FindCommonAncestor(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_findCommonAncestor_const", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_Object_findCommonAncestor_const(_Underlying *_this, MR.Const_Object._Underlying *other);
            var __ret = __MR_Object_findCommonAncestor_const(_UnderlyingPtr, other._UnderlyingPtr);
            return __ret is not null ? new MR.Const_Object(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::Object::children`.
        public unsafe MR.Std.Const_Vector_StdSharedPtrConstMRObject Children()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_children_const", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_StdSharedPtrConstMRObject._Underlying *__MR_Object_children_const(_Underlying *_this);
            return new(__MR_Object_children_const(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Object::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isSelected", ExactSpelling = true)]
            extern static byte __MR_Object_isSelected(_Underlying *_this);
            return __MR_Object_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Object::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isAncillary", ExactSpelling = true)]
            extern static byte __MR_Object_isAncillary(_Underlying *_this);
            return __MR_Object_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::Object::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_Object_isGlobalAncillary(_Underlying *_this);
            return __MR_Object_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::Object::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_isVisible", ExactSpelling = true)]
            extern static byte __MR_Object_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_Object_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::Object::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_Object_visibilityMask(_Underlying *_this);
            return new(__MR_Object_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// this method virtual because others data model types could have dirty flags or something
        /// Generated from method `MR::Object::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_Object_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *_1);
            return __MR_Object_getRedrawFlag(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::Object::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_Object_resetRedrawFlag(_Underlying *_this);
            __MR_Object_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::Object::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_Object_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones current object only, without parent and/or children
        /// Generated from method `MR::Object::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_Object_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::Object::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_Object_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones current object only, without parent and/or children
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::Object::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_Object_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// return several info lines that can better describe object in the UI
        /// Generated from method `MR::Object::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_Object_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_Object_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// returns bounding box of this object in world coordinates for default or specific viewport
        /// Generated from method `MR::Object::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_Object_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_Object_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::Object::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_Object_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_Object_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::Object::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_Object_hasVisualRepresentation(_Underlying *_this);
            return __MR_Object_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::Object::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_hasModel", ExactSpelling = true)]
            extern static byte __MR_Object_hasModel(_Underlying *_this);
            return __MR_Object_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::Object::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_Object_tags(_Underlying *_this);
            return new(__MR_Object_tags(_UnderlyingPtr), is_owning: false);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Object::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Object_heapBytes(_Underlying *_this);
            return __MR_Object_heapBytes(_UnderlyingPtr);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::Object::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_sameModels", ExactSpelling = true)]
            extern static byte __MR_Object_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_Object_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::Object::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_Object_getModelHash(_Underlying *_this);
            return __MR_Object_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::Object::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_Object_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// named object in the data model
    /// Generated from class `MR::Object`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SceneRootObject`
    ///     `MR::VisualObject`
    ///   Indirect: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::AngleMeasurementObject`
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::FeatureObject`
    ///     `MR::LineObject`
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLabel`
    ///     `MR::ObjectLines`
    ///     `MR::ObjectLinesHolder`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectMeshHolder`
    ///     `MR::ObjectPoints`
    ///     `MR::ObjectPointsHolder`
    ///     `MR::ObjectVoxels`
    ///     `MR::PlaneObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::PointObject`
    ///     `MR::RadiusMeasurementObject`
    ///     `MR::SphereObject`
    /// This is the non-const half of the class.
    public class Object : Const_Object
    {
        internal unsafe Object(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe Object(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(Object self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_Object_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_Object_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Object() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_DefaultConstruct();
            _LateMakeShared(__MR_Object_DefaultConstruct());
        }

        /// Generated from constructor `MR::Object::Object`.
        public unsafe Object(MR._ByValue_Object _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Object._Underlying *_other);
            _LateMakeShared(__MR_Object_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::Object::operator=`.
        public unsafe MR.Object Assign(MR._ByValue_Object _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Object._Underlying *_other);
            return new(__MR_Object_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::Object::asType<MR::VisualObject>`.
        public unsafe new MR.VisualObject? AsType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_asType", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_Object_asType(_Underlying *_this);
            var __ret = __MR_Object_asType(_UnderlyingPtr);
            return __ret is not null ? new MR.VisualObject(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::Object::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setName", ExactSpelling = true)]
            extern static void __MR_Object_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_Object_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::Object::find`.
        public unsafe new MR.Misc._Moved<MR.Object> Find(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_find", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_Object_find(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                return MR.Misc.Move(new MR.Object(__MR_Object_find(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name), is_owning: true));
            }
        }

        /// Generated from method `MR::Object::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setXf", ExactSpelling = true)]
            extern static void __MR_Object_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_Object_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::Object::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_resetXf", ExactSpelling = true)]
            extern static void __MR_Object_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_Object_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::Object::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_Object_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_Object_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Object::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setWorldXf", ExactSpelling = true)]
            extern static void __MR_Object_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_Object_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::Object::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_applyScale", ExactSpelling = true)]
            extern static void __MR_Object_applyScale(_Underlying *_this, float scaleFactor);
            __MR_Object_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::Object::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_Object_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_Object_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Object::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setLocked", ExactSpelling = true)]
            extern static void __MR_Object_setLocked(_Underlying *_this, byte on);
            __MR_Object_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Object::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setParentLocked", ExactSpelling = true)]
            extern static void __MR_Object_setParentLocked(_Underlying *_this, byte lock_);
            __MR_Object_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::Object::parent`.
        public unsafe new MR.Object? Parent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_parent", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_parent(_Underlying *_this);
            var __ret = __MR_Object_parent(_UnderlyingPtr);
            return __ret is not null ? new MR.Object(__ret, is_owning: false) : null;
        }

        /// Find a common ancestor between this object and the other one.
        /// Returns null on failure (which is impossible if both are children of the scene root).
        /// Will return `this` if `other` matches `this`.
        /// Generated from method `MR::Object::findCommonAncestor`.
        public unsafe MR.Object? FindCommonAncestor(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_findCommonAncestor", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_Object_findCommonAncestor(_Underlying *_this, MR.Object._Underlying *other);
            var __ret = __MR_Object_findCommonAncestor(_UnderlyingPtr, other._UnderlyingPtr);
            return __ret is not null ? new MR.Object(__ret, is_owning: false) : null;
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::Object::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_Object_detachFromParent(_Underlying *_this);
            return __MR_Object_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// an object can hold other sub-objects
        /// Generated from method `MR::Object::children`.
        public unsafe new MR.Std.Const_Vector_StdSharedPtrMRObject Children()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_children", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_StdSharedPtrMRObject._Underlying *__MR_Object_children(_Underlying *_this);
            return new(__MR_Object_children(_UnderlyingPtr), is_owning: false);
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::Object::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_addChild", ExactSpelling = true)]
            extern static byte __MR_Object_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_Object_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::Object::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_Object_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_Object_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// returns false if it was not child of this
        /// Generated from method `MR::Object::removeChild`.
        public unsafe bool RemoveChild(MR.Const_Object child)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_removeChild_std_shared_ptr_MR_Object", ExactSpelling = true)]
            extern static byte __MR_Object_removeChild_std_shared_ptr_MR_Object(_Underlying *_this, MR.Const_Object._UnderlyingShared *child);
            return __MR_Object_removeChild_std_shared_ptr_MR_Object(_UnderlyingPtr, child._UnderlyingSharedPtr) != 0;
        }

        /// Generated from method `MR::Object::removeChild`.
        public unsafe bool RemoveChild(MR.Object? child)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_removeChild_MR_Object_ptr", ExactSpelling = true)]
            extern static byte __MR_Object_removeChild_MR_Object_ptr(_Underlying *_this, MR.Object._Underlying *child);
            return __MR_Object_removeChild_MR_Object_ptr(_UnderlyingPtr, child is not null ? child._UnderlyingPtr : null) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::Object::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_Object_removeAllChildren(_Underlying *_this);
            __MR_Object_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::Object::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_sortChildren", ExactSpelling = true)]
            extern static void __MR_Object_sortChildren(_Underlying *_this);
            __MR_Object_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::Object::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_select", ExactSpelling = true)]
            extern static byte __MR_Object_select(_Underlying *_this, byte on);
            return __MR_Object_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::Object::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setAncillary", ExactSpelling = true)]
            extern static void __MR_Object_setAncillary(_Underlying *_this, byte ancillary);
            __MR_Object_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::Object::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setVisible", ExactSpelling = true)]
            extern static void __MR_Object_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_Object_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::Object::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_Object_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_Object_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::Object::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_swap", ExactSpelling = true)]
            extern static void __MR_Object_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_Object_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::Object::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_addTag", ExactSpelling = true)]
            extern static byte __MR_Object_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_Object_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::Object::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_removeTag", ExactSpelling = true)]
            extern static byte __MR_Object_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_Object_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `Object` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Object`/`Const_Object` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Object
    {
        internal readonly Const_Object? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Object() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Object(MR.Misc._Moved<Object> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Object(MR.Misc._Moved<Object> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Object` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Object`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Object`/`Const_Object` directly.
    public class _InOptMut_Object
    {
        public Object? Opt;

        public _InOptMut_Object() {}
        public _InOptMut_Object(Object value) {Opt = value;}
        public static implicit operator _InOptMut_Object(Object value) {return new(value);}
    }

    /// This is used for optional parameters of class `Object` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Object`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Object`/`Const_Object` to pass it to the function.
    public class _InOptConst_Object
    {
        public Const_Object? Opt;

        public _InOptConst_Object() {}
        public _InOptConst_Object(Const_Object value) {Opt = value;}
        public static implicit operator _InOptConst_Object(Const_Object value) {return new(value);}
    }
}
