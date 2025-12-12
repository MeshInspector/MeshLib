public static partial class MR
{
    // A base class for a data-model object that is a feature/measurement that can be compared between two models.
    /// Generated from class `MR::ObjectComparableWithReference`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::PointMeasurementObject`
    /// This is the const half of the class.
    public class Const_ObjectComparableWithReference : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectComparableWithReference_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectComparableWithReference_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectComparableWithReference_UseCount();
                return __MR_std_shared_ptr_MR_ObjectComparableWithReference_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectComparableWithReference(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectComparableWithReference_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectComparableWithReference_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectComparableWithReference(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectComparableWithReference _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectComparableWithReference_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectComparableWithReference_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectComparableWithReference_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectComparableWithReference_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectComparableWithReference_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectComparableWithReference_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectComparableWithReference() {Dispose(false);}

        // When comparing this object with a reference, how many different properties can we output?
        /// Generated from method `MR::ObjectComparableWithReference::numComparableProperties`.
        public unsafe ulong NumComparableProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_numComparableProperties", ExactSpelling = true)]
            extern static ulong __MR_ObjectComparableWithReference_numComparableProperties(_Underlying *_this);
            return __MR_ObjectComparableWithReference_numComparableProperties(_UnderlyingPtr);
        }

        // `i` goes up to `numComparableProperties()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::getComparablePropertyName`.
        public unsafe MR.Std.StringView GetComparablePropertyName(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_getComparablePropertyName", ExactSpelling = true)]
            extern static MR.Std.StringView._Underlying *__MR_ObjectComparableWithReference_getComparablePropertyName(_Underlying *_this, ulong i);
            return new(__MR_ObjectComparableWithReference_getComparablePropertyName(_UnderlyingPtr, i), is_owning: true);
        }

        // Compute a value of a property.
        // Compare `value` and `referenceValue` using the tolerance.
        // This can return null if the value is impossible to compute, e.g. for some types if the reference isn't set (e.g. if
        //   we're computing the distance to a reference point).
        // `i` goes up to `numComparableProperties()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::computeComparableProperty`.
        public unsafe MR.Std.Optional_MRObjectComparableWithReferenceComparableProperty ComputeComparableProperty(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_computeComparableProperty", ExactSpelling = true)]
            extern static MR.Std.Optional_MRObjectComparableWithReferenceComparableProperty._Underlying *__MR_ObjectComparableWithReference_computeComparableProperty(_Underlying *_this, ulong i);
            return new(__MR_ObjectComparableWithReference_computeComparableProperty(_UnderlyingPtr, i), is_owning: true);
        }

        // Returns the tolerance for a specific comparable property. Returns null if not set.
        // `i` goes up to `numComparableProperties()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::getComparisonTolerence`.
        public unsafe MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance GetComparisonTolerence(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_getComparisonTolerence", ExactSpelling = true)]
            extern static MR.Std.Optional_MRObjectComparableWithReferenceComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_getComparisonTolerence(_Underlying *_this, ulong i);
            return new(__MR_ObjectComparableWithReference_getComparisonTolerence(_UnderlyingPtr, i), is_owning: true);
        }

        // If true, indicates that the getter will always return zero negative tolerance, and the setter will ignore the negative tolerance.
        // `i` goes up to `numComparableProperties()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::comparisonToleranceIsAlwaysOnlyPositive`.
        public unsafe bool ComparisonToleranceIsAlwaysOnlyPositive(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_comparisonToleranceIsAlwaysOnlyPositive", ExactSpelling = true)]
            extern static byte __MR_ObjectComparableWithReference_comparisonToleranceIsAlwaysOnlyPositive(_Underlying *_this, ulong i);
            return __MR_ObjectComparableWithReference_comparisonToleranceIsAlwaysOnlyPositive(_UnderlyingPtr, i) != 0;
        }

        // The number and types of reference values can be entirely different compared to `numComparableProperties()`.
        /// Generated from method `MR::ObjectComparableWithReference::numComparisonReferenceValues`.
        public unsafe ulong NumComparisonReferenceValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_numComparisonReferenceValues", ExactSpelling = true)]
            extern static ulong __MR_ObjectComparableWithReference_numComparisonReferenceValues(_Underlying *_this);
            return __MR_ObjectComparableWithReference_numComparisonReferenceValues(_UnderlyingPtr);
        }

        // `i` goes up to `numComparisonReferenceValues()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::getComparisonReferenceValueName`.
        public unsafe MR.Std.StringView GetComparisonReferenceValueName(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_getComparisonReferenceValueName", ExactSpelling = true)]
            extern static MR.Std.StringView._Underlying *__MR_ObjectComparableWithReference_getComparisonReferenceValueName(_Underlying *_this, ulong i);
            return new(__MR_ObjectComparableWithReference_getComparisonReferenceValueName(_UnderlyingPtr, i), is_owning: true);
        }

        // Returns the internal reference value.
        // If the value wasn't set yet (as indicated by `isSet == false`), you can still use the returned variant to get the expected type.
        // `i` goes up to `numComparisonReferenceValues()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::getComparisonReferenceValue`.
        public unsafe MR.ObjectComparableWithReference.ComparisonReferenceValue GetComparisonReferenceValue(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_getComparisonReferenceValue", ExactSpelling = true)]
            extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_getComparisonReferenceValue(_Underlying *_this, ulong i);
            return new(__MR_ObjectComparableWithReference_getComparisonReferenceValue(_UnderlyingPtr, i), is_owning: true);
        }

        /// Generated from class `MR::ObjectComparableWithReference::ComparableProperty`.
        /// This is the const half of the class.
        public class Const_ComparableProperty : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ComparableProperty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_Destroy", ExactSpelling = true)]
                extern static void __MR_ObjectComparableWithReference_ComparableProperty_Destroy(_Underlying *_this);
                __MR_ObjectComparableWithReference_ComparableProperty_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ComparableProperty() {Dispose(false);}

            public unsafe float Value
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_Get_value", ExactSpelling = true)]
                    extern static float *__MR_ObjectComparableWithReference_ComparableProperty_Get_value(_Underlying *_this);
                    return *__MR_ObjectComparableWithReference_ComparableProperty_Get_value(_UnderlyingPtr);
                }
            }

            // This can be null if the reference value isn't set, or something else is wrong.
            // This can match whatever is set via `get/setComparisonReferenceValue()`, but not necessarily.
            // E.g. for point coordinates, those functions act on the reference coordinates (three optional floats), but this number is always zero,
            //   and the `value` is the distance to those coordinates.
            public unsafe MR.Std.Const_Optional_Float ReferenceValue
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_Get_referenceValue", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_Float._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_Get_referenceValue(_Underlying *_this);
                    return new(__MR_ObjectComparableWithReference_ComparableProperty_Get_referenceValue(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ComparableProperty() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparableProperty_DefaultConstruct();
            }

            /// Constructs `MR::ObjectComparableWithReference::ComparableProperty` elementwise.
            public unsafe Const_ComparableProperty(float value, float? referenceValue) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_ConstructFrom(float value, float *referenceValue);
                float __deref_referenceValue = referenceValue.GetValueOrDefault();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparableProperty_ConstructFrom(value, referenceValue.HasValue ? &__deref_referenceValue : null);
            }

            /// Generated from constructor `MR::ObjectComparableWithReference::ComparableProperty::ComparableProperty`.
            public unsafe Const_ComparableProperty(MR.ObjectComparableWithReference.Const_ComparableProperty _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_ConstructFromAnother(MR.ObjectComparableWithReference.ComparableProperty._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparableProperty_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::ObjectComparableWithReference::ComparableProperty`.
        /// This is the non-const half of the class.
        public class ComparableProperty : Const_ComparableProperty
        {
            internal unsafe ComparableProperty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe ref float Value
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_GetMutable_value", ExactSpelling = true)]
                    extern static float *__MR_ObjectComparableWithReference_ComparableProperty_GetMutable_value(_Underlying *_this);
                    return ref *__MR_ObjectComparableWithReference_ComparableProperty_GetMutable_value(_UnderlyingPtr);
                }
            }

            // This can be null if the reference value isn't set, or something else is wrong.
            // This can match whatever is set via `get/setComparisonReferenceValue()`, but not necessarily.
            // E.g. for point coordinates, those functions act on the reference coordinates (three optional floats), but this number is always zero,
            //   and the `value` is the distance to those coordinates.
            public new unsafe MR.Std.Optional_Float ReferenceValue
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_GetMutable_referenceValue", ExactSpelling = true)]
                    extern static MR.Std.Optional_Float._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_GetMutable_referenceValue(_Underlying *_this);
                    return new(__MR_ObjectComparableWithReference_ComparableProperty_GetMutable_referenceValue(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ComparableProperty() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparableProperty_DefaultConstruct();
            }

            /// Constructs `MR::ObjectComparableWithReference::ComparableProperty` elementwise.
            public unsafe ComparableProperty(float value, float? referenceValue) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_ConstructFrom(float value, float *referenceValue);
                float __deref_referenceValue = referenceValue.GetValueOrDefault();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparableProperty_ConstructFrom(value, referenceValue.HasValue ? &__deref_referenceValue : null);
            }

            /// Generated from constructor `MR::ObjectComparableWithReference::ComparableProperty::ComparableProperty`.
            public unsafe ComparableProperty(MR.ObjectComparableWithReference.Const_ComparableProperty _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_ConstructFromAnother(MR.ObjectComparableWithReference.ComparableProperty._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparableProperty_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::ObjectComparableWithReference::ComparableProperty::operator=`.
            public unsafe MR.ObjectComparableWithReference.ComparableProperty Assign(MR.ObjectComparableWithReference.Const_ComparableProperty _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparableProperty_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparableProperty._Underlying *__MR_ObjectComparableWithReference_ComparableProperty_AssignFromAnother(_Underlying *_this, MR.ObjectComparableWithReference.ComparableProperty._Underlying *_other);
                return new(__MR_ObjectComparableWithReference_ComparableProperty_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ComparableProperty` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ComparableProperty`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ComparableProperty`/`Const_ComparableProperty` directly.
        public class _InOptMut_ComparableProperty
        {
            public ComparableProperty? Opt;

            public _InOptMut_ComparableProperty() {}
            public _InOptMut_ComparableProperty(ComparableProperty value) {Opt = value;}
            public static implicit operator _InOptMut_ComparableProperty(ComparableProperty value) {return new(value);}
        }

        /// This is used for optional parameters of class `ComparableProperty` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ComparableProperty`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ComparableProperty`/`Const_ComparableProperty` to pass it to the function.
        public class _InOptConst_ComparableProperty
        {
            public Const_ComparableProperty? Opt;

            public _InOptConst_ComparableProperty() {}
            public _InOptConst_ComparableProperty(Const_ComparableProperty value) {Opt = value;}
            public static implicit operator _InOptConst_ComparableProperty(Const_ComparableProperty value) {return new(value);}
        }

        // This can't be `std::optional<Var>`, because we still need the variant to know the correct type.
        /// Generated from class `MR::ObjectComparableWithReference::ComparisonReferenceValue`.
        /// This is the const half of the class.
        public class Const_ComparisonReferenceValue : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ComparisonReferenceValue(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_Destroy", ExactSpelling = true)]
                extern static void __MR_ObjectComparableWithReference_ComparisonReferenceValue_Destroy(_Underlying *_this);
                __MR_ObjectComparableWithReference_ComparisonReferenceValue_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ComparisonReferenceValue() {Dispose(false);}

            public unsafe bool IsSet
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_Get_isSet", ExactSpelling = true)]
                    extern static bool *__MR_ObjectComparableWithReference_ComparisonReferenceValue_Get_isSet(_Underlying *_this);
                    return *__MR_ObjectComparableWithReference_ComparisonReferenceValue_Get_isSet(_UnderlyingPtr);
                }
            }

            // If `isSet == false`, this will hold zeroes, or some other default values.
            public unsafe MR.Std.Const_Variant_Float_MRVector3f Var
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_Get_var", ExactSpelling = true)]
                    extern static MR.Std.Const_Variant_Float_MRVector3f._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_Get_var(_Underlying *_this);
                    return new(__MR_ObjectComparableWithReference_ComparisonReferenceValue_Get_var(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ComparisonReferenceValue() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonReferenceValue_DefaultConstruct();
            }

            /// Constructs `MR::ObjectComparableWithReference::ComparisonReferenceValue` elementwise.
            public unsafe Const_ComparisonReferenceValue(bool isSet, MR.Std.Const_Variant_Float_MRVector3f var) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFrom(byte isSet, MR.Std.Variant_Float_MRVector3f._Underlying *var);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFrom(isSet ? (byte)1 : (byte)0, var._UnderlyingPtr);
            }

            /// Generated from constructor `MR::ObjectComparableWithReference::ComparisonReferenceValue::ComparisonReferenceValue`.
            public unsafe Const_ComparisonReferenceValue(MR.ObjectComparableWithReference.Const_ComparisonReferenceValue _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFromAnother(MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        // This can't be `std::optional<Var>`, because we still need the variant to know the correct type.
        /// Generated from class `MR::ObjectComparableWithReference::ComparisonReferenceValue`.
        /// This is the non-const half of the class.
        public class ComparisonReferenceValue : Const_ComparisonReferenceValue
        {
            internal unsafe ComparisonReferenceValue(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe ref bool IsSet
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_GetMutable_isSet", ExactSpelling = true)]
                    extern static bool *__MR_ObjectComparableWithReference_ComparisonReferenceValue_GetMutable_isSet(_Underlying *_this);
                    return ref *__MR_ObjectComparableWithReference_ComparisonReferenceValue_GetMutable_isSet(_UnderlyingPtr);
                }
            }

            // If `isSet == false`, this will hold zeroes, or some other default values.
            public new unsafe MR.Std.Variant_Float_MRVector3f Var
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_GetMutable_var", ExactSpelling = true)]
                    extern static MR.Std.Variant_Float_MRVector3f._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_GetMutable_var(_Underlying *_this);
                    return new(__MR_ObjectComparableWithReference_ComparisonReferenceValue_GetMutable_var(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ComparisonReferenceValue() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonReferenceValue_DefaultConstruct();
            }

            /// Constructs `MR::ObjectComparableWithReference::ComparisonReferenceValue` elementwise.
            public unsafe ComparisonReferenceValue(bool isSet, MR.Std.Const_Variant_Float_MRVector3f var) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFrom(byte isSet, MR.Std.Variant_Float_MRVector3f._Underlying *var);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFrom(isSet ? (byte)1 : (byte)0, var._UnderlyingPtr);
            }

            /// Generated from constructor `MR::ObjectComparableWithReference::ComparisonReferenceValue::ComparisonReferenceValue`.
            public unsafe ComparisonReferenceValue(MR.ObjectComparableWithReference.Const_ComparisonReferenceValue _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFromAnother(MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonReferenceValue_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::ObjectComparableWithReference::ComparisonReferenceValue::operator=`.
            public unsafe MR.ObjectComparableWithReference.ComparisonReferenceValue Assign(MR.ObjectComparableWithReference.Const_ComparisonReferenceValue _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonReferenceValue_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *__MR_ObjectComparableWithReference_ComparisonReferenceValue_AssignFromAnother(_Underlying *_this, MR.ObjectComparableWithReference.ComparisonReferenceValue._Underlying *_other);
                return new(__MR_ObjectComparableWithReference_ComparisonReferenceValue_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ComparisonReferenceValue` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ComparisonReferenceValue`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ComparisonReferenceValue`/`Const_ComparisonReferenceValue` directly.
        public class _InOptMut_ComparisonReferenceValue
        {
            public ComparisonReferenceValue? Opt;

            public _InOptMut_ComparisonReferenceValue() {}
            public _InOptMut_ComparisonReferenceValue(ComparisonReferenceValue value) {Opt = value;}
            public static implicit operator _InOptMut_ComparisonReferenceValue(ComparisonReferenceValue value) {return new(value);}
        }

        /// This is used for optional parameters of class `ComparisonReferenceValue` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ComparisonReferenceValue`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ComparisonReferenceValue`/`Const_ComparisonReferenceValue` to pass it to the function.
        public class _InOptConst_ComparisonReferenceValue
        {
            public Const_ComparisonReferenceValue? Opt;

            public _InOptConst_ComparisonReferenceValue() {}
            public _InOptConst_ComparisonReferenceValue(Const_ComparisonReferenceValue value) {Opt = value;}
            public static implicit operator _InOptConst_ComparisonReferenceValue(Const_ComparisonReferenceValue value) {return new(value);}
        }

        // Tolerances:
        /// Generated from class `MR::ObjectComparableWithReference::ComparisonTolerance`.
        /// This is the const half of the class.
        public class Const_ComparisonTolerance : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ComparisonTolerance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_Destroy", ExactSpelling = true)]
                extern static void __MR_ObjectComparableWithReference_ComparisonTolerance_Destroy(_Underlying *_this);
                __MR_ObjectComparableWithReference_ComparisonTolerance_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ComparisonTolerance() {Dispose(false);}

            // How much larger can this value be compared to the reference?
            public unsafe float Positive
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_Get_positive", ExactSpelling = true)]
                    extern static float *__MR_ObjectComparableWithReference_ComparisonTolerance_Get_positive(_Underlying *_this);
                    return *__MR_ObjectComparableWithReference_ComparisonTolerance_Get_positive(_UnderlyingPtr);
                }
            }

            // How much smaller can this value be compared to the reference?
            // This number should normally be zero or negative.
            public unsafe float Negative
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_Get_negative", ExactSpelling = true)]
                    extern static float *__MR_ObjectComparableWithReference_ComparisonTolerance_Get_negative(_Underlying *_this);
                    return *__MR_ObjectComparableWithReference_ComparisonTolerance_Get_negative(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ComparisonTolerance() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
            }

            /// Constructs `MR::ObjectComparableWithReference::ComparisonTolerance` elementwise.
            public unsafe Const_ComparisonTolerance(float positive, float negative) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(float positive, float negative);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(positive, negative);
            }

            /// Generated from constructor `MR::ObjectComparableWithReference::ComparisonTolerance::ComparisonTolerance`.
            public unsafe Const_ComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        // Tolerances:
        /// Generated from class `MR::ObjectComparableWithReference::ComparisonTolerance`.
        /// This is the non-const half of the class.
        public class ComparisonTolerance : Const_ComparisonTolerance
        {
            internal unsafe ComparisonTolerance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // How much larger can this value be compared to the reference?
            public new unsafe ref float Positive
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_GetMutable_positive", ExactSpelling = true)]
                    extern static float *__MR_ObjectComparableWithReference_ComparisonTolerance_GetMutable_positive(_Underlying *_this);
                    return ref *__MR_ObjectComparableWithReference_ComparisonTolerance_GetMutable_positive(_UnderlyingPtr);
                }
            }

            // How much smaller can this value be compared to the reference?
            // This number should normally be zero or negative.
            public new unsafe ref float Negative
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_GetMutable_negative", ExactSpelling = true)]
                    extern static float *__MR_ObjectComparableWithReference_ComparisonTolerance_GetMutable_negative(_Underlying *_this);
                    return ref *__MR_ObjectComparableWithReference_ComparisonTolerance_GetMutable_negative(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ComparisonTolerance() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonTolerance_DefaultConstruct();
            }

            /// Constructs `MR::ObjectComparableWithReference::ComparisonTolerance` elementwise.
            public unsafe ComparisonTolerance(float positive, float negative) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(float positive, float negative);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFrom(positive, negative);
            }

            /// Generated from constructor `MR::ObjectComparableWithReference::ComparisonTolerance::ComparisonTolerance`.
            public unsafe ComparisonTolerance(MR.ObjectComparableWithReference.Const_ComparisonTolerance _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectComparableWithReference_ComparisonTolerance_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::ObjectComparableWithReference::ComparisonTolerance::operator=`.
            public unsafe MR.ObjectComparableWithReference.ComparisonTolerance Assign(MR.ObjectComparableWithReference.Const_ComparisonTolerance _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_ComparisonTolerance_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *__MR_ObjectComparableWithReference_ComparisonTolerance_AssignFromAnother(_Underlying *_this, MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *_other);
                return new(__MR_ObjectComparableWithReference_ComparisonTolerance_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `ComparisonTolerance` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ComparisonTolerance`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ComparisonTolerance`/`Const_ComparisonTolerance` directly.
        public class _InOptMut_ComparisonTolerance
        {
            public ComparisonTolerance? Opt;

            public _InOptMut_ComparisonTolerance() {}
            public _InOptMut_ComparisonTolerance(ComparisonTolerance value) {Opt = value;}
            public static implicit operator _InOptMut_ComparisonTolerance(ComparisonTolerance value) {return new(value);}
        }

        /// This is used for optional parameters of class `ComparisonTolerance` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ComparisonTolerance`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ComparisonTolerance`/`Const_ComparisonTolerance` to pass it to the function.
        public class _InOptConst_ComparisonTolerance
        {
            public Const_ComparisonTolerance? Opt;

            public _InOptConst_ComparisonTolerance() {}
            public _InOptConst_ComparisonTolerance(Const_ComparisonTolerance value) {Opt = value;}
            public static implicit operator _InOptConst_ComparisonTolerance(Const_ComparisonTolerance value) {return new(value);}
        }
    }

    // A base class for a data-model object that is a feature/measurement that can be compared between two models.
    /// Generated from class `MR::ObjectComparableWithReference`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::PointMeasurementObject`
    /// This is the non-const half of the class.
    public class ObjectComparableWithReference : Const_ObjectComparableWithReference
    {
        internal unsafe ObjectComparableWithReference(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectComparableWithReference(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Sets the tolerance for a specific comparable property.
        // `i` goes up to `numComparableProperties()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::setComparisonTolerance`.
        public unsafe void SetComparisonTolerance(ulong i, MR.ObjectComparableWithReference.Const_ComparisonTolerance? newTolerance)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_setComparisonTolerance", ExactSpelling = true)]
            extern static void __MR_ObjectComparableWithReference_setComparisonTolerance(_Underlying *_this, ulong i, MR.ObjectComparableWithReference.ComparisonTolerance._Underlying *newTolerance);
            __MR_ObjectComparableWithReference_setComparisonTolerance(_UnderlyingPtr, i, newTolerance is not null ? newTolerance._UnderlyingPtr : null);
        }

        // Sets the internal reference value. Makes `hasComparisonReferenceValue()` return true.
        // If you pass nullopt, removes this reference value.
        // Only a certain variant type is legal to pass, depending on the derived class and the index. Use `getComparisonReferenceValue()` to determine that type.
        // `i` goes up to `numComparisonReferenceValues()`, exclusive.
        /// Generated from method `MR::ObjectComparableWithReference::setComparisonReferenceValue`.
        public unsafe void SetComparisonReferenceValue(ulong i, MR.Std.Const_Variant_Float_MRVector3f? value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_setComparisonReferenceValue", ExactSpelling = true)]
            extern static void __MR_ObjectComparableWithReference_setComparisonReferenceValue(_Underlying *_this, ulong i, MR.Std.Variant_Float_MRVector3f._Underlying *value);
            __MR_ObjectComparableWithReference_setComparisonReferenceValue(_UnderlyingPtr, i, value is not null ? value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectComparableWithReference::setComparisonReferenceVal`.
        public unsafe void SetComparisonReferenceVal(ulong i, MR.ObjectComparableWithReference.Const_ComparisonReferenceValue value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectComparableWithReference_setComparisonReferenceVal", ExactSpelling = true)]
            extern static void __MR_ObjectComparableWithReference_setComparisonReferenceVal(_Underlying *_this, ulong i, MR.ObjectComparableWithReference.Const_ComparisonReferenceValue._Underlying *value);
            __MR_ObjectComparableWithReference_setComparisonReferenceVal(_UnderlyingPtr, i, value._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectComparableWithReference` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectComparableWithReference`/`Const_ObjectComparableWithReference` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectComparableWithReference
    {
        internal readonly Const_ObjectComparableWithReference? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `ObjectComparableWithReference` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectComparableWithReference`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectComparableWithReference`/`Const_ObjectComparableWithReference` directly.
    public class _InOptMut_ObjectComparableWithReference
    {
        public ObjectComparableWithReference? Opt;

        public _InOptMut_ObjectComparableWithReference() {}
        public _InOptMut_ObjectComparableWithReference(ObjectComparableWithReference value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectComparableWithReference(ObjectComparableWithReference value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectComparableWithReference` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectComparableWithReference`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectComparableWithReference`/`Const_ObjectComparableWithReference` to pass it to the function.
    public class _InOptConst_ObjectComparableWithReference
    {
        public Const_ObjectComparableWithReference? Opt;

        public _InOptConst_ObjectComparableWithReference() {}
        public _InOptConst_ObjectComparableWithReference(Const_ObjectComparableWithReference value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectComparableWithReference(Const_ObjectComparableWithReference value) {return new(value);}
    }
}
