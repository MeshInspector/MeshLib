public static partial class MR
{
    // Note! Must use `MRMESH_CLASS` on this enum and all enums that extend this one,
    // otherwise you'll get silent wrong behavior on Mac.
    public enum VisualizeMaskType : int
    {
        Visibility = 0,
        InvertedNormals = 1,
        Name = 2,
        ClippedByPlane = 3,
        DepthTest = 4,
        Count = 5,
    }

    // If a type derived from `VisualObject` wants to extend `VisualizeMaskType`, it must create a separate enum and specialize this to `true` for it.
    // NOTE! All those enums can start from 0, don't worry about collisions.
    /// Generated from class `MR::IsVisualizeMaskEnum<MR::AnyVisualizeMaskEnum>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *__MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::AnyVisualizeMaskEnum>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(MR.Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *__MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // If a type derived from `VisualObject` wants to extend `VisualizeMaskType`, it must create a separate enum and specialize this to `true` for it.
    // NOTE! All those enums can start from 0, don't worry about collisions.
    /// Generated from class `MR::IsVisualizeMaskEnum<MR::AnyVisualizeMaskEnum>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum : Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum
    {
        internal unsafe IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *__MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::AnyVisualizeMaskEnum>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(MR.Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *__MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::AnyVisualizeMaskEnum>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum Assign(MR.Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *__MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_AnyVisualizeMaskEnum_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum`/`Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum
    {
        public IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum() {}
        public _InOptMut_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum`/`Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum
    {
        public Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum() {}
        public _InOptConst_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum(Const_IsVisualizeMaskEnum_MRAnyVisualizeMaskEnum value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::VisualizeMaskType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRVisualizeMaskType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRVisualizeMaskType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRVisualizeMaskType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRVisualizeMaskType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *__MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::VisualizeMaskType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRVisualizeMaskType(MR.Const_IsVisualizeMaskEnum_MRVisualizeMaskType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *__MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::VisualizeMaskType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRVisualizeMaskType : Const_IsVisualizeMaskEnum_MRVisualizeMaskType
    {
        internal unsafe IsVisualizeMaskEnum_MRVisualizeMaskType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRVisualizeMaskType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *__MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::VisualizeMaskType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRVisualizeMaskType(MR.Const_IsVisualizeMaskEnum_MRVisualizeMaskType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *__MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::VisualizeMaskType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRVisualizeMaskType Assign(MR.Const_IsVisualizeMaskEnum_MRVisualizeMaskType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *__MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRVisualizeMaskType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_VisualizeMaskType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRVisualizeMaskType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRVisualizeMaskType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRVisualizeMaskType`/`Const_IsVisualizeMaskEnum_MRVisualizeMaskType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRVisualizeMaskType
    {
        public IsVisualizeMaskEnum_MRVisualizeMaskType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRVisualizeMaskType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRVisualizeMaskType(IsVisualizeMaskEnum_MRVisualizeMaskType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRVisualizeMaskType(IsVisualizeMaskEnum_MRVisualizeMaskType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRVisualizeMaskType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRVisualizeMaskType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRVisualizeMaskType`/`Const_IsVisualizeMaskEnum_MRVisualizeMaskType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRVisualizeMaskType
    {
        public Const_IsVisualizeMaskEnum_MRVisualizeMaskType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRVisualizeMaskType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRVisualizeMaskType(Const_IsVisualizeMaskEnum_MRVisualizeMaskType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRVisualizeMaskType(Const_IsVisualizeMaskEnum_MRVisualizeMaskType value) {return new(value);}
    }

    // Stores a `VisualizeMaskType` or any other enum that extends it (i.e. which specializes `IsVisualizeMaskEnum`).
    // To extract the value, do this:
    //     if ( auto value = x.tryGet<MyEnum>() )
    //     {
    //         switch ( *value )
    //         {
    //             case MyEnum::foo: ...
    //             case MyEnum::bar: ...
    //         }
    //     }
    //     else // forward to the parent class
    /// Generated from class `MR::AnyVisualizeMaskEnum`.
    /// This is the const half of the class.
    public class Const_AnyVisualizeMaskEnum : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_AnyVisualizeMaskEnum(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Destroy", ExactSpelling = true)]
            extern static void __MR_AnyVisualizeMaskEnum_Destroy(_Underlying *_this);
            __MR_AnyVisualizeMaskEnum_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_AnyVisualizeMaskEnum() {Dispose(false);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe Const_AnyVisualizeMaskEnum(MR.Const_AnyVisualizeMaskEnum _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_ConstructFromAnother(MR.AnyVisualizeMaskEnum._Underlying *_other);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe Const_AnyVisualizeMaskEnum(MR.VisualizeMaskType value) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Construct_MR_VisualizeMaskType", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_Construct_MR_VisualizeMaskType(MR.VisualizeMaskType value);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_Construct_MR_VisualizeMaskType(value);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator Const_AnyVisualizeMaskEnum(MR.VisualizeMaskType value) {return new(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe Const_AnyVisualizeMaskEnum(MR.MeshVisualizePropertyType value) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Construct_MR_MeshVisualizePropertyType", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_Construct_MR_MeshVisualizePropertyType(MR.MeshVisualizePropertyType value);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_Construct_MR_MeshVisualizePropertyType(value);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator Const_AnyVisualizeMaskEnum(MR.MeshVisualizePropertyType value) {return new(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe Const_AnyVisualizeMaskEnum(MR.DimensionsVisualizePropertyType value) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Construct_MR_DimensionsVisualizePropertyType", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_Construct_MR_DimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType value);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_Construct_MR_DimensionsVisualizePropertyType(value);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator Const_AnyVisualizeMaskEnum(MR.DimensionsVisualizePropertyType value) {return new(value);}

        /// Generated from method `MR::AnyVisualizeMaskEnum::tryGet<MR::DimensionsVisualizePropertyType>`.
        public unsafe MR.Std.Optional_MRDimensionsVisualizePropertyType TryGet()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_tryGet", ExactSpelling = true)]
            extern static MR.Std.Optional_MRDimensionsVisualizePropertyType._Underlying *__MR_AnyVisualizeMaskEnum_tryGet(_Underlying *_this);
            return new(__MR_AnyVisualizeMaskEnum_tryGet(_UnderlyingPtr), is_owning: true);
        }
    }

    // Stores a `VisualizeMaskType` or any other enum that extends it (i.e. which specializes `IsVisualizeMaskEnum`).
    // To extract the value, do this:
    //     if ( auto value = x.tryGet<MyEnum>() )
    //     {
    //         switch ( *value )
    //         {
    //             case MyEnum::foo: ...
    //             case MyEnum::bar: ...
    //         }
    //     }
    //     else // forward to the parent class
    /// Generated from class `MR::AnyVisualizeMaskEnum`.
    /// This is the non-const half of the class.
    public class AnyVisualizeMaskEnum : Const_AnyVisualizeMaskEnum
    {
        internal unsafe AnyVisualizeMaskEnum(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe AnyVisualizeMaskEnum(MR.Const_AnyVisualizeMaskEnum _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_ConstructFromAnother(MR.AnyVisualizeMaskEnum._Underlying *_other);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe AnyVisualizeMaskEnum(MR.VisualizeMaskType value) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Construct_MR_VisualizeMaskType", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_Construct_MR_VisualizeMaskType(MR.VisualizeMaskType value);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_Construct_MR_VisualizeMaskType(value);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator AnyVisualizeMaskEnum(MR.VisualizeMaskType value) {return new(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe AnyVisualizeMaskEnum(MR.MeshVisualizePropertyType value) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Construct_MR_MeshVisualizePropertyType", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_Construct_MR_MeshVisualizePropertyType(MR.MeshVisualizePropertyType value);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_Construct_MR_MeshVisualizePropertyType(value);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator AnyVisualizeMaskEnum(MR.MeshVisualizePropertyType value) {return new(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public unsafe AnyVisualizeMaskEnum(MR.DimensionsVisualizePropertyType value) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_Construct_MR_DimensionsVisualizePropertyType", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_Construct_MR_DimensionsVisualizePropertyType(MR.DimensionsVisualizePropertyType value);
            _UnderlyingPtr = __MR_AnyVisualizeMaskEnum_Construct_MR_DimensionsVisualizePropertyType(value);
        }

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator AnyVisualizeMaskEnum(MR.DimensionsVisualizePropertyType value) {return new(value);}

        /// Generated from method `MR::AnyVisualizeMaskEnum::operator=`.
        public unsafe MR.AnyVisualizeMaskEnum Assign(MR.Const_AnyVisualizeMaskEnum _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_AnyVisualizeMaskEnum_AssignFromAnother", ExactSpelling = true)]
            extern static MR.AnyVisualizeMaskEnum._Underlying *__MR_AnyVisualizeMaskEnum_AssignFromAnother(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *_other);
            return new(__MR_AnyVisualizeMaskEnum_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `AnyVisualizeMaskEnum` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_AnyVisualizeMaskEnum`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AnyVisualizeMaskEnum`/`Const_AnyVisualizeMaskEnum` directly.
    public class _InOptMut_AnyVisualizeMaskEnum
    {
        public AnyVisualizeMaskEnum? Opt;

        public _InOptMut_AnyVisualizeMaskEnum() {}
        public _InOptMut_AnyVisualizeMaskEnum(AnyVisualizeMaskEnum value) {Opt = value;}
        public static implicit operator _InOptMut_AnyVisualizeMaskEnum(AnyVisualizeMaskEnum value) {return new(value);}
    }

    /// This is used for optional parameters of class `AnyVisualizeMaskEnum` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_AnyVisualizeMaskEnum`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `AnyVisualizeMaskEnum`/`Const_AnyVisualizeMaskEnum` to pass it to the function.
    public class _InOptConst_AnyVisualizeMaskEnum
    {
        public Const_AnyVisualizeMaskEnum? Opt;

        public _InOptConst_AnyVisualizeMaskEnum() {}
        public _InOptConst_AnyVisualizeMaskEnum(Const_AnyVisualizeMaskEnum value) {Opt = value;}
        public static implicit operator _InOptConst_AnyVisualizeMaskEnum(Const_AnyVisualizeMaskEnum value) {return new(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator _InOptConst_AnyVisualizeMaskEnum(MR.VisualizeMaskType value) {return new MR.AnyVisualizeMaskEnum(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator _InOptConst_AnyVisualizeMaskEnum(MR.MeshVisualizePropertyType value) {return new MR.AnyVisualizeMaskEnum(value);}

        /// Generated from constructor `MR::AnyVisualizeMaskEnum::AnyVisualizeMaskEnum`.
        public static unsafe implicit operator _InOptConst_AnyVisualizeMaskEnum(MR.DimensionsVisualizePropertyType value) {return new MR.AnyVisualizeMaskEnum(value);}
    }

    public enum DirtyFlags : int
    {
        DIRTYNONE = 0,
        DIRTYPOSITION = 1,
        DIRTYUV = 2,
        //< gl normals
        DIRTYVERTSRENDERNORMAL = 4,
        ///< gl normals
        DIRTYFACESRENDERNORMAL = 8,
        ///< gl normals
        DIRTYCORNERSRENDERNORMAL = 16,
        DIRTYRENDERNORMALS = 28,
        DIRTYSELECTION = 32,
        DIRTYTEXTURE = 64,
        DIRTYPRIMITIVES = 128,
        DIRTYFACE = 128,
        DIRTYVERTSCOLORMAP = 256,
        DIRTYPRIMITIVECOLORMAP = 512,
        DIRTYFACESCOLORMAP = 512,
        DIRTYTEXTUREPERFACE = 1024,
        DIRTYMESH = 2047,
        DIRTYBOUNDINGBOX = 2048,
        DIRTYBORDERLINES = 4096,
        DIRTYEDGESSELECTION = 8192,
        DIRTYCACHES = 2048,
        DIRTYVOLUME = 16384,
        DIRTYALL = 32767,
    }

    /// Marks dirty buffers that need to be uploaded to OpenGL.
    /// Dirty flags must be moved together with renderObj_,
    /// but not copied since renderObj_ is not copied as well
    /// Generated from class `MR::Dirty`.
    /// This is the const half of the class.
    public class Const_Dirty : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Dirty(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_Destroy", ExactSpelling = true)]
            extern static void __MR_Dirty_Destroy(_Underlying *_this);
            __MR_Dirty_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Dirty() {Dispose(false);}

        public unsafe uint F
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_Get_f", ExactSpelling = true)]
                extern static uint *__MR_Dirty_Get_f(_Underlying *_this);
                return *__MR_Dirty_Get_f(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Dirty() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Dirty._Underlying *__MR_Dirty_DefaultConstruct();
            _UnderlyingPtr = __MR_Dirty_DefaultConstruct();
        }

        /// Generated from constructor `MR::Dirty::Dirty`.
        public unsafe Const_Dirty(MR.Const_Dirty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Dirty._Underlying *__MR_Dirty_ConstructFromAnother(MR.Dirty._Underlying *_other);
            _UnderlyingPtr = __MR_Dirty_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::Dirty::operator unsigned int`.
        public static unsafe implicit operator uint(MR.Const_Dirty _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_ConvertTo_unsigned_int", ExactSpelling = true)]
            extern static uint __MR_Dirty_ConvertTo_unsigned_int(MR.Const_Dirty._Underlying *_this);
            return __MR_Dirty_ConvertTo_unsigned_int(_this._UnderlyingPtr);
        }
    }

    /// Marks dirty buffers that need to be uploaded to OpenGL.
    /// Dirty flags must be moved together with renderObj_,
    /// but not copied since renderObj_ is not copied as well
    /// Generated from class `MR::Dirty`.
    /// This is the non-const half of the class.
    public class Dirty : Const_Dirty
    {
        internal unsafe Dirty(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref uint F
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_GetMutable_f", ExactSpelling = true)]
                extern static uint *__MR_Dirty_GetMutable_f(_Underlying *_this);
                return ref *__MR_Dirty_GetMutable_f(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Dirty() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Dirty._Underlying *__MR_Dirty_DefaultConstruct();
            _UnderlyingPtr = __MR_Dirty_DefaultConstruct();
        }

        /// Generated from constructor `MR::Dirty::Dirty`.
        public unsafe Dirty(MR.Const_Dirty _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Dirty._Underlying *__MR_Dirty_ConstructFromAnother(MR.Dirty._Underlying *_other);
            _UnderlyingPtr = __MR_Dirty_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::Dirty::operator unsigned int &`.
        public unsafe ref uint ConvertTo_UnsignedIntRef()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_ConvertTo_unsigned_int_ref", ExactSpelling = true)]
            extern static uint *__MR_Dirty_ConvertTo_unsigned_int_ref(_Underlying *_this);
            return ref *__MR_Dirty_ConvertTo_unsigned_int_ref(_UnderlyingPtr);
        }

        /// Generated from method `MR::Dirty::operator=`.
        public unsafe MR.Dirty Assign(MR.Const_Dirty _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Dirty._Underlying *__MR_Dirty_AssignFromAnother(_Underlying *_this, MR.Dirty._Underlying *_other);
            return new(__MR_Dirty_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Dirty::operator=`.
        public unsafe MR.Dirty Assign(uint b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dirty_assign", ExactSpelling = true)]
            extern static MR.Dirty._Underlying *__MR_Dirty_assign(_Underlying *_this, uint b);
            return new(__MR_Dirty_assign(_UnderlyingPtr, b), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Dirty` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Dirty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Dirty`/`Const_Dirty` directly.
    public class _InOptMut_Dirty
    {
        public Dirty? Opt;

        public _InOptMut_Dirty() {}
        public _InOptMut_Dirty(Dirty value) {Opt = value;}
        public static implicit operator _InOptMut_Dirty(Dirty value) {return new(value);}
    }

    /// This is used for optional parameters of class `Dirty` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Dirty`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Dirty`/`Const_Dirty` to pass it to the function.
    public class _InOptConst_Dirty
    {
        public Const_Dirty? Opt;

        public _InOptConst_Dirty() {}
        public _InOptConst_Dirty(Const_Dirty value) {Opt = value;}
        public static implicit operator _InOptConst_Dirty(Const_Dirty value) {return new(value);}
    }

    /// Visual Object
    /// Generated from class `MR::VisualObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Object`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectLabel`
    ///     `MR::ObjectLinesHolder`
    ///     `MR::ObjectMeshHolder`
    ///     `MR::ObjectPointsHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::AngleMeasurementObject`
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::LineObject`
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLines`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectPoints`
    ///     `MR::ObjectVoxels`
    ///     `MR::PlaneObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::PointObject`
    ///     `MR::RadiusMeasurementObject`
    ///     `MR::SphereObject`
    /// This is the const half of the class.
    public class Const_VisualObject : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_VisualObject_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_VisualObject_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_VisualObject_UseCount();
                return __MR_std_shared_ptr_MR_VisualObject_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_VisualObject_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_VisualObject_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_VisualObject(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_VisualObject_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_VisualObject_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_VisualObject_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_VisualObject_ConstructNonOwning(ptr);
        }

        internal unsafe Const_VisualObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe VisualObject _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_VisualObject_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_VisualObject_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_VisualObject_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_VisualObject_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_VisualObject_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_VisualObject_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_VisualObject_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VisualObject() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_VisualObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_VisualObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_VisualObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_VisualObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_VisualObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_VisualObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_VisualObject?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_VisualObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_VisualObject(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_VisualObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VisualObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_VisualObject_DefaultConstruct();
            _LateMakeShared(__MR_VisualObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::VisualObject::VisualObject`.
        public unsafe Const_VisualObject(MR._ByValue_VisualObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_VisualObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VisualObject._Underlying *_other);
            _LateMakeShared(__MR_VisualObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::VisualObject::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_VisualObject_StaticTypeName();
            var __ret = __MR_VisualObject_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::VisualObject::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_typeName", ExactSpelling = true)]
            extern static byte *__MR_VisualObject_typeName(_Underlying *_this);
            var __ret = __MR_VisualObject_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::VisualObject::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_VisualObject_StaticClassName();
            var __ret = __MR_VisualObject_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::VisualObject::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_VisualObject_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_VisualObject_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::VisualObject::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_VisualObject_StaticClassNameInPlural();
            var __ret = __MR_VisualObject_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::VisualObject::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_VisualObject_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_VisualObject_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Returns true if this class supports the property `type`. Otherwise passing it to the functions below is illegal.
        /// Generated from method `MR::VisualObject::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_VisualObject_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_VisualObject_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::VisualObject::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_VisualObject_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_VisualObject_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::VisualObject::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_VisualObject_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_VisualObject_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// get all visualize properties masks
        /// Generated from method `MR::VisualObject::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_VisualObject_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_VisualObject_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::VisualObject::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_VisualObject_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_VisualObject_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::VisualObject::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_VisualObject_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_VisualObject_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns whether object name is shown in any viewport
        /// Generated from method `MR::VisualObject::showName`.
        public unsafe bool ShowName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_showName_const", ExactSpelling = true)]
            extern static byte __MR_VisualObject_showName_const(_Underlying *_this);
            return __MR_VisualObject_showName_const(_UnderlyingPtr) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::VisualObject::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_VisualObject_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_VisualObject_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::VisualObject::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_VisualObject_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_VisualObject_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::VisualObject::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_VisualObject_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_VisualObject_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::VisualObject::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_VisualObject_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_VisualObject_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::VisualObject::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_VisualObject_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_VisualObject_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::VisualObject::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_VisualObject_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_VisualObject_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::VisualObject::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_VisualObject_getDirtyFlags(_Underlying *_this);
            return __MR_VisualObject_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::VisualObject::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_resetDirty", ExactSpelling = true)]
            extern static void __MR_VisualObject_resetDirty(_Underlying *_this);
            __MR_VisualObject_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::VisualObject::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_VisualObject_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_VisualObject_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::VisualObject::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_VisualObject_getBoundingBox(_Underlying *_this);
            return __MR_VisualObject_getBoundingBox(_UnderlyingPtr);
        }

        /// returns bounding box of this object in given viewport in world coordinates,
        /// to get world bounding box of the object with all child objects, please call Object::getWorldTreeBox method
        /// Generated from method `MR::VisualObject::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_VisualObject_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_VisualObject_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns true if the object must be redrawn (due to dirty flags) in one of specified viewports
        /// Generated from method `MR::VisualObject::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_VisualObject_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_VisualObject_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::VisualObject::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isPickable", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_VisualObject_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::VisualObject::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_VisualObject_getColoringType(_Underlying *_this);
            return __MR_VisualObject_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::VisualObject::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getShininess", ExactSpelling = true)]
            extern static float __MR_VisualObject_getShininess(_Underlying *_this);
            return __MR_VisualObject_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::VisualObject::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_VisualObject_getSpecularStrength(_Underlying *_this);
            return __MR_VisualObject_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::VisualObject::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_VisualObject_getAmbientStrength(_Underlying *_this);
            return __MR_VisualObject_getAmbientStrength(_UnderlyingPtr);
        }

        /// clones this object only, without its children,
        /// making new object the owner of all copied resources
        /// Generated from method `MR::VisualObject::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_VisualObject_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_VisualObject_clone(_UnderlyingPtr), is_owning: true));
        }

        /// clones this object only, without its children,
        /// making new object to share resources with this object
        /// Generated from method `MR::VisualObject::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_VisualObject_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_VisualObject_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::VisualObject::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_render", ExactSpelling = true)]
            extern static byte __MR_VisualObject_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_VisualObject_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::VisualObject::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_renderForPicker", ExactSpelling = true)]
            extern static void __MR_VisualObject_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_VisualObject_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::VisualObject::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_renderUi", ExactSpelling = true)]
            extern static void __MR_VisualObject_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_VisualObject_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::VisualObject::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_VisualObject_heapBytes(_Underlying *_this);
            return __MR_VisualObject_heapBytes(_UnderlyingPtr);
        }

        /// return several info lines that can better describe the object in the UI
        /// Generated from method `MR::VisualObject::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_VisualObject_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_VisualObject_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::VisualObject::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_VisualObject_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_VisualObject_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VisualObject::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_VisualObject_name(_Underlying *_this);
            return new(__MR_VisualObject_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::VisualObject::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_VisualObject_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_VisualObject_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::VisualObject::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_VisualObject_xfsForAllViewports(_Underlying *_this);
            return new(__MR_VisualObject_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::VisualObject::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_VisualObject_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_VisualObject_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::VisualObject::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_VisualObject_globalVisibilityMask(_Underlying *_this);
            return new(__MR_VisualObject_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::VisualObject::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_VisualObject_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_VisualObject_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::VisualObject::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isLocked", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isLocked(_Underlying *_this);
            return __MR_VisualObject_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::VisualObject::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isParentLocked(_Underlying *_this);
            return __MR_VisualObject_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::VisualObject::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isAncestor", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_VisualObject_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::VisualObject::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isSelected", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isSelected(_Underlying *_this);
            return __MR_VisualObject_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::VisualObject::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isAncillary", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isAncillary(_Underlying *_this);
            return __MR_VisualObject_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::VisualObject::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isGlobalAncillary(_Underlying *_this);
            return __MR_VisualObject_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::VisualObject::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_isVisible", ExactSpelling = true)]
            extern static byte __MR_VisualObject_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_VisualObject_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::VisualObject::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_VisualObject_visibilityMask(_Underlying *_this);
            return new(__MR_VisualObject_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::VisualObject::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_VisualObject_resetRedrawFlag(_Underlying *_this);
            __MR_VisualObject_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::VisualObject::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_VisualObject_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_VisualObject_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::VisualObject::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_VisualObject_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_VisualObject_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::VisualObject::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_VisualObject_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_VisualObject_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
        /// Generated from method `MR::VisualObject::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_VisualObject_hasVisualRepresentation(_Underlying *_this);
            return __MR_VisualObject_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// does the object have any model available (but possibly empty),
        /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
        /// Generated from method `MR::VisualObject::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_hasModel", ExactSpelling = true)]
            extern static byte __MR_VisualObject_hasModel(_Underlying *_this);
            return __MR_VisualObject_hasModel(_UnderlyingPtr) != 0;
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::VisualObject::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_VisualObject_tags(_Underlying *_this);
            return new(__MR_VisualObject_tags(_UnderlyingPtr), is_owning: false);
        }

        // return true if model of current object equals to model (the same) of other
        /// Generated from method `MR::VisualObject::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_sameModels", ExactSpelling = true)]
            extern static byte __MR_VisualObject_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_VisualObject_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        // return hash of model (or hash object pointer if object has no model)
        /// Generated from method `MR::VisualObject::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_VisualObject_getModelHash(_Underlying *_this);
            return __MR_VisualObject_getModelHash(_UnderlyingPtr);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::VisualObject::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_VisualObject_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_VisualObject_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }
    }

    /// Visual Object
    /// Generated from class `MR::VisualObject`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Object`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::FeatureObject`
    ///     `MR::MeasurementObject`
    ///     `MR::ObjectLabel`
    ///     `MR::ObjectLinesHolder`
    ///     `MR::ObjectMeshHolder`
    ///     `MR::ObjectPointsHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::angle, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter, MR::DimensionsVisualizePropertyType::length>`
    ///     `MR::AddVisualProperties<MR::FeatureObject, MR::DimensionsVisualizePropertyType::diameter>`
    ///     `MR::AngleMeasurementObject`
    ///     `MR::CircleObject`
    ///     `MR::ConeObject`
    ///     `MR::CylinderObject`
    ///     `MR::DistanceMeasurementObject`
    ///     `MR::LineObject`
    ///     `MR::ObjectDistanceMap`
    ///     `MR::ObjectGcode`
    ///     `MR::ObjectLines`
    ///     `MR::ObjectMesh`
    ///     `MR::ObjectPoints`
    ///     `MR::ObjectVoxels`
    ///     `MR::PlaneObject`
    ///     `MR::PointMeasurementObject`
    ///     `MR::PointObject`
    ///     `MR::RadiusMeasurementObject`
    ///     `MR::SphereObject`
    /// This is the non-const half of the class.
    public class VisualObject : Const_VisualObject
    {
        internal unsafe VisualObject(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe VisualObject(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(VisualObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_VisualObject_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_VisualObject_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(VisualObject self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_VisualObject_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_VisualObject_UpcastTo_MR_Object(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator VisualObject?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_VisualObject", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_VisualObject(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_VisualObject(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe VisualObject() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_VisualObject_DefaultConstruct();
            _LateMakeShared(__MR_VisualObject_DefaultConstruct());
        }

        /// Generated from constructor `MR::VisualObject::VisualObject`.
        public unsafe VisualObject(MR._ByValue_VisualObject _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_VisualObject_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VisualObject._Underlying *_other);
            _LateMakeShared(__MR_VisualObject_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::VisualObject::operator=`.
        public unsafe MR.VisualObject Assign(MR._ByValue_VisualObject _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_VisualObject_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VisualObject._Underlying *_other);
            return new(__MR_VisualObject_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::VisualObject::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_VisualObject_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::VisualObject::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_VisualObject_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::VisualObject::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_VisualObject_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::VisualObject::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_VisualObject_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_VisualObject_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::VisualObject::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_VisualObject other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_VisualObject_copyAllSolidColors(_Underlying *_this, MR.Const_VisualObject._Underlying *other);
            __MR_VisualObject_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::VisualObject::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_VisualObject_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// shows/hides object name in all viewports
        /// Generated from method `MR::VisualObject::showName`.
        public unsafe void ShowName(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_showName", ExactSpelling = true)]
            extern static void __MR_VisualObject_showName(_Underlying *_this, byte on);
            __MR_VisualObject_showName(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::VisualObject::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setFrontColor", ExactSpelling = true)]
            extern static void __MR_VisualObject_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_VisualObject_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::VisualObject::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_VisualObject_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VisualObject_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::VisualObject::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_VisualObject_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_VisualObject_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::VisualObject::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setBackColor", ExactSpelling = true)]
            extern static void __MR_VisualObject_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_VisualObject_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::VisualObject::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_VisualObject_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_VisualObject_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::VisualObject::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_VisualObject_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_VisualObject_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets some dirty flags for the object (to force its visual update)
        /// \param mask is a union of DirtyFlags flags
        /// \param invalidateCaches whether to automatically invalidate model caches (pass false here if you manually update the caches)
        /// Generated from method `MR::VisualObject::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_VisualObject_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_VisualObject_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::VisualObject::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setPickable", ExactSpelling = true)]
            extern static void __MR_VisualObject_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::VisualObject::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setColoringType", ExactSpelling = true)]
            extern static void __MR_VisualObject_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_VisualObject_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::VisualObject::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setShininess", ExactSpelling = true)]
            extern static void __MR_VisualObject_setShininess(_Underlying *_this, float shininess);
            __MR_VisualObject_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::VisualObject::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_VisualObject_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_VisualObject_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::VisualObject::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_VisualObject_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_VisualObject_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::VisualObject::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_VisualObject_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_VisualObject_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::VisualObject::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_VisualObject_resetFrontColor(_Underlying *_this);
            __MR_VisualObject_resetFrontColor(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::VisualObject::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_resetColors", ExactSpelling = true)]
            extern static void __MR_VisualObject_resetColors(_Underlying *_this);
            __MR_VisualObject_resetColors(_UnderlyingPtr);
        }

        /// Generated from method `MR::VisualObject::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setName", ExactSpelling = true)]
            extern static void __MR_VisualObject_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_VisualObject_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::VisualObject::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setXf", ExactSpelling = true)]
            extern static void __MR_VisualObject_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_VisualObject_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::VisualObject::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_resetXf", ExactSpelling = true)]
            extern static void __MR_VisualObject_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_VisualObject_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::VisualObject::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_VisualObject_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_VisualObject_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VisualObject::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setWorldXf", ExactSpelling = true)]
            extern static void __MR_VisualObject_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_VisualObject_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// scale object size (all point positions)
        /// Generated from method `MR::VisualObject::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_applyScale", ExactSpelling = true)]
            extern static void __MR_VisualObject_applyScale(_Underlying *_this, float scaleFactor);
            __MR_VisualObject_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::VisualObject::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_VisualObject_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VisualObject::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setLocked", ExactSpelling = true)]
            extern static void __MR_VisualObject_setLocked(_Underlying *_this, byte on);
            __MR_VisualObject_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::VisualObject::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setParentLocked", ExactSpelling = true)]
            extern static void __MR_VisualObject_setParentLocked(_Underlying *_this, byte lock_);
            __MR_VisualObject_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::VisualObject::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_VisualObject_detachFromParent(_Underlying *_this);
            return __MR_VisualObject_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::VisualObject::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_addChild", ExactSpelling = true)]
            extern static byte __MR_VisualObject_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_VisualObject_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::VisualObject::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_VisualObject_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_VisualObject_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::VisualObject::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_VisualObject_removeAllChildren(_Underlying *_this);
            __MR_VisualObject_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::VisualObject::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_sortChildren", ExactSpelling = true)]
            extern static void __MR_VisualObject_sortChildren(_Underlying *_this);
            __MR_VisualObject_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::VisualObject::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_select", ExactSpelling = true)]
            extern static byte __MR_VisualObject_select(_Underlying *_this, byte on);
            return __MR_VisualObject_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::VisualObject::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setAncillary", ExactSpelling = true)]
            extern static void __MR_VisualObject_setAncillary(_Underlying *_this, byte ancillary);
            __MR_VisualObject_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::VisualObject::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setVisible", ExactSpelling = true)]
            extern static void __MR_VisualObject_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::VisualObject::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_VisualObject_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_VisualObject_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::VisualObject::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_swap", ExactSpelling = true)]
            extern static void __MR_VisualObject_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_VisualObject_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::VisualObject::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_addTag", ExactSpelling = true)]
            extern static byte __MR_VisualObject_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_VisualObject_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::VisualObject::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_removeTag", ExactSpelling = true)]
            extern static byte __MR_VisualObject_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_VisualObject_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `VisualObject` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VisualObject`/`Const_VisualObject` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VisualObject
    {
        internal readonly Const_VisualObject? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VisualObject() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VisualObject(MR.Misc._Moved<VisualObject> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VisualObject(MR.Misc._Moved<VisualObject> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VisualObject` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VisualObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VisualObject`/`Const_VisualObject` directly.
    public class _InOptMut_VisualObject
    {
        public VisualObject? Opt;

        public _InOptMut_VisualObject() {}
        public _InOptMut_VisualObject(VisualObject value) {Opt = value;}
        public static implicit operator _InOptMut_VisualObject(VisualObject value) {return new(value);}
    }

    /// This is used for optional parameters of class `VisualObject` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VisualObject`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VisualObject`/`Const_VisualObject` to pass it to the function.
    public class _InOptConst_VisualObject
    {
        public Const_VisualObject? Opt;

        public _InOptConst_VisualObject() {}
        public _InOptConst_VisualObject(Const_VisualObject value) {Opt = value;}
        public static implicit operator _InOptConst_VisualObject(Const_VisualObject value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::MeshVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::MeshVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::MeshVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRMeshVisualizePropertyType : Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRMeshVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRMeshVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::MeshVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRMeshVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::MeshVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRMeshVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_MeshVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRMeshVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRMeshVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRMeshVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRMeshVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRMeshVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRMeshVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRMeshVisualizePropertyType(IsVisualizeMaskEnum_MRMeshVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRMeshVisualizePropertyType(IsVisualizeMaskEnum_MRMeshVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRMeshVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRMeshVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRMeshVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRMeshVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRMeshVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRMeshVisualizePropertyType(Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRMeshVisualizePropertyType(Const_IsVisualizeMaskEnum_MRMeshVisualizePropertyType value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::LinesVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::LinesVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::LinesVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRLinesVisualizePropertyType : Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRLinesVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRLinesVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::LinesVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRLinesVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::LinesVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRLinesVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_LinesVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRLinesVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRLinesVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRLinesVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRLinesVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRLinesVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRLinesVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRLinesVisualizePropertyType(IsVisualizeMaskEnum_MRLinesVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRLinesVisualizePropertyType(IsVisualizeMaskEnum_MRLinesVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRLinesVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRLinesVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRLinesVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRLinesVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRLinesVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRLinesVisualizePropertyType(Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRLinesVisualizePropertyType(Const_IsVisualizeMaskEnum_MRLinesVisualizePropertyType value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::PointsVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::PointsVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::PointsVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRPointsVisualizePropertyType : Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRPointsVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRPointsVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::PointsVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRPointsVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::PointsVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRPointsVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_PointsVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRPointsVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRPointsVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRPointsVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRPointsVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRPointsVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRPointsVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRPointsVisualizePropertyType(IsVisualizeMaskEnum_MRPointsVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRPointsVisualizePropertyType(IsVisualizeMaskEnum_MRPointsVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRPointsVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRPointsVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRPointsVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRPointsVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRPointsVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRPointsVisualizePropertyType(Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRPointsVisualizePropertyType(Const_IsVisualizeMaskEnum_MRPointsVisualizePropertyType value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::DimensionsVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::DimensionsVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::DimensionsVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType : Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::DimensionsVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::DimensionsVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_DimensionsVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType(Const_IsVisualizeMaskEnum_MRDimensionsVisualizePropertyType value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::FeatureVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::FeatureVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::FeatureVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRFeatureVisualizePropertyType : Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRFeatureVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::FeatureVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::FeatureVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRFeatureVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_FeatureVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRFeatureVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRFeatureVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRFeatureVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(IsVisualizeMaskEnum_MRFeatureVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(IsVisualizeMaskEnum_MRFeatureVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRFeatureVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRFeatureVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType(Const_IsVisualizeMaskEnum_MRFeatureVisualizePropertyType value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::PointMeasurementVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::PointMeasurementVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::PointMeasurementVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType : Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::PointMeasurementVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::PointMeasurementVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_PointMeasurementVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType(Const_IsVisualizeMaskEnum_MRPointMeasurementVisualizePropertyType value) {return new(value);}
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::LabelVisualizePropertyType>`.
    /// This is the const half of the class.
    public class Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_Destroy", ExactSpelling = true)]
            extern static void __MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_Destroy(_Underlying *_this);
            __MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::LabelVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::IsVisualizeMaskEnum<MR::LabelVisualizePropertyType>`.
    /// This is the non-const half of the class.
    public class IsVisualizeMaskEnum_MRLabelVisualizePropertyType : Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType
    {
        internal unsafe IsVisualizeMaskEnum_MRLabelVisualizePropertyType(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe IsVisualizeMaskEnum_MRLabelVisualizePropertyType() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_DefaultConstruct", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_DefaultConstruct();
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_DefaultConstruct();
        }

        /// Generated from constructor `MR::IsVisualizeMaskEnum<MR::LabelVisualizePropertyType>::IsVisualizeMaskEnum`.
        public unsafe IsVisualizeMaskEnum_MRLabelVisualizePropertyType(MR.Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_ConstructFromAnother(MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *_other);
            _UnderlyingPtr = __MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::IsVisualizeMaskEnum<MR::LabelVisualizePropertyType>::operator=`.
        public unsafe MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType Assign(MR.Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_AssignFromAnother", ExactSpelling = true)]
            extern static MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *__MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_AssignFromAnother(_Underlying *_this, MR.IsVisualizeMaskEnum_MRLabelVisualizePropertyType._Underlying *_other);
            return new(__MR_IsVisualizeMaskEnum_MR_LabelVisualizePropertyType_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRLabelVisualizePropertyType` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IsVisualizeMaskEnum_MRLabelVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRLabelVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType` directly.
    public class _InOptMut_IsVisualizeMaskEnum_MRLabelVisualizePropertyType
    {
        public IsVisualizeMaskEnum_MRLabelVisualizePropertyType? Opt;

        public _InOptMut_IsVisualizeMaskEnum_MRLabelVisualizePropertyType() {}
        public _InOptMut_IsVisualizeMaskEnum_MRLabelVisualizePropertyType(IsVisualizeMaskEnum_MRLabelVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptMut_IsVisualizeMaskEnum_MRLabelVisualizePropertyType(IsVisualizeMaskEnum_MRLabelVisualizePropertyType value) {return new(value);}
    }

    /// This is used for optional parameters of class `IsVisualizeMaskEnum_MRLabelVisualizePropertyType` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IsVisualizeMaskEnum_MRLabelVisualizePropertyType`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IsVisualizeMaskEnum_MRLabelVisualizePropertyType`/`Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType` to pass it to the function.
    public class _InOptConst_IsVisualizeMaskEnum_MRLabelVisualizePropertyType
    {
        public Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType? Opt;

        public _InOptConst_IsVisualizeMaskEnum_MRLabelVisualizePropertyType() {}
        public _InOptConst_IsVisualizeMaskEnum_MRLabelVisualizePropertyType(Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType value) {Opt = value;}
        public static implicit operator _InOptConst_IsVisualizeMaskEnum_MRLabelVisualizePropertyType(Const_IsVisualizeMaskEnum_MRLabelVisualizePropertyType value) {return new(value);}
    }
}
