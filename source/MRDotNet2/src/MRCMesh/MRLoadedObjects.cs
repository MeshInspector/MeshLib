public static partial class MR
{
    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<>`.
    /// This is the const half of the class.
    public class Const_LoadedObjectT : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LoadedObjectT(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_Destroy", ExactSpelling = true)]
            extern static void __MR_LoadedObjectT_Destroy(_Underlying *_this);
            __MR_LoadedObjectT_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LoadedObjectT() {Dispose(false);}

        public unsafe MR.Const_Object Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_Get_obj", ExactSpelling = true)]
                extern static MR.Const_Object._UnderlyingShared *__MR_LoadedObjectT_Get_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_Get_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public unsafe MR.Std.Const_String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_Get_warnings", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_LoadedObjectT_Get_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_Get_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_Get_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LoadedObjectT() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<>` elementwise.
        public unsafe Const_LoadedObjectT(MR._ByValue_Object obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.Object._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<>::LoadedObjectT`.
        public unsafe Const_LoadedObjectT(MR._ByValue_LoadedObjectT _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<>`.
    /// This is the non-const half of the class.
    public class LoadedObjectT : Const_LoadedObjectT
    {
        internal unsafe LoadedObjectT(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Object Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_GetMutable_obj", ExactSpelling = true)]
                extern static MR.Object._UnderlyingShared *__MR_LoadedObjectT_GetMutable_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_GetMutable_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public new unsafe MR.Std.String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_GetMutable_warnings", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_LoadedObjectT_GetMutable_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_GetMutable_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LoadedObjectT() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<>` elementwise.
        public unsafe LoadedObjectT(MR._ByValue_Object obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.Object._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<>::LoadedObjectT`.
        public unsafe LoadedObjectT(MR._ByValue_LoadedObjectT _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LoadedObjectT<>::operator=`.
        public unsafe MR.LoadedObjectT Assign(MR._ByValue_LoadedObjectT _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT._Underlying *__MR_LoadedObjectT_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT._Underlying *_other);
            return new(__MR_LoadedObjectT_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LoadedObjectT` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LoadedObjectT`/`Const_LoadedObjectT` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LoadedObjectT
    {
        internal readonly Const_LoadedObjectT? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LoadedObjectT() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LoadedObjectT(Const_LoadedObjectT new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LoadedObjectT(Const_LoadedObjectT arg) {return new(arg);}
        public _ByValue_LoadedObjectT(MR.Misc._Moved<LoadedObjectT> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LoadedObjectT(MR.Misc._Moved<LoadedObjectT> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LoadedObjectT` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadedObjectT`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT`/`Const_LoadedObjectT` directly.
    public class _InOptMut_LoadedObjectT
    {
        public LoadedObjectT? Opt;

        public _InOptMut_LoadedObjectT() {}
        public _InOptMut_LoadedObjectT(LoadedObjectT value) {Opt = value;}
        public static implicit operator _InOptMut_LoadedObjectT(LoadedObjectT value) {return new(value);}
    }

    /// This is used for optional parameters of class `LoadedObjectT` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadedObjectT`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT`/`Const_LoadedObjectT` to pass it to the function.
    public class _InOptConst_LoadedObjectT
    {
        public Const_LoadedObjectT? Opt;

        public _InOptConst_LoadedObjectT() {}
        public _InOptConst_LoadedObjectT(Const_LoadedObjectT value) {Opt = value;}
        public static implicit operator _InOptConst_LoadedObjectT(Const_LoadedObjectT value) {return new(value);}
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectMesh>`.
    /// This is the const half of the class.
    public class Const_LoadedObjectT_MRObjectMesh : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LoadedObjectT_MRObjectMesh(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_Destroy", ExactSpelling = true)]
            extern static void __MR_LoadedObjectT_MR_ObjectMesh_Destroy(_Underlying *_this);
            __MR_LoadedObjectT_MR_ObjectMesh_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LoadedObjectT_MRObjectMesh() {Dispose(false);}

        public unsafe MR.Const_ObjectMesh Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_Get_obj", ExactSpelling = true)]
                extern static MR.Const_ObjectMesh._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectMesh_Get_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectMesh_Get_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public unsafe MR.Std.Const_String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_Get_warnings", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_Get_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectMesh_Get_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_Get_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectMesh_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LoadedObjectT_MRObjectMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectMesh_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectMesh>` elementwise.
        public unsafe Const_LoadedObjectT_MRObjectMesh(MR._ByValue_ObjectMesh obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectMesh_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectMesh>::LoadedObjectT`.
        public unsafe Const_LoadedObjectT_MRObjectMesh(MR._ByValue_LoadedObjectT_MRObjectMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectMesh._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectMesh>`.
    /// This is the non-const half of the class.
    public class LoadedObjectT_MRObjectMesh : Const_LoadedObjectT_MRObjectMesh
    {
        internal unsafe LoadedObjectT_MRObjectMesh(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.ObjectMesh Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_GetMutable_obj", ExactSpelling = true)]
                extern static MR.ObjectMesh._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectMesh_GetMutable_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectMesh_GetMutable_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public new unsafe MR.Std.String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_GetMutable_warnings", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_GetMutable_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectMesh_GetMutable_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectMesh_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LoadedObjectT_MRObjectMesh() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectMesh_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectMesh>` elementwise.
        public unsafe LoadedObjectT_MRObjectMesh(MR._ByValue_ObjectMesh obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectMesh._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectMesh_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectMesh>::LoadedObjectT`.
        public unsafe LoadedObjectT_MRObjectMesh(MR._ByValue_LoadedObjectT_MRObjectMesh _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectMesh._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectMesh_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LoadedObjectT<MR::ObjectMesh>::operator=`.
        public unsafe MR.LoadedObjectT_MRObjectMesh Assign(MR._ByValue_LoadedObjectT_MRObjectMesh _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectMesh_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectMesh._Underlying *__MR_LoadedObjectT_MR_ObjectMesh_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectMesh._Underlying *_other);
            return new(__MR_LoadedObjectT_MR_ObjectMesh_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LoadedObjectT_MRObjectMesh` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LoadedObjectT_MRObjectMesh`/`Const_LoadedObjectT_MRObjectMesh` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LoadedObjectT_MRObjectMesh
    {
        internal readonly Const_LoadedObjectT_MRObjectMesh? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LoadedObjectT_MRObjectMesh() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LoadedObjectT_MRObjectMesh(Const_LoadedObjectT_MRObjectMesh new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectMesh(Const_LoadedObjectT_MRObjectMesh arg) {return new(arg);}
        public _ByValue_LoadedObjectT_MRObjectMesh(MR.Misc._Moved<LoadedObjectT_MRObjectMesh> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectMesh(MR.Misc._Moved<LoadedObjectT_MRObjectMesh> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectMesh` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadedObjectT_MRObjectMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectMesh`/`Const_LoadedObjectT_MRObjectMesh` directly.
    public class _InOptMut_LoadedObjectT_MRObjectMesh
    {
        public LoadedObjectT_MRObjectMesh? Opt;

        public _InOptMut_LoadedObjectT_MRObjectMesh() {}
        public _InOptMut_LoadedObjectT_MRObjectMesh(LoadedObjectT_MRObjectMesh value) {Opt = value;}
        public static implicit operator _InOptMut_LoadedObjectT_MRObjectMesh(LoadedObjectT_MRObjectMesh value) {return new(value);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectMesh` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadedObjectT_MRObjectMesh`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectMesh`/`Const_LoadedObjectT_MRObjectMesh` to pass it to the function.
    public class _InOptConst_LoadedObjectT_MRObjectMesh
    {
        public Const_LoadedObjectT_MRObjectMesh? Opt;

        public _InOptConst_LoadedObjectT_MRObjectMesh() {}
        public _InOptConst_LoadedObjectT_MRObjectMesh(Const_LoadedObjectT_MRObjectMesh value) {Opt = value;}
        public static implicit operator _InOptConst_LoadedObjectT_MRObjectMesh(Const_LoadedObjectT_MRObjectMesh value) {return new(value);}
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectPoints>`.
    /// This is the const half of the class.
    public class Const_LoadedObjectT_MRObjectPoints : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LoadedObjectT_MRObjectPoints(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_Destroy", ExactSpelling = true)]
            extern static void __MR_LoadedObjectT_MR_ObjectPoints_Destroy(_Underlying *_this);
            __MR_LoadedObjectT_MR_ObjectPoints_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LoadedObjectT_MRObjectPoints() {Dispose(false);}

        public unsafe MR.Const_ObjectPoints Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_Get_obj", ExactSpelling = true)]
                extern static MR.Const_ObjectPoints._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectPoints_Get_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectPoints_Get_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public unsafe MR.Std.Const_String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_Get_warnings", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_Get_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectPoints_Get_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_Get_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectPoints_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LoadedObjectT_MRObjectPoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectPoints_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectPoints>` elementwise.
        public unsafe Const_LoadedObjectT_MRObjectPoints(MR._ByValue_ObjectPoints obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectPoints._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectPoints_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectPoints>::LoadedObjectT`.
        public unsafe Const_LoadedObjectT_MRObjectPoints(MR._ByValue_LoadedObjectT_MRObjectPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectPoints._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectPoints>`.
    /// This is the non-const half of the class.
    public class LoadedObjectT_MRObjectPoints : Const_LoadedObjectT_MRObjectPoints
    {
        internal unsafe LoadedObjectT_MRObjectPoints(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.ObjectPoints Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_GetMutable_obj", ExactSpelling = true)]
                extern static MR.ObjectPoints._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectPoints_GetMutable_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectPoints_GetMutable_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public new unsafe MR.Std.String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_GetMutable_warnings", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_GetMutable_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectPoints_GetMutable_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectPoints_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LoadedObjectT_MRObjectPoints() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectPoints_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectPoints>` elementwise.
        public unsafe LoadedObjectT_MRObjectPoints(MR._ByValue_ObjectPoints obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectPoints._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectPoints_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectPoints>::LoadedObjectT`.
        public unsafe LoadedObjectT_MRObjectPoints(MR._ByValue_LoadedObjectT_MRObjectPoints _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectPoints._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectPoints_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LoadedObjectT<MR::ObjectPoints>::operator=`.
        public unsafe MR.LoadedObjectT_MRObjectPoints Assign(MR._ByValue_LoadedObjectT_MRObjectPoints _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectPoints_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectPoints._Underlying *__MR_LoadedObjectT_MR_ObjectPoints_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectPoints._Underlying *_other);
            return new(__MR_LoadedObjectT_MR_ObjectPoints_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LoadedObjectT_MRObjectPoints` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LoadedObjectT_MRObjectPoints`/`Const_LoadedObjectT_MRObjectPoints` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LoadedObjectT_MRObjectPoints
    {
        internal readonly Const_LoadedObjectT_MRObjectPoints? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LoadedObjectT_MRObjectPoints() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LoadedObjectT_MRObjectPoints(Const_LoadedObjectT_MRObjectPoints new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectPoints(Const_LoadedObjectT_MRObjectPoints arg) {return new(arg);}
        public _ByValue_LoadedObjectT_MRObjectPoints(MR.Misc._Moved<LoadedObjectT_MRObjectPoints> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectPoints(MR.Misc._Moved<LoadedObjectT_MRObjectPoints> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectPoints` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadedObjectT_MRObjectPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectPoints`/`Const_LoadedObjectT_MRObjectPoints` directly.
    public class _InOptMut_LoadedObjectT_MRObjectPoints
    {
        public LoadedObjectT_MRObjectPoints? Opt;

        public _InOptMut_LoadedObjectT_MRObjectPoints() {}
        public _InOptMut_LoadedObjectT_MRObjectPoints(LoadedObjectT_MRObjectPoints value) {Opt = value;}
        public static implicit operator _InOptMut_LoadedObjectT_MRObjectPoints(LoadedObjectT_MRObjectPoints value) {return new(value);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectPoints` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadedObjectT_MRObjectPoints`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectPoints`/`Const_LoadedObjectT_MRObjectPoints` to pass it to the function.
    public class _InOptConst_LoadedObjectT_MRObjectPoints
    {
        public Const_LoadedObjectT_MRObjectPoints? Opt;

        public _InOptConst_LoadedObjectT_MRObjectPoints() {}
        public _InOptConst_LoadedObjectT_MRObjectPoints(Const_LoadedObjectT_MRObjectPoints value) {Opt = value;}
        public static implicit operator _InOptConst_LoadedObjectT_MRObjectPoints(Const_LoadedObjectT_MRObjectPoints value) {return new(value);}
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectLines>`.
    /// This is the const half of the class.
    public class Const_LoadedObjectT_MRObjectLines : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LoadedObjectT_MRObjectLines(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_Destroy", ExactSpelling = true)]
            extern static void __MR_LoadedObjectT_MR_ObjectLines_Destroy(_Underlying *_this);
            __MR_LoadedObjectT_MR_ObjectLines_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LoadedObjectT_MRObjectLines() {Dispose(false);}

        public unsafe MR.Const_ObjectLines Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_Get_obj", ExactSpelling = true)]
                extern static MR.Const_ObjectLines._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectLines_Get_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectLines_Get_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public unsafe MR.Std.Const_String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_Get_warnings", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_LoadedObjectT_MR_ObjectLines_Get_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectLines_Get_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectLines_Get_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectLines_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LoadedObjectT_MRObjectLines() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectLines_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectLines>` elementwise.
        public unsafe Const_LoadedObjectT_MRObjectLines(MR._ByValue_ObjectLines obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectLines._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectLines_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectLines>::LoadedObjectT`.
        public unsafe Const_LoadedObjectT_MRObjectLines(MR._ByValue_LoadedObjectT_MRObjectLines _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectLines._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectLines_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectLines>`.
    /// This is the non-const half of the class.
    public class LoadedObjectT_MRObjectLines : Const_LoadedObjectT_MRObjectLines
    {
        internal unsafe LoadedObjectT_MRObjectLines(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.ObjectLines Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_GetMutable_obj", ExactSpelling = true)]
                extern static MR.ObjectLines._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectLines_GetMutable_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectLines_GetMutable_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public new unsafe MR.Std.String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_GetMutable_warnings", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_LoadedObjectT_MR_ObjectLines_GetMutable_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectLines_GetMutable_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectLines_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectLines_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LoadedObjectT_MRObjectLines() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectLines_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectLines>` elementwise.
        public unsafe LoadedObjectT_MRObjectLines(MR._ByValue_ObjectLines obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectLines._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectLines_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectLines>::LoadedObjectT`.
        public unsafe LoadedObjectT_MRObjectLines(MR._ByValue_LoadedObjectT_MRObjectLines _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectLines._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectLines_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LoadedObjectT<MR::ObjectLines>::operator=`.
        public unsafe MR.LoadedObjectT_MRObjectLines Assign(MR._ByValue_LoadedObjectT_MRObjectLines _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectLines_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectLines._Underlying *__MR_LoadedObjectT_MR_ObjectLines_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectLines._Underlying *_other);
            return new(__MR_LoadedObjectT_MR_ObjectLines_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LoadedObjectT_MRObjectLines` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LoadedObjectT_MRObjectLines`/`Const_LoadedObjectT_MRObjectLines` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LoadedObjectT_MRObjectLines
    {
        internal readonly Const_LoadedObjectT_MRObjectLines? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LoadedObjectT_MRObjectLines() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LoadedObjectT_MRObjectLines(Const_LoadedObjectT_MRObjectLines new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectLines(Const_LoadedObjectT_MRObjectLines arg) {return new(arg);}
        public _ByValue_LoadedObjectT_MRObjectLines(MR.Misc._Moved<LoadedObjectT_MRObjectLines> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectLines(MR.Misc._Moved<LoadedObjectT_MRObjectLines> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectLines` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadedObjectT_MRObjectLines`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectLines`/`Const_LoadedObjectT_MRObjectLines` directly.
    public class _InOptMut_LoadedObjectT_MRObjectLines
    {
        public LoadedObjectT_MRObjectLines? Opt;

        public _InOptMut_LoadedObjectT_MRObjectLines() {}
        public _InOptMut_LoadedObjectT_MRObjectLines(LoadedObjectT_MRObjectLines value) {Opt = value;}
        public static implicit operator _InOptMut_LoadedObjectT_MRObjectLines(LoadedObjectT_MRObjectLines value) {return new(value);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectLines` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadedObjectT_MRObjectLines`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectLines`/`Const_LoadedObjectT_MRObjectLines` to pass it to the function.
    public class _InOptConst_LoadedObjectT_MRObjectLines
    {
        public Const_LoadedObjectT_MRObjectLines? Opt;

        public _InOptConst_LoadedObjectT_MRObjectLines() {}
        public _InOptConst_LoadedObjectT_MRObjectLines(Const_LoadedObjectT_MRObjectLines value) {Opt = value;}
        public static implicit operator _InOptConst_LoadedObjectT_MRObjectLines(Const_LoadedObjectT_MRObjectLines value) {return new(value);}
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectVoxels>`.
    /// This is the const half of the class.
    public class Const_LoadedObjectT_MRObjectVoxels : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LoadedObjectT_MRObjectVoxels(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_Destroy", ExactSpelling = true)]
            extern static void __MR_LoadedObjectT_MR_ObjectVoxels_Destroy(_Underlying *_this);
            __MR_LoadedObjectT_MR_ObjectVoxels_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LoadedObjectT_MRObjectVoxels() {Dispose(false);}

        public unsafe MR.Const_ObjectVoxels Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_Get_obj", ExactSpelling = true)]
                extern static MR.Const_ObjectVoxels._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectVoxels_Get_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectVoxels_Get_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public unsafe MR.Std.Const_String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_Get_warnings", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_Get_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectVoxels_Get_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_Get_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectVoxels_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LoadedObjectT_MRObjectVoxels() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectVoxels_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectVoxels>` elementwise.
        public unsafe Const_LoadedObjectT_MRObjectVoxels(MR._ByValue_ObjectVoxels obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectVoxels._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectVoxels_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectVoxels>::LoadedObjectT`.
        public unsafe Const_LoadedObjectT_MRObjectVoxels(MR._ByValue_LoadedObjectT_MRObjectVoxels _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectVoxels._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectVoxels_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// result of loading (e.g. from a file) as one object (with possible subobjects)
    /// Generated from class `MR::LoadedObjectT<MR::ObjectVoxels>`.
    /// This is the non-const half of the class.
    public class LoadedObjectT_MRObjectVoxels : Const_LoadedObjectT_MRObjectVoxels
    {
        internal unsafe LoadedObjectT_MRObjectVoxels(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.ObjectVoxels Obj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_obj", ExactSpelling = true)]
                extern static MR.ObjectVoxels._UnderlyingShared *__MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_obj(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_obj(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public new unsafe MR.Std.String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_warnings", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_warnings(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjectT_MR_ObjectVoxels_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LoadedObjectT_MRObjectVoxels() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectVoxels_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjectT<MR::ObjectVoxels>` elementwise.
        public unsafe LoadedObjectT_MRObjectVoxels(MR._ByValue_ObjectVoxels obj, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_ConstructFrom(MR.Misc._PassBy obj_pass_by, MR.ObjectVoxels._UnderlyingShared *obj, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectVoxels_ConstructFrom(obj.PassByMode, obj.Value is not null ? obj.Value._UnderlyingSharedPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjectT<MR::ObjectVoxels>::LoadedObjectT`.
        public unsafe LoadedObjectT_MRObjectVoxels(MR._ByValue_LoadedObjectT_MRObjectVoxels _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectVoxels._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjectT_MR_ObjectVoxels_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LoadedObjectT<MR::ObjectVoxels>::operator=`.
        public unsafe MR.LoadedObjectT_MRObjectVoxels Assign(MR._ByValue_LoadedObjectT_MRObjectVoxels _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjectT_MR_ObjectVoxels_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjectT_MRObjectVoxels._Underlying *__MR_LoadedObjectT_MR_ObjectVoxels_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LoadedObjectT_MRObjectVoxels._Underlying *_other);
            return new(__MR_LoadedObjectT_MR_ObjectVoxels_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LoadedObjectT_MRObjectVoxels` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LoadedObjectT_MRObjectVoxels`/`Const_LoadedObjectT_MRObjectVoxels` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LoadedObjectT_MRObjectVoxels
    {
        internal readonly Const_LoadedObjectT_MRObjectVoxels? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LoadedObjectT_MRObjectVoxels() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LoadedObjectT_MRObjectVoxels(Const_LoadedObjectT_MRObjectVoxels new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectVoxels(Const_LoadedObjectT_MRObjectVoxels arg) {return new(arg);}
        public _ByValue_LoadedObjectT_MRObjectVoxels(MR.Misc._Moved<LoadedObjectT_MRObjectVoxels> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LoadedObjectT_MRObjectVoxels(MR.Misc._Moved<LoadedObjectT_MRObjectVoxels> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectVoxels` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadedObjectT_MRObjectVoxels`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectVoxels`/`Const_LoadedObjectT_MRObjectVoxels` directly.
    public class _InOptMut_LoadedObjectT_MRObjectVoxels
    {
        public LoadedObjectT_MRObjectVoxels? Opt;

        public _InOptMut_LoadedObjectT_MRObjectVoxels() {}
        public _InOptMut_LoadedObjectT_MRObjectVoxels(LoadedObjectT_MRObjectVoxels value) {Opt = value;}
        public static implicit operator _InOptMut_LoadedObjectT_MRObjectVoxels(LoadedObjectT_MRObjectVoxels value) {return new(value);}
    }

    /// This is used for optional parameters of class `LoadedObjectT_MRObjectVoxels` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadedObjectT_MRObjectVoxels`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjectT_MRObjectVoxels`/`Const_LoadedObjectT_MRObjectVoxels` to pass it to the function.
    public class _InOptConst_LoadedObjectT_MRObjectVoxels
    {
        public Const_LoadedObjectT_MRObjectVoxels? Opt;

        public _InOptConst_LoadedObjectT_MRObjectVoxels() {}
        public _InOptConst_LoadedObjectT_MRObjectVoxels(Const_LoadedObjectT_MRObjectVoxels value) {Opt = value;}
        public static implicit operator _InOptConst_LoadedObjectT_MRObjectVoxels(Const_LoadedObjectT_MRObjectVoxels value) {return new(value);}
    }

    /// result of loading (e.g. from a file) as a number of objects
    /// Generated from class `MR::LoadedObjects`.
    /// This is the const half of the class.
    public class Const_LoadedObjects : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LoadedObjects(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_Destroy", ExactSpelling = true)]
            extern static void __MR_LoadedObjects_Destroy(_Underlying *_this);
            __MR_LoadedObjects_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LoadedObjects() {Dispose(false);}

        public unsafe MR.Std.Const_Vector_StdSharedPtrMRObject Objs
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_Get_objs", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_StdSharedPtrMRObject._Underlying *__MR_LoadedObjects_Get_objs(_Underlying *_this);
                return new(__MR_LoadedObjects_Get_objs(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public unsafe MR.Std.Const_String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_Get_warnings", ExactSpelling = true)]
                extern static MR.Std.Const_String._Underlying *__MR_LoadedObjects_Get_warnings(_Underlying *_this);
                return new(__MR_LoadedObjects_Get_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_Get_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_LoadedObjects_Get_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjects_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LoadedObjects() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjects_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjects` elementwise.
        public unsafe Const_LoadedObjects(MR.Std._ByValue_Vector_StdSharedPtrMRObject objs, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_ConstructFrom(MR.Misc._PassBy objs_pass_by, MR.Std.Vector_StdSharedPtrMRObject._Underlying *objs, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjects_ConstructFrom(objs.PassByMode, objs.Value is not null ? objs.Value._UnderlyingPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjects::LoadedObjects`.
        public unsafe Const_LoadedObjects(MR._ByValue_LoadedObjects _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjects._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjects_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// result of loading (e.g. from a file) as a number of objects
    /// Generated from class `MR::LoadedObjects`.
    /// This is the non-const half of the class.
    public class LoadedObjects : Const_LoadedObjects
    {
        internal unsafe LoadedObjects(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Vector_StdSharedPtrMRObject Objs
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_GetMutable_objs", ExactSpelling = true)]
                extern static MR.Std.Vector_StdSharedPtrMRObject._Underlying *__MR_LoadedObjects_GetMutable_objs(_Underlying *_this);
                return new(__MR_LoadedObjects_GetMutable_objs(_UnderlyingPtr), is_owning: false);
            }
        }

        //either empty or ends with '\n'
        public new unsafe MR.Std.String Warnings
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_GetMutable_warnings", ExactSpelling = true)]
                extern static MR.Std.String._Underlying *__MR_LoadedObjects_GetMutable_warnings(_Underlying *_this);
                return new(__MR_LoadedObjects_GetMutable_warnings(_UnderlyingPtr), is_owning: false);
            }
        }

        /// units of object coordinates and transformations (if known)
        public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_GetMutable_lengthUnit", ExactSpelling = true)]
                extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_LoadedObjects_GetMutable_lengthUnit(_Underlying *_this);
                return new(__MR_LoadedObjects_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LoadedObjects() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_DefaultConstruct();
            _UnderlyingPtr = __MR_LoadedObjects_DefaultConstruct();
        }

        /// Constructs `MR::LoadedObjects` elementwise.
        public unsafe LoadedObjects(MR.Std._ByValue_Vector_StdSharedPtrMRObject objs, ReadOnlySpan<char> warnings, MR.LengthUnit? lengthUnit) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_ConstructFrom", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_ConstructFrom(MR.Misc._PassBy objs_pass_by, MR.Std.Vector_StdSharedPtrMRObject._Underlying *objs, byte *warnings, byte *warnings_end, MR.LengthUnit *lengthUnit);
            byte[] __bytes_warnings = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warnings.Length)];
            int __len_warnings = System.Text.Encoding.UTF8.GetBytes(warnings, __bytes_warnings);
            fixed (byte *__ptr_warnings = __bytes_warnings)
            {
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_LoadedObjects_ConstructFrom(objs.PassByMode, objs.Value is not null ? objs.Value._UnderlyingPtr : null, __ptr_warnings, __ptr_warnings + __len_warnings, lengthUnit.HasValue ? &__deref_lengthUnit : null);
            }
        }

        /// Generated from constructor `MR::LoadedObjects::LoadedObjects`.
        public unsafe LoadedObjects(MR._ByValue_LoadedObjects _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LoadedObjects._Underlying *_other);
            _UnderlyingPtr = __MR_LoadedObjects_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LoadedObjects::operator=`.
        public unsafe MR.LoadedObjects Assign(MR._ByValue_LoadedObjects _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LoadedObjects_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LoadedObjects._Underlying *__MR_LoadedObjects_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LoadedObjects._Underlying *_other);
            return new(__MR_LoadedObjects_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LoadedObjects` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LoadedObjects`/`Const_LoadedObjects` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LoadedObjects
    {
        internal readonly Const_LoadedObjects? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LoadedObjects() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LoadedObjects(Const_LoadedObjects new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LoadedObjects(Const_LoadedObjects arg) {return new(arg);}
        public _ByValue_LoadedObjects(MR.Misc._Moved<LoadedObjects> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LoadedObjects(MR.Misc._Moved<LoadedObjects> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LoadedObjects` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LoadedObjects`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjects`/`Const_LoadedObjects` directly.
    public class _InOptMut_LoadedObjects
    {
        public LoadedObjects? Opt;

        public _InOptMut_LoadedObjects() {}
        public _InOptMut_LoadedObjects(LoadedObjects value) {Opt = value;}
        public static implicit operator _InOptMut_LoadedObjects(LoadedObjects value) {return new(value);}
    }

    /// This is used for optional parameters of class `LoadedObjects` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LoadedObjects`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LoadedObjects`/`Const_LoadedObjects` to pass it to the function.
    public class _InOptConst_LoadedObjects
    {
        public Const_LoadedObjects? Opt;

        public _InOptConst_LoadedObjects() {}
        public _InOptConst_LoadedObjects(Const_LoadedObjects value) {Opt = value;}
        public static implicit operator _InOptConst_LoadedObjects(Const_LoadedObjects value) {return new(value);}
    }
}
