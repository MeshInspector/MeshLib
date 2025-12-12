public static partial class MR
{
    // class to process g-code source and generate toolpath
    /// Generated from class `MR::GcodeProcessor`.
    /// This is the const half of the class.
    public class Const_GcodeProcessor : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_GcodeProcessor(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Destroy", ExactSpelling = true)]
            extern static void __MR_GcodeProcessor_Destroy(_Underlying *_this);
            __MR_GcodeProcessor_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_GcodeProcessor() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_GcodeProcessor() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GcodeProcessor._Underlying *__MR_GcodeProcessor_DefaultConstruct();
            _UnderlyingPtr = __MR_GcodeProcessor_DefaultConstruct();
        }

        /// Generated from constructor `MR::GcodeProcessor::GcodeProcessor`.
        public unsafe Const_GcodeProcessor(MR._ByValue_GcodeProcessor _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GcodeProcessor._Underlying *__MR_GcodeProcessor_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor._Underlying *_other);
            _UnderlyingPtr = __MR_GcodeProcessor_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from class `MR::GcodeProcessor::BaseAction<MR::Vector2f>`.
        /// This is the const half of the class.
        public class Const_BaseAction_MRVector2f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BaseAction_MRVector2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_Destroy", ExactSpelling = true)]
                extern static void __MR_GcodeProcessor_BaseAction_MR_Vector2f_Destroy(_Underlying *_this);
                __MR_GcodeProcessor_BaseAction_MR_Vector2f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BaseAction_MRVector2f() {Dispose(false);}

            // tool movement parsed from gcode
            public unsafe MR.Std.Const_Vector_MRVector2f Path
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_Get_path", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_Get_path(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector2f_Get_path(_UnderlyingPtr), is_owning: false);
                }
            }

            // parser warning
            public unsafe MR.Std.Const_String Warning
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_Get_warning", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_Get_warning(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector2f_Get_warning(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BaseAction_MRVector2f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector2f_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::BaseAction<MR::Vector2f>` elementwise.
            public unsafe Const_BaseAction_MRVector2f(MR.Std._ByValue_Vector_MRVector2f path, ReadOnlySpan<char> warning) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFrom(MR.Misc._PassBy path_pass_by, MR.Std.Vector_MRVector2f._Underlying *path, byte *warning, byte *warning_end);
                byte[] __bytes_warning = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warning.Length)];
                int __len_warning = System.Text.Encoding.UTF8.GetBytes(warning, __bytes_warning);
                fixed (byte *__ptr_warning = __bytes_warning)
                {
                    _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFrom(path.PassByMode, path.Value is not null ? path.Value._UnderlyingPtr : null, __ptr_warning, __ptr_warning + __len_warning);
                }
            }

            /// Generated from constructor `MR::GcodeProcessor::BaseAction<MR::Vector2f>::BaseAction`.
            public unsafe Const_BaseAction_MRVector2f(MR.GcodeProcessor._ByValue_BaseAction_MRVector2f _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::GcodeProcessor::BaseAction<MR::Vector2f>`.
        /// This is the non-const half of the class.
        public class BaseAction_MRVector2f : Const_BaseAction_MRVector2f
        {
            internal unsafe BaseAction_MRVector2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // tool movement parsed from gcode
            public new unsafe MR.Std.Vector_MRVector2f Path
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_GetMutable_path", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_GetMutable_path(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector2f_GetMutable_path(_UnderlyingPtr), is_owning: false);
                }
            }

            // parser warning
            public new unsafe MR.Std.String Warning
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_GetMutable_warning", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_GetMutable_warning(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector2f_GetMutable_warning(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BaseAction_MRVector2f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector2f_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::BaseAction<MR::Vector2f>` elementwise.
            public unsafe BaseAction_MRVector2f(MR.Std._ByValue_Vector_MRVector2f path, ReadOnlySpan<char> warning) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFrom(MR.Misc._PassBy path_pass_by, MR.Std.Vector_MRVector2f._Underlying *path, byte *warning, byte *warning_end);
                byte[] __bytes_warning = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warning.Length)];
                int __len_warning = System.Text.Encoding.UTF8.GetBytes(warning, __bytes_warning);
                fixed (byte *__ptr_warning = __bytes_warning)
                {
                    _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFrom(path.PassByMode, path.Value is not null ? path.Value._UnderlyingPtr : null, __ptr_warning, __ptr_warning + __len_warning);
                }
            }

            /// Generated from constructor `MR::GcodeProcessor::BaseAction<MR::Vector2f>::BaseAction`.
            public unsafe BaseAction_MRVector2f(MR.GcodeProcessor._ByValue_BaseAction_MRVector2f _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector2f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::GcodeProcessor::BaseAction<MR::Vector2f>::operator=`.
            public unsafe MR.GcodeProcessor.BaseAction_MRVector2f Assign(MR.GcodeProcessor._ByValue_BaseAction_MRVector2f _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector2f_AssignFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector2f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.BaseAction_MRVector2f._Underlying *_other);
                return new(__MR_GcodeProcessor_BaseAction_MR_Vector2f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `BaseAction_MRVector2f` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `BaseAction_MRVector2f`/`Const_BaseAction_MRVector2f` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_BaseAction_MRVector2f
        {
            internal readonly Const_BaseAction_MRVector2f? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_BaseAction_MRVector2f() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_BaseAction_MRVector2f(Const_BaseAction_MRVector2f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_BaseAction_MRVector2f(Const_BaseAction_MRVector2f arg) {return new(arg);}
            public _ByValue_BaseAction_MRVector2f(MR.Misc._Moved<BaseAction_MRVector2f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_BaseAction_MRVector2f(MR.Misc._Moved<BaseAction_MRVector2f> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `BaseAction_MRVector2f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BaseAction_MRVector2f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BaseAction_MRVector2f`/`Const_BaseAction_MRVector2f` directly.
        public class _InOptMut_BaseAction_MRVector2f
        {
            public BaseAction_MRVector2f? Opt;

            public _InOptMut_BaseAction_MRVector2f() {}
            public _InOptMut_BaseAction_MRVector2f(BaseAction_MRVector2f value) {Opt = value;}
            public static implicit operator _InOptMut_BaseAction_MRVector2f(BaseAction_MRVector2f value) {return new(value);}
        }

        /// This is used for optional parameters of class `BaseAction_MRVector2f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BaseAction_MRVector2f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BaseAction_MRVector2f`/`Const_BaseAction_MRVector2f` to pass it to the function.
        public class _InOptConst_BaseAction_MRVector2f
        {
            public Const_BaseAction_MRVector2f? Opt;

            public _InOptConst_BaseAction_MRVector2f() {}
            public _InOptConst_BaseAction_MRVector2f(Const_BaseAction_MRVector2f value) {Opt = value;}
            public static implicit operator _InOptConst_BaseAction_MRVector2f(Const_BaseAction_MRVector2f value) {return new(value);}
        }

        /// Generated from class `MR::GcodeProcessor::BaseAction<MR::Vector3f>`.
        /// This is the const half of the class.
        public class Const_BaseAction_MRVector3f : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BaseAction_MRVector3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_Destroy", ExactSpelling = true)]
                extern static void __MR_GcodeProcessor_BaseAction_MR_Vector3f_Destroy(_Underlying *_this);
                __MR_GcodeProcessor_BaseAction_MR_Vector3f_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BaseAction_MRVector3f() {Dispose(false);}

            // tool movement parsed from gcode
            public unsafe MR.Std.Const_Vector_MRVector3f Path
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_Get_path", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_Get_path(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector3f_Get_path(_UnderlyingPtr), is_owning: false);
                }
            }

            // parser warning
            public unsafe MR.Std.Const_String Warning
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_Get_warning", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_Get_warning(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector3f_Get_warning(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BaseAction_MRVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector3f_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::BaseAction<MR::Vector3f>` elementwise.
            public unsafe Const_BaseAction_MRVector3f(MR.Std._ByValue_Vector_MRVector3f path, ReadOnlySpan<char> warning) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFrom(MR.Misc._PassBy path_pass_by, MR.Std.Vector_MRVector3f._Underlying *path, byte *warning, byte *warning_end);
                byte[] __bytes_warning = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warning.Length)];
                int __len_warning = System.Text.Encoding.UTF8.GetBytes(warning, __bytes_warning);
                fixed (byte *__ptr_warning = __bytes_warning)
                {
                    _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFrom(path.PassByMode, path.Value is not null ? path.Value._UnderlyingPtr : null, __ptr_warning, __ptr_warning + __len_warning);
                }
            }

            /// Generated from constructor `MR::GcodeProcessor::BaseAction<MR::Vector3f>::BaseAction`.
            public unsafe Const_BaseAction_MRVector3f(MR.GcodeProcessor._ByValue_BaseAction_MRVector3f _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::GcodeProcessor::BaseAction<MR::Vector3f>`.
        /// This is the non-const half of the class.
        public class BaseAction_MRVector3f : Const_BaseAction_MRVector3f
        {
            internal unsafe BaseAction_MRVector3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // tool movement parsed from gcode
            public new unsafe MR.Std.Vector_MRVector3f Path
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_GetMutable_path", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_GetMutable_path(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector3f_GetMutable_path(_UnderlyingPtr), is_owning: false);
                }
            }

            // parser warning
            public new unsafe MR.Std.String Warning
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_GetMutable_warning", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_GetMutable_warning(_Underlying *_this);
                    return new(__MR_GcodeProcessor_BaseAction_MR_Vector3f_GetMutable_warning(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BaseAction_MRVector3f() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector3f_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::BaseAction<MR::Vector3f>` elementwise.
            public unsafe BaseAction_MRVector3f(MR.Std._ByValue_Vector_MRVector3f path, ReadOnlySpan<char> warning) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFrom(MR.Misc._PassBy path_pass_by, MR.Std.Vector_MRVector3f._Underlying *path, byte *warning, byte *warning_end);
                byte[] __bytes_warning = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warning.Length)];
                int __len_warning = System.Text.Encoding.UTF8.GetBytes(warning, __bytes_warning);
                fixed (byte *__ptr_warning = __bytes_warning)
                {
                    _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFrom(path.PassByMode, path.Value is not null ? path.Value._UnderlyingPtr : null, __ptr_warning, __ptr_warning + __len_warning);
                }
            }

            /// Generated from constructor `MR::GcodeProcessor::BaseAction<MR::Vector3f>::BaseAction`.
            public unsafe BaseAction_MRVector3f(MR.GcodeProcessor._ByValue_BaseAction_MRVector3f _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_BaseAction_MR_Vector3f_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::GcodeProcessor::BaseAction<MR::Vector3f>::operator=`.
            public unsafe MR.GcodeProcessor.BaseAction_MRVector3f Assign(MR.GcodeProcessor._ByValue_BaseAction_MRVector3f _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_BaseAction_MR_Vector3f_AssignFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_BaseAction_MR_Vector3f_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *_other);
                return new(__MR_GcodeProcessor_BaseAction_MR_Vector3f_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `BaseAction_MRVector3f` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `BaseAction_MRVector3f`/`Const_BaseAction_MRVector3f` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_BaseAction_MRVector3f
        {
            internal readonly Const_BaseAction_MRVector3f? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_BaseAction_MRVector3f() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_BaseAction_MRVector3f(Const_BaseAction_MRVector3f new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_BaseAction_MRVector3f(Const_BaseAction_MRVector3f arg) {return new(arg);}
            public _ByValue_BaseAction_MRVector3f(MR.Misc._Moved<BaseAction_MRVector3f> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_BaseAction_MRVector3f(MR.Misc._Moved<BaseAction_MRVector3f> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `BaseAction_MRVector3f` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BaseAction_MRVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BaseAction_MRVector3f`/`Const_BaseAction_MRVector3f` directly.
        public class _InOptMut_BaseAction_MRVector3f
        {
            public BaseAction_MRVector3f? Opt;

            public _InOptMut_BaseAction_MRVector3f() {}
            public _InOptMut_BaseAction_MRVector3f(BaseAction_MRVector3f value) {Opt = value;}
            public static implicit operator _InOptMut_BaseAction_MRVector3f(BaseAction_MRVector3f value) {return new(value);}
        }

        /// This is used for optional parameters of class `BaseAction_MRVector3f` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BaseAction_MRVector3f`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BaseAction_MRVector3f`/`Const_BaseAction_MRVector3f` to pass it to the function.
        public class _InOptConst_BaseAction_MRVector3f
        {
            public Const_BaseAction_MRVector3f? Opt;

            public _InOptConst_BaseAction_MRVector3f() {}
            public _InOptConst_BaseAction_MRVector3f(Const_BaseAction_MRVector3f value) {Opt = value;}
            public static implicit operator _InOptConst_BaseAction_MRVector3f(Const_BaseAction_MRVector3f value) {return new(value);}
        }

        /// Generated from class `MR::GcodeProcessor::Command`.
        /// This is the const half of the class.
        public class Const_Command : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Command(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_Destroy", ExactSpelling = true)]
                extern static void __MR_GcodeProcessor_Command_Destroy(_Underlying *_this);
                __MR_GcodeProcessor_Command_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Command() {Dispose(false);}

            // in lowercase
            public unsafe byte Key
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_Get_key", ExactSpelling = true)]
                    extern static byte *__MR_GcodeProcessor_Command_Get_key(_Underlying *_this);
                    return *__MR_GcodeProcessor_Command_Get_key(_UnderlyingPtr);
                }
            }

            public unsafe float Value
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_Get_value", ExactSpelling = true)]
                    extern static float *__MR_GcodeProcessor_Command_Get_value(_Underlying *_this);
                    return *__MR_GcodeProcessor_Command_Get_value(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Command() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_Command_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::Command` elementwise.
            public unsafe Const_Command(byte key, float value) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_ConstructFrom(byte key, float value);
                _UnderlyingPtr = __MR_GcodeProcessor_Command_ConstructFrom(key, value);
            }

            /// Generated from constructor `MR::GcodeProcessor::Command::Command`.
            public unsafe Const_Command(MR.GcodeProcessor.Const_Command _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_ConstructFromAnother(MR.GcodeProcessor.Command._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_Command_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::GcodeProcessor::Command`.
        /// This is the non-const half of the class.
        public class Command : Const_Command
        {
            internal unsafe Command(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // in lowercase
            public new unsafe ref byte Key
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_GetMutable_key", ExactSpelling = true)]
                    extern static byte *__MR_GcodeProcessor_Command_GetMutable_key(_Underlying *_this);
                    return ref *__MR_GcodeProcessor_Command_GetMutable_key(_UnderlyingPtr);
                }
            }

            public new unsafe ref float Value
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_GetMutable_value", ExactSpelling = true)]
                    extern static float *__MR_GcodeProcessor_Command_GetMutable_value(_Underlying *_this);
                    return ref *__MR_GcodeProcessor_Command_GetMutable_value(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Command() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_Command_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::Command` elementwise.
            public unsafe Command(byte key, float value) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_ConstructFrom(byte key, float value);
                _UnderlyingPtr = __MR_GcodeProcessor_Command_ConstructFrom(key, value);
            }

            /// Generated from constructor `MR::GcodeProcessor::Command::Command`.
            public unsafe Command(MR.GcodeProcessor.Const_Command _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_ConstructFromAnother(MR.GcodeProcessor.Command._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_Command_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::GcodeProcessor::Command::operator=`.
            public unsafe MR.GcodeProcessor.Command Assign(MR.GcodeProcessor.Const_Command _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_Command_AssignFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.Command._Underlying *__MR_GcodeProcessor_Command_AssignFromAnother(_Underlying *_this, MR.GcodeProcessor.Command._Underlying *_other);
                return new(__MR_GcodeProcessor_Command_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Command` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Command`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Command`/`Const_Command` directly.
        public class _InOptMut_Command
        {
            public Command? Opt;

            public _InOptMut_Command() {}
            public _InOptMut_Command(Command value) {Opt = value;}
            public static implicit operator _InOptMut_Command(Command value) {return new(value);}
        }

        /// This is used for optional parameters of class `Command` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Command`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Command`/`Const_Command` to pass it to the function.
        public class _InOptConst_Command
        {
            public Const_Command? Opt;

            public _InOptConst_Command() {}
            public _InOptConst_Command(Const_Command value) {Opt = value;}
            public static implicit operator _InOptConst_Command(Const_Command value) {return new(value);}
        }

        // structure that stores information about the movement of the tool, specified by some string of commands
        /// Generated from class `MR::GcodeProcessor::MoveAction`.
        /// This is the const half of the class.
        public class Const_MoveAction : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_MoveAction(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_Destroy", ExactSpelling = true)]
                extern static void __MR_GcodeProcessor_MoveAction_Destroy(_Underlying *_this);
                __MR_GcodeProcessor_MoveAction_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_MoveAction() {Dispose(false);}

            public unsafe MR.GcodeProcessor.Const_BaseAction_MRVector3f Action
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_Get_action", ExactSpelling = true)]
                    extern static MR.GcodeProcessor.Const_BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_MoveAction_Get_action(_Underlying *_this);
                    return new(__MR_GcodeProcessor_MoveAction_Get_action(_UnderlyingPtr), is_owning: false);
                }
            }

            // tool direction for each point from action.path
            public unsafe MR.Std.Const_Vector_MRVector3f ToolDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_Get_toolDirection", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRVector3f._Underlying *__MR_GcodeProcessor_MoveAction_Get_toolDirection(_Underlying *_this);
                    return new(__MR_GcodeProcessor_MoveAction_Get_toolDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe bool Idle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_Get_idle", ExactSpelling = true)]
                    extern static bool *__MR_GcodeProcessor_MoveAction_Get_idle(_Underlying *_this);
                    return *__MR_GcodeProcessor_MoveAction_Get_idle(_UnderlyingPtr);
                }
            }

            public unsafe float Feedrate
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_Get_feedrate", ExactSpelling = true)]
                    extern static float *__MR_GcodeProcessor_MoveAction_Get_feedrate(_Underlying *_this);
                    return *__MR_GcodeProcessor_MoveAction_Get_feedrate(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_MoveAction() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_MoveAction_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::MoveAction` elementwise.
            public unsafe Const_MoveAction(MR.GcodeProcessor._ByValue_BaseAction_MRVector3f action, MR.Std._ByValue_Vector_MRVector3f toolDirection, bool idle, float feedrate) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_ConstructFrom(MR.Misc._PassBy action_pass_by, MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *action, MR.Misc._PassBy toolDirection_pass_by, MR.Std.Vector_MRVector3f._Underlying *toolDirection, byte idle, float feedrate);
                _UnderlyingPtr = __MR_GcodeProcessor_MoveAction_ConstructFrom(action.PassByMode, action.Value is not null ? action.Value._UnderlyingPtr : null, toolDirection.PassByMode, toolDirection.Value is not null ? toolDirection.Value._UnderlyingPtr : null, idle ? (byte)1 : (byte)0, feedrate);
            }

            /// Generated from constructor `MR::GcodeProcessor::MoveAction::MoveAction`.
            public unsafe Const_MoveAction(MR.GcodeProcessor._ByValue_MoveAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.MoveAction._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_MoveAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from conversion operator `MR::GcodeProcessor::MoveAction::operator bool`.
            public static unsafe implicit operator bool(MR.GcodeProcessor.Const_MoveAction _this)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_ConvertTo_bool", ExactSpelling = true)]
                extern static byte __MR_GcodeProcessor_MoveAction_ConvertTo_bool(MR.GcodeProcessor.Const_MoveAction._Underlying *_this);
                return __MR_GcodeProcessor_MoveAction_ConvertTo_bool(_this._UnderlyingPtr) != 0;
            }

            // return true if operation was parsed without warnings
            /// Generated from method `MR::GcodeProcessor::MoveAction::valid`.
            public unsafe bool Valid()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_valid", ExactSpelling = true)]
                extern static byte __MR_GcodeProcessor_MoveAction_valid(_Underlying *_this);
                return __MR_GcodeProcessor_MoveAction_valid(_UnderlyingPtr) != 0;
            }
        }

        // structure that stores information about the movement of the tool, specified by some string of commands
        /// Generated from class `MR::GcodeProcessor::MoveAction`.
        /// This is the non-const half of the class.
        public class MoveAction : Const_MoveAction
        {
            internal unsafe MoveAction(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.GcodeProcessor.BaseAction_MRVector3f Action
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_GetMutable_action", ExactSpelling = true)]
                    extern static MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *__MR_GcodeProcessor_MoveAction_GetMutable_action(_Underlying *_this);
                    return new(__MR_GcodeProcessor_MoveAction_GetMutable_action(_UnderlyingPtr), is_owning: false);
                }
            }

            // tool direction for each point from action.path
            public new unsafe MR.Std.Vector_MRVector3f ToolDirection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_GetMutable_toolDirection", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVector3f._Underlying *__MR_GcodeProcessor_MoveAction_GetMutable_toolDirection(_Underlying *_this);
                    return new(__MR_GcodeProcessor_MoveAction_GetMutable_toolDirection(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe ref bool Idle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_GetMutable_idle", ExactSpelling = true)]
                    extern static bool *__MR_GcodeProcessor_MoveAction_GetMutable_idle(_Underlying *_this);
                    return ref *__MR_GcodeProcessor_MoveAction_GetMutable_idle(_UnderlyingPtr);
                }
            }

            public new unsafe ref float Feedrate
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_GetMutable_feedrate", ExactSpelling = true)]
                    extern static float *__MR_GcodeProcessor_MoveAction_GetMutable_feedrate(_Underlying *_this);
                    return ref *__MR_GcodeProcessor_MoveAction_GetMutable_feedrate(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe MoveAction() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_DefaultConstruct", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_DefaultConstruct();
                _UnderlyingPtr = __MR_GcodeProcessor_MoveAction_DefaultConstruct();
            }

            /// Constructs `MR::GcodeProcessor::MoveAction` elementwise.
            public unsafe MoveAction(MR.GcodeProcessor._ByValue_BaseAction_MRVector3f action, MR.Std._ByValue_Vector_MRVector3f toolDirection, bool idle, float feedrate) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_ConstructFrom", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_ConstructFrom(MR.Misc._PassBy action_pass_by, MR.GcodeProcessor.BaseAction_MRVector3f._Underlying *action, MR.Misc._PassBy toolDirection_pass_by, MR.Std.Vector_MRVector3f._Underlying *toolDirection, byte idle, float feedrate);
                _UnderlyingPtr = __MR_GcodeProcessor_MoveAction_ConstructFrom(action.PassByMode, action.Value is not null ? action.Value._UnderlyingPtr : null, toolDirection.PassByMode, toolDirection.Value is not null ? toolDirection.Value._UnderlyingPtr : null, idle ? (byte)1 : (byte)0, feedrate);
            }

            /// Generated from constructor `MR::GcodeProcessor::MoveAction::MoveAction`.
            public unsafe MoveAction(MR.GcodeProcessor._ByValue_MoveAction _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.MoveAction._Underlying *_other);
                _UnderlyingPtr = __MR_GcodeProcessor_MoveAction_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::GcodeProcessor::MoveAction::operator=`.
            public unsafe MR.GcodeProcessor.MoveAction Assign(MR.GcodeProcessor._ByValue_MoveAction _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_MoveAction_AssignFromAnother", ExactSpelling = true)]
                extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_MoveAction_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor.MoveAction._Underlying *_other);
                return new(__MR_GcodeProcessor_MoveAction_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `MoveAction` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `MoveAction`/`Const_MoveAction` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_MoveAction
        {
            internal readonly Const_MoveAction? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_MoveAction() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_MoveAction(Const_MoveAction new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_MoveAction(Const_MoveAction arg) {return new(arg);}
            public _ByValue_MoveAction(MR.Misc._Moved<MoveAction> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_MoveAction(MR.Misc._Moved<MoveAction> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `MoveAction` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_MoveAction`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `MoveAction`/`Const_MoveAction` directly.
        public class _InOptMut_MoveAction
        {
            public MoveAction? Opt;

            public _InOptMut_MoveAction() {}
            public _InOptMut_MoveAction(MoveAction value) {Opt = value;}
            public static implicit operator _InOptMut_MoveAction(MoveAction value) {return new(value);}
        }

        /// This is used for optional parameters of class `MoveAction` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_MoveAction`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `MoveAction`/`Const_MoveAction` to pass it to the function.
        public class _InOptConst_MoveAction
        {
            public Const_MoveAction? Opt;

            public _InOptConst_MoveAction() {}
            public _InOptConst_MoveAction(Const_MoveAction value) {Opt = value;}
            public static implicit operator _InOptConst_MoveAction(Const_MoveAction value) {return new(value);}
        }
    }

    // class to process g-code source and generate toolpath
    /// Generated from class `MR::GcodeProcessor`.
    /// This is the non-const half of the class.
    public class GcodeProcessor : Const_GcodeProcessor
    {
        internal unsafe GcodeProcessor(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe GcodeProcessor() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_DefaultConstruct", ExactSpelling = true)]
            extern static MR.GcodeProcessor._Underlying *__MR_GcodeProcessor_DefaultConstruct();
            _UnderlyingPtr = __MR_GcodeProcessor_DefaultConstruct();
        }

        /// Generated from constructor `MR::GcodeProcessor::GcodeProcessor`.
        public unsafe GcodeProcessor(MR._ByValue_GcodeProcessor _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.GcodeProcessor._Underlying *__MR_GcodeProcessor_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor._Underlying *_other);
            _UnderlyingPtr = __MR_GcodeProcessor_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::GcodeProcessor::operator=`.
        public unsafe MR.GcodeProcessor Assign(MR._ByValue_GcodeProcessor _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_AssignFromAnother", ExactSpelling = true)]
            extern static MR.GcodeProcessor._Underlying *__MR_GcodeProcessor_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.GcodeProcessor._Underlying *_other);
            return new(__MR_GcodeProcessor_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        // reset internal states
        /// Generated from method `MR::GcodeProcessor::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_reset", ExactSpelling = true)]
            extern static void __MR_GcodeProcessor_reset(_Underlying *_this);
            __MR_GcodeProcessor_reset(_UnderlyingPtr);
        }

        // set g-code source
        /// Generated from method `MR::GcodeProcessor::setGcodeSource`.
        public unsafe void SetGcodeSource(MR.Std.Const_Vector_StdString gcodeSource)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_setGcodeSource", ExactSpelling = true)]
            extern static void __MR_GcodeProcessor_setGcodeSource(_Underlying *_this, MR.Std.Const_Vector_StdString._Underlying *gcodeSource);
            __MR_GcodeProcessor_setGcodeSource(_UnderlyingPtr, gcodeSource._UnderlyingPtr);
        }

        // process all lines g-code source and generate corresponding move actions
        /// Generated from method `MR::GcodeProcessor::processSource`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRGcodeProcessorMoveAction> ProcessSource()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_processSource", ExactSpelling = true)]
            extern static MR.Std.Vector_MRGcodeProcessorMoveAction._Underlying *__MR_GcodeProcessor_processSource(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRGcodeProcessorMoveAction(__MR_GcodeProcessor_processSource(_UnderlyingPtr), is_owning: true));
        }

        // process all commands from one line g-code source and generate corresponding move action;
        // \param externalStorage to avoid memory allocation on each line
        /// Generated from method `MR::GcodeProcessor::processLine`.
        public unsafe MR.Misc._Moved<MR.GcodeProcessor.MoveAction> ProcessLine(ReadOnlySpan<char> line, MR.Std.Vector_MRGcodeProcessorCommand externalStorage)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_processLine", ExactSpelling = true)]
            extern static MR.GcodeProcessor.MoveAction._Underlying *__MR_GcodeProcessor_processLine(_Underlying *_this, byte *line, byte *line_end, MR.Std.Vector_MRGcodeProcessorCommand._Underlying *externalStorage);
            byte[] __bytes_line = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(line.Length)];
            int __len_line = System.Text.Encoding.UTF8.GetBytes(line, __bytes_line);
            fixed (byte *__ptr_line = __bytes_line)
            {
                return MR.Misc.Move(new MR.GcodeProcessor.MoveAction(__MR_GcodeProcessor_processLine(_UnderlyingPtr, __ptr_line, __ptr_line + __len_line, externalStorage._UnderlyingPtr), is_owning: true));
            }
        }

        // settings
        /// Generated from method `MR::GcodeProcessor::setCNCMachineSettings`.
        public unsafe void SetCNCMachineSettings(MR.Const_CNCMachineSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_setCNCMachineSettings", ExactSpelling = true)]
            extern static void __MR_GcodeProcessor_setCNCMachineSettings(_Underlying *_this, MR.Const_CNCMachineSettings._Underlying *settings);
            __MR_GcodeProcessor_setCNCMachineSettings(_UnderlyingPtr, settings._UnderlyingPtr);
        }

        /// Generated from method `MR::GcodeProcessor::getCNCMachineSettings`.
        public unsafe MR.Const_CNCMachineSettings GetCNCMachineSettings()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_GcodeProcessor_getCNCMachineSettings", ExactSpelling = true)]
            extern static MR.Const_CNCMachineSettings._Underlying *__MR_GcodeProcessor_getCNCMachineSettings(_Underlying *_this);
            return new(__MR_GcodeProcessor_getCNCMachineSettings(_UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `GcodeProcessor` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `GcodeProcessor`/`Const_GcodeProcessor` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_GcodeProcessor
    {
        internal readonly Const_GcodeProcessor? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_GcodeProcessor() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_GcodeProcessor(Const_GcodeProcessor new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_GcodeProcessor(Const_GcodeProcessor arg) {return new(arg);}
        public _ByValue_GcodeProcessor(MR.Misc._Moved<GcodeProcessor> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_GcodeProcessor(MR.Misc._Moved<GcodeProcessor> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `GcodeProcessor` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_GcodeProcessor`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GcodeProcessor`/`Const_GcodeProcessor` directly.
    public class _InOptMut_GcodeProcessor
    {
        public GcodeProcessor? Opt;

        public _InOptMut_GcodeProcessor() {}
        public _InOptMut_GcodeProcessor(GcodeProcessor value) {Opt = value;}
        public static implicit operator _InOptMut_GcodeProcessor(GcodeProcessor value) {return new(value);}
    }

    /// This is used for optional parameters of class `GcodeProcessor` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_GcodeProcessor`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `GcodeProcessor`/`Const_GcodeProcessor` to pass it to the function.
    public class _InOptConst_GcodeProcessor
    {
        public Const_GcodeProcessor? Opt;

        public _InOptConst_GcodeProcessor() {}
        public _InOptConst_GcodeProcessor(Const_GcodeProcessor value) {Opt = value;}
        public static implicit operator _InOptConst_GcodeProcessor(Const_GcodeProcessor value) {return new(value);}
    }
}
