public static partial class MR
{
    /// setting for polyline loading from external format, and locations of optional output data
    /// Generated from class `MR::LinesLoadSettings`.
    /// This is the const half of the class.
    public class Const_LinesLoadSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_LinesLoadSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_LinesLoadSettings_Destroy(_Underlying *_this);
            __MR_LinesLoadSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_LinesLoadSettings() {Dispose(false);}

        ///< optional load artifact: per-vertex color map
        public unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_Get_colors", ExactSpelling = true)]
                extern static void **__MR_LinesLoadSettings_Get_colors(_Underlying *_this);
                return ref *__MR_LinesLoadSettings_Get_colors(_UnderlyingPtr);
            }
        }

        ///< callback for set progress and stop process
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_LinesLoadSettings_Get_callback(_Underlying *_this);
                return new(__MR_LinesLoadSettings_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_LinesLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_LinesLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::LinesLoadSettings` elementwise.
        public unsafe Const_LinesLoadSettings(MR.VertColors? colors, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_ConstructFrom(MR.VertColors._Underlying *colors, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_LinesLoadSettings_ConstructFrom(colors is not null ? colors._UnderlyingPtr : null, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::LinesLoadSettings::LinesLoadSettings`.
        public unsafe Const_LinesLoadSettings(MR._ByValue_LinesLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LinesLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_LinesLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// setting for polyline loading from external format, and locations of optional output data
    /// Generated from class `MR::LinesLoadSettings`.
    /// This is the non-const half of the class.
    public class LinesLoadSettings : Const_LinesLoadSettings
    {
        internal unsafe LinesLoadSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< optional load artifact: per-vertex color map
        public new unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_GetMutable_colors", ExactSpelling = true)]
                extern static void **__MR_LinesLoadSettings_GetMutable_colors(_Underlying *_this);
                return ref *__MR_LinesLoadSettings_GetMutable_colors(_UnderlyingPtr);
            }
        }

        ///< callback for set progress and stop process
        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_LinesLoadSettings_GetMutable_callback(_Underlying *_this);
                return new(__MR_LinesLoadSettings_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe LinesLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_LinesLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::LinesLoadSettings` elementwise.
        public unsafe LinesLoadSettings(MR.VertColors? colors, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_ConstructFrom(MR.VertColors._Underlying *colors, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_LinesLoadSettings_ConstructFrom(colors is not null ? colors._UnderlyingPtr : null, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::LinesLoadSettings::LinesLoadSettings`.
        public unsafe LinesLoadSettings(MR._ByValue_LinesLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.LinesLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_LinesLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::LinesLoadSettings::operator=`.
        public unsafe MR.LinesLoadSettings Assign(MR._ByValue_LinesLoadSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_LinesLoadSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.LinesLoadSettings._Underlying *__MR_LinesLoadSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.LinesLoadSettings._Underlying *_other);
            return new(__MR_LinesLoadSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `LinesLoadSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `LinesLoadSettings`/`Const_LinesLoadSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_LinesLoadSettings
    {
        internal readonly Const_LinesLoadSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_LinesLoadSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_LinesLoadSettings(Const_LinesLoadSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_LinesLoadSettings(Const_LinesLoadSettings arg) {return new(arg);}
        public _ByValue_LinesLoadSettings(MR.Misc._Moved<LinesLoadSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_LinesLoadSettings(MR.Misc._Moved<LinesLoadSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `LinesLoadSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_LinesLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LinesLoadSettings`/`Const_LinesLoadSettings` directly.
    public class _InOptMut_LinesLoadSettings
    {
        public LinesLoadSettings? Opt;

        public _InOptMut_LinesLoadSettings() {}
        public _InOptMut_LinesLoadSettings(LinesLoadSettings value) {Opt = value;}
        public static implicit operator _InOptMut_LinesLoadSettings(LinesLoadSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `LinesLoadSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_LinesLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `LinesLoadSettings`/`Const_LinesLoadSettings` to pass it to the function.
    public class _InOptConst_LinesLoadSettings
    {
        public Const_LinesLoadSettings? Opt;

        public _InOptConst_LinesLoadSettings() {}
        public _InOptConst_LinesLoadSettings(Const_LinesLoadSettings value) {Opt = value;}
        public static implicit operator _InOptConst_LinesLoadSettings(Const_LinesLoadSettings value) {return new(value);}
    }
}
