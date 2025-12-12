public static partial class MR
{
    // structure with settings and side output parameters for loading point cloud
    /// Generated from class `MR::PointsLoadSettings`.
    /// This is the const half of the class.
    public class Const_PointsLoadSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointsLoadSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_PointsLoadSettings_Destroy(_Underlying *_this);
            __MR_PointsLoadSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointsLoadSettings() {Dispose(false);}

        ///< points where to load point color map
        public unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_Get_colors", ExactSpelling = true)]
                extern static void **__MR_PointsLoadSettings_Get_colors(_Underlying *_this);
                return ref *__MR_PointsLoadSettings_Get_colors(_UnderlyingPtr);
            }
        }

        ///< transform for the loaded point cloud
        public unsafe ref MR.AffineXf3f * OutXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_Get_outXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_PointsLoadSettings_Get_outXf(_Underlying *_this);
                return ref *__MR_PointsLoadSettings_Get_outXf(_UnderlyingPtr);
            }
        }

        ///< callback for set progress and stop process
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_PointsLoadSettings_Get_callback(_Underlying *_this);
                return new(__MR_PointsLoadSettings_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointsLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::PointsLoadSettings` elementwise.
        public unsafe Const_PointsLoadSettings(MR.VertColors? colors, MR.Mut_AffineXf3f? outXf, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_ConstructFrom(MR.VertColors._Underlying *colors, MR.Mut_AffineXf3f._Underlying *outXf, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_PointsLoadSettings_ConstructFrom(colors is not null ? colors._UnderlyingPtr : null, outXf is not null ? outXf._UnderlyingPtr : null, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PointsLoadSettings::PointsLoadSettings`.
        public unsafe Const_PointsLoadSettings(MR._ByValue_PointsLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_PointsLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    // structure with settings and side output parameters for loading point cloud
    /// Generated from class `MR::PointsLoadSettings`.
    /// This is the non-const half of the class.
    public class PointsLoadSettings : Const_PointsLoadSettings
    {
        internal unsafe PointsLoadSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< points where to load point color map
        public new unsafe ref void * Colors
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_GetMutable_colors", ExactSpelling = true)]
                extern static void **__MR_PointsLoadSettings_GetMutable_colors(_Underlying *_this);
                return ref *__MR_PointsLoadSettings_GetMutable_colors(_UnderlyingPtr);
            }
        }

        ///< transform for the loaded point cloud
        public new unsafe ref MR.AffineXf3f * OutXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_GetMutable_outXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_PointsLoadSettings_GetMutable_outXf(_Underlying *_this);
                return ref *__MR_PointsLoadSettings_GetMutable_outXf(_UnderlyingPtr);
            }
        }

        ///< callback for set progress and stop process
        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_PointsLoadSettings_GetMutable_callback(_Underlying *_this);
                return new(__MR_PointsLoadSettings_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointsLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::PointsLoadSettings` elementwise.
        public unsafe PointsLoadSettings(MR.VertColors? colors, MR.Mut_AffineXf3f? outXf, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_ConstructFrom(MR.VertColors._Underlying *colors, MR.Mut_AffineXf3f._Underlying *outXf, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_PointsLoadSettings_ConstructFrom(colors is not null ? colors._UnderlyingPtr : null, outXf is not null ? outXf._UnderlyingPtr : null, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PointsLoadSettings::PointsLoadSettings`.
        public unsafe PointsLoadSettings(MR._ByValue_PointsLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_PointsLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointsLoadSettings::operator=`.
        public unsafe MR.PointsLoadSettings Assign(MR._ByValue_PointsLoadSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsLoadSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointsLoadSettings._Underlying *__MR_PointsLoadSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsLoadSettings._Underlying *_other);
            return new(__MR_PointsLoadSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointsLoadSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointsLoadSettings`/`Const_PointsLoadSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointsLoadSettings
    {
        internal readonly Const_PointsLoadSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointsLoadSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointsLoadSettings(Const_PointsLoadSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointsLoadSettings(Const_PointsLoadSettings arg) {return new(arg);}
        public _ByValue_PointsLoadSettings(MR.Misc._Moved<PointsLoadSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointsLoadSettings(MR.Misc._Moved<PointsLoadSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointsLoadSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointsLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsLoadSettings`/`Const_PointsLoadSettings` directly.
    public class _InOptMut_PointsLoadSettings
    {
        public PointsLoadSettings? Opt;

        public _InOptMut_PointsLoadSettings() {}
        public _InOptMut_PointsLoadSettings(PointsLoadSettings value) {Opt = value;}
        public static implicit operator _InOptMut_PointsLoadSettings(PointsLoadSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointsLoadSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointsLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsLoadSettings`/`Const_PointsLoadSettings` to pass it to the function.
    public class _InOptConst_PointsLoadSettings
    {
        public Const_PointsLoadSettings? Opt;

        public _InOptConst_PointsLoadSettings() {}
        public _InOptConst_PointsLoadSettings(Const_PointsLoadSettings value) {Opt = value;}
        public static implicit operator _InOptConst_PointsLoadSettings(Const_PointsLoadSettings value) {return new(value);}
    }
}
