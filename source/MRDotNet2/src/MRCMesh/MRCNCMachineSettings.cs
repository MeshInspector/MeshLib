public static partial class MR
{
    /// class with CNC machine emulation settings
    /// Generated from class `MR::CNCMachineSettings`.
    /// This is the const half of the class.
    public class Const_CNCMachineSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_CNCMachineSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_CNCMachineSettings_Destroy(_Underlying *_this);
            __MR_CNCMachineSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_CNCMachineSettings() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_CNCMachineSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CNCMachineSettings._Underlying *__MR_CNCMachineSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_CNCMachineSettings_DefaultConstruct();
        }

        /// Generated from constructor `MR::CNCMachineSettings::CNCMachineSettings`.
        public unsafe Const_CNCMachineSettings(MR._ByValue_CNCMachineSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CNCMachineSettings._Underlying *__MR_CNCMachineSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CNCMachineSettings._Underlying *_other);
            _UnderlyingPtr = __MR_CNCMachineSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CNCMachineSettings::getAxesCount`.
        public static int GetAxesCount()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_getAxesCount", ExactSpelling = true)]
            extern static int __MR_CNCMachineSettings_getAxesCount();
            return __MR_CNCMachineSettings_getAxesCount();
        }

        /// Generated from method `MR::CNCMachineSettings::getRotationAxis`.
        public unsafe MR.Const_Vector3f GetRotationAxis(MR.CNCMachineSettings.RotationAxisName paramName)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_getRotationAxis", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_CNCMachineSettings_getRotationAxis(_Underlying *_this, MR.CNCMachineSettings.RotationAxisName paramName);
            return new(__MR_CNCMachineSettings_getRotationAxis(_UnderlyingPtr, paramName), is_owning: false);
        }

        /// Generated from method `MR::CNCMachineSettings::getRotationLimits`.
        public unsafe MR.Std.Const_Optional_MRVector2f GetRotationLimits(MR.CNCMachineSettings.RotationAxisName paramName)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_getRotationLimits", ExactSpelling = true)]
            extern static MR.Std.Const_Optional_MRVector2f._Underlying *__MR_CNCMachineSettings_getRotationLimits(_Underlying *_this, MR.CNCMachineSettings.RotationAxisName paramName);
            return new(__MR_CNCMachineSettings_getRotationLimits(_UnderlyingPtr, paramName), is_owning: false);
        }

        /// Generated from method `MR::CNCMachineSettings::getRotationOrder`.
        public unsafe MR.Std.Const_Vector_MRCNCMachineSettingsRotationAxisName GetRotationOrder()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_getRotationOrder", ExactSpelling = true)]
            extern static MR.Std.Const_Vector_MRCNCMachineSettingsRotationAxisName._Underlying *__MR_CNCMachineSettings_getRotationOrder(_Underlying *_this);
            return new(__MR_CNCMachineSettings_getRotationOrder(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::CNCMachineSettings::getFeedrateIdle`.
        public unsafe float GetFeedrateIdle()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_getFeedrateIdle", ExactSpelling = true)]
            extern static float __MR_CNCMachineSettings_getFeedrateIdle(_Underlying *_this);
            return __MR_CNCMachineSettings_getFeedrateIdle(_UnderlyingPtr);
        }

        /// Generated from method `MR::CNCMachineSettings::getHomePosition`.
        public unsafe MR.Const_Vector3f GetHomePosition()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_getHomePosition", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_CNCMachineSettings_getHomePosition(_Underlying *_this);
            return new(__MR_CNCMachineSettings_getHomePosition(_UnderlyingPtr), is_owning: false);
        }

        // enumeration of axes of rotation
        public enum RotationAxisName : int
        {
            A = 0,
            B = 1,
            C = 2,
        }
    }

    /// class with CNC machine emulation settings
    /// Generated from class `MR::CNCMachineSettings`.
    /// This is the non-const half of the class.
    public class CNCMachineSettings : Const_CNCMachineSettings, System.IEquatable<MR.Const_CNCMachineSettings>
    {
        internal unsafe CNCMachineSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe CNCMachineSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.CNCMachineSettings._Underlying *__MR_CNCMachineSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_CNCMachineSettings_DefaultConstruct();
        }

        /// Generated from constructor `MR::CNCMachineSettings::CNCMachineSettings`.
        public unsafe CNCMachineSettings(MR._ByValue_CNCMachineSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.CNCMachineSettings._Underlying *__MR_CNCMachineSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.CNCMachineSettings._Underlying *_other);
            _UnderlyingPtr = __MR_CNCMachineSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::CNCMachineSettings::operator=`.
        public unsafe MR.CNCMachineSettings Assign(MR._ByValue_CNCMachineSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.CNCMachineSettings._Underlying *__MR_CNCMachineSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.CNCMachineSettings._Underlying *_other);
            return new(__MR_CNCMachineSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        // rotationAxis length will be more then 0.01
        /// Generated from method `MR::CNCMachineSettings::setRotationAxis`.
        public unsafe void SetRotationAxis(MR.CNCMachineSettings.RotationAxisName paramName, MR.Const_Vector3f rotationAxis)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_setRotationAxis", ExactSpelling = true)]
            extern static void __MR_CNCMachineSettings_setRotationAxis(_Underlying *_this, MR.CNCMachineSettings.RotationAxisName paramName, MR.Const_Vector3f._Underlying *rotationAxis);
            __MR_CNCMachineSettings_setRotationAxis(_UnderlyingPtr, paramName, rotationAxis._UnderlyingPtr);
        }

        // rotationLimits = {min, max}
        // valid range -180 <= min < max <= 180
        /// Generated from method `MR::CNCMachineSettings::setRotationLimits`.
        public unsafe void SetRotationLimits(MR.CNCMachineSettings.RotationAxisName paramName, MR._InOpt_Vector2f rotationLimits)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_setRotationLimits", ExactSpelling = true)]
            extern static void __MR_CNCMachineSettings_setRotationLimits(_Underlying *_this, MR.CNCMachineSettings.RotationAxisName paramName, MR.Vector2f *rotationLimits);
            __MR_CNCMachineSettings_setRotationLimits(_UnderlyingPtr, paramName, rotationLimits.HasValue ? &rotationLimits.Object : null);
        }

        // duplicated values will be removed (ABAAC - > ABC)
        /// Generated from method `MR::CNCMachineSettings::setRotationOrder`.
        public unsafe void SetRotationOrder(MR.Std.Const_Vector_MRCNCMachineSettingsRotationAxisName rotationAxesOrder)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_setRotationOrder", ExactSpelling = true)]
            extern static void __MR_CNCMachineSettings_setRotationOrder(_Underlying *_this, MR.Std.Const_Vector_MRCNCMachineSettingsRotationAxisName._Underlying *rotationAxesOrder);
            __MR_CNCMachineSettings_setRotationOrder(_UnderlyingPtr, rotationAxesOrder._UnderlyingPtr);
        }

        // set feedrate idle. valid range - [0, 100000]
        // 0 - feedrate idle set as maximum feedrate on any action, or 100 if feedrate is not set in any action
        /// Generated from method `MR::CNCMachineSettings::setFeedrateIdle`.
        public unsafe void SetFeedrateIdle(float feedrateIdle)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_setFeedrateIdle", ExactSpelling = true)]
            extern static void __MR_CNCMachineSettings_setFeedrateIdle(_Underlying *_this, float feedrateIdle);
            __MR_CNCMachineSettings_setFeedrateIdle(_UnderlyingPtr, feedrateIdle);
        }

        /// Generated from method `MR::CNCMachineSettings::setHomePosition`.
        public unsafe void SetHomePosition(MR.Const_Vector3f homePosition)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_CNCMachineSettings_setHomePosition", ExactSpelling = true)]
            extern static void __MR_CNCMachineSettings_setHomePosition(_Underlying *_this, MR.Const_Vector3f._Underlying *homePosition);
            __MR_CNCMachineSettings_setHomePosition(_UnderlyingPtr, homePosition._UnderlyingPtr);
        }

        /// Generated from method `MR::CNCMachineSettings::operator==`.
        public static unsafe bool operator==(MR.CNCMachineSettings _this, MR.Const_CNCMachineSettings rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_CNCMachineSettings", ExactSpelling = true)]
            extern static byte __MR_equal_MR_CNCMachineSettings(MR.CNCMachineSettings._Underlying *_this, MR.Const_CNCMachineSettings._Underlying *rhs);
            return __MR_equal_MR_CNCMachineSettings(_this._UnderlyingPtr, rhs._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.CNCMachineSettings _this, MR.Const_CNCMachineSettings rhs)
        {
            return !(_this == rhs);
        }

        // IEquatable:

        public bool Equals(MR.Const_CNCMachineSettings? rhs)
        {
            if (rhs is null)
                return false;
            return this == rhs;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_CNCMachineSettings)
                return this == (MR.Const_CNCMachineSettings)other;
            return false;
        }
    }

    /// This is used as a function parameter when the underlying function receives `CNCMachineSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `CNCMachineSettings`/`Const_CNCMachineSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_CNCMachineSettings
    {
        internal readonly Const_CNCMachineSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_CNCMachineSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_CNCMachineSettings(Const_CNCMachineSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_CNCMachineSettings(Const_CNCMachineSettings arg) {return new(arg);}
        public _ByValue_CNCMachineSettings(MR.Misc._Moved<CNCMachineSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_CNCMachineSettings(MR.Misc._Moved<CNCMachineSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `CNCMachineSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_CNCMachineSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CNCMachineSettings`/`Const_CNCMachineSettings` directly.
    public class _InOptMut_CNCMachineSettings
    {
        public CNCMachineSettings? Opt;

        public _InOptMut_CNCMachineSettings() {}
        public _InOptMut_CNCMachineSettings(CNCMachineSettings value) {Opt = value;}
        public static implicit operator _InOptMut_CNCMachineSettings(CNCMachineSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `CNCMachineSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_CNCMachineSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `CNCMachineSettings`/`Const_CNCMachineSettings` to pass it to the function.
    public class _InOptConst_CNCMachineSettings
    {
        public Const_CNCMachineSettings? Opt;

        public _InOptConst_CNCMachineSettings() {}
        public _InOptConst_CNCMachineSettings(Const_CNCMachineSettings value) {Opt = value;}
        public static implicit operator _InOptConst_CNCMachineSettings(Const_CNCMachineSettings value) {return new(value);}
    }
}
