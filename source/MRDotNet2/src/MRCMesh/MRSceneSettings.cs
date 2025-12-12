public static partial class MR
{
    /// This singleton struct contains default settings for scene objects
    /// Generated from class `MR::SceneSettings`.
    /// This is the const half of the class.
    public class Const_SceneSettings : MR.Misc.Object
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SceneSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        // Reset all scene settings to default values
        /// Generated from method `MR::SceneSettings::reset`.
        public static void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_reset", ExactSpelling = true)]
            extern static void __MR_SceneSettings_reset();
            __MR_SceneSettings_reset();
        }

        /// Generated from method `MR::SceneSettings::get`.
        public static bool Get(MR.SceneSettings.BoolType type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_get_MR_SceneSettings_BoolType", ExactSpelling = true)]
            extern static byte __MR_SceneSettings_get_MR_SceneSettings_BoolType(MR.SceneSettings.BoolType type);
            return __MR_SceneSettings_get_MR_SceneSettings_BoolType(type) != 0;
        }

        /// Generated from method `MR::SceneSettings::get`.
        public static float Get(MR.SceneSettings.FloatType type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_get_MR_SceneSettings_FloatType", ExactSpelling = true)]
            extern static float __MR_SceneSettings_get_MR_SceneSettings_FloatType(MR.SceneSettings.FloatType type);
            return __MR_SceneSettings_get_MR_SceneSettings_FloatType(type);
        }

        /// Generated from method `MR::SceneSettings::set`.
        public static void Set(MR.SceneSettings.BoolType type, bool value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_set_MR_SceneSettings_BoolType", ExactSpelling = true)]
            extern static void __MR_SceneSettings_set_MR_SceneSettings_BoolType(MR.SceneSettings.BoolType type, byte value);
            __MR_SceneSettings_set_MR_SceneSettings_BoolType(type, value ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::SceneSettings::set`.
        public static void Set(MR.SceneSettings.FloatType type, float value)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_set_MR_SceneSettings_FloatType", ExactSpelling = true)]
            extern static void __MR_SceneSettings_set_MR_SceneSettings_FloatType(MR.SceneSettings.FloatType type, float value);
            __MR_SceneSettings_set_MR_SceneSettings_FloatType(type, value);
        }

        /// Default shading mode for new mesh objects, or imported form files
        /// Tools may consider this setting when creating new meshes
        /// `AutoDetect`: choose depending of file format and mesh shape, fallback to smooth
        /// Generated from method `MR::SceneSettings::getDefaultShadingMode`.
        public static MR.SceneSettings.ShadingMode GetDefaultShadingMode()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_getDefaultShadingMode", ExactSpelling = true)]
            extern static MR.SceneSettings.ShadingMode __MR_SceneSettings_getDefaultShadingMode();
            return __MR_SceneSettings_getDefaultShadingMode();
        }

        /// Generated from method `MR::SceneSettings::setDefaultShadingMode`.
        public static void SetDefaultShadingMode(MR.SceneSettings.ShadingMode mode)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_setDefaultShadingMode", ExactSpelling = true)]
            extern static void __MR_SceneSettings_setDefaultShadingMode(MR.SceneSettings.ShadingMode mode);
            __MR_SceneSettings_setDefaultShadingMode(mode);
        }

        /// Generated from method `MR::SceneSettings::getCNCMachineSettings`.
        public static unsafe MR.Const_CNCMachineSettings GetCNCMachineSettings()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_getCNCMachineSettings", ExactSpelling = true)]
            extern static MR.Const_CNCMachineSettings._Underlying *__MR_SceneSettings_getCNCMachineSettings();
            return new(__MR_SceneSettings_getCNCMachineSettings(), is_owning: false);
        }

        /// Generated from method `MR::SceneSettings::setCNCMachineSettings`.
        public static unsafe void SetCNCMachineSettings(MR.Const_CNCMachineSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneSettings_setCNCMachineSettings", ExactSpelling = true)]
            extern static void __MR_SceneSettings_setCNCMachineSettings(MR.Const_CNCMachineSettings._Underlying *settings);
            __MR_SceneSettings_setCNCMachineSettings(settings._UnderlyingPtr);
        }

        public enum BoolType : int
        {
            /// on deserialization replace object properties with default values from SceneSettings and SceneColors
            UseDefaultScenePropertiesOnDeserialization = 0,
            /// total count
            Count = 1,
        }

        public enum FloatType : int
        {
            FeaturePointsAlpha = 0,
            FeatureLinesAlpha = 1,
            FeatureMeshAlpha = 2,
            FeatureSubPointsAlpha = 3,
            FeatureSubLinesAlpha = 4,
            FeatureSubMeshAlpha = 5,
            // Line width of line features (line, circle, ...).
            FeatureLineWidth = 6,
            // Line width of line subfeatures (axes, base circles, ...).
            FeatureSubLineWidth = 7,
            // Size of the point feature.
            FeaturePointSize = 8,
            // Size of point subfeatures (various centers).
            FeatureSubPointSize = 9,
            // Ambient multiplication coefficient for ambientStrength for selected objects
            AmbientCoefSelectedObj = 10,
            // Ambient multiplication coefficient for ambientStrength for selected objects
            Count = 11,
        }

        /// Mesh faces shading mode
        public enum ShadingMode : int
        {
            AutoDetect = 0,
            Smooth = 1,
            Flat = 2,
        }
    }

    /// This singleton struct contains default settings for scene objects
    /// Generated from class `MR::SceneSettings`.
    /// This is the non-const half of the class.
    public class SceneSettings : Const_SceneSettings
    {
        internal unsafe SceneSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}
    }

    /// This is used as a function parameter when the underlying function receives `SceneSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SceneSettings`/`Const_SceneSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SceneSettings
    {
        internal readonly Const_SceneSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SceneSettings(Const_SceneSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SceneSettings(Const_SceneSettings arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SceneSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SceneSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneSettings`/`Const_SceneSettings` directly.
    public class _InOptMut_SceneSettings
    {
        public SceneSettings? Opt;

        public _InOptMut_SceneSettings() {}
        public _InOptMut_SceneSettings(SceneSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SceneSettings(SceneSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SceneSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SceneSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneSettings`/`Const_SceneSettings` to pass it to the function.
    public class _InOptConst_SceneSettings
    {
        public Const_SceneSettings? Opt;

        public _InOptConst_SceneSettings() {}
        public _InOptConst_SceneSettings(Const_SceneSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SceneSettings(Const_SceneSettings value) {return new(value);}
    }
}
