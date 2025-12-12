public static partial class MR
{
    /// This singleton struct contains default colors for scene objects
    /// Generated from class `MR::SceneColors`.
    /// This is the const half of the class.
    public class Const_SceneColors : MR.Misc.Object
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SceneColors(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        /// Generated from method `MR::SceneColors::get`.
        public static unsafe MR.Const_Color Get(MR.SceneColors.Type type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneColors_get", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_SceneColors_get(MR.SceneColors.Type type);
            return new(__MR_SceneColors_get(type), is_owning: false);
        }

        /// Generated from method `MR::SceneColors::set`.
        public static unsafe void Set(MR.SceneColors.Type type, MR.Const_Color color)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneColors_set", ExactSpelling = true)]
            extern static void __MR_SceneColors_set(MR.SceneColors.Type type, MR.Const_Color._Underlying *color);
            __MR_SceneColors_set(type, color._UnderlyingPtr);
        }

        /// Generated from method `MR::SceneColors::getName`.
        public static unsafe byte? GetName(MR.SceneColors.Type type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneColors_getName", ExactSpelling = true)]
            extern static byte *__MR_SceneColors_getName(MR.SceneColors.Type type);
            var __ret = __MR_SceneColors_getName(type);
            return __ret is not null ? *__ret : null;
        }

        public enum Type : int
        {
            SelectedObjectMesh = 0,
            UnselectedObjectMesh = 1,
            SelectedObjectPoints = 2,
            UnselectedObjectPoints = 3,
            SelectedObjectLines = 4,
            UnselectedObjectLines = 5,
            SelectedObjectVoxels = 6,
            UnselectedObjectVoxels = 7,
            SelectedObjectDistanceMap = 8,
            UnselectedObjectDistanceMap = 9,
            BackFaces = 10,
            Labels = 11,
            // Typically green.
            LabelsGood = 12,
            // Typically red.
            LabelsBad = 13,
            Edges = 14,
            Points = 15,
            SelectedFaces = 16,
            SelectedEdges = 17,
            SelectedPoints = 18,
            SelectedFeatures = 19,
            UnselectedFeatures = 20,
            FeatureBackFaces = 21,
            SelectedFeatureDecorations = 22,
            UnselectedFeatureDecorations = 23,
            SelectedMeasurements = 24,
            UnselectedMeasurements = 25,
            UnselectedMeasurementsX = 26,
            UnselectedMeasurementsY = 27,
            UnselectedMeasurementsZ = 28,
            SelectedTemporaryMeasurements = 29,
            UnselectedTemporaryMeasurements = 30,
            Count = 31,
        }
    }

    /// This singleton struct contains default colors for scene objects
    /// Generated from class `MR::SceneColors`.
    /// This is the non-const half of the class.
    public class SceneColors : Const_SceneColors
    {
        internal unsafe SceneColors(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}
    }

    /// This is used for optional parameters of class `SceneColors` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SceneColors`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneColors`/`Const_SceneColors` directly.
    public class _InOptMut_SceneColors
    {
        public SceneColors? Opt;

        public _InOptMut_SceneColors() {}
        public _InOptMut_SceneColors(SceneColors value) {Opt = value;}
        public static implicit operator _InOptMut_SceneColors(SceneColors value) {return new(value);}
    }

    /// This is used for optional parameters of class `SceneColors` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SceneColors`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SceneColors`/`Const_SceneColors` to pass it to the function.
    public class _InOptConst_SceneColors
    {
        public Const_SceneColors? Opt;

        public _InOptConst_SceneColors() {}
        public _InOptConst_SceneColors(Const_SceneColors value) {Opt = value;}
        public static implicit operator _InOptConst_SceneColors(Const_SceneColors value) {return new(value);}
    }
}
