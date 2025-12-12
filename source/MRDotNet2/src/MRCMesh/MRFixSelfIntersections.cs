public static partial class MR
{
    public static partial class SelfIntersections
    {
        /// Setting set for mesh self-intersections fix
        /// Generated from class `MR::SelfIntersections::Settings`.
        /// This is the const half of the class.
        public class Const_Settings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Settings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Destroy", ExactSpelling = true)]
                extern static void __MR_SelfIntersections_Settings_Destroy(_Underlying *_this);
                __MR_SelfIntersections_Settings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Settings() {Dispose(false);}

            /// If true then count touching faces as self-intersections
            public unsafe bool TouchIsIntersection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Get_touchIsIntersection", ExactSpelling = true)]
                    extern static bool *__MR_SelfIntersections_Settings_Get_touchIsIntersection(_Underlying *_this);
                    return *__MR_SelfIntersections_Settings_Get_touchIsIntersection(_UnderlyingPtr);
                }
            }

            public unsafe MR.SelfIntersections.Settings.Method Method_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Get_method", ExactSpelling = true)]
                    extern static MR.SelfIntersections.Settings.Method *__MR_SelfIntersections_Settings_Get_method(_Underlying *_this);
                    return *__MR_SelfIntersections_Settings_Get_method(_UnderlyingPtr);
                }
            }

            /// Maximum relax iterations
            public unsafe int RelaxIterations
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Get_relaxIterations", ExactSpelling = true)]
                    extern static int *__MR_SelfIntersections_Settings_Get_relaxIterations(_Underlying *_this);
                    return *__MR_SelfIntersections_Settings_Get_relaxIterations(_UnderlyingPtr);
                }
            }

            /// Maximum expand count (edge steps from self-intersecting faces), should be > 0
            public unsafe int MaxExpand
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Get_maxExpand", ExactSpelling = true)]
                    extern static int *__MR_SelfIntersections_Settings_Get_maxExpand(_Underlying *_this);
                    return *__MR_SelfIntersections_Settings_Get_maxExpand(_UnderlyingPtr);
                }
            }

            /// Edge length for subdivision of holes covers (0.0f means auto)
            /// FLT_MAX to disable subdivision
            public unsafe float SubdivideEdgeLen
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Get_subdivideEdgeLen", ExactSpelling = true)]
                    extern static float *__MR_SelfIntersections_Settings_Get_subdivideEdgeLen(_Underlying *_this);
                    return *__MR_SelfIntersections_Settings_Get_subdivideEdgeLen(_UnderlyingPtr);
                }
            }

            /// Callback function
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_Get_callback", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_SelfIntersections_Settings_Get_callback(_Underlying *_this);
                    return new(__MR_SelfIntersections_Settings_Get_callback(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_SelfIntersections_Settings_DefaultConstruct();
            }

            /// Constructs `MR::SelfIntersections::Settings` elementwise.
            public unsafe Const_Settings(bool touchIsIntersection, MR.SelfIntersections.Settings.Method method, int relaxIterations, int maxExpand, float subdivideEdgeLen, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_ConstructFrom(byte touchIsIntersection, MR.SelfIntersections.Settings.Method method, int relaxIterations, int maxExpand, float subdivideEdgeLen, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
                _UnderlyingPtr = __MR_SelfIntersections_Settings_ConstructFrom(touchIsIntersection ? (byte)1 : (byte)0, method, relaxIterations, maxExpand, subdivideEdgeLen, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::SelfIntersections::Settings::Settings`.
            public unsafe Const_Settings(MR.SelfIntersections._ByValue_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SelfIntersections.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_SelfIntersections_Settings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Fix method
            public enum Method : int
            {
                /// Relax mesh around self-intersections
                Relax = 0,
                /// Cut and re-fill regions around self-intersections (may fall back to `Relax`)
                CutAndFill = 1,
            }
        }

        /// Setting set for mesh self-intersections fix
        /// Generated from class `MR::SelfIntersections::Settings`.
        /// This is the non-const half of the class.
        public class Settings : Const_Settings
        {
            internal unsafe Settings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// If true then count touching faces as self-intersections
            public new unsafe ref bool TouchIsIntersection
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_GetMutable_touchIsIntersection", ExactSpelling = true)]
                    extern static bool *__MR_SelfIntersections_Settings_GetMutable_touchIsIntersection(_Underlying *_this);
                    return ref *__MR_SelfIntersections_Settings_GetMutable_touchIsIntersection(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.SelfIntersections.Settings.Method Method_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_GetMutable_method", ExactSpelling = true)]
                    extern static MR.SelfIntersections.Settings.Method *__MR_SelfIntersections_Settings_GetMutable_method(_Underlying *_this);
                    return ref *__MR_SelfIntersections_Settings_GetMutable_method(_UnderlyingPtr);
                }
            }

            /// Maximum relax iterations
            public new unsafe ref int RelaxIterations
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_GetMutable_relaxIterations", ExactSpelling = true)]
                    extern static int *__MR_SelfIntersections_Settings_GetMutable_relaxIterations(_Underlying *_this);
                    return ref *__MR_SelfIntersections_Settings_GetMutable_relaxIterations(_UnderlyingPtr);
                }
            }

            /// Maximum expand count (edge steps from self-intersecting faces), should be > 0
            public new unsafe ref int MaxExpand
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_GetMutable_maxExpand", ExactSpelling = true)]
                    extern static int *__MR_SelfIntersections_Settings_GetMutable_maxExpand(_Underlying *_this);
                    return ref *__MR_SelfIntersections_Settings_GetMutable_maxExpand(_UnderlyingPtr);
                }
            }

            /// Edge length for subdivision of holes covers (0.0f means auto)
            /// FLT_MAX to disable subdivision
            public new unsafe ref float SubdivideEdgeLen
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_GetMutable_subdivideEdgeLen", ExactSpelling = true)]
                    extern static float *__MR_SelfIntersections_Settings_GetMutable_subdivideEdgeLen(_Underlying *_this);
                    return ref *__MR_SelfIntersections_Settings_GetMutable_subdivideEdgeLen(_UnderlyingPtr);
                }
            }

            /// Callback function
            public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_GetMutable_callback", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_SelfIntersections_Settings_GetMutable_callback(_Underlying *_this);
                    return new(__MR_SelfIntersections_Settings_GetMutable_callback(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_SelfIntersections_Settings_DefaultConstruct();
            }

            /// Constructs `MR::SelfIntersections::Settings` elementwise.
            public unsafe Settings(bool touchIsIntersection, MR.SelfIntersections.Settings.Method method, int relaxIterations, int maxExpand, float subdivideEdgeLen, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_ConstructFrom(byte touchIsIntersection, MR.SelfIntersections.Settings.Method method, int relaxIterations, int maxExpand, float subdivideEdgeLen, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
                _UnderlyingPtr = __MR_SelfIntersections_Settings_ConstructFrom(touchIsIntersection ? (byte)1 : (byte)0, method, relaxIterations, maxExpand, subdivideEdgeLen, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::SelfIntersections::Settings::Settings`.
            public unsafe Settings(MR.SelfIntersections._ByValue_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SelfIntersections.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_SelfIntersections_Settings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::SelfIntersections::Settings::operator=`.
            public unsafe MR.SelfIntersections.Settings Assign(MR.SelfIntersections._ByValue_Settings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_Settings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.SelfIntersections.Settings._Underlying *__MR_SelfIntersections_Settings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SelfIntersections.Settings._Underlying *_other);
                return new(__MR_SelfIntersections_Settings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Settings` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Settings`/`Const_Settings` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Settings
        {
            internal readonly Const_Settings? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Settings() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Settings(Const_Settings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Settings(Const_Settings arg) {return new(arg);}
            public _ByValue_Settings(MR.Misc._Moved<Settings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Settings(MR.Misc._Moved<Settings> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Settings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Settings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Settings`/`Const_Settings` directly.
        public class _InOptMut_Settings
        {
            public Settings? Opt;

            public _InOptMut_Settings() {}
            public _InOptMut_Settings(Settings value) {Opt = value;}
            public static implicit operator _InOptMut_Settings(Settings value) {return new(value);}
        }

        /// This is used for optional parameters of class `Settings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Settings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Settings`/`Const_Settings` to pass it to the function.
        public class _InOptConst_Settings
        {
            public Const_Settings? Opt;

            public _InOptConst_Settings() {}
            public _InOptConst_Settings(Const_Settings value) {Opt = value;}
            public static implicit operator _InOptConst_Settings(Const_Settings value) {return new(value);}
        }

        /// Find all self-intersections faces component-wise
        /// Generated from function `MR::SelfIntersections::getFaces`.
        /// Parameter `touchIsIntersection` defaults to `true`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> GetFaces(MR.Const_Mesh mesh, bool? touchIsIntersection = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_getFaces", ExactSpelling = true)]
            extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_SelfIntersections_getFaces(MR.Const_Mesh._Underlying *mesh, byte *touchIsIntersection, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            byte __deref_touchIsIntersection = touchIsIntersection.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_SelfIntersections_getFaces(mesh._UnderlyingPtr, touchIsIntersection.HasValue ? &__deref_touchIsIntersection : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Finds and fixes self-intersections per component:
        /// Generated from function `MR::SelfIntersections::fix`.
        public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> Fix(MR.Mesh mesh, MR.SelfIntersections.Const_Settings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SelfIntersections_fix", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_SelfIntersections_fix(MR.Mesh._Underlying *mesh, MR.SelfIntersections.Const_Settings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_SelfIntersections_fix(mesh._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
        }
    }
}
