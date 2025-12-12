public static partial class MR
{
    public static partial class SceneLoad
    {
        /// Generated from class `MR::SceneLoad::Settings`.
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
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_Destroy", ExactSpelling = true)]
                extern static void __MR_SceneLoad_Settings_Destroy(_Underlying *_this);
                __MR_SceneLoad_Settings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Settings() {Dispose(false);}

            /// if both targetUnit and loadedObject.lengthUnit are not nullopt,
            /// then adjusts transformations of the loaded objects to match target units
            public unsafe MR.Std.Const_Optional_MRLengthUnit TargetUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_Get_targetUnit", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_SceneLoad_Settings_Get_targetUnit(_Underlying *_this);
                    return new(__MR_SceneLoad_Settings_Get_targetUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report loading progress and allow the user to cancel it
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_SceneLoad_Settings_Get_progress(_Underlying *_this);
                    return new(__MR_SceneLoad_Settings_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_SceneLoad_Settings_DefaultConstruct();
            }

            /// Constructs `MR::SceneLoad::Settings` elementwise.
            public unsafe Const_Settings(MR.LengthUnit? targetUnit, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_ConstructFrom(MR.LengthUnit *targetUnit, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                MR.LengthUnit __deref_targetUnit = targetUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_SceneLoad_Settings_ConstructFrom(targetUnit.HasValue ? &__deref_targetUnit : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::SceneLoad::Settings::Settings`.
            public unsafe Const_Settings(MR.SceneLoad._ByValue_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneLoad.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_SceneLoad_Settings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::SceneLoad::Settings`.
        /// This is the non-const half of the class.
        public class Settings : Const_Settings
        {
            internal unsafe Settings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// if both targetUnit and loadedObject.lengthUnit are not nullopt,
            /// then adjusts transformations of the loaded objects to match target units
            public new unsafe MR.Std.Optional_MRLengthUnit TargetUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_GetMutable_targetUnit", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_SceneLoad_Settings_GetMutable_targetUnit(_Underlying *_this);
                    return new(__MR_SceneLoad_Settings_GetMutable_targetUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report loading progress and allow the user to cancel it
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_SceneLoad_Settings_GetMutable_progress(_Underlying *_this);
                    return new(__MR_SceneLoad_Settings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_SceneLoad_Settings_DefaultConstruct();
            }

            /// Constructs `MR::SceneLoad::Settings` elementwise.
            public unsafe Settings(MR.LengthUnit? targetUnit, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_ConstructFrom(MR.LengthUnit *targetUnit, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                MR.LengthUnit __deref_targetUnit = targetUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_SceneLoad_Settings_ConstructFrom(targetUnit.HasValue ? &__deref_targetUnit : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::SceneLoad::Settings::Settings`.
            public unsafe Settings(MR.SceneLoad._ByValue_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneLoad.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_SceneLoad_Settings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::SceneLoad::Settings::operator=`.
            public unsafe MR.SceneLoad.Settings Assign(MR.SceneLoad._ByValue_Settings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Settings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.SceneLoad.Settings._Underlying *__MR_SceneLoad_Settings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SceneLoad.Settings._Underlying *_other);
                return new(__MR_SceneLoad_Settings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
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

        /// Scene loading result
        /// Generated from class `MR::SceneLoad::Result`.
        /// This is the const half of the class.
        public class Const_Result : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Result(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_Destroy", ExactSpelling = true)]
                extern static void __MR_SceneLoad_Result_Destroy(_Underlying *_this);
                __MR_SceneLoad_Result_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Result() {Dispose(false);}

            /// The loaded scene or empty object
            public unsafe MR.Const_SceneRootObject Scene
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_Get_scene", ExactSpelling = true)]
                    extern static MR.Const_SceneRootObject._UnderlyingShared *__MR_SceneLoad_Result_Get_scene(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_Get_scene(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Marks whether the scene was loaded from a single file (false) or was built from scratch (true)
            public unsafe bool IsSceneConstructed
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_Get_isSceneConstructed", ExactSpelling = true)]
                    extern static bool *__MR_SceneLoad_Result_Get_isSceneConstructed(_Underlying *_this);
                    return *__MR_SceneLoad_Result_Get_isSceneConstructed(_UnderlyingPtr);
                }
            }

            /// List of successfully loaded files
            public unsafe MR.Std.Const_Vector_StdFilesystemPath LoadedFiles
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_Get_loadedFiles", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_StdFilesystemPath._Underlying *__MR_SceneLoad_Result_Get_loadedFiles(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_Get_loadedFiles(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Error summary text
            // TODO: user-defined error format
            public unsafe MR.Std.Const_String ErrorSummary
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_Get_errorSummary", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_SceneLoad_Result_Get_errorSummary(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_Get_errorSummary(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Warning summary text
            // TODO: user-defined warning format
            public unsafe MR.Std.Const_String WarningSummary
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_Get_warningSummary", ExactSpelling = true)]
                    extern static MR.Std.Const_String._Underlying *__MR_SceneLoad_Result_Get_warningSummary(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_Get_warningSummary(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Result() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_DefaultConstruct();
                _UnderlyingPtr = __MR_SceneLoad_Result_DefaultConstruct();
            }

            /// Constructs `MR::SceneLoad::Result` elementwise.
            public unsafe Const_Result(MR._ByValue_SceneRootObject scene, bool isSceneConstructed, MR.Std._ByValue_Vector_StdFilesystemPath loadedFiles, ReadOnlySpan<char> errorSummary, ReadOnlySpan<char> warningSummary) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_ConstructFrom", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_ConstructFrom(MR.Misc._PassBy scene_pass_by, MR.SceneRootObject._UnderlyingShared *scene, byte isSceneConstructed, MR.Misc._PassBy loadedFiles_pass_by, MR.Std.Vector_StdFilesystemPath._Underlying *loadedFiles, byte *errorSummary, byte *errorSummary_end, byte *warningSummary, byte *warningSummary_end);
                byte[] __bytes_errorSummary = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(errorSummary.Length)];
                int __len_errorSummary = System.Text.Encoding.UTF8.GetBytes(errorSummary, __bytes_errorSummary);
                fixed (byte *__ptr_errorSummary = __bytes_errorSummary)
                {
                    byte[] __bytes_warningSummary = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warningSummary.Length)];
                    int __len_warningSummary = System.Text.Encoding.UTF8.GetBytes(warningSummary, __bytes_warningSummary);
                    fixed (byte *__ptr_warningSummary = __bytes_warningSummary)
                    {
                        _UnderlyingPtr = __MR_SceneLoad_Result_ConstructFrom(scene.PassByMode, scene.Value is not null ? scene.Value._UnderlyingSharedPtr : null, isSceneConstructed ? (byte)1 : (byte)0, loadedFiles.PassByMode, loadedFiles.Value is not null ? loadedFiles.Value._UnderlyingPtr : null, __ptr_errorSummary, __ptr_errorSummary + __len_errorSummary, __ptr_warningSummary, __ptr_warningSummary + __len_warningSummary);
                    }
                }
            }

            /// Generated from constructor `MR::SceneLoad::Result::Result`.
            public unsafe Const_Result(MR.SceneLoad._ByValue_Result _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneLoad.Result._Underlying *_other);
                _UnderlyingPtr = __MR_SceneLoad_Result_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Scene loading result
        /// Generated from class `MR::SceneLoad::Result`.
        /// This is the non-const half of the class.
        public class Result : Const_Result
        {
            internal unsafe Result(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// The loaded scene or empty object
            public new unsafe MR.SceneRootObject Scene
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_GetMutable_scene", ExactSpelling = true)]
                    extern static MR.SceneRootObject._UnderlyingShared *__MR_SceneLoad_Result_GetMutable_scene(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_GetMutable_scene(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Marks whether the scene was loaded from a single file (false) or was built from scratch (true)
            public new unsafe ref bool IsSceneConstructed
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_GetMutable_isSceneConstructed", ExactSpelling = true)]
                    extern static bool *__MR_SceneLoad_Result_GetMutable_isSceneConstructed(_Underlying *_this);
                    return ref *__MR_SceneLoad_Result_GetMutable_isSceneConstructed(_UnderlyingPtr);
                }
            }

            /// List of successfully loaded files
            public new unsafe MR.Std.Vector_StdFilesystemPath LoadedFiles
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_GetMutable_loadedFiles", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdFilesystemPath._Underlying *__MR_SceneLoad_Result_GetMutable_loadedFiles(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_GetMutable_loadedFiles(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Error summary text
            // TODO: user-defined error format
            public new unsafe MR.Std.String ErrorSummary
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_GetMutable_errorSummary", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_SceneLoad_Result_GetMutable_errorSummary(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_GetMutable_errorSummary(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Warning summary text
            // TODO: user-defined warning format
            public new unsafe MR.Std.String WarningSummary
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_GetMutable_warningSummary", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_SceneLoad_Result_GetMutable_warningSummary(_Underlying *_this);
                    return new(__MR_SceneLoad_Result_GetMutable_warningSummary(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Result() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_DefaultConstruct", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_DefaultConstruct();
                _UnderlyingPtr = __MR_SceneLoad_Result_DefaultConstruct();
            }

            /// Constructs `MR::SceneLoad::Result` elementwise.
            public unsafe Result(MR._ByValue_SceneRootObject scene, bool isSceneConstructed, MR.Std._ByValue_Vector_StdFilesystemPath loadedFiles, ReadOnlySpan<char> errorSummary, ReadOnlySpan<char> warningSummary) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_ConstructFrom", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_ConstructFrom(MR.Misc._PassBy scene_pass_by, MR.SceneRootObject._UnderlyingShared *scene, byte isSceneConstructed, MR.Misc._PassBy loadedFiles_pass_by, MR.Std.Vector_StdFilesystemPath._Underlying *loadedFiles, byte *errorSummary, byte *errorSummary_end, byte *warningSummary, byte *warningSummary_end);
                byte[] __bytes_errorSummary = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(errorSummary.Length)];
                int __len_errorSummary = System.Text.Encoding.UTF8.GetBytes(errorSummary, __bytes_errorSummary);
                fixed (byte *__ptr_errorSummary = __bytes_errorSummary)
                {
                    byte[] __bytes_warningSummary = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(warningSummary.Length)];
                    int __len_warningSummary = System.Text.Encoding.UTF8.GetBytes(warningSummary, __bytes_warningSummary);
                    fixed (byte *__ptr_warningSummary = __bytes_warningSummary)
                    {
                        _UnderlyingPtr = __MR_SceneLoad_Result_ConstructFrom(scene.PassByMode, scene.Value is not null ? scene.Value._UnderlyingSharedPtr : null, isSceneConstructed ? (byte)1 : (byte)0, loadedFiles.PassByMode, loadedFiles.Value is not null ? loadedFiles.Value._UnderlyingPtr : null, __ptr_errorSummary, __ptr_errorSummary + __len_errorSummary, __ptr_warningSummary, __ptr_warningSummary + __len_warningSummary);
                    }
                }
            }

            /// Generated from constructor `MR::SceneLoad::Result::Result`.
            public unsafe Result(MR.SceneLoad._ByValue_Result _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SceneLoad.Result._Underlying *_other);
                _UnderlyingPtr = __MR_SceneLoad_Result_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::SceneLoad::Result::operator=`.
            public unsafe MR.SceneLoad.Result Assign(MR.SceneLoad._ByValue_Result _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_Result_AssignFromAnother", ExactSpelling = true)]
                extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_Result_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SceneLoad.Result._Underlying *_other);
                return new(__MR_SceneLoad_Result_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `Result` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `Result`/`Const_Result` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_Result
        {
            internal readonly Const_Result? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_Result() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_Result(Const_Result new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_Result(Const_Result arg) {return new(arg);}
            public _ByValue_Result(MR.Misc._Moved<Result> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_Result(MR.Misc._Moved<Result> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `Result` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Result`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Result`/`Const_Result` directly.
        public class _InOptMut_Result
        {
            public Result? Opt;

            public _InOptMut_Result() {}
            public _InOptMut_Result(Result value) {Opt = value;}
            public static implicit operator _InOptMut_Result(Result value) {return new(value);}
        }

        /// This is used for optional parameters of class `Result` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Result`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Result`/`Const_Result` to pass it to the function.
        public class _InOptConst_Result
        {
            public Const_Result? Opt;

            public _InOptConst_Result() {}
            public _InOptConst_Result(Const_Result value) {Opt = value;}
            public static implicit operator _InOptConst_Result(Const_Result value) {return new(value);}
        }

        /// Load scene from file;
        /// if both targetUnit and loadedObject.lengthUnit are not nullopt, then adjusts transformations of the loaded objects
        /// Generated from function `MR::SceneLoad::fromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.SceneLoad.Result> FromAnySupportedFormat(MR.Std.Const_Vector_StdFilesystemPath files, MR.SceneLoad.Const_Settings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_fromAnySupportedFormat", ExactSpelling = true)]
            extern static MR.SceneLoad.Result._Underlying *__MR_SceneLoad_fromAnySupportedFormat(MR.Std.Const_Vector_StdFilesystemPath._Underlying *files, MR.SceneLoad.Const_Settings._Underlying *settings);
            return MR.Misc.Move(new MR.SceneLoad.Result(__MR_SceneLoad_fromAnySupportedFormat(files._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::SceneLoad::asyncFromAnySupportedFormat`.
        /// Parameter `settings` defaults to `{}`.
        public static unsafe void AsyncFromAnySupportedFormat(MR.Std.Const_Vector_StdFilesystemPath files, MR.Std.Const_Function_VoidFuncFromMRSceneLoadResult postLoadCallback, MR.SceneLoad.Const_Settings? settings = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SceneLoad_asyncFromAnySupportedFormat", ExactSpelling = true)]
            extern static void __MR_SceneLoad_asyncFromAnySupportedFormat(MR.Std.Const_Vector_StdFilesystemPath._Underlying *files, MR.Std.Const_Function_VoidFuncFromMRSceneLoadResult._Underlying *postLoadCallback, MR.SceneLoad.Const_Settings._Underlying *settings);
            __MR_SceneLoad_asyncFromAnySupportedFormat(files._UnderlyingPtr, postLoadCallback._UnderlyingPtr, settings is not null ? settings._UnderlyingPtr : null);
        }
    }
}
