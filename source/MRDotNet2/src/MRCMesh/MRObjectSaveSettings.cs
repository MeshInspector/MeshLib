public static partial class MR
{
    public static partial class ObjectSave
    {
        /// Generated from class `MR::ObjectSave::Settings`.
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
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_Destroy", ExactSpelling = true)]
                extern static void __MR_ObjectSave_Settings_Destroy(_Underlying *_this);
                __MR_ObjectSave_Settings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Settings() {Dispose(false);}

            /// units of input coordinates and transformation, to be serialized if the format supports it
            public unsafe MR.Std.Const_Optional_MRLengthUnit LengthUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_Get_lengthUnit", ExactSpelling = true)]
                    extern static MR.Std.Const_Optional_MRLengthUnit._Underlying *__MR_ObjectSave_Settings_Get_lengthUnit(_Underlying *_this);
                    return new(__MR_ObjectSave_Settings_Get_lengthUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report loading progress and allow the user to cancel it
            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_Get_progress", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ObjectSave_Settings_Get_progress(_Underlying *_this);
                    return new(__MR_ObjectSave_Settings_Get_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectSave_Settings_DefaultConstruct();
            }

            /// Constructs `MR::ObjectSave::Settings` elementwise.
            public unsafe Const_Settings(MR.LengthUnit? lengthUnit, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_ConstructFrom(MR.LengthUnit *lengthUnit, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_ObjectSave_Settings_ConstructFrom(lengthUnit.HasValue ? &__deref_lengthUnit : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ObjectSave::Settings::Settings`.
            public unsafe Const_Settings(MR.ObjectSave._ByValue_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectSave.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectSave_Settings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::ObjectSave::Settings`.
        /// This is the non-const half of the class.
        public class Settings : Const_Settings
        {
            internal unsafe Settings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// units of input coordinates and transformation, to be serialized if the format supports it
            public new unsafe MR.Std.Optional_MRLengthUnit LengthUnit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_GetMutable_lengthUnit", ExactSpelling = true)]
                    extern static MR.Std.Optional_MRLengthUnit._Underlying *__MR_ObjectSave_Settings_GetMutable_lengthUnit(_Underlying *_this);
                    return new(__MR_ObjectSave_Settings_GetMutable_lengthUnit(_UnderlyingPtr), is_owning: false);
                }
            }

            /// to report loading progress and allow the user to cancel it
            public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_GetMutable_progress", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ObjectSave_Settings_GetMutable_progress(_Underlying *_this);
                    return new(__MR_ObjectSave_Settings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectSave_Settings_DefaultConstruct();
            }

            /// Constructs `MR::ObjectSave::Settings` elementwise.
            public unsafe Settings(MR.LengthUnit? lengthUnit, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_ConstructFrom(MR.LengthUnit *lengthUnit, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
                MR.LengthUnit __deref_lengthUnit = lengthUnit.GetValueOrDefault();
                _UnderlyingPtr = __MR_ObjectSave_Settings_ConstructFrom(lengthUnit.HasValue ? &__deref_lengthUnit : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ObjectSave::Settings::Settings`.
            public unsafe Settings(MR.ObjectSave._ByValue_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectSave.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectSave_Settings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::ObjectSave::Settings::operator=`.
            public unsafe MR.ObjectSave.Settings Assign(MR.ObjectSave._ByValue_Settings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectSave_Settings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ObjectSave.Settings._Underlying *__MR_ObjectSave_Settings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectSave.Settings._Underlying *_other);
                return new(__MR_ObjectSave_Settings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
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
    }
}
