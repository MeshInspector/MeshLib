public static partial class MR
{
    /// Generated from class `MR::NoiseSettings`.
    /// This is the const half of the class.
    public class Const_NoiseSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_NoiseSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_NoiseSettings_Destroy(_Underlying *_this);
            __MR_NoiseSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_NoiseSettings() {Dispose(false);}

        public unsafe float Sigma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_Get_sigma", ExactSpelling = true)]
                extern static float *__MR_NoiseSettings_Get_sigma(_Underlying *_this);
                return *__MR_NoiseSettings_Get_sigma(_UnderlyingPtr);
            }
        }

        // start state of the generator engine
        public unsafe uint Seed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_Get_seed", ExactSpelling = true)]
                extern static uint *__MR_NoiseSettings_Get_seed(_Underlying *_this);
                return *__MR_NoiseSettings_Get_seed(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_Get_callback", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_NoiseSettings_Get_callback(_Underlying *_this);
                return new(__MR_NoiseSettings_Get_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_NoiseSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_NoiseSettings_DefaultConstruct();
        }

        /// Constructs `MR::NoiseSettings` elementwise.
        public unsafe Const_NoiseSettings(float sigma, uint seed, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_ConstructFrom(float sigma, uint seed, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_NoiseSettings_ConstructFrom(sigma, seed, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::NoiseSettings::NoiseSettings`.
        public unsafe Const_NoiseSettings(MR._ByValue_NoiseSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NoiseSettings._Underlying *_other);
            _UnderlyingPtr = __MR_NoiseSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::NoiseSettings`.
    /// This is the non-const half of the class.
    public class NoiseSettings : Const_NoiseSettings
    {
        internal unsafe NoiseSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref float Sigma
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_GetMutable_sigma", ExactSpelling = true)]
                extern static float *__MR_NoiseSettings_GetMutable_sigma(_Underlying *_this);
                return ref *__MR_NoiseSettings_GetMutable_sigma(_UnderlyingPtr);
            }
        }

        // start state of the generator engine
        public new unsafe ref uint Seed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_GetMutable_seed", ExactSpelling = true)]
                extern static uint *__MR_NoiseSettings_GetMutable_seed(_Underlying *_this);
                return ref *__MR_NoiseSettings_GetMutable_seed(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Callback
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_GetMutable_callback", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_NoiseSettings_GetMutable_callback(_Underlying *_this);
                return new(__MR_NoiseSettings_GetMutable_callback(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe NoiseSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_NoiseSettings_DefaultConstruct();
        }

        /// Constructs `MR::NoiseSettings` elementwise.
        public unsafe NoiseSettings(float sigma, uint seed, MR.Std._ByValue_Function_BoolFuncFromFloat callback) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_ConstructFrom(float sigma, uint seed, MR.Misc._PassBy callback_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *callback);
            _UnderlyingPtr = __MR_NoiseSettings_ConstructFrom(sigma, seed, callback.PassByMode, callback.Value is not null ? callback.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::NoiseSettings::NoiseSettings`.
        public unsafe NoiseSettings(MR._ByValue_NoiseSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.NoiseSettings._Underlying *_other);
            _UnderlyingPtr = __MR_NoiseSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::NoiseSettings::operator=`.
        public unsafe MR.NoiseSettings Assign(MR._ByValue_NoiseSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_NoiseSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.NoiseSettings._Underlying *__MR_NoiseSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.NoiseSettings._Underlying *_other);
            return new(__MR_NoiseSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `NoiseSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `NoiseSettings`/`Const_NoiseSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_NoiseSettings
    {
        internal readonly Const_NoiseSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_NoiseSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_NoiseSettings(Const_NoiseSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_NoiseSettings(Const_NoiseSettings arg) {return new(arg);}
        public _ByValue_NoiseSettings(MR.Misc._Moved<NoiseSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_NoiseSettings(MR.Misc._Moved<NoiseSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `NoiseSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_NoiseSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoiseSettings`/`Const_NoiseSettings` directly.
    public class _InOptMut_NoiseSettings
    {
        public NoiseSettings? Opt;

        public _InOptMut_NoiseSettings() {}
        public _InOptMut_NoiseSettings(NoiseSettings value) {Opt = value;}
        public static implicit operator _InOptMut_NoiseSettings(NoiseSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `NoiseSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_NoiseSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `NoiseSettings`/`Const_NoiseSettings` to pass it to the function.
    public class _InOptConst_NoiseSettings
    {
        public Const_NoiseSettings? Opt;

        public _InOptConst_NoiseSettings() {}
        public _InOptConst_NoiseSettings(Const_NoiseSettings value) {Opt = value;}
        public static implicit operator _InOptConst_NoiseSettings(Const_NoiseSettings value) {return new(value);}
    }

    // Adds noise to the points, using a normal distribution
    /// Generated from function `MR::addNoise`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> AddNoise(MR.VertCoords points, MR.Const_VertBitSet validVerts, MR._ByValue_NoiseSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_addNoise_MR_VertCoords", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_addNoise_MR_VertCoords(MR.VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *validVerts, MR.Misc._PassBy settings_pass_by, MR.NoiseSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_addNoise_MR_VertCoords(points._UnderlyingPtr, validVerts._UnderlyingPtr, settings.PassByMode, settings.Value is not null ? settings.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::addNoise`.
    /// Parameter `settings` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> AddNoise(MR.Mesh mesh, MR.Const_VertBitSet? region = null, MR.Const_NoiseSettings? settings = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_addNoise_MR_Mesh", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_addNoise_MR_Mesh(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region, MR.Const_NoiseSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_addNoise_MR_Mesh(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null, settings is not null ? settings._UnderlyingPtr : null), is_owning: true));
    }
}
