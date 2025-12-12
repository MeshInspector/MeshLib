public static partial class MR
{
    /// parameters for MR::findOverhangs
    /// Generated from class `MR::FindOverhangsSettings`.
    /// This is the const half of the class.
    public class Const_FindOverhangsSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FindOverhangsSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_FindOverhangsSettings_Destroy(_Underlying *_this);
            __MR_FindOverhangsSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FindOverhangsSettings() {Dispose(false);}

        /// base axis marking the up direction
        public unsafe MR.Const_Vector3f Axis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Get_axis", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_FindOverhangsSettings_Get_axis(_Underlying *_this);
                return new(__MR_FindOverhangsSettings_Get_axis(_UnderlyingPtr), is_owning: false);
            }
        }

        /// height of a layer
        public unsafe float LayerHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Get_layerHeight", ExactSpelling = true)]
                extern static float *__MR_FindOverhangsSettings_Get_layerHeight(_Underlying *_this);
                return *__MR_FindOverhangsSettings_Get_layerHeight(_UnderlyingPtr);
            }
        }

        /// maximum allowed overhang distance within a layer
        public unsafe float MaxOverhangDistance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Get_maxOverhangDistance", ExactSpelling = true)]
                extern static float *__MR_FindOverhangsSettings_Get_maxOverhangDistance(_Underlying *_this);
                return *__MR_FindOverhangsSettings_Get_maxOverhangDistance(_UnderlyingPtr);
            }
        }

        /// number of hops used to smooth out the overhang regions (0 - disable smoothing)
        public unsafe int Hops
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Get_hops", ExactSpelling = true)]
                extern static int *__MR_FindOverhangsSettings_Get_hops(_Underlying *_this);
                return *__MR_FindOverhangsSettings_Get_hops(_UnderlyingPtr);
            }
        }

        /// mesh transform
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_FindOverhangsSettings_Get_xf(_Underlying *_this);
                return ref *__MR_FindOverhangsSettings_Get_xf(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_Get_progressCb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FindOverhangsSettings_Get_progressCb(_Underlying *_this);
                return new(__MR_FindOverhangsSettings_Get_progressCb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FindOverhangsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindOverhangsSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindOverhangsSettings` elementwise.
        public unsafe Const_FindOverhangsSettings(MR.Vector3f axis, float layerHeight, float maxOverhangDistance, int hops, MR.Const_AffineXf3f? xf, MR.Std._ByValue_Function_BoolFuncFromFloat progressCb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_ConstructFrom(MR.Vector3f axis, float layerHeight, float maxOverhangDistance, int hops, MR.Const_AffineXf3f._Underlying *xf, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            _UnderlyingPtr = __MR_FindOverhangsSettings_ConstructFrom(axis, layerHeight, maxOverhangDistance, hops, xf is not null ? xf._UnderlyingPtr : null, progressCb.PassByMode, progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindOverhangsSettings::FindOverhangsSettings`.
        public unsafe Const_FindOverhangsSettings(MR._ByValue_FindOverhangsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindOverhangsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindOverhangsSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// parameters for MR::findOverhangs
    /// Generated from class `MR::FindOverhangsSettings`.
    /// This is the non-const half of the class.
    public class FindOverhangsSettings : Const_FindOverhangsSettings
    {
        internal unsafe FindOverhangsSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// base axis marking the up direction
        public new unsafe MR.Mut_Vector3f Axis
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_GetMutable_axis", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_FindOverhangsSettings_GetMutable_axis(_Underlying *_this);
                return new(__MR_FindOverhangsSettings_GetMutable_axis(_UnderlyingPtr), is_owning: false);
            }
        }

        /// height of a layer
        public new unsafe ref float LayerHeight
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_GetMutable_layerHeight", ExactSpelling = true)]
                extern static float *__MR_FindOverhangsSettings_GetMutable_layerHeight(_Underlying *_this);
                return ref *__MR_FindOverhangsSettings_GetMutable_layerHeight(_UnderlyingPtr);
            }
        }

        /// maximum allowed overhang distance within a layer
        public new unsafe ref float MaxOverhangDistance
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_GetMutable_maxOverhangDistance", ExactSpelling = true)]
                extern static float *__MR_FindOverhangsSettings_GetMutable_maxOverhangDistance(_Underlying *_this);
                return ref *__MR_FindOverhangsSettings_GetMutable_maxOverhangDistance(_UnderlyingPtr);
            }
        }

        /// number of hops used to smooth out the overhang regions (0 - disable smoothing)
        public new unsafe ref int Hops
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_GetMutable_hops", ExactSpelling = true)]
                extern static int *__MR_FindOverhangsSettings_GetMutable_hops(_Underlying *_this);
                return ref *__MR_FindOverhangsSettings_GetMutable_hops(_UnderlyingPtr);
            }
        }

        /// mesh transform
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_FindOverhangsSettings_GetMutable_xf(_Underlying *_this);
                return ref *__MR_FindOverhangsSettings_GetMutable_xf(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_GetMutable_progressCb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FindOverhangsSettings_GetMutable_progressCb(_Underlying *_this);
                return new(__MR_FindOverhangsSettings_GetMutable_progressCb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FindOverhangsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindOverhangsSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindOverhangsSettings` elementwise.
        public unsafe FindOverhangsSettings(MR.Vector3f axis, float layerHeight, float maxOverhangDistance, int hops, MR.Const_AffineXf3f? xf, MR.Std._ByValue_Function_BoolFuncFromFloat progressCb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_ConstructFrom(MR.Vector3f axis, float layerHeight, float maxOverhangDistance, int hops, MR.Const_AffineXf3f._Underlying *xf, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            _UnderlyingPtr = __MR_FindOverhangsSettings_ConstructFrom(axis, layerHeight, maxOverhangDistance, hops, xf is not null ? xf._UnderlyingPtr : null, progressCb.PassByMode, progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindOverhangsSettings::FindOverhangsSettings`.
        public unsafe FindOverhangsSettings(MR._ByValue_FindOverhangsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindOverhangsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindOverhangsSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FindOverhangsSettings::operator=`.
        public unsafe MR.FindOverhangsSettings Assign(MR._ByValue_FindOverhangsSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverhangsSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FindOverhangsSettings._Underlying *__MR_FindOverhangsSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FindOverhangsSettings._Underlying *_other);
            return new(__MR_FindOverhangsSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FindOverhangsSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FindOverhangsSettings`/`Const_FindOverhangsSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FindOverhangsSettings
    {
        internal readonly Const_FindOverhangsSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FindOverhangsSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FindOverhangsSettings(Const_FindOverhangsSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FindOverhangsSettings(Const_FindOverhangsSettings arg) {return new(arg);}
        public _ByValue_FindOverhangsSettings(MR.Misc._Moved<FindOverhangsSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FindOverhangsSettings(MR.Misc._Moved<FindOverhangsSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FindOverhangsSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FindOverhangsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindOverhangsSettings`/`Const_FindOverhangsSettings` directly.
    public class _InOptMut_FindOverhangsSettings
    {
        public FindOverhangsSettings? Opt;

        public _InOptMut_FindOverhangsSettings() {}
        public _InOptMut_FindOverhangsSettings(FindOverhangsSettings value) {Opt = value;}
        public static implicit operator _InOptMut_FindOverhangsSettings(FindOverhangsSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `FindOverhangsSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FindOverhangsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindOverhangsSettings`/`Const_FindOverhangsSettings` to pass it to the function.
    public class _InOptConst_FindOverhangsSettings
    {
        public Const_FindOverhangsSettings? Opt;

        public _InOptConst_FindOverhangsSettings() {}
        public _InOptConst_FindOverhangsSettings(Const_FindOverhangsSettings value) {Opt = value;}
        public static implicit operator _InOptConst_FindOverhangsSettings(Const_FindOverhangsSettings value) {return new(value);}
    }

    /// \brief Find face regions that might create overhangs
    /// \param mesh - source mesh
    /// \param settings - parameters
    /// \return face regions
    /// Generated from function `MR::findOverhangs`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorMRFaceBitSet_StdString> FindOverhangs(MR.Const_Mesh mesh, MR.Const_FindOverhangsSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findOverhangs", ExactSpelling = true)]
        extern static MR.Expected_StdVectorMRFaceBitSet_StdString._Underlying *__MR_findOverhangs(MR.Const_Mesh._Underlying *mesh, MR.Const_FindOverhangsSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_StdVectorMRFaceBitSet_StdString(__MR_findOverhangs(mesh._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
