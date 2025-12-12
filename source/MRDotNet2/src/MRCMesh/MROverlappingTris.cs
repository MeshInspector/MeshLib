public static partial class MR
{
    /// Generated from class `MR::FindOverlappingSettings`.
    /// This is the const half of the class.
    public class Const_FindOverlappingSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FindOverlappingSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_FindOverlappingSettings_Destroy(_Underlying *_this);
            __MR_FindOverlappingSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FindOverlappingSettings() {Dispose(false);}

        // suggestion: multiply it on mesh.getBoundingBox().size().lengthSq();
        public unsafe float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_Get_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_FindOverlappingSettings_Get_maxDistSq(_Underlying *_this);
                return *__MR_FindOverlappingSettings_Get_maxDistSq(_UnderlyingPtr);
            }
        }

        /// maximal dot product of one triangle and another overlapping triangle normals
        public unsafe float MaxNormalDot
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_Get_maxNormalDot", ExactSpelling = true)]
                extern static float *__MR_FindOverlappingSettings_Get_maxNormalDot(_Underlying *_this);
                return *__MR_FindOverlappingSettings_Get_maxNormalDot(_UnderlyingPtr);
            }
        }

        /// consider triangle as overlapping only if the area of the oppositely oriented triangle is at least given fraction of the triangle's area
        public unsafe float MinAreaFraction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_Get_minAreaFraction", ExactSpelling = true)]
                extern static float *__MR_FindOverlappingSettings_Get_minAreaFraction(_Underlying *_this);
                return *__MR_FindOverlappingSettings_Get_minAreaFraction(_UnderlyingPtr);
            }
        }

        /// for reporting current progress and allowing the user to cancel the algorithm
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FindOverlappingSettings_Get_cb(_Underlying *_this);
                return new(__MR_FindOverlappingSettings_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FindOverlappingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindOverlappingSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindOverlappingSettings` elementwise.
        public unsafe Const_FindOverlappingSettings(float maxDistSq, float maxNormalDot, float minAreaFraction, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_ConstructFrom(float maxDistSq, float maxNormalDot, float minAreaFraction, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FindOverlappingSettings_ConstructFrom(maxDistSq, maxNormalDot, minAreaFraction, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindOverlappingSettings::FindOverlappingSettings`.
        public unsafe Const_FindOverlappingSettings(MR._ByValue_FindOverlappingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindOverlappingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindOverlappingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::FindOverlappingSettings`.
    /// This is the non-const half of the class.
    public class FindOverlappingSettings : Const_FindOverlappingSettings
    {
        internal unsafe FindOverlappingSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // suggestion: multiply it on mesh.getBoundingBox().size().lengthSq();
        public new unsafe ref float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_GetMutable_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_FindOverlappingSettings_GetMutable_maxDistSq(_Underlying *_this);
                return ref *__MR_FindOverlappingSettings_GetMutable_maxDistSq(_UnderlyingPtr);
            }
        }

        /// maximal dot product of one triangle and another overlapping triangle normals
        public new unsafe ref float MaxNormalDot
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_GetMutable_maxNormalDot", ExactSpelling = true)]
                extern static float *__MR_FindOverlappingSettings_GetMutable_maxNormalDot(_Underlying *_this);
                return ref *__MR_FindOverlappingSettings_GetMutable_maxNormalDot(_UnderlyingPtr);
            }
        }

        /// consider triangle as overlapping only if the area of the oppositely oriented triangle is at least given fraction of the triangle's area
        public new unsafe ref float MinAreaFraction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_GetMutable_minAreaFraction", ExactSpelling = true)]
                extern static float *__MR_FindOverlappingSettings_GetMutable_minAreaFraction(_Underlying *_this);
                return ref *__MR_FindOverlappingSettings_GetMutable_minAreaFraction(_UnderlyingPtr);
            }
        }

        /// for reporting current progress and allowing the user to cancel the algorithm
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FindOverlappingSettings_GetMutable_cb(_Underlying *_this);
                return new(__MR_FindOverlappingSettings_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FindOverlappingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindOverlappingSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindOverlappingSettings` elementwise.
        public unsafe FindOverlappingSettings(float maxDistSq, float maxNormalDot, float minAreaFraction, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_ConstructFrom(float maxDistSq, float maxNormalDot, float minAreaFraction, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FindOverlappingSettings_ConstructFrom(maxDistSq, maxNormalDot, minAreaFraction, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindOverlappingSettings::FindOverlappingSettings`.
        public unsafe FindOverlappingSettings(MR._ByValue_FindOverlappingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindOverlappingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindOverlappingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FindOverlappingSettings::operator=`.
        public unsafe MR.FindOverlappingSettings Assign(MR._ByValue_FindOverlappingSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindOverlappingSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FindOverlappingSettings._Underlying *__MR_FindOverlappingSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FindOverlappingSettings._Underlying *_other);
            return new(__MR_FindOverlappingSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FindOverlappingSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FindOverlappingSettings`/`Const_FindOverlappingSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FindOverlappingSettings
    {
        internal readonly Const_FindOverlappingSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FindOverlappingSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FindOverlappingSettings(Const_FindOverlappingSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FindOverlappingSettings(Const_FindOverlappingSettings arg) {return new(arg);}
        public _ByValue_FindOverlappingSettings(MR.Misc._Moved<FindOverlappingSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FindOverlappingSettings(MR.Misc._Moved<FindOverlappingSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FindOverlappingSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FindOverlappingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindOverlappingSettings`/`Const_FindOverlappingSettings` directly.
    public class _InOptMut_FindOverlappingSettings
    {
        public FindOverlappingSettings? Opt;

        public _InOptMut_FindOverlappingSettings() {}
        public _InOptMut_FindOverlappingSettings(FindOverlappingSettings value) {Opt = value;}
        public static implicit operator _InOptMut_FindOverlappingSettings(FindOverlappingSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `FindOverlappingSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FindOverlappingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindOverlappingSettings`/`Const_FindOverlappingSettings` to pass it to the function.
    public class _InOptConst_FindOverlappingSettings
    {
        public Const_FindOverlappingSettings? Opt;

        public _InOptConst_FindOverlappingSettings() {}
        public _InOptConst_FindOverlappingSettings(Const_FindOverlappingSettings value) {Opt = value;}
        public static implicit operator _InOptConst_FindOverlappingSettings(Const_FindOverlappingSettings value) {return new(value);}
    }

    /// finds all triangles that have oppositely oriented close triangle in the mesh
    /// Generated from function `MR::findOverlappingTris`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> FindOverlappingTris(MR.Const_MeshPart mp, MR.Const_FindOverlappingSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findOverlappingTris", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_findOverlappingTris(MR.Const_MeshPart._Underlying *mp, MR.Const_FindOverlappingSettings._Underlying *settings);
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_findOverlappingTris(mp._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
    }
}
