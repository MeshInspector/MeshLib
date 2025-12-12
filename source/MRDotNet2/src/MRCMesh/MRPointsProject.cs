public static partial class MR
{
    /// Generated from class `MR::PointsProjectionResult`.
    /// This is the const half of the class.
    public class Const_PointsProjectionResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointsProjectionResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_Destroy", ExactSpelling = true)]
            extern static void __MR_PointsProjectionResult_Destroy(_Underlying *_this);
            __MR_PointsProjectionResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointsProjectionResult() {Dispose(false);}

        /// squared distance from pt to proj
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_PointsProjectionResult_Get_distSq(_Underlying *_this);
                return *__MR_PointsProjectionResult_Get_distSq(_UnderlyingPtr);
            }
        }

        /// the closest vertex in point cloud
        public unsafe MR.Const_VertId VId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_Get_vId", ExactSpelling = true)]
                extern static MR.Const_VertId._Underlying *__MR_PointsProjectionResult_Get_vId(_Underlying *_this);
                return new(__MR_PointsProjectionResult_Get_vId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointsProjectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsProjectionResult_DefaultConstruct();
        }

        /// Constructs `MR::PointsProjectionResult` elementwise.
        public unsafe Const_PointsProjectionResult(float distSq, MR.VertId vId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_ConstructFrom(float distSq, MR.VertId vId);
            _UnderlyingPtr = __MR_PointsProjectionResult_ConstructFrom(distSq, vId);
        }

        /// Generated from constructor `MR::PointsProjectionResult::PointsProjectionResult`.
        public unsafe Const_PointsProjectionResult(MR.Const_PointsProjectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_ConstructFromAnother(MR.PointsProjectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_PointsProjectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PointsProjectionResult`.
    /// This is the non-const half of the class.
    public class PointsProjectionResult : Const_PointsProjectionResult
    {
        internal unsafe PointsProjectionResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// squared distance from pt to proj
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_PointsProjectionResult_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_PointsProjectionResult_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// the closest vertex in point cloud
        public new unsafe MR.Mut_VertId VId
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_GetMutable_vId", ExactSpelling = true)]
                extern static MR.Mut_VertId._Underlying *__MR_PointsProjectionResult_GetMutable_vId(_Underlying *_this);
                return new(__MR_PointsProjectionResult_GetMutable_vId(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointsProjectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsProjectionResult_DefaultConstruct();
        }

        /// Constructs `MR::PointsProjectionResult` elementwise.
        public unsafe PointsProjectionResult(float distSq, MR.VertId vId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_ConstructFrom(float distSq, MR.VertId vId);
            _UnderlyingPtr = __MR_PointsProjectionResult_ConstructFrom(distSq, vId);
        }

        /// Generated from constructor `MR::PointsProjectionResult::PointsProjectionResult`.
        public unsafe PointsProjectionResult(MR.Const_PointsProjectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_ConstructFromAnother(MR.PointsProjectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_PointsProjectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointsProjectionResult::operator=`.
        public unsafe MR.PointsProjectionResult Assign(MR.Const_PointsProjectionResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjectionResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointsProjectionResult._Underlying *__MR_PointsProjectionResult_AssignFromAnother(_Underlying *_this, MR.PointsProjectionResult._Underlying *_other);
            return new(__MR_PointsProjectionResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PointsProjectionResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointsProjectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsProjectionResult`/`Const_PointsProjectionResult` directly.
    public class _InOptMut_PointsProjectionResult
    {
        public PointsProjectionResult? Opt;

        public _InOptMut_PointsProjectionResult() {}
        public _InOptMut_PointsProjectionResult(PointsProjectionResult value) {Opt = value;}
        public static implicit operator _InOptMut_PointsProjectionResult(PointsProjectionResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointsProjectionResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointsProjectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsProjectionResult`/`Const_PointsProjectionResult` to pass it to the function.
    public class _InOptConst_PointsProjectionResult
    {
        public Const_PointsProjectionResult? Opt;

        public _InOptConst_PointsProjectionResult() {}
        public _InOptConst_PointsProjectionResult(Const_PointsProjectionResult value) {Opt = value;}
        public static implicit operator _InOptConst_PointsProjectionResult(Const_PointsProjectionResult value) {return new(value);}
    }

    /// settings for \ref IPointsProjector::findProjections
    /// Generated from class `MR::FindProjectionOnPointsSettings`.
    /// This is the const half of the class.
    public class Const_FindProjectionOnPointsSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FindProjectionOnPointsSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_FindProjectionOnPointsSettings_Destroy(_Underlying *_this);
            __MR_FindProjectionOnPointsSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FindProjectionOnPointsSettings() {Dispose(false);}

        /// bitset of valid input points
        public unsafe ref readonly void * Valid
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Get_valid", ExactSpelling = true)]
                extern static void **__MR_FindProjectionOnPointsSettings_Get_valid(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_Get_valid(_UnderlyingPtr);
            }
        }

        /// affine transformation for input points
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_FindProjectionOnPointsSettings_Get_xf(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_Get_xf(_UnderlyingPtr);
            }
        }

        /// upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
        public unsafe float UpDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Get_upDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_FindProjectionOnPointsSettings_Get_upDistLimitSq(_Underlying *_this);
                return *__MR_FindProjectionOnPointsSettings_Get_upDistLimitSq(_UnderlyingPtr);
            }
        }

        /// low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
        public unsafe float LoDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Get_loDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_FindProjectionOnPointsSettings_Get_loDistLimitSq(_Underlying *_this);
                return *__MR_FindProjectionOnPointsSettings_Get_loDistLimitSq(_UnderlyingPtr);
            }
        }

        /// if true, discards a projection candidate with the same index as the target point
        public unsafe bool SkipSameIndex
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Get_skipSameIndex", ExactSpelling = true)]
                extern static bool *__MR_FindProjectionOnPointsSettings_Get_skipSameIndex(_Underlying *_this);
                return *__MR_FindProjectionOnPointsSettings_Get_skipSameIndex(_UnderlyingPtr);
            }
        }

        /// progress callback
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FindProjectionOnPointsSettings_Get_cb(_Underlying *_this);
                return new(__MR_FindProjectionOnPointsSettings_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FindProjectionOnPointsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindProjectionOnPointsSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindProjectionOnPointsSettings` elementwise.
        public unsafe Const_FindProjectionOnPointsSettings(MR.Const_BitSet? valid, MR.Const_AffineXf3f? xf, float upDistLimitSq, float loDistLimitSq, bool skipSameIndex, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_ConstructFrom(MR.Const_BitSet._Underlying *valid, MR.Const_AffineXf3f._Underlying *xf, float upDistLimitSq, float loDistLimitSq, byte skipSameIndex, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FindProjectionOnPointsSettings_ConstructFrom(valid is not null ? valid._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null, upDistLimitSq, loDistLimitSq, skipSameIndex ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindProjectionOnPointsSettings::FindProjectionOnPointsSettings`.
        public unsafe Const_FindProjectionOnPointsSettings(MR._ByValue_FindProjectionOnPointsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindProjectionOnPointsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindProjectionOnPointsSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// settings for \ref IPointsProjector::findProjections
    /// Generated from class `MR::FindProjectionOnPointsSettings`.
    /// This is the non-const half of the class.
    public class FindProjectionOnPointsSettings : Const_FindProjectionOnPointsSettings
    {
        internal unsafe FindProjectionOnPointsSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// bitset of valid input points
        public new unsafe ref readonly void * Valid
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_GetMutable_valid", ExactSpelling = true)]
                extern static void **__MR_FindProjectionOnPointsSettings_GetMutable_valid(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_GetMutable_valid(_UnderlyingPtr);
            }
        }

        /// affine transformation for input points
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_FindProjectionOnPointsSettings_GetMutable_xf(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_GetMutable_xf(_UnderlyingPtr);
            }
        }

        /// upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
        public new unsafe ref float UpDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_GetMutable_upDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_FindProjectionOnPointsSettings_GetMutable_upDistLimitSq(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_GetMutable_upDistLimitSq(_UnderlyingPtr);
            }
        }

        /// low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
        public new unsafe ref float LoDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_GetMutable_loDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_FindProjectionOnPointsSettings_GetMutable_loDistLimitSq(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_GetMutable_loDistLimitSq(_UnderlyingPtr);
            }
        }

        /// if true, discards a projection candidate with the same index as the target point
        public new unsafe ref bool SkipSameIndex
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_GetMutable_skipSameIndex", ExactSpelling = true)]
                extern static bool *__MR_FindProjectionOnPointsSettings_GetMutable_skipSameIndex(_Underlying *_this);
                return ref *__MR_FindProjectionOnPointsSettings_GetMutable_skipSameIndex(_UnderlyingPtr);
            }
        }

        /// progress callback
        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FindProjectionOnPointsSettings_GetMutable_cb(_Underlying *_this);
                return new(__MR_FindProjectionOnPointsSettings_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FindProjectionOnPointsSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_FindProjectionOnPointsSettings_DefaultConstruct();
        }

        /// Constructs `MR::FindProjectionOnPointsSettings` elementwise.
        public unsafe FindProjectionOnPointsSettings(MR.Const_BitSet? valid, MR.Const_AffineXf3f? xf, float upDistLimitSq, float loDistLimitSq, bool skipSameIndex, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_ConstructFrom(MR.Const_BitSet._Underlying *valid, MR.Const_AffineXf3f._Underlying *xf, float upDistLimitSq, float loDistLimitSq, byte skipSameIndex, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FindProjectionOnPointsSettings_ConstructFrom(valid is not null ? valid._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null, upDistLimitSq, loDistLimitSq, skipSameIndex ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindProjectionOnPointsSettings::FindProjectionOnPointsSettings`.
        public unsafe FindProjectionOnPointsSettings(MR._ByValue_FindProjectionOnPointsSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindProjectionOnPointsSettings._Underlying *_other);
            _UnderlyingPtr = __MR_FindProjectionOnPointsSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FindProjectionOnPointsSettings::operator=`.
        public unsafe MR.FindProjectionOnPointsSettings Assign(MR._ByValue_FindProjectionOnPointsSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindProjectionOnPointsSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FindProjectionOnPointsSettings._Underlying *__MR_FindProjectionOnPointsSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FindProjectionOnPointsSettings._Underlying *_other);
            return new(__MR_FindProjectionOnPointsSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FindProjectionOnPointsSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FindProjectionOnPointsSettings`/`Const_FindProjectionOnPointsSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FindProjectionOnPointsSettings
    {
        internal readonly Const_FindProjectionOnPointsSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FindProjectionOnPointsSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FindProjectionOnPointsSettings(Const_FindProjectionOnPointsSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FindProjectionOnPointsSettings(Const_FindProjectionOnPointsSettings arg) {return new(arg);}
        public _ByValue_FindProjectionOnPointsSettings(MR.Misc._Moved<FindProjectionOnPointsSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FindProjectionOnPointsSettings(MR.Misc._Moved<FindProjectionOnPointsSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FindProjectionOnPointsSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FindProjectionOnPointsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindProjectionOnPointsSettings`/`Const_FindProjectionOnPointsSettings` directly.
    public class _InOptMut_FindProjectionOnPointsSettings
    {
        public FindProjectionOnPointsSettings? Opt;

        public _InOptMut_FindProjectionOnPointsSettings() {}
        public _InOptMut_FindProjectionOnPointsSettings(FindProjectionOnPointsSettings value) {Opt = value;}
        public static implicit operator _InOptMut_FindProjectionOnPointsSettings(FindProjectionOnPointsSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `FindProjectionOnPointsSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FindProjectionOnPointsSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindProjectionOnPointsSettings`/`Const_FindProjectionOnPointsSettings` to pass it to the function.
    public class _InOptConst_FindProjectionOnPointsSettings
    {
        public Const_FindProjectionOnPointsSettings? Opt;

        public _InOptConst_FindProjectionOnPointsSettings() {}
        public _InOptConst_FindProjectionOnPointsSettings(Const_FindProjectionOnPointsSettings value) {Opt = value;}
        public static implicit operator _InOptConst_FindProjectionOnPointsSettings(Const_FindProjectionOnPointsSettings value) {return new(value);}
    }

    /// abstract class for computing the closest points of point clouds
    /// Generated from class `MR::IPointsProjector`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointsProjector`
    /// This is the const half of the class.
    public class Const_IPointsProjector : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IPointsProjector(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsProjector_Destroy", ExactSpelling = true)]
            extern static void __MR_IPointsProjector_Destroy(_Underlying *_this);
            __MR_IPointsProjector_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IPointsProjector() {Dispose(false);}

        /// computes the closest points on point cloud to given points
        /// Generated from method `MR::IPointsProjector::findProjections`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> FindProjections(MR.Std.Vector_MRPointsProjectionResult results, MR.Std.Const_Vector_MRVector3f points, MR.Const_FindProjectionOnPointsSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsProjector_findProjections", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IPointsProjector_findProjections(_Underlying *_this, MR.Std.Vector_MRPointsProjectionResult._Underlying *results, MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_FindProjectionOnPointsSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IPointsProjector_findProjections(_UnderlyingPtr, results._UnderlyingPtr, points._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
        }

        /// Returns amount of memory needed to compute projections
        /// Generated from method `MR::IPointsProjector::projectionsHeapBytes`.
        public unsafe ulong ProjectionsHeapBytes(ulong numProjections)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsProjector_projectionsHeapBytes", ExactSpelling = true)]
            extern static ulong __MR_IPointsProjector_projectionsHeapBytes(_Underlying *_this, ulong numProjections);
            return __MR_IPointsProjector_projectionsHeapBytes(_UnderlyingPtr, numProjections);
        }
    }

    /// abstract class for computing the closest points of point clouds
    /// Generated from class `MR::IPointsProjector`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointsProjector`
    /// This is the non-const half of the class.
    public class IPointsProjector : Const_IPointsProjector
    {
        internal unsafe IPointsProjector(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// sets the reference point cloud
        /// Generated from method `MR::IPointsProjector::setPointCloud`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SetPointCloud(MR.Const_PointCloud pointCloud)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsProjector_setPointCloud", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IPointsProjector_setPointCloud(_Underlying *_this, MR.Const_PointCloud._Underlying *pointCloud);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IPointsProjector_setPointCloud(_UnderlyingPtr, pointCloud._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used for optional parameters of class `IPointsProjector` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IPointsProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IPointsProjector`/`Const_IPointsProjector` directly.
    public class _InOptMut_IPointsProjector
    {
        public IPointsProjector? Opt;

        public _InOptMut_IPointsProjector() {}
        public _InOptMut_IPointsProjector(IPointsProjector value) {Opt = value;}
        public static implicit operator _InOptMut_IPointsProjector(IPointsProjector value) {return new(value);}
    }

    /// This is used for optional parameters of class `IPointsProjector` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IPointsProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IPointsProjector`/`Const_IPointsProjector` to pass it to the function.
    public class _InOptConst_IPointsProjector
    {
        public Const_IPointsProjector? Opt;

        public _InOptConst_IPointsProjector() {}
        public _InOptConst_IPointsProjector(Const_IPointsProjector value) {Opt = value;}
        public static implicit operator _InOptConst_IPointsProjector(Const_IPointsProjector value) {return new(value);}
    }

    /// default implementation of IPointsProjector
    /// Generated from class `MR::PointsProjector`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointsProjector`
    /// This is the const half of the class.
    public class Const_PointsProjector : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointsProjector(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_Destroy", ExactSpelling = true)]
            extern static void __MR_PointsProjector_Destroy(_Underlying *_this);
            __MR_PointsProjector_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointsProjector() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_IPointsProjector(Const_PointsProjector self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_UpcastTo_MR_IPointsProjector", ExactSpelling = true)]
            extern static MR.Const_IPointsProjector._Underlying *__MR_PointsProjector_UpcastTo_MR_IPointsProjector(_Underlying *_this);
            MR.Const_IPointsProjector ret = new(__MR_PointsProjector_UpcastTo_MR_IPointsProjector(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Downcasts:
        public static unsafe explicit operator Const_PointsProjector?(MR.Const_IPointsProjector parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsProjector_DynamicDowncastTo_MR_PointsProjector", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointsProjector_DynamicDowncastTo_MR_PointsProjector(MR.Const_IPointsProjector._Underlying *_this);
            var ptr = __MR_IPointsProjector_DynamicDowncastTo_MR_PointsProjector(parent._UnderlyingPtr);
            if (ptr is null) return null;
            Const_PointsProjector ret = new(ptr, is_owning: false);
            ret._KeepAlive(parent);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointsProjector() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsProjector._Underlying *__MR_PointsProjector_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsProjector_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointsProjector::PointsProjector`.
        public unsafe Const_PointsProjector(MR._ByValue_PointsProjector _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsProjector._Underlying *__MR_PointsProjector_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsProjector._Underlying *_other);
            _UnderlyingPtr = __MR_PointsProjector_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// computes the closest points on point cloud to given points
        /// Generated from method `MR::PointsProjector::findProjections`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> FindProjections(MR.Std.Vector_MRPointsProjectionResult results, MR.Std.Const_Vector_MRVector3f points, MR.Const_FindProjectionOnPointsSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_findProjections", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsProjector_findProjections(_Underlying *_this, MR.Std.Vector_MRPointsProjectionResult._Underlying *results, MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_FindProjectionOnPointsSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsProjector_findProjections(_UnderlyingPtr, results._UnderlyingPtr, points._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
        }

        /// Returns amount of memory needed to compute projections
        /// Generated from method `MR::PointsProjector::projectionsHeapBytes`.
        public unsafe ulong ProjectionsHeapBytes(ulong numProjections)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_projectionsHeapBytes", ExactSpelling = true)]
            extern static ulong __MR_PointsProjector_projectionsHeapBytes(_Underlying *_this, ulong numProjections);
            return __MR_PointsProjector_projectionsHeapBytes(_UnderlyingPtr, numProjections);
        }
    }

    /// default implementation of IPointsProjector
    /// Generated from class `MR::PointsProjector`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointsProjector`
    /// This is the non-const half of the class.
    public class PointsProjector : Const_PointsProjector
    {
        internal unsafe PointsProjector(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.IPointsProjector(PointsProjector self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_UpcastTo_MR_IPointsProjector", ExactSpelling = true)]
            extern static MR.IPointsProjector._Underlying *__MR_PointsProjector_UpcastTo_MR_IPointsProjector(_Underlying *_this);
            MR.IPointsProjector ret = new(__MR_PointsProjector_UpcastTo_MR_IPointsProjector(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        // Downcasts:
        public static unsafe explicit operator PointsProjector?(MR.IPointsProjector parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsProjector_DynamicDowncastTo_MR_PointsProjector", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointsProjector_DynamicDowncastTo_MR_PointsProjector(MR.IPointsProjector._Underlying *_this);
            var ptr = __MR_IPointsProjector_DynamicDowncastTo_MR_PointsProjector(parent._UnderlyingPtr);
            if (ptr is null) return null;
            PointsProjector ret = new(ptr, is_owning: false);
            ret._KeepAlive(parent);
            return ret;
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointsProjector() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsProjector._Underlying *__MR_PointsProjector_DefaultConstruct();
            _UnderlyingPtr = __MR_PointsProjector_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointsProjector::PointsProjector`.
        public unsafe PointsProjector(MR._ByValue_PointsProjector _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsProjector._Underlying *__MR_PointsProjector_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsProjector._Underlying *_other);
            _UnderlyingPtr = __MR_PointsProjector_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointsProjector::operator=`.
        public unsafe MR.PointsProjector Assign(MR._ByValue_PointsProjector _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointsProjector._Underlying *__MR_PointsProjector_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsProjector._Underlying *_other);
            return new(__MR_PointsProjector_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// sets the reference point cloud
        /// Generated from method `MR::PointsProjector::setPointCloud`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> SetPointCloud(MR.Const_PointCloud pointCloud)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsProjector_setPointCloud", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_PointsProjector_setPointCloud(_Underlying *_this, MR.Const_PointCloud._Underlying *pointCloud);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_PointsProjector_setPointCloud(_UnderlyingPtr, pointCloud._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointsProjector` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointsProjector`/`Const_PointsProjector` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointsProjector
    {
        internal readonly Const_PointsProjector? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointsProjector() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointsProjector(Const_PointsProjector new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointsProjector(Const_PointsProjector arg) {return new(arg);}
        public _ByValue_PointsProjector(MR.Misc._Moved<PointsProjector> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointsProjector(MR.Misc._Moved<PointsProjector> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointsProjector` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointsProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsProjector`/`Const_PointsProjector` directly.
    public class _InOptMut_PointsProjector
    {
        public PointsProjector? Opt;

        public _InOptMut_PointsProjector() {}
        public _InOptMut_PointsProjector(PointsProjector value) {Opt = value;}
        public static implicit operator _InOptMut_PointsProjector(PointsProjector value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointsProjector` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointsProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsProjector`/`Const_PointsProjector` to pass it to the function.
    public class _InOptConst_PointsProjector
    {
        public Const_PointsProjector? Opt;

        public _InOptConst_PointsProjector() {}
        public _InOptConst_PointsProjector(Const_PointsProjector value) {Opt = value;}
        public static implicit operator _InOptConst_PointsProjector(Const_PointsProjector value) {return new(value);}
    }

    /**
    * \brief computes the closest point on point cloud to given point
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    * \param skipCb callback to discard VertId projection candidate
    */
    /// Generated from function `MR::findProjectionOnPoints`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    /// Parameter `skipCb` defaults to `{}`.
    public static unsafe MR.PointsProjectionResult FindProjectionOnPoints(MR.Const_Vector3f pt, MR.Const_PointCloudPart pcp, float? upDistLimitSq = null, MR.Const_AffineXf3f? xf = null, float? loDistLimitSq = null, MR.Std._ByValue_Function_BoolFuncFromMRVertId? skipCb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPoints_6", ExactSpelling = true)]
        extern static MR.PointsProjectionResult._Underlying *__MR_findProjectionOnPoints_6(MR.Const_Vector3f._Underlying *pt, MR.Const_PointCloudPart._Underlying *pcp, float *upDistLimitSq, MR.Const_AffineXf3f._Underlying *xf, float *loDistLimitSq, MR.Misc._PassBy skipCb_pass_by, MR.Std.Function_BoolFuncFromMRVertId._Underlying *skipCb);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnPoints_6(pt._UnderlyingPtr, pcp._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null, skipCb is not null ? skipCb.PassByMode : MR.Misc._PassBy.default_arg, skipCb is not null && skipCb.Value is not null ? skipCb.Value._UnderlyingPtr : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on AABBTreePoints to given point
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    * \param region if not nullptr, all points not from the given region will be ignored
    * \param skipCb callback to discard VertId projection candidate
    */
    /// Generated from function `MR::findProjectionOnPoints`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    /// Parameter `skipCb` defaults to `{}`.
    public static unsafe MR.PointsProjectionResult FindProjectionOnPoints(MR.Const_Vector3f pt, MR.Const_AABBTreePoints tree, float? upDistLimitSq = null, MR.Const_AffineXf3f? xf = null, float? loDistLimitSq = null, MR.Const_VertBitSet? region = null, MR.Std._ByValue_Function_BoolFuncFromMRVertId? skipCb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionOnPoints_7", ExactSpelling = true)]
        extern static MR.PointsProjectionResult._Underlying *__MR_findProjectionOnPoints_7(MR.Const_Vector3f._Underlying *pt, MR.Const_AABBTreePoints._Underlying *tree, float *upDistLimitSq, MR.Const_AffineXf3f._Underlying *xf, float *loDistLimitSq, MR.Const_VertBitSet._Underlying *region, MR.Misc._PassBy skipCb_pass_by, MR.Std.Function_BoolFuncFromMRVertId._Underlying *skipCb);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionOnPoints_7(pt._UnderlyingPtr, tree._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null, region is not null ? region._UnderlyingPtr : null, skipCb is not null ? skipCb.PassByMode : MR.Misc._PassBy.default_arg, skipCb is not null && skipCb.Value is not null ? skipCb.Value._UnderlyingPtr : null), is_owning: true);
    }

    /**
    * \brief finds a number of the closest points in the cloud (as configured in \param res) to given point
    * \param upDistLimitSq upper limit on the distance in question, points with larger distance than it will not be returned
    * \param xf pointcloud-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, the algorithm can return given number of points within this distance even skipping closer ones
    */
    /// Generated from function `MR::findFewClosestPoints`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe void FindFewClosestPoints(MR.Const_Vector3f pt, MR.Const_PointCloud pc, MR.FewSmallest_MRPointsProjectionResult res, float? upDistLimitSq = null, MR.Const_AffineXf3f? xf = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findFewClosestPoints", ExactSpelling = true)]
        extern static void __MR_findFewClosestPoints(MR.Const_Vector3f._Underlying *pt, MR.Const_PointCloud._Underlying *pc, MR.FewSmallest_MRPointsProjectionResult._Underlying *res, float *upDistLimitSq, MR.Const_AffineXf3f._Underlying *xf, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        __MR_findFewClosestPoints(pt._UnderlyingPtr, pc._UnderlyingPtr, res._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null);
    }

    /**
    * \brief finds given number of closest points (excluding itself) to each valid point in the cloud;
    * \param numNei the number of closest points to find for each point
    * \return a buffer where for every valid point with index `i` its neighbours are stored at indices [i*numNei; (i+1)*numNei)
    */
    /// Generated from function `MR::findNClosestPointsPerPoint`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Buffer_MRVertId> FindNClosestPointsPerPoint(MR.Const_PointCloud pc, int numNei, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findNClosestPointsPerPoint", ExactSpelling = true)]
        extern static MR.Buffer_MRVertId._Underlying *__MR_findNClosestPointsPerPoint(MR.Const_PointCloud._Underlying *pc, int numNei, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.Buffer_MRVertId(__MR_findNClosestPointsPerPoint(pc._UnderlyingPtr, numNei, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /// finds two closest points (first id < second id) in whole point cloud
    /// Generated from function `MR::findTwoClosestPoints`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Std.Pair_MRVertId_MRVertId FindTwoClosestPoints(MR.Const_PointCloud pc, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTwoClosestPoints", ExactSpelling = true)]
        extern static MR.Std.Pair_MRVertId_MRVertId._Underlying *__MR_findTwoClosestPoints(MR.Const_PointCloud._Underlying *pc, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return new(__MR_findTwoClosestPoints(pc._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true);
    }
}
