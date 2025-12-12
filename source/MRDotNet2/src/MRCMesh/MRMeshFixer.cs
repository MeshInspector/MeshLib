public static partial class MR
{
    /// Generated from class `MR::FixMeshDegeneraciesParams`.
    /// This is the const half of the class.
    public class Const_FixMeshDegeneraciesParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FixMeshDegeneraciesParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Destroy", ExactSpelling = true)]
            extern static void __MR_FixMeshDegeneraciesParams_Destroy(_Underlying *_this);
            __MR_FixMeshDegeneraciesParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FixMeshDegeneraciesParams() {Dispose(false);}

        /// maximum permitted deviation from the original surface
        public unsafe float MaxDeviation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_maxDeviation", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_Get_maxDeviation(_Underlying *_this);
                return *__MR_FixMeshDegeneraciesParams_Get_maxDeviation(_UnderlyingPtr);
            }
        }

        /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
        public unsafe float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_Get_tinyEdgeLength(_Underlying *_this);
                return *__MR_FixMeshDegeneraciesParams_Get_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public unsafe float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_Get_criticalTriAspectRatio(_Underlying *_this);
                return *__MR_FixMeshDegeneraciesParams_Get_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// Permit edge flips if it does not change dihedral angle more than on this value
        public unsafe float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_Get_maxAngleChange(_Underlying *_this);
                return *__MR_FixMeshDegeneraciesParams_Get_maxAngleChange(_UnderlyingPtr);
            }
        }

        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_Get_stabilizer(_Underlying *_this);
                return *__MR_FixMeshDegeneraciesParams_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// degenerations will be fixed only in given region, it is updated during the operation
        public unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_FixMeshDegeneraciesParams_Get_region(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_Get_region(_UnderlyingPtr);
            }
        }

        public unsafe MR.FixMeshDegeneraciesParams.Mode Mode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_mode", ExactSpelling = true)]
                extern static MR.FixMeshDegeneraciesParams.Mode *__MR_FixMeshDegeneraciesParams_Get_mode(_Underlying *_this);
                return *__MR_FixMeshDegeneraciesParams_Get_mode(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FixMeshDegeneraciesParams_Get_cb(_Underlying *_this);
                return new(__MR_FixMeshDegeneraciesParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FixMeshDegeneraciesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FixMeshDegeneraciesParams_DefaultConstruct();
        }

        /// Constructs `MR::FixMeshDegeneraciesParams` elementwise.
        public unsafe Const_FixMeshDegeneraciesParams(float maxDeviation, float tinyEdgeLength, float criticalTriAspectRatio, float maxAngleChange, float stabilizer, MR.FaceBitSet? region, MR.FixMeshDegeneraciesParams.Mode mode, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_ConstructFrom(float maxDeviation, float tinyEdgeLength, float criticalTriAspectRatio, float maxAngleChange, float stabilizer, MR.FaceBitSet._Underlying *region, MR.FixMeshDegeneraciesParams.Mode mode, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FixMeshDegeneraciesParams_ConstructFrom(maxDeviation, tinyEdgeLength, criticalTriAspectRatio, maxAngleChange, stabilizer, region is not null ? region._UnderlyingPtr : null, mode, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FixMeshDegeneraciesParams::FixMeshDegeneraciesParams`.
        public unsafe Const_FixMeshDegeneraciesParams(MR._ByValue_FixMeshDegeneraciesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FixMeshDegeneraciesParams._Underlying *_other);
            _UnderlyingPtr = __MR_FixMeshDegeneraciesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        public enum Mode : int
        {
            ///< use decimation only to fix degeneracies
            Decimate = 0,
            ///< if decimation does not succeed, perform subdivision too
            Remesh = 1,
            ///< if both decimation and subdivision does not succeed, removes degenerate areas and fills occurred holes
            RemeshPatch = 2,
        }
    }

    /// Generated from class `MR::FixMeshDegeneraciesParams`.
    /// This is the non-const half of the class.
    public class FixMeshDegeneraciesParams : Const_FixMeshDegeneraciesParams
    {
        internal unsafe FixMeshDegeneraciesParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// maximum permitted deviation from the original surface
        public new unsafe ref float MaxDeviation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_maxDeviation", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_GetMutable_maxDeviation(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_maxDeviation(_UnderlyingPtr);
            }
        }

        /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
        public new unsafe ref float TinyEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_tinyEdgeLength", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_GetMutable_tinyEdgeLength(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_tinyEdgeLength(_UnderlyingPtr);
            }
        }

        /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
        /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
        public new unsafe ref float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_GetMutable_criticalTriAspectRatio(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// Permit edge flips if it does not change dihedral angle more than on this value
        public new unsafe ref float MaxAngleChange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_maxAngleChange", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_GetMutable_maxAngleChange(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_maxAngleChange(_UnderlyingPtr);
            }
        }

        /// Small stabilizer is important to achieve good results on completely planar mesh parts,
        /// if your mesh is not-planer everywhere, then you can set it to zero
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_FixMeshDegeneraciesParams_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// degenerations will be fixed only in given region, it is updated during the operation
        public new unsafe ref void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_FixMeshDegeneraciesParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.FixMeshDegeneraciesParams.Mode Mode_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_mode", ExactSpelling = true)]
                extern static MR.FixMeshDegeneraciesParams.Mode *__MR_FixMeshDegeneraciesParams_GetMutable_mode(_Underlying *_this);
                return ref *__MR_FixMeshDegeneraciesParams_GetMutable_mode(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FixMeshDegeneraciesParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_FixMeshDegeneraciesParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FixMeshDegeneraciesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FixMeshDegeneraciesParams_DefaultConstruct();
        }

        /// Constructs `MR::FixMeshDegeneraciesParams` elementwise.
        public unsafe FixMeshDegeneraciesParams(float maxDeviation, float tinyEdgeLength, float criticalTriAspectRatio, float maxAngleChange, float stabilizer, MR.FaceBitSet? region, MR.FixMeshDegeneraciesParams.Mode mode, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_ConstructFrom(float maxDeviation, float tinyEdgeLength, float criticalTriAspectRatio, float maxAngleChange, float stabilizer, MR.FaceBitSet._Underlying *region, MR.FixMeshDegeneraciesParams.Mode mode, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FixMeshDegeneraciesParams_ConstructFrom(maxDeviation, tinyEdgeLength, criticalTriAspectRatio, maxAngleChange, stabilizer, region is not null ? region._UnderlyingPtr : null, mode, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FixMeshDegeneraciesParams::FixMeshDegeneraciesParams`.
        public unsafe FixMeshDegeneraciesParams(MR._ByValue_FixMeshDegeneraciesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FixMeshDegeneraciesParams._Underlying *_other);
            _UnderlyingPtr = __MR_FixMeshDegeneraciesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FixMeshDegeneraciesParams::operator=`.
        public unsafe MR.FixMeshDegeneraciesParams Assign(MR._ByValue_FixMeshDegeneraciesParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixMeshDegeneraciesParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FixMeshDegeneraciesParams._Underlying *__MR_FixMeshDegeneraciesParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FixMeshDegeneraciesParams._Underlying *_other);
            return new(__MR_FixMeshDegeneraciesParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FixMeshDegeneraciesParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FixMeshDegeneraciesParams`/`Const_FixMeshDegeneraciesParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FixMeshDegeneraciesParams
    {
        internal readonly Const_FixMeshDegeneraciesParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FixMeshDegeneraciesParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FixMeshDegeneraciesParams(Const_FixMeshDegeneraciesParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FixMeshDegeneraciesParams(Const_FixMeshDegeneraciesParams arg) {return new(arg);}
        public _ByValue_FixMeshDegeneraciesParams(MR.Misc._Moved<FixMeshDegeneraciesParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FixMeshDegeneraciesParams(MR.Misc._Moved<FixMeshDegeneraciesParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FixMeshDegeneraciesParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FixMeshDegeneraciesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FixMeshDegeneraciesParams`/`Const_FixMeshDegeneraciesParams` directly.
    public class _InOptMut_FixMeshDegeneraciesParams
    {
        public FixMeshDegeneraciesParams? Opt;

        public _InOptMut_FixMeshDegeneraciesParams() {}
        public _InOptMut_FixMeshDegeneraciesParams(FixMeshDegeneraciesParams value) {Opt = value;}
        public static implicit operator _InOptMut_FixMeshDegeneraciesParams(FixMeshDegeneraciesParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `FixMeshDegeneraciesParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FixMeshDegeneraciesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FixMeshDegeneraciesParams`/`Const_FixMeshDegeneraciesParams` to pass it to the function.
    public class _InOptConst_FixMeshDegeneraciesParams
    {
        public Const_FixMeshDegeneraciesParams? Opt;

        public _InOptConst_FixMeshDegeneraciesParams() {}
        public _InOptConst_FixMeshDegeneraciesParams(Const_FixMeshDegeneraciesParams value) {Opt = value;}
        public static implicit operator _InOptConst_FixMeshDegeneraciesParams(Const_FixMeshDegeneraciesParams value) {return new(value);}
    }

    /// Parameters structure for `fixMeshCreases` function
    /// Generated from class `MR::FixCreasesParams`.
    /// This is the const half of the class.
    public class Const_FixCreasesParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FixCreasesParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_Destroy", ExactSpelling = true)]
            extern static void __MR_FixCreasesParams_Destroy(_Underlying *_this);
            __MR_FixCreasesParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FixCreasesParams() {Dispose(false);}

        /// edges with dihedral angle sharper this will be considered as creases
        public unsafe float CreaseAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_Get_creaseAngle", ExactSpelling = true)]
                extern static float *__MR_FixCreasesParams_Get_creaseAngle(_Underlying *_this);
                return *__MR_FixCreasesParams_Get_creaseAngle(_UnderlyingPtr);
            }
        }

        /// planar check is skipped for faces with worse aspect ratio
        public unsafe float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_Get_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_FixCreasesParams_Get_criticalTriAspectRatio(_Underlying *_this);
                return *__MR_FixCreasesParams_Get_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// maximum number of algorithm iterations
        public unsafe int MaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_Get_maxIters", ExactSpelling = true)]
                extern static int *__MR_FixCreasesParams_Get_maxIters(_Underlying *_this);
                return *__MR_FixCreasesParams_Get_maxIters(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FixCreasesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FixCreasesParams_DefaultConstruct();
        }

        /// Constructs `MR::FixCreasesParams` elementwise.
        public unsafe Const_FixCreasesParams(float creaseAngle, float criticalTriAspectRatio, int maxIters) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_ConstructFrom(float creaseAngle, float criticalTriAspectRatio, int maxIters);
            _UnderlyingPtr = __MR_FixCreasesParams_ConstructFrom(creaseAngle, criticalTriAspectRatio, maxIters);
        }

        /// Generated from constructor `MR::FixCreasesParams::FixCreasesParams`.
        public unsafe Const_FixCreasesParams(MR.Const_FixCreasesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_ConstructFromAnother(MR.FixCreasesParams._Underlying *_other);
            _UnderlyingPtr = __MR_FixCreasesParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Parameters structure for `fixMeshCreases` function
    /// Generated from class `MR::FixCreasesParams`.
    /// This is the non-const half of the class.
    public class FixCreasesParams : Const_FixCreasesParams
    {
        internal unsafe FixCreasesParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// edges with dihedral angle sharper this will be considered as creases
        public new unsafe ref float CreaseAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_GetMutable_creaseAngle", ExactSpelling = true)]
                extern static float *__MR_FixCreasesParams_GetMutable_creaseAngle(_Underlying *_this);
                return ref *__MR_FixCreasesParams_GetMutable_creaseAngle(_UnderlyingPtr);
            }
        }

        /// planar check is skipped for faces with worse aspect ratio
        public new unsafe ref float CriticalTriAspectRatio
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_GetMutable_criticalTriAspectRatio", ExactSpelling = true)]
                extern static float *__MR_FixCreasesParams_GetMutable_criticalTriAspectRatio(_Underlying *_this);
                return ref *__MR_FixCreasesParams_GetMutable_criticalTriAspectRatio(_UnderlyingPtr);
            }
        }

        /// maximum number of algorithm iterations
        public new unsafe ref int MaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_GetMutable_maxIters", ExactSpelling = true)]
                extern static int *__MR_FixCreasesParams_GetMutable_maxIters(_Underlying *_this);
                return ref *__MR_FixCreasesParams_GetMutable_maxIters(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FixCreasesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FixCreasesParams_DefaultConstruct();
        }

        /// Constructs `MR::FixCreasesParams` elementwise.
        public unsafe FixCreasesParams(float creaseAngle, float criticalTriAspectRatio, int maxIters) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_ConstructFrom(float creaseAngle, float criticalTriAspectRatio, int maxIters);
            _UnderlyingPtr = __MR_FixCreasesParams_ConstructFrom(creaseAngle, criticalTriAspectRatio, maxIters);
        }

        /// Generated from constructor `MR::FixCreasesParams::FixCreasesParams`.
        public unsafe FixCreasesParams(MR.Const_FixCreasesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_ConstructFromAnother(MR.FixCreasesParams._Underlying *_other);
            _UnderlyingPtr = __MR_FixCreasesParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::FixCreasesParams::operator=`.
        public unsafe MR.FixCreasesParams Assign(MR.Const_FixCreasesParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FixCreasesParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FixCreasesParams._Underlying *__MR_FixCreasesParams_AssignFromAnother(_Underlying *_this, MR.FixCreasesParams._Underlying *_other);
            return new(__MR_FixCreasesParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `FixCreasesParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FixCreasesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FixCreasesParams`/`Const_FixCreasesParams` directly.
    public class _InOptMut_FixCreasesParams
    {
        public FixCreasesParams? Opt;

        public _InOptMut_FixCreasesParams() {}
        public _InOptMut_FixCreasesParams(FixCreasesParams value) {Opt = value;}
        public static implicit operator _InOptMut_FixCreasesParams(FixCreasesParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `FixCreasesParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FixCreasesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FixCreasesParams`/`Const_FixCreasesParams` to pass it to the function.
    public class _InOptConst_FixCreasesParams
    {
        public Const_FixCreasesParams? Opt;

        public _InOptConst_FixCreasesParams() {}
        public _InOptConst_FixCreasesParams(Const_FixCreasesParams value) {Opt = value;}
        public static implicit operator _InOptConst_FixCreasesParams(Const_FixCreasesParams value) {return new(value);}
    }

    /// Parameters for `findDisorientedFaces` function
    /// Generated from class `MR::FindDisorientationParams`.
    /// This is the const half of the class.
    public class Const_FindDisorientationParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FindDisorientationParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_Destroy", ExactSpelling = true)]
            extern static void __MR_FindDisorientationParams_Destroy(_Underlying *_this);
            __MR_FindDisorientationParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FindDisorientationParams() {Dispose(false);}

        public unsafe MR.FindDisorientationParams.RayMode Mode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_Get_mode", ExactSpelling = true)]
                extern static MR.FindDisorientationParams.RayMode *__MR_FindDisorientationParams_Get_mode(_Underlying *_this);
                return *__MR_FindDisorientationParams_Get_mode(_UnderlyingPtr);
            }
        }

        /// if set - copy mesh, and fills holes for better quality in case of ray going out through hole
        public unsafe bool VirtualFillHoles
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_Get_virtualFillHoles", ExactSpelling = true)]
                extern static bool *__MR_FindDisorientationParams_Get_virtualFillHoles(_Underlying *_this);
                return *__MR_FindDisorientationParams_Get_virtualFillHoles(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_Get_cb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_FindDisorientationParams_Get_cb(_Underlying *_this);
                return new(__MR_FindDisorientationParams_Get_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FindDisorientationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FindDisorientationParams_DefaultConstruct();
        }

        /// Constructs `MR::FindDisorientationParams` elementwise.
        public unsafe Const_FindDisorientationParams(MR.FindDisorientationParams.RayMode mode, bool virtualFillHoles, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_ConstructFrom(MR.FindDisorientationParams.RayMode mode, byte virtualFillHoles, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FindDisorientationParams_ConstructFrom(mode, virtualFillHoles ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindDisorientationParams::FindDisorientationParams`.
        public unsafe Const_FindDisorientationParams(MR._ByValue_FindDisorientationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindDisorientationParams._Underlying *_other);
            _UnderlyingPtr = __MR_FindDisorientationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Mode of detecting disoriented face
        public enum RayMode : int
        {
            ///< positive (normal) direction of face should have even number of intersections
            Positive = 0,
            ///< positive or negative (normal or -normal) direction (the one with lowest number of intersections) should have even/odd number of intersections
            Shallowest = 1,
            ///< both direction should have correct number of intersections (positive - even; negative - odd)
            Both = 2,
        }
    }

    /// Parameters for `findDisorientedFaces` function
    /// Generated from class `MR::FindDisorientationParams`.
    /// This is the non-const half of the class.
    public class FindDisorientationParams : Const_FindDisorientationParams
    {
        internal unsafe FindDisorientationParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.FindDisorientationParams.RayMode Mode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_GetMutable_mode", ExactSpelling = true)]
                extern static MR.FindDisorientationParams.RayMode *__MR_FindDisorientationParams_GetMutable_mode(_Underlying *_this);
                return ref *__MR_FindDisorientationParams_GetMutable_mode(_UnderlyingPtr);
            }
        }

        /// if set - copy mesh, and fills holes for better quality in case of ray going out through hole
        public new unsafe ref bool VirtualFillHoles
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_GetMutable_virtualFillHoles", ExactSpelling = true)]
                extern static bool *__MR_FindDisorientationParams_GetMutable_virtualFillHoles(_Underlying *_this);
                return ref *__MR_FindDisorientationParams_GetMutable_virtualFillHoles(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_GetMutable_cb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_FindDisorientationParams_GetMutable_cb(_Underlying *_this);
                return new(__MR_FindDisorientationParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe FindDisorientationParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_DefaultConstruct();
            _UnderlyingPtr = __MR_FindDisorientationParams_DefaultConstruct();
        }

        /// Constructs `MR::FindDisorientationParams` elementwise.
        public unsafe FindDisorientationParams(MR.FindDisorientationParams.RayMode mode, bool virtualFillHoles, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_ConstructFrom(MR.FindDisorientationParams.RayMode mode, byte virtualFillHoles, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            _UnderlyingPtr = __MR_FindDisorientationParams_ConstructFrom(mode, virtualFillHoles ? (byte)1 : (byte)0, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FindDisorientationParams::FindDisorientationParams`.
        public unsafe FindDisorientationParams(MR._ByValue_FindDisorientationParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FindDisorientationParams._Underlying *_other);
            _UnderlyingPtr = __MR_FindDisorientationParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FindDisorientationParams::operator=`.
        public unsafe MR.FindDisorientationParams Assign(MR._ByValue_FindDisorientationParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FindDisorientationParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FindDisorientationParams._Underlying *__MR_FindDisorientationParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FindDisorientationParams._Underlying *_other);
            return new(__MR_FindDisorientationParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `FindDisorientationParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FindDisorientationParams`/`Const_FindDisorientationParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FindDisorientationParams
    {
        internal readonly Const_FindDisorientationParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FindDisorientationParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FindDisorientationParams(Const_FindDisorientationParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FindDisorientationParams(Const_FindDisorientationParams arg) {return new(arg);}
        public _ByValue_FindDisorientationParams(MR.Misc._Moved<FindDisorientationParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FindDisorientationParams(MR.Misc._Moved<FindDisorientationParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FindDisorientationParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FindDisorientationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindDisorientationParams`/`Const_FindDisorientationParams` directly.
    public class _InOptMut_FindDisorientationParams
    {
        public FindDisorientationParams? Opt;

        public _InOptMut_FindDisorientationParams() {}
        public _InOptMut_FindDisorientationParams(FindDisorientationParams value) {Opt = value;}
        public static implicit operator _InOptMut_FindDisorientationParams(FindDisorientationParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `FindDisorientationParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FindDisorientationParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FindDisorientationParams`/`Const_FindDisorientationParams` to pass it to the function.
    public class _InOptConst_FindDisorientationParams
    {
        public Const_FindDisorientationParams? Opt;

        public _InOptConst_FindDisorientationParams() {}
        public _InOptConst_FindDisorientationParams(Const_FindDisorientationParams value) {Opt = value;}
        public static implicit operator _InOptConst_FindDisorientationParams(Const_FindDisorientationParams value) {return new(value);}
    }

    /// Duplicates all vertices having more than two boundary edges (and returns the number of duplications);
    /// Generated from function `MR::duplicateMultiHoleVertices`.
    public static unsafe int DuplicateMultiHoleVertices(MR.Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_duplicateMultiHoleVertices", ExactSpelling = true)]
        extern static int __MR_duplicateMultiHoleVertices(MR.Mesh._Underlying *mesh);
        return __MR_duplicateMultiHoleVertices(mesh._UnderlyingPtr);
    }

    /// Generated from function `MR::findMultipleEdges`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdVectorStdPairMRVertIdMRVertId_StdString> FindMultipleEdges(MR.Const_MeshTopology topology, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findMultipleEdges", ExactSpelling = true)]
        extern static MR.Expected_StdVectorStdPairMRVertIdMRVertId_StdString._Underlying *__MR_findMultipleEdges(MR.Const_MeshTopology._Underlying *topology, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_StdVectorStdPairMRVertIdMRVertId_StdString(__MR_findMultipleEdges(topology._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Generated from function `MR::hasMultipleEdges`.
    public static unsafe bool HasMultipleEdges(MR.Const_MeshTopology topology)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_hasMultipleEdges", ExactSpelling = true)]
        extern static byte __MR_hasMultipleEdges(MR.Const_MeshTopology._Underlying *topology);
        return __MR_hasMultipleEdges(topology._UnderlyingPtr) != 0;
    }

    /// resolves given multiple edges, but splitting all but one edge in each group
    /// Generated from function `MR::fixMultipleEdges`.
    public static unsafe void FixMultipleEdges(MR.Mesh mesh, MR.Std.Const_Vector_StdPairMRVertIdMRVertId multipleEdges)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fixMultipleEdges_2", ExactSpelling = true)]
        extern static void __MR_fixMultipleEdges_2(MR.Mesh._Underlying *mesh, MR.Std.Const_Vector_StdPairMRVertIdMRVertId._Underlying *multipleEdges);
        __MR_fixMultipleEdges_2(mesh._UnderlyingPtr, multipleEdges._UnderlyingPtr);
    }

    /// finds and resolves multiple edges
    /// Generated from function `MR::fixMultipleEdges`.
    public static unsafe void FixMultipleEdges(MR.Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fixMultipleEdges_1", ExactSpelling = true)]
        extern static void __MR_fixMultipleEdges_1(MR.Mesh._Underlying *mesh);
        __MR_fixMultipleEdges_1(mesh._UnderlyingPtr);
    }

    /// finds faces having aspect ratio >= criticalAspectRatio
    /// Generated from function `MR::findDegenerateFaces`.
    /// Parameter `criticalAspectRatio` defaults to `3.40282347e38f`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> FindDegenerateFaces(MR.Const_MeshPart mp, float? criticalAspectRatio = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDegenerateFaces", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_findDegenerateFaces(MR.Const_MeshPart._Underlying *mp, float *criticalAspectRatio, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        float __deref_criticalAspectRatio = criticalAspectRatio.GetValueOrDefault();
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_findDegenerateFaces(mp._UnderlyingPtr, criticalAspectRatio.HasValue ? &__deref_criticalAspectRatio : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// finds edges having length <= criticalLength
    /// Generated from function `MR::findShortEdges`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRUndirectedEdgeBitSet_StdString> FindShortEdges(MR.Const_MeshPart mp, float criticalLength, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findShortEdges", ExactSpelling = true)]
        extern static MR.Expected_MRUndirectedEdgeBitSet_StdString._Underlying *__MR_findShortEdges(MR.Const_MeshPart._Underlying *mp, float criticalLength, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRUndirectedEdgeBitSet_StdString(__MR_findShortEdges(mp._UnderlyingPtr, criticalLength, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// Fixes degenerate faces and short edges in mesh (changes topology)
    /// Generated from function `MR::fixMeshDegeneracies`.
    public static unsafe MR.Misc._Moved<MR.Expected_Void_StdString> FixMeshDegeneracies(MR.Mesh mesh, MR.Const_FixMeshDegeneraciesParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fixMeshDegeneracies", ExactSpelling = true)]
        extern static MR.Expected_Void_StdString._Underlying *__MR_fixMeshDegeneracies(MR.Mesh._Underlying *mesh, MR.Const_FixMeshDegeneraciesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_fixMeshDegeneracies(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// finds vertices in region with complete ring of N edges
    /// Generated from function `MR::findNRingVerts`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> FindNRingVerts(MR.Const_MeshTopology topology, int n, MR.Const_VertBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findNRingVerts", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_findNRingVerts(MR.Const_MeshTopology._Underlying *topology, int n, MR.Const_VertBitSet._Underlying *region);
        return MR.Misc.Move(new MR.VertBitSet(__MR_findNRingVerts(topology._UnderlyingPtr, n, region is not null ? region._UnderlyingPtr : null), is_owning: true));
    }

    /// returns true if the edge e has both left and right triangular faces and the degree of dest( e ) is 2
    /// Generated from function `MR::isEdgeBetweenDoubleTris`.
    public static unsafe bool IsEdgeBetweenDoubleTris(MR.Const_MeshTopology topology, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isEdgeBetweenDoubleTris", ExactSpelling = true)]
        extern static byte __MR_isEdgeBetweenDoubleTris(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId e);
        return __MR_isEdgeBetweenDoubleTris(topology._UnderlyingPtr, e) != 0;
    }

    /// if the edge e has both left and right triangular faces and the degree of dest( e ) is 2,
    /// then eliminates left( e ), right( e ), e, e.sym(), next( e ), dest( e ), and returns prev( e );
    /// if region is provided then eliminated faces are excluded from it;
    /// otherwise returns invalid edge
    /// Generated from function `MR::eliminateDoubleTris`.
    public static unsafe MR.EdgeId EliminateDoubleTris(MR.MeshTopology topology, MR.EdgeId e, MR.FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_eliminateDoubleTris", ExactSpelling = true)]
        extern static MR.EdgeId __MR_eliminateDoubleTris(MR.MeshTopology._Underlying *topology, MR.EdgeId e, MR.FaceBitSet._Underlying *region);
        return __MR_eliminateDoubleTris(topology._UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
    }

    /// eliminates all double triangles around given vertex preserving vertex valid;
    /// if region is provided then eliminated triangles are excluded from it
    /// Generated from function `MR::eliminateDoubleTrisAround`.
    public static unsafe void EliminateDoubleTrisAround(MR.MeshTopology topology, MR.VertId v, MR.FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_eliminateDoubleTrisAround", ExactSpelling = true)]
        extern static void __MR_eliminateDoubleTrisAround(MR.MeshTopology._Underlying *topology, MR.VertId v, MR.FaceBitSet._Underlying *region);
        __MR_eliminateDoubleTrisAround(topology._UnderlyingPtr, v, region is not null ? region._UnderlyingPtr : null);
    }

    /// returns true if the destination of given edge has degree 3 and 3 incident triangles
    /// Generated from function `MR::isDegree3Dest`.
    public static unsafe bool IsDegree3Dest(MR.Const_MeshTopology topology, MR.EdgeId e)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isDegree3Dest", ExactSpelling = true)]
        extern static byte __MR_isDegree3Dest(MR.Const_MeshTopology._Underlying *topology, MR.EdgeId e);
        return __MR_isDegree3Dest(topology._UnderlyingPtr, e) != 0;
    }

    /// if the destination of given edge has degree 3 and 3 incident triangles,
    /// then eliminates the destination vertex with all its edges and all but one faces, and returns valid remaining edge with same origin as e;
    /// if region is provided then eliminated triangles are excluded from it;
    /// otherwise returns invalid edge
    /// Generated from function `MR::eliminateDegree3Dest`.
    public static unsafe MR.EdgeId EliminateDegree3Dest(MR.MeshTopology topology, MR.EdgeId e, MR.FaceBitSet? region = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_eliminateDegree3Dest", ExactSpelling = true)]
        extern static MR.EdgeId __MR_eliminateDegree3Dest(MR.MeshTopology._Underlying *topology, MR.EdgeId e, MR.FaceBitSet._Underlying *region);
        return __MR_eliminateDegree3Dest(topology._UnderlyingPtr, e, region is not null ? region._UnderlyingPtr : null);
    }

    /// eliminates from the mesh all vertices having degree 3 and 3 incident triangles from given region (which is updated);
    /// if \param fs is provided then eliminated triangles are excluded from it;
    /// \return the number of vertices eliminated
    /// Generated from function `MR::eliminateDegree3Vertices`.
    public static unsafe int EliminateDegree3Vertices(MR.MeshTopology topology, MR.VertBitSet region, MR.FaceBitSet? fs = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_eliminateDegree3Vertices", ExactSpelling = true)]
        extern static int __MR_eliminateDegree3Vertices(MR.MeshTopology._Underlying *topology, MR.VertBitSet._Underlying *region, MR.FaceBitSet._Underlying *fs);
        return __MR_eliminateDegree3Vertices(topology._UnderlyingPtr, region._UnderlyingPtr, fs is not null ? fs._UnderlyingPtr : null);
    }

    /// if given vertex is present on the boundary of some hole several times then returns an edge of this hole (without left);
    /// returns invalid edge otherwise (not a boundary vertex, or it is present only once on the boundary of each hole it pertains to)
    /// Generated from function `MR::isVertexRepeatedOnHoleBd`.
    public static unsafe MR.EdgeId IsVertexRepeatedOnHoleBd(MR.Const_MeshTopology topology, MR.VertId v)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isVertexRepeatedOnHoleBd", ExactSpelling = true)]
        extern static MR.EdgeId __MR_isVertexRepeatedOnHoleBd(MR.Const_MeshTopology._Underlying *topology, MR.VertId v);
        return __MR_isVertexRepeatedOnHoleBd(topology._UnderlyingPtr, v);
    }

    /// returns set bits for all vertices present on the boundary of a hole several times;
    /// Generated from function `MR::findRepeatedVertsOnHoleBd`.
    public static unsafe MR.Misc._Moved<MR.VertBitSet> FindRepeatedVertsOnHoleBd(MR.Const_MeshTopology topology)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findRepeatedVertsOnHoleBd", ExactSpelling = true)]
        extern static MR.VertBitSet._Underlying *__MR_findRepeatedVertsOnHoleBd(MR.Const_MeshTopology._Underlying *topology);
        return MR.Misc.Move(new MR.VertBitSet(__MR_findRepeatedVertsOnHoleBd(topology._UnderlyingPtr), is_owning: true));
    }

    /// returns all faces that complicate one of mesh holes;
    /// hole is complicated if it passes via one vertex more than once;
    /// deleting such faces simplifies the holes and makes them easier to fill
    /// Generated from function `MR::findHoleComplicatingFaces`.
    public static unsafe MR.Misc._Moved<MR.FaceBitSet> FindHoleComplicatingFaces(MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findHoleComplicatingFaces", ExactSpelling = true)]
        extern static MR.FaceBitSet._Underlying *__MR_findHoleComplicatingFaces(MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.FaceBitSet(__MR_findHoleComplicatingFaces(mesh._UnderlyingPtr), is_owning: true));
    }

    /// Finds creases edges and re-triangulates planar areas around them, useful to fix double faces
    /// Generated from function `MR::fixMeshCreases`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe void FixMeshCreases(MR.Mesh mesh, MR.Const_FixCreasesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_fixMeshCreases", ExactSpelling = true)]
        extern static void __MR_fixMeshCreases(MR.Mesh._Underlying *mesh, MR.Const_FixCreasesParams._Underlying *params_);
        __MR_fixMeshCreases(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null);
    }

    /// returns all faces that are oriented inconsistently, based on number of ray intersections
    /// Generated from function `MR::findDisorientedFaces`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> FindDisorientedFaces(MR.Const_Mesh mesh, MR.Const_FindDisorientationParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findDisorientedFaces", ExactSpelling = true)]
        extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_findDisorientedFaces(MR.Const_Mesh._Underlying *mesh, MR.Const_FindDisorientationParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_findDisorientedFaces(mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
