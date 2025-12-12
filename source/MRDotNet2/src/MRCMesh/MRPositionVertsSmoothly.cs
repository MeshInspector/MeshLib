public static partial class MR
{
    /// Generated from class `MR::PositionVertsSmoothlyParams`.
    /// This is the const half of the class.
    public class Const_PositionVertsSmoothlyParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PositionVertsSmoothlyParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_Destroy", ExactSpelling = true)]
            extern static void __MR_PositionVertsSmoothlyParams_Destroy(_Underlying *_this);
            __MR_PositionVertsSmoothlyParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PositionVertsSmoothlyParams() {Dispose(false);}

        /// which vertices on mesh are smoothed, nullptr means all vertices;
        /// it must not include all vertices of a mesh connected component unless stabilizer > 0
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_Get_region", ExactSpelling = true)]
                extern static void **__MR_PositionVertsSmoothlyParams_Get_region(_Underlying *_this);
                return ref *__MR_PositionVertsSmoothlyParams_Get_region(_UnderlyingPtr);
            }
        }

        /// optional additional shifts of each vertex relative to smooth position
        public unsafe ref readonly void * VertShifts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_Get_vertShifts", ExactSpelling = true)]
                extern static void **__MR_PositionVertsSmoothlyParams_Get_vertShifts(_Underlying *_this);
                return ref *__MR_PositionVertsSmoothlyParams_Get_vertShifts(_UnderlyingPtr);
            }
        }

        /// the more the value, the bigger attraction of each vertex to its original position
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_PositionVertsSmoothlyParams_Get_stabilizer(_Underlying *_this);
                return *__MR_PositionVertsSmoothlyParams_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// if specified then it is used instead of \p stabilizer
        public unsafe MR.Std.Const_Function_FloatFuncFromMRVertId VertStabilizers
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_Get_vertStabilizers", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromMRVertId._Underlying *__MR_PositionVertsSmoothlyParams_Get_vertStabilizers(_Underlying *_this);
                return new(__MR_PositionVertsSmoothlyParams_Get_vertStabilizers(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if specified then it is used for edge weights instead of default 1
        public unsafe MR.Std.Const_Function_FloatFuncFromMRUndirectedEdgeId EdgeWeights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_Get_edgeWeights", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromMRUndirectedEdgeId._Underlying *__MR_PositionVertsSmoothlyParams_Get_edgeWeights(_Underlying *_this);
                return new(__MR_PositionVertsSmoothlyParams_Get_edgeWeights(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PositionVertsSmoothlyParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PositionVertsSmoothlyParams_DefaultConstruct();
        }

        /// Constructs `MR::PositionVertsSmoothlyParams` elementwise.
        public unsafe Const_PositionVertsSmoothlyParams(MR.Const_VertBitSet? region, MR.Const_VertCoords? vertShifts, float stabilizer, MR.Std._ByValue_Function_FloatFuncFromMRVertId vertStabilizers, MR.Std._ByValue_Function_FloatFuncFromMRUndirectedEdgeId edgeWeights) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_ConstructFrom(MR.Const_VertBitSet._Underlying *region, MR.Const_VertCoords._Underlying *vertShifts, float stabilizer, MR.Misc._PassBy vertStabilizers_pass_by, MR.Std.Function_FloatFuncFromMRVertId._Underlying *vertStabilizers, MR.Misc._PassBy edgeWeights_pass_by, MR.Std.Function_FloatFuncFromMRUndirectedEdgeId._Underlying *edgeWeights);
            _UnderlyingPtr = __MR_PositionVertsSmoothlyParams_ConstructFrom(region is not null ? region._UnderlyingPtr : null, vertShifts is not null ? vertShifts._UnderlyingPtr : null, stabilizer, vertStabilizers.PassByMode, vertStabilizers.Value is not null ? vertStabilizers.Value._UnderlyingPtr : null, edgeWeights.PassByMode, edgeWeights.Value is not null ? edgeWeights.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PositionVertsSmoothlyParams::PositionVertsSmoothlyParams`.
        public unsafe Const_PositionVertsSmoothlyParams(MR._ByValue_PositionVertsSmoothlyParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PositionVertsSmoothlyParams._Underlying *_other);
            _UnderlyingPtr = __MR_PositionVertsSmoothlyParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::PositionVertsSmoothlyParams`.
    /// This is the non-const half of the class.
    public class PositionVertsSmoothlyParams : Const_PositionVertsSmoothlyParams
    {
        internal unsafe PositionVertsSmoothlyParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// which vertices on mesh are smoothed, nullptr means all vertices;
        /// it must not include all vertices of a mesh connected component unless stabilizer > 0
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_PositionVertsSmoothlyParams_GetMutable_region(_Underlying *_this);
                return ref *__MR_PositionVertsSmoothlyParams_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// optional additional shifts of each vertex relative to smooth position
        public new unsafe ref readonly void * VertShifts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_GetMutable_vertShifts", ExactSpelling = true)]
                extern static void **__MR_PositionVertsSmoothlyParams_GetMutable_vertShifts(_Underlying *_this);
                return ref *__MR_PositionVertsSmoothlyParams_GetMutable_vertShifts(_UnderlyingPtr);
            }
        }

        /// the more the value, the bigger attraction of each vertex to its original position
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_PositionVertsSmoothlyParams_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_PositionVertsSmoothlyParams_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// if specified then it is used instead of \p stabilizer
        public new unsafe MR.Std.Function_FloatFuncFromMRVertId VertStabilizers
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_GetMutable_vertStabilizers", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromMRVertId._Underlying *__MR_PositionVertsSmoothlyParams_GetMutable_vertStabilizers(_Underlying *_this);
                return new(__MR_PositionVertsSmoothlyParams_GetMutable_vertStabilizers(_UnderlyingPtr), is_owning: false);
            }
        }

        /// if specified then it is used for edge weights instead of default 1
        public new unsafe MR.Std.Function_FloatFuncFromMRUndirectedEdgeId EdgeWeights
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_GetMutable_edgeWeights", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromMRUndirectedEdgeId._Underlying *__MR_PositionVertsSmoothlyParams_GetMutable_edgeWeights(_Underlying *_this);
                return new(__MR_PositionVertsSmoothlyParams_GetMutable_edgeWeights(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PositionVertsSmoothlyParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_DefaultConstruct();
            _UnderlyingPtr = __MR_PositionVertsSmoothlyParams_DefaultConstruct();
        }

        /// Constructs `MR::PositionVertsSmoothlyParams` elementwise.
        public unsafe PositionVertsSmoothlyParams(MR.Const_VertBitSet? region, MR.Const_VertCoords? vertShifts, float stabilizer, MR.Std._ByValue_Function_FloatFuncFromMRVertId vertStabilizers, MR.Std._ByValue_Function_FloatFuncFromMRUndirectedEdgeId edgeWeights) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_ConstructFrom(MR.Const_VertBitSet._Underlying *region, MR.Const_VertCoords._Underlying *vertShifts, float stabilizer, MR.Misc._PassBy vertStabilizers_pass_by, MR.Std.Function_FloatFuncFromMRVertId._Underlying *vertStabilizers, MR.Misc._PassBy edgeWeights_pass_by, MR.Std.Function_FloatFuncFromMRUndirectedEdgeId._Underlying *edgeWeights);
            _UnderlyingPtr = __MR_PositionVertsSmoothlyParams_ConstructFrom(region is not null ? region._UnderlyingPtr : null, vertShifts is not null ? vertShifts._UnderlyingPtr : null, stabilizer, vertStabilizers.PassByMode, vertStabilizers.Value is not null ? vertStabilizers.Value._UnderlyingPtr : null, edgeWeights.PassByMode, edgeWeights.Value is not null ? edgeWeights.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PositionVertsSmoothlyParams::PositionVertsSmoothlyParams`.
        public unsafe PositionVertsSmoothlyParams(MR._ByValue_PositionVertsSmoothlyParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PositionVertsSmoothlyParams._Underlying *_other);
            _UnderlyingPtr = __MR_PositionVertsSmoothlyParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PositionVertsSmoothlyParams::operator=`.
        public unsafe MR.PositionVertsSmoothlyParams Assign(MR._ByValue_PositionVertsSmoothlyParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PositionVertsSmoothlyParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PositionVertsSmoothlyParams._Underlying *__MR_PositionVertsSmoothlyParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PositionVertsSmoothlyParams._Underlying *_other);
            return new(__MR_PositionVertsSmoothlyParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PositionVertsSmoothlyParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PositionVertsSmoothlyParams`/`Const_PositionVertsSmoothlyParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PositionVertsSmoothlyParams
    {
        internal readonly Const_PositionVertsSmoothlyParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PositionVertsSmoothlyParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PositionVertsSmoothlyParams(Const_PositionVertsSmoothlyParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PositionVertsSmoothlyParams(Const_PositionVertsSmoothlyParams arg) {return new(arg);}
        public _ByValue_PositionVertsSmoothlyParams(MR.Misc._Moved<PositionVertsSmoothlyParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PositionVertsSmoothlyParams(MR.Misc._Moved<PositionVertsSmoothlyParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PositionVertsSmoothlyParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PositionVertsSmoothlyParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PositionVertsSmoothlyParams`/`Const_PositionVertsSmoothlyParams` directly.
    public class _InOptMut_PositionVertsSmoothlyParams
    {
        public PositionVertsSmoothlyParams? Opt;

        public _InOptMut_PositionVertsSmoothlyParams() {}
        public _InOptMut_PositionVertsSmoothlyParams(PositionVertsSmoothlyParams value) {Opt = value;}
        public static implicit operator _InOptMut_PositionVertsSmoothlyParams(PositionVertsSmoothlyParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `PositionVertsSmoothlyParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PositionVertsSmoothlyParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PositionVertsSmoothlyParams`/`Const_PositionVertsSmoothlyParams` to pass it to the function.
    public class _InOptConst_PositionVertsSmoothlyParams
    {
        public Const_PositionVertsSmoothlyParams? Opt;

        public _InOptConst_PositionVertsSmoothlyParams() {}
        public _InOptConst_PositionVertsSmoothlyParams(Const_PositionVertsSmoothlyParams value) {Opt = value;}
        public static implicit operator _InOptConst_PositionVertsSmoothlyParams(Const_PositionVertsSmoothlyParams value) {return new(value);}
    }

    /// Generated from class `MR::SpacingSettings`.
    /// This is the const half of the class.
    public class Const_SpacingSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SpacingSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_SpacingSettings_Destroy(_Underlying *_this);
            __MR_SpacingSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SpacingSettings() {Dispose(false);}

        /// vertices to be moved by the algorithm, nullptr means all valid vertices
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Get_region", ExactSpelling = true)]
                extern static void **__MR_SpacingSettings_Get_region(_Underlying *_this);
                return ref *__MR_SpacingSettings_Get_region(_UnderlyingPtr);
            }
        }

        // must be defined by the caller
        public unsafe MR.Std.Const_Function_FloatFuncFromMRUndirectedEdgeId Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Get_dist", ExactSpelling = true)]
                extern static MR.Std.Const_Function_FloatFuncFromMRUndirectedEdgeId._Underlying *__MR_SpacingSettings_Get_dist(_Underlying *_this);
                return new(__MR_SpacingSettings_Get_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the algorithm is iterative, the more iterations the closer result to exact solution
        public unsafe int NumIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Get_numIters", ExactSpelling = true)]
                extern static int *__MR_SpacingSettings_Get_numIters(_Underlying *_this);
                return *__MR_SpacingSettings_Get_numIters(_UnderlyingPtr);
            }
        }

        /// too small number here can lead to instability, too large - to slow convergence
        public unsafe float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Get_stabilizer", ExactSpelling = true)]
                extern static float *__MR_SpacingSettings_Get_stabilizer(_Underlying *_this);
                return *__MR_SpacingSettings_Get_stabilizer(_UnderlyingPtr);
            }
        }

        /// maximum sum of minus negative weights, if it is exceeded then stabilizer is increased automatically
        public unsafe float MaxSumNegW
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Get_maxSumNegW", ExactSpelling = true)]
                extern static float *__MR_SpacingSettings_Get_maxSumNegW(_Underlying *_this);
                return *__MR_SpacingSettings_Get_maxSumNegW(_UnderlyingPtr);
            }
        }

        /// if this predicated is given, then all inverted faces will be converted in degenerate faces at the end of each iteration
        public unsafe MR.Std.Const_Function_BoolFuncFromMRFaceId IsInverted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_Get_isInverted", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *__MR_SpacingSettings_Get_isInverted(_Underlying *_this);
                return new(__MR_SpacingSettings_Get_isInverted(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SpacingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SpacingSettings_DefaultConstruct();
        }

        /// Constructs `MR::SpacingSettings` elementwise.
        public unsafe Const_SpacingSettings(MR.Const_VertBitSet? region, MR.Std._ByValue_Function_FloatFuncFromMRUndirectedEdgeId dist, int numIters, float stabilizer, float maxSumNegW, MR.Std._ByValue_Function_BoolFuncFromMRFaceId isInverted) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_ConstructFrom(MR.Const_VertBitSet._Underlying *region, MR.Misc._PassBy dist_pass_by, MR.Std.Function_FloatFuncFromMRUndirectedEdgeId._Underlying *dist, int numIters, float stabilizer, float maxSumNegW, MR.Misc._PassBy isInverted_pass_by, MR.Std.Function_BoolFuncFromMRFaceId._Underlying *isInverted);
            _UnderlyingPtr = __MR_SpacingSettings_ConstructFrom(region is not null ? region._UnderlyingPtr : null, dist.PassByMode, dist.Value is not null ? dist.Value._UnderlyingPtr : null, numIters, stabilizer, maxSumNegW, isInverted.PassByMode, isInverted.Value is not null ? isInverted.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SpacingSettings::SpacingSettings`.
        public unsafe Const_SpacingSettings(MR._ByValue_SpacingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SpacingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SpacingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::SpacingSettings`.
    /// This is the non-const half of the class.
    public class SpacingSettings : Const_SpacingSettings
    {
        internal unsafe SpacingSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// vertices to be moved by the algorithm, nullptr means all valid vertices
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_SpacingSettings_GetMutable_region(_Underlying *_this);
                return ref *__MR_SpacingSettings_GetMutable_region(_UnderlyingPtr);
            }
        }

        // must be defined by the caller
        public new unsafe MR.Std.Function_FloatFuncFromMRUndirectedEdgeId Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_GetMutable_dist", ExactSpelling = true)]
                extern static MR.Std.Function_FloatFuncFromMRUndirectedEdgeId._Underlying *__MR_SpacingSettings_GetMutable_dist(_Underlying *_this);
                return new(__MR_SpacingSettings_GetMutable_dist(_UnderlyingPtr), is_owning: false);
            }
        }

        /// the algorithm is iterative, the more iterations the closer result to exact solution
        public new unsafe ref int NumIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_GetMutable_numIters", ExactSpelling = true)]
                extern static int *__MR_SpacingSettings_GetMutable_numIters(_Underlying *_this);
                return ref *__MR_SpacingSettings_GetMutable_numIters(_UnderlyingPtr);
            }
        }

        /// too small number here can lead to instability, too large - to slow convergence
        public new unsafe ref float Stabilizer
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_GetMutable_stabilizer", ExactSpelling = true)]
                extern static float *__MR_SpacingSettings_GetMutable_stabilizer(_Underlying *_this);
                return ref *__MR_SpacingSettings_GetMutable_stabilizer(_UnderlyingPtr);
            }
        }

        /// maximum sum of minus negative weights, if it is exceeded then stabilizer is increased automatically
        public new unsafe ref float MaxSumNegW
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_GetMutable_maxSumNegW", ExactSpelling = true)]
                extern static float *__MR_SpacingSettings_GetMutable_maxSumNegW(_Underlying *_this);
                return ref *__MR_SpacingSettings_GetMutable_maxSumNegW(_UnderlyingPtr);
            }
        }

        /// if this predicated is given, then all inverted faces will be converted in degenerate faces at the end of each iteration
        public new unsafe MR.Std.Function_BoolFuncFromMRFaceId IsInverted
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_GetMutable_isInverted", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMRFaceId._Underlying *__MR_SpacingSettings_GetMutable_isInverted(_Underlying *_this);
                return new(__MR_SpacingSettings_GetMutable_isInverted(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SpacingSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_SpacingSettings_DefaultConstruct();
        }

        /// Constructs `MR::SpacingSettings` elementwise.
        public unsafe SpacingSettings(MR.Const_VertBitSet? region, MR.Std._ByValue_Function_FloatFuncFromMRUndirectedEdgeId dist, int numIters, float stabilizer, float maxSumNegW, MR.Std._ByValue_Function_BoolFuncFromMRFaceId isInverted) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_ConstructFrom(MR.Const_VertBitSet._Underlying *region, MR.Misc._PassBy dist_pass_by, MR.Std.Function_FloatFuncFromMRUndirectedEdgeId._Underlying *dist, int numIters, float stabilizer, float maxSumNegW, MR.Misc._PassBy isInverted_pass_by, MR.Std.Function_BoolFuncFromMRFaceId._Underlying *isInverted);
            _UnderlyingPtr = __MR_SpacingSettings_ConstructFrom(region is not null ? region._UnderlyingPtr : null, dist.PassByMode, dist.Value is not null ? dist.Value._UnderlyingPtr : null, numIters, stabilizer, maxSumNegW, isInverted.PassByMode, isInverted.Value is not null ? isInverted.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::SpacingSettings::SpacingSettings`.
        public unsafe SpacingSettings(MR._ByValue_SpacingSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.SpacingSettings._Underlying *_other);
            _UnderlyingPtr = __MR_SpacingSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::SpacingSettings::operator=`.
        public unsafe MR.SpacingSettings Assign(MR._ByValue_SpacingSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SpacingSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SpacingSettings._Underlying *__MR_SpacingSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.SpacingSettings._Underlying *_other);
            return new(__MR_SpacingSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `SpacingSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `SpacingSettings`/`Const_SpacingSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_SpacingSettings
    {
        internal readonly Const_SpacingSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_SpacingSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_SpacingSettings(Const_SpacingSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_SpacingSettings(Const_SpacingSettings arg) {return new(arg);}
        public _ByValue_SpacingSettings(MR.Misc._Moved<SpacingSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_SpacingSettings(MR.Misc._Moved<SpacingSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `SpacingSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SpacingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SpacingSettings`/`Const_SpacingSettings` directly.
    public class _InOptMut_SpacingSettings
    {
        public SpacingSettings? Opt;

        public _InOptMut_SpacingSettings() {}
        public _InOptMut_SpacingSettings(SpacingSettings value) {Opt = value;}
        public static implicit operator _InOptMut_SpacingSettings(SpacingSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `SpacingSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SpacingSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SpacingSettings`/`Const_SpacingSettings` to pass it to the function.
    public class _InOptConst_SpacingSettings
    {
        public Const_SpacingSettings? Opt;

        public _InOptConst_SpacingSettings() {}
        public _InOptConst_SpacingSettings(Const_SpacingSettings value) {Opt = value;}
        public static implicit operator _InOptConst_SpacingSettings(Const_SpacingSettings value) {return new(value);}
    }

    /// Generated from class `MR::InflateSettings`.
    /// This is the const half of the class.
    public class Const_InflateSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_InflateSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_InflateSettings_Destroy(_Underlying *_this);
            __MR_InflateSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_InflateSettings() {Dispose(false);}

        /// the amount of pressure applied to mesh region:
        /// positive pressure moves the vertices outside, negative - inside;
        /// please specify a value by magnitude about the region diagonal
        public unsafe float Pressure
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_Get_pressure", ExactSpelling = true)]
                extern static float *__MR_InflateSettings_Get_pressure(_Underlying *_this);
                return *__MR_InflateSettings_Get_pressure(_UnderlyingPtr);
            }
        }

        /// the number of internal iterations (>=1);
        /// larger number of iterations makes the performance slower, but the quality better
        public unsafe int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_Get_iterations", ExactSpelling = true)]
                extern static int *__MR_InflateSettings_Get_iterations(_Underlying *_this);
                return *__MR_InflateSettings_Get_iterations(_UnderlyingPtr);
            }
        }

        /// smooths the area before starting inflation;
        /// please set to false only if the region is known to be already smooth
        public unsafe bool PreSmooth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_Get_preSmooth", ExactSpelling = true)]
                extern static bool *__MR_InflateSettings_Get_preSmooth(_Underlying *_this);
                return *__MR_InflateSettings_Get_preSmooth(_UnderlyingPtr);
            }
        }

        /// whether to increase the pressure gradually during the iterations (recommended for best quality)
        public unsafe bool GradualPressureGrowth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_Get_gradualPressureGrowth", ExactSpelling = true)]
                extern static bool *__MR_InflateSettings_Get_gradualPressureGrowth(_Underlying *_this);
                return *__MR_InflateSettings_Get_gradualPressureGrowth(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_InflateSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_InflateSettings_DefaultConstruct();
        }

        /// Constructs `MR::InflateSettings` elementwise.
        public unsafe Const_InflateSettings(float pressure, int iterations, bool preSmooth, bool gradualPressureGrowth) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_ConstructFrom(float pressure, int iterations, byte preSmooth, byte gradualPressureGrowth);
            _UnderlyingPtr = __MR_InflateSettings_ConstructFrom(pressure, iterations, preSmooth ? (byte)1 : (byte)0, gradualPressureGrowth ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::InflateSettings::InflateSettings`.
        public unsafe Const_InflateSettings(MR.Const_InflateSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_ConstructFromAnother(MR.InflateSettings._Underlying *_other);
            _UnderlyingPtr = __MR_InflateSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::InflateSettings`.
    /// This is the non-const half of the class.
    public class InflateSettings : Const_InflateSettings
    {
        internal unsafe InflateSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the amount of pressure applied to mesh region:
        /// positive pressure moves the vertices outside, negative - inside;
        /// please specify a value by magnitude about the region diagonal
        public new unsafe ref float Pressure
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_GetMutable_pressure", ExactSpelling = true)]
                extern static float *__MR_InflateSettings_GetMutable_pressure(_Underlying *_this);
                return ref *__MR_InflateSettings_GetMutable_pressure(_UnderlyingPtr);
            }
        }

        /// the number of internal iterations (>=1);
        /// larger number of iterations makes the performance slower, but the quality better
        public new unsafe ref int Iterations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_GetMutable_iterations", ExactSpelling = true)]
                extern static int *__MR_InflateSettings_GetMutable_iterations(_Underlying *_this);
                return ref *__MR_InflateSettings_GetMutable_iterations(_UnderlyingPtr);
            }
        }

        /// smooths the area before starting inflation;
        /// please set to false only if the region is known to be already smooth
        public new unsafe ref bool PreSmooth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_GetMutable_preSmooth", ExactSpelling = true)]
                extern static bool *__MR_InflateSettings_GetMutable_preSmooth(_Underlying *_this);
                return ref *__MR_InflateSettings_GetMutable_preSmooth(_UnderlyingPtr);
            }
        }

        /// whether to increase the pressure gradually during the iterations (recommended for best quality)
        public new unsafe ref bool GradualPressureGrowth
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_GetMutable_gradualPressureGrowth", ExactSpelling = true)]
                extern static bool *__MR_InflateSettings_GetMutable_gradualPressureGrowth(_Underlying *_this);
                return ref *__MR_InflateSettings_GetMutable_gradualPressureGrowth(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe InflateSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_InflateSettings_DefaultConstruct();
        }

        /// Constructs `MR::InflateSettings` elementwise.
        public unsafe InflateSettings(float pressure, int iterations, bool preSmooth, bool gradualPressureGrowth) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_ConstructFrom(float pressure, int iterations, byte preSmooth, byte gradualPressureGrowth);
            _UnderlyingPtr = __MR_InflateSettings_ConstructFrom(pressure, iterations, preSmooth ? (byte)1 : (byte)0, gradualPressureGrowth ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::InflateSettings::InflateSettings`.
        public unsafe InflateSettings(MR.Const_InflateSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_ConstructFromAnother(MR.InflateSettings._Underlying *_other);
            _UnderlyingPtr = __MR_InflateSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::InflateSettings::operator=`.
        public unsafe MR.InflateSettings Assign(MR.Const_InflateSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InflateSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.InflateSettings._Underlying *__MR_InflateSettings_AssignFromAnother(_Underlying *_this, MR.InflateSettings._Underlying *_other);
            return new(__MR_InflateSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `InflateSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_InflateSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InflateSettings`/`Const_InflateSettings` directly.
    public class _InOptMut_InflateSettings
    {
        public InflateSettings? Opt;

        public _InOptMut_InflateSettings() {}
        public _InOptMut_InflateSettings(InflateSettings value) {Opt = value;}
        public static implicit operator _InOptMut_InflateSettings(InflateSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `InflateSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_InflateSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InflateSettings`/`Const_InflateSettings` to pass it to the function.
    public class _InOptConst_InflateSettings
    {
        public Const_InflateSettings? Opt;

        public _InOptConst_InflateSettings() {}
        public _InOptConst_InflateSettings(Const_InflateSettings value) {Opt = value;}
        public static implicit operator _InOptConst_InflateSettings(Const_InflateSettings value) {return new(value);}
    }

    /// Puts given vertices in such positions to make smooth surface both inside verts-region and on its boundary;
    /// \param verts must not include all vertices of a mesh connected component
    /// \param fixedSharpVertices in these vertices the surface can be not-smooth
    /// Generated from function `MR::positionVertsSmoothly`.
    /// Parameter `edgeWeights` defaults to `EdgeWeights::Cotan`.
    /// Parameter `vmass` defaults to `VertexMass::Unit`.
    public static unsafe void PositionVertsSmoothly(MR.Mesh mesh, MR.Const_VertBitSet verts, MR.EdgeWeights? edgeWeights = null, MR.VertexMass? vmass = null, MR.Const_VertBitSet? fixedSharpVertices = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsSmoothly_5", ExactSpelling = true)]
        extern static void __MR_positionVertsSmoothly_5(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *verts, MR.EdgeWeights *edgeWeights, MR.VertexMass *vmass, MR.Const_VertBitSet._Underlying *fixedSharpVertices);
        MR.EdgeWeights __deref_edgeWeights = edgeWeights.GetValueOrDefault();
        MR.VertexMass __deref_vmass = vmass.GetValueOrDefault();
        __MR_positionVertsSmoothly_5(mesh._UnderlyingPtr, verts._UnderlyingPtr, edgeWeights.HasValue ? &__deref_edgeWeights : null, vmass.HasValue ? &__deref_vmass : null, fixedSharpVertices is not null ? fixedSharpVertices._UnderlyingPtr : null);
    }

    /// Generated from function `MR::positionVertsSmoothly`.
    /// Parameter `edgeWeights` defaults to `EdgeWeights::Cotan`.
    /// Parameter `vmass` defaults to `VertexMass::Unit`.
    public static unsafe void PositionVertsSmoothly(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_VertBitSet verts, MR.EdgeWeights? edgeWeights = null, MR.VertexMass? vmass = null, MR.Const_VertBitSet? fixedSharpVertices = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsSmoothly_6", ExactSpelling = true)]
        extern static void __MR_positionVertsSmoothly_6(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *verts, MR.EdgeWeights *edgeWeights, MR.VertexMass *vmass, MR.Const_VertBitSet._Underlying *fixedSharpVertices);
        MR.EdgeWeights __deref_edgeWeights = edgeWeights.GetValueOrDefault();
        MR.VertexMass __deref_vmass = vmass.GetValueOrDefault();
        __MR_positionVertsSmoothly_6(topology._UnderlyingPtr, points._UnderlyingPtr, verts._UnderlyingPtr, edgeWeights.HasValue ? &__deref_edgeWeights : null, vmass.HasValue ? &__deref_vmass : null, fixedSharpVertices is not null ? fixedSharpVertices._UnderlyingPtr : null);
    }

    /// Puts given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary;
    /// Generated from function `MR::positionVertsSmoothlySharpBd`.
    public static unsafe void PositionVertsSmoothlySharpBd(MR.Mesh mesh, MR.Const_PositionVertsSmoothlyParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsSmoothlySharpBd_2_MR_PositionVertsSmoothlyParams", ExactSpelling = true)]
        extern static void __MR_positionVertsSmoothlySharpBd_2_MR_PositionVertsSmoothlyParams(MR.Mesh._Underlying *mesh, MR.Const_PositionVertsSmoothlyParams._Underlying *params_);
        __MR_positionVertsSmoothlySharpBd_2_MR_PositionVertsSmoothlyParams(mesh._UnderlyingPtr, params_._UnderlyingPtr);
    }

    /// Generated from function `MR::positionVertsSmoothlySharpBd`.
    public static unsafe void PositionVertsSmoothlySharpBd(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_PositionVertsSmoothlyParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsSmoothlySharpBd_3", ExactSpelling = true)]
        extern static void __MR_positionVertsSmoothlySharpBd_3(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_PositionVertsSmoothlyParams._Underlying *params_);
        __MR_positionVertsSmoothlySharpBd_3(topology._UnderlyingPtr, points._UnderlyingPtr, params_._UnderlyingPtr);
    }

    /// Generated from function `MR::positionVertsSmoothlySharpBd`.
    [Obsolete]
    public static unsafe void PositionVertsSmoothlySharpBd(MR.Mesh mesh, MR.Const_VertBitSet verts)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsSmoothlySharpBd_2_MR_VertBitSet", ExactSpelling = true)]
        extern static void __MR_positionVertsSmoothlySharpBd_2_MR_VertBitSet(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *verts);
        __MR_positionVertsSmoothlySharpBd_2_MR_VertBitSet(mesh._UnderlyingPtr, verts._UnderlyingPtr);
    }

    /// Moves given vertices to make the distances between them as specified
    /// Generated from function `MR::positionVertsWithSpacing`.
    public static unsafe void PositionVertsWithSpacing(MR.Mesh mesh, MR.Const_SpacingSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsWithSpacing_2", ExactSpelling = true)]
        extern static void __MR_positionVertsWithSpacing_2(MR.Mesh._Underlying *mesh, MR.Const_SpacingSettings._Underlying *settings);
        __MR_positionVertsWithSpacing_2(mesh._UnderlyingPtr, settings._UnderlyingPtr);
    }

    /// Generated from function `MR::positionVertsWithSpacing`.
    public static unsafe void PositionVertsWithSpacing(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_SpacingSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_positionVertsWithSpacing_3", ExactSpelling = true)]
        extern static void __MR_positionVertsWithSpacing_3(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_SpacingSettings._Underlying *settings);
        __MR_positionVertsWithSpacing_3(topology._UnderlyingPtr, points._UnderlyingPtr, settings._UnderlyingPtr);
    }

    /// Inflates (in one of two sides) given mesh region,
    /// putting given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary;
    /// \param verts must not include all vertices of a mesh connected component
    /// Generated from function `MR::inflate`.
    public static unsafe void Inflate(MR.Mesh mesh, MR.Const_VertBitSet verts, MR.Const_InflateSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_inflate_3", ExactSpelling = true)]
        extern static void __MR_inflate_3(MR.Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *verts, MR.Const_InflateSettings._Underlying *settings);
        __MR_inflate_3(mesh._UnderlyingPtr, verts._UnderlyingPtr, settings._UnderlyingPtr);
    }

    /// Generated from function `MR::inflate`.
    public static unsafe void Inflate(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_VertBitSet verts, MR.Const_InflateSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_inflate_4", ExactSpelling = true)]
        extern static void __MR_inflate_4(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *verts, MR.Const_InflateSettings._Underlying *settings);
        __MR_inflate_4(topology._UnderlyingPtr, points._UnderlyingPtr, verts._UnderlyingPtr, settings._UnderlyingPtr);
    }

    /// Inflates (in one of two sides) given mesh region,
    /// putting given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary;
    /// this function makes just 1 iteration of inflation and is used inside inflate(...)
    /// Generated from function `MR::inflate1`.
    public static unsafe void Inflate1(MR.Const_MeshTopology topology, MR.VertCoords points, MR.Const_VertBitSet verts, float pressure)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_inflate1", ExactSpelling = true)]
        extern static void __MR_inflate1(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points, MR.Const_VertBitSet._Underlying *verts, float pressure);
        __MR_inflate1(topology._UnderlyingPtr, points._UnderlyingPtr, verts._UnderlyingPtr, pressure);
    }
}
