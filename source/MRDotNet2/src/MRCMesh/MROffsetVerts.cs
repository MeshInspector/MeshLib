public static partial class MR
{
    /// Generated from class `MR::ThickenParams`.
    /// This is the const half of the class.
    public class Const_ThickenParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ThickenParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ThickenParams_Destroy(_Underlying *_this);
            __MR_ThickenParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ThickenParams() {Dispose(false);}

        /// the amount of offset for original mesh vertices
        public unsafe float OutsideOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_Get_outsideOffset", ExactSpelling = true)]
                extern static float *__MR_ThickenParams_Get_outsideOffset(_Underlying *_this);
                return *__MR_ThickenParams_Get_outsideOffset(_UnderlyingPtr);
            }
        }

        /// the amount of offset for cloned mirrored mesh vertices in the opposite direction
        public unsafe float InsideOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_Get_insideOffset", ExactSpelling = true)]
                extern static float *__MR_ThickenParams_Get_insideOffset(_Underlying *_this);
                return *__MR_ThickenParams_Get_insideOffset(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ThickenParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ThickenParams_DefaultConstruct();
        }

        /// Constructs `MR::ThickenParams` elementwise.
        public unsafe Const_ThickenParams(float outsideOffset, float insideOffset) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_ConstructFrom(float outsideOffset, float insideOffset);
            _UnderlyingPtr = __MR_ThickenParams_ConstructFrom(outsideOffset, insideOffset);
        }

        /// Generated from constructor `MR::ThickenParams::ThickenParams`.
        public unsafe Const_ThickenParams(MR.Const_ThickenParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_ConstructFromAnother(MR.ThickenParams._Underlying *_other);
            _UnderlyingPtr = __MR_ThickenParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::ThickenParams`.
    /// This is the non-const half of the class.
    public class ThickenParams : Const_ThickenParams
    {
        internal unsafe ThickenParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the amount of offset for original mesh vertices
        public new unsafe ref float OutsideOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_GetMutable_outsideOffset", ExactSpelling = true)]
                extern static float *__MR_ThickenParams_GetMutable_outsideOffset(_Underlying *_this);
                return ref *__MR_ThickenParams_GetMutable_outsideOffset(_UnderlyingPtr);
            }
        }

        /// the amount of offset for cloned mirrored mesh vertices in the opposite direction
        public new unsafe ref float InsideOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_GetMutable_insideOffset", ExactSpelling = true)]
                extern static float *__MR_ThickenParams_GetMutable_insideOffset(_Underlying *_this);
                return ref *__MR_ThickenParams_GetMutable_insideOffset(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ThickenParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ThickenParams_DefaultConstruct();
        }

        /// Constructs `MR::ThickenParams` elementwise.
        public unsafe ThickenParams(float outsideOffset, float insideOffset) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_ConstructFrom(float outsideOffset, float insideOffset);
            _UnderlyingPtr = __MR_ThickenParams_ConstructFrom(outsideOffset, insideOffset);
        }

        /// Generated from constructor `MR::ThickenParams::ThickenParams`.
        public unsafe ThickenParams(MR.Const_ThickenParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_ConstructFromAnother(MR.ThickenParams._Underlying *_other);
            _UnderlyingPtr = __MR_ThickenParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ThickenParams::operator=`.
        public unsafe MR.ThickenParams Assign(MR.Const_ThickenParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ThickenParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ThickenParams._Underlying *__MR_ThickenParams_AssignFromAnother(_Underlying *_this, MR.ThickenParams._Underlying *_other);
            return new(__MR_ThickenParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ThickenParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ThickenParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ThickenParams`/`Const_ThickenParams` directly.
    public class _InOptMut_ThickenParams
    {
        public ThickenParams? Opt;

        public _InOptMut_ThickenParams() {}
        public _InOptMut_ThickenParams(ThickenParams value) {Opt = value;}
        public static implicit operator _InOptMut_ThickenParams(ThickenParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ThickenParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ThickenParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ThickenParams`/`Const_ThickenParams` to pass it to the function.
    public class _InOptConst_ThickenParams
    {
        public Const_ThickenParams? Opt;

        public _InOptConst_ThickenParams() {}
        public _InOptConst_ThickenParams(Const_ThickenParams value) {Opt = value;}
        public static implicit operator _InOptConst_ThickenParams(Const_ThickenParams value) {return new(value);}
    }

    /// Generated from class `MR::ZCompensateParams`.
    /// This is the const half of the class.
    public class Const_ZCompensateParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ZCompensateParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ZCompensateParams_Destroy(_Underlying *_this);
            __MR_ZCompensateParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ZCompensateParams() {Dispose(false);}

        /// shift of mesh parts orthogonal to Z-axis with normal against Z-axis;
        /// for other mesh parts the shift will be less and will depend on the angle between point pseudo-normal and Z-axis
        public unsafe float MaxShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_Get_maxShift", ExactSpelling = true)]
                extern static float *__MR_ZCompensateParams_Get_maxShift(_Underlying *_this);
                return *__MR_ZCompensateParams_Get_maxShift(_UnderlyingPtr);
            }
        }

        /// if true, limits the movement of each vertex to reduce self-intersections in the mesh
        public unsafe bool ReduceSelfIntersections
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_Get_reduceSelfIntersections", ExactSpelling = true)]
                extern static bool *__MR_ZCompensateParams_Get_reduceSelfIntersections(_Underlying *_this);
                return *__MR_ZCompensateParams_Get_reduceSelfIntersections(_UnderlyingPtr);
            }
        }

        /// only if (reduceSelfIntersections = true), avoids moving a vertex closer than this distance to another triangle
        public unsafe float MinThickness
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_Get_minThickness", ExactSpelling = true)]
                extern static float *__MR_ZCompensateParams_Get_minThickness(_Underlying *_this);
                return *__MR_ZCompensateParams_Get_minThickness(_UnderlyingPtr);
            }
        }

        /// to report progress and cancel processing
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_ZCompensateParams_Get_progress(_Underlying *_this);
                return new(__MR_ZCompensateParams_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ZCompensateParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ZCompensateParams_DefaultConstruct();
        }

        /// Constructs `MR::ZCompensateParams` elementwise.
        public unsafe Const_ZCompensateParams(float maxShift, bool reduceSelfIntersections, float minThickness, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_ConstructFrom(float maxShift, byte reduceSelfIntersections, float minThickness, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_ZCompensateParams_ConstructFrom(maxShift, reduceSelfIntersections ? (byte)1 : (byte)0, minThickness, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ZCompensateParams::ZCompensateParams`.
        public unsafe Const_ZCompensateParams(MR._ByValue_ZCompensateParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ZCompensateParams._Underlying *_other);
            _UnderlyingPtr = __MR_ZCompensateParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::ZCompensateParams`.
    /// This is the non-const half of the class.
    public class ZCompensateParams : Const_ZCompensateParams
    {
        internal unsafe ZCompensateParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// shift of mesh parts orthogonal to Z-axis with normal against Z-axis;
        /// for other mesh parts the shift will be less and will depend on the angle between point pseudo-normal and Z-axis
        public new unsafe ref float MaxShift
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_GetMutable_maxShift", ExactSpelling = true)]
                extern static float *__MR_ZCompensateParams_GetMutable_maxShift(_Underlying *_this);
                return ref *__MR_ZCompensateParams_GetMutable_maxShift(_UnderlyingPtr);
            }
        }

        /// if true, limits the movement of each vertex to reduce self-intersections in the mesh
        public new unsafe ref bool ReduceSelfIntersections
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_GetMutable_reduceSelfIntersections", ExactSpelling = true)]
                extern static bool *__MR_ZCompensateParams_GetMutable_reduceSelfIntersections(_Underlying *_this);
                return ref *__MR_ZCompensateParams_GetMutable_reduceSelfIntersections(_UnderlyingPtr);
            }
        }

        /// only if (reduceSelfIntersections = true), avoids moving a vertex closer than this distance to another triangle
        public new unsafe ref float MinThickness
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_GetMutable_minThickness", ExactSpelling = true)]
                extern static float *__MR_ZCompensateParams_GetMutable_minThickness(_Underlying *_this);
                return ref *__MR_ZCompensateParams_GetMutable_minThickness(_UnderlyingPtr);
            }
        }

        /// to report progress and cancel processing
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_ZCompensateParams_GetMutable_progress(_Underlying *_this);
                return new(__MR_ZCompensateParams_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ZCompensateParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ZCompensateParams_DefaultConstruct();
        }

        /// Constructs `MR::ZCompensateParams` elementwise.
        public unsafe ZCompensateParams(float maxShift, bool reduceSelfIntersections, float minThickness, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_ConstructFrom(float maxShift, byte reduceSelfIntersections, float minThickness, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_ZCompensateParams_ConstructFrom(maxShift, reduceSelfIntersections ? (byte)1 : (byte)0, minThickness, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::ZCompensateParams::ZCompensateParams`.
        public unsafe ZCompensateParams(MR._ByValue_ZCompensateParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ZCompensateParams._Underlying *_other);
            _UnderlyingPtr = __MR_ZCompensateParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ZCompensateParams::operator=`.
        public unsafe MR.ZCompensateParams Assign(MR._ByValue_ZCompensateParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ZCompensateParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ZCompensateParams._Underlying *__MR_ZCompensateParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ZCompensateParams._Underlying *_other);
            return new(__MR_ZCompensateParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `ZCompensateParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ZCompensateParams`/`Const_ZCompensateParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ZCompensateParams
    {
        internal readonly Const_ZCompensateParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ZCompensateParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ZCompensateParams(Const_ZCompensateParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_ZCompensateParams(Const_ZCompensateParams arg) {return new(arg);}
        public _ByValue_ZCompensateParams(MR.Misc._Moved<ZCompensateParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ZCompensateParams(MR.Misc._Moved<ZCompensateParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ZCompensateParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ZCompensateParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ZCompensateParams`/`Const_ZCompensateParams` directly.
    public class _InOptMut_ZCompensateParams
    {
        public ZCompensateParams? Opt;

        public _InOptMut_ZCompensateParams() {}
        public _InOptMut_ZCompensateParams(ZCompensateParams value) {Opt = value;}
        public static implicit operator _InOptMut_ZCompensateParams(ZCompensateParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ZCompensateParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ZCompensateParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ZCompensateParams`/`Const_ZCompensateParams` to pass it to the function.
    public class _InOptConst_ZCompensateParams
    {
        public Const_ZCompensateParams? Opt;

        public _InOptConst_ZCompensateParams() {}
        public _InOptConst_ZCompensateParams(Const_ZCompensateParams value) {Opt = value;}
        public static implicit operator _InOptConst_ZCompensateParams(Const_ZCompensateParams value) {return new(value);}
    }

    /// Modifies \p mesh shifting each vertex along its pseudonormal by the corresponding \p offset
    /// @return false if cancelled.
    /// Generated from function `MR::offsetVerts`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe bool OffsetVerts(MR.Mesh mesh, MR.Std.Const_Function_FloatFuncFromMRVertId offset, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_offsetVerts", ExactSpelling = true)]
        extern static byte __MR_offsetVerts(MR.Mesh._Underlying *mesh, MR.Std.Const_Function_FloatFuncFromMRVertId._Underlying *offset, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
        return __MR_offsetVerts(mesh._UnderlyingPtr, offset._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null) != 0;
    }

    /// given a mesh \p m, representing a surface,
    /// creates new closed mesh by cloning mirrored mesh, and shifting original part and cloned part in different directions according to \p params,
    /// if original mesh was open then stitches corresponding boundaries of two parts
    /// Generated from function `MR::makeThickMesh`.
    public static unsafe MR.Misc._Moved<MR.Mesh> MakeThickMesh(MR.Const_Mesh m, MR.Const_ThickenParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeThickMesh", ExactSpelling = true)]
        extern static MR.Mesh._Underlying *__MR_makeThickMesh(MR.Const_Mesh._Underlying *m, MR.Const_ThickenParams._Underlying *params_);
        return MR.Misc.Move(new MR.Mesh(__MR_makeThickMesh(m._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// For 3D printers: shifts every vertex with normal having negative projection on Z-axis, along Z-axis;
    /// mesh's topology is preserved unchanged
    /// @return false if cancelled.
    /// Generated from function `MR::zCompensate`.
    public static unsafe bool ZCompensate(MR.Mesh mesh, MR.Const_ZCompensateParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_zCompensate", ExactSpelling = true)]
        extern static byte __MR_zCompensate(MR.Mesh._Underlying *mesh, MR.Const_ZCompensateParams._Underlying *params_);
        return __MR_zCompensate(mesh._UnderlyingPtr, params_._UnderlyingPtr) != 0;
    }

    /// finds the shift along z-axis for each vertex without modifying the mesh
    /// Generated from function `MR::findZcompensationShifts`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertScalars> FindZcompensationShifts(MR.Const_Mesh mesh, MR.Const_ZCompensateParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findZcompensationShifts", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertScalars._Underlying *__MR_findZcompensationShifts(MR.Const_Mesh._Underlying *mesh, MR.Const_ZCompensateParams._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Optional_MRVertScalars(__MR_findZcompensationShifts(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }

    /// finds vertices positions of the mesh after z-compensation without modifying the mesh
    /// Generated from function `MR::findZcompensatedPositions`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertCoords> FindZcompensatedPositions(MR.Const_Mesh mesh, MR.Const_ZCompensateParams params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findZcompensatedPositions", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertCoords._Underlying *__MR_findZcompensatedPositions(MR.Const_Mesh._Underlying *mesh, MR.Const_ZCompensateParams._Underlying *params_);
        return MR.Misc.Move(new MR.Std.Optional_MRVertCoords(__MR_findZcompensatedPositions(mesh._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
