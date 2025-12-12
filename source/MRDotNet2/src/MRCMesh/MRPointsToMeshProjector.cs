public static partial class MR
{
    /// Abstract class, computes the closest point on mesh to each of given points. Pure virtual functions must be implemented
    /// Generated from class `MR::IPointsToMeshProjector`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointsToMeshProjector`
    /// This is the const half of the class.
    public class Const_IPointsToMeshProjector : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_IPointsToMeshProjector_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_IPointsToMeshProjector_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_IPointsToMeshProjector_UseCount();
                return __MR_std_shared_ptr_MR_IPointsToMeshProjector_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_IPointsToMeshProjector(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IPointsToMeshProjector_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_IPointsToMeshProjector_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructNonOwning(ptr);
        }

        internal unsafe Const_IPointsToMeshProjector(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe IPointsToMeshProjector _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_IPointsToMeshProjector_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IPointsToMeshProjector_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_IPointsToMeshProjector_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IPointsToMeshProjector_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_IPointsToMeshProjector_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_IPointsToMeshProjector_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IPointsToMeshProjector() {Dispose(false);}

        /// Returns amount of memory needed to compute projections
        /// Generated from method `MR::IPointsToMeshProjector::projectionsHeapBytes`.
        public unsafe ulong ProjectionsHeapBytes(ulong numProjections)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsToMeshProjector_projectionsHeapBytes", ExactSpelling = true)]
            extern static ulong __MR_IPointsToMeshProjector_projectionsHeapBytes(_Underlying *_this, ulong numProjections);
            return __MR_IPointsToMeshProjector_projectionsHeapBytes(_UnderlyingPtr, numProjections);
        }
    }

    /// Abstract class, computes the closest point on mesh to each of given points. Pure virtual functions must be implemented
    /// Generated from class `MR::IPointsToMeshProjector`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::PointsToMeshProjector`
    /// This is the non-const half of the class.
    public class IPointsToMeshProjector : Const_IPointsToMeshProjector
    {
        internal unsafe IPointsToMeshProjector(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe IPointsToMeshProjector(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// Updates all data related to the referencing mesh
        /// Generated from method `MR::IPointsToMeshProjector::updateMeshData`.
        public unsafe void UpdateMeshData(MR.Const_Mesh? mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsToMeshProjector_updateMeshData", ExactSpelling = true)]
            extern static void __MR_IPointsToMeshProjector_updateMeshData(_Underlying *_this, MR.Const_Mesh._Underlying *mesh);
            __MR_IPointsToMeshProjector_updateMeshData(_UnderlyingPtr, mesh is not null ? mesh._UnderlyingPtr : null);
        }

        /// Computes the closest point on mesh to each of given points
        /// Generated from method `MR::IPointsToMeshProjector::findProjections`.
        /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
        /// Parameter `loDistLimitSq` defaults to `0.0f`.
        public unsafe void FindProjections(MR.Std.Vector_MRMeshProjectionResult result, MR.Std.Const_Vector_MRVector3f points, MR.Const_AffineXf3f? worldXf = null, MR.Const_AffineXf3f? worldRefXf = null, float? upDistLimitSq = null, float? loDistLimitSq = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsToMeshProjector_findProjections", ExactSpelling = true)]
            extern static void __MR_IPointsToMeshProjector_findProjections(_Underlying *_this, MR.Std.Vector_MRMeshProjectionResult._Underlying *result, MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_AffineXf3f._Underlying *worldXf, MR.Const_AffineXf3f._Underlying *worldRefXf, float *upDistLimitSq, float *loDistLimitSq);
            float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
            float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
            __MR_IPointsToMeshProjector_findProjections(_UnderlyingPtr, result._UnderlyingPtr, points._UnderlyingPtr, worldXf is not null ? worldXf._UnderlyingPtr : null, worldRefXf is not null ? worldRefXf._UnderlyingPtr : null, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null);
        }
    }

    /// This is used as a function parameter when the underlying function receives `IPointsToMeshProjector` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `IPointsToMeshProjector`/`Const_IPointsToMeshProjector` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_IPointsToMeshProjector
    {
        internal readonly Const_IPointsToMeshProjector? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `IPointsToMeshProjector` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IPointsToMeshProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IPointsToMeshProjector`/`Const_IPointsToMeshProjector` directly.
    public class _InOptMut_IPointsToMeshProjector
    {
        public IPointsToMeshProjector? Opt;

        public _InOptMut_IPointsToMeshProjector() {}
        public _InOptMut_IPointsToMeshProjector(IPointsToMeshProjector value) {Opt = value;}
        public static implicit operator _InOptMut_IPointsToMeshProjector(IPointsToMeshProjector value) {return new(value);}
    }

    /// This is used for optional parameters of class `IPointsToMeshProjector` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IPointsToMeshProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IPointsToMeshProjector`/`Const_IPointsToMeshProjector` to pass it to the function.
    public class _InOptConst_IPointsToMeshProjector
    {
        public Const_IPointsToMeshProjector? Opt;

        public _InOptConst_IPointsToMeshProjector() {}
        public _InOptConst_IPointsToMeshProjector(Const_IPointsToMeshProjector value) {Opt = value;}
        public static implicit operator _InOptConst_IPointsToMeshProjector(Const_IPointsToMeshProjector value) {return new(value);}
    }

    /// Generated from class `MR::MeshProjectionParameters`.
    /// This is the const half of the class.
    public class Const_MeshProjectionParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshProjectionParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshProjectionParameters_Destroy(_Underlying *_this);
            __MR_MeshProjectionParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshProjectionParameters() {Dispose(false);}

        /// minimum squared distance from a test point to mesh to be computed precisely,
        /// if a mesh point is found within this distance then it is immediately returned without searching for a closer one
        public unsafe float LoDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_Get_loDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_MeshProjectionParameters_Get_loDistLimitSq(_Underlying *_this);
                return *__MR_MeshProjectionParameters_Get_loDistLimitSq(_UnderlyingPtr);
            }
        }

        /// maximum squared distance from a test point to mesh to be computed precisely,
        /// if actual distance is larger than upDistLimit will be returned with not-trusted sign
        public unsafe float UpDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_Get_upDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_MeshProjectionParameters_Get_upDistLimitSq(_Underlying *_this);
                return *__MR_MeshProjectionParameters_Get_upDistLimitSq(_UnderlyingPtr);
            }
        }

        /// optional reference mesh to world transformation
        public unsafe ref readonly MR.AffineXf3f * RefXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_Get_refXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionParameters_Get_refXf(_Underlying *_this);
                return ref *__MR_MeshProjectionParameters_Get_refXf(_UnderlyingPtr);
            }
        }

        /// optional test points to world transformation
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionParameters_Get_xf(_Underlying *_this);
                return ref *__MR_MeshProjectionParameters_Get_xf(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshProjectionParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshProjectionParameters_DefaultConstruct();
        }

        /// Constructs `MR::MeshProjectionParameters` elementwise.
        public unsafe Const_MeshProjectionParameters(float loDistLimitSq, float upDistLimitSq, MR.Const_AffineXf3f? refXf, MR.Const_AffineXf3f? xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_ConstructFrom(float loDistLimitSq, float upDistLimitSq, MR.Const_AffineXf3f._Underlying *refXf, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_MeshProjectionParameters_ConstructFrom(loDistLimitSq, upDistLimitSq, refXf is not null ? refXf._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshProjectionParameters::MeshProjectionParameters`.
        public unsafe Const_MeshProjectionParameters(MR.Const_MeshProjectionParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_ConstructFromAnother(MR.MeshProjectionParameters._Underlying *_other);
            _UnderlyingPtr = __MR_MeshProjectionParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshProjectionParameters`.
    /// This is the non-const half of the class.
    public class MeshProjectionParameters : Const_MeshProjectionParameters
    {
        internal unsafe MeshProjectionParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// minimum squared distance from a test point to mesh to be computed precisely,
        /// if a mesh point is found within this distance then it is immediately returned without searching for a closer one
        public new unsafe ref float LoDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_GetMutable_loDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_MeshProjectionParameters_GetMutable_loDistLimitSq(_Underlying *_this);
                return ref *__MR_MeshProjectionParameters_GetMutable_loDistLimitSq(_UnderlyingPtr);
            }
        }

        /// maximum squared distance from a test point to mesh to be computed precisely,
        /// if actual distance is larger than upDistLimit will be returned with not-trusted sign
        public new unsafe ref float UpDistLimitSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_GetMutable_upDistLimitSq", ExactSpelling = true)]
                extern static float *__MR_MeshProjectionParameters_GetMutable_upDistLimitSq(_Underlying *_this);
                return ref *__MR_MeshProjectionParameters_GetMutable_upDistLimitSq(_UnderlyingPtr);
            }
        }

        /// optional reference mesh to world transformation
        public new unsafe ref readonly MR.AffineXf3f * RefXf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_GetMutable_refXf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionParameters_GetMutable_refXf(_Underlying *_this);
                return ref *__MR_MeshProjectionParameters_GetMutable_refXf(_UnderlyingPtr);
            }
        }

        /// optional test points to world transformation
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionParameters_GetMutable_xf(_Underlying *_this);
                return ref *__MR_MeshProjectionParameters_GetMutable_xf(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshProjectionParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshProjectionParameters_DefaultConstruct();
        }

        /// Constructs `MR::MeshProjectionParameters` elementwise.
        public unsafe MeshProjectionParameters(float loDistLimitSq, float upDistLimitSq, MR.Const_AffineXf3f? refXf, MR.Const_AffineXf3f? xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_ConstructFrom(float loDistLimitSq, float upDistLimitSq, MR.Const_AffineXf3f._Underlying *refXf, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_MeshProjectionParameters_ConstructFrom(loDistLimitSq, upDistLimitSq, refXf is not null ? refXf._UnderlyingPtr : null, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshProjectionParameters::MeshProjectionParameters`.
        public unsafe MeshProjectionParameters(MR.Const_MeshProjectionParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_ConstructFromAnother(MR.MeshProjectionParameters._Underlying *_other);
            _UnderlyingPtr = __MR_MeshProjectionParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshProjectionParameters::operator=`.
        public unsafe MR.MeshProjectionParameters Assign(MR.Const_MeshProjectionParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionParameters._Underlying *__MR_MeshProjectionParameters_AssignFromAnother(_Underlying *_this, MR.MeshProjectionParameters._Underlying *_other);
            return new(__MR_MeshProjectionParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshProjectionParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshProjectionParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshProjectionParameters`/`Const_MeshProjectionParameters` directly.
    public class _InOptMut_MeshProjectionParameters
    {
        public MeshProjectionParameters? Opt;

        public _InOptMut_MeshProjectionParameters() {}
        public _InOptMut_MeshProjectionParameters(MeshProjectionParameters value) {Opt = value;}
        public static implicit operator _InOptMut_MeshProjectionParameters(MeshProjectionParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshProjectionParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshProjectionParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshProjectionParameters`/`Const_MeshProjectionParameters` to pass it to the function.
    public class _InOptConst_MeshProjectionParameters
    {
        public Const_MeshProjectionParameters? Opt;

        public _InOptConst_MeshProjectionParameters() {}
        public _InOptConst_MeshProjectionParameters(Const_MeshProjectionParameters value) {Opt = value;}
        public static implicit operator _InOptConst_MeshProjectionParameters(Const_MeshProjectionParameters value) {return new(value);}
    }

    /// Computes the closest point on mesh to each of given points on CPU
    /// Generated from class `MR::PointsToMeshProjector`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointsToMeshProjector`
    /// This is the const half of the class.
    public class Const_PointsToMeshProjector : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_PointsToMeshProjector_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_PointsToMeshProjector_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_PointsToMeshProjector_UseCount();
                return __MR_std_shared_ptr_MR_PointsToMeshProjector_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_PointsToMeshProjector(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointsToMeshProjector_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointsToMeshProjector_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructNonOwning(ptr);
        }

        internal unsafe Const_PointsToMeshProjector(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe PointsToMeshProjector _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_PointsToMeshProjector_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_PointsToMeshProjector_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_PointsToMeshProjector_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_PointsToMeshProjector_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_PointsToMeshProjector_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_PointsToMeshProjector_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointsToMeshProjector() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_IPointsToMeshProjector(Const_PointsToMeshProjector self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_UpcastTo_MR_IPointsToMeshProjector", ExactSpelling = true)]
            extern static MR.Const_IPointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_UpcastTo_MR_IPointsToMeshProjector(_Underlying *_this);
            return MR.Const_IPointsToMeshProjector._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointsToMeshProjector_UpcastTo_MR_IPointsToMeshProjector(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_PointsToMeshProjector?(MR.Const_IPointsToMeshProjector parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsToMeshProjector_DynamicDowncastTo_MR_PointsToMeshProjector", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointsToMeshProjector_DynamicDowncastTo_MR_PointsToMeshProjector(MR.Const_IPointsToMeshProjector._Underlying *_this);
            var ptr = __MR_IPointsToMeshProjector_DynamicDowncastTo_MR_PointsToMeshProjector(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_IPointsToMeshProjector._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointsToMeshProjector() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_DefaultConstruct();
            _LateMakeShared(__MR_PointsToMeshProjector_DefaultConstruct());
        }

        /// Generated from constructor `MR::PointsToMeshProjector::PointsToMeshProjector`.
        public unsafe Const_PointsToMeshProjector(MR._ByValue_PointsToMeshProjector _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsToMeshProjector._Underlying *_other);
            _LateMakeShared(__MR_PointsToMeshProjector_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Returns amount of additional memory needed to compute projections
        /// Generated from method `MR::PointsToMeshProjector::projectionsHeapBytes`.
        public unsafe ulong ProjectionsHeapBytes(ulong numProjections)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_projectionsHeapBytes", ExactSpelling = true)]
            extern static ulong __MR_PointsToMeshProjector_projectionsHeapBytes(_Underlying *_this, ulong numProjections);
            return __MR_PointsToMeshProjector_projectionsHeapBytes(_UnderlyingPtr, numProjections);
        }
    }

    /// Computes the closest point on mesh to each of given points on CPU
    /// Generated from class `MR::PointsToMeshProjector`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IPointsToMeshProjector`
    /// This is the non-const half of the class.
    public class PointsToMeshProjector : Const_PointsToMeshProjector
    {
        internal unsafe PointsToMeshProjector(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe PointsToMeshProjector(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.IPointsToMeshProjector(PointsToMeshProjector self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_UpcastTo_MR_IPointsToMeshProjector", ExactSpelling = true)]
            extern static MR.IPointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_UpcastTo_MR_IPointsToMeshProjector(_Underlying *_this);
            return MR.IPointsToMeshProjector._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_PointsToMeshProjector_UpcastTo_MR_IPointsToMeshProjector(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator PointsToMeshProjector?(MR.IPointsToMeshProjector parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IPointsToMeshProjector_DynamicDowncastTo_MR_PointsToMeshProjector", ExactSpelling = true)]
            extern static _Underlying *__MR_IPointsToMeshProjector_DynamicDowncastTo_MR_PointsToMeshProjector(MR.IPointsToMeshProjector._Underlying *_this);
            var ptr = __MR_IPointsToMeshProjector_DynamicDowncastTo_MR_PointsToMeshProjector(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.IPointsToMeshProjector._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointsToMeshProjector() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_DefaultConstruct();
            _LateMakeShared(__MR_PointsToMeshProjector_DefaultConstruct());
        }

        /// Generated from constructor `MR::PointsToMeshProjector::PointsToMeshProjector`.
        public unsafe PointsToMeshProjector(MR._ByValue_PointsToMeshProjector _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointsToMeshProjector._Underlying *_other);
            _LateMakeShared(__MR_PointsToMeshProjector_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::PointsToMeshProjector::operator=`.
        public unsafe MR.PointsToMeshProjector Assign(MR._ByValue_PointsToMeshProjector _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointsToMeshProjector._Underlying *__MR_PointsToMeshProjector_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointsToMeshProjector._Underlying *_other);
            return new(__MR_PointsToMeshProjector_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// update all data related to the referencing mesh
        /// Generated from method `MR::PointsToMeshProjector::updateMeshData`.
        public unsafe void UpdateMeshData(MR.Const_Mesh? mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_updateMeshData", ExactSpelling = true)]
            extern static void __MR_PointsToMeshProjector_updateMeshData(_Underlying *_this, MR.Const_Mesh._Underlying *mesh);
            __MR_PointsToMeshProjector_updateMeshData(_UnderlyingPtr, mesh is not null ? mesh._UnderlyingPtr : null);
        }

        /// <summary>
        /// Computes the closest point on mesh to each of given points
        /// </summary>
        /// <param name="result">vector pf projections</param>
        /// <param name="points">vector of points to project</param>
        /// <param name="objXf">transform applied to points</param>
        /// <param name="refObjXf">transform applied to referencing mesh</param>
        /// <param name="upDistLimitSq">maximal squared distance from point to mesh</param>
        /// <param name="loDistLimitSq">minimal squared distance from point to mesh</param>
        /// Generated from method `MR::PointsToMeshProjector::findProjections`.
        public unsafe void FindProjections(MR.Std.Vector_MRMeshProjectionResult result, MR.Std.Const_Vector_MRVector3f points, MR.Const_AffineXf3f? objXf, MR.Const_AffineXf3f? refObjXf, float upDistLimitSq, float loDistLimitSq)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointsToMeshProjector_findProjections", ExactSpelling = true)]
            extern static void __MR_PointsToMeshProjector_findProjections(_Underlying *_this, MR.Std.Vector_MRMeshProjectionResult._Underlying *result, MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_AffineXf3f._Underlying *objXf, MR.Const_AffineXf3f._Underlying *refObjXf, float upDistLimitSq, float loDistLimitSq);
            __MR_PointsToMeshProjector_findProjections(_UnderlyingPtr, result._UnderlyingPtr, points._UnderlyingPtr, objXf is not null ? objXf._UnderlyingPtr : null, refObjXf is not null ? refObjXf._UnderlyingPtr : null, upDistLimitSq, loDistLimitSq);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointsToMeshProjector` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointsToMeshProjector`/`Const_PointsToMeshProjector` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointsToMeshProjector
    {
        internal readonly Const_PointsToMeshProjector? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointsToMeshProjector() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointsToMeshProjector(Const_PointsToMeshProjector new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointsToMeshProjector(Const_PointsToMeshProjector arg) {return new(arg);}
        public _ByValue_PointsToMeshProjector(MR.Misc._Moved<PointsToMeshProjector> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointsToMeshProjector(MR.Misc._Moved<PointsToMeshProjector> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointsToMeshProjector` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointsToMeshProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsToMeshProjector`/`Const_PointsToMeshProjector` directly.
    public class _InOptMut_PointsToMeshProjector
    {
        public PointsToMeshProjector? Opt;

        public _InOptMut_PointsToMeshProjector() {}
        public _InOptMut_PointsToMeshProjector(PointsToMeshProjector value) {Opt = value;}
        public static implicit operator _InOptMut_PointsToMeshProjector(PointsToMeshProjector value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointsToMeshProjector` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointsToMeshProjector`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointsToMeshProjector`/`Const_PointsToMeshProjector` to pass it to the function.
    public class _InOptConst_PointsToMeshProjector
    {
        public Const_PointsToMeshProjector? Opt;

        public _InOptConst_PointsToMeshProjector() {}
        public _InOptConst_PointsToMeshProjector(Const_PointsToMeshProjector value) {Opt = value;}
        public static implicit operator _InOptConst_PointsToMeshProjector(Const_PointsToMeshProjector value) {return new(value);}
    }

    /// Computes signed distances from given test points to the closest point on the reference mesh:
    /// positive value - outside reference mesh, negative - inside reference mesh;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    /// Generated from function `MR::findSignedDistances`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> FindSignedDistances(MR.Const_Mesh refMesh, MR.Const_VertCoords testPoints, MR.Const_VertBitSet? validTestPoints = null, MR.Const_MeshProjectionParameters? params_ = null, MR.IPointsToMeshProjector? projector = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSignedDistances_5", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_findSignedDistances_5(MR.Const_Mesh._Underlying *refMesh, MR.Const_VertCoords._Underlying *testPoints, MR.Const_VertBitSet._Underlying *validTestPoints, MR.Const_MeshProjectionParameters._Underlying *params_, MR.IPointsToMeshProjector._Underlying *projector);
        return MR.Misc.Move(new MR.VertScalars(__MR_findSignedDistances_5(refMesh._UnderlyingPtr, testPoints._UnderlyingPtr, validTestPoints is not null ? validTestPoints._UnderlyingPtr : null, params_ is not null ? params_._UnderlyingPtr : null, projector is not null ? projector._UnderlyingPtr : null), is_owning: true));
    }

    /// Computes signed distances from valid vertices of test mesh to the closest point on the reference mesh:
    /// positive value - outside reference mesh, negative - inside reference mesh;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    /// Generated from function `MR::findSignedDistances`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> FindSignedDistances(MR.Const_Mesh refMesh, MR.Const_Mesh mesh, MR.Const_MeshProjectionParameters? params_ = null, MR.IPointsToMeshProjector? projector = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSignedDistances_4", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_findSignedDistances_4(MR.Const_Mesh._Underlying *refMesh, MR.Const_Mesh._Underlying *mesh, MR.Const_MeshProjectionParameters._Underlying *params_, MR.IPointsToMeshProjector._Underlying *projector);
        return MR.Misc.Move(new MR.VertScalars(__MR_findSignedDistances_4(refMesh._UnderlyingPtr, mesh._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null, projector is not null ? projector._UnderlyingPtr : null), is_owning: true));
    }
}
