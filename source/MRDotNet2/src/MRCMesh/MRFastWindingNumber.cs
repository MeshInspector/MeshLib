public static partial class MR
{
    /// Abstract class for fast approximate computation of generalized winding number for a mesh (using its AABB tree)
    /// Generated from class `MR::IFastWindingNumber`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::FastWindingNumber`
    /// This is the const half of the class.
    public class Const_IFastWindingNumber : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_IFastWindingNumber_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_IFastWindingNumber_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_IFastWindingNumber_UseCount();
                return __MR_std_shared_ptr_MR_IFastWindingNumber_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IFastWindingNumber_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_IFastWindingNumber_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_IFastWindingNumber(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IFastWindingNumber_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IFastWindingNumber_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_IFastWindingNumber_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_IFastWindingNumber_ConstructNonOwning(ptr);
        }

        internal unsafe Const_IFastWindingNumber(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe IFastWindingNumber _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IFastWindingNumber_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_IFastWindingNumber_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_IFastWindingNumber_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_IFastWindingNumber_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_IFastWindingNumber_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_IFastWindingNumber_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_IFastWindingNumber_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IFastWindingNumber() {Dispose(false);}
    }

    /// Abstract class for fast approximate computation of generalized winding number for a mesh (using its AABB tree)
    /// Generated from class `MR::IFastWindingNumber`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::FastWindingNumber`
    /// This is the non-const half of the class.
    public class IFastWindingNumber : Const_IFastWindingNumber
    {
        internal unsafe IFastWindingNumber(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe IFastWindingNumber(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        /// <summary>
        /// calculates winding numbers in the points from given vector
        /// </summary>
        /// <param name="res">resulting winding numbers, will be resized automatically</param>
        /// <param name="points">incoming points</param>
        /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
        /// <param name="skipFace">this triangle (if it is close to `q`) will be skipped from summation</param>
        /// Generated from method `MR::IFastWindingNumber::calcFromVector`.
        /// Parameter `skipFace` defaults to `{}`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromVector(MR.Std.Vector_Float res, MR.Std.Const_Vector_MRVector3f points, float beta, MR._InOpt_FaceId skipFace = default, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumber_calcFromVector", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IFastWindingNumber_calcFromVector(_Underlying *_this, MR.Std.Vector_Float._Underlying *res, MR.Std.Const_Vector_MRVector3f._Underlying *points, float beta, MR.FaceId *skipFace, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IFastWindingNumber_calcFromVector(_UnderlyingPtr, res._UnderlyingPtr, points._UnderlyingPtr, beta, skipFace.HasValue ? &skipFace.Object : null, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// <summary>
        /// calculates winding numbers for all centers of mesh's triangles. if winding number is less than 0 or greater then 1, that face is marked as self-intersected
        /// </summary>
        /// <param name="res">resulting bit set</param>
        /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
        /// Generated from method `MR::IFastWindingNumber::calcSelfIntersections`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcSelfIntersections(MR.FaceBitSet res, float beta, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumber_calcSelfIntersections", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IFastWindingNumber_calcSelfIntersections(_Underlying *_this, MR.FaceBitSet._Underlying *res, float beta, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IFastWindingNumber_calcSelfIntersections(_UnderlyingPtr, res._UnderlyingPtr, beta, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// <summary>
        /// calculates winding numbers in each point from a three-dimensional grid
        /// </summary>
        /// <param name="res">resulting winding numbers, will be resized automatically</param>
        /// <param name="dims">dimensions of the grid</param>
        /// <param name="gridToMeshXf">transform from integer grid locations to voxel's centers in mesh reference frame</param>
        /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
        /// Generated from method `MR::IFastWindingNumber::calcFromGrid`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromGrid(MR.Std.Vector_Float res, MR.Const_Vector3i dims, MR.Const_AffineXf3f gridToMeshXf, float beta, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumber_calcFromGrid", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IFastWindingNumber_calcFromGrid(_Underlying *_this, MR.Std.Vector_Float._Underlying *res, MR.Const_Vector3i._Underlying *dims, MR.Const_AffineXf3f._Underlying *gridToMeshXf, float beta, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IFastWindingNumber_calcFromGrid(_UnderlyingPtr, res._UnderlyingPtr, dims._UnderlyingPtr, gridToMeshXf._UnderlyingPtr, beta, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
        }

        /// <summary>
        /// calculates distances with the sign obtained from generalized winding number in each point from a three-dimensional grid;
        /// if sqr(res) < minDistSq or sqr(res) >= maxDistSq, then NaN is returned for such point
        /// </summary>
        /// <param name="res">resulting signed distances, will be resized automatically</param>
        /// <param name="dims">dimensions of the grid</param>
        /// Generated from method `MR::IFastWindingNumber::calcFromGridWithDistances`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromGridWithDistances(MR.Std.Vector_Float res, MR.Const_Vector3i dims, MR.Const_AffineXf3f gridToMeshXf, MR.Const_DistanceToMeshOptions options, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumber_calcFromGridWithDistances", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IFastWindingNumber_calcFromGridWithDistances(_Underlying *_this, MR.Std.Vector_Float._Underlying *res, MR.Const_Vector3i._Underlying *dims, MR.Const_AffineXf3f._Underlying *gridToMeshXf, MR.Const_DistanceToMeshOptions._Underlying *options, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IFastWindingNumber_calcFromGridWithDistances(_UnderlyingPtr, res._UnderlyingPtr, dims._UnderlyingPtr, gridToMeshXf._UnderlyingPtr, options._UnderlyingPtr, cb._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `IFastWindingNumber` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `IFastWindingNumber`/`Const_IFastWindingNumber` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_IFastWindingNumber
    {
        internal readonly Const_IFastWindingNumber? Value;
        internal readonly MR.Misc._PassBy PassByMode;
    }

    /// This is used for optional parameters of class `IFastWindingNumber` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IFastWindingNumber`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IFastWindingNumber`/`Const_IFastWindingNumber` directly.
    public class _InOptMut_IFastWindingNumber
    {
        public IFastWindingNumber? Opt;

        public _InOptMut_IFastWindingNumber() {}
        public _InOptMut_IFastWindingNumber(IFastWindingNumber value) {Opt = value;}
        public static implicit operator _InOptMut_IFastWindingNumber(IFastWindingNumber value) {return new(value);}
    }

    /// This is used for optional parameters of class `IFastWindingNumber` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IFastWindingNumber`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IFastWindingNumber`/`Const_IFastWindingNumber` to pass it to the function.
    public class _InOptConst_IFastWindingNumber
    {
        public Const_IFastWindingNumber? Opt;

        public _InOptConst_IFastWindingNumber() {}
        public _InOptConst_IFastWindingNumber(Const_IFastWindingNumber value) {Opt = value;}
        public static implicit operator _InOptConst_IFastWindingNumber(Const_IFastWindingNumber value) {return new(value);}
    }

    /// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
    /// Note, this used to be `[[nodiscard]]`, but GCC 12 doesn't understand both `[[...]]` and `__attribute__(...)` on the same class.
    /// A possible fix is to change `MRMESH_CLASS` globally to `[[__gnu__::__visibility__("default")]]`.
    /// Generated from class `MR::FastWindingNumber`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IFastWindingNumber`
    /// This is the const half of the class.
    public class Const_FastWindingNumber : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_FastWindingNumber_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_FastWindingNumber_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_FastWindingNumber_UseCount();
                return __MR_std_shared_ptr_MR_FastWindingNumber_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FastWindingNumber_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_FastWindingNumber_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_FastWindingNumber(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FastWindingNumber_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FastWindingNumber_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_FastWindingNumber_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_FastWindingNumber_ConstructNonOwning(ptr);
        }

        internal unsafe Const_FastWindingNumber(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe FastWindingNumber _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FastWindingNumber_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_FastWindingNumber_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_FastWindingNumber_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_FastWindingNumber_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_FastWindingNumber_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_FastWindingNumber_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_FastWindingNumber_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FastWindingNumber() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_IFastWindingNumber(Const_FastWindingNumber self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_UpcastTo_MR_IFastWindingNumber", ExactSpelling = true)]
            extern static MR.Const_IFastWindingNumber._Underlying *__MR_FastWindingNumber_UpcastTo_MR_IFastWindingNumber(_Underlying *_this);
            return MR.Const_IFastWindingNumber._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FastWindingNumber_UpcastTo_MR_IFastWindingNumber(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_FastWindingNumber?(MR.Const_IFastWindingNumber parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumber_DynamicDowncastTo_MR_FastWindingNumber", ExactSpelling = true)]
            extern static _Underlying *__MR_IFastWindingNumber_DynamicDowncastTo_MR_FastWindingNumber(MR.Const_IFastWindingNumber._Underlying *_this);
            var ptr = __MR_IFastWindingNumber_DynamicDowncastTo_MR_FastWindingNumber(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_IFastWindingNumber._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public unsafe Const_FastWindingNumber(MR._ByValue_FastWindingNumber _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FastWindingNumber._Underlying *__MR_FastWindingNumber_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FastWindingNumber._Underlying *_other);
            _LateMakeShared(__MR_FastWindingNumber_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// constructs this from AABB tree of given mesh;
        /// this remains valid only if tree is valid
        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public unsafe Const_FastWindingNumber(MR.Const_Mesh mesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_Construct", ExactSpelling = true)]
            extern static MR.FastWindingNumber._Underlying *__MR_FastWindingNumber_Construct(MR.Const_Mesh._Underlying *mesh);
            _LateMakeShared(__MR_FastWindingNumber_Construct(mesh._UnderlyingPtr));
        }

        /// constructs this from AABB tree of given mesh;
        /// this remains valid only if tree is valid
        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public static unsafe implicit operator Const_FastWindingNumber(MR.Const_Mesh mesh) {return new(mesh);}
    }

    /// the class for fast approximate computation of winding number for a mesh (using its AABB tree)
    /// Note, this used to be `[[nodiscard]]`, but GCC 12 doesn't understand both `[[...]]` and `__attribute__(...)` on the same class.
    /// A possible fix is to change `MRMESH_CLASS` globally to `[[__gnu__::__visibility__("default")]]`.
    /// Generated from class `MR::FastWindingNumber`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::IFastWindingNumber`
    /// This is the non-const half of the class.
    public class FastWindingNumber : Const_FastWindingNumber
    {
        internal unsafe FastWindingNumber(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe FastWindingNumber(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.IFastWindingNumber(FastWindingNumber self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_UpcastTo_MR_IFastWindingNumber", ExactSpelling = true)]
            extern static MR.IFastWindingNumber._Underlying *__MR_FastWindingNumber_UpcastTo_MR_IFastWindingNumber(_Underlying *_this);
            return MR.IFastWindingNumber._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_FastWindingNumber_UpcastTo_MR_IFastWindingNumber(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator FastWindingNumber?(MR.IFastWindingNumber parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumber_DynamicDowncastTo_MR_FastWindingNumber", ExactSpelling = true)]
            extern static _Underlying *__MR_IFastWindingNumber_DynamicDowncastTo_MR_FastWindingNumber(MR.IFastWindingNumber._Underlying *_this);
            var ptr = __MR_IFastWindingNumber_DynamicDowncastTo_MR_FastWindingNumber(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.IFastWindingNumber._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public unsafe FastWindingNumber(MR._ByValue_FastWindingNumber _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FastWindingNumber._Underlying *__MR_FastWindingNumber_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FastWindingNumber._Underlying *_other);
            _LateMakeShared(__MR_FastWindingNumber_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// constructs this from AABB tree of given mesh;
        /// this remains valid only if tree is valid
        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public unsafe FastWindingNumber(MR.Const_Mesh mesh) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_Construct", ExactSpelling = true)]
            extern static MR.FastWindingNumber._Underlying *__MR_FastWindingNumber_Construct(MR.Const_Mesh._Underlying *mesh);
            _LateMakeShared(__MR_FastWindingNumber_Construct(mesh._UnderlyingPtr));
        }

        /// constructs this from AABB tree of given mesh;
        /// this remains valid only if tree is valid
        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public static unsafe implicit operator FastWindingNumber(MR.Const_Mesh mesh) {return new(mesh);}

        // see methods' descriptions in IFastWindingNumber
        /// Generated from method `MR::FastWindingNumber::calcFromVector`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromVector(MR.Std.Vector_Float res, MR.Std.Const_Vector_MRVector3f points, float beta, MR.FaceId skipFace, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_calcFromVector", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_FastWindingNumber_calcFromVector(_Underlying *_this, MR.Std.Vector_Float._Underlying *res, MR.Std.Const_Vector_MRVector3f._Underlying *points, float beta, MR.FaceId skipFace, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_FastWindingNumber_calcFromVector(_UnderlyingPtr, res._UnderlyingPtr, points._UnderlyingPtr, beta, skipFace, cb._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FastWindingNumber::calcSelfIntersections`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcSelfIntersections(MR.FaceBitSet res, float beta, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_calcSelfIntersections", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_FastWindingNumber_calcSelfIntersections(_Underlying *_this, MR.FaceBitSet._Underlying *res, float beta, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_FastWindingNumber_calcSelfIntersections(_UnderlyingPtr, res._UnderlyingPtr, beta, cb._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FastWindingNumber::calcFromGrid`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromGrid(MR.Std.Vector_Float res, MR.Const_Vector3i dims, MR.Const_AffineXf3f gridToMeshXf, float beta, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_calcFromGrid", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_FastWindingNumber_calcFromGrid(_Underlying *_this, MR.Std.Vector_Float._Underlying *res, MR.Const_Vector3i._Underlying *dims, MR.Const_AffineXf3f._Underlying *gridToMeshXf, float beta, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_FastWindingNumber_calcFromGrid(_UnderlyingPtr, res._UnderlyingPtr, dims._UnderlyingPtr, gridToMeshXf._UnderlyingPtr, beta, cb._UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::FastWindingNumber::calcWithDistances`.
        public unsafe float CalcWithDistances(MR.Const_Vector3f p, MR.Const_DistanceToMeshOptions options)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_calcWithDistances", ExactSpelling = true)]
            extern static float __MR_FastWindingNumber_calcWithDistances(_Underlying *_this, MR.Const_Vector3f._Underlying *p, MR.Const_DistanceToMeshOptions._Underlying *options);
            return __MR_FastWindingNumber_calcWithDistances(_UnderlyingPtr, p._UnderlyingPtr, options._UnderlyingPtr);
        }

        /// Generated from method `MR::FastWindingNumber::calcFromGridWithDistances`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromGridWithDistances(MR.Std.Vector_Float res, MR.Const_Vector3i dims, MR.Const_AffineXf3f gridToMeshXf, MR.Const_DistanceToMeshOptions options, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FastWindingNumber_calcFromGridWithDistances", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_FastWindingNumber_calcFromGridWithDistances(_Underlying *_this, MR.Std.Vector_Float._Underlying *res, MR.Const_Vector3i._Underlying *dims, MR.Const_AffineXf3f._Underlying *gridToMeshXf, MR.Const_DistanceToMeshOptions._Underlying *options, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_FastWindingNumber_calcFromGridWithDistances(_UnderlyingPtr, res._UnderlyingPtr, dims._UnderlyingPtr, gridToMeshXf._UnderlyingPtr, options._UnderlyingPtr, cb._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `FastWindingNumber` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FastWindingNumber`/`Const_FastWindingNumber` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FastWindingNumber
    {
        internal readonly Const_FastWindingNumber? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FastWindingNumber(Const_FastWindingNumber new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FastWindingNumber(Const_FastWindingNumber arg) {return new(arg);}
        public _ByValue_FastWindingNumber(MR.Misc._Moved<FastWindingNumber> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FastWindingNumber(MR.Misc._Moved<FastWindingNumber> arg) {return new(arg);}

        /// constructs this from AABB tree of given mesh;
        /// this remains valid only if tree is valid
        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public static unsafe implicit operator _ByValue_FastWindingNumber(MR.Const_Mesh mesh) {return new MR.FastWindingNumber(mesh);}
    }

    /// This is used for optional parameters of class `FastWindingNumber` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FastWindingNumber`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FastWindingNumber`/`Const_FastWindingNumber` directly.
    public class _InOptMut_FastWindingNumber
    {
        public FastWindingNumber? Opt;

        public _InOptMut_FastWindingNumber() {}
        public _InOptMut_FastWindingNumber(FastWindingNumber value) {Opt = value;}
        public static implicit operator _InOptMut_FastWindingNumber(FastWindingNumber value) {return new(value);}
    }

    /// This is used for optional parameters of class `FastWindingNumber` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FastWindingNumber`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FastWindingNumber`/`Const_FastWindingNumber` to pass it to the function.
    public class _InOptConst_FastWindingNumber
    {
        public Const_FastWindingNumber? Opt;

        public _InOptConst_FastWindingNumber() {}
        public _InOptConst_FastWindingNumber(Const_FastWindingNumber value) {Opt = value;}
        public static implicit operator _InOptConst_FastWindingNumber(Const_FastWindingNumber value) {return new(value);}

        /// constructs this from AABB tree of given mesh;
        /// this remains valid only if tree is valid
        /// Generated from constructor `MR::FastWindingNumber::FastWindingNumber`.
        public static unsafe implicit operator _InOptConst_FastWindingNumber(MR.Const_Mesh mesh) {return new MR.FastWindingNumber(mesh);}
    }

    /// Abstract class that complements \ref IFastWindingNumber with chunked processing variants of its methods
    /// Generated from class `MR::IFastWindingNumberByParts`.
    /// This is the const half of the class.
    public class Const_IFastWindingNumberByParts : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_IFastWindingNumberByParts(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumberByParts_Destroy", ExactSpelling = true)]
            extern static void __MR_IFastWindingNumberByParts_Destroy(_Underlying *_this);
            __MR_IFastWindingNumberByParts_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_IFastWindingNumberByParts() {Dispose(false);}
    }

    /// Abstract class that complements \ref IFastWindingNumber with chunked processing variants of its methods
    /// Generated from class `MR::IFastWindingNumberByParts`.
    /// This is the non-const half of the class.
    public class IFastWindingNumberByParts : Const_IFastWindingNumberByParts
    {
        internal unsafe IFastWindingNumberByParts(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// <summary>
        /// calculates winding numbers in each point from a three-dimensional grid
        /// </summary>
        /// <param name="resFunc">callback that gets a block of resulting winding numbers</param>
        /// <param name="dims">dimensions of the grid</param>
        /// <param name="gridToMeshXf">transform from integer grid locations to voxel's centers in mesh reference frame</param>
        /// <param name="beta">determines the precision of the approximation: the more the better, recommended value 2 or more</param>
        /// <param name="layerOverlap">overlap between two blocks of the grid, set as XY layer count</param>
        /// Generated from method `MR::IFastWindingNumberByParts::calcFromGridByParts`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromGridByParts(MR.Std._ByValue_Function_ExpectedVoidStdStringFuncFromStdVectorFloatRvalueRefConstMRVector3iRefInt resFunc, MR.Const_Vector3i dims, MR.Const_AffineXf3f gridToMeshXf, float beta, int layerOverlap, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumberByParts_calcFromGridByParts", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IFastWindingNumberByParts_calcFromGridByParts(_Underlying *_this, MR.Misc._PassBy resFunc_pass_by, MR.Std.Function_ExpectedVoidStdStringFuncFromStdVectorFloatRvalueRefConstMRVector3iRefInt._Underlying *resFunc, MR.Const_Vector3i._Underlying *dims, MR.Const_AffineXf3f._Underlying *gridToMeshXf, float beta, int layerOverlap, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IFastWindingNumberByParts_calcFromGridByParts(_UnderlyingPtr, resFunc.PassByMode, resFunc.Value is not null ? resFunc.Value._UnderlyingPtr : null, dims._UnderlyingPtr, gridToMeshXf._UnderlyingPtr, beta, layerOverlap, cb._UnderlyingPtr), is_owning: true));
        }

        /// <summary>
        /// calculates distances with the sign obtained from generalized winding number in each point from a three-dimensional grid;
        /// if sqr(res) < minDistSq or sqr(res) >= maxDistSq, then NaN is returned for such point
        /// </summary>
        /// <param name="resFunc">callback that gets a block of resulting winding numbers</param>
        /// <param name="dims">dimensions of the grid</param>
        /// <param name="gridToMeshXf">transform from integer grid locations to voxel's centers in mesh reference frame</param>
        /// <param name="layerOverlap">overlap between two blocks of the grid, set as XY layer count</param>
        /// Generated from method `MR::IFastWindingNumberByParts::calcFromGridWithDistancesByParts`.
        public unsafe MR.Misc._Moved<MR.Expected_Void_StdString> CalcFromGridWithDistancesByParts(MR.Std._ByValue_Function_ExpectedVoidStdStringFuncFromStdVectorFloatRvalueRefConstMRVector3iRefInt resFunc, MR.Const_Vector3i dims, MR.Const_AffineXf3f gridToMeshXf, MR.Const_DistanceToMeshOptions options, int layerOverlap, MR.Std.Const_Function_BoolFuncFromFloat cb)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_IFastWindingNumberByParts_calcFromGridWithDistancesByParts", ExactSpelling = true)]
            extern static MR.Expected_Void_StdString._Underlying *__MR_IFastWindingNumberByParts_calcFromGridWithDistancesByParts(_Underlying *_this, MR.Misc._PassBy resFunc_pass_by, MR.Std.Function_ExpectedVoidStdStringFuncFromStdVectorFloatRvalueRefConstMRVector3iRefInt._Underlying *resFunc, MR.Const_Vector3i._Underlying *dims, MR.Const_AffineXf3f._Underlying *gridToMeshXf, MR.Const_DistanceToMeshOptions._Underlying *options, int layerOverlap, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_Void_StdString(__MR_IFastWindingNumberByParts_calcFromGridWithDistancesByParts(_UnderlyingPtr, resFunc.PassByMode, resFunc.Value is not null ? resFunc.Value._UnderlyingPtr : null, dims._UnderlyingPtr, gridToMeshXf._UnderlyingPtr, options._UnderlyingPtr, layerOverlap, cb._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used for optional parameters of class `IFastWindingNumberByParts` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_IFastWindingNumberByParts`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IFastWindingNumberByParts`/`Const_IFastWindingNumberByParts` directly.
    public class _InOptMut_IFastWindingNumberByParts
    {
        public IFastWindingNumberByParts? Opt;

        public _InOptMut_IFastWindingNumberByParts() {}
        public _InOptMut_IFastWindingNumberByParts(IFastWindingNumberByParts value) {Opt = value;}
        public static implicit operator _InOptMut_IFastWindingNumberByParts(IFastWindingNumberByParts value) {return new(value);}
    }

    /// This is used for optional parameters of class `IFastWindingNumberByParts` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_IFastWindingNumberByParts`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `IFastWindingNumberByParts`/`Const_IFastWindingNumberByParts` to pass it to the function.
    public class _InOptConst_IFastWindingNumberByParts
    {
        public Const_IFastWindingNumberByParts? Opt;

        public _InOptConst_IFastWindingNumberByParts() {}
        public _InOptConst_IFastWindingNumberByParts(Const_IFastWindingNumberByParts value) {Opt = value;}
        public static implicit operator _InOptConst_IFastWindingNumberByParts(Const_IFastWindingNumberByParts value) {return new(value);}
    }
}
