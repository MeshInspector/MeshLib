public static partial class MR
{
    /// wrapper class that helps mrbind to avoid excess MRVDBFloatGrid.h includes
    /// Generated from class `MR::FloatGrid`.
    /// This is the const half of the class.
    public class Const_FloatGrid : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FloatGrid(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_Destroy", ExactSpelling = true)]
            extern static void __MR_FloatGrid_Destroy(_Underlying *_this);
            __MR_FloatGrid_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FloatGrid() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FloatGrid() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_DefaultConstruct();
            _UnderlyingPtr = __MR_FloatGrid_DefaultConstruct();
        }

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public unsafe Const_FloatGrid(MR._ByValue_FloatGrid _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FloatGrid._Underlying *_other);
            _UnderlyingPtr = __MR_FloatGrid_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public unsafe Const_FloatGrid(MR._ByValue_OpenVdbFloatGrid ptr) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_Construct", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_Construct(MR.Misc._PassBy ptr_pass_by, MR.OpenVdbFloatGrid._UnderlyingShared *ptr);
            _UnderlyingPtr = __MR_FloatGrid_Construct(ptr.PassByMode, ptr.Value is not null ? ptr.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public static unsafe implicit operator Const_FloatGrid(MR._ByValue_OpenVdbFloatGrid ptr) {return new(ptr);}

        /// Generated from conversion operator `MR::FloatGrid::operator bool`.
        public static unsafe explicit operator bool(MR.Const_FloatGrid _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_FloatGrid_ConvertTo_bool(MR.Const_FloatGrid._Underlying *_this);
            return __MR_FloatGrid_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::FloatGrid::get`.
        public unsafe MR.OpenVdbFloatGrid? Get()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_get", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_FloatGrid_get(_Underlying *_this);
            var __ret = __MR_FloatGrid_get(_UnderlyingPtr);
            return __ret is not null ? new MR.OpenVdbFloatGrid(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::FloatGrid::operator*`.
        public unsafe MR.OpenVdbFloatGrid Deref()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_deref_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_deref_MR_FloatGrid(_Underlying *_this);
            return new(__MR_deref_MR_FloatGrid(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::FloatGrid::operator->`.
        public unsafe MR.OpenVdbFloatGrid? Arrow()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_arrow", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._Underlying *__MR_FloatGrid_arrow(_Underlying *_this);
            var __ret = __MR_FloatGrid_arrow(_UnderlyingPtr);
            return __ret is not null ? new MR.OpenVdbFloatGrid(__ret, is_owning: false) : null;
        }

        /// Generated from method `MR::FloatGrid::toVdb`.
        public unsafe MR.Misc._Moved<MR.OpenVdbFloatGrid> ToVdb()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_toVdb", ExactSpelling = true)]
            extern static MR.OpenVdbFloatGrid._UnderlyingShared *__MR_FloatGrid_toVdb(_Underlying *_this);
            return MR.Misc.Move(new MR.OpenVdbFloatGrid(__MR_FloatGrid_toVdb(_UnderlyingPtr), is_owning: true));
        }
    }

    /// wrapper class that helps mrbind to avoid excess MRVDBFloatGrid.h includes
    /// Generated from class `MR::FloatGrid`.
    /// This is the non-const half of the class.
    public class FloatGrid : Const_FloatGrid
    {
        internal unsafe FloatGrid(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe FloatGrid() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_DefaultConstruct();
            _UnderlyingPtr = __MR_FloatGrid_DefaultConstruct();
        }

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public unsafe FloatGrid(MR._ByValue_FloatGrid _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FloatGrid._Underlying *_other);
            _UnderlyingPtr = __MR_FloatGrid_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public unsafe FloatGrid(MR._ByValue_OpenVdbFloatGrid ptr) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_Construct", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_Construct(MR.Misc._PassBy ptr_pass_by, MR.OpenVdbFloatGrid._UnderlyingShared *ptr);
            _UnderlyingPtr = __MR_FloatGrid_Construct(ptr.PassByMode, ptr.Value is not null ? ptr.Value._UnderlyingSharedPtr : null);
        }

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public static unsafe implicit operator FloatGrid(MR._ByValue_OpenVdbFloatGrid ptr) {return new(ptr);}

        /// Generated from method `MR::FloatGrid::operator=`.
        public unsafe MR.FloatGrid Assign(MR._ByValue_FloatGrid _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_FloatGrid_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FloatGrid._Underlying *_other);
            return new(__MR_FloatGrid_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::FloatGrid::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_reset", ExactSpelling = true)]
            extern static void __MR_FloatGrid_reset(_Underlying *_this);
            __MR_FloatGrid_reset(_UnderlyingPtr);
        }

        /// Generated from method `MR::FloatGrid::swap`.
        public unsafe void Swap(MR.FloatGrid other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FloatGrid_swap", ExactSpelling = true)]
            extern static void __MR_FloatGrid_swap(_Underlying *_this, MR.FloatGrid._Underlying *other);
            __MR_FloatGrid_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        // union operation on volumetric representation of two meshes
        /// Generated from function `MR::operator+=`.
        public unsafe MR.Misc._Moved<MR.FloatGrid> AddAssign(MR.Const_FloatGrid b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_add_assign_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_add_assign_MR_FloatGrid(_Underlying *a, MR.Const_FloatGrid._Underlying *b);
            return MR.Misc.Move(new MR.FloatGrid(__MR_add_assign_MR_FloatGrid(_UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        // difference operation on volumetric representation of two meshes
        /// Generated from function `MR::operator-=`.
        public unsafe MR.Misc._Moved<MR.FloatGrid> SubAssign(MR.Const_FloatGrid b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_assign_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_sub_assign_MR_FloatGrid(_Underlying *a, MR.Const_FloatGrid._Underlying *b);
            return MR.Misc.Move(new MR.FloatGrid(__MR_sub_assign_MR_FloatGrid(_UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }

        // intersection operation on volumetric representation of two meshes
        /// Generated from function `MR::operator*=`.
        public unsafe MR.Misc._Moved<MR.FloatGrid> MulAssign(MR.Const_FloatGrid b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_FloatGrid", ExactSpelling = true)]
            extern static MR.FloatGrid._Underlying *__MR_mul_assign_MR_FloatGrid(_Underlying *a, MR.Const_FloatGrid._Underlying *b);
            return MR.Misc.Move(new MR.FloatGrid(__MR_mul_assign_MR_FloatGrid(_UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `FloatGrid` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FloatGrid`/`Const_FloatGrid` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FloatGrid
    {
        internal readonly Const_FloatGrid? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FloatGrid() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FloatGrid(Const_FloatGrid new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FloatGrid(Const_FloatGrid arg) {return new(arg);}
        public _ByValue_FloatGrid(MR.Misc._Moved<FloatGrid> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FloatGrid(MR.Misc._Moved<FloatGrid> arg) {return new(arg);}

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public static unsafe implicit operator _ByValue_FloatGrid(MR._ByValue_OpenVdbFloatGrid ptr) {return new MR.FloatGrid(ptr);}
    }

    /// This is used for optional parameters of class `FloatGrid` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FloatGrid`/`Const_FloatGrid` directly.
    public class _InOptMut_FloatGrid
    {
        public FloatGrid? Opt;

        public _InOptMut_FloatGrid() {}
        public _InOptMut_FloatGrid(FloatGrid value) {Opt = value;}
        public static implicit operator _InOptMut_FloatGrid(FloatGrid value) {return new(value);}
    }

    /// This is used for optional parameters of class `FloatGrid` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FloatGrid`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FloatGrid`/`Const_FloatGrid` to pass it to the function.
    public class _InOptConst_FloatGrid
    {
        public Const_FloatGrid? Opt;

        public _InOptConst_FloatGrid() {}
        public _InOptConst_FloatGrid(Const_FloatGrid value) {Opt = value;}
        public static implicit operator _InOptConst_FloatGrid(Const_FloatGrid value) {return new(value);}

        /// Generated from constructor `MR::FloatGrid::FloatGrid`.
        public static unsafe implicit operator _InOptConst_FloatGrid(MR._ByValue_OpenVdbFloatGrid ptr) {return new MR.FloatGrid(ptr);}
    }

    /// returns the amount of heap memory occupied by grid
    /// Generated from function `MR::heapBytes`.
    public static unsafe ulong HeapBytes(MR.Const_FloatGrid grid)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_heapBytes_MR_FloatGrid", ExactSpelling = true)]
        extern static ulong __MR_heapBytes_MR_FloatGrid(MR.Const_FloatGrid._Underlying *grid);
        return __MR_heapBytes_MR_FloatGrid(grid._UnderlyingPtr);
    }

    /// resample this grid to fit voxelScale
    /// Generated from function `MR::resampled`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> Resampled(MR.Const_FloatGrid grid, float voxelScale, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_resampled_float", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_resampled_float(MR.Const_FloatGrid._Underlying *grid, float voxelScale, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.FloatGrid(__MR_resampled_float(grid._UnderlyingPtr, voxelScale, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// resample this grid to fit voxelScale
    /// Generated from function `MR::resampled`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> Resampled(MR.Const_FloatGrid grid, MR.Const_Vector3f voxelScale, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_resampled_MR_Vector3f", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_resampled_MR_Vector3f(MR.Const_FloatGrid._Underlying *grid, MR.Const_Vector3f._Underlying *voxelScale, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.FloatGrid(__MR_resampled_MR_Vector3f(grid._UnderlyingPtr, voxelScale._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// returns cropped grid
    /// Generated from function `MR::cropped`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> Cropped(MR.Const_FloatGrid grid, MR.Const_Box3i box, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_cropped", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_cropped(MR.Const_FloatGrid._Underlying *grid, MR.Const_Box3i._Underlying *box, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.FloatGrid(__MR_cropped(grid._UnderlyingPtr, box._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// returns grid with gaussian filter applied
    /// Generated from function `MR::gaussianFilter`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe void GaussianFilter(MR.FloatGrid grid, int width, int iters, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_gaussianFilter", ExactSpelling = true)]
        extern static void __MR_gaussianFilter(MR.FloatGrid._Underlying *grid, int width, int iters, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        __MR_gaussianFilter(grid._UnderlyingPtr, width, iters, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null);
    }

    /// Generated from function `MR::gaussianFiltered`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.FloatGrid> GaussianFiltered(MR.Const_FloatGrid grid, int width, int iters, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_gaussianFiltered", ExactSpelling = true)]
        extern static MR.FloatGrid._Underlying *__MR_gaussianFiltered(MR.Const_FloatGrid._Underlying *grid, int width, int iters, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.FloatGrid(__MR_gaussianFiltered(grid._UnderlyingPtr, width, iters, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// returns the value at given voxel
    /// Generated from function `MR::getValue`.
    public static unsafe float GetValue(MR.Const_FloatGrid grid, MR.Const_Vector3i p)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getValue", ExactSpelling = true)]
        extern static float __MR_getValue(MR.Const_FloatGrid._Underlying *grid, MR.Const_Vector3i._Underlying *p);
        return __MR_getValue(grid._UnderlyingPtr, p._UnderlyingPtr);
    }

    /// sets given region voxels value
    /// \note region is in grid space (0 voxel id is minimum active voxel in grid)
    /// Generated from function `MR::setValue`.
    public static unsafe void SetValue(MR.FloatGrid grid, MR.Const_Vector3i p, float value)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setValue_MR_Vector3i", ExactSpelling = true)]
        extern static void __MR_setValue_MR_Vector3i(MR.FloatGrid._Underlying *grid, MR.Const_Vector3i._Underlying *p, float value);
        __MR_setValue_MR_Vector3i(grid._UnderlyingPtr, p._UnderlyingPtr, value);
    }

    /// sets given region voxels value
    /// \note region is in grid space (0 voxel id is minimum active voxel in grid)
    /// Generated from function `MR::setValue`.
    public static unsafe void SetValue(MR.FloatGrid grid, MR.Const_VoxelBitSet region, float value)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setValue_MR_VoxelBitSet", ExactSpelling = true)]
        extern static void __MR_setValue_MR_VoxelBitSet(MR.FloatGrid._Underlying *grid, MR.Const_VoxelBitSet._Underlying *region, float value);
        __MR_setValue_MR_VoxelBitSet(grid._UnderlyingPtr, region._UnderlyingPtr, value);
    }

    /// sets type of this grid as LEVEL SET (for normal flipping)
    /// Generated from function `MR::setLevelSetType`.
    public static unsafe void SetLevelSetType(MR.FloatGrid grid)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setLevelSetType", ExactSpelling = true)]
        extern static void __MR_setLevelSetType(MR.FloatGrid._Underlying *grid);
        __MR_setLevelSetType(grid._UnderlyingPtr);
    }
}
