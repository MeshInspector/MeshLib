public static partial class MR
{
    /// VoxelsVolumeAccessor specialization for simple volumes with min/max
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelsVolumeAccessor<MR::SimpleVolume>`
    /// This is the const half of the class.
    public class Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_VoxelsVolumeAccessor_MRSimpleVolume(Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_UpcastTo_MR_VoxelsVolumeAccessor_MR_SimpleVolume", ExactSpelling = true)]
            extern static MR.Const_VoxelsVolumeAccessor_MRSimpleVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_UpcastTo_MR_VoxelsVolumeAccessor_MR_SimpleVolume(_Underlying *_this);
            MR.Const_VoxelsVolumeAccessor_MRSimpleVolume ret = new(__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_UpcastTo_MR_VoxelsVolumeAccessor_MR_SimpleVolume(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        ///< caching results of this accessor does not make any sense since it returns values from a simple container
        public static unsafe bool CacheEffective
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Get_cacheEffective", ExactSpelling = true)]
                extern static bool *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Get_cacheEffective();
                return *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Get_cacheEffective();
            }
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_ConstructFromAnother(MR.VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(MR.Const_SimpleVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct(MR.Const_SimpleVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct(volume._UnderlyingPtr);
        }

        /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
        /// Generated from method `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>::shift`.
        public unsafe MR.Vector3f Shift()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_shift", ExactSpelling = true)]
            extern static MR.Vector3f __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_shift(_Underlying *_this);
            return __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_shift(_UnderlyingPtr);
        }
    }

    /// VoxelsVolumeAccessor specialization for simple volumes with min/max
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelsVolumeAccessor<MR::SimpleVolume>`
    /// This is the non-const half of the class.
    public class VoxelsVolumeAccessor_MRSimpleVolumeMinMax : Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax
    {
        internal unsafe VoxelsVolumeAccessor_MRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.VoxelsVolumeAccessor_MRSimpleVolume(VoxelsVolumeAccessor_MRSimpleVolumeMinMax self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_UpcastTo_MR_VoxelsVolumeAccessor_MR_SimpleVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_UpcastTo_MR_VoxelsVolumeAccessor_MR_SimpleVolume(_Underlying *_this);
            MR.VoxelsVolumeAccessor_MRSimpleVolume ret = new(__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_UpcastTo_MR_VoxelsVolumeAccessor_MR_SimpleVolume(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRSimpleVolumeMinMax(MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_ConstructFromAnother(MR.VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRSimpleVolumeMinMax(MR.Const_SimpleVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct(MR.Const_SimpleVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct(volume._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRSimpleVolumeMinMax` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeAccessor_MRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRSimpleVolumeMinMax`/`Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax` directly.
    public class _InOptMut_VoxelsVolumeAccessor_MRSimpleVolumeMinMax
    {
        public VoxelsVolumeAccessor_MRSimpleVolumeMinMax? Opt;

        public _InOptMut_VoxelsVolumeAccessor_MRSimpleVolumeMinMax() {}
        public _InOptMut_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(VoxelsVolumeAccessor_MRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(VoxelsVolumeAccessor_MRSimpleVolumeMinMax value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRSimpleVolumeMinMax` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeAccessor_MRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRSimpleVolumeMinMax`/`Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax` to pass it to the function.
    public class _InOptConst_VoxelsVolumeAccessor_MRSimpleVolumeMinMax
    {
        public Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax? Opt;

        public _InOptConst_VoxelsVolumeAccessor_MRSimpleVolumeMinMax() {}
        public _InOptConst_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeAccessor_MRSimpleVolumeMinMax(Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax value) {return new(value);}
    }

    /// VoxelsVolumeAccessor specialization for simple volumes
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::SimpleVolume>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>`
    /// This is the const half of the class.
    public class Const_VoxelsVolumeAccessor_MRSimpleVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeAccessor_MRSimpleVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeAccessor_MR_SimpleVolume_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeAccessor_MR_SimpleVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeAccessor_MRSimpleVolume() {Dispose(false);}

        ///< caching results of this accessor does not make any sense since it returns values from a simple container
        public static unsafe bool CacheEffective
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_Get_cacheEffective", ExactSpelling = true)]
                extern static bool *__MR_VoxelsVolumeAccessor_MR_SimpleVolume_Get_cacheEffective();
                return *__MR_VoxelsVolumeAccessor_MR_SimpleVolume_Get_cacheEffective();
            }
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRSimpleVolume(MR.Const_VoxelsVolumeAccessor_MRSimpleVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolume_ConstructFromAnother(MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolume_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRSimpleVolume(MR.Const_SimpleVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolume_Construct(MR.Const_SimpleVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolume_Construct(volume._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::get`.
        public unsafe float Get(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_get_MR_Vector3i", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeAccessor_MR_SimpleVolume_get_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return __MR_VoxelsVolumeAccessor_MR_SimpleVolume_get_MR_Vector3i(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::get`.
        public unsafe float Get(MR.Const_VoxelLocation loc)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_get_MR_VoxelLocation", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeAccessor_MR_SimpleVolume_get_MR_VoxelLocation(_Underlying *_this, MR.Const_VoxelLocation._Underlying *loc);
            return __MR_VoxelsVolumeAccessor_MR_SimpleVolume_get_MR_VoxelLocation(_UnderlyingPtr, loc._UnderlyingPtr);
        }

        /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
        /// Generated from method `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::shift`.
        public unsafe MR.Vector3f Shift()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_shift", ExactSpelling = true)]
            extern static MR.Vector3f __MR_VoxelsVolumeAccessor_MR_SimpleVolume_shift(_Underlying *_this);
            return __MR_VoxelsVolumeAccessor_MR_SimpleVolume_shift(_UnderlyingPtr);
        }
    }

    /// VoxelsVolumeAccessor specialization for simple volumes
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::SimpleVolume>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>`
    /// This is the non-const half of the class.
    public class VoxelsVolumeAccessor_MRSimpleVolume : Const_VoxelsVolumeAccessor_MRSimpleVolume
    {
        internal unsafe VoxelsVolumeAccessor_MRSimpleVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRSimpleVolume(MR.Const_VoxelsVolumeAccessor_MRSimpleVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolume_ConstructFromAnother(MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolume_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::SimpleVolume>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRSimpleVolume(MR.Const_SimpleVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_SimpleVolume_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRSimpleVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_SimpleVolume_Construct(MR.Const_SimpleVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_SimpleVolume_Construct(volume._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRSimpleVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeAccessor_MRSimpleVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRSimpleVolume`/`Const_VoxelsVolumeAccessor_MRSimpleVolume` directly.
    public class _InOptMut_VoxelsVolumeAccessor_MRSimpleVolume
    {
        public VoxelsVolumeAccessor_MRSimpleVolume? Opt;

        public _InOptMut_VoxelsVolumeAccessor_MRSimpleVolume() {}
        public _InOptMut_VoxelsVolumeAccessor_MRSimpleVolume(VoxelsVolumeAccessor_MRSimpleVolume value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeAccessor_MRSimpleVolume(VoxelsVolumeAccessor_MRSimpleVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRSimpleVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeAccessor_MRSimpleVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRSimpleVolume`/`Const_VoxelsVolumeAccessor_MRSimpleVolume` to pass it to the function.
    public class _InOptConst_VoxelsVolumeAccessor_MRSimpleVolume
    {
        public Const_VoxelsVolumeAccessor_MRSimpleVolume? Opt;

        public _InOptConst_VoxelsVolumeAccessor_MRSimpleVolume() {}
        public _InOptConst_VoxelsVolumeAccessor_MRSimpleVolume(Const_VoxelsVolumeAccessor_MRSimpleVolume value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeAccessor_MRSimpleVolume(Const_VoxelsVolumeAccessor_MRSimpleVolume value) {return new(value);}
    }

    /// VoxelsVolumeAccessor specialization for value getters
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::FunctionVolume>`.
    /// This is the const half of the class.
    public class Const_VoxelsVolumeAccessor_MRFunctionVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeAccessor_MRFunctionVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeAccessor_MR_FunctionVolume_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeAccessor_MR_FunctionVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeAccessor_MRFunctionVolume() {Dispose(false);}

        ///< caching results of this accessor can improve performance
        public static unsafe bool CacheEffective
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_Get_cacheEffective", ExactSpelling = true)]
                extern static bool *__MR_VoxelsVolumeAccessor_MR_FunctionVolume_Get_cacheEffective();
                return *__MR_VoxelsVolumeAccessor_MR_FunctionVolume_Get_cacheEffective();
            }
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRFunctionVolume(MR.Const_VoxelsVolumeAccessor_MRFunctionVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRFunctionVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_FunctionVolume_ConstructFromAnother(MR.VoxelsVolumeAccessor_MRFunctionVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_FunctionVolume_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRFunctionVolume(MR.Const_FunctionVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRFunctionVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct(MR.Const_FunctionVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct(volume._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::get`.
        public unsafe float Get(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_get_MR_Vector3i", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeAccessor_MR_FunctionVolume_get_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return __MR_VoxelsVolumeAccessor_MR_FunctionVolume_get_MR_Vector3i(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::get`.
        public unsafe float Get(MR.Const_VoxelLocation loc)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_get_MR_VoxelLocation", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeAccessor_MR_FunctionVolume_get_MR_VoxelLocation(_Underlying *_this, MR.Const_VoxelLocation._Underlying *loc);
            return __MR_VoxelsVolumeAccessor_MR_FunctionVolume_get_MR_VoxelLocation(_UnderlyingPtr, loc._UnderlyingPtr);
        }

        /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
        /// Generated from method `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::shift`.
        public unsafe MR.Vector3f Shift()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_shift", ExactSpelling = true)]
            extern static MR.Vector3f __MR_VoxelsVolumeAccessor_MR_FunctionVolume_shift(_Underlying *_this);
            return __MR_VoxelsVolumeAccessor_MR_FunctionVolume_shift(_UnderlyingPtr);
        }
    }

    /// VoxelsVolumeAccessor specialization for value getters
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::FunctionVolume>`.
    /// This is the non-const half of the class.
    public class VoxelsVolumeAccessor_MRFunctionVolume : Const_VoxelsVolumeAccessor_MRFunctionVolume
    {
        internal unsafe VoxelsVolumeAccessor_MRFunctionVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRFunctionVolume(MR.Const_VoxelsVolumeAccessor_MRFunctionVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRFunctionVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_FunctionVolume_ConstructFromAnother(MR.VoxelsVolumeAccessor_MRFunctionVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_FunctionVolume_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::FunctionVolume>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRFunctionVolume(MR.Const_FunctionVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRFunctionVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct(MR.Const_FunctionVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct(volume._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRFunctionVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeAccessor_MRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRFunctionVolume`/`Const_VoxelsVolumeAccessor_MRFunctionVolume` directly.
    public class _InOptMut_VoxelsVolumeAccessor_MRFunctionVolume
    {
        public VoxelsVolumeAccessor_MRFunctionVolume? Opt;

        public _InOptMut_VoxelsVolumeAccessor_MRFunctionVolume() {}
        public _InOptMut_VoxelsVolumeAccessor_MRFunctionVolume(VoxelsVolumeAccessor_MRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeAccessor_MRFunctionVolume(VoxelsVolumeAccessor_MRFunctionVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRFunctionVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeAccessor_MRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRFunctionVolume`/`Const_VoxelsVolumeAccessor_MRFunctionVolume` to pass it to the function.
    public class _InOptConst_VoxelsVolumeAccessor_MRFunctionVolume
    {
        public Const_VoxelsVolumeAccessor_MRFunctionVolume? Opt;

        public _InOptConst_VoxelsVolumeAccessor_MRFunctionVolume() {}
        public _InOptConst_VoxelsVolumeAccessor_MRFunctionVolume(Const_VoxelsVolumeAccessor_MRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeAccessor_MRFunctionVolume(Const_VoxelsVolumeAccessor_MRFunctionVolume value) {return new(value);}
    }

    /// VoxelsVolumeAccessor specialization for VDB volume
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::VdbVolume>`.
    /// This is the const half of the class.
    public class Const_VoxelsVolumeAccessor_MRVdbVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeAccessor_MRVdbVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeAccessor_MR_VdbVolume_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeAccessor_MR_VdbVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeAccessor_MRVdbVolume() {Dispose(false);}

        ///< caching results of this accessor can improve performance
        public static unsafe bool CacheEffective
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_Get_cacheEffective", ExactSpelling = true)]
                extern static bool *__MR_VoxelsVolumeAccessor_MR_VdbVolume_Get_cacheEffective();
                return *__MR_VoxelsVolumeAccessor_MR_VdbVolume_Get_cacheEffective();
            }
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::VdbVolume>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRVdbVolume(MR._ByValue_VoxelsVolumeAccessor_MRVdbVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_VdbVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::VdbVolume>::VoxelsVolumeAccessor`.
        public unsafe Const_VoxelsVolumeAccessor_MRVdbVolume(MR.Const_VdbVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct(MR.Const_VdbVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct(volume._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::VdbVolume>::get`.
        public unsafe float Get(MR.Const_Vector3i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_get_MR_Vector3i", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeAccessor_MR_VdbVolume_get_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pos);
            return __MR_VoxelsVolumeAccessor_MR_VdbVolume_get_MR_Vector3i(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::VdbVolume>::get`.
        public unsafe float Get(MR.Const_VoxelLocation loc)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_get_MR_VoxelLocation", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeAccessor_MR_VdbVolume_get_MR_VoxelLocation(_Underlying *_this, MR.Const_VoxelLocation._Underlying *loc);
            return __MR_VoxelsVolumeAccessor_MR_VdbVolume_get_MR_VoxelLocation(_UnderlyingPtr, loc._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::VdbVolume>::minCoord`.
        public unsafe MR.Const_Vector3i MinCoord()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_minCoord", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_VoxelsVolumeAccessor_MR_VdbVolume_minCoord(_Underlying *_this);
            return new(__MR_VoxelsVolumeAccessor_MR_VdbVolume_minCoord(_UnderlyingPtr), is_owning: false);
        }

        /// this additional shift shall be added to integer voxel coordinates during transformation in 3D space
        /// Generated from method `MR::VoxelsVolumeAccessor<MR::VdbVolume>::shift`.
        public unsafe MR.Vector3f Shift()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_shift", ExactSpelling = true)]
            extern static MR.Vector3f __MR_VoxelsVolumeAccessor_MR_VdbVolume_shift(_Underlying *_this);
            return __MR_VoxelsVolumeAccessor_MR_VdbVolume_shift(_UnderlyingPtr);
        }
    }

    /// VoxelsVolumeAccessor specialization for VDB volume
    /// Generated from class `MR::VoxelsVolumeAccessor<MR::VdbVolume>`.
    /// This is the non-const half of the class.
    public class VoxelsVolumeAccessor_MRVdbVolume : Const_VoxelsVolumeAccessor_MRVdbVolume
    {
        internal unsafe VoxelsVolumeAccessor_MRVdbVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::VdbVolume>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRVdbVolume(MR._ByValue_VoxelsVolumeAccessor_MRVdbVolume _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_VdbVolume_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *_other);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_VdbVolume_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::VoxelsVolumeAccessor<MR::VdbVolume>::VoxelsVolumeAccessor`.
        public unsafe VoxelsVolumeAccessor_MRVdbVolume(MR.Const_VdbVolume volume) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct(MR.Const_VdbVolume._Underlying *volume);
            _UnderlyingPtr = __MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct(volume._UnderlyingPtr);
        }

        /// Generated from method `MR::VoxelsVolumeAccessor<MR::VdbVolume>::operator=`.
        public unsafe MR.VoxelsVolumeAccessor_MRVdbVolume Assign(MR._ByValue_VoxelsVolumeAccessor_MRVdbVolume _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeAccessor_MR_VdbVolume_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *__MR_VoxelsVolumeAccessor_MR_VdbVolume_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VoxelsVolumeAccessor_MRVdbVolume._Underlying *_other);
            return new(__MR_VoxelsVolumeAccessor_MR_VdbVolume_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `VoxelsVolumeAccessor_MRVdbVolume` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRVdbVolume`/`Const_VoxelsVolumeAccessor_MRVdbVolume` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VoxelsVolumeAccessor_MRVdbVolume
    {
        internal readonly Const_VoxelsVolumeAccessor_MRVdbVolume? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VoxelsVolumeAccessor_MRVdbVolume(Const_VoxelsVolumeAccessor_MRVdbVolume new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VoxelsVolumeAccessor_MRVdbVolume(Const_VoxelsVolumeAccessor_MRVdbVolume arg) {return new(arg);}
        public _ByValue_VoxelsVolumeAccessor_MRVdbVolume(MR.Misc._Moved<VoxelsVolumeAccessor_MRVdbVolume> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VoxelsVolumeAccessor_MRVdbVolume(MR.Misc._Moved<VoxelsVolumeAccessor_MRVdbVolume> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRVdbVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeAccessor_MRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRVdbVolume`/`Const_VoxelsVolumeAccessor_MRVdbVolume` directly.
    public class _InOptMut_VoxelsVolumeAccessor_MRVdbVolume
    {
        public VoxelsVolumeAccessor_MRVdbVolume? Opt;

        public _InOptMut_VoxelsVolumeAccessor_MRVdbVolume() {}
        public _InOptMut_VoxelsVolumeAccessor_MRVdbVolume(VoxelsVolumeAccessor_MRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeAccessor_MRVdbVolume(VoxelsVolumeAccessor_MRVdbVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeAccessor_MRVdbVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeAccessor_MRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeAccessor_MRVdbVolume`/`Const_VoxelsVolumeAccessor_MRVdbVolume` to pass it to the function.
    public class _InOptConst_VoxelsVolumeAccessor_MRVdbVolume
    {
        public Const_VoxelsVolumeAccessor_MRVdbVolume? Opt;

        public _InOptConst_VoxelsVolumeAccessor_MRVdbVolume() {}
        public _InOptConst_VoxelsVolumeAccessor_MRVdbVolume(Const_VoxelsVolumeAccessor_MRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeAccessor_MRVdbVolume(Const_VoxelsVolumeAccessor_MRVdbVolume value) {return new(value);}
    }
}
