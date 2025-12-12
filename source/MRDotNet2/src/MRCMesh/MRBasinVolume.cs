public static partial class MR
{
    /// the class to compute the volume of water some basin can accumulate,
    /// considering that water upper surface has constant z-level
    /// Generated from class `MR::BasinVolumeCalculator`.
    /// This is the const half of the class.
    public class Const_BasinVolumeCalculator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_BasinVolumeCalculator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_Destroy", ExactSpelling = true)]
            extern static void __MR_BasinVolumeCalculator_Destroy(_Underlying *_this);
            __MR_BasinVolumeCalculator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_BasinVolumeCalculator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_BasinVolumeCalculator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BasinVolumeCalculator._Underlying *__MR_BasinVolumeCalculator_DefaultConstruct();
            _UnderlyingPtr = __MR_BasinVolumeCalculator_DefaultConstruct();
        }

        /// Generated from constructor `MR::BasinVolumeCalculator::BasinVolumeCalculator`.
        public unsafe Const_BasinVolumeCalculator(MR.Const_BasinVolumeCalculator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BasinVolumeCalculator._Underlying *__MR_BasinVolumeCalculator_ConstructFromAnother(MR.BasinVolumeCalculator._Underlying *_other);
            _UnderlyingPtr = __MR_BasinVolumeCalculator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// call it after all addTerrainTri to get the volume
        /// Generated from method `MR::BasinVolumeCalculator::getVolume`.
        public unsafe double GetVolume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_getVolume", ExactSpelling = true)]
            extern static double __MR_BasinVolumeCalculator_getVolume(_Underlying *_this);
            return __MR_BasinVolumeCalculator_getVolume(_UnderlyingPtr);
        }
    }

    /// the class to compute the volume of water some basin can accumulate,
    /// considering that water upper surface has constant z-level
    /// Generated from class `MR::BasinVolumeCalculator`.
    /// This is the non-const half of the class.
    public class BasinVolumeCalculator : Const_BasinVolumeCalculator
    {
        internal unsafe BasinVolumeCalculator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe BasinVolumeCalculator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.BasinVolumeCalculator._Underlying *__MR_BasinVolumeCalculator_DefaultConstruct();
            _UnderlyingPtr = __MR_BasinVolumeCalculator_DefaultConstruct();
        }

        /// Generated from constructor `MR::BasinVolumeCalculator::BasinVolumeCalculator`.
        public unsafe BasinVolumeCalculator(MR.Const_BasinVolumeCalculator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.BasinVolumeCalculator._Underlying *__MR_BasinVolumeCalculator_ConstructFromAnother(MR.BasinVolumeCalculator._Underlying *_other);
            _UnderlyingPtr = __MR_BasinVolumeCalculator_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::BasinVolumeCalculator::operator=`.
        public unsafe MR.BasinVolumeCalculator Assign(MR.Const_BasinVolumeCalculator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.BasinVolumeCalculator._Underlying *__MR_BasinVolumeCalculator_AssignFromAnother(_Underlying *_this, MR.BasinVolumeCalculator._Underlying *_other);
            return new(__MR_BasinVolumeCalculator_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// pass every triangle of the basin here, and the water level;
        /// \return true if the triangle is at least partially below the water level and influences on the volume
        /// Generated from method `MR::BasinVolumeCalculator::addTerrainTri`.
        public unsafe bool AddTerrainTri(MR.Std.Array_MRVector3f_3 t, float level)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_BasinVolumeCalculator_addTerrainTri", ExactSpelling = true)]
            extern static byte __MR_BasinVolumeCalculator_addTerrainTri(_Underlying *_this, MR.Std.Array_MRVector3f_3 t, float level);
            return __MR_BasinVolumeCalculator_addTerrainTri(_UnderlyingPtr, t, level) != 0;
        }
    }

    /// This is used for optional parameters of class `BasinVolumeCalculator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_BasinVolumeCalculator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BasinVolumeCalculator`/`Const_BasinVolumeCalculator` directly.
    public class _InOptMut_BasinVolumeCalculator
    {
        public BasinVolumeCalculator? Opt;

        public _InOptMut_BasinVolumeCalculator() {}
        public _InOptMut_BasinVolumeCalculator(BasinVolumeCalculator value) {Opt = value;}
        public static implicit operator _InOptMut_BasinVolumeCalculator(BasinVolumeCalculator value) {return new(value);}
    }

    /// This is used for optional parameters of class `BasinVolumeCalculator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_BasinVolumeCalculator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `BasinVolumeCalculator`/`Const_BasinVolumeCalculator` to pass it to the function.
    public class _InOptConst_BasinVolumeCalculator
    {
        public Const_BasinVolumeCalculator? Opt;

        public _InOptConst_BasinVolumeCalculator() {}
        public _InOptConst_BasinVolumeCalculator(Const_BasinVolumeCalculator value) {Opt = value;}
        public static implicit operator _InOptConst_BasinVolumeCalculator(Const_BasinVolumeCalculator value) {return new(value);}
    }

    /// computes the volume of given mesh basin below given water level;
    /// \param faces shall include all basin faces at least partially below the water level
    /// Generated from function `MR::computeBasinVolume`.
    public static unsafe double ComputeBasinVolume(MR.Const_Mesh mesh, MR.Const_FaceBitSet faces, float level)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeBasinVolume", ExactSpelling = true)]
        extern static double __MR_computeBasinVolume(MR.Const_Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *faces, float level);
        return __MR_computeBasinVolume(mesh._UnderlyingPtr, faces._UnderlyingPtr, level);
    }
}
