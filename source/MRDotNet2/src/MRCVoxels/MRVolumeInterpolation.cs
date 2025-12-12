public static partial class MR
{
    /// helper class for generalized access to voxel volume data with trilinear interpolation
    /// coordinate: 0       voxelSize
    ///             |       |
    ///             I---*---I---*---I---
    ///             |       |       |
    /// value:     [0]     [1]     [2] ...
    /// note: this class is as thread-safe as the underlying Accessor
    /// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
    /// Generated from class `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>`.
    /// This is the const half of the class.
    public class Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume() {Dispose(false);}

        /// create an accessor instance that stores references to volume and its accessor
        /// the volume should not modified while it is accessed by this class
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(MR.Const_VdbVolume volume, MR.Const_VoxelsVolumeAccessor_MRVdbVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VdbVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VdbVolume(MR.Const_VdbVolume._Underlying *volume, MR.Const_VoxelsVolumeAccessor_MRVdbVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VdbVolume(volume._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// a copying-like constructor with explicitly provided accessor
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume other, MR.Const_VoxelsVolumeAccessor_MRVdbVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume._Underlying *other, MR.Const_VoxelsVolumeAccessor_MRVdbVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume(other._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// get value at specified coordinates
        /// Generated from method `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>::get`.
        public unsafe float Get(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_get", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_get(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_get(_UnderlyingPtr, pos._UnderlyingPtr);
        }
    }

    /// helper class for generalized access to voxel volume data with trilinear interpolation
    /// coordinate: 0       voxelSize
    ///             |       |
    ///             I---*---I---*---I---
    ///             |       |       |
    /// value:     [0]     [1]     [2] ...
    /// note: this class is as thread-safe as the underlying Accessor
    /// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
    /// Generated from class `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>`.
    /// This is the non-const half of the class.
    public class VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume : Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume
    {
        internal unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// create an accessor instance that stores references to volume and its accessor
        /// the volume should not modified while it is accessed by this class
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(MR.Const_VdbVolume volume, MR.Const_VoxelsVolumeAccessor_MRVdbVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VdbVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VdbVolume(MR.Const_VdbVolume._Underlying *volume, MR.Const_VoxelsVolumeAccessor_MRVdbVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VdbVolume(volume._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// a copying-like constructor with explicitly provided accessor
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::VdbVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume other, MR.Const_VoxelsVolumeAccessor_MRVdbVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume._Underlying *other, MR.Const_VoxelsVolumeAccessor_MRVdbVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_VdbVolume(other._UnderlyingPtr, accessor._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume`/`Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume` directly.
    public class _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume
    {
        public VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume? Opt;

        public _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume() {}
        public _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume`/`Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume` to pass it to the function.
    public class _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume
    {
        public Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume? Opt;

        public _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume() {}
        public _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume(Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRVdbVolume value) {return new(value);}
    }

    /// helper class for generalized access to voxel volume data with trilinear interpolation
    /// coordinate: 0       voxelSize
    ///             |       |
    ///             I---*---I---*---I---
    ///             |       |       |
    /// value:     [0]     [1]     [2] ...
    /// note: this class is as thread-safe as the underlying Accessor
    /// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
    /// Generated from class `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>`.
    /// This is the const half of the class.
    public class Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax() {Dispose(false);}

        /// create an accessor instance that stores references to volume and its accessor
        /// the volume should not modified while it is accessed by this class
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(MR.Const_SimpleVolumeMinMax volume, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_SimpleVolumeMinMax", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_SimpleVolumeMinMax(MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_SimpleVolumeMinMax(volume._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// a copying-like constructor with explicitly provided accessor
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax other, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax._Underlying *other, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax(other._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// get value at specified coordinates
        /// Generated from method `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>::get`.
        public unsafe float Get(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_get", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_get(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_get(_UnderlyingPtr, pos._UnderlyingPtr);
        }
    }

    /// helper class for generalized access to voxel volume data with trilinear interpolation
    /// coordinate: 0       voxelSize
    ///             |       |
    ///             I---*---I---*---I---
    ///             |       |       |
    /// value:     [0]     [1]     [2] ...
    /// note: this class is as thread-safe as the underlying Accessor
    /// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
    /// Generated from class `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>`.
    /// This is the non-const half of the class.
    public class VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax : Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax
    {
        internal unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// create an accessor instance that stores references to volume and its accessor
        /// the volume should not modified while it is accessed by this class
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(MR.Const_SimpleVolumeMinMax volume, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_SimpleVolumeMinMax", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_SimpleVolumeMinMax(MR.Const_SimpleVolumeMinMax._Underlying *volume, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_SimpleVolumeMinMax(volume._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// a copying-like constructor with explicitly provided accessor
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::SimpleVolumeMinMax>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax other, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax._Underlying *other, MR.Const_VoxelsVolumeAccessor_MRSimpleVolumeMinMax._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_SimpleVolumeMinMax(other._UnderlyingPtr, accessor._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax`/`Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax` directly.
    public class _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax
    {
        public VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax? Opt;

        public _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax() {}
        public _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax`/`Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax` to pass it to the function.
    public class _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax
    {
        public Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax? Opt;

        public _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax() {}
        public _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax(Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRSimpleVolumeMinMax value) {return new(value);}
    }

    /// helper class for generalized access to voxel volume data with trilinear interpolation
    /// coordinate: 0       voxelSize
    ///             |       |
    ///             I---*---I---*---I---
    ///             |       |       |
    /// value:     [0]     [1]     [2] ...
    /// note: this class is as thread-safe as the underlying Accessor
    /// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
    /// Generated from class `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>`.
    /// This is the const half of the class.
    public class Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Destroy", ExactSpelling = true)]
            extern static void __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Destroy(_Underlying *_this);
            __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume() {Dispose(false);}

        /// create an accessor instance that stores references to volume and its accessor
        /// the volume should not modified while it is accessed by this class
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(MR.Const_FunctionVolume volume, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_FunctionVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_FunctionVolume(MR.Const_FunctionVolume._Underlying *volume, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_FunctionVolume(volume._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// a copying-like constructor with explicitly provided accessor
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume other, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume._Underlying *other, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume(other._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// get value at specified coordinates
        /// Generated from method `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>::get`.
        public unsafe float Get(MR.Const_Vector3f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_get", ExactSpelling = true)]
            extern static float __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_get(_Underlying *_this, MR.Const_Vector3f._Underlying *pos);
            return __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_get(_UnderlyingPtr, pos._UnderlyingPtr);
        }
    }

    /// helper class for generalized access to voxel volume data with trilinear interpolation
    /// coordinate: 0       voxelSize
    ///             |       |
    ///             I---*---I---*---I---
    ///             |       |       |
    /// value:     [0]     [1]     [2] ...
    /// note: this class is as thread-safe as the underlying Accessor
    /// e.g. VoxelsVolumeAccessor<VdbVolume> is not thread-safe (but several instances on same volume is thread-safe)
    /// Generated from class `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>`.
    /// This is the non-const half of the class.
    public class VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume : Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume
    {
        internal unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// create an accessor instance that stores references to volume and its accessor
        /// the volume should not modified while it is accessed by this class
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(MR.Const_FunctionVolume volume, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_FunctionVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_FunctionVolume(MR.Const_FunctionVolume._Underlying *volume, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_FunctionVolume(volume._UnderlyingPtr, accessor._UnderlyingPtr);
        }

        /// a copying-like constructor with explicitly provided accessor
        /// Generated from constructor `MR::VoxelsVolumeInterpolatedAccessor<MR::VoxelsVolumeAccessor<MR::FunctionVolume>>::VoxelsVolumeInterpolatedAccessor`.
        public unsafe VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume other, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume accessor) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume", ExactSpelling = true)]
            extern static MR.VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume._Underlying *__MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume(MR.Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume._Underlying *other, MR.Const_VoxelsVolumeAccessor_MRFunctionVolume._Underlying *accessor);
            _UnderlyingPtr = __MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume_Construct_MR_VoxelsVolumeInterpolatedAccessor_MR_VoxelsVolumeAccessor_MR_FunctionVolume(other._UnderlyingPtr, accessor._UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume`/`Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume` directly.
    public class _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume
    {
        public VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume? Opt;

        public _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume() {}
        public _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume value) {return new(value);}
    }

    /// This is used for optional parameters of class `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume`/`Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume` to pass it to the function.
    public class _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume
    {
        public Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume? Opt;

        public _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume() {}
        public _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume value) {Opt = value;}
        public static implicit operator _InOptConst_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume(Const_VoxelsVolumeInterpolatedAccessor_MRVoxelsVolumeAccessorMRFunctionVolume value) {return new(value);}
    }
}
