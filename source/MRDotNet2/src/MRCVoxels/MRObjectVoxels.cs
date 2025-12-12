public static partial class MR
{
    /// This class stores information about voxels object
    /// Generated from class `MR::ObjectVoxels`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectMeshHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the const half of the class.
    public class Const_ObjectVoxels : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_ObjectVoxels_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_ObjectVoxels_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_ObjectVoxels_UseCount();
                return __MR_std_shared_ptr_MR_ObjectVoxels_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectVoxels_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_ObjectVoxels_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_ObjectVoxels(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectVoxels_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectVoxels_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectVoxels_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectVoxels_ConstructNonOwning(ptr);
        }

        internal unsafe Const_ObjectVoxels(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe ObjectVoxels _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectVoxels_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_ObjectVoxels_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_ObjectVoxels_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_ObjectVoxels_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_ObjectVoxels_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_ObjectVoxels_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_ObjectVoxels_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ObjectVoxels() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_ObjectChildrenHolder(Const_ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectChildrenHolder._Underlying *__MR_ObjectVoxels_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.Const_ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_Object(Const_ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Const_Object._Underlying *__MR_ObjectVoxels_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_VisualObject(Const_ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.Const_VisualObject._Underlying *__MR_ObjectVoxels_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Const_ObjectMeshHolder(Const_ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static MR.Const_ObjectMeshHolder._Underlying *__MR_ObjectVoxels_UpcastTo_MR_ObjectMeshHolder(_Underlying *_this);
            return MR.Const_ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_ObjectMeshHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator Const_ObjectVoxels?(MR.Const_Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectVoxels", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectVoxels(MR.Const_Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectVoxels(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectVoxels?(MR.Const_VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectVoxels", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectVoxels(MR.Const_VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectVoxels(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator Const_ObjectVoxels?(MR.Const_ObjectMeshHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectVoxels", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectVoxels(MR.Const_ObjectMeshHolder._Underlying *_this);
            var ptr = __MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectVoxels(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Const_ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ObjectVoxels() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectVoxels._Underlying *__MR_ObjectVoxels_DefaultConstruct();
            _LateMakeShared(__MR_ObjectVoxels_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectVoxels::ObjectVoxels`.
        public unsafe Const_ObjectVoxels(MR._ByValue_ObjectVoxels _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectVoxels._Underlying *__MR_ObjectVoxels_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectVoxels._Underlying *_other);
            _LateMakeShared(__MR_ObjectVoxels_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectVoxels::StaticTypeName`.
        public static unsafe byte? StaticTypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_StaticTypeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_StaticTypeName();
            var __ret = __MR_ObjectVoxels_StaticTypeName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectVoxels::typeName`.
        public unsafe byte? TypeName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_typeName", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_typeName(_Underlying *_this);
            var __ret = __MR_ObjectVoxels_typeName(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectVoxels::StaticClassName`.
        public static unsafe byte? StaticClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_StaticClassName", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_StaticClassName();
            var __ret = __MR_ObjectVoxels_StaticClassName();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectVoxels::className`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassName()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_className", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectVoxels_className(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectVoxels_className(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectVoxels::StaticClassNameInPlural`.
        public static unsafe byte? StaticClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_StaticClassNameInPlural", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_StaticClassNameInPlural();
            var __ret = __MR_ObjectVoxels_StaticClassNameInPlural();
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectVoxels::classNameInPlural`.
        public unsafe MR.Misc._Moved<MR.Std.String> ClassNameInPlural()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_classNameInPlural", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_ObjectVoxels_classNameInPlural(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.String(__MR_ObjectVoxels_classNameInPlural(_UnderlyingPtr), is_owning: true));
        }

        /// Returns iso surface, empty if iso value is not set
        /// Generated from method `MR::ObjectVoxels::surface`.
        public unsafe MR.Const_Mesh Surface()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_surface", ExactSpelling = true)]
            extern static MR.Const_Mesh._UnderlyingShared *__MR_ObjectVoxels_surface(_Underlying *_this);
            return new(__MR_ObjectVoxels_surface(_UnderlyingPtr), is_owning: false);
        }

        /// Return VdbVolume
        /// Generated from method `MR::ObjectVoxels::vdbVolume`.
        public unsafe MR.Const_VdbVolume VdbVolume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_vdbVolume", ExactSpelling = true)]
            extern static MR.Const_VdbVolume._Underlying *__MR_ObjectVoxels_vdbVolume(_Underlying *_this);
            return new(__MR_ObjectVoxels_vdbVolume(_UnderlyingPtr), is_owning: false);
        }

        /// Returns Float grid which contains voxels data, see more on openvdb::FloatGrid
        /// Generated from method `MR::ObjectVoxels::grid`.
        public unsafe MR.Const_FloatGrid Grid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_grid", ExactSpelling = true)]
            extern static MR.Const_FloatGrid._Underlying *__MR_ObjectVoxels_grid(_Underlying *_this);
            return new(__MR_ObjectVoxels_grid(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::hasModel`.
        public unsafe bool HasModel()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_hasModel", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_hasModel(_Underlying *_this);
            return __MR_ObjectVoxels_hasModel(_UnderlyingPtr) != 0;
        }

        /// Returns dimensions of voxel objects
        /// Generated from method `MR::ObjectVoxels::dimensions`.
        public unsafe MR.Const_Vector3i Dimensions()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_dimensions", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_ObjectVoxels_dimensions(_Underlying *_this);
            return new(__MR_ObjectVoxels_dimensions(_UnderlyingPtr), is_owning: false);
        }

        /// Returns current iso value
        /// Generated from method `MR::ObjectVoxels::getIsoValue`.
        public unsafe float GetIsoValue()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getIsoValue", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_getIsoValue(_Underlying *_this);
            return __MR_ObjectVoxels_getIsoValue(_UnderlyingPtr);
        }

        /// Returns histogram
        /// Generated from method `MR::ObjectVoxels::histogram`.
        public unsafe MR.Const_Histogram Histogram()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_histogram", ExactSpelling = true)]
            extern static MR.Const_Histogram._Underlying *__MR_ObjectVoxels_histogram(_Underlying *_this);
            return new(__MR_ObjectVoxels_histogram(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::voxelSize`.
        public unsafe MR.Const_Vector3f VoxelSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_voxelSize", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_ObjectVoxels_voxelSize(_Underlying *_this);
            return new(__MR_ObjectVoxels_voxelSize(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getInfoLines`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetInfoLines()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getInfoLines", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_ObjectVoxels_getInfoLines(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_ObjectVoxels_getInfoLines(_UnderlyingPtr), is_owning: true));
        }

        /// Calculates and return new mesh or error message
        /// Generated from method `MR::ObjectVoxels::recalculateIsoSurface`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_StdSharedPtrMRMesh_StdString> RecalculateIsoSurface(float iso, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_recalculateIsoSurface_2", ExactSpelling = true)]
            extern static MR.Expected_StdSharedPtrMRMesh_StdString._Underlying *__MR_ObjectVoxels_recalculateIsoSurface_2(_Underlying *_this, float iso, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_StdSharedPtrMRMesh_StdString(__MR_ObjectVoxels_recalculateIsoSurface_2(_UnderlyingPtr, iso, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Same as above, but takes external volume
        /// Generated from method `MR::ObjectVoxels::recalculateIsoSurface`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Expected_StdSharedPtrMRMesh_StdString> RecalculateIsoSurface(MR.Const_VdbVolume volume, float iso, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_recalculateIsoSurface_3", ExactSpelling = true)]
            extern static MR.Expected_StdSharedPtrMRMesh_StdString._Underlying *__MR_ObjectVoxels_recalculateIsoSurface_3(_Underlying *_this, MR.Const_VdbVolume._Underlying *volume, float iso, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Expected_StdSharedPtrMRMesh_StdString(__MR_ObjectVoxels_recalculateIsoSurface_3(_UnderlyingPtr, volume._UnderlyingPtr, iso, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Calculates and returns new histogram
        /// Generated from method `MR::ObjectVoxels::recalculateHistogram`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe MR.Misc._Moved<MR.Histogram> RecalculateHistogram(MR._InOpt_Vector2f minmax, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_recalculateHistogram", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_ObjectVoxels_recalculateHistogram(_Underlying *_this, MR.Vector2f *minmax, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Histogram(__MR_ObjectVoxels_recalculateHistogram(_UnderlyingPtr, minmax.HasValue ? &minmax.Object : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// returns true if the iso-surface is built using Dual Marching Cubes algorithm or false if using Standard Marching Cubes
        /// Generated from method `MR::ObjectVoxels::getDualMarchingCubes`.
        public unsafe bool GetDualMarchingCubes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getDualMarchingCubes", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_getDualMarchingCubes(_Underlying *_this);
            return __MR_ObjectVoxels_getDualMarchingCubes(_UnderlyingPtr) != 0;
        }

        /// Returns active bounds (max excluded)
        /// active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty
        /// Generated from method `MR::ObjectVoxels::getActiveBounds`.
        public unsafe MR.Const_Box3i GetActiveBounds()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getActiveBounds", ExactSpelling = true)]
            extern static MR.Const_Box3i._Underlying *__MR_ObjectVoxels_getActiveBounds(_Underlying *_this);
            return new(__MR_ObjectVoxels_getActiveBounds(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getSelectedVoxels`.
        public unsafe MR.Const_VoxelBitSet GetSelectedVoxels()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedVoxels", ExactSpelling = true)]
            extern static MR.Const_VoxelBitSet._Underlying *__MR_ObjectVoxels_getSelectedVoxels(_Underlying *_this);
            return new(__MR_ObjectVoxels_getSelectedVoxels(_UnderlyingPtr), is_owning: false);
        }

        /// get active (visible) voxels
        /// Generated from method `MR::ObjectVoxels::getVolumeRenderActiveVoxels`.
        public unsafe MR.Const_VoxelBitSet GetVolumeRenderActiveVoxels()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVolumeRenderActiveVoxels", ExactSpelling = true)]
            extern static MR.Const_VoxelBitSet._Underlying *__MR_ObjectVoxels_getVolumeRenderActiveVoxels(_Underlying *_this);
            return new(__MR_ObjectVoxels_getVolumeRenderActiveVoxels(_UnderlyingPtr), is_owning: false);
        }

        /// VoxelId is numerical representation of voxel
        /// Coordinate is {x,y,z} indices of voxels in box (base dimensions space, NOT active dimensions)
        /// Point is local space coordinate of point in scene
        /// Generated from method `MR::ObjectVoxels::getVoxelIdByCoordinate`.
        public unsafe MR.VoxelId GetVoxelIdByCoordinate(MR.Const_Vector3i coord)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVoxelIdByCoordinate", ExactSpelling = true)]
            extern static MR.VoxelId __MR_ObjectVoxels_getVoxelIdByCoordinate(_Underlying *_this, MR.Const_Vector3i._Underlying *coord);
            return __MR_ObjectVoxels_getVoxelIdByCoordinate(_UnderlyingPtr, coord._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::getVoxelIdByPoint`.
        public unsafe MR.VoxelId GetVoxelIdByPoint(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVoxelIdByPoint", ExactSpelling = true)]
            extern static MR.VoxelId __MR_ObjectVoxels_getVoxelIdByPoint(_Underlying *_this, MR.Const_Vector3f._Underlying *point);
            return __MR_ObjectVoxels_getVoxelIdByPoint(_UnderlyingPtr, point._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::getCoordinateByVoxelId`.
        public unsafe MR.Vector3i GetCoordinateByVoxelId(MR.VoxelId id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getCoordinateByVoxelId", ExactSpelling = true)]
            extern static MR.Vector3i __MR_ObjectVoxels_getCoordinateByVoxelId(_Underlying *_this, MR.VoxelId id);
            return __MR_ObjectVoxels_getCoordinateByVoxelId(_UnderlyingPtr, id);
        }

        /// Returns indexer with more options
        /// Generated from method `MR::ObjectVoxels::getVolumeIndexer`.
        public unsafe MR.Const_VolumeIndexer GetVolumeIndexer()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVolumeIndexer", ExactSpelling = true)]
            extern static MR.Const_VolumeIndexer._Underlying *__MR_ObjectVoxels_getVolumeIndexer(_Underlying *_this);
            return new(__MR_ObjectVoxels_getVolumeIndexer(_UnderlyingPtr), is_owning: false);
        }

        // prepare data for volume rendering
        // returns false if canceled or voxel data is empty
        /// Generated from method `MR::ObjectVoxels::prepareDataForVolumeRendering`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe bool PrepareDataForVolumeRendering(MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_prepareDataForVolumeRendering", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_prepareDataForVolumeRendering(_Underlying *_this, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return __MR_ObjectVoxels_prepareDataForVolumeRendering(_UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectVoxels::isVolumeRenderingEnabled`.
        public unsafe bool IsVolumeRenderingEnabled()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isVolumeRenderingEnabled", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isVolumeRenderingEnabled(_Underlying *_this);
            return __MR_ObjectVoxels_isVolumeRenderingEnabled(_UnderlyingPtr) != 0;
        }

        // move volume rendering data to caller: basically used in RenderVolumeObject
        /// Generated from method `MR::ObjectVoxels::getVolumeRenderingData`.
        public unsafe void *GetVolumeRenderingData()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVolumeRenderingData", ExactSpelling = true)]
            extern static void *__MR_ObjectVoxels_getVolumeRenderingData(_Underlying *_this);
            return __MR_ObjectVoxels_getVolumeRenderingData(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::getVolumeRenderingParams`.
        public unsafe MR.ObjectVoxels.Const_VolumeRenderingParams GetVolumeRenderingParams()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVolumeRenderingParams", ExactSpelling = true)]
            extern static MR.ObjectVoxels.Const_VolumeRenderingParams._Underlying *__MR_ObjectVoxels_getVolumeRenderingParams(_Underlying *_this);
            return new(__MR_ObjectVoxels_getVolumeRenderingParams(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::hasVisualRepresentation`.
        public unsafe bool HasVisualRepresentation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_hasVisualRepresentation", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_hasVisualRepresentation(_Underlying *_this);
            return __MR_ObjectVoxels_hasVisualRepresentation(_UnderlyingPtr) != 0;
        }

        /// gets top limit on the number of vertices in the iso-surface
        /// Generated from method `MR::ObjectVoxels::getMaxSurfaceVertices`.
        public unsafe int GetMaxSurfaceVertices()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getMaxSurfaceVertices", ExactSpelling = true)]
            extern static int __MR_ObjectVoxels_getMaxSurfaceVertices(_Underlying *_this);
            return __MR_ObjectVoxels_getMaxSurfaceVertices(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::clone`.
        public unsafe MR.Misc._Moved<MR.Object> Clone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_clone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectVoxels_clone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectVoxels_clone(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from method `MR::ObjectVoxels::shallowClone`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowClone()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_shallowClone", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectVoxels_shallowClone(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectVoxels_shallowClone(_UnderlyingPtr), is_owning: true));
        }

        /// returns cached information about the number of active voxels
        /// Generated from method `MR::ObjectVoxels::activeVoxels`.
        public unsafe ulong ActiveVoxels()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_activeVoxels", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_activeVoxels(_Underlying *_this);
            return __MR_ObjectVoxels_activeVoxels(_UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::ObjectVoxels::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_heapBytes(_Underlying *_this);
            return __MR_ObjectVoxels_heapBytes(_UnderlyingPtr);
        }

        /// returns overriden file extension used to serialize voxels inside this object, nullptr means defaultSerializeVoxelsFormat()
        /// Generated from method `MR::ObjectVoxels::serializeFormat`.
        public unsafe byte? SerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_serializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_serializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectVoxels_serializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectVoxels::mesh`.
        public unsafe MR.Const_Mesh Mesh()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_mesh", ExactSpelling = true)]
            extern static MR.Const_Mesh._UnderlyingShared *__MR_ObjectVoxels_mesh(_Underlying *_this);
            return new(__MR_ObjectVoxels_mesh(_UnderlyingPtr), is_owning: false);
        }

        /// \return the pair ( mesh, selected triangles ) if any triangle is selected or whole mesh otherwise
        /// Generated from method `MR::ObjectVoxels::meshPart`.
        public unsafe MR.MeshPart MeshPart()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_meshPart", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_ObjectVoxels_meshPart(_Underlying *_this);
            return new(__MR_ObjectVoxels_meshPart(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectVoxels::getSelectedFaces`.
        public unsafe MR.Const_FaceBitSet GetSelectedFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedFaces", ExactSpelling = true)]
            extern static MR.Const_FaceBitSet._Underlying *__MR_ObjectVoxels_getSelectedFaces(_Underlying *_this);
            return new(__MR_ObjectVoxels_getSelectedFaces(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected triangles
        /// Generated from method `MR::ObjectVoxels::getSelectedFacesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedFacesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedFacesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getSelectedFacesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectVoxels_getSelectedFacesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getSelectedEdges`.
        public unsafe MR.Const_UndirectedEdgeBitSet GetSelectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedEdges", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectVoxels_getSelectedEdges(_Underlying *_this);
            return new(__MR_ObjectVoxels_getSelectedEdges(_UnderlyingPtr), is_owning: false);
        }

        /// returns colors of selected edges
        /// Generated from method `MR::ObjectVoxels::getSelectedEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetSelectedEdgesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedEdgesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getSelectedEdgesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectVoxels_getSelectedEdgesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getSelectedEdgesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedEdgesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getSelectedEdgesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getSelectedEdgesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getSelectedFacesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetSelectedFacesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSelectedFacesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getSelectedFacesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getSelectedFacesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getEdgesColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetEdgesColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getEdgesColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getEdgesColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getPointsColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetPointsColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getPointsColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getPointsColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getPointsColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getBordersColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBordersColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getBordersColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getBordersColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getBordersColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// Edges on mesh, that will have sharp visualization even with smooth shading
        /// Generated from method `MR::ObjectVoxels::creases`.
        public unsafe MR.Const_UndirectedEdgeBitSet Creases()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_creases", ExactSpelling = true)]
            extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ObjectVoxels_creases(_Underlying *_this);
            return new(__MR_ObjectVoxels_creases(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::flatShading`.
        public unsafe bool FlatShading()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_flatShading", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_flatShading(_Underlying *_this);
            return __MR_ObjectVoxels_flatShading(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectVoxels::supportsVisualizeProperty`.
        public unsafe bool SupportsVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_supportsVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_supportsVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return __MR_ObjectVoxels_supportsVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr) != 0;
        }

        /// get all visualize properties masks
        /// Generated from method `MR::ObjectVoxels::getAllVisualizeProperties`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_MRViewportMask> GetAllVisualizeProperties()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getAllVisualizeProperties", ExactSpelling = true)]
            extern static MR.Std.Vector_MRViewportMask._Underlying *__MR_ObjectVoxels_getAllVisualizeProperties(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_MRViewportMask(__MR_ObjectVoxels_getAllVisualizeProperties(_UnderlyingPtr), is_owning: true));
        }

        /// returns mask of viewports where given property is set
        /// Generated from method `MR::ObjectVoxels::getVisualizePropertyMask`.
        public unsafe MR.Const_ViewportMask GetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVisualizePropertyMask", ExactSpelling = true)]
            extern static MR.Const_ViewportMask._Underlying *__MR_ObjectVoxels_getVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type);
            return new(__MR_ObjectVoxels_getVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr), is_owning: false);
        }

        /// provides read-only access to whole ObjectMeshData
        /// Generated from method `MR::ObjectVoxels::data`.
        public unsafe MR.Const_ObjectMeshData Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_data", ExactSpelling = true)]
            extern static MR.Const_ObjectMeshData._Underlying *__MR_ObjectVoxels_data(_Underlying *_this);
            return new(__MR_ObjectVoxels_data(_UnderlyingPtr), is_owning: false);
        }

        /// returns per-vertex colors of the object
        /// Generated from method `MR::ObjectVoxels::getVertsColorMap`.
        public unsafe MR.Const_VertColors GetVertsColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVertsColorMap", ExactSpelling = true)]
            extern static MR.Const_VertColors._Underlying *__MR_ObjectVoxels_getVertsColorMap(_Underlying *_this);
            return new(__MR_ObjectVoxels_getVertsColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getFacesColorMap`.
        public unsafe MR.Const_FaceColors GetFacesColorMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getFacesColorMap", ExactSpelling = true)]
            extern static MR.Const_FaceColors._Underlying *__MR_ObjectVoxels_getFacesColorMap(_Underlying *_this);
            return new(__MR_ObjectVoxels_getFacesColorMap(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getEdgeWidth`.
        public unsafe float GetEdgeWidth()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getEdgeWidth", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_getEdgeWidth(_Underlying *_this);
            return __MR_ObjectVoxels_getEdgeWidth(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::getPointSize`.
        public unsafe float GetPointSize()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getPointSize", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_getPointSize(_Underlying *_this);
            return __MR_ObjectVoxels_getPointSize(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::getEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetEdgesColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getEdgesColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getEdgesColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectVoxels_getEdgesColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getPointsColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetPointsColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getPointsColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getPointsColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectVoxels_getPointsColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getBordersColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_Color GetBordersColor(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getBordersColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getBordersColor(_Underlying *_this, MR.ViewportId *id);
            return new(__MR_ObjectVoxels_getBordersColor(_UnderlyingPtr, id.HasValue ? &id.Object : null), is_owning: false);
        }

        /// returns first texture in the vector. If there is no textures, returns empty texture
        /// Generated from method `MR::ObjectVoxels::getTexture`.
        public unsafe MR.Const_MeshTexture GetTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getTexture", ExactSpelling = true)]
            extern static MR.Const_MeshTexture._Underlying *__MR_ObjectVoxels_getTexture(_Underlying *_this);
            return new(__MR_ObjectVoxels_getTexture(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getTextures`.
        public unsafe MR.Const_Vector_MRMeshTexture_MRTextureId GetTextures()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getTextures", ExactSpelling = true)]
            extern static MR.Const_Vector_MRMeshTexture_MRTextureId._Underlying *__MR_ObjectVoxels_getTextures(_Underlying *_this);
            return new(__MR_ObjectVoxels_getTextures(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getTexturePerFace`.
        public unsafe MR.Const_TexturePerFace GetTexturePerFace()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getTexturePerFace", ExactSpelling = true)]
            extern static MR.Const_TexturePerFace._Underlying *__MR_ObjectVoxels_getTexturePerFace(_Underlying *_this);
            return new(__MR_ObjectVoxels_getTexturePerFace(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getUVCoords`.
        public unsafe MR.Const_VertCoords2 GetUVCoords()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getUVCoords", ExactSpelling = true)]
            extern static MR.Const_VertCoords2._Underlying *__MR_ObjectVoxels_getUVCoords(_Underlying *_this);
            return new(__MR_ObjectVoxels_getUVCoords(_UnderlyingPtr), is_owning: false);
        }

        // ancillary texture can be used to have custom features visualization without affecting real one
        /// Generated from method `MR::ObjectVoxels::getAncillaryTexture`.
        public unsafe MR.Const_MeshTexture GetAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getAncillaryTexture", ExactSpelling = true)]
            extern static MR.Const_MeshTexture._Underlying *__MR_ObjectVoxels_getAncillaryTexture(_Underlying *_this);
            return new(__MR_ObjectVoxels_getAncillaryTexture(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::getAncillaryUVCoords`.
        public unsafe MR.Const_VertCoords2 GetAncillaryUVCoords()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getAncillaryUVCoords", ExactSpelling = true)]
            extern static MR.Const_VertCoords2._Underlying *__MR_ObjectVoxels_getAncillaryUVCoords(_Underlying *_this);
            return new(__MR_ObjectVoxels_getAncillaryUVCoords(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::hasAncillaryTexture`.
        public unsafe bool HasAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_hasAncillaryTexture", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_hasAncillaryTexture(_Underlying *_this);
            return __MR_ObjectVoxels_hasAncillaryTexture(_UnderlyingPtr) != 0;
        }

        /// returns dirty flag of currently using normal type if they are dirty in render representation
        /// Generated from method `MR::ObjectVoxels::getNeededNormalsRenderDirtyValue`.
        public unsafe uint GetNeededNormalsRenderDirtyValue(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getNeededNormalsRenderDirtyValue", ExactSpelling = true)]
            extern static uint __MR_ObjectVoxels_getNeededNormalsRenderDirtyValue(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_getNeededNormalsRenderDirtyValue(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::getRedrawFlag`.
        public unsafe bool GetRedrawFlag(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getRedrawFlag", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_getRedrawFlag(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_getRedrawFlag(_UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns cached information whether the mesh is closed
        /// Generated from method `MR::ObjectVoxels::isMeshClosed`.
        public unsafe bool IsMeshClosed()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isMeshClosed", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isMeshClosed(_Underlying *_this);
            return __MR_ObjectVoxels_isMeshClosed(_UnderlyingPtr) != 0;
        }

        /// returns cached bounding box of this mesh object in world coordinates;
        /// if you need bounding box in local coordinates please call getBoundingBox()
        /// Generated from method `MR::ObjectVoxels::getWorldBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getWorldBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectVoxels_getWorldBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectVoxels_getWorldBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// returns cached information about the number of selected faces in the mesh
        /// Generated from method `MR::ObjectVoxels::numSelectedFaces`.
        public unsafe ulong NumSelectedFaces()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numSelectedFaces", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numSelectedFaces(_Underlying *_this);
            return __MR_ObjectVoxels_numSelectedFaces(_UnderlyingPtr);
        }

        /// returns cached information about the number of selected undirected edges in the mesh
        /// Generated from method `MR::ObjectVoxels::numSelectedEdges`.
        public unsafe ulong NumSelectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numSelectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numSelectedEdges(_Underlying *_this);
            return __MR_ObjectVoxels_numSelectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of crease undirected edges in the mesh
        /// Generated from method `MR::ObjectVoxels::numCreaseEdges`.
        public unsafe ulong NumCreaseEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numCreaseEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numCreaseEdges(_Underlying *_this);
            return __MR_ObjectVoxels_numCreaseEdges(_UnderlyingPtr);
        }

        /// returns cached summed area of mesh triangles
        /// Generated from method `MR::ObjectVoxels::totalArea`.
        public unsafe double TotalArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_totalArea", ExactSpelling = true)]
            extern static double __MR_ObjectVoxels_totalArea(_Underlying *_this);
            return __MR_ObjectVoxels_totalArea(_UnderlyingPtr);
        }

        /// returns cached area of selected triangles
        /// Generated from method `MR::ObjectVoxels::selectedArea`.
        public unsafe double SelectedArea()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_selectedArea", ExactSpelling = true)]
            extern static double __MR_ObjectVoxels_selectedArea(_Underlying *_this);
            return __MR_ObjectVoxels_selectedArea(_UnderlyingPtr);
        }

        /// returns cached volume of space surrounded by the mesh, which is valid only if mesh is closed
        /// Generated from method `MR::ObjectVoxels::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_volume", ExactSpelling = true)]
            extern static double __MR_ObjectVoxels_volume(_Underlying *_this);
            return __MR_ObjectVoxels_volume(_UnderlyingPtr);
        }

        /// returns cached average edge length
        /// Generated from method `MR::ObjectVoxels::avgEdgeLen`.
        public unsafe float AvgEdgeLen()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_avgEdgeLen", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_avgEdgeLen(_Underlying *_this);
            return __MR_ObjectVoxels_avgEdgeLen(_UnderlyingPtr);
        }

        /// returns cached information about the number of undirected edges in the mesh
        /// Generated from method `MR::ObjectVoxels::numUndirectedEdges`.
        public unsafe ulong NumUndirectedEdges()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numUndirectedEdges", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numUndirectedEdges(_Underlying *_this);
            return __MR_ObjectVoxels_numUndirectedEdges(_UnderlyingPtr);
        }

        /// returns cached information about the number of holes in the mesh
        /// Generated from method `MR::ObjectVoxels::numHoles`.
        public unsafe ulong NumHoles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numHoles", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numHoles(_Underlying *_this);
            return __MR_ObjectVoxels_numHoles(_UnderlyingPtr);
        }

        /// returns cached information about the number of components in the mesh
        /// Generated from method `MR::ObjectVoxels::numComponents`.
        public unsafe ulong NumComponents()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numComponents", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numComponents(_Underlying *_this);
            return __MR_ObjectVoxels_numComponents(_UnderlyingPtr);
        }

        /// returns cached information about the number of handles in the mesh
        /// Generated from method `MR::ObjectVoxels::numHandles`.
        public unsafe ulong NumHandles()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_numHandles", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_numHandles(_Underlying *_this);
            return __MR_ObjectVoxels_numHandles(_UnderlyingPtr);
        }

        /// returns overriden file extension used to serialize mesh inside this object if set, or defaultSerializeMeshFormat().c_str() otherwise; never returns nullptr
        /// Generated from method `MR::ObjectVoxels::actualSerializeFormat`.
        public unsafe byte? ActualSerializeFormat()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_actualSerializeFormat", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_actualSerializeFormat(_Underlying *_this);
            var __ret = __MR_ObjectVoxels_actualSerializeFormat(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /// Generated from method `MR::ObjectVoxels::getModelHash`.
        public unsafe ulong GetModelHash()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getModelHash", ExactSpelling = true)]
            extern static ulong __MR_ObjectVoxels_getModelHash(_Underlying *_this);
            return __MR_ObjectVoxels_getModelHash(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::sameModels`.
        public unsafe bool SameModels(MR.Const_Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_sameModels", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_sameModels(_Underlying *_this, MR.Const_Object._Underlying *other);
            return __MR_ObjectVoxels_sameModels(_UnderlyingPtr, other._UnderlyingPtr) != 0;
        }

        /// returns true if the property is set at least in one viewport specified by the mask
        /// Generated from method `MR::ObjectVoxels::getVisualizeProperty`.
        public unsafe bool GetVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getVisualizeProperty", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_getVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_getVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr) != 0;
        }

        /// returns all viewports where this object or any of its parents is clipped by plane
        /// Generated from method `MR::ObjectVoxels::globalClippedByPlaneMask`.
        public unsafe MR.ViewportMask GlobalClippedByPlaneMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_globalClippedByPlaneMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectVoxels_globalClippedByPlaneMask(_Underlying *_this);
            return new(__MR_ObjectVoxels_globalClippedByPlaneMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object or any of its parents is clipped by plane in any of given viewports
        /// Generated from method `MR::ObjectVoxels::globalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalClippedByPlane(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_globalClippedByPlane", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_globalClippedByPlane(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_globalClippedByPlane(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectVoxels::getFrontColor`.
        /// Parameter `selected` defaults to `true`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetFrontColor(bool? selected = null, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getFrontColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getFrontColor(_Underlying *_this, byte *selected, MR.ViewportId *viewportId);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectVoxels_getFrontColor(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectVoxels::getFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe MR.Const_ViewportProperty_MRColor GetFrontColorsForAllViewports(bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getFrontColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getFrontColorsForAllViewports(_Underlying *_this, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            return new(__MR_ObjectVoxels_getFrontColorsForAllViewports(_UnderlyingPtr, selected.HasValue ? &__deref_selected : null), is_owning: false);
        }

        /// returns backward color of object in all viewports
        /// Generated from method `MR::ObjectVoxels::getBackColorsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRColor GetBackColorsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getBackColorsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRColor._Underlying *__MR_ObjectVoxels_getBackColorsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getBackColorsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns backward color of object in given viewport
        /// Generated from method `MR::ObjectVoxels::getBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe MR.Const_Color GetBackColor(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getBackColor", ExactSpelling = true)]
            extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_getBackColor(_Underlying *_this, MR.ViewportId *viewportId);
            return new(__MR_ObjectVoxels_getBackColor(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null), is_owning: false);
        }

        /// returns global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectVoxels::getGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe byte GetGlobalAlpha(MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getGlobalAlpha", ExactSpelling = true)]
            extern static byte *__MR_ObjectVoxels_getGlobalAlpha(_Underlying *_this, MR.ViewportId *viewportId);
            return *__MR_ObjectVoxels_getGlobalAlpha(_UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// returns global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectVoxels::getGlobalAlphaForAllViewports`.
        public unsafe MR.Const_ViewportProperty_UnsignedChar GetGlobalAlphaForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_UnsignedChar._Underlying *__MR_ObjectVoxels_getGlobalAlphaForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_getGlobalAlphaForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// returns current dirty flags for the object
        /// Generated from method `MR::ObjectVoxels::getDirtyFlags`.
        public unsafe uint GetDirtyFlags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getDirtyFlags", ExactSpelling = true)]
            extern static uint __MR_ObjectVoxels_getDirtyFlags(_Underlying *_this);
            return __MR_ObjectVoxels_getDirtyFlags(_UnderlyingPtr);
        }

        /// resets all dirty flags (except for cache flags that will be reset automatically on cache update)
        /// Generated from method `MR::ObjectVoxels::resetDirty`.
        public unsafe void ResetDirty()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_resetDirty", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_resetDirty(_Underlying *_this);
            __MR_ObjectVoxels_resetDirty(_UnderlyingPtr);
        }

        /// reset dirty flags without some specific bits (useful for lazy normals update)
        /// Generated from method `MR::ObjectVoxels::resetDirtyExceptMask`.
        public unsafe void ResetDirtyExceptMask(uint mask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_resetDirtyExceptMask", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_resetDirtyExceptMask(_Underlying *_this, uint mask);
            __MR_ObjectVoxels_resetDirtyExceptMask(_UnderlyingPtr, mask);
        }

        /// returns cached bounding box of this object in local coordinates
        /// Generated from method `MR::ObjectVoxels::getBoundingBox`.
        public unsafe MR.Box3f GetBoundingBox()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getBoundingBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectVoxels_getBoundingBox(_Underlying *_this);
            return __MR_ObjectVoxels_getBoundingBox(_UnderlyingPtr);
        }

        /// whether the object can be picked (by mouse) in any of given viewports
        /// Generated from method `MR::ObjectVoxels::isPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsPickable(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isPickable", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isPickable(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_isPickable(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// returns the current coloring mode of the object
        /// Generated from method `MR::ObjectVoxels::getColoringType`.
        public unsafe MR.ColoringType GetColoringType()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getColoringType", ExactSpelling = true)]
            extern static MR.ColoringType __MR_ObjectVoxels_getColoringType(_Underlying *_this);
            return __MR_ObjectVoxels_getColoringType(_UnderlyingPtr);
        }

        /// returns the current shininess visual value
        /// Generated from method `MR::ObjectVoxels::getShininess`.
        public unsafe float GetShininess()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getShininess", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_getShininess(_Underlying *_this);
            return __MR_ObjectVoxels_getShininess(_UnderlyingPtr);
        }

        /// returns intensity of reflections
        /// Generated from method `MR::ObjectVoxels::getSpecularStrength`.
        public unsafe float GetSpecularStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSpecularStrength", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_getSpecularStrength(_Underlying *_this);
            return __MR_ObjectVoxels_getSpecularStrength(_UnderlyingPtr);
        }

        /// returns intensity of non-directional light
        /// Generated from method `MR::ObjectVoxels::getAmbientStrength`.
        public unsafe float GetAmbientStrength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getAmbientStrength", ExactSpelling = true)]
            extern static float __MR_ObjectVoxels_getAmbientStrength(_Underlying *_this);
            return __MR_ObjectVoxels_getAmbientStrength(_UnderlyingPtr);
        }

        /// draws this object for visualization
        /// Returns true if something was drawn.
        /// Generated from method `MR::ObjectVoxels::render`.
        public unsafe bool Render(MR.Const_ModelRenderParams _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_render", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_render(_Underlying *_this, MR.Const_ModelRenderParams._Underlying *_1);
            return __MR_ObjectVoxels_render(_UnderlyingPtr, _1._UnderlyingPtr) != 0;
        }

        /// draws this object for picking
        /// Generated from method `MR::ObjectVoxels::renderForPicker`.
        public unsafe void RenderForPicker(MR.Const_ModelBaseRenderParams _1, uint _2)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_renderForPicker", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_renderForPicker(_Underlying *_this, MR.Const_ModelBaseRenderParams._Underlying *_1, uint _2);
            __MR_ObjectVoxels_renderForPicker(_UnderlyingPtr, _1._UnderlyingPtr, _2);
        }

        /// draws this object for 2d UI
        /// Generated from method `MR::ObjectVoxels::renderUi`.
        public unsafe void RenderUi(MR.Const_UiRenderParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_renderUi", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_renderUi(_Underlying *_this, MR.Const_UiRenderParams._Underlying *params_);
            __MR_ObjectVoxels_renderUi(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectVoxels::useDefaultScenePropertiesOnDeserialization`.
        public unsafe bool UseDefaultScenePropertiesOnDeserialization()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_useDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_useDefaultScenePropertiesOnDeserialization(_Underlying *_this);
            return __MR_ObjectVoxels_useDefaultScenePropertiesOnDeserialization(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectVoxels::name`.
        public unsafe MR.Std.Const_String Name()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_name", ExactSpelling = true)]
            extern static MR.Std.Const_String._Underlying *__MR_ObjectVoxels_name(_Underlying *_this);
            return new(__MR_ObjectVoxels_name(_UnderlyingPtr), is_owning: false);
        }

        /// this space to parent space transformation (to world space if no parent) for default or given viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
        /// Generated from method `MR::ObjectVoxels::xf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.Const_AffineXf3f Xf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_xf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_ObjectVoxels_xf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectVoxels_xf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return new(__ret, is_owning: false);
        }

        /// returns xfs for all viewports, combined into a single object
        /// Generated from method `MR::ObjectVoxels::xfsForAllViewports`.
        public unsafe MR.Const_ViewportProperty_MRAffineXf3f XfsForAllViewports()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_xfsForAllViewports", ExactSpelling = true)]
            extern static MR.Const_ViewportProperty_MRAffineXf3f._Underlying *__MR_ObjectVoxels_xfsForAllViewports(_Underlying *_this);
            return new(__MR_ObjectVoxels_xfsForAllViewports(_UnderlyingPtr), is_owning: false);
        }

        /// this space to world space transformation for default or specific viewport
        /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
        /// Generated from method `MR::ObjectVoxels::worldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe MR.AffineXf3f WorldXf(MR._InOpt_ViewportId id = default, MR.Misc.InOut<bool>? isDef = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_worldXf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ObjectVoxels_worldXf(_Underlying *_this, MR.ViewportId *id, bool *isDef);
            bool __value_isDef = isDef is not null ? isDef.Value : default(bool);
            var __ret = __MR_ObjectVoxels_worldXf(_UnderlyingPtr, id.HasValue ? &id.Object : null, isDef is not null ? &__value_isDef : null);
            if (isDef is not null) isDef.Value = __value_isDef;
            return __ret;
        }

        /// returns all viewports where this object is visible together with all its parents
        /// Generated from method `MR::ObjectVoxels::globalVisibilityMask`.
        public unsafe MR.ViewportMask GlobalVisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_globalVisibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectVoxels_globalVisibilityMask(_Underlying *_this);
            return new(__MR_ObjectVoxels_globalVisibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// returns true if this object is visible together with all its parents in any of given viewports
        /// Generated from method `MR::ObjectVoxels::globalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool GlobalVisibility(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_globalVisibility", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_globalVisibility(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_globalVisibility(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// object properties lock for UI
        /// Generated from method `MR::ObjectVoxels::isLocked`.
        public unsafe bool IsLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isLocked(_Underlying *_this);
            return __MR_ObjectVoxels_isLocked(_UnderlyingPtr) != 0;
        }

        /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
        /// Defaults to false.
        /// Generated from method `MR::ObjectVoxels::isParentLocked`.
        public unsafe bool IsParentLocked()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isParentLocked", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isParentLocked(_Underlying *_this);
            return __MR_ObjectVoxels_isParentLocked(_UnderlyingPtr) != 0;
        }

        /// return true if given object is ancestor of this one, false otherwise
        /// Generated from method `MR::ObjectVoxels::isAncestor`.
        public unsafe bool IsAncestor(MR.Const_Object? ancestor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isAncestor", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isAncestor(_Underlying *_this, MR.Const_Object._Underlying *ancestor);
            return __MR_ObjectVoxels_isAncestor(_UnderlyingPtr, ancestor is not null ? ancestor._UnderlyingPtr : null) != 0;
        }

        /// Generated from method `MR::ObjectVoxels::isSelected`.
        public unsafe bool IsSelected()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isSelected", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isSelected(_Underlying *_this);
            return __MR_ObjectVoxels_isSelected(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::ObjectVoxels::isAncillary`.
        public unsafe bool IsAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isAncillary(_Underlying *_this);
            return __MR_ObjectVoxels_isAncillary(_UnderlyingPtr) != 0;
        }

        /// returns true if the object or any of its ancestors are ancillary
        /// Generated from method `MR::ObjectVoxels::isGlobalAncillary`.
        public unsafe bool IsGlobalAncillary()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isGlobalAncillary", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isGlobalAncillary(_Underlying *_this);
            return __MR_ObjectVoxels_isGlobalAncillary(_UnderlyingPtr) != 0;
        }

        /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
        /// Generated from method `MR::ObjectVoxels::isVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe bool IsVisible(MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_isVisible", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_isVisible(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            return __MR_ObjectVoxels_isVisible(_UnderlyingPtr, viewportMask is not null ? viewportMask._UnderlyingPtr : null) != 0;
        }

        /// gets object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectVoxels::visibilityMask`.
        public unsafe MR.ViewportMask VisibilityMask()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_visibilityMask", ExactSpelling = true)]
            extern static MR.ViewportMask._Underlying *__MR_ObjectVoxels_visibilityMask(_Underlying *_this);
            return new(__MR_ObjectVoxels_visibilityMask(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::ObjectVoxels::resetRedrawFlag`.
        public unsafe void ResetRedrawFlag()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_resetRedrawFlag", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_resetRedrawFlag(_Underlying *_this);
            __MR_ObjectVoxels_resetRedrawFlag(_UnderlyingPtr);
        }

        /// clones all tree of this object (except ancillary and unrecognized children)
        /// Generated from method `MR::ObjectVoxels::cloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> CloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_cloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectVoxels_cloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectVoxels_cloneTree(_UnderlyingPtr), is_owning: true));
        }

        /// clones all tree of this object (except ancillary and unrecognied children)
        /// clones only pointers to mesh, points or voxels
        /// Generated from method `MR::ObjectVoxels::shallowCloneTree`.
        public unsafe MR.Misc._Moved<MR.Object> ShallowCloneTree()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_shallowCloneTree", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectVoxels_shallowCloneTree(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectVoxels_shallowCloneTree(_UnderlyingPtr), is_owning: true));
        }

        ///empty box
        /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
        /// Generated from method `MR::ObjectVoxels::getWorldTreeBox`.
        /// Parameter `_1` defaults to `{}`.
        public unsafe MR.Box3f GetWorldTreeBox(MR._InOpt_ViewportId _1 = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getWorldTreeBox", ExactSpelling = true)]
            extern static MR.Box3f __MR_ObjectVoxels_getWorldTreeBox(_Underlying *_this, MR.ViewportId *_1);
            return __MR_ObjectVoxels_getWorldTreeBox(_UnderlyingPtr, _1.HasValue ? &_1.Object : null);
        }

        /// provides read-only access to the tag storage
        /// the storage is a set of unique strings
        /// Generated from method `MR::ObjectVoxels::tags`.
        public unsafe MR.Std.Const_Set_StdString Tags()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_tags", ExactSpelling = true)]
            extern static MR.Std.Const_Set_StdString._Underlying *__MR_ObjectVoxels_tags(_Underlying *_this);
            return new(__MR_ObjectVoxels_tags(_UnderlyingPtr), is_owning: false);
        }

        // returns this Object as shared_ptr
        // finds it among its parent's recognized children
        /// Generated from method `MR::ObjectVoxels::getSharedPtr`.
        public unsafe MR.Misc._Moved<MR.Object> GetSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_getSharedPtr", ExactSpelling = true)]
            extern static MR.Object._UnderlyingShared *__MR_ObjectVoxels_getSharedPtr(_Underlying *_this);
            return MR.Misc.Move(new MR.Object(__MR_ObjectVoxels_getSharedPtr(_UnderlyingPtr), is_owning: true));
        }

        // struct to control volume rendering texture
        /// Generated from class `MR::ObjectVoxels::VolumeRenderingParams`.
        /// This is the const half of the class.
        public class Const_VolumeRenderingParams : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.ObjectVoxels.Const_VolumeRenderingParams>
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_VolumeRenderingParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Destroy", ExactSpelling = true)]
                extern static void __MR_ObjectVoxels_VolumeRenderingParams_Destroy(_Underlying *_this);
                __MR_ObjectVoxels_VolumeRenderingParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_VolumeRenderingParams() {Dispose(false);}

            // volume texture smoothing
            public unsafe MR.FilterType VolumeFilterType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_volumeFilterType", ExactSpelling = true)]
                    extern static MR.FilterType *__MR_ObjectVoxels_VolumeRenderingParams_Get_volumeFilterType(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_volumeFilterType(_UnderlyingPtr);
                }
            }

            public unsafe MR.ObjectVoxels.VolumeRenderingParams.ShadingType ShadingType_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_shadingType", ExactSpelling = true)]
                    extern static MR.ObjectVoxels.VolumeRenderingParams.ShadingType *__MR_ObjectVoxels_VolumeRenderingParams_Get_shadingType(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_shadingType(_UnderlyingPtr);
                }
            }

            public unsafe MR.ObjectVoxels.VolumeRenderingParams.LutType LutType_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_lutType", ExactSpelling = true)]
                    extern static MR.ObjectVoxels.VolumeRenderingParams.LutType *__MR_ObjectVoxels_VolumeRenderingParams_Get_lutType(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_lutType(_UnderlyingPtr);
                }
            }

            // color that is used for OneColor mode
            public unsafe MR.Const_Color OneColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_oneColor", ExactSpelling = true)]
                    extern static MR.Const_Color._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_Get_oneColor(_Underlying *_this);
                    return new(__MR_ObjectVoxels_VolumeRenderingParams_Get_oneColor(_UnderlyingPtr), is_owning: false);
                }
            }

            // minimum colored value (voxels with lower values are transparent)
            public unsafe float Min
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_min", ExactSpelling = true)]
                    extern static float *__MR_ObjectVoxels_VolumeRenderingParams_Get_min(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_min(_UnderlyingPtr);
                }
            }

            // maximum colored value (voxels with higher values are transparent)
            public unsafe float Max
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_max", ExactSpelling = true)]
                    extern static float *__MR_ObjectVoxels_VolumeRenderingParams_Get_max(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_max(_UnderlyingPtr);
                }
            }

            // step to sample each ray with
            // if <= 0 then default sampling is used
            public unsafe float SamplingStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_samplingStep", ExactSpelling = true)]
                    extern static float *__MR_ObjectVoxels_VolumeRenderingParams_Get_samplingStep(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_samplingStep(_UnderlyingPtr);
                }
            }

            public unsafe MR.ObjectVoxels.VolumeRenderingParams.AlphaType AlphaType_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_alphaType", ExactSpelling = true)]
                    extern static MR.ObjectVoxels.VolumeRenderingParams.AlphaType *__MR_ObjectVoxels_VolumeRenderingParams_Get_alphaType(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_alphaType(_UnderlyingPtr);
                }
            }

            public unsafe byte AlphaLimit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_Get_alphaLimit", ExactSpelling = true)]
                    extern static byte *__MR_ObjectVoxels_VolumeRenderingParams_Get_alphaLimit(_Underlying *_this);
                    return *__MR_ObjectVoxels_VolumeRenderingParams_Get_alphaLimit(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_VolumeRenderingParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectVoxels_VolumeRenderingParams_DefaultConstruct();
            }

            /// Constructs `MR::ObjectVoxels::VolumeRenderingParams` elementwise.
            public unsafe Const_VolumeRenderingParams(MR.FilterType volumeFilterType, MR.ObjectVoxels.VolumeRenderingParams.ShadingType shadingType, MR.ObjectVoxels.VolumeRenderingParams.LutType lutType, MR.Color oneColor, float min, float max, float samplingStep, MR.ObjectVoxels.VolumeRenderingParams.AlphaType alphaType, byte alphaLimit) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_ConstructFrom(MR.FilterType volumeFilterType, MR.ObjectVoxels.VolumeRenderingParams.ShadingType shadingType, MR.ObjectVoxels.VolumeRenderingParams.LutType lutType, MR.Color oneColor, float min, float max, float samplingStep, MR.ObjectVoxels.VolumeRenderingParams.AlphaType alphaType, byte alphaLimit);
                _UnderlyingPtr = __MR_ObjectVoxels_VolumeRenderingParams_ConstructFrom(volumeFilterType, shadingType, lutType, oneColor, min, max, samplingStep, alphaType, alphaLimit);
            }

            /// Generated from constructor `MR::ObjectVoxels::VolumeRenderingParams::VolumeRenderingParams`.
            public unsafe Const_VolumeRenderingParams(MR.ObjectVoxels.Const_VolumeRenderingParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_ConstructFromAnother(MR.ObjectVoxels.VolumeRenderingParams._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectVoxels_VolumeRenderingParams_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::ObjectVoxels::VolumeRenderingParams::operator==`.
            public static unsafe bool operator==(MR.ObjectVoxels.Const_VolumeRenderingParams _this, MR.ObjectVoxels.Const_VolumeRenderingParams _1)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_ObjectVoxels_VolumeRenderingParams", ExactSpelling = true)]
                extern static byte __MR_equal_MR_ObjectVoxels_VolumeRenderingParams(MR.ObjectVoxels.Const_VolumeRenderingParams._Underlying *_this, MR.ObjectVoxels.Const_VolumeRenderingParams._Underlying *_1);
                return __MR_equal_MR_ObjectVoxels_VolumeRenderingParams(_this._UnderlyingPtr, _1._UnderlyingPtr) != 0;
            }

            public static unsafe bool operator!=(MR.ObjectVoxels.Const_VolumeRenderingParams _this, MR.ObjectVoxels.Const_VolumeRenderingParams _1)
            {
                return !(_this == _1);
            }

            // type of alpha function on texture
            public enum AlphaType : int
            {
                Constant = 0,
                LinearIncreasing = 1,
                LinearDecreasing = 2,
            }

            // coloring type
            public enum LutType : int
            {
                GrayShades = 0,
                Rainbow = 1,
                OneColor = 2,
            }

            // shading model
            public enum ShadingType : int
            {
                None = 0,
                ValueGradient = 1,
                AlphaGradient = 2,
            }

            // IEquatable:

            public bool Equals(MR.ObjectVoxels.Const_VolumeRenderingParams? _1)
            {
                if (_1 is null)
                    return false;
                return this == _1;
            }

            public override bool Equals(object? other)
            {
                if (other is null)
                    return false;
                if (other is MR.ObjectVoxels.Const_VolumeRenderingParams)
                    return this == (MR.ObjectVoxels.Const_VolumeRenderingParams)other;
                return false;
            }
        }

        // struct to control volume rendering texture
        /// Generated from class `MR::ObjectVoxels::VolumeRenderingParams`.
        /// This is the non-const half of the class.
        public class VolumeRenderingParams : Const_VolumeRenderingParams
        {
            internal unsafe VolumeRenderingParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // volume texture smoothing
            public new unsafe ref MR.FilterType VolumeFilterType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_volumeFilterType", ExactSpelling = true)]
                    extern static MR.FilterType *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_volumeFilterType(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_volumeFilterType(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.ObjectVoxels.VolumeRenderingParams.ShadingType ShadingType_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_shadingType", ExactSpelling = true)]
                    extern static MR.ObjectVoxels.VolumeRenderingParams.ShadingType *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_shadingType(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_shadingType(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.ObjectVoxels.VolumeRenderingParams.LutType LutType_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_lutType", ExactSpelling = true)]
                    extern static MR.ObjectVoxels.VolumeRenderingParams.LutType *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_lutType(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_lutType(_UnderlyingPtr);
                }
            }

            // color that is used for OneColor mode
            public new unsafe MR.Mut_Color OneColor
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_oneColor", ExactSpelling = true)]
                    extern static MR.Mut_Color._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_oneColor(_Underlying *_this);
                    return new(__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_oneColor(_UnderlyingPtr), is_owning: false);
                }
            }

            // minimum colored value (voxels with lower values are transparent)
            public new unsafe ref float Min
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_min", ExactSpelling = true)]
                    extern static float *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_min(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_min(_UnderlyingPtr);
                }
            }

            // maximum colored value (voxels with higher values are transparent)
            public new unsafe ref float Max
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_max", ExactSpelling = true)]
                    extern static float *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_max(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_max(_UnderlyingPtr);
                }
            }

            // step to sample each ray with
            // if <= 0 then default sampling is used
            public new unsafe ref float SamplingStep
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_samplingStep", ExactSpelling = true)]
                    extern static float *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_samplingStep(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_samplingStep(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.ObjectVoxels.VolumeRenderingParams.AlphaType AlphaType_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_alphaType", ExactSpelling = true)]
                    extern static MR.ObjectVoxels.VolumeRenderingParams.AlphaType *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_alphaType(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_alphaType(_UnderlyingPtr);
                }
            }

            public new unsafe ref byte AlphaLimit
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_GetMutable_alphaLimit", ExactSpelling = true)]
                    extern static byte *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_alphaLimit(_Underlying *_this);
                    return ref *__MR_ObjectVoxels_VolumeRenderingParams_GetMutable_alphaLimit(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe VolumeRenderingParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_DefaultConstruct();
                _UnderlyingPtr = __MR_ObjectVoxels_VolumeRenderingParams_DefaultConstruct();
            }

            /// Constructs `MR::ObjectVoxels::VolumeRenderingParams` elementwise.
            public unsafe VolumeRenderingParams(MR.FilterType volumeFilterType, MR.ObjectVoxels.VolumeRenderingParams.ShadingType shadingType, MR.ObjectVoxels.VolumeRenderingParams.LutType lutType, MR.Color oneColor, float min, float max, float samplingStep, MR.ObjectVoxels.VolumeRenderingParams.AlphaType alphaType, byte alphaLimit) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_ConstructFrom(MR.FilterType volumeFilterType, MR.ObjectVoxels.VolumeRenderingParams.ShadingType shadingType, MR.ObjectVoxels.VolumeRenderingParams.LutType lutType, MR.Color oneColor, float min, float max, float samplingStep, MR.ObjectVoxels.VolumeRenderingParams.AlphaType alphaType, byte alphaLimit);
                _UnderlyingPtr = __MR_ObjectVoxels_VolumeRenderingParams_ConstructFrom(volumeFilterType, shadingType, lutType, oneColor, min, max, samplingStep, alphaType, alphaLimit);
            }

            /// Generated from constructor `MR::ObjectVoxels::VolumeRenderingParams::VolumeRenderingParams`.
            public unsafe VolumeRenderingParams(MR.ObjectVoxels.Const_VolumeRenderingParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_ConstructFromAnother(MR.ObjectVoxels.VolumeRenderingParams._Underlying *_other);
                _UnderlyingPtr = __MR_ObjectVoxels_VolumeRenderingParams_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::ObjectVoxels::VolumeRenderingParams::operator=`.
            public unsafe MR.ObjectVoxels.VolumeRenderingParams Assign(MR.ObjectVoxels.Const_VolumeRenderingParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_VolumeRenderingParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ObjectVoxels.VolumeRenderingParams._Underlying *__MR_ObjectVoxels_VolumeRenderingParams_AssignFromAnother(_Underlying *_this, MR.ObjectVoxels.VolumeRenderingParams._Underlying *_other);
                return new(__MR_ObjectVoxels_VolumeRenderingParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `VolumeRenderingParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_VolumeRenderingParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VolumeRenderingParams`/`Const_VolumeRenderingParams` directly.
        public class _InOptMut_VolumeRenderingParams
        {
            public VolumeRenderingParams? Opt;

            public _InOptMut_VolumeRenderingParams() {}
            public _InOptMut_VolumeRenderingParams(VolumeRenderingParams value) {Opt = value;}
            public static implicit operator _InOptMut_VolumeRenderingParams(VolumeRenderingParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `VolumeRenderingParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_VolumeRenderingParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VolumeRenderingParams`/`Const_VolumeRenderingParams` to pass it to the function.
        public class _InOptConst_VolumeRenderingParams
        {
            public Const_VolumeRenderingParams? Opt;

            public _InOptConst_VolumeRenderingParams() {}
            public _InOptConst_VolumeRenderingParams(Const_VolumeRenderingParams value) {Opt = value;}
            public static implicit operator _InOptConst_VolumeRenderingParams(Const_VolumeRenderingParams value) {return new(value);}
        }
    }

    /// This class stores information about voxels object
    /// Generated from class `MR::ObjectVoxels`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::ObjectMeshHolder`
    ///   Indirect: (non-virtual)
    ///     `MR::ObjectChildrenHolder`
    ///     `MR::Object`
    ///     `MR::VisualObject`
    /// This is the non-const half of the class.
    public class ObjectVoxels : Const_ObjectVoxels
    {
        internal unsafe ObjectVoxels(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe ObjectVoxels(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.ObjectChildrenHolder(ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_ObjectChildrenHolder", ExactSpelling = true)]
            extern static MR.ObjectChildrenHolder._Underlying *__MR_ObjectVoxels_UpcastTo_MR_ObjectChildrenHolder(_Underlying *_this);
            return MR.ObjectChildrenHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_ObjectChildrenHolder(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.Object(ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_Object", ExactSpelling = true)]
            extern static MR.Object._Underlying *__MR_ObjectVoxels_UpcastTo_MR_Object(_Underlying *_this);
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_Object(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.VisualObject(ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_VisualObject", ExactSpelling = true)]
            extern static MR.VisualObject._Underlying *__MR_ObjectVoxels_UpcastTo_MR_VisualObject(_Underlying *_this);
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_VisualObject(self._UnderlyingPtr));
        }
        public static unsafe implicit operator MR.ObjectMeshHolder(ObjectVoxels self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_UpcastTo_MR_ObjectMeshHolder", ExactSpelling = true)]
            extern static MR.ObjectMeshHolder._Underlying *__MR_ObjectVoxels_UpcastTo_MR_ObjectMeshHolder(_Underlying *_this);
            return MR.ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_ObjectVoxels_UpcastTo_MR_ObjectMeshHolder(self._UnderlyingPtr));
        }

        // Downcasts:
        public static unsafe explicit operator ObjectVoxels?(MR.Object parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Object_DynamicDowncastTo_MR_ObjectVoxels", ExactSpelling = true)]
            extern static _Underlying *__MR_Object_DynamicDowncastTo_MR_ObjectVoxels(MR.Object._Underlying *_this);
            var ptr = __MR_Object_DynamicDowncastTo_MR_ObjectVoxels(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.Object._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectVoxels?(MR.VisualObject parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VisualObject_DynamicDowncastTo_MR_ObjectVoxels", ExactSpelling = true)]
            extern static _Underlying *__MR_VisualObject_DynamicDowncastTo_MR_ObjectVoxels(MR.VisualObject._Underlying *_this);
            var ptr = __MR_VisualObject_DynamicDowncastTo_MR_ObjectVoxels(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.VisualObject._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }
        public static unsafe explicit operator ObjectVoxels?(MR.ObjectMeshHolder parent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectVoxels", ExactSpelling = true)]
            extern static _Underlying *__MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectVoxels(MR.ObjectMeshHolder._Underlying *_this);
            var ptr = __MR_ObjectMeshHolder_DynamicDowncastTo_MR_ObjectVoxels(parent._UnderlyingPtr);
            if (ptr is null) return null;
            return MR.ObjectMeshHolder._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, ptr);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ObjectVoxels() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ObjectVoxels._Underlying *__MR_ObjectVoxels_DefaultConstruct();
            _LateMakeShared(__MR_ObjectVoxels_DefaultConstruct());
        }

        /// Generated from constructor `MR::ObjectVoxels::ObjectVoxels`.
        public unsafe ObjectVoxels(MR._ByValue_ObjectVoxels _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ObjectVoxels._Underlying *__MR_ObjectVoxels_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ObjectVoxels._Underlying *_other);
            _LateMakeShared(__MR_ObjectVoxels_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Generated from method `MR::ObjectVoxels::operator=`.
        public unsafe MR.ObjectVoxels Assign(MR._ByValue_ObjectVoxels _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ObjectVoxels._Underlying *__MR_ObjectVoxels_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ObjectVoxels._Underlying *_other);
            return new(__MR_ObjectVoxels_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::ObjectVoxels::applyScale`.
        public unsafe void ApplyScale(float scaleFactor)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_applyScale", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_applyScale(_Underlying *_this, float scaleFactor);
            __MR_ObjectVoxels_applyScale(_UnderlyingPtr, scaleFactor);
        }

        /// Generated from method `MR::ObjectVoxels::varVdbVolume`.
        public unsafe MR.VdbVolume VarVdbVolume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_varVdbVolume", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_ObjectVoxels_varVdbVolume(_Underlying *_this);
            return new(__MR_ObjectVoxels_varVdbVolume(_UnderlyingPtr), is_owning: false);
        }

        /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
        /// \param normalPlusGrad true means that iso-surface normals will be along gradient, false means opposite direction
        /// \param minmax optional data about known min and max values
        /// set a new background for the VdbVolume, if normalPlusGrad = true, use the maximum value, otherwise the minimum value
        /// Generated from method `MR::ObjectVoxels::construct`.
        /// Parameter `minmax` defaults to `{}`.
        /// Parameter `cb` defaults to `{}`.
        /// Parameter `normalPlusGrad` defaults to `false`.
        public unsafe void Construct(MR.Const_SimpleVolume simpleVolume, MR.Std.Const_Optional_MRVector2f? minmax = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, bool? normalPlusGrad = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_construct_4", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_construct_4(_Underlying *_this, MR.Const_SimpleVolume._Underlying *simpleVolume, MR.Std.Const_Optional_MRVector2f._Underlying *minmax, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, byte *normalPlusGrad);
            byte __deref_normalPlusGrad = normalPlusGrad.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectVoxels_construct_4(_UnderlyingPtr, simpleVolume._UnderlyingPtr, minmax is not null ? minmax._UnderlyingPtr : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, normalPlusGrad.HasValue ? &__deref_normalPlusGrad : null);
        }

        /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
        /// \param normalPlusGrad true means that iso-surface normals will be along gradient, false means opposite direction
        /// set a new background for the VdbVolume, if normalPlusGrad = true, use the maximum value, otherwise the minimum value
        /// Generated from method `MR::ObjectVoxels::construct`.
        /// Parameter `cb` defaults to `{}`.
        /// Parameter `normalPlusGrad` defaults to `false`.
        public unsafe void Construct(MR.Const_SimpleVolumeMinMax simpleVolumeMinMax, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, bool? normalPlusGrad = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_construct_3_MR_SimpleVolumeMinMax", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_construct_3_MR_SimpleVolumeMinMax(_Underlying *_this, MR.Const_SimpleVolumeMinMax._Underlying *simpleVolumeMinMax, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, byte *normalPlusGrad);
            byte __deref_normalPlusGrad = normalPlusGrad.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectVoxels_construct_3_MR_SimpleVolumeMinMax(_UnderlyingPtr, simpleVolumeMinMax._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, normalPlusGrad.HasValue ? &__deref_normalPlusGrad : null);
        }

        /// Clears all internal data and then remembers grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
        /// \param minmax optional data about known min and max values
        /// Generated from method `MR::ObjectVoxels::construct`.
        /// Parameter `minmax` defaults to `{}`.
        public unsafe void Construct(MR.Const_FloatGrid grid, MR.Const_Vector3f voxelSize, MR.Std.Const_Optional_MRVector2f? minmax = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_construct_3_MR_FloatGrid", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_construct_3_MR_FloatGrid(_Underlying *_this, MR.Const_FloatGrid._Underlying *grid, MR.Const_Vector3f._Underlying *voxelSize, MR.Std.Const_Optional_MRVector2f._Underlying *minmax);
            __MR_ObjectVoxels_construct_3_MR_FloatGrid(_UnderlyingPtr, grid._UnderlyingPtr, voxelSize._UnderlyingPtr, minmax is not null ? minmax._UnderlyingPtr : null);
        }

        /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
        /// Generated from method `MR::ObjectVoxels::construct`.
        public unsafe void Construct(MR.Const_VdbVolume vdbVolume)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_construct_1", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_construct_1(_Underlying *_this, MR.Const_VdbVolume._Underlying *vdbVolume);
            __MR_ObjectVoxels_construct_1(_UnderlyingPtr, vdbVolume._UnderlyingPtr);
        }

        /// Updates histogram, by stored grid (evals min and max values from grid)
        /// rebuild iso surface if it is present
        /// Generated from method `MR::ObjectVoxels::updateHistogramAndSurface`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe void UpdateHistogramAndSurface(MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateHistogramAndSurface", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateHistogramAndSurface(_Underlying *_this, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            __MR_ObjectVoxels_updateHistogramAndSurface(_UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// Sets iso value and updates iso-surfaces if needed:
        /// Returns true if iso-value was updated, false - otherwise
        /// Generated from method `MR::ObjectVoxels::setIsoValue`.
        /// Parameter `cb` defaults to `{}`.
        /// Parameter `updateSurface` defaults to `true`.
        public unsafe MR.Misc._Moved<MR.Expected_Bool_StdString> SetIsoValue(float iso, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, bool? updateSurface = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setIsoValue", ExactSpelling = true)]
            extern static MR.Expected_Bool_StdString._Underlying *__MR_ObjectVoxels_setIsoValue(_Underlying *_this, float iso, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, byte *updateSurface);
            byte __deref_updateSurface = updateSurface.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.Expected_Bool_StdString(__MR_ObjectVoxels_setIsoValue(_UnderlyingPtr, iso, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, updateSurface.HasValue ? &__deref_updateSurface : null), is_owning: true));
        }

        /// Sets external surface mesh for this object
        /// and returns back previous mesh of this
        /// Generated from method `MR::ObjectVoxels::updateIsoSurface`.
        public unsafe MR.Misc._Moved<MR.Mesh> UpdateIsoSurface(MR._ByValue_Mesh mesh)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateIsoSurface", ExactSpelling = true)]
            extern static MR.Mesh._UnderlyingShared *__MR_ObjectVoxels_updateIsoSurface(_Underlying *_this, MR.Misc._PassBy mesh_pass_by, MR.Mesh._UnderlyingShared *mesh);
            return MR.Misc.Move(new MR.Mesh(__MR_ObjectVoxels_updateIsoSurface(_UnderlyingPtr, mesh.PassByMode, mesh.Value is not null ? mesh.Value._UnderlyingSharedPtr : null), is_owning: true));
        }

        /// Sets external vdb volume for this object
        /// and returns back previous vdb volume of this
        /// Generated from method `MR::ObjectVoxels::updateVdbVolume`.
        public unsafe MR.Misc._Moved<MR.VdbVolume> UpdateVdbVolume(MR._ByValue_VdbVolume vdbVolume)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateVdbVolume", ExactSpelling = true)]
            extern static MR.VdbVolume._Underlying *__MR_ObjectVoxels_updateVdbVolume(_Underlying *_this, MR.Misc._PassBy vdbVolume_pass_by, MR.VdbVolume._Underlying *vdbVolume);
            return MR.Misc.Move(new MR.VdbVolume(__MR_ObjectVoxels_updateVdbVolume(_UnderlyingPtr, vdbVolume.PassByMode, vdbVolume.Value is not null ? vdbVolume.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// Sets external histogram for this object
        /// and returns back previous histogram of this
        /// Generated from method `MR::ObjectVoxels::updateHistogram`.
        public unsafe MR.Misc._Moved<MR.Histogram> UpdateHistogram(MR._ByValue_Histogram histogram)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateHistogram", ExactSpelling = true)]
            extern static MR.Histogram._Underlying *__MR_ObjectVoxels_updateHistogram(_Underlying *_this, MR.Misc._PassBy histogram_pass_by, MR.Histogram._Underlying *histogram);
            return MR.Misc.Move(new MR.Histogram(__MR_ObjectVoxels_updateHistogram(_UnderlyingPtr, histogram.PassByMode, histogram.Value is not null ? histogram.Value._UnderlyingPtr : null), is_owning: true));
        }

        /// sets whether to use Dual Marching Cubes algorithm for visualization (true) or Standard Marching Cubes (false);
        /// \param updateSurface forces immediate update
        /// Generated from method `MR::ObjectVoxels::setDualMarchingCubes`.
        /// Parameter `updateSurface` defaults to `true`.
        /// Parameter `cb` defaults to `{}`.
        public unsafe void SetDualMarchingCubes(bool on, bool? updateSurface = null, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setDualMarchingCubes", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setDualMarchingCubes(_Underlying *_this, byte on, byte *updateSurface, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            byte __deref_updateSurface = updateSurface.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectVoxels_setDualMarchingCubes(_UnderlyingPtr, on ? (byte)1 : (byte)0, updateSurface.HasValue ? &__deref_updateSurface : null, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null);
        }

        /// set voxel point positioner for Marching Cubes (only for Standard Marching Cubes)
        /// Generated from method `MR::ObjectVoxels::setVoxelPointPositioner`.
        public unsafe void SetVoxelPointPositioner(MR.Std._ByValue_Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat positioner)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVoxelPointPositioner", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVoxelPointPositioner(_Underlying *_this, MR.Misc._PassBy positioner_pass_by, MR.Std.Function_MRVector3fFuncFromConstMRVector3fRefConstMRVector3fRefFloatFloatFloat._Underlying *positioner);
            __MR_ObjectVoxels_setVoxelPointPositioner(_UnderlyingPtr, positioner.PassByMode, positioner.Value is not null ? positioner.Value._UnderlyingPtr : null);
        }

        /// Sets active bounds for some simplifications (max excluded)
        /// active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty
        /// NOTE: don't forget to call `invalidateActiveBoundsCaches` if you call this function from progress bar thread
        /// Generated from method `MR::ObjectVoxels::setActiveBounds`.
        /// Parameter `cb` defaults to `{}`.
        /// Parameter `updateSurface` defaults to `true`.
        public unsafe void SetActiveBounds(MR.Const_Box3i activeBox, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, bool? updateSurface = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setActiveBounds", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setActiveBounds(_Underlying *_this, MR.Const_Box3i._Underlying *activeBox, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, byte *updateSurface);
            byte __deref_updateSurface = updateSurface.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectVoxels_setActiveBounds(_UnderlyingPtr, activeBox._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, updateSurface.HasValue ? &__deref_updateSurface : null);
        }

        /// Call this function in main thread post processing if you call setActiveBounds from progress bar thread
        /// Generated from method `MR::ObjectVoxels::invalidateActiveBoundsCaches`.
        public unsafe void InvalidateActiveBoundsCaches()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_invalidateActiveBoundsCaches", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_invalidateActiveBoundsCaches(_Underlying *_this);
            __MR_ObjectVoxels_invalidateActiveBoundsCaches(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::selectVoxels`.
        public unsafe void SelectVoxels(MR.Const_VoxelBitSet selectedVoxels)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_selectVoxels", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_selectVoxels(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *selectedVoxels);
            __MR_ObjectVoxels_selectVoxels(_UnderlyingPtr, selectedVoxels._UnderlyingPtr);
        }

        /// set active (visible) voxels (using only in Volume Rendering mode)
        /// Generated from method `MR::ObjectVoxels::setVolumeRenderActiveVoxels`.
        public unsafe void SetVolumeRenderActiveVoxels(MR.Const_VoxelBitSet activeVoxels)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVolumeRenderActiveVoxels", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVolumeRenderActiveVoxels(_Underlying *_this, MR.Const_VoxelBitSet._Underlying *activeVoxels);
            __MR_ObjectVoxels_setVolumeRenderActiveVoxels(_UnderlyingPtr, activeVoxels._UnderlyingPtr);
        }

        // this function should only be called from GUI thread because it changes rendering object,
        // it can take some time to prepare data, so you can prepare data with progress callback
        // by calling `prepareDataForVolumeRendering(cb)` function before calling this one
        /// Generated from method `MR::ObjectVoxels::enableVolumeRendering`.
        public unsafe void EnableVolumeRendering(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_enableVolumeRendering", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_enableVolumeRendering(_Underlying *_this, byte on);
            __MR_ObjectVoxels_enableVolumeRendering(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectVoxels::setVolumeRenderingParams`.
        public unsafe void SetVolumeRenderingParams(MR.ObjectVoxels.Const_VolumeRenderingParams params_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVolumeRenderingParams", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVolumeRenderingParams(_Underlying *_this, MR.ObjectVoxels.Const_VolumeRenderingParams._Underlying *params_);
            __MR_ObjectVoxels_setVolumeRenderingParams(_UnderlyingPtr, params_._UnderlyingPtr);
        }

        /// sets top limit on the number of vertices in the iso-surface
        /// Generated from method `MR::ObjectVoxels::setMaxSurfaceVertices`.
        public unsafe void SetMaxSurfaceVertices(int maxVerts)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setMaxSurfaceVertices", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setMaxSurfaceVertices(_Underlying *_this, int maxVerts);
            __MR_ObjectVoxels_setMaxSurfaceVertices(_UnderlyingPtr, maxVerts);
        }

        /// Generated from method `MR::ObjectVoxels::setDirtyFlags`.
        /// Parameter `invalidateCaches` defaults to `true`.
        public unsafe void SetDirtyFlags(uint mask, bool? invalidateCaches = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setDirtyFlags", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setDirtyFlags(_Underlying *_this, uint mask, byte *invalidateCaches);
            byte __deref_invalidateCaches = invalidateCaches.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectVoxels_setDirtyFlags(_UnderlyingPtr, mask, invalidateCaches.HasValue ? &__deref_invalidateCaches : null);
        }

        /// overrides file extension used to serialize voxels inside this object: must start from '.',
        /// nullptr means serialize in defaultSerializeVoxelsFormat()
        /// Generated from method `MR::ObjectVoxels::setSerializeFormat`.
        public unsafe void SetSerializeFormat(byte? newFormat)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setSerializeFormat", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setSerializeFormat(_Underlying *_this, byte *newFormat);
            byte __deref_newFormat = newFormat.GetValueOrDefault();
            __MR_ObjectVoxels_setSerializeFormat(_UnderlyingPtr, newFormat.HasValue ? &__deref_newFormat : null);
        }

        /// reset basic object colors to their default values from the current theme
        /// Generated from method `MR::ObjectVoxels::resetFrontColor`.
        public unsafe void ResetFrontColor()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_resetFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_resetFrontColor(_Underlying *_this);
            __MR_ObjectVoxels_resetFrontColor(_UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::selectFaces`.
        public unsafe void SelectFaces(MR._ByValue_FaceBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_selectFaces", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_selectFaces(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.FaceBitSet._Underlying *newSelection);
            __MR_ObjectVoxels_selectFaces(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// sets colors of selected triangles
        /// Generated from method `MR::ObjectVoxels::setSelectedFacesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedFacesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setSelectedFacesColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setSelectedFacesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectVoxels_setSelectedFacesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectVoxels::selectEdges`.
        public unsafe void SelectEdges(MR._ByValue_UndirectedEdgeBitSet newSelection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_selectEdges", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_selectEdges(_Underlying *_this, MR.Misc._PassBy newSelection_pass_by, MR.UndirectedEdgeBitSet._Underlying *newSelection);
            __MR_ObjectVoxels_selectEdges(_UnderlyingPtr, newSelection.PassByMode, newSelection.Value is not null ? newSelection.Value._UnderlyingPtr : null);
        }

        /// sets colors of selected edges
        /// Generated from method `MR::ObjectVoxels::setSelectedEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetSelectedEdgesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setSelectedEdgesColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setSelectedEdgesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectVoxels_setSelectedEdgesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectVoxels::setSelectedEdgesColorsForAllViewports`.
        public unsafe void SetSelectedEdgesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setSelectedEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setSelectedEdgesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectVoxels_setSelectedEdgesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setSelectedFacesColorsForAllViewports`.
        public unsafe void SetSelectedFacesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setSelectedFacesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setSelectedFacesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectVoxels_setSelectedFacesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setEdgesColorsForAllViewports`.
        public unsafe void SetEdgesColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setEdgesColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setEdgesColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectVoxels_setEdgesColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setPointsColorsForAllViewports`.
        public unsafe void SetPointsColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setPointsColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setPointsColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectVoxels_setPointsColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setBordersColorsForAllViewports`.
        public unsafe void SetBordersColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setBordersColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setBordersColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectVoxels_setBordersColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// set all object solid colors (front/back/etc.) from other object for all viewports
        /// Generated from method `MR::ObjectVoxels::copyAllSolidColors`.
        public unsafe void CopyAllSolidColors(MR.Const_ObjectMeshHolder other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_copyAllSolidColors", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_copyAllSolidColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *other);
            __MR_ObjectVoxels_copyAllSolidColors(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::setCreases`.
        public unsafe void SetCreases(MR._ByValue_UndirectedEdgeBitSet creases)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setCreases", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setCreases(_Underlying *_this, MR.Misc._PassBy creases_pass_by, MR.UndirectedEdgeBitSet._Underlying *creases);
            __MR_ObjectVoxels_setCreases(_UnderlyingPtr, creases.PassByMode, creases.Value is not null ? creases.Value._UnderlyingPtr : null);
        }

        /// sets flat (true) or smooth (false) shading
        /// Generated from method `MR::ObjectVoxels::setFlatShading`.
        public unsafe void SetFlatShading(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setFlatShading", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setFlatShading(_Underlying *_this, byte on);
            __MR_ObjectVoxels_setFlatShading(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// sets whole new ObjectMeshData
        /// Generated from method `MR::ObjectVoxels::setData`.
        public unsafe void SetData(MR.Misc._Moved<MR.ObjectMeshData> data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setData", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setData(_Underlying *_this, MR.ObjectMeshData._Underlying *data);
            __MR_ObjectVoxels_setData(_UnderlyingPtr, data.Value._UnderlyingPtr);
        }

        /// swaps whole ObjectMeshData with given argument
        /// Generated from method `MR::ObjectVoxels::updateData`.
        public unsafe void UpdateData(MR.ObjectMeshData data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateData", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateData(_Underlying *_this, MR.ObjectMeshData._Underlying *data);
            __MR_ObjectVoxels_updateData(_UnderlyingPtr, data._UnderlyingPtr);
        }

        /// sets per-vertex colors of the object
        /// Generated from method `MR::ObjectVoxels::setVertsColorMap`.
        public unsafe void SetVertsColorMap(MR._ByValue_VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVertsColorMap(_Underlying *_this, MR.Misc._PassBy vertsColorMap_pass_by, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectVoxels_setVertsColorMap(_UnderlyingPtr, vertsColorMap.PassByMode, vertsColorMap.Value is not null ? vertsColorMap.Value._UnderlyingPtr : null);
        }

        /// swaps per-vertex colors of the object with given argument
        /// Generated from method `MR::ObjectVoxels::updateVertsColorMap`.
        public unsafe void UpdateVertsColorMap(MR.VertColors vertsColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateVertsColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateVertsColorMap(_Underlying *_this, MR.VertColors._Underlying *vertsColorMap);
            __MR_ObjectVoxels_updateVertsColorMap(_UnderlyingPtr, vertsColorMap._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::setFacesColorMap`.
        public unsafe void SetFacesColorMap(MR._ByValue_FaceColors facesColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setFacesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setFacesColorMap(_Underlying *_this, MR.Misc._PassBy facesColorMap_pass_by, MR.FaceColors._Underlying *facesColorMap);
            __MR_ObjectVoxels_setFacesColorMap(_UnderlyingPtr, facesColorMap.PassByMode, facesColorMap.Value is not null ? facesColorMap.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::updateFacesColorMap`.
        public unsafe void UpdateFacesColorMap(MR.FaceColors updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateFacesColorMap", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateFacesColorMap(_Underlying *_this, MR.FaceColors._Underlying *updated);
            __MR_ObjectVoxels_updateFacesColorMap(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::setEdgeWidth`.
        public unsafe void SetEdgeWidth(float edgeWidth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setEdgeWidth", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setEdgeWidth(_Underlying *_this, float edgeWidth);
            __MR_ObjectVoxels_setEdgeWidth(_UnderlyingPtr, edgeWidth);
        }

        /// Generated from method `MR::ObjectVoxels::setPointSize`.
        public unsafe void SetPointSize(float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setPointSize", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setPointSize(_Underlying *_this, float size);
            __MR_ObjectVoxels_setPointSize(_UnderlyingPtr, size);
        }

        /// Generated from method `MR::ObjectVoxels::setEdgesColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetEdgesColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setEdgesColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setEdgesColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectVoxels_setEdgesColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectVoxels::setPointsColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetPointsColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setPointsColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setPointsColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectVoxels_setPointsColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectVoxels::setBordersColor`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetBordersColor(MR.Const_Color color, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setBordersColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setBordersColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *id);
            __MR_ObjectVoxels_setBordersColor(_UnderlyingPtr, color._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// Generated from method `MR::ObjectVoxels::setTextures`.
        public unsafe void SetTextures(MR._ByValue_Vector_MRMeshTexture_MRTextureId texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setTextures", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setTextures(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.Vector_MRMeshTexture_MRTextureId._Underlying *texture);
            __MR_ObjectVoxels_setTextures(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::updateTextures`.
        public unsafe void UpdateTextures(MR.Vector_MRMeshTexture_MRTextureId updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateTextures", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateTextures(_Underlying *_this, MR.Vector_MRMeshTexture_MRTextureId._Underlying *updated);
            __MR_ObjectVoxels_updateTextures(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// the texture ids for the faces if more than one texture is used to texture the object
        /// texture coordinates (data_.uvCoordinates) at a point can belong to different textures, depending on which face the point belongs to
        /// Generated from method `MR::ObjectVoxels::setTexturePerFace`.
        public unsafe void SetTexturePerFace(MR._ByValue_TexturePerFace texturePerFace)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setTexturePerFace", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setTexturePerFace(_Underlying *_this, MR.Misc._PassBy texturePerFace_pass_by, MR.TexturePerFace._Underlying *texturePerFace);
            __MR_ObjectVoxels_setTexturePerFace(_UnderlyingPtr, texturePerFace.PassByMode, texturePerFace.Value is not null ? texturePerFace.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::updateTexturePerFace`.
        public unsafe void UpdateTexturePerFace(MR.TexturePerFace texturePerFace)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateTexturePerFace", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateTexturePerFace(_Underlying *_this, MR.TexturePerFace._Underlying *texturePerFace);
            __MR_ObjectVoxels_updateTexturePerFace(_UnderlyingPtr, texturePerFace._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::addTexture`.
        public unsafe void AddTexture(MR._ByValue_MeshTexture texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_addTexture", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_addTexture(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.MeshTexture._Underlying *texture);
            __MR_ObjectVoxels_addTexture(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setUVCoords`.
        public unsafe void SetUVCoords(MR._ByValue_VertCoords2 uvCoordinates)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setUVCoords(_Underlying *_this, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates);
            __MR_ObjectVoxels_setUVCoords(_UnderlyingPtr, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::updateUVCoords`.
        public unsafe void UpdateUVCoords(MR.VertCoords2 updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateUVCoords(_Underlying *_this, MR.VertCoords2._Underlying *updated);
            __MR_ObjectVoxels_updateUVCoords(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// copies texture, UV-coordinates and vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectVoxels::copyTextureAndColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyTextureAndColors(MR.Const_ObjectMeshHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_copyTextureAndColors", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_copyTextureAndColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectVoxels_copyTextureAndColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// copies vertex colors from given source object \param src using given map \param thisToSrc
        /// Generated from method `MR::ObjectVoxels::copyColors`.
        /// Parameter `thisToSrcFaces` defaults to `{}`.
        public unsafe void CopyColors(MR.Const_ObjectMeshHolder src, MR.Const_VertMap thisToSrc, MR.Const_FaceMap? thisToSrcFaces = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_copyColors", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_copyColors(_Underlying *_this, MR.Const_ObjectMeshHolder._Underlying *src, MR.Const_VertMap._Underlying *thisToSrc, MR.Const_FaceMap._Underlying *thisToSrcFaces);
            __MR_ObjectVoxels_copyColors(_UnderlyingPtr, src._UnderlyingPtr, thisToSrc._UnderlyingPtr, thisToSrcFaces is not null ? thisToSrcFaces._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setAncillaryTexture`.
        public unsafe void SetAncillaryTexture(MR._ByValue_MeshTexture texture)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setAncillaryTexture", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setAncillaryTexture(_Underlying *_this, MR.Misc._PassBy texture_pass_by, MR.MeshTexture._Underlying *texture);
            __MR_ObjectVoxels_setAncillaryTexture(_UnderlyingPtr, texture.PassByMode, texture.Value is not null ? texture.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setAncillaryUVCoords`.
        public unsafe void SetAncillaryUVCoords(MR._ByValue_VertCoords2 uvCoordinates)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setAncillaryUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setAncillaryUVCoords(_Underlying *_this, MR.Misc._PassBy uvCoordinates_pass_by, MR.VertCoords2._Underlying *uvCoordinates);
            __MR_ObjectVoxels_setAncillaryUVCoords(_UnderlyingPtr, uvCoordinates.PassByMode, uvCoordinates.Value is not null ? uvCoordinates.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::updateAncillaryUVCoords`.
        public unsafe void UpdateAncillaryUVCoords(MR.VertCoords2 updated)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_updateAncillaryUVCoords", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_updateAncillaryUVCoords(_Underlying *_this, MR.VertCoords2._Underlying *updated);
            __MR_ObjectVoxels_updateAncillaryUVCoords(_UnderlyingPtr, updated._UnderlyingPtr);
        }

        /// Generated from method `MR::ObjectVoxels::clearAncillaryTexture`.
        public unsafe void ClearAncillaryTexture()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_clearAncillaryTexture", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_clearAncillaryTexture(_Underlying *_this);
            __MR_ObjectVoxels_clearAncillaryTexture(_UnderlyingPtr);
        }

        /// reset all object colors to their default values from the current theme
        /// Generated from method `MR::ObjectVoxels::resetColors`.
        public unsafe void ResetColors()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_resetColors", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_resetColors(_Underlying *_this);
            __MR_ObjectVoxels_resetColors(_UnderlyingPtr);
        }

        /// set visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectVoxels::setVisualizeProperty`.
        public unsafe void SetVisualizeProperty(bool value, MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVisualizeProperty(_Underlying *_this, byte value, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setVisualizeProperty(_UnderlyingPtr, value ? (byte)1 : (byte)0, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set visual property mask
        /// Generated from method `MR::ObjectVoxels::setVisualizePropertyMask`.
        public unsafe void SetVisualizePropertyMask(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVisualizePropertyMask", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVisualizePropertyMask(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setVisualizePropertyMask(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// toggle visual property in all viewports specified by the mask
        /// Generated from method `MR::ObjectVoxels::toggleVisualizeProperty`.
        public unsafe void ToggleVisualizeProperty(MR.Const_AnyVisualizeMaskEnum type, MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_toggleVisualizeProperty", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_toggleVisualizeProperty(_Underlying *_this, MR.AnyVisualizeMaskEnum._Underlying *type, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_toggleVisualizeProperty(_UnderlyingPtr, type._UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// set all visualize properties masks
        /// Generated from method `MR::ObjectVoxels::setAllVisualizeProperties`.
        public unsafe void SetAllVisualizeProperties(MR.Std.Const_Vector_MRViewportMask properties)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setAllVisualizeProperties", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setAllVisualizeProperties(_Underlying *_this, MR.Std.Const_Vector_MRViewportMask._Underlying *properties);
            __MR_ObjectVoxels_setAllVisualizeProperties(_UnderlyingPtr, properties._UnderlyingPtr);
        }

        /// if false deactivates clipped-by-plane for this object and all of its parents, otherwise sets clipped-by-plane for this this object only
        /// Generated from method `MR::ObjectVoxels::setGlobalClippedByPlane`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetGlobalClippedByPlane(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setGlobalClippedByPlane", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setGlobalClippedByPlane(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setGlobalClippedByPlane(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in given viewport
        /// Generated from method `MR::ObjectVoxels::setFrontColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetFrontColor(MR.Const_Color color, bool selected, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setFrontColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setFrontColor(_Underlying *_this, MR.Const_Color._Underlying *color, byte selected, MR.ViewportId *viewportId);
            __MR_ObjectVoxels_setFrontColor(_UnderlyingPtr, color._UnderlyingPtr, selected ? (byte)1 : (byte)0, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets color of object when it is selected/not-selected (depending on argument) in all viewports
        /// Generated from method `MR::ObjectVoxels::setFrontColorsForAllViewports`.
        /// Parameter `selected` defaults to `true`.
        public unsafe void SetFrontColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val, bool? selected = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setFrontColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setFrontColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val, byte *selected);
            byte __deref_selected = selected.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_ObjectVoxels_setFrontColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null, selected.HasValue ? &__deref_selected : null);
        }

        /// sets backward color of object in all viewports
        /// Generated from method `MR::ObjectVoxels::setBackColorsForAllViewports`.
        public unsafe void SetBackColorsForAllViewports(MR._ByValue_ViewportProperty_MRColor val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setBackColorsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setBackColorsForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_MRColor._Underlying *val);
            __MR_ObjectVoxels_setBackColorsForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets backward color of object in given viewport
        /// Generated from method `MR::ObjectVoxels::setBackColor`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetBackColor(MR.Const_Color color, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setBackColor", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setBackColor(_Underlying *_this, MR.Const_Color._Underlying *color, MR.ViewportId *viewportId);
            __MR_ObjectVoxels_setBackColor(_UnderlyingPtr, color._UnderlyingPtr, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in given viewport
        /// Generated from method `MR::ObjectVoxels::setGlobalAlpha`.
        /// Parameter `viewportId` defaults to `{}`.
        public unsafe void SetGlobalAlpha(byte alpha, MR._InOpt_ViewportId viewportId = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setGlobalAlpha", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setGlobalAlpha(_Underlying *_this, byte alpha, MR.ViewportId *viewportId);
            __MR_ObjectVoxels_setGlobalAlpha(_UnderlyingPtr, alpha, viewportId.HasValue ? &viewportId.Object : null);
        }

        /// sets global transparency alpha of object in all viewports
        /// Generated from method `MR::ObjectVoxels::setGlobalAlphaForAllViewports`.
        public unsafe void SetGlobalAlphaForAllViewports(MR._ByValue_ViewportProperty_UnsignedChar val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setGlobalAlphaForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setGlobalAlphaForAllViewports(_Underlying *_this, MR.Misc._PassBy val_pass_by, MR.ViewportProperty_UnsignedChar._Underlying *val);
            __MR_ObjectVoxels_setGlobalAlphaForAllViewports(_UnderlyingPtr, val.PassByMode, val.Value is not null ? val.Value._UnderlyingPtr : null);
        }

        /// sets the object as can/cannot be picked (by mouse) in all of given viewports
        /// Generated from method `MR::ObjectVoxels::setPickable`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetPickable(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setPickable", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setPickable(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setPickable(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// sets coloring mode of the object with given argument
        /// Generated from method `MR::ObjectVoxels::setColoringType`.
        public unsafe void SetColoringType(MR.ColoringType coloringType)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setColoringType", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setColoringType(_Underlying *_this, MR.ColoringType coloringType);
            __MR_ObjectVoxels_setColoringType(_UnderlyingPtr, coloringType);
        }

        /// sets shininess visual value of the object with given argument
        /// Generated from method `MR::ObjectVoxels::setShininess`.
        public unsafe void SetShininess(float shininess)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setShininess", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setShininess(_Underlying *_this, float shininess);
            __MR_ObjectVoxels_setShininess(_UnderlyingPtr, shininess);
        }

        /// sets intensity of reflections
        /// Generated from method `MR::ObjectVoxels::setSpecularStrength`.
        public unsafe void SetSpecularStrength(float specularStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setSpecularStrength", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setSpecularStrength(_Underlying *_this, float specularStrength);
            __MR_ObjectVoxels_setSpecularStrength(_UnderlyingPtr, specularStrength);
        }

        /// sets intensity of non-directional light
        /// Generated from method `MR::ObjectVoxels::setAmbientStrength`.
        public unsafe void SetAmbientStrength(float ambientStrength)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setAmbientStrength", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setAmbientStrength(_Underlying *_this, float ambientStrength);
            __MR_ObjectVoxels_setAmbientStrength(_UnderlyingPtr, ambientStrength);
        }

        /// set whether the scene-related properties should get their values from SceneColors and SceneSettings instances
        /// rather than from the input data on deserialization
        /// Generated from method `MR::ObjectVoxels::setUseDefaultScenePropertiesOnDeserialization`.
        public unsafe void SetUseDefaultScenePropertiesOnDeserialization(bool useDefaultScenePropertiesOnDeserialization)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setUseDefaultScenePropertiesOnDeserialization", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setUseDefaultScenePropertiesOnDeserialization(_Underlying *_this, byte useDefaultScenePropertiesOnDeserialization);
            __MR_ObjectVoxels_setUseDefaultScenePropertiesOnDeserialization(_UnderlyingPtr, useDefaultScenePropertiesOnDeserialization ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectVoxels::setName`.
        public unsafe void SetName(ReadOnlySpan<char> name)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setName", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setName(_Underlying *_this, byte *name, byte *name_end);
            byte[] __bytes_name = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(name.Length)];
            int __len_name = System.Text.Encoding.UTF8.GetBytes(name, __bytes_name);
            fixed (byte *__ptr_name = __bytes_name)
            {
                __MR_ObjectVoxels_setName(_UnderlyingPtr, __ptr_name, __ptr_name + __len_name);
            }
        }

        /// Generated from method `MR::ObjectVoxels::setXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setXf", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectVoxels_setXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
        /// Generated from method `MR::ObjectVoxels::resetXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void ResetXf(MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_resetXf", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_resetXf(_Underlying *_this, MR.ViewportId *id);
            __MR_ObjectVoxels_resetXf(_UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// modifies xfs for all viewports at once
        /// Generated from method `MR::ObjectVoxels::setXfsForAllViewports`.
        public unsafe void SetXfsForAllViewports(MR._ByValue_ViewportProperty_MRAffineXf3f xf)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setXfsForAllViewports", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setXfsForAllViewports(_Underlying *_this, MR.Misc._PassBy xf_pass_by, MR.ViewportProperty_MRAffineXf3f._Underlying *xf);
            __MR_ObjectVoxels_setXfsForAllViewports(_UnderlyingPtr, xf.PassByMode, xf.Value is not null ? xf.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setWorldXf`.
        /// Parameter `id` defaults to `{}`.
        public unsafe void SetWorldXf(MR.Const_AffineXf3f xf, MR._InOpt_ViewportId id = default)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setWorldXf", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setWorldXf(_Underlying *_this, MR.Const_AffineXf3f._Underlying *xf, MR.ViewportId *id);
            __MR_ObjectVoxels_setWorldXf(_UnderlyingPtr, xf._UnderlyingPtr, id.HasValue ? &id.Object : null);
        }

        /// if true sets all predecessors visible, otherwise sets this object invisible
        /// Generated from method `MR::ObjectVoxels::setGlobalVisibility`.
        /// Parameter `viewportMask` defaults to `ViewportMask::any()`.
        public unsafe void SetGlobalVisibility(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setGlobalVisibility", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setGlobalVisibility(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setGlobalVisibility(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// Generated from method `MR::ObjectVoxels::setLocked`.
        public unsafe void SetLocked(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setLocked", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setLocked(_Underlying *_this, byte on);
            __MR_ObjectVoxels_setLocked(_UnderlyingPtr, on ? (byte)1 : (byte)0);
        }

        /// Generated from method `MR::ObjectVoxels::setParentLocked`.
        public unsafe void SetParentLocked(bool lock_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setParentLocked", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setParentLocked(_Underlying *_this, byte lock_);
            __MR_ObjectVoxels_setParentLocked(_UnderlyingPtr, lock_ ? (byte)1 : (byte)0);
        }

        /// removes this from its parent children list
        /// returns false if it was already orphan
        /// Generated from method `MR::ObjectVoxels::detachFromParent`.
        public unsafe bool DetachFromParent()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_detachFromParent", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_detachFromParent(_Underlying *_this);
            return __MR_ObjectVoxels_detachFromParent(_UnderlyingPtr) != 0;
        }

        /// adds given object at the end of children (recognized or not);
        /// returns false if it was already child of this, of if given pointer is empty;
        /// child object will always report this as parent after the call;
        /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
        /// Generated from method `MR::ObjectVoxels::addChild`.
        /// Parameter `recognizedChild` defaults to `true`.
        public unsafe bool AddChild(MR._ByValue_Object child, bool? recognizedChild = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_addChild", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_addChild(_Underlying *_this, MR.Misc._PassBy child_pass_by, MR.Object._UnderlyingShared *child, byte *recognizedChild);
            byte __deref_recognizedChild = recognizedChild.GetValueOrDefault() ? (byte)1 : (byte)0;
            return __MR_ObjectVoxels_addChild(_UnderlyingPtr, child.PassByMode, child.Value is not null ? child.Value._UnderlyingSharedPtr : null, recognizedChild.HasValue ? &__deref_recognizedChild : null) != 0;
        }

        /// adds given object in the recognized children before existingChild;
        /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
        /// returns false if newChild is nullptr, or existingChild is not a child of this
        /// Generated from method `MR::ObjectVoxels::addChildBefore`.
        public unsafe bool AddChildBefore(MR._ByValue_Object newChild, MR.Const_Object existingChild)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_addChildBefore", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_addChildBefore(_Underlying *_this, MR.Misc._PassBy newChild_pass_by, MR.Object._UnderlyingShared *newChild, MR.Const_Object._UnderlyingShared *existingChild);
            return __MR_ObjectVoxels_addChildBefore(_UnderlyingPtr, newChild.PassByMode, newChild.Value is not null ? newChild.Value._UnderlyingSharedPtr : null, existingChild._UnderlyingSharedPtr) != 0;
        }

        /// detaches all recognized children from this, keeping all unrecognized ones
        /// Generated from method `MR::ObjectVoxels::removeAllChildren`.
        public unsafe void RemoveAllChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_removeAllChildren", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_removeAllChildren(_Underlying *_this);
            __MR_ObjectVoxels_removeAllChildren(_UnderlyingPtr);
        }

        /// sort recognized children by name
        /// Generated from method `MR::ObjectVoxels::sortChildren`.
        public unsafe void SortChildren()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_sortChildren", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_sortChildren(_Underlying *_this);
            __MR_ObjectVoxels_sortChildren(_UnderlyingPtr);
        }

        /// selects the object, returns true if value changed, otherwise returns false
        /// Generated from method `MR::ObjectVoxels::select`.
        public unsafe bool Select(bool on)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_select", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_select(_Underlying *_this, byte on);
            return __MR_ObjectVoxels_select(_UnderlyingPtr, on ? (byte)1 : (byte)0) != 0;
        }

        /// ancillary object is an object hidden (in scene menu) from a regular user
        /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
        /// Generated from method `MR::ObjectVoxels::setAncillary`.
        public unsafe void SetAncillary(bool ancillary)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setAncillary", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setAncillary(_Underlying *_this, byte ancillary);
            __MR_ObjectVoxels_setAncillary(_UnderlyingPtr, ancillary ? (byte)1 : (byte)0);
        }

        /// sets the object visible in the viewports specified by the mask (by default in all viewports)
        /// Generated from method `MR::ObjectVoxels::setVisible`.
        /// Parameter `viewportMask` defaults to `ViewportMask::all()`.
        public unsafe void SetVisible(bool on, MR.Const_ViewportMask? viewportMask = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVisible", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVisible(_Underlying *_this, byte on, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setVisible(_UnderlyingPtr, on ? (byte)1 : (byte)0, viewportMask is not null ? viewportMask._UnderlyingPtr : null);
        }

        /// specifies object visibility as bitmask of viewports
        /// Generated from method `MR::ObjectVoxels::setVisibilityMask`.
        public unsafe void SetVisibilityMask(MR.Const_ViewportMask viewportMask)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_setVisibilityMask", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_setVisibilityMask(_Underlying *_this, MR.ViewportMask._Underlying *viewportMask);
            __MR_ObjectVoxels_setVisibilityMask(_UnderlyingPtr, viewportMask._UnderlyingPtr);
        }

        /// swaps this object with other
        /// note: do not swap object signals, so listeners will get notifications from swapped object
        /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
        /// Generated from method `MR::ObjectVoxels::swap`.
        public unsafe void Swap(MR.Object other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_swap", ExactSpelling = true)]
            extern static void __MR_ObjectVoxels_swap(_Underlying *_this, MR.Object._Underlying *other);
            __MR_ObjectVoxels_swap(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// adds tag to the object's tag storage
        /// additionally calls ObjectTagManager::tagAddedSignal
        /// NOTE: tags starting with a dot are considered as service ones and might be hidden from UI
        /// Generated from method `MR::ObjectVoxels::addTag`.
        public unsafe bool AddTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_addTag", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_addTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectVoxels_addTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }

        /// removes tag from the object's tag storage
        /// additionally calls ObjectTagManager::tagRemovedSignal
        /// Generated from method `MR::ObjectVoxels::removeTag`.
        public unsafe bool RemoveTag(ReadOnlySpan<char> tag)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ObjectVoxels_removeTag", ExactSpelling = true)]
            extern static byte __MR_ObjectVoxels_removeTag(_Underlying *_this, byte *tag, byte *tag_end);
            byte[] __bytes_tag = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(tag.Length)];
            int __len_tag = System.Text.Encoding.UTF8.GetBytes(tag, __bytes_tag);
            fixed (byte *__ptr_tag = __bytes_tag)
            {
                return __MR_ObjectVoxels_removeTag(_UnderlyingPtr, __ptr_tag, __ptr_tag + __len_tag) != 0;
            }
        }
    }

    /// This is used as a function parameter when the underlying function receives `ObjectVoxels` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `ObjectVoxels`/`Const_ObjectVoxels` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_ObjectVoxels
    {
        internal readonly Const_ObjectVoxels? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_ObjectVoxels() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_ObjectVoxels(MR.Misc._Moved<ObjectVoxels> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_ObjectVoxels(MR.Misc._Moved<ObjectVoxels> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `ObjectVoxels` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ObjectVoxels`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectVoxels`/`Const_ObjectVoxels` directly.
    public class _InOptMut_ObjectVoxels
    {
        public ObjectVoxels? Opt;

        public _InOptMut_ObjectVoxels() {}
        public _InOptMut_ObjectVoxels(ObjectVoxels value) {Opt = value;}
        public static implicit operator _InOptMut_ObjectVoxels(ObjectVoxels value) {return new(value);}
    }

    /// This is used for optional parameters of class `ObjectVoxels` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ObjectVoxels`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ObjectVoxels`/`Const_ObjectVoxels` to pass it to the function.
    public class _InOptConst_ObjectVoxels
    {
        public Const_ObjectVoxels? Opt;

        public _InOptConst_ObjectVoxels() {}
        public _InOptConst_ObjectVoxels(Const_ObjectVoxels value) {Opt = value;}
        public static implicit operator _InOptConst_ObjectVoxels(Const_ObjectVoxels value) {return new(value);}
    }

    /// returns file extension used to serialize ObjectVoxels by default (if not overridden in specific object),
    /// the string starts with '.'
    /// Generated from function `MR::defaultSerializeVoxelsFormat`.
    public static unsafe MR.Std.Const_String DefaultSerializeVoxelsFormat()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_defaultSerializeVoxelsFormat", ExactSpelling = true)]
        extern static MR.Std.Const_String._Underlying *__MR_defaultSerializeVoxelsFormat();
        return new(__MR_defaultSerializeVoxelsFormat(), is_owning: false);
    }

    /// sets file extension used to serialize serialize ObjectVoxels by default (if not overridden in specific object),
    /// the string must start from '.'
    /// Generated from function `MR::setDefaultSerializeVoxelsFormat`.
    public static unsafe void SetDefaultSerializeVoxelsFormat(ReadOnlySpan<char> newFormat)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_setDefaultSerializeVoxelsFormat", ExactSpelling = true)]
        extern static void __MR_setDefaultSerializeVoxelsFormat(byte *newFormat, byte *newFormat_end);
        byte[] __bytes_newFormat = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(newFormat.Length)];
        int __len_newFormat = System.Text.Encoding.UTF8.GetBytes(newFormat, __bytes_newFormat);
        fixed (byte *__ptr_newFormat = __bytes_newFormat)
        {
            __MR_setDefaultSerializeVoxelsFormat(__ptr_newFormat, __ptr_newFormat + __len_newFormat);
        }
    }
}
