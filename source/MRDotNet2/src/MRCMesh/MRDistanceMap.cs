public static partial class MR
{
    /// this class allows to store distances from the plane in particular pixels
    /// validVerts keeps only pixels with mesh-intersecting rays from them
    /// Generated from class `MR::DistanceMap`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RectIndexer`
    /// This is the const half of the class.
    public class Const_DistanceMap : MR.Misc.SharedObject, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.
        internal struct _UnderlyingShared; // Represents the underlying shared pointer C++ type.

        internal unsafe _UnderlyingShared *_UnderlyingSharedPtr;
        internal unsafe _Underlying *_UnderlyingPtr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_Get", ExactSpelling = true)]
                extern static _Underlying *__MR_std_shared_ptr_MR_DistanceMap_Get(_UnderlyingShared *_this);
                return __MR_std_shared_ptr_MR_DistanceMap_Get(_UnderlyingSharedPtr);
            }
        }

        /// Check if the underlying shared pointer is owning or not.
        public override bool _IsOwning
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_UseCount", ExactSpelling = true)]
                extern static int __MR_std_shared_ptr_MR_DistanceMap_UseCount();
                return __MR_std_shared_ptr_MR_DistanceMap_UseCount() > 0;
            }
        }

        /// Clones the underlying shared pointer. Returns an owning pointer to shared pointer (which itself isn't necessarily owning).
        internal unsafe _UnderlyingShared *_CloneUnderlyingSharedPtr()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_ConstructFromAnother", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_DistanceMap_ConstructFromAnother(MR.Misc._PassBy other_pass_by, _UnderlyingShared *other);
            return __MR_std_shared_ptr_MR_DistanceMap_ConstructFromAnother(MR.Misc._PassBy.copy, _UnderlyingSharedPtr);
        }

        internal unsafe Const_DistanceMap(_Underlying *ptr, bool is_owning) : base(true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_DistanceMap_Construct(_Underlying *other);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_ConstructNonOwning", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_DistanceMap_ConstructNonOwning(_Underlying *other);
            if (is_owning)
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_DistanceMap_Construct(ptr);
            else
                _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_DistanceMap_ConstructNonOwning(ptr);
        }

        internal unsafe Const_DistanceMap(_UnderlyingShared *shared_ptr, bool is_owning) : base(is_owning) {_UnderlyingSharedPtr = shared_ptr;}

        internal static unsafe DistanceMap _MakeAliasing(MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_ConstructAliasing", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_DistanceMap_ConstructAliasing(MR.Misc._PassBy ownership_pass_by, MR.Std.Const_SharedPtr_ConstVoid._Underlying *ownership, _Underlying *ptr);
            return new(__MR_std_shared_ptr_MR_DistanceMap_ConstructAliasing(MR.Misc._PassBy.copy, ownership, ptr), is_owning: true);
        }

        private protected unsafe void _LateMakeShared(_Underlying *ptr)
        {
            System.Diagnostics.Trace.Assert(_IsOwningVal == true);
            System.Diagnostics.Trace.Assert(_UnderlyingSharedPtr is null);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_Construct", ExactSpelling = true)]
            extern static _UnderlyingShared *__MR_std_shared_ptr_MR_DistanceMap_Construct(_Underlying *other);
            _UnderlyingSharedPtr = __MR_std_shared_ptr_MR_DistanceMap_Construct(ptr);
        }

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingSharedPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_std_shared_ptr_MR_DistanceMap_Destroy", ExactSpelling = true)]
            extern static void __MR_std_shared_ptr_MR_DistanceMap_Destroy(_UnderlyingShared *_this);
            __MR_std_shared_ptr_MR_DistanceMap_Destroy(_UnderlyingSharedPtr);
            _UnderlyingSharedPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceMap() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_RectIndexer(Const_DistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_UpcastTo_MR_RectIndexer", ExactSpelling = true)]
            extern static MR.Const_RectIndexer._Underlying *__MR_DistanceMap_UpcastTo_MR_RectIndexer(_Underlying *_this);
            return MR.Const_RectIndexer._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_DistanceMap_UpcastTo_MR_RectIndexer(self._UnderlyingPtr));
        }

        /// a constant that is treated as 'no value' or 'invalid value'
        public static unsafe float NOTVALIDVALUE
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_Get_NOT_VALID_VALUE", ExactSpelling = true)]
                extern static float *__MR_DistanceMap_Get_NOT_VALID_VALUE();
                return *__MR_DistanceMap_Get_NOT_VALID_VALUE();
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceMap() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_DefaultConstruct();
            _LateMakeShared(__MR_DistanceMap_DefaultConstruct());
        }

        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public unsafe Const_DistanceMap(MR._ByValue_DistanceMap _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceMap._Underlying *_other);
            _LateMakeShared(__MR_DistanceMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Preferable constructor with resolution arguments
        /// Access by the index (i) is equal to (y*resX + x)
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public unsafe Const_DistanceMap(ulong resX, ulong resY) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_Construct_2", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_Construct_2(ulong resX, ulong resY);
            _LateMakeShared(__MR_DistanceMap_Construct_2(resX, resY));
        }

        /// make from 2d array
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public unsafe Const_DistanceMap(MR.Const_Matrix_Float m) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_Construct_1", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_Construct_1(MR.Const_Matrix_Float._Underlying *m);
            _LateMakeShared(__MR_DistanceMap_Construct_1(m._UnderlyingPtr));
        }

        /// make from 2d array
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public static unsafe implicit operator Const_DistanceMap(MR.Const_Matrix_Float m) {return new(m);}

        /// checks if X,Y element is valid (i.e. not `NOT_VALID_VALUE`; passing invalid coords to this is UB)
        /// Generated from method `MR::DistanceMap::isValid`.
        public unsafe bool IsValid(ulong x, ulong y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_isValid_2", ExactSpelling = true)]
            extern static byte __MR_DistanceMap_isValid_2(_Underlying *_this, ulong x, ulong y);
            return __MR_DistanceMap_isValid_2(_UnderlyingPtr, x, y) != 0;
        }

        /// checks if index element is valid (i.e. not `NOT_VALID_VALUE`; passing an invalid coord to this is UB)
        /// Generated from method `MR::DistanceMap::isValid`.
        public unsafe bool IsValid(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_isValid_1", ExactSpelling = true)]
            extern static byte __MR_DistanceMap_isValid_1(_Underlying *_this, ulong i);
            return __MR_DistanceMap_isValid_1(_UnderlyingPtr, i) != 0;
        }

        /// Returns true if (X,Y) coordinates are in bounds.
        /// Generated from method `MR::DistanceMap::isInBounds`.
        public unsafe bool IsInBounds(ulong x, ulong y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_isInBounds_2", ExactSpelling = true)]
            extern static byte __MR_DistanceMap_isInBounds_2(_Underlying *_this, ulong x, ulong y);
            return __MR_DistanceMap_isInBounds_2(_UnderlyingPtr, x, y) != 0;
        }

        /// Returns true if a flattened coordinate is in bounds.
        /// Generated from method `MR::DistanceMap::isInBounds`.
        public unsafe bool IsInBounds(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_isInBounds_1", ExactSpelling = true)]
            extern static byte __MR_DistanceMap_isInBounds_1(_Underlying *_this, ulong i);
            return __MR_DistanceMap_isInBounds_1(_UnderlyingPtr, i) != 0;
        }

        /// returns value in (X,Y) element, returns nullopt if not valid (see `isValid()`), UB if out of bounds.
        /// Generated from method `MR::DistanceMap::get`.
        public unsafe MR.Std.Optional_Float Get(ulong x, ulong y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_get_2", ExactSpelling = true)]
            extern static MR.Std.Optional_Float._Underlying *__MR_DistanceMap_get_2(_Underlying *_this, ulong x, ulong y);
            return new(__MR_DistanceMap_get_2(_UnderlyingPtr, x, y), is_owning: true);
        }

        /// returns value of index element, returns nullopt if not valid (see `isValid()`), UB if out of bounds.
        /// Generated from method `MR::DistanceMap::get`.
        public unsafe MR.Std.Optional_Float Get(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_get_1", ExactSpelling = true)]
            extern static MR.Std.Optional_Float._Underlying *__MR_DistanceMap_get_1(_Underlying *_this, ulong i);
            return new(__MR_DistanceMap_get_1(_UnderlyingPtr, i), is_owning: true);
        }

        /// Generated from method `MR::DistanceMap::getValue`.
        public unsafe float GetValue(ulong x, ulong y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getValue_const_2", ExactSpelling = true)]
            extern static float __MR_DistanceMap_getValue_const_2(_Underlying *_this, ulong x, ulong y);
            return __MR_DistanceMap_getValue_const_2(_UnderlyingPtr, x, y);
        }

        /// Generated from method `MR::DistanceMap::getValue`.
        public unsafe float GetValue(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getValue_const_1", ExactSpelling = true)]
            extern static float __MR_DistanceMap_getValue_const_1(_Underlying *_this, ulong i);
            return __MR_DistanceMap_getValue_const_1(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::DistanceMap::data`.
        public unsafe float? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_data_const", ExactSpelling = true)]
            extern static float *__MR_DistanceMap_data_const(_Underlying *_this);
            var __ret = __MR_DistanceMap_data_const(_UnderlyingPtr);
            return __ret is not null ? *__ret : null;
        }

        /**
        * \brief finds interpolated value.
        * \details https://en.wikipedia.org/wiki/Bilinear_interpolation
        * getInterpolated( 0.5f, 0.5f ) == get( 0, 0 )
        * see https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates for details
        * all 4 elements around this point should be valid, returns nullopt if at least one is not valid
        * \param x,y should be in resolution range [0;resX][0;resY].
        * If x,y are out of bounds, returns nullopt.
        */
        /// Generated from method `MR::DistanceMap::getInterpolated`.
        public unsafe MR.Std.Optional_Float GetInterpolated(float x, float y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getInterpolated", ExactSpelling = true)]
            extern static MR.Std.Optional_Float._Underlying *__MR_DistanceMap_getInterpolated(_Underlying *_this, float x, float y);
            return new(__MR_DistanceMap_getInterpolated(_UnderlyingPtr, x, y), is_owning: true);
        }

        /// finds 3d coordinates of the Point on the model surface for the (x,y) pixel
        /// Use the same params with distance map creation
        /// (x,y) must be in bounds, the behavior is undefined otherwise.
        /// Generated from method `MR::DistanceMap::unproject`.
        public unsafe MR.Std.Optional_MRVector3f Unproject(ulong x, ulong y, MR.Const_AffineXf3f toWorld)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_unproject", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVector3f._Underlying *__MR_DistanceMap_unproject(_Underlying *_this, ulong x, ulong y, MR.Const_AffineXf3f._Underlying *toWorld);
            return new(__MR_DistanceMap_unproject(_UnderlyingPtr, x, y, toWorld._UnderlyingPtr), is_owning: true);
        }

        /**
        * \brief finds 3d coordinates of the Point on the model surface for the (x,y) interpolated value
        * \param x,y should be in resolution range [0;resX][0;resY].
        * \details getInterpolated( 0.5f, 0.5f ) == get( 0, 0 )
        * see https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates for details
        * all 4 elements around this point should be valid, returns nullopt if at least one is not valid
        * If x,y are out of bounds, returns nullopt.
        */
        /// Generated from method `MR::DistanceMap::unprojectInterpolated`.
        public unsafe MR.Std.Optional_MRVector3f UnprojectInterpolated(float x, float y, MR.Const_AffineXf3f toWorld)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_unprojectInterpolated", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVector3f._Underlying *__MR_DistanceMap_unprojectInterpolated(_Underlying *_this, float x, float y, MR.Const_AffineXf3f._Underlying *toWorld);
            return new(__MR_DistanceMap_unprojectInterpolated(_UnderlyingPtr, x, y, toWorld._UnderlyingPtr), is_owning: true);
        }

        /// boolean operators
        /// returns new Distance Map with cell-wise maximum values. Invalid values remain only if both corresponding cells are invalid
        /// Generated from method `MR::DistanceMap::max`.
        public unsafe MR.Misc._Moved<MR.DistanceMap> Max(MR.Const_DistanceMap rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_max", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_max(_Underlying *_this, MR.Const_DistanceMap._Underlying *rhs);
            return MR.Misc.Move(new MR.DistanceMap(__MR_DistanceMap_max(_UnderlyingPtr, rhs._UnderlyingPtr), is_owning: true));
        }

        /// returns new Distance Map with cell-wise minimum values. Invalid values remain only if both corresponding cells are invalid
        /// Generated from method `MR::DistanceMap::min`.
        public unsafe MR.Misc._Moved<MR.DistanceMap> Min(MR.Const_DistanceMap rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_min", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_min(_Underlying *_this, MR.Const_DistanceMap._Underlying *rhs);
            return MR.Misc.Move(new MR.DistanceMap(__MR_DistanceMap_min(_UnderlyingPtr, rhs._UnderlyingPtr), is_owning: true));
        }

        /// returns new Distance Map with cell-wise subtracted values. Invalid values remain only if both corresponding cells are invalid
        /// Generated from method `MR::DistanceMap::operator-`.
        public static unsafe MR.Misc._Moved<MR.DistanceMap> operator-(MR.Const_DistanceMap _this, MR.Const_DistanceMap rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sub_MR_DistanceMap", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_sub_MR_DistanceMap(MR.Const_DistanceMap._Underlying *_this, MR.Const_DistanceMap._Underlying *rhs);
            return MR.Misc.Move(new MR.DistanceMap(__MR_sub_MR_DistanceMap(_this._UnderlyingPtr, rhs._UnderlyingPtr), is_owning: true));
        }

        /// returns new derivatives map without directions
        /// Generated from method `MR::DistanceMap::getDerivativeMap`.
        public unsafe MR.Misc._Moved<MR.DistanceMap> GetDerivativeMap()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getDerivativeMap", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_getDerivativeMap(_Underlying *_this);
            return MR.Misc.Move(new MR.DistanceMap(__MR_DistanceMap_getDerivativeMap(_UnderlyingPtr), is_owning: true));
        }

        /// returns new derivative maps with X and Y axes direction
        /// Generated from method `MR::DistanceMap::getXYDerivativeMaps`.
        public unsafe MR.Misc._Moved<MR.Std.Pair_MRDistanceMap_MRDistanceMap> GetXYDerivativeMaps()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getXYDerivativeMaps", ExactSpelling = true)]
            extern static MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *__MR_DistanceMap_getXYDerivativeMaps(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Pair_MRDistanceMap_MRDistanceMap(__MR_DistanceMap_getXYDerivativeMaps(_UnderlyingPtr), is_owning: true));
        }

        /// computes single derivative map from XY spaces combined. Returns local maximums then
        /// Generated from method `MR::DistanceMap::getLocalMaximums`.
        public unsafe MR.Misc._Moved<MR.Std.Vector_StdPairMRUint64TMRUint64T> GetLocalMaximums()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getLocalMaximums", ExactSpelling = true)]
            extern static MR.Std.Vector_StdPairMRUint64TMRUint64T._Underlying *__MR_DistanceMap_getLocalMaximums(_Underlying *_this);
            return MR.Misc.Move(new MR.Std.Vector_StdPairMRUint64TMRUint64T(__MR_DistanceMap_getLocalMaximums(_UnderlyingPtr), is_owning: true));
        }

        ///returns X resolution
        /// Generated from method `MR::DistanceMap::resX`.
        public unsafe ulong ResX()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_resX", ExactSpelling = true)]
            extern static ulong __MR_DistanceMap_resX(_Underlying *_this);
            return __MR_DistanceMap_resX(_UnderlyingPtr);
        }

        ///returns Y resolution
        /// Generated from method `MR::DistanceMap::resY`.
        public unsafe ulong ResY()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_resY", ExactSpelling = true)]
            extern static ulong __MR_DistanceMap_resY(_Underlying *_this);
            return __MR_DistanceMap_resY(_UnderlyingPtr);
        }

        ///returns the number of pixels
        /// Generated from method `MR::DistanceMap::numPoints`.
        public unsafe ulong NumPoints()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_numPoints", ExactSpelling = true)]
            extern static ulong __MR_DistanceMap_numPoints(_Underlying *_this);
            return __MR_DistanceMap_numPoints(_UnderlyingPtr);
        }

        /// finds minimum and maximum values
        /// returns min_float and max_float if all values are invalid
        /// Generated from method `MR::DistanceMap::getMinMaxValues`.
        public unsafe MR.Std.Pair_Float_Float GetMinMaxValues()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getMinMaxValues", ExactSpelling = true)]
            extern static MR.Std.Pair_Float_Float._Underlying *__MR_DistanceMap_getMinMaxValues(_Underlying *_this);
            return new(__MR_DistanceMap_getMinMaxValues(_UnderlyingPtr), is_owning: true);
        }

        /// finds minimum value X,Y
        /// returns [-1.-1] if all values are invalid
        /// Generated from method `MR::DistanceMap::getMinIndex`.
        public unsafe MR.Std.Pair_MRUint64T_MRUint64T GetMinIndex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getMinIndex", ExactSpelling = true)]
            extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_DistanceMap_getMinIndex(_Underlying *_this);
            return new(__MR_DistanceMap_getMinIndex(_UnderlyingPtr), is_owning: true);
        }

        /// finds maximum value X,Y
        /// returns [-1.-1] if all values are invalid
        /// Generated from method `MR::DistanceMap::getMaxIndex`.
        public unsafe MR.Std.Pair_MRUint64T_MRUint64T GetMaxIndex()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getMaxIndex", ExactSpelling = true)]
            extern static MR.Std.Pair_MRUint64T_MRUint64T._Underlying *__MR_DistanceMap_getMaxIndex(_Underlying *_this);
            return new(__MR_DistanceMap_getMaxIndex(_UnderlyingPtr), is_owning: true);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::DistanceMap::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_DistanceMap_heapBytes(_Underlying *_this);
            return __MR_DistanceMap_heapBytes(_UnderlyingPtr);
        }

        /// Generated from method `MR::DistanceMap::dims`.
        public unsafe MR.Const_Vector2i Dims()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_dims", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_DistanceMap_dims(_Underlying *_this);
            return new(__MR_DistanceMap_dims(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::DistanceMap::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_size", ExactSpelling = true)]
            extern static ulong __MR_DistanceMap_size(_Underlying *_this);
            return __MR_DistanceMap_size(_UnderlyingPtr);
        }

        /// Generated from method `MR::DistanceMap::toPixelId`.
        public unsafe MR.PixelId ToPixelId(MR.Const_Vector2i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_toPixelId", ExactSpelling = true)]
            extern static MR.PixelId __MR_DistanceMap_toPixelId(_Underlying *_this, MR.Const_Vector2i._Underlying *pos);
            return __MR_DistanceMap_toPixelId(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Generated from method `MR::DistanceMap::toIndex`.
        public unsafe ulong ToIndex(MR.Const_Vector2i pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_toIndex", ExactSpelling = true)]
            extern static ulong __MR_DistanceMap_toIndex(_Underlying *_this, MR.Const_Vector2i._Underlying *pos);
            return __MR_DistanceMap_toIndex(_UnderlyingPtr, pos._UnderlyingPtr);
        }
    }

    /// this class allows to store distances from the plane in particular pixels
    /// validVerts keeps only pixels with mesh-intersecting rays from them
    /// Generated from class `MR::DistanceMap`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::RectIndexer`
    /// This is the non-const half of the class.
    public class DistanceMap : Const_DistanceMap
    {
        internal unsafe DistanceMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        internal unsafe DistanceMap(_UnderlyingShared *shared_ptr, bool is_owning) : base(shared_ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.RectIndexer(DistanceMap self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_UpcastTo_MR_RectIndexer", ExactSpelling = true)]
            extern static MR.RectIndexer._Underlying *__MR_DistanceMap_UpcastTo_MR_RectIndexer(_Underlying *_this);
            return MR.RectIndexer._MakeAliasing((MR.Std.Const_SharedPtr_ConstVoid._Underlying *)self._UnderlyingSharedPtr, __MR_DistanceMap_UpcastTo_MR_RectIndexer(self._UnderlyingPtr));
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceMap() : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_DefaultConstruct();
            _LateMakeShared(__MR_DistanceMap_DefaultConstruct());
        }

        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public unsafe DistanceMap(MR._ByValue_DistanceMap _other) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceMap._Underlying *_other);
            _LateMakeShared(__MR_DistanceMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null));
        }

        /// Preferable constructor with resolution arguments
        /// Access by the index (i) is equal to (y*resX + x)
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public unsafe DistanceMap(ulong resX, ulong resY) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_Construct_2", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_Construct_2(ulong resX, ulong resY);
            _LateMakeShared(__MR_DistanceMap_Construct_2(resX, resY));
        }

        /// make from 2d array
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public unsafe DistanceMap(MR.Const_Matrix_Float m) : this(shared_ptr: null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_Construct_1", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_Construct_1(MR.Const_Matrix_Float._Underlying *m);
            _LateMakeShared(__MR_DistanceMap_Construct_1(m._UnderlyingPtr));
        }

        /// make from 2d array
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public static unsafe implicit operator DistanceMap(MR.Const_Matrix_Float m) {return new(m);}

        /// Generated from method `MR::DistanceMap::operator=`.
        public unsafe MR.DistanceMap Assign(MR._ByValue_DistanceMap _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMap._Underlying *__MR_DistanceMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceMap._Underlying *_other);
            return new(__MR_DistanceMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// returns value in (X,Y) element without check on valid
        /// use this only if you sure that distance map has no invalid values or for serialization
        /// Generated from method `MR::DistanceMap::getValue`.
        public unsafe new ref float GetValue(ulong x, ulong y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getValue_2", ExactSpelling = true)]
            extern static float *__MR_DistanceMap_getValue_2(_Underlying *_this, ulong x, ulong y);
            return ref *__MR_DistanceMap_getValue_2(_UnderlyingPtr, x, y);
        }

        /// Generated from method `MR::DistanceMap::getValue`.
        public unsafe new ref float GetValue(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_getValue_1", ExactSpelling = true)]
            extern static float *__MR_DistanceMap_getValue_1(_Underlying *_this, ulong i);
            return ref *__MR_DistanceMap_getValue_1(_UnderlyingPtr, i);
        }

        /// Generated from method `MR::DistanceMap::data`.
        public unsafe new MR.Misc.Ref<float>? Data()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_data", ExactSpelling = true)]
            extern static float *__MR_DistanceMap_data(_Underlying *_this);
            var __ret = __MR_DistanceMap_data(_UnderlyingPtr);
            return __ret is not null ? new MR.Misc.Ref<float>(__ret) : null;
        }

        /// replaces every valid element in the map with its negative value
        /// Generated from method `MR::DistanceMap::negate`.
        public unsafe void Negate()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_negate", ExactSpelling = true)]
            extern static void __MR_DistanceMap_negate(_Underlying *_this);
            __MR_DistanceMap_negate(_UnderlyingPtr);
        }

        /// replaces values with cell-wise maximum values. Invalid values remain only if both corresponding cells are invalid
        /// Generated from method `MR::DistanceMap::mergeMax`.
        public unsafe MR.Const_DistanceMap MergeMax(MR.Const_DistanceMap rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_mergeMax", ExactSpelling = true)]
            extern static MR.Const_DistanceMap._Underlying *__MR_DistanceMap_mergeMax(_Underlying *_this, MR.Const_DistanceMap._Underlying *rhs);
            return new(__MR_DistanceMap_mergeMax(_UnderlyingPtr, rhs._UnderlyingPtr), is_owning: false);
        }

        /// replaces values with cell-wise minimum values. Invalid values remain only if both corresponding cells are invalid
        /// Generated from method `MR::DistanceMap::mergeMin`.
        public unsafe MR.Const_DistanceMap MergeMin(MR.Const_DistanceMap rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_mergeMin", ExactSpelling = true)]
            extern static MR.Const_DistanceMap._Underlying *__MR_DistanceMap_mergeMin(_Underlying *_this, MR.Const_DistanceMap._Underlying *rhs);
            return new(__MR_DistanceMap_mergeMin(_UnderlyingPtr, rhs._UnderlyingPtr), is_owning: false);
        }

        /// replaces values with cell-wise subtracted values. Invalid values remain only if both corresponding cells are invalid
        /// Generated from method `MR::DistanceMap::operator-=`.
        public unsafe MR.Const_DistanceMap SubAssign(MR.Const_DistanceMap rhs)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_sub_assign", ExactSpelling = true)]
            extern static MR.Const_DistanceMap._Underlying *__MR_DistanceMap_sub_assign(_Underlying *_this, MR.Const_DistanceMap._Underlying *rhs);
            return new(__MR_DistanceMap_sub_assign(_UnderlyingPtr, rhs._UnderlyingPtr), is_owning: false);
        }

        /// sets value in (X,Y) element (the coords must be valid, UB otherwise)
        /// Generated from method `MR::DistanceMap::set`.
        public unsafe void Set(ulong x, ulong y, float val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_set_3", ExactSpelling = true)]
            extern static void __MR_DistanceMap_set_3(_Underlying *_this, ulong x, ulong y, float val);
            __MR_DistanceMap_set_3(_UnderlyingPtr, x, y, val);
        }

        /// sets value in index element (the coord must be valid, UB otherwise)
        /// Generated from method `MR::DistanceMap::set`.
        public unsafe void Set(ulong i, float val)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_set_2", ExactSpelling = true)]
            extern static void __MR_DistanceMap_set_2(_Underlying *_this, ulong i, float val);
            __MR_DistanceMap_set_2(_UnderlyingPtr, i, val);
        }

        /// sets all values at one time
        /// Generated from method `MR::DistanceMap::set`.
        public unsafe void Set(MR.Std._ByValue_Vector_Float data)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_set_1", ExactSpelling = true)]
            extern static void __MR_DistanceMap_set_1(_Underlying *_this, MR.Misc._PassBy data_pass_by, MR.Std.Vector_Float._Underlying *data);
            __MR_DistanceMap_set_1(_UnderlyingPtr, data.PassByMode, data.Value is not null ? data.Value._UnderlyingPtr : null);
        }

        /// invalidates value in (X,Y) element (the coords must be valid, UB otherwise)
        /// Generated from method `MR::DistanceMap::unset`.
        public unsafe void Unset(ulong x, ulong y)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_unset_2", ExactSpelling = true)]
            extern static void __MR_DistanceMap_unset_2(_Underlying *_this, ulong x, ulong y);
            __MR_DistanceMap_unset_2(_UnderlyingPtr, x, y);
        }

        /// invalidates value in index element (the coord must be valid, UB otherwise)
        /// Generated from method `MR::DistanceMap::unset`.
        public unsafe void Unset(ulong i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_unset_1", ExactSpelling = true)]
            extern static void __MR_DistanceMap_unset_1(_Underlying *_this, ulong i);
            __MR_DistanceMap_unset_1(_UnderlyingPtr, i);
        }

        /// invalidates all elements
        /// Generated from method `MR::DistanceMap::invalidateAll`.
        public unsafe void InvalidateAll()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_invalidateAll", ExactSpelling = true)]
            extern static void __MR_DistanceMap_invalidateAll(_Underlying *_this);
            __MR_DistanceMap_invalidateAll(_UnderlyingPtr);
        }

        /// clears data, sets resolutions to zero
        /// Generated from method `MR::DistanceMap::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_clear", ExactSpelling = true)]
            extern static void __MR_DistanceMap_clear(_Underlying *_this);
            __MR_DistanceMap_clear(_UnderlyingPtr);
        }

        /// Generated from method `MR::DistanceMap::resize`.
        public unsafe void Resize(MR.Const_Vector2i dims)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMap_resize", ExactSpelling = true)]
            extern static void __MR_DistanceMap_resize(_Underlying *_this, MR.Const_Vector2i._Underlying *dims);
            __MR_DistanceMap_resize(_UnderlyingPtr, dims._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DistanceMap` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DistanceMap`/`Const_DistanceMap` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DistanceMap
    {
        internal readonly Const_DistanceMap? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DistanceMap() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DistanceMap(Const_DistanceMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DistanceMap(Const_DistanceMap arg) {return new(arg);}
        public _ByValue_DistanceMap(MR.Misc._Moved<DistanceMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DistanceMap(MR.Misc._Moved<DistanceMap> arg) {return new(arg);}

        /// make from 2d array
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public static unsafe implicit operator _ByValue_DistanceMap(MR.Const_Matrix_Float m) {return new MR.DistanceMap(m);}
    }

    /// This is used for optional parameters of class `DistanceMap` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMap`/`Const_DistanceMap` directly.
    public class _InOptMut_DistanceMap
    {
        public DistanceMap? Opt;

        public _InOptMut_DistanceMap() {}
        public _InOptMut_DistanceMap(DistanceMap value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceMap(DistanceMap value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceMap` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceMap`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMap`/`Const_DistanceMap` to pass it to the function.
    public class _InOptConst_DistanceMap
    {
        public Const_DistanceMap? Opt;

        public _InOptConst_DistanceMap() {}
        public _InOptConst_DistanceMap(Const_DistanceMap value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceMap(Const_DistanceMap value) {return new(value);}

        /// make from 2d array
        /// Generated from constructor `MR::DistanceMap::DistanceMap`.
        public static unsafe implicit operator _InOptConst_DistanceMap(MR.Const_Matrix_Float m) {return new MR.DistanceMap(m);}
    }

    /// Structure with parameters for optional offset in `distanceMapFromContours` function
    /// Generated from class `MR::ContoursDistanceMapOffset`.
    /// This is the const half of the class.
    public class Const_ContoursDistanceMapOffset : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ContoursDistanceMapOffset(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_Destroy", ExactSpelling = true)]
            extern static void __MR_ContoursDistanceMapOffset_Destroy(_Underlying *_this);
            __MR_ContoursDistanceMapOffset_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ContoursDistanceMapOffset() {Dispose(false);}

        /// offset values for each undirected edge of given polyline
        public unsafe MR.Const_UndirectedEdgeScalars PerEdgeOffset
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_Get_perEdgeOffset", ExactSpelling = true)]
                extern static MR.Const_UndirectedEdgeScalars._Underlying *__MR_ContoursDistanceMapOffset_Get_perEdgeOffset(_Underlying *_this);
                return new(__MR_ContoursDistanceMapOffset_Get_perEdgeOffset(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.ContoursDistanceMapOffset.OffsetType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_Get_type", ExactSpelling = true)]
                extern static MR.ContoursDistanceMapOffset.OffsetType *__MR_ContoursDistanceMapOffset_Get_type(_Underlying *_this);
                return *__MR_ContoursDistanceMapOffset_Get_type(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::ContoursDistanceMapOffset::ContoursDistanceMapOffset`.
        public unsafe Const_ContoursDistanceMapOffset(MR.Const_ContoursDistanceMapOffset _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOffset._Underlying *__MR_ContoursDistanceMapOffset_ConstructFromAnother(MR.ContoursDistanceMapOffset._Underlying *_other);
            _UnderlyingPtr = __MR_ContoursDistanceMapOffset_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Constructs `MR::ContoursDistanceMapOffset` elementwise.
        public unsafe Const_ContoursDistanceMapOffset(MR.Const_UndirectedEdgeScalars perEdgeOffset, MR.ContoursDistanceMapOffset.OffsetType type) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_ConstructFrom", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOffset._Underlying *__MR_ContoursDistanceMapOffset_ConstructFrom(MR.Const_UndirectedEdgeScalars._Underlying *perEdgeOffset, MR.ContoursDistanceMapOffset.OffsetType type);
            _UnderlyingPtr = __MR_ContoursDistanceMapOffset_ConstructFrom(perEdgeOffset._UnderlyingPtr, type);
        }

        public enum OffsetType : int
        {
            ///< distance map from given polyline with values offset
            Normal = 0,
            ///< distance map from shell of given polyline (perEdgeOffset should not have negative values )
            Shell = 1,
        }
    }

    /// Structure with parameters for optional offset in `distanceMapFromContours` function
    /// Generated from class `MR::ContoursDistanceMapOffset`.
    /// This is the non-const half of the class.
    public class ContoursDistanceMapOffset : Const_ContoursDistanceMapOffset
    {
        internal unsafe ContoursDistanceMapOffset(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.ContoursDistanceMapOffset.OffsetType Type
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_GetMutable_type", ExactSpelling = true)]
                extern static MR.ContoursDistanceMapOffset.OffsetType *__MR_ContoursDistanceMapOffset_GetMutable_type(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOffset_GetMutable_type(_UnderlyingPtr);
            }
        }

        /// Generated from constructor `MR::ContoursDistanceMapOffset::ContoursDistanceMapOffset`.
        public unsafe ContoursDistanceMapOffset(MR.Const_ContoursDistanceMapOffset _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOffset._Underlying *__MR_ContoursDistanceMapOffset_ConstructFromAnother(MR.ContoursDistanceMapOffset._Underlying *_other);
            _UnderlyingPtr = __MR_ContoursDistanceMapOffset_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Constructs `MR::ContoursDistanceMapOffset` elementwise.
        public unsafe ContoursDistanceMapOffset(MR.Const_UndirectedEdgeScalars perEdgeOffset, MR.ContoursDistanceMapOffset.OffsetType type) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOffset_ConstructFrom", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOffset._Underlying *__MR_ContoursDistanceMapOffset_ConstructFrom(MR.Const_UndirectedEdgeScalars._Underlying *perEdgeOffset, MR.ContoursDistanceMapOffset.OffsetType type);
            _UnderlyingPtr = __MR_ContoursDistanceMapOffset_ConstructFrom(perEdgeOffset._UnderlyingPtr, type);
        }
    }

    /// This is used for optional parameters of class `ContoursDistanceMapOffset` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ContoursDistanceMapOffset`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContoursDistanceMapOffset`/`Const_ContoursDistanceMapOffset` directly.
    public class _InOptMut_ContoursDistanceMapOffset
    {
        public ContoursDistanceMapOffset? Opt;

        public _InOptMut_ContoursDistanceMapOffset() {}
        public _InOptMut_ContoursDistanceMapOffset(ContoursDistanceMapOffset value) {Opt = value;}
        public static implicit operator _InOptMut_ContoursDistanceMapOffset(ContoursDistanceMapOffset value) {return new(value);}
    }

    /// This is used for optional parameters of class `ContoursDistanceMapOffset` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ContoursDistanceMapOffset`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContoursDistanceMapOffset`/`Const_ContoursDistanceMapOffset` to pass it to the function.
    public class _InOptConst_ContoursDistanceMapOffset
    {
        public Const_ContoursDistanceMapOffset? Opt;

        public _InOptConst_ContoursDistanceMapOffset() {}
        public _InOptConst_ContoursDistanceMapOffset(Const_ContoursDistanceMapOffset value) {Opt = value;}
        public static implicit operator _InOptConst_ContoursDistanceMapOffset(Const_ContoursDistanceMapOffset value) {return new(value);}
    }

    /// Generated from class `MR::ContoursDistanceMapOptions`.
    /// This is the const half of the class.
    public class Const_ContoursDistanceMapOptions : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ContoursDistanceMapOptions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Destroy", ExactSpelling = true)]
            extern static void __MR_ContoursDistanceMapOptions_Destroy(_Underlying *_this);
            __MR_ContoursDistanceMapOptions_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ContoursDistanceMapOptions() {Dispose(false);}

        public unsafe MR.ContoursDistanceMapOptions.SignedDetectionMethod SignMethod
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Get_signMethod", ExactSpelling = true)]
                extern static MR.ContoursDistanceMapOptions.SignedDetectionMethod *__MR_ContoursDistanceMapOptions_Get_signMethod(_Underlying *_this);
                return *__MR_ContoursDistanceMapOptions_Get_signMethod(_UnderlyingPtr);
            }
        }

        /// optional input offset for each edges of polyline, find more on `ContoursDistanceMapOffset` structure description
        public unsafe ref readonly void * OffsetParameters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Get_offsetParameters", ExactSpelling = true)]
                extern static void **__MR_ContoursDistanceMapOptions_Get_offsetParameters(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_Get_offsetParameters(_UnderlyingPtr);
            }
        }

        /// if pointer is valid, then only these pixels will be filled
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Get_region", ExactSpelling = true)]
                extern static void **__MR_ContoursDistanceMapOptions_Get_region(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_Get_region(_UnderlyingPtr);
            }
        }

        /// optional output vector of closest polyline edge per each pixel of distance map
        public unsafe ref void * OutClosestEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Get_outClosestEdges", ExactSpelling = true)]
                extern static void **__MR_ContoursDistanceMapOptions_Get_outClosestEdges(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_Get_outClosestEdges(_UnderlyingPtr);
            }
        }

        /// minimum value (or absolute value if offsetParameters == nullptr) in a pixel of distance map (lower values can be present but they are not precise)
        public unsafe float MinDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Get_minDist", ExactSpelling = true)]
                extern static float *__MR_ContoursDistanceMapOptions_Get_minDist(_Underlying *_this);
                return *__MR_ContoursDistanceMapOptions_Get_minDist(_UnderlyingPtr);
            }
        }

        /// maximum value (or absolute value if offsetParameters == nullptr) in a pixel of distance map (larger values cannot be present)
        public unsafe float MaxDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_Get_maxDist", ExactSpelling = true)]
                extern static float *__MR_ContoursDistanceMapOptions_Get_maxDist(_Underlying *_this);
                return *__MR_ContoursDistanceMapOptions_Get_maxDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ContoursDistanceMapOptions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_DefaultConstruct();
            _UnderlyingPtr = __MR_ContoursDistanceMapOptions_DefaultConstruct();
        }

        /// Constructs `MR::ContoursDistanceMapOptions` elementwise.
        public unsafe Const_ContoursDistanceMapOptions(MR.ContoursDistanceMapOptions.SignedDetectionMethod signMethod, MR.Const_ContoursDistanceMapOffset? offsetParameters, MR.Const_PixelBitSet? region, MR.Std.Vector_MRUndirectedEdgeId? outClosestEdges, float minDist, float maxDist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_ConstructFrom", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_ConstructFrom(MR.ContoursDistanceMapOptions.SignedDetectionMethod signMethod, MR.Const_ContoursDistanceMapOffset._Underlying *offsetParameters, MR.Const_PixelBitSet._Underlying *region, MR.Std.Vector_MRUndirectedEdgeId._Underlying *outClosestEdges, float minDist, float maxDist);
            _UnderlyingPtr = __MR_ContoursDistanceMapOptions_ConstructFrom(signMethod, offsetParameters is not null ? offsetParameters._UnderlyingPtr : null, region is not null ? region._UnderlyingPtr : null, outClosestEdges is not null ? outClosestEdges._UnderlyingPtr : null, minDist, maxDist);
        }

        /// Generated from constructor `MR::ContoursDistanceMapOptions::ContoursDistanceMapOptions`.
        public unsafe Const_ContoursDistanceMapOptions(MR.Const_ContoursDistanceMapOptions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_ConstructFromAnother(MR.ContoursDistanceMapOptions._Underlying *_other);
            _UnderlyingPtr = __MR_ContoursDistanceMapOptions_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// method to calculate sign
        public enum SignedDetectionMethod : int
        {
            /// detect sign of distance based on closest contour's edge turn\n
            /// (recommended for good contours with no self-intersections)
            /// \note that polyline topology should be consistently oriented \n
            ContourOrientation = 0,
            /// detect sign of distance based on number of ray intersections with contours\n
            /// (recommended for contours with self-intersections)
            WindingRule = 1,
        }
    }

    /// Generated from class `MR::ContoursDistanceMapOptions`.
    /// This is the non-const half of the class.
    public class ContoursDistanceMapOptions : Const_ContoursDistanceMapOptions
    {
        internal unsafe ContoursDistanceMapOptions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.ContoursDistanceMapOptions.SignedDetectionMethod SignMethod
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_GetMutable_signMethod", ExactSpelling = true)]
                extern static MR.ContoursDistanceMapOptions.SignedDetectionMethod *__MR_ContoursDistanceMapOptions_GetMutable_signMethod(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_GetMutable_signMethod(_UnderlyingPtr);
            }
        }

        /// optional input offset for each edges of polyline, find more on `ContoursDistanceMapOffset` structure description
        public new unsafe ref readonly void * OffsetParameters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_GetMutable_offsetParameters", ExactSpelling = true)]
                extern static void **__MR_ContoursDistanceMapOptions_GetMutable_offsetParameters(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_GetMutable_offsetParameters(_UnderlyingPtr);
            }
        }

        /// if pointer is valid, then only these pixels will be filled
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_ContoursDistanceMapOptions_GetMutable_region(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_GetMutable_region(_UnderlyingPtr);
            }
        }

        /// optional output vector of closest polyline edge per each pixel of distance map
        public new unsafe ref void * OutClosestEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_GetMutable_outClosestEdges", ExactSpelling = true)]
                extern static void **__MR_ContoursDistanceMapOptions_GetMutable_outClosestEdges(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_GetMutable_outClosestEdges(_UnderlyingPtr);
            }
        }

        /// minimum value (or absolute value if offsetParameters == nullptr) in a pixel of distance map (lower values can be present but they are not precise)
        public new unsafe ref float MinDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_GetMutable_minDist", ExactSpelling = true)]
                extern static float *__MR_ContoursDistanceMapOptions_GetMutable_minDist(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_GetMutable_minDist(_UnderlyingPtr);
            }
        }

        /// maximum value (or absolute value if offsetParameters == nullptr) in a pixel of distance map (larger values cannot be present)
        public new unsafe ref float MaxDist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_GetMutable_maxDist", ExactSpelling = true)]
                extern static float *__MR_ContoursDistanceMapOptions_GetMutable_maxDist(_Underlying *_this);
                return ref *__MR_ContoursDistanceMapOptions_GetMutable_maxDist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ContoursDistanceMapOptions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_DefaultConstruct();
            _UnderlyingPtr = __MR_ContoursDistanceMapOptions_DefaultConstruct();
        }

        /// Constructs `MR::ContoursDistanceMapOptions` elementwise.
        public unsafe ContoursDistanceMapOptions(MR.ContoursDistanceMapOptions.SignedDetectionMethod signMethod, MR.Const_ContoursDistanceMapOffset? offsetParameters, MR.Const_PixelBitSet? region, MR.Std.Vector_MRUndirectedEdgeId? outClosestEdges, float minDist, float maxDist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_ConstructFrom", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_ConstructFrom(MR.ContoursDistanceMapOptions.SignedDetectionMethod signMethod, MR.Const_ContoursDistanceMapOffset._Underlying *offsetParameters, MR.Const_PixelBitSet._Underlying *region, MR.Std.Vector_MRUndirectedEdgeId._Underlying *outClosestEdges, float minDist, float maxDist);
            _UnderlyingPtr = __MR_ContoursDistanceMapOptions_ConstructFrom(signMethod, offsetParameters is not null ? offsetParameters._UnderlyingPtr : null, region is not null ? region._UnderlyingPtr : null, outClosestEdges is not null ? outClosestEdges._UnderlyingPtr : null, minDist, maxDist);
        }

        /// Generated from constructor `MR::ContoursDistanceMapOptions::ContoursDistanceMapOptions`.
        public unsafe ContoursDistanceMapOptions(MR.Const_ContoursDistanceMapOptions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_ConstructFromAnother(MR.ContoursDistanceMapOptions._Underlying *_other);
            _UnderlyingPtr = __MR_ContoursDistanceMapOptions_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::ContoursDistanceMapOptions::operator=`.
        public unsafe MR.ContoursDistanceMapOptions Assign(MR.Const_ContoursDistanceMapOptions _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContoursDistanceMapOptions_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ContoursDistanceMapOptions._Underlying *__MR_ContoursDistanceMapOptions_AssignFromAnother(_Underlying *_this, MR.ContoursDistanceMapOptions._Underlying *_other);
            return new(__MR_ContoursDistanceMapOptions_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ContoursDistanceMapOptions` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ContoursDistanceMapOptions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContoursDistanceMapOptions`/`Const_ContoursDistanceMapOptions` directly.
    public class _InOptMut_ContoursDistanceMapOptions
    {
        public ContoursDistanceMapOptions? Opt;

        public _InOptMut_ContoursDistanceMapOptions() {}
        public _InOptMut_ContoursDistanceMapOptions(ContoursDistanceMapOptions value) {Opt = value;}
        public static implicit operator _InOptMut_ContoursDistanceMapOptions(ContoursDistanceMapOptions value) {return new(value);}
    }

    /// This is used for optional parameters of class `ContoursDistanceMapOptions` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ContoursDistanceMapOptions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContoursDistanceMapOptions`/`Const_ContoursDistanceMapOptions` to pass it to the function.
    public class _InOptConst_ContoursDistanceMapOptions
    {
        public Const_ContoursDistanceMapOptions? Opt;

        public _InOptConst_ContoursDistanceMapOptions() {}
        public _InOptConst_ContoursDistanceMapOptions(Const_ContoursDistanceMapOptions value) {Opt = value;}
        public static implicit operator _InOptConst_ContoursDistanceMapOptions(Const_ContoursDistanceMapOptions value) {return new(value);}
    }

    /// fill another distance map pair with gradients across X and Y axes of the argument map
    /// Generated from function `MR::combineXYderivativeMaps`.
    public static unsafe MR.Misc._Moved<MR.DistanceMap> CombineXYderivativeMaps(MR.Std._ByValue_Pair_MRDistanceMap_MRDistanceMap XYderivativeMaps)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_combineXYderivativeMaps", ExactSpelling = true)]
        extern static MR.DistanceMap._Underlying *__MR_combineXYderivativeMaps(MR.Misc._PassBy XYderivativeMaps_pass_by, MR.Std.Pair_MRDistanceMap_MRDistanceMap._Underlying *XYderivativeMaps);
        return MR.Misc.Move(new MR.DistanceMap(__MR_combineXYderivativeMaps(XYderivativeMaps.PassByMode, XYderivativeMaps.Value is not null ? XYderivativeMaps.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// computes distance (height) map for given projection parameters
    /// using float-precision for finding ray-mesh intersections, which is faster but less reliable
    /// Generated from function `MR::computeDistanceMap`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.DistanceMap> ComputeDistanceMap(MR.Const_MeshPart mp, MR.Const_MeshToDistanceMapParams params_, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Std.Vector_MRMeshTriPoint? outSamples = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeDistanceMap", ExactSpelling = true)]
        extern static MR.DistanceMap._Underlying *__MR_computeDistanceMap(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshToDistanceMapParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Std.Vector_MRMeshTriPoint._Underlying *outSamples);
        return MR.Misc.Move(new MR.DistanceMap(__MR_computeDistanceMap(mp._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, outSamples is not null ? outSamples._UnderlyingPtr : null), is_owning: true));
    }

    /// computes distance (height) map for given projection parameters
    /// using double-precision for finding ray-mesh intersections, which is slower but more reliable
    /// Generated from function `MR::computeDistanceMapD`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.DistanceMap> ComputeDistanceMapD(MR.Const_MeshPart mp, MR.Const_MeshToDistanceMapParams params_, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null, MR.Std.Vector_MRMeshTriPoint? outSamples = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeDistanceMapD", ExactSpelling = true)]
        extern static MR.DistanceMap._Underlying *__MR_computeDistanceMapD(MR.Const_MeshPart._Underlying *mp, MR.Const_MeshToDistanceMapParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb, MR.Std.Vector_MRMeshTriPoint._Underlying *outSamples);
        return MR.Misc.Move(new MR.DistanceMap(__MR_computeDistanceMapD(mp._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null, outSamples is not null ? outSamples._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Computes distance of 2d contours according ContourToDistanceMapParams
    * \param options - optional input and output options for distance map calculation, find more \ref ContoursDistanceMapOptions
    */
    /// Generated from function `MR::distanceMapFromContours`.
    /// Parameter `options` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.DistanceMap> DistanceMapFromContours(MR.Const_Polyline2 contours, MR.Const_ContourToDistanceMapParams params_, MR.Const_ContoursDistanceMapOptions? options = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapFromContours_3", ExactSpelling = true)]
        extern static MR.DistanceMap._Underlying *__MR_distanceMapFromContours_3(MR.Const_Polyline2._Underlying *contours, MR.Const_ContourToDistanceMapParams._Underlying *params_, MR.Const_ContoursDistanceMapOptions._Underlying *options);
        return MR.Misc.Move(new MR.DistanceMap(__MR_distanceMapFromContours_3(contours._UnderlyingPtr, params_._UnderlyingPtr, options is not null ? options._UnderlyingPtr : null), is_owning: true));
    }

    /**
    * \brief Computes distance of 2d contours according ContourToDistanceMapParams
    * \param distMap - preallocated distance map
    * \param options - optional input and output options for distance map calculation, find more \ref ContoursDistanceMapOptions
    */
    /// Generated from function `MR::distanceMapFromContours`.
    /// Parameter `options` defaults to `{}`.
    public static unsafe void DistanceMapFromContours(MR.DistanceMap distMap, MR.Const_Polyline2 polyline, MR.Const_ContourToDistanceMapParams params_, MR.Const_ContoursDistanceMapOptions? options = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapFromContours_4", ExactSpelling = true)]
        extern static void __MR_distanceMapFromContours_4(MR.DistanceMap._Underlying *distMap, MR.Const_Polyline2._Underlying *polyline, MR.Const_ContourToDistanceMapParams._Underlying *params_, MR.Const_ContoursDistanceMapOptions._Underlying *options);
        __MR_distanceMapFromContours_4(distMap._UnderlyingPtr, polyline._UnderlyingPtr, params_._UnderlyingPtr, options is not null ? options._UnderlyingPtr : null);
    }

    /// Makes distance map and filter out pixels with large (>threshold) distance between closest points on contour in neighbor pixels
    /// Converts such points back in 3d space and return
    /// \note that polyline topology should be consistently oriented
    /// Generated from function `MR::edgePointsFromContours`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> EdgePointsFromContours(MR.Const_Polyline2 polyline, float pixelSize, float threshold)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_edgePointsFromContours", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_edgePointsFromContours(MR.Const_Polyline2._Underlying *polyline, float pixelSize, float threshold);
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_edgePointsFromContours(polyline._UnderlyingPtr, pixelSize, threshold), is_owning: true));
    }

    /// converts distance map to 2d iso-lines:
    /// iso-lines are created in space DistanceMap ( plane OXY with pixelSize = (1, 1) )
    /// Generated from function `MR::distanceMapTo2DIsoPolyline`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> DistanceMapTo2DIsoPolyline(MR.Const_DistanceMap distMap, float isoValue)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapTo2DIsoPolyline_2", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_distanceMapTo2DIsoPolyline_2(MR.Const_DistanceMap._Underlying *distMap, float isoValue);
        return MR.Misc.Move(new MR.Polyline2(__MR_distanceMapTo2DIsoPolyline_2(distMap._UnderlyingPtr, isoValue), is_owning: true));
    }

    /// iso-lines are created in real space ( plane OXY with parameters according ContourToDistanceMapParams )
    /// Generated from function `MR::distanceMapTo2DIsoPolyline`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> DistanceMapTo2DIsoPolyline(MR.Const_DistanceMap distMap, MR.Const_ContourToDistanceMapParams params_, float isoValue)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapTo2DIsoPolyline_3_MR_ContourToDistanceMapParams", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_distanceMapTo2DIsoPolyline_3_MR_ContourToDistanceMapParams(MR.Const_DistanceMap._Underlying *distMap, MR.Const_ContourToDistanceMapParams._Underlying *params_, float isoValue);
        return MR.Misc.Move(new MR.Polyline2(__MR_distanceMapTo2DIsoPolyline_3_MR_ContourToDistanceMapParams(distMap._UnderlyingPtr, params_._UnderlyingPtr, isoValue), is_owning: true));
    }

    /// computes iso-lines of distance map corresponding to given iso-value;
    /// in second returns the transformation from 0XY plane to world;
    /// \param useDepth true - the isolines will be located on distance map surface, false - isolines for any iso-value will be located on the common plane xf(0XY)
    /// Generated from function `MR::distanceMapTo2DIsoPolyline`.
    /// Parameter `useDepth` defaults to `false`.
    public static unsafe MR.Misc._Moved<MR.Std.Pair_MRPolyline2_MRAffineXf3f> DistanceMapTo2DIsoPolyline(MR.Const_DistanceMap distMap, MR.Const_AffineXf3f xf, float isoValue, bool? useDepth = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapTo2DIsoPolyline_4", ExactSpelling = true)]
        extern static MR.Std.Pair_MRPolyline2_MRAffineXf3f._Underlying *__MR_distanceMapTo2DIsoPolyline_4(MR.Const_DistanceMap._Underlying *distMap, MR.Const_AffineXf3f._Underlying *xf, float isoValue, byte *useDepth);
        byte __deref_useDepth = useDepth.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Std.Pair_MRPolyline2_MRAffineXf3f(__MR_distanceMapTo2DIsoPolyline_4(distMap._UnderlyingPtr, xf._UnderlyingPtr, isoValue, useDepth.HasValue ? &__deref_useDepth : null), is_owning: true));
    }

    /// Generated from function `MR::distanceMapTo2DIsoPolyline`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> DistanceMapTo2DIsoPolyline(MR.Const_DistanceMap distMap, float pixelSize, float isoValue)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapTo2DIsoPolyline_3_float", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_distanceMapTo2DIsoPolyline_3_float(MR.Const_DistanceMap._Underlying *distMap, float pixelSize, float isoValue);
        return MR.Misc.Move(new MR.Polyline2(__MR_distanceMapTo2DIsoPolyline_3_float(distMap._UnderlyingPtr, pixelSize, isoValue), is_owning: true));
    }

    /// constructs an offset contour for given polyline
    /// Generated from function `MR::polylineOffset`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> PolylineOffset(MR.Const_Polyline2 polyline, float pixelSize, float offset)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_polylineOffset", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_polylineOffset(MR.Const_Polyline2._Underlying *polyline, float pixelSize, float offset);
        return MR.Misc.Move(new MR.Polyline2(__MR_polylineOffset(polyline._UnderlyingPtr, pixelSize, offset), is_owning: true));
    }

    /**
    * \brief computes the union of the shapes bounded by input 2d contours
    * \return the boundary of the union
    * \details input contours must be closed within the area of distance map and be consistently oriented (clockwise, that is leaving the bounded shapes from the left).
    * the value of params.withSign must be true (checked with assert() inside the function)
    * \note that polyline topology should be consistently oriented
    */
    /// Generated from function `MR::contourUnion`.
    /// Parameter `offsetInside` defaults to `0`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> ContourUnion(MR.Const_Polyline2 contoursA, MR.Const_Polyline2 contoursB, MR.Const_ContourToDistanceMapParams params_, float? offsetInside = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_contourUnion", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_contourUnion(MR.Const_Polyline2._Underlying *contoursA, MR.Const_Polyline2._Underlying *contoursB, MR.Const_ContourToDistanceMapParams._Underlying *params_, float *offsetInside);
        float __deref_offsetInside = offsetInside.GetValueOrDefault();
        return MR.Misc.Move(new MR.Polyline2(__MR_contourUnion(contoursA._UnderlyingPtr, contoursB._UnderlyingPtr, params_._UnderlyingPtr, offsetInside.HasValue ? &__deref_offsetInside : null), is_owning: true));
    }

    /**
    * \brief computes the intersection of the shapes bounded by input 2d contours
    * \return the boundary of the intersection
    * \details input contours must be closed within the area of distance map and be consistently oriented (clockwise, that is leaving the bounded shapes from the left).
    * the value of params.withSign must be true (checked with assert() inside the function)
    * \note that polyline topology should be consistently oriented
    */
    /// Generated from function `MR::contourIntersection`.
    /// Parameter `offsetInside` defaults to `0.0f`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> ContourIntersection(MR.Const_Polyline2 contoursA, MR.Const_Polyline2 contoursB, MR.Const_ContourToDistanceMapParams params_, float? offsetInside = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_contourIntersection", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_contourIntersection(MR.Const_Polyline2._Underlying *contoursA, MR.Const_Polyline2._Underlying *contoursB, MR.Const_ContourToDistanceMapParams._Underlying *params_, float *offsetInside);
        float __deref_offsetInside = offsetInside.GetValueOrDefault();
        return MR.Misc.Move(new MR.Polyline2(__MR_contourIntersection(contoursA._UnderlyingPtr, contoursB._UnderlyingPtr, params_._UnderlyingPtr, offsetInside.HasValue ? &__deref_offsetInside : null), is_owning: true));
    }

    /**
    * \brief computes the difference between the shapes bounded by contoursA and the shapes bounded by contoursB
    * \return the boundary of the difference
    * \details input contours must be closed within the area of distance map and be consistently oriented (clockwise, that is leaving the bounded shapes from the left).
    * the value of params.withSign must be true (checked with assert() inside the function)
    * \note that polyline topology should be consistently oriented
    */
    /// Generated from function `MR::contourSubtract`.
    /// Parameter `offsetInside` defaults to `0.0f`.
    public static unsafe MR.Misc._Moved<MR.Polyline2> ContourSubtract(MR.Const_Polyline2 contoursA, MR.Const_Polyline2 contoursB, MR.Const_ContourToDistanceMapParams params_, float? offsetInside = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_contourSubtract", ExactSpelling = true)]
        extern static MR.Polyline2._Underlying *__MR_contourSubtract(MR.Const_Polyline2._Underlying *contoursA, MR.Const_Polyline2._Underlying *contoursB, MR.Const_ContourToDistanceMapParams._Underlying *params_, float *offsetInside);
        float __deref_offsetInside = offsetInside.GetValueOrDefault();
        return MR.Misc.Move(new MR.Polyline2(__MR_contourSubtract(contoursA._UnderlyingPtr, contoursB._UnderlyingPtr, params_._UnderlyingPtr, offsetInside.HasValue ? &__deref_offsetInside : null), is_owning: true));
    }

    /// converts distance map into mesh and applies a transformation to all points
    /// Generated from function `MR::distanceMapToMesh`.
    /// Parameter `cb` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> DistanceMapToMesh(MR.Const_DistanceMap distMap, MR.Const_AffineXf3f toWorld, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_distanceMapToMesh", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_distanceMapToMesh(MR.Const_DistanceMap._Underlying *distMap, MR.Const_AffineXf3f._Underlying *toWorld, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_distanceMapToMesh(distMap._UnderlyingPtr, toWorld._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
    }

    /// export distance map to a grayscale image
    /// \param threshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid
    /// Generated from function `MR::convertDistanceMapToImage`.
    /// Parameter `threshold` defaults to `1.0f/255`.
    public static unsafe MR.Misc._Moved<MR.Image> ConvertDistanceMapToImage(MR.Const_DistanceMap distMap, float? threshold = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertDistanceMapToImage", ExactSpelling = true)]
        extern static MR.Image._Underlying *__MR_convertDistanceMapToImage(MR.Const_DistanceMap._Underlying *distMap, float *threshold);
        float __deref_threshold = threshold.GetValueOrDefault();
        return MR.Misc.Move(new MR.Image(__MR_convertDistanceMapToImage(distMap._UnderlyingPtr, threshold.HasValue ? &__deref_threshold : null), is_owning: true));
    }

    /// load distance map from a grayscale image:
    /// \param threshold - threshold of valid values [0.; 1.]. pixel with color less then threshold set invalid
    /// \param invert - whether to invert values (min is white) or leave them as is (min is block)
    /// Generated from function `MR::convertImageToDistanceMap`.
    /// Parameter `threshold` defaults to `1.0f/255`.
    /// Parameter `invert` defaults to `true`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRDistanceMap_StdString> ConvertImageToDistanceMap(MR.Const_Image image, float? threshold = null, bool? invert = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_convertImageToDistanceMap", ExactSpelling = true)]
        extern static MR.Expected_MRDistanceMap_StdString._Underlying *__MR_convertImageToDistanceMap(MR.Const_Image._Underlying *image, float *threshold, byte *invert);
        float __deref_threshold = threshold.GetValueOrDefault();
        byte __deref_invert = invert.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.Expected_MRDistanceMap_StdString(__MR_convertImageToDistanceMap(image._UnderlyingPtr, threshold.HasValue ? &__deref_threshold : null, invert.HasValue ? &__deref_invert : null), is_owning: true));
    }
}
