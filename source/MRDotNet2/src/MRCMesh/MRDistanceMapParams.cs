public static partial class MR
{
    /// Generated from class `MR::MeshToDistanceMapParams`.
    /// This is the const half of the class.
    public class Const_MeshToDistanceMapParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshToDistanceMapParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshToDistanceMapParams_Destroy(_Underlying *_this);
            __MR_MeshToDistanceMapParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshToDistanceMapParams() {Dispose(false);}

        ///< Cartesian range vector between distance map borders in X direction
        public unsafe MR.Const_Vector3f XRange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_xRange", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshToDistanceMapParams_Get_xRange(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_Get_xRange(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< Cartesian range vector between distance map borders in Y direction
        public unsafe MR.Const_Vector3f YRange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_yRange", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshToDistanceMapParams_Get_yRange(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_Get_yRange(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< direction of intersection ray
        public unsafe MR.Const_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_direction", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshToDistanceMapParams_Get_direction(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_Get_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< location of (0,0) pixel with value 0.f
        public unsafe MR.Const_Vector3f OrgPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_orgPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshToDistanceMapParams_Get_orgPoint(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_Get_orgPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< out of limits intersections will be set to non-valid
        public unsafe bool UseDistanceLimits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_useDistanceLimits", ExactSpelling = true)]
                extern static bool *__MR_MeshToDistanceMapParams_Get_useDistanceLimits(_Underlying *_this);
                return *__MR_MeshToDistanceMapParams_Get_useDistanceLimits(_UnderlyingPtr);
            }
        }

        ///< allows to find intersections in backward to direction vector with negative values
        public unsafe bool AllowNegativeValues
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_allowNegativeValues", ExactSpelling = true)]
                extern static bool *__MR_MeshToDistanceMapParams_Get_allowNegativeValues(_Underlying *_this);
                return *__MR_MeshToDistanceMapParams_Get_allowNegativeValues(_UnderlyingPtr);
            }
        }

        ///< Using of this parameter depends on useDistanceLimits
        public unsafe float MinValue
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_minValue", ExactSpelling = true)]
                extern static float *__MR_MeshToDistanceMapParams_Get_minValue(_Underlying *_this);
                return *__MR_MeshToDistanceMapParams_Get_minValue(_UnderlyingPtr);
            }
        }

        ///< Using of this parameter depends on useDistanceLimits
        public unsafe float MaxValue
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_maxValue", ExactSpelling = true)]
                extern static float *__MR_MeshToDistanceMapParams_Get_maxValue(_Underlying *_this);
                return *__MR_MeshToDistanceMapParams_Get_maxValue(_UnderlyingPtr);
            }
        }

        ///< resolution of distance map
        public unsafe MR.Const_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Get_resolution", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_MeshToDistanceMapParams_Get_resolution(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_Get_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshToDistanceMapParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_MeshToDistanceMapParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_ConstructFromAnother(MR.MeshToDistanceMapParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// direction vector shows the projections vector to the distance map for points on model
        /// yRange and xRange directions make orthonormal basis with direction
        /// see Vector3<T>::perpendicular() for more details
        /// All Output Distance map values will be positive
        /// usePreciseBoundingBox false (fast): use general (cached) bounding box with applied rotation
        /// usePreciseBoundingBox true (slow): compute bounding box from points with respect to rotation
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        /// Parameter `usePreciseBoundingBox` defaults to `false`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_Vector3f direction, MR.Const_Vector2i resolution, MR.Const_MeshPart mp, bool? usePreciseBoundingBox = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2i", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2i(MR.Const_Vector3f._Underlying *direction, MR.Const_Vector2i._Underlying *resolution, MR.Const_MeshPart._Underlying *mp, byte *usePreciseBoundingBox);
            byte __deref_usePreciseBoundingBox = usePreciseBoundingBox.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2i(direction._UnderlyingPtr, resolution._UnderlyingPtr, mp._UnderlyingPtr, usePreciseBoundingBox.HasValue ? &__deref_usePreciseBoundingBox : null);
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        /// Parameter `usePreciseBoundingBox` defaults to `false`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_Vector3f direction, MR.Const_Vector2f pixelSize, MR.Const_MeshPart mp, bool? usePreciseBoundingBox = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2f", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2f(MR.Const_Vector3f._Underlying *direction, MR.Const_Vector2f._Underlying *pixelSize, MR.Const_MeshPart._Underlying *mp, byte *usePreciseBoundingBox);
            byte __deref_usePreciseBoundingBox = usePreciseBoundingBox.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2f(direction._UnderlyingPtr, pixelSize._UnderlyingPtr, mp._UnderlyingPtr, usePreciseBoundingBox.HasValue ? &__deref_usePreciseBoundingBox : null);
        }

        /// input matrix should be orthonormal!
        /// rotation.z - direction
        /// rotation.x * (box X length) - xRange
        /// rotation.y * (box Y length) - yRange
        /// All Output Distance map values will be positive
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_Matrix3f rotation, MR.Const_Vector3f origin, MR.Const_Vector2i resolution, MR.Const_Vector2f size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2i", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2i(MR.Const_Matrix3f._Underlying *rotation, MR.Const_Vector3f._Underlying *origin, MR.Const_Vector2i._Underlying *resolution, MR.Const_Vector2f._Underlying *size);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2i(rotation._UnderlyingPtr, origin._UnderlyingPtr, resolution._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_Matrix3f rotation, MR.Const_Vector3f origin, MR.Const_Vector2f pixelSize, MR.Const_Vector2i resolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2f", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2f(MR.Const_Matrix3f._Underlying *rotation, MR.Const_Vector3f._Underlying *origin, MR.Const_Vector2f._Underlying *pixelSize, MR.Const_Vector2i._Underlying *resolution);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2f(rotation._UnderlyingPtr, origin._UnderlyingPtr, pixelSize._UnderlyingPtr, resolution._UnderlyingPtr);
        }

        /// input matrix should be orthonormal!
        /// rotation.z - direction
        /// rotation.x * (box X length) - xRange
        /// rotation.y * (box Y length) - yRange
        /// All Output Distance map values will be positive
        /// usePreciseBoundingBox false (fast): use general (cached) bounding box with applied rotation
        /// usePreciseBoundingBox true (slow): compute bounding box from points with respect to rotation
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        /// Parameter `usePreciseBoundingBox` defaults to `false`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_Matrix3f rotation, MR.Const_Vector2i resolution, MR.Const_MeshPart mp, bool? usePreciseBoundingBox = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector2i_ref", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector2i_ref(MR.Const_Matrix3f._Underlying *rotation, MR.Const_Vector2i._Underlying *resolution, MR.Const_MeshPart._Underlying *mp, byte *usePreciseBoundingBox);
            byte __deref_usePreciseBoundingBox = usePreciseBoundingBox.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector2i_ref(rotation._UnderlyingPtr, resolution._UnderlyingPtr, mp._UnderlyingPtr, usePreciseBoundingBox.HasValue ? &__deref_usePreciseBoundingBox : null);
        }

        /// The most general constructor. Use it if you have to find special view, resolution,
        /// distance map with visual the part of the model etc.
        /// All params match is in the user responsibility
        /// xf.b - origin point: pixel(0,0) with value 0.
        /// xf.A.z - direction
        /// xf.A.x - xRange
        /// xf.A.y - yRange
        /// All Output Distance map values could be positive and negative by default. Set allowNegativeValues to false if negative values are not required
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_AffineXf3f xf, MR.Const_Vector2i resolution, MR.Const_Vector2f size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_3_MR_Vector2i", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_3_MR_Vector2i(MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector2i._Underlying *resolution, MR.Const_Vector2f._Underlying *size);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_3_MR_Vector2i(xf._UnderlyingPtr, resolution._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe Const_MeshToDistanceMapParams(MR.Const_AffineXf3f xf, MR.Const_Vector2f pixelSize, MR.Const_Vector2i resolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_3_MR_Vector2f", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_3_MR_Vector2f(MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector2f._Underlying *pixelSize, MR.Const_Vector2i._Underlying *resolution);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_3_MR_Vector2f(xf._UnderlyingPtr, pixelSize._UnderlyingPtr, resolution._UnderlyingPtr);
        }

        /// converts in transformation
        /// Generated from conversion operator `MR::MeshToDistanceMapParams::operator MR::AffineXf3f`.
        public static unsafe implicit operator MR.AffineXf3f(MR.Const_MeshToDistanceMapParams _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_ConvertTo_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshToDistanceMapParams_ConvertTo_MR_AffineXf3f(MR.Const_MeshToDistanceMapParams._Underlying *_this);
            return __MR_MeshToDistanceMapParams_ConvertTo_MR_AffineXf3f(_this._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshToDistanceMapParams::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_xf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_MeshToDistanceMapParams_xf(_Underlying *_this);
            return __MR_MeshToDistanceMapParams_xf(_UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshToDistanceMapParams`.
    /// This is the non-const half of the class.
    public class MeshToDistanceMapParams : Const_MeshToDistanceMapParams
    {
        internal unsafe MeshToDistanceMapParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< Cartesian range vector between distance map borders in X direction
        public new unsafe MR.Mut_Vector3f XRange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_xRange", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshToDistanceMapParams_GetMutable_xRange(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_GetMutable_xRange(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< Cartesian range vector between distance map borders in Y direction
        public new unsafe MR.Mut_Vector3f YRange
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_yRange", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshToDistanceMapParams_GetMutable_yRange(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_GetMutable_yRange(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< direction of intersection ray
        public new unsafe MR.Mut_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_direction", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshToDistanceMapParams_GetMutable_direction(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_GetMutable_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< location of (0,0) pixel with value 0.f
        public new unsafe MR.Mut_Vector3f OrgPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_orgPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshToDistanceMapParams_GetMutable_orgPoint(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_GetMutable_orgPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< out of limits intersections will be set to non-valid
        public new unsafe ref bool UseDistanceLimits
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_useDistanceLimits", ExactSpelling = true)]
                extern static bool *__MR_MeshToDistanceMapParams_GetMutable_useDistanceLimits(_Underlying *_this);
                return ref *__MR_MeshToDistanceMapParams_GetMutable_useDistanceLimits(_UnderlyingPtr);
            }
        }

        ///< allows to find intersections in backward to direction vector with negative values
        public new unsafe ref bool AllowNegativeValues
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_allowNegativeValues", ExactSpelling = true)]
                extern static bool *__MR_MeshToDistanceMapParams_GetMutable_allowNegativeValues(_Underlying *_this);
                return ref *__MR_MeshToDistanceMapParams_GetMutable_allowNegativeValues(_UnderlyingPtr);
            }
        }

        ///< Using of this parameter depends on useDistanceLimits
        public new unsafe ref float MinValue
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_minValue", ExactSpelling = true)]
                extern static float *__MR_MeshToDistanceMapParams_GetMutable_minValue(_Underlying *_this);
                return ref *__MR_MeshToDistanceMapParams_GetMutable_minValue(_UnderlyingPtr);
            }
        }

        ///< Using of this parameter depends on useDistanceLimits
        public new unsafe ref float MaxValue
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_maxValue", ExactSpelling = true)]
                extern static float *__MR_MeshToDistanceMapParams_GetMutable_maxValue(_Underlying *_this);
                return ref *__MR_MeshToDistanceMapParams_GetMutable_maxValue(_UnderlyingPtr);
            }
        }

        ///< resolution of distance map
        public new unsafe MR.Mut_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_GetMutable_resolution", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_MeshToDistanceMapParams_GetMutable_resolution(_Underlying *_this);
                return new(__MR_MeshToDistanceMapParams_GetMutable_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshToDistanceMapParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe MeshToDistanceMapParams(MR.Const_MeshToDistanceMapParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_ConstructFromAnother(MR.MeshToDistanceMapParams._Underlying *_other);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// direction vector shows the projections vector to the distance map for points on model
        /// yRange and xRange directions make orthonormal basis with direction
        /// see Vector3<T>::perpendicular() for more details
        /// All Output Distance map values will be positive
        /// usePreciseBoundingBox false (fast): use general (cached) bounding box with applied rotation
        /// usePreciseBoundingBox true (slow): compute bounding box from points with respect to rotation
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        /// Parameter `usePreciseBoundingBox` defaults to `false`.
        public unsafe MeshToDistanceMapParams(MR.Const_Vector3f direction, MR.Const_Vector2i resolution, MR.Const_MeshPart mp, bool? usePreciseBoundingBox = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2i", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2i(MR.Const_Vector3f._Underlying *direction, MR.Const_Vector2i._Underlying *resolution, MR.Const_MeshPart._Underlying *mp, byte *usePreciseBoundingBox);
            byte __deref_usePreciseBoundingBox = usePreciseBoundingBox.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2i(direction._UnderlyingPtr, resolution._UnderlyingPtr, mp._UnderlyingPtr, usePreciseBoundingBox.HasValue ? &__deref_usePreciseBoundingBox : null);
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        /// Parameter `usePreciseBoundingBox` defaults to `false`.
        public unsafe MeshToDistanceMapParams(MR.Const_Vector3f direction, MR.Const_Vector2f pixelSize, MR.Const_MeshPart mp, bool? usePreciseBoundingBox = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2f", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2f(MR.Const_Vector3f._Underlying *direction, MR.Const_Vector2f._Underlying *pixelSize, MR.Const_MeshPart._Underlying *mp, byte *usePreciseBoundingBox);
            byte __deref_usePreciseBoundingBox = usePreciseBoundingBox.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Vector3f_ref_MR_Vector2f(direction._UnderlyingPtr, pixelSize._UnderlyingPtr, mp._UnderlyingPtr, usePreciseBoundingBox.HasValue ? &__deref_usePreciseBoundingBox : null);
        }

        /// input matrix should be orthonormal!
        /// rotation.z - direction
        /// rotation.x * (box X length) - xRange
        /// rotation.y * (box Y length) - yRange
        /// All Output Distance map values will be positive
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe MeshToDistanceMapParams(MR.Const_Matrix3f rotation, MR.Const_Vector3f origin, MR.Const_Vector2i resolution, MR.Const_Vector2f size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2i", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2i(MR.Const_Matrix3f._Underlying *rotation, MR.Const_Vector3f._Underlying *origin, MR.Const_Vector2i._Underlying *resolution, MR.Const_Vector2f._Underlying *size);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2i(rotation._UnderlyingPtr, origin._UnderlyingPtr, resolution._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe MeshToDistanceMapParams(MR.Const_Matrix3f rotation, MR.Const_Vector3f origin, MR.Const_Vector2f pixelSize, MR.Const_Vector2i resolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2f", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2f(MR.Const_Matrix3f._Underlying *rotation, MR.Const_Vector3f._Underlying *origin, MR.Const_Vector2f._Underlying *pixelSize, MR.Const_Vector2i._Underlying *resolution);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector3f_ref_MR_Vector2f(rotation._UnderlyingPtr, origin._UnderlyingPtr, pixelSize._UnderlyingPtr, resolution._UnderlyingPtr);
        }

        /// input matrix should be orthonormal!
        /// rotation.z - direction
        /// rotation.x * (box X length) - xRange
        /// rotation.y * (box Y length) - yRange
        /// All Output Distance map values will be positive
        /// usePreciseBoundingBox false (fast): use general (cached) bounding box with applied rotation
        /// usePreciseBoundingBox true (slow): compute bounding box from points with respect to rotation
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        /// Parameter `usePreciseBoundingBox` defaults to `false`.
        public unsafe MeshToDistanceMapParams(MR.Const_Matrix3f rotation, MR.Const_Vector2i resolution, MR.Const_MeshPart mp, bool? usePreciseBoundingBox = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector2i_ref", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector2i_ref(MR.Const_Matrix3f._Underlying *rotation, MR.Const_Vector2i._Underlying *resolution, MR.Const_MeshPart._Underlying *mp, byte *usePreciseBoundingBox);
            byte __deref_usePreciseBoundingBox = usePreciseBoundingBox.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_4_const_MR_Matrix3f_ref_const_MR_Vector2i_ref(rotation._UnderlyingPtr, resolution._UnderlyingPtr, mp._UnderlyingPtr, usePreciseBoundingBox.HasValue ? &__deref_usePreciseBoundingBox : null);
        }

        /// The most general constructor. Use it if you have to find special view, resolution,
        /// distance map with visual the part of the model etc.
        /// All params match is in the user responsibility
        /// xf.b - origin point: pixel(0,0) with value 0.
        /// xf.A.z - direction
        /// xf.A.x - xRange
        /// xf.A.y - yRange
        /// All Output Distance map values could be positive and negative by default. Set allowNegativeValues to false if negative values are not required
        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe MeshToDistanceMapParams(MR.Const_AffineXf3f xf, MR.Const_Vector2i resolution, MR.Const_Vector2f size) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_3_MR_Vector2i", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_3_MR_Vector2i(MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector2i._Underlying *resolution, MR.Const_Vector2f._Underlying *size);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_3_MR_Vector2i(xf._UnderlyingPtr, resolution._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshToDistanceMapParams::MeshToDistanceMapParams`.
        public unsafe MeshToDistanceMapParams(MR.Const_AffineXf3f xf, MR.Const_Vector2f pixelSize, MR.Const_Vector2i resolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_Construct_3_MR_Vector2f", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_Construct_3_MR_Vector2f(MR.Const_AffineXf3f._Underlying *xf, MR.Const_Vector2f._Underlying *pixelSize, MR.Const_Vector2i._Underlying *resolution);
            _UnderlyingPtr = __MR_MeshToDistanceMapParams_Construct_3_MR_Vector2f(xf._UnderlyingPtr, pixelSize._UnderlyingPtr, resolution._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshToDistanceMapParams::operator=`.
        public unsafe MR.MeshToDistanceMapParams Assign(MR.Const_MeshToDistanceMapParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshToDistanceMapParams._Underlying *__MR_MeshToDistanceMapParams_AssignFromAnother(_Underlying *_this, MR.MeshToDistanceMapParams._Underlying *_other);
            return new(__MR_MeshToDistanceMapParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// if distance is not in set range, pixel became invalid
        /// default value: false. Any distance will be applied (include negative)
        /// Generated from method `MR::MeshToDistanceMapParams::setDistanceLimits`.
        public unsafe void SetDistanceLimits(float min, float max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshToDistanceMapParams_setDistanceLimits", ExactSpelling = true)]
            extern static void __MR_MeshToDistanceMapParams_setDistanceLimits(_Underlying *_this, float min, float max);
            __MR_MeshToDistanceMapParams_setDistanceLimits(_UnderlyingPtr, min, max);
        }
    }

    /// This is used for optional parameters of class `MeshToDistanceMapParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshToDistanceMapParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToDistanceMapParams`/`Const_MeshToDistanceMapParams` directly.
    public class _InOptMut_MeshToDistanceMapParams
    {
        public MeshToDistanceMapParams? Opt;

        public _InOptMut_MeshToDistanceMapParams() {}
        public _InOptMut_MeshToDistanceMapParams(MeshToDistanceMapParams value) {Opt = value;}
        public static implicit operator _InOptMut_MeshToDistanceMapParams(MeshToDistanceMapParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshToDistanceMapParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshToDistanceMapParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshToDistanceMapParams`/`Const_MeshToDistanceMapParams` to pass it to the function.
    public class _InOptConst_MeshToDistanceMapParams
    {
        public Const_MeshToDistanceMapParams? Opt;

        public _InOptConst_MeshToDistanceMapParams() {}
        public _InOptConst_MeshToDistanceMapParams(Const_MeshToDistanceMapParams value) {Opt = value;}
        public static implicit operator _InOptConst_MeshToDistanceMapParams(Const_MeshToDistanceMapParams value) {return new(value);}
    }

    /// Structure with parameters to generate DistanceMap by Contours
    /// Generated from class `MR::ContourToDistanceMapParams`.
    /// This is the const half of the class.
    public class Const_ContourToDistanceMapParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_ContourToDistanceMapParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Destroy", ExactSpelling = true)]
            extern static void __MR_ContourToDistanceMapParams_Destroy(_Underlying *_this);
            __MR_ContourToDistanceMapParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_ContourToDistanceMapParams() {Dispose(false);}

        ///< pixel size
        public unsafe MR.Const_Vector2f PixelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Get_pixelSize", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_ContourToDistanceMapParams_Get_pixelSize(_Underlying *_this);
                return new(__MR_ContourToDistanceMapParams_Get_pixelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< distance map size
        public unsafe MR.Const_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Get_resolution", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_ContourToDistanceMapParams_Get_resolution(_Underlying *_this);
                return new(__MR_ContourToDistanceMapParams_Get_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< coordinates of origin area corner
        public unsafe MR.Const_Vector2f OrgPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Get_orgPoint", ExactSpelling = true)]
                extern static MR.Const_Vector2f._Underlying *__MR_ContourToDistanceMapParams_Get_orgPoint(_Underlying *_this);
                return new(__MR_ContourToDistanceMapParams_Get_orgPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< allows calculate negative values of distance (inside closed and correctly oriented (CW) contours)
        public unsafe bool WithSign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Get_withSign", ExactSpelling = true)]
                extern static bool *__MR_ContourToDistanceMapParams_Get_withSign(_Underlying *_this);
                return *__MR_ContourToDistanceMapParams_Get_withSign(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_ContourToDistanceMapParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        public unsafe Const_ContourToDistanceMapParams(MR.Const_ContourToDistanceMapParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_ConstructFromAnother(MR.ContourToDistanceMapParams._Underlying *_other);
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Ctor, calculating pixelSize by areaSize & dmapSize
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe Const_ContourToDistanceMapParams(MR.Const_Vector2i resolution, MR.Const_Vector2f oriPoint, MR.Const_Vector2f areaSize, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_MR_Vector2f", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_MR_Vector2f(MR.Const_Vector2i._Underlying *resolution, MR.Const_Vector2f._Underlying *oriPoint, MR.Const_Vector2f._Underlying *areaSize, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_MR_Vector2f(resolution._UnderlyingPtr, oriPoint._UnderlyingPtr, areaSize._UnderlyingPtr, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Ctor, calculating pixelSize & oriPoint by box parameters
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe Const_ContourToDistanceMapParams(MR.Const_Vector2i resolution, MR.Const_Box2f box, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_3", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_3(MR.Const_Vector2i._Underlying *resolution, MR.Const_Box2f._Underlying *box, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_3(resolution._UnderlyingPtr, box._UnderlyingPtr, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Ctor, calculating pixelSize & oriPoint by contours box + offset
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe Const_ContourToDistanceMapParams(MR.Const_Vector2i resolution, MR.Std.Const_Vector_StdVectorMRVector2f contours, float offset, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_std_vector_std_vector_MR_Vector2f(MR.Const_Vector2i._Underlying *resolution, MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, float offset, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_std_vector_std_vector_MR_Vector2f(resolution._UnderlyingPtr, contours._UnderlyingPtr, offset, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Ctor, calculating resolution & oriPoint by contours box + offset
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe Const_ContourToDistanceMapParams(float pixelSize, MR.Std.Const_Vector_StdVectorMRVector2f contours, float offset, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_4_float", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_4_float(float pixelSize, MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, float offset, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_4_float(pixelSize, contours._UnderlyingPtr, offset, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        public unsafe Const_ContourToDistanceMapParams(MR.Const_DistanceMapToWorld toWorld) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_1", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_1(MR.Const_DistanceMapToWorld._Underlying *toWorld);
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_1(toWorld._UnderlyingPtr);
        }

        /// converts in transformation
        /// Generated from conversion operator `MR::ContourToDistanceMapParams::operator MR::AffineXf3f`.
        public static unsafe implicit operator MR.AffineXf3f(MR.Const_ContourToDistanceMapParams _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_ConvertTo_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ContourToDistanceMapParams_ConvertTo_MR_AffineXf3f(MR.Const_ContourToDistanceMapParams._Underlying *_this);
            return __MR_ContourToDistanceMapParams_ConvertTo_MR_AffineXf3f(_this._UnderlyingPtr);
        }

        /// get world 2d coordinate (respects origin point and pixel size)
        /// point - coordinate on distance map
        /// Generated from method `MR::ContourToDistanceMapParams::toWorld`.
        public unsafe MR.Vector2f ToWorld(MR.Vector2f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_toWorld", ExactSpelling = true)]
            extern static MR.Vector2f __MR_ContourToDistanceMapParams_toWorld(_Underlying *_this, MR.Vector2f point);
            return __MR_ContourToDistanceMapParams_toWorld(_UnderlyingPtr, point);
        }

        /// Generated from method `MR::ContourToDistanceMapParams::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_xf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_ContourToDistanceMapParams_xf(_Underlying *_this);
            return __MR_ContourToDistanceMapParams_xf(_UnderlyingPtr);
        }
    }

    /// Structure with parameters to generate DistanceMap by Contours
    /// Generated from class `MR::ContourToDistanceMapParams`.
    /// This is the non-const half of the class.
    public class ContourToDistanceMapParams : Const_ContourToDistanceMapParams
    {
        internal unsafe ContourToDistanceMapParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< pixel size
        public new unsafe MR.Mut_Vector2f PixelSize
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_GetMutable_pixelSize", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_ContourToDistanceMapParams_GetMutable_pixelSize(_Underlying *_this);
                return new(__MR_ContourToDistanceMapParams_GetMutable_pixelSize(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< distance map size
        public new unsafe MR.Mut_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_GetMutable_resolution", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_ContourToDistanceMapParams_GetMutable_resolution(_Underlying *_this);
                return new(__MR_ContourToDistanceMapParams_GetMutable_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< coordinates of origin area corner
        public new unsafe MR.Mut_Vector2f OrgPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_GetMutable_orgPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector2f._Underlying *__MR_ContourToDistanceMapParams_GetMutable_orgPoint(_Underlying *_this);
                return new(__MR_ContourToDistanceMapParams_GetMutable_orgPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< allows calculate negative values of distance (inside closed and correctly oriented (CW) contours)
        public new unsafe ref bool WithSign
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_GetMutable_withSign", ExactSpelling = true)]
                extern static bool *__MR_ContourToDistanceMapParams_GetMutable_withSign(_Underlying *_this);
                return ref *__MR_ContourToDistanceMapParams_GetMutable_withSign(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe ContourToDistanceMapParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_DefaultConstruct();
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_DefaultConstruct();
        }

        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        public unsafe ContourToDistanceMapParams(MR.Const_ContourToDistanceMapParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_ConstructFromAnother(MR.ContourToDistanceMapParams._Underlying *_other);
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Ctor, calculating pixelSize by areaSize & dmapSize
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe ContourToDistanceMapParams(MR.Const_Vector2i resolution, MR.Const_Vector2f oriPoint, MR.Const_Vector2f areaSize, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_MR_Vector2f", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_MR_Vector2f(MR.Const_Vector2i._Underlying *resolution, MR.Const_Vector2f._Underlying *oriPoint, MR.Const_Vector2f._Underlying *areaSize, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_MR_Vector2f(resolution._UnderlyingPtr, oriPoint._UnderlyingPtr, areaSize._UnderlyingPtr, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Ctor, calculating pixelSize & oriPoint by box parameters
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe ContourToDistanceMapParams(MR.Const_Vector2i resolution, MR.Const_Box2f box, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_3", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_3(MR.Const_Vector2i._Underlying *resolution, MR.Const_Box2f._Underlying *box, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_3(resolution._UnderlyingPtr, box._UnderlyingPtr, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Ctor, calculating pixelSize & oriPoint by contours box + offset
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe ContourToDistanceMapParams(MR.Const_Vector2i resolution, MR.Std.Const_Vector_StdVectorMRVector2f contours, float offset, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_std_vector_std_vector_MR_Vector2f(MR.Const_Vector2i._Underlying *resolution, MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, float offset, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_4_const_MR_Vector2i_ref_std_vector_std_vector_MR_Vector2f(resolution._UnderlyingPtr, contours._UnderlyingPtr, offset, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Ctor, calculating resolution & oriPoint by contours box + offset
        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        /// Parameter `withSign` defaults to `false`.
        public unsafe ContourToDistanceMapParams(float pixelSize, MR.Std.Const_Vector_StdVectorMRVector2f contours, float offset, bool? withSign = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_4_float", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_4_float(float pixelSize, MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, float offset, byte *withSign);
            byte __deref_withSign = withSign.GetValueOrDefault() ? (byte)1 : (byte)0;
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_4_float(pixelSize, contours._UnderlyingPtr, offset, withSign.HasValue ? &__deref_withSign : null);
        }

        /// Generated from constructor `MR::ContourToDistanceMapParams::ContourToDistanceMapParams`.
        public unsafe ContourToDistanceMapParams(MR.Const_DistanceMapToWorld toWorld) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_Construct_1", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_Construct_1(MR.Const_DistanceMapToWorld._Underlying *toWorld);
            _UnderlyingPtr = __MR_ContourToDistanceMapParams_Construct_1(toWorld._UnderlyingPtr);
        }

        /// Generated from method `MR::ContourToDistanceMapParams::operator=`.
        public unsafe MR.ContourToDistanceMapParams Assign(MR.Const_ContourToDistanceMapParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ContourToDistanceMapParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.ContourToDistanceMapParams._Underlying *__MR_ContourToDistanceMapParams_AssignFromAnother(_Underlying *_this, MR.ContourToDistanceMapParams._Underlying *_other);
            return new(__MR_ContourToDistanceMapParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `ContourToDistanceMapParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_ContourToDistanceMapParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContourToDistanceMapParams`/`Const_ContourToDistanceMapParams` directly.
    public class _InOptMut_ContourToDistanceMapParams
    {
        public ContourToDistanceMapParams? Opt;

        public _InOptMut_ContourToDistanceMapParams() {}
        public _InOptMut_ContourToDistanceMapParams(ContourToDistanceMapParams value) {Opt = value;}
        public static implicit operator _InOptMut_ContourToDistanceMapParams(ContourToDistanceMapParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `ContourToDistanceMapParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_ContourToDistanceMapParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `ContourToDistanceMapParams`/`Const_ContourToDistanceMapParams` to pass it to the function.
    public class _InOptConst_ContourToDistanceMapParams
    {
        public Const_ContourToDistanceMapParams? Opt;

        public _InOptConst_ContourToDistanceMapParams() {}
        public _InOptConst_ContourToDistanceMapParams(Const_ContourToDistanceMapParams value) {Opt = value;}
        public static implicit operator _InOptConst_ContourToDistanceMapParams(Const_ContourToDistanceMapParams value) {return new(value);}
    }

    /// This structure store data to transform distance map to world coordinates
    /// Generated from class `MR::DistanceMapToWorld`.
    /// This is the const half of the class.
    public class Const_DistanceMapToWorld : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceMapToWorld(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceMapToWorld_Destroy(_Underlying *_this);
            __MR_DistanceMapToWorld_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceMapToWorld() {Dispose(false);}

        /// world coordinates of distance map origin corner
        public unsafe MR.Const_Vector3f OrgPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Get_orgPoint", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_DistanceMapToWorld_Get_orgPoint(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_Get_orgPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// vector in world space of pixel x positive direction
        /// length is equal to pixel size
        /// \note typically it should be orthogonal to `pixelYVec`
        public unsafe MR.Const_Vector3f PixelXVec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Get_pixelXVec", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_DistanceMapToWorld_Get_pixelXVec(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_Get_pixelXVec(_UnderlyingPtr), is_owning: false);
            }
        }

        /// vector in world space of pixel y positive direction
        /// length is equal to pixel size
        /// \note typically it should be orthogonal to `pixelXVec`
        public unsafe MR.Const_Vector3f PixelYVec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Get_pixelYVec", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_DistanceMapToWorld_Get_pixelYVec(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_Get_pixelYVec(_UnderlyingPtr), is_owning: false);
            }
        }

        /// vector of depth direction
        /// \note typically it should be normalized and orthogonal to `pixelXVec` `pixelYVec` plane
        public unsafe MR.Const_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Get_direction", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_DistanceMapToWorld_Get_direction(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_Get_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceMapToWorld() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceMapToWorld_DefaultConstruct();
        }

        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe Const_DistanceMapToWorld(MR.Const_DistanceMapToWorld _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_ConstructFromAnother(MR.DistanceMapToWorld._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceMapToWorld_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Init fields by `MeshToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe Const_DistanceMapToWorld(MR.Const_MeshToDistanceMapParams params_) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Construct_MR_MeshToDistanceMapParams", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_Construct_MR_MeshToDistanceMapParams(MR.Const_MeshToDistanceMapParams._Underlying *params_);
            _UnderlyingPtr = __MR_DistanceMapToWorld_Construct_MR_MeshToDistanceMapParams(params_._UnderlyingPtr);
        }

        /// Init fields by `MeshToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator Const_DistanceMapToWorld(MR.Const_MeshToDistanceMapParams params_) {return new(params_);}

        /// Init fields by `ContourToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe Const_DistanceMapToWorld(MR.Const_ContourToDistanceMapParams params_) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Construct_MR_ContourToDistanceMapParams", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_Construct_MR_ContourToDistanceMapParams(MR.Const_ContourToDistanceMapParams._Underlying *params_);
            _UnderlyingPtr = __MR_DistanceMapToWorld_Construct_MR_ContourToDistanceMapParams(params_._UnderlyingPtr);
        }

        /// Init fields by `ContourToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator Const_DistanceMapToWorld(MR.Const_ContourToDistanceMapParams params_) {return new(params_);}

        /// Converts from AffineXf3f
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe Const_DistanceMapToWorld(MR.Const_AffineXf3f xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Construct_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_Construct_MR_AffineXf3f(MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DistanceMapToWorld_Construct_MR_AffineXf3f(xf._UnderlyingPtr);
        }

        /// Converts from AffineXf3f
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator Const_DistanceMapToWorld(MR.Const_AffineXf3f xf) {return new(xf);}

        /// converts in transformation X: X(p) == toWorld( p.x, p.y, p.z )
        /// Generated from conversion operator `MR::DistanceMapToWorld::operator MR::AffineXf3f`.
        public static unsafe implicit operator MR.AffineXf3f(MR.Const_DistanceMapToWorld _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_ConvertTo_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_DistanceMapToWorld_ConvertTo_MR_AffineXf3f(MR.Const_DistanceMapToWorld._Underlying *_this);
            return __MR_DistanceMapToWorld_ConvertTo_MR_AffineXf3f(_this._UnderlyingPtr);
        }

        /// get world coordinate by depth map info
        /// x - float X coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)
        /// y - float Y coordinate of depth map: (0.0f - left corner of pixel 0, 1.0 - right corner of pixel 0 and left corner of pixel 1)
        /// float depth value (value in distance map, represent depth in world)
        /// Generated from method `MR::DistanceMapToWorld::toWorld`.
        public unsafe MR.Vector3f ToWorld(float x, float y, float depth)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_toWorld", ExactSpelling = true)]
            extern static MR.Vector3f __MR_DistanceMapToWorld_toWorld(_Underlying *_this, float x, float y, float depth);
            return __MR_DistanceMapToWorld_toWorld(_UnderlyingPtr, x, y, depth);
        }

        /// Generated from method `MR::DistanceMapToWorld::xf`.
        public unsafe MR.AffineXf3f Xf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_xf", ExactSpelling = true)]
            extern static MR.AffineXf3f __MR_DistanceMapToWorld_xf(_Underlying *_this);
            return __MR_DistanceMapToWorld_xf(_UnderlyingPtr);
        }
    }

    /// This structure store data to transform distance map to world coordinates
    /// Generated from class `MR::DistanceMapToWorld`.
    /// This is the non-const half of the class.
    public class DistanceMapToWorld : Const_DistanceMapToWorld
    {
        internal unsafe DistanceMapToWorld(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// world coordinates of distance map origin corner
        public new unsafe MR.Mut_Vector3f OrgPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_GetMutable_orgPoint", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_DistanceMapToWorld_GetMutable_orgPoint(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_GetMutable_orgPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// vector in world space of pixel x positive direction
        /// length is equal to pixel size
        /// \note typically it should be orthogonal to `pixelYVec`
        public new unsafe MR.Mut_Vector3f PixelXVec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_GetMutable_pixelXVec", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_DistanceMapToWorld_GetMutable_pixelXVec(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_GetMutable_pixelXVec(_UnderlyingPtr), is_owning: false);
            }
        }

        /// vector in world space of pixel y positive direction
        /// length is equal to pixel size
        /// \note typically it should be orthogonal to `pixelXVec`
        public new unsafe MR.Mut_Vector3f PixelYVec
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_GetMutable_pixelYVec", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_DistanceMapToWorld_GetMutable_pixelYVec(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_GetMutable_pixelYVec(_UnderlyingPtr), is_owning: false);
            }
        }

        /// vector of depth direction
        /// \note typically it should be normalized and orthogonal to `pixelXVec` `pixelYVec` plane
        public new unsafe MR.Mut_Vector3f Direction
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_GetMutable_direction", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_DistanceMapToWorld_GetMutable_direction(_Underlying *_this);
                return new(__MR_DistanceMapToWorld_GetMutable_direction(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceMapToWorld() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceMapToWorld_DefaultConstruct();
        }

        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe DistanceMapToWorld(MR.Const_DistanceMapToWorld _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_ConstructFromAnother(MR.DistanceMapToWorld._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceMapToWorld_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Init fields by `MeshToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe DistanceMapToWorld(MR.Const_MeshToDistanceMapParams params_) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Construct_MR_MeshToDistanceMapParams", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_Construct_MR_MeshToDistanceMapParams(MR.Const_MeshToDistanceMapParams._Underlying *params_);
            _UnderlyingPtr = __MR_DistanceMapToWorld_Construct_MR_MeshToDistanceMapParams(params_._UnderlyingPtr);
        }

        /// Init fields by `MeshToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator DistanceMapToWorld(MR.Const_MeshToDistanceMapParams params_) {return new(params_);}

        /// Init fields by `ContourToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe DistanceMapToWorld(MR.Const_ContourToDistanceMapParams params_) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Construct_MR_ContourToDistanceMapParams", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_Construct_MR_ContourToDistanceMapParams(MR.Const_ContourToDistanceMapParams._Underlying *params_);
            _UnderlyingPtr = __MR_DistanceMapToWorld_Construct_MR_ContourToDistanceMapParams(params_._UnderlyingPtr);
        }

        /// Init fields by `ContourToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator DistanceMapToWorld(MR.Const_ContourToDistanceMapParams params_) {return new(params_);}

        /// Converts from AffineXf3f
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public unsafe DistanceMapToWorld(MR.Const_AffineXf3f xf) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_Construct_MR_AffineXf3f", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_Construct_MR_AffineXf3f(MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DistanceMapToWorld_Construct_MR_AffineXf3f(xf._UnderlyingPtr);
        }

        /// Converts from AffineXf3f
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator DistanceMapToWorld(MR.Const_AffineXf3f xf) {return new(xf);}

        /// Generated from method `MR::DistanceMapToWorld::operator=`.
        public unsafe MR.DistanceMapToWorld Assign(MR.Const_DistanceMapToWorld _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapToWorld_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapToWorld._Underlying *__MR_DistanceMapToWorld_AssignFromAnother(_Underlying *_this, MR.DistanceMapToWorld._Underlying *_other);
            return new(__MR_DistanceMapToWorld_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DistanceMapToWorld` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceMapToWorld`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMapToWorld`/`Const_DistanceMapToWorld` directly.
    public class _InOptMut_DistanceMapToWorld
    {
        public DistanceMapToWorld? Opt;

        public _InOptMut_DistanceMapToWorld() {}
        public _InOptMut_DistanceMapToWorld(DistanceMapToWorld value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceMapToWorld(DistanceMapToWorld value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceMapToWorld` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceMapToWorld`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMapToWorld`/`Const_DistanceMapToWorld` to pass it to the function.
    public class _InOptConst_DistanceMapToWorld
    {
        public Const_DistanceMapToWorld? Opt;

        public _InOptConst_DistanceMapToWorld() {}
        public _InOptConst_DistanceMapToWorld(Const_DistanceMapToWorld value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceMapToWorld(Const_DistanceMapToWorld value) {return new(value);}

        /// Init fields by `MeshToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator _InOptConst_DistanceMapToWorld(MR.Const_MeshToDistanceMapParams params_) {return new MR.DistanceMapToWorld(params_);}

        /// Init fields by `ContourToDistanceMapParams` struct
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator _InOptConst_DistanceMapToWorld(MR.Const_ContourToDistanceMapParams params_) {return new MR.DistanceMapToWorld(params_);}

        /// Converts from AffineXf3f
        /// Generated from constructor `MR::DistanceMapToWorld::DistanceMapToWorld`.
        public static unsafe implicit operator _InOptConst_DistanceMapToWorld(MR.Const_AffineXf3f xf) {return new MR.DistanceMapToWorld(xf);}
    }

    /// settings for loading distance maps from external formats
    /// Generated from class `MR::DistanceMapLoadSettings`.
    /// This is the const half of the class.
    public class Const_DistanceMapLoadSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceMapLoadSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceMapLoadSettings_Destroy(_Underlying *_this);
            __MR_DistanceMapLoadSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceMapLoadSettings() {Dispose(false);}

        /// optional output: distance map to world transform
        public unsafe ref void * DistanceMapToWorld
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_Get_distanceMapToWorld", ExactSpelling = true)]
                extern static void **__MR_DistanceMapLoadSettings_Get_distanceMapToWorld(_Underlying *_this);
                return ref *__MR_DistanceMapLoadSettings_Get_distanceMapToWorld(_UnderlyingPtr);
            }
        }

        /// to report load progress and cancel loading if user desires
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DistanceMapLoadSettings_Get_progress(_Underlying *_this);
                return new(__MR_DistanceMapLoadSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceMapLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceMapLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::DistanceMapLoadSettings` elementwise.
        public unsafe Const_DistanceMapLoadSettings(MR.DistanceMapToWorld? distanceMapToWorld, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_ConstructFrom(MR.DistanceMapToWorld._Underlying *distanceMapToWorld, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DistanceMapLoadSettings_ConstructFrom(distanceMapToWorld is not null ? distanceMapToWorld._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DistanceMapLoadSettings::DistanceMapLoadSettings`.
        public unsafe Const_DistanceMapLoadSettings(MR._ByValue_DistanceMapLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceMapLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceMapLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// settings for loading distance maps from external formats
    /// Generated from class `MR::DistanceMapLoadSettings`.
    /// This is the non-const half of the class.
    public class DistanceMapLoadSettings : Const_DistanceMapLoadSettings
    {
        internal unsafe DistanceMapLoadSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// optional output: distance map to world transform
        public new unsafe ref void * DistanceMapToWorld
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_GetMutable_distanceMapToWorld", ExactSpelling = true)]
                extern static void **__MR_DistanceMapLoadSettings_GetMutable_distanceMapToWorld(_Underlying *_this);
                return ref *__MR_DistanceMapLoadSettings_GetMutable_distanceMapToWorld(_UnderlyingPtr);
            }
        }

        /// to report load progress and cancel loading if user desires
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DistanceMapLoadSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_DistanceMapLoadSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceMapLoadSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceMapLoadSettings_DefaultConstruct();
        }

        /// Constructs `MR::DistanceMapLoadSettings` elementwise.
        public unsafe DistanceMapLoadSettings(MR.DistanceMapToWorld? distanceMapToWorld, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_ConstructFrom(MR.DistanceMapToWorld._Underlying *distanceMapToWorld, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DistanceMapLoadSettings_ConstructFrom(distanceMapToWorld is not null ? distanceMapToWorld._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DistanceMapLoadSettings::DistanceMapLoadSettings`.
        public unsafe DistanceMapLoadSettings(MR._ByValue_DistanceMapLoadSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceMapLoadSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceMapLoadSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DistanceMapLoadSettings::operator=`.
        public unsafe MR.DistanceMapLoadSettings Assign(MR._ByValue_DistanceMapLoadSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapLoadSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapLoadSettings._Underlying *__MR_DistanceMapLoadSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceMapLoadSettings._Underlying *_other);
            return new(__MR_DistanceMapLoadSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DistanceMapLoadSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DistanceMapLoadSettings`/`Const_DistanceMapLoadSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DistanceMapLoadSettings
    {
        internal readonly Const_DistanceMapLoadSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DistanceMapLoadSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DistanceMapLoadSettings(Const_DistanceMapLoadSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DistanceMapLoadSettings(Const_DistanceMapLoadSettings arg) {return new(arg);}
        public _ByValue_DistanceMapLoadSettings(MR.Misc._Moved<DistanceMapLoadSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DistanceMapLoadSettings(MR.Misc._Moved<DistanceMapLoadSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DistanceMapLoadSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceMapLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMapLoadSettings`/`Const_DistanceMapLoadSettings` directly.
    public class _InOptMut_DistanceMapLoadSettings
    {
        public DistanceMapLoadSettings? Opt;

        public _InOptMut_DistanceMapLoadSettings() {}
        public _InOptMut_DistanceMapLoadSettings(DistanceMapLoadSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceMapLoadSettings(DistanceMapLoadSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceMapLoadSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceMapLoadSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMapLoadSettings`/`Const_DistanceMapLoadSettings` to pass it to the function.
    public class _InOptConst_DistanceMapLoadSettings
    {
        public Const_DistanceMapLoadSettings? Opt;

        public _InOptConst_DistanceMapLoadSettings() {}
        public _InOptConst_DistanceMapLoadSettings(Const_DistanceMapLoadSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceMapLoadSettings(Const_DistanceMapLoadSettings value) {return new(value);}
    }

    /// determines how to save distance maps
    /// Generated from class `MR::DistanceMapSaveSettings`.
    /// This is the const half of the class.
    public class Const_DistanceMapSaveSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceMapSaveSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceMapSaveSettings_Destroy(_Underlying *_this);
            __MR_DistanceMapSaveSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceMapSaveSettings() {Dispose(false);}

        /// optional distance map to world transform
        public unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_Get_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_DistanceMapSaveSettings_Get_xf(_Underlying *_this);
                return ref *__MR_DistanceMapSaveSettings_Get_xf(_UnderlyingPtr);
            }
        }

        /// to report save progress and cancel saving if user desires
        public unsafe MR.Std.Const_Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_Get_progress", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_DistanceMapSaveSettings_Get_progress(_Underlying *_this);
                return new(__MR_DistanceMapSaveSettings_Get_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceMapSaveSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceMapSaveSettings_DefaultConstruct();
        }

        /// Constructs `MR::DistanceMapSaveSettings` elementwise.
        public unsafe Const_DistanceMapSaveSettings(MR.Const_AffineXf3f? xf, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_ConstructFrom(MR.Const_AffineXf3f._Underlying *xf, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DistanceMapSaveSettings_ConstructFrom(xf is not null ? xf._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DistanceMapSaveSettings::DistanceMapSaveSettings`.
        public unsafe Const_DistanceMapSaveSettings(MR._ByValue_DistanceMapSaveSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceMapSaveSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceMapSaveSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// determines how to save distance maps
    /// Generated from class `MR::DistanceMapSaveSettings`.
    /// This is the non-const half of the class.
    public class DistanceMapSaveSettings : Const_DistanceMapSaveSettings
    {
        internal unsafe DistanceMapSaveSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// optional distance map to world transform
        public new unsafe ref readonly MR.AffineXf3f * Xf
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_GetMutable_xf", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_DistanceMapSaveSettings_GetMutable_xf(_Underlying *_this);
                return ref *__MR_DistanceMapSaveSettings_GetMutable_xf(_UnderlyingPtr);
            }
        }

        /// to report save progress and cancel saving if user desires
        public new unsafe MR.Std.Function_BoolFuncFromFloat Progress
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_GetMutable_progress", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_DistanceMapSaveSettings_GetMutable_progress(_Underlying *_this);
                return new(__MR_DistanceMapSaveSettings_GetMutable_progress(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceMapSaveSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceMapSaveSettings_DefaultConstruct();
        }

        /// Constructs `MR::DistanceMapSaveSettings` elementwise.
        public unsafe DistanceMapSaveSettings(MR.Const_AffineXf3f? xf, MR.Std._ByValue_Function_BoolFuncFromFloat progress) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_ConstructFrom(MR.Const_AffineXf3f._Underlying *xf, MR.Misc._PassBy progress_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progress);
            _UnderlyingPtr = __MR_DistanceMapSaveSettings_ConstructFrom(xf is not null ? xf._UnderlyingPtr : null, progress.PassByMode, progress.Value is not null ? progress.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::DistanceMapSaveSettings::DistanceMapSaveSettings`.
        public unsafe DistanceMapSaveSettings(MR._ByValue_DistanceMapSaveSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.DistanceMapSaveSettings._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceMapSaveSettings_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DistanceMapSaveSettings::operator=`.
        public unsafe MR.DistanceMapSaveSettings Assign(MR._ByValue_DistanceMapSaveSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceMapSaveSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceMapSaveSettings._Underlying *__MR_DistanceMapSaveSettings_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.DistanceMapSaveSettings._Underlying *_other);
            return new(__MR_DistanceMapSaveSettings_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `DistanceMapSaveSettings` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `DistanceMapSaveSettings`/`Const_DistanceMapSaveSettings` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_DistanceMapSaveSettings
    {
        internal readonly Const_DistanceMapSaveSettings? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_DistanceMapSaveSettings() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_DistanceMapSaveSettings(Const_DistanceMapSaveSettings new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_DistanceMapSaveSettings(Const_DistanceMapSaveSettings arg) {return new(arg);}
        public _ByValue_DistanceMapSaveSettings(MR.Misc._Moved<DistanceMapSaveSettings> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_DistanceMapSaveSettings(MR.Misc._Moved<DistanceMapSaveSettings> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `DistanceMapSaveSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceMapSaveSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMapSaveSettings`/`Const_DistanceMapSaveSettings` directly.
    public class _InOptMut_DistanceMapSaveSettings
    {
        public DistanceMapSaveSettings? Opt;

        public _InOptMut_DistanceMapSaveSettings() {}
        public _InOptMut_DistanceMapSaveSettings(DistanceMapSaveSettings value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceMapSaveSettings(DistanceMapSaveSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceMapSaveSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceMapSaveSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceMapSaveSettings`/`Const_DistanceMapSaveSettings` to pass it to the function.
    public class _InOptConst_DistanceMapSaveSettings
    {
        public Const_DistanceMapSaveSettings? Opt;

        public _InOptConst_DistanceMapSaveSettings() {}
        public _InOptConst_DistanceMapSaveSettings(Const_DistanceMapSaveSettings value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceMapSaveSettings(Const_DistanceMapSaveSettings value) {return new(value);}
    }
}
