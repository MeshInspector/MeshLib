public static partial class MR
{
    /// Generated from class `MR::PolylineIntersectionResult2`.
    /// This is the const half of the class.
    public class Const_PolylineIntersectionResult2 : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PolylineIntersectionResult2(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_Destroy", ExactSpelling = true)]
            extern static void __MR_PolylineIntersectionResult2_Destroy(_Underlying *_this);
            __MR_PolylineIntersectionResult2_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PolylineIntersectionResult2() {Dispose(false);}

        /// intersection point in polyline
        public unsafe MR.Const_EdgePoint EdgePoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_Get_edgePoint", ExactSpelling = true)]
                extern static MR.Const_EdgePoint._Underlying *__MR_PolylineIntersectionResult2_Get_edgePoint(_Underlying *_this);
                return new(__MR_PolylineIntersectionResult2_Get_edgePoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores the distance from ray origin to the intersection point in direction units
        public unsafe float DistanceAlongLine
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_Get_distanceAlongLine", ExactSpelling = true)]
                extern static float *__MR_PolylineIntersectionResult2_Get_distanceAlongLine(_Underlying *_this);
                return *__MR_PolylineIntersectionResult2_Get_distanceAlongLine(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PolylineIntersectionResult2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineIntersectionResult2_DefaultConstruct();
        }

        /// Constructs `MR::PolylineIntersectionResult2` elementwise.
        public unsafe Const_PolylineIntersectionResult2(MR.Const_EdgePoint edgePoint, float distanceAlongLine) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_ConstructFrom(MR.EdgePoint._Underlying *edgePoint, float distanceAlongLine);
            _UnderlyingPtr = __MR_PolylineIntersectionResult2_ConstructFrom(edgePoint._UnderlyingPtr, distanceAlongLine);
        }

        /// Generated from constructor `MR::PolylineIntersectionResult2::PolylineIntersectionResult2`.
        public unsafe Const_PolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_ConstructFromAnother(MR.PolylineIntersectionResult2._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineIntersectionResult2_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::PolylineIntersectionResult2`.
    /// This is the non-const half of the class.
    public class PolylineIntersectionResult2 : Const_PolylineIntersectionResult2
    {
        internal unsafe PolylineIntersectionResult2(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// intersection point in polyline
        public new unsafe MR.EdgePoint EdgePoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_GetMutable_edgePoint", ExactSpelling = true)]
                extern static MR.EdgePoint._Underlying *__MR_PolylineIntersectionResult2_GetMutable_edgePoint(_Underlying *_this);
                return new(__MR_PolylineIntersectionResult2_GetMutable_edgePoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// stores the distance from ray origin to the intersection point in direction units
        public new unsafe ref float DistanceAlongLine
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_GetMutable_distanceAlongLine", ExactSpelling = true)]
                extern static float *__MR_PolylineIntersectionResult2_GetMutable_distanceAlongLine(_Underlying *_this);
                return ref *__MR_PolylineIntersectionResult2_GetMutable_distanceAlongLine(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PolylineIntersectionResult2() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_DefaultConstruct();
            _UnderlyingPtr = __MR_PolylineIntersectionResult2_DefaultConstruct();
        }

        /// Constructs `MR::PolylineIntersectionResult2` elementwise.
        public unsafe PolylineIntersectionResult2(MR.Const_EdgePoint edgePoint, float distanceAlongLine) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_ConstructFrom", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_ConstructFrom(MR.EdgePoint._Underlying *edgePoint, float distanceAlongLine);
            _UnderlyingPtr = __MR_PolylineIntersectionResult2_ConstructFrom(edgePoint._UnderlyingPtr, distanceAlongLine);
        }

        /// Generated from constructor `MR::PolylineIntersectionResult2::PolylineIntersectionResult2`.
        public unsafe PolylineIntersectionResult2(MR.Const_PolylineIntersectionResult2 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_ConstructFromAnother(MR.PolylineIntersectionResult2._Underlying *_other);
            _UnderlyingPtr = __MR_PolylineIntersectionResult2_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PolylineIntersectionResult2::operator=`.
        public unsafe MR.PolylineIntersectionResult2 Assign(MR.Const_PolylineIntersectionResult2 _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PolylineIntersectionResult2_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PolylineIntersectionResult2._Underlying *__MR_PolylineIntersectionResult2_AssignFromAnother(_Underlying *_this, MR.PolylineIntersectionResult2._Underlying *_other);
            return new(__MR_PolylineIntersectionResult2_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `PolylineIntersectionResult2` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PolylineIntersectionResult2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineIntersectionResult2`/`Const_PolylineIntersectionResult2` directly.
    public class _InOptMut_PolylineIntersectionResult2
    {
        public PolylineIntersectionResult2? Opt;

        public _InOptMut_PolylineIntersectionResult2() {}
        public _InOptMut_PolylineIntersectionResult2(PolylineIntersectionResult2 value) {Opt = value;}
        public static implicit operator _InOptMut_PolylineIntersectionResult2(PolylineIntersectionResult2 value) {return new(value);}
    }

    /// This is used for optional parameters of class `PolylineIntersectionResult2` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PolylineIntersectionResult2`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PolylineIntersectionResult2`/`Const_PolylineIntersectionResult2` to pass it to the function.
    public class _InOptConst_PolylineIntersectionResult2
    {
        public Const_PolylineIntersectionResult2? Opt;

        public _InOptConst_PolylineIntersectionResult2() {}
        public _InOptConst_PolylineIntersectionResult2(Const_PolylineIntersectionResult2 value) {Opt = value;}
        public static implicit operator _InOptConst_PolylineIntersectionResult2(Const_PolylineIntersectionResult2 value) {return new(value);}
    }

    /**
    * \brief detect if given point is inside polyline, by counting ray intersections
    * \param polyline input polyline
    * \param point input point
    */
    /// Generated from function `MR::isPointInsidePolyline`.
    public static unsafe bool IsPointInsidePolyline(MR.Const_Polyline2 polyline, MR.Const_Vector2f point)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_isPointInsidePolyline", ExactSpelling = true)]
        extern static byte __MR_isPointInsidePolyline(MR.Const_Polyline2._Underlying *polyline, MR.Const_Vector2f._Underlying *point);
        return __MR_isPointInsidePolyline(polyline._UnderlyingPtr, point._UnderlyingPtr) != 0;
    }

    /// Finds ray and polyline intersection in float-precision.
    /// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
    /// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
    /// Finds the closest intersection to ray origin (line param=0)
    /// or any intersection for better performance if \p !closestIntersect.
    /// Generated from function `MR::rayPolylineIntersect`.
    /// Parameter `rayStart` defaults to `0`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    /// Parameter `closestIntersect` defaults to `true`.
    public static unsafe MR.Std.Optional_MRPolylineIntersectionResult2 RayPolylineIntersect(MR.Const_Polyline2 polyline, MR.Const_Line2f line, float? rayStart = null, float? rayEnd = null, MR.Const_IntersectionPrecomputes2_Float? prec = null, bool? closestIntersect = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayPolylineIntersect_MR_Line2f", ExactSpelling = true)]
        extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_rayPolylineIntersect_MR_Line2f(MR.Const_Polyline2._Underlying *polyline, MR.Const_Line2f._Underlying *line, float *rayStart, float *rayEnd, MR.Const_IntersectionPrecomputes2_Float._Underlying *prec, byte *closestIntersect);
        float __deref_rayStart = rayStart.GetValueOrDefault();
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        byte __deref_closestIntersect = closestIntersect.GetValueOrDefault() ? (byte)1 : (byte)0;
        return new(__MR_rayPolylineIntersect_MR_Line2f(polyline._UnderlyingPtr, line._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null, closestIntersect.HasValue ? &__deref_closestIntersect : null), is_owning: true);
    }

    /// Finds ray and polyline intersection in double-precision.
    /// \p rayStart and \p rayEnd define the interval on the ray to detect an intersection.
    /// \p prec can be specified to reuse some precomputations (e.g. for checking many parallel rays).
    /// Finds the closest intersection to ray origin (line param=0)
    /// or any intersection for better performance if \p !closestIntersect.
    /// Generated from function `MR::rayPolylineIntersect`.
    /// Parameter `rayStart` defaults to `0`.
    /// Parameter `rayEnd` defaults to `1.7976931348623157e308`.
    /// Parameter `closestIntersect` defaults to `true`.
    public static unsafe MR.Std.Optional_MRPolylineIntersectionResult2 RayPolylineIntersect(MR.Const_Polyline2 polyline, MR.Const_Line2d line, double? rayStart = null, double? rayEnd = null, MR.Const_IntersectionPrecomputes2_Double? prec = null, bool? closestIntersect = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayPolylineIntersect_MR_Line2d", ExactSpelling = true)]
        extern static MR.Std.Optional_MRPolylineIntersectionResult2._Underlying *__MR_rayPolylineIntersect_MR_Line2d(MR.Const_Polyline2._Underlying *polyline, MR.Const_Line2d._Underlying *line, double *rayStart, double *rayEnd, MR.Const_IntersectionPrecomputes2_Double._Underlying *prec, byte *closestIntersect);
        double __deref_rayStart = rayStart.GetValueOrDefault();
        double __deref_rayEnd = rayEnd.GetValueOrDefault();
        byte __deref_closestIntersect = closestIntersect.GetValueOrDefault() ? (byte)1 : (byte)0;
        return new(__MR_rayPolylineIntersect_MR_Line2d(polyline._UnderlyingPtr, line._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null, closestIntersect.HasValue ? &__deref_closestIntersect : null), is_owning: true);
    }

    /// Intersects 2D ray and polyline in single-precision.
    /// Reports all intersections via given callback with the tendency to do it from ray start to ray end, but without guarantee of exact order.
    /// Generated from function `MR::rayPolylineIntersectAll`.
    /// Parameter `rayStart` defaults to `0.0f`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    public static unsafe void RayPolylineIntersectAll(MR.Const_Polyline2 polyline, MR.Const_Line2f line, MR.Std.Const_Function_MRProcessingFuncFromConstMREdgePointRefFloatFloatRefFloatRef callback, float? rayStart = null, float? rayEnd = null, MR.Const_IntersectionPrecomputes2_Float? prec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayPolylineIntersectAll_MR_Line2f", ExactSpelling = true)]
        extern static void __MR_rayPolylineIntersectAll_MR_Line2f(MR.Const_Polyline2._Underlying *polyline, MR.Const_Line2f._Underlying *line, MR.Std.Const_Function_MRProcessingFuncFromConstMREdgePointRefFloatFloatRefFloatRef._Underlying *callback, float *rayStart, float *rayEnd, MR.Const_IntersectionPrecomputes2_Float._Underlying *prec);
        float __deref_rayStart = rayStart.GetValueOrDefault();
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        __MR_rayPolylineIntersectAll_MR_Line2f(polyline._UnderlyingPtr, line._UnderlyingPtr, callback._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null);
    }

    /// Intersects 2D ray and polyline in double-precision.
    /// Reports all intersections via given callback with the tendency to do it from ray start to ray end, but without guarantee of exact order.
    /// Generated from function `MR::rayPolylineIntersectAll`.
    /// Parameter `rayStart` defaults to `0.0`.
    /// Parameter `rayEnd` defaults to `1.7976931348623157e308`.
    public static unsafe void RayPolylineIntersectAll(MR.Const_Polyline2 polyline, MR.Const_Line2d line, MR.Std.Const_Function_MRProcessingFuncFromConstMREdgePointRefDoubleDoubleRefDoubleRef callback, double? rayStart = null, double? rayEnd = null, MR.Const_IntersectionPrecomputes2_Double? prec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayPolylineIntersectAll_MR_Line2d", ExactSpelling = true)]
        extern static void __MR_rayPolylineIntersectAll_MR_Line2d(MR.Const_Polyline2._Underlying *polyline, MR.Const_Line2d._Underlying *line, MR.Std.Const_Function_MRProcessingFuncFromConstMREdgePointRefDoubleDoubleRefDoubleRef._Underlying *callback, double *rayStart, double *rayEnd, MR.Const_IntersectionPrecomputes2_Double._Underlying *prec);
        double __deref_rayStart = rayStart.GetValueOrDefault();
        double __deref_rayEnd = rayEnd.GetValueOrDefault();
        __MR_rayPolylineIntersectAll_MR_Line2d(polyline._UnderlyingPtr, line._UnderlyingPtr, callback._UnderlyingPtr, rayStart.HasValue ? &__deref_rayStart : null, rayEnd.HasValue ? &__deref_rayEnd : null, prec is not null ? prec._UnderlyingPtr : null);
    }
}
