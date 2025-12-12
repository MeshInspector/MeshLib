public static partial class MR
{
    public static partial class Features
    {
        public static partial class Primitives
        {
            /// Generated from class `MR::Features::Primitives::Plane`.
            /// This is the const half of the class.
            public class Const_Plane : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Features.Primitives.Const_Plane>
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Plane(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Primitives_Plane_Destroy(_Underlying *_this);
                    __MR_Features_Primitives_Plane_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Plane() {Dispose(false);}

                public unsafe MR.Const_Vector3f Center
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_Get_center", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_Primitives_Plane_Get_center(_Underlying *_this);
                        return new(__MR_Features_Primitives_Plane_Get_center(_UnderlyingPtr), is_owning: false);
                    }
                }

                // This must be normalized. The sign doesn't matter.
                public unsafe MR.Const_Vector3f Normal
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_Get_normal", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_Primitives_Plane_Get_normal(_Underlying *_this);
                        return new(__MR_Features_Primitives_Plane_Get_normal(_UnderlyingPtr), is_owning: false);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Plane() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Primitives_Plane_DefaultConstruct();
                }

                /// Constructs `MR::Features::Primitives::Plane` elementwise.
                public unsafe Const_Plane(MR.Vector3f center, MR.Vector3f normal) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_ConstructFrom(MR.Vector3f center, MR.Vector3f normal);
                    _UnderlyingPtr = __MR_Features_Primitives_Plane_ConstructFrom(center, normal);
                }

                /// Generated from constructor `MR::Features::Primitives::Plane::Plane`.
                public unsafe Const_Plane(MR.Features.Primitives.Const_Plane _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_ConstructFromAnother(MR.Features.Primitives.Plane._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Primitives_Plane_ConstructFromAnother(_other._UnderlyingPtr);
                }

                // Returns an infinite line, with the center in a sane location.
                /// Generated from method `MR::Features::Primitives::Plane::intersectWithPlane`.
                public unsafe MR.Features.Primitives.ConeSegment IntersectWithPlane(MR.Features.Primitives.Const_Plane other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_intersectWithPlane", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_Plane_intersectWithPlane(_Underlying *_this, MR.Features.Primitives.Const_Plane._Underlying *other);
                    return new(__MR_Features_Primitives_Plane_intersectWithPlane(_UnderlyingPtr, other._UnderlyingPtr), is_owning: true);
                }

                // Intersects the plane with a line, returns a point (zero radius sphere).
                // Only `center` and `dir` are used from `line` (so if `line` is a cone/cylinder, its axis is used,
                // and the line is extended to infinity).
                /// Generated from method `MR::Features::Primitives::Plane::intersectWithLine`.
                public unsafe MR.Sphere3f IntersectWithLine(MR.Features.Primitives.Const_ConeSegment line)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_intersectWithLine", ExactSpelling = true)]
                    extern static MR.Sphere3f._Underlying *__MR_Features_Primitives_Plane_intersectWithLine(_Underlying *_this, MR.Features.Primitives.Const_ConeSegment._Underlying *line);
                    return new(__MR_Features_Primitives_Plane_intersectWithLine(_UnderlyingPtr, line._UnderlyingPtr), is_owning: true);
                }

                /// Generated from function `MR::Features::Primitives::operator==`.
                public static unsafe bool operator==(MR.Features.Primitives.Const_Plane _1, MR.Features.Primitives.Const_Plane _2)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Features_Primitives_Plane", ExactSpelling = true)]
                    extern static byte __MR_equal_MR_Features_Primitives_Plane(MR.Features.Primitives.Const_Plane._Underlying *_1, MR.Features.Primitives.Const_Plane._Underlying *_2);
                    return __MR_equal_MR_Features_Primitives_Plane(_1._UnderlyingPtr, _2._UnderlyingPtr) != 0;
                }

                public static unsafe bool operator!=(MR.Features.Primitives.Const_Plane _1, MR.Features.Primitives.Const_Plane _2)
                {
                    return !(_1 == _2);
                }

                // IEquatable:

                public bool Equals(MR.Features.Primitives.Const_Plane? _2)
                {
                    if (_2 is null)
                        return false;
                    return this == _2;
                }

                public override bool Equals(object? other)
                {
                    if (other is null)
                        return false;
                    if (other is MR.Features.Primitives.Const_Plane)
                        return this == (MR.Features.Primitives.Const_Plane)other;
                    return false;
                }
            }

            /// Generated from class `MR::Features::Primitives::Plane`.
            /// This is the non-const half of the class.
            public class Plane : Const_Plane
            {
                internal unsafe Plane(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                public new unsafe MR.Mut_Vector3f Center
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_GetMutable_center", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_Primitives_Plane_GetMutable_center(_Underlying *_this);
                        return new(__MR_Features_Primitives_Plane_GetMutable_center(_UnderlyingPtr), is_owning: false);
                    }
                }

                // This must be normalized. The sign doesn't matter.
                public new unsafe MR.Mut_Vector3f Normal
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_GetMutable_normal", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_Primitives_Plane_GetMutable_normal(_Underlying *_this);
                        return new(__MR_Features_Primitives_Plane_GetMutable_normal(_UnderlyingPtr), is_owning: false);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Plane() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Primitives_Plane_DefaultConstruct();
                }

                /// Constructs `MR::Features::Primitives::Plane` elementwise.
                public unsafe Plane(MR.Vector3f center, MR.Vector3f normal) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_ConstructFrom(MR.Vector3f center, MR.Vector3f normal);
                    _UnderlyingPtr = __MR_Features_Primitives_Plane_ConstructFrom(center, normal);
                }

                /// Generated from constructor `MR::Features::Primitives::Plane::Plane`.
                public unsafe Plane(MR.Features.Primitives.Const_Plane _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_ConstructFromAnother(MR.Features.Primitives.Plane._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Primitives_Plane_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Primitives::Plane::operator=`.
                public unsafe MR.Features.Primitives.Plane Assign(MR.Features.Primitives.Const_Plane _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_Plane_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_Plane_AssignFromAnother(_Underlying *_this, MR.Features.Primitives.Plane._Underlying *_other);
                    return new(__MR_Features_Primitives_Plane_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Plane` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Plane`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Plane`/`Const_Plane` directly.
            public class _InOptMut_Plane
            {
                public Plane? Opt;

                public _InOptMut_Plane() {}
                public _InOptMut_Plane(Plane value) {Opt = value;}
                public static implicit operator _InOptMut_Plane(Plane value) {return new(value);}
            }

            /// This is used for optional parameters of class `Plane` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Plane`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Plane`/`Const_Plane` to pass it to the function.
            public class _InOptConst_Plane
            {
                public Const_Plane? Opt;

                public _InOptConst_Plane() {}
                public _InOptConst_Plane(Const_Plane value) {Opt = value;}
                public static implicit operator _InOptConst_Plane(Const_Plane value) {return new(value);}
            }

            //! Can have infinite length in one or two directions.
            //! The top and/or bottom can be flat or pointy.
            //! Doubles as a cylinder, line (finite or infinite), and a circle.
            /// Generated from class `MR::Features::Primitives::ConeSegment`.
            /// This is the const half of the class.
            public class Const_ConeSegment : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Features.Primitives.Const_ConeSegment>
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_ConeSegment(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Primitives_ConeSegment_Destroy(_Underlying *_this);
                    __MR_Features_Primitives_ConeSegment_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_ConeSegment() {Dispose(false);}

                //! Some point on the axis, but not necessarily the true center point. Use `centerPoint()` for that.
                public unsafe MR.Const_Vector3f ReferencePoint
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_referencePoint", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_Primitives_ConeSegment_Get_referencePoint(_Underlying *_this);
                        return new(__MR_Features_Primitives_ConeSegment_Get_referencePoint(_UnderlyingPtr), is_owning: false);
                    }
                }

                //! The axis direction. Must be normalized.
                public unsafe MR.Const_Vector3f Dir
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_dir", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_Primitives_ConeSegment_Get_dir(_Underlying *_this);
                        return new(__MR_Features_Primitives_ConeSegment_Get_dir(_UnderlyingPtr), is_owning: false);
                    }
                }

                //! Cap radius in the `dir` direction.
                public unsafe float PositiveSideRadius
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_positiveSideRadius", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_Get_positiveSideRadius(_Underlying *_this);
                        return *__MR_Features_Primitives_ConeSegment_Get_positiveSideRadius(_UnderlyingPtr);
                    }
                }

                //! Cap radius in the direction opposite to `dir`.
                public unsafe float NegativeSideRadius
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_negativeSideRadius", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_Get_negativeSideRadius(_Underlying *_this);
                        return *__MR_Features_Primitives_ConeSegment_Get_negativeSideRadius(_UnderlyingPtr);
                    }
                }

                //! Distance from the `center` to the cap in the `dir` direction.
                public unsafe float PositiveLength
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_positiveLength", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_Get_positiveLength(_Underlying *_this);
                        return *__MR_Features_Primitives_ConeSegment_Get_positiveLength(_UnderlyingPtr);
                    }
                }

                //! Distance from the `center` to the cap in the direction opposite to `dir`.
                public unsafe float NegativeLength
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_negativeLength", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_Get_negativeLength(_Underlying *_this);
                        return *__MR_Features_Primitives_ConeSegment_Get_negativeLength(_UnderlyingPtr);
                    }
                }

                // If true, the cone has no caps and no volume, and all distances (to the conical surface, that is) are positive.
                public unsafe bool Hollow
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_Get_hollow", ExactSpelling = true)]
                        extern static bool *__MR_Features_Primitives_ConeSegment_Get_hollow(_Underlying *_this);
                        return *__MR_Features_Primitives_ConeSegment_Get_hollow(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_ConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Constructs `MR::Features::Primitives::ConeSegment` elementwise.
                public unsafe Const_ConeSegment(MR.Vector3f referencePoint, MR.Vector3f dir, float positiveSideRadius, float negativeSideRadius, float positiveLength, float negativeLength, bool hollow) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_ConstructFrom(MR.Vector3f referencePoint, MR.Vector3f dir, float positiveSideRadius, float negativeSideRadius, float positiveLength, float negativeLength, byte hollow);
                    _UnderlyingPtr = __MR_Features_Primitives_ConeSegment_ConstructFrom(referencePoint, dir, positiveSideRadius, negativeSideRadius, positiveLength, negativeLength, hollow ? (byte)1 : (byte)0);
                }

                /// Generated from constructor `MR::Features::Primitives::ConeSegment::ConeSegment`.
                public unsafe Const_ConeSegment(MR.Features.Primitives.Const_ConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Primitives.ConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Primitives::ConeSegment::isZeroRadius`.
                public unsafe bool IsZeroRadius()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_isZeroRadius", ExactSpelling = true)]
                    extern static byte __MR_Features_Primitives_ConeSegment_isZeroRadius(_Underlying *_this);
                    return __MR_Features_Primitives_ConeSegment_isZeroRadius(_UnderlyingPtr) != 0;
                }

                /// Generated from method `MR::Features::Primitives::ConeSegment::isCircle`.
                public unsafe bool IsCircle()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_isCircle", ExactSpelling = true)]
                    extern static byte __MR_Features_Primitives_ConeSegment_isCircle(_Underlying *_this);
                    return __MR_Features_Primitives_ConeSegment_isCircle(_UnderlyingPtr) != 0;
                }

                // Returns the length. Can be infinite.
                /// Generated from method `MR::Features::Primitives::ConeSegment::length`.
                public unsafe float Length()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_length", ExactSpelling = true)]
                    extern static float __MR_Features_Primitives_ConeSegment_length(_Underlying *_this);
                    return __MR_Features_Primitives_ConeSegment_length(_UnderlyingPtr);
                }

                // Returns the center point (unlike `referencePoint`, which can actually be off-center).
                // For half-infinite objects, returns the finite end.
                /// Generated from method `MR::Features::Primitives::ConeSegment::centerPoint`.
                public unsafe MR.Sphere3f CenterPoint()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_centerPoint", ExactSpelling = true)]
                    extern static MR.Sphere3f._Underlying *__MR_Features_Primitives_ConeSegment_centerPoint(_Underlying *_this);
                    return new(__MR_Features_Primitives_ConeSegment_centerPoint(_UnderlyingPtr), is_owning: true);
                }

                // Extends the object to infinity in one direction. The radius in the extended direction becomes equal to the radius in the opposite direction.
                /// Generated from method `MR::Features::Primitives::ConeSegment::extendToInfinity`.
                public unsafe MR.Features.Primitives.ConeSegment ExtendToInfinity(bool negative)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_extendToInfinity_1", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_extendToInfinity_1(_Underlying *_this, byte negative);
                    return new(__MR_Features_Primitives_ConeSegment_extendToInfinity_1(_UnderlyingPtr, negative ? (byte)1 : (byte)0), is_owning: true);
                }

                // Extends the object to infinity in both directions. This is equivalent to `.extendToInfinity(false).extendToInfinity(true)`,
                // except that calling it with `positiveSideRadius != negativeSideRadius` is illegal and triggers an assertion.
                /// Generated from method `MR::Features::Primitives::ConeSegment::extendToInfinity`.
                public unsafe MR.Features.Primitives.ConeSegment ExtendToInfinity()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_extendToInfinity_0", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_extendToInfinity_0(_Underlying *_this);
                    return new(__MR_Features_Primitives_ConeSegment_extendToInfinity_0(_UnderlyingPtr), is_owning: true);
                }

                // Untruncates a truncated cone. If it's not a cone at all, returns the object unchanged and triggers an assertion.
                /// Generated from method `MR::Features::Primitives::ConeSegment::untruncateCone`.
                public unsafe MR.Features.Primitives.ConeSegment UntruncateCone()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_untruncateCone", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_untruncateCone(_Underlying *_this);
                    return new(__MR_Features_Primitives_ConeSegment_untruncateCone(_UnderlyingPtr), is_owning: true);
                }

                // Returns a finite axis. For circles, you might want to immediately `extendToInfinity()` it.
                /// Generated from method `MR::Features::Primitives::ConeSegment::axis`.
                public unsafe MR.Features.Primitives.ConeSegment Axis()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_axis", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_axis(_Underlying *_this);
                    return new(__MR_Features_Primitives_ConeSegment_axis(_UnderlyingPtr), is_owning: true);
                }

                // Returns a center of one of the two base circles.
                /// Generated from method `MR::Features::Primitives::ConeSegment::basePoint`.
                public unsafe MR.Sphere3f BasePoint(bool negative)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_basePoint", ExactSpelling = true)]
                    extern static MR.Sphere3f._Underlying *__MR_Features_Primitives_ConeSegment_basePoint(_Underlying *_this, byte negative);
                    return new(__MR_Features_Primitives_ConeSegment_basePoint(_UnderlyingPtr, negative ? (byte)1 : (byte)0), is_owning: true);
                }

                // Returns one of the two base planes.
                /// Generated from method `MR::Features::Primitives::ConeSegment::basePlane`.
                public unsafe MR.Features.Primitives.Plane BasePlane(bool negative)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_basePlane", ExactSpelling = true)]
                    extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_Primitives_ConeSegment_basePlane(_Underlying *_this, byte negative);
                    return new(__MR_Features_Primitives_ConeSegment_basePlane(_UnderlyingPtr, negative ? (byte)1 : (byte)0), is_owning: true);
                }

                // Returns one of the two base circles.
                /// Generated from method `MR::Features::Primitives::ConeSegment::baseCircle`.
                public unsafe MR.Features.Primitives.ConeSegment BaseCircle(bool negative)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_baseCircle", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_baseCircle(_Underlying *_this, byte negative);
                    return new(__MR_Features_Primitives_ConeSegment_baseCircle(_UnderlyingPtr, negative ? (byte)1 : (byte)0), is_owning: true);
                }

                /// Generated from function `MR::Features::Primitives::operator==`.
                public static unsafe bool operator==(MR.Features.Primitives.Const_ConeSegment _1, MR.Features.Primitives.Const_ConeSegment _2)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
                    extern static byte __MR_equal_MR_Features_Primitives_ConeSegment(MR.Features.Primitives.Const_ConeSegment._Underlying *_1, MR.Features.Primitives.Const_ConeSegment._Underlying *_2);
                    return __MR_equal_MR_Features_Primitives_ConeSegment(_1._UnderlyingPtr, _2._UnderlyingPtr) != 0;
                }

                public static unsafe bool operator!=(MR.Features.Primitives.Const_ConeSegment _1, MR.Features.Primitives.Const_ConeSegment _2)
                {
                    return !(_1 == _2);
                }

                // IEquatable:

                public bool Equals(MR.Features.Primitives.Const_ConeSegment? _2)
                {
                    if (_2 is null)
                        return false;
                    return this == _2;
                }

                public override bool Equals(object? other)
                {
                    if (other is null)
                        return false;
                    if (other is MR.Features.Primitives.Const_ConeSegment)
                        return this == (MR.Features.Primitives.Const_ConeSegment)other;
                    return false;
                }
            }

            //! Can have infinite length in one or two directions.
            //! The top and/or bottom can be flat or pointy.
            //! Doubles as a cylinder, line (finite or infinite), and a circle.
            /// Generated from class `MR::Features::Primitives::ConeSegment`.
            /// This is the non-const half of the class.
            public class ConeSegment : Const_ConeSegment
            {
                internal unsafe ConeSegment(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                //! Some point on the axis, but not necessarily the true center point. Use `centerPoint()` for that.
                public new unsafe MR.Mut_Vector3f ReferencePoint
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_referencePoint", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_Primitives_ConeSegment_GetMutable_referencePoint(_Underlying *_this);
                        return new(__MR_Features_Primitives_ConeSegment_GetMutable_referencePoint(_UnderlyingPtr), is_owning: false);
                    }
                }

                //! The axis direction. Must be normalized.
                public new unsafe MR.Mut_Vector3f Dir
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_dir", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_Primitives_ConeSegment_GetMutable_dir(_Underlying *_this);
                        return new(__MR_Features_Primitives_ConeSegment_GetMutable_dir(_UnderlyingPtr), is_owning: false);
                    }
                }

                //! Cap radius in the `dir` direction.
                public new unsafe ref float PositiveSideRadius
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_positiveSideRadius", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_GetMutable_positiveSideRadius(_Underlying *_this);
                        return ref *__MR_Features_Primitives_ConeSegment_GetMutable_positiveSideRadius(_UnderlyingPtr);
                    }
                }

                //! Cap radius in the direction opposite to `dir`.
                public new unsafe ref float NegativeSideRadius
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_negativeSideRadius", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_GetMutable_negativeSideRadius(_Underlying *_this);
                        return ref *__MR_Features_Primitives_ConeSegment_GetMutable_negativeSideRadius(_UnderlyingPtr);
                    }
                }

                //! Distance from the `center` to the cap in the `dir` direction.
                public new unsafe ref float PositiveLength
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_positiveLength", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_GetMutable_positiveLength(_Underlying *_this);
                        return ref *__MR_Features_Primitives_ConeSegment_GetMutable_positiveLength(_UnderlyingPtr);
                    }
                }

                //! Distance from the `center` to the cap in the direction opposite to `dir`.
                public new unsafe ref float NegativeLength
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_negativeLength", ExactSpelling = true)]
                        extern static float *__MR_Features_Primitives_ConeSegment_GetMutable_negativeLength(_Underlying *_this);
                        return ref *__MR_Features_Primitives_ConeSegment_GetMutable_negativeLength(_UnderlyingPtr);
                    }
                }

                // If true, the cone has no caps and no volume, and all distances (to the conical surface, that is) are positive.
                public new unsafe ref bool Hollow
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_GetMutable_hollow", ExactSpelling = true)]
                        extern static bool *__MR_Features_Primitives_ConeSegment_GetMutable_hollow(_Underlying *_this);
                        return ref *__MR_Features_Primitives_ConeSegment_GetMutable_hollow(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe ConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Constructs `MR::Features::Primitives::ConeSegment` elementwise.
                public unsafe ConeSegment(MR.Vector3f referencePoint, MR.Vector3f dir, float positiveSideRadius, float negativeSideRadius, float positiveLength, float negativeLength, bool hollow) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_ConstructFrom(MR.Vector3f referencePoint, MR.Vector3f dir, float positiveSideRadius, float negativeSideRadius, float positiveLength, float negativeLength, byte hollow);
                    _UnderlyingPtr = __MR_Features_Primitives_ConeSegment_ConstructFrom(referencePoint, dir, positiveSideRadius, negativeSideRadius, positiveLength, negativeLength, hollow ? (byte)1 : (byte)0);
                }

                /// Generated from constructor `MR::Features::Primitives::ConeSegment::ConeSegment`.
                public unsafe ConeSegment(MR.Features.Primitives.Const_ConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Primitives.ConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Primitives::ConeSegment::operator=`.
                public unsafe MR.Features.Primitives.ConeSegment Assign(MR.Features.Primitives.Const_ConeSegment _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Primitives_ConeSegment_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_Primitives_ConeSegment_AssignFromAnother(_Underlying *_this, MR.Features.Primitives.ConeSegment._Underlying *_other);
                    return new(__MR_Features_Primitives_ConeSegment_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `ConeSegment` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_ConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `ConeSegment`/`Const_ConeSegment` directly.
            public class _InOptMut_ConeSegment
            {
                public ConeSegment? Opt;

                public _InOptMut_ConeSegment() {}
                public _InOptMut_ConeSegment(ConeSegment value) {Opt = value;}
                public static implicit operator _InOptMut_ConeSegment(ConeSegment value) {return new(value);}
            }

            /// This is used for optional parameters of class `ConeSegment` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_ConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `ConeSegment`/`Const_ConeSegment` to pass it to the function.
            public class _InOptConst_ConeSegment
            {
                public Const_ConeSegment? Opt;

                public _InOptConst_ConeSegment() {}
                public _InOptConst_ConeSegment(Const_ConeSegment value) {Opt = value;}
                public static implicit operator _InOptConst_ConeSegment(Const_ConeSegment value) {return new(value);}
            }
        }

        //! Stores the results of measuring two objects relative to one another.
        /// Generated from class `MR::Features::MeasureResult`.
        /// This is the const half of the class.
        public class Const_MeasureResult : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_MeasureResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Destroy", ExactSpelling = true)]
                extern static void __MR_Features_MeasureResult_Destroy(_Underlying *_this);
                __MR_Features_MeasureResult_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_MeasureResult() {Dispose(false);}

            // Exact distance.
            public unsafe MR.Features.MeasureResult.Const_Distance Distance_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Get_distance", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Const_Distance._Underlying *__MR_Features_MeasureResult_Get_distance(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_Get_distance(_UnderlyingPtr), is_owning: false);
                }
            }

            // Some approximation of the distance.
            // For planes and lines, this expects them to be mostly parallel. For everything else, it just takes the feature center.
            public unsafe MR.Features.MeasureResult.Const_Distance CenterDistance
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Get_centerDistance", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Const_Distance._Underlying *__MR_Features_MeasureResult_Get_centerDistance(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_Get_centerDistance(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Features.MeasureResult.Const_Angle Angle_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Get_angle", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Const_Angle._Underlying *__MR_Features_MeasureResult_Get_angle(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_Get_angle(_UnderlyingPtr), is_owning: false);
                }
            }

            // The primitives obtained from intersecting those two.
            public unsafe MR.Std.Const_Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane Intersections
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Get_intersections", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_Features_MeasureResult_Get_intersections(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_Get_intersections(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_MeasureResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_DefaultConstruct();
                _UnderlyingPtr = __MR_Features_MeasureResult_DefaultConstruct();
            }

            /// Constructs `MR::Features::MeasureResult` elementwise.
            public unsafe Const_MeasureResult(MR.Features.MeasureResult.Const_Distance distance, MR.Features.MeasureResult.Const_Distance centerDistance, MR.Features.MeasureResult.Const_Angle angle, MR.Std._ByValue_Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane intersections) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_ConstructFrom(MR.Features.MeasureResult.Distance._Underlying *distance, MR.Features.MeasureResult.Distance._Underlying *centerDistance, MR.Features.MeasureResult.Angle._Underlying *angle, MR.Misc._PassBy intersections_pass_by, MR.Std.Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *intersections);
                _UnderlyingPtr = __MR_Features_MeasureResult_ConstructFrom(distance._UnderlyingPtr, centerDistance._UnderlyingPtr, angle._UnderlyingPtr, intersections.PassByMode, intersections.Value is not null ? intersections.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::Features::MeasureResult::MeasureResult`.
            public unsafe Const_MeasureResult(MR.Features._ByValue_MeasureResult _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Features.MeasureResult._Underlying *_other);
                _UnderlyingPtr = __MR_Features_MeasureResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from class `MR::Features::MeasureResult::Angle`.
            /// Base classes:
            ///   Direct: (non-virtual)
            ///     `MR::Features::MeasureResult::BasicPart`
            /// This is the const half of the class.
            public class Const_Angle : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Angle(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_MeasureResult_Angle_Destroy(_Underlying *_this);
                    __MR_Features_MeasureResult_Angle_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Angle() {Dispose(false);}

                // Upcasts:
                public static unsafe implicit operator MR.Features.MeasureResult.Const_BasicPart(Const_Angle self)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_UpcastTo_MR_Features_MeasureResult_BasicPart", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Const_BasicPart._Underlying *__MR_Features_MeasureResult_Angle_UpcastTo_MR_Features_MeasureResult_BasicPart(_Underlying *_this);
                    MR.Features.MeasureResult.Const_BasicPart ret = new(__MR_Features_MeasureResult_Angle_UpcastTo_MR_Features_MeasureResult_BasicPart(self._UnderlyingPtr), is_owning: false);
                    ret._KeepAlive(self);
                    return ret;
                }

                public unsafe MR.Const_Vector3f PointA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_pointA", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_Get_pointA(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_Get_pointA(_UnderlyingPtr), is_owning: false);
                    }
                }

                public unsafe MR.Const_Vector3f PointB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_pointB", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_Get_pointB(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_Get_pointB(_UnderlyingPtr), is_owning: false);
                    }
                }

                // Normalized.
                public unsafe MR.Const_Vector3f DirA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_dirA", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_Get_dirA(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_Get_dirA(_UnderlyingPtr), is_owning: false);
                    }
                }

                // ^
                public unsafe MR.Const_Vector3f DirB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_dirB", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_Get_dirB(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_Get_dirB(_UnderlyingPtr), is_owning: false);
                    }
                }

                /// Whether `dir{A,B}` is a surface normal or a line direction.
                public unsafe bool IsSurfaceNormalA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_isSurfaceNormalA", ExactSpelling = true)]
                        extern static bool *__MR_Features_MeasureResult_Angle_Get_isSurfaceNormalA(_Underlying *_this);
                        return *__MR_Features_MeasureResult_Angle_Get_isSurfaceNormalA(_UnderlyingPtr);
                    }
                }

                public unsafe bool IsSurfaceNormalB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_isSurfaceNormalB", ExactSpelling = true)]
                        extern static bool *__MR_Features_MeasureResult_Angle_Get_isSurfaceNormalB(_Underlying *_this);
                        return *__MR_Features_MeasureResult_Angle_Get_isSurfaceNormalB(_UnderlyingPtr);
                    }
                }

                public unsafe MR.Features.MeasureResult.Status Status
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_Get_status", ExactSpelling = true)]
                        extern static MR.Features.MeasureResult.Status *__MR_Features_MeasureResult_Angle_Get_status(_Underlying *_this);
                        return *__MR_Features_MeasureResult_Angle_Get_status(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Angle() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Angle._Underlying *__MR_Features_MeasureResult_Angle_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_MeasureResult_Angle_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::MeasureResult::Angle::Angle`.
                public unsafe Const_Angle(MR.Features.MeasureResult.Const_Angle _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Angle._Underlying *__MR_Features_MeasureResult_Angle_ConstructFromAnother(MR.Features.MeasureResult.Angle._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_MeasureResult_Angle_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from conversion operator `MR::Features::MeasureResult::Angle::operator bool`.
                public static unsafe implicit operator bool(MR.Features.MeasureResult.Const_Angle _this)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_ConvertTo_bool", ExactSpelling = true)]
                    extern static byte __MR_Features_MeasureResult_Angle_ConvertTo_bool(MR.Features.MeasureResult.Const_Angle._Underlying *_this);
                    return __MR_Features_MeasureResult_Angle_ConvertTo_bool(_this._UnderlyingPtr) != 0;
                }

                /// Generated from method `MR::Features::MeasureResult::Angle::pointFor`.
                public unsafe MR.Vector3f PointFor(bool b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_pointFor", ExactSpelling = true)]
                    extern static MR.Vector3f __MR_Features_MeasureResult_Angle_pointFor(_Underlying *_this, byte b);
                    return __MR_Features_MeasureResult_Angle_pointFor(_UnderlyingPtr, b ? (byte)1 : (byte)0);
                }

                /// Generated from method `MR::Features::MeasureResult::Angle::dirFor`.
                public unsafe MR.Vector3f DirFor(bool b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_dirFor", ExactSpelling = true)]
                    extern static MR.Vector3f __MR_Features_MeasureResult_Angle_dirFor(_Underlying *_this, byte b);
                    return __MR_Features_MeasureResult_Angle_dirFor(_UnderlyingPtr, b ? (byte)1 : (byte)0);
                }

                /// Generated from method `MR::Features::MeasureResult::Angle::isSurfaceNormalFor`.
                public unsafe bool IsSurfaceNormalFor(bool b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_isSurfaceNormalFor", ExactSpelling = true)]
                    extern static byte __MR_Features_MeasureResult_Angle_isSurfaceNormalFor(_Underlying *_this, byte b);
                    return __MR_Features_MeasureResult_Angle_isSurfaceNormalFor(_UnderlyingPtr, b ? (byte)1 : (byte)0) != 0;
                }

                /// Generated from method `MR::Features::MeasureResult::Angle::computeAngleInRadians`.
                public unsafe float ComputeAngleInRadians()
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_computeAngleInRadians", ExactSpelling = true)]
                    extern static float __MR_Features_MeasureResult_Angle_computeAngleInRadians(_Underlying *_this);
                    return __MR_Features_MeasureResult_Angle_computeAngleInRadians(_UnderlyingPtr);
                }
            }

            /// Generated from class `MR::Features::MeasureResult::Angle`.
            /// Base classes:
            ///   Direct: (non-virtual)
            ///     `MR::Features::MeasureResult::BasicPart`
            /// This is the non-const half of the class.
            public class Angle : Const_Angle
            {
                internal unsafe Angle(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                // Upcasts:
                public static unsafe implicit operator MR.Features.MeasureResult.BasicPart(Angle self)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_UpcastTo_MR_Features_MeasureResult_BasicPart", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_Angle_UpcastTo_MR_Features_MeasureResult_BasicPart(_Underlying *_this);
                    MR.Features.MeasureResult.BasicPart ret = new(__MR_Features_MeasureResult_Angle_UpcastTo_MR_Features_MeasureResult_BasicPart(self._UnderlyingPtr), is_owning: false);
                    ret._KeepAlive(self);
                    return ret;
                }

                public new unsafe MR.Mut_Vector3f PointA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_pointA", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_GetMutable_pointA(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_GetMutable_pointA(_UnderlyingPtr), is_owning: false);
                    }
                }

                public new unsafe MR.Mut_Vector3f PointB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_pointB", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_GetMutable_pointB(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_GetMutable_pointB(_UnderlyingPtr), is_owning: false);
                    }
                }

                // Normalized.
                public new unsafe MR.Mut_Vector3f DirA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_dirA", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_GetMutable_dirA(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_GetMutable_dirA(_UnderlyingPtr), is_owning: false);
                    }
                }

                // ^
                public new unsafe MR.Mut_Vector3f DirB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_dirB", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_MeasureResult_Angle_GetMutable_dirB(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Angle_GetMutable_dirB(_UnderlyingPtr), is_owning: false);
                    }
                }

                /// Whether `dir{A,B}` is a surface normal or a line direction.
                public new unsafe ref bool IsSurfaceNormalA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_isSurfaceNormalA", ExactSpelling = true)]
                        extern static bool *__MR_Features_MeasureResult_Angle_GetMutable_isSurfaceNormalA(_Underlying *_this);
                        return ref *__MR_Features_MeasureResult_Angle_GetMutable_isSurfaceNormalA(_UnderlyingPtr);
                    }
                }

                public new unsafe ref bool IsSurfaceNormalB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_isSurfaceNormalB", ExactSpelling = true)]
                        extern static bool *__MR_Features_MeasureResult_Angle_GetMutable_isSurfaceNormalB(_Underlying *_this);
                        return ref *__MR_Features_MeasureResult_Angle_GetMutable_isSurfaceNormalB(_UnderlyingPtr);
                    }
                }

                public new unsafe ref MR.Features.MeasureResult.Status Status
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_GetMutable_status", ExactSpelling = true)]
                        extern static MR.Features.MeasureResult.Status *__MR_Features_MeasureResult_Angle_GetMutable_status(_Underlying *_this);
                        return ref *__MR_Features_MeasureResult_Angle_GetMutable_status(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Angle() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Angle._Underlying *__MR_Features_MeasureResult_Angle_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_MeasureResult_Angle_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::MeasureResult::Angle::Angle`.
                public unsafe Angle(MR.Features.MeasureResult.Const_Angle _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Angle._Underlying *__MR_Features_MeasureResult_Angle_ConstructFromAnother(MR.Features.MeasureResult.Angle._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_MeasureResult_Angle_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::MeasureResult::Angle::operator=`.
                public unsafe MR.Features.MeasureResult.Angle Assign(MR.Features.MeasureResult.Const_Angle _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Angle_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Angle._Underlying *__MR_Features_MeasureResult_Angle_AssignFromAnother(_Underlying *_this, MR.Features.MeasureResult.Angle._Underlying *_other);
                    return new(__MR_Features_MeasureResult_Angle_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Angle` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Angle`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Angle`/`Const_Angle` directly.
            public class _InOptMut_Angle
            {
                public Angle? Opt;

                public _InOptMut_Angle() {}
                public _InOptMut_Angle(Angle value) {Opt = value;}
                public static implicit operator _InOptMut_Angle(Angle value) {return new(value);}
            }

            /// This is used for optional parameters of class `Angle` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Angle`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Angle`/`Const_Angle` to pass it to the function.
            public class _InOptConst_Angle
            {
                public Const_Angle? Opt;

                public _InOptConst_Angle() {}
                public _InOptConst_Angle(Const_Angle value) {Opt = value;}
                public static implicit operator _InOptConst_Angle(Const_Angle value) {return new(value);}
            }

            /// Generated from class `MR::Features::MeasureResult::BasicPart`.
            /// Derived classes:
            ///   Direct: (non-virtual)
            ///     `MR::Features::MeasureResult::Angle`
            ///     `MR::Features::MeasureResult::Distance`
            /// This is the const half of the class.
            public class Const_BasicPart : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_BasicPart(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_MeasureResult_BasicPart_Destroy(_Underlying *_this);
                    __MR_Features_MeasureResult_BasicPart_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_BasicPart() {Dispose(false);}

                public unsafe MR.Features.MeasureResult.Status Status
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_Get_status", ExactSpelling = true)]
                        extern static MR.Features.MeasureResult.Status *__MR_Features_MeasureResult_BasicPart_Get_status(_Underlying *_this);
                        return *__MR_Features_MeasureResult_BasicPart_Get_status(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_BasicPart() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_MeasureResult_BasicPart_DefaultConstruct();
                }

                /// Constructs `MR::Features::MeasureResult::BasicPart` elementwise.
                public unsafe Const_BasicPart(MR.Features.MeasureResult.Status status) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_ConstructFrom(MR.Features.MeasureResult.Status status);
                    _UnderlyingPtr = __MR_Features_MeasureResult_BasicPart_ConstructFrom(status);
                }

                /// Generated from constructor `MR::Features::MeasureResult::BasicPart::BasicPart`.
                public unsafe Const_BasicPart(MR.Features.MeasureResult.Const_BasicPart _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_ConstructFromAnother(MR.Features.MeasureResult.BasicPart._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_MeasureResult_BasicPart_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from conversion operator `MR::Features::MeasureResult::BasicPart::operator bool`.
                public static unsafe implicit operator bool(MR.Features.MeasureResult.Const_BasicPart _this)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_ConvertTo_bool", ExactSpelling = true)]
                    extern static byte __MR_Features_MeasureResult_BasicPart_ConvertTo_bool(MR.Features.MeasureResult.Const_BasicPart._Underlying *_this);
                    return __MR_Features_MeasureResult_BasicPart_ConvertTo_bool(_this._UnderlyingPtr) != 0;
                }
            }

            /// Generated from class `MR::Features::MeasureResult::BasicPart`.
            /// Derived classes:
            ///   Direct: (non-virtual)
            ///     `MR::Features::MeasureResult::Angle`
            ///     `MR::Features::MeasureResult::Distance`
            /// This is the non-const half of the class.
            public class BasicPart : Const_BasicPart
            {
                internal unsafe BasicPart(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                public new unsafe ref MR.Features.MeasureResult.Status Status
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_GetMutable_status", ExactSpelling = true)]
                        extern static MR.Features.MeasureResult.Status *__MR_Features_MeasureResult_BasicPart_GetMutable_status(_Underlying *_this);
                        return ref *__MR_Features_MeasureResult_BasicPart_GetMutable_status(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe BasicPart() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_MeasureResult_BasicPart_DefaultConstruct();
                }

                /// Constructs `MR::Features::MeasureResult::BasicPart` elementwise.
                public unsafe BasicPart(MR.Features.MeasureResult.Status status) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_ConstructFrom", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_ConstructFrom(MR.Features.MeasureResult.Status status);
                    _UnderlyingPtr = __MR_Features_MeasureResult_BasicPart_ConstructFrom(status);
                }

                /// Generated from constructor `MR::Features::MeasureResult::BasicPart::BasicPart`.
                public unsafe BasicPart(MR.Features.MeasureResult.Const_BasicPart _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_ConstructFromAnother(MR.Features.MeasureResult.BasicPart._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_MeasureResult_BasicPart_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::MeasureResult::BasicPart::operator=`.
                public unsafe MR.Features.MeasureResult.BasicPart Assign(MR.Features.MeasureResult.Const_BasicPart _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_BasicPart_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_BasicPart_AssignFromAnother(_Underlying *_this, MR.Features.MeasureResult.BasicPart._Underlying *_other);
                    return new(__MR_Features_MeasureResult_BasicPart_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `BasicPart` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_BasicPart`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `BasicPart`/`Const_BasicPart` directly.
            public class _InOptMut_BasicPart
            {
                public BasicPart? Opt;

                public _InOptMut_BasicPart() {}
                public _InOptMut_BasicPart(BasicPart value) {Opt = value;}
                public static implicit operator _InOptMut_BasicPart(BasicPart value) {return new(value);}
            }

            /// This is used for optional parameters of class `BasicPart` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_BasicPart`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `BasicPart`/`Const_BasicPart` to pass it to the function.
            public class _InOptConst_BasicPart
            {
                public Const_BasicPart? Opt;

                public _InOptConst_BasicPart() {}
                public _InOptConst_BasicPart(Const_BasicPart value) {Opt = value;}
                public static implicit operator _InOptConst_BasicPart(Const_BasicPart value) {return new(value);}
            }

            /// Generated from class `MR::Features::MeasureResult::Distance`.
            /// Base classes:
            ///   Direct: (non-virtual)
            ///     `MR::Features::MeasureResult::BasicPart`
            /// This is the const half of the class.
            public class Const_Distance : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Distance(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_MeasureResult_Distance_Destroy(_Underlying *_this);
                    __MR_Features_MeasureResult_Distance_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Distance() {Dispose(false);}

                // Upcasts:
                public static unsafe implicit operator MR.Features.MeasureResult.Const_BasicPart(Const_Distance self)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_UpcastTo_MR_Features_MeasureResult_BasicPart", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Const_BasicPart._Underlying *__MR_Features_MeasureResult_Distance_UpcastTo_MR_Features_MeasureResult_BasicPart(_Underlying *_this);
                    MR.Features.MeasureResult.Const_BasicPart ret = new(__MR_Features_MeasureResult_Distance_UpcastTo_MR_Features_MeasureResult_BasicPart(self._UnderlyingPtr), is_owning: false);
                    ret._KeepAlive(self);
                    return ret;
                }

                // This is a separate field because it can be negative.
                public unsafe float Distance_
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_Get_distance", ExactSpelling = true)]
                        extern static float *__MR_Features_MeasureResult_Distance_Get_distance(_Underlying *_this);
                        return *__MR_Features_MeasureResult_Distance_Get_distance(_UnderlyingPtr);
                    }
                }

                public unsafe MR.Const_Vector3f ClosestPointA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_Get_closestPointA", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_MeasureResult_Distance_Get_closestPointA(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Distance_Get_closestPointA(_UnderlyingPtr), is_owning: false);
                    }
                }

                public unsafe MR.Const_Vector3f ClosestPointB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_Get_closestPointB", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_Features_MeasureResult_Distance_Get_closestPointB(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Distance_Get_closestPointB(_UnderlyingPtr), is_owning: false);
                    }
                }

                public unsafe MR.Features.MeasureResult.Status Status
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_Get_status", ExactSpelling = true)]
                        extern static MR.Features.MeasureResult.Status *__MR_Features_MeasureResult_Distance_Get_status(_Underlying *_this);
                        return *__MR_Features_MeasureResult_Distance_Get_status(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Distance() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_Distance_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_MeasureResult_Distance_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::MeasureResult::Distance::Distance`.
                public unsafe Const_Distance(MR.Features.MeasureResult.Const_Distance _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_Distance_ConstructFromAnother(MR.Features.MeasureResult.Distance._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_MeasureResult_Distance_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from conversion operator `MR::Features::MeasureResult::Distance::operator bool`.
                public static unsafe implicit operator bool(MR.Features.MeasureResult.Const_Distance _this)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_ConvertTo_bool", ExactSpelling = true)]
                    extern static byte __MR_Features_MeasureResult_Distance_ConvertTo_bool(MR.Features.MeasureResult.Const_Distance._Underlying *_this);
                    return __MR_Features_MeasureResult_Distance_ConvertTo_bool(_this._UnderlyingPtr) != 0;
                }

                /// Generated from method `MR::Features::MeasureResult::Distance::closestPointFor`.
                public unsafe MR.Vector3f ClosestPointFor(bool b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_closestPointFor", ExactSpelling = true)]
                    extern static MR.Vector3f __MR_Features_MeasureResult_Distance_closestPointFor(_Underlying *_this, byte b);
                    return __MR_Features_MeasureResult_Distance_closestPointFor(_UnderlyingPtr, b ? (byte)1 : (byte)0);
                }

                /// Generated from method `MR::Features::MeasureResult::Distance::distanceAlongAxis`.
                public unsafe float DistanceAlongAxis(int i)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_distanceAlongAxis", ExactSpelling = true)]
                    extern static float __MR_Features_MeasureResult_Distance_distanceAlongAxis(_Underlying *_this, int i);
                    return __MR_Features_MeasureResult_Distance_distanceAlongAxis(_UnderlyingPtr, i);
                }

                /// Generated from method `MR::Features::MeasureResult::Distance::distanceAlongAxisAbs`.
                public unsafe float DistanceAlongAxisAbs(int i)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_distanceAlongAxisAbs", ExactSpelling = true)]
                    extern static float __MR_Features_MeasureResult_Distance_distanceAlongAxisAbs(_Underlying *_this, int i);
                    return __MR_Features_MeasureResult_Distance_distanceAlongAxisAbs(_UnderlyingPtr, i);
                }
            }

            /// Generated from class `MR::Features::MeasureResult::Distance`.
            /// Base classes:
            ///   Direct: (non-virtual)
            ///     `MR::Features::MeasureResult::BasicPart`
            /// This is the non-const half of the class.
            public class Distance : Const_Distance
            {
                internal unsafe Distance(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                // Upcasts:
                public static unsafe implicit operator MR.Features.MeasureResult.BasicPart(Distance self)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_UpcastTo_MR_Features_MeasureResult_BasicPart", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.BasicPart._Underlying *__MR_Features_MeasureResult_Distance_UpcastTo_MR_Features_MeasureResult_BasicPart(_Underlying *_this);
                    MR.Features.MeasureResult.BasicPart ret = new(__MR_Features_MeasureResult_Distance_UpcastTo_MR_Features_MeasureResult_BasicPart(self._UnderlyingPtr), is_owning: false);
                    ret._KeepAlive(self);
                    return ret;
                }

                // This is a separate field because it can be negative.
                public new unsafe ref float Distance_
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_GetMutable_distance", ExactSpelling = true)]
                        extern static float *__MR_Features_MeasureResult_Distance_GetMutable_distance(_Underlying *_this);
                        return ref *__MR_Features_MeasureResult_Distance_GetMutable_distance(_UnderlyingPtr);
                    }
                }

                public new unsafe MR.Mut_Vector3f ClosestPointA
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_GetMutable_closestPointA", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_MeasureResult_Distance_GetMutable_closestPointA(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Distance_GetMutable_closestPointA(_UnderlyingPtr), is_owning: false);
                    }
                }

                public new unsafe MR.Mut_Vector3f ClosestPointB
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_GetMutable_closestPointB", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_Features_MeasureResult_Distance_GetMutable_closestPointB(_Underlying *_this);
                        return new(__MR_Features_MeasureResult_Distance_GetMutable_closestPointB(_UnderlyingPtr), is_owning: false);
                    }
                }

                public new unsafe ref MR.Features.MeasureResult.Status Status
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_GetMutable_status", ExactSpelling = true)]
                        extern static MR.Features.MeasureResult.Status *__MR_Features_MeasureResult_Distance_GetMutable_status(_Underlying *_this);
                        return ref *__MR_Features_MeasureResult_Distance_GetMutable_status(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Distance() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_Distance_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_MeasureResult_Distance_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::MeasureResult::Distance::Distance`.
                public unsafe Distance(MR.Features.MeasureResult.Const_Distance _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_Distance_ConstructFromAnother(MR.Features.MeasureResult.Distance._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_MeasureResult_Distance_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::MeasureResult::Distance::operator=`.
                public unsafe MR.Features.MeasureResult.Distance Assign(MR.Features.MeasureResult.Const_Distance _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_Distance_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_Distance_AssignFromAnother(_Underlying *_this, MR.Features.MeasureResult.Distance._Underlying *_other);
                    return new(__MR_Features_MeasureResult_Distance_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Distance` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Distance`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Distance`/`Const_Distance` directly.
            public class _InOptMut_Distance
            {
                public Distance? Opt;

                public _InOptMut_Distance() {}
                public _InOptMut_Distance(Distance value) {Opt = value;}
                public static implicit operator _InOptMut_Distance(Distance value) {return new(value);}
            }

            /// This is used for optional parameters of class `Distance` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Distance`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Distance`/`Const_Distance` to pass it to the function.
            public class _InOptConst_Distance
            {
                public Const_Distance? Opt;

                public _InOptConst_Distance() {}
                public _InOptConst_Distance(Const_Distance value) {Opt = value;}
                public static implicit operator _InOptConst_Distance(Const_Distance value) {return new(value);}
            }

            public enum Status : int
            {
                Ok = 0,
                //! Algorithms set this if this when something isn't yet implemented.
                NotImplemented = 1,
                //! Algorithms set this when the calculation doesn't make sense for those object types.
                //! This result can be based on object parameters, but not on their relative location.
                BadFeaturePair = 2,
                //! Can't be computed because of how the objects are located relative to each other.
                BadRelativeLocation = 3,
                //! The result was not finite. This is set automatically if you return non-finite values, but you can also set this manually.
                NotFinite = 4,
            }
        }

        //! Stores the results of measuring two objects relative to one another.
        /// Generated from class `MR::Features::MeasureResult`.
        /// This is the non-const half of the class.
        public class MeasureResult : Const_MeasureResult
        {
            internal unsafe MeasureResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // Exact distance.
            public new unsafe MR.Features.MeasureResult.Distance Distance_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_GetMutable_distance", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_GetMutable_distance(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_GetMutable_distance(_UnderlyingPtr), is_owning: false);
                }
            }

            // Some approximation of the distance.
            // For planes and lines, this expects them to be mostly parallel. For everything else, it just takes the feature center.
            public new unsafe MR.Features.MeasureResult.Distance CenterDistance
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_GetMutable_centerDistance", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Distance._Underlying *__MR_Features_MeasureResult_GetMutable_centerDistance(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_GetMutable_centerDistance(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Features.MeasureResult.Angle Angle_
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_GetMutable_angle", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult.Angle._Underlying *__MR_Features_MeasureResult_GetMutable_angle(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_GetMutable_angle(_UnderlyingPtr), is_owning: false);
                }
            }

            // The primitives obtained from intersecting those two.
            public new unsafe MR.Std.Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane Intersections
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_GetMutable_intersections", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_Features_MeasureResult_GetMutable_intersections(_Underlying *_this);
                    return new(__MR_Features_MeasureResult_GetMutable_intersections(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe MeasureResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_DefaultConstruct();
                _UnderlyingPtr = __MR_Features_MeasureResult_DefaultConstruct();
            }

            /// Constructs `MR::Features::MeasureResult` elementwise.
            public unsafe MeasureResult(MR.Features.MeasureResult.Const_Distance distance, MR.Features.MeasureResult.Const_Distance centerDistance, MR.Features.MeasureResult.Const_Angle angle, MR.Std._ByValue_Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane intersections) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_ConstructFrom(MR.Features.MeasureResult.Distance._Underlying *distance, MR.Features.MeasureResult.Distance._Underlying *centerDistance, MR.Features.MeasureResult.Angle._Underlying *angle, MR.Misc._PassBy intersections_pass_by, MR.Std.Vector_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *intersections);
                _UnderlyingPtr = __MR_Features_MeasureResult_ConstructFrom(distance._UnderlyingPtr, centerDistance._UnderlyingPtr, angle._UnderlyingPtr, intersections.PassByMode, intersections.Value is not null ? intersections.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::Features::MeasureResult::MeasureResult`.
            public unsafe MeasureResult(MR.Features._ByValue_MeasureResult _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Features.MeasureResult._Underlying *_other);
                _UnderlyingPtr = __MR_Features_MeasureResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::Features::MeasureResult::operator=`.
            public unsafe MR.Features.MeasureResult Assign(MR.Features._ByValue_MeasureResult _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Features.MeasureResult._Underlying *__MR_Features_MeasureResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Features.MeasureResult._Underlying *_other);
                return new(__MR_Features_MeasureResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }

            // Modifies the object to swap A and B;
            /// Generated from method `MR::Features::MeasureResult::swapObjects`.
            public unsafe void SwapObjects()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_MeasureResult_swapObjects", ExactSpelling = true)]
                extern static void __MR_Features_MeasureResult_swapObjects(_Underlying *_this);
                __MR_Features_MeasureResult_swapObjects(_UnderlyingPtr);
            }
        }

        /// This is used as a function parameter when the underlying function receives `MeasureResult` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `MeasureResult`/`Const_MeasureResult` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_MeasureResult
        {
            internal readonly Const_MeasureResult? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_MeasureResult() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_MeasureResult(Const_MeasureResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_MeasureResult(Const_MeasureResult arg) {return new(arg);}
            public _ByValue_MeasureResult(MR.Misc._Moved<MeasureResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_MeasureResult(MR.Misc._Moved<MeasureResult> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `MeasureResult` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_MeasureResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `MeasureResult`/`Const_MeasureResult` directly.
        public class _InOptMut_MeasureResult
        {
            public MeasureResult? Opt;

            public _InOptMut_MeasureResult() {}
            public _InOptMut_MeasureResult(MeasureResult value) {Opt = value;}
            public static implicit operator _InOptMut_MeasureResult(MeasureResult value) {return new(value);}
        }

        /// This is used for optional parameters of class `MeasureResult` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_MeasureResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `MeasureResult`/`Const_MeasureResult` to pass it to the function.
        public class _InOptConst_MeasureResult
        {
            public Const_MeasureResult? Opt;

            public _InOptConst_MeasureResult() {}
            public _InOptConst_MeasureResult(Const_MeasureResult value) {Opt = value;}
            public static implicit operator _InOptConst_MeasureResult(Const_MeasureResult value) {return new(value);}
        }

        public static partial class Traits
        {
            /// Generated from class `MR::Features::Traits::Unary<MR::Sphere3f>`.
            /// This is the const half of the class.
            public class Const_Unary_MRSphere3f : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Unary_MRSphere3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Unary_MR_Sphere3f_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Unary_MR_Sphere3f_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Unary_MRSphere3f() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Unary_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRSphere3f._Underlying *__MR_Features_Traits_Unary_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Unary<MR::Sphere3f>::Unary`.
                public unsafe Const_Unary_MRSphere3f(MR.Features.Traits.Const_Unary_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRSphere3f._Underlying *__MR_Features_Traits_Unary_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Unary_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Unary<MR::Sphere3f>::name`.
                public unsafe MR.Misc._Moved<MR.Std.String> Name(MR.Const_Sphere3f prim)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Features_Traits_Unary_MR_Sphere3f_name(_Underlying *_this, MR.Const_Sphere3f._Underlying *prim);
                    return MR.Misc.Move(new MR.Std.String(__MR_Features_Traits_Unary_MR_Sphere3f_name(_UnderlyingPtr, prim._UnderlyingPtr), is_owning: true));
                }
            }

            /// Generated from class `MR::Features::Traits::Unary<MR::Sphere3f>`.
            /// This is the non-const half of the class.
            public class Unary_MRSphere3f : Const_Unary_MRSphere3f
            {
                internal unsafe Unary_MRSphere3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Unary_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRSphere3f._Underlying *__MR_Features_Traits_Unary_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Unary<MR::Sphere3f>::Unary`.
                public unsafe Unary_MRSphere3f(MR.Features.Traits.Const_Unary_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRSphere3f._Underlying *__MR_Features_Traits_Unary_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Unary_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Unary<MR::Sphere3f>::operator=`.
                public unsafe MR.Features.Traits.Unary_MRSphere3f Assign(MR.Features.Traits.Const_Unary_MRSphere3f _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Sphere3f_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRSphere3f._Underlying *__MR_Features_Traits_Unary_MR_Sphere3f_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Unary_MRSphere3f._Underlying *_other);
                    return new(__MR_Features_Traits_Unary_MR_Sphere3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Unary_MRSphere3f` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Unary_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Unary_MRSphere3f`/`Const_Unary_MRSphere3f` directly.
            public class _InOptMut_Unary_MRSphere3f
            {
                public Unary_MRSphere3f? Opt;

                public _InOptMut_Unary_MRSphere3f() {}
                public _InOptMut_Unary_MRSphere3f(Unary_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptMut_Unary_MRSphere3f(Unary_MRSphere3f value) {return new(value);}
            }

            /// This is used for optional parameters of class `Unary_MRSphere3f` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Unary_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Unary_MRSphere3f`/`Const_Unary_MRSphere3f` to pass it to the function.
            public class _InOptConst_Unary_MRSphere3f
            {
                public Const_Unary_MRSphere3f? Opt;

                public _InOptConst_Unary_MRSphere3f() {}
                public _InOptConst_Unary_MRSphere3f(Const_Unary_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptConst_Unary_MRSphere3f(Const_Unary_MRSphere3f value) {return new(value);}
            }

            /// Generated from class `MR::Features::Traits::Unary<MR::Features::Primitives::ConeSegment>`.
            /// This is the const half of the class.
            public class Const_Unary_MRFeaturesPrimitivesConeSegment : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Unary_MRFeaturesPrimitivesConeSegment(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Unary_MRFeaturesPrimitivesConeSegment() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Unary_MRFeaturesPrimitivesConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Unary<MR::Features::Primitives::ConeSegment>::Unary`.
                public unsafe Const_Unary_MRFeaturesPrimitivesConeSegment(MR.Features.Traits.Const_Unary_MRFeaturesPrimitivesConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Unary<MR::Features::Primitives::ConeSegment>::name`.
                public unsafe MR.Misc._Moved<MR.Std.String> Name(MR.Features.Primitives.Const_ConeSegment prim)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_name(_Underlying *_this, MR.Features.Primitives.Const_ConeSegment._Underlying *prim);
                    return MR.Misc.Move(new MR.Std.String(__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_name(_UnderlyingPtr, prim._UnderlyingPtr), is_owning: true));
                }
            }

            /// Generated from class `MR::Features::Traits::Unary<MR::Features::Primitives::ConeSegment>`.
            /// This is the non-const half of the class.
            public class Unary_MRFeaturesPrimitivesConeSegment : Const_Unary_MRFeaturesPrimitivesConeSegment
            {
                internal unsafe Unary_MRFeaturesPrimitivesConeSegment(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Unary_MRFeaturesPrimitivesConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Unary<MR::Features::Primitives::ConeSegment>::Unary`.
                public unsafe Unary_MRFeaturesPrimitivesConeSegment(MR.Features.Traits.Const_Unary_MRFeaturesPrimitivesConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Unary<MR::Features::Primitives::ConeSegment>::operator=`.
                public unsafe MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment Assign(MR.Features.Traits.Const_Unary_MRFeaturesPrimitivesConeSegment _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Unary_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    return new(__MR_Features_Traits_Unary_MR_Features_Primitives_ConeSegment_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Unary_MRFeaturesPrimitivesConeSegment` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Unary_MRFeaturesPrimitivesConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Unary_MRFeaturesPrimitivesConeSegment`/`Const_Unary_MRFeaturesPrimitivesConeSegment` directly.
            public class _InOptMut_Unary_MRFeaturesPrimitivesConeSegment
            {
                public Unary_MRFeaturesPrimitivesConeSegment? Opt;

                public _InOptMut_Unary_MRFeaturesPrimitivesConeSegment() {}
                public _InOptMut_Unary_MRFeaturesPrimitivesConeSegment(Unary_MRFeaturesPrimitivesConeSegment value) {Opt = value;}
                public static implicit operator _InOptMut_Unary_MRFeaturesPrimitivesConeSegment(Unary_MRFeaturesPrimitivesConeSegment value) {return new(value);}
            }

            /// This is used for optional parameters of class `Unary_MRFeaturesPrimitivesConeSegment` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Unary_MRFeaturesPrimitivesConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Unary_MRFeaturesPrimitivesConeSegment`/`Const_Unary_MRFeaturesPrimitivesConeSegment` to pass it to the function.
            public class _InOptConst_Unary_MRFeaturesPrimitivesConeSegment
            {
                public Const_Unary_MRFeaturesPrimitivesConeSegment? Opt;

                public _InOptConst_Unary_MRFeaturesPrimitivesConeSegment() {}
                public _InOptConst_Unary_MRFeaturesPrimitivesConeSegment(Const_Unary_MRFeaturesPrimitivesConeSegment value) {Opt = value;}
                public static implicit operator _InOptConst_Unary_MRFeaturesPrimitivesConeSegment(Const_Unary_MRFeaturesPrimitivesConeSegment value) {return new(value);}
            }

            /// Generated from class `MR::Features::Traits::Unary<MR::Features::Primitives::Plane>`.
            /// This is the const half of the class.
            public class Const_Unary_MRFeaturesPrimitivesPlane : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Unary_MRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Unary_MR_Features_Primitives_Plane_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Unary_MR_Features_Primitives_Plane_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Unary_MRFeaturesPrimitivesPlane() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Unary_MRFeaturesPrimitivesPlane() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_Plane_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Unary<MR::Features::Primitives::Plane>::Unary`.
                public unsafe Const_Unary_MRFeaturesPrimitivesPlane(MR.Features.Traits.Const_Unary_MRFeaturesPrimitivesPlane _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_Plane_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Unary<MR::Features::Primitives::Plane>::name`.
                public unsafe MR.Misc._Moved<MR.Std.String> Name(MR.Features.Primitives.Const_Plane prim)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_name", ExactSpelling = true)]
                    extern static MR.Std.String._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_name(_Underlying *_this, MR.Features.Primitives.Const_Plane._Underlying *prim);
                    return MR.Misc.Move(new MR.Std.String(__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_name(_UnderlyingPtr, prim._UnderlyingPtr), is_owning: true));
                }
            }

            /// Generated from class `MR::Features::Traits::Unary<MR::Features::Primitives::Plane>`.
            /// This is the non-const half of the class.
            public class Unary_MRFeaturesPrimitivesPlane : Const_Unary_MRFeaturesPrimitivesPlane
            {
                internal unsafe Unary_MRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Unary_MRFeaturesPrimitivesPlane() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_Plane_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Unary<MR::Features::Primitives::Plane>::Unary`.
                public unsafe Unary_MRFeaturesPrimitivesPlane(MR.Features.Traits.Const_Unary_MRFeaturesPrimitivesPlane _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Unary_MR_Features_Primitives_Plane_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Unary<MR::Features::Primitives::Plane>::operator=`.
                public unsafe MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane Assign(MR.Features.Traits.Const_Unary_MRFeaturesPrimitivesPlane _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Unary_MR_Features_Primitives_Plane_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Unary_MRFeaturesPrimitivesPlane._Underlying *_other);
                    return new(__MR_Features_Traits_Unary_MR_Features_Primitives_Plane_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Unary_MRFeaturesPrimitivesPlane` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Unary_MRFeaturesPrimitivesPlane`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Unary_MRFeaturesPrimitivesPlane`/`Const_Unary_MRFeaturesPrimitivesPlane` directly.
            public class _InOptMut_Unary_MRFeaturesPrimitivesPlane
            {
                public Unary_MRFeaturesPrimitivesPlane? Opt;

                public _InOptMut_Unary_MRFeaturesPrimitivesPlane() {}
                public _InOptMut_Unary_MRFeaturesPrimitivesPlane(Unary_MRFeaturesPrimitivesPlane value) {Opt = value;}
                public static implicit operator _InOptMut_Unary_MRFeaturesPrimitivesPlane(Unary_MRFeaturesPrimitivesPlane value) {return new(value);}
            }

            /// This is used for optional parameters of class `Unary_MRFeaturesPrimitivesPlane` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Unary_MRFeaturesPrimitivesPlane`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Unary_MRFeaturesPrimitivesPlane`/`Const_Unary_MRFeaturesPrimitivesPlane` to pass it to the function.
            public class _InOptConst_Unary_MRFeaturesPrimitivesPlane
            {
                public Const_Unary_MRFeaturesPrimitivesPlane? Opt;

                public _InOptConst_Unary_MRFeaturesPrimitivesPlane() {}
                public _InOptConst_Unary_MRFeaturesPrimitivesPlane(Const_Unary_MRFeaturesPrimitivesPlane value) {Opt = value;}
                public static implicit operator _InOptConst_Unary_MRFeaturesPrimitivesPlane(Const_Unary_MRFeaturesPrimitivesPlane value) {return new(value);}
            }

            // ?? <-> Sphere
            /// Generated from class `MR::Features::Traits::Binary<MR::Sphere3f, MR::Sphere3f>`.
            /// This is the const half of the class.
            public class Const_Binary_MRSphere3f_MRSphere3f : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Binary_MRSphere3f_MRSphere3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Binary_MRSphere3f_MRSphere3f() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Binary_MRSphere3f_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Sphere3f, MR::Sphere3f>::Binary`.
                public unsafe Const_Binary_MRSphere3f_MRSphere3f(MR.Features.Traits.Const_Binary_MRSphere3f_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Sphere3f, MR::Sphere3f>::measure`.
                public unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Const_Sphere3f a, MR.Const_Sphere3f b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_measure", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult._Underlying *__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_measure(_Underlying *_this, MR.Const_Sphere3f._Underlying *a, MR.Const_Sphere3f._Underlying *b);
                    return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_measure(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
                }
            }

            // ?? <-> Sphere
            /// Generated from class `MR::Features::Traits::Binary<MR::Sphere3f, MR::Sphere3f>`.
            /// This is the non-const half of the class.
            public class Binary_MRSphere3f_MRSphere3f : Const_Binary_MRSphere3f_MRSphere3f
            {
                internal unsafe Binary_MRSphere3f_MRSphere3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Binary_MRSphere3f_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Sphere3f, MR::Sphere3f>::Binary`.
                public unsafe Binary_MRSphere3f_MRSphere3f(MR.Features.Traits.Const_Binary_MRSphere3f_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Sphere3f, MR::Sphere3f>::operator=`.
                public unsafe MR.Features.Traits.Binary_MRSphere3f_MRSphere3f Assign(MR.Features.Traits.Const_Binary_MRSphere3f_MRSphere3f _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Binary_MRSphere3f_MRSphere3f._Underlying *_other);
                    return new(__MR_Features_Traits_Binary_MR_Sphere3f_MR_Sphere3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Binary_MRSphere3f_MRSphere3f` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Binary_MRSphere3f_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRSphere3f_MRSphere3f`/`Const_Binary_MRSphere3f_MRSphere3f` directly.
            public class _InOptMut_Binary_MRSphere3f_MRSphere3f
            {
                public Binary_MRSphere3f_MRSphere3f? Opt;

                public _InOptMut_Binary_MRSphere3f_MRSphere3f() {}
                public _InOptMut_Binary_MRSphere3f_MRSphere3f(Binary_MRSphere3f_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptMut_Binary_MRSphere3f_MRSphere3f(Binary_MRSphere3f_MRSphere3f value) {return new(value);}
            }

            /// This is used for optional parameters of class `Binary_MRSphere3f_MRSphere3f` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Binary_MRSphere3f_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRSphere3f_MRSphere3f`/`Const_Binary_MRSphere3f_MRSphere3f` to pass it to the function.
            public class _InOptConst_Binary_MRSphere3f_MRSphere3f
            {
                public Const_Binary_MRSphere3f_MRSphere3f? Opt;

                public _InOptConst_Binary_MRSphere3f_MRSphere3f() {}
                public _InOptConst_Binary_MRSphere3f_MRSphere3f(Const_Binary_MRSphere3f_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptConst_Binary_MRSphere3f_MRSphere3f(Const_Binary_MRSphere3f_MRSphere3f value) {return new(value);}
            }

            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Sphere3f>`.
            /// This is the const half of the class.
            public class Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Sphere3f>::Binary`.
                public unsafe Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Sphere3f>::measure`.
                public unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Features.Primitives.Const_ConeSegment a, MR.Const_Sphere3f b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_measure", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_measure(_Underlying *_this, MR.Features.Primitives.Const_ConeSegment._Underlying *a, MR.Const_Sphere3f._Underlying *b);
                    return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_measure(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
                }
            }

            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Sphere3f>`.
            /// This is the non-const half of the class.
            public class Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f : Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f
            {
                internal unsafe Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Sphere3f>::Binary`.
                public unsafe Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Sphere3f>::operator=`.
                public unsafe MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f Assign(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f._Underlying *_other);
                    return new(__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Sphere3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f`/`Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f` directly.
            public class _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f
            {
                public Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f? Opt;

                public _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f() {}
                public _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f value) {return new(value);}
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f`/`Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f` to pass it to the function.
            public class _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f
            {
                public Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f? Opt;

                public _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f() {}
                public _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f(Const_Binary_MRFeaturesPrimitivesConeSegment_MRSphere3f value) {return new(value);}
            }

            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Sphere3f>`.
            /// This is the const half of the class.
            public class Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Sphere3f>::Binary`.
                public unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Sphere3f>::measure`.
                public unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Features.Primitives.Const_Plane a, MR.Const_Sphere3f b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_measure", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_measure(_Underlying *_this, MR.Features.Primitives.Const_Plane._Underlying *a, MR.Const_Sphere3f._Underlying *b);
                    return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_measure(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
                }
            }

            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Sphere3f>`.
            /// This is the non-const half of the class.
            public class Binary_MRFeaturesPrimitivesPlane_MRSphere3f : Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f
            {
                internal unsafe Binary_MRFeaturesPrimitivesPlane_MRSphere3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Binary_MRFeaturesPrimitivesPlane_MRSphere3f() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Sphere3f>::Binary`.
                public unsafe Binary_MRFeaturesPrimitivesPlane_MRSphere3f(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Sphere3f>::operator=`.
                public unsafe MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f Assign(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRSphere3f._Underlying *_other);
                    return new(__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Sphere3f_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesPlane_MRSphere3f` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Binary_MRFeaturesPrimitivesPlane_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesPlane_MRSphere3f`/`Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f` directly.
            public class _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRSphere3f
            {
                public Binary_MRFeaturesPrimitivesPlane_MRSphere3f? Opt;

                public _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRSphere3f() {}
                public _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRSphere3f(Binary_MRFeaturesPrimitivesPlane_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRSphere3f(Binary_MRFeaturesPrimitivesPlane_MRSphere3f value) {return new(value);}
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesPlane_MRSphere3f` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Binary_MRFeaturesPrimitivesPlane_MRSphere3f`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesPlane_MRSphere3f`/`Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f` to pass it to the function.
            public class _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRSphere3f
            {
                public Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f? Opt;

                public _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRSphere3f() {}
                public _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRSphere3f(Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f value) {Opt = value;}
                public static implicit operator _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRSphere3f(Const_Binary_MRFeaturesPrimitivesPlane_MRSphere3f value) {return new(value);}
            }

            // ?? <-> Cone
            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Features::Primitives::ConeSegment>`.
            /// This is the const half of the class.
            public class Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Features::Primitives::ConeSegment>::Binary`.
                public unsafe Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Features::Primitives::ConeSegment>::measure`.
                public unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Features.Primitives.Const_ConeSegment a, MR.Features.Primitives.Const_ConeSegment b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_measure", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_measure(_Underlying *_this, MR.Features.Primitives.Const_ConeSegment._Underlying *a, MR.Features.Primitives.Const_ConeSegment._Underlying *b);
                    return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_measure(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
                }
            }

            // ?? <-> Cone
            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Features::Primitives::ConeSegment>`.
            /// This is the non-const half of the class.
            public class Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment : Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment
            {
                internal unsafe Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Features::Primitives::ConeSegment>::Binary`.
                public unsafe Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::ConeSegment, MR::Features::Primitives::ConeSegment>::operator=`.
                public unsafe MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment Assign(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    return new(__MR_Features_Traits_Binary_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_ConeSegment_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment`/`Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment` directly.
            public class _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment
            {
                public Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment? Opt;

                public _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment() {}
                public _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment value) {Opt = value;}
                public static implicit operator _InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment value) {return new(value);}
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment`/`Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment` to pass it to the function.
            public class _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment
            {
                public Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment? Opt;

                public _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment() {}
                public _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment value) {Opt = value;}
                public static implicit operator _InOptConst_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment(Const_Binary_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesConeSegment value) {return new(value);}
            }

            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::ConeSegment>`.
            /// This is the const half of the class.
            public class Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::ConeSegment>::Binary`.
                public unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::ConeSegment>::measure`.
                public unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Features.Primitives.Const_Plane a, MR.Features.Primitives.Const_ConeSegment b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_measure", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_measure(_Underlying *_this, MR.Features.Primitives.Const_Plane._Underlying *a, MR.Features.Primitives.Const_ConeSegment._Underlying *b);
                    return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_measure(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
                }
            }

            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::ConeSegment>`.
            /// This is the non-const half of the class.
            public class Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment : Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment
            {
                internal unsafe Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::ConeSegment>::Binary`.
                public unsafe Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::ConeSegment>::operator=`.
                public unsafe MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment Assign(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment._Underlying *_other);
                    return new(__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_ConeSegment_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment`/`Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment` directly.
            public class _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment
            {
                public Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment? Opt;

                public _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment() {}
                public _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment value) {Opt = value;}
                public static implicit operator _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment value) {return new(value);}
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment`/`Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment` to pass it to the function.
            public class _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment
            {
                public Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment? Opt;

                public _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment() {}
                public _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment value) {Opt = value;}
                public static implicit operator _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment(Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesConeSegment value) {return new(value);}
            }

            // ?? <-> Plane
            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::Plane>`.
            /// This is the const half of the class.
            public class Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_Destroy", ExactSpelling = true)]
                    extern static void __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_Destroy(_Underlying *_this);
                    __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane() {Dispose(false);}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::Plane>::Binary`.
                public unsafe Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::Plane>::measure`.
                public unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Features.Primitives.Const_Plane a, MR.Features.Primitives.Const_Plane b)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_measure", ExactSpelling = true)]
                    extern static MR.Features.MeasureResult._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_measure(_Underlying *_this, MR.Features.Primitives.Const_Plane._Underlying *a, MR.Features.Primitives.Const_Plane._Underlying *b);
                    return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_measure(_UnderlyingPtr, a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
                }
            }

            // ?? <-> Plane
            /// Generated from class `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::Plane>`.
            /// This is the non-const half of the class.
            public class Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane : Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane
            {
                internal unsafe Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                /// Constructs an empty (default-constructed) instance.
                public unsafe Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_DefaultConstruct();
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_DefaultConstruct();
                }

                /// Generated from constructor `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::Plane>::Binary`.
                public unsafe Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_ConstructFromAnother(MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *_other);
                    _UnderlyingPtr = __MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::Features::Traits::Binary<MR::Features::Primitives::Plane, MR::Features::Primitives::Plane>::operator=`.
                public unsafe MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane Assign(MR.Features.Traits.Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_AssignFromAnother(_Underlying *_this, MR.Features.Traits.Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane._Underlying *_other);
                    return new(__MR_Features_Traits_Binary_MR_Features_Primitives_Plane_MR_Features_Primitives_Plane_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane`/`Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane` directly.
            public class _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane
            {
                public Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane? Opt;

                public _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane() {}
                public _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane value) {Opt = value;}
                public static implicit operator _InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane value) {return new(value);}
            }

            /// This is used for optional parameters of class `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane`/`Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane` to pass it to the function.
            public class _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane
            {
                public Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane? Opt;

                public _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane() {}
                public _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane value) {Opt = value;}
                public static implicit operator _InOptConst_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane(Const_Binary_MRFeaturesPrimitivesPlane_MRFeaturesPrimitivesPlane value) {return new(value);}
            }
        }

        // Those map various MR types to our primitives. Some of those are identity functions.
        /// Generated from function `MR::Features::toPrimitive`.
        public static unsafe MR.Sphere3f ToPrimitive(MR.Const_Vector3f point)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toPrimitive_MR_Vector3f", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Features_toPrimitive_MR_Vector3f(MR.Const_Vector3f._Underlying *point);
            return new(__MR_Features_toPrimitive_MR_Vector3f(point._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::toPrimitive`.
        public static unsafe MR.Sphere3f ToPrimitive(MR.Const_Sphere3f sphere)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toPrimitive_MR_Sphere3f", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Features_toPrimitive_MR_Sphere3f(MR.Const_Sphere3f._Underlying *sphere);
            return new(__MR_Features_toPrimitive_MR_Sphere3f(sphere._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::toPrimitive`.
        public static unsafe MR.Features.Primitives.ConeSegment ToPrimitive(MR.Const_Line3f line)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toPrimitive_MR_Line3f", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_toPrimitive_MR_Line3f(MR.Const_Line3f._Underlying *line);
            return new(__MR_Features_toPrimitive_MR_Line3f(line._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::toPrimitive`.
        public static unsafe MR.Features.Primitives.ConeSegment ToPrimitive(MR.Const_LineSegm3f segm)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toPrimitive_MR_LineSegm3f", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_toPrimitive_MR_LineSegm3f(MR.Const_LineSegm3f._Underlying *segm);
            return new(__MR_Features_toPrimitive_MR_LineSegm3f(segm._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::toPrimitive`.
        public static unsafe MR.Features.Primitives.ConeSegment ToPrimitive(MR.Const_Cylinder3f cyl)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toPrimitive_MR_Cylinder3f", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_toPrimitive_MR_Cylinder3f(MR.Const_Cylinder3f._Underlying *cyl);
            return new(__MR_Features_toPrimitive_MR_Cylinder3f(cyl._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::toPrimitive`.
        public static unsafe MR.Features.Primitives.ConeSegment ToPrimitive(MR.Const_Cone3f cone)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toPrimitive_MR_Cone3f", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_toPrimitive_MR_Cone3f(MR.Const_Cone3f._Underlying *cone);
            return new(__MR_Features_toPrimitive_MR_Cone3f(cone._UnderlyingPtr), is_owning: true);
        }

        //! `normal` doesn't need to be normalized.
        /// Generated from function `MR::Features::primitiveCircle`.
        public static unsafe MR.Features.Primitives.ConeSegment PrimitiveCircle(MR.Const_Vector3f point, MR.Const_Vector3f normal, float rad)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_primitiveCircle", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_primitiveCircle(MR.Const_Vector3f._Underlying *point, MR.Const_Vector3f._Underlying *normal, float rad);
            return new(__MR_Features_primitiveCircle(point._UnderlyingPtr, normal._UnderlyingPtr, rad), is_owning: true);
        }

        //! `a` and `b` are centers of the sides.
        /// Generated from function `MR::Features::primitiveCylinder`.
        public static unsafe MR.Features.Primitives.ConeSegment PrimitiveCylinder(MR.Const_Vector3f a, MR.Const_Vector3f b, float rad)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_primitiveCylinder", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_primitiveCylinder(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, float rad);
            return new(__MR_Features_primitiveCylinder(a._UnderlyingPtr, b._UnderlyingPtr, rad), is_owning: true);
        }

        //! `a` is the center of the base, `b` is the pointy end.
        /// Generated from function `MR::Features::primitiveCone`.
        public static unsafe MR.Features.Primitives.ConeSegment PrimitiveCone(MR.Const_Vector3f a, MR.Const_Vector3f b, float rad)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_primitiveCone", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_primitiveCone(MR.Const_Vector3f._Underlying *a, MR.Const_Vector3f._Underlying *b, float rad);
            return new(__MR_Features_primitiveCone(a._UnderlyingPtr, b._UnderlyingPtr, rad), is_owning: true);
        }

        // Returns null if the object type is unknown. This overload ignores the parent xf.
        /// Generated from function `MR::Features::primitiveFromObject`.
        public static unsafe MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane PrimitiveFromObject(MR.Const_Object object_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_primitiveFromObject", ExactSpelling = true)]
            extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_Features_primitiveFromObject(MR.Const_Object._Underlying *object_);
            return new(__MR_Features_primitiveFromObject(object_._UnderlyingPtr), is_owning: true);
        }

        // Returns null if the object type is unknown. This overload respects the parent's `worldXf()`.
        /// Generated from function `MR::Features::primitiveFromObjectWithWorldXf`.
        public static unsafe MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane PrimitiveFromObjectWithWorldXf(MR.Const_Object object_)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_primitiveFromObjectWithWorldXf", ExactSpelling = true)]
            extern static MR.Std.Optional_StdVariantMRSphere3fMRFeaturesPrimitivesConeSegmentMRFeaturesPrimitivesPlane._Underlying *__MR_Features_primitiveFromObjectWithWorldXf(MR.Const_Object._Underlying *object_);
            return new(__MR_Features_primitiveFromObjectWithWorldXf(object_._UnderlyingPtr), is_owning: true);
        }

        // Can return null on some primitive configurations.
        // `infiniteExtent` is how large we make "infinite" objects. Half-infinite objects divide this by 2.
        /// Generated from function `MR::Features::primitiveToObject`.
        public static unsafe MR.Misc._Moved<MR.FeatureObject> PrimitiveToObject(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane primitive, float infiniteExtent)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_primitiveToObject", ExactSpelling = true)]
            extern static MR.FeatureObject._UnderlyingShared *__MR_Features_primitiveToObject(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *primitive, float infiniteExtent);
            return MR.Misc.Move(new MR.FeatureObject(__MR_Features_primitiveToObject(primitive._UnderlyingPtr, infiniteExtent), is_owning: true));
        }

        // Transform a primitive by an xf.
        // Non-uniform scaling and skewing are not supported.
        /// Generated from function `MR::Features::transformPrimitive`.
        public static unsafe MR.Sphere3f TransformPrimitive(MR.Const_AffineXf3f xf, MR.Const_Sphere3f primitive)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_transformPrimitive_MR_Sphere3f", ExactSpelling = true)]
            extern static MR.Sphere3f._Underlying *__MR_Features_transformPrimitive_MR_Sphere3f(MR.Const_AffineXf3f._Underlying *xf, MR.Const_Sphere3f._Underlying *primitive);
            return new(__MR_Features_transformPrimitive_MR_Sphere3f(xf._UnderlyingPtr, primitive._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::transformPrimitive`.
        public static unsafe MR.Features.Primitives.Plane TransformPrimitive(MR.Const_AffineXf3f xf, MR.Features.Primitives.Const_Plane primitive)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_transformPrimitive_MR_Features_Primitives_Plane", ExactSpelling = true)]
            extern static MR.Features.Primitives.Plane._Underlying *__MR_Features_transformPrimitive_MR_Features_Primitives_Plane(MR.Const_AffineXf3f._Underlying *xf, MR.Features.Primitives.Const_Plane._Underlying *primitive);
            return new(__MR_Features_transformPrimitive_MR_Features_Primitives_Plane(xf._UnderlyingPtr, primitive._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::transformPrimitive`.
        public static unsafe MR.Features.Primitives.ConeSegment TransformPrimitive(MR.Const_AffineXf3f xf, MR.Features.Primitives.Const_ConeSegment primitive)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_transformPrimitive_MR_Features_Primitives_ConeSegment", ExactSpelling = true)]
            extern static MR.Features.Primitives.ConeSegment._Underlying *__MR_Features_transformPrimitive_MR_Features_Primitives_ConeSegment(MR.Const_AffineXf3f._Underlying *xf, MR.Features.Primitives.Const_ConeSegment._Underlying *primitive);
            return new(__MR_Features_transformPrimitive_MR_Features_Primitives_ConeSegment(xf._UnderlyingPtr, primitive._UnderlyingPtr), is_owning: true);
        }

        /// Generated from function `MR::Features::transformPrimitive`.
        public static unsafe MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane TransformPrimitive(MR.Const_AffineXf3f xf, MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane primitive)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_transformPrimitive_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane", ExactSpelling = true)]
            extern static MR.Std.Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *__MR_Features_transformPrimitive_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane(MR.Const_AffineXf3f._Underlying *xf, MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *primitive);
            return new(__MR_Features_transformPrimitive_std_variant_MR_Sphere3f_MR_Features_Primitives_ConeSegment_MR_Features_Primitives_Plane(xf._UnderlyingPtr, primitive._UnderlyingPtr), is_owning: true);
        }

        // `MeasureResult::Status` enum to string.
        /// Generated from function `MR::Features::toString`.
        public static unsafe MR.Std.StringView ToString(MR.Features.MeasureResult.Status status)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_toString", ExactSpelling = true)]
            extern static MR.Std.StringView._Underlying *__MR_Features_toString(MR.Features.MeasureResult.Status status);
            return new(__MR_Features_toString(status), is_owning: true);
        }

        // Same but for a variant.
        /// Generated from function `MR::Features::name`.
        public static unsafe MR.Misc._Moved<MR.Std.String> Name(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane var)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_name", ExactSpelling = true)]
            extern static MR.Std.String._Underlying *__MR_Features_name(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *var);
            return MR.Misc.Move(new MR.Std.String(__MR_Features_name(var._UnderlyingPtr), is_owning: true));
        }

        // Same, but with variants as both argument.
        /// Generated from function `MR::Features::measure`.
        public static unsafe MR.Misc._Moved<MR.Features.MeasureResult> Measure(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane a, MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Features_measure", ExactSpelling = true)]
            extern static MR.Features.MeasureResult._Underlying *__MR_Features_measure(MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *a, MR.Std.Const_Variant_MRSphere3f_MRFeaturesPrimitivesConeSegment_MRFeaturesPrimitivesPlane._Underlying *b);
            return MR.Misc.Move(new MR.Features.MeasureResult(__MR_Features_measure(a._UnderlyingPtr, b._UnderlyingPtr), is_owning: true));
        }
    }
}
