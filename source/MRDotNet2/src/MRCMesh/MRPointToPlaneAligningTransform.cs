public static partial class MR
{
    /// This class and its main method can be used to solve the problem of 3D shape alignment.
    /// This algorithm uses a point-to-plane error metric in which the object of minimization is the sum of
    /// the squared distance between a point and the tangent plane at its correspondence point.
    /// To use this technique it's need to have small rotation angles. So there is an approximate solution.
    /// The result of this algorithm is the transformation of first points (p1) which aligns it to the second ones (p2).
    /// Generated from class `MR::PointToPlaneAligningTransform`.
    /// This is the const half of the class.
    public class Const_PointToPlaneAligningTransform : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointToPlaneAligningTransform(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_Destroy", ExactSpelling = true)]
            extern static void __MR_PointToPlaneAligningTransform_Destroy(_Underlying *_this);
            __MR_PointToPlaneAligningTransform_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointToPlaneAligningTransform() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointToPlaneAligningTransform() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointToPlaneAligningTransform._Underlying *__MR_PointToPlaneAligningTransform_DefaultConstruct();
            _UnderlyingPtr = __MR_PointToPlaneAligningTransform_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointToPlaneAligningTransform::PointToPlaneAligningTransform`.
        public unsafe Const_PointToPlaneAligningTransform(MR._ByValue_PointToPlaneAligningTransform _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointToPlaneAligningTransform._Underlying *__MR_PointToPlaneAligningTransform_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointToPlaneAligningTransform._Underlying *_other);
            _UnderlyingPtr = __MR_PointToPlaneAligningTransform_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Compute transformation as the solution to a least squares optimization problem:
        /// xf( p1_i ) = p2_i
        /// this version searches for best rigid body transformation
        /// Generated from method `MR::PointToPlaneAligningTransform::findBestRigidXf`.
        public unsafe MR.AffineXf3d FindBestRigidXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_findBestRigidXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPlaneAligningTransform_findBestRigidXf(_Underlying *_this);
            return __MR_PointToPlaneAligningTransform_findBestRigidXf(_UnderlyingPtr);
        }

        /// this version searches for best rigid body transformation with uniform scaling
        /// Generated from method `MR::PointToPlaneAligningTransform::findBestRigidScaleXf`.
        public unsafe MR.AffineXf3d FindBestRigidScaleXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_findBestRigidScaleXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPlaneAligningTransform_findBestRigidScaleXf(_Underlying *_this);
            return __MR_PointToPlaneAligningTransform_findBestRigidScaleXf(_UnderlyingPtr);
        }

        /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
        /// Generated from method `MR::PointToPlaneAligningTransform::findBestRigidXfFixedRotationAxis`.
        public unsafe MR.AffineXf3d FindBestRigidXfFixedRotationAxis(MR.Const_Vector3d axis)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_findBestRigidXfFixedRotationAxis", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPlaneAligningTransform_findBestRigidXfFixedRotationAxis(_Underlying *_this, MR.Const_Vector3d._Underlying *axis);
            return __MR_PointToPlaneAligningTransform_findBestRigidXfFixedRotationAxis(_UnderlyingPtr, axis._UnderlyingPtr);
        }

        /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
        /// Generated from method `MR::PointToPlaneAligningTransform::findBestRigidXfOrthogonalRotationAxis`.
        public unsafe MR.AffineXf3d FindBestRigidXfOrthogonalRotationAxis(MR.Const_Vector3d ort)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_findBestRigidXfOrthogonalRotationAxis", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPlaneAligningTransform_findBestRigidXfOrthogonalRotationAxis(_Underlying *_this, MR.Const_Vector3d._Underlying *ort);
            return __MR_PointToPlaneAligningTransform_findBestRigidXfOrthogonalRotationAxis(_UnderlyingPtr, ort._UnderlyingPtr);
        }

        /// this version searches for best translational part of affine transformation with given linear part
        /// Generated from method `MR::PointToPlaneAligningTransform::findBestTranslation`.
        /// Parameter `rotAngles` defaults to `{}`.
        /// Parameter `scale` defaults to `1`.
        public unsafe MR.Vector3d FindBestTranslation(MR._InOpt_Vector3d rotAngles = default, double? scale = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_findBestTranslation", ExactSpelling = true)]
            extern static MR.Vector3d __MR_PointToPlaneAligningTransform_findBestTranslation(_Underlying *_this, MR.Vector3d *rotAngles, double *scale);
            double __deref_scale = scale.GetValueOrDefault();
            return __MR_PointToPlaneAligningTransform_findBestTranslation(_UnderlyingPtr, rotAngles.HasValue ? &rotAngles.Object : null, scale.HasValue ? &__deref_scale : null);
        }

        /// Compute transformation relative to given approximation and return it as angles and shift (scale = 1)
        /// Generated from method `MR::PointToPlaneAligningTransform::calculateAmendment`.
        public unsafe MR.RigidScaleXf3d CalculateAmendment()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_calculateAmendment", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_PointToPlaneAligningTransform_calculateAmendment(_Underlying *_this);
            return new(__MR_PointToPlaneAligningTransform_calculateAmendment(_UnderlyingPtr), is_owning: true);
        }

        /// Compute transformation relative to given approximation and return it as scale, angles and shift
        /// Generated from method `MR::PointToPlaneAligningTransform::calculateAmendmentWithScale`.
        public unsafe MR.RigidScaleXf3d CalculateAmendmentWithScale()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_calculateAmendmentWithScale", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_PointToPlaneAligningTransform_calculateAmendmentWithScale(_Underlying *_this);
            return new(__MR_PointToPlaneAligningTransform_calculateAmendmentWithScale(_UnderlyingPtr), is_owning: true);
        }

        /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
        /// Generated from method `MR::PointToPlaneAligningTransform::calculateFixedAxisAmendment`.
        public unsafe MR.RigidScaleXf3d CalculateFixedAxisAmendment(MR.Const_Vector3d axis)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_calculateFixedAxisAmendment", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_PointToPlaneAligningTransform_calculateFixedAxisAmendment(_Underlying *_this, MR.Const_Vector3d._Underlying *axis);
            return new(__MR_PointToPlaneAligningTransform_calculateFixedAxisAmendment(_UnderlyingPtr, axis._UnderlyingPtr), is_owning: true);
        }

        /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
        /// Generated from method `MR::PointToPlaneAligningTransform::calculateOrthogonalAxisAmendment`.
        public unsafe MR.RigidScaleXf3d CalculateOrthogonalAxisAmendment(MR.Const_Vector3d ort)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_calculateOrthogonalAxisAmendment", ExactSpelling = true)]
            extern static MR.RigidScaleXf3d._Underlying *__MR_PointToPlaneAligningTransform_calculateOrthogonalAxisAmendment(_Underlying *_this, MR.Const_Vector3d._Underlying *ort);
            return new(__MR_PointToPlaneAligningTransform_calculateOrthogonalAxisAmendment(_UnderlyingPtr, ort._UnderlyingPtr), is_owning: true);
        }
    }

    /// This class and its main method can be used to solve the problem of 3D shape alignment.
    /// This algorithm uses a point-to-plane error metric in which the object of minimization is the sum of
    /// the squared distance between a point and the tangent plane at its correspondence point.
    /// To use this technique it's need to have small rotation angles. So there is an approximate solution.
    /// The result of this algorithm is the transformation of first points (p1) which aligns it to the second ones (p2).
    /// Generated from class `MR::PointToPlaneAligningTransform`.
    /// This is the non-const half of the class.
    public class PointToPlaneAligningTransform : Const_PointToPlaneAligningTransform
    {
        internal unsafe PointToPlaneAligningTransform(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointToPlaneAligningTransform() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointToPlaneAligningTransform._Underlying *__MR_PointToPlaneAligningTransform_DefaultConstruct();
            _UnderlyingPtr = __MR_PointToPlaneAligningTransform_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointToPlaneAligningTransform::PointToPlaneAligningTransform`.
        public unsafe PointToPlaneAligningTransform(MR._ByValue_PointToPlaneAligningTransform _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointToPlaneAligningTransform._Underlying *__MR_PointToPlaneAligningTransform_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PointToPlaneAligningTransform._Underlying *_other);
            _UnderlyingPtr = __MR_PointToPlaneAligningTransform_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::PointToPlaneAligningTransform::operator=`.
        public unsafe MR.PointToPlaneAligningTransform Assign(MR._ByValue_PointToPlaneAligningTransform _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointToPlaneAligningTransform._Underlying *__MR_PointToPlaneAligningTransform_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PointToPlaneAligningTransform._Underlying *_other);
            return new(__MR_PointToPlaneAligningTransform_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Add a pair of corresponding points and the normal of the tangent plane at the second point
        /// Generated from method `MR::PointToPlaneAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(MR.Const_Vector3d p1, MR.Const_Vector3d p2, MR.Const_Vector3d normal2, double? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_add_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_PointToPlaneAligningTransform_add_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *p1, MR.Const_Vector3d._Underlying *p2, MR.Const_Vector3d._Underlying *normal2, double *w);
            double __deref_w = w.GetValueOrDefault();
            __MR_PointToPlaneAligningTransform_add_MR_Vector3d(_UnderlyingPtr, p1._UnderlyingPtr, p2._UnderlyingPtr, normal2._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// Add a pair of corresponding points and the normal of the tangent plane at the second point
        /// Generated from method `MR::PointToPlaneAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(MR.Const_Vector3f p1, MR.Const_Vector3f p2, MR.Const_Vector3f normal2, float? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_add_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_PointToPlaneAligningTransform_add_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *p1, MR.Const_Vector3f._Underlying *p2, MR.Const_Vector3f._Underlying *normal2, float *w);
            float __deref_w = w.GetValueOrDefault();
            __MR_PointToPlaneAligningTransform_add_MR_Vector3f(_UnderlyingPtr, p1._UnderlyingPtr, p2._UnderlyingPtr, normal2._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// this method must be called after add() and before constant find...()/calculate...() to make the matrix symmetric
        /// Generated from method `MR::PointToPlaneAligningTransform::prepare`.
        public unsafe void Prepare()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_prepare", ExactSpelling = true)]
            extern static void __MR_PointToPlaneAligningTransform_prepare(_Underlying *_this);
            __MR_PointToPlaneAligningTransform_prepare(_UnderlyingPtr);
        }

        /// Clear points and normals data
        /// Generated from method `MR::PointToPlaneAligningTransform::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPlaneAligningTransform_clear", ExactSpelling = true)]
            extern static void __MR_PointToPlaneAligningTransform_clear(_Underlying *_this);
            __MR_PointToPlaneAligningTransform_clear(_UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `PointToPlaneAligningTransform` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `PointToPlaneAligningTransform`/`Const_PointToPlaneAligningTransform` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_PointToPlaneAligningTransform
    {
        internal readonly Const_PointToPlaneAligningTransform? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_PointToPlaneAligningTransform() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_PointToPlaneAligningTransform(Const_PointToPlaneAligningTransform new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_PointToPlaneAligningTransform(Const_PointToPlaneAligningTransform arg) {return new(arg);}
        public _ByValue_PointToPlaneAligningTransform(MR.Misc._Moved<PointToPlaneAligningTransform> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_PointToPlaneAligningTransform(MR.Misc._Moved<PointToPlaneAligningTransform> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `PointToPlaneAligningTransform` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointToPlaneAligningTransform`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointToPlaneAligningTransform`/`Const_PointToPlaneAligningTransform` directly.
    public class _InOptMut_PointToPlaneAligningTransform
    {
        public PointToPlaneAligningTransform? Opt;

        public _InOptMut_PointToPlaneAligningTransform() {}
        public _InOptMut_PointToPlaneAligningTransform(PointToPlaneAligningTransform value) {Opt = value;}
        public static implicit operator _InOptMut_PointToPlaneAligningTransform(PointToPlaneAligningTransform value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointToPlaneAligningTransform` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointToPlaneAligningTransform`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointToPlaneAligningTransform`/`Const_PointToPlaneAligningTransform` to pass it to the function.
    public class _InOptConst_PointToPlaneAligningTransform
    {
        public Const_PointToPlaneAligningTransform? Opt;

        public _InOptConst_PointToPlaneAligningTransform() {}
        public _InOptConst_PointToPlaneAligningTransform(Const_PointToPlaneAligningTransform value) {Opt = value;}
        public static implicit operator _InOptConst_PointToPlaneAligningTransform(Const_PointToPlaneAligningTransform value) {return new(value);}
    }
}
