public static partial class MR
{
    /// This class and its main method can be used to solve the problem well-known as the absolute orientation problem.
    /// It means computing the transformation that aligns two sets of points for which correspondence is known.
    /// Generated from class `MR::PointToPointAligningTransform`.
    /// This is the const half of the class.
    public class Const_PointToPointAligningTransform : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PointToPointAligningTransform(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_Destroy", ExactSpelling = true)]
            extern static void __MR_PointToPointAligningTransform_Destroy(_Underlying *_this);
            __MR_PointToPointAligningTransform_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PointToPointAligningTransform() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PointToPointAligningTransform() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointToPointAligningTransform._Underlying *__MR_PointToPointAligningTransform_DefaultConstruct();
            _UnderlyingPtr = __MR_PointToPointAligningTransform_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointToPointAligningTransform::PointToPointAligningTransform`.
        public unsafe Const_PointToPointAligningTransform(MR.Const_PointToPointAligningTransform _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointToPointAligningTransform._Underlying *__MR_PointToPointAligningTransform_ConstructFromAnother(MR.PointToPointAligningTransform._Underlying *_other);
            _UnderlyingPtr = __MR_PointToPointAligningTransform_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns weighted centroid of points p1 accumulated so far
        /// Generated from method `MR::PointToPointAligningTransform::centroid1`.
        public unsafe MR.Vector3d Centroid1()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_centroid1", ExactSpelling = true)]
            extern static MR.Vector3d __MR_PointToPointAligningTransform_centroid1(_Underlying *_this);
            return __MR_PointToPointAligningTransform_centroid1(_UnderlyingPtr);
        }

        /// returns weighted centroid of points p2 accumulated so far
        /// Generated from method `MR::PointToPointAligningTransform::centroid2`.
        public unsafe MR.Vector3d Centroid2()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_centroid2", ExactSpelling = true)]
            extern static MR.Vector3d __MR_PointToPointAligningTransform_centroid2(_Underlying *_this);
            return __MR_PointToPointAligningTransform_centroid2(_UnderlyingPtr);
        }

        /// returns summed weight of points accumulated so far
        /// Generated from method `MR::PointToPointAligningTransform::totalWeight`.
        public unsafe double TotalWeight()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_totalWeight", ExactSpelling = true)]
            extern static double __MR_PointToPointAligningTransform_totalWeight(_Underlying *_this);
            return __MR_PointToPointAligningTransform_totalWeight(_UnderlyingPtr);
        }

        /// Compute transformation as the solution to a least squares formulation of the problem:
        /// xf( p1_i ) = p2_i
        /// this version searches for best rigid body transformation
        /// Generated from method `MR::PointToPointAligningTransform::findBestRigidXf`.
        public unsafe MR.AffineXf3d FindBestRigidXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_findBestRigidXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPointAligningTransform_findBestRigidXf(_Underlying *_this);
            return __MR_PointToPointAligningTransform_findBestRigidXf(_UnderlyingPtr);
        }

        /// this version searches for best rigid body transformation with uniform scaling
        /// Generated from method `MR::PointToPointAligningTransform::findBestRigidScaleXf`.
        public unsafe MR.AffineXf3d FindBestRigidScaleXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_findBestRigidScaleXf", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPointAligningTransform_findBestRigidScaleXf(_Underlying *_this);
            return __MR_PointToPointAligningTransform_findBestRigidScaleXf(_UnderlyingPtr);
        }

        /// this version searches for best transformation where rotation is allowed only around given axis and with arbitrary translation
        /// Generated from method `MR::PointToPointAligningTransform::findBestRigidXfFixedRotationAxis`.
        public unsafe MR.AffineXf3d FindBestRigidXfFixedRotationAxis(MR.Const_Vector3d axis)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_findBestRigidXfFixedRotationAxis", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPointAligningTransform_findBestRigidXfFixedRotationAxis(_Underlying *_this, MR.Const_Vector3d._Underlying *axis);
            return __MR_PointToPointAligningTransform_findBestRigidXfFixedRotationAxis(_UnderlyingPtr, axis._UnderlyingPtr);
        }

        /// this version searches for best transformation where rotation is allowed only around axes orthogonal to given one
        /// Generated from method `MR::PointToPointAligningTransform::findBestRigidXfOrthogonalRotationAxis`.
        public unsafe MR.AffineXf3d FindBestRigidXfOrthogonalRotationAxis(MR.Const_Vector3d ort)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_findBestRigidXfOrthogonalRotationAxis", ExactSpelling = true)]
            extern static MR.AffineXf3d __MR_PointToPointAligningTransform_findBestRigidXfOrthogonalRotationAxis(_Underlying *_this, MR.Const_Vector3d._Underlying *ort);
            return __MR_PointToPointAligningTransform_findBestRigidXfOrthogonalRotationAxis(_UnderlyingPtr, ort._UnderlyingPtr);
        }

        /// Simplified solution for translational part only
        /// Generated from method `MR::PointToPointAligningTransform::findBestTranslation`.
        public unsafe MR.Vector3d FindBestTranslation()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_findBestTranslation", ExactSpelling = true)]
            extern static MR.Vector3d __MR_PointToPointAligningTransform_findBestTranslation(_Underlying *_this);
            return __MR_PointToPointAligningTransform_findBestTranslation(_UnderlyingPtr);
        }
    }

    /// This class and its main method can be used to solve the problem well-known as the absolute orientation problem.
    /// It means computing the transformation that aligns two sets of points for which correspondence is known.
    /// Generated from class `MR::PointToPointAligningTransform`.
    /// This is the non-const half of the class.
    public class PointToPointAligningTransform : Const_PointToPointAligningTransform
    {
        internal unsafe PointToPointAligningTransform(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe PointToPointAligningTransform() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PointToPointAligningTransform._Underlying *__MR_PointToPointAligningTransform_DefaultConstruct();
            _UnderlyingPtr = __MR_PointToPointAligningTransform_DefaultConstruct();
        }

        /// Generated from constructor `MR::PointToPointAligningTransform::PointToPointAligningTransform`.
        public unsafe PointToPointAligningTransform(MR.Const_PointToPointAligningTransform _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PointToPointAligningTransform._Underlying *__MR_PointToPointAligningTransform_ConstructFromAnother(MR.PointToPointAligningTransform._Underlying *_other);
            _UnderlyingPtr = __MR_PointToPointAligningTransform_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PointToPointAligningTransform::operator=`.
        public unsafe MR.PointToPointAligningTransform Assign(MR.Const_PointToPointAligningTransform _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PointToPointAligningTransform._Underlying *__MR_PointToPointAligningTransform_AssignFromAnother(_Underlying *_this, MR.PointToPointAligningTransform._Underlying *_other);
            return new(__MR_PointToPointAligningTransform_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Add one pair of points in the set
        /// Generated from method `MR::PointToPointAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(MR.Const_Vector3d p1, MR.Const_Vector3d p2, double? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_add_3_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_PointToPointAligningTransform_add_3_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *p1, MR.Const_Vector3d._Underlying *p2, double *w);
            double __deref_w = w.GetValueOrDefault();
            __MR_PointToPointAligningTransform_add_3_MR_Vector3d(_UnderlyingPtr, p1._UnderlyingPtr, p2._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// Add one pair of points in the set
        /// Generated from method `MR::PointToPointAligningTransform::add`.
        /// Parameter `w` defaults to `1`.
        public unsafe void Add(MR.Const_Vector3f p1, MR.Const_Vector3f p2, float? w = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_add_3_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_PointToPointAligningTransform_add_3_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *p1, MR.Const_Vector3f._Underlying *p2, float *w);
            float __deref_w = w.GetValueOrDefault();
            __MR_PointToPointAligningTransform_add_3_MR_Vector3f(_UnderlyingPtr, p1._UnderlyingPtr, p2._UnderlyingPtr, w.HasValue ? &__deref_w : null);
        }

        /// Add another two sets of points.
        /// Generated from method `MR::PointToPointAligningTransform::add`.
        public unsafe void Add(MR.Const_PointToPointAligningTransform other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_add_1", ExactSpelling = true)]
            extern static void __MR_PointToPointAligningTransform_add_1(_Underlying *_this, MR.Const_PointToPointAligningTransform._Underlying *other);
            __MR_PointToPointAligningTransform_add_1(_UnderlyingPtr, other._UnderlyingPtr);
        }

        /// Clear sets.
        /// Generated from method `MR::PointToPointAligningTransform::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PointToPointAligningTransform_clear", ExactSpelling = true)]
            extern static void __MR_PointToPointAligningTransform_clear(_Underlying *_this);
            __MR_PointToPointAligningTransform_clear(_UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `PointToPointAligningTransform` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PointToPointAligningTransform`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointToPointAligningTransform`/`Const_PointToPointAligningTransform` directly.
    public class _InOptMut_PointToPointAligningTransform
    {
        public PointToPointAligningTransform? Opt;

        public _InOptMut_PointToPointAligningTransform() {}
        public _InOptMut_PointToPointAligningTransform(PointToPointAligningTransform value) {Opt = value;}
        public static implicit operator _InOptMut_PointToPointAligningTransform(PointToPointAligningTransform value) {return new(value);}
    }

    /// This is used for optional parameters of class `PointToPointAligningTransform` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PointToPointAligningTransform`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PointToPointAligningTransform`/`Const_PointToPointAligningTransform` to pass it to the function.
    public class _InOptConst_PointToPointAligningTransform
    {
        public Const_PointToPointAligningTransform? Opt;

        public _InOptConst_PointToPointAligningTransform() {}
        public _InOptConst_PointToPointAligningTransform(Const_PointToPointAligningTransform value) {Opt = value;}
        public static implicit operator _InOptConst_PointToPointAligningTransform(Const_PointToPointAligningTransform value) {return new(value);}
    }
}
