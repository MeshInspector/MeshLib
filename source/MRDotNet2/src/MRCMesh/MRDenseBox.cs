public static partial class MR
{
    /// Structure to hold and work with dense box
    /// \details Scalar operations that are not provided in this struct can be called via `box()`
    /// For example `box().size()`, `box().diagonal()` or `box().volume()`
    /// Non const operations are not allowed for dense box because it can spoil density
    /// Generated from class `MR::DenseBox`.
    /// This is the const half of the class.
    public class Const_DenseBox : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DenseBox(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Destroy", ExactSpelling = true)]
            extern static void __MR_DenseBox_Destroy(_Underlying *_this);
            __MR_DenseBox_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DenseBox() {Dispose(false);}

        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe Const_DenseBox(MR.Const_DenseBox _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_ConstructFromAnother(MR.DenseBox._Underlying *_other);
            _UnderlyingPtr = __MR_DenseBox_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Include given points into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe Const_DenseBox(MR.Std.Const_Vector_MRVector3f points, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_std_vector_MR_Vector3f", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_std_vector_MR_Vector3f(points._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include given weighed points into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe Const_DenseBox(MR.Std.Const_Vector_MRVector3f points, MR.Std.Const_Vector_Float weights, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_3", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_3(MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Std.Const_Vector_Float._Underlying *weights, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_3(points._UnderlyingPtr, weights._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include mesh part into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe Const_DenseBox(MR.Const_MeshPart meshPart, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_MR_MeshPart", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_MR_MeshPart(MR.Const_MeshPart._Underlying *meshPart, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_MR_MeshPart(meshPart._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include point into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe Const_DenseBox(MR.Const_PointCloud points, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_MR_PointCloud", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_MR_PointCloud(MR.Const_PointCloud._Underlying *points, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_MR_PointCloud(points._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include line into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe Const_DenseBox(MR.Const_Polyline3 line, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_MR_Polyline3", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_MR_Polyline3(MR.Const_Polyline3._Underlying *line, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_MR_Polyline3(line._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// returns center of dense box
        /// Generated from method `MR::DenseBox::center`.
        public unsafe MR.Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_center", ExactSpelling = true)]
            extern static MR.Vector3f __MR_DenseBox_center(_Underlying *_this);
            return __MR_DenseBox_center(_UnderlyingPtr);
        }

        /// returns corner of dense box, each index value means: false - min, true - max
        /// example: {false, false, flase} - min point, {true, true, true} - max point
        /// Generated from method `MR::DenseBox::corner`.
        public unsafe MR.Vector3f Corner(MR.Const_Vector3b index)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_corner", ExactSpelling = true)]
            extern static MR.Vector3f __MR_DenseBox_corner(_Underlying *_this, MR.Const_Vector3b._Underlying *index);
            return __MR_DenseBox_corner(_UnderlyingPtr, index._UnderlyingPtr);
        }

        /// returns true if dense box contains given point
        /// Generated from method `MR::DenseBox::contains`.
        public unsafe bool Contains(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_contains", ExactSpelling = true)]
            extern static byte __MR_DenseBox_contains(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_DenseBox_contains(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// return box in its space
        /// Generated from method `MR::DenseBox::box`.
        public unsafe MR.Const_Box3f Box()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_box", ExactSpelling = true)]
            extern static MR.Const_Box3f._Underlying *__MR_DenseBox_box(_Underlying *_this);
            return new(__MR_DenseBox_box(_UnderlyingPtr), is_owning: false);
        }

        /// transform box space to world space 
        /// Generated from method `MR::DenseBox::basisXf`.
        public unsafe MR.Const_AffineXf3f BasisXf()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_basisXf", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_DenseBox_basisXf(_Underlying *_this);
            return new(__MR_DenseBox_basisXf(_UnderlyingPtr), is_owning: false);
        }

        /// transform world space to box space
        /// Generated from method `MR::DenseBox::basisXfInv`.
        public unsafe MR.Const_AffineXf3f BasisXfInv()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_basisXfInv", ExactSpelling = true)]
            extern static MR.Const_AffineXf3f._Underlying *__MR_DenseBox_basisXfInv(_Underlying *_this);
            return new(__MR_DenseBox_basisXfInv(_UnderlyingPtr), is_owning: false);
        }
    }

    /// Structure to hold and work with dense box
    /// \details Scalar operations that are not provided in this struct can be called via `box()`
    /// For example `box().size()`, `box().diagonal()` or `box().volume()`
    /// Non const operations are not allowed for dense box because it can spoil density
    /// Generated from class `MR::DenseBox`.
    /// This is the non-const half of the class.
    public class DenseBox : Const_DenseBox
    {
        internal unsafe DenseBox(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe DenseBox(MR.Const_DenseBox _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_ConstructFromAnother(MR.DenseBox._Underlying *_other);
            _UnderlyingPtr = __MR_DenseBox_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Include given points into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe DenseBox(MR.Std.Const_Vector_MRVector3f points, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_std_vector_MR_Vector3f", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_std_vector_MR_Vector3f(MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_std_vector_MR_Vector3f(points._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include given weighed points into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe DenseBox(MR.Std.Const_Vector_MRVector3f points, MR.Std.Const_Vector_Float weights, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_3", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_3(MR.Std.Const_Vector_MRVector3f._Underlying *points, MR.Std.Const_Vector_Float._Underlying *weights, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_3(points._UnderlyingPtr, weights._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include mesh part into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe DenseBox(MR.Const_MeshPart meshPart, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_MR_MeshPart", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_MR_MeshPart(MR.Const_MeshPart._Underlying *meshPart, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_MR_MeshPart(meshPart._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include point into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe DenseBox(MR.Const_PointCloud points, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_MR_PointCloud", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_MR_PointCloud(MR.Const_PointCloud._Underlying *points, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_MR_PointCloud(points._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Include line into this dense box
        /// Generated from constructor `MR::DenseBox::DenseBox`.
        public unsafe DenseBox(MR.Const_Polyline3 line, MR.Const_AffineXf3f? xf = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_Construct_2_MR_Polyline3", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_Construct_2_MR_Polyline3(MR.Const_Polyline3._Underlying *line, MR.Const_AffineXf3f._Underlying *xf);
            _UnderlyingPtr = __MR_DenseBox_Construct_2_MR_Polyline3(line._UnderlyingPtr, xf is not null ? xf._UnderlyingPtr : null);
        }

        /// Generated from method `MR::DenseBox::operator=`.
        public unsafe MR.DenseBox Assign(MR.Const_DenseBox _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DenseBox_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DenseBox._Underlying *__MR_DenseBox_AssignFromAnother(_Underlying *_this, MR.DenseBox._Underlying *_other);
            return new(__MR_DenseBox_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DenseBox` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DenseBox`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DenseBox`/`Const_DenseBox` directly.
    public class _InOptMut_DenseBox
    {
        public DenseBox? Opt;

        public _InOptMut_DenseBox() {}
        public _InOptMut_DenseBox(DenseBox value) {Opt = value;}
        public static implicit operator _InOptMut_DenseBox(DenseBox value) {return new(value);}
    }

    /// This is used for optional parameters of class `DenseBox` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DenseBox`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DenseBox`/`Const_DenseBox` to pass it to the function.
    public class _InOptConst_DenseBox
    {
        public Const_DenseBox? Opt;

        public _InOptConst_DenseBox() {}
        public _InOptConst_DenseBox(Const_DenseBox value) {Opt = value;}
        public static implicit operator _InOptConst_DenseBox(Const_DenseBox value) {return new(value);}
    }
}
