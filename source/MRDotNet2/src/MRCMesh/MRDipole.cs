public static partial class MR
{
    /// Single oriented point or two oppositely charged points close together, representing a mesh part (one or more triangles)
    /// https://www.dgp.toronto.edu/projects/fast-winding-numbers/fast-winding-numbers-for-soups-and-clouds-siggraph-2018-barill-et-al.pdf
    /// Generated from class `MR::Dipole`.
    /// This is the const half of the class.
    public class Const_Dipole : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Dipole(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_Destroy", ExactSpelling = true)]
            extern static void __MR_Dipole_Destroy(_Underlying *_this);
            __MR_Dipole_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Dipole() {Dispose(false);}

        public unsafe MR.Const_Vector3f Pos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_Get_pos", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Dipole_Get_pos(_Underlying *_this);
                return new(__MR_Dipole_Get_pos(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Area
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_Get_area", ExactSpelling = true)]
                extern static float *__MR_Dipole_Get_area(_Underlying *_this);
                return *__MR_Dipole_Get_area(_UnderlyingPtr);
            }
        }

        public unsafe MR.Const_Vector3f DirArea
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_Get_dirArea", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_Dipole_Get_dirArea(_Underlying *_this);
                return new(__MR_Dipole_Get_dirArea(_UnderlyingPtr), is_owning: false);
            }
        }

        // maximum squared distance from pos to any corner of the bounding box
        public unsafe float Rr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_Get_rr", ExactSpelling = true)]
                extern static float *__MR_Dipole_Get_rr(_Underlying *_this);
                return *__MR_Dipole_Get_rr(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Dipole() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_DefaultConstruct();
            _UnderlyingPtr = __MR_Dipole_DefaultConstruct();
        }

        /// Constructs `MR::Dipole` elementwise.
        public unsafe Const_Dipole(MR.Vector3f pos, float area, MR.Vector3f dirArea, float rr) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_ConstructFrom", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_ConstructFrom(MR.Vector3f pos, float area, MR.Vector3f dirArea, float rr);
            _UnderlyingPtr = __MR_Dipole_ConstructFrom(pos, area, dirArea, rr);
        }

        /// Generated from constructor `MR::Dipole::Dipole`.
        public unsafe Const_Dipole(MR.Const_Dipole _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_ConstructFromAnother(MR.Dipole._Underlying *_other);
            _UnderlyingPtr = __MR_Dipole_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// returns true if this dipole is good approximation for a point \param q;
        /// and adds the contribution of this dipole to the winding number at point \param q to \param addTo
        /// Generated from method `MR::Dipole::addIfGoodApprox`.
        public unsafe bool AddIfGoodApprox(MR.Const_Vector3f q, float betaSq, ref float addTo)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_addIfGoodApprox", ExactSpelling = true)]
            extern static byte __MR_Dipole_addIfGoodApprox(_Underlying *_this, MR.Const_Vector3f._Underlying *q, float betaSq, float *addTo);
            fixed (float *__ptr_addTo = &addTo)
            {
                return __MR_Dipole_addIfGoodApprox(_UnderlyingPtr, q._UnderlyingPtr, betaSq, __ptr_addTo) != 0;
            }
        }
    }

    /// Single oriented point or two oppositely charged points close together, representing a mesh part (one or more triangles)
    /// https://www.dgp.toronto.edu/projects/fast-winding-numbers/fast-winding-numbers-for-soups-and-clouds-siggraph-2018-barill-et-al.pdf
    /// Generated from class `MR::Dipole`.
    /// This is the non-const half of the class.
    public class Dipole : Const_Dipole
    {
        internal unsafe Dipole(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f Pos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_GetMutable_pos", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Dipole_GetMutable_pos(_Underlying *_this);
                return new(__MR_Dipole_GetMutable_pos(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Area
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_GetMutable_area", ExactSpelling = true)]
                extern static float *__MR_Dipole_GetMutable_area(_Underlying *_this);
                return ref *__MR_Dipole_GetMutable_area(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Mut_Vector3f DirArea
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_GetMutable_dirArea", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_Dipole_GetMutable_dirArea(_Underlying *_this);
                return new(__MR_Dipole_GetMutable_dirArea(_UnderlyingPtr), is_owning: false);
            }
        }

        // maximum squared distance from pos to any corner of the bounding box
        public new unsafe ref float Rr
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_GetMutable_rr", ExactSpelling = true)]
                extern static float *__MR_Dipole_GetMutable_rr(_Underlying *_this);
                return ref *__MR_Dipole_GetMutable_rr(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Dipole() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_DefaultConstruct();
            _UnderlyingPtr = __MR_Dipole_DefaultConstruct();
        }

        /// Constructs `MR::Dipole` elementwise.
        public unsafe Dipole(MR.Vector3f pos, float area, MR.Vector3f dirArea, float rr) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_ConstructFrom", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_ConstructFrom(MR.Vector3f pos, float area, MR.Vector3f dirArea, float rr);
            _UnderlyingPtr = __MR_Dipole_ConstructFrom(pos, area, dirArea, rr);
        }

        /// Generated from constructor `MR::Dipole::Dipole`.
        public unsafe Dipole(MR.Const_Dipole _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_ConstructFromAnother(MR.Dipole._Underlying *_other);
            _UnderlyingPtr = __MR_Dipole_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::Dipole::operator=`.
        public unsafe MR.Dipole Assign(MR.Const_Dipole _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Dipole_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Dipole._Underlying *__MR_Dipole_AssignFromAnother(_Underlying *_this, MR.Dipole._Underlying *_other);
            return new(__MR_Dipole_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Dipole` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Dipole`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Dipole`/`Const_Dipole` directly.
    public class _InOptMut_Dipole
    {
        public Dipole? Opt;

        public _InOptMut_Dipole() {}
        public _InOptMut_Dipole(Dipole value) {Opt = value;}
        public static implicit operator _InOptMut_Dipole(Dipole value) {return new(value);}
    }

    /// This is used for optional parameters of class `Dipole` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Dipole`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Dipole`/`Const_Dipole` to pass it to the function.
    public class _InOptConst_Dipole
    {
        public Const_Dipole? Opt;

        public _InOptConst_Dipole() {}
        public _InOptConst_Dipole(Const_Dipole value) {Opt = value;}
        public static implicit operator _InOptConst_Dipole(Const_Dipole value) {return new(value);}
    }

    /// calculates dipoles for given mesh and AABB-tree
    /// Generated from function `MR::calcDipoles`.
    public static unsafe void CalcDipoles(MR.Dipoles dipoles, MR.Const_AABBTree tree, MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcDipoles_3", ExactSpelling = true)]
        extern static void __MR_calcDipoles_3(MR.Dipoles._Underlying *dipoles, MR.Const_AABBTree._Underlying *tree, MR.Const_Mesh._Underlying *mesh);
        __MR_calcDipoles_3(dipoles._UnderlyingPtr, tree._UnderlyingPtr, mesh._UnderlyingPtr);
    }

    /// Generated from function `MR::calcDipoles`.
    public static unsafe MR.Misc._Moved<MR.Dipoles> CalcDipoles(MR.Const_AABBTree tree, MR.Const_Mesh mesh)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcDipoles_2", ExactSpelling = true)]
        extern static MR.Dipoles._Underlying *__MR_calcDipoles_2(MR.Const_AABBTree._Underlying *tree, MR.Const_Mesh._Underlying *mesh);
        return MR.Misc.Move(new MR.Dipoles(__MR_calcDipoles_2(tree._UnderlyingPtr, mesh._UnderlyingPtr), is_owning: true));
    }

    /// compute approximate winding number at \param q;
    /// \param beta determines the precision of the approximation: the more the better, recommended value 2 or more;
    /// if distance from q to the center of some triangle group is more than beta times the distance from the center to most distance triangle in the group then we use approximate formula
    /// \param skipFace this triangle (if it is close to \param q) will be skipped from summation
    /// Generated from function `MR::calcFastWindingNumber`.
    public static unsafe float CalcFastWindingNumber(MR.Const_Dipoles dipoles, MR.Const_AABBTree tree, MR.Const_Mesh mesh, MR.Const_Vector3f q, float beta, MR.FaceId skipFace)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_calcFastWindingNumber", ExactSpelling = true)]
        extern static float __MR_calcFastWindingNumber(MR.Const_Dipoles._Underlying *dipoles, MR.Const_AABBTree._Underlying *tree, MR.Const_Mesh._Underlying *mesh, MR.Const_Vector3f._Underlying *q, float beta, MR.FaceId skipFace);
        return __MR_calcFastWindingNumber(dipoles._UnderlyingPtr, tree._UnderlyingPtr, mesh._UnderlyingPtr, q._UnderlyingPtr, beta, skipFace);
    }
}
