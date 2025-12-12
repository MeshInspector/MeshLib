public static partial class MR
{
    /// Generated from class `MR::MeshProjectionResult`.
    /// This is the const half of the class.
    public class Const_MeshProjectionResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshProjectionResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshProjectionResult_Destroy(_Underlying *_this);
            __MR_MeshProjectionResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshProjectionResult() {Dispose(false);}

        /// the closest point on mesh, transformed by xf if it is given
        public unsafe MR.Const_PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_Get_proj", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_MeshProjectionResult_Get_proj(_Underlying *_this);
                return new(__MR_MeshProjectionResult_Get_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// its barycentric representation
        public unsafe MR.Const_MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_Get_mtp", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_MeshProjectionResult_Get_mtp(_Underlying *_this);
                return new(__MR_MeshProjectionResult_Get_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from original projected point to proj
        public unsafe float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_Get_distSq", ExactSpelling = true)]
                extern static float *__MR_MeshProjectionResult_Get_distSq(_Underlying *_this);
                return *__MR_MeshProjectionResult_Get_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshProjectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshProjectionResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshProjectionResult` elementwise.
        public unsafe Const_MeshProjectionResult(MR.Const_PointOnFace proj, MR.Const_MeshTriPoint mtp, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_ConstructFrom(MR.PointOnFace._Underlying *proj, MR.MeshTriPoint._Underlying *mtp, float distSq);
            _UnderlyingPtr = __MR_MeshProjectionResult_ConstructFrom(proj._UnderlyingPtr, mtp._UnderlyingPtr, distSq);
        }

        /// Generated from constructor `MR::MeshProjectionResult::MeshProjectionResult`.
        public unsafe Const_MeshProjectionResult(MR.Const_MeshProjectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_ConstructFromAnother(MR.MeshProjectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshProjectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from conversion operator `MR::MeshProjectionResult::operator bool`.
        public static unsafe explicit operator bool(MR.Const_MeshProjectionResult _this)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_ConvertTo_bool", ExactSpelling = true)]
            extern static byte __MR_MeshProjectionResult_ConvertTo_bool(MR.Const_MeshProjectionResult._Underlying *_this);
            return __MR_MeshProjectionResult_ConvertTo_bool(_this._UnderlyingPtr) != 0;
        }

        /// check for validity, otherwise the projection was not found
        /// Generated from method `MR::MeshProjectionResult::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_valid", ExactSpelling = true)]
            extern static byte __MR_MeshProjectionResult_valid(_Underlying *_this);
            return __MR_MeshProjectionResult_valid(_UnderlyingPtr) != 0;
        }
    }

    /// Generated from class `MR::MeshProjectionResult`.
    /// This is the non-const half of the class.
    public class MeshProjectionResult : Const_MeshProjectionResult
    {
        internal unsafe MeshProjectionResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the closest point on mesh, transformed by xf if it is given
        public new unsafe MR.PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_GetMutable_proj", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_MeshProjectionResult_GetMutable_proj(_Underlying *_this);
                return new(__MR_MeshProjectionResult_GetMutable_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// its barycentric representation
        public new unsafe MR.MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_GetMutable_mtp", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_MeshProjectionResult_GetMutable_mtp(_Underlying *_this);
                return new(__MR_MeshProjectionResult_GetMutable_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// squared distance from original projected point to proj
        public new unsafe ref float DistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_GetMutable_distSq", ExactSpelling = true)]
                extern static float *__MR_MeshProjectionResult_GetMutable_distSq(_Underlying *_this);
                return ref *__MR_MeshProjectionResult_GetMutable_distSq(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshProjectionResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshProjectionResult_DefaultConstruct();
        }

        /// Constructs `MR::MeshProjectionResult` elementwise.
        public unsafe MeshProjectionResult(MR.Const_PointOnFace proj, MR.Const_MeshTriPoint mtp, float distSq) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_ConstructFrom(MR.PointOnFace._Underlying *proj, MR.MeshTriPoint._Underlying *mtp, float distSq);
            _UnderlyingPtr = __MR_MeshProjectionResult_ConstructFrom(proj._UnderlyingPtr, mtp._UnderlyingPtr, distSq);
        }

        /// Generated from constructor `MR::MeshProjectionResult::MeshProjectionResult`.
        public unsafe MeshProjectionResult(MR.Const_MeshProjectionResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_ConstructFromAnother(MR.MeshProjectionResult._Underlying *_other);
            _UnderlyingPtr = __MR_MeshProjectionResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshProjectionResult::operator=`.
        public unsafe MR.MeshProjectionResult Assign(MR.Const_MeshProjectionResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionResult._Underlying *__MR_MeshProjectionResult_AssignFromAnother(_Underlying *_this, MR.MeshProjectionResult._Underlying *_other);
            return new(__MR_MeshProjectionResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshProjectionResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshProjectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshProjectionResult`/`Const_MeshProjectionResult` directly.
    public class _InOptMut_MeshProjectionResult
    {
        public MeshProjectionResult? Opt;

        public _InOptMut_MeshProjectionResult() {}
        public _InOptMut_MeshProjectionResult(MeshProjectionResult value) {Opt = value;}
        public static implicit operator _InOptMut_MeshProjectionResult(MeshProjectionResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshProjectionResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshProjectionResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshProjectionResult`/`Const_MeshProjectionResult` to pass it to the function.
    public class _InOptConst_MeshProjectionResult
    {
        public Const_MeshProjectionResult? Opt;

        public _InOptConst_MeshProjectionResult() {}
        public _InOptConst_MeshProjectionResult(Const_MeshProjectionResult value) {Opt = value;}
        public static implicit operator _InOptConst_MeshProjectionResult(Const_MeshProjectionResult value) {return new(value);}
    }

    /// Generated from class `MR::MeshProjectionTransforms`.
    /// This is the const half of the class.
    public class Const_MeshProjectionTransforms : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshProjectionTransforms(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshProjectionTransforms_Destroy(_Underlying *_this);
            __MR_MeshProjectionTransforms_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshProjectionTransforms() {Dispose(false);}

        ///< this xf is applied to point to move it into projection space
        public unsafe ref readonly MR.AffineXf3f * RigidXfPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_Get_rigidXfPoint", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionTransforms_Get_rigidXfPoint(_Underlying *_this);
                return ref *__MR_MeshProjectionTransforms_Get_rigidXfPoint(_UnderlyingPtr);
            }
        }

        ///< this xf is applied to AABB tree to move it into projection space
        public unsafe ref readonly MR.AffineXf3f * NonRigidXfTree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_Get_nonRigidXfTree", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionTransforms_Get_nonRigidXfTree(_Underlying *_this);
                return ref *__MR_MeshProjectionTransforms_Get_nonRigidXfTree(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshProjectionTransforms() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshProjectionTransforms_DefaultConstruct();
        }

        /// Constructs `MR::MeshProjectionTransforms` elementwise.
        public unsafe Const_MeshProjectionTransforms(MR.Const_AffineXf3f? rigidXfPoint, MR.Const_AffineXf3f? nonRigidXfTree) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_ConstructFrom(MR.Const_AffineXf3f._Underlying *rigidXfPoint, MR.Const_AffineXf3f._Underlying *nonRigidXfTree);
            _UnderlyingPtr = __MR_MeshProjectionTransforms_ConstructFrom(rigidXfPoint is not null ? rigidXfPoint._UnderlyingPtr : null, nonRigidXfTree is not null ? nonRigidXfTree._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshProjectionTransforms::MeshProjectionTransforms`.
        public unsafe Const_MeshProjectionTransforms(MR.Const_MeshProjectionTransforms _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_ConstructFromAnother(MR.MeshProjectionTransforms._Underlying *_other);
            _UnderlyingPtr = __MR_MeshProjectionTransforms_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshProjectionTransforms`.
    /// This is the non-const half of the class.
    public class MeshProjectionTransforms : Const_MeshProjectionTransforms
    {
        internal unsafe MeshProjectionTransforms(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< this xf is applied to point to move it into projection space
        public new unsafe ref readonly MR.AffineXf3f * RigidXfPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_GetMutable_rigidXfPoint", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionTransforms_GetMutable_rigidXfPoint(_Underlying *_this);
                return ref *__MR_MeshProjectionTransforms_GetMutable_rigidXfPoint(_UnderlyingPtr);
            }
        }

        ///< this xf is applied to AABB tree to move it into projection space
        public new unsafe ref readonly MR.AffineXf3f * NonRigidXfTree
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_GetMutable_nonRigidXfTree", ExactSpelling = true)]
                extern static MR.AffineXf3f **__MR_MeshProjectionTransforms_GetMutable_nonRigidXfTree(_Underlying *_this);
                return ref *__MR_MeshProjectionTransforms_GetMutable_nonRigidXfTree(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshProjectionTransforms() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshProjectionTransforms_DefaultConstruct();
        }

        /// Constructs `MR::MeshProjectionTransforms` elementwise.
        public unsafe MeshProjectionTransforms(MR.Const_AffineXf3f? rigidXfPoint, MR.Const_AffineXf3f? nonRigidXfTree) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_ConstructFrom(MR.Const_AffineXf3f._Underlying *rigidXfPoint, MR.Const_AffineXf3f._Underlying *nonRigidXfTree);
            _UnderlyingPtr = __MR_MeshProjectionTransforms_ConstructFrom(rigidXfPoint is not null ? rigidXfPoint._UnderlyingPtr : null, nonRigidXfTree is not null ? nonRigidXfTree._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshProjectionTransforms::MeshProjectionTransforms`.
        public unsafe MeshProjectionTransforms(MR.Const_MeshProjectionTransforms _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_ConstructFromAnother(MR.MeshProjectionTransforms._Underlying *_other);
            _UnderlyingPtr = __MR_MeshProjectionTransforms_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MeshProjectionTransforms::operator=`.
        public unsafe MR.MeshProjectionTransforms Assign(MR.Const_MeshProjectionTransforms _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshProjectionTransforms_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshProjectionTransforms._Underlying *__MR_MeshProjectionTransforms_AssignFromAnother(_Underlying *_this, MR.MeshProjectionTransforms._Underlying *_other);
            return new(__MR_MeshProjectionTransforms_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MeshProjectionTransforms` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshProjectionTransforms`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshProjectionTransforms`/`Const_MeshProjectionTransforms` directly.
    public class _InOptMut_MeshProjectionTransforms
    {
        public MeshProjectionTransforms? Opt;

        public _InOptMut_MeshProjectionTransforms() {}
        public _InOptMut_MeshProjectionTransforms(MeshProjectionTransforms value) {Opt = value;}
        public static implicit operator _InOptMut_MeshProjectionTransforms(MeshProjectionTransforms value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshProjectionTransforms` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshProjectionTransforms`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshProjectionTransforms`/`Const_MeshProjectionTransforms` to pass it to the function.
    public class _InOptConst_MeshProjectionTransforms
    {
        public Const_MeshProjectionTransforms? Opt;

        public _InOptConst_MeshProjectionTransforms() {}
        public _InOptConst_MeshProjectionTransforms(Const_MeshProjectionTransforms value) {Opt = value;}
        public static implicit operator _InOptConst_MeshProjectionTransforms(Const_MeshProjectionTransforms value) {return new(value);}
    }

    /// Generated from class `MR::SignedDistanceToMeshResult`.
    /// This is the const half of the class.
    public class Const_SignedDistanceToMeshResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SignedDistanceToMeshResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_Destroy", ExactSpelling = true)]
            extern static void __MR_SignedDistanceToMeshResult_Destroy(_Underlying *_this);
            __MR_SignedDistanceToMeshResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SignedDistanceToMeshResult() {Dispose(false);}

        /// the closest point on mesh
        public unsafe MR.Const_PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_Get_proj", ExactSpelling = true)]
                extern static MR.Const_PointOnFace._Underlying *__MR_SignedDistanceToMeshResult_Get_proj(_Underlying *_this);
                return new(__MR_SignedDistanceToMeshResult_Get_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// its barycentric representation
        public unsafe MR.Const_MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_Get_mtp", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_SignedDistanceToMeshResult_Get_mtp(_Underlying *_this);
                return new(__MR_SignedDistanceToMeshResult_Get_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// distance from pt to proj (positive - outside, negative - inside the mesh)
        public unsafe float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_Get_dist", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshResult_Get_dist(_Underlying *_this);
                return *__MR_SignedDistanceToMeshResult_Get_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SignedDistanceToMeshResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_DefaultConstruct();
            _UnderlyingPtr = __MR_SignedDistanceToMeshResult_DefaultConstruct();
        }

        /// Constructs `MR::SignedDistanceToMeshResult` elementwise.
        public unsafe Const_SignedDistanceToMeshResult(MR.Const_PointOnFace proj, MR.Const_MeshTriPoint mtp, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_ConstructFrom(MR.PointOnFace._Underlying *proj, MR.MeshTriPoint._Underlying *mtp, float dist);
            _UnderlyingPtr = __MR_SignedDistanceToMeshResult_ConstructFrom(proj._UnderlyingPtr, mtp._UnderlyingPtr, dist);
        }

        /// Generated from constructor `MR::SignedDistanceToMeshResult::SignedDistanceToMeshResult`.
        public unsafe Const_SignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_ConstructFromAnother(MR.SignedDistanceToMeshResult._Underlying *_other);
            _UnderlyingPtr = __MR_SignedDistanceToMeshResult_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::SignedDistanceToMeshResult`.
    /// This is the non-const half of the class.
    public class SignedDistanceToMeshResult : Const_SignedDistanceToMeshResult
    {
        internal unsafe SignedDistanceToMeshResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// the closest point on mesh
        public new unsafe MR.PointOnFace Proj
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_GetMutable_proj", ExactSpelling = true)]
                extern static MR.PointOnFace._Underlying *__MR_SignedDistanceToMeshResult_GetMutable_proj(_Underlying *_this);
                return new(__MR_SignedDistanceToMeshResult_GetMutable_proj(_UnderlyingPtr), is_owning: false);
            }
        }

        /// its barycentric representation
        public new unsafe MR.MeshTriPoint Mtp
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_GetMutable_mtp", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_SignedDistanceToMeshResult_GetMutable_mtp(_Underlying *_this);
                return new(__MR_SignedDistanceToMeshResult_GetMutable_mtp(_UnderlyingPtr), is_owning: false);
            }
        }

        /// distance from pt to proj (positive - outside, negative - inside the mesh)
        public new unsafe ref float Dist
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_GetMutable_dist", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshResult_GetMutable_dist(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshResult_GetMutable_dist(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SignedDistanceToMeshResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_DefaultConstruct();
            _UnderlyingPtr = __MR_SignedDistanceToMeshResult_DefaultConstruct();
        }

        /// Constructs `MR::SignedDistanceToMeshResult` elementwise.
        public unsafe SignedDistanceToMeshResult(MR.Const_PointOnFace proj, MR.Const_MeshTriPoint mtp, float dist) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_ConstructFrom(MR.PointOnFace._Underlying *proj, MR.MeshTriPoint._Underlying *mtp, float dist);
            _UnderlyingPtr = __MR_SignedDistanceToMeshResult_ConstructFrom(proj._UnderlyingPtr, mtp._UnderlyingPtr, dist);
        }

        /// Generated from constructor `MR::SignedDistanceToMeshResult::SignedDistanceToMeshResult`.
        public unsafe SignedDistanceToMeshResult(MR.Const_SignedDistanceToMeshResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_ConstructFromAnother(MR.SignedDistanceToMeshResult._Underlying *_other);
            _UnderlyingPtr = __MR_SignedDistanceToMeshResult_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SignedDistanceToMeshResult::operator=`.
        public unsafe MR.SignedDistanceToMeshResult Assign(MR.Const_SignedDistanceToMeshResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshResult._Underlying *__MR_SignedDistanceToMeshResult_AssignFromAnother(_Underlying *_this, MR.SignedDistanceToMeshResult._Underlying *_other);
            return new(__MR_SignedDistanceToMeshResult_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SignedDistanceToMeshResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SignedDistanceToMeshResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SignedDistanceToMeshResult`/`Const_SignedDistanceToMeshResult` directly.
    public class _InOptMut_SignedDistanceToMeshResult
    {
        public SignedDistanceToMeshResult? Opt;

        public _InOptMut_SignedDistanceToMeshResult() {}
        public _InOptMut_SignedDistanceToMeshResult(SignedDistanceToMeshResult value) {Opt = value;}
        public static implicit operator _InOptMut_SignedDistanceToMeshResult(SignedDistanceToMeshResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `SignedDistanceToMeshResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SignedDistanceToMeshResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SignedDistanceToMeshResult`/`Const_SignedDistanceToMeshResult` to pass it to the function.
    public class _InOptConst_SignedDistanceToMeshResult
    {
        public Const_SignedDistanceToMeshResult? Opt;

        public _InOptConst_SignedDistanceToMeshResult() {}
        public _InOptConst_SignedDistanceToMeshResult(Const_SignedDistanceToMeshResult value) {Opt = value;}
        public static implicit operator _InOptConst_SignedDistanceToMeshResult(Const_SignedDistanceToMeshResult value) {return new(value);}
    }

    /// <summary>
    /// Creates structure with simplified transforms for projection functions, with `rigidXfPoint` applied to point, and `nonRigidXfTree` applied to tree
    /// </summary>
    /// <param name="storageXf">this argument will hold modified transfrom</param>
    /// <param name="pointXf">transform for points to be projected</param>
    /// <param name="treeXf">transform for tree's boxes</param>
    /// <returns>structure with simplified transforms</returns>
    /// Generated from function `MR::createProjectionTransforms`.
    public static unsafe MR.MeshProjectionTransforms CreateProjectionTransforms(MR.Mut_AffineXf3f storageXf, MR.Const_AffineXf3f? pointXf, MR.Const_AffineXf3f? treeXf)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_createProjectionTransforms", ExactSpelling = true)]
        extern static MR.MeshProjectionTransforms._Underlying *__MR_createProjectionTransforms(MR.Mut_AffineXf3f._Underlying *storageXf, MR.Const_AffineXf3f._Underlying *pointXf, MR.Const_AffineXf3f._Underlying *treeXf);
        return new(__MR_createProjectionTransforms(storageXf._UnderlyingPtr, pointXf is not null ? pointXf._UnderlyingPtr : null, treeXf is not null ? treeXf._UnderlyingPtr : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on mesh (or its region) to given point
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    * \param validFaces if provided then only faces from there will be considered as projections
    * \param validProjections if provided then only projections passed this test can be returned
    */
    /// Generated from function `MR::findProjection`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    /// Parameter `validFaces` defaults to `{}`.
    /// Parameter `validProjections` defaults to `{}`.
    public static unsafe MR.MeshProjectionResult FindProjection(MR.Const_Vector3f pt, MR.Const_MeshPart mp, float? upDistLimitSq = null, MR.Const_AffineXf3f? xf = null, float? loDistLimitSq = null, MR.Std.Const_Function_BoolFuncFromMRFaceId? validFaces = null, MR.Std.Const_Function_BoolFuncFromConstMRMeshProjectionResultRef? validProjections = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjection", ExactSpelling = true)]
        extern static MR.MeshProjectionResult._Underlying *__MR_findProjection(MR.Const_Vector3f._Underlying *pt, MR.Const_MeshPart._Underlying *mp, float *upDistLimitSq, MR.Const_AffineXf3f._Underlying *xf, float *loDistLimitSq, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *validFaces, MR.Std.Const_Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *validProjections);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjection(pt._UnderlyingPtr, mp._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null, validFaces is not null ? validFaces._UnderlyingPtr : null, validProjections is not null ? validProjections._UnderlyingPtr : null), is_owning: true);
    }

    /**
    * \brief computes the closest point on mesh (or its region) to given point
    * \param tree explicitly given BVH-tree for whole mesh or part of mesh we are searching projection on,
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger than the function exits returning upDistLimitSq and no valid point
    * \param xf mesh-to-point transformation, if not specified then identity transformation is assumed
    * \param loDistLimitSq low limit on the distance in question, if a point is found within this distance then it is immediately returned without searching for a closer one
    * \param validFaces if provided then only faces from there will be considered as projections
    * \param validProjections if provided then only projections passed this test can be returned
    */
    /// Generated from function `MR::findProjectionSubtree`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    /// Parameter `validFaces` defaults to `{}`.
    /// Parameter `validProjections` defaults to `{}`.
    public static unsafe MR.MeshProjectionResult FindProjectionSubtree(MR.Const_Vector3f pt, MR.Const_MeshPart mp, MR.Const_AABBTree tree, float? upDistLimitSq = null, MR.Const_AffineXf3f? xf = null, float? loDistLimitSq = null, MR.Std.Const_Function_BoolFuncFromMRFaceId? validFaces = null, MR.Std.Const_Function_BoolFuncFromConstMRMeshProjectionResultRef? validProjections = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findProjectionSubtree", ExactSpelling = true)]
        extern static MR.MeshProjectionResult._Underlying *__MR_findProjectionSubtree(MR.Const_Vector3f._Underlying *pt, MR.Const_MeshPart._Underlying *mp, MR.Const_AABBTree._Underlying *tree, float *upDistLimitSq, MR.Const_AffineXf3f._Underlying *xf, float *loDistLimitSq, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *validFaces, MR.Std.Const_Function_BoolFuncFromConstMRMeshProjectionResultRef._Underlying *validProjections);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findProjectionSubtree(pt._UnderlyingPtr, mp._UnderlyingPtr, tree._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, xf is not null ? xf._UnderlyingPtr : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null, validFaces is not null ? validFaces._UnderlyingPtr : null, validProjections is not null ? validProjections._UnderlyingPtr : null), is_owning: true);
    }

    /// enumerates all triangles with bounding boxes at least partially in the ball (the triangles themselves can be fully out of ball)
    /// until callback returns Stop;
    /// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
    /// Generated from function `MR::findBoxedTrisInBall`.
    public static unsafe void FindBoxedTrisInBall(MR.Const_MeshPart mp, MR.Const_Ball3f ball, MR.Std.Const_Function_MRProcessingFuncFromMRFaceIdMRBall3fRef foundCallback)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findBoxedTrisInBall", ExactSpelling = true)]
        extern static void __MR_findBoxedTrisInBall(MR.Const_MeshPart._Underlying *mp, MR.Ball3f._Underlying *ball, MR.Std.Const_Function_MRProcessingFuncFromMRFaceIdMRBall3fRef._Underlying *foundCallback);
        __MR_findBoxedTrisInBall(mp._UnderlyingPtr, ball._UnderlyingPtr, foundCallback._UnderlyingPtr);
    }

    /// enumerates all triangles within the ball until callback returns Stop;
    /// the ball during enumeration can shrink (new ball is always within the previous one) but never expand
    /// Generated from function `MR::findTrisInBall`.
    /// Parameter `validFaces` defaults to `{}`.
    public static unsafe void FindTrisInBall(MR.Const_MeshPart mp, MR.Const_Ball3f ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRMeshProjectionResultRefMRBall3fRef foundCallback, MR.Std.Const_Function_BoolFuncFromMRFaceId? validFaces = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findTrisInBall", ExactSpelling = true)]
        extern static void __MR_findTrisInBall(MR.Const_MeshPart._Underlying *mp, MR.Const_Ball3f._Underlying *ball, MR.Std.Const_Function_MRProcessingFuncFromConstMRMeshProjectionResultRefMRBall3fRef._Underlying *foundCallback, MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *validFaces);
        __MR_findTrisInBall(mp._UnderlyingPtr, ball._UnderlyingPtr, foundCallback._UnderlyingPtr, validFaces is not null ? validFaces._UnderlyingPtr : null);
    }

    /**
    * \brief computes the closest point on mesh (or its region) to given point,
    * and finds the distance with sign to it (positive - outside, negative - inside the mesh)
    * \param upDistLimitSq upper limit on the distance in question, if the real distance is larger then the function exits returning nullopt
    * \param loDistLimitSq low limit on the distance in question, if the real distance smaller then the function exits returning nullopt
    */
    /// Generated from function `MR::findSignedDistance`.
    /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
    /// Parameter `loDistLimitSq` defaults to `0`.
    public static unsafe MR.Std.Optional_MRSignedDistanceToMeshResult FindSignedDistance(MR.Const_Vector3f pt, MR.Const_MeshPart mp, float? upDistLimitSq = null, float? loDistLimitSq = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSignedDistance_MR_Vector3f", ExactSpelling = true)]
        extern static MR.Std.Optional_MRSignedDistanceToMeshResult._Underlying *__MR_findSignedDistance_MR_Vector3f(MR.Const_Vector3f._Underlying *pt, MR.Const_MeshPart._Underlying *mp, float *upDistLimitSq, float *loDistLimitSq);
        float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
        float __deref_loDistLimitSq = loDistLimitSq.GetValueOrDefault();
        return new(__MR_findSignedDistance_MR_Vector3f(pt._UnderlyingPtr, mp._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null, loDistLimitSq.HasValue ? &__deref_loDistLimitSq : null), is_owning: true);
    }
}
