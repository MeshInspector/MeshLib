public static partial class MR
{
    /// mapping among elements of source mesh, from which a part is taken, and target mesh
    /// Generated from class `MR::PartMapping`.
    /// This is the const half of the class.
    public class Const_PartMapping : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_PartMapping(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Destroy", ExactSpelling = true)]
            extern static void __MR_PartMapping_Destroy(_Underlying *_this);
            __MR_PartMapping_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_PartMapping() {Dispose(false);}

        // source.id -> target.id
        // each map here can be either dense vector or hash map, the type is set by the user and preserved by mesh copying functions;
        // dense maps are better by speed and memory when source mesh is packed and must be copied entirely;
        // hash maps minimize memory consumption when only a small portion of source mesh is copied
        public unsafe ref void * Src2tgtFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Get_src2tgtFaces", ExactSpelling = true)]
                extern static void **__MR_PartMapping_Get_src2tgtFaces(_Underlying *_this);
                return ref *__MR_PartMapping_Get_src2tgtFaces(_UnderlyingPtr);
            }
        }

        public unsafe ref void * Src2tgtVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Get_src2tgtVerts", ExactSpelling = true)]
                extern static void **__MR_PartMapping_Get_src2tgtVerts(_Underlying *_this);
                return ref *__MR_PartMapping_Get_src2tgtVerts(_UnderlyingPtr);
            }
        }

        public unsafe ref void * Src2tgtEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Get_src2tgtEdges", ExactSpelling = true)]
                extern static void **__MR_PartMapping_Get_src2tgtEdges(_Underlying *_this);
                return ref *__MR_PartMapping_Get_src2tgtEdges(_UnderlyingPtr);
            }
        }

        // target.id -> source.id
        // dense vectors are better by speed and memory when target mesh was empty before copying
        public unsafe ref void * Tgt2srcFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Get_tgt2srcFaces", ExactSpelling = true)]
                extern static void **__MR_PartMapping_Get_tgt2srcFaces(_Underlying *_this);
                return ref *__MR_PartMapping_Get_tgt2srcFaces(_UnderlyingPtr);
            }
        }

        public unsafe ref void * Tgt2srcVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Get_tgt2srcVerts", ExactSpelling = true)]
                extern static void **__MR_PartMapping_Get_tgt2srcVerts(_Underlying *_this);
                return ref *__MR_PartMapping_Get_tgt2srcVerts(_UnderlyingPtr);
            }
        }

        public unsafe ref void * Tgt2srcEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_Get_tgt2srcEdges", ExactSpelling = true)]
                extern static void **__MR_PartMapping_Get_tgt2srcEdges(_Underlying *_this);
                return ref *__MR_PartMapping_Get_tgt2srcEdges(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_PartMapping() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_DefaultConstruct();
            _UnderlyingPtr = __MR_PartMapping_DefaultConstruct();
        }

        /// Constructs `MR::PartMapping` elementwise.
        public unsafe Const_PartMapping(MR.MapOrHashMap_MRFaceId_MRFaceId? src2tgtFaces, MR.MapOrHashMap_MRVertId_MRVertId? src2tgtVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId? src2tgtEdges, MR.MapOrHashMap_MRFaceId_MRFaceId? tgt2srcFaces, MR.MapOrHashMap_MRVertId_MRVertId? tgt2srcVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId? tgt2srcEdges) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_ConstructFrom", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_ConstructFrom(MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *src2tgtFaces, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *src2tgtVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *src2tgtEdges, MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *tgt2srcFaces, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *tgt2srcVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *tgt2srcEdges);
            _UnderlyingPtr = __MR_PartMapping_ConstructFrom(src2tgtFaces is not null ? src2tgtFaces._UnderlyingPtr : null, src2tgtVerts is not null ? src2tgtVerts._UnderlyingPtr : null, src2tgtEdges is not null ? src2tgtEdges._UnderlyingPtr : null, tgt2srcFaces is not null ? tgt2srcFaces._UnderlyingPtr : null, tgt2srcVerts is not null ? tgt2srcVerts._UnderlyingPtr : null, tgt2srcEdges is not null ? tgt2srcEdges._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PartMapping::PartMapping`.
        public unsafe Const_PartMapping(MR.Const_PartMapping _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_ConstructFromAnother(MR.PartMapping._Underlying *_other);
            _UnderlyingPtr = __MR_PartMapping_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// mapping among elements of source mesh, from which a part is taken, and target mesh
    /// Generated from class `MR::PartMapping`.
    /// This is the non-const half of the class.
    public class PartMapping : Const_PartMapping
    {
        internal unsafe PartMapping(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // source.id -> target.id
        // each map here can be either dense vector or hash map, the type is set by the user and preserved by mesh copying functions;
        // dense maps are better by speed and memory when source mesh is packed and must be copied entirely;
        // hash maps minimize memory consumption when only a small portion of source mesh is copied
        public new unsafe ref void * Src2tgtFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_GetMutable_src2tgtFaces", ExactSpelling = true)]
                extern static void **__MR_PartMapping_GetMutable_src2tgtFaces(_Underlying *_this);
                return ref *__MR_PartMapping_GetMutable_src2tgtFaces(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * Src2tgtVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_GetMutable_src2tgtVerts", ExactSpelling = true)]
                extern static void **__MR_PartMapping_GetMutable_src2tgtVerts(_Underlying *_this);
                return ref *__MR_PartMapping_GetMutable_src2tgtVerts(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * Src2tgtEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_GetMutable_src2tgtEdges", ExactSpelling = true)]
                extern static void **__MR_PartMapping_GetMutable_src2tgtEdges(_Underlying *_this);
                return ref *__MR_PartMapping_GetMutable_src2tgtEdges(_UnderlyingPtr);
            }
        }

        // target.id -> source.id
        // dense vectors are better by speed and memory when target mesh was empty before copying
        public new unsafe ref void * Tgt2srcFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_GetMutable_tgt2srcFaces", ExactSpelling = true)]
                extern static void **__MR_PartMapping_GetMutable_tgt2srcFaces(_Underlying *_this);
                return ref *__MR_PartMapping_GetMutable_tgt2srcFaces(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * Tgt2srcVerts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_GetMutable_tgt2srcVerts", ExactSpelling = true)]
                extern static void **__MR_PartMapping_GetMutable_tgt2srcVerts(_Underlying *_this);
                return ref *__MR_PartMapping_GetMutable_tgt2srcVerts(_UnderlyingPtr);
            }
        }

        public new unsafe ref void * Tgt2srcEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_GetMutable_tgt2srcEdges", ExactSpelling = true)]
                extern static void **__MR_PartMapping_GetMutable_tgt2srcEdges(_Underlying *_this);
                return ref *__MR_PartMapping_GetMutable_tgt2srcEdges(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe PartMapping() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_DefaultConstruct", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_DefaultConstruct();
            _UnderlyingPtr = __MR_PartMapping_DefaultConstruct();
        }

        /// Constructs `MR::PartMapping` elementwise.
        public unsafe PartMapping(MR.MapOrHashMap_MRFaceId_MRFaceId? src2tgtFaces, MR.MapOrHashMap_MRVertId_MRVertId? src2tgtVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId? src2tgtEdges, MR.MapOrHashMap_MRFaceId_MRFaceId? tgt2srcFaces, MR.MapOrHashMap_MRVertId_MRVertId? tgt2srcVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId? tgt2srcEdges) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_ConstructFrom", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_ConstructFrom(MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *src2tgtFaces, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *src2tgtVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *src2tgtEdges, MR.MapOrHashMap_MRFaceId_MRFaceId._Underlying *tgt2srcFaces, MR.MapOrHashMap_MRVertId_MRVertId._Underlying *tgt2srcVerts, MR.MapOrHashMap_MRUndirectedEdgeId_MREdgeId._Underlying *tgt2srcEdges);
            _UnderlyingPtr = __MR_PartMapping_ConstructFrom(src2tgtFaces is not null ? src2tgtFaces._UnderlyingPtr : null, src2tgtVerts is not null ? src2tgtVerts._UnderlyingPtr : null, src2tgtEdges is not null ? src2tgtEdges._UnderlyingPtr : null, tgt2srcFaces is not null ? tgt2srcFaces._UnderlyingPtr : null, tgt2srcVerts is not null ? tgt2srcVerts._UnderlyingPtr : null, tgt2srcEdges is not null ? tgt2srcEdges._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::PartMapping::PartMapping`.
        public unsafe PartMapping(MR.Const_PartMapping _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_ConstructFromAnother(MR.PartMapping._Underlying *_other);
            _UnderlyingPtr = __MR_PartMapping_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::PartMapping::operator=`.
        public unsafe MR.PartMapping Assign(MR.Const_PartMapping _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_AssignFromAnother", ExactSpelling = true)]
            extern static MR.PartMapping._Underlying *__MR_PartMapping_AssignFromAnother(_Underlying *_this, MR.PartMapping._Underlying *_other);
            return new(__MR_PartMapping_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// clears all member maps
        /// Generated from method `MR::PartMapping::clear`.
        public unsafe void Clear()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PartMapping_clear", ExactSpelling = true)]
            extern static void __MR_PartMapping_clear(_Underlying *_this);
            __MR_PartMapping_clear(_UnderlyingPtr);
        }
    }

    /// This is used for optional parameters of class `PartMapping` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_PartMapping`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartMapping`/`Const_PartMapping` directly.
    public class _InOptMut_PartMapping
    {
        public PartMapping? Opt;

        public _InOptMut_PartMapping() {}
        public _InOptMut_PartMapping(PartMapping value) {Opt = value;}
        public static implicit operator _InOptMut_PartMapping(PartMapping value) {return new(value);}
    }

    /// This is used for optional parameters of class `PartMapping` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_PartMapping`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `PartMapping`/`Const_PartMapping` to pass it to the function.
    public class _InOptConst_PartMapping
    {
        public Const_PartMapping? Opt;

        public _InOptConst_PartMapping() {}
        public _InOptConst_PartMapping(Const_PartMapping value) {Opt = value;}
        public static implicit operator _InOptConst_PartMapping(Const_PartMapping value) {return new(value);}
    }
}
