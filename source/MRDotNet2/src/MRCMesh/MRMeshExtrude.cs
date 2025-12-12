public static partial class MR
{
    // holds together settings for makeDegenerateBandAroundRegion
    /// Generated from class `MR::MakeDegenerateBandAroundRegionParams`.
    /// This is the const half of the class.
    public class Const_MakeDegenerateBandAroundRegionParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MakeDegenerateBandAroundRegionParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_Destroy", ExactSpelling = true)]
            extern static void __MR_MakeDegenerateBandAroundRegionParams_Destroy(_Underlying *_this);
            __MR_MakeDegenerateBandAroundRegionParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MakeDegenerateBandAroundRegionParams() {Dispose(false);}

        // (optional) output newly generated faces
        public unsafe ref void * OutNewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_Get_outNewFaces", ExactSpelling = true)]
                extern static void **__MR_MakeDegenerateBandAroundRegionParams_Get_outNewFaces(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_Get_outNewFaces(_UnderlyingPtr);
            }
        }

        // (optional) output edges orthogonal to the boundary
        public unsafe ref void * OutExtrudedEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_Get_outExtrudedEdges", ExactSpelling = true)]
                extern static void **__MR_MakeDegenerateBandAroundRegionParams_Get_outExtrudedEdges(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_Get_outExtrudedEdges(_UnderlyingPtr);
            }
        }

        // (optional) return legth of the longest edges from the boundary of the region
        public unsafe ref float * MaxEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_Get_maxEdgeLength", ExactSpelling = true)]
                extern static float **__MR_MakeDegenerateBandAroundRegionParams_Get_maxEdgeLength(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_Get_maxEdgeLength(_UnderlyingPtr);
            }
        }

        // (optional) map of new vertices to old ones
        public unsafe ref void * New2OldMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_Get_new2OldMap", ExactSpelling = true)]
                extern static void **__MR_MakeDegenerateBandAroundRegionParams_Get_new2OldMap(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_Get_new2OldMap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MakeDegenerateBandAroundRegionParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MakeDegenerateBandAroundRegionParams_DefaultConstruct();
        }

        /// Constructs `MR::MakeDegenerateBandAroundRegionParams` elementwise.
        public unsafe Const_MakeDegenerateBandAroundRegionParams(MR.FaceBitSet? outNewFaces, MR.UndirectedEdgeBitSet? outExtrudedEdges, MR.Misc.InOut<float>? maxEdgeLength, MR.Phmap.FlatHashMap_MRVertId_MRVertId? new2OldMap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_ConstructFrom(MR.FaceBitSet._Underlying *outNewFaces, MR.UndirectedEdgeBitSet._Underlying *outExtrudedEdges, float *maxEdgeLength, MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *new2OldMap);
            float __value_maxEdgeLength = maxEdgeLength is not null ? maxEdgeLength.Value : default(float);
            _UnderlyingPtr = __MR_MakeDegenerateBandAroundRegionParams_ConstructFrom(outNewFaces is not null ? outNewFaces._UnderlyingPtr : null, outExtrudedEdges is not null ? outExtrudedEdges._UnderlyingPtr : null, maxEdgeLength is not null ? &__value_maxEdgeLength : null, new2OldMap is not null ? new2OldMap._UnderlyingPtr : null);
            if (maxEdgeLength is not null) maxEdgeLength.Value = __value_maxEdgeLength;
        }

        /// Generated from constructor `MR::MakeDegenerateBandAroundRegionParams::MakeDegenerateBandAroundRegionParams`.
        public unsafe Const_MakeDegenerateBandAroundRegionParams(MR.Const_MakeDegenerateBandAroundRegionParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_ConstructFromAnother(MR.MakeDegenerateBandAroundRegionParams._Underlying *_other);
            _UnderlyingPtr = __MR_MakeDegenerateBandAroundRegionParams_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // holds together settings for makeDegenerateBandAroundRegion
    /// Generated from class `MR::MakeDegenerateBandAroundRegionParams`.
    /// This is the non-const half of the class.
    public class MakeDegenerateBandAroundRegionParams : Const_MakeDegenerateBandAroundRegionParams
    {
        internal unsafe MakeDegenerateBandAroundRegionParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // (optional) output newly generated faces
        public new unsafe ref void * OutNewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_GetMutable_outNewFaces", ExactSpelling = true)]
                extern static void **__MR_MakeDegenerateBandAroundRegionParams_GetMutable_outNewFaces(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_GetMutable_outNewFaces(_UnderlyingPtr);
            }
        }

        // (optional) output edges orthogonal to the boundary
        public new unsafe ref void * OutExtrudedEdges
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_GetMutable_outExtrudedEdges", ExactSpelling = true)]
                extern static void **__MR_MakeDegenerateBandAroundRegionParams_GetMutable_outExtrudedEdges(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_GetMutable_outExtrudedEdges(_UnderlyingPtr);
            }
        }

        // (optional) return legth of the longest edges from the boundary of the region
        public new unsafe ref float * MaxEdgeLength
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_GetMutable_maxEdgeLength", ExactSpelling = true)]
                extern static float **__MR_MakeDegenerateBandAroundRegionParams_GetMutable_maxEdgeLength(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_GetMutable_maxEdgeLength(_UnderlyingPtr);
            }
        }

        // (optional) map of new vertices to old ones
        public new unsafe ref void * New2OldMap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_GetMutable_new2OldMap", ExactSpelling = true)]
                extern static void **__MR_MakeDegenerateBandAroundRegionParams_GetMutable_new2OldMap(_Underlying *_this);
                return ref *__MR_MakeDegenerateBandAroundRegionParams_GetMutable_new2OldMap(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MakeDegenerateBandAroundRegionParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_DefaultConstruct();
            _UnderlyingPtr = __MR_MakeDegenerateBandAroundRegionParams_DefaultConstruct();
        }

        /// Constructs `MR::MakeDegenerateBandAroundRegionParams` elementwise.
        public unsafe MakeDegenerateBandAroundRegionParams(MR.FaceBitSet? outNewFaces, MR.UndirectedEdgeBitSet? outExtrudedEdges, MR.Misc.InOut<float>? maxEdgeLength, MR.Phmap.FlatHashMap_MRVertId_MRVertId? new2OldMap) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_ConstructFrom(MR.FaceBitSet._Underlying *outNewFaces, MR.UndirectedEdgeBitSet._Underlying *outExtrudedEdges, float *maxEdgeLength, MR.Phmap.FlatHashMap_MRVertId_MRVertId._Underlying *new2OldMap);
            float __value_maxEdgeLength = maxEdgeLength is not null ? maxEdgeLength.Value : default(float);
            _UnderlyingPtr = __MR_MakeDegenerateBandAroundRegionParams_ConstructFrom(outNewFaces is not null ? outNewFaces._UnderlyingPtr : null, outExtrudedEdges is not null ? outExtrudedEdges._UnderlyingPtr : null, maxEdgeLength is not null ? &__value_maxEdgeLength : null, new2OldMap is not null ? new2OldMap._UnderlyingPtr : null);
            if (maxEdgeLength is not null) maxEdgeLength.Value = __value_maxEdgeLength;
        }

        /// Generated from constructor `MR::MakeDegenerateBandAroundRegionParams::MakeDegenerateBandAroundRegionParams`.
        public unsafe MakeDegenerateBandAroundRegionParams(MR.Const_MakeDegenerateBandAroundRegionParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_ConstructFromAnother(MR.MakeDegenerateBandAroundRegionParams._Underlying *_other);
            _UnderlyingPtr = __MR_MakeDegenerateBandAroundRegionParams_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::MakeDegenerateBandAroundRegionParams::operator=`.
        public unsafe MR.MakeDegenerateBandAroundRegionParams Assign(MR.Const_MakeDegenerateBandAroundRegionParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MakeDegenerateBandAroundRegionParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MakeDegenerateBandAroundRegionParams._Underlying *__MR_MakeDegenerateBandAroundRegionParams_AssignFromAnother(_Underlying *_this, MR.MakeDegenerateBandAroundRegionParams._Underlying *_other);
            return new(__MR_MakeDegenerateBandAroundRegionParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `MakeDegenerateBandAroundRegionParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MakeDegenerateBandAroundRegionParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MakeDegenerateBandAroundRegionParams`/`Const_MakeDegenerateBandAroundRegionParams` directly.
    public class _InOptMut_MakeDegenerateBandAroundRegionParams
    {
        public MakeDegenerateBandAroundRegionParams? Opt;

        public _InOptMut_MakeDegenerateBandAroundRegionParams() {}
        public _InOptMut_MakeDegenerateBandAroundRegionParams(MakeDegenerateBandAroundRegionParams value) {Opt = value;}
        public static implicit operator _InOptMut_MakeDegenerateBandAroundRegionParams(MakeDegenerateBandAroundRegionParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `MakeDegenerateBandAroundRegionParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MakeDegenerateBandAroundRegionParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MakeDegenerateBandAroundRegionParams`/`Const_MakeDegenerateBandAroundRegionParams` to pass it to the function.
    public class _InOptConst_MakeDegenerateBandAroundRegionParams
    {
        public Const_MakeDegenerateBandAroundRegionParams? Opt;

        public _InOptConst_MakeDegenerateBandAroundRegionParams() {}
        public _InOptConst_MakeDegenerateBandAroundRegionParams(Const_MakeDegenerateBandAroundRegionParams value) {Opt = value;}
        public static implicit operator _InOptConst_MakeDegenerateBandAroundRegionParams(Const_MakeDegenerateBandAroundRegionParams value) {return new(value);}
    }

    /**
    * \brief Create a band of degenerate faces along the border of the specified region and the rest of the mesh
    * \details The function is useful for extruding the region without changing the existing faces and creating holes
    *
    * @param mesh - the target mesh
    * @param region - the region required to be separated by a band of degenerate faces
    * @param params - optional output parameters
    */
    /// Generated from function `MR::makeDegenerateBandAroundRegion`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe void MakeDegenerateBandAroundRegion(MR.Mesh mesh, MR.Const_FaceBitSet region, MR.Const_MakeDegenerateBandAroundRegionParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_makeDegenerateBandAroundRegion", ExactSpelling = true)]
        extern static void __MR_makeDegenerateBandAroundRegion(MR.Mesh._Underlying *mesh, MR.Const_FaceBitSet._Underlying *region, MR.Const_MakeDegenerateBandAroundRegionParams._Underlying *params_);
        __MR_makeDegenerateBandAroundRegion(mesh._UnderlyingPtr, region._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null);
    }
}
