public static partial class MR
{
    /// this class represents a portion of the sky, and its radiation
    /// Generated from class `MR::SkyPatch`.
    /// This is the const half of the class.
    public class Const_SkyPatch : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SkyPatch(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_Destroy", ExactSpelling = true)]
            extern static void __MR_SkyPatch_Destroy(_Underlying *_this);
            __MR_SkyPatch_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SkyPatch() {Dispose(false);}

        /// direction toward the center of the patch
        public unsafe MR.Const_Vector3f Dir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_Get_dir", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_SkyPatch_Get_dir(_Underlying *_this);
                return new(__MR_SkyPatch_Get_dir(_UnderlyingPtr), is_owning: false);
            }
        }

        /// radiation of the patch depending on Sun's position, sky clearness and brightness, etc
        public unsafe float Radiation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_Get_radiation", ExactSpelling = true)]
                extern static float *__MR_SkyPatch_Get_radiation(_Underlying *_this);
                return *__MR_SkyPatch_Get_radiation(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SkyPatch() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_DefaultConstruct();
            _UnderlyingPtr = __MR_SkyPatch_DefaultConstruct();
        }

        /// Constructs `MR::SkyPatch` elementwise.
        public unsafe Const_SkyPatch(MR.Vector3f dir, float radiation) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_ConstructFrom", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_ConstructFrom(MR.Vector3f dir, float radiation);
            _UnderlyingPtr = __MR_SkyPatch_ConstructFrom(dir, radiation);
        }

        /// Generated from constructor `MR::SkyPatch::SkyPatch`.
        public unsafe Const_SkyPatch(MR.Const_SkyPatch _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_ConstructFromAnother(MR.SkyPatch._Underlying *_other);
            _UnderlyingPtr = __MR_SkyPatch_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// this class represents a portion of the sky, and its radiation
    /// Generated from class `MR::SkyPatch`.
    /// This is the non-const half of the class.
    public class SkyPatch : Const_SkyPatch
    {
        internal unsafe SkyPatch(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// direction toward the center of the patch
        public new unsafe MR.Mut_Vector3f Dir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_GetMutable_dir", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_SkyPatch_GetMutable_dir(_Underlying *_this);
                return new(__MR_SkyPatch_GetMutable_dir(_UnderlyingPtr), is_owning: false);
            }
        }

        /// radiation of the patch depending on Sun's position, sky clearness and brightness, etc
        public new unsafe ref float Radiation
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_GetMutable_radiation", ExactSpelling = true)]
                extern static float *__MR_SkyPatch_GetMutable_radiation(_Underlying *_this);
                return ref *__MR_SkyPatch_GetMutable_radiation(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SkyPatch() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_DefaultConstruct();
            _UnderlyingPtr = __MR_SkyPatch_DefaultConstruct();
        }

        /// Constructs `MR::SkyPatch` elementwise.
        public unsafe SkyPatch(MR.Vector3f dir, float radiation) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_ConstructFrom", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_ConstructFrom(MR.Vector3f dir, float radiation);
            _UnderlyingPtr = __MR_SkyPatch_ConstructFrom(dir, radiation);
        }

        /// Generated from constructor `MR::SkyPatch::SkyPatch`.
        public unsafe SkyPatch(MR.Const_SkyPatch _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_ConstructFromAnother(MR.SkyPatch._Underlying *_other);
            _UnderlyingPtr = __MR_SkyPatch_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SkyPatch::operator=`.
        public unsafe MR.SkyPatch Assign(MR.Const_SkyPatch _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SkyPatch_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SkyPatch._Underlying *__MR_SkyPatch_AssignFromAnother(_Underlying *_this, MR.SkyPatch._Underlying *_other);
            return new(__MR_SkyPatch_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SkyPatch` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SkyPatch`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SkyPatch`/`Const_SkyPatch` directly.
    public class _InOptMut_SkyPatch
    {
        public SkyPatch? Opt;

        public _InOptMut_SkyPatch() {}
        public _InOptMut_SkyPatch(SkyPatch value) {Opt = value;}
        public static implicit operator _InOptMut_SkyPatch(SkyPatch value) {return new(value);}
    }

    /// This is used for optional parameters of class `SkyPatch` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SkyPatch`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SkyPatch`/`Const_SkyPatch` to pass it to the function.
    public class _InOptConst_SkyPatch
    {
        public Const_SkyPatch? Opt;

        public _InOptConst_SkyPatch() {}
        public _InOptConst_SkyPatch(Const_SkyPatch value) {Opt = value;}
        public static implicit operator _InOptConst_SkyPatch(Const_SkyPatch value) {return new(value);}
    }

    /// returns quasi-uniform 145 samples on unit half-sphere z>0
    /// Generated from function `MR::sampleHalfSphere`.
    public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVector3f> SampleHalfSphere()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_sampleHalfSphere", ExactSpelling = true)]
        extern static MR.Std.Vector_MRVector3f._Underlying *__MR_sampleHalfSphere();
        return MR.Misc.Move(new MR.Std.Vector_MRVector3f(__MR_sampleHalfSphere(), is_owning: true));
    }

    /// computes relative radiation in each valid sample point by emitting rays from that point in the sky:
    /// the radiation is 1.0f if all rays reach the sky not hitting the terrain;
    /// the radiation is 0.0f if all rays do not reach the sky because they are intercepted by the terrain;
    /// \param outSkyRays - optional output bitset where for every valid sample #i its rays are stored at indices [i*numPatches; (i+1)*numPatches),
    ///                     0s for occluded rays (hitting the terrain) and 1s for the ones which don't hit anything and reach the sky
    /// \param outIntersections - optional output vector of MeshIntersectionResult for every valid sample point
    /// Generated from function `MR::computeSkyViewFactor`.
    public static unsafe MR.Misc._Moved<MR.VertScalars> ComputeSkyViewFactor(MR.Const_Mesh terrain, MR.Const_VertCoords samples, MR.Const_VertBitSet validSamples, MR.Std.Const_Vector_MRSkyPatch skyPatches, MR.BitSet? outSkyRays = null, MR.Std.Vector_MRMeshIntersectionResult? outIntersections = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeSkyViewFactor", ExactSpelling = true)]
        extern static MR.VertScalars._Underlying *__MR_computeSkyViewFactor(MR.Const_Mesh._Underlying *terrain, MR.Const_VertCoords._Underlying *samples, MR.Const_VertBitSet._Underlying *validSamples, MR.Std.Const_Vector_MRSkyPatch._Underlying *skyPatches, MR.BitSet._Underlying *outSkyRays, MR.Std.Vector_MRMeshIntersectionResult._Underlying *outIntersections);
        return MR.Misc.Move(new MR.VertScalars(__MR_computeSkyViewFactor(terrain._UnderlyingPtr, samples._UnderlyingPtr, validSamples._UnderlyingPtr, skyPatches._UnderlyingPtr, outSkyRays is not null ? outSkyRays._UnderlyingPtr : null, outIntersections is not null ? outIntersections._UnderlyingPtr : null), is_owning: true));
    }

    /// In each valid sample point tests the rays from that point in the sky;
    /// \return bitset where for every valid sample #i its rays are stored at indices [i*numPatches; (i+1)*numPatches),
    ///         0s for occluded rays (hitting the terrain) and 1s for the ones which don't hit anything and reach the sky
    /// \param outIntersections - optional output vector of MeshIntersectionResult for every valid sample point
    /// Generated from function `MR::findSkyRays`.
    public static unsafe MR.Misc._Moved<MR.BitSet> FindSkyRays(MR.Const_Mesh terrain, MR.Const_VertCoords samples, MR.Const_VertBitSet validSamples, MR.Std.Const_Vector_MRSkyPatch skyPatches, MR.Std.Vector_MRMeshIntersectionResult? outIntersections = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findSkyRays", ExactSpelling = true)]
        extern static MR.BitSet._Underlying *__MR_findSkyRays(MR.Const_Mesh._Underlying *terrain, MR.Const_VertCoords._Underlying *samples, MR.Const_VertBitSet._Underlying *validSamples, MR.Std.Const_Vector_MRSkyPatch._Underlying *skyPatches, MR.Std.Vector_MRMeshIntersectionResult._Underlying *outIntersections);
        return MR.Misc.Move(new MR.BitSet(__MR_findSkyRays(terrain._UnderlyingPtr, samples._UnderlyingPtr, validSamples._UnderlyingPtr, skyPatches._UnderlyingPtr, outIntersections is not null ? outIntersections._UnderlyingPtr : null), is_owning: true));
    }
}
