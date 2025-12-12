public static partial class MR
{
    /// Laplacian to smoothly deform a region preserving mesh fine details.
    /// How to use:
    /// 1. Initialize Laplacian for the region being deformed, here region properties are remembered.
    /// 2. Change positions of some vertices within the region and call fixVertex for them.
    /// 3. Optionally call updateSolver()
    /// 4. Call apply() to change the remaining vertices within the region
    /// Then steps 1-4 or 2-4 can be repeated.
    /// \snippet cpp-samples/LaplacianDeformation.cpp 0
    /// Generated from class `MR::Laplacian`.
    /// This is the const half of the class.
    public class Const_Laplacian : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Laplacian(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_Destroy", ExactSpelling = true)]
            extern static void __MR_Laplacian_Destroy(_Underlying *_this);
            __MR_Laplacian_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Laplacian() {Dispose(false);}

        /// Generated from constructor `MR::Laplacian::Laplacian`.
        public unsafe Const_Laplacian(MR._ByValue_Laplacian _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Laplacian._Underlying *__MR_Laplacian_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Laplacian._Underlying *_other);
            _UnderlyingPtr = __MR_Laplacian_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Laplacian::Laplacian`.
        public unsafe Const_Laplacian(MR.Mesh mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_Construct_1", ExactSpelling = true)]
            extern static MR.Laplacian._Underlying *__MR_Laplacian_Construct_1(MR.Mesh._Underlying *mesh);
            _UnderlyingPtr = __MR_Laplacian_Construct_1(mesh._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Laplacian::Laplacian`.
        public unsafe Const_Laplacian(MR.Const_MeshTopology topology, MR.VertCoords points) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_Construct_2", ExactSpelling = true)]
            extern static MR.Laplacian._Underlying *__MR_Laplacian_Construct_2(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points);
            _UnderlyingPtr = __MR_Laplacian_Construct_2(topology._UnderlyingPtr, points._UnderlyingPtr);
        }

        /// return all initially free vertices and the first layer of vertices around them
        /// Generated from method `MR::Laplacian::region`.
        public unsafe MR.Const_VertBitSet Region()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_region", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_Laplacian_region(_Underlying *_this);
            return new(__MR_Laplacian_region(_UnderlyingPtr), is_owning: false);
        }

        /// return currently free vertices
        /// Generated from method `MR::Laplacian::freeVerts`.
        public unsafe MR.Const_VertBitSet FreeVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_freeVerts", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_Laplacian_freeVerts(_Underlying *_this);
            return new(__MR_Laplacian_freeVerts(_UnderlyingPtr), is_owning: false);
        }

        /// return fixed vertices from the first layer around free vertices
        /// Generated from method `MR::Laplacian::firstLayerFixedVerts`.
        public unsafe MR.Const_VertBitSet FirstLayerFixedVerts()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_firstLayerFixedVerts", ExactSpelling = true)]
            extern static MR.Const_VertBitSet._Underlying *__MR_Laplacian_firstLayerFixedVerts(_Underlying *_this);
            return new(__MR_Laplacian_firstLayerFixedVerts(_UnderlyingPtr), is_owning: false);
        }

        public enum RememberShape : int
        {
            // true Laplacian mode when initial mesh shape is remembered and copied in apply
            Yes = 0,
            // ignore initial mesh shape in the region and just position vertices smoothly in the region
            No = 1,
        }
    }

    /// Laplacian to smoothly deform a region preserving mesh fine details.
    /// How to use:
    /// 1. Initialize Laplacian for the region being deformed, here region properties are remembered.
    /// 2. Change positions of some vertices within the region and call fixVertex for them.
    /// 3. Optionally call updateSolver()
    /// 4. Call apply() to change the remaining vertices within the region
    /// Then steps 1-4 or 2-4 can be repeated.
    /// \snippet cpp-samples/LaplacianDeformation.cpp 0
    /// Generated from class `MR::Laplacian`.
    /// This is the non-const half of the class.
    public class Laplacian : Const_Laplacian
    {
        internal unsafe Laplacian(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::Laplacian::Laplacian`.
        public unsafe Laplacian(MR._ByValue_Laplacian _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Laplacian._Underlying *__MR_Laplacian_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Laplacian._Underlying *_other);
            _UnderlyingPtr = __MR_Laplacian_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::Laplacian::Laplacian`.
        public unsafe Laplacian(MR.Mesh mesh) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_Construct_1", ExactSpelling = true)]
            extern static MR.Laplacian._Underlying *__MR_Laplacian_Construct_1(MR.Mesh._Underlying *mesh);
            _UnderlyingPtr = __MR_Laplacian_Construct_1(mesh._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Laplacian::Laplacian`.
        public unsafe Laplacian(MR.Const_MeshTopology topology, MR.VertCoords points) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_Construct_2", ExactSpelling = true)]
            extern static MR.Laplacian._Underlying *__MR_Laplacian_Construct_2(MR.Const_MeshTopology._Underlying *topology, MR.VertCoords._Underlying *points);
            _UnderlyingPtr = __MR_Laplacian_Construct_2(topology._UnderlyingPtr, points._UnderlyingPtr);
        }

        /// initialize Laplacian for the region being deformed, here region properties are remembered and precomputed;
        /// \param freeVerts must not include all vertices of a mesh connected component
        /// Generated from method `MR::Laplacian::init`.
        /// Parameter `vmass` defaults to `VertexMass::Unit`.
        /// Parameter `rem` defaults to `Laplacian::RememberShape::Yes`.
        public unsafe void Init(MR.Const_VertBitSet freeVerts, MR.EdgeWeights weights, MR.VertexMass? vmass = null, MR.Laplacian.RememberShape? rem = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_init", ExactSpelling = true)]
            extern static void __MR_Laplacian_init(_Underlying *_this, MR.Const_VertBitSet._Underlying *freeVerts, MR.EdgeWeights weights, MR.VertexMass *vmass, MR.Laplacian.RememberShape *rem);
            MR.VertexMass __deref_vmass = vmass.GetValueOrDefault();
            MR.Laplacian.RememberShape __deref_rem = rem.GetValueOrDefault();
            __MR_Laplacian_init(_UnderlyingPtr, freeVerts._UnderlyingPtr, weights, vmass.HasValue ? &__deref_vmass : null, rem.HasValue ? &__deref_rem : null);
        }

        /// notify Laplacian that given vertex has changed after init and must be fixed during apply;
        /// \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
        /// Generated from method `MR::Laplacian::fixVertex`.
        /// Parameter `smooth` defaults to `true`.
        public unsafe void FixVertex(MR.VertId v, bool? smooth = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_fixVertex_2", ExactSpelling = true)]
            extern static void __MR_Laplacian_fixVertex_2(_Underlying *_this, MR.VertId v, byte *smooth);
            byte __deref_smooth = smooth.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Laplacian_fixVertex_2(_UnderlyingPtr, v, smooth.HasValue ? &__deref_smooth : null);
        }

        /// sets position of given vertex after init and it must be fixed during apply (THIS METHOD CHANGES THE MESH);
        /// \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
        /// Generated from method `MR::Laplacian::fixVertex`.
        /// Parameter `smooth` defaults to `true`.
        public unsafe void FixVertex(MR.VertId v, MR.Const_Vector3f fixedPos, bool? smooth = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_fixVertex_3", ExactSpelling = true)]
            extern static void __MR_Laplacian_fixVertex_3(_Underlying *_this, MR.VertId v, MR.Const_Vector3f._Underlying *fixedPos, byte *smooth);
            byte __deref_smooth = smooth.GetValueOrDefault() ? (byte)1 : (byte)0;
            __MR_Laplacian_fixVertex_3(_UnderlyingPtr, v, fixedPos._UnderlyingPtr, smooth.HasValue ? &__deref_smooth : null);
        }

        /// if you manually call this method after initialization and fixing vertices then next apply call will be much faster
        /// Generated from method `MR::Laplacian::updateSolver`.
        public unsafe void UpdateSolver()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_updateSolver", ExactSpelling = true)]
            extern static void __MR_Laplacian_updateSolver(_Underlying *_this);
            __MR_Laplacian_updateSolver(_UnderlyingPtr);
        }

        /// given fixed vertices, computes positions of remaining region vertices
        /// Generated from method `MR::Laplacian::apply`.
        public unsafe void Apply()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_apply", ExactSpelling = true)]
            extern static void __MR_Laplacian_apply(_Underlying *_this);
            __MR_Laplacian_apply(_UnderlyingPtr);
        }

        /// given a pre-resized scalar field with set values in fixed vertices, computes the values in free vertices
        /// Generated from method `MR::Laplacian::applyToScalar`.
        public unsafe void ApplyToScalar(MR.VertScalars scalarField)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Laplacian_applyToScalar", ExactSpelling = true)]
            extern static void __MR_Laplacian_applyToScalar(_Underlying *_this, MR.VertScalars._Underlying *scalarField);
            __MR_Laplacian_applyToScalar(_UnderlyingPtr, scalarField._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Laplacian` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Laplacian`/`Const_Laplacian` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Laplacian
    {
        internal readonly Const_Laplacian? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Laplacian(MR.Misc._Moved<Laplacian> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Laplacian(MR.Misc._Moved<Laplacian> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Laplacian` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Laplacian`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Laplacian`/`Const_Laplacian` directly.
    public class _InOptMut_Laplacian
    {
        public Laplacian? Opt;

        public _InOptMut_Laplacian() {}
        public _InOptMut_Laplacian(Laplacian value) {Opt = value;}
        public static implicit operator _InOptMut_Laplacian(Laplacian value) {return new(value);}
    }

    /// This is used for optional parameters of class `Laplacian` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Laplacian`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Laplacian`/`Const_Laplacian` to pass it to the function.
    public class _InOptConst_Laplacian
    {
        public Const_Laplacian? Opt;

        public _InOptConst_Laplacian() {}
        public _InOptConst_Laplacian(Const_Laplacian value) {Opt = value;}
        public static implicit operator _InOptConst_Laplacian(Const_Laplacian value) {return new(value);}
    }
}
