public static partial class MR
{
    /// Mode of processing components
    public enum NestedComponenetsMode : int
    {
        ///< Default: separate nested meshes and remove them, just like union operation should do, use this if input meshes are single component
        Remove = 0,
        ///< merge nested meshes, useful if input meshes are components of single object
        Merge = 1,
        ///< does not separate components and call union for all input meshes, works slower than Remove and Merge method but returns valid result if input meshes has multiple components
        Union = 2,
    }

    /// Parameters structure for uniteManyMeshes function
    /// Generated from class `MR::UniteManyMeshesParams`.
    /// This is the const half of the class.
    public class Const_UniteManyMeshesParams : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UniteManyMeshesParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Destroy", ExactSpelling = true)]
            extern static void __MR_UniteManyMeshesParams_Destroy(_Underlying *_this);
            __MR_UniteManyMeshesParams_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UniteManyMeshesParams() {Dispose(false);}

        /// Apply random shift to each mesh, to prevent degenerations on coincident surfaces
        public unsafe bool UseRandomShifts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_useRandomShifts", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_Get_useRandomShifts(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_useRandomShifts(_UnderlyingPtr);
            }
        }

        /// Try fix degenerations after each boolean step, to prevent boolean failure due to high amount of degenerated faces
        /// useful on meshes with many coincident surfaces 
        /// (useRandomShifts used for same issue)
        public unsafe bool FixDegenerations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_fixDegenerations", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_Get_fixDegenerations(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_fixDegenerations(_UnderlyingPtr);
            }
        }

        /// Max allowed random shifts in each direction, and max allowed deviation after degeneration fixing
        /// not used if both flags (useRandomShifts,fixDegenerations) are false
        public unsafe float MaxAllowedError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_maxAllowedError", ExactSpelling = true)]
                extern static float *__MR_UniteManyMeshesParams_Get_maxAllowedError(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_maxAllowedError(_UnderlyingPtr);
            }
        }

        /// Seed that is used for random shifts
        public unsafe uint RandomShiftsSeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_randomShiftsSeed", ExactSpelling = true)]
                extern static uint *__MR_UniteManyMeshesParams_Get_randomShiftsSeed(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_randomShiftsSeed(_UnderlyingPtr);
            }
        }

        /// If set, the bitset will store new faces created by boolean operations
        public unsafe ref void * NewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_newFaces", ExactSpelling = true)]
                extern static void **__MR_UniteManyMeshesParams_Get_newFaces(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_Get_newFaces(_UnderlyingPtr);
            }
        }

        /// By default function separate nested meshes and remove them, just like union operation should do
        /// read comment of NestedComponenetsMode enum for more information
        public unsafe MR.NestedComponenetsMode NestedComponentsMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_nestedComponentsMode", ExactSpelling = true)]
                extern static MR.NestedComponenetsMode *__MR_UniteManyMeshesParams_Get_nestedComponentsMode(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_nestedComponentsMode(_UnderlyingPtr);
            }
        }

        /// If set - merges meshes instead of booleaning it if boolean operation fails
        public unsafe bool MergeOnFail
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_mergeOnFail", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_Get_mergeOnFail(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_mergeOnFail(_UnderlyingPtr);
            }
        }

        /// If this option is enabled boolean will try to cut meshes even if there are self-intersections in intersecting area
        /// it might work in some cases, but in general it might prevent fast error report and lead to other errors along the way
        /// \warning not recommended in most cases
        public unsafe bool ForceCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_forceCut", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_Get_forceCut(_Underlying *_this);
                return *__MR_UniteManyMeshesParams_Get_forceCut(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Function_BoolFuncFromFloat ProgressCb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_Get_progressCb", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_UniteManyMeshesParams_Get_progressCb(_Underlying *_this);
                return new(__MR_UniteManyMeshesParams_Get_progressCb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UniteManyMeshesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_UniteManyMeshesParams_DefaultConstruct();
        }

        /// Constructs `MR::UniteManyMeshesParams` elementwise.
        public unsafe Const_UniteManyMeshesParams(bool useRandomShifts, bool fixDegenerations, float maxAllowedError, uint randomShiftsSeed, MR.FaceBitSet? newFaces, MR.NestedComponenetsMode nestedComponentsMode, bool mergeOnFail, bool forceCut, MR.Std._ByValue_Function_BoolFuncFromFloat progressCb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_ConstructFrom(byte useRandomShifts, byte fixDegenerations, float maxAllowedError, uint randomShiftsSeed, MR.FaceBitSet._Underlying *newFaces, MR.NestedComponenetsMode nestedComponentsMode, byte mergeOnFail, byte forceCut, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            _UnderlyingPtr = __MR_UniteManyMeshesParams_ConstructFrom(useRandomShifts ? (byte)1 : (byte)0, fixDegenerations ? (byte)1 : (byte)0, maxAllowedError, randomShiftsSeed, newFaces is not null ? newFaces._UnderlyingPtr : null, nestedComponentsMode, mergeOnFail ? (byte)1 : (byte)0, forceCut ? (byte)1 : (byte)0, progressCb.PassByMode, progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::UniteManyMeshesParams::UniteManyMeshesParams`.
        public unsafe Const_UniteManyMeshesParams(MR._ByValue_UniteManyMeshesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniteManyMeshesParams._Underlying *_other);
            _UnderlyingPtr = __MR_UniteManyMeshesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Parameters structure for uniteManyMeshes function
    /// Generated from class `MR::UniteManyMeshesParams`.
    /// This is the non-const half of the class.
    public class UniteManyMeshesParams : Const_UniteManyMeshesParams
    {
        internal unsafe UniteManyMeshesParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Apply random shift to each mesh, to prevent degenerations on coincident surfaces
        public new unsafe ref bool UseRandomShifts
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_useRandomShifts", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_GetMutable_useRandomShifts(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_useRandomShifts(_UnderlyingPtr);
            }
        }

        /// Try fix degenerations after each boolean step, to prevent boolean failure due to high amount of degenerated faces
        /// useful on meshes with many coincident surfaces 
        /// (useRandomShifts used for same issue)
        public new unsafe ref bool FixDegenerations
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_fixDegenerations", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_GetMutable_fixDegenerations(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_fixDegenerations(_UnderlyingPtr);
            }
        }

        /// Max allowed random shifts in each direction, and max allowed deviation after degeneration fixing
        /// not used if both flags (useRandomShifts,fixDegenerations) are false
        public new unsafe ref float MaxAllowedError
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_maxAllowedError", ExactSpelling = true)]
                extern static float *__MR_UniteManyMeshesParams_GetMutable_maxAllowedError(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_maxAllowedError(_UnderlyingPtr);
            }
        }

        /// Seed that is used for random shifts
        public new unsafe ref uint RandomShiftsSeed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_randomShiftsSeed", ExactSpelling = true)]
                extern static uint *__MR_UniteManyMeshesParams_GetMutable_randomShiftsSeed(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_randomShiftsSeed(_UnderlyingPtr);
            }
        }

        /// If set, the bitset will store new faces created by boolean operations
        public new unsafe ref void * NewFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_newFaces", ExactSpelling = true)]
                extern static void **__MR_UniteManyMeshesParams_GetMutable_newFaces(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_newFaces(_UnderlyingPtr);
            }
        }

        /// By default function separate nested meshes and remove them, just like union operation should do
        /// read comment of NestedComponenetsMode enum for more information
        public new unsafe ref MR.NestedComponenetsMode NestedComponentsMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_nestedComponentsMode", ExactSpelling = true)]
                extern static MR.NestedComponenetsMode *__MR_UniteManyMeshesParams_GetMutable_nestedComponentsMode(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_nestedComponentsMode(_UnderlyingPtr);
            }
        }

        /// If set - merges meshes instead of booleaning it if boolean operation fails
        public new unsafe ref bool MergeOnFail
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_mergeOnFail", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_GetMutable_mergeOnFail(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_mergeOnFail(_UnderlyingPtr);
            }
        }

        /// If this option is enabled boolean will try to cut meshes even if there are self-intersections in intersecting area
        /// it might work in some cases, but in general it might prevent fast error report and lead to other errors along the way
        /// \warning not recommended in most cases
        public new unsafe ref bool ForceCut
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_forceCut", ExactSpelling = true)]
                extern static bool *__MR_UniteManyMeshesParams_GetMutable_forceCut(_Underlying *_this);
                return ref *__MR_UniteManyMeshesParams_GetMutable_forceCut(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Function_BoolFuncFromFloat ProgressCb
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_GetMutable_progressCb", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_UniteManyMeshesParams_GetMutable_progressCb(_Underlying *_this);
                return new(__MR_UniteManyMeshesParams_GetMutable_progressCb(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe UniteManyMeshesParams() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_DefaultConstruct();
            _UnderlyingPtr = __MR_UniteManyMeshesParams_DefaultConstruct();
        }

        /// Constructs `MR::UniteManyMeshesParams` elementwise.
        public unsafe UniteManyMeshesParams(bool useRandomShifts, bool fixDegenerations, float maxAllowedError, uint randomShiftsSeed, MR.FaceBitSet? newFaces, MR.NestedComponenetsMode nestedComponentsMode, bool mergeOnFail, bool forceCut, MR.Std._ByValue_Function_BoolFuncFromFloat progressCb) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_ConstructFrom", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_ConstructFrom(byte useRandomShifts, byte fixDegenerations, float maxAllowedError, uint randomShiftsSeed, MR.FaceBitSet._Underlying *newFaces, MR.NestedComponenetsMode nestedComponentsMode, byte mergeOnFail, byte forceCut, MR.Misc._PassBy progressCb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *progressCb);
            _UnderlyingPtr = __MR_UniteManyMeshesParams_ConstructFrom(useRandomShifts ? (byte)1 : (byte)0, fixDegenerations ? (byte)1 : (byte)0, maxAllowedError, randomShiftsSeed, newFaces is not null ? newFaces._UnderlyingPtr : null, nestedComponentsMode, mergeOnFail ? (byte)1 : (byte)0, forceCut ? (byte)1 : (byte)0, progressCb.PassByMode, progressCb.Value is not null ? progressCb.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::UniteManyMeshesParams::UniteManyMeshesParams`.
        public unsafe UniteManyMeshesParams(MR._ByValue_UniteManyMeshesParams _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UniteManyMeshesParams._Underlying *_other);
            _UnderlyingPtr = __MR_UniteManyMeshesParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::UniteManyMeshesParams::operator=`.
        public unsafe MR.UniteManyMeshesParams Assign(MR._ByValue_UniteManyMeshesParams _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UniteManyMeshesParams_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UniteManyMeshesParams._Underlying *__MR_UniteManyMeshesParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UniteManyMeshesParams._Underlying *_other);
            return new(__MR_UniteManyMeshesParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `UniteManyMeshesParams` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UniteManyMeshesParams`/`Const_UniteManyMeshesParams` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UniteManyMeshesParams
    {
        internal readonly Const_UniteManyMeshesParams? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UniteManyMeshesParams() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UniteManyMeshesParams(Const_UniteManyMeshesParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UniteManyMeshesParams(Const_UniteManyMeshesParams arg) {return new(arg);}
        public _ByValue_UniteManyMeshesParams(MR.Misc._Moved<UniteManyMeshesParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UniteManyMeshesParams(MR.Misc._Moved<UniteManyMeshesParams> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UniteManyMeshesParams` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UniteManyMeshesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniteManyMeshesParams`/`Const_UniteManyMeshesParams` directly.
    public class _InOptMut_UniteManyMeshesParams
    {
        public UniteManyMeshesParams? Opt;

        public _InOptMut_UniteManyMeshesParams() {}
        public _InOptMut_UniteManyMeshesParams(UniteManyMeshesParams value) {Opt = value;}
        public static implicit operator _InOptMut_UniteManyMeshesParams(UniteManyMeshesParams value) {return new(value);}
    }

    /// This is used for optional parameters of class `UniteManyMeshesParams` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UniteManyMeshesParams`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UniteManyMeshesParams`/`Const_UniteManyMeshesParams` to pass it to the function.
    public class _InOptConst_UniteManyMeshesParams
    {
        public Const_UniteManyMeshesParams? Opt;

        public _InOptConst_UniteManyMeshesParams() {}
        public _InOptConst_UniteManyMeshesParams(Const_UniteManyMeshesParams value) {Opt = value;}
        public static implicit operator _InOptConst_UniteManyMeshesParams(Const_UniteManyMeshesParams value) {return new(value);}
    }

    // Computes the surface of objects' union each of which is defined by its own surface mesh
    // - merge non intersecting meshes first
    // - unite merged groups
    /// Generated from function `MR::uniteManyMeshes`.
    /// Parameter `params_` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> UniteManyMeshes(MR.Std.Const_Vector_ConstMRMeshPtr meshes, MR.Const_UniteManyMeshesParams? params_ = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_uniteManyMeshes", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_uniteManyMeshes(MR.Std.Const_Vector_ConstMRMeshPtr._Underlying *meshes, MR.Const_UniteManyMeshesParams._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_uniteManyMeshes(meshes._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
    }
}
