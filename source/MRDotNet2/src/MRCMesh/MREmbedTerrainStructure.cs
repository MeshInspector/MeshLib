public static partial class MR
{
    // Parameters of structure embedding in terrain
    /// Generated from class `MR::EmbeddedStructureParameters`.
    /// This is the const half of the class.
    public class Const_EmbeddedStructureParameters : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EmbeddedStructureParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Destroy", ExactSpelling = true)]
            extern static void __MR_EmbeddedStructureParameters_Destroy(_Underlying *_this);
            __MR_EmbeddedStructureParameters_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EmbeddedStructureParameters() {Dispose(false);}

        // angle of fill cone (mound)
        public unsafe float FillAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_fillAngle", ExactSpelling = true)]
                extern static float *__MR_EmbeddedStructureParameters_Get_fillAngle(_Underlying *_this);
                return *__MR_EmbeddedStructureParameters_Get_fillAngle(_UnderlyingPtr);
            }
        }

        // angle of cut cone (pit)
        public unsafe float CutAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_cutAngle", ExactSpelling = true)]
                extern static float *__MR_EmbeddedStructureParameters_Get_cutAngle(_Underlying *_this);
                return *__MR_EmbeddedStructureParameters_Get_cutAngle(_UnderlyingPtr);
            }
        }

        // 20 deg
        public unsafe float MinAnglePrecision
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_minAnglePrecision", ExactSpelling = true)]
                extern static float *__MR_EmbeddedStructureParameters_Get_minAnglePrecision(_Underlying *_this);
                return *__MR_EmbeddedStructureParameters_Get_minAnglePrecision(_UnderlyingPtr);
            }
        }

        // optional out new faces of embedded structure 
        public unsafe ref void * OutStructFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_outStructFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_Get_outStructFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_Get_outStructFaces(_UnderlyingPtr);
            }
        }

        // optional out new faces of fill part
        public unsafe ref void * OutFillFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_outFillFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_Get_outFillFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_Get_outFillFaces(_UnderlyingPtr);
            }
        }

        // optional out new faces of cut part
        public unsafe ref void * OutCutFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_outCutFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_Get_outCutFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_Get_outCutFaces(_UnderlyingPtr);
            }
        }

        // optional out map new terrain faces to old terrain faces
        public unsafe ref void * New2oldFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_Get_new2oldFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_Get_new2oldFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_Get_new2oldFaces(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EmbeddedStructureParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_EmbeddedStructureParameters_DefaultConstruct();
        }

        /// Constructs `MR::EmbeddedStructureParameters` elementwise.
        public unsafe Const_EmbeddedStructureParameters(float fillAngle, float cutAngle, float minAnglePrecision, MR.FaceBitSet? outStructFaces, MR.FaceBitSet? outFillFaces, MR.FaceBitSet? outCutFaces, MR.FaceMap? new2oldFaces) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_ConstructFrom(float fillAngle, float cutAngle, float minAnglePrecision, MR.FaceBitSet._Underlying *outStructFaces, MR.FaceBitSet._Underlying *outFillFaces, MR.FaceBitSet._Underlying *outCutFaces, MR.FaceMap._Underlying *new2oldFaces);
            _UnderlyingPtr = __MR_EmbeddedStructureParameters_ConstructFrom(fillAngle, cutAngle, minAnglePrecision, outStructFaces is not null ? outStructFaces._UnderlyingPtr : null, outFillFaces is not null ? outFillFaces._UnderlyingPtr : null, outCutFaces is not null ? outCutFaces._UnderlyingPtr : null, new2oldFaces is not null ? new2oldFaces._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EmbeddedStructureParameters::EmbeddedStructureParameters`.
        public unsafe Const_EmbeddedStructureParameters(MR.Const_EmbeddedStructureParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_ConstructFromAnother(MR.EmbeddedStructureParameters._Underlying *_other);
            _UnderlyingPtr = __MR_EmbeddedStructureParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // Parameters of structure embedding in terrain
    /// Generated from class `MR::EmbeddedStructureParameters`.
    /// This is the non-const half of the class.
    public class EmbeddedStructureParameters : Const_EmbeddedStructureParameters
    {
        internal unsafe EmbeddedStructureParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // angle of fill cone (mound)
        public new unsafe ref float FillAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_fillAngle", ExactSpelling = true)]
                extern static float *__MR_EmbeddedStructureParameters_GetMutable_fillAngle(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_fillAngle(_UnderlyingPtr);
            }
        }

        // angle of cut cone (pit)
        public new unsafe ref float CutAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_cutAngle", ExactSpelling = true)]
                extern static float *__MR_EmbeddedStructureParameters_GetMutable_cutAngle(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_cutAngle(_UnderlyingPtr);
            }
        }

        // 20 deg
        public new unsafe ref float MinAnglePrecision
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_minAnglePrecision", ExactSpelling = true)]
                extern static float *__MR_EmbeddedStructureParameters_GetMutable_minAnglePrecision(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_minAnglePrecision(_UnderlyingPtr);
            }
        }

        // optional out new faces of embedded structure 
        public new unsafe ref void * OutStructFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_outStructFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_GetMutable_outStructFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_outStructFaces(_UnderlyingPtr);
            }
        }

        // optional out new faces of fill part
        public new unsafe ref void * OutFillFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_outFillFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_GetMutable_outFillFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_outFillFaces(_UnderlyingPtr);
            }
        }

        // optional out new faces of cut part
        public new unsafe ref void * OutCutFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_outCutFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_GetMutable_outCutFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_outCutFaces(_UnderlyingPtr);
            }
        }

        // optional out map new terrain faces to old terrain faces
        public new unsafe ref void * New2oldFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_GetMutable_new2oldFaces", ExactSpelling = true)]
                extern static void **__MR_EmbeddedStructureParameters_GetMutable_new2oldFaces(_Underlying *_this);
                return ref *__MR_EmbeddedStructureParameters_GetMutable_new2oldFaces(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EmbeddedStructureParameters() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_DefaultConstruct();
            _UnderlyingPtr = __MR_EmbeddedStructureParameters_DefaultConstruct();
        }

        /// Constructs `MR::EmbeddedStructureParameters` elementwise.
        public unsafe EmbeddedStructureParameters(float fillAngle, float cutAngle, float minAnglePrecision, MR.FaceBitSet? outStructFaces, MR.FaceBitSet? outFillFaces, MR.FaceBitSet? outCutFaces, MR.FaceMap? new2oldFaces) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_ConstructFrom", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_ConstructFrom(float fillAngle, float cutAngle, float minAnglePrecision, MR.FaceBitSet._Underlying *outStructFaces, MR.FaceBitSet._Underlying *outFillFaces, MR.FaceBitSet._Underlying *outCutFaces, MR.FaceMap._Underlying *new2oldFaces);
            _UnderlyingPtr = __MR_EmbeddedStructureParameters_ConstructFrom(fillAngle, cutAngle, minAnglePrecision, outStructFaces is not null ? outStructFaces._UnderlyingPtr : null, outFillFaces is not null ? outFillFaces._UnderlyingPtr : null, outCutFaces is not null ? outCutFaces._UnderlyingPtr : null, new2oldFaces is not null ? new2oldFaces._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::EmbeddedStructureParameters::EmbeddedStructureParameters`.
        public unsafe EmbeddedStructureParameters(MR.Const_EmbeddedStructureParameters _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_ConstructFromAnother(MR.EmbeddedStructureParameters._Underlying *_other);
            _UnderlyingPtr = __MR_EmbeddedStructureParameters_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::EmbeddedStructureParameters::operator=`.
        public unsafe MR.EmbeddedStructureParameters Assign(MR.Const_EmbeddedStructureParameters _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EmbeddedStructureParameters_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EmbeddedStructureParameters._Underlying *__MR_EmbeddedStructureParameters_AssignFromAnother(_Underlying *_this, MR.EmbeddedStructureParameters._Underlying *_other);
            return new(__MR_EmbeddedStructureParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `EmbeddedStructureParameters` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EmbeddedStructureParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EmbeddedStructureParameters`/`Const_EmbeddedStructureParameters` directly.
    public class _InOptMut_EmbeddedStructureParameters
    {
        public EmbeddedStructureParameters? Opt;

        public _InOptMut_EmbeddedStructureParameters() {}
        public _InOptMut_EmbeddedStructureParameters(EmbeddedStructureParameters value) {Opt = value;}
        public static implicit operator _InOptMut_EmbeddedStructureParameters(EmbeddedStructureParameters value) {return new(value);}
    }

    /// This is used for optional parameters of class `EmbeddedStructureParameters` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EmbeddedStructureParameters`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EmbeddedStructureParameters`/`Const_EmbeddedStructureParameters` to pass it to the function.
    public class _InOptConst_EmbeddedStructureParameters
    {
        public Const_EmbeddedStructureParameters? Opt;

        public _InOptConst_EmbeddedStructureParameters() {}
        public _InOptConst_EmbeddedStructureParameters(Const_EmbeddedStructureParameters value) {Opt = value;}
        public static implicit operator _InOptConst_EmbeddedStructureParameters(Const_EmbeddedStructureParameters value) {return new(value);}
    }

    // Returns terrain mesh with structure embedded to it, or error string
    // terrain - mesh with +Z normal (not-closed mesh is expected)
    // structure - mesh with one open contour and +Z normal, that will be embedded in terrain
    /// Generated from function `MR::embedStructureToTerrain`.
    public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> EmbedStructureToTerrain(MR.Const_Mesh terrain, MR.Const_Mesh structure, MR.Const_EmbeddedStructureParameters params_)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_embedStructureToTerrain", ExactSpelling = true)]
        extern static MR.Expected_MRMesh_StdString._Underlying *__MR_embedStructureToTerrain(MR.Const_Mesh._Underlying *terrain, MR.Const_Mesh._Underlying *structure, MR.Const_EmbeddedStructureParameters._Underlying *params_);
        return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_embedStructureToTerrain(terrain._UnderlyingPtr, structure._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
    }
}
