public static partial class MR
{
    /// struct to determine transparent rendering mode
    /// Generated from class `MR::TransparencyMode`.
    /// This is the const half of the class.
    public class Const_TransparencyMode : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TransparencyMode(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_Destroy", ExactSpelling = true)]
            extern static void __MR_TransparencyMode_Destroy(_Underlying *_this);
            __MR_TransparencyMode_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TransparencyMode() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TransparencyMode() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_DefaultConstruct();
            _UnderlyingPtr = __MR_TransparencyMode_DefaultConstruct();
        }

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public unsafe Const_TransparencyMode(MR.Const_TransparencyMode _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_ConstructFromAnother(MR.TransparencyMode._Underlying *_other);
            _UnderlyingPtr = __MR_TransparencyMode_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public unsafe Const_TransparencyMode(bool alphaSort) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_Construct_1", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_Construct_1(byte alphaSort);
            _UnderlyingPtr = __MR_TransparencyMode_Construct_1(alphaSort ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public static unsafe implicit operator Const_TransparencyMode(bool alphaSort) {return new(alphaSort);}

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public unsafe Const_TransparencyMode(uint bgDepthTexId, uint fgColorTexId, uint fgDepthTexId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_Construct_3", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_Construct_3(uint bgDepthTexId, uint fgColorTexId, uint fgDepthTexId);
            _UnderlyingPtr = __MR_TransparencyMode_Construct_3(bgDepthTexId, fgColorTexId, fgDepthTexId);
        }

        /// Generated from method `MR::TransparencyMode::isAlphaSortEnabled`.
        public unsafe bool IsAlphaSortEnabled()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_isAlphaSortEnabled", ExactSpelling = true)]
            extern static byte __MR_TransparencyMode_isAlphaSortEnabled(_Underlying *_this);
            return __MR_TransparencyMode_isAlphaSortEnabled(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TransparencyMode::isDepthPeelingEnabled`.
        public unsafe bool IsDepthPeelingEnabled()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_isDepthPeelingEnabled", ExactSpelling = true)]
            extern static byte __MR_TransparencyMode_isDepthPeelingEnabled(_Underlying *_this);
            return __MR_TransparencyMode_isDepthPeelingEnabled(_UnderlyingPtr) != 0;
        }

        /// Generated from method `MR::TransparencyMode::getBGDepthPeelingDepthTextureId`.
        public unsafe uint GetBGDepthPeelingDepthTextureId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_getBGDepthPeelingDepthTextureId", ExactSpelling = true)]
            extern static uint __MR_TransparencyMode_getBGDepthPeelingDepthTextureId(_Underlying *_this);
            return __MR_TransparencyMode_getBGDepthPeelingDepthTextureId(_UnderlyingPtr);
        }

        /// Generated from method `MR::TransparencyMode::getFGDepthPeelingColorTextureId`.
        public unsafe uint GetFGDepthPeelingColorTextureId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_getFGDepthPeelingColorTextureId", ExactSpelling = true)]
            extern static uint __MR_TransparencyMode_getFGDepthPeelingColorTextureId(_Underlying *_this);
            return __MR_TransparencyMode_getFGDepthPeelingColorTextureId(_UnderlyingPtr);
        }

        /// Generated from method `MR::TransparencyMode::getFGDepthPeelingDepthTextureId`.
        public unsafe uint GetFGDepthPeelingDepthTextureId()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_getFGDepthPeelingDepthTextureId", ExactSpelling = true)]
            extern static uint __MR_TransparencyMode_getFGDepthPeelingDepthTextureId(_Underlying *_this);
            return __MR_TransparencyMode_getFGDepthPeelingDepthTextureId(_UnderlyingPtr);
        }
    }

    /// struct to determine transparent rendering mode
    /// Generated from class `MR::TransparencyMode`.
    /// This is the non-const half of the class.
    public class TransparencyMode : Const_TransparencyMode
    {
        internal unsafe TransparencyMode(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe TransparencyMode() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_DefaultConstruct();
            _UnderlyingPtr = __MR_TransparencyMode_DefaultConstruct();
        }

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public unsafe TransparencyMode(MR.Const_TransparencyMode _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_ConstructFromAnother(MR.TransparencyMode._Underlying *_other);
            _UnderlyingPtr = __MR_TransparencyMode_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public unsafe TransparencyMode(bool alphaSort) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_Construct_1", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_Construct_1(byte alphaSort);
            _UnderlyingPtr = __MR_TransparencyMode_Construct_1(alphaSort ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public static unsafe implicit operator TransparencyMode(bool alphaSort) {return new(alphaSort);}

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public unsafe TransparencyMode(uint bgDepthTexId, uint fgColorTexId, uint fgDepthTexId) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_Construct_3", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_Construct_3(uint bgDepthTexId, uint fgColorTexId, uint fgDepthTexId);
            _UnderlyingPtr = __MR_TransparencyMode_Construct_3(bgDepthTexId, fgColorTexId, fgDepthTexId);
        }

        /// Generated from method `MR::TransparencyMode::operator=`.
        public unsafe MR.TransparencyMode Assign(MR.Const_TransparencyMode _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransparencyMode_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TransparencyMode._Underlying *__MR_TransparencyMode_AssignFromAnother(_Underlying *_this, MR.TransparencyMode._Underlying *_other);
            return new(__MR_TransparencyMode_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `TransparencyMode` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TransparencyMode`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TransparencyMode`/`Const_TransparencyMode` directly.
    public class _InOptMut_TransparencyMode
    {
        public TransparencyMode? Opt;

        public _InOptMut_TransparencyMode() {}
        public _InOptMut_TransparencyMode(TransparencyMode value) {Opt = value;}
        public static implicit operator _InOptMut_TransparencyMode(TransparencyMode value) {return new(value);}
    }

    /// This is used for optional parameters of class `TransparencyMode` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TransparencyMode`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TransparencyMode`/`Const_TransparencyMode` to pass it to the function.
    public class _InOptConst_TransparencyMode
    {
        public Const_TransparencyMode? Opt;

        public _InOptConst_TransparencyMode() {}
        public _InOptConst_TransparencyMode(Const_TransparencyMode value) {Opt = value;}
        public static implicit operator _InOptConst_TransparencyMode(Const_TransparencyMode value) {return new(value);}

        /// Generated from constructor `MR::TransparencyMode::TransparencyMode`.
        public static unsafe implicit operator _InOptConst_TransparencyMode(bool alphaSort) {return new MR.TransparencyMode(alphaSort);}
    }

    /// Various passes of the 3D rendering.
    public enum RenderModelPassMask : int
    {
        Opaque = 1,
        Transparent = 2,
        VolumeRendering = 4,
        NoDepthTest = 8,
        All = 15,
    }

    /// Generated from function `MR::operator&`.
    public static MR.RenderModelPassMask Bitand(MR.RenderModelPassMask a, MR.RenderModelPassMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_MR_RenderModelPassMask", ExactSpelling = true)]
        extern static MR.RenderModelPassMask __MR_bitand_MR_RenderModelPassMask(MR.RenderModelPassMask a, MR.RenderModelPassMask b);
        return __MR_bitand_MR_RenderModelPassMask(a, b);
    }

    /// Generated from function `MR::operator|`.
    public static MR.RenderModelPassMask Bitor(MR.RenderModelPassMask a, MR.RenderModelPassMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_MR_RenderModelPassMask", ExactSpelling = true)]
        extern static MR.RenderModelPassMask __MR_bitor_MR_RenderModelPassMask(MR.RenderModelPassMask a, MR.RenderModelPassMask b);
        return __MR_bitor_MR_RenderModelPassMask(a, b);
    }

    /// Generated from function `MR::operator~`.
    public static MR.RenderModelPassMask Compl(MR.RenderModelPassMask a)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_compl_MR_RenderModelPassMask", ExactSpelling = true)]
        extern static MR.RenderModelPassMask __MR_compl_MR_RenderModelPassMask(MR.RenderModelPassMask a);
        return __MR_compl_MR_RenderModelPassMask(a);
    }

    /// Generated from function `MR::operator&=`.
    public static unsafe ref MR.RenderModelPassMask BitandAssign(ref MR.RenderModelPassMask a, MR.RenderModelPassMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitand_assign_MR_RenderModelPassMask", ExactSpelling = true)]
        extern static MR.RenderModelPassMask *__MR_bitand_assign_MR_RenderModelPassMask(MR.RenderModelPassMask *a, MR.RenderModelPassMask b);
        fixed (MR.RenderModelPassMask *__ptr_a = &a)
        {
            return ref *__MR_bitand_assign_MR_RenderModelPassMask(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator|=`.
    public static unsafe ref MR.RenderModelPassMask BitorAssign(ref MR.RenderModelPassMask a, MR.RenderModelPassMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_bitor_assign_MR_RenderModelPassMask", ExactSpelling = true)]
        extern static MR.RenderModelPassMask *__MR_bitor_assign_MR_RenderModelPassMask(MR.RenderModelPassMask *a, MR.RenderModelPassMask b);
        fixed (MR.RenderModelPassMask *__ptr_a = &a)
        {
            return ref *__MR_bitor_assign_MR_RenderModelPassMask(__ptr_a, b);
        }
    }

    /// Generated from function `MR::operator*`.
    public static MR.RenderModelPassMask Mul(MR.RenderModelPassMask a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_MR_RenderModelPassMask_bool", ExactSpelling = true)]
        extern static MR.RenderModelPassMask __MR_mul_MR_RenderModelPassMask_bool(MR.RenderModelPassMask a, byte b);
        return __MR_mul_MR_RenderModelPassMask_bool(a, b ? (byte)1 : (byte)0);
    }

    /// Generated from function `MR::operator*`.
    public static MR.RenderModelPassMask Mul(bool a, MR.RenderModelPassMask b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_bool_MR_RenderModelPassMask", ExactSpelling = true)]
        extern static MR.RenderModelPassMask __MR_mul_bool_MR_RenderModelPassMask(byte a, MR.RenderModelPassMask b);
        return __MR_mul_bool_MR_RenderModelPassMask(a ? (byte)1 : (byte)0, b);
    }

    /// Generated from function `MR::operator*=`.
    public static unsafe ref MR.RenderModelPassMask MulAssign(ref MR.RenderModelPassMask a, bool b)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_mul_assign_MR_RenderModelPassMask_bool", ExactSpelling = true)]
        extern static MR.RenderModelPassMask *__MR_mul_assign_MR_RenderModelPassMask_bool(MR.RenderModelPassMask *a, byte b);
        fixed (MR.RenderModelPassMask *__ptr_a = &a)
        {
            return ref *__MR_mul_assign_MR_RenderModelPassMask_bool(__ptr_a, b ? (byte)1 : (byte)0);
        }
    }
}
