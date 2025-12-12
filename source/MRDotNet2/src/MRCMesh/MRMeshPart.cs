public static partial class MR
{
    /// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
    /// Generated from class `MR::MeshPart`.
    /// This is the const half of the class.
    public class Const_MeshPart : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshPart(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshPart_Destroy(_Underlying *_this);
            __MR_MeshPart_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshPart() {Dispose(false);}

        public unsafe MR.Const_Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_Get_mesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_MeshPart_Get_mesh(_Underlying *_this);
                return new(__MR_MeshPart_Get_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        // nullptr here means whole mesh
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_Get_region", ExactSpelling = true)]
                extern static void **__MR_MeshPart_Get_region(_Underlying *_this);
                return ref *__MR_MeshPart_Get_region(_UnderlyingPtr);
            }
        }

        // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
        /// Generated from constructor `MR::MeshPart::MeshPart`.
        public unsafe Const_MeshPart(MR.Const_MeshPart other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_MeshPart_ConstructFromAnother(MR.MeshPart._Underlying *other);
            _UnderlyingPtr = __MR_MeshPart_ConstructFromAnother(other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshPart::MeshPart`.
        public unsafe Const_MeshPart(MR.Const_Mesh m, MR.Const_FaceBitSet? bs = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_Construct", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_MeshPart_Construct(MR.Const_Mesh._Underlying *m, MR.Const_FaceBitSet._Underlying *bs);
            _UnderlyingPtr = __MR_MeshPart_Construct(m._UnderlyingPtr, bs is not null ? bs._UnderlyingPtr : null);
        }
    }

    /// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
    /// Generated from class `MR::MeshPart`.
    /// This is the non-const half of the class.
    public class MeshPart : Const_MeshPart
    {
        internal unsafe MeshPart(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // nullptr here means whole mesh
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_MeshPart_GetMutable_region(_Underlying *_this);
                return ref *__MR_MeshPart_GetMutable_region(_UnderlyingPtr);
            }
        }

        // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
        /// Generated from constructor `MR::MeshPart::MeshPart`.
        public unsafe MeshPart(MR.Const_MeshPart other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_MeshPart_ConstructFromAnother(MR.MeshPart._Underlying *other);
            _UnderlyingPtr = __MR_MeshPart_ConstructFromAnother(other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshPart::MeshPart`.
        public unsafe MeshPart(MR.Const_Mesh m, MR.Const_FaceBitSet? bs = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPart_Construct", ExactSpelling = true)]
            extern static MR.MeshPart._Underlying *__MR_MeshPart_Construct(MR.Const_Mesh._Underlying *m, MR.Const_FaceBitSet._Underlying *bs);
            _UnderlyingPtr = __MR_MeshPart_Construct(m._UnderlyingPtr, bs is not null ? bs._UnderlyingPtr : null);
        }
    }

    /// This is used for optional parameters of class `MeshPart` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshPart`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshPart`/`Const_MeshPart` directly.
    public class _InOptMut_MeshPart
    {
        public MeshPart? Opt;

        public _InOptMut_MeshPart() {}
        public _InOptMut_MeshPart(MeshPart value) {Opt = value;}
        public static implicit operator _InOptMut_MeshPart(MeshPart value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshPart` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshPart`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshPart`/`Const_MeshPart` to pass it to the function.
    public class _InOptConst_MeshPart
    {
        public Const_MeshPart? Opt;

        public _InOptConst_MeshPart() {}
        public _InOptConst_MeshPart(Const_MeshPart value) {Opt = value;}
        public static implicit operator _InOptConst_MeshPart(Const_MeshPart value) {return new(value);}
    }

    /// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
    /// Generated from class `MR::MeshVertPart`.
    /// This is the const half of the class.
    public class Const_MeshVertPart : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshVertPart(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshVertPart_Destroy(_Underlying *_this);
            __MR_MeshVertPart_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshVertPart() {Dispose(false);}

        public unsafe MR.Const_Mesh Mesh
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_Get_mesh", ExactSpelling = true)]
                extern static MR.Const_Mesh._Underlying *__MR_MeshVertPart_Get_mesh(_Underlying *_this);
                return new(__MR_MeshVertPart_Get_mesh(_UnderlyingPtr), is_owning: false);
            }
        }

        // nullptr here means whole mesh
        public unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_Get_region", ExactSpelling = true)]
                extern static void **__MR_MeshVertPart_Get_region(_Underlying *_this);
                return ref *__MR_MeshVertPart_Get_region(_UnderlyingPtr);
            }
        }

        // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
        /// Generated from constructor `MR::MeshVertPart::MeshVertPart`.
        public unsafe Const_MeshVertPart(MR.Const_MeshVertPart other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshVertPart._Underlying *__MR_MeshVertPart_ConstructFromAnother(MR.MeshVertPart._Underlying *other);
            _UnderlyingPtr = __MR_MeshVertPart_ConstructFromAnother(other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshVertPart::MeshVertPart`.
        public unsafe Const_MeshVertPart(MR.Const_Mesh m, MR.Const_VertBitSet? bs = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_Construct", ExactSpelling = true)]
            extern static MR.MeshVertPart._Underlying *__MR_MeshVertPart_Construct(MR.Const_Mesh._Underlying *m, MR.Const_VertBitSet._Underlying *bs);
            _UnderlyingPtr = __MR_MeshVertPart_Construct(m._UnderlyingPtr, bs is not null ? bs._UnderlyingPtr : null);
        }
    }

    /// stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)
    /// Generated from class `MR::MeshVertPart`.
    /// This is the non-const half of the class.
    public class MeshVertPart : Const_MeshVertPart
    {
        internal unsafe MeshVertPart(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // nullptr here means whole mesh
        public new unsafe ref readonly void * Region
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_GetMutable_region", ExactSpelling = true)]
                extern static void **__MR_MeshVertPart_GetMutable_region(_Underlying *_this);
                return ref *__MR_MeshVertPart_GetMutable_region(_UnderlyingPtr);
            }
        }

        // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
        /// Generated from constructor `MR::MeshVertPart::MeshVertPart`.
        public unsafe MeshVertPart(MR.Const_MeshVertPart other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshVertPart._Underlying *__MR_MeshVertPart_ConstructFromAnother(MR.MeshVertPart._Underlying *other);
            _UnderlyingPtr = __MR_MeshVertPart_ConstructFromAnother(other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::MeshVertPart::MeshVertPart`.
        public unsafe MeshVertPart(MR.Const_Mesh m, MR.Const_VertBitSet? bs = null) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshVertPart_Construct", ExactSpelling = true)]
            extern static MR.MeshVertPart._Underlying *__MR_MeshVertPart_Construct(MR.Const_Mesh._Underlying *m, MR.Const_VertBitSet._Underlying *bs);
            _UnderlyingPtr = __MR_MeshVertPart_Construct(m._UnderlyingPtr, bs is not null ? bs._UnderlyingPtr : null);
        }
    }

    /// This is used for optional parameters of class `MeshVertPart` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshVertPart`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshVertPart`/`Const_MeshVertPart` directly.
    public class _InOptMut_MeshVertPart
    {
        public MeshVertPart? Opt;

        public _InOptMut_MeshVertPart() {}
        public _InOptMut_MeshVertPart(MeshVertPart value) {Opt = value;}
        public static implicit operator _InOptMut_MeshVertPart(MeshVertPart value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshVertPart` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshVertPart`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshVertPart`/`Const_MeshVertPart` to pass it to the function.
    public class _InOptConst_MeshVertPart
    {
        public Const_MeshVertPart? Opt;

        public _InOptConst_MeshVertPart() {}
        public _InOptConst_MeshVertPart(Const_MeshVertPart value) {Opt = value;}
        public static implicit operator _InOptConst_MeshVertPart(Const_MeshVertPart value) {return new(value);}
    }
}
