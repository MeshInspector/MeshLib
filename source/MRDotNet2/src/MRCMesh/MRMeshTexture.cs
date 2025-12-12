public static partial class MR
{
    /// Generated from class `MR::MeshTexture`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Image`
    /// This is the const half of the class.
    public class Const_MeshTexture : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshTexture(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshTexture_Destroy(_Underlying *_this);
            __MR_MeshTexture_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshTexture() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_Image(Const_MeshTexture self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_UpcastTo_MR_Image", ExactSpelling = true)]
            extern static MR.Const_Image._Underlying *__MR_MeshTexture_UpcastTo_MR_Image(_Underlying *_this);
            MR.Const_Image ret = new(__MR_MeshTexture_UpcastTo_MR_Image(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public unsafe MR.FilterType Filter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_Get_filter", ExactSpelling = true)]
                extern static MR.FilterType *__MR_MeshTexture_Get_filter(_Underlying *_this);
                return *__MR_MeshTexture_Get_filter(_UnderlyingPtr);
            }
        }

        public unsafe MR.WrapType Wrap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_Get_wrap", ExactSpelling = true)]
                extern static MR.WrapType *__MR_MeshTexture_Get_wrap(_Underlying *_this);
                return *__MR_MeshTexture_Get_wrap(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_Vector_MRColor Pixels
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_Get_pixels", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRColor._Underlying *__MR_MeshTexture_Get_pixels(_Underlying *_this);
                return new(__MR_MeshTexture_Get_pixels(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_Get_resolution", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_MeshTexture_Get_resolution(_Underlying *_this);
                return new(__MR_MeshTexture_Get_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshTexture() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTexture._Underlying *__MR_MeshTexture_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTexture_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTexture::MeshTexture`.
        public unsafe Const_MeshTexture(MR._ByValue_MeshTexture _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTexture._Underlying *__MR_MeshTexture_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshTexture._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTexture_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::MeshTexture::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_MeshTexture_heapBytes(_Underlying *_this);
            return __MR_MeshTexture_heapBytes(_UnderlyingPtr);
        }

        /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
        /// returns the color of the closest pixel
        /// Generated from method `MR::MeshTexture::sampleDiscrete`.
        public unsafe MR.Color SampleDiscrete(MR.Const_Vector2f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_sampleDiscrete", ExactSpelling = true)]
            extern static MR.Color __MR_MeshTexture_sampleDiscrete(_Underlying *_this, MR.Const_Vector2f._Underlying *pos);
            return __MR_MeshTexture_sampleDiscrete(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
        /// returns bilinear interpolated color at it
        /// Generated from method `MR::MeshTexture::sampleBilinear`.
        public unsafe MR.Color SampleBilinear(MR.Const_Vector2f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_sampleBilinear", ExactSpelling = true)]
            extern static MR.Color __MR_MeshTexture_sampleBilinear(_Underlying *_this, MR.Const_Vector2f._Underlying *pos);
            return __MR_MeshTexture_sampleBilinear(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
        /// returns sampled color at it according to given filter
        /// Generated from method `MR::MeshTexture::sample`.
        public unsafe MR.Color Sample(MR.FilterType filter, MR.Const_Vector2f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_sample", ExactSpelling = true)]
            extern static MR.Color __MR_MeshTexture_sample(_Underlying *_this, MR.FilterType filter, MR.Const_Vector2f._Underlying *pos);
            return __MR_MeshTexture_sample(_UnderlyingPtr, filter, pos._UnderlyingPtr);
        }
    }

    /// Generated from class `MR::MeshTexture`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::Image`
    /// This is the non-const half of the class.
    public class MeshTexture : Const_MeshTexture
    {
        internal unsafe MeshTexture(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.Image(MeshTexture self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_UpcastTo_MR_Image", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_MeshTexture_UpcastTo_MR_Image(_Underlying *_this);
            MR.Image ret = new(__MR_MeshTexture_UpcastTo_MR_Image(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        public new unsafe ref MR.FilterType Filter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_GetMutable_filter", ExactSpelling = true)]
                extern static MR.FilterType *__MR_MeshTexture_GetMutable_filter(_Underlying *_this);
                return ref *__MR_MeshTexture_GetMutable_filter(_UnderlyingPtr);
            }
        }

        public new unsafe ref MR.WrapType Wrap
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_GetMutable_wrap", ExactSpelling = true)]
                extern static MR.WrapType *__MR_MeshTexture_GetMutable_wrap(_Underlying *_this);
                return ref *__MR_MeshTexture_GetMutable_wrap(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.Vector_MRColor Pixels
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_GetMutable_pixels", ExactSpelling = true)]
                extern static MR.Std.Vector_MRColor._Underlying *__MR_MeshTexture_GetMutable_pixels(_Underlying *_this);
                return new(__MR_MeshTexture_GetMutable_pixels(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_GetMutable_resolution", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_MeshTexture_GetMutable_resolution(_Underlying *_this);
                return new(__MR_MeshTexture_GetMutable_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshTexture() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshTexture._Underlying *__MR_MeshTexture_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshTexture_DefaultConstruct();
        }

        /// Generated from constructor `MR::MeshTexture::MeshTexture`.
        public unsafe MeshTexture(MR._ByValue_MeshTexture _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshTexture._Underlying *__MR_MeshTexture_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshTexture._Underlying *_other);
            _UnderlyingPtr = __MR_MeshTexture_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshTexture::operator=`.
        public unsafe MR.MeshTexture Assign(MR._ByValue_MeshTexture _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshTexture_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshTexture._Underlying *__MR_MeshTexture_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshTexture._Underlying *_other);
            return new(__MR_MeshTexture_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshTexture` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshTexture`/`Const_MeshTexture` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshTexture
    {
        internal readonly Const_MeshTexture? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshTexture() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshTexture(Const_MeshTexture new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshTexture(Const_MeshTexture arg) {return new(arg);}
        public _ByValue_MeshTexture(MR.Misc._Moved<MeshTexture> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshTexture(MR.Misc._Moved<MeshTexture> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshTexture` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshTexture`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTexture`/`Const_MeshTexture` directly.
    public class _InOptMut_MeshTexture
    {
        public MeshTexture? Opt;

        public _InOptMut_MeshTexture() {}
        public _InOptMut_MeshTexture(MeshTexture value) {Opt = value;}
        public static implicit operator _InOptMut_MeshTexture(MeshTexture value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshTexture` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshTexture`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshTexture`/`Const_MeshTexture` to pass it to the function.
    public class _InOptConst_MeshTexture
    {
        public Const_MeshTexture? Opt;

        public _InOptConst_MeshTexture() {}
        public _InOptConst_MeshTexture(Const_MeshTexture value) {Opt = value;}
        public static implicit operator _InOptConst_MeshTexture(Const_MeshTexture value) {return new(value);}
    }
}
