public static partial class MR
{
    /// struct to hold Image data
    /// Generated from class `MR::Image`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshTexture`
    /// This is the const half of the class.
    public class Const_Image : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Image(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_Destroy", ExactSpelling = true)]
            extern static void __MR_Image_Destroy(_Underlying *_this);
            __MR_Image_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Image() {Dispose(false);}

        public unsafe MR.Std.Const_Vector_MRColor Pixels
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_Get_pixels", ExactSpelling = true)]
                extern static MR.Std.Const_Vector_MRColor._Underlying *__MR_Image_Get_pixels(_Underlying *_this);
                return new(__MR_Image_Get_pixels(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe MR.Const_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_Get_resolution", ExactSpelling = true)]
                extern static MR.Const_Vector2i._Underlying *__MR_Image_Get_resolution(_Underlying *_this);
                return new(__MR_Image_Get_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Image() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_DefaultConstruct();
            _UnderlyingPtr = __MR_Image_DefaultConstruct();
        }

        /// Constructs `MR::Image` elementwise.
        public unsafe Const_Image(MR.Std._ByValue_Vector_MRColor pixels, MR.Vector2i resolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_ConstructFrom", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_ConstructFrom(MR.Misc._PassBy pixels_pass_by, MR.Std.Vector_MRColor._Underlying *pixels, MR.Vector2i resolution);
            _UnderlyingPtr = __MR_Image_ConstructFrom(pixels.PassByMode, pixels.Value is not null ? pixels.Value._UnderlyingPtr : null, resolution);
        }

        /// Generated from constructor `MR::Image::Image`.
        public unsafe Const_Image(MR._ByValue_Image _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Image._Underlying *_other);
            _UnderlyingPtr = __MR_Image_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Image::operator[]`.
        public unsafe MR.Color Index(MR.Const_Vector2i p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_index_const", ExactSpelling = true)]
            extern static MR.Color __MR_Image_index_const(_Underlying *_this, MR.Const_Vector2i._Underlying *p);
            return __MR_Image_index_const(_UnderlyingPtr, p._UnderlyingPtr);
        }

        /// returns the amount of memory this object occupies on heap
        /// Generated from method `MR::Image::heapBytes`.
        public unsafe ulong HeapBytes()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_heapBytes", ExactSpelling = true)]
            extern static ulong __MR_Image_heapBytes(_Underlying *_this);
            return __MR_Image_heapBytes(_UnderlyingPtr);
        }

        /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
        /// returns the color of the closest pixel
        /// Generated from method `MR::Image::sampleDiscrete`.
        public unsafe MR.Color SampleDiscrete(MR.Const_Vector2f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_sampleDiscrete", ExactSpelling = true)]
            extern static MR.Color __MR_Image_sampleDiscrete(_Underlying *_this, MR.Const_Vector2f._Underlying *pos);
            return __MR_Image_sampleDiscrete(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
        /// returns bilinear interpolated color at it
        /// Generated from method `MR::Image::sampleBilinear`.
        public unsafe MR.Color SampleBilinear(MR.Const_Vector2f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_sampleBilinear", ExactSpelling = true)]
            extern static MR.Color __MR_Image_sampleBilinear(_Underlying *_this, MR.Const_Vector2f._Underlying *pos);
            return __MR_Image_sampleBilinear(_UnderlyingPtr, pos._UnderlyingPtr);
        }

        /// Given texture position in [0,1]x[0,1] (which is clamped if necessary),
        /// returns sampled color at it according to given filter
        /// Generated from method `MR::Image::sample`.
        public unsafe MR.Color Sample(MR.FilterType filter, MR.Const_Vector2f pos)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_sample", ExactSpelling = true)]
            extern static MR.Color __MR_Image_sample(_Underlying *_this, MR.FilterType filter, MR.Const_Vector2f._Underlying *pos);
            return __MR_Image_sample(_UnderlyingPtr, filter, pos._UnderlyingPtr);
        }
    }

    /// struct to hold Image data
    /// Generated from class `MR::Image`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::MeshTexture`
    /// This is the non-const half of the class.
    public class Image : Const_Image
    {
        internal unsafe Image(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Std.Vector_MRColor Pixels
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_GetMutable_pixels", ExactSpelling = true)]
                extern static MR.Std.Vector_MRColor._Underlying *__MR_Image_GetMutable_pixels(_Underlying *_this);
                return new(__MR_Image_GetMutable_pixels(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe MR.Mut_Vector2i Resolution
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_GetMutable_resolution", ExactSpelling = true)]
                extern static MR.Mut_Vector2i._Underlying *__MR_Image_GetMutable_resolution(_Underlying *_this);
                return new(__MR_Image_GetMutable_resolution(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Image() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_DefaultConstruct();
            _UnderlyingPtr = __MR_Image_DefaultConstruct();
        }

        /// Constructs `MR::Image` elementwise.
        public unsafe Image(MR.Std._ByValue_Vector_MRColor pixels, MR.Vector2i resolution) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_ConstructFrom", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_ConstructFrom(MR.Misc._PassBy pixels_pass_by, MR.Std.Vector_MRColor._Underlying *pixels, MR.Vector2i resolution);
            _UnderlyingPtr = __MR_Image_ConstructFrom(pixels.PassByMode, pixels.Value is not null ? pixels.Value._UnderlyingPtr : null, resolution);
        }

        /// Generated from constructor `MR::Image::Image`.
        public unsafe Image(MR._ByValue_Image _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Image._Underlying *_other);
            _UnderlyingPtr = __MR_Image_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Image::operator=`.
        public unsafe MR.Image Assign(MR._ByValue_Image _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Image._Underlying *__MR_Image_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Image._Underlying *_other);
            return new(__MR_Image_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// fetches some texture element specified by integer coordinates
        /// Generated from method `MR::Image::operator[]`.
        public unsafe new MR.Mut_Color Index(MR.Const_Vector2i p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Image_index", ExactSpelling = true)]
            extern static MR.Mut_Color._Underlying *__MR_Image_index(_Underlying *_this, MR.Const_Vector2i._Underlying *p);
            return new(__MR_Image_index(_UnderlyingPtr, p._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Image` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Image`/`Const_Image` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Image
    {
        internal readonly Const_Image? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Image() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_Image(Const_Image new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Image(Const_Image arg) {return new(arg);}
        public _ByValue_Image(MR.Misc._Moved<Image> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Image(MR.Misc._Moved<Image> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Image` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Image`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Image`/`Const_Image` directly.
    public class _InOptMut_Image
    {
        public Image? Opt;

        public _InOptMut_Image() {}
        public _InOptMut_Image(Image value) {Opt = value;}
        public static implicit operator _InOptMut_Image(Image value) {return new(value);}
    }

    /// This is used for optional parameters of class `Image` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Image`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Image`/`Const_Image` to pass it to the function.
    public class _InOptConst_Image
    {
        public Const_Image? Opt;

        public _InOptConst_Image() {}
        public _InOptConst_Image(Const_Image value) {Opt = value;}
        public static implicit operator _InOptConst_Image(Const_Image value) {return new(value);}
    }
}
