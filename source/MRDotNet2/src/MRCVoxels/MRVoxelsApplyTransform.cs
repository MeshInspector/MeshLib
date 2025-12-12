public static partial class MR
{
    /// Generated from class `MR::TransformVdbVolumeResult`.
    /// This is the const half of the class.
    public class Const_TransformVdbVolumeResult : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TransformVdbVolumeResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_Destroy", ExactSpelling = true)]
            extern static void __MR_TransformVdbVolumeResult_Destroy(_Underlying *_this);
            __MR_TransformVdbVolumeResult_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TransformVdbVolumeResult() {Dispose(false);}

        public unsafe MR.Const_VdbVolume Volume
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_Get_volume", ExactSpelling = true)]
                extern static MR.Const_VdbVolume._Underlying *__MR_TransformVdbVolumeResult_Get_volume(_Underlying *_this);
                return new(__MR_TransformVdbVolumeResult_Get_volume(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe bool BoxFixed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_Get_boxFixed", ExactSpelling = true)]
                extern static bool *__MR_TransformVdbVolumeResult_Get_boxFixed(_Underlying *_this);
                return *__MR_TransformVdbVolumeResult_Get_boxFixed(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_TransformVdbVolumeResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_DefaultConstruct();
            _UnderlyingPtr = __MR_TransformVdbVolumeResult_DefaultConstruct();
        }

        /// Constructs `MR::TransformVdbVolumeResult` elementwise.
        public unsafe Const_TransformVdbVolumeResult(MR._ByValue_VdbVolume volume, bool boxFixed) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_ConstructFrom(MR.Misc._PassBy volume_pass_by, MR.VdbVolume._Underlying *volume, byte boxFixed);
            _UnderlyingPtr = __MR_TransformVdbVolumeResult_ConstructFrom(volume.PassByMode, volume.Value is not null ? volume.Value._UnderlyingPtr : null, boxFixed ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::TransformVdbVolumeResult::TransformVdbVolumeResult`.
        public unsafe Const_TransformVdbVolumeResult(MR._ByValue_TransformVdbVolumeResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TransformVdbVolumeResult._Underlying *_other);
            _UnderlyingPtr = __MR_TransformVdbVolumeResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// Generated from class `MR::TransformVdbVolumeResult`.
    /// This is the non-const half of the class.
    public class TransformVdbVolumeResult : Const_TransformVdbVolumeResult
    {
        internal unsafe TransformVdbVolumeResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.VdbVolume Volume
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_GetMutable_volume", ExactSpelling = true)]
                extern static MR.VdbVolume._Underlying *__MR_TransformVdbVolumeResult_GetMutable_volume(_Underlying *_this);
                return new(__MR_TransformVdbVolumeResult_GetMutable_volume(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref bool BoxFixed
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_GetMutable_boxFixed", ExactSpelling = true)]
                extern static bool *__MR_TransformVdbVolumeResult_GetMutable_boxFixed(_Underlying *_this);
                return ref *__MR_TransformVdbVolumeResult_GetMutable_boxFixed(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe TransformVdbVolumeResult() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_DefaultConstruct", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_DefaultConstruct();
            _UnderlyingPtr = __MR_TransformVdbVolumeResult_DefaultConstruct();
        }

        /// Constructs `MR::TransformVdbVolumeResult` elementwise.
        public unsafe TransformVdbVolumeResult(MR._ByValue_VdbVolume volume, bool boxFixed) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_ConstructFrom", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_ConstructFrom(MR.Misc._PassBy volume_pass_by, MR.VdbVolume._Underlying *volume, byte boxFixed);
            _UnderlyingPtr = __MR_TransformVdbVolumeResult_ConstructFrom(volume.PassByMode, volume.Value is not null ? volume.Value._UnderlyingPtr : null, boxFixed ? (byte)1 : (byte)0);
        }

        /// Generated from constructor `MR::TransformVdbVolumeResult::TransformVdbVolumeResult`.
        public unsafe TransformVdbVolumeResult(MR._ByValue_TransformVdbVolumeResult _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TransformVdbVolumeResult._Underlying *_other);
            _UnderlyingPtr = __MR_TransformVdbVolumeResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::TransformVdbVolumeResult::operator=`.
        public unsafe MR.TransformVdbVolumeResult Assign(MR._ByValue_TransformVdbVolumeResult _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TransformVdbVolumeResult_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TransformVdbVolumeResult._Underlying *__MR_TransformVdbVolumeResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TransformVdbVolumeResult._Underlying *_other);
            return new(__MR_TransformVdbVolumeResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TransformVdbVolumeResult` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TransformVdbVolumeResult`/`Const_TransformVdbVolumeResult` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TransformVdbVolumeResult
    {
        internal readonly Const_TransformVdbVolumeResult? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TransformVdbVolumeResult() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_TransformVdbVolumeResult(Const_TransformVdbVolumeResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TransformVdbVolumeResult(Const_TransformVdbVolumeResult arg) {return new(arg);}
        public _ByValue_TransformVdbVolumeResult(MR.Misc._Moved<TransformVdbVolumeResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TransformVdbVolumeResult(MR.Misc._Moved<TransformVdbVolumeResult> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TransformVdbVolumeResult` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TransformVdbVolumeResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TransformVdbVolumeResult`/`Const_TransformVdbVolumeResult` directly.
    public class _InOptMut_TransformVdbVolumeResult
    {
        public TransformVdbVolumeResult? Opt;

        public _InOptMut_TransformVdbVolumeResult() {}
        public _InOptMut_TransformVdbVolumeResult(TransformVdbVolumeResult value) {Opt = value;}
        public static implicit operator _InOptMut_TransformVdbVolumeResult(TransformVdbVolumeResult value) {return new(value);}
    }

    /// This is used for optional parameters of class `TransformVdbVolumeResult` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TransformVdbVolumeResult`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TransformVdbVolumeResult`/`Const_TransformVdbVolumeResult` to pass it to the function.
    public class _InOptConst_TransformVdbVolumeResult
    {
        public Const_TransformVdbVolumeResult? Opt;

        public _InOptConst_TransformVdbVolumeResult() {}
        public _InOptConst_TransformVdbVolumeResult(Const_TransformVdbVolumeResult value) {Opt = value;}
        public static implicit operator _InOptConst_TransformVdbVolumeResult(Const_TransformVdbVolumeResult value) {return new(value);}
    }

    /// Transform volume
    /// @param volume Volume to transform
    /// @param xf The transformation
    /// @param fixBox If true, and if \p box is valid and represents the bounding box of the \p volume, then
    ///               the result will be shifted so that no data has negative coordinate by any of dimensions
    /// Generated from function `MR::transformVdbVolume`.
    /// Parameter `fixBox` defaults to `false`.
    /// Parameter `box` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.TransformVdbVolumeResult> TransformVdbVolume(MR.Const_VdbVolume volume, MR.Const_AffineXf3f xf, bool? fixBox = null, MR.Const_Box3f? box = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_transformVdbVolume", ExactSpelling = true)]
        extern static MR.TransformVdbVolumeResult._Underlying *__MR_transformVdbVolume(MR.Const_VdbVolume._Underlying *volume, MR.Const_AffineXf3f._Underlying *xf, byte *fixBox, MR.Const_Box3f._Underlying *box);
        byte __deref_fixBox = fixBox.GetValueOrDefault() ? (byte)1 : (byte)0;
        return MR.Misc.Move(new MR.TransformVdbVolumeResult(__MR_transformVdbVolume(volume._UnderlyingPtr, xf._UnderlyingPtr, fixBox.HasValue ? &__deref_fixBox : null, box is not null ? box._UnderlyingPtr : null), is_owning: true));
    }

    /// Same as above but for the SceneObject
    /// @return true, if \p fixBox is true and the box was "fixed" (see parameter `fixBox` of \ref transformVdbVolume)
    /// Generated from function `MR::voxelsApplyTransform`.
    public static unsafe bool VoxelsApplyTransform(MR.ObjectVoxels obj, MR.Const_AffineXf3f xf, bool fixBox)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_voxelsApplyTransform", ExactSpelling = true)]
        extern static byte __MR_voxelsApplyTransform(MR.ObjectVoxels._Underlying *obj, MR.Const_AffineXf3f._Underlying *xf, byte fixBox);
        return __MR_voxelsApplyTransform(obj._UnderlyingPtr, xf._UnderlyingPtr, fixBox ? (byte)1 : (byte)0) != 0;
    }
}
