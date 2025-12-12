public static partial class MR
{
    /// This class represents tooth id
    /// Generated from class `MR::DentalId`.
    /// This is the const half of the class.
    public class Const_DentalId : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DentalId(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DentalId_Destroy", ExactSpelling = true)]
            extern static void __MR_DentalId_Destroy(_Underlying *_this);
            __MR_DentalId_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DentalId() {Dispose(false);}

        /// Generated from constructor `MR::DentalId::DentalId`.
        public unsafe Const_DentalId(MR.Const_DentalId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DentalId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DentalId._Underlying *__MR_DentalId_ConstructFromAnother(MR.DentalId._Underlying *_other);
            _UnderlyingPtr = __MR_DentalId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Creates id from FDI number known only at runtime
        /// Generated from method `MR::DentalId::fromFDI`.
        public static unsafe MR.Std.Optional_MRDentalId FromFDI(int id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DentalId_fromFDI", ExactSpelling = true)]
            extern static MR.Std.Optional_MRDentalId._Underlying *__MR_DentalId_fromFDI(int id);
            return new(__MR_DentalId_fromFDI(id), is_owning: true);
        }

        /// Returns FDI representation of the id
        /// Generated from method `MR::DentalId::fdi`.
        public unsafe int Fdi()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DentalId_fdi", ExactSpelling = true)]
            extern static int __MR_DentalId_fdi(_Underlying *_this);
            return __MR_DentalId_fdi(_UnderlyingPtr);
        }
    }

    /// This class represents tooth id
    /// Generated from class `MR::DentalId`.
    /// This is the non-const half of the class.
    public class DentalId : Const_DentalId
    {
        internal unsafe DentalId(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::DentalId::DentalId`.
        public unsafe DentalId(MR.Const_DentalId _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DentalId_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DentalId._Underlying *__MR_DentalId_ConstructFromAnother(MR.DentalId._Underlying *_other);
            _UnderlyingPtr = __MR_DentalId_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::DentalId::operator=`.
        public unsafe MR.DentalId Assign(MR.Const_DentalId _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DentalId_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DentalId._Underlying *__MR_DentalId_AssignFromAnother(_Underlying *_this, MR.DentalId._Underlying *_other);
            return new(__MR_DentalId_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DentalId` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DentalId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DentalId`/`Const_DentalId` directly.
    public class _InOptMut_DentalId
    {
        public DentalId? Opt;

        public _InOptMut_DentalId() {}
        public _InOptMut_DentalId(DentalId value) {Opt = value;}
        public static implicit operator _InOptMut_DentalId(DentalId value) {return new(value);}
    }

    /// This is used for optional parameters of class `DentalId` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DentalId`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DentalId`/`Const_DentalId` to pass it to the function.
    public class _InOptConst_DentalId
    {
        public Const_DentalId? Opt;

        public _InOptConst_DentalId() {}
        public _InOptConst_DentalId(Const_DentalId value) {Opt = value;}
        public static implicit operator _InOptConst_DentalId(Const_DentalId value) {return new(value);}
    }

    /// This class is an alternative to directly invoking \ref meshToDirectionVolume for the mesh retrieved from the teeth mask.
    /// It is better because when a single mesh is created from mask, some neighboring teeth might fuse together, creating incorrect mask.
    /// This class invokes meshing for each teeth separately, thus eliminating this problem.
    /// Generated from class `MR::TeethMaskToDirectionVolumeConvertor`.
    /// This is the const half of the class.
    public class Const_TeethMaskToDirectionVolumeConvertor : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_TeethMaskToDirectionVolumeConvertor(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_Destroy", ExactSpelling = true)]
            extern static void __MR_TeethMaskToDirectionVolumeConvertor_Destroy(_Underlying *_this);
            __MR_TeethMaskToDirectionVolumeConvertor_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_TeethMaskToDirectionVolumeConvertor() {Dispose(false);}

        /// Generated from constructor `MR::TeethMaskToDirectionVolumeConvertor::TeethMaskToDirectionVolumeConvertor`.
        public unsafe Const_TeethMaskToDirectionVolumeConvertor(MR._ByValue_TeethMaskToDirectionVolumeConvertor _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TeethMaskToDirectionVolumeConvertor._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TeethMaskToDirectionVolumeConvertor._Underlying *_other);
            _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Initialize class
        /// @param volume Voxel mask
        /// @param additionalIds List of additional ids (besides teeth) to convert
        /// Generated from method `MR::TeethMaskToDirectionVolumeConvertor::create`.
        /// Parameter `additionalIds` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRTeethMaskToDirectionVolumeConvertor_StdString> Create(MR.Const_VdbVolume volume, MR.Std.Const_Vector_Int? additionalIds = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_create", ExactSpelling = true)]
            extern static MR.Expected_MRTeethMaskToDirectionVolumeConvertor_StdString._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_create(MR.Const_VdbVolume._Underlying *volume, MR.Std.Const_Vector_Int._Underlying *additionalIds);
            return MR.Misc.Move(new MR.Expected_MRTeethMaskToDirectionVolumeConvertor_StdString(__MR_TeethMaskToDirectionVolumeConvertor_create(volume._UnderlyingPtr, additionalIds is not null ? additionalIds._UnderlyingPtr : null), is_owning: true));
        }

        /// Returns all the objects present in volume and corresponding bounding boxes
        /// Generated from method `MR::TeethMaskToDirectionVolumeConvertor::getObjectBounds`.
        public unsafe MR.Phmap.Const_FlatHashMap_Int_MRBox3i_PhmapHashInt32T GetObjectBounds()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_getObjectBounds", ExactSpelling = true)]
            extern static MR.Phmap.Const_FlatHashMap_Int_MRBox3i_PhmapHashInt32T._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_getObjectBounds(_Underlying *_this);
            return new(__MR_TeethMaskToDirectionVolumeConvertor_getObjectBounds(_UnderlyingPtr), is_owning: false);
        }

        /// Converts single object into direction volume
        /// Generated from method `MR::TeethMaskToDirectionVolumeConvertor::convertObject`.
        public unsafe MR.Misc._Moved<MR.Expected_MRTeethMaskToDirectionVolumeConvertorProcessResult_StdString> ConvertObject(int id)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_convertObject", ExactSpelling = true)]
            extern static MR.Expected_MRTeethMaskToDirectionVolumeConvertorProcessResult_StdString._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_convertObject(_Underlying *_this, int id);
            return MR.Misc.Move(new MR.Expected_MRTeethMaskToDirectionVolumeConvertorProcessResult_StdString(__MR_TeethMaskToDirectionVolumeConvertor_convertObject(_UnderlyingPtr, id), is_owning: true));
        }

        /// Converts all the objects into direction volume
        /// Generated from method `MR::TeethMaskToDirectionVolumeConvertor::convertAll`.
        public unsafe MR.Misc._Moved<MR.Expected_MRTeethMaskToDirectionVolumeConvertorProcessResult_StdString> ConvertAll()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_convertAll", ExactSpelling = true)]
            extern static MR.Expected_MRTeethMaskToDirectionVolumeConvertorProcessResult_StdString._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_convertAll(_Underlying *_this);
            return MR.Misc.Move(new MR.Expected_MRTeethMaskToDirectionVolumeConvertorProcessResult_StdString(__MR_TeethMaskToDirectionVolumeConvertor_convertAll(_UnderlyingPtr), is_owning: true));
        }

        /// Generated from class `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult`.
        /// This is the const half of the class.
        public class Const_ProcessResult : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ProcessResult(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Destroy", ExactSpelling = true)]
                extern static void __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Destroy(_Underlying *_this);
                __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ProcessResult() {Dispose(false);}

            public unsafe MR.Std.Const_Array_MRSimpleVolumeMinMax_3 Volume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Get_volume", ExactSpelling = true)]
                    extern static MR.Std.Const_Array_MRSimpleVolumeMinMax_3._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Get_volume(_Underlying *_this);
                    return new(__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Get_volume(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Get_xf", ExactSpelling = true)]
                    extern static MR.Const_AffineXf3f._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Get_xf(_Underlying *_this);
                    return new(__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_Get_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ProcessResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_DefaultConstruct();
                _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_DefaultConstruct();
            }

            /// Constructs `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult` elementwise.
            public unsafe Const_ProcessResult(MR.Std._ByValue_Array_MRSimpleVolumeMinMax_3 volume, MR.AffineXf3f xf) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFrom(MR.Misc._PassBy volume_pass_by, MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *volume, MR.AffineXf3f xf);
                _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFrom(volume.PassByMode, volume.Value is not null ? volume.Value._UnderlyingPtr : null, xf);
            }

            /// Generated from constructor `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult::ProcessResult`.
            public unsafe Const_ProcessResult(MR.TeethMaskToDirectionVolumeConvertor._ByValue_ProcessResult _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *_other);
                _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult`.
        /// This is the non-const half of the class.
        public class ProcessResult : Const_ProcessResult
        {
            internal unsafe ProcessResult(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.Array_MRSimpleVolumeMinMax_3 Volume
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_GetMutable_volume", ExactSpelling = true)]
                    extern static MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_GetMutable_volume(_Underlying *_this);
                    return new(__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_GetMutable_volume(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_AffineXf3f Xf
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_GetMutable_xf", ExactSpelling = true)]
                    extern static MR.Mut_AffineXf3f._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_GetMutable_xf(_Underlying *_this);
                    return new(__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_GetMutable_xf(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ProcessResult() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_DefaultConstruct();
                _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_DefaultConstruct();
            }

            /// Constructs `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult` elementwise.
            public unsafe ProcessResult(MR.Std._ByValue_Array_MRSimpleVolumeMinMax_3 volume, MR.AffineXf3f xf) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFrom", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFrom(MR.Misc._PassBy volume_pass_by, MR.Std.Array_MRSimpleVolumeMinMax_3._Underlying *volume, MR.AffineXf3f xf);
                _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFrom(volume.PassByMode, volume.Value is not null ? volume.Value._UnderlyingPtr : null, xf);
            }

            /// Generated from constructor `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult::ProcessResult`.
            public unsafe ProcessResult(MR.TeethMaskToDirectionVolumeConvertor._ByValue_ProcessResult _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *_other);
                _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::TeethMaskToDirectionVolumeConvertor::ProcessResult::operator=`.
            public unsafe MR.TeethMaskToDirectionVolumeConvertor.ProcessResult Assign(MR.TeethMaskToDirectionVolumeConvertor._ByValue_ProcessResult _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_AssignFromAnother", ExactSpelling = true)]
                extern static MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TeethMaskToDirectionVolumeConvertor.ProcessResult._Underlying *_other);
                return new(__MR_TeethMaskToDirectionVolumeConvertor_ProcessResult_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ProcessResult` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ProcessResult`/`Const_ProcessResult` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ProcessResult
        {
            internal readonly Const_ProcessResult? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ProcessResult() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ProcessResult(Const_ProcessResult new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ProcessResult(Const_ProcessResult arg) {return new(arg);}
            public _ByValue_ProcessResult(MR.Misc._Moved<ProcessResult> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ProcessResult(MR.Misc._Moved<ProcessResult> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ProcessResult` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ProcessResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ProcessResult`/`Const_ProcessResult` directly.
        public class _InOptMut_ProcessResult
        {
            public ProcessResult? Opt;

            public _InOptMut_ProcessResult() {}
            public _InOptMut_ProcessResult(ProcessResult value) {Opt = value;}
            public static implicit operator _InOptMut_ProcessResult(ProcessResult value) {return new(value);}
        }

        /// This is used for optional parameters of class `ProcessResult` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ProcessResult`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ProcessResult`/`Const_ProcessResult` to pass it to the function.
        public class _InOptConst_ProcessResult
        {
            public Const_ProcessResult? Opt;

            public _InOptConst_ProcessResult() {}
            public _InOptConst_ProcessResult(Const_ProcessResult value) {Opt = value;}
            public static implicit operator _InOptConst_ProcessResult(Const_ProcessResult value) {return new(value);}
        }
    }

    /// This class is an alternative to directly invoking \ref meshToDirectionVolume for the mesh retrieved from the teeth mask.
    /// It is better because when a single mesh is created from mask, some neighboring teeth might fuse together, creating incorrect mask.
    /// This class invokes meshing for each teeth separately, thus eliminating this problem.
    /// Generated from class `MR::TeethMaskToDirectionVolumeConvertor`.
    /// This is the non-const half of the class.
    public class TeethMaskToDirectionVolumeConvertor : Const_TeethMaskToDirectionVolumeConvertor
    {
        internal unsafe TeethMaskToDirectionVolumeConvertor(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::TeethMaskToDirectionVolumeConvertor::TeethMaskToDirectionVolumeConvertor`.
        public unsafe TeethMaskToDirectionVolumeConvertor(MR._ByValue_TeethMaskToDirectionVolumeConvertor _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.TeethMaskToDirectionVolumeConvertor._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TeethMaskToDirectionVolumeConvertor._Underlying *_other);
            _UnderlyingPtr = __MR_TeethMaskToDirectionVolumeConvertor_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::TeethMaskToDirectionVolumeConvertor::operator=`.
        public unsafe MR.TeethMaskToDirectionVolumeConvertor Assign(MR._ByValue_TeethMaskToDirectionVolumeConvertor _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TeethMaskToDirectionVolumeConvertor_AssignFromAnother", ExactSpelling = true)]
            extern static MR.TeethMaskToDirectionVolumeConvertor._Underlying *__MR_TeethMaskToDirectionVolumeConvertor_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TeethMaskToDirectionVolumeConvertor._Underlying *_other);
            return new(__MR_TeethMaskToDirectionVolumeConvertor_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }
    }

    /// This is used as a function parameter when the underlying function receives `TeethMaskToDirectionVolumeConvertor` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `TeethMaskToDirectionVolumeConvertor`/`Const_TeethMaskToDirectionVolumeConvertor` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_TeethMaskToDirectionVolumeConvertor
    {
        internal readonly Const_TeethMaskToDirectionVolumeConvertor? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_TeethMaskToDirectionVolumeConvertor(Const_TeethMaskToDirectionVolumeConvertor new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_TeethMaskToDirectionVolumeConvertor(Const_TeethMaskToDirectionVolumeConvertor arg) {return new(arg);}
        public _ByValue_TeethMaskToDirectionVolumeConvertor(MR.Misc._Moved<TeethMaskToDirectionVolumeConvertor> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_TeethMaskToDirectionVolumeConvertor(MR.Misc._Moved<TeethMaskToDirectionVolumeConvertor> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `TeethMaskToDirectionVolumeConvertor` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_TeethMaskToDirectionVolumeConvertor`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TeethMaskToDirectionVolumeConvertor`/`Const_TeethMaskToDirectionVolumeConvertor` directly.
    public class _InOptMut_TeethMaskToDirectionVolumeConvertor
    {
        public TeethMaskToDirectionVolumeConvertor? Opt;

        public _InOptMut_TeethMaskToDirectionVolumeConvertor() {}
        public _InOptMut_TeethMaskToDirectionVolumeConvertor(TeethMaskToDirectionVolumeConvertor value) {Opt = value;}
        public static implicit operator _InOptMut_TeethMaskToDirectionVolumeConvertor(TeethMaskToDirectionVolumeConvertor value) {return new(value);}
    }

    /// This is used for optional parameters of class `TeethMaskToDirectionVolumeConvertor` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_TeethMaskToDirectionVolumeConvertor`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `TeethMaskToDirectionVolumeConvertor`/`Const_TeethMaskToDirectionVolumeConvertor` to pass it to the function.
    public class _InOptConst_TeethMaskToDirectionVolumeConvertor
    {
        public Const_TeethMaskToDirectionVolumeConvertor? Opt;

        public _InOptConst_TeethMaskToDirectionVolumeConvertor() {}
        public _InOptConst_TeethMaskToDirectionVolumeConvertor(Const_TeethMaskToDirectionVolumeConvertor value) {Opt = value;}
        public static implicit operator _InOptConst_TeethMaskToDirectionVolumeConvertor(Const_TeethMaskToDirectionVolumeConvertor value) {return new(value);}
    }

    /// A shortcut for \ref TeethMaskToDirectionVolumeConvertor::create and \ref TeethMaskToDirectionVolumeConvertor::convertAll
    /// Generated from function `MR::teethMaskToDirectionVolume`.
    /// Parameter `additionalIds` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Expected_StdArrayMRSimpleVolumeMinMax3_StdString> TeethMaskToDirectionVolume(MR.Const_VdbVolume volume, MR.Std.Const_Vector_Int? additionalIds = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_teethMaskToDirectionVolume", ExactSpelling = true)]
        extern static MR.Expected_StdArrayMRSimpleVolumeMinMax3_StdString._Underlying *__MR_teethMaskToDirectionVolume(MR.Const_VdbVolume._Underlying *volume, MR.Std.Const_Vector_Int._Underlying *additionalIds);
        return MR.Misc.Move(new MR.Expected_StdArrayMRSimpleVolumeMinMax3_StdString(__MR_teethMaskToDirectionVolume(volume._UnderlyingPtr, additionalIds is not null ? additionalIds._UnderlyingPtr : null), is_owning: true));
    }
}
