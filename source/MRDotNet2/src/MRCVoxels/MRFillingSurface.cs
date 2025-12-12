public static partial class MR
{
    public static partial class FillingSurface
    {
        public static partial class TPMS
        {
            /// Supported types of TPMS (Triply Periodic Minimal Surfaces)
            public enum Type : int
            {
                SchwartzP = 0,
                ThickSchwartzP = 1,
                DoubleGyroid = 2,
                ThickGyroid = 3,
                Count = 4,
            }

            /// Generated from class `MR::FillingSurface::TPMS::VolumeParams`.
            /// Derived classes:
            ///   Direct: (non-virtual)
            ///     `MR::FillingSurface::TPMS::MeshParams`
            /// This is the const half of the class.
            public class Const_VolumeParams : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_VolumeParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_Destroy", ExactSpelling = true)]
                    extern static void __MR_FillingSurface_TPMS_VolumeParams_Destroy(_Underlying *_this);
                    __MR_FillingSurface_TPMS_VolumeParams_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_VolumeParams() {Dispose(false);}

                // Type of the surface
                public unsafe MR.FillingSurface.TPMS.Type Type
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_Get_type", ExactSpelling = true)]
                        extern static MR.FillingSurface.TPMS.Type *__MR_FillingSurface_TPMS_VolumeParams_Get_type(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_VolumeParams_Get_type(_UnderlyingPtr);
                    }
                }

                // Frequency of oscillations (determines size of the "cells" in the "grid")
                public unsafe float Frequency
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_Get_frequency", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_VolumeParams_Get_frequency(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_VolumeParams_Get_frequency(_UnderlyingPtr);
                    }
                }

                // Ratio `n / T`, between the number of voxels and period of oscillations
                public unsafe float Resolution
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_Get_resolution", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_VolumeParams_Get_resolution(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_VolumeParams_Get_resolution(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_VolumeParams() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_DefaultConstruct();
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_VolumeParams_DefaultConstruct();
                }

                /// Constructs `MR::FillingSurface::TPMS::VolumeParams` elementwise.
                public unsafe Const_VolumeParams(MR.FillingSurface.TPMS.Type type, float frequency, float resolution) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_ConstructFrom", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_ConstructFrom(MR.FillingSurface.TPMS.Type type, float frequency, float resolution);
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_VolumeParams_ConstructFrom(type, frequency, resolution);
                }

                /// Generated from constructor `MR::FillingSurface::TPMS::VolumeParams::VolumeParams`.
                public unsafe Const_VolumeParams(MR.FillingSurface.TPMS.Const_VolumeParams _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_ConstructFromAnother(MR.FillingSurface.TPMS.VolumeParams._Underlying *_other);
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_VolumeParams_ConstructFromAnother(_other._UnderlyingPtr);
                }
            }

            /// Generated from class `MR::FillingSurface::TPMS::VolumeParams`.
            /// Derived classes:
            ///   Direct: (non-virtual)
            ///     `MR::FillingSurface::TPMS::MeshParams`
            /// This is the non-const half of the class.
            public class VolumeParams : Const_VolumeParams
            {
                internal unsafe VolumeParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                // Type of the surface
                public new unsafe ref MR.FillingSurface.TPMS.Type Type
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_GetMutable_type", ExactSpelling = true)]
                        extern static MR.FillingSurface.TPMS.Type *__MR_FillingSurface_TPMS_VolumeParams_GetMutable_type(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_VolumeParams_GetMutable_type(_UnderlyingPtr);
                    }
                }

                // Frequency of oscillations (determines size of the "cells" in the "grid")
                public new unsafe ref float Frequency
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_GetMutable_frequency", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_VolumeParams_GetMutable_frequency(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_VolumeParams_GetMutable_frequency(_UnderlyingPtr);
                    }
                }

                // Ratio `n / T`, between the number of voxels and period of oscillations
                public new unsafe ref float Resolution
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_GetMutable_resolution", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_VolumeParams_GetMutable_resolution(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_VolumeParams_GetMutable_resolution(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe VolumeParams() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_DefaultConstruct();
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_VolumeParams_DefaultConstruct();
                }

                /// Constructs `MR::FillingSurface::TPMS::VolumeParams` elementwise.
                public unsafe VolumeParams(MR.FillingSurface.TPMS.Type type, float frequency, float resolution) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_ConstructFrom", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_ConstructFrom(MR.FillingSurface.TPMS.Type type, float frequency, float resolution);
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_VolumeParams_ConstructFrom(type, frequency, resolution);
                }

                /// Generated from constructor `MR::FillingSurface::TPMS::VolumeParams::VolumeParams`.
                public unsafe VolumeParams(MR.FillingSurface.TPMS.Const_VolumeParams _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_ConstructFromAnother(MR.FillingSurface.TPMS.VolumeParams._Underlying *_other);
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_VolumeParams_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::FillingSurface::TPMS::VolumeParams::operator=`.
                public unsafe MR.FillingSurface.TPMS.VolumeParams Assign(MR.FillingSurface.TPMS.Const_VolumeParams _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_VolumeParams_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_VolumeParams_AssignFromAnother(_Underlying *_this, MR.FillingSurface.TPMS.VolumeParams._Underlying *_other);
                    return new(__MR_FillingSurface_TPMS_VolumeParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `VolumeParams` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_VolumeParams`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `VolumeParams`/`Const_VolumeParams` directly.
            public class _InOptMut_VolumeParams
            {
                public VolumeParams? Opt;

                public _InOptMut_VolumeParams() {}
                public _InOptMut_VolumeParams(VolumeParams value) {Opt = value;}
                public static implicit operator _InOptMut_VolumeParams(VolumeParams value) {return new(value);}
            }

            /// This is used for optional parameters of class `VolumeParams` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_VolumeParams`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `VolumeParams`/`Const_VolumeParams` to pass it to the function.
            public class _InOptConst_VolumeParams
            {
                public Const_VolumeParams? Opt;

                public _InOptConst_VolumeParams() {}
                public _InOptConst_VolumeParams(Const_VolumeParams value) {Opt = value;}
                public static implicit operator _InOptConst_VolumeParams(Const_VolumeParams value) {return new(value);}
            }

            /// Generated from class `MR::FillingSurface::TPMS::MeshParams`.
            /// Base classes:
            ///   Direct: (non-virtual)
            ///     `MR::FillingSurface::TPMS::VolumeParams`
            /// This is the const half of the class.
            public class Const_MeshParams : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_MeshParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_Destroy", ExactSpelling = true)]
                    extern static void __MR_FillingSurface_TPMS_MeshParams_Destroy(_Underlying *_this);
                    __MR_FillingSurface_TPMS_MeshParams_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_MeshParams() {Dispose(false);}

                // Upcasts:
                public static unsafe implicit operator MR.FillingSurface.TPMS.Const_VolumeParams(Const_MeshParams self)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_UpcastTo_MR_FillingSurface_TPMS_VolumeParams", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.Const_VolumeParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_UpcastTo_MR_FillingSurface_TPMS_VolumeParams(_Underlying *_this);
                    MR.FillingSurface.TPMS.Const_VolumeParams ret = new(__MR_FillingSurface_TPMS_MeshParams_UpcastTo_MR_FillingSurface_TPMS_VolumeParams(self._UnderlyingPtr), is_owning: false);
                    ret._KeepAlive(self);
                    return ret;
                }

                public unsafe float Iso
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_Get_iso", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_MeshParams_Get_iso(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_MeshParams_Get_iso(_UnderlyingPtr);
                    }
                }

                public unsafe bool Decimate
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_Get_decimate", ExactSpelling = true)]
                        extern static bool *__MR_FillingSurface_TPMS_MeshParams_Get_decimate(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_MeshParams_Get_decimate(_UnderlyingPtr);
                    }
                }

                // Type of the surface
                public unsafe MR.FillingSurface.TPMS.Type Type
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_Get_type", ExactSpelling = true)]
                        extern static MR.FillingSurface.TPMS.Type *__MR_FillingSurface_TPMS_MeshParams_Get_type(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_MeshParams_Get_type(_UnderlyingPtr);
                    }
                }

                // Frequency of oscillations (determines size of the "cells" in the "grid")
                public unsafe float Frequency
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_Get_frequency", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_MeshParams_Get_frequency(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_MeshParams_Get_frequency(_UnderlyingPtr);
                    }
                }

                // Ratio `n / T`, between the number of voxels and period of oscillations
                public unsafe float Resolution
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_Get_resolution", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_MeshParams_Get_resolution(_Underlying *_this);
                        return *__MR_FillingSurface_TPMS_MeshParams_Get_resolution(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_MeshParams() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.MeshParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_DefaultConstruct();
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_MeshParams_DefaultConstruct();
                }

                /// Generated from constructor `MR::FillingSurface::TPMS::MeshParams::MeshParams`.
                public unsafe Const_MeshParams(MR.FillingSurface.TPMS.Const_MeshParams _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.MeshParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_ConstructFromAnother(MR.FillingSurface.TPMS.MeshParams._Underlying *_other);
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_MeshParams_ConstructFromAnother(_other._UnderlyingPtr);
                }
            }

            /// Generated from class `MR::FillingSurface::TPMS::MeshParams`.
            /// Base classes:
            ///   Direct: (non-virtual)
            ///     `MR::FillingSurface::TPMS::VolumeParams`
            /// This is the non-const half of the class.
            public class MeshParams : Const_MeshParams
            {
                internal unsafe MeshParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                // Upcasts:
                public static unsafe implicit operator MR.FillingSurface.TPMS.VolumeParams(MeshParams self)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_UpcastTo_MR_FillingSurface_TPMS_VolumeParams", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.VolumeParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_UpcastTo_MR_FillingSurface_TPMS_VolumeParams(_Underlying *_this);
                    MR.FillingSurface.TPMS.VolumeParams ret = new(__MR_FillingSurface_TPMS_MeshParams_UpcastTo_MR_FillingSurface_TPMS_VolumeParams(self._UnderlyingPtr), is_owning: false);
                    ret._KeepAlive(self);
                    return ret;
                }

                public new unsafe ref float Iso
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_GetMutable_iso", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_MeshParams_GetMutable_iso(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_MeshParams_GetMutable_iso(_UnderlyingPtr);
                    }
                }

                public new unsafe ref bool Decimate
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_GetMutable_decimate", ExactSpelling = true)]
                        extern static bool *__MR_FillingSurface_TPMS_MeshParams_GetMutable_decimate(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_MeshParams_GetMutable_decimate(_UnderlyingPtr);
                    }
                }

                // Type of the surface
                public new unsafe ref MR.FillingSurface.TPMS.Type Type
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_GetMutable_type", ExactSpelling = true)]
                        extern static MR.FillingSurface.TPMS.Type *__MR_FillingSurface_TPMS_MeshParams_GetMutable_type(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_MeshParams_GetMutable_type(_UnderlyingPtr);
                    }
                }

                // Frequency of oscillations (determines size of the "cells" in the "grid")
                public new unsafe ref float Frequency
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_GetMutable_frequency", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_MeshParams_GetMutable_frequency(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_MeshParams_GetMutable_frequency(_UnderlyingPtr);
                    }
                }

                // Ratio `n / T`, between the number of voxels and period of oscillations
                public new unsafe ref float Resolution
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_GetMutable_resolution", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_TPMS_MeshParams_GetMutable_resolution(_Underlying *_this);
                        return ref *__MR_FillingSurface_TPMS_MeshParams_GetMutable_resolution(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe MeshParams() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.MeshParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_DefaultConstruct();
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_MeshParams_DefaultConstruct();
                }

                /// Generated from constructor `MR::FillingSurface::TPMS::MeshParams::MeshParams`.
                public unsafe MeshParams(MR.FillingSurface.TPMS.Const_MeshParams _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.MeshParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_ConstructFromAnother(MR.FillingSurface.TPMS.MeshParams._Underlying *_other);
                    _UnderlyingPtr = __MR_FillingSurface_TPMS_MeshParams_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::FillingSurface::TPMS::MeshParams::operator=`.
                public unsafe MR.FillingSurface.TPMS.MeshParams Assign(MR.FillingSurface.TPMS.Const_MeshParams _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_MeshParams_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.TPMS.MeshParams._Underlying *__MR_FillingSurface_TPMS_MeshParams_AssignFromAnother(_Underlying *_this, MR.FillingSurface.TPMS.MeshParams._Underlying *_other);
                    return new(__MR_FillingSurface_TPMS_MeshParams_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `MeshParams` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshParams`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `MeshParams`/`Const_MeshParams` directly.
            public class _InOptMut_MeshParams
            {
                public MeshParams? Opt;

                public _InOptMut_MeshParams() {}
                public _InOptMut_MeshParams(MeshParams value) {Opt = value;}
                public static implicit operator _InOptMut_MeshParams(MeshParams value) {return new(value);}
            }

            /// This is used for optional parameters of class `MeshParams` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshParams`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `MeshParams`/`Const_MeshParams` to pass it to the function.
            public class _InOptConst_MeshParams
            {
                public Const_MeshParams? Opt;

                public _InOptConst_MeshParams() {}
                public _InOptConst_MeshParams(Const_MeshParams value) {Opt = value;}
                public static implicit operator _InOptConst_MeshParams(Const_MeshParams value) {return new(value);}
            }
        }

        public static partial class CellularSurface
        {
            /// Type of cellular surface base element
            public enum Type : int
            {
                Cylinder = 0,
                Rect = 1,
            }

            /// Generated from class `MR::FillingSurface::CellularSurface::Params`.
            /// This is the const half of the class.
            public class Const_Params : MR.Misc.Object, System.IDisposable
            {
                internal struct _Underlying; // Represents the underlying C++ type.

                internal unsafe _Underlying *_UnderlyingPtr;

                internal unsafe Const_Params(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

                protected virtual unsafe void Dispose(bool disposing)
                {
                    if (_UnderlyingPtr is null || !_IsOwningVal)
                        return;
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Destroy", ExactSpelling = true)]
                    extern static void __MR_FillingSurface_CellularSurface_Params_Destroy(_Underlying *_this);
                    __MR_FillingSurface_CellularSurface_Params_Destroy(_UnderlyingPtr);
                    _UnderlyingPtr = null;
                }
                public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
                ~Const_Params() {Dispose(false);}

                ///< the type of the base element
                public unsafe MR.FillingSurface.CellularSurface.Type Type
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Get_type", ExactSpelling = true)]
                        extern static MR.FillingSurface.CellularSurface.Type *__MR_FillingSurface_CellularSurface_Params_Get_type(_Underlying *_this);
                        return *__MR_FillingSurface_CellularSurface_Params_Get_type(_UnderlyingPtr);
                    }
                }

                ///< the distance between consecutive cylinders in each direction
                public unsafe MR.Const_Vector3f Period
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Get_period", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_FillingSurface_CellularSurface_Params_Get_period(_Underlying *_this);
                        return new(__MR_FillingSurface_CellularSurface_Params_Get_period(_UnderlyingPtr), is_owning: false);
                    }
                }

                ///< the width of cylinders in each direction
                public unsafe MR.Const_Vector3f Width
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Get_width", ExactSpelling = true)]
                        extern static MR.Const_Vector3f._Underlying *__MR_FillingSurface_CellularSurface_Params_Get_width(_Underlying *_this);
                        return new(__MR_FillingSurface_CellularSurface_Params_Get_width(_UnderlyingPtr), is_owning: false);
                    }
                }

                ///< the radius of uniting spheres
                public unsafe float R
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Get_r", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_CellularSurface_Params_Get_r(_Underlying *_this);
                        return *__MR_FillingSurface_CellularSurface_Params_Get_r(_UnderlyingPtr);
                    }
                }

                // used in tests in order to make surfaces close to their analytical expression
                // recommended to be false for real usage for better performance
                public unsafe bool HighRes
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Get_highRes", ExactSpelling = true)]
                        extern static bool *__MR_FillingSurface_CellularSurface_Params_Get_highRes(_Underlying *_this);
                        return *__MR_FillingSurface_CellularSurface_Params_Get_highRes(_UnderlyingPtr);
                    }
                }

                // Used in tests for roughly the same purpose: the computations of density estimation are made under the assumption of an infinite surface.
                // Thus, we must impose "boundary conditions" that inflict the "tips" of the bars (cylinders or cubes) to be preserved on the boundary of the
                // generated filling surface. However, for the aesthetic reasons, it was requested that the tips must be cut in the UI. And here comes this flag.
                // Note that for the estimation of density in UI the influence of "tips" is not significant (it tends to zero with growing size), however
                // we cannot afford to run tests on too big surfaces as it takes too long.
                public unsafe bool PreserveTips
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_Get_preserveTips", ExactSpelling = true)]
                        extern static bool *__MR_FillingSurface_CellularSurface_Params_Get_preserveTips(_Underlying *_this);
                        return *__MR_FillingSurface_CellularSurface_Params_Get_preserveTips(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Const_Params() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_DefaultConstruct();
                    _UnderlyingPtr = __MR_FillingSurface_CellularSurface_Params_DefaultConstruct();
                }

                /// Constructs `MR::FillingSurface::CellularSurface::Params` elementwise.
                public unsafe Const_Params(MR.FillingSurface.CellularSurface.Type type, MR.Vector3f period, MR.Vector3f width, float r, bool highRes, bool preserveTips) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_ConstructFrom", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_ConstructFrom(MR.FillingSurface.CellularSurface.Type type, MR.Vector3f period, MR.Vector3f width, float r, byte highRes, byte preserveTips);
                    _UnderlyingPtr = __MR_FillingSurface_CellularSurface_Params_ConstructFrom(type, period, width, r, highRes ? (byte)1 : (byte)0, preserveTips ? (byte)1 : (byte)0);
                }

                /// Generated from constructor `MR::FillingSurface::CellularSurface::Params::Params`.
                public unsafe Const_Params(MR.FillingSurface.CellularSurface.Const_Params _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_ConstructFromAnother(MR.FillingSurface.CellularSurface.Params._Underlying *_other);
                    _UnderlyingPtr = __MR_FillingSurface_CellularSurface_Params_ConstructFromAnother(_other._UnderlyingPtr);
                }
            }

            /// Generated from class `MR::FillingSurface::CellularSurface::Params`.
            /// This is the non-const half of the class.
            public class Params : Const_Params
            {
                internal unsafe Params(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

                ///< the type of the base element
                public new unsafe ref MR.FillingSurface.CellularSurface.Type Type
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_GetMutable_type", ExactSpelling = true)]
                        extern static MR.FillingSurface.CellularSurface.Type *__MR_FillingSurface_CellularSurface_Params_GetMutable_type(_Underlying *_this);
                        return ref *__MR_FillingSurface_CellularSurface_Params_GetMutable_type(_UnderlyingPtr);
                    }
                }

                ///< the distance between consecutive cylinders in each direction
                public new unsafe MR.Mut_Vector3f Period
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_GetMutable_period", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_FillingSurface_CellularSurface_Params_GetMutable_period(_Underlying *_this);
                        return new(__MR_FillingSurface_CellularSurface_Params_GetMutable_period(_UnderlyingPtr), is_owning: false);
                    }
                }

                ///< the width of cylinders in each direction
                public new unsafe MR.Mut_Vector3f Width
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_GetMutable_width", ExactSpelling = true)]
                        extern static MR.Mut_Vector3f._Underlying *__MR_FillingSurface_CellularSurface_Params_GetMutable_width(_Underlying *_this);
                        return new(__MR_FillingSurface_CellularSurface_Params_GetMutable_width(_UnderlyingPtr), is_owning: false);
                    }
                }

                ///< the radius of uniting spheres
                public new unsafe ref float R
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_GetMutable_r", ExactSpelling = true)]
                        extern static float *__MR_FillingSurface_CellularSurface_Params_GetMutable_r(_Underlying *_this);
                        return ref *__MR_FillingSurface_CellularSurface_Params_GetMutable_r(_UnderlyingPtr);
                    }
                }

                // used in tests in order to make surfaces close to their analytical expression
                // recommended to be false for real usage for better performance
                public new unsafe ref bool HighRes
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_GetMutable_highRes", ExactSpelling = true)]
                        extern static bool *__MR_FillingSurface_CellularSurface_Params_GetMutable_highRes(_Underlying *_this);
                        return ref *__MR_FillingSurface_CellularSurface_Params_GetMutable_highRes(_UnderlyingPtr);
                    }
                }

                // Used in tests for roughly the same purpose: the computations of density estimation are made under the assumption of an infinite surface.
                // Thus, we must impose "boundary conditions" that inflict the "tips" of the bars (cylinders or cubes) to be preserved on the boundary of the
                // generated filling surface. However, for the aesthetic reasons, it was requested that the tips must be cut in the UI. And here comes this flag.
                // Note that for the estimation of density in UI the influence of "tips" is not significant (it tends to zero with growing size), however
                // we cannot afford to run tests on too big surfaces as it takes too long.
                public new unsafe ref bool PreserveTips
                {
                    get
                    {
                        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_GetMutable_preserveTips", ExactSpelling = true)]
                        extern static bool *__MR_FillingSurface_CellularSurface_Params_GetMutable_preserveTips(_Underlying *_this);
                        return ref *__MR_FillingSurface_CellularSurface_Params_GetMutable_preserveTips(_UnderlyingPtr);
                    }
                }

                /// Constructs an empty (default-constructed) instance.
                public unsafe Params() : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_DefaultConstruct", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_DefaultConstruct();
                    _UnderlyingPtr = __MR_FillingSurface_CellularSurface_Params_DefaultConstruct();
                }

                /// Constructs `MR::FillingSurface::CellularSurface::Params` elementwise.
                public unsafe Params(MR.FillingSurface.CellularSurface.Type type, MR.Vector3f period, MR.Vector3f width, float r, bool highRes, bool preserveTips) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_ConstructFrom", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_ConstructFrom(MR.FillingSurface.CellularSurface.Type type, MR.Vector3f period, MR.Vector3f width, float r, byte highRes, byte preserveTips);
                    _UnderlyingPtr = __MR_FillingSurface_CellularSurface_Params_ConstructFrom(type, period, width, r, highRes ? (byte)1 : (byte)0, preserveTips ? (byte)1 : (byte)0);
                }

                /// Generated from constructor `MR::FillingSurface::CellularSurface::Params::Params`.
                public unsafe Params(MR.FillingSurface.CellularSurface.Const_Params _other) : this(null, is_owning: true)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_ConstructFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_ConstructFromAnother(MR.FillingSurface.CellularSurface.Params._Underlying *_other);
                    _UnderlyingPtr = __MR_FillingSurface_CellularSurface_Params_ConstructFromAnother(_other._UnderlyingPtr);
                }

                /// Generated from method `MR::FillingSurface::CellularSurface::Params::operator=`.
                public unsafe MR.FillingSurface.CellularSurface.Params Assign(MR.FillingSurface.CellularSurface.Const_Params _other)
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_Params_AssignFromAnother", ExactSpelling = true)]
                    extern static MR.FillingSurface.CellularSurface.Params._Underlying *__MR_FillingSurface_CellularSurface_Params_AssignFromAnother(_Underlying *_this, MR.FillingSurface.CellularSurface.Params._Underlying *_other);
                    return new(__MR_FillingSurface_CellularSurface_Params_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
                }
            }

            /// This is used for optional parameters of class `Params` with default arguments.
            /// This is only used mutable parameters. For const ones we have `_InOptConst_Params`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Params`/`Const_Params` directly.
            public class _InOptMut_Params
            {
                public Params? Opt;

                public _InOptMut_Params() {}
                public _InOptMut_Params(Params value) {Opt = value;}
                public static implicit operator _InOptMut_Params(Params value) {return new(value);}
            }

            /// This is used for optional parameters of class `Params` with default arguments.
            /// This is only used const parameters. For non-const ones we have `_InOptMut_Params`.
            /// Usage:
            /// * Pass `null` to use the default argument.
            /// * Pass `new()` to pass no object.
            /// * Pass an instance of `Params`/`Const_Params` to pass it to the function.
            public class _InOptConst_Params
            {
                public Const_Params? Opt;

                public _InOptConst_Params() {}
                public _InOptConst_Params(Const_Params value) {Opt = value;}
                public static implicit operator _InOptConst_Params(Const_Params value) {return new(value);}
            }
        }

        // Different kinds of filling surface
        public enum Kind : int
        {
            TPMS = 0,
            Cellular = 1,
        }

        public static partial class TPMS
        {
            /// Returns the names for each type of filling
            /// Generated from function `MR::FillingSurface::TPMS::getTypeNames`.
            public static unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetTypeNames()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_getTypeNames", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_FillingSurface_TPMS_getTypeNames();
                return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_FillingSurface_TPMS_getTypeNames(), is_owning: true));
            }

            /// Returns the tooltips for each type of filling
            /// Generated from function `MR::FillingSurface::TPMS::getTypeTooltips`.
            public static unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetTypeTooltips()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_getTypeTooltips", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_FillingSurface_TPMS_getTypeTooltips();
                return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_FillingSurface_TPMS_getTypeTooltips(), is_owning: true));
            }

            /// Returns true if the \p type is thick
            /// Generated from function `MR::FillingSurface::TPMS::isThick`.
            public static bool IsThick(MR.FillingSurface.TPMS.Type type)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_isThick", ExactSpelling = true)]
                extern static byte __MR_FillingSurface_TPMS_isThick(MR.FillingSurface.TPMS.Type type);
                return __MR_FillingSurface_TPMS_isThick(type) != 0;
            }

            /// Construct TPMS using implicit function (https://www.researchgate.net/publication/350658078_Computational_method_and_program_for_generating_a_porous_scaffold_based_on_implicit_surfaces)
            /// @param size Size of the cube with the surface
            /// @return Distance-volume starting at (0, 0, 0) and having specified @p size
            /// Generated from function `MR::FillingSurface::TPMS::buildVolume`.
            public static unsafe MR.Misc._Moved<MR.FunctionVolume> BuildVolume(MR.Const_Vector3f size, MR.FillingSurface.TPMS.Const_VolumeParams params_)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_buildVolume", ExactSpelling = true)]
                extern static MR.FunctionVolume._Underlying *__MR_FillingSurface_TPMS_buildVolume(MR.Const_Vector3f._Underlying *size, MR.FillingSurface.TPMS.Const_VolumeParams._Underlying *params_);
                return MR.Misc.Move(new MR.FunctionVolume(__MR_FillingSurface_TPMS_buildVolume(size._UnderlyingPtr, params_._UnderlyingPtr), is_owning: true));
            }

            /// Constructs TPMS level-set and then convert it to mesh
            /// Generated from function `MR::FillingSurface::TPMS::build`.
            /// Parameter `cb` defaults to `{}`.
            public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> Build(MR.Const_Vector3f size, MR.FillingSurface.TPMS.Const_MeshParams params_, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_build", ExactSpelling = true)]
                extern static MR.Expected_MRMesh_StdString._Underlying *__MR_FillingSurface_TPMS_build(MR.Const_Vector3f._Underlying *size, MR.FillingSurface.TPMS.Const_MeshParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_FillingSurface_TPMS_build(size._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
            }

            /// Constructs TPMS-filling for the given @p mesh
            /// Generated from function `MR::FillingSurface::TPMS::fill`.
            /// Parameter `cb` defaults to `{}`.
            public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> Fill(MR.Const_Mesh mesh, MR.FillingSurface.TPMS.Const_MeshParams params_, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_fill", ExactSpelling = true)]
                extern static MR.Expected_MRMesh_StdString._Underlying *__MR_FillingSurface_TPMS_fill(MR.Const_Mesh._Underlying *mesh, MR.FillingSurface.TPMS.Const_MeshParams._Underlying *params_, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_FillingSurface_TPMS_fill(mesh._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
            }

            /// Returns number of voxels that would be used to perform \ref fillWithTPMS
            /// Generated from function `MR::FillingSurface::TPMS::getNumberOfVoxels`.
            public static unsafe ulong GetNumberOfVoxels(MR.Const_Mesh mesh, float frequency, float resolution)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_getNumberOfVoxels_MR_Mesh", ExactSpelling = true)]
                extern static ulong __MR_FillingSurface_TPMS_getNumberOfVoxels_MR_Mesh(MR.Const_Mesh._Underlying *mesh, float frequency, float resolution);
                return __MR_FillingSurface_TPMS_getNumberOfVoxels_MR_Mesh(mesh._UnderlyingPtr, frequency, resolution);
            }

            /// Returns number of voxels that would be used to perform \ref buildTPMS or \ref buildTPMSVolume
            /// Generated from function `MR::FillingSurface::TPMS::getNumberOfVoxels`.
            public static unsafe ulong GetNumberOfVoxels(MR.Const_Vector3f size, float frequency, float resolution)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_getNumberOfVoxels_MR_Vector3f", ExactSpelling = true)]
                extern static ulong __MR_FillingSurface_TPMS_getNumberOfVoxels_MR_Vector3f(MR.Const_Vector3f._Underlying *size, float frequency, float resolution);
                return __MR_FillingSurface_TPMS_getNumberOfVoxels_MR_Vector3f(size._UnderlyingPtr, frequency, resolution);
            }

            /// Returns approximated ISO value corresponding to the given density
            /// @param targetDensity value in [0; 1]
            /// @return Value in [-1; 1]
            /// Generated from function `MR::FillingSurface::TPMS::estimateIso`.
            public static float EstimateIso(MR.FillingSurface.TPMS.Type type, float targetDensity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_estimateIso", ExactSpelling = true)]
                extern static float __MR_FillingSurface_TPMS_estimateIso(MR.FillingSurface.TPMS.Type type, float targetDensity);
                return __MR_FillingSurface_TPMS_estimateIso(type, targetDensity);
            }

            /// Returns approximate density corresponding to the given ISO value
            /// @param targetIso value in [-1; 1]
            /// @return Value in [0; 1]
            /// Generated from function `MR::FillingSurface::TPMS::estimateDensity`.
            public static float EstimateDensity(MR.FillingSurface.TPMS.Type type, float targetIso)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_estimateDensity", ExactSpelling = true)]
                extern static float __MR_FillingSurface_TPMS_estimateDensity(MR.FillingSurface.TPMS.Type type, float targetIso);
                return __MR_FillingSurface_TPMS_estimateDensity(type, targetIso);
            }

            /// Returns minimal reasonable resolution for given parameters
            /// Generated from function `MR::FillingSurface::TPMS::getMinimalResolution`.
            public static float GetMinimalResolution(MR.FillingSurface.TPMS.Type type, float iso)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_TPMS_getMinimalResolution", ExactSpelling = true)]
                extern static float __MR_FillingSurface_TPMS_getMinimalResolution(MR.FillingSurface.TPMS.Type type, float iso);
                return __MR_FillingSurface_TPMS_getMinimalResolution(type, iso);
            }
        }

        public static partial class CellularSurface
        {
            /// Returns the names for each type of filling
            /// Generated from function `MR::FillingSurface::CellularSurface::getTypeNames`.
            public static unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetTypeNames()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_getTypeNames", ExactSpelling = true)]
                extern static MR.Std.Vector_StdString._Underlying *__MR_FillingSurface_CellularSurface_getTypeNames();
                return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_FillingSurface_CellularSurface_getTypeNames(), is_owning: true));
            }

            /// Build a cellular surface of size \p size
            /// Generated from function `MR::FillingSurface::CellularSurface::build`.
            /// Parameter `cb` defaults to `{}`.
            public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> Build(MR.Const_Vector3f size, MR.FillingSurface.CellularSurface.Const_Params params_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_build", ExactSpelling = true)]
                extern static MR.Expected_MRMesh_StdString._Underlying *__MR_FillingSurface_CellularSurface_build(MR.Const_Vector3f._Underlying *size, MR.FillingSurface.CellularSurface.Const_Params._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_FillingSurface_CellularSurface_build(size._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }

            /// Fill given mesh with a cellular surface
            /// Generated from function `MR::FillingSurface::CellularSurface::fill`.
            /// Parameter `cb` defaults to `{}`.
            public static unsafe MR.Misc._Moved<MR.Expected_MRMesh_StdString> Fill(MR.Const_Mesh mesh, MR.FillingSurface.CellularSurface.Const_Params params_, MR.Std.Const_Function_BoolFuncFromFloat? cb = null)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_fill", ExactSpelling = true)]
                extern static MR.Expected_MRMesh_StdString._Underlying *__MR_FillingSurface_CellularSurface_fill(MR.Const_Mesh._Underlying *mesh, MR.FillingSurface.CellularSurface.Const_Params._Underlying *params_, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *cb);
                return MR.Misc.Move(new MR.Expected_MRMesh_StdString(__MR_FillingSurface_CellularSurface_fill(mesh._UnderlyingPtr, params_._UnderlyingPtr, cb is not null ? cb._UnderlyingPtr : null), is_owning: true));
            }

            /// Estimate the density of the cellular surface
            /// Generated from function `MR::FillingSurface::CellularSurface::estimateDensity`.
            public static float EstimateDensity(float period, float width, float r)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_estimateDensity", ExactSpelling = true)]
                extern static float __MR_FillingSurface_CellularSurface_estimateDensity(float period, float width, float r);
                return __MR_FillingSurface_CellularSurface_estimateDensity(period, width, r);
            }

            /// Estimate the width that is needed to attain the \p targetDensity. Inverse of \ref estimateDensity.
            /// \note The width is not unique in general, no guarantees are made about which value among possible will be returned.
            //    Due to the simplification of the formula (sphere must either fully contain the intersection of cylinders or be inside it), solution not always exists.
            /// Generated from function `MR::FillingSurface::CellularSurface::estimateWidth`.
            public static unsafe MR.Std.Optional_Float EstimateWidth(float period, float r, float targetDensity)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_CellularSurface_estimateWidth", ExactSpelling = true)]
                extern static MR.Std.Optional_Float._Underlying *__MR_FillingSurface_CellularSurface_estimateWidth(float period, float r, float targetDensity);
                return new(__MR_FillingSurface_CellularSurface_estimateWidth(period, r, targetDensity), is_owning: true);
            }
        }

        /// Generated from function `MR::FillingSurface::getKindNames`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_StdString> GetKindNames()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FillingSurface_getKindNames", ExactSpelling = true)]
            extern static MR.Std.Vector_StdString._Underlying *__MR_FillingSurface_getKindNames();
            return MR.Misc.Move(new MR.Std.Vector_StdString(__MR_FillingSurface_getKindNames(), is_owning: true));
        }
    }
}
