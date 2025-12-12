public static partial class MR
{
    /// end mill cutter specifications
    /// Generated from class `MR::EndMillCutter`.
    /// This is the const half of the class.
    public class Const_EndMillCutter : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EndMillCutter(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_Destroy", ExactSpelling = true)]
            extern static void __MR_EndMillCutter_Destroy(_Underlying *_this);
            __MR_EndMillCutter_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EndMillCutter() {Dispose(false);}

        public unsafe MR.EndMillCutter.Type Type_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_Get_type", ExactSpelling = true)]
                extern static MR.EndMillCutter.Type *__MR_EndMillCutter_Get_type(_Underlying *_this);
                return *__MR_EndMillCutter_Get_type(_UnderlyingPtr);
            }
        }

        /// (bull nose) corner radius
        public unsafe float CornerRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_Get_cornerRadius", ExactSpelling = true)]
                extern static float *__MR_EndMillCutter_Get_cornerRadius(_Underlying *_this);
                return *__MR_EndMillCutter_Get_cornerRadius(_UnderlyingPtr);
            }
        }

        /// (chamfer) cutting angle
        public unsafe float CuttingAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_Get_cuttingAngle", ExactSpelling = true)]
                extern static float *__MR_EndMillCutter_Get_cuttingAngle(_Underlying *_this);
                return *__MR_EndMillCutter_Get_cuttingAngle(_UnderlyingPtr);
            }
        }

        /// (chamfer) end diameter
        public unsafe float EndDiameter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_Get_endDiameter", ExactSpelling = true)]
                extern static float *__MR_EndMillCutter_Get_endDiameter(_Underlying *_this);
                return *__MR_EndMillCutter_Get_endDiameter(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EndMillCutter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_DefaultConstruct();
            _UnderlyingPtr = __MR_EndMillCutter_DefaultConstruct();
        }

        /// Constructs `MR::EndMillCutter` elementwise.
        public unsafe Const_EndMillCutter(MR.EndMillCutter.Type type, float cornerRadius, float cuttingAngle, float endDiameter) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_ConstructFrom", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_ConstructFrom(MR.EndMillCutter.Type type, float cornerRadius, float cuttingAngle, float endDiameter);
            _UnderlyingPtr = __MR_EndMillCutter_ConstructFrom(type, cornerRadius, cuttingAngle, endDiameter);
        }

        /// Generated from constructor `MR::EndMillCutter::EndMillCutter`.
        public unsafe Const_EndMillCutter(MR.Const_EndMillCutter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_ConstructFromAnother(MR.EndMillCutter._Underlying *_other);
            _UnderlyingPtr = __MR_EndMillCutter_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// cutter type
        public enum Type : int
        {
            /// flat end mill
            Flat = 0,
            /// ball end mill
            Ball = 1,
            /// bull nose end mill
            BullNose = 2,
            /// chamfer end mill
            Chamfer = 3,
            /// chamfer end mill
            Count = 4,
        }
    }

    /// end mill cutter specifications
    /// Generated from class `MR::EndMillCutter`.
    /// This is the non-const half of the class.
    public class EndMillCutter : Const_EndMillCutter
    {
        internal unsafe EndMillCutter(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref MR.EndMillCutter.Type Type_
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_GetMutable_type", ExactSpelling = true)]
                extern static MR.EndMillCutter.Type *__MR_EndMillCutter_GetMutable_type(_Underlying *_this);
                return ref *__MR_EndMillCutter_GetMutable_type(_UnderlyingPtr);
            }
        }

        /// (bull nose) corner radius
        public new unsafe ref float CornerRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_GetMutable_cornerRadius", ExactSpelling = true)]
                extern static float *__MR_EndMillCutter_GetMutable_cornerRadius(_Underlying *_this);
                return ref *__MR_EndMillCutter_GetMutable_cornerRadius(_UnderlyingPtr);
            }
        }

        /// (chamfer) cutting angle
        public new unsafe ref float CuttingAngle
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_GetMutable_cuttingAngle", ExactSpelling = true)]
                extern static float *__MR_EndMillCutter_GetMutable_cuttingAngle(_Underlying *_this);
                return ref *__MR_EndMillCutter_GetMutable_cuttingAngle(_UnderlyingPtr);
            }
        }

        /// (chamfer) end diameter
        public new unsafe ref float EndDiameter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_GetMutable_endDiameter", ExactSpelling = true)]
                extern static float *__MR_EndMillCutter_GetMutable_endDiameter(_Underlying *_this);
                return ref *__MR_EndMillCutter_GetMutable_endDiameter(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EndMillCutter() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_DefaultConstruct();
            _UnderlyingPtr = __MR_EndMillCutter_DefaultConstruct();
        }

        /// Constructs `MR::EndMillCutter` elementwise.
        public unsafe EndMillCutter(MR.EndMillCutter.Type type, float cornerRadius, float cuttingAngle, float endDiameter) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_ConstructFrom", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_ConstructFrom(MR.EndMillCutter.Type type, float cornerRadius, float cuttingAngle, float endDiameter);
            _UnderlyingPtr = __MR_EndMillCutter_ConstructFrom(type, cornerRadius, cuttingAngle, endDiameter);
        }

        /// Generated from constructor `MR::EndMillCutter::EndMillCutter`.
        public unsafe EndMillCutter(MR.Const_EndMillCutter _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_ConstructFromAnother(MR.EndMillCutter._Underlying *_other);
            _UnderlyingPtr = __MR_EndMillCutter_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::EndMillCutter::operator=`.
        public unsafe MR.EndMillCutter Assign(MR.Const_EndMillCutter _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillCutter_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EndMillCutter._Underlying *__MR_EndMillCutter_AssignFromAnother(_Underlying *_this, MR.EndMillCutter._Underlying *_other);
            return new(__MR_EndMillCutter_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `EndMillCutter` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EndMillCutter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EndMillCutter`/`Const_EndMillCutter` directly.
    public class _InOptMut_EndMillCutter
    {
        public EndMillCutter? Opt;

        public _InOptMut_EndMillCutter() {}
        public _InOptMut_EndMillCutter(EndMillCutter value) {Opt = value;}
        public static implicit operator _InOptMut_EndMillCutter(EndMillCutter value) {return new(value);}
    }

    /// This is used for optional parameters of class `EndMillCutter` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EndMillCutter`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EndMillCutter`/`Const_EndMillCutter` to pass it to the function.
    public class _InOptConst_EndMillCutter
    {
        public Const_EndMillCutter? Opt;

        public _InOptConst_EndMillCutter() {}
        public _InOptConst_EndMillCutter(Const_EndMillCutter value) {Opt = value;}
        public static implicit operator _InOptConst_EndMillCutter(Const_EndMillCutter value) {return new(value);}
    }

    /// end mill tool specifications
    /// Generated from class `MR::EndMillTool`.
    /// This is the const half of the class.
    public class Const_EndMillTool : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_EndMillTool(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_Destroy", ExactSpelling = true)]
            extern static void __MR_EndMillTool_Destroy(_Underlying *_this);
            __MR_EndMillTool_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_EndMillTool() {Dispose(false);}

        /// overall tool length
        public unsafe float Length
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_Get_length", ExactSpelling = true)]
                extern static float *__MR_EndMillTool_Get_length(_Underlying *_this);
                return *__MR_EndMillTool_Get_length(_UnderlyingPtr);
            }
        }

        /// tool diameter
        public unsafe float Diameter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_Get_diameter", ExactSpelling = true)]
                extern static float *__MR_EndMillTool_Get_diameter(_Underlying *_this);
                return *__MR_EndMillTool_Get_diameter(_UnderlyingPtr);
            }
        }

        /// cutter
        public unsafe MR.Const_EndMillCutter Cutter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_Get_cutter", ExactSpelling = true)]
                extern static MR.Const_EndMillCutter._Underlying *__MR_EndMillTool_Get_cutter(_Underlying *_this);
                return new(__MR_EndMillTool_Get_cutter(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_EndMillTool() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_DefaultConstruct();
            _UnderlyingPtr = __MR_EndMillTool_DefaultConstruct();
        }

        /// Constructs `MR::EndMillTool` elementwise.
        public unsafe Const_EndMillTool(float length, float diameter, MR.Const_EndMillCutter cutter) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_ConstructFrom", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_ConstructFrom(float length, float diameter, MR.EndMillCutter._Underlying *cutter);
            _UnderlyingPtr = __MR_EndMillTool_ConstructFrom(length, diameter, cutter._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EndMillTool::EndMillTool`.
        public unsafe Const_EndMillTool(MR.Const_EndMillTool _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_ConstructFromAnother(MR.EndMillTool._Underlying *_other);
            _UnderlyingPtr = __MR_EndMillTool_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// compute the minimal cut length based on the cutter parameters
        /// Generated from method `MR::EndMillTool::getMinimalCutLength`.
        public unsafe float GetMinimalCutLength()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_getMinimalCutLength", ExactSpelling = true)]
            extern static float __MR_EndMillTool_getMinimalCutLength(_Underlying *_this);
            return __MR_EndMillTool_getMinimalCutLength(_UnderlyingPtr);
        }

        /// create a tool mesh
        /// Generated from method `MR::EndMillTool::toMesh`.
        /// Parameter `horizontalResolution` defaults to `32`.
        /// Parameter `verticalResolution` defaults to `32`.
        public unsafe MR.Misc._Moved<MR.Mesh> ToMesh(int? horizontalResolution = null, int? verticalResolution = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_toMesh", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_EndMillTool_toMesh(_Underlying *_this, int *horizontalResolution, int *verticalResolution);
            int __deref_horizontalResolution = horizontalResolution.GetValueOrDefault();
            int __deref_verticalResolution = verticalResolution.GetValueOrDefault();
            return MR.Misc.Move(new MR.Mesh(__MR_EndMillTool_toMesh(_UnderlyingPtr, horizontalResolution.HasValue ? &__deref_horizontalResolution : null, verticalResolution.HasValue ? &__deref_verticalResolution : null), is_owning: true));
        }
    }

    /// end mill tool specifications
    /// Generated from class `MR::EndMillTool`.
    /// This is the non-const half of the class.
    public class EndMillTool : Const_EndMillTool
    {
        internal unsafe EndMillTool(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// overall tool length
        public new unsafe ref float Length
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_GetMutable_length", ExactSpelling = true)]
                extern static float *__MR_EndMillTool_GetMutable_length(_Underlying *_this);
                return ref *__MR_EndMillTool_GetMutable_length(_UnderlyingPtr);
            }
        }

        /// tool diameter
        public new unsafe ref float Diameter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_GetMutable_diameter", ExactSpelling = true)]
                extern static float *__MR_EndMillTool_GetMutable_diameter(_Underlying *_this);
                return ref *__MR_EndMillTool_GetMutable_diameter(_UnderlyingPtr);
            }
        }

        /// cutter
        public new unsafe MR.EndMillCutter Cutter
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_GetMutable_cutter", ExactSpelling = true)]
                extern static MR.EndMillCutter._Underlying *__MR_EndMillTool_GetMutable_cutter(_Underlying *_this);
                return new(__MR_EndMillTool_GetMutable_cutter(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe EndMillTool() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_DefaultConstruct", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_DefaultConstruct();
            _UnderlyingPtr = __MR_EndMillTool_DefaultConstruct();
        }

        /// Constructs `MR::EndMillTool` elementwise.
        public unsafe EndMillTool(float length, float diameter, MR.Const_EndMillCutter cutter) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_ConstructFrom", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_ConstructFrom(float length, float diameter, MR.EndMillCutter._Underlying *cutter);
            _UnderlyingPtr = __MR_EndMillTool_ConstructFrom(length, diameter, cutter._UnderlyingPtr);
        }

        /// Generated from constructor `MR::EndMillTool::EndMillTool`.
        public unsafe EndMillTool(MR.Const_EndMillTool _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_ConstructFromAnother(MR.EndMillTool._Underlying *_other);
            _UnderlyingPtr = __MR_EndMillTool_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::EndMillTool::operator=`.
        public unsafe MR.EndMillTool Assign(MR.Const_EndMillTool _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_EndMillTool_AssignFromAnother", ExactSpelling = true)]
            extern static MR.EndMillTool._Underlying *__MR_EndMillTool_AssignFromAnother(_Underlying *_this, MR.EndMillTool._Underlying *_other);
            return new(__MR_EndMillTool_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `EndMillTool` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_EndMillTool`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EndMillTool`/`Const_EndMillTool` directly.
    public class _InOptMut_EndMillTool
    {
        public EndMillTool? Opt;

        public _InOptMut_EndMillTool() {}
        public _InOptMut_EndMillTool(EndMillTool value) {Opt = value;}
        public static implicit operator _InOptMut_EndMillTool(EndMillTool value) {return new(value);}
    }

    /// This is used for optional parameters of class `EndMillTool` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_EndMillTool`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `EndMillTool`/`Const_EndMillTool` to pass it to the function.
    public class _InOptConst_EndMillTool
    {
        public Const_EndMillTool? Opt;

        public _InOptConst_EndMillTool() {}
        public _InOptConst_EndMillTool(Const_EndMillTool value) {Opt = value;}
        public static implicit operator _InOptConst_EndMillTool(Const_EndMillTool value) {return new(value);}
    }
}
