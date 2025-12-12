public static partial class MR
{
    // A stub measurement unit representing no unit.
    public enum NoUnit : int
    {
        Count = 0,
    }

    // Measurement units of length.
    public enum LengthUnit : int
    {
        Microns = 0,
        Millimeters = 1,
        Centimeters = 2,
        Meters = 3,
        Inches = 4,
        Feet = 5,
        Count = 6,
    }

    // Measurement units of angle.
    public enum AngleUnit : int
    {
        Radians = 0,
        Degrees = 1,
        Count = 2,
    }

    // Measurement units of screen sizes.
    public enum PixelSizeUnit : int
    {
        Pixels = 0,
        Count = 1,
    }

    // Measurement units for factors / ratios.
    public enum RatioUnit : int
    {
        // 0..1 x
        Factor = 0,
        // 0..100 %
        Percents = 1,
        Count = 2,
    }

    // Measurement units for time.
    public enum TimeUnit : int
    {
        Seconds = 0,
        Milliseconds = 1,
        Count = 2,
    }

    // Measurement units for movement speed.
    public enum MovementSpeedUnit : int
    {
        MicronsPerSecond = 0,
        MillimetersPerSecond = 1,
        CentimetersPerSecond = 2,
        MetersPerSecond = 3,
        InchesPerSecond = 4,
        FeetPerSecond = 5,
        Count = 6,
    }

    // Measurement units for surface area.
    public enum AreaUnit : int
    {
        Microns2 = 0,
        Millimeters2 = 1,
        Centimeters2 = 2,
        Meters2 = 3,
        Inches2 = 4,
        Feet2 = 5,
        Count = 6,
    }

    // Measurement units for body volume.
    public enum VolumeUnit : int
    {
        Microns3 = 0,
        Millimeters3 = 1,
        Centimeters3 = 2,
        Meters3 = 3,
        Inches3 = 4,
        Feet3 = 5,
        Count = 6,
    }

    // Measurement units for 1/length.
    public enum InvLengthUnit : int
    {
        InvMicrons = 0,
        InvMillimeters = 1,
        InvCentimeters = 2,
        InvMeters = 3,
        InvInches = 4,
        InvFeet = 5,
        Count = 6,
    }

    // Information about a single measurement unit.
    /// Generated from class `MR::UnitInfo`.
    /// This is the const half of the class.
    public class Const_UnitInfo : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UnitInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_Destroy", ExactSpelling = true)]
            extern static void __MR_UnitInfo_Destroy(_Underlying *_this);
            __MR_UnitInfo_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UnitInfo() {Dispose(false);}

        // This is used to convert between units.
        // To convert from A to B, multiply by A's factor and divide by B's.
        public unsafe float ConversionFactor
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_Get_conversionFactor", ExactSpelling = true)]
                extern static float *__MR_UnitInfo_Get_conversionFactor(_Underlying *_this);
                return *__MR_UnitInfo_Get_conversionFactor(_UnderlyingPtr);
            }
        }

        public unsafe MR.Std.Const_StringView PrettyName
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_Get_prettyName", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_UnitInfo_Get_prettyName(_Underlying *_this);
                return new(__MR_UnitInfo_Get_prettyName(_UnderlyingPtr), is_owning: false);
            }
        }

        // The short unit name that's placed after values.
        // This may or may not start with a space.
        public unsafe MR.Std.Const_StringView UnitSuffix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_Get_unitSuffix", ExactSpelling = true)]
                extern static MR.Std.Const_StringView._Underlying *__MR_UnitInfo_Get_unitSuffix(_Underlying *_this);
                return new(__MR_UnitInfo_Get_unitSuffix(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UnitInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_UnitInfo_DefaultConstruct();
        }

        /// Constructs `MR::UnitInfo` elementwise.
        public unsafe Const_UnitInfo(float conversionFactor, ReadOnlySpan<char> prettyName, ReadOnlySpan<char> unitSuffix) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_ConstructFrom", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_ConstructFrom(float conversionFactor, byte *prettyName, byte *prettyName_end, byte *unitSuffix, byte *unitSuffix_end);
            byte[] __bytes_prettyName = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(prettyName.Length)];
            int __len_prettyName = System.Text.Encoding.UTF8.GetBytes(prettyName, __bytes_prettyName);
            fixed (byte *__ptr_prettyName = __bytes_prettyName)
            {
                byte[] __bytes_unitSuffix = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(unitSuffix.Length)];
                int __len_unitSuffix = System.Text.Encoding.UTF8.GetBytes(unitSuffix, __bytes_unitSuffix);
                fixed (byte *__ptr_unitSuffix = __bytes_unitSuffix)
                {
                    _UnderlyingPtr = __MR_UnitInfo_ConstructFrom(conversionFactor, __ptr_prettyName, __ptr_prettyName + __len_prettyName, __ptr_unitSuffix, __ptr_unitSuffix + __len_unitSuffix);
                }
            }
        }

        /// Generated from constructor `MR::UnitInfo::UnitInfo`.
        public unsafe Const_UnitInfo(MR.Const_UnitInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_ConstructFromAnother(MR.UnitInfo._Underlying *_other);
            _UnderlyingPtr = __MR_UnitInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    // Information about a single measurement unit.
    /// Generated from class `MR::UnitInfo`.
    /// This is the non-const half of the class.
    public class UnitInfo : Const_UnitInfo
    {
        internal unsafe UnitInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // This is used to convert between units.
        // To convert from A to B, multiply by A's factor and divide by B's.
        public new unsafe ref float ConversionFactor
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_GetMutable_conversionFactor", ExactSpelling = true)]
                extern static float *__MR_UnitInfo_GetMutable_conversionFactor(_Underlying *_this);
                return ref *__MR_UnitInfo_GetMutable_conversionFactor(_UnderlyingPtr);
            }
        }

        public new unsafe MR.Std.StringView PrettyName
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_GetMutable_prettyName", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_UnitInfo_GetMutable_prettyName(_Underlying *_this);
                return new(__MR_UnitInfo_GetMutable_prettyName(_UnderlyingPtr), is_owning: false);
            }
        }

        // The short unit name that's placed after values.
        // This may or may not start with a space.
        public new unsafe MR.Std.StringView UnitSuffix
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_GetMutable_unitSuffix", ExactSpelling = true)]
                extern static MR.Std.StringView._Underlying *__MR_UnitInfo_GetMutable_unitSuffix(_Underlying *_this);
                return new(__MR_UnitInfo_GetMutable_unitSuffix(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe UnitInfo() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_DefaultConstruct();
            _UnderlyingPtr = __MR_UnitInfo_DefaultConstruct();
        }

        /// Constructs `MR::UnitInfo` elementwise.
        public unsafe UnitInfo(float conversionFactor, ReadOnlySpan<char> prettyName, ReadOnlySpan<char> unitSuffix) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_ConstructFrom", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_ConstructFrom(float conversionFactor, byte *prettyName, byte *prettyName_end, byte *unitSuffix, byte *unitSuffix_end);
            byte[] __bytes_prettyName = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(prettyName.Length)];
            int __len_prettyName = System.Text.Encoding.UTF8.GetBytes(prettyName, __bytes_prettyName);
            fixed (byte *__ptr_prettyName = __bytes_prettyName)
            {
                byte[] __bytes_unitSuffix = new byte[System.Text.Encoding.UTF8.GetMaxByteCount(unitSuffix.Length)];
                int __len_unitSuffix = System.Text.Encoding.UTF8.GetBytes(unitSuffix, __bytes_unitSuffix);
                fixed (byte *__ptr_unitSuffix = __bytes_unitSuffix)
                {
                    _UnderlyingPtr = __MR_UnitInfo_ConstructFrom(conversionFactor, __ptr_prettyName, __ptr_prettyName + __len_prettyName, __ptr_unitSuffix, __ptr_unitSuffix + __len_unitSuffix);
                }
            }
        }

        /// Generated from constructor `MR::UnitInfo::UnitInfo`.
        public unsafe UnitInfo(MR.Const_UnitInfo _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_ConstructFromAnother(MR.UnitInfo._Underlying *_other);
            _UnderlyingPtr = __MR_UnitInfo_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::UnitInfo::operator=`.
        public unsafe MR.UnitInfo Assign(MR.Const_UnitInfo _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UnitInfo_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UnitInfo._Underlying *__MR_UnitInfo_AssignFromAnother(_Underlying *_this, MR.UnitInfo._Underlying *_other);
            return new(__MR_UnitInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `UnitInfo` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UnitInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnitInfo`/`Const_UnitInfo` directly.
    public class _InOptMut_UnitInfo
    {
        public UnitInfo? Opt;

        public _InOptMut_UnitInfo() {}
        public _InOptMut_UnitInfo(UnitInfo value) {Opt = value;}
        public static implicit operator _InOptMut_UnitInfo(UnitInfo value) {return new(value);}
    }

    /// This is used for optional parameters of class `UnitInfo` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UnitInfo`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UnitInfo`/`Const_UnitInfo` to pass it to the function.
    public class _InOptConst_UnitInfo
    {
        public Const_UnitInfo? Opt;

        public _InOptConst_UnitInfo() {}
        public _InOptConst_UnitInfo(Const_UnitInfo value) {Opt = value;}
        public static implicit operator _InOptConst_UnitInfo(Const_UnitInfo value) {return new(value);}
    }
}
