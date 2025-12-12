public static partial class MR
{
    /// options determining computation of distance from a point to a mesh
    /// Generated from class `MR::DistanceToMeshOptions`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SignedDistanceToMeshOptions`
    /// This is the const half of the class.
    public class Const_DistanceToMeshOptions : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_DistanceToMeshOptions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_Destroy", ExactSpelling = true)]
            extern static void __MR_DistanceToMeshOptions_Destroy(_Underlying *_this);
            __MR_DistanceToMeshOptions_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_DistanceToMeshOptions() {Dispose(false);}

        /// minimum squared distance from a point to mesh to be computed precisely
        public unsafe float MinDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_Get_minDistSq", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_Get_minDistSq(_Underlying *_this);
                return *__MR_DistanceToMeshOptions_Get_minDistSq(_UnderlyingPtr);
            }
        }

        /// maximum squared distance from a point to mesh to be computed precisely
        public unsafe float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_Get_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_Get_maxDistSq(_Underlying *_this);
                return *__MR_DistanceToMeshOptions_Get_maxDistSq(_UnderlyingPtr);
            }
        }

        /// what to do if actual distance is outside [min, max) range:
        /// true - return std::nullopt for std::optional<float> or NaN for float,
        /// false - return approximate value of the distance (with correct sign in case of SignDetectionMode::HoleWindingRule);
        /// please note that in HoleWindingRule the sign can change even for too small or too large distances,
        /// so if you would like to get closed mesh from marching cubes, set false here
        public unsafe bool NullOutsideMinMax
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_Get_nullOutsideMinMax", ExactSpelling = true)]
                extern static bool *__MR_DistanceToMeshOptions_Get_nullOutsideMinMax(_Underlying *_this);
                return *__MR_DistanceToMeshOptions_Get_nullOutsideMinMax(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_DistanceToMeshOptions_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_DistanceToMeshOptions_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_DistanceToMeshOptions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceToMeshOptions_DefaultConstruct();
        }

        /// Constructs `MR::DistanceToMeshOptions` elementwise.
        public unsafe Const_DistanceToMeshOptions(float minDistSq, float maxDistSq, bool nullOutsideMinMax, float windingNumberThreshold, float windingNumberBeta) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_ConstructFrom(float minDistSq, float maxDistSq, byte nullOutsideMinMax, float windingNumberThreshold, float windingNumberBeta);
            _UnderlyingPtr = __MR_DistanceToMeshOptions_ConstructFrom(minDistSq, maxDistSq, nullOutsideMinMax ? (byte)1 : (byte)0, windingNumberThreshold, windingNumberBeta);
        }

        /// Generated from constructor `MR::DistanceToMeshOptions::DistanceToMeshOptions`.
        public unsafe Const_DistanceToMeshOptions(MR.Const_DistanceToMeshOptions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_ConstructFromAnother(MR.DistanceToMeshOptions._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceToMeshOptions_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// options determining computation of distance from a point to a mesh
    /// Generated from class `MR::DistanceToMeshOptions`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SignedDistanceToMeshOptions`
    /// This is the non-const half of the class.
    public class DistanceToMeshOptions : Const_DistanceToMeshOptions
    {
        internal unsafe DistanceToMeshOptions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// minimum squared distance from a point to mesh to be computed precisely
        public new unsafe ref float MinDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_GetMutable_minDistSq", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_GetMutable_minDistSq(_Underlying *_this);
                return ref *__MR_DistanceToMeshOptions_GetMutable_minDistSq(_UnderlyingPtr);
            }
        }

        /// maximum squared distance from a point to mesh to be computed precisely
        public new unsafe ref float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_GetMutable_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_GetMutable_maxDistSq(_Underlying *_this);
                return ref *__MR_DistanceToMeshOptions_GetMutable_maxDistSq(_UnderlyingPtr);
            }
        }

        /// what to do if actual distance is outside [min, max) range:
        /// true - return std::nullopt for std::optional<float> or NaN for float,
        /// false - return approximate value of the distance (with correct sign in case of SignDetectionMode::HoleWindingRule);
        /// please note that in HoleWindingRule the sign can change even for too small or too large distances,
        /// so if you would like to get closed mesh from marching cubes, set false here
        public new unsafe ref bool NullOutsideMinMax
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_GetMutable_nullOutsideMinMax", ExactSpelling = true)]
                extern static bool *__MR_DistanceToMeshOptions_GetMutable_nullOutsideMinMax(_Underlying *_this);
                return ref *__MR_DistanceToMeshOptions_GetMutable_nullOutsideMinMax(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_DistanceToMeshOptions_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_DistanceToMeshOptions_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_DistanceToMeshOptions_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe DistanceToMeshOptions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_DefaultConstruct();
            _UnderlyingPtr = __MR_DistanceToMeshOptions_DefaultConstruct();
        }

        /// Constructs `MR::DistanceToMeshOptions` elementwise.
        public unsafe DistanceToMeshOptions(float minDistSq, float maxDistSq, bool nullOutsideMinMax, float windingNumberThreshold, float windingNumberBeta) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_ConstructFrom", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_ConstructFrom(float minDistSq, float maxDistSq, byte nullOutsideMinMax, float windingNumberThreshold, float windingNumberBeta);
            _UnderlyingPtr = __MR_DistanceToMeshOptions_ConstructFrom(minDistSq, maxDistSq, nullOutsideMinMax ? (byte)1 : (byte)0, windingNumberThreshold, windingNumberBeta);
        }

        /// Generated from constructor `MR::DistanceToMeshOptions::DistanceToMeshOptions`.
        public unsafe DistanceToMeshOptions(MR.Const_DistanceToMeshOptions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_ConstructFromAnother(MR.DistanceToMeshOptions._Underlying *_other);
            _UnderlyingPtr = __MR_DistanceToMeshOptions_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::DistanceToMeshOptions::operator=`.
        public unsafe MR.DistanceToMeshOptions Assign(MR.Const_DistanceToMeshOptions _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_DistanceToMeshOptions_AssignFromAnother", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_DistanceToMeshOptions_AssignFromAnother(_Underlying *_this, MR.DistanceToMeshOptions._Underlying *_other);
            return new(__MR_DistanceToMeshOptions_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `DistanceToMeshOptions` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_DistanceToMeshOptions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceToMeshOptions`/`Const_DistanceToMeshOptions` directly.
    public class _InOptMut_DistanceToMeshOptions
    {
        public DistanceToMeshOptions? Opt;

        public _InOptMut_DistanceToMeshOptions() {}
        public _InOptMut_DistanceToMeshOptions(DistanceToMeshOptions value) {Opt = value;}
        public static implicit operator _InOptMut_DistanceToMeshOptions(DistanceToMeshOptions value) {return new(value);}
    }

    /// This is used for optional parameters of class `DistanceToMeshOptions` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_DistanceToMeshOptions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `DistanceToMeshOptions`/`Const_DistanceToMeshOptions` to pass it to the function.
    public class _InOptConst_DistanceToMeshOptions
    {
        public Const_DistanceToMeshOptions? Opt;

        public _InOptConst_DistanceToMeshOptions() {}
        public _InOptConst_DistanceToMeshOptions(Const_DistanceToMeshOptions value) {Opt = value;}
        public static implicit operator _InOptConst_DistanceToMeshOptions(Const_DistanceToMeshOptions value) {return new(value);}
    }

    /// options determining computation of signed distance from a point to a mesh
    /// Generated from class `MR::SignedDistanceToMeshOptions`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceToMeshOptions`
    /// This is the const half of the class.
    public class Const_SignedDistanceToMeshOptions : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_SignedDistanceToMeshOptions(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Destroy", ExactSpelling = true)]
            extern static void __MR_SignedDistanceToMeshOptions_Destroy(_Underlying *_this);
            __MR_SignedDistanceToMeshOptions_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_SignedDistanceToMeshOptions() {Dispose(false);}

        // Upcasts:
        public static unsafe implicit operator MR.Const_DistanceToMeshOptions(Const_SignedDistanceToMeshOptions self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_UpcastTo_MR_DistanceToMeshOptions", ExactSpelling = true)]
            extern static MR.Const_DistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_UpcastTo_MR_DistanceToMeshOptions(_Underlying *_this);
            MR.Const_DistanceToMeshOptions ret = new(__MR_SignedDistanceToMeshOptions_UpcastTo_MR_DistanceToMeshOptions(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// the method to compute distance sign
        public unsafe MR.SignDetectionMode SignMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Get_signMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_SignedDistanceToMeshOptions_Get_signMode(_Underlying *_this);
                return *__MR_SignedDistanceToMeshOptions_Get_signMode(_UnderlyingPtr);
            }
        }

        /// minimum squared distance from a point to mesh to be computed precisely
        public unsafe float MinDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Get_minDistSq", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_Get_minDistSq(_Underlying *_this);
                return *__MR_SignedDistanceToMeshOptions_Get_minDistSq(_UnderlyingPtr);
            }
        }

        /// maximum squared distance from a point to mesh to be computed precisely
        public unsafe float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Get_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_Get_maxDistSq(_Underlying *_this);
                return *__MR_SignedDistanceToMeshOptions_Get_maxDistSq(_UnderlyingPtr);
            }
        }

        /// what to do if actual distance is outside [min, max) range:
        /// true - return std::nullopt for std::optional<float> or NaN for float,
        /// false - return approximate value of the distance (with correct sign in case of SignDetectionMode::HoleWindingRule);
        /// please note that in HoleWindingRule the sign can change even for too small or too large distances,
        /// so if you would like to get closed mesh from marching cubes, set false here
        public unsafe bool NullOutsideMinMax
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Get_nullOutsideMinMax", ExactSpelling = true)]
                extern static bool *__MR_SignedDistanceToMeshOptions_Get_nullOutsideMinMax(_Underlying *_this);
                return *__MR_SignedDistanceToMeshOptions_Get_nullOutsideMinMax(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public unsafe float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Get_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_Get_windingNumberThreshold(_Underlying *_this);
                return *__MR_SignedDistanceToMeshOptions_Get_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public unsafe float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_Get_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_Get_windingNumberBeta(_Underlying *_this);
                return *__MR_SignedDistanceToMeshOptions_Get_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_SignedDistanceToMeshOptions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_DefaultConstruct();
            _UnderlyingPtr = __MR_SignedDistanceToMeshOptions_DefaultConstruct();
        }

        /// Generated from constructor `MR::SignedDistanceToMeshOptions::SignedDistanceToMeshOptions`.
        public unsafe Const_SignedDistanceToMeshOptions(MR.Const_SignedDistanceToMeshOptions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_ConstructFromAnother(MR.SignedDistanceToMeshOptions._Underlying *_other);
            _UnderlyingPtr = __MR_SignedDistanceToMeshOptions_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// options determining computation of signed distance from a point to a mesh
    /// Generated from class `MR::SignedDistanceToMeshOptions`.
    /// Base classes:
    ///   Direct: (non-virtual)
    ///     `MR::DistanceToMeshOptions`
    /// This is the non-const half of the class.
    public class SignedDistanceToMeshOptions : Const_SignedDistanceToMeshOptions
    {
        internal unsafe SignedDistanceToMeshOptions(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        // Upcasts:
        public static unsafe implicit operator MR.DistanceToMeshOptions(SignedDistanceToMeshOptions self)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_UpcastTo_MR_DistanceToMeshOptions", ExactSpelling = true)]
            extern static MR.DistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_UpcastTo_MR_DistanceToMeshOptions(_Underlying *_this);
            MR.DistanceToMeshOptions ret = new(__MR_SignedDistanceToMeshOptions_UpcastTo_MR_DistanceToMeshOptions(self._UnderlyingPtr), is_owning: false);
            ret._KeepAlive(self);
            return ret;
        }

        /// the method to compute distance sign
        public new unsafe ref MR.SignDetectionMode SignMode
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_GetMutable_signMode", ExactSpelling = true)]
                extern static MR.SignDetectionMode *__MR_SignedDistanceToMeshOptions_GetMutable_signMode(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshOptions_GetMutable_signMode(_UnderlyingPtr);
            }
        }

        /// minimum squared distance from a point to mesh to be computed precisely
        public new unsafe ref float MinDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_GetMutable_minDistSq", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_GetMutable_minDistSq(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshOptions_GetMutable_minDistSq(_UnderlyingPtr);
            }
        }

        /// maximum squared distance from a point to mesh to be computed precisely
        public new unsafe ref float MaxDistSq
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_GetMutable_maxDistSq", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_GetMutable_maxDistSq(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshOptions_GetMutable_maxDistSq(_UnderlyingPtr);
            }
        }

        /// what to do if actual distance is outside [min, max) range:
        /// true - return std::nullopt for std::optional<float> or NaN for float,
        /// false - return approximate value of the distance (with correct sign in case of SignDetectionMode::HoleWindingRule);
        /// please note that in HoleWindingRule the sign can change even for too small or too large distances,
        /// so if you would like to get closed mesh from marching cubes, set false here
        public new unsafe ref bool NullOutsideMinMax
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_GetMutable_nullOutsideMinMax", ExactSpelling = true)]
                extern static bool *__MR_SignedDistanceToMeshOptions_GetMutable_nullOutsideMinMax(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshOptions_GetMutable_nullOutsideMinMax(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// positive distance if winding number below or equal this threshold;
        /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
        public new unsafe ref float WindingNumberThreshold
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_GetMutable_windingNumberThreshold", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_GetMutable_windingNumberThreshold(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshOptions_GetMutable_windingNumberThreshold(_UnderlyingPtr);
            }
        }

        /// only for SignDetectionMode::HoleWindingRule:
        /// determines the precision of fast approximation: the more the better, minimum value is 1
        public new unsafe ref float WindingNumberBeta
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_GetMutable_windingNumberBeta", ExactSpelling = true)]
                extern static float *__MR_SignedDistanceToMeshOptions_GetMutable_windingNumberBeta(_Underlying *_this);
                return ref *__MR_SignedDistanceToMeshOptions_GetMutable_windingNumberBeta(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe SignedDistanceToMeshOptions() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_DefaultConstruct", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_DefaultConstruct();
            _UnderlyingPtr = __MR_SignedDistanceToMeshOptions_DefaultConstruct();
        }

        /// Generated from constructor `MR::SignedDistanceToMeshOptions::SignedDistanceToMeshOptions`.
        public unsafe SignedDistanceToMeshOptions(MR.Const_SignedDistanceToMeshOptions _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_ConstructFromAnother(MR.SignedDistanceToMeshOptions._Underlying *_other);
            _UnderlyingPtr = __MR_SignedDistanceToMeshOptions_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::SignedDistanceToMeshOptions::operator=`.
        public unsafe MR.SignedDistanceToMeshOptions Assign(MR.Const_SignedDistanceToMeshOptions _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_SignedDistanceToMeshOptions_AssignFromAnother", ExactSpelling = true)]
            extern static MR.SignedDistanceToMeshOptions._Underlying *__MR_SignedDistanceToMeshOptions_AssignFromAnother(_Underlying *_this, MR.SignedDistanceToMeshOptions._Underlying *_other);
            return new(__MR_SignedDistanceToMeshOptions_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `SignedDistanceToMeshOptions` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_SignedDistanceToMeshOptions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SignedDistanceToMeshOptions`/`Const_SignedDistanceToMeshOptions` directly.
    public class _InOptMut_SignedDistanceToMeshOptions
    {
        public SignedDistanceToMeshOptions? Opt;

        public _InOptMut_SignedDistanceToMeshOptions() {}
        public _InOptMut_SignedDistanceToMeshOptions(SignedDistanceToMeshOptions value) {Opt = value;}
        public static implicit operator _InOptMut_SignedDistanceToMeshOptions(SignedDistanceToMeshOptions value) {return new(value);}
    }

    /// This is used for optional parameters of class `SignedDistanceToMeshOptions` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_SignedDistanceToMeshOptions`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `SignedDistanceToMeshOptions`/`Const_SignedDistanceToMeshOptions` to pass it to the function.
    public class _InOptConst_SignedDistanceToMeshOptions
    {
        public Const_SignedDistanceToMeshOptions? Opt;

        public _InOptConst_SignedDistanceToMeshOptions() {}
        public _InOptConst_SignedDistanceToMeshOptions(Const_SignedDistanceToMeshOptions value) {Opt = value;}
        public static implicit operator _InOptConst_SignedDistanceToMeshOptions(Const_SignedDistanceToMeshOptions value) {return new(value);}
    }
}
