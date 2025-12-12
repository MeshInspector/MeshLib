public static partial class MR
{
    /// describes the point of measurement on mesh
    /// Generated from class `MR::MeshPoint`.
    /// This is the const half of the class.
    public class Const_MeshPoint : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_MeshPoint(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_Destroy", ExactSpelling = true)]
            extern static void __MR_MeshPoint_Destroy(_Underlying *_this);
            __MR_MeshPoint_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_MeshPoint() {Dispose(false);}

        ///< relative position on mesh
        public unsafe MR.Const_MeshTriPoint TriPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_Get_triPoint", ExactSpelling = true)]
                extern static MR.Const_MeshTriPoint._Underlying *__MR_MeshPoint_Get_triPoint(_Underlying *_this);
                return new(__MR_MeshPoint_Get_triPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< 3d coordinates
        public unsafe MR.Const_Vector3f Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_Get_pt", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshPoint_Get_pt(_Underlying *_this);
                return new(__MR_MeshPoint_Get_pt(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< unit direction inside the mesh = minus normal
        public unsafe MR.Const_Vector3f InDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_Get_inDir", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_MeshPoint_Get_inDir(_Underlying *_this);
                return new(__MR_MeshPoint_Get_inDir(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< predicate that returns true for mesh faces not-incident to the point
        public unsafe MR.Std.Const_Function_BoolFuncFromMRFaceId NotIncidentFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_Get_notIncidentFaces", ExactSpelling = true)]
                extern static MR.Std.Const_Function_BoolFuncFromMRFaceId._Underlying *__MR_MeshPoint_Get_notIncidentFaces(_Underlying *_this);
                return new(__MR_MeshPoint_Get_notIncidentFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_MeshPoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshPoint_DefaultConstruct();
        }

        /// Constructs `MR::MeshPoint` elementwise.
        public unsafe Const_MeshPoint(MR.Const_MeshTriPoint triPoint, MR.Vector3f pt, MR.Vector3f inDir, MR.Std._ByValue_Function_BoolFuncFromMRFaceId notIncidentFaces) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_ConstructFrom(MR.MeshTriPoint._Underlying *triPoint, MR.Vector3f pt, MR.Vector3f inDir, MR.Misc._PassBy notIncidentFaces_pass_by, MR.Std.Function_BoolFuncFromMRFaceId._Underlying *notIncidentFaces);
            _UnderlyingPtr = __MR_MeshPoint_ConstructFrom(triPoint._UnderlyingPtr, pt, inDir, notIncidentFaces.PassByMode, notIncidentFaces.Value is not null ? notIncidentFaces.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshPoint::MeshPoint`.
        public unsafe Const_MeshPoint(MR._ByValue_MeshPoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshPoint._Underlying *_other);
            _UnderlyingPtr = __MR_MeshPoint_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /// describes the point of measurement on mesh
    /// Generated from class `MR::MeshPoint`.
    /// This is the non-const half of the class.
    public class MeshPoint : Const_MeshPoint
    {
        internal unsafe MeshPoint(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        ///< relative position on mesh
        public new unsafe MR.MeshTriPoint TriPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_GetMutable_triPoint", ExactSpelling = true)]
                extern static MR.MeshTriPoint._Underlying *__MR_MeshPoint_GetMutable_triPoint(_Underlying *_this);
                return new(__MR_MeshPoint_GetMutable_triPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< 3d coordinates
        public new unsafe MR.Mut_Vector3f Pt
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_GetMutable_pt", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshPoint_GetMutable_pt(_Underlying *_this);
                return new(__MR_MeshPoint_GetMutable_pt(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< unit direction inside the mesh = minus normal
        public new unsafe MR.Mut_Vector3f InDir
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_GetMutable_inDir", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_MeshPoint_GetMutable_inDir(_Underlying *_this);
                return new(__MR_MeshPoint_GetMutable_inDir(_UnderlyingPtr), is_owning: false);
            }
        }

        ///< predicate that returns true for mesh faces not-incident to the point
        public new unsafe MR.Std.Function_BoolFuncFromMRFaceId NotIncidentFaces
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_GetMutable_notIncidentFaces", ExactSpelling = true)]
                extern static MR.Std.Function_BoolFuncFromMRFaceId._Underlying *__MR_MeshPoint_GetMutable_notIncidentFaces(_Underlying *_this);
                return new(__MR_MeshPoint_GetMutable_notIncidentFaces(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe MeshPoint() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_DefaultConstruct", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_DefaultConstruct();
            _UnderlyingPtr = __MR_MeshPoint_DefaultConstruct();
        }

        /// Constructs `MR::MeshPoint` elementwise.
        public unsafe MeshPoint(MR.Const_MeshTriPoint triPoint, MR.Vector3f pt, MR.Vector3f inDir, MR.Std._ByValue_Function_BoolFuncFromMRFaceId notIncidentFaces) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_ConstructFrom", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_ConstructFrom(MR.MeshTriPoint._Underlying *triPoint, MR.Vector3f pt, MR.Vector3f inDir, MR.Misc._PassBy notIncidentFaces_pass_by, MR.Std.Function_BoolFuncFromMRFaceId._Underlying *notIncidentFaces);
            _UnderlyingPtr = __MR_MeshPoint_ConstructFrom(triPoint._UnderlyingPtr, pt, inDir, notIncidentFaces.PassByMode, notIncidentFaces.Value is not null ? notIncidentFaces.Value._UnderlyingPtr : null);
        }

        /// Generated from constructor `MR::MeshPoint::MeshPoint`.
        public unsafe MeshPoint(MR._ByValue_MeshPoint _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshPoint._Underlying *_other);
            _UnderlyingPtr = __MR_MeshPoint_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::MeshPoint::operator=`.
        public unsafe MR.MeshPoint Assign(MR._ByValue_MeshPoint _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_AssignFromAnother", ExactSpelling = true)]
            extern static MR.MeshPoint._Underlying *__MR_MeshPoint_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshPoint._Underlying *_other);
            return new(__MR_MeshPoint_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// Generated from method `MR::MeshPoint::set`.
        public unsafe void Set(MR.Const_Mesh mesh, MR.Const_MeshTriPoint p)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshPoint_set", ExactSpelling = true)]
            extern static void __MR_MeshPoint_set(_Underlying *_this, MR.Const_Mesh._Underlying *mesh, MR.Const_MeshTriPoint._Underlying *p);
            __MR_MeshPoint_set(_UnderlyingPtr, mesh._UnderlyingPtr, p._UnderlyingPtr);
        }
    }

    /// This is used as a function parameter when the underlying function receives `MeshPoint` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `MeshPoint`/`Const_MeshPoint` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_MeshPoint
    {
        internal readonly Const_MeshPoint? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_MeshPoint() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_MeshPoint(Const_MeshPoint new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_MeshPoint(Const_MeshPoint arg) {return new(arg);}
        public _ByValue_MeshPoint(MR.Misc._Moved<MeshPoint> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_MeshPoint(MR.Misc._Moved<MeshPoint> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `MeshPoint` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_MeshPoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshPoint`/`Const_MeshPoint` directly.
    public class _InOptMut_MeshPoint
    {
        public MeshPoint? Opt;

        public _InOptMut_MeshPoint() {}
        public _InOptMut_MeshPoint(MeshPoint value) {Opt = value;}
        public static implicit operator _InOptMut_MeshPoint(MeshPoint value) {return new(value);}
    }

    /// This is used for optional parameters of class `MeshPoint` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_MeshPoint`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `MeshPoint`/`Const_MeshPoint` to pass it to the function.
    public class _InOptConst_MeshPoint
    {
        public Const_MeshPoint? Opt;

        public _InOptConst_MeshPoint() {}
        public _InOptConst_MeshPoint(Const_MeshPoint value) {Opt = value;}
        public static implicit operator _InOptConst_MeshPoint(Const_MeshPoint value) {return new(value);}
    }

    /// controls the finding of maximal inscribed sphere in mesh
    /// Generated from class `MR::InSphereSearchSettings`.
    /// This is the const half of the class.
    public class Const_InSphereSearchSettings : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_InSphereSearchSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_Destroy", ExactSpelling = true)]
            extern static void __MR_InSphereSearchSettings_Destroy(_Underlying *_this);
            __MR_InSphereSearchSettings_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_InSphereSearchSettings() {Dispose(false);}

        /// if false then searches for the maximal inscribed sphere in mesh;
        /// if true then searches for both a) maximal inscribed sphere, and b) maximal sphere outside the mesh touching it at two points;
        ///              and returns the smaller of two, and if it is b) then with minus sign
        public unsafe bool InsideAndOutside
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_Get_insideAndOutside", ExactSpelling = true)]
                extern static bool *__MR_InSphereSearchSettings_Get_insideAndOutside(_Underlying *_this);
                return *__MR_InSphereSearchSettings_Get_insideAndOutside(_UnderlyingPtr);
            }
        }

        /// maximum allowed radius of the sphere;
        /// for almost closed meshes the article recommends maxRadius = 0.5f * std::min( { boxSize.x, boxSize.y, boxSize.z } )
        public unsafe float MaxRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_Get_maxRadius", ExactSpelling = true)]
                extern static float *__MR_InSphereSearchSettings_Get_maxRadius(_Underlying *_this);
                return *__MR_InSphereSearchSettings_Get_maxRadius(_UnderlyingPtr);
            }
        }

        /// maximum number of shrinking iterations for one triangle
        public unsafe int MaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_Get_maxIters", ExactSpelling = true)]
                extern static int *__MR_InSphereSearchSettings_Get_maxIters(_Underlying *_this);
                return *__MR_InSphereSearchSettings_Get_maxIters(_UnderlyingPtr);
            }
        }

        /// iterations stop if next radius is larger than minShrinkage times previous radius
        public unsafe float MinShrinkage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_Get_minShrinkage", ExactSpelling = true)]
                extern static float *__MR_InSphereSearchSettings_Get_minShrinkage(_Underlying *_this);
                return *__MR_InSphereSearchSettings_Get_minShrinkage(_UnderlyingPtr);
            }
        }

        /// minimum cosine of the angle between two unit directions:
        /// 1) search unit direction (m.inDir),
        /// 2) unit direction from sphere's center to the other found touch point;
        /// -1 value means no filtering by this angle;
        /// the increase of this value helps avoiding too small spheres on noisy surfaces
        public unsafe float MinAngleCos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_Get_minAngleCos", ExactSpelling = true)]
                extern static float *__MR_InSphereSearchSettings_Get_minAngleCos(_Underlying *_this);
                return *__MR_InSphereSearchSettings_Get_minAngleCos(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_InSphereSearchSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_InSphereSearchSettings_DefaultConstruct();
        }

        /// Constructs `MR::InSphereSearchSettings` elementwise.
        public unsafe Const_InSphereSearchSettings(bool insideAndOutside, float maxRadius, int maxIters, float minShrinkage, float minAngleCos) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_ConstructFrom(byte insideAndOutside, float maxRadius, int maxIters, float minShrinkage, float minAngleCos);
            _UnderlyingPtr = __MR_InSphereSearchSettings_ConstructFrom(insideAndOutside ? (byte)1 : (byte)0, maxRadius, maxIters, minShrinkage, minAngleCos);
        }

        /// Generated from constructor `MR::InSphereSearchSettings::InSphereSearchSettings`.
        public unsafe Const_InSphereSearchSettings(MR.Const_InSphereSearchSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_ConstructFromAnother(MR.InSphereSearchSettings._Underlying *_other);
            _UnderlyingPtr = __MR_InSphereSearchSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// controls the finding of maximal inscribed sphere in mesh
    /// Generated from class `MR::InSphereSearchSettings`.
    /// This is the non-const half of the class.
    public class InSphereSearchSettings : Const_InSphereSearchSettings
    {
        internal unsafe InSphereSearchSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// if false then searches for the maximal inscribed sphere in mesh;
        /// if true then searches for both a) maximal inscribed sphere, and b) maximal sphere outside the mesh touching it at two points;
        ///              and returns the smaller of two, and if it is b) then with minus sign
        public new unsafe ref bool InsideAndOutside
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_GetMutable_insideAndOutside", ExactSpelling = true)]
                extern static bool *__MR_InSphereSearchSettings_GetMutable_insideAndOutside(_Underlying *_this);
                return ref *__MR_InSphereSearchSettings_GetMutable_insideAndOutside(_UnderlyingPtr);
            }
        }

        /// maximum allowed radius of the sphere;
        /// for almost closed meshes the article recommends maxRadius = 0.5f * std::min( { boxSize.x, boxSize.y, boxSize.z } )
        public new unsafe ref float MaxRadius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_GetMutable_maxRadius", ExactSpelling = true)]
                extern static float *__MR_InSphereSearchSettings_GetMutable_maxRadius(_Underlying *_this);
                return ref *__MR_InSphereSearchSettings_GetMutable_maxRadius(_UnderlyingPtr);
            }
        }

        /// maximum number of shrinking iterations for one triangle
        public new unsafe ref int MaxIters
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_GetMutable_maxIters", ExactSpelling = true)]
                extern static int *__MR_InSphereSearchSettings_GetMutable_maxIters(_Underlying *_this);
                return ref *__MR_InSphereSearchSettings_GetMutable_maxIters(_UnderlyingPtr);
            }
        }

        /// iterations stop if next radius is larger than minShrinkage times previous radius
        public new unsafe ref float MinShrinkage
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_GetMutable_minShrinkage", ExactSpelling = true)]
                extern static float *__MR_InSphereSearchSettings_GetMutable_minShrinkage(_Underlying *_this);
                return ref *__MR_InSphereSearchSettings_GetMutable_minShrinkage(_UnderlyingPtr);
            }
        }

        /// minimum cosine of the angle between two unit directions:
        /// 1) search unit direction (m.inDir),
        /// 2) unit direction from sphere's center to the other found touch point;
        /// -1 value means no filtering by this angle;
        /// the increase of this value helps avoiding too small spheres on noisy surfaces
        public new unsafe ref float MinAngleCos
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_GetMutable_minAngleCos", ExactSpelling = true)]
                extern static float *__MR_InSphereSearchSettings_GetMutable_minAngleCos(_Underlying *_this);
                return ref *__MR_InSphereSearchSettings_GetMutable_minAngleCos(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe InSphereSearchSettings() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_DefaultConstruct", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_DefaultConstruct();
            _UnderlyingPtr = __MR_InSphereSearchSettings_DefaultConstruct();
        }

        /// Constructs `MR::InSphereSearchSettings` elementwise.
        public unsafe InSphereSearchSettings(bool insideAndOutside, float maxRadius, int maxIters, float minShrinkage, float minAngleCos) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_ConstructFrom", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_ConstructFrom(byte insideAndOutside, float maxRadius, int maxIters, float minShrinkage, float minAngleCos);
            _UnderlyingPtr = __MR_InSphereSearchSettings_ConstructFrom(insideAndOutside ? (byte)1 : (byte)0, maxRadius, maxIters, minShrinkage, minAngleCos);
        }

        /// Generated from constructor `MR::InSphereSearchSettings::InSphereSearchSettings`.
        public unsafe InSphereSearchSettings(MR.Const_InSphereSearchSettings _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_ConstructFromAnother(MR.InSphereSearchSettings._Underlying *_other);
            _UnderlyingPtr = __MR_InSphereSearchSettings_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::InSphereSearchSettings::operator=`.
        public unsafe MR.InSphereSearchSettings Assign(MR.Const_InSphereSearchSettings _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphereSearchSettings_AssignFromAnother", ExactSpelling = true)]
            extern static MR.InSphereSearchSettings._Underlying *__MR_InSphereSearchSettings_AssignFromAnother(_Underlying *_this, MR.InSphereSearchSettings._Underlying *_other);
            return new(__MR_InSphereSearchSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `InSphereSearchSettings` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_InSphereSearchSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InSphereSearchSettings`/`Const_InSphereSearchSettings` directly.
    public class _InOptMut_InSphereSearchSettings
    {
        public InSphereSearchSettings? Opt;

        public _InOptMut_InSphereSearchSettings() {}
        public _InOptMut_InSphereSearchSettings(InSphereSearchSettings value) {Opt = value;}
        public static implicit operator _InOptMut_InSphereSearchSettings(InSphereSearchSettings value) {return new(value);}
    }

    /// This is used for optional parameters of class `InSphereSearchSettings` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_InSphereSearchSettings`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InSphereSearchSettings`/`Const_InSphereSearchSettings` to pass it to the function.
    public class _InOptConst_InSphereSearchSettings
    {
        public Const_InSphereSearchSettings? Opt;

        public _InOptConst_InSphereSearchSettings() {}
        public _InOptConst_InSphereSearchSettings(Const_InSphereSearchSettings value) {Opt = value;}
        public static implicit operator _InOptConst_InSphereSearchSettings(Const_InSphereSearchSettings value) {return new(value);}
    }

    /// found maximal inscribed sphere touching input point with center along given direction
    /// Generated from class `MR::InSphere`.
    /// This is the const half of the class.
    public class Const_InSphere : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_InSphere(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_Destroy", ExactSpelling = true)]
            extern static void __MR_InSphere_Destroy(_Underlying *_this);
            __MR_InSphere_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_InSphere() {Dispose(false);}

        public unsafe MR.Const_Vector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_Get_center", ExactSpelling = true)]
                extern static MR.Const_Vector3f._Underlying *__MR_InSphere_Get_center(_Underlying *_this);
                return new(__MR_InSphere_Get_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public unsafe float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_Get_radius", ExactSpelling = true)]
                extern static float *__MR_InSphere_Get_radius(_Underlying *_this);
                return *__MR_InSphere_Get_radius(_UnderlyingPtr);
            }
        }

        ///< excluding input point and incident triangles, distSq - squared distance to sphere's center
        public unsafe MR.Const_MeshProjectionResult OppositeTouchPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_Get_oppositeTouchPoint", ExactSpelling = true)]
                extern static MR.Const_MeshProjectionResult._Underlying *__MR_InSphere_Get_oppositeTouchPoint(_Underlying *_this);
                return new(__MR_InSphere_Get_oppositeTouchPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_InSphere() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_DefaultConstruct", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_DefaultConstruct();
            _UnderlyingPtr = __MR_InSphere_DefaultConstruct();
        }

        /// Constructs `MR::InSphere` elementwise.
        public unsafe Const_InSphere(MR.Vector3f center, float radius, MR.Const_MeshProjectionResult oppositeTouchPoint) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_ConstructFrom", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_ConstructFrom(MR.Vector3f center, float radius, MR.MeshProjectionResult._Underlying *oppositeTouchPoint);
            _UnderlyingPtr = __MR_InSphere_ConstructFrom(center, radius, oppositeTouchPoint._UnderlyingPtr);
        }

        /// Generated from constructor `MR::InSphere::InSphere`.
        public unsafe Const_InSphere(MR.Const_InSphere _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_ConstructFromAnother(MR.InSphere._Underlying *_other);
            _UnderlyingPtr = __MR_InSphere_ConstructFromAnother(_other._UnderlyingPtr);
        }
    }

    /// found maximal inscribed sphere touching input point with center along given direction
    /// Generated from class `MR::InSphere`.
    /// This is the non-const half of the class.
    public class InSphere : Const_InSphere
    {
        internal unsafe InSphere(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe MR.Mut_Vector3f Center
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_GetMutable_center", ExactSpelling = true)]
                extern static MR.Mut_Vector3f._Underlying *__MR_InSphere_GetMutable_center(_Underlying *_this);
                return new(__MR_InSphere_GetMutable_center(_UnderlyingPtr), is_owning: false);
            }
        }

        public new unsafe ref float Radius
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_GetMutable_radius", ExactSpelling = true)]
                extern static float *__MR_InSphere_GetMutable_radius(_Underlying *_this);
                return ref *__MR_InSphere_GetMutable_radius(_UnderlyingPtr);
            }
        }

        ///< excluding input point and incident triangles, distSq - squared distance to sphere's center
        public new unsafe MR.MeshProjectionResult OppositeTouchPoint
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_GetMutable_oppositeTouchPoint", ExactSpelling = true)]
                extern static MR.MeshProjectionResult._Underlying *__MR_InSphere_GetMutable_oppositeTouchPoint(_Underlying *_this);
                return new(__MR_InSphere_GetMutable_oppositeTouchPoint(_UnderlyingPtr), is_owning: false);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe InSphere() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_DefaultConstruct", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_DefaultConstruct();
            _UnderlyingPtr = __MR_InSphere_DefaultConstruct();
        }

        /// Constructs `MR::InSphere` elementwise.
        public unsafe InSphere(MR.Vector3f center, float radius, MR.Const_MeshProjectionResult oppositeTouchPoint) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_ConstructFrom", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_ConstructFrom(MR.Vector3f center, float radius, MR.MeshProjectionResult._Underlying *oppositeTouchPoint);
            _UnderlyingPtr = __MR_InSphere_ConstructFrom(center, radius, oppositeTouchPoint._UnderlyingPtr);
        }

        /// Generated from constructor `MR::InSphere::InSphere`.
        public unsafe InSphere(MR.Const_InSphere _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_ConstructFromAnother(MR.InSphere._Underlying *_other);
            _UnderlyingPtr = __MR_InSphere_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from method `MR::InSphere::operator=`.
        public unsafe MR.InSphere Assign(MR.Const_InSphere _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_InSphere_AssignFromAnother", ExactSpelling = true)]
            extern static MR.InSphere._Underlying *__MR_InSphere_AssignFromAnother(_Underlying *_this, MR.InSphere._Underlying *_other);
            return new(__MR_InSphere_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `InSphere` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_InSphere`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InSphere`/`Const_InSphere` directly.
    public class _InOptMut_InSphere
    {
        public InSphere? Opt;

        public _InOptMut_InSphere() {}
        public _InOptMut_InSphere(InSphere value) {Opt = value;}
        public static implicit operator _InOptMut_InSphere(InSphere value) {return new(value);}
    }

    /// This is used for optional parameters of class `InSphere` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_InSphere`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `InSphere`/`Const_InSphere` to pass it to the function.
    public class _InOptConst_InSphere
    {
        public Const_InSphere? Opt;

        public _InOptConst_InSphere() {}
        public _InOptConst_InSphere(Const_InSphere value) {Opt = value;}
        public static implicit operator _InOptConst_InSphere(Const_InSphere value) {return new(value);}
    }

    /// returns the distance from each vertex along minus normal to the nearest mesh intersection (or FLT_MAX if no intersection found)
    /// Generated from function `MR::computeRayThicknessAtVertices`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertScalars> ComputeRayThicknessAtVertices(MR.Const_Mesh mesh, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeRayThicknessAtVertices", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertScalars._Underlying *__MR_computeRayThicknessAtVertices(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.Std.Optional_MRVertScalars(__MR_computeRayThicknessAtVertices(mesh._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }

    /// returns the nearest intersection between the mesh and the ray from given point along minus normal (inside the mesh)
    /// Generated from function `MR::rayInsideIntersect`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    public static unsafe MR.MeshIntersectionResult RayInsideIntersect(MR.Const_Mesh mesh, MR.Const_MeshPoint m, float? rayEnd = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayInsideIntersect_MR_MeshPoint", ExactSpelling = true)]
        extern static MR.MeshIntersectionResult._Underlying *__MR_rayInsideIntersect_MR_MeshPoint(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshPoint._Underlying *m, float *rayEnd);
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        return new(__MR_rayInsideIntersect_MR_MeshPoint(mesh._UnderlyingPtr, m._UnderlyingPtr, rayEnd.HasValue ? &__deref_rayEnd : null), is_owning: true);
    }

    /// Generated from function `MR::rayInsideIntersect`.
    /// Parameter `rayEnd` defaults to `3.40282347e38f`.
    public static unsafe MR.MeshIntersectionResult RayInsideIntersect(MR.Const_Mesh mesh, MR.VertId v, float? rayEnd = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_rayInsideIntersect_MR_VertId", ExactSpelling = true)]
        extern static MR.MeshIntersectionResult._Underlying *__MR_rayInsideIntersect_MR_VertId(MR.Const_Mesh._Underlying *mesh, MR.VertId v, float *rayEnd);
        float __deref_rayEnd = rayEnd.GetValueOrDefault();
        return new(__MR_rayInsideIntersect_MR_VertId(mesh._UnderlyingPtr, v, rayEnd.HasValue ? &__deref_rayEnd : null), is_owning: true);
    }

    /// finds maximal sphere inscribed in the mesh touching point (p) with center along the normal at (p)
    /// Generated from function `MR::findInSphere`.
    public static unsafe MR.InSphere FindInSphere(MR.Const_Mesh mesh, MR.Const_MeshPoint m, MR.Const_InSphereSearchSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findInSphere_MR_MeshPoint", ExactSpelling = true)]
        extern static MR.InSphere._Underlying *__MR_findInSphere_MR_MeshPoint(MR.Const_Mesh._Underlying *mesh, MR.Const_MeshPoint._Underlying *m, MR.Const_InSphereSearchSettings._Underlying *settings);
        return new(__MR_findInSphere_MR_MeshPoint(mesh._UnderlyingPtr, m._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true);
    }

    /// Generated from function `MR::findInSphere`.
    public static unsafe MR.InSphere FindInSphere(MR.Const_Mesh mesh, MR.VertId v, MR.Const_InSphereSearchSettings settings)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_findInSphere_MR_VertId", ExactSpelling = true)]
        extern static MR.InSphere._Underlying *__MR_findInSphere_MR_VertId(MR.Const_Mesh._Underlying *mesh, MR.VertId v, MR.Const_InSphereSearchSettings._Underlying *settings);
        return new(__MR_findInSphere_MR_VertId(mesh._UnderlyingPtr, v, settings._UnderlyingPtr), is_owning: true);
    }

    /// returns the thickness at each vertex as the diameter of the maximal inscribed sphere
    /// Generated from function `MR::computeInSphereThicknessAtVertices`.
    /// Parameter `progress` defaults to `{}`.
    public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertScalars> ComputeInSphereThicknessAtVertices(MR.Const_Mesh mesh, MR.Const_InSphereSearchSettings settings, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_computeInSphereThicknessAtVertices", ExactSpelling = true)]
        extern static MR.Std.Optional_MRVertScalars._Underlying *__MR_computeInSphereThicknessAtVertices(MR.Const_Mesh._Underlying *mesh, MR.Const_InSphereSearchSettings._Underlying *settings, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
        return MR.Misc.Move(new MR.Std.Optional_MRVertScalars(__MR_computeInSphereThicknessAtVertices(mesh._UnderlyingPtr, settings._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
    }
}
