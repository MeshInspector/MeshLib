public static partial class MR
{
    public static partial class MeshBuilder
    {
        /// mesh triangle represented by its three vertices and by its face ID
        /// Generated from class `MR::MeshBuilder::Triangle`.
        /// This is the const half of the class.
        public class Const_Triangle : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.MeshBuilder.Const_Triangle>
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Triangle(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_Triangle_Destroy(_Underlying *_this);
                __MR_MeshBuilder_Triangle_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Triangle() {Dispose(false);}

            public unsafe MR.Std.Const_Array_MRVertId_3 V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_Get_v", ExactSpelling = true)]
                    extern static MR.Std.Const_Array_MRVertId_3._Underlying *__MR_MeshBuilder_Triangle_Get_v(_Underlying *_this);
                    return new(__MR_MeshBuilder_Triangle_Get_v(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe MR.Const_FaceId F
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_Get_f", ExactSpelling = true)]
                    extern static MR.Const_FaceId._Underlying *__MR_MeshBuilder_Triangle_Get_f(_Underlying *_this);
                    return new(__MR_MeshBuilder_Triangle_Get_f(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Triangle() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_Triangle_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshBuilder::Triangle::Triangle`.
            public unsafe Const_Triangle(MR.MeshBuilder.Const_Triangle _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_ConstructFromAnother(MR.MeshBuilder.Triangle._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_Triangle_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from constructor `MR::MeshBuilder::Triangle::Triangle`.
            public unsafe Const_Triangle(MR.VertId a, MR.VertId b, MR.VertId c, MR.FaceId f) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_Construct", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_Construct(MR.VertId a, MR.VertId b, MR.VertId c, MR.FaceId f);
                _UnderlyingPtr = __MR_MeshBuilder_Triangle_Construct(a, b, c, f);
            }

            /// Generated from method `MR::MeshBuilder::Triangle::operator==`.
            public static unsafe bool operator==(MR.MeshBuilder.Const_Triangle _this, MR.MeshBuilder.Const_Triangle other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_MeshBuilder_Triangle", ExactSpelling = true)]
                extern static byte __MR_equal_MR_MeshBuilder_Triangle(MR.MeshBuilder.Const_Triangle._Underlying *_this, MR.MeshBuilder.Const_Triangle._Underlying *other);
                return __MR_equal_MR_MeshBuilder_Triangle(_this._UnderlyingPtr, other._UnderlyingPtr) != 0;
            }

            public static unsafe bool operator!=(MR.MeshBuilder.Const_Triangle _this, MR.MeshBuilder.Const_Triangle other)
            {
                return !(_this == other);
            }

            // IEquatable:

            public bool Equals(MR.MeshBuilder.Const_Triangle? other)
            {
                if (other is null)
                    return false;
                return this == other;
            }

            public override bool Equals(object? other)
            {
                if (other is null)
                    return false;
                if (other is MR.MeshBuilder.Const_Triangle)
                    return this == (MR.MeshBuilder.Const_Triangle)other;
                return false;
            }
        }

        /// mesh triangle represented by its three vertices and by its face ID
        /// Generated from class `MR::MeshBuilder::Triangle`.
        /// This is the non-const half of the class.
        public class Triangle : Const_Triangle
        {
            internal unsafe Triangle(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Std.Mut_Array_MRVertId_3 V
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_GetMutable_v", ExactSpelling = true)]
                    extern static MR.Std.Mut_Array_MRVertId_3._Underlying *__MR_MeshBuilder_Triangle_GetMutable_v(_Underlying *_this);
                    return new(__MR_MeshBuilder_Triangle_GetMutable_v(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe MR.Mut_FaceId F
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_GetMutable_f", ExactSpelling = true)]
                    extern static MR.Mut_FaceId._Underlying *__MR_MeshBuilder_Triangle_GetMutable_f(_Underlying *_this);
                    return new(__MR_MeshBuilder_Triangle_GetMutable_f(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Triangle() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_Triangle_DefaultConstruct();
            }

            /// Generated from constructor `MR::MeshBuilder::Triangle::Triangle`.
            public unsafe Triangle(MR.MeshBuilder.Const_Triangle _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_ConstructFromAnother(MR.MeshBuilder.Triangle._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_Triangle_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from constructor `MR::MeshBuilder::Triangle::Triangle`.
            public unsafe Triangle(MR.VertId a, MR.VertId b, MR.VertId c, MR.FaceId f) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_Construct", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_Construct(MR.VertId a, MR.VertId b, MR.VertId c, MR.FaceId f);
                _UnderlyingPtr = __MR_MeshBuilder_Triangle_Construct(a, b, c, f);
            }

            /// Generated from method `MR::MeshBuilder::Triangle::operator=`.
            public unsafe MR.MeshBuilder.Triangle Assign(MR.MeshBuilder.Const_Triangle _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_Triangle_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.Triangle._Underlying *__MR_MeshBuilder_Triangle_AssignFromAnother(_Underlying *_this, MR.MeshBuilder.Triangle._Underlying *_other);
                return new(__MR_MeshBuilder_Triangle_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Triangle` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Triangle`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Triangle`/`Const_Triangle` directly.
        public class _InOptMut_Triangle
        {
            public Triangle? Opt;

            public _InOptMut_Triangle() {}
            public _InOptMut_Triangle(Triangle value) {Opt = value;}
            public static implicit operator _InOptMut_Triangle(Triangle value) {return new(value);}
        }

        /// This is used for optional parameters of class `Triangle` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Triangle`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Triangle`/`Const_Triangle` to pass it to the function.
        public class _InOptConst_Triangle
        {
            public Const_Triangle? Opt;

            public _InOptConst_Triangle() {}
            public _InOptConst_Triangle(Const_Triangle value) {Opt = value;}
            public static implicit operator _InOptConst_Triangle(Const_Triangle value) {return new(value);}
        }

        /// Generated from class `MR::MeshBuilder::BuildSettings`.
        /// This is the const half of the class.
        public class Const_BuildSettings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BuildSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_BuildSettings_Destroy(_Underlying *_this);
                __MR_MeshBuilder_BuildSettings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BuildSettings() {Dispose(false);}

            /// if region is given then on input it contains the faces to be added, and on output the faces failed to be added
            public unsafe ref void * Region
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_Get_region", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_BuildSettings_Get_region(_Underlying *_this);
                    return ref *__MR_MeshBuilder_BuildSettings_Get_region(_UnderlyingPtr);
                }
            }

            /// this value to be added to every faceId before its inclusion in the topology
            public unsafe int ShiftFaceId
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_Get_shiftFaceId", ExactSpelling = true)]
                    extern static int *__MR_MeshBuilder_BuildSettings_Get_shiftFaceId(_Underlying *_this);
                    return *__MR_MeshBuilder_BuildSettings_Get_shiftFaceId(_UnderlyingPtr);
                }
            }

            /// whether to permit non-manifold edges in the resulting topology
            public unsafe bool AllowNonManifoldEdge
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_Get_allowNonManifoldEdge", ExactSpelling = true)]
                    extern static bool *__MR_MeshBuilder_BuildSettings_Get_allowNonManifoldEdge(_Underlying *_this);
                    return *__MR_MeshBuilder_BuildSettings_Get_allowNonManifoldEdge(_UnderlyingPtr);
                }
            }

            /// optional output: counter of skipped faces during mesh creation
            public unsafe ref int * SkippedFaceCount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_Get_skippedFaceCount", ExactSpelling = true)]
                    extern static int **__MR_MeshBuilder_BuildSettings_Get_skippedFaceCount(_Underlying *_this);
                    return ref *__MR_MeshBuilder_BuildSettings_Get_skippedFaceCount(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BuildSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_BuildSettings_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::BuildSettings` elementwise.
            public unsafe Const_BuildSettings(MR.FaceBitSet? region, int shiftFaceId, bool allowNonManifoldEdge, MR.Misc.InOut<int>? skippedFaceCount) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_ConstructFrom(MR.FaceBitSet._Underlying *region, int shiftFaceId, byte allowNonManifoldEdge, int *skippedFaceCount);
                int __value_skippedFaceCount = skippedFaceCount is not null ? skippedFaceCount.Value : default(int);
                _UnderlyingPtr = __MR_MeshBuilder_BuildSettings_ConstructFrom(region is not null ? region._UnderlyingPtr : null, shiftFaceId, allowNonManifoldEdge ? (byte)1 : (byte)0, skippedFaceCount is not null ? &__value_skippedFaceCount : null);
                if (skippedFaceCount is not null) skippedFaceCount.Value = __value_skippedFaceCount;
            }

            /// Generated from constructor `MR::MeshBuilder::BuildSettings::BuildSettings`.
            public unsafe Const_BuildSettings(MR.MeshBuilder.Const_BuildSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_ConstructFromAnother(MR.MeshBuilder.BuildSettings._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_BuildSettings_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::MeshBuilder::BuildSettings`.
        /// This is the non-const half of the class.
        public class BuildSettings : Const_BuildSettings
        {
            internal unsafe BuildSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// if region is given then on input it contains the faces to be added, and on output the faces failed to be added
            public new unsafe ref void * Region
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_GetMutable_region", ExactSpelling = true)]
                    extern static void **__MR_MeshBuilder_BuildSettings_GetMutable_region(_Underlying *_this);
                    return ref *__MR_MeshBuilder_BuildSettings_GetMutable_region(_UnderlyingPtr);
                }
            }

            /// this value to be added to every faceId before its inclusion in the topology
            public new unsafe ref int ShiftFaceId
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_GetMutable_shiftFaceId", ExactSpelling = true)]
                    extern static int *__MR_MeshBuilder_BuildSettings_GetMutable_shiftFaceId(_Underlying *_this);
                    return ref *__MR_MeshBuilder_BuildSettings_GetMutable_shiftFaceId(_UnderlyingPtr);
                }
            }

            /// whether to permit non-manifold edges in the resulting topology
            public new unsafe ref bool AllowNonManifoldEdge
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_GetMutable_allowNonManifoldEdge", ExactSpelling = true)]
                    extern static bool *__MR_MeshBuilder_BuildSettings_GetMutable_allowNonManifoldEdge(_Underlying *_this);
                    return ref *__MR_MeshBuilder_BuildSettings_GetMutable_allowNonManifoldEdge(_UnderlyingPtr);
                }
            }

            /// optional output: counter of skipped faces during mesh creation
            public new unsafe ref int * SkippedFaceCount
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_GetMutable_skippedFaceCount", ExactSpelling = true)]
                    extern static int **__MR_MeshBuilder_BuildSettings_GetMutable_skippedFaceCount(_Underlying *_this);
                    return ref *__MR_MeshBuilder_BuildSettings_GetMutable_skippedFaceCount(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BuildSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_BuildSettings_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::BuildSettings` elementwise.
            public unsafe BuildSettings(MR.FaceBitSet? region, int shiftFaceId, bool allowNonManifoldEdge, MR.Misc.InOut<int>? skippedFaceCount) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_ConstructFrom(MR.FaceBitSet._Underlying *region, int shiftFaceId, byte allowNonManifoldEdge, int *skippedFaceCount);
                int __value_skippedFaceCount = skippedFaceCount is not null ? skippedFaceCount.Value : default(int);
                _UnderlyingPtr = __MR_MeshBuilder_BuildSettings_ConstructFrom(region is not null ? region._UnderlyingPtr : null, shiftFaceId, allowNonManifoldEdge ? (byte)1 : (byte)0, skippedFaceCount is not null ? &__value_skippedFaceCount : null);
                if (skippedFaceCount is not null) skippedFaceCount.Value = __value_skippedFaceCount;
            }

            /// Generated from constructor `MR::MeshBuilder::BuildSettings::BuildSettings`.
            public unsafe BuildSettings(MR.MeshBuilder.Const_BuildSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_ConstructFromAnother(MR.MeshBuilder.BuildSettings._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_BuildSettings_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshBuilder::BuildSettings::operator=`.
            public unsafe MR.MeshBuilder.BuildSettings Assign(MR.MeshBuilder.Const_BuildSettings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_BuildSettings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.BuildSettings._Underlying *__MR_MeshBuilder_BuildSettings_AssignFromAnother(_Underlying *_this, MR.MeshBuilder.BuildSettings._Underlying *_other);
                return new(__MR_MeshBuilder_BuildSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `BuildSettings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BuildSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BuildSettings`/`Const_BuildSettings` directly.
        public class _InOptMut_BuildSettings
        {
            public BuildSettings? Opt;

            public _InOptMut_BuildSettings() {}
            public _InOptMut_BuildSettings(BuildSettings value) {Opt = value;}
            public static implicit operator _InOptMut_BuildSettings(BuildSettings value) {return new(value);}
        }

        /// This is used for optional parameters of class `BuildSettings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BuildSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BuildSettings`/`Const_BuildSettings` to pass it to the function.
        public class _InOptConst_BuildSettings
        {
            public Const_BuildSettings? Opt;

            public _InOptConst_BuildSettings() {}
            public _InOptConst_BuildSettings(Const_BuildSettings value) {Opt = value;}
            public static implicit operator _InOptConst_BuildSettings(Const_BuildSettings value) {return new(value);}
        }

        // each face is surrounded by a closed contour of vertices [fistVertex, lastVertex)
        /// Generated from class `MR::MeshBuilder::VertSpan`.
        /// This is the const half of the class.
        public class Const_VertSpan : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_VertSpan(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshBuilder_VertSpan_Destroy(_Underlying *_this);
                __MR_MeshBuilder_VertSpan_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_VertSpan() {Dispose(false);}

            public unsafe int FirstVertex
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_Get_firstVertex", ExactSpelling = true)]
                    extern static int *__MR_MeshBuilder_VertSpan_Get_firstVertex(_Underlying *_this);
                    return *__MR_MeshBuilder_VertSpan_Get_firstVertex(_UnderlyingPtr);
                }
            }

            public unsafe int LastVertex
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_Get_lastVertex", ExactSpelling = true)]
                    extern static int *__MR_MeshBuilder_VertSpan_Get_lastVertex(_Underlying *_this);
                    return *__MR_MeshBuilder_VertSpan_Get_lastVertex(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_VertSpan() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_VertSpan_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::VertSpan` elementwise.
            public unsafe Const_VertSpan(int firstVertex, int lastVertex) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_ConstructFrom(int firstVertex, int lastVertex);
                _UnderlyingPtr = __MR_MeshBuilder_VertSpan_ConstructFrom(firstVertex, lastVertex);
            }

            /// Generated from constructor `MR::MeshBuilder::VertSpan::VertSpan`.
            public unsafe Const_VertSpan(MR.MeshBuilder.Const_VertSpan _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_ConstructFromAnother(MR.MeshBuilder.VertSpan._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_VertSpan_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        // each face is surrounded by a closed contour of vertices [fistVertex, lastVertex)
        /// Generated from class `MR::MeshBuilder::VertSpan`.
        /// This is the non-const half of the class.
        public class VertSpan : Const_VertSpan
        {
            internal unsafe VertSpan(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe ref int FirstVertex
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_GetMutable_firstVertex", ExactSpelling = true)]
                    extern static int *__MR_MeshBuilder_VertSpan_GetMutable_firstVertex(_Underlying *_this);
                    return ref *__MR_MeshBuilder_VertSpan_GetMutable_firstVertex(_UnderlyingPtr);
                }
            }

            public new unsafe ref int LastVertex
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_GetMutable_lastVertex", ExactSpelling = true)]
                    extern static int *__MR_MeshBuilder_VertSpan_GetMutable_lastVertex(_Underlying *_this);
                    return ref *__MR_MeshBuilder_VertSpan_GetMutable_lastVertex(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe VertSpan() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshBuilder_VertSpan_DefaultConstruct();
            }

            /// Constructs `MR::MeshBuilder::VertSpan` elementwise.
            public unsafe VertSpan(int firstVertex, int lastVertex) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_ConstructFrom(int firstVertex, int lastVertex);
                _UnderlyingPtr = __MR_MeshBuilder_VertSpan_ConstructFrom(firstVertex, lastVertex);
            }

            /// Generated from constructor `MR::MeshBuilder::VertSpan::VertSpan`.
            public unsafe VertSpan(MR.MeshBuilder.Const_VertSpan _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_ConstructFromAnother(MR.MeshBuilder.VertSpan._Underlying *_other);
                _UnderlyingPtr = __MR_MeshBuilder_VertSpan_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshBuilder::VertSpan::operator=`.
            public unsafe MR.MeshBuilder.VertSpan Assign(MR.MeshBuilder.Const_VertSpan _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshBuilder_VertSpan_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshBuilder.VertSpan._Underlying *__MR_MeshBuilder_VertSpan_AssignFromAnother(_Underlying *_this, MR.MeshBuilder.VertSpan._Underlying *_other);
                return new(__MR_MeshBuilder_VertSpan_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `VertSpan` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_VertSpan`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VertSpan`/`Const_VertSpan` directly.
        public class _InOptMut_VertSpan
        {
            public VertSpan? Opt;

            public _InOptMut_VertSpan() {}
            public _InOptMut_VertSpan(VertSpan value) {Opt = value;}
            public static implicit operator _InOptMut_VertSpan(VertSpan value) {return new(value);}
        }

        /// This is used for optional parameters of class `VertSpan` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_VertSpan`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `VertSpan`/`Const_VertSpan` to pass it to the function.
        public class _InOptConst_VertSpan
        {
            public Const_VertSpan? Opt;

            public _InOptConst_VertSpan() {}
            public _InOptConst_VertSpan(Const_VertSpan value) {Opt = value;}
            public static implicit operator _InOptConst_VertSpan(Const_VertSpan value) {return new(value);}
        }
    }
}
