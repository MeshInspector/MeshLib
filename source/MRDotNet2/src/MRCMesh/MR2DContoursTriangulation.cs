public static partial class MR
{
    public static partial class PlanarTriangulation
    {
        /// Specify mode of detecting inside and outside parts of triangulation
        public enum WindingMode : int
        {
            NonZero = 0,
            Positive = 1,
            Negative = 2,
        }

        /// Info about intersection point for mapping
        /// Generated from class `MR::PlanarTriangulation::IntersectionInfo`.
        /// This is the const half of the class.
        public class Const_IntersectionInfo : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_IntersectionInfo(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Destroy", ExactSpelling = true)]
                extern static void __MR_PlanarTriangulation_IntersectionInfo_Destroy(_Underlying *_this);
                __MR_PlanarTriangulation_IntersectionInfo_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_IntersectionInfo() {Dispose(false);}

            /// if lDest is invalid then lOrg is id of input vertex
            /// ids of lower intersection edge vertices
            public unsafe MR.Const_VertId LOrg
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Get_lOrg", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_Get_lOrg(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_Get_lOrg(_UnderlyingPtr), is_owning: false);
                }
            }

            /// if lDest is invalid then lOrg is id of input vertex
            /// ids of lower intersection edge vertices
            public unsafe MR.Const_VertId LDest
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Get_lDest", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_Get_lDest(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_Get_lDest(_UnderlyingPtr), is_owning: false);
                }
            }

            /// ids of upper intersection edge vertices
            public unsafe MR.Const_VertId UOrg
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Get_uOrg", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_Get_uOrg(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_Get_uOrg(_UnderlyingPtr), is_owning: false);
                }
            }

            /// ids of upper intersection edge vertices
            public unsafe MR.Const_VertId UDest
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Get_uDest", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_Get_uDest(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_Get_uDest(_UnderlyingPtr), is_owning: false);
                }
            }

            // ratio of intersection
            // 0.0 -> point is lOrg
            // 1.0 -> point is lDest
            public unsafe float LRatio
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Get_lRatio", ExactSpelling = true)]
                    extern static float *__MR_PlanarTriangulation_IntersectionInfo_Get_lRatio(_Underlying *_this);
                    return *__MR_PlanarTriangulation_IntersectionInfo_Get_lRatio(_UnderlyingPtr);
                }
            }

            // 0.0 -> point is uOrg
            // 1.0 -> point is uDest
            public unsafe float URatio
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_Get_uRatio", ExactSpelling = true)]
                    extern static float *__MR_PlanarTriangulation_IntersectionInfo_Get_uRatio(_Underlying *_this);
                    return *__MR_PlanarTriangulation_IntersectionInfo_Get_uRatio(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_IntersectionInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionInfo_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::IntersectionInfo` elementwise.
            public unsafe Const_IntersectionInfo(MR.VertId lOrg, MR.VertId lDest, MR.VertId uOrg, MR.VertId uDest, float lRatio, float uRatio) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_ConstructFrom(MR.VertId lOrg, MR.VertId lDest, MR.VertId uOrg, MR.VertId uDest, float lRatio, float uRatio);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionInfo_ConstructFrom(lOrg, lDest, uOrg, uDest, lRatio, uRatio);
            }

            /// Generated from constructor `MR::PlanarTriangulation::IntersectionInfo::IntersectionInfo`.
            public unsafe Const_IntersectionInfo(MR.PlanarTriangulation.Const_IntersectionInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_ConstructFromAnother(MR.PlanarTriangulation.IntersectionInfo._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionInfo_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::PlanarTriangulation::IntersectionInfo::isIntersection`.
            public unsafe bool IsIntersection()
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_isIntersection", ExactSpelling = true)]
                extern static byte __MR_PlanarTriangulation_IntersectionInfo_isIntersection(_Underlying *_this);
                return __MR_PlanarTriangulation_IntersectionInfo_isIntersection(_UnderlyingPtr) != 0;
            }
        }

        /// Info about intersection point for mapping
        /// Generated from class `MR::PlanarTriangulation::IntersectionInfo`.
        /// This is the non-const half of the class.
        public class IntersectionInfo : Const_IntersectionInfo
        {
            internal unsafe IntersectionInfo(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// if lDest is invalid then lOrg is id of input vertex
            /// ids of lower intersection edge vertices
            public new unsafe MR.Mut_VertId LOrg
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_GetMutable_lOrg", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_lOrg(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_GetMutable_lOrg(_UnderlyingPtr), is_owning: false);
                }
            }

            /// if lDest is invalid then lOrg is id of input vertex
            /// ids of lower intersection edge vertices
            public new unsafe MR.Mut_VertId LDest
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_GetMutable_lDest", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_lDest(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_GetMutable_lDest(_UnderlyingPtr), is_owning: false);
                }
            }

            /// ids of upper intersection edge vertices
            public new unsafe MR.Mut_VertId UOrg
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_GetMutable_uOrg", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_uOrg(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_GetMutable_uOrg(_UnderlyingPtr), is_owning: false);
                }
            }

            /// ids of upper intersection edge vertices
            public new unsafe MR.Mut_VertId UDest
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_GetMutable_uDest", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_uDest(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionInfo_GetMutable_uDest(_UnderlyingPtr), is_owning: false);
                }
            }

            // ratio of intersection
            // 0.0 -> point is lOrg
            // 1.0 -> point is lDest
            public new unsafe ref float LRatio
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_GetMutable_lRatio", ExactSpelling = true)]
                    extern static float *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_lRatio(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_lRatio(_UnderlyingPtr);
                }
            }

            // 0.0 -> point is uOrg
            // 1.0 -> point is uDest
            public new unsafe ref float URatio
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_GetMutable_uRatio", ExactSpelling = true)]
                    extern static float *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_uRatio(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_IntersectionInfo_GetMutable_uRatio(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe IntersectionInfo() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionInfo_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::IntersectionInfo` elementwise.
            public unsafe IntersectionInfo(MR.VertId lOrg, MR.VertId lDest, MR.VertId uOrg, MR.VertId uDest, float lRatio, float uRatio) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_ConstructFrom(MR.VertId lOrg, MR.VertId lDest, MR.VertId uOrg, MR.VertId uDest, float lRatio, float uRatio);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionInfo_ConstructFrom(lOrg, lDest, uOrg, uDest, lRatio, uRatio);
            }

            /// Generated from constructor `MR::PlanarTriangulation::IntersectionInfo::IntersectionInfo`.
            public unsafe IntersectionInfo(MR.PlanarTriangulation.Const_IntersectionInfo _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_ConstructFromAnother(MR.PlanarTriangulation.IntersectionInfo._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionInfo_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::PlanarTriangulation::IntersectionInfo::operator=`.
            public unsafe MR.PlanarTriangulation.IntersectionInfo Assign(MR.PlanarTriangulation.Const_IntersectionInfo _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionInfo_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionInfo_AssignFromAnother(_Underlying *_this, MR.PlanarTriangulation.IntersectionInfo._Underlying *_other);
                return new(__MR_PlanarTriangulation_IntersectionInfo_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `IntersectionInfo` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_IntersectionInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `IntersectionInfo`/`Const_IntersectionInfo` directly.
        public class _InOptMut_IntersectionInfo
        {
            public IntersectionInfo? Opt;

            public _InOptMut_IntersectionInfo() {}
            public _InOptMut_IntersectionInfo(IntersectionInfo value) {Opt = value;}
            public static implicit operator _InOptMut_IntersectionInfo(IntersectionInfo value) {return new(value);}
        }

        /// This is used for optional parameters of class `IntersectionInfo` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_IntersectionInfo`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `IntersectionInfo`/`Const_IntersectionInfo` to pass it to the function.
        public class _InOptConst_IntersectionInfo
        {
            public Const_IntersectionInfo? Opt;

            public _InOptConst_IntersectionInfo() {}
            public _InOptConst_IntersectionInfo(Const_IntersectionInfo value) {Opt = value;}
            public static implicit operator _InOptConst_IntersectionInfo(Const_IntersectionInfo value) {return new(value);}
        }

        /// struct to map new vertices (only appear on intersections) of the outline to it's edges
        /// Generated from class `MR::PlanarTriangulation::IntersectionsMap`.
        /// This is the const half of the class.
        public class Const_IntersectionsMap : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_IntersectionsMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_Destroy", ExactSpelling = true)]
                extern static void __MR_PlanarTriangulation_IntersectionsMap_Destroy(_Underlying *_this);
                __MR_PlanarTriangulation_IntersectionsMap_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_IntersectionsMap() {Dispose(false);}

            /// shift of index
            public unsafe ulong Shift
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_Get_shift", ExactSpelling = true)]
                    extern static ulong *__MR_PlanarTriangulation_IntersectionsMap_Get_shift(_Underlying *_this);
                    return *__MR_PlanarTriangulation_IntersectionsMap_Get_shift(_UnderlyingPtr);
                }
            }

            /// map[id-shift] = {lower intersection edge, upper intersection edge}
            public unsafe MR.Std.Const_Vector_MRPlanarTriangulationIntersectionInfo Map
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_Get_map", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRPlanarTriangulationIntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionsMap_Get_map(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionsMap_Get_map(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_IntersectionsMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionsMap_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::IntersectionsMap` elementwise.
            public unsafe Const_IntersectionsMap(ulong shift, MR.Std._ByValue_Vector_MRPlanarTriangulationIntersectionInfo map) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_ConstructFrom(ulong shift, MR.Misc._PassBy map_pass_by, MR.Std.Vector_MRPlanarTriangulationIntersectionInfo._Underlying *map);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionsMap_ConstructFrom(shift, map.PassByMode, map.Value is not null ? map.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::PlanarTriangulation::IntersectionsMap::IntersectionsMap`.
            public unsafe Const_IntersectionsMap(MR.PlanarTriangulation._ByValue_IntersectionsMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PlanarTriangulation.IntersectionsMap._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionsMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// struct to map new vertices (only appear on intersections) of the outline to it's edges
        /// Generated from class `MR::PlanarTriangulation::IntersectionsMap`.
        /// This is the non-const half of the class.
        public class IntersectionsMap : Const_IntersectionsMap
        {
            internal unsafe IntersectionsMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// shift of index
            public new unsafe ref ulong Shift
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_GetMutable_shift", ExactSpelling = true)]
                    extern static ulong *__MR_PlanarTriangulation_IntersectionsMap_GetMutable_shift(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_IntersectionsMap_GetMutable_shift(_UnderlyingPtr);
                }
            }

            /// map[id-shift] = {lower intersection edge, upper intersection edge}
            public new unsafe MR.Std.Vector_MRPlanarTriangulationIntersectionInfo Map
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_GetMutable_map", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRPlanarTriangulationIntersectionInfo._Underlying *__MR_PlanarTriangulation_IntersectionsMap_GetMutable_map(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_IntersectionsMap_GetMutable_map(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe IntersectionsMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionsMap_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::IntersectionsMap` elementwise.
            public unsafe IntersectionsMap(ulong shift, MR.Std._ByValue_Vector_MRPlanarTriangulationIntersectionInfo map) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_ConstructFrom(ulong shift, MR.Misc._PassBy map_pass_by, MR.Std.Vector_MRPlanarTriangulationIntersectionInfo._Underlying *map);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionsMap_ConstructFrom(shift, map.PassByMode, map.Value is not null ? map.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::PlanarTriangulation::IntersectionsMap::IntersectionsMap`.
            public unsafe IntersectionsMap(MR.PlanarTriangulation._ByValue_IntersectionsMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.PlanarTriangulation.IntersectionsMap._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_IntersectionsMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::PlanarTriangulation::IntersectionsMap::operator=`.
            public unsafe MR.PlanarTriangulation.IntersectionsMap Assign(MR.PlanarTriangulation._ByValue_IntersectionsMap _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_IntersectionsMap_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.IntersectionsMap._Underlying *__MR_PlanarTriangulation_IntersectionsMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.PlanarTriangulation.IntersectionsMap._Underlying *_other);
                return new(__MR_PlanarTriangulation_IntersectionsMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `IntersectionsMap` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `IntersectionsMap`/`Const_IntersectionsMap` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_IntersectionsMap
        {
            internal readonly Const_IntersectionsMap? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_IntersectionsMap() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_IntersectionsMap(Const_IntersectionsMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_IntersectionsMap(Const_IntersectionsMap arg) {return new(arg);}
            public _ByValue_IntersectionsMap(MR.Misc._Moved<IntersectionsMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_IntersectionsMap(MR.Misc._Moved<IntersectionsMap> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `IntersectionsMap` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_IntersectionsMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `IntersectionsMap`/`Const_IntersectionsMap` directly.
        public class _InOptMut_IntersectionsMap
        {
            public IntersectionsMap? Opt;

            public _InOptMut_IntersectionsMap() {}
            public _InOptMut_IntersectionsMap(IntersectionsMap value) {Opt = value;}
            public static implicit operator _InOptMut_IntersectionsMap(IntersectionsMap value) {return new(value);}
        }

        /// This is used for optional parameters of class `IntersectionsMap` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_IntersectionsMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `IntersectionsMap`/`Const_IntersectionsMap` to pass it to the function.
        public class _InOptConst_IntersectionsMap
        {
            public Const_IntersectionsMap? Opt;

            public _InOptConst_IntersectionsMap() {}
            public _InOptConst_IntersectionsMap(Const_IntersectionsMap value) {Opt = value;}
            public static implicit operator _InOptConst_IntersectionsMap(Const_IntersectionsMap value) {return new(value);}
        }

        /// Generated from class `MR::PlanarTriangulation::BaseOutlineParameters`.
        /// This is the const half of the class.
        public class Const_BaseOutlineParameters : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_BaseOutlineParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_Destroy", ExactSpelling = true)]
                extern static void __MR_PlanarTriangulation_BaseOutlineParameters_Destroy(_Underlying *_this);
                __MR_PlanarTriangulation_BaseOutlineParameters_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_BaseOutlineParameters() {Dispose(false);}

            ///< allow to merge vertices with same coordinates
            public unsafe bool AllowMerge
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_Get_allowMerge", ExactSpelling = true)]
                    extern static bool *__MR_PlanarTriangulation_BaseOutlineParameters_Get_allowMerge(_Underlying *_this);
                    return *__MR_PlanarTriangulation_BaseOutlineParameters_Get_allowMerge(_UnderlyingPtr);
                }
            }

            ///< what to mark as inner part
            public unsafe MR.PlanarTriangulation.WindingMode InnerType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_Get_innerType", ExactSpelling = true)]
                    extern static MR.PlanarTriangulation.WindingMode *__MR_PlanarTriangulation_BaseOutlineParameters_Get_innerType(_Underlying *_this);
                    return *__MR_PlanarTriangulation_BaseOutlineParameters_Get_innerType(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_BaseOutlineParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_BaseOutlineParameters_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::BaseOutlineParameters` elementwise.
            public unsafe Const_BaseOutlineParameters(bool allowMerge, MR.PlanarTriangulation.WindingMode innerType) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_ConstructFrom(byte allowMerge, MR.PlanarTriangulation.WindingMode innerType);
                _UnderlyingPtr = __MR_PlanarTriangulation_BaseOutlineParameters_ConstructFrom(allowMerge ? (byte)1 : (byte)0, innerType);
            }

            /// Generated from constructor `MR::PlanarTriangulation::BaseOutlineParameters::BaseOutlineParameters`.
            public unsafe Const_BaseOutlineParameters(MR.PlanarTriangulation.Const_BaseOutlineParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_ConstructFromAnother(MR.PlanarTriangulation.BaseOutlineParameters._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_BaseOutlineParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::PlanarTriangulation::BaseOutlineParameters`.
        /// This is the non-const half of the class.
        public class BaseOutlineParameters : Const_BaseOutlineParameters
        {
            internal unsafe BaseOutlineParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            ///< allow to merge vertices with same coordinates
            public new unsafe ref bool AllowMerge
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_GetMutable_allowMerge", ExactSpelling = true)]
                    extern static bool *__MR_PlanarTriangulation_BaseOutlineParameters_GetMutable_allowMerge(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_BaseOutlineParameters_GetMutable_allowMerge(_UnderlyingPtr);
                }
            }

            ///< what to mark as inner part
            public new unsafe ref MR.PlanarTriangulation.WindingMode InnerType
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_GetMutable_innerType", ExactSpelling = true)]
                    extern static MR.PlanarTriangulation.WindingMode *__MR_PlanarTriangulation_BaseOutlineParameters_GetMutable_innerType(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_BaseOutlineParameters_GetMutable_innerType(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe BaseOutlineParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_BaseOutlineParameters_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::BaseOutlineParameters` elementwise.
            public unsafe BaseOutlineParameters(bool allowMerge, MR.PlanarTriangulation.WindingMode innerType) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_ConstructFrom(byte allowMerge, MR.PlanarTriangulation.WindingMode innerType);
                _UnderlyingPtr = __MR_PlanarTriangulation_BaseOutlineParameters_ConstructFrom(allowMerge ? (byte)1 : (byte)0, innerType);
            }

            /// Generated from constructor `MR::PlanarTriangulation::BaseOutlineParameters::BaseOutlineParameters`.
            public unsafe BaseOutlineParameters(MR.PlanarTriangulation.Const_BaseOutlineParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_ConstructFromAnother(MR.PlanarTriangulation.BaseOutlineParameters._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_BaseOutlineParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::PlanarTriangulation::BaseOutlineParameters::operator=`.
            public unsafe MR.PlanarTriangulation.BaseOutlineParameters Assign(MR.PlanarTriangulation.Const_BaseOutlineParameters _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_BaseOutlineParameters_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_BaseOutlineParameters_AssignFromAnother(_Underlying *_this, MR.PlanarTriangulation.BaseOutlineParameters._Underlying *_other);
                return new(__MR_PlanarTriangulation_BaseOutlineParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `BaseOutlineParameters` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_BaseOutlineParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BaseOutlineParameters`/`Const_BaseOutlineParameters` directly.
        public class _InOptMut_BaseOutlineParameters
        {
            public BaseOutlineParameters? Opt;

            public _InOptMut_BaseOutlineParameters() {}
            public _InOptMut_BaseOutlineParameters(BaseOutlineParameters value) {Opt = value;}
            public static implicit operator _InOptMut_BaseOutlineParameters(BaseOutlineParameters value) {return new(value);}
        }

        /// This is used for optional parameters of class `BaseOutlineParameters` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_BaseOutlineParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `BaseOutlineParameters`/`Const_BaseOutlineParameters` to pass it to the function.
        public class _InOptConst_BaseOutlineParameters
        {
            public Const_BaseOutlineParameters? Opt;

            public _InOptConst_BaseOutlineParameters() {}
            public _InOptConst_BaseOutlineParameters(Const_BaseOutlineParameters value) {Opt = value;}
            public static implicit operator _InOptConst_BaseOutlineParameters(Const_BaseOutlineParameters value) {return new(value);}
        }

        /// Generated from class `MR::PlanarTriangulation::OutlineParameters`.
        /// This is the const half of the class.
        public class Const_OutlineParameters : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_OutlineParameters(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_Destroy", ExactSpelling = true)]
                extern static void __MR_PlanarTriangulation_OutlineParameters_Destroy(_Underlying *_this);
                __MR_PlanarTriangulation_OutlineParameters_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_OutlineParameters() {Dispose(false);}

            ///< optional output from result contour ids to input ones
            public unsafe ref void * IndicesMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_Get_indicesMap", ExactSpelling = true)]
                    extern static void **__MR_PlanarTriangulation_OutlineParameters_Get_indicesMap(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_OutlineParameters_Get_indicesMap(_UnderlyingPtr);
                }
            }

            public unsafe MR.PlanarTriangulation.Const_BaseOutlineParameters BaseParams
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_Get_baseParams", ExactSpelling = true)]
                    extern static MR.PlanarTriangulation.Const_BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_Get_baseParams(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_OutlineParameters_Get_baseParams(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_OutlineParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_OutlineParameters_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::OutlineParameters` elementwise.
            public unsafe Const_OutlineParameters(MR.Std.Vector_StdVectorMRPlanarTriangulationIntersectionInfo? indicesMap, MR.PlanarTriangulation.Const_BaseOutlineParameters baseParams) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_ConstructFrom(MR.Std.Vector_StdVectorMRPlanarTriangulationIntersectionInfo._Underlying *indicesMap, MR.PlanarTriangulation.BaseOutlineParameters._Underlying *baseParams);
                _UnderlyingPtr = __MR_PlanarTriangulation_OutlineParameters_ConstructFrom(indicesMap is not null ? indicesMap._UnderlyingPtr : null, baseParams._UnderlyingPtr);
            }

            /// Generated from constructor `MR::PlanarTriangulation::OutlineParameters::OutlineParameters`.
            public unsafe Const_OutlineParameters(MR.PlanarTriangulation.Const_OutlineParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_ConstructFromAnother(MR.PlanarTriangulation.OutlineParameters._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_OutlineParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::PlanarTriangulation::OutlineParameters`.
        /// This is the non-const half of the class.
        public class OutlineParameters : Const_OutlineParameters
        {
            internal unsafe OutlineParameters(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            ///< optional output from result contour ids to input ones
            public new unsafe ref void * IndicesMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_GetMutable_indicesMap", ExactSpelling = true)]
                    extern static void **__MR_PlanarTriangulation_OutlineParameters_GetMutable_indicesMap(_Underlying *_this);
                    return ref *__MR_PlanarTriangulation_OutlineParameters_GetMutable_indicesMap(_UnderlyingPtr);
                }
            }

            public new unsafe MR.PlanarTriangulation.BaseOutlineParameters BaseParams
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_GetMutable_baseParams", ExactSpelling = true)]
                    extern static MR.PlanarTriangulation.BaseOutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_GetMutable_baseParams(_Underlying *_this);
                    return new(__MR_PlanarTriangulation_OutlineParameters_GetMutable_baseParams(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe OutlineParameters() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_DefaultConstruct", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_DefaultConstruct();
                _UnderlyingPtr = __MR_PlanarTriangulation_OutlineParameters_DefaultConstruct();
            }

            /// Constructs `MR::PlanarTriangulation::OutlineParameters` elementwise.
            public unsafe OutlineParameters(MR.Std.Vector_StdVectorMRPlanarTriangulationIntersectionInfo? indicesMap, MR.PlanarTriangulation.Const_BaseOutlineParameters baseParams) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_ConstructFrom", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_ConstructFrom(MR.Std.Vector_StdVectorMRPlanarTriangulationIntersectionInfo._Underlying *indicesMap, MR.PlanarTriangulation.BaseOutlineParameters._Underlying *baseParams);
                _UnderlyingPtr = __MR_PlanarTriangulation_OutlineParameters_ConstructFrom(indicesMap is not null ? indicesMap._UnderlyingPtr : null, baseParams._UnderlyingPtr);
            }

            /// Generated from constructor `MR::PlanarTriangulation::OutlineParameters::OutlineParameters`.
            public unsafe OutlineParameters(MR.PlanarTriangulation.Const_OutlineParameters _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_ConstructFromAnother(MR.PlanarTriangulation.OutlineParameters._Underlying *_other);
                _UnderlyingPtr = __MR_PlanarTriangulation_OutlineParameters_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::PlanarTriangulation::OutlineParameters::operator=`.
            public unsafe MR.PlanarTriangulation.OutlineParameters Assign(MR.PlanarTriangulation.Const_OutlineParameters _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_OutlineParameters_AssignFromAnother", ExactSpelling = true)]
                extern static MR.PlanarTriangulation.OutlineParameters._Underlying *__MR_PlanarTriangulation_OutlineParameters_AssignFromAnother(_Underlying *_this, MR.PlanarTriangulation.OutlineParameters._Underlying *_other);
                return new(__MR_PlanarTriangulation_OutlineParameters_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `OutlineParameters` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_OutlineParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `OutlineParameters`/`Const_OutlineParameters` directly.
        public class _InOptMut_OutlineParameters
        {
            public OutlineParameters? Opt;

            public _InOptMut_OutlineParameters() {}
            public _InOptMut_OutlineParameters(OutlineParameters value) {Opt = value;}
            public static implicit operator _InOptMut_OutlineParameters(OutlineParameters value) {return new(value);}
        }

        /// This is used for optional parameters of class `OutlineParameters` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_OutlineParameters`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `OutlineParameters`/`Const_OutlineParameters` to pass it to the function.
        public class _InOptConst_OutlineParameters
        {
            public Const_OutlineParameters? Opt;

            public _InOptConst_OutlineParameters() {}
            public _InOptConst_OutlineParameters(Const_OutlineParameters value) {Opt = value;}
            public static implicit operator _InOptConst_OutlineParameters(Const_OutlineParameters value) {return new(value);}
        }

        /// return vertices of holes that correspond internal contours representation of PlanarTriangulation
        /// Generated from function `MR::PlanarTriangulation::findHoleVertIdsByHoleEdges`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVertId> FindHoleVertIdsByHoleEdges(MR.Const_MeshTopology tp, MR.Std.Const_Vector_StdVectorMREdgeId holePaths)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_findHoleVertIdsByHoleEdges", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMRVertId._Underlying *__MR_PlanarTriangulation_findHoleVertIdsByHoleEdges(MR.Const_MeshTopology._Underlying *tp, MR.Std.Const_Vector_StdVectorMREdgeId._Underlying *holePaths);
            return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVertId(__MR_PlanarTriangulation_findHoleVertIdsByHoleEdges(tp._UnderlyingPtr, holePaths._UnderlyingPtr), is_owning: true));
        }

        /// returns Mesh with boundaries representing outline if input contours
        /// interMap optional output intersection map
        /// Generated from function `MR::PlanarTriangulation::getOutlineMesh`.
        /// Parameter `params_` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Mesh> GetOutlineMesh(MR.Std.Const_Vector_StdVectorMRVector2f contours, MR.PlanarTriangulation.IntersectionsMap? interMap = null, MR.PlanarTriangulation.Const_BaseOutlineParameters? params_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_getOutlineMesh_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_PlanarTriangulation_getOutlineMesh_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, MR.PlanarTriangulation.IntersectionsMap._Underlying *interMap, MR.PlanarTriangulation.Const_BaseOutlineParameters._Underlying *params_);
            return MR.Misc.Move(new MR.Mesh(__MR_PlanarTriangulation_getOutlineMesh_std_vector_std_vector_MR_Vector2f(contours._UnderlyingPtr, interMap is not null ? interMap._UnderlyingPtr : null, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::PlanarTriangulation::getOutlineMesh`.
        /// Parameter `params_` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Mesh> GetOutlineMesh(MR.Std.Const_Vector_StdVectorMRVector2d contours, MR.PlanarTriangulation.IntersectionsMap? interMap = null, MR.PlanarTriangulation.Const_BaseOutlineParameters? params_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_getOutlineMesh_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_PlanarTriangulation_getOutlineMesh_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *contours, MR.PlanarTriangulation.IntersectionsMap._Underlying *interMap, MR.PlanarTriangulation.Const_BaseOutlineParameters._Underlying *params_);
            return MR.Misc.Move(new MR.Mesh(__MR_PlanarTriangulation_getOutlineMesh_std_vector_std_vector_MR_Vector2d(contours._UnderlyingPtr, interMap is not null ? interMap._UnderlyingPtr : null, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
        }

        /// returns Contour representing outline if input contours
        /// Generated from function `MR::PlanarTriangulation::getOutline`.
        /// Parameter `params_` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> GetOutline(MR.Std.Const_Vector_StdVectorMRVector2f contours, MR.PlanarTriangulation.Const_OutlineParameters? params_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_getOutline_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_PlanarTriangulation_getOutline_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, MR.PlanarTriangulation.Const_OutlineParameters._Underlying *params_);
            return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_PlanarTriangulation_getOutline_std_vector_std_vector_MR_Vector2f(contours._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::PlanarTriangulation::getOutline`.
        /// Parameter `params_` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_StdVectorMRVector2f> GetOutline(MR.Std.Const_Vector_StdVectorMRVector2d contours, MR.PlanarTriangulation.Const_OutlineParameters? params_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_getOutline_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Std.Vector_StdVectorMRVector2f._Underlying *__MR_PlanarTriangulation_getOutline_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *contours, MR.PlanarTriangulation.Const_OutlineParameters._Underlying *params_);
            return MR.Misc.Move(new MR.Std.Vector_StdVectorMRVector2f(__MR_PlanarTriangulation_getOutline_std_vector_std_vector_MR_Vector2d(contours._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
        }

        /**
        * @brief triangulate 2d contours
        * only closed contours are allowed (first point of each contour should be the same as last point of the contour)
        * @param holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates
        * @return return created mesh
        */
        /// Generated from function `MR::PlanarTriangulation::triangulateContours`.
        public static unsafe MR.Misc._Moved<MR.Mesh> TriangulateContours(MR.Std.Const_Vector_StdVectorMRVector2d contours, MR.Std.Const_Vector_StdVectorMRVertId? holeVertsIds = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *contours, MR.Std.Const_Vector_StdVectorMRVertId._Underlying *holeVertsIds);
            return MR.Misc.Move(new MR.Mesh(__MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2d(contours._UnderlyingPtr, holeVertsIds is not null ? holeVertsIds._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::PlanarTriangulation::triangulateContours`.
        public static unsafe MR.Misc._Moved<MR.Mesh> TriangulateContours(MR.Std.Const_Vector_StdVectorMRVector2f contours, MR.Std.Const_Vector_StdVectorMRVertId? holeVertsIds = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Mesh._Underlying *__MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, MR.Std.Const_Vector_StdVectorMRVertId._Underlying *holeVertsIds);
            return MR.Misc.Move(new MR.Mesh(__MR_PlanarTriangulation_triangulateContours_std_vector_std_vector_MR_Vector2f(contours._UnderlyingPtr, holeVertsIds is not null ? holeVertsIds._UnderlyingPtr : null), is_owning: true));
        }

        /**
        * @brief triangulate 2d contours
        * only closed contours are allowed (first point of each contour should be the same as last point of the contour)
        * @param holeVertsIds if set merge only points with same vertex id, otherwise merge all points with same coordinates
        * @param outBoundaries optional output EdgePaths that correspond to initial contours
        * @return std::optional<Mesh> : if some contours intersect return false, otherwise return created mesh
        */
        /// Generated from function `MR::PlanarTriangulation::triangulateDisjointContours`.
        public static unsafe MR.Misc._Moved<MR.Std.Optional_MRMesh> TriangulateDisjointContours(MR.Std.Const_Vector_StdVectorMRVector2d contours, MR.Std.Const_Vector_StdVectorMRVertId? holeVertsIds = null, MR.Std.Vector_StdVectorMREdgeId? outBoundaries = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_triangulateDisjointContours_std_vector_std_vector_MR_Vector2d", ExactSpelling = true)]
            extern static MR.Std.Optional_MRMesh._Underlying *__MR_PlanarTriangulation_triangulateDisjointContours_std_vector_std_vector_MR_Vector2d(MR.Std.Const_Vector_StdVectorMRVector2d._Underlying *contours, MR.Std.Const_Vector_StdVectorMRVertId._Underlying *holeVertsIds, MR.Std.Vector_StdVectorMREdgeId._Underlying *outBoundaries);
            return MR.Misc.Move(new MR.Std.Optional_MRMesh(__MR_PlanarTriangulation_triangulateDisjointContours_std_vector_std_vector_MR_Vector2d(contours._UnderlyingPtr, holeVertsIds is not null ? holeVertsIds._UnderlyingPtr : null, outBoundaries is not null ? outBoundaries._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::PlanarTriangulation::triangulateDisjointContours`.
        public static unsafe MR.Misc._Moved<MR.Std.Optional_MRMesh> TriangulateDisjointContours(MR.Std.Const_Vector_StdVectorMRVector2f contours, MR.Std.Const_Vector_StdVectorMRVertId? holeVertsIds = null, MR.Std.Vector_StdVectorMREdgeId? outBoundaries = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_PlanarTriangulation_triangulateDisjointContours_std_vector_std_vector_MR_Vector2f", ExactSpelling = true)]
            extern static MR.Std.Optional_MRMesh._Underlying *__MR_PlanarTriangulation_triangulateDisjointContours_std_vector_std_vector_MR_Vector2f(MR.Std.Const_Vector_StdVectorMRVector2f._Underlying *contours, MR.Std.Const_Vector_StdVectorMRVertId._Underlying *holeVertsIds, MR.Std.Vector_StdVectorMREdgeId._Underlying *outBoundaries);
            return MR.Misc.Move(new MR.Std.Optional_MRMesh(__MR_PlanarTriangulation_triangulateDisjointContours_std_vector_std_vector_MR_Vector2f(contours._UnderlyingPtr, holeVertsIds is not null ? holeVertsIds._UnderlyingPtr : null, outBoundaries is not null ? outBoundaries._UnderlyingPtr : null), is_owning: true));
        }
    }
}
