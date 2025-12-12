public static partial class MR
{
    public static partial class TriangulationHelpers
    {
        /// Generated from class `MR::TriangulationHelpers::FanOptimizerQueueElement`.
        /// This is the const half of the class.
        public class Const_FanOptimizerQueueElement : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.TriangulationHelpers.Const_FanOptimizerQueueElement>
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_FanOptimizerQueueElement(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_Destroy", ExactSpelling = true)]
                extern static void __MR_TriangulationHelpers_FanOptimizerQueueElement_Destroy(_Underlying *_this);
                __MR_TriangulationHelpers_FanOptimizerQueueElement_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_FanOptimizerQueueElement() {Dispose(false);}

            // profit of flipping this edge
            public unsafe float Weight
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_Get_weight", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_weight(_Underlying *_this);
                    return *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_weight(_UnderlyingPtr);
                }
            }

            // index
            public unsafe int Id
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_Get_id", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_id(_Underlying *_this);
                    return *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_id(_UnderlyingPtr);
                }
            }

            // id of prev neighbor
            public unsafe int PrevId
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_Get_prevId", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_prevId(_Underlying *_this);
                    return *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_prevId(_UnderlyingPtr);
                }
            }

            // id of next neighbor
            public unsafe int NextId
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_Get_nextId", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_nextId(_Underlying *_this);
                    return *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_nextId(_UnderlyingPtr);
                }
            }

            // if this flag is true, edge cannot be flipped
            public unsafe bool Stable
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_Get_stable", ExactSpelling = true)]
                    extern static bool *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_stable(_Underlying *_this);
                    return *__MR_TriangulationHelpers_FanOptimizerQueueElement_Get_stable(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_FanOptimizerQueueElement() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_DefaultConstruct();
                _UnderlyingPtr = __MR_TriangulationHelpers_FanOptimizerQueueElement_DefaultConstruct();
            }

            /// Constructs `MR::TriangulationHelpers::FanOptimizerQueueElement` elementwise.
            public unsafe Const_FanOptimizerQueueElement(float weight, int id, int prevId, int nextId, bool stable) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFrom", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFrom(float weight, int id, int prevId, int nextId, byte stable);
                _UnderlyingPtr = __MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFrom(weight, id, prevId, nextId, stable ? (byte)1 : (byte)0);
            }

            /// Generated from constructor `MR::TriangulationHelpers::FanOptimizerQueueElement::FanOptimizerQueueElement`.
            public unsafe Const_FanOptimizerQueueElement(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFromAnother(MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *_other);
                _UnderlyingPtr = __MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::TriangulationHelpers::FanOptimizerQueueElement::operator<`.
            public static unsafe bool operator<(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_less_MR_TriangulationHelpers_FanOptimizerQueueElement", ExactSpelling = true)]
                extern static byte __MR_less_MR_TriangulationHelpers_FanOptimizerQueueElement(MR.TriangulationHelpers.Const_FanOptimizerQueueElement._Underlying *_this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement._Underlying *other);
                return __MR_less_MR_TriangulationHelpers_FanOptimizerQueueElement(_this._UnderlyingPtr, other._UnderlyingPtr) != 0;
            }

            public static unsafe bool operator>(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement other)
            {
                return other < _this;
            }

            public static unsafe bool operator<=(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement other)
            {
                return !(other < _this);
            }

            public static unsafe bool operator>=(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement other)
            {
                return !(_this < other);
            }

            /// Generated from method `MR::TriangulationHelpers::FanOptimizerQueueElement::operator==`.
            public static unsafe bool operator==(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_TriangulationHelpers_FanOptimizerQueueElement", ExactSpelling = true)]
                extern static byte __MR_equal_MR_TriangulationHelpers_FanOptimizerQueueElement(MR.TriangulationHelpers.Const_FanOptimizerQueueElement._Underlying *_this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement._Underlying *other);
                return __MR_equal_MR_TriangulationHelpers_FanOptimizerQueueElement(_this._UnderlyingPtr, other._UnderlyingPtr) != 0;
            }

            public static unsafe bool operator!=(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _this, MR.TriangulationHelpers.Const_FanOptimizerQueueElement other)
            {
                return !(_this == other);
            }

            /// Generated from method `MR::TriangulationHelpers::FanOptimizerQueueElement::isOutdated`.
            public unsafe bool IsOutdated(MR.Std.Const_Vector_MRVertId neighbors)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_isOutdated", ExactSpelling = true)]
                extern static byte __MR_TriangulationHelpers_FanOptimizerQueueElement_isOutdated(_Underlying *_this, MR.Std.Const_Vector_MRVertId._Underlying *neighbors);
                return __MR_TriangulationHelpers_FanOptimizerQueueElement_isOutdated(_UnderlyingPtr, neighbors._UnderlyingPtr) != 0;
            }

            // IEquatable:

            public bool Equals(MR.TriangulationHelpers.Const_FanOptimizerQueueElement? other)
            {
                if (other is null)
                    return false;
                return this == other;
            }

            public override bool Equals(object? other)
            {
                if (other is null)
                    return false;
                if (other is MR.TriangulationHelpers.Const_FanOptimizerQueueElement)
                    return this == (MR.TriangulationHelpers.Const_FanOptimizerQueueElement)other;
                return false;
            }
        }

        /// Generated from class `MR::TriangulationHelpers::FanOptimizerQueueElement`.
        /// This is the non-const half of the class.
        public class FanOptimizerQueueElement : Const_FanOptimizerQueueElement
        {
            internal unsafe FanOptimizerQueueElement(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // profit of flipping this edge
            public new unsafe ref float Weight
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_weight", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_weight(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_weight(_UnderlyingPtr);
                }
            }

            // index
            public new unsafe ref int Id
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_id", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_id(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_id(_UnderlyingPtr);
                }
            }

            // id of prev neighbor
            public new unsafe ref int PrevId
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_prevId", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_prevId(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_prevId(_UnderlyingPtr);
                }
            }

            // id of next neighbor
            public new unsafe ref int NextId
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_nextId", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_nextId(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_nextId(_UnderlyingPtr);
                }
            }

            // if this flag is true, edge cannot be flipped
            public new unsafe ref bool Stable
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_stable", ExactSpelling = true)]
                    extern static bool *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_stable(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_FanOptimizerQueueElement_GetMutable_stable(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe FanOptimizerQueueElement() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_DefaultConstruct();
                _UnderlyingPtr = __MR_TriangulationHelpers_FanOptimizerQueueElement_DefaultConstruct();
            }

            /// Constructs `MR::TriangulationHelpers::FanOptimizerQueueElement` elementwise.
            public unsafe FanOptimizerQueueElement(float weight, int id, int prevId, int nextId, bool stable) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFrom", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFrom(float weight, int id, int prevId, int nextId, byte stable);
                _UnderlyingPtr = __MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFrom(weight, id, prevId, nextId, stable ? (byte)1 : (byte)0);
            }

            /// Generated from constructor `MR::TriangulationHelpers::FanOptimizerQueueElement::FanOptimizerQueueElement`.
            public unsafe FanOptimizerQueueElement(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFromAnother(MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *_other);
                _UnderlyingPtr = __MR_TriangulationHelpers_FanOptimizerQueueElement_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::TriangulationHelpers::FanOptimizerQueueElement::operator=`.
            public unsafe MR.TriangulationHelpers.FanOptimizerQueueElement Assign(MR.TriangulationHelpers.Const_FanOptimizerQueueElement _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_FanOptimizerQueueElement_AssignFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *__MR_TriangulationHelpers_FanOptimizerQueueElement_AssignFromAnother(_Underlying *_this, MR.TriangulationHelpers.FanOptimizerQueueElement._Underlying *_other);
                return new(__MR_TriangulationHelpers_FanOptimizerQueueElement_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `FanOptimizerQueueElement` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_FanOptimizerQueueElement`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FanOptimizerQueueElement`/`Const_FanOptimizerQueueElement` directly.
        public class _InOptMut_FanOptimizerQueueElement
        {
            public FanOptimizerQueueElement? Opt;

            public _InOptMut_FanOptimizerQueueElement() {}
            public _InOptMut_FanOptimizerQueueElement(FanOptimizerQueueElement value) {Opt = value;}
            public static implicit operator _InOptMut_FanOptimizerQueueElement(FanOptimizerQueueElement value) {return new(value);}
        }

        /// This is used for optional parameters of class `FanOptimizerQueueElement` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_FanOptimizerQueueElement`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `FanOptimizerQueueElement`/`Const_FanOptimizerQueueElement` to pass it to the function.
        public class _InOptConst_FanOptimizerQueueElement
        {
            public Const_FanOptimizerQueueElement? Opt;

            public _InOptConst_FanOptimizerQueueElement() {}
            public _InOptConst_FanOptimizerQueueElement(Const_FanOptimizerQueueElement value) {Opt = value;}
            public static implicit operator _InOptConst_FanOptimizerQueueElement(Const_FanOptimizerQueueElement value) {return new(value);}
        }

        /**
        * \brief Data with caches for optimizing fan triangulation
        *
        */
        /// Generated from class `MR::TriangulationHelpers::TriangulatedFanData`.
        /// This is the const half of the class.
        public class Const_TriangulatedFanData : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_TriangulatedFanData(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_Destroy", ExactSpelling = true)]
                extern static void __MR_TriangulationHelpers_TriangulatedFanData_Destroy(_Underlying *_this);
                __MR_TriangulationHelpers_TriangulatedFanData_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_TriangulatedFanData() {Dispose(false);}

            /// clockwise points around center point in (optimized) triangle fan,
            /// each pair of points (as well as back()-front() pair) together with the center form a fan triangle
            public unsafe MR.Std.Const_Vector_MRVertId Neighbors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_Get_neighbors", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_MRVertId._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_Get_neighbors(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_Get_neighbors(_UnderlyingPtr), is_owning: false);
                }
            }

            /// temporary reusable storage to avoid allocations for each point
            public unsafe MR.Std.Const_Vector_StdPairDoubleInt CacheAngleOrder
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_Get_cacheAngleOrder", ExactSpelling = true)]
                    extern static MR.Std.Const_Vector_StdPairDoubleInt._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_Get_cacheAngleOrder(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_Get_cacheAngleOrder(_UnderlyingPtr), is_owning: false);
                }
            }

            /// first border edge (invalid if the center point is not on the boundary)
            /// triangle associated with this point is absent
            public unsafe MR.Const_VertId Border
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_Get_border", ExactSpelling = true)]
                    extern static MR.Const_VertId._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_Get_border(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_Get_border(_UnderlyingPtr), is_owning: false);
                }
            }

            /// the storage to collect n-nearest neighbours, here to avoid allocations for each point
            public unsafe MR.Const_FewSmallest_MRPointsProjectionResult NearesetPoints
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_Get_nearesetPoints", ExactSpelling = true)]
                    extern static MR.Const_FewSmallest_MRPointsProjectionResult._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_Get_nearesetPoints(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_Get_nearesetPoints(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_TriangulatedFanData() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.TriangulatedFanData._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_DefaultConstruct();
                _UnderlyingPtr = __MR_TriangulationHelpers_TriangulatedFanData_DefaultConstruct();
            }

            /// Generated from constructor `MR::TriangulationHelpers::TriangulatedFanData::TriangulatedFanData`.
            public unsafe Const_TriangulatedFanData(MR.TriangulationHelpers._ByValue_TriangulatedFanData _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.TriangulatedFanData._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TriangulationHelpers.TriangulatedFanData._Underlying *_other);
                _UnderlyingPtr = __MR_TriangulationHelpers_TriangulatedFanData_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /**
        * \brief Data with caches for optimizing fan triangulation
        *
        */
        /// Generated from class `MR::TriangulationHelpers::TriangulatedFanData`.
        /// This is the non-const half of the class.
        public class TriangulatedFanData : Const_TriangulatedFanData
        {
            internal unsafe TriangulatedFanData(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// clockwise points around center point in (optimized) triangle fan,
            /// each pair of points (as well as back()-front() pair) together with the center form a fan triangle
            public new unsafe MR.Std.Vector_MRVertId Neighbors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_GetMutable_neighbors", ExactSpelling = true)]
                    extern static MR.Std.Vector_MRVertId._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_neighbors(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_neighbors(_UnderlyingPtr), is_owning: false);
                }
            }

            /// temporary reusable storage to avoid allocations for each point
            public new unsafe MR.Std.Vector_StdPairDoubleInt CacheAngleOrder
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_GetMutable_cacheAngleOrder", ExactSpelling = true)]
                    extern static MR.Std.Vector_StdPairDoubleInt._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_cacheAngleOrder(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_cacheAngleOrder(_UnderlyingPtr), is_owning: false);
                }
            }

            /// first border edge (invalid if the center point is not on the boundary)
            /// triangle associated with this point is absent
            public new unsafe MR.Mut_VertId Border
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_GetMutable_border", ExactSpelling = true)]
                    extern static MR.Mut_VertId._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_border(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_border(_UnderlyingPtr), is_owning: false);
                }
            }

            /// the storage to collect n-nearest neighbours, here to avoid allocations for each point
            public new unsafe MR.FewSmallest_MRPointsProjectionResult NearesetPoints
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_GetMutable_nearesetPoints", ExactSpelling = true)]
                    extern static MR.FewSmallest_MRPointsProjectionResult._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_nearesetPoints(_Underlying *_this);
                    return new(__MR_TriangulationHelpers_TriangulatedFanData_GetMutable_nearesetPoints(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe TriangulatedFanData() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.TriangulatedFanData._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_DefaultConstruct();
                _UnderlyingPtr = __MR_TriangulationHelpers_TriangulatedFanData_DefaultConstruct();
            }

            /// Generated from constructor `MR::TriangulationHelpers::TriangulatedFanData::TriangulatedFanData`.
            public unsafe TriangulatedFanData(MR.TriangulationHelpers._ByValue_TriangulatedFanData _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.TriangulatedFanData._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.TriangulationHelpers.TriangulatedFanData._Underlying *_other);
                _UnderlyingPtr = __MR_TriangulationHelpers_TriangulatedFanData_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::TriangulationHelpers::TriangulatedFanData::operator=`.
            public unsafe MR.TriangulationHelpers.TriangulatedFanData Assign(MR.TriangulationHelpers._ByValue_TriangulatedFanData _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_TriangulatedFanData_AssignFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.TriangulatedFanData._Underlying *__MR_TriangulationHelpers_TriangulatedFanData_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.TriangulationHelpers.TriangulatedFanData._Underlying *_other);
                return new(__MR_TriangulationHelpers_TriangulatedFanData_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `TriangulatedFanData` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `TriangulatedFanData`/`Const_TriangulatedFanData` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_TriangulatedFanData
        {
            internal readonly Const_TriangulatedFanData? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_TriangulatedFanData() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_TriangulatedFanData(Const_TriangulatedFanData new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_TriangulatedFanData(Const_TriangulatedFanData arg) {return new(arg);}
            public _ByValue_TriangulatedFanData(MR.Misc._Moved<TriangulatedFanData> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_TriangulatedFanData(MR.Misc._Moved<TriangulatedFanData> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `TriangulatedFanData` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_TriangulatedFanData`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `TriangulatedFanData`/`Const_TriangulatedFanData` directly.
        public class _InOptMut_TriangulatedFanData
        {
            public TriangulatedFanData? Opt;

            public _InOptMut_TriangulatedFanData() {}
            public _InOptMut_TriangulatedFanData(TriangulatedFanData value) {Opt = value;}
            public static implicit operator _InOptMut_TriangulatedFanData(TriangulatedFanData value) {return new(value);}
        }

        /// This is used for optional parameters of class `TriangulatedFanData` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_TriangulatedFanData`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `TriangulatedFanData`/`Const_TriangulatedFanData` to pass it to the function.
        public class _InOptConst_TriangulatedFanData
        {
            public Const_TriangulatedFanData? Opt;

            public _InOptConst_TriangulatedFanData() {}
            public _InOptConst_TriangulatedFanData(Const_TriangulatedFanData value) {Opt = value;}
            public static implicit operator _InOptConst_TriangulatedFanData(Const_TriangulatedFanData value) {return new(value);}
        }

        /// Generated from class `MR::TriangulationHelpers::Settings`.
        /// This is the const half of the class.
        public class Const_Settings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Settings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Destroy", ExactSpelling = true)]
                extern static void __MR_TriangulationHelpers_Settings_Destroy(_Underlying *_this);
                __MR_TriangulationHelpers_Settings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Settings() {Dispose(false);}

            /// initial radius of search for neighbours, it can be increased automatically;
            /// if radius is positive then numNeis must be zero
            public unsafe float Radius
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_radius", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_Settings_Get_radius(_Underlying *_this);
                    return *__MR_TriangulationHelpers_Settings_Get_radius(_UnderlyingPtr);
                }
            }

            /// initially selects given number of nearest neighbours;
            /// if numNeis is positive then radius must be zero
            public unsafe int NumNeis
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_numNeis", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_Settings_Get_numNeis(_Underlying *_this);
                    return *__MR_TriangulationHelpers_Settings_Get_numNeis(_UnderlyingPtr);
                }
            }

            /// max allowed angle for triangles in fan
            public unsafe float CritAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_critAngle", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_Settings_Get_critAngle(_Underlying *_this);
                    return *__MR_TriangulationHelpers_Settings_Get_critAngle(_UnderlyingPtr);
                }
            }

            /// the vertex is considered as boundary if its neighbor ring has angle more than this value
            public unsafe float BoundaryAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_boundaryAngle", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_Settings_Get_boundaryAngle(_Underlying *_this);
                    return *__MR_TriangulationHelpers_Settings_Get_boundaryAngle(_UnderlyingPtr);
                }
            }

            /// if oriented normals are known, they will be used for neighbor points selection
            /// except for the ones indicated by untrustedNormals
            public unsafe ref readonly void * OrientedNormals
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_orientedNormals", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_Get_orientedNormals(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_Get_orientedNormals(_UnderlyingPtr);
                }
            }

            public unsafe ref readonly void * UntrustedNormals
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_untrustedNormals", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_Get_untrustedNormals(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_Get_untrustedNormals(_UnderlyingPtr);
                }
            }

            /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
            public unsafe bool AutomaticRadiusIncrease
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_automaticRadiusIncrease", ExactSpelling = true)]
                    extern static bool *__MR_TriangulationHelpers_Settings_Get_automaticRadiusIncrease(_Underlying *_this);
                    return *__MR_TriangulationHelpers_Settings_Get_automaticRadiusIncrease(_UnderlyingPtr);
                }
            }

            /// the maximum number of optimization steps (removals) in local triangulation
            public unsafe int MaxRemoves
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_maxRemoves", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_Settings_Get_maxRemoves(_Underlying *_this);
                    return *__MR_TriangulationHelpers_Settings_Get_maxRemoves(_UnderlyingPtr);
                }
            }

            /// optional output of considered neighbor points after filtering but before triangulation/optimization
            public unsafe ref void * AllNeighbors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_allNeighbors", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_Get_allNeighbors(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_Get_allNeighbors(_UnderlyingPtr);
                }
            }

            /// optional output: actual radius of neighbor search (after increase if any)
            public unsafe ref float * ActualRadius
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_actualRadius", ExactSpelling = true)]
                    extern static float **__MR_TriangulationHelpers_Settings_Get_actualRadius(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_Get_actualRadius(_UnderlyingPtr);
                }
            }

            /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
            public unsafe ref readonly void * SearchNeighbors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_Get_searchNeighbors", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_Get_searchNeighbors(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_Get_searchNeighbors(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_TriangulationHelpers_Settings_DefaultConstruct();
            }

            /// Constructs `MR::TriangulationHelpers::Settings` elementwise.
            public unsafe Const_Settings(float radius, int numNeis, float critAngle, float boundaryAngle, MR.Const_VertCoords? orientedNormals, MR.Const_VertBitSet? untrustedNormals, bool automaticRadiusIncrease, int maxRemoves, MR.Std.Vector_MRVertId? allNeighbors, MR.Misc.InOut<float>? actualRadius, MR.Const_PointCloud? searchNeighbors) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_ConstructFrom(float radius, int numNeis, float critAngle, float boundaryAngle, MR.Const_VertCoords._Underlying *orientedNormals, MR.Const_VertBitSet._Underlying *untrustedNormals, byte automaticRadiusIncrease, int maxRemoves, MR.Std.Vector_MRVertId._Underlying *allNeighbors, float *actualRadius, MR.Const_PointCloud._Underlying *searchNeighbors);
                float __value_actualRadius = actualRadius is not null ? actualRadius.Value : default(float);
                _UnderlyingPtr = __MR_TriangulationHelpers_Settings_ConstructFrom(radius, numNeis, critAngle, boundaryAngle, orientedNormals is not null ? orientedNormals._UnderlyingPtr : null, untrustedNormals is not null ? untrustedNormals._UnderlyingPtr : null, automaticRadiusIncrease ? (byte)1 : (byte)0, maxRemoves, allNeighbors is not null ? allNeighbors._UnderlyingPtr : null, actualRadius is not null ? &__value_actualRadius : null, searchNeighbors is not null ? searchNeighbors._UnderlyingPtr : null);
                if (actualRadius is not null) actualRadius.Value = __value_actualRadius;
            }

            /// Generated from constructor `MR::TriangulationHelpers::Settings::Settings`.
            public unsafe Const_Settings(MR.TriangulationHelpers.Const_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_ConstructFromAnother(MR.TriangulationHelpers.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_TriangulationHelpers_Settings_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::TriangulationHelpers::Settings`.
        /// This is the non-const half of the class.
        public class Settings : Const_Settings
        {
            internal unsafe Settings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// initial radius of search for neighbours, it can be increased automatically;
            /// if radius is positive then numNeis must be zero
            public new unsafe ref float Radius
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_radius", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_Settings_GetMutable_radius(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_radius(_UnderlyingPtr);
                }
            }

            /// initially selects given number of nearest neighbours;
            /// if numNeis is positive then radius must be zero
            public new unsafe ref int NumNeis
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_numNeis", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_Settings_GetMutable_numNeis(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_numNeis(_UnderlyingPtr);
                }
            }

            /// max allowed angle for triangles in fan
            public new unsafe ref float CritAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_critAngle", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_Settings_GetMutable_critAngle(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_critAngle(_UnderlyingPtr);
                }
            }

            /// the vertex is considered as boundary if its neighbor ring has angle more than this value
            public new unsafe ref float BoundaryAngle
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_boundaryAngle", ExactSpelling = true)]
                    extern static float *__MR_TriangulationHelpers_Settings_GetMutable_boundaryAngle(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_boundaryAngle(_UnderlyingPtr);
                }
            }

            /// if oriented normals are known, they will be used for neighbor points selection
            /// except for the ones indicated by untrustedNormals
            public new unsafe ref readonly void * OrientedNormals
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_orientedNormals", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_GetMutable_orientedNormals(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_orientedNormals(_UnderlyingPtr);
                }
            }

            public new unsafe ref readonly void * UntrustedNormals
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_untrustedNormals", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_GetMutable_untrustedNormals(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_untrustedNormals(_UnderlyingPtr);
                }
            }

            /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
            public new unsafe ref bool AutomaticRadiusIncrease
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_automaticRadiusIncrease", ExactSpelling = true)]
                    extern static bool *__MR_TriangulationHelpers_Settings_GetMutable_automaticRadiusIncrease(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_automaticRadiusIncrease(_UnderlyingPtr);
                }
            }

            /// the maximum number of optimization steps (removals) in local triangulation
            public new unsafe ref int MaxRemoves
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_maxRemoves", ExactSpelling = true)]
                    extern static int *__MR_TriangulationHelpers_Settings_GetMutable_maxRemoves(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_maxRemoves(_UnderlyingPtr);
                }
            }

            /// optional output of considered neighbor points after filtering but before triangulation/optimization
            public new unsafe ref void * AllNeighbors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_allNeighbors", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_GetMutable_allNeighbors(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_allNeighbors(_UnderlyingPtr);
                }
            }

            /// optional output: actual radius of neighbor search (after increase if any)
            public new unsafe ref float * ActualRadius
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_actualRadius", ExactSpelling = true)]
                    extern static float **__MR_TriangulationHelpers_Settings_GetMutable_actualRadius(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_actualRadius(_UnderlyingPtr);
                }
            }

            /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
            public new unsafe ref readonly void * SearchNeighbors
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_GetMutable_searchNeighbors", ExactSpelling = true)]
                    extern static void **__MR_TriangulationHelpers_Settings_GetMutable_searchNeighbors(_Underlying *_this);
                    return ref *__MR_TriangulationHelpers_Settings_GetMutable_searchNeighbors(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Settings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_DefaultConstruct();
                _UnderlyingPtr = __MR_TriangulationHelpers_Settings_DefaultConstruct();
            }

            /// Constructs `MR::TriangulationHelpers::Settings` elementwise.
            public unsafe Settings(float radius, int numNeis, float critAngle, float boundaryAngle, MR.Const_VertCoords? orientedNormals, MR.Const_VertBitSet? untrustedNormals, bool automaticRadiusIncrease, int maxRemoves, MR.Std.Vector_MRVertId? allNeighbors, MR.Misc.InOut<float>? actualRadius, MR.Const_PointCloud? searchNeighbors) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_ConstructFrom", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_ConstructFrom(float radius, int numNeis, float critAngle, float boundaryAngle, MR.Const_VertCoords._Underlying *orientedNormals, MR.Const_VertBitSet._Underlying *untrustedNormals, byte automaticRadiusIncrease, int maxRemoves, MR.Std.Vector_MRVertId._Underlying *allNeighbors, float *actualRadius, MR.Const_PointCloud._Underlying *searchNeighbors);
                float __value_actualRadius = actualRadius is not null ? actualRadius.Value : default(float);
                _UnderlyingPtr = __MR_TriangulationHelpers_Settings_ConstructFrom(radius, numNeis, critAngle, boundaryAngle, orientedNormals is not null ? orientedNormals._UnderlyingPtr : null, untrustedNormals is not null ? untrustedNormals._UnderlyingPtr : null, automaticRadiusIncrease ? (byte)1 : (byte)0, maxRemoves, allNeighbors is not null ? allNeighbors._UnderlyingPtr : null, actualRadius is not null ? &__value_actualRadius : null, searchNeighbors is not null ? searchNeighbors._UnderlyingPtr : null);
                if (actualRadius is not null) actualRadius.Value = __value_actualRadius;
            }

            /// Generated from constructor `MR::TriangulationHelpers::Settings::Settings`.
            public unsafe Settings(MR.TriangulationHelpers.Const_Settings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_ConstructFromAnother(MR.TriangulationHelpers.Settings._Underlying *_other);
                _UnderlyingPtr = __MR_TriangulationHelpers_Settings_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::TriangulationHelpers::Settings::operator=`.
            public unsafe MR.TriangulationHelpers.Settings Assign(MR.TriangulationHelpers.Const_Settings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_Settings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.TriangulationHelpers.Settings._Underlying *__MR_TriangulationHelpers_Settings_AssignFromAnother(_Underlying *_this, MR.TriangulationHelpers.Settings._Underlying *_other);
                return new(__MR_TriangulationHelpers_Settings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Settings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Settings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Settings`/`Const_Settings` directly.
        public class _InOptMut_Settings
        {
            public Settings? Opt;

            public _InOptMut_Settings() {}
            public _InOptMut_Settings(Settings value) {Opt = value;}
            public static implicit operator _InOptMut_Settings(Settings value) {return new(value);}
        }

        /// This is used for optional parameters of class `Settings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Settings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Settings`/`Const_Settings` to pass it to the function.
        public class _InOptConst_Settings
        {
            public Const_Settings? Opt;

            public _InOptConst_Settings() {}
            public _InOptConst_Settings(Const_Settings value) {Opt = value;}
            public static implicit operator _InOptConst_Settings(Const_Settings value) {return new(value);}
        }

        /**
        * \brief Finds max radius of neighbors search, for possible better local triangulation
        * \param borderV first boundary vertex in \param fan (next VertId in fan is also boundary but first is enough)
        *
        */
        /// Generated from function `MR::TriangulationHelpers::updateNeighborsRadius`.
        public static unsafe float UpdateNeighborsRadius(MR.Const_VertCoords points, MR.VertId v, MR.VertId boundaryV, MR.Std.Const_Vector_MRVertId fan, float baseRadius)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_updateNeighborsRadius", ExactSpelling = true)]
            extern static float __MR_TriangulationHelpers_updateNeighborsRadius(MR.Const_VertCoords._Underlying *points, MR.VertId v, MR.VertId boundaryV, MR.Std.Const_Vector_MRVertId._Underlying *fan, float baseRadius);
            return __MR_TriangulationHelpers_updateNeighborsRadius(points._UnderlyingPtr, v, boundaryV, fan._UnderlyingPtr, baseRadius);
        }

        /**
        * \brief Finds all neighbors of v in given radius (v excluded)
        *
        */
        /// Generated from function `MR::TriangulationHelpers::findNeighborsInBall`.
        public static unsafe void FindNeighborsInBall(MR.Const_PointCloud pointCloud, MR.VertId v, float radius, MR.Std.Vector_MRVertId neighbors)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_findNeighborsInBall", ExactSpelling = true)]
            extern static void __MR_TriangulationHelpers_findNeighborsInBall(MR.Const_PointCloud._Underlying *pointCloud, MR.VertId v, float radius, MR.Std.Vector_MRVertId._Underlying *neighbors);
            __MR_TriangulationHelpers_findNeighborsInBall(pointCloud._UnderlyingPtr, v, radius, neighbors._UnderlyingPtr);
        }

        /**
        * \brief Finds at most given number of neighbors of v (v excluded)
        * \param tmp temporary storage to avoid its allocation
        * \param upDistLimitSq upper limit on the distance in question, points with larger distance than it will not be returned
        * \return maxDistSq to the furthest returned neighbor (or 0 if no neighbours are returned)
        *
        */
        /// Generated from function `MR::TriangulationHelpers::findNumNeighbors`.
        /// Parameter `upDistLimitSq` defaults to `3.40282347e38f`.
        public static unsafe float FindNumNeighbors(MR.Const_PointCloud pointCloud, MR.VertId v, int numNeis, MR.Std.Vector_MRVertId neighbors, MR.FewSmallest_MRPointsProjectionResult tmp, float? upDistLimitSq = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_findNumNeighbors", ExactSpelling = true)]
            extern static float __MR_TriangulationHelpers_findNumNeighbors(MR.Const_PointCloud._Underlying *pointCloud, MR.VertId v, int numNeis, MR.Std.Vector_MRVertId._Underlying *neighbors, MR.FewSmallest_MRPointsProjectionResult._Underlying *tmp, float *upDistLimitSq);
            float __deref_upDistLimitSq = upDistLimitSq.GetValueOrDefault();
            return __MR_TriangulationHelpers_findNumNeighbors(pointCloud._UnderlyingPtr, v, numNeis, neighbors._UnderlyingPtr, tmp._UnderlyingPtr, upDistLimitSq.HasValue ? &__deref_upDistLimitSq : null);
        }

        /**
        * \brief Filter neighbors with crossing normals
        *
        */
        /// Generated from function `MR::TriangulationHelpers::filterNeighbors`.
        public static unsafe void FilterNeighbors(MR.Const_VertCoords orientedNormals, MR.Const_VertBitSet? untrustedNormals, MR.VertId v, MR.Std.Vector_MRVertId neighbors)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_filterNeighbors", ExactSpelling = true)]
            extern static void __MR_TriangulationHelpers_filterNeighbors(MR.Const_VertCoords._Underlying *orientedNormals, MR.Const_VertBitSet._Underlying *untrustedNormals, MR.VertId v, MR.Std.Vector_MRVertId._Underlying *neighbors);
            __MR_TriangulationHelpers_filterNeighbors(orientedNormals._UnderlyingPtr, untrustedNormals is not null ? untrustedNormals._UnderlyingPtr : null, v, neighbors._UnderlyingPtr);
        }

        /// constructs local triangulation around given point
        /// Generated from function `MR::TriangulationHelpers::buildLocalTriangulation`.
        public static unsafe void BuildLocalTriangulation(MR.Const_PointCloud cloud, MR.VertId v, MR.TriangulationHelpers.Const_Settings settings, MR.TriangulationHelpers.TriangulatedFanData fanData)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_buildLocalTriangulation", ExactSpelling = true)]
            extern static void __MR_TriangulationHelpers_buildLocalTriangulation(MR.Const_PointCloud._Underlying *cloud, MR.VertId v, MR.TriangulationHelpers.Const_Settings._Underlying *settings, MR.TriangulationHelpers.TriangulatedFanData._Underlying *fanData);
            __MR_TriangulationHelpers_buildLocalTriangulation(cloud._UnderlyingPtr, v, settings._UnderlyingPtr, fanData._UnderlyingPtr);
        }

        /// computes all local triangulations of all points in the cloud, and returns them distributed among
        /// a set of SomeLocalTriangulations objects
        /// Generated from function `MR::TriangulationHelpers::buildLocalTriangulations`.
        /// Parameter `progress` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Optional_StdVectorMRSomeLocalTriangulations> BuildLocalTriangulations(MR.Const_PointCloud cloud, MR.TriangulationHelpers.Const_Settings settings, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_buildLocalTriangulations", ExactSpelling = true)]
            extern static MR.Std.Optional_StdVectorMRSomeLocalTriangulations._Underlying *__MR_TriangulationHelpers_buildLocalTriangulations(MR.Const_PointCloud._Underlying *cloud, MR.TriangulationHelpers.Const_Settings._Underlying *settings, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
            return MR.Misc.Move(new MR.Std.Optional_StdVectorMRSomeLocalTriangulations(__MR_TriangulationHelpers_buildLocalTriangulations(cloud._UnderlyingPtr, settings._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
        }

        //// computes local triangulations of all points in the cloud united in one struct
        /// Generated from function `MR::TriangulationHelpers::buildUnitedLocalTriangulations`.
        /// Parameter `progress` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Optional_MRAllLocalTriangulations> BuildUnitedLocalTriangulations(MR.Const_PointCloud cloud, MR.TriangulationHelpers.Const_Settings settings, MR.Std.Const_Function_BoolFuncFromFloat? progress = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_buildUnitedLocalTriangulations", ExactSpelling = true)]
            extern static MR.Std.Optional_MRAllLocalTriangulations._Underlying *__MR_TriangulationHelpers_buildUnitedLocalTriangulations(MR.Const_PointCloud._Underlying *cloud, MR.TriangulationHelpers.Const_Settings._Underlying *settings, MR.Std.Const_Function_BoolFuncFromFloat._Underlying *progress);
            return MR.Misc.Move(new MR.Std.Optional_MRAllLocalTriangulations(__MR_TriangulationHelpers_buildUnitedLocalTriangulations(cloud._UnderlyingPtr, settings._UnderlyingPtr, progress is not null ? progress._UnderlyingPtr : null), is_owning: true));
        }

        /**
        * \brief Checks if given vertex is on boundary of the point cloud, by constructing local triangulation around it
        *
        * \param cloud input point cloud
        * \param v vertex id to check
        * \param settings all parameters of the computation
        * \param fanData cache structure for neighbors, not to allocate for multiple calls
        * \returns true if vertex is boundary, false otherwise
        */
        /// Generated from function `MR::TriangulationHelpers::isBoundaryPoint`.
        public static unsafe bool IsBoundaryPoint(MR.Const_PointCloud cloud, MR.VertId v, MR.TriangulationHelpers.Const_Settings settings, MR.TriangulationHelpers.TriangulatedFanData fanData)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_isBoundaryPoint", ExactSpelling = true)]
            extern static byte __MR_TriangulationHelpers_isBoundaryPoint(MR.Const_PointCloud._Underlying *cloud, MR.VertId v, MR.TriangulationHelpers.Const_Settings._Underlying *settings, MR.TriangulationHelpers.TriangulatedFanData._Underlying *fanData);
            return __MR_TriangulationHelpers_isBoundaryPoint(cloud._UnderlyingPtr, v, settings._UnderlyingPtr, fanData._UnderlyingPtr) != 0;
        }

        /// Returns bit set of points that are considered as boundary by calling isBoundaryPoint in each
        /// Generated from function `MR::TriangulationHelpers::findBoundaryPoints`.
        /// Parameter `cb` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Std.Optional_MRVertBitSet> FindBoundaryPoints(MR.Const_PointCloud pointCloud, MR.TriangulationHelpers.Const_Settings settings, MR.Std._ByValue_Function_BoolFuncFromFloat? cb = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_TriangulationHelpers_findBoundaryPoints", ExactSpelling = true)]
            extern static MR.Std.Optional_MRVertBitSet._Underlying *__MR_TriangulationHelpers_findBoundaryPoints(MR.Const_PointCloud._Underlying *pointCloud, MR.TriangulationHelpers.Const_Settings._Underlying *settings, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
            return MR.Misc.Move(new MR.Std.Optional_MRVertBitSet(__MR_TriangulationHelpers_findBoundaryPoints(pointCloud._UnderlyingPtr, settings._UnderlyingPtr, cb is not null ? cb.PassByMode : MR.Misc._PassBy.default_arg, cb is not null && cb.Value is not null ? cb.Value._UnderlyingPtr : null), is_owning: true));
        }
    }
}
