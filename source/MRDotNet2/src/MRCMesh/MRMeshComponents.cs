public static partial class MR
{
    public static partial class MeshComponents
    {
        /// Face incidence type
        public enum FaceIncidence : int
        {
            ///< face can have neighbor only via edge
            PerEdge = 0,
            ///< face can have neighbor via vertex
            PerVertex = 1,
        }

        /// Generated from class `MR::MeshComponents::ExpandToComponentsParams`.
        /// This is the const half of the class.
        public class Const_ExpandToComponentsParams : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_ExpandToComponentsParams(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshComponents_ExpandToComponentsParams_Destroy(_Underlying *_this);
                __MR_MeshComponents_ExpandToComponentsParams_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_ExpandToComponentsParams() {Dispose(false);}

            /// expands only if seeds cover at least this ratio of the component area
            /// <=0 - expands all seeds
            /// > 1 - none
            public unsafe float CoverRatio
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_Get_coverRatio", ExactSpelling = true)]
                    extern static float *__MR_MeshComponents_ExpandToComponentsParams_Get_coverRatio(_Underlying *_this);
                    return *__MR_MeshComponents_ExpandToComponentsParams_Get_coverRatio(_UnderlyingPtr);
                }
            }

            public unsafe MR.MeshComponents.FaceIncidence Incidence
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_Get_incidence", ExactSpelling = true)]
                    extern static MR.MeshComponents.FaceIncidence *__MR_MeshComponents_ExpandToComponentsParams_Get_incidence(_Underlying *_this);
                    return *__MR_MeshComponents_ExpandToComponentsParams_Get_incidence(_UnderlyingPtr);
                }
            }

            /// optional predicate of boundaries between components
            public unsafe ref readonly void * IsCompBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_Get_isCompBd", ExactSpelling = true)]
                    extern static void **__MR_MeshComponents_ExpandToComponentsParams_Get_isCompBd(_Underlying *_this);
                    return ref *__MR_MeshComponents_ExpandToComponentsParams_Get_isCompBd(_UnderlyingPtr);
                }
            }

            /// optional output number of components
            public unsafe ref int * OptOutNumComponents
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_Get_optOutNumComponents", ExactSpelling = true)]
                    extern static int **__MR_MeshComponents_ExpandToComponentsParams_Get_optOutNumComponents(_Underlying *_this);
                    return ref *__MR_MeshComponents_ExpandToComponentsParams_Get_optOutNumComponents(_UnderlyingPtr);
                }
            }

            public unsafe MR.Std.Const_Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_Get_cb", ExactSpelling = true)]
                    extern static MR.Std.Const_Function_BoolFuncFromFloat._Underlying *__MR_MeshComponents_ExpandToComponentsParams_Get_cb(_Underlying *_this);
                    return new(__MR_MeshComponents_ExpandToComponentsParams_Get_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_ExpandToComponentsParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshComponents_ExpandToComponentsParams_DefaultConstruct();
            }

            /// Constructs `MR::MeshComponents::ExpandToComponentsParams` elementwise.
            public unsafe Const_ExpandToComponentsParams(float coverRatio, MR.MeshComponents.FaceIncidence incidence, MR.Const_UndirectedEdgeBitSet? isCompBd, MR.Misc.InOut<int>? optOutNumComponents, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_ConstructFrom(float coverRatio, MR.MeshComponents.FaceIncidence incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd, int *optOutNumComponents, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                int __value_optOutNumComponents = optOutNumComponents is not null ? optOutNumComponents.Value : default(int);
                _UnderlyingPtr = __MR_MeshComponents_ExpandToComponentsParams_ConstructFrom(coverRatio, incidence, isCompBd is not null ? isCompBd._UnderlyingPtr : null, optOutNumComponents is not null ? &__value_optOutNumComponents : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
                if (optOutNumComponents is not null) optOutNumComponents.Value = __value_optOutNumComponents;
            }

            /// Generated from constructor `MR::MeshComponents::ExpandToComponentsParams::ExpandToComponentsParams`.
            public unsafe Const_ExpandToComponentsParams(MR.MeshComponents._ByValue_ExpandToComponentsParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshComponents.ExpandToComponentsParams._Underlying *_other);
                _UnderlyingPtr = __MR_MeshComponents_ExpandToComponentsParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// Generated from class `MR::MeshComponents::ExpandToComponentsParams`.
        /// This is the non-const half of the class.
        public class ExpandToComponentsParams : Const_ExpandToComponentsParams
        {
            internal unsafe ExpandToComponentsParams(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// expands only if seeds cover at least this ratio of the component area
            /// <=0 - expands all seeds
            /// > 1 - none
            public new unsafe ref float CoverRatio
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_GetMutable_coverRatio", ExactSpelling = true)]
                    extern static float *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_coverRatio(_Underlying *_this);
                    return ref *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_coverRatio(_UnderlyingPtr);
                }
            }

            public new unsafe ref MR.MeshComponents.FaceIncidence Incidence
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_GetMutable_incidence", ExactSpelling = true)]
                    extern static MR.MeshComponents.FaceIncidence *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_incidence(_Underlying *_this);
                    return ref *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_incidence(_UnderlyingPtr);
                }
            }

            /// optional predicate of boundaries between components
            public new unsafe ref readonly void * IsCompBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_GetMutable_isCompBd", ExactSpelling = true)]
                    extern static void **__MR_MeshComponents_ExpandToComponentsParams_GetMutable_isCompBd(_Underlying *_this);
                    return ref *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_isCompBd(_UnderlyingPtr);
                }
            }

            /// optional output number of components
            public new unsafe ref int * OptOutNumComponents
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_GetMutable_optOutNumComponents", ExactSpelling = true)]
                    extern static int **__MR_MeshComponents_ExpandToComponentsParams_GetMutable_optOutNumComponents(_Underlying *_this);
                    return ref *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_optOutNumComponents(_UnderlyingPtr);
                }
            }

            public new unsafe MR.Std.Function_BoolFuncFromFloat Cb
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_GetMutable_cb", ExactSpelling = true)]
                    extern static MR.Std.Function_BoolFuncFromFloat._Underlying *__MR_MeshComponents_ExpandToComponentsParams_GetMutable_cb(_Underlying *_this);
                    return new(__MR_MeshComponents_ExpandToComponentsParams_GetMutable_cb(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe ExpandToComponentsParams() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshComponents_ExpandToComponentsParams_DefaultConstruct();
            }

            /// Constructs `MR::MeshComponents::ExpandToComponentsParams` elementwise.
            public unsafe ExpandToComponentsParams(float coverRatio, MR.MeshComponents.FaceIncidence incidence, MR.Const_UndirectedEdgeBitSet? isCompBd, MR.Misc.InOut<int>? optOutNumComponents, MR.Std._ByValue_Function_BoolFuncFromFloat cb) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_ConstructFrom(float coverRatio, MR.MeshComponents.FaceIncidence incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd, int *optOutNumComponents, MR.Misc._PassBy cb_pass_by, MR.Std.Function_BoolFuncFromFloat._Underlying *cb);
                int __value_optOutNumComponents = optOutNumComponents is not null ? optOutNumComponents.Value : default(int);
                _UnderlyingPtr = __MR_MeshComponents_ExpandToComponentsParams_ConstructFrom(coverRatio, incidence, isCompBd is not null ? isCompBd._UnderlyingPtr : null, optOutNumComponents is not null ? &__value_optOutNumComponents : null, cb.PassByMode, cb.Value is not null ? cb.Value._UnderlyingPtr : null);
                if (optOutNumComponents is not null) optOutNumComponents.Value = __value_optOutNumComponents;
            }

            /// Generated from constructor `MR::MeshComponents::ExpandToComponentsParams::ExpandToComponentsParams`.
            public unsafe ExpandToComponentsParams(MR.MeshComponents._ByValue_ExpandToComponentsParams _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.MeshComponents.ExpandToComponentsParams._Underlying *_other);
                _UnderlyingPtr = __MR_MeshComponents_ExpandToComponentsParams_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::MeshComponents::ExpandToComponentsParams::operator=`.
            public unsafe MR.MeshComponents.ExpandToComponentsParams Assign(MR.MeshComponents._ByValue_ExpandToComponentsParams _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_ExpandToComponentsParams_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshComponents.ExpandToComponentsParams._Underlying *__MR_MeshComponents_ExpandToComponentsParams_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.MeshComponents.ExpandToComponentsParams._Underlying *_other);
                return new(__MR_MeshComponents_ExpandToComponentsParams_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `ExpandToComponentsParams` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `ExpandToComponentsParams`/`Const_ExpandToComponentsParams` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_ExpandToComponentsParams
        {
            internal readonly Const_ExpandToComponentsParams? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_ExpandToComponentsParams() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_ExpandToComponentsParams(Const_ExpandToComponentsParams new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_ExpandToComponentsParams(Const_ExpandToComponentsParams arg) {return new(arg);}
            public _ByValue_ExpandToComponentsParams(MR.Misc._Moved<ExpandToComponentsParams> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_ExpandToComponentsParams(MR.Misc._Moved<ExpandToComponentsParams> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `ExpandToComponentsParams` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_ExpandToComponentsParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ExpandToComponentsParams`/`Const_ExpandToComponentsParams` directly.
        public class _InOptMut_ExpandToComponentsParams
        {
            public ExpandToComponentsParams? Opt;

            public _InOptMut_ExpandToComponentsParams() {}
            public _InOptMut_ExpandToComponentsParams(ExpandToComponentsParams value) {Opt = value;}
            public static implicit operator _InOptMut_ExpandToComponentsParams(ExpandToComponentsParams value) {return new(value);}
        }

        /// This is used for optional parameters of class `ExpandToComponentsParams` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_ExpandToComponentsParams`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `ExpandToComponentsParams`/`Const_ExpandToComponentsParams` to pass it to the function.
        public class _InOptConst_ExpandToComponentsParams
        {
            public Const_ExpandToComponentsParams? Opt;

            public _InOptConst_ExpandToComponentsParams() {}
            public _InOptConst_ExpandToComponentsParams(Const_ExpandToComponentsParams value) {Opt = value;}
            public static implicit operator _InOptConst_ExpandToComponentsParams(Const_ExpandToComponentsParams value) {return new(value);}
        }

        /// Generated from class `MR::MeshComponents::LargeByAreaComponentsSettings`.
        /// This is the const half of the class.
        public class Const_LargeByAreaComponentsSettings : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_LargeByAreaComponentsSettings(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_Destroy", ExactSpelling = true)]
                extern static void __MR_MeshComponents_LargeByAreaComponentsSettings_Destroy(_Underlying *_this);
                __MR_MeshComponents_LargeByAreaComponentsSettings_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_LargeByAreaComponentsSettings() {Dispose(false);}

            /// return at most given number of largest by area connected components
            public unsafe int MaxLargeComponents
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_Get_maxLargeComponents", ExactSpelling = true)]
                    extern static int *__MR_MeshComponents_LargeByAreaComponentsSettings_Get_maxLargeComponents(_Underlying *_this);
                    return *__MR_MeshComponents_LargeByAreaComponentsSettings_Get_maxLargeComponents(_UnderlyingPtr);
                }
            }

            /// optional output: the number of components in addition to returned ones
            public unsafe ref int * NumSmallerComponents
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_Get_numSmallerComponents", ExactSpelling = true)]
                    extern static int **__MR_MeshComponents_LargeByAreaComponentsSettings_Get_numSmallerComponents(_Underlying *_this);
                    return ref *__MR_MeshComponents_LargeByAreaComponentsSettings_Get_numSmallerComponents(_UnderlyingPtr);
                }
            }

            /// do not consider a component large if its area is below this value
            public unsafe float MinArea
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_Get_minArea", ExactSpelling = true)]
                    extern static float *__MR_MeshComponents_LargeByAreaComponentsSettings_Get_minArea(_Underlying *_this);
                    return *__MR_MeshComponents_LargeByAreaComponentsSettings_Get_minArea(_UnderlyingPtr);
                }
            }

            /// optional predicate of boundaries between components
            public unsafe ref readonly void * IsCompBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_Get_isCompBd", ExactSpelling = true)]
                    extern static void **__MR_MeshComponents_LargeByAreaComponentsSettings_Get_isCompBd(_Underlying *_this);
                    return ref *__MR_MeshComponents_LargeByAreaComponentsSettings_Get_isCompBd(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_LargeByAreaComponentsSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshComponents_LargeByAreaComponentsSettings_DefaultConstruct();
            }

            /// Constructs `MR::MeshComponents::LargeByAreaComponentsSettings` elementwise.
            public unsafe Const_LargeByAreaComponentsSettings(int maxLargeComponents, MR.Misc.InOut<int>? numSmallerComponents, float minArea, MR.Const_UndirectedEdgeBitSet? isCompBd) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFrom(int maxLargeComponents, int *numSmallerComponents, float minArea, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
                int __value_numSmallerComponents = numSmallerComponents is not null ? numSmallerComponents.Value : default(int);
                _UnderlyingPtr = __MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFrom(maxLargeComponents, numSmallerComponents is not null ? &__value_numSmallerComponents : null, minArea, isCompBd is not null ? isCompBd._UnderlyingPtr : null);
                if (numSmallerComponents is not null) numSmallerComponents.Value = __value_numSmallerComponents;
            }

            /// Generated from constructor `MR::MeshComponents::LargeByAreaComponentsSettings::LargeByAreaComponentsSettings`.
            public unsafe Const_LargeByAreaComponentsSettings(MR.MeshComponents.Const_LargeByAreaComponentsSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFromAnother(MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *_other);
                _UnderlyingPtr = __MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::MeshComponents::LargeByAreaComponentsSettings`.
        /// This is the non-const half of the class.
        public class LargeByAreaComponentsSettings : Const_LargeByAreaComponentsSettings
        {
            internal unsafe LargeByAreaComponentsSettings(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            /// return at most given number of largest by area connected components
            public new unsafe ref int MaxLargeComponents
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_maxLargeComponents", ExactSpelling = true)]
                    extern static int *__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_maxLargeComponents(_Underlying *_this);
                    return ref *__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_maxLargeComponents(_UnderlyingPtr);
                }
            }

            /// optional output: the number of components in addition to returned ones
            public new unsafe ref int * NumSmallerComponents
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_numSmallerComponents", ExactSpelling = true)]
                    extern static int **__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_numSmallerComponents(_Underlying *_this);
                    return ref *__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_numSmallerComponents(_UnderlyingPtr);
                }
            }

            /// do not consider a component large if its area is below this value
            public new unsafe ref float MinArea
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_minArea", ExactSpelling = true)]
                    extern static float *__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_minArea(_Underlying *_this);
                    return ref *__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_minArea(_UnderlyingPtr);
                }
            }

            /// optional predicate of boundaries between components
            public new unsafe ref readonly void * IsCompBd
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_isCompBd", ExactSpelling = true)]
                    extern static void **__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_isCompBd(_Underlying *_this);
                    return ref *__MR_MeshComponents_LargeByAreaComponentsSettings_GetMutable_isCompBd(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe LargeByAreaComponentsSettings() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_DefaultConstruct", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_DefaultConstruct();
                _UnderlyingPtr = __MR_MeshComponents_LargeByAreaComponentsSettings_DefaultConstruct();
            }

            /// Constructs `MR::MeshComponents::LargeByAreaComponentsSettings` elementwise.
            public unsafe LargeByAreaComponentsSettings(int maxLargeComponents, MR.Misc.InOut<int>? numSmallerComponents, float minArea, MR.Const_UndirectedEdgeBitSet? isCompBd) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFrom", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFrom(int maxLargeComponents, int *numSmallerComponents, float minArea, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
                int __value_numSmallerComponents = numSmallerComponents is not null ? numSmallerComponents.Value : default(int);
                _UnderlyingPtr = __MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFrom(maxLargeComponents, numSmallerComponents is not null ? &__value_numSmallerComponents : null, minArea, isCompBd is not null ? isCompBd._UnderlyingPtr : null);
                if (numSmallerComponents is not null) numSmallerComponents.Value = __value_numSmallerComponents;
            }

            /// Generated from constructor `MR::MeshComponents::LargeByAreaComponentsSettings::LargeByAreaComponentsSettings`.
            public unsafe LargeByAreaComponentsSettings(MR.MeshComponents.Const_LargeByAreaComponentsSettings _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFromAnother(MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *_other);
                _UnderlyingPtr = __MR_MeshComponents_LargeByAreaComponentsSettings_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::MeshComponents::LargeByAreaComponentsSettings::operator=`.
            public unsafe MR.MeshComponents.LargeByAreaComponentsSettings Assign(MR.MeshComponents.Const_LargeByAreaComponentsSettings _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_LargeByAreaComponentsSettings_AssignFromAnother", ExactSpelling = true)]
                extern static MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *__MR_MeshComponents_LargeByAreaComponentsSettings_AssignFromAnother(_Underlying *_this, MR.MeshComponents.LargeByAreaComponentsSettings._Underlying *_other);
                return new(__MR_MeshComponents_LargeByAreaComponentsSettings_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `LargeByAreaComponentsSettings` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_LargeByAreaComponentsSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `LargeByAreaComponentsSettings`/`Const_LargeByAreaComponentsSettings` directly.
        public class _InOptMut_LargeByAreaComponentsSettings
        {
            public LargeByAreaComponentsSettings? Opt;

            public _InOptMut_LargeByAreaComponentsSettings() {}
            public _InOptMut_LargeByAreaComponentsSettings(LargeByAreaComponentsSettings value) {Opt = value;}
            public static implicit operator _InOptMut_LargeByAreaComponentsSettings(LargeByAreaComponentsSettings value) {return new(value);}
        }

        /// This is used for optional parameters of class `LargeByAreaComponentsSettings` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_LargeByAreaComponentsSettings`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `LargeByAreaComponentsSettings`/`Const_LargeByAreaComponentsSettings` to pass it to the function.
        public class _InOptConst_LargeByAreaComponentsSettings
        {
            public Const_LargeByAreaComponentsSettings? Opt;

            public _InOptConst_LargeByAreaComponentsSettings() {}
            public _InOptConst_LargeByAreaComponentsSettings(Const_LargeByAreaComponentsSettings value) {Opt = value;}
            public static implicit operator _InOptConst_LargeByAreaComponentsSettings(Const_LargeByAreaComponentsSettings value) {return new(value);}
        }

        /// returns one connected component containing given face, 
        /// not effective to call more than once, if several components are needed use getAllComponents
        /// Generated from function `MR::MeshComponents::getComponent`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetComponent(MR.Const_MeshPart meshPart, MR.FaceId id, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getComponent", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshComponents_getComponent(MR.Const_MeshPart._Underlying *meshPart, MR.FaceId id, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshComponents_getComponent(meshPart._UnderlyingPtr, id, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// returns one connected component containing given vertex, 
        /// not effective to call more than once, if several components are needed use getAllComponentsVerts
        /// Generated from function `MR::MeshComponents::getComponentVerts`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> GetComponentVerts(MR.Const_Mesh mesh, MR.VertId id, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getComponentVerts", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_MeshComponents_getComponentVerts(MR.Const_Mesh._Underlying *mesh, MR.VertId id, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.VertBitSet(__MR_MeshComponents_getComponentVerts(mesh._UnderlyingPtr, id, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the largest by number of elements component
        /// Generated from function `MR::MeshComponents::getLargestComponentVerts`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> GetLargestComponentVerts(MR.Const_Mesh mesh, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargestComponentVerts", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_MeshComponents_getLargestComponentVerts(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.VertBitSet(__MR_MeshComponents_getLargestComponentVerts(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the union of vertex connected components, each having at least \param minVerts vertices
        /// Generated from function `MR::MeshComponents::getLargeComponentVerts`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> GetLargeComponentVerts(MR.Const_Mesh mesh, int minVerts, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargeComponentVerts", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_MeshComponents_getLargeComponentVerts(MR.Const_Mesh._Underlying *mesh, int minVerts, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.VertBitSet(__MR_MeshComponents_getLargeComponentVerts(mesh._UnderlyingPtr, minVerts, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the largest by surface area component or empty set if its area is smaller than \param minArea
        /// Generated from function `MR::MeshComponents::getLargestComponent`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        /// Parameter `minArea` defaults to `0`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetLargestComponent(MR.Const_MeshPart meshPart, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null, float? minArea = null, MR.Misc.InOut<int>? numSmallerComponents = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargestComponent", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshComponents_getLargestComponent(MR.Const_MeshPart._Underlying *meshPart, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd, float *minArea, int *numSmallerComponents);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            float __deref_minArea = minArea.GetValueOrDefault();
            int __value_numSmallerComponents = numSmallerComponents is not null ? numSmallerComponents.Value : default(int);
            var __ret = __MR_MeshComponents_getLargestComponent(meshPart._UnderlyingPtr, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null, minArea.HasValue ? &__deref_minArea : null, numSmallerComponents is not null ? &__value_numSmallerComponents : null);
            if (numSmallerComponents is not null) numSmallerComponents.Value = __value_numSmallerComponents;
            return MR.Misc.Move(new MR.FaceBitSet(__ret, is_owning: true));
        }

        /// returns union of connected components, each of which contains at least one seed face
        /// Generated from function `MR::MeshComponents::getComponents`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetComponents(MR.Const_MeshPart meshPart, MR.Const_FaceBitSet seeds, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getComponents", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshComponents_getComponents(MR.Const_MeshPart._Underlying *meshPart, MR.Const_FaceBitSet._Underlying *seeds, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshComponents_getComponents(meshPart._UnderlyingPtr, seeds._UnderlyingPtr, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the union of connected components, each having at least given area
        /// Generated from function `MR::MeshComponents::getLargeByAreaComponents`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetLargeByAreaComponents(MR.Const_MeshPart meshPart, float minArea, MR.Const_UndirectedEdgeBitSet? isCompBd)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargeByAreaComponents_3", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshComponents_getLargeByAreaComponents_3(MR.Const_MeshPart._Underlying *meshPart, float minArea, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshComponents_getLargeByAreaComponents_3(meshPart._UnderlyingPtr, minArea, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// given prepared union-find structure returns the union of connected components, each having at least given area
        /// Generated from function `MR::MeshComponents::getLargeByAreaComponents`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetLargeByAreaComponents(MR.Const_MeshPart meshPart, MR.UnionFind_MRFaceId unionFind, float minArea, MR.UndirectedEdgeBitSet? outBdEdgesBetweenLargeComps = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargeByAreaComponents_4", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshComponents_getLargeByAreaComponents_4(MR.Const_MeshPart._Underlying *meshPart, MR.UnionFind_MRFaceId._Underlying *unionFind, float minArea, MR.UndirectedEdgeBitSet._Underlying *outBdEdgesBetweenLargeComps);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshComponents_getLargeByAreaComponents_4(meshPart._UnderlyingPtr, unionFind._UnderlyingPtr, minArea, outBdEdgesBetweenLargeComps is not null ? outBdEdgesBetweenLargeComps._UnderlyingPtr : null), is_owning: true));
        }

        /// expands given seeds to whole components
        /// Generated from function `MR::MeshComponents::expandToComponents`.
        /// Parameter `params_` defaults to `{}`.
        public static unsafe MR.Misc._Moved<MR.Expected_MRFaceBitSet_StdString> ExpandToComponents(MR.Const_MeshPart mp, MR.Const_FaceBitSet seeds, MR.MeshComponents.Const_ExpandToComponentsParams? params_ = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_expandToComponents", ExactSpelling = true)]
            extern static MR.Expected_MRFaceBitSet_StdString._Underlying *__MR_MeshComponents_expandToComponents(MR.Const_MeshPart._Underlying *mp, MR.Const_FaceBitSet._Underlying *seeds, MR.MeshComponents.Const_ExpandToComponentsParams._Underlying *params_);
            return MR.Misc.Move(new MR.Expected_MRFaceBitSet_StdString(__MR_MeshComponents_expandToComponents(mp._UnderlyingPtr, seeds._UnderlyingPtr, params_ is not null ? params_._UnderlyingPtr : null), is_owning: true));
        }

        /// returns requested number of largest by area connected components in descending by area order
        /// Generated from function `MR::MeshComponents::getNLargeByAreaComponents`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRFaceBitSet> GetNLargeByAreaComponents(MR.Const_MeshPart meshPart, MR.MeshComponents.Const_LargeByAreaComponentsSettings settings)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getNLargeByAreaComponents", ExactSpelling = true)]
            extern static MR.Std.Vector_MRFaceBitSet._Underlying *__MR_MeshComponents_getNLargeByAreaComponents(MR.Const_MeshPart._Underlying *meshPart, MR.MeshComponents.Const_LargeByAreaComponentsSettings._Underlying *settings);
            return MR.Misc.Move(new MR.Std.Vector_MRFaceBitSet(__MR_MeshComponents_getNLargeByAreaComponents(meshPart._UnderlyingPtr, settings._UnderlyingPtr), is_owning: true));
        }

        /// returns the union of connected components, each having at least given area,
        /// and any two faces in a connected component have a path along the surface across the edges, where surface does not deviate from plane more than on given angle
        /// Generated from function `MR::MeshComponents::getLargeByAreaSmoothComponents`.
        public static unsafe MR.Misc._Moved<MR.FaceBitSet> GetLargeByAreaSmoothComponents(MR.Const_MeshPart meshPart, float minArea, float angleFromPlanar, MR.UndirectedEdgeBitSet? outBdEdgesBetweenLargeComps = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargeByAreaSmoothComponents", ExactSpelling = true)]
            extern static MR.FaceBitSet._Underlying *__MR_MeshComponents_getLargeByAreaSmoothComponents(MR.Const_MeshPart._Underlying *meshPart, float minArea, float angleFromPlanar, MR.UndirectedEdgeBitSet._Underlying *outBdEdgesBetweenLargeComps);
            return MR.Misc.Move(new MR.FaceBitSet(__MR_MeshComponents_getLargeByAreaSmoothComponents(meshPart._UnderlyingPtr, minArea, angleFromPlanar, outBdEdgesBetweenLargeComps is not null ? outBdEdgesBetweenLargeComps._UnderlyingPtr : null), is_owning: true));
        }

        /// returns union of connected components, each of which contains at least one seed vert
        /// Generated from function `MR::MeshComponents::getComponentsVerts`.
        public static unsafe MR.Misc._Moved<MR.VertBitSet> GetComponentsVerts(MR.Const_Mesh mesh, MR.Const_VertBitSet seeds, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getComponentsVerts", ExactSpelling = true)]
            extern static MR.VertBitSet._Underlying *__MR_MeshComponents_getComponentsVerts(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *seeds, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.VertBitSet(__MR_MeshComponents_getComponentsVerts(mesh._UnderlyingPtr, seeds._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// returns the number of connected components in mesh part
        /// Generated from function `MR::MeshComponents::getNumComponents`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe ulong GetNumComponents(MR.Const_MeshPart meshPart, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getNumComponents", ExactSpelling = true)]
            extern static ulong __MR_MeshComponents_getNumComponents(MR.Const_MeshPart._Underlying *meshPart, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return __MR_MeshComponents_getNumComponents(meshPart._UnderlyingPtr, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null);
        }

        /// gets all connected components of mesh part
        /// \note be careful, if mesh is large enough and has many components, the memory overflow will occur
        /// Generated from function `MR::MeshComponents::getAllComponents`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRFaceBitSet> GetAllComponents(MR.Const_MeshPart meshPart, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponents_3", ExactSpelling = true)]
            extern static MR.Std.Vector_MRFaceBitSet._Underlying *__MR_MeshComponents_getAllComponents_3(MR.Const_MeshPart._Underlying *meshPart, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return MR.Misc.Move(new MR.Std.Vector_MRFaceBitSet(__MR_MeshComponents_getAllComponents_3(meshPart._UnderlyingPtr, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// gets all connected components of mesh part
        /// \detail if components  number more than the maxComponentCount, they will be combined into groups of the same size
        /// \param maxComponentCount should be more then 1
        /// \return pair components bitsets vector and number components in one group if components number more than maxComponentCount
        /// Generated from function `MR::MeshComponents::getAllComponents`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe MR.Misc._Moved<MR.Std.Pair_StdVectorMRFaceBitSet_Int> GetAllComponents(MR.Const_MeshPart meshPart, int maxComponentCount, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponents_4_MR_MeshPart", ExactSpelling = true)]
            extern static MR.Std.Pair_StdVectorMRFaceBitSet_Int._Underlying *__MR_MeshComponents_getAllComponents_4_MR_MeshPart(MR.Const_MeshPart._Underlying *meshPart, int maxComponentCount, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return MR.Misc.Move(new MR.Std.Pair_StdVectorMRFaceBitSet_Int(__MR_MeshComponents_getAllComponents_4_MR_MeshPart(meshPart._UnderlyingPtr, maxComponentCount, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// gets all connected components from components map ( FaceId => RegionId )
        /// \detail if components  number more than the maxComponentCount, they will be combined into groups of the same size (this similarly changes componentsMap)
        /// \param maxComponentCount should be more then 1
        /// \return components bitsets vector
        /// Generated from function `MR::MeshComponents::getAllComponents`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRFaceBitSet> GetAllComponents(MR.Face2RegionMap componentsMap, int componentsCount, MR.Const_FaceBitSet region, int maxComponentCount)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponents_4_MR_Face2RegionMap", ExactSpelling = true)]
            extern static MR.Std.Vector_MRFaceBitSet._Underlying *__MR_MeshComponents_getAllComponents_4_MR_Face2RegionMap(MR.Face2RegionMap._Underlying *componentsMap, int componentsCount, MR.Const_FaceBitSet._Underlying *region, int maxComponentCount);
            return MR.Misc.Move(new MR.Std.Vector_MRFaceBitSet(__MR_MeshComponents_getAllComponents_4_MR_Face2RegionMap(componentsMap._UnderlyingPtr, componentsCount, region._UnderlyingPtr, maxComponentCount), is_owning: true));
        }

        /// gets all connected components of mesh part as
        /// 1. the mapping: FaceId -> Component ID in [0, 1, 2, ...)
        /// 2. the total number of components
        /// Generated from function `MR::MeshComponents::getAllComponentsMap`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe MR.Misc._Moved<MR.Std.Pair_MRFace2RegionMap_Int> GetAllComponentsMap(MR.Const_MeshPart meshPart, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponentsMap", ExactSpelling = true)]
            extern static MR.Std.Pair_MRFace2RegionMap_Int._Underlying *__MR_MeshComponents_getAllComponentsMap(MR.Const_MeshPart._Underlying *meshPart, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return MR.Misc.Move(new MR.Std.Pair_MRFace2RegionMap_Int(__MR_MeshComponents_getAllComponentsMap(meshPart._UnderlyingPtr, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// computes the area of each region given via the map
        /// Generated from function `MR::MeshComponents::getRegionAreas`.
        public static unsafe MR.Misc._Moved<MR.Vector_Double_MRRegionId> GetRegionAreas(MR.Const_MeshPart meshPart, MR.Const_Face2RegionMap regionMap, int numRegions)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getRegionAreas", ExactSpelling = true)]
            extern static MR.Vector_Double_MRRegionId._Underlying *__MR_MeshComponents_getRegionAreas(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Face2RegionMap._Underlying *regionMap, int numRegions);
            return MR.Misc.Move(new MR.Vector_Double_MRRegionId(__MR_MeshComponents_getRegionAreas(meshPart._UnderlyingPtr, regionMap._UnderlyingPtr, numRegions), is_owning: true));
        }

        /// returns
        /// 1. the union of all regions with area >= minArea
        /// 2. the number of such regions
        /// Generated from function `MR::MeshComponents::getLargeByAreaRegions`.
        public static unsafe MR.Misc._Moved<MR.Std.Pair_MRFaceBitSet_Int> GetLargeByAreaRegions(MR.Const_MeshPart meshPart, MR.Const_Face2RegionMap regionMap, int numRegions, float minArea)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getLargeByAreaRegions", ExactSpelling = true)]
            extern static MR.Std.Pair_MRFaceBitSet_Int._Underlying *__MR_MeshComponents_getLargeByAreaRegions(MR.Const_MeshPart._Underlying *meshPart, MR.Const_Face2RegionMap._Underlying *regionMap, int numRegions, float minArea);
            return MR.Misc.Move(new MR.Std.Pair_MRFaceBitSet_Int(__MR_MeshComponents_getLargeByAreaRegions(meshPart._UnderlyingPtr, regionMap._UnderlyingPtr, numRegions, minArea), is_owning: true));
        }

        /// gets all connected components of mesh part
        /// Generated from function `MR::MeshComponents::getAllComponentsVerts`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVertBitSet> GetAllComponentsVerts(MR.Const_Mesh mesh, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponentsVerts", ExactSpelling = true)]
            extern static MR.Std.Vector_MRVertBitSet._Underlying *__MR_MeshComponents_getAllComponentsVerts(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.Std.Vector_MRVertBitSet(__MR_MeshComponents_getAllComponentsVerts(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// gets all connected components, separating vertices by given path (either closed or from boundary to boundary)
        /// Generated from function `MR::MeshComponents::getAllComponentsVertsSeparatedByPath`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVertBitSet> GetAllComponentsVertsSeparatedByPath(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint path)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponentsVertsSeparatedByPath", ExactSpelling = true)]
            extern static MR.Std.Vector_MRVertBitSet._Underlying *__MR_MeshComponents_getAllComponentsVertsSeparatedByPath(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *path);
            return MR.Misc.Move(new MR.Std.Vector_MRVertBitSet(__MR_MeshComponents_getAllComponentsVertsSeparatedByPath(mesh._UnderlyingPtr, path._UnderlyingPtr), is_owning: true));
        }

        /// gets all connected components, separating vertices by given paths (either closed or from boundary to boundary)
        /// Generated from function `MR::MeshComponents::getAllComponentsVertsSeparatedByPaths`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRVertBitSet> GetAllComponentsVertsSeparatedByPaths(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMREdgePoint paths)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponentsVertsSeparatedByPaths", ExactSpelling = true)]
            extern static MR.Std.Vector_MRVertBitSet._Underlying *__MR_MeshComponents_getAllComponentsVertsSeparatedByPaths(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMREdgePoint._Underlying *paths);
            return MR.Misc.Move(new MR.Std.Vector_MRVertBitSet(__MR_MeshComponents_getAllComponentsVertsSeparatedByPaths(mesh._UnderlyingPtr, paths._UnderlyingPtr), is_owning: true));
        }

        /// subdivides given edges on connected components
        /// Generated from function `MR::MeshComponents::getAllComponentsEdges`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MREdgeBitSet> GetAllComponentsEdges(MR.Const_Mesh mesh, MR.Const_EdgeBitSet edges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponentsEdges", ExactSpelling = true)]
            extern static MR.Std.Vector_MREdgeBitSet._Underlying *__MR_MeshComponents_getAllComponentsEdges(MR.Const_Mesh._Underlying *mesh, MR.Const_EdgeBitSet._Underlying *edges);
            return MR.Misc.Move(new MR.Std.Vector_MREdgeBitSet(__MR_MeshComponents_getAllComponentsEdges(mesh._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
        }

        /// subdivides given edges on connected components
        /// Generated from function `MR::MeshComponents::getAllComponentsUndirectedEdges`.
        public static unsafe MR.Misc._Moved<MR.Std.Vector_MRUndirectedEdgeBitSet> GetAllComponentsUndirectedEdges(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet edges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getAllComponentsUndirectedEdges", ExactSpelling = true)]
            extern static MR.Std.Vector_MRUndirectedEdgeBitSet._Underlying *__MR_MeshComponents_getAllComponentsUndirectedEdges(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
            return MR.Misc.Move(new MR.Std.Vector_MRUndirectedEdgeBitSet(__MR_MeshComponents_getAllComponentsUndirectedEdges(mesh._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
        }

        /// returns true if all vertices of a mesh connected component are present in selection
        /// Generated from function `MR::MeshComponents::hasFullySelectedComponent`.
        public static unsafe bool HasFullySelectedComponent(MR.Const_Mesh mesh, MR.Const_VertBitSet selection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_hasFullySelectedComponent_MR_Mesh", ExactSpelling = true)]
            extern static byte __MR_MeshComponents_hasFullySelectedComponent_MR_Mesh(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *selection);
            return __MR_MeshComponents_hasFullySelectedComponent_MR_Mesh(mesh._UnderlyingPtr, selection._UnderlyingPtr) != 0;
        }

        /// Generated from function `MR::MeshComponents::hasFullySelectedComponent`.
        public static unsafe bool HasFullySelectedComponent(MR.Const_MeshTopology topology, MR.Const_VertBitSet selection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_hasFullySelectedComponent_MR_MeshTopology", ExactSpelling = true)]
            extern static byte __MR_MeshComponents_hasFullySelectedComponent_MR_MeshTopology(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertBitSet._Underlying *selection);
            return __MR_MeshComponents_hasFullySelectedComponent_MR_MeshTopology(topology._UnderlyingPtr, selection._UnderlyingPtr) != 0;
        }

        /// if all vertices of a mesh connected component are present in selection, excludes these vertices
        /// Generated from function `MR::MeshComponents::excludeFullySelectedComponents`.
        public static unsafe void ExcludeFullySelectedComponents(MR.Const_Mesh mesh, MR.VertBitSet selection)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_excludeFullySelectedComponents", ExactSpelling = true)]
            extern static void __MR_MeshComponents_excludeFullySelectedComponents(MR.Const_Mesh._Underlying *mesh, MR.VertBitSet._Underlying *selection);
            __MR_MeshComponents_excludeFullySelectedComponents(mesh._UnderlyingPtr, selection._UnderlyingPtr);
        }

        /// gets union-find structure for faces with different options of face-connectivity
        /// Generated from function `MR::MeshComponents::getUnionFindStructureFaces`.
        /// Parameter `incidence` defaults to `FaceIncidence::PerEdge`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRFaceId> GetUnionFindStructureFaces(MR.Const_MeshPart meshPart, MR.MeshComponents.FaceIncidence? incidence = null, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureFaces", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_MeshComponents_getUnionFindStructureFaces(MR.Const_MeshPart._Underlying *meshPart, MR.MeshComponents.FaceIncidence *incidence, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            MR.MeshComponents.FaceIncidence __deref_incidence = incidence.GetValueOrDefault();
            return MR.Misc.Move(new MR.UnionFind_MRFaceId(__MR_MeshComponents_getUnionFindStructureFaces(meshPart._UnderlyingPtr, incidence.HasValue ? &__deref_incidence : null, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// gets union-find structure for faces with connectivity by shared edge, and optional edge predicate whether to skip uniting components over it
        /// it is guaranteed that isCompBd is invoked in a thread-safe manner (that left and right face are always processed by one thread)
        /// Generated from function `MR::MeshComponents::getUnionFindStructureFacesPerEdge`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRFaceId> GetUnionFindStructureFacesPerEdge(MR.Const_MeshPart meshPart, MR.Const_UndirectedEdgeBitSet? isCompBd = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureFacesPerEdge", ExactSpelling = true)]
            extern static MR.UnionFind_MRFaceId._Underlying *__MR_MeshComponents_getUnionFindStructureFacesPerEdge(MR.Const_MeshPart._Underlying *meshPart, MR.Const_UndirectedEdgeBitSet._Underlying *isCompBd);
            return MR.Misc.Move(new MR.UnionFind_MRFaceId(__MR_MeshComponents_getUnionFindStructureFacesPerEdge(meshPart._UnderlyingPtr, isCompBd is not null ? isCompBd._UnderlyingPtr : null), is_owning: true));
        }

        /// gets union-find structure for vertices
        /// Generated from function `MR::MeshComponents::getUnionFindStructureVerts`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVerts(MR.Const_Mesh mesh, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_const_MR_VertBitSet_ptr", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_const_MR_VertBitSet_ptr(MR.Const_Mesh._Underlying *mesh, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_const_MR_VertBitSet_ptr(mesh._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::MeshComponents::getUnionFindStructureVerts`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVerts(MR.Const_MeshTopology topology, MR.Const_VertBitSet? region = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVerts_const_MR_MeshTopology_ref", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVerts_const_MR_MeshTopology_ref(MR.Const_MeshTopology._Underlying *topology, MR.Const_VertBitSet._Underlying *region);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVerts_const_MR_MeshTopology_ref(topology._UnderlyingPtr, region is not null ? region._UnderlyingPtr : null), is_owning: true));
        }

        /// gets union-find structure for vertices, considering connections by given edges only
        /// Generated from function `MR::MeshComponents::getUnionFindStructureVerts`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVerts(MR.Const_Mesh mesh, MR.Const_EdgeBitSet edges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_MR_EdgeBitSet", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_MR_EdgeBitSet(MR.Const_Mesh._Underlying *mesh, MR.Const_EdgeBitSet._Underlying *edges);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_MR_EdgeBitSet(mesh._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
        }

        /// gets union-find structure for vertices, considering connections by given undirected edges only
        /// Generated from function `MR::MeshComponents::getUnionFindStructureVerts`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVerts(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet edges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_MR_UndirectedEdgeBitSet", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_MR_UndirectedEdgeBitSet(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *edges);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVerts_const_MR_Mesh_ref_MR_UndirectedEdgeBitSet(mesh._UnderlyingPtr, edges._UnderlyingPtr), is_owning: true));
        }

        /// gets union-find structure for vertices, considering connections by all edges excluding given ones
        /// Generated from function `MR::MeshComponents::getUnionFindStructureVertsEx`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVertsEx(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet ignoreEdges)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVertsEx", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVertsEx(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *ignoreEdges);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVertsEx(mesh._UnderlyingPtr, ignoreEdges._UnderlyingPtr), is_owning: true));
        }

        /**
        * \brief gets union-find structure for vertices, separating vertices by given path (either closed or from boundary to boundary)
        * \param outPathVerts this set receives all vertices passed by the path
        */
        /// Generated from function `MR::MeshComponents::getUnionFindStructureVertsSeparatedByPath`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVertsSeparatedByPath(MR.Const_Mesh mesh, MR.Std.Const_Vector_MREdgePoint path, MR.VertBitSet? outPathVerts = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVertsSeparatedByPath", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVertsSeparatedByPath(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_MREdgePoint._Underlying *path, MR.VertBitSet._Underlying *outPathVerts);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVertsSeparatedByPath(mesh._UnderlyingPtr, path._UnderlyingPtr, outPathVerts is not null ? outPathVerts._UnderlyingPtr : null), is_owning: true));
        }

        /// Generated from function `MR::MeshComponents::getUnionFindStructureVertsSeparatedByPaths`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRVertId> GetUnionFindStructureVertsSeparatedByPaths(MR.Const_Mesh mesh, MR.Std.Const_Vector_StdVectorMREdgePoint paths, MR.VertBitSet? outPathVerts = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureVertsSeparatedByPaths", ExactSpelling = true)]
            extern static MR.UnionFind_MRVertId._Underlying *__MR_MeshComponents_getUnionFindStructureVertsSeparatedByPaths(MR.Const_Mesh._Underlying *mesh, MR.Std.Const_Vector_StdVectorMREdgePoint._Underlying *paths, MR.VertBitSet._Underlying *outPathVerts);
            return MR.Misc.Move(new MR.UnionFind_MRVertId(__MR_MeshComponents_getUnionFindStructureVertsSeparatedByPaths(mesh._UnderlyingPtr, paths._UnderlyingPtr, outPathVerts is not null ? outPathVerts._UnderlyingPtr : null), is_owning: true));
        }

        /// gets union-find structure for all undirected edges in \param mesh
        /// \param allPointToRoots if true, then every element in the structure will point directly to the root of its respective component
        /// Generated from function `MR::MeshComponents::getUnionFindStructureUndirectedEdges`.
        /// Parameter `allPointToRoots` defaults to `false`.
        public static unsafe MR.Misc._Moved<MR.UnionFind_MRUndirectedEdgeId> GetUnionFindStructureUndirectedEdges(MR.Const_Mesh mesh, bool? allPointToRoots = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getUnionFindStructureUndirectedEdges", ExactSpelling = true)]
            extern static MR.UnionFind_MRUndirectedEdgeId._Underlying *__MR_MeshComponents_getUnionFindStructureUndirectedEdges(MR.Const_Mesh._Underlying *mesh, byte *allPointToRoots);
            byte __deref_allPointToRoots = allPointToRoots.GetValueOrDefault() ? (byte)1 : (byte)0;
            return MR.Misc.Move(new MR.UnionFind_MRUndirectedEdgeId(__MR_MeshComponents_getUnionFindStructureUndirectedEdges(mesh._UnderlyingPtr, allPointToRoots.HasValue ? &__deref_allPointToRoots : null), is_owning: true));
        }

        /// returns union of connected components, each of which contains at least one seed edge
        /// Generated from function `MR::MeshComponents::getComponentsUndirectedEdges`.
        public static unsafe MR.Misc._Moved<MR.UndirectedEdgeBitSet> GetComponentsUndirectedEdges(MR.Const_Mesh mesh, MR.Const_UndirectedEdgeBitSet seeds)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_MeshComponents_getComponentsUndirectedEdges", ExactSpelling = true)]
            extern static MR.UndirectedEdgeBitSet._Underlying *__MR_MeshComponents_getComponentsUndirectedEdges(MR.Const_Mesh._Underlying *mesh, MR.Const_UndirectedEdgeBitSet._Underlying *seeds);
            return MR.Misc.Move(new MR.UndirectedEdgeBitSet(__MR_MeshComponents_getComponentsUndirectedEdges(mesh._UnderlyingPtr, seeds._UnderlyingPtr), is_owning: true));
        }
    }
}
