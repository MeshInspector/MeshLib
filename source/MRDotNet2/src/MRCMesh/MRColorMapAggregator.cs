public static partial class MR
{
    public static partial class ColorMapAggregator_MRVertTag
    {
        /// partial color map
        /// Generated from class `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap`.
        /// This is the const half of the class.
        public class Const_PartialColorMap : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_PartialColorMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Destroy", ExactSpelling = true)]
                extern static void __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Destroy(_Underlying *_this);
                __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_PartialColorMap() {Dispose(false);}

            // color map
            public unsafe MR.Const_VertColors ColorMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Get_colorMap", ExactSpelling = true)]
                    extern static MR.Const_VertColors._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Get_colorMap(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Get_colorMap(_UnderlyingPtr), is_owning: false);
                }
            }

            // bitset of elements for which the color map is applied
            public unsafe MR.Const_VertBitSet Elements
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Get_elements", ExactSpelling = true)]
                    extern static MR.Const_VertBitSet._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Get_elements(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_Get_elements(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_PartialColorMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_DefaultConstruct();
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_DefaultConstruct();
            }

            /// Constructs `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap` elementwise.
            public unsafe Const_PartialColorMap(MR._ByValue_VertColors colorMap, MR._ByValue_VertBitSet elements) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFrom(MR.Misc._PassBy colorMap_pass_by, MR.VertColors._Underlying *colorMap, MR.Misc._PassBy elements_pass_by, MR.VertBitSet._Underlying *elements);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFrom(colorMap.PassByMode, colorMap.Value is not null ? colorMap.Value._UnderlyingPtr : null, elements.PassByMode, elements.Value is not null ? elements.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap::PartialColorMap`.
            public unsafe Const_PartialColorMap(MR.ColorMapAggregator_MRVertTag._ByValue_PartialColorMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *_other);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// partial color map
        /// Generated from class `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap`.
        /// This is the non-const half of the class.
        public class PartialColorMap : Const_PartialColorMap
        {
            internal unsafe PartialColorMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // color map
            public new unsafe MR.VertColors ColorMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_GetMutable_colorMap", ExactSpelling = true)]
                    extern static MR.VertColors._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_GetMutable_colorMap(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_GetMutable_colorMap(_UnderlyingPtr), is_owning: false);
                }
            }

            // bitset of elements for which the color map is applied
            public new unsafe MR.VertBitSet Elements
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_GetMutable_elements", ExactSpelling = true)]
                    extern static MR.VertBitSet._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_GetMutable_elements(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_GetMutable_elements(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe PartialColorMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_DefaultConstruct();
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_DefaultConstruct();
            }

            /// Constructs `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap` elementwise.
            public unsafe PartialColorMap(MR._ByValue_VertColors colorMap, MR._ByValue_VertBitSet elements) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFrom(MR.Misc._PassBy colorMap_pass_by, MR.VertColors._Underlying *colorMap, MR.Misc._PassBy elements_pass_by, MR.VertBitSet._Underlying *elements);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFrom(colorMap.PassByMode, colorMap.Value is not null ? colorMap.Value._UnderlyingPtr : null, elements.PassByMode, elements.Value is not null ? elements.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap::PartialColorMap`.
            public unsafe PartialColorMap(MR.ColorMapAggregator_MRVertTag._ByValue_PartialColorMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *_other);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_VertTag_PartialColorMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::ColorMapAggregator<MR::VertTag>::PartialColorMap::operator=`.
            public unsafe MR.ColorMapAggregator_MRVertTag.PartialColorMap Assign(MR.ColorMapAggregator_MRVertTag._ByValue_PartialColorMap _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_VertTag_PartialColorMap_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRVertTag.PartialColorMap._Underlying *_other);
                return new(__MR_ColorMapAggregator_MR_VertTag_PartialColorMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `PartialColorMap` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_PartialColorMap
        {
            internal readonly Const_PartialColorMap? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_PartialColorMap() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_PartialColorMap(Const_PartialColorMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_PartialColorMap(Const_PartialColorMap arg) {return new(arg);}
            public _ByValue_PartialColorMap(MR.Misc._Moved<PartialColorMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_PartialColorMap(MR.Misc._Moved<PartialColorMap> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `PartialColorMap` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialColorMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` directly.
        public class _InOptMut_PartialColorMap
        {
            public PartialColorMap? Opt;

            public _InOptMut_PartialColorMap() {}
            public _InOptMut_PartialColorMap(PartialColorMap value) {Opt = value;}
            public static implicit operator _InOptMut_PartialColorMap(PartialColorMap value) {return new(value);}
        }

        /// This is used for optional parameters of class `PartialColorMap` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialColorMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` to pass it to the function.
        public class _InOptConst_PartialColorMap
        {
            public Const_PartialColorMap? Opt;

            public _InOptConst_PartialColorMap() {}
            public _InOptConst_PartialColorMap(Const_PartialColorMap value) {Opt = value;}
            public static implicit operator _InOptConst_PartialColorMap(Const_PartialColorMap value) {return new(value);}
        }

        /// color map aggregating mode
        public enum AggregateMode : int
        {
            Overlay = 0,
            /// result element color is element color of more priority color map (or default color, if there isn't color map for this element)
            Blending = 1,
        }
    }

    /**
    * @brief Class for aggregate several color map in one
    * Color maps are aggregated according order
    */
    /// Generated from class `MR::VertColorMapAggregator`.
    /// This is the const half of the class.
    public class Const_VertColorMapAggregator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_VertColorMapAggregator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_Destroy", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_Destroy(_Underlying *_this);
            __MR_VertColorMapAggregator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_VertColorMapAggregator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_VertColorMapAggregator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertColorMapAggregator._Underlying *__MR_VertColorMapAggregator_DefaultConstruct();
            _UnderlyingPtr = __MR_VertColorMapAggregator_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertColorMapAggregator::VertColorMapAggregator`.
        public unsafe Const_VertColorMapAggregator(MR._ByValue_VertColorMapAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertColorMapAggregator._Underlying *__MR_VertColorMapAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertColorMapAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_VertColorMapAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * @brief Class for aggregate several color map in one
    * Color maps are aggregated according order
    */
    /// Generated from class `MR::VertColorMapAggregator`.
    /// This is the non-const half of the class.
    public class VertColorMapAggregator : Const_VertColorMapAggregator
    {
        internal unsafe VertColorMapAggregator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe VertColorMapAggregator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.VertColorMapAggregator._Underlying *__MR_VertColorMapAggregator_DefaultConstruct();
            _UnderlyingPtr = __MR_VertColorMapAggregator_DefaultConstruct();
        }

        /// Generated from constructor `MR::VertColorMapAggregator::VertColorMapAggregator`.
        public unsafe VertColorMapAggregator(MR._ByValue_VertColorMapAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.VertColorMapAggregator._Underlying *__MR_VertColorMapAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.VertColorMapAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_VertColorMapAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::VertColorMapAggregator::operator=`.
        public unsafe MR.VertColorMapAggregator Assign(MR._ByValue_VertColorMapAggregator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.VertColorMapAggregator._Underlying *__MR_VertColorMapAggregator_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.VertColorMapAggregator._Underlying *_other);
            return new(__MR_VertColorMapAggregator_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// set default (background) color
        /// Generated from method `MR::VertColorMapAggregator::setDefaultColor`.
        public unsafe void SetDefaultColor(MR.Const_Color color)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_setDefaultColor", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_setDefaultColor(_Underlying *_this, MR.Const_Color._Underlying *color);
            __MR_VertColorMapAggregator_setDefaultColor(_UnderlyingPtr, color._UnderlyingPtr);
        }

        /// add color map after all (more priority)
        /// Generated from method `MR::VertColorMapAggregator::pushBack`.
        public unsafe void PushBack(MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_pushBack", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_pushBack(_Underlying *_this, MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_VertColorMapAggregator_pushBack(_UnderlyingPtr, partitialColorMap._UnderlyingPtr);
        }

        /// insert color map before element #i (0 - minimum priority)
        /// Generated from method `MR::VertColorMapAggregator::insert`.
        public unsafe void Insert(int i, MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_insert", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_insert(_Underlying *_this, int i, MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_VertColorMapAggregator_insert(_UnderlyingPtr, i, partitialColorMap._UnderlyingPtr);
        }

        /// replace color map in #i position
        /// Generated from method `MR::VertColorMapAggregator::replace`.
        public unsafe void Replace(int i, MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_replace", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_replace(_Underlying *_this, int i, MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_VertColorMapAggregator_replace(_UnderlyingPtr, i, partitialColorMap._UnderlyingPtr);
        }

        /// reset all accumulated color map
        /// Generated from method `MR::VertColorMapAggregator::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_reset", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_reset(_Underlying *_this);
            __MR_VertColorMapAggregator_reset(_UnderlyingPtr);
        }

        /// get number of accumulated color maps
        /// Generated from method `MR::VertColorMapAggregator::getColorMapNumber`.
        public unsafe ulong GetColorMapNumber()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_getColorMapNumber", ExactSpelling = true)]
            extern static ulong __MR_VertColorMapAggregator_getColorMapNumber(_Underlying *_this);
            return __MR_VertColorMapAggregator_getColorMapNumber(_UnderlyingPtr);
        }

        /// get partial color map map by index
        /// Generated from method `MR::VertColorMapAggregator::getPartialColorMap`.
        public unsafe MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap GetPartialColorMap(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_getPartialColorMap", ExactSpelling = true)]
            extern static MR.ColorMapAggregator_MRVertTag.Const_PartialColorMap._Underlying *__MR_VertColorMapAggregator_getPartialColorMap(_Underlying *_this, int i);
            return new(__MR_VertColorMapAggregator_getPartialColorMap(_UnderlyingPtr, i), is_owning: false);
        }

        /// erase n color map from #i 
        /// Generated from method `MR::VertColorMapAggregator::erase`.
        /// Parameter `n` defaults to `1`.
        public unsafe void Erase(int i, int? n = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_erase", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_erase(_Underlying *_this, int i, int *n);
            int __deref_n = n.GetValueOrDefault();
            __MR_VertColorMapAggregator_erase(_UnderlyingPtr, i, n.HasValue ? &__deref_n : null);
        }

        /// set color map aggregating mode
        /// Generated from method `MR::VertColorMapAggregator::setMode`.
        public unsafe void SetMode(MR.ColorMapAggregator_MRVertTag.AggregateMode mode)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_setMode", ExactSpelling = true)]
            extern static void __MR_VertColorMapAggregator_setMode(_Underlying *_this, MR.ColorMapAggregator_MRVertTag.AggregateMode mode);
            __MR_VertColorMapAggregator_setMode(_UnderlyingPtr, mode);
        }

        /// get aggregated color map for active elements
        /// Generated from method `MR::VertColorMapAggregator::aggregate`.
        public unsafe MR.Misc._Moved<MR.VertColors> Aggregate(MR.Const_VertBitSet elementBitSet)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_VertColorMapAggregator_aggregate", ExactSpelling = true)]
            extern static MR.VertColors._Underlying *__MR_VertColorMapAggregator_aggregate(_Underlying *_this, MR.Const_VertBitSet._Underlying *elementBitSet);
            return MR.Misc.Move(new MR.VertColors(__MR_VertColorMapAggregator_aggregate(_UnderlyingPtr, elementBitSet._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `VertColorMapAggregator` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `VertColorMapAggregator`/`Const_VertColorMapAggregator` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_VertColorMapAggregator
    {
        internal readonly Const_VertColorMapAggregator? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_VertColorMapAggregator() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_VertColorMapAggregator(Const_VertColorMapAggregator new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_VertColorMapAggregator(Const_VertColorMapAggregator arg) {return new(arg);}
        public _ByValue_VertColorMapAggregator(MR.Misc._Moved<VertColorMapAggregator> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_VertColorMapAggregator(MR.Misc._Moved<VertColorMapAggregator> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `VertColorMapAggregator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_VertColorMapAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertColorMapAggregator`/`Const_VertColorMapAggregator` directly.
    public class _InOptMut_VertColorMapAggregator
    {
        public VertColorMapAggregator? Opt;

        public _InOptMut_VertColorMapAggregator() {}
        public _InOptMut_VertColorMapAggregator(VertColorMapAggregator value) {Opt = value;}
        public static implicit operator _InOptMut_VertColorMapAggregator(VertColorMapAggregator value) {return new(value);}
    }

    /// This is used for optional parameters of class `VertColorMapAggregator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_VertColorMapAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `VertColorMapAggregator`/`Const_VertColorMapAggregator` to pass it to the function.
    public class _InOptConst_VertColorMapAggregator
    {
        public Const_VertColorMapAggregator? Opt;

        public _InOptConst_VertColorMapAggregator() {}
        public _InOptConst_VertColorMapAggregator(Const_VertColorMapAggregator value) {Opt = value;}
        public static implicit operator _InOptConst_VertColorMapAggregator(Const_VertColorMapAggregator value) {return new(value);}
    }

    public static partial class ColorMapAggregator_MRUndirectedEdgeTag
    {
        /// partial color map
        /// Generated from class `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap`.
        /// This is the const half of the class.
        public class Const_PartialColorMap : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_PartialColorMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Destroy", ExactSpelling = true)]
                extern static void __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Destroy(_Underlying *_this);
                __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_PartialColorMap() {Dispose(false);}

            // color map
            public unsafe MR.Const_UndirectedEdgeColors ColorMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Get_colorMap", ExactSpelling = true)]
                    extern static MR.Const_UndirectedEdgeColors._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Get_colorMap(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Get_colorMap(_UnderlyingPtr), is_owning: false);
                }
            }

            // bitset of elements for which the color map is applied
            public unsafe MR.Const_UndirectedEdgeBitSet Elements
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Get_elements", ExactSpelling = true)]
                    extern static MR.Const_UndirectedEdgeBitSet._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Get_elements(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_Get_elements(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_PartialColorMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_DefaultConstruct();
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_DefaultConstruct();
            }

            /// Constructs `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap` elementwise.
            public unsafe Const_PartialColorMap(MR._ByValue_UndirectedEdgeColors colorMap, MR._ByValue_UndirectedEdgeBitSet elements) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFrom(MR.Misc._PassBy colorMap_pass_by, MR.UndirectedEdgeColors._Underlying *colorMap, MR.Misc._PassBy elements_pass_by, MR.UndirectedEdgeBitSet._Underlying *elements);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFrom(colorMap.PassByMode, colorMap.Value is not null ? colorMap.Value._UnderlyingPtr : null, elements.PassByMode, elements.Value is not null ? elements.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap::PartialColorMap`.
            public unsafe Const_PartialColorMap(MR.ColorMapAggregator_MRUndirectedEdgeTag._ByValue_PartialColorMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *_other);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// partial color map
        /// Generated from class `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap`.
        /// This is the non-const half of the class.
        public class PartialColorMap : Const_PartialColorMap
        {
            internal unsafe PartialColorMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // color map
            public new unsafe MR.UndirectedEdgeColors ColorMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_GetMutable_colorMap", ExactSpelling = true)]
                    extern static MR.UndirectedEdgeColors._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_GetMutable_colorMap(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_GetMutable_colorMap(_UnderlyingPtr), is_owning: false);
                }
            }

            // bitset of elements for which the color map is applied
            public new unsafe MR.UndirectedEdgeBitSet Elements
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_GetMutable_elements", ExactSpelling = true)]
                    extern static MR.UndirectedEdgeBitSet._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_GetMutable_elements(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_GetMutable_elements(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe PartialColorMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_DefaultConstruct();
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_DefaultConstruct();
            }

            /// Constructs `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap` elementwise.
            public unsafe PartialColorMap(MR._ByValue_UndirectedEdgeColors colorMap, MR._ByValue_UndirectedEdgeBitSet elements) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFrom(MR.Misc._PassBy colorMap_pass_by, MR.UndirectedEdgeColors._Underlying *colorMap, MR.Misc._PassBy elements_pass_by, MR.UndirectedEdgeBitSet._Underlying *elements);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFrom(colorMap.PassByMode, colorMap.Value is not null ? colorMap.Value._UnderlyingPtr : null, elements.PassByMode, elements.Value is not null ? elements.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap::PartialColorMap`.
            public unsafe PartialColorMap(MR.ColorMapAggregator_MRUndirectedEdgeTag._ByValue_PartialColorMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *_other);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::ColorMapAggregator<MR::UndirectedEdgeTag>::PartialColorMap::operator=`.
            public unsafe MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap Assign(MR.ColorMapAggregator_MRUndirectedEdgeTag._ByValue_PartialColorMap _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRUndirectedEdgeTag.PartialColorMap._Underlying *_other);
                return new(__MR_ColorMapAggregator_MR_UndirectedEdgeTag_PartialColorMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `PartialColorMap` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_PartialColorMap
        {
            internal readonly Const_PartialColorMap? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_PartialColorMap() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_PartialColorMap(Const_PartialColorMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_PartialColorMap(Const_PartialColorMap arg) {return new(arg);}
            public _ByValue_PartialColorMap(MR.Misc._Moved<PartialColorMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_PartialColorMap(MR.Misc._Moved<PartialColorMap> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `PartialColorMap` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialColorMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` directly.
        public class _InOptMut_PartialColorMap
        {
            public PartialColorMap? Opt;

            public _InOptMut_PartialColorMap() {}
            public _InOptMut_PartialColorMap(PartialColorMap value) {Opt = value;}
            public static implicit operator _InOptMut_PartialColorMap(PartialColorMap value) {return new(value);}
        }

        /// This is used for optional parameters of class `PartialColorMap` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialColorMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` to pass it to the function.
        public class _InOptConst_PartialColorMap
        {
            public Const_PartialColorMap? Opt;

            public _InOptConst_PartialColorMap() {}
            public _InOptConst_PartialColorMap(Const_PartialColorMap value) {Opt = value;}
            public static implicit operator _InOptConst_PartialColorMap(Const_PartialColorMap value) {return new(value);}
        }

        /// color map aggregating mode
        public enum AggregateMode : int
        {
            Overlay = 0,
            /// result element color is element color of more priority color map (or default color, if there isn't color map for this element)
            Blending = 1,
        }
    }

    /**
    * @brief Class for aggregate several color map in one
    * Color maps are aggregated according order
    */
    /// Generated from class `MR::UndirEdgeColorMapAggregator`.
    /// This is the const half of the class.
    public class Const_UndirEdgeColorMapAggregator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_UndirEdgeColorMapAggregator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_Destroy", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_Destroy(_Underlying *_this);
            __MR_UndirEdgeColorMapAggregator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_UndirEdgeColorMapAggregator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_UndirEdgeColorMapAggregator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirEdgeColorMapAggregator._Underlying *__MR_UndirEdgeColorMapAggregator_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirEdgeColorMapAggregator_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirEdgeColorMapAggregator::UndirEdgeColorMapAggregator`.
        public unsafe Const_UndirEdgeColorMapAggregator(MR._ByValue_UndirEdgeColorMapAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirEdgeColorMapAggregator._Underlying *__MR_UndirEdgeColorMapAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UndirEdgeColorMapAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_UndirEdgeColorMapAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * @brief Class for aggregate several color map in one
    * Color maps are aggregated according order
    */
    /// Generated from class `MR::UndirEdgeColorMapAggregator`.
    /// This is the non-const half of the class.
    public class UndirEdgeColorMapAggregator : Const_UndirEdgeColorMapAggregator
    {
        internal unsafe UndirEdgeColorMapAggregator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe UndirEdgeColorMapAggregator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.UndirEdgeColorMapAggregator._Underlying *__MR_UndirEdgeColorMapAggregator_DefaultConstruct();
            _UnderlyingPtr = __MR_UndirEdgeColorMapAggregator_DefaultConstruct();
        }

        /// Generated from constructor `MR::UndirEdgeColorMapAggregator::UndirEdgeColorMapAggregator`.
        public unsafe UndirEdgeColorMapAggregator(MR._ByValue_UndirEdgeColorMapAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.UndirEdgeColorMapAggregator._Underlying *__MR_UndirEdgeColorMapAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.UndirEdgeColorMapAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_UndirEdgeColorMapAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::UndirEdgeColorMapAggregator::operator=`.
        public unsafe MR.UndirEdgeColorMapAggregator Assign(MR._ByValue_UndirEdgeColorMapAggregator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.UndirEdgeColorMapAggregator._Underlying *__MR_UndirEdgeColorMapAggregator_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.UndirEdgeColorMapAggregator._Underlying *_other);
            return new(__MR_UndirEdgeColorMapAggregator_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// set default (background) color
        /// Generated from method `MR::UndirEdgeColorMapAggregator::setDefaultColor`.
        public unsafe void SetDefaultColor(MR.Const_Color color)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_setDefaultColor", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_setDefaultColor(_Underlying *_this, MR.Const_Color._Underlying *color);
            __MR_UndirEdgeColorMapAggregator_setDefaultColor(_UnderlyingPtr, color._UnderlyingPtr);
        }

        /// add color map after all (more priority)
        /// Generated from method `MR::UndirEdgeColorMapAggregator::pushBack`.
        public unsafe void PushBack(MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_pushBack", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_pushBack(_Underlying *_this, MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_UndirEdgeColorMapAggregator_pushBack(_UnderlyingPtr, partitialColorMap._UnderlyingPtr);
        }

        /// insert color map before element #i (0 - minimum priority)
        /// Generated from method `MR::UndirEdgeColorMapAggregator::insert`.
        public unsafe void Insert(int i, MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_insert", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_insert(_Underlying *_this, int i, MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_UndirEdgeColorMapAggregator_insert(_UnderlyingPtr, i, partitialColorMap._UnderlyingPtr);
        }

        /// replace color map in #i position
        /// Generated from method `MR::UndirEdgeColorMapAggregator::replace`.
        public unsafe void Replace(int i, MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_replace", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_replace(_Underlying *_this, int i, MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_UndirEdgeColorMapAggregator_replace(_UnderlyingPtr, i, partitialColorMap._UnderlyingPtr);
        }

        /// reset all accumulated color map
        /// Generated from method `MR::UndirEdgeColorMapAggregator::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_reset", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_reset(_Underlying *_this);
            __MR_UndirEdgeColorMapAggregator_reset(_UnderlyingPtr);
        }

        /// get number of accumulated color maps
        /// Generated from method `MR::UndirEdgeColorMapAggregator::getColorMapNumber`.
        public unsafe ulong GetColorMapNumber()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_getColorMapNumber", ExactSpelling = true)]
            extern static ulong __MR_UndirEdgeColorMapAggregator_getColorMapNumber(_Underlying *_this);
            return __MR_UndirEdgeColorMapAggregator_getColorMapNumber(_UnderlyingPtr);
        }

        /// get partial color map map by index
        /// Generated from method `MR::UndirEdgeColorMapAggregator::getPartialColorMap`.
        public unsafe MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap GetPartialColorMap(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_getPartialColorMap", ExactSpelling = true)]
            extern static MR.ColorMapAggregator_MRUndirectedEdgeTag.Const_PartialColorMap._Underlying *__MR_UndirEdgeColorMapAggregator_getPartialColorMap(_Underlying *_this, int i);
            return new(__MR_UndirEdgeColorMapAggregator_getPartialColorMap(_UnderlyingPtr, i), is_owning: false);
        }

        /// erase n color map from #i 
        /// Generated from method `MR::UndirEdgeColorMapAggregator::erase`.
        /// Parameter `n` defaults to `1`.
        public unsafe void Erase(int i, int? n = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_erase", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_erase(_Underlying *_this, int i, int *n);
            int __deref_n = n.GetValueOrDefault();
            __MR_UndirEdgeColorMapAggregator_erase(_UnderlyingPtr, i, n.HasValue ? &__deref_n : null);
        }

        /// set color map aggregating mode
        /// Generated from method `MR::UndirEdgeColorMapAggregator::setMode`.
        public unsafe void SetMode(MR.ColorMapAggregator_MRUndirectedEdgeTag.AggregateMode mode)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_setMode", ExactSpelling = true)]
            extern static void __MR_UndirEdgeColorMapAggregator_setMode(_Underlying *_this, MR.ColorMapAggregator_MRUndirectedEdgeTag.AggregateMode mode);
            __MR_UndirEdgeColorMapAggregator_setMode(_UnderlyingPtr, mode);
        }

        /// get aggregated color map for active elements
        /// Generated from method `MR::UndirEdgeColorMapAggregator::aggregate`.
        public unsafe MR.Misc._Moved<MR.UndirectedEdgeColors> Aggregate(MR.Const_UndirectedEdgeBitSet elementBitSet)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_UndirEdgeColorMapAggregator_aggregate", ExactSpelling = true)]
            extern static MR.UndirectedEdgeColors._Underlying *__MR_UndirEdgeColorMapAggregator_aggregate(_Underlying *_this, MR.Const_UndirectedEdgeBitSet._Underlying *elementBitSet);
            return MR.Misc.Move(new MR.UndirectedEdgeColors(__MR_UndirEdgeColorMapAggregator_aggregate(_UnderlyingPtr, elementBitSet._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `UndirEdgeColorMapAggregator` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `UndirEdgeColorMapAggregator`/`Const_UndirEdgeColorMapAggregator` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_UndirEdgeColorMapAggregator
    {
        internal readonly Const_UndirEdgeColorMapAggregator? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_UndirEdgeColorMapAggregator() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_UndirEdgeColorMapAggregator(Const_UndirEdgeColorMapAggregator new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_UndirEdgeColorMapAggregator(Const_UndirEdgeColorMapAggregator arg) {return new(arg);}
        public _ByValue_UndirEdgeColorMapAggregator(MR.Misc._Moved<UndirEdgeColorMapAggregator> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_UndirEdgeColorMapAggregator(MR.Misc._Moved<UndirEdgeColorMapAggregator> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `UndirEdgeColorMapAggregator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_UndirEdgeColorMapAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirEdgeColorMapAggregator`/`Const_UndirEdgeColorMapAggregator` directly.
    public class _InOptMut_UndirEdgeColorMapAggregator
    {
        public UndirEdgeColorMapAggregator? Opt;

        public _InOptMut_UndirEdgeColorMapAggregator() {}
        public _InOptMut_UndirEdgeColorMapAggregator(UndirEdgeColorMapAggregator value) {Opt = value;}
        public static implicit operator _InOptMut_UndirEdgeColorMapAggregator(UndirEdgeColorMapAggregator value) {return new(value);}
    }

    /// This is used for optional parameters of class `UndirEdgeColorMapAggregator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_UndirEdgeColorMapAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `UndirEdgeColorMapAggregator`/`Const_UndirEdgeColorMapAggregator` to pass it to the function.
    public class _InOptConst_UndirEdgeColorMapAggregator
    {
        public Const_UndirEdgeColorMapAggregator? Opt;

        public _InOptConst_UndirEdgeColorMapAggregator() {}
        public _InOptConst_UndirEdgeColorMapAggregator(Const_UndirEdgeColorMapAggregator value) {Opt = value;}
        public static implicit operator _InOptConst_UndirEdgeColorMapAggregator(Const_UndirEdgeColorMapAggregator value) {return new(value);}
    }

    public static partial class ColorMapAggregator_MRFaceTag
    {
        /// partial color map
        /// Generated from class `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap`.
        /// This is the const half of the class.
        public class Const_PartialColorMap : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_PartialColorMap(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Destroy", ExactSpelling = true)]
                extern static void __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Destroy(_Underlying *_this);
                __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_PartialColorMap() {Dispose(false);}

            // color map
            public unsafe MR.Const_FaceColors ColorMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Get_colorMap", ExactSpelling = true)]
                    extern static MR.Const_FaceColors._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Get_colorMap(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Get_colorMap(_UnderlyingPtr), is_owning: false);
                }
            }

            // bitset of elements for which the color map is applied
            public unsafe MR.Const_FaceBitSet Elements
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Get_elements", ExactSpelling = true)]
                    extern static MR.Const_FaceBitSet._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Get_elements(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_Get_elements(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_PartialColorMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_DefaultConstruct();
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_DefaultConstruct();
            }

            /// Constructs `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap` elementwise.
            public unsafe Const_PartialColorMap(MR._ByValue_FaceColors colorMap, MR._ByValue_FaceBitSet elements) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFrom(MR.Misc._PassBy colorMap_pass_by, MR.FaceColors._Underlying *colorMap, MR.Misc._PassBy elements_pass_by, MR.FaceBitSet._Underlying *elements);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFrom(colorMap.PassByMode, colorMap.Value is not null ? colorMap.Value._UnderlyingPtr : null, elements.PassByMode, elements.Value is not null ? elements.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap::PartialColorMap`.
            public unsafe Const_PartialColorMap(MR.ColorMapAggregator_MRFaceTag._ByValue_PartialColorMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *_other);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }
        }

        /// partial color map
        /// Generated from class `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap`.
        /// This is the non-const half of the class.
        public class PartialColorMap : Const_PartialColorMap
        {
            internal unsafe PartialColorMap(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            // color map
            public new unsafe MR.FaceColors ColorMap
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_GetMutable_colorMap", ExactSpelling = true)]
                    extern static MR.FaceColors._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_GetMutable_colorMap(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_GetMutable_colorMap(_UnderlyingPtr), is_owning: false);
                }
            }

            // bitset of elements for which the color map is applied
            public new unsafe MR.FaceBitSet Elements
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_GetMutable_elements", ExactSpelling = true)]
                    extern static MR.FaceBitSet._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_GetMutable_elements(_Underlying *_this);
                    return new(__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_GetMutable_elements(_UnderlyingPtr), is_owning: false);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe PartialColorMap() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_DefaultConstruct", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_DefaultConstruct();
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_DefaultConstruct();
            }

            /// Constructs `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap` elementwise.
            public unsafe PartialColorMap(MR._ByValue_FaceColors colorMap, MR._ByValue_FaceBitSet elements) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFrom", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFrom(MR.Misc._PassBy colorMap_pass_by, MR.FaceColors._Underlying *colorMap, MR.Misc._PassBy elements_pass_by, MR.FaceBitSet._Underlying *elements);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFrom(colorMap.PassByMode, colorMap.Value is not null ? colorMap.Value._UnderlyingPtr : null, elements.PassByMode, elements.Value is not null ? elements.Value._UnderlyingPtr : null);
            }

            /// Generated from constructor `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap::PartialColorMap`.
            public unsafe PartialColorMap(MR.ColorMapAggregator_MRFaceTag._ByValue_PartialColorMap _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *_other);
                _UnderlyingPtr = __MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
            }

            /// Generated from method `MR::ColorMapAggregator<MR::FaceTag>::PartialColorMap::operator=`.
            public unsafe MR.ColorMapAggregator_MRFaceTag.PartialColorMap Assign(MR.ColorMapAggregator_MRFaceTag._ByValue_PartialColorMap _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_AssignFromAnother", ExactSpelling = true)]
                extern static MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.ColorMapAggregator_MRFaceTag.PartialColorMap._Underlying *_other);
                return new(__MR_ColorMapAggregator_MR_FaceTag_PartialColorMap_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
            }
        }

        /// This is used as a function parameter when the underlying function receives `PartialColorMap` by value.
        /// Usage:
        /// * Pass `new()` to default-construct the instance.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` to copy it into the function.
        /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
        ///   Be careful if your input isn't a unique reference to this object.
        /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
        public class _ByValue_PartialColorMap
        {
            internal readonly Const_PartialColorMap? Value;
            internal readonly MR.Misc._PassBy PassByMode;
            public _ByValue_PartialColorMap() {PassByMode = MR.Misc._PassBy.default_construct;}
            public _ByValue_PartialColorMap(Const_PartialColorMap new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
            public static implicit operator _ByValue_PartialColorMap(Const_PartialColorMap arg) {return new(arg);}
            public _ByValue_PartialColorMap(MR.Misc._Moved<PartialColorMap> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
            public static implicit operator _ByValue_PartialColorMap(MR.Misc._Moved<PartialColorMap> arg) {return new(arg);}
        }

        /// This is used for optional parameters of class `PartialColorMap` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_PartialColorMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` directly.
        public class _InOptMut_PartialColorMap
        {
            public PartialColorMap? Opt;

            public _InOptMut_PartialColorMap() {}
            public _InOptMut_PartialColorMap(PartialColorMap value) {Opt = value;}
            public static implicit operator _InOptMut_PartialColorMap(PartialColorMap value) {return new(value);}
        }

        /// This is used for optional parameters of class `PartialColorMap` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_PartialColorMap`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `PartialColorMap`/`Const_PartialColorMap` to pass it to the function.
        public class _InOptConst_PartialColorMap
        {
            public Const_PartialColorMap? Opt;

            public _InOptConst_PartialColorMap() {}
            public _InOptConst_PartialColorMap(Const_PartialColorMap value) {Opt = value;}
            public static implicit operator _InOptConst_PartialColorMap(Const_PartialColorMap value) {return new(value);}
        }

        /// color map aggregating mode
        public enum AggregateMode : int
        {
            Overlay = 0,
            /// result element color is element color of more priority color map (or default color, if there isn't color map for this element)
            Blending = 1,
        }
    }

    /**
    * @brief Class for aggregate several color map in one
    * Color maps are aggregated according order
    */
    /// Generated from class `MR::FaceColorMapAggregator`.
    /// This is the const half of the class.
    public class Const_FaceColorMapAggregator : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_FaceColorMapAggregator(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_Destroy", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_Destroy(_Underlying *_this);
            __MR_FaceColorMapAggregator_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_FaceColorMapAggregator() {Dispose(false);}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_FaceColorMapAggregator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceColorMapAggregator._Underlying *__MR_FaceColorMapAggregator_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceColorMapAggregator_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceColorMapAggregator::FaceColorMapAggregator`.
        public unsafe Const_FaceColorMapAggregator(MR._ByValue_FaceColorMapAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceColorMapAggregator._Underlying *__MR_FaceColorMapAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceColorMapAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_FaceColorMapAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }
    }

    /**
    * @brief Class for aggregate several color map in one
    * Color maps are aggregated according order
    */
    /// Generated from class `MR::FaceColorMapAggregator`.
    /// This is the non-const half of the class.
    public class FaceColorMapAggregator : Const_FaceColorMapAggregator
    {
        internal unsafe FaceColorMapAggregator(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Constructs an empty (default-constructed) instance.
        public unsafe FaceColorMapAggregator() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_DefaultConstruct", ExactSpelling = true)]
            extern static MR.FaceColorMapAggregator._Underlying *__MR_FaceColorMapAggregator_DefaultConstruct();
            _UnderlyingPtr = __MR_FaceColorMapAggregator_DefaultConstruct();
        }

        /// Generated from constructor `MR::FaceColorMapAggregator::FaceColorMapAggregator`.
        public unsafe FaceColorMapAggregator(MR._ByValue_FaceColorMapAggregator _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.FaceColorMapAggregator._Underlying *__MR_FaceColorMapAggregator_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.FaceColorMapAggregator._Underlying *_other);
            _UnderlyingPtr = __MR_FaceColorMapAggregator_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::FaceColorMapAggregator::operator=`.
        public unsafe MR.FaceColorMapAggregator Assign(MR._ByValue_FaceColorMapAggregator _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_AssignFromAnother", ExactSpelling = true)]
            extern static MR.FaceColorMapAggregator._Underlying *__MR_FaceColorMapAggregator_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.FaceColorMapAggregator._Underlying *_other);
            return new(__MR_FaceColorMapAggregator_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// set default (background) color
        /// Generated from method `MR::FaceColorMapAggregator::setDefaultColor`.
        public unsafe void SetDefaultColor(MR.Const_Color color)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_setDefaultColor", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_setDefaultColor(_Underlying *_this, MR.Const_Color._Underlying *color);
            __MR_FaceColorMapAggregator_setDefaultColor(_UnderlyingPtr, color._UnderlyingPtr);
        }

        /// add color map after all (more priority)
        /// Generated from method `MR::FaceColorMapAggregator::pushBack`.
        public unsafe void PushBack(MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_pushBack", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_pushBack(_Underlying *_this, MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_FaceColorMapAggregator_pushBack(_UnderlyingPtr, partitialColorMap._UnderlyingPtr);
        }

        /// insert color map before element #i (0 - minimum priority)
        /// Generated from method `MR::FaceColorMapAggregator::insert`.
        public unsafe void Insert(int i, MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_insert", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_insert(_Underlying *_this, int i, MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_FaceColorMapAggregator_insert(_UnderlyingPtr, i, partitialColorMap._UnderlyingPtr);
        }

        /// replace color map in #i position
        /// Generated from method `MR::FaceColorMapAggregator::replace`.
        public unsafe void Replace(int i, MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap partitialColorMap)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_replace", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_replace(_Underlying *_this, int i, MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap._Underlying *partitialColorMap);
            __MR_FaceColorMapAggregator_replace(_UnderlyingPtr, i, partitialColorMap._UnderlyingPtr);
        }

        /// reset all accumulated color map
        /// Generated from method `MR::FaceColorMapAggregator::reset`.
        public unsafe void Reset()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_reset", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_reset(_Underlying *_this);
            __MR_FaceColorMapAggregator_reset(_UnderlyingPtr);
        }

        /// get number of accumulated color maps
        /// Generated from method `MR::FaceColorMapAggregator::getColorMapNumber`.
        public unsafe ulong GetColorMapNumber()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_getColorMapNumber", ExactSpelling = true)]
            extern static ulong __MR_FaceColorMapAggregator_getColorMapNumber(_Underlying *_this);
            return __MR_FaceColorMapAggregator_getColorMapNumber(_UnderlyingPtr);
        }

        /// get partial color map map by index
        /// Generated from method `MR::FaceColorMapAggregator::getPartialColorMap`.
        public unsafe MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap GetPartialColorMap(int i)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_getPartialColorMap", ExactSpelling = true)]
            extern static MR.ColorMapAggregator_MRFaceTag.Const_PartialColorMap._Underlying *__MR_FaceColorMapAggregator_getPartialColorMap(_Underlying *_this, int i);
            return new(__MR_FaceColorMapAggregator_getPartialColorMap(_UnderlyingPtr, i), is_owning: false);
        }

        /// erase n color map from #i 
        /// Generated from method `MR::FaceColorMapAggregator::erase`.
        /// Parameter `n` defaults to `1`.
        public unsafe void Erase(int i, int? n = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_erase", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_erase(_Underlying *_this, int i, int *n);
            int __deref_n = n.GetValueOrDefault();
            __MR_FaceColorMapAggregator_erase(_UnderlyingPtr, i, n.HasValue ? &__deref_n : null);
        }

        /// set color map aggregating mode
        /// Generated from method `MR::FaceColorMapAggregator::setMode`.
        public unsafe void SetMode(MR.ColorMapAggregator_MRFaceTag.AggregateMode mode)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_setMode", ExactSpelling = true)]
            extern static void __MR_FaceColorMapAggregator_setMode(_Underlying *_this, MR.ColorMapAggregator_MRFaceTag.AggregateMode mode);
            __MR_FaceColorMapAggregator_setMode(_UnderlyingPtr, mode);
        }

        /// get aggregated color map for active elements
        /// Generated from method `MR::FaceColorMapAggregator::aggregate`.
        public unsafe MR.Misc._Moved<MR.FaceColors> Aggregate(MR.Const_FaceBitSet elementBitSet)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_FaceColorMapAggregator_aggregate", ExactSpelling = true)]
            extern static MR.FaceColors._Underlying *__MR_FaceColorMapAggregator_aggregate(_Underlying *_this, MR.Const_FaceBitSet._Underlying *elementBitSet);
            return MR.Misc.Move(new MR.FaceColors(__MR_FaceColorMapAggregator_aggregate(_UnderlyingPtr, elementBitSet._UnderlyingPtr), is_owning: true));
        }
    }

    /// This is used as a function parameter when the underlying function receives `FaceColorMapAggregator` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `FaceColorMapAggregator`/`Const_FaceColorMapAggregator` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_FaceColorMapAggregator
    {
        internal readonly Const_FaceColorMapAggregator? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_FaceColorMapAggregator() {PassByMode = MR.Misc._PassBy.default_construct;}
        public _ByValue_FaceColorMapAggregator(Const_FaceColorMapAggregator new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_FaceColorMapAggregator(Const_FaceColorMapAggregator arg) {return new(arg);}
        public _ByValue_FaceColorMapAggregator(MR.Misc._Moved<FaceColorMapAggregator> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_FaceColorMapAggregator(MR.Misc._Moved<FaceColorMapAggregator> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `FaceColorMapAggregator` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_FaceColorMapAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceColorMapAggregator`/`Const_FaceColorMapAggregator` directly.
    public class _InOptMut_FaceColorMapAggregator
    {
        public FaceColorMapAggregator? Opt;

        public _InOptMut_FaceColorMapAggregator() {}
        public _InOptMut_FaceColorMapAggregator(FaceColorMapAggregator value) {Opt = value;}
        public static implicit operator _InOptMut_FaceColorMapAggregator(FaceColorMapAggregator value) {return new(value);}
    }

    /// This is used for optional parameters of class `FaceColorMapAggregator` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_FaceColorMapAggregator`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `FaceColorMapAggregator`/`Const_FaceColorMapAggregator` to pass it to the function.
    public class _InOptConst_FaceColorMapAggregator
    {
        public Const_FaceColorMapAggregator? Opt;

        public _InOptConst_FaceColorMapAggregator() {}
        public _InOptConst_FaceColorMapAggregator(Const_FaceColorMapAggregator value) {Opt = value;}
        public static implicit operator _InOptConst_FaceColorMapAggregator(Const_FaceColorMapAggregator value) {return new(value);}
    }
}
