public static partial class MR
{
    /**
    * \brief stores map from element id in[0, size) to T;
    * \details provides two operations:
    * 1) change the value of any element;
    * 2) find the element with the largest value
    */
    /// Generated from class `MR::Heap<float, MR::GraphVertId, std::greater<float>>`.
    /// This is the const half of the class.
    public class Const_Heap_Float_MRGraphVertId_StdGreaterFloat : MR.Misc.Object, System.IDisposable
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Heap_Float_MRGraphVertId_StdGreaterFloat(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Destroy", ExactSpelling = true)]
            extern static void __MR_Heap_float_MR_GraphVertId_std_greater_float_Destroy(_Underlying *_this);
            __MR_Heap_float_MR_GraphVertId_std_greater_float_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Heap_Float_MRGraphVertId_StdGreaterFloat() {Dispose(false);}

        /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Heap`.
        public unsafe Const_Heap_Float_MRGraphVertId_StdGreaterFloat(MR._ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *_other);
            _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// constructs heap for given number of elements, assigning given default value to each element
        /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Heap`.
        /// Parameter `def` defaults to `{}`.
        public unsafe Const_Heap_Float_MRGraphVertId_StdGreaterFloat(ulong size, float? def = null, MR.Std.Greater_Float pred = default) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_3", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_3(ulong size, float *def);
            float __deref_def = def.GetValueOrDefault();
            _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_3(size, def.HasValue ? &__deref_def : null);
        }

        /// constructs heap from given elements (id's shall not repeat and have spaces, but can be arbitrary shuffled)
        /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Heap`.
        public unsafe Const_Heap_Float_MRGraphVertId_StdGreaterFloat(MR.Std._ByValue_Vector_MRHeapFloatMRGraphVertIdStdGreaterFloatElement elms, MR.Std.Greater_Float pred = default) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_2", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_2(MR.Misc._PassBy elms_pass_by, MR.Std.Vector_MRHeapFloatMRGraphVertIdStdGreaterFloatElement._Underlying *elms);
            _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_2(elms.PassByMode, elms.Value is not null ? elms.Value._UnderlyingPtr : null);
        }

        /// returns the size of the heap
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::size`.
        public unsafe ulong Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_size", ExactSpelling = true)]
            extern static ulong __MR_Heap_float_MR_GraphVertId_std_greater_float_size(_Underlying *_this);
            return __MR_Heap_float_MR_GraphVertId_std_greater_float_size(_UnderlyingPtr);
        }

        /// returns the value associated with given element
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::value`.
        public unsafe float Value(MR.GraphVertId elemId)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_value", ExactSpelling = true)]
            extern static float *__MR_Heap_float_MR_GraphVertId_std_greater_float_value(_Underlying *_this, MR.GraphVertId elemId);
            return *__MR_Heap_float_MR_GraphVertId_std_greater_float_value(_UnderlyingPtr, elemId);
        }

        /// returns the element with the largest value
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::top`.
        public unsafe MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Const_Element Top()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_top", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Const_Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_top(_Underlying *_this);
            return new(__MR_Heap_float_MR_GraphVertId_std_greater_float_top(_UnderlyingPtr), is_owning: false);
        }

        /// Generated from class `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element`.
        /// This is the const half of the class.
        public class Const_Element : MR.Misc.Object, System.IDisposable
        {
            internal struct _Underlying; // Represents the underlying C++ type.

            internal unsafe _Underlying *_UnderlyingPtr;

            internal unsafe Const_Element(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

            protected virtual unsafe void Dispose(bool disposing)
            {
                if (_UnderlyingPtr is null || !_IsOwningVal)
                    return;
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Destroy", ExactSpelling = true)]
                extern static void __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Destroy(_Underlying *_this);
                __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Destroy(_UnderlyingPtr);
                _UnderlyingPtr = null;
            }
            public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
            ~Const_Element() {Dispose(false);}

            public unsafe MR.Const_GraphVertId Id
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Get_id", ExactSpelling = true)]
                    extern static MR.Const_GraphVertId._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Get_id(_Underlying *_this);
                    return new(__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Get_id(_UnderlyingPtr), is_owning: false);
                }
            }

            public unsafe float Val
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Get_val", ExactSpelling = true)]
                    extern static float *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Get_val(_Underlying *_this);
                    return *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_Get_val(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Const_Element() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_DefaultConstruct();
                _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_DefaultConstruct();
            }

            /// Constructs `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element` elementwise.
            public unsafe Const_Element(MR.GraphVertId id, float val) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFrom", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFrom(MR.GraphVertId id, float val);
                _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFrom(id, val);
            }

            /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element::Element`.
            public unsafe Const_Element(MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Const_Element _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFromAnother(MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *_other);
                _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFromAnother(_other._UnderlyingPtr);
            }
        }

        /// Generated from class `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element`.
        /// This is the non-const half of the class.
        public class Element : Const_Element
        {
            internal unsafe Element(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

            public new unsafe MR.Mut_GraphVertId Id
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_GetMutable_id", ExactSpelling = true)]
                    extern static MR.Mut_GraphVertId._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_GetMutable_id(_Underlying *_this);
                    return new(__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_GetMutable_id(_UnderlyingPtr), is_owning: false);
                }
            }

            public new unsafe ref float Val
            {
                get
                {
                    [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_GetMutable_val", ExactSpelling = true)]
                    extern static float *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_GetMutable_val(_Underlying *_this);
                    return ref *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_GetMutable_val(_UnderlyingPtr);
                }
            }

            /// Constructs an empty (default-constructed) instance.
            public unsafe Element() : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_DefaultConstruct", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_DefaultConstruct();
                _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_DefaultConstruct();
            }

            /// Constructs `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element` elementwise.
            public unsafe Element(MR.GraphVertId id, float val) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFrom", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFrom(MR.GraphVertId id, float val);
                _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFrom(id, val);
            }

            /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element::Element`.
            public unsafe Element(MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Const_Element _other) : this(null, is_owning: true)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFromAnother", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFromAnother(MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *_other);
                _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Element_ConstructFromAnother(_other._UnderlyingPtr);
            }

            /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Element::operator=`.
            public unsafe MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element Assign(MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Const_Element _other)
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Element_AssignFromAnother", ExactSpelling = true)]
                extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_AssignFromAnother(_Underlying *_this, MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *_other);
                return new(__MR_Heap_float_MR_GraphVertId_std_greater_float_Element_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
            }
        }

        /// This is used for optional parameters of class `Element` with default arguments.
        /// This is only used mutable parameters. For const ones we have `_InOptConst_Element`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Element`/`Const_Element` directly.
        public class _InOptMut_Element
        {
            public Element? Opt;

            public _InOptMut_Element() {}
            public _InOptMut_Element(Element value) {Opt = value;}
            public static implicit operator _InOptMut_Element(Element value) {return new(value);}
        }

        /// This is used for optional parameters of class `Element` with default arguments.
        /// This is only used const parameters. For non-const ones we have `_InOptMut_Element`.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `Element`/`Const_Element` to pass it to the function.
        public class _InOptConst_Element
        {
            public Const_Element? Opt;

            public _InOptConst_Element() {}
            public _InOptConst_Element(Const_Element value) {Opt = value;}
            public static implicit operator _InOptConst_Element(Const_Element value) {return new(value);}
        }
    }

    /**
    * \brief stores map from element id in[0, size) to T;
    * \details provides two operations:
    * 1) change the value of any element;
    * 2) find the element with the largest value
    */
    /// Generated from class `MR::Heap<float, MR::GraphVertId, std::greater<float>>`.
    /// This is the non-const half of the class.
    public class Heap_Float_MRGraphVertId_StdGreaterFloat : Const_Heap_Float_MRGraphVertId_StdGreaterFloat
    {
        internal unsafe Heap_Float_MRGraphVertId_StdGreaterFloat(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Heap`.
        public unsafe Heap_Float_MRGraphVertId_StdGreaterFloat(MR._ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_ConstructFromAnother(MR.Misc._PassBy _other_pass_by, MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *_other);
            _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_ConstructFromAnother(_other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null);
        }

        /// constructs heap for given number of elements, assigning given default value to each element
        /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Heap`.
        /// Parameter `def` defaults to `{}`.
        public unsafe Heap_Float_MRGraphVertId_StdGreaterFloat(ulong size, float? def = null, MR.Std.Greater_Float pred = default) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_3", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_3(ulong size, float *def);
            float __deref_def = def.GetValueOrDefault();
            _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_3(size, def.HasValue ? &__deref_def : null);
        }

        /// constructs heap from given elements (id's shall not repeat and have spaces, but can be arbitrary shuffled)
        /// Generated from constructor `MR::Heap<float, MR::GraphVertId, std::greater<float>>::Heap`.
        public unsafe Heap_Float_MRGraphVertId_StdGreaterFloat(MR.Std._ByValue_Vector_MRHeapFloatMRGraphVertIdStdGreaterFloatElement elms, MR.Std.Greater_Float pred = default) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_2", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_2(MR.Misc._PassBy elms_pass_by, MR.Std.Vector_MRHeapFloatMRGraphVertIdStdGreaterFloatElement._Underlying *elms);
            _UnderlyingPtr = __MR_Heap_float_MR_GraphVertId_std_greater_float_Construct_2(elms.PassByMode, elms.Value is not null ? elms.Value._UnderlyingPtr : null);
        }

        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::operator=`.
        public unsafe MR.Heap_Float_MRGraphVertId_StdGreaterFloat Assign(MR._ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_AssignFromAnother(_Underlying *_this, MR.Misc._PassBy _other_pass_by, MR.Heap_Float_MRGraphVertId_StdGreaterFloat._Underlying *_other);
            return new(__MR_Heap_float_MR_GraphVertId_std_greater_float_AssignFromAnother(_UnderlyingPtr, _other.PassByMode, _other.Value is not null ? _other.Value._UnderlyingPtr : null), is_owning: false);
        }

        /// increases the size of the heap by adding elements at the end
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::resize`.
        /// Parameter `def` defaults to `{}`.
        public unsafe void Resize(ulong size, float? def = null)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_resize", ExactSpelling = true)]
            extern static void __MR_Heap_float_MR_GraphVertId_std_greater_float_resize(_Underlying *_this, ulong size, float *def);
            float __deref_def = def.GetValueOrDefault();
            __MR_Heap_float_MR_GraphVertId_std_greater_float_resize(_UnderlyingPtr, size, def.HasValue ? &__deref_def : null);
        }

        /// sets new value to given element
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::setValue`.
        public unsafe void SetValue(MR.GraphVertId elemId, float newVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_setValue", ExactSpelling = true)]
            extern static void __MR_Heap_float_MR_GraphVertId_std_greater_float_setValue(_Underlying *_this, MR.GraphVertId elemId, float *newVal);
            __MR_Heap_float_MR_GraphVertId_std_greater_float_setValue(_UnderlyingPtr, elemId, &newVal);
        }

        /// sets new value to given element, which shall be larger/smaller than the current value
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::setLargerValue`.
        public unsafe void SetLargerValue(MR.GraphVertId elemId, float newVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_setLargerValue", ExactSpelling = true)]
            extern static void __MR_Heap_float_MR_GraphVertId_std_greater_float_setLargerValue(_Underlying *_this, MR.GraphVertId elemId, float *newVal);
            __MR_Heap_float_MR_GraphVertId_std_greater_float_setLargerValue(_UnderlyingPtr, elemId, &newVal);
        }

        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::setSmallerValue`.
        public unsafe void SetSmallerValue(MR.GraphVertId elemId, float newVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_setSmallerValue", ExactSpelling = true)]
            extern static void __MR_Heap_float_MR_GraphVertId_std_greater_float_setSmallerValue(_Underlying *_this, MR.GraphVertId elemId, float *newVal);
            __MR_Heap_float_MR_GraphVertId_std_greater_float_setSmallerValue(_UnderlyingPtr, elemId, &newVal);
        }

        /// sets new value to the current top element, returning its previous value
        /// Generated from method `MR::Heap<float, MR::GraphVertId, std::greater<float>>::setTopValue`.
        public unsafe MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element SetTopValue(float newVal)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Heap_float_MR_GraphVertId_std_greater_float_setTopValue", ExactSpelling = true)]
            extern static MR.Heap_Float_MRGraphVertId_StdGreaterFloat.Element._Underlying *__MR_Heap_float_MR_GraphVertId_std_greater_float_setTopValue(_Underlying *_this, float *newVal);
            return new(__MR_Heap_float_MR_GraphVertId_std_greater_float_setTopValue(_UnderlyingPtr, &newVal), is_owning: true);
        }
    }

    /// This is used as a function parameter when the underlying function receives `Heap_Float_MRGraphVertId_StdGreaterFloat` by value.
    /// Usage:
    /// * Pass `new()` to default-construct the instance.
    /// * Pass an instance of `Heap_Float_MRGraphVertId_StdGreaterFloat`/`Const_Heap_Float_MRGraphVertId_StdGreaterFloat` to copy it into the function.
    /// * Pass `Move(instance)` to move it into the function. This is a more efficient form of copying that might invalidate the input object.
    ///   Be careful if your input isn't a unique reference to this object.
    /// * Pass `null` to use the default argument, assuming the parameter has a default argument (has `?` in the type).
    public class _ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat
    {
        internal readonly Const_Heap_Float_MRGraphVertId_StdGreaterFloat? Value;
        internal readonly MR.Misc._PassBy PassByMode;
        public _ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat(Const_Heap_Float_MRGraphVertId_StdGreaterFloat new_value) {Value = new_value; PassByMode = MR.Misc._PassBy.copy;}
        public static implicit operator _ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat(Const_Heap_Float_MRGraphVertId_StdGreaterFloat arg) {return new(arg);}
        public _ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat(MR.Misc._Moved<Heap_Float_MRGraphVertId_StdGreaterFloat> moved) {Value = moved.Value; PassByMode = MR.Misc._PassBy.move;}
        public static implicit operator _ByValue_Heap_Float_MRGraphVertId_StdGreaterFloat(MR.Misc._Moved<Heap_Float_MRGraphVertId_StdGreaterFloat> arg) {return new(arg);}
    }

    /// This is used for optional parameters of class `Heap_Float_MRGraphVertId_StdGreaterFloat` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Heap_Float_MRGraphVertId_StdGreaterFloat`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Heap_Float_MRGraphVertId_StdGreaterFloat`/`Const_Heap_Float_MRGraphVertId_StdGreaterFloat` directly.
    public class _InOptMut_Heap_Float_MRGraphVertId_StdGreaterFloat
    {
        public Heap_Float_MRGraphVertId_StdGreaterFloat? Opt;

        public _InOptMut_Heap_Float_MRGraphVertId_StdGreaterFloat() {}
        public _InOptMut_Heap_Float_MRGraphVertId_StdGreaterFloat(Heap_Float_MRGraphVertId_StdGreaterFloat value) {Opt = value;}
        public static implicit operator _InOptMut_Heap_Float_MRGraphVertId_StdGreaterFloat(Heap_Float_MRGraphVertId_StdGreaterFloat value) {return new(value);}
    }

    /// This is used for optional parameters of class `Heap_Float_MRGraphVertId_StdGreaterFloat` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Heap_Float_MRGraphVertId_StdGreaterFloat`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Heap_Float_MRGraphVertId_StdGreaterFloat`/`Const_Heap_Float_MRGraphVertId_StdGreaterFloat` to pass it to the function.
    public class _InOptConst_Heap_Float_MRGraphVertId_StdGreaterFloat
    {
        public Const_Heap_Float_MRGraphVertId_StdGreaterFloat? Opt;

        public _InOptConst_Heap_Float_MRGraphVertId_StdGreaterFloat() {}
        public _InOptConst_Heap_Float_MRGraphVertId_StdGreaterFloat(Const_Heap_Float_MRGraphVertId_StdGreaterFloat value) {Opt = value;}
        public static implicit operator _InOptConst_Heap_Float_MRGraphVertId_StdGreaterFloat(Const_Heap_Float_MRGraphVertId_StdGreaterFloat value) {return new(value);}
    }
}
