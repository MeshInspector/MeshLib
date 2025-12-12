public static partial class MR
{
    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1i`.
    /// This is the const reference to the struct.
    public class Const_Box1i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box1i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box1i UnderlyingStruct => ref *(Box1i *)_UnderlyingPtr;

        internal unsafe Const_Box1i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Destroy", ExactSpelling = true)]
            extern static void __MR_Box1i_Destroy(_Underlying *_this);
            __MR_Box1i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box1i() {Dispose(false);}

        public ref readonly int Min => ref UnderlyingStruct.Min;

        public ref readonly int Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box1i_Get_elements();
                return *__MR_Box1i_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box1i(Const_Box1i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box1i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1i _ctor_result = __MR_Box1i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1i::Box1i`.
        public unsafe Const_Box1i(int min, int max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Construct_2", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_Construct_2(int *min, int *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1i _ctor_result = __MR_Box1i_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1i::Box1i`.
        public unsafe Const_Box1i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Construct_1", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1i _ctor_result = __MR_Box1i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1i::operator[]`.
        public unsafe int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_index_const", ExactSpelling = true)]
            extern static int *__MR_Box1i_index_const(_Underlying *_this, int e);
            return *__MR_Box1i_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Box1i::fromMinAndSize`.
        public static unsafe MR.Box1i FromMinAndSize(int min, int size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_fromMinAndSize(int *min, int *size);
            return __MR_Box1i_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1i::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_valid", ExactSpelling = true)]
            extern static byte __MR_Box1i_valid(_Underlying *_this);
            return __MR_Box1i_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box1i::center`.
        public unsafe int Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_center", ExactSpelling = true)]
            extern static int __MR_Box1i_center(_Underlying *_this);
            return __MR_Box1i_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1i::corner`.
        public unsafe int Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_corner", ExactSpelling = true)]
            extern static int __MR_Box1i_corner(_Underlying *_this, bool *c);
            return __MR_Box1i_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1i::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(int n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i_getMinBoxCorner(int *n);
            return __MR_Box1i_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1i::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(int n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i_getMaxBoxCorner(int *n);
            return __MR_Box1i_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1i::size`.
        public unsafe int Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_size", ExactSpelling = true)]
            extern static int __MR_Box1i_size(_Underlying *_this);
            return __MR_Box1i_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1i::diagonal`.
        public unsafe int Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_diagonal", ExactSpelling = true)]
            extern static int __MR_Box1i_diagonal(_Underlying *_this);
            return __MR_Box1i_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1i::volume`.
        public unsafe int Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_volume", ExactSpelling = true)]
            extern static int __MR_Box1i_volume(_Underlying *_this);
            return __MR_Box1i_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1i::contains`.
        public unsafe bool Contains(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_contains_int", ExactSpelling = true)]
            extern static byte __MR_Box1i_contains_int(_Underlying *_this, int *pt);
            return __MR_Box1i_contains_int(_UnderlyingPtr, &pt) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1i::contains`.
        public unsafe bool Contains(MR.Const_Box1i otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_contains_MR_Box1i", ExactSpelling = true)]
            extern static byte __MR_Box1i_contains_MR_Box1i(_Underlying *_this, MR.Const_Box1i._Underlying *otherbox);
            return __MR_Box1i_contains_MR_Box1i(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1i::getBoxClosestPointTo`.
        public unsafe int GetBoxClosestPointTo(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getBoxClosestPointTo", ExactSpelling = true)]
            extern static int __MR_Box1i_getBoxClosestPointTo(_Underlying *_this, int *pt);
            return __MR_Box1i_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1i::intersects`.
        public unsafe bool Intersects(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1i_intersects(_Underlying *_this, MR.Const_Box1i._Underlying *b);
            return __MR_Box1i_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1i::intersection`.
        public unsafe MR.Box1i Intersection(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_intersection", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_intersection(_Underlying *_this, MR.Const_Box1i._Underlying *b);
            return __MR_Box1i_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getDistanceSq_MR_Box1i", ExactSpelling = true)]
            extern static int __MR_Box1i_getDistanceSq_MR_Box1i(_Underlying *_this, MR.Const_Box1i._Underlying *b);
            return __MR_Box1i_getDistanceSq_MR_Box1i(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1i::getDistanceSq`.
        public unsafe int GetDistanceSq(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getDistanceSq_int", ExactSpelling = true)]
            extern static int __MR_Box1i_getDistanceSq_int(_Underlying *_this, int *pt);
            return __MR_Box1i_getDistanceSq_int(_UnderlyingPtr, &pt);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1i::getProjection`.
        public unsafe int GetProjection(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getProjection", ExactSpelling = true)]
            extern static int __MR_Box1i_getProjection(_Underlying *_this, int *pt);
            return __MR_Box1i_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1i::expanded`.
        public unsafe MR.Box1i Expanded(int expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_expanded", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_expanded(_Underlying *_this, int *expansion);
            return __MR_Box1i_expanded(_UnderlyingPtr, &expansion);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1i::insignificantlyExpanded`.
        public unsafe MR.Box1i InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box1i_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box1i::operator==`.
        public static unsafe bool operator==(MR.Const_Box1i _this, MR.Const_Box1i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1i(MR.Const_Box1i._Underlying *_this, MR.Const_Box1i._Underlying *a);
            return __MR_equal_MR_Box1i(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box1i _this, MR.Const_Box1i a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box1i? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box1i)
                return this == (MR.Const_Box1i)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1i`.
    /// This is the non-const reference to the struct.
    public class Mut_Box1i : Const_Box1i
    {
        /// Get the underlying struct.
        public unsafe new ref Box1i UnderlyingStruct => ref *(Box1i *)_UnderlyingPtr;

        internal unsafe Mut_Box1i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref int Min => ref UnderlyingStruct.Min;

        public new ref int Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box1i(Const_Box1i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box1i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1i _ctor_result = __MR_Box1i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1i::Box1i`.
        public unsafe Mut_Box1i(int min, int max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Construct_2", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_Construct_2(int *min, int *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1i _ctor_result = __MR_Box1i_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1i::Box1i`.
        public unsafe Mut_Box1i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Construct_1", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1i _ctor_result = __MR_Box1i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::Box1i::operator[]`.
        public unsafe new ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_index", ExactSpelling = true)]
            extern static int *__MR_Box1i_index(_Underlying *_this, int e);
            return ref *__MR_Box1i_index(_UnderlyingPtr, e);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1i::include`.
        public unsafe void Include(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_include_int", ExactSpelling = true)]
            extern static void __MR_Box1i_include_int(_Underlying *_this, int *pt);
            __MR_Box1i_include_int(_UnderlyingPtr, &pt);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1i::include`.
        public unsafe void Include(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_include_MR_Box1i", ExactSpelling = true)]
            extern static void __MR_Box1i_include_MR_Box1i(_Underlying *_this, MR.Const_Box1i._Underlying *b);
            __MR_Box1i_include_MR_Box1i(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box1i::intersect`.
        public unsafe MR.Mut_Box1i Intersect(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1i._Underlying *__MR_Box1i_intersect(_Underlying *_this, MR.Const_Box1i._Underlying *b);
            return new(__MR_Box1i_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
    public struct Box1i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box1i(Const_Box1i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box1i(Box1i other) => new(new Mut_Box1i((Mut_Box1i._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public int Min;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public int Max;

        /// Generated copy constructor.
        public Box1i(Box1i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box1i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_DefaultConstruct();
            this = __MR_Box1i_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box1i::Box1i`.
        public unsafe Box1i(int min, int max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Construct_2", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_Construct_2(int *min, int *max);
            this = __MR_Box1i_Construct_2(&min, &max);
        }

        /// Generated from constructor `MR::Box1i::Box1i`.
        public unsafe Box1i(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_Construct_1", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box1i_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1i::operator[]`.
        public unsafe int Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_index_const", ExactSpelling = true)]
            extern static int *__MR_Box1i_index_const(MR.Box1i *_this, int e);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return *__MR_Box1i_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1i::operator[]`.
        public unsafe ref int Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_index", ExactSpelling = true)]
            extern static int *__MR_Box1i_index(MR.Box1i *_this, int e);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return ref *__MR_Box1i_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1i::fromMinAndSize`.
        public static unsafe MR.Box1i FromMinAndSize(int min, int size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_fromMinAndSize(int *min, int *size);
            return __MR_Box1i_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1i::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_valid", ExactSpelling = true)]
            extern static byte __MR_Box1i_valid(MR.Box1i *_this);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box1i::center`.
        public unsafe int Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_center", ExactSpelling = true)]
            extern static int __MR_Box1i_center(MR.Box1i *_this);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1i::corner`.
        public unsafe int Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_corner", ExactSpelling = true)]
            extern static int __MR_Box1i_corner(MR.Box1i *_this, bool *c);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_corner(__ptr__this, &c);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1i::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(int n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i_getMinBoxCorner(int *n);
            return __MR_Box1i_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1i::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(int n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i_getMaxBoxCorner(int *n);
            return __MR_Box1i_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1i::size`.
        public unsafe int Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_size", ExactSpelling = true)]
            extern static int __MR_Box1i_size(MR.Box1i *_this);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1i::diagonal`.
        public unsafe int Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_diagonal", ExactSpelling = true)]
            extern static int __MR_Box1i_diagonal(MR.Box1i *_this);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1i::volume`.
        public unsafe int Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_volume", ExactSpelling = true)]
            extern static int __MR_Box1i_volume(MR.Box1i *_this);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1i::include`.
        public unsafe void Include(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_include_int", ExactSpelling = true)]
            extern static void __MR_Box1i_include_int(MR.Box1i *_this, int *pt);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                __MR_Box1i_include_int(__ptr__this, &pt);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1i::include`.
        public unsafe void Include(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_include_MR_Box1i", ExactSpelling = true)]
            extern static void __MR_Box1i_include_MR_Box1i(MR.Box1i *_this, MR.Const_Box1i._Underlying *b);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                __MR_Box1i_include_MR_Box1i(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1i::contains`.
        public unsafe bool Contains(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_contains_int", ExactSpelling = true)]
            extern static byte __MR_Box1i_contains_int(MR.Box1i *_this, int *pt);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_contains_int(__ptr__this, &pt) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1i::contains`.
        public unsafe bool Contains(MR.Const_Box1i otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_contains_MR_Box1i", ExactSpelling = true)]
            extern static byte __MR_Box1i_contains_MR_Box1i(MR.Box1i *_this, MR.Const_Box1i._Underlying *otherbox);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_contains_MR_Box1i(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1i::getBoxClosestPointTo`.
        public unsafe int GetBoxClosestPointTo(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getBoxClosestPointTo", ExactSpelling = true)]
            extern static int __MR_Box1i_getBoxClosestPointTo(MR.Box1i *_this, int *pt);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_getBoxClosestPointTo(__ptr__this, &pt);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1i::intersects`.
        public unsafe bool Intersects(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1i_intersects(MR.Box1i *_this, MR.Const_Box1i._Underlying *b);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1i::intersection`.
        public unsafe MR.Box1i Intersection(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_intersection", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_intersection(MR.Box1i *_this, MR.Const_Box1i._Underlying *b);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box1i::intersect`.
        public unsafe MR.Mut_Box1i Intersect(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1i._Underlying *__MR_Box1i_intersect(MR.Box1i *_this, MR.Const_Box1i._Underlying *b);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return new(__MR_Box1i_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Box1i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getDistanceSq_MR_Box1i", ExactSpelling = true)]
            extern static int __MR_Box1i_getDistanceSq_MR_Box1i(MR.Box1i *_this, MR.Const_Box1i._Underlying *b);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_getDistanceSq_MR_Box1i(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1i::getDistanceSq`.
        public unsafe int GetDistanceSq(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getDistanceSq_int", ExactSpelling = true)]
            extern static int __MR_Box1i_getDistanceSq_int(MR.Box1i *_this, int *pt);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_getDistanceSq_int(__ptr__this, &pt);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1i::getProjection`.
        public unsafe int GetProjection(int pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_getProjection", ExactSpelling = true)]
            extern static int __MR_Box1i_getProjection(MR.Box1i *_this, int *pt);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_getProjection(__ptr__this, &pt);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1i::expanded`.
        public unsafe MR.Box1i Expanded(int expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_expanded", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_expanded(MR.Box1i *_this, int *expansion);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_expanded(__ptr__this, &expansion);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1i::insignificantlyExpanded`.
        public unsafe MR.Box1i InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1i __MR_Box1i_insignificantlyExpanded(MR.Box1i *_this);
            fixed (MR.Box1i *__ptr__this = &this)
            {
                return __MR_Box1i_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box1i::operator==`.
        public static unsafe bool operator==(MR.Box1i _this, MR.Box1i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1i(MR.Const_Box1i._Underlying *_this, MR.Const_Box1i._Underlying *a);
            return __MR_equal_MR_Box1i((MR.Mut_Box1i._Underlying *)&_this, (MR.Mut_Box1i._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box1i _this, MR.Box1i a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box1i a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box1i)
                return this == (MR.Box1i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box1i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box1i`/`Const_Box1i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box1i
    {
        public readonly bool HasValue;
        internal readonly Box1i Object;
        public Box1i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box1i() {HasValue = false;}
        public _InOpt_Box1i(Box1i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box1i(Box1i new_value) {return new(new_value);}
        public _InOpt_Box1i(Const_Box1i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box1i(Const_Box1i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box1i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box1i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1i`/`Const_Box1i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box1i`.
    public class _InOptMut_Box1i
    {
        public Mut_Box1i? Opt;

        public _InOptMut_Box1i() {}
        public _InOptMut_Box1i(Mut_Box1i value) {Opt = value;}
        public static implicit operator _InOptMut_Box1i(Mut_Box1i value) {return new(value);}
        public unsafe _InOptMut_Box1i(ref Box1i value)
        {
            fixed (Box1i *value_ptr = &value)
            {
                Opt = new((Const_Box1i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box1i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box1i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1i`/`Const_Box1i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box1i`.
    public class _InOptConst_Box1i
    {
        public Const_Box1i? Opt;

        public _InOptConst_Box1i() {}
        public _InOptConst_Box1i(Const_Box1i value) {Opt = value;}
        public static implicit operator _InOptConst_Box1i(Const_Box1i value) {return new(value);}
        public unsafe _InOptConst_Box1i(ref readonly Box1i value)
        {
            fixed (Box1i *value_ptr = &value)
            {
                Opt = new((Const_Box1i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1i64`.
    /// This is the const reference to the struct.
    public class Const_Box1i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box1i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box1i64 UnderlyingStruct => ref *(Box1i64 *)_UnderlyingPtr;

        internal unsafe Const_Box1i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Box1i64_Destroy(_Underlying *_this);
            __MR_Box1i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box1i64() {Dispose(false);}

        public ref readonly long Min => ref UnderlyingStruct.Min;

        public ref readonly long Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box1i64_Get_elements();
                return *__MR_Box1i64_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box1i64(Const_Box1i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box1i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1i64 _ctor_result = __MR_Box1i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1i64::Box1i64`.
        public unsafe Const_Box1i64(long min, long max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_Construct_2(long *min, long *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1i64 _ctor_result = __MR_Box1i64_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1i64::Box1i64`.
        public unsafe Const_Box1i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1i64 _ctor_result = __MR_Box1i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1i64::operator[]`.
        public unsafe long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Box1i64_index_const(_Underlying *_this, int e);
            return *__MR_Box1i64_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Box1i64::fromMinAndSize`.
        public static unsafe MR.Box1i64 FromMinAndSize(long min, long size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_fromMinAndSize(long *min, long *size);
            return __MR_Box1i64_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1i64::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_valid", ExactSpelling = true)]
            extern static byte __MR_Box1i64_valid(_Underlying *_this);
            return __MR_Box1i64_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box1i64::center`.
        public unsafe long Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_center", ExactSpelling = true)]
            extern static long __MR_Box1i64_center(_Underlying *_this);
            return __MR_Box1i64_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1i64::corner`.
        public unsafe long Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_corner", ExactSpelling = true)]
            extern static long __MR_Box1i64_corner(_Underlying *_this, bool *c);
            return __MR_Box1i64_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1i64::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(long n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i64_getMinBoxCorner(long *n);
            return __MR_Box1i64_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1i64::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(long n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i64_getMaxBoxCorner(long *n);
            return __MR_Box1i64_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1i64::size`.
        public unsafe long Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_size", ExactSpelling = true)]
            extern static long __MR_Box1i64_size(_Underlying *_this);
            return __MR_Box1i64_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1i64::diagonal`.
        public unsafe long Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_diagonal", ExactSpelling = true)]
            extern static long __MR_Box1i64_diagonal(_Underlying *_this);
            return __MR_Box1i64_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1i64::volume`.
        public unsafe long Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_volume", ExactSpelling = true)]
            extern static long __MR_Box1i64_volume(_Underlying *_this);
            return __MR_Box1i64_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1i64::contains`.
        public unsafe bool Contains(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_contains_int64_t", ExactSpelling = true)]
            extern static byte __MR_Box1i64_contains_int64_t(_Underlying *_this, long *pt);
            return __MR_Box1i64_contains_int64_t(_UnderlyingPtr, &pt) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1i64::contains`.
        public unsafe bool Contains(MR.Const_Box1i64 otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_contains_MR_Box1i64", ExactSpelling = true)]
            extern static byte __MR_Box1i64_contains_MR_Box1i64(_Underlying *_this, MR.Const_Box1i64._Underlying *otherbox);
            return __MR_Box1i64_contains_MR_Box1i64(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1i64::getBoxClosestPointTo`.
        public unsafe long GetBoxClosestPointTo(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getBoxClosestPointTo", ExactSpelling = true)]
            extern static long __MR_Box1i64_getBoxClosestPointTo(_Underlying *_this, long *pt);
            return __MR_Box1i64_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1i64::intersects`.
        public unsafe bool Intersects(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1i64_intersects(_Underlying *_this, MR.Const_Box1i64._Underlying *b);
            return __MR_Box1i64_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1i64::intersection`.
        public unsafe MR.Box1i64 Intersection(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_intersection", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_intersection(_Underlying *_this, MR.Const_Box1i64._Underlying *b);
            return __MR_Box1i64_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getDistanceSq_MR_Box1i64", ExactSpelling = true)]
            extern static long __MR_Box1i64_getDistanceSq_MR_Box1i64(_Underlying *_this, MR.Const_Box1i64._Underlying *b);
            return __MR_Box1i64_getDistanceSq_MR_Box1i64(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1i64::getDistanceSq`.
        public unsafe long GetDistanceSq(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getDistanceSq_int64_t", ExactSpelling = true)]
            extern static long __MR_Box1i64_getDistanceSq_int64_t(_Underlying *_this, long *pt);
            return __MR_Box1i64_getDistanceSq_int64_t(_UnderlyingPtr, &pt);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1i64::getProjection`.
        public unsafe long GetProjection(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getProjection", ExactSpelling = true)]
            extern static long __MR_Box1i64_getProjection(_Underlying *_this, long *pt);
            return __MR_Box1i64_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1i64::expanded`.
        public unsafe MR.Box1i64 Expanded(long expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_expanded", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_expanded(_Underlying *_this, long *expansion);
            return __MR_Box1i64_expanded(_UnderlyingPtr, &expansion);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1i64::insignificantlyExpanded`.
        public unsafe MR.Box1i64 InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box1i64_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box1i64::operator==`.
        public static unsafe bool operator==(MR.Const_Box1i64 _this, MR.Const_Box1i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1i64(MR.Const_Box1i64._Underlying *_this, MR.Const_Box1i64._Underlying *a);
            return __MR_equal_MR_Box1i64(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box1i64 _this, MR.Const_Box1i64 a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box1i64? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box1i64)
                return this == (MR.Const_Box1i64)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Box1i64 : Const_Box1i64
    {
        /// Get the underlying struct.
        public unsafe new ref Box1i64 UnderlyingStruct => ref *(Box1i64 *)_UnderlyingPtr;

        internal unsafe Mut_Box1i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref long Min => ref UnderlyingStruct.Min;

        public new ref long Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box1i64(Const_Box1i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box1i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1i64 _ctor_result = __MR_Box1i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1i64::Box1i64`.
        public unsafe Mut_Box1i64(long min, long max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_Construct_2(long *min, long *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1i64 _ctor_result = __MR_Box1i64_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1i64::Box1i64`.
        public unsafe Mut_Box1i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1i64 _ctor_result = __MR_Box1i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Box1i64::operator[]`.
        public unsafe new ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_index", ExactSpelling = true)]
            extern static long *__MR_Box1i64_index(_Underlying *_this, int e);
            return ref *__MR_Box1i64_index(_UnderlyingPtr, e);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1i64::include`.
        public unsafe void Include(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_include_int64_t", ExactSpelling = true)]
            extern static void __MR_Box1i64_include_int64_t(_Underlying *_this, long *pt);
            __MR_Box1i64_include_int64_t(_UnderlyingPtr, &pt);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1i64::include`.
        public unsafe void Include(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_include_MR_Box1i64", ExactSpelling = true)]
            extern static void __MR_Box1i64_include_MR_Box1i64(_Underlying *_this, MR.Const_Box1i64._Underlying *b);
            __MR_Box1i64_include_MR_Box1i64(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box1i64::intersect`.
        public unsafe MR.Mut_Box1i64 Intersect(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1i64._Underlying *__MR_Box1i64_intersect(_Underlying *_this, MR.Const_Box1i64._Underlying *b);
            return new(__MR_Box1i64_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Box1i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box1i64(Const_Box1i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box1i64(Box1i64 other) => new(new Mut_Box1i64((Mut_Box1i64._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public long Min;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public long Max;

        /// Generated copy constructor.
        public Box1i64(Box1i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box1i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_DefaultConstruct();
            this = __MR_Box1i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box1i64::Box1i64`.
        public unsafe Box1i64(long min, long max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_Construct_2(long *min, long *max);
            this = __MR_Box1i64_Construct_2(&min, &max);
        }

        /// Generated from constructor `MR::Box1i64::Box1i64`.
        public unsafe Box1i64(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box1i64_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1i64::operator[]`.
        public unsafe long Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_index_const", ExactSpelling = true)]
            extern static long *__MR_Box1i64_index_const(MR.Box1i64 *_this, int e);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return *__MR_Box1i64_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1i64::operator[]`.
        public unsafe ref long Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_index", ExactSpelling = true)]
            extern static long *__MR_Box1i64_index(MR.Box1i64 *_this, int e);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return ref *__MR_Box1i64_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1i64::fromMinAndSize`.
        public static unsafe MR.Box1i64 FromMinAndSize(long min, long size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_fromMinAndSize(long *min, long *size);
            return __MR_Box1i64_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1i64::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_valid", ExactSpelling = true)]
            extern static byte __MR_Box1i64_valid(MR.Box1i64 *_this);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box1i64::center`.
        public unsafe long Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_center", ExactSpelling = true)]
            extern static long __MR_Box1i64_center(MR.Box1i64 *_this);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1i64::corner`.
        public unsafe long Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_corner", ExactSpelling = true)]
            extern static long __MR_Box1i64_corner(MR.Box1i64 *_this, bool *c);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_corner(__ptr__this, &c);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1i64::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(long n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i64_getMinBoxCorner(long *n);
            return __MR_Box1i64_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1i64::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(long n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1i64_getMaxBoxCorner(long *n);
            return __MR_Box1i64_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1i64::size`.
        public unsafe long Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_size", ExactSpelling = true)]
            extern static long __MR_Box1i64_size(MR.Box1i64 *_this);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1i64::diagonal`.
        public unsafe long Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_diagonal", ExactSpelling = true)]
            extern static long __MR_Box1i64_diagonal(MR.Box1i64 *_this);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1i64::volume`.
        public unsafe long Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_volume", ExactSpelling = true)]
            extern static long __MR_Box1i64_volume(MR.Box1i64 *_this);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1i64::include`.
        public unsafe void Include(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_include_int64_t", ExactSpelling = true)]
            extern static void __MR_Box1i64_include_int64_t(MR.Box1i64 *_this, long *pt);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                __MR_Box1i64_include_int64_t(__ptr__this, &pt);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1i64::include`.
        public unsafe void Include(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_include_MR_Box1i64", ExactSpelling = true)]
            extern static void __MR_Box1i64_include_MR_Box1i64(MR.Box1i64 *_this, MR.Const_Box1i64._Underlying *b);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                __MR_Box1i64_include_MR_Box1i64(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1i64::contains`.
        public unsafe bool Contains(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_contains_int64_t", ExactSpelling = true)]
            extern static byte __MR_Box1i64_contains_int64_t(MR.Box1i64 *_this, long *pt);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_contains_int64_t(__ptr__this, &pt) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1i64::contains`.
        public unsafe bool Contains(MR.Const_Box1i64 otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_contains_MR_Box1i64", ExactSpelling = true)]
            extern static byte __MR_Box1i64_contains_MR_Box1i64(MR.Box1i64 *_this, MR.Const_Box1i64._Underlying *otherbox);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_contains_MR_Box1i64(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1i64::getBoxClosestPointTo`.
        public unsafe long GetBoxClosestPointTo(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getBoxClosestPointTo", ExactSpelling = true)]
            extern static long __MR_Box1i64_getBoxClosestPointTo(MR.Box1i64 *_this, long *pt);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_getBoxClosestPointTo(__ptr__this, &pt);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1i64::intersects`.
        public unsafe bool Intersects(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1i64_intersects(MR.Box1i64 *_this, MR.Const_Box1i64._Underlying *b);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1i64::intersection`.
        public unsafe MR.Box1i64 Intersection(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_intersection", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_intersection(MR.Box1i64 *_this, MR.Const_Box1i64._Underlying *b);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box1i64::intersect`.
        public unsafe MR.Mut_Box1i64 Intersect(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1i64._Underlying *__MR_Box1i64_intersect(MR.Box1i64 *_this, MR.Const_Box1i64._Underlying *b);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return new(__MR_Box1i64_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Box1i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getDistanceSq_MR_Box1i64", ExactSpelling = true)]
            extern static long __MR_Box1i64_getDistanceSq_MR_Box1i64(MR.Box1i64 *_this, MR.Const_Box1i64._Underlying *b);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_getDistanceSq_MR_Box1i64(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1i64::getDistanceSq`.
        public unsafe long GetDistanceSq(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getDistanceSq_int64_t", ExactSpelling = true)]
            extern static long __MR_Box1i64_getDistanceSq_int64_t(MR.Box1i64 *_this, long *pt);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_getDistanceSq_int64_t(__ptr__this, &pt);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1i64::getProjection`.
        public unsafe long GetProjection(long pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_getProjection", ExactSpelling = true)]
            extern static long __MR_Box1i64_getProjection(MR.Box1i64 *_this, long *pt);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_getProjection(__ptr__this, &pt);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1i64::expanded`.
        public unsafe MR.Box1i64 Expanded(long expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_expanded", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_expanded(MR.Box1i64 *_this, long *expansion);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_expanded(__ptr__this, &expansion);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1i64::insignificantlyExpanded`.
        public unsafe MR.Box1i64 InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1i64_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1i64 __MR_Box1i64_insignificantlyExpanded(MR.Box1i64 *_this);
            fixed (MR.Box1i64 *__ptr__this = &this)
            {
                return __MR_Box1i64_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box1i64::operator==`.
        public static unsafe bool operator==(MR.Box1i64 _this, MR.Box1i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1i64(MR.Const_Box1i64._Underlying *_this, MR.Const_Box1i64._Underlying *a);
            return __MR_equal_MR_Box1i64((MR.Mut_Box1i64._Underlying *)&_this, (MR.Mut_Box1i64._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box1i64 _this, MR.Box1i64 a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box1i64 a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box1i64)
                return this == (MR.Box1i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box1i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box1i64`/`Const_Box1i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box1i64
    {
        public readonly bool HasValue;
        internal readonly Box1i64 Object;
        public Box1i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box1i64() {HasValue = false;}
        public _InOpt_Box1i64(Box1i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box1i64(Box1i64 new_value) {return new(new_value);}
        public _InOpt_Box1i64(Const_Box1i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box1i64(Const_Box1i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box1i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box1i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1i64`/`Const_Box1i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box1i64`.
    public class _InOptMut_Box1i64
    {
        public Mut_Box1i64? Opt;

        public _InOptMut_Box1i64() {}
        public _InOptMut_Box1i64(Mut_Box1i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Box1i64(Mut_Box1i64 value) {return new(value);}
        public unsafe _InOptMut_Box1i64(ref Box1i64 value)
        {
            fixed (Box1i64 *value_ptr = &value)
            {
                Opt = new((Const_Box1i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box1i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box1i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1i64`/`Const_Box1i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box1i64`.
    public class _InOptConst_Box1i64
    {
        public Const_Box1i64? Opt;

        public _InOptConst_Box1i64() {}
        public _InOptConst_Box1i64(Const_Box1i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Box1i64(Const_Box1i64 value) {return new(value);}
        public unsafe _InOptConst_Box1i64(ref readonly Box1i64 value)
        {
            fixed (Box1i64 *value_ptr = &value)
            {
                Opt = new((Const_Box1i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1f`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMax`
    ///     `MR::VdbVolume`
    /// This is the const reference to the struct.
    public class Const_Box1f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box1f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box1f UnderlyingStruct => ref *(Box1f *)_UnderlyingPtr;

        internal unsafe Const_Box1f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Destroy", ExactSpelling = true)]
            extern static void __MR_Box1f_Destroy(_Underlying *_this);
            __MR_Box1f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box1f() {Dispose(false);}

        public ref readonly float Min => ref UnderlyingStruct.Min;

        public ref readonly float Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box1f_Get_elements();
                return *__MR_Box1f_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box1f(Const_Box1f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box1f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1f _ctor_result = __MR_Box1f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1f::Box1f`.
        public unsafe Const_Box1f(float min, float max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Construct_2", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_Construct_2(float *min, float *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1f _ctor_result = __MR_Box1f_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1f::Box1f`.
        public unsafe Const_Box1f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Construct_1", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1f _ctor_result = __MR_Box1f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1f::operator[]`.
        public unsafe float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_index_const", ExactSpelling = true)]
            extern static float *__MR_Box1f_index_const(_Underlying *_this, int e);
            return *__MR_Box1f_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Box1f::fromMinAndSize`.
        public static unsafe MR.Box1f FromMinAndSize(float min, float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_fromMinAndSize(float *min, float *size);
            return __MR_Box1f_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1f::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_valid", ExactSpelling = true)]
            extern static byte __MR_Box1f_valid(_Underlying *_this);
            return __MR_Box1f_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box1f::center`.
        public unsafe float Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_center", ExactSpelling = true)]
            extern static float __MR_Box1f_center(_Underlying *_this);
            return __MR_Box1f_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1f::corner`.
        public unsafe float Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_corner", ExactSpelling = true)]
            extern static float __MR_Box1f_corner(_Underlying *_this, bool *c);
            return __MR_Box1f_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1f::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1f_getMinBoxCorner(float *n);
            return __MR_Box1f_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1f::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1f_getMaxBoxCorner(float *n);
            return __MR_Box1f_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1f::size`.
        public unsafe float Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_size", ExactSpelling = true)]
            extern static float __MR_Box1f_size(_Underlying *_this);
            return __MR_Box1f_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1f::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_diagonal", ExactSpelling = true)]
            extern static float __MR_Box1f_diagonal(_Underlying *_this);
            return __MR_Box1f_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1f::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_volume", ExactSpelling = true)]
            extern static float __MR_Box1f_volume(_Underlying *_this);
            return __MR_Box1f_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1f::contains`.
        public unsafe bool Contains(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_contains_float", ExactSpelling = true)]
            extern static byte __MR_Box1f_contains_float(_Underlying *_this, float *pt);
            return __MR_Box1f_contains_float(_UnderlyingPtr, &pt) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1f::contains`.
        public unsafe bool Contains(MR.Const_Box1f otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_contains_MR_Box1f", ExactSpelling = true)]
            extern static byte __MR_Box1f_contains_MR_Box1f(_Underlying *_this, MR.Const_Box1f._Underlying *otherbox);
            return __MR_Box1f_contains_MR_Box1f(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1f::getBoxClosestPointTo`.
        public unsafe float GetBoxClosestPointTo(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getBoxClosestPointTo", ExactSpelling = true)]
            extern static float __MR_Box1f_getBoxClosestPointTo(_Underlying *_this, float *pt);
            return __MR_Box1f_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1f::intersects`.
        public unsafe bool Intersects(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1f_intersects(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_Box1f_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1f::intersection`.
        public unsafe MR.Box1f Intersection(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_intersection", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_intersection(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_Box1f_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getDistanceSq_MR_Box1f", ExactSpelling = true)]
            extern static float __MR_Box1f_getDistanceSq_MR_Box1f(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return __MR_Box1f_getDistanceSq_MR_Box1f(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1f::getDistanceSq`.
        public unsafe float GetDistanceSq(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getDistanceSq_float", ExactSpelling = true)]
            extern static float __MR_Box1f_getDistanceSq_float(_Underlying *_this, float *pt);
            return __MR_Box1f_getDistanceSq_float(_UnderlyingPtr, &pt);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1f::getProjection`.
        public unsafe float GetProjection(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getProjection", ExactSpelling = true)]
            extern static float __MR_Box1f_getProjection(_Underlying *_this, float *pt);
            return __MR_Box1f_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1f::expanded`.
        public unsafe MR.Box1f Expanded(float expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_expanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_expanded(_Underlying *_this, float *expansion);
            return __MR_Box1f_expanded(_UnderlyingPtr, &expansion);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1f::insignificantlyExpanded`.
        public unsafe MR.Box1f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box1f_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box1f::operator==`.
        public static unsafe bool operator==(MR.Const_Box1f _this, MR.Const_Box1f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1f(MR.Const_Box1f._Underlying *_this, MR.Const_Box1f._Underlying *a);
            return __MR_equal_MR_Box1f(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box1f _this, MR.Const_Box1f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box1f? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box1f)
                return this == (MR.Const_Box1f)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1f`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMax`
    ///     `MR::VdbVolume`
    /// This is the non-const reference to the struct.
    public class Mut_Box1f : Const_Box1f
    {
        /// Get the underlying struct.
        public unsafe new ref Box1f UnderlyingStruct => ref *(Box1f *)_UnderlyingPtr;

        internal unsafe Mut_Box1f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref float Min => ref UnderlyingStruct.Min;

        public new ref float Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box1f(Const_Box1f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 8);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box1f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1f _ctor_result = __MR_Box1f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1f::Box1f`.
        public unsafe Mut_Box1f(float min, float max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Construct_2", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_Construct_2(float *min, float *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1f _ctor_result = __MR_Box1f_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from constructor `MR::Box1f::Box1f`.
        public unsafe Mut_Box1f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Construct_1", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(8);
            MR.Box1f _ctor_result = __MR_Box1f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 8);
        }

        /// Generated from method `MR::Box1f::operator[]`.
        public unsafe new ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_index", ExactSpelling = true)]
            extern static float *__MR_Box1f_index(_Underlying *_this, int e);
            return ref *__MR_Box1f_index(_UnderlyingPtr, e);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1f::include`.
        public unsafe void Include(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_include_float", ExactSpelling = true)]
            extern static void __MR_Box1f_include_float(_Underlying *_this, float *pt);
            __MR_Box1f_include_float(_UnderlyingPtr, &pt);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1f::include`.
        public unsafe void Include(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_include_MR_Box1f", ExactSpelling = true)]
            extern static void __MR_Box1f_include_MR_Box1f(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            __MR_Box1f_include_MR_Box1f(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box1f::intersect`.
        public unsafe MR.Mut_Box1f Intersect(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1f._Underlying *__MR_Box1f_intersect(_Underlying *_this, MR.Const_Box1f._Underlying *b);
            return new(__MR_Box1f_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1f`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMax`
    ///     `MR::VdbVolume`
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
    public struct Box1f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box1f(Const_Box1f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box1f(Box1f other) => new(new Mut_Box1f((Mut_Box1f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public float Min;

        [System.Runtime.InteropServices.FieldOffset(4)]
        public float Max;

        /// Generated copy constructor.
        public Box1f(Box1f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box1f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_DefaultConstruct();
            this = __MR_Box1f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box1f::Box1f`.
        public unsafe Box1f(float min, float max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Construct_2", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_Construct_2(float *min, float *max);
            this = __MR_Box1f_Construct_2(&min, &max);
        }

        /// Generated from constructor `MR::Box1f::Box1f`.
        public unsafe Box1f(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_Construct_1", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box1f_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1f::operator[]`.
        public unsafe float Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_index_const", ExactSpelling = true)]
            extern static float *__MR_Box1f_index_const(MR.Box1f *_this, int e);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return *__MR_Box1f_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1f::operator[]`.
        public unsafe ref float Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_index", ExactSpelling = true)]
            extern static float *__MR_Box1f_index(MR.Box1f *_this, int e);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return ref *__MR_Box1f_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1f::fromMinAndSize`.
        public static unsafe MR.Box1f FromMinAndSize(float min, float size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_fromMinAndSize(float *min, float *size);
            return __MR_Box1f_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1f::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_valid", ExactSpelling = true)]
            extern static byte __MR_Box1f_valid(MR.Box1f *_this);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box1f::center`.
        public unsafe float Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_center", ExactSpelling = true)]
            extern static float __MR_Box1f_center(MR.Box1f *_this);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1f::corner`.
        public unsafe float Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_corner", ExactSpelling = true)]
            extern static float __MR_Box1f_corner(MR.Box1f *_this, bool *c);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_corner(__ptr__this, &c);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1f::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1f_getMinBoxCorner(float *n);
            return __MR_Box1f_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1f::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(float n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1f_getMaxBoxCorner(float *n);
            return __MR_Box1f_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1f::size`.
        public unsafe float Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_size", ExactSpelling = true)]
            extern static float __MR_Box1f_size(MR.Box1f *_this);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1f::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_diagonal", ExactSpelling = true)]
            extern static float __MR_Box1f_diagonal(MR.Box1f *_this);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1f::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_volume", ExactSpelling = true)]
            extern static float __MR_Box1f_volume(MR.Box1f *_this);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1f::include`.
        public unsafe void Include(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_include_float", ExactSpelling = true)]
            extern static void __MR_Box1f_include_float(MR.Box1f *_this, float *pt);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                __MR_Box1f_include_float(__ptr__this, &pt);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1f::include`.
        public unsafe void Include(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_include_MR_Box1f", ExactSpelling = true)]
            extern static void __MR_Box1f_include_MR_Box1f(MR.Box1f *_this, MR.Const_Box1f._Underlying *b);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                __MR_Box1f_include_MR_Box1f(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1f::contains`.
        public unsafe bool Contains(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_contains_float", ExactSpelling = true)]
            extern static byte __MR_Box1f_contains_float(MR.Box1f *_this, float *pt);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_contains_float(__ptr__this, &pt) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1f::contains`.
        public unsafe bool Contains(MR.Const_Box1f otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_contains_MR_Box1f", ExactSpelling = true)]
            extern static byte __MR_Box1f_contains_MR_Box1f(MR.Box1f *_this, MR.Const_Box1f._Underlying *otherbox);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_contains_MR_Box1f(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1f::getBoxClosestPointTo`.
        public unsafe float GetBoxClosestPointTo(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getBoxClosestPointTo", ExactSpelling = true)]
            extern static float __MR_Box1f_getBoxClosestPointTo(MR.Box1f *_this, float *pt);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_getBoxClosestPointTo(__ptr__this, &pt);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1f::intersects`.
        public unsafe bool Intersects(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1f_intersects(MR.Box1f *_this, MR.Const_Box1f._Underlying *b);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1f::intersection`.
        public unsafe MR.Box1f Intersection(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_intersection", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_intersection(MR.Box1f *_this, MR.Const_Box1f._Underlying *b);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box1f::intersect`.
        public unsafe MR.Mut_Box1f Intersect(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1f._Underlying *__MR_Box1f_intersect(MR.Box1f *_this, MR.Const_Box1f._Underlying *b);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return new(__MR_Box1f_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Box1f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getDistanceSq_MR_Box1f", ExactSpelling = true)]
            extern static float __MR_Box1f_getDistanceSq_MR_Box1f(MR.Box1f *_this, MR.Const_Box1f._Underlying *b);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_getDistanceSq_MR_Box1f(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1f::getDistanceSq`.
        public unsafe float GetDistanceSq(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getDistanceSq_float", ExactSpelling = true)]
            extern static float __MR_Box1f_getDistanceSq_float(MR.Box1f *_this, float *pt);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_getDistanceSq_float(__ptr__this, &pt);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1f::getProjection`.
        public unsafe float GetProjection(float pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_getProjection", ExactSpelling = true)]
            extern static float __MR_Box1f_getProjection(MR.Box1f *_this, float *pt);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_getProjection(__ptr__this, &pt);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1f::expanded`.
        public unsafe MR.Box1f Expanded(float expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_expanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_expanded(MR.Box1f *_this, float *expansion);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_expanded(__ptr__this, &expansion);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1f::insignificantlyExpanded`.
        public unsafe MR.Box1f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1f_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1f __MR_Box1f_insignificantlyExpanded(MR.Box1f *_this);
            fixed (MR.Box1f *__ptr__this = &this)
            {
                return __MR_Box1f_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box1f::operator==`.
        public static unsafe bool operator==(MR.Box1f _this, MR.Box1f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1f(MR.Const_Box1f._Underlying *_this, MR.Const_Box1f._Underlying *a);
            return __MR_equal_MR_Box1f((MR.Mut_Box1f._Underlying *)&_this, (MR.Mut_Box1f._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box1f _this, MR.Box1f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box1f a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box1f)
                return this == (MR.Box1f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box1f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box1f`/`Const_Box1f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box1f
    {
        public readonly bool HasValue;
        internal readonly Box1f Object;
        public Box1f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box1f() {HasValue = false;}
        public _InOpt_Box1f(Box1f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box1f(Box1f new_value) {return new(new_value);}
        public _InOpt_Box1f(Const_Box1f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box1f(Const_Box1f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box1f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box1f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1f`/`Const_Box1f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box1f`.
    public class _InOptMut_Box1f
    {
        public Mut_Box1f? Opt;

        public _InOptMut_Box1f() {}
        public _InOptMut_Box1f(Mut_Box1f value) {Opt = value;}
        public static implicit operator _InOptMut_Box1f(Mut_Box1f value) {return new(value);}
        public unsafe _InOptMut_Box1f(ref Box1f value)
        {
            fixed (Box1f *value_ptr = &value)
            {
                Opt = new((Const_Box1f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box1f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box1f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1f`/`Const_Box1f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box1f`.
    public class _InOptConst_Box1f
    {
        public Const_Box1f? Opt;

        public _InOptConst_Box1f() {}
        public _InOptConst_Box1f(Const_Box1f value) {Opt = value;}
        public static implicit operator _InOptConst_Box1f(Const_Box1f value) {return new(value);}
        public unsafe _InOptConst_Box1f(ref readonly Box1f value)
        {
            fixed (Box1f *value_ptr = &value)
            {
                Opt = new((Const_Box1f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1d`.
    /// This is the const reference to the struct.
    public class Const_Box1d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box1d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box1d UnderlyingStruct => ref *(Box1d *)_UnderlyingPtr;

        internal unsafe Const_Box1d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Destroy", ExactSpelling = true)]
            extern static void __MR_Box1d_Destroy(_Underlying *_this);
            __MR_Box1d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box1d() {Dispose(false);}

        public ref readonly double Min => ref UnderlyingStruct.Min;

        public ref readonly double Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box1d_Get_elements();
                return *__MR_Box1d_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box1d(Const_Box1d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box1d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1d _ctor_result = __MR_Box1d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1d::Box1d`.
        public unsafe Const_Box1d(double min, double max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Construct_2", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_Construct_2(double *min, double *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1d _ctor_result = __MR_Box1d_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1d::Box1d`.
        public unsafe Const_Box1d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Construct_1", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1d _ctor_result = __MR_Box1d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1d::operator[]`.
        public unsafe double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_index_const", ExactSpelling = true)]
            extern static double *__MR_Box1d_index_const(_Underlying *_this, int e);
            return *__MR_Box1d_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Box1d::fromMinAndSize`.
        public static unsafe MR.Box1d FromMinAndSize(double min, double size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_fromMinAndSize(double *min, double *size);
            return __MR_Box1d_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1d::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_valid", ExactSpelling = true)]
            extern static byte __MR_Box1d_valid(_Underlying *_this);
            return __MR_Box1d_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box1d::center`.
        public unsafe double Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_center", ExactSpelling = true)]
            extern static double __MR_Box1d_center(_Underlying *_this);
            return __MR_Box1d_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1d::corner`.
        public unsafe double Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_corner", ExactSpelling = true)]
            extern static double __MR_Box1d_corner(_Underlying *_this, bool *c);
            return __MR_Box1d_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1d::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(double n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1d_getMinBoxCorner(double *n);
            return __MR_Box1d_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1d::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(double n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1d_getMaxBoxCorner(double *n);
            return __MR_Box1d_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1d::size`.
        public unsafe double Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_size", ExactSpelling = true)]
            extern static double __MR_Box1d_size(_Underlying *_this);
            return __MR_Box1d_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1d::diagonal`.
        public unsafe double Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_diagonal", ExactSpelling = true)]
            extern static double __MR_Box1d_diagonal(_Underlying *_this);
            return __MR_Box1d_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1d::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_volume", ExactSpelling = true)]
            extern static double __MR_Box1d_volume(_Underlying *_this);
            return __MR_Box1d_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1d::contains`.
        public unsafe bool Contains(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_contains_double", ExactSpelling = true)]
            extern static byte __MR_Box1d_contains_double(_Underlying *_this, double *pt);
            return __MR_Box1d_contains_double(_UnderlyingPtr, &pt) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1d::contains`.
        public unsafe bool Contains(MR.Const_Box1d otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_contains_MR_Box1d", ExactSpelling = true)]
            extern static byte __MR_Box1d_contains_MR_Box1d(_Underlying *_this, MR.Const_Box1d._Underlying *otherbox);
            return __MR_Box1d_contains_MR_Box1d(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1d::getBoxClosestPointTo`.
        public unsafe double GetBoxClosestPointTo(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getBoxClosestPointTo", ExactSpelling = true)]
            extern static double __MR_Box1d_getBoxClosestPointTo(_Underlying *_this, double *pt);
            return __MR_Box1d_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1d::intersects`.
        public unsafe bool Intersects(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1d_intersects(_Underlying *_this, MR.Const_Box1d._Underlying *b);
            return __MR_Box1d_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1d::intersection`.
        public unsafe MR.Box1d Intersection(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_intersection", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_intersection(_Underlying *_this, MR.Const_Box1d._Underlying *b);
            return __MR_Box1d_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getDistanceSq_MR_Box1d", ExactSpelling = true)]
            extern static double __MR_Box1d_getDistanceSq_MR_Box1d(_Underlying *_this, MR.Const_Box1d._Underlying *b);
            return __MR_Box1d_getDistanceSq_MR_Box1d(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1d::getDistanceSq`.
        public unsafe double GetDistanceSq(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getDistanceSq_double", ExactSpelling = true)]
            extern static double __MR_Box1d_getDistanceSq_double(_Underlying *_this, double *pt);
            return __MR_Box1d_getDistanceSq_double(_UnderlyingPtr, &pt);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1d::getProjection`.
        public unsafe double GetProjection(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getProjection", ExactSpelling = true)]
            extern static double __MR_Box1d_getProjection(_Underlying *_this, double *pt);
            return __MR_Box1d_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1d::expanded`.
        public unsafe MR.Box1d Expanded(double expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_expanded", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_expanded(_Underlying *_this, double *expansion);
            return __MR_Box1d_expanded(_UnderlyingPtr, &expansion);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1d::insignificantlyExpanded`.
        public unsafe MR.Box1d InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box1d_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box1d::operator==`.
        public static unsafe bool operator==(MR.Const_Box1d _this, MR.Const_Box1d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1d(MR.Const_Box1d._Underlying *_this, MR.Const_Box1d._Underlying *a);
            return __MR_equal_MR_Box1d(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box1d _this, MR.Const_Box1d a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box1d? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box1d)
                return this == (MR.Const_Box1d)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1d`.
    /// This is the non-const reference to the struct.
    public class Mut_Box1d : Const_Box1d
    {
        /// Get the underlying struct.
        public unsafe new ref Box1d UnderlyingStruct => ref *(Box1d *)_UnderlyingPtr;

        internal unsafe Mut_Box1d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref double Min => ref UnderlyingStruct.Min;

        public new ref double Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box1d(Const_Box1d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box1d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1d _ctor_result = __MR_Box1d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1d::Box1d`.
        public unsafe Mut_Box1d(double min, double max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Construct_2", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_Construct_2(double *min, double *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1d _ctor_result = __MR_Box1d_Construct_2(&min, &max);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box1d::Box1d`.
        public unsafe Mut_Box1d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Construct_1", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box1d _ctor_result = __MR_Box1d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Box1d::operator[]`.
        public unsafe new ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_index", ExactSpelling = true)]
            extern static double *__MR_Box1d_index(_Underlying *_this, int e);
            return ref *__MR_Box1d_index(_UnderlyingPtr, e);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1d::include`.
        public unsafe void Include(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_include_double", ExactSpelling = true)]
            extern static void __MR_Box1d_include_double(_Underlying *_this, double *pt);
            __MR_Box1d_include_double(_UnderlyingPtr, &pt);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1d::include`.
        public unsafe void Include(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_include_MR_Box1d", ExactSpelling = true)]
            extern static void __MR_Box1d_include_MR_Box1d(_Underlying *_this, MR.Const_Box1d._Underlying *b);
            __MR_Box1d_include_MR_Box1d(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box1d::intersect`.
        public unsafe MR.Mut_Box1d Intersect(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1d._Underlying *__MR_Box1d_intersect(_Underlying *_this, MR.Const_Box1d._Underlying *b);
            return new(__MR_Box1d_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box1d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Box1d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box1d(Const_Box1d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box1d(Box1d other) => new(new Mut_Box1d((Mut_Box1d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public double Min;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public double Max;

        /// Generated copy constructor.
        public Box1d(Box1d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box1d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_DefaultConstruct();
            this = __MR_Box1d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box1d::Box1d`.
        public unsafe Box1d(double min, double max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Construct_2", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_Construct_2(double *min, double *max);
            this = __MR_Box1d_Construct_2(&min, &max);
        }

        /// Generated from constructor `MR::Box1d::Box1d`.
        public unsafe Box1d(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_Construct_1", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box1d_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box1d::operator[]`.
        public unsafe double Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_index_const", ExactSpelling = true)]
            extern static double *__MR_Box1d_index_const(MR.Box1d *_this, int e);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return *__MR_Box1d_index_const(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1d::operator[]`.
        public unsafe ref double Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_index", ExactSpelling = true)]
            extern static double *__MR_Box1d_index(MR.Box1d *_this, int e);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return ref *__MR_Box1d_index(__ptr__this, e);
            }
        }

        /// Generated from method `MR::Box1d::fromMinAndSize`.
        public static unsafe MR.Box1d FromMinAndSize(double min, double size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_fromMinAndSize(double *min, double *size);
            return __MR_Box1d_fromMinAndSize(&min, &size);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box1d::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_valid", ExactSpelling = true)]
            extern static byte __MR_Box1d_valid(MR.Box1d *_this);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box1d::center`.
        public unsafe double Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_center", ExactSpelling = true)]
            extern static double __MR_Box1d_center(MR.Box1d *_this);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box1d::corner`.
        public unsafe double Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_corner", ExactSpelling = true)]
            extern static double __MR_Box1d_corner(MR.Box1d *_this, bool *c);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_corner(__ptr__this, &c);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box1d::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(double n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1d_getMinBoxCorner(double *n);
            return __MR_Box1d_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box1d::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(double n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box1d_getMaxBoxCorner(double *n);
            return __MR_Box1d_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box1d::size`.
        public unsafe double Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_size", ExactSpelling = true)]
            extern static double __MR_Box1d_size(MR.Box1d *_this);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box1d::diagonal`.
        public unsafe double Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_diagonal", ExactSpelling = true)]
            extern static double __MR_Box1d_diagonal(MR.Box1d *_this);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box1d::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_volume", ExactSpelling = true)]
            extern static double __MR_Box1d_volume(MR.Box1d *_this);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box1d::include`.
        public unsafe void Include(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_include_double", ExactSpelling = true)]
            extern static void __MR_Box1d_include_double(MR.Box1d *_this, double *pt);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                __MR_Box1d_include_double(__ptr__this, &pt);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box1d::include`.
        public unsafe void Include(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_include_MR_Box1d", ExactSpelling = true)]
            extern static void __MR_Box1d_include_MR_Box1d(MR.Box1d *_this, MR.Const_Box1d._Underlying *b);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                __MR_Box1d_include_MR_Box1d(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box1d::contains`.
        public unsafe bool Contains(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_contains_double", ExactSpelling = true)]
            extern static byte __MR_Box1d_contains_double(MR.Box1d *_this, double *pt);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_contains_double(__ptr__this, &pt) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box1d::contains`.
        public unsafe bool Contains(MR.Const_Box1d otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_contains_MR_Box1d", ExactSpelling = true)]
            extern static byte __MR_Box1d_contains_MR_Box1d(MR.Box1d *_this, MR.Const_Box1d._Underlying *otherbox);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_contains_MR_Box1d(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box1d::getBoxClosestPointTo`.
        public unsafe double GetBoxClosestPointTo(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getBoxClosestPointTo", ExactSpelling = true)]
            extern static double __MR_Box1d_getBoxClosestPointTo(MR.Box1d *_this, double *pt);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_getBoxClosestPointTo(__ptr__this, &pt);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box1d::intersects`.
        public unsafe bool Intersects(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_intersects", ExactSpelling = true)]
            extern static byte __MR_Box1d_intersects(MR.Box1d *_this, MR.Const_Box1d._Underlying *b);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box1d::intersection`.
        public unsafe MR.Box1d Intersection(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_intersection", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_intersection(MR.Box1d *_this, MR.Const_Box1d._Underlying *b);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box1d::intersect`.
        public unsafe MR.Mut_Box1d Intersect(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box1d._Underlying *__MR_Box1d_intersect(MR.Box1d *_this, MR.Const_Box1d._Underlying *b);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return new(__MR_Box1d_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box1d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Box1d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getDistanceSq_MR_Box1d", ExactSpelling = true)]
            extern static double __MR_Box1d_getDistanceSq_MR_Box1d(MR.Box1d *_this, MR.Const_Box1d._Underlying *b);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_getDistanceSq_MR_Box1d(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box1d::getDistanceSq`.
        public unsafe double GetDistanceSq(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getDistanceSq_double", ExactSpelling = true)]
            extern static double __MR_Box1d_getDistanceSq_double(MR.Box1d *_this, double *pt);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_getDistanceSq_double(__ptr__this, &pt);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box1d::getProjection`.
        public unsafe double GetProjection(double pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_getProjection", ExactSpelling = true)]
            extern static double __MR_Box1d_getProjection(MR.Box1d *_this, double *pt);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_getProjection(__ptr__this, &pt);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box1d::expanded`.
        public unsafe MR.Box1d Expanded(double expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_expanded", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_expanded(MR.Box1d *_this, double *expansion);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_expanded(__ptr__this, &expansion);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box1d::insignificantlyExpanded`.
        public unsafe MR.Box1d InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box1d_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box1d __MR_Box1d_insignificantlyExpanded(MR.Box1d *_this);
            fixed (MR.Box1d *__ptr__this = &this)
            {
                return __MR_Box1d_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box1d::operator==`.
        public static unsafe bool operator==(MR.Box1d _this, MR.Box1d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box1d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box1d(MR.Const_Box1d._Underlying *_this, MR.Const_Box1d._Underlying *a);
            return __MR_equal_MR_Box1d((MR.Mut_Box1d._Underlying *)&_this, (MR.Mut_Box1d._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box1d _this, MR.Box1d a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box1d a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box1d)
                return this == (MR.Box1d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box1d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box1d`/`Const_Box1d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box1d
    {
        public readonly bool HasValue;
        internal readonly Box1d Object;
        public Box1d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box1d() {HasValue = false;}
        public _InOpt_Box1d(Box1d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box1d(Box1d new_value) {return new(new_value);}
        public _InOpt_Box1d(Const_Box1d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box1d(Const_Box1d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box1d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box1d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1d`/`Const_Box1d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box1d`.
    public class _InOptMut_Box1d
    {
        public Mut_Box1d? Opt;

        public _InOptMut_Box1d() {}
        public _InOptMut_Box1d(Mut_Box1d value) {Opt = value;}
        public static implicit operator _InOptMut_Box1d(Mut_Box1d value) {return new(value);}
        public unsafe _InOptMut_Box1d(ref Box1d value)
        {
            fixed (Box1d *value_ptr = &value)
            {
                Opt = new((Const_Box1d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box1d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box1d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box1d`/`Const_Box1d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box1d`.
    public class _InOptConst_Box1d
    {
        public Const_Box1d? Opt;

        public _InOptConst_Box1d() {}
        public _InOptConst_Box1d(Const_Box1d value) {Opt = value;}
        public static implicit operator _InOptConst_Box1d(Const_Box1d value) {return new(value);}
        public unsafe _InOptConst_Box1d(ref readonly Box1d value)
        {
            fixed (Box1d *value_ptr = &value)
            {
                Opt = new((Const_Box1d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2i`.
    /// This is the const reference to the struct.
    public class Const_Box2i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box2i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box2i UnderlyingStruct => ref *(Box2i *)_UnderlyingPtr;

        internal unsafe Const_Box2i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Destroy", ExactSpelling = true)]
            extern static void __MR_Box2i_Destroy(_Underlying *_this);
            __MR_Box2i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box2i() {Dispose(false);}

        public ref readonly MR.Vector2i Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector2i Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box2i_Get_elements();
                return *__MR_Box2i_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box2i(Const_Box2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2i _ctor_result = __MR_Box2i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box2i::Box2i`.
        public unsafe Const_Box2i(MR.Const_Vector2i min, MR.Const_Vector2i max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Construct_2", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_Construct_2(MR.Const_Vector2i._Underlying *min, MR.Const_Vector2i._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2i _ctor_result = __MR_Box2i_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2i::Box2i`.
        public unsafe Const_Box2i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Construct_1", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2i _ctor_result = __MR_Box2i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2i::operator[]`.
        public unsafe MR.Const_Vector2i Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_Box2i_index_const(_Underlying *_this, int e);
            return new(__MR_Box2i_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box2i::fromMinAndSize`.
        public static unsafe MR.Box2i FromMinAndSize(MR.Const_Vector2i min, MR.Const_Vector2i size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_fromMinAndSize(MR.Const_Vector2i._Underlying *min, MR.Const_Vector2i._Underlying *size);
            return __MR_Box2i_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2i::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_valid", ExactSpelling = true)]
            extern static byte __MR_Box2i_valid(_Underlying *_this);
            return __MR_Box2i_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box2i::center`.
        public unsafe MR.Vector2i Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_center", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_center(_Underlying *_this);
            return __MR_Box2i_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2i::corner`.
        public unsafe MR.Vector2i Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_corner", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_corner(_Underlying *_this, MR.Const_Vector2b._Underlying *c);
            return __MR_Box2i_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2i::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i_getMinBoxCorner(MR.Const_Vector2i._Underlying *n);
            return __MR_Box2i_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2i::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i_getMaxBoxCorner(MR.Const_Vector2i._Underlying *n);
            return __MR_Box2i_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2i::size`.
        public unsafe MR.Vector2i Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_size", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_size(_Underlying *_this);
            return __MR_Box2i_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2i::diagonal`.
        public unsafe int Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_diagonal", ExactSpelling = true)]
            extern static int __MR_Box2i_diagonal(_Underlying *_this);
            return __MR_Box2i_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2i::volume`.
        public unsafe int Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_volume", ExactSpelling = true)]
            extern static int __MR_Box2i_volume(_Underlying *_this);
            return __MR_Box2i_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2i::contains`.
        public unsafe bool Contains(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_contains_MR_Vector2i", ExactSpelling = true)]
            extern static byte __MR_Box2i_contains_MR_Vector2i(_Underlying *_this, MR.Const_Vector2i._Underlying *pt);
            return __MR_Box2i_contains_MR_Vector2i(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2i::contains`.
        public unsafe bool Contains(MR.Const_Box2i otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_contains_MR_Box2i", ExactSpelling = true)]
            extern static byte __MR_Box2i_contains_MR_Box2i(_Underlying *_this, MR.Const_Box2i._Underlying *otherbox);
            return __MR_Box2i_contains_MR_Box2i(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2i::getBoxClosestPointTo`.
        public unsafe MR.Vector2i GetBoxClosestPointTo(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector2i._Underlying *pt);
            return __MR_Box2i_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2i::intersects`.
        public unsafe bool Intersects(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2i_intersects(_Underlying *_this, MR.Const_Box2i._Underlying *b);
            return __MR_Box2i_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2i::intersection`.
        public unsafe MR.Box2i Intersection(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_intersection", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_intersection(_Underlying *_this, MR.Const_Box2i._Underlying *b);
            return __MR_Box2i_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getDistanceSq_MR_Box2i", ExactSpelling = true)]
            extern static int __MR_Box2i_getDistanceSq_MR_Box2i(_Underlying *_this, MR.Const_Box2i._Underlying *b);
            return __MR_Box2i_getDistanceSq_MR_Box2i(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getDistanceSq_MR_Vector2i", ExactSpelling = true)]
            extern static int __MR_Box2i_getDistanceSq_MR_Vector2i(_Underlying *_this, MR.Const_Vector2i._Underlying *pt);
            return __MR_Box2i_getDistanceSq_MR_Vector2i(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2i::getProjection`.
        public unsafe MR.Vector2i GetProjection(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getProjection", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_getProjection(_Underlying *_this, MR.Const_Vector2i._Underlying *pt);
            return __MR_Box2i_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2i::expanded`.
        public unsafe MR.Box2i Expanded(MR.Const_Vector2i expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_expanded", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_expanded(_Underlying *_this, MR.Const_Vector2i._Underlying *expansion);
            return __MR_Box2i_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2i::insignificantlyExpanded`.
        public unsafe MR.Box2i InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box2i_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box2i::operator==`.
        public static unsafe bool operator==(MR.Const_Box2i _this, MR.Const_Box2i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2i(MR.Const_Box2i._Underlying *_this, MR.Const_Box2i._Underlying *a);
            return __MR_equal_MR_Box2i(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box2i _this, MR.Const_Box2i a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box2i? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box2i)
                return this == (MR.Const_Box2i)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2i`.
    /// This is the non-const reference to the struct.
    public class Mut_Box2i : Const_Box2i
    {
        /// Get the underlying struct.
        public unsafe new ref Box2i UnderlyingStruct => ref *(Box2i *)_UnderlyingPtr;

        internal unsafe Mut_Box2i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector2i Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector2i Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box2i(Const_Box2i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box2i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2i _ctor_result = __MR_Box2i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box2i::Box2i`.
        public unsafe Mut_Box2i(MR.Const_Vector2i min, MR.Const_Vector2i max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Construct_2", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_Construct_2(MR.Const_Vector2i._Underlying *min, MR.Const_Vector2i._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2i _ctor_result = __MR_Box2i_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2i::Box2i`.
        public unsafe Mut_Box2i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Construct_1", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2i _ctor_result = __MR_Box2i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Box2i::operator[]`.
        public unsafe new MR.Mut_Vector2i Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_Box2i_index(_Underlying *_this, int e);
            return new(__MR_Box2i_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2i::include`.
        public unsafe void Include(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_include_MR_Vector2i", ExactSpelling = true)]
            extern static void __MR_Box2i_include_MR_Vector2i(_Underlying *_this, MR.Const_Vector2i._Underlying *pt);
            __MR_Box2i_include_MR_Vector2i(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2i::include`.
        public unsafe void Include(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_include_MR_Box2i", ExactSpelling = true)]
            extern static void __MR_Box2i_include_MR_Box2i(_Underlying *_this, MR.Const_Box2i._Underlying *b);
            __MR_Box2i_include_MR_Box2i(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box2i::intersect`.
        public unsafe MR.Mut_Box2i Intersect(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2i._Underlying *__MR_Box2i_intersect(_Underlying *_this, MR.Const_Box2i._Underlying *b);
            return new(__MR_Box2i_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Box2i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box2i(Const_Box2i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box2i(Box2i other) => new(new Mut_Box2i((Mut_Box2i._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2i Min;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public MR.Vector2i Max;

        /// Generated copy constructor.
        public Box2i(Box2i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box2i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_DefaultConstruct();
            this = __MR_Box2i_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box2i::Box2i`.
        public unsafe Box2i(MR.Const_Vector2i min, MR.Const_Vector2i max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Construct_2", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_Construct_2(MR.Const_Vector2i._Underlying *min, MR.Const_Vector2i._Underlying *max);
            this = __MR_Box2i_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2i::Box2i`.
        public unsafe Box2i(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_Construct_1", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box2i_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2i::operator[]`.
        public unsafe MR.Const_Vector2i Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i._Underlying *__MR_Box2i_index_const(MR.Box2i *_this, int e);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return new(__MR_Box2i_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2i::operator[]`.
        public unsafe MR.Mut_Vector2i Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i._Underlying *__MR_Box2i_index(MR.Box2i *_this, int e);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return new(__MR_Box2i_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2i::fromMinAndSize`.
        public static unsafe MR.Box2i FromMinAndSize(MR.Const_Vector2i min, MR.Const_Vector2i size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_fromMinAndSize(MR.Const_Vector2i._Underlying *min, MR.Const_Vector2i._Underlying *size);
            return __MR_Box2i_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2i::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_valid", ExactSpelling = true)]
            extern static byte __MR_Box2i_valid(MR.Box2i *_this);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box2i::center`.
        public unsafe MR.Vector2i Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_center", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_center(MR.Box2i *_this);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2i::corner`.
        public unsafe MR.Vector2i Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_corner", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_corner(MR.Box2i *_this, MR.Const_Vector2b._Underlying *c);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2i::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i_getMinBoxCorner(MR.Const_Vector2i._Underlying *n);
            return __MR_Box2i_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2i::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i_getMaxBoxCorner(MR.Const_Vector2i._Underlying *n);
            return __MR_Box2i_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2i::size`.
        public unsafe MR.Vector2i Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_size", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_size(MR.Box2i *_this);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2i::diagonal`.
        public unsafe int Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_diagonal", ExactSpelling = true)]
            extern static int __MR_Box2i_diagonal(MR.Box2i *_this);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2i::volume`.
        public unsafe int Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_volume", ExactSpelling = true)]
            extern static int __MR_Box2i_volume(MR.Box2i *_this);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2i::include`.
        public unsafe void Include(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_include_MR_Vector2i", ExactSpelling = true)]
            extern static void __MR_Box2i_include_MR_Vector2i(MR.Box2i *_this, MR.Const_Vector2i._Underlying *pt);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                __MR_Box2i_include_MR_Vector2i(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2i::include`.
        public unsafe void Include(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_include_MR_Box2i", ExactSpelling = true)]
            extern static void __MR_Box2i_include_MR_Box2i(MR.Box2i *_this, MR.Const_Box2i._Underlying *b);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                __MR_Box2i_include_MR_Box2i(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2i::contains`.
        public unsafe bool Contains(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_contains_MR_Vector2i", ExactSpelling = true)]
            extern static byte __MR_Box2i_contains_MR_Vector2i(MR.Box2i *_this, MR.Const_Vector2i._Underlying *pt);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_contains_MR_Vector2i(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2i::contains`.
        public unsafe bool Contains(MR.Const_Box2i otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_contains_MR_Box2i", ExactSpelling = true)]
            extern static byte __MR_Box2i_contains_MR_Box2i(MR.Box2i *_this, MR.Const_Box2i._Underlying *otherbox);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_contains_MR_Box2i(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2i::getBoxClosestPointTo`.
        public unsafe MR.Vector2i GetBoxClosestPointTo(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_getBoxClosestPointTo(MR.Box2i *_this, MR.Const_Vector2i._Underlying *pt);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2i::intersects`.
        public unsafe bool Intersects(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2i_intersects(MR.Box2i *_this, MR.Const_Box2i._Underlying *b);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2i::intersection`.
        public unsafe MR.Box2i Intersection(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_intersection", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_intersection(MR.Box2i *_this, MR.Const_Box2i._Underlying *b);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box2i::intersect`.
        public unsafe MR.Mut_Box2i Intersect(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2i._Underlying *__MR_Box2i_intersect(MR.Box2i *_this, MR.Const_Box2i._Underlying *b);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return new(__MR_Box2i_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Box2i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getDistanceSq_MR_Box2i", ExactSpelling = true)]
            extern static int __MR_Box2i_getDistanceSq_MR_Box2i(MR.Box2i *_this, MR.Const_Box2i._Underlying *b);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_getDistanceSq_MR_Box2i(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getDistanceSq_MR_Vector2i", ExactSpelling = true)]
            extern static int __MR_Box2i_getDistanceSq_MR_Vector2i(MR.Box2i *_this, MR.Const_Vector2i._Underlying *pt);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_getDistanceSq_MR_Vector2i(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2i::getProjection`.
        public unsafe MR.Vector2i GetProjection(MR.Const_Vector2i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_getProjection", ExactSpelling = true)]
            extern static MR.Vector2i __MR_Box2i_getProjection(MR.Box2i *_this, MR.Const_Vector2i._Underlying *pt);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2i::expanded`.
        public unsafe MR.Box2i Expanded(MR.Const_Vector2i expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_expanded", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_expanded(MR.Box2i *_this, MR.Const_Vector2i._Underlying *expansion);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2i::insignificantlyExpanded`.
        public unsafe MR.Box2i InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2i __MR_Box2i_insignificantlyExpanded(MR.Box2i *_this);
            fixed (MR.Box2i *__ptr__this = &this)
            {
                return __MR_Box2i_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box2i::operator==`.
        public static unsafe bool operator==(MR.Box2i _this, MR.Box2i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2i(MR.Const_Box2i._Underlying *_this, MR.Const_Box2i._Underlying *a);
            return __MR_equal_MR_Box2i((MR.Mut_Box2i._Underlying *)&_this, (MR.Mut_Box2i._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box2i _this, MR.Box2i a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box2i a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box2i)
                return this == (MR.Box2i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box2i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box2i`/`Const_Box2i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box2i
    {
        public readonly bool HasValue;
        internal readonly Box2i Object;
        public Box2i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box2i() {HasValue = false;}
        public _InOpt_Box2i(Box2i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box2i(Box2i new_value) {return new(new_value);}
        public _InOpt_Box2i(Const_Box2i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box2i(Const_Box2i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box2i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2i`/`Const_Box2i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box2i`.
    public class _InOptMut_Box2i
    {
        public Mut_Box2i? Opt;

        public _InOptMut_Box2i() {}
        public _InOptMut_Box2i(Mut_Box2i value) {Opt = value;}
        public static implicit operator _InOptMut_Box2i(Mut_Box2i value) {return new(value);}
        public unsafe _InOptMut_Box2i(ref Box2i value)
        {
            fixed (Box2i *value_ptr = &value)
            {
                Opt = new((Const_Box2i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box2i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box2i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2i`/`Const_Box2i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box2i`.
    public class _InOptConst_Box2i
    {
        public Const_Box2i? Opt;

        public _InOptConst_Box2i() {}
        public _InOptConst_Box2i(Const_Box2i value) {Opt = value;}
        public static implicit operator _InOptConst_Box2i(Const_Box2i value) {return new(value);}
        public unsafe _InOptConst_Box2i(ref readonly Box2i value)
        {
            fixed (Box2i *value_ptr = &value)
            {
                Opt = new((Const_Box2i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2i64`.
    /// This is the const reference to the struct.
    public class Const_Box2i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box2i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box2i64 UnderlyingStruct => ref *(Box2i64 *)_UnderlyingPtr;

        internal unsafe Const_Box2i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Box2i64_Destroy(_Underlying *_this);
            __MR_Box2i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box2i64() {Dispose(false);}

        public ref readonly MR.Vector2i64 Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector2i64 Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box2i64_Get_elements();
                return *__MR_Box2i64_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box2i64(Const_Box2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2i64 _ctor_result = __MR_Box2i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Box2i64::Box2i64`.
        public unsafe Const_Box2i64(MR.Const_Vector2i64 min, MR.Const_Vector2i64 max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_Construct_2(MR.Const_Vector2i64._Underlying *min, MR.Const_Vector2i64._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2i64 _ctor_result = __MR_Box2i64_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2i64::Box2i64`.
        public unsafe Const_Box2i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2i64 _ctor_result = __MR_Box2i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2i64::operator[]`.
        public unsafe MR.Const_Vector2i64 Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i64._Underlying *__MR_Box2i64_index_const(_Underlying *_this, int e);
            return new(__MR_Box2i64_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box2i64::fromMinAndSize`.
        public static unsafe MR.Box2i64 FromMinAndSize(MR.Const_Vector2i64 min, MR.Const_Vector2i64 size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_fromMinAndSize(MR.Const_Vector2i64._Underlying *min, MR.Const_Vector2i64._Underlying *size);
            return __MR_Box2i64_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2i64::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_valid", ExactSpelling = true)]
            extern static byte __MR_Box2i64_valid(_Underlying *_this);
            return __MR_Box2i64_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box2i64::center`.
        public unsafe MR.Vector2i64 Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_center", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_center(_Underlying *_this);
            return __MR_Box2i64_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2i64::corner`.
        public unsafe MR.Vector2i64 Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_corner", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_corner(_Underlying *_this, MR.Const_Vector2b._Underlying *c);
            return __MR_Box2i64_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2i64::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i64_getMinBoxCorner(MR.Const_Vector2i64._Underlying *n);
            return __MR_Box2i64_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2i64::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i64_getMaxBoxCorner(MR.Const_Vector2i64._Underlying *n);
            return __MR_Box2i64_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2i64::size`.
        public unsafe MR.Vector2i64 Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_size", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_size(_Underlying *_this);
            return __MR_Box2i64_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2i64::diagonal`.
        public unsafe long Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_diagonal", ExactSpelling = true)]
            extern static long __MR_Box2i64_diagonal(_Underlying *_this);
            return __MR_Box2i64_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2i64::volume`.
        public unsafe long Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_volume", ExactSpelling = true)]
            extern static long __MR_Box2i64_volume(_Underlying *_this);
            return __MR_Box2i64_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2i64::contains`.
        public unsafe bool Contains(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_contains_MR_Vector2i64", ExactSpelling = true)]
            extern static byte __MR_Box2i64_contains_MR_Vector2i64(_Underlying *_this, MR.Const_Vector2i64._Underlying *pt);
            return __MR_Box2i64_contains_MR_Vector2i64(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2i64::contains`.
        public unsafe bool Contains(MR.Const_Box2i64 otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_contains_MR_Box2i64", ExactSpelling = true)]
            extern static byte __MR_Box2i64_contains_MR_Box2i64(_Underlying *_this, MR.Const_Box2i64._Underlying *otherbox);
            return __MR_Box2i64_contains_MR_Box2i64(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2i64::getBoxClosestPointTo`.
        public unsafe MR.Vector2i64 GetBoxClosestPointTo(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector2i64._Underlying *pt);
            return __MR_Box2i64_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2i64::intersects`.
        public unsafe bool Intersects(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2i64_intersects(_Underlying *_this, MR.Const_Box2i64._Underlying *b);
            return __MR_Box2i64_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2i64::intersection`.
        public unsafe MR.Box2i64 Intersection(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_intersection", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_intersection(_Underlying *_this, MR.Const_Box2i64._Underlying *b);
            return __MR_Box2i64_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getDistanceSq_MR_Box2i64", ExactSpelling = true)]
            extern static long __MR_Box2i64_getDistanceSq_MR_Box2i64(_Underlying *_this, MR.Const_Box2i64._Underlying *b);
            return __MR_Box2i64_getDistanceSq_MR_Box2i64(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getDistanceSq_MR_Vector2i64", ExactSpelling = true)]
            extern static long __MR_Box2i64_getDistanceSq_MR_Vector2i64(_Underlying *_this, MR.Const_Vector2i64._Underlying *pt);
            return __MR_Box2i64_getDistanceSq_MR_Vector2i64(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2i64::getProjection`.
        public unsafe MR.Vector2i64 GetProjection(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getProjection", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_getProjection(_Underlying *_this, MR.Const_Vector2i64._Underlying *pt);
            return __MR_Box2i64_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2i64::expanded`.
        public unsafe MR.Box2i64 Expanded(MR.Const_Vector2i64 expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_expanded", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_expanded(_Underlying *_this, MR.Const_Vector2i64._Underlying *expansion);
            return __MR_Box2i64_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2i64::insignificantlyExpanded`.
        public unsafe MR.Box2i64 InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box2i64_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box2i64::operator==`.
        public static unsafe bool operator==(MR.Const_Box2i64 _this, MR.Const_Box2i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2i64(MR.Const_Box2i64._Underlying *_this, MR.Const_Box2i64._Underlying *a);
            return __MR_equal_MR_Box2i64(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box2i64 _this, MR.Const_Box2i64 a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box2i64? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box2i64)
                return this == (MR.Const_Box2i64)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Box2i64 : Const_Box2i64
    {
        /// Get the underlying struct.
        public unsafe new ref Box2i64 UnderlyingStruct => ref *(Box2i64 *)_UnderlyingPtr;

        internal unsafe Mut_Box2i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector2i64 Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector2i64 Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box2i64(Const_Box2i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box2i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2i64 _ctor_result = __MR_Box2i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Box2i64::Box2i64`.
        public unsafe Mut_Box2i64(MR.Const_Vector2i64 min, MR.Const_Vector2i64 max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_Construct_2(MR.Const_Vector2i64._Underlying *min, MR.Const_Vector2i64._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2i64 _ctor_result = __MR_Box2i64_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2i64::Box2i64`.
        public unsafe Mut_Box2i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2i64 _ctor_result = __MR_Box2i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Box2i64::operator[]`.
        public unsafe new MR.Mut_Vector2i64 Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_Box2i64_index(_Underlying *_this, int e);
            return new(__MR_Box2i64_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2i64::include`.
        public unsafe void Include(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_include_MR_Vector2i64", ExactSpelling = true)]
            extern static void __MR_Box2i64_include_MR_Vector2i64(_Underlying *_this, MR.Const_Vector2i64._Underlying *pt);
            __MR_Box2i64_include_MR_Vector2i64(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2i64::include`.
        public unsafe void Include(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_include_MR_Box2i64", ExactSpelling = true)]
            extern static void __MR_Box2i64_include_MR_Box2i64(_Underlying *_this, MR.Const_Box2i64._Underlying *b);
            __MR_Box2i64_include_MR_Box2i64(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box2i64::intersect`.
        public unsafe MR.Mut_Box2i64 Intersect(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2i64._Underlying *__MR_Box2i64_intersect(_Underlying *_this, MR.Const_Box2i64._Underlying *b);
            return new(__MR_Box2i64_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
    public struct Box2i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box2i64(Const_Box2i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box2i64(Box2i64 other) => new(new Mut_Box2i64((Mut_Box2i64._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2i64 Min;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector2i64 Max;

        /// Generated copy constructor.
        public Box2i64(Box2i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box2i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_DefaultConstruct();
            this = __MR_Box2i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box2i64::Box2i64`.
        public unsafe Box2i64(MR.Const_Vector2i64 min, MR.Const_Vector2i64 max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_Construct_2(MR.Const_Vector2i64._Underlying *min, MR.Const_Vector2i64._Underlying *max);
            this = __MR_Box2i64_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2i64::Box2i64`.
        public unsafe Box2i64(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box2i64_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2i64::operator[]`.
        public unsafe MR.Const_Vector2i64 Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2i64._Underlying *__MR_Box2i64_index_const(MR.Box2i64 *_this, int e);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return new(__MR_Box2i64_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2i64::operator[]`.
        public unsafe MR.Mut_Vector2i64 Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2i64._Underlying *__MR_Box2i64_index(MR.Box2i64 *_this, int e);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return new(__MR_Box2i64_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2i64::fromMinAndSize`.
        public static unsafe MR.Box2i64 FromMinAndSize(MR.Const_Vector2i64 min, MR.Const_Vector2i64 size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_fromMinAndSize(MR.Const_Vector2i64._Underlying *min, MR.Const_Vector2i64._Underlying *size);
            return __MR_Box2i64_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2i64::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_valid", ExactSpelling = true)]
            extern static byte __MR_Box2i64_valid(MR.Box2i64 *_this);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box2i64::center`.
        public unsafe MR.Vector2i64 Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_center", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_center(MR.Box2i64 *_this);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2i64::corner`.
        public unsafe MR.Vector2i64 Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_corner", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_corner(MR.Box2i64 *_this, MR.Const_Vector2b._Underlying *c);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2i64::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i64_getMinBoxCorner(MR.Const_Vector2i64._Underlying *n);
            return __MR_Box2i64_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2i64::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2i64_getMaxBoxCorner(MR.Const_Vector2i64._Underlying *n);
            return __MR_Box2i64_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2i64::size`.
        public unsafe MR.Vector2i64 Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_size", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_size(MR.Box2i64 *_this);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2i64::diagonal`.
        public unsafe long Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_diagonal", ExactSpelling = true)]
            extern static long __MR_Box2i64_diagonal(MR.Box2i64 *_this);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2i64::volume`.
        public unsafe long Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_volume", ExactSpelling = true)]
            extern static long __MR_Box2i64_volume(MR.Box2i64 *_this);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2i64::include`.
        public unsafe void Include(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_include_MR_Vector2i64", ExactSpelling = true)]
            extern static void __MR_Box2i64_include_MR_Vector2i64(MR.Box2i64 *_this, MR.Const_Vector2i64._Underlying *pt);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                __MR_Box2i64_include_MR_Vector2i64(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2i64::include`.
        public unsafe void Include(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_include_MR_Box2i64", ExactSpelling = true)]
            extern static void __MR_Box2i64_include_MR_Box2i64(MR.Box2i64 *_this, MR.Const_Box2i64._Underlying *b);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                __MR_Box2i64_include_MR_Box2i64(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2i64::contains`.
        public unsafe bool Contains(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_contains_MR_Vector2i64", ExactSpelling = true)]
            extern static byte __MR_Box2i64_contains_MR_Vector2i64(MR.Box2i64 *_this, MR.Const_Vector2i64._Underlying *pt);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_contains_MR_Vector2i64(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2i64::contains`.
        public unsafe bool Contains(MR.Const_Box2i64 otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_contains_MR_Box2i64", ExactSpelling = true)]
            extern static byte __MR_Box2i64_contains_MR_Box2i64(MR.Box2i64 *_this, MR.Const_Box2i64._Underlying *otherbox);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_contains_MR_Box2i64(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2i64::getBoxClosestPointTo`.
        public unsafe MR.Vector2i64 GetBoxClosestPointTo(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_getBoxClosestPointTo(MR.Box2i64 *_this, MR.Const_Vector2i64._Underlying *pt);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2i64::intersects`.
        public unsafe bool Intersects(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2i64_intersects(MR.Box2i64 *_this, MR.Const_Box2i64._Underlying *b);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2i64::intersection`.
        public unsafe MR.Box2i64 Intersection(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_intersection", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_intersection(MR.Box2i64 *_this, MR.Const_Box2i64._Underlying *b);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box2i64::intersect`.
        public unsafe MR.Mut_Box2i64 Intersect(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2i64._Underlying *__MR_Box2i64_intersect(MR.Box2i64 *_this, MR.Const_Box2i64._Underlying *b);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return new(__MR_Box2i64_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Box2i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getDistanceSq_MR_Box2i64", ExactSpelling = true)]
            extern static long __MR_Box2i64_getDistanceSq_MR_Box2i64(MR.Box2i64 *_this, MR.Const_Box2i64._Underlying *b);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_getDistanceSq_MR_Box2i64(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getDistanceSq_MR_Vector2i64", ExactSpelling = true)]
            extern static long __MR_Box2i64_getDistanceSq_MR_Vector2i64(MR.Box2i64 *_this, MR.Const_Vector2i64._Underlying *pt);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_getDistanceSq_MR_Vector2i64(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2i64::getProjection`.
        public unsafe MR.Vector2i64 GetProjection(MR.Const_Vector2i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_getProjection", ExactSpelling = true)]
            extern static MR.Vector2i64 __MR_Box2i64_getProjection(MR.Box2i64 *_this, MR.Const_Vector2i64._Underlying *pt);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2i64::expanded`.
        public unsafe MR.Box2i64 Expanded(MR.Const_Vector2i64 expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_expanded", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_expanded(MR.Box2i64 *_this, MR.Const_Vector2i64._Underlying *expansion);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2i64::insignificantlyExpanded`.
        public unsafe MR.Box2i64 InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2i64_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2i64 __MR_Box2i64_insignificantlyExpanded(MR.Box2i64 *_this);
            fixed (MR.Box2i64 *__ptr__this = &this)
            {
                return __MR_Box2i64_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box2i64::operator==`.
        public static unsafe bool operator==(MR.Box2i64 _this, MR.Box2i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2i64(MR.Const_Box2i64._Underlying *_this, MR.Const_Box2i64._Underlying *a);
            return __MR_equal_MR_Box2i64((MR.Mut_Box2i64._Underlying *)&_this, (MR.Mut_Box2i64._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box2i64 _this, MR.Box2i64 a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box2i64 a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box2i64)
                return this == (MR.Box2i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box2i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box2i64`/`Const_Box2i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box2i64
    {
        public readonly bool HasValue;
        internal readonly Box2i64 Object;
        public Box2i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box2i64() {HasValue = false;}
        public _InOpt_Box2i64(Box2i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box2i64(Box2i64 new_value) {return new(new_value);}
        public _InOpt_Box2i64(Const_Box2i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box2i64(Const_Box2i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box2i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2i64`/`Const_Box2i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box2i64`.
    public class _InOptMut_Box2i64
    {
        public Mut_Box2i64? Opt;

        public _InOptMut_Box2i64() {}
        public _InOptMut_Box2i64(Mut_Box2i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Box2i64(Mut_Box2i64 value) {return new(value);}
        public unsafe _InOptMut_Box2i64(ref Box2i64 value)
        {
            fixed (Box2i64 *value_ptr = &value)
            {
                Opt = new((Const_Box2i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box2i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box2i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2i64`/`Const_Box2i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box2i64`.
    public class _InOptConst_Box2i64
    {
        public Const_Box2i64? Opt;

        public _InOptConst_Box2i64() {}
        public _InOptConst_Box2i64(Const_Box2i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Box2i64(Const_Box2i64 value) {return new(value);}
        public unsafe _InOptConst_Box2i64(ref readonly Box2i64 value)
        {
            fixed (Box2i64 *value_ptr = &value)
            {
                Opt = new((Const_Box2i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2f`.
    /// This is the const reference to the struct.
    public class Const_Box2f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box2f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box2f UnderlyingStruct => ref *(Box2f *)_UnderlyingPtr;

        internal unsafe Const_Box2f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Destroy", ExactSpelling = true)]
            extern static void __MR_Box2f_Destroy(_Underlying *_this);
            __MR_Box2f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box2f() {Dispose(false);}

        public ref readonly MR.Vector2f Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector2f Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box2f_Get_elements();
                return *__MR_Box2f_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box2f(Const_Box2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2f _ctor_result = __MR_Box2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box2f::Box2f`.
        public unsafe Const_Box2f(MR.Const_Vector2f min, MR.Const_Vector2f max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Construct_2", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_Construct_2(MR.Const_Vector2f._Underlying *min, MR.Const_Vector2f._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2f _ctor_result = __MR_Box2f_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2f::Box2f`.
        public unsafe Const_Box2f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Construct_1", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2f _ctor_result = __MR_Box2f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2f::operator[]`.
        public unsafe MR.Const_Vector2f Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2f._Underlying *__MR_Box2f_index_const(_Underlying *_this, int e);
            return new(__MR_Box2f_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box2f::fromMinAndSize`.
        public static unsafe MR.Box2f FromMinAndSize(MR.Const_Vector2f min, MR.Const_Vector2f size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_fromMinAndSize(MR.Const_Vector2f._Underlying *min, MR.Const_Vector2f._Underlying *size);
            return __MR_Box2f_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2f::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_valid", ExactSpelling = true)]
            extern static byte __MR_Box2f_valid(_Underlying *_this);
            return __MR_Box2f_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box2f::center`.
        public unsafe MR.Vector2f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_center", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_center(_Underlying *_this);
            return __MR_Box2f_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2f::corner`.
        public unsafe MR.Vector2f Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_corner", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_corner(_Underlying *_this, MR.Const_Vector2b._Underlying *c);
            return __MR_Box2f_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2f::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2f_getMinBoxCorner(MR.Const_Vector2f._Underlying *n);
            return __MR_Box2f_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2f::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2f_getMaxBoxCorner(MR.Const_Vector2f._Underlying *n);
            return __MR_Box2f_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2f::size`.
        public unsafe MR.Vector2f Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_size", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_size(_Underlying *_this);
            return __MR_Box2f_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2f::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_diagonal", ExactSpelling = true)]
            extern static float __MR_Box2f_diagonal(_Underlying *_this);
            return __MR_Box2f_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2f::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_volume", ExactSpelling = true)]
            extern static float __MR_Box2f_volume(_Underlying *_this);
            return __MR_Box2f_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2f::contains`.
        public unsafe bool Contains(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_contains_MR_Vector2f", ExactSpelling = true)]
            extern static byte __MR_Box2f_contains_MR_Vector2f(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            return __MR_Box2f_contains_MR_Vector2f(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2f::contains`.
        public unsafe bool Contains(MR.Const_Box2f otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_contains_MR_Box2f", ExactSpelling = true)]
            extern static byte __MR_Box2f_contains_MR_Box2f(_Underlying *_this, MR.Const_Box2f._Underlying *otherbox);
            return __MR_Box2f_contains_MR_Box2f(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2f::getBoxClosestPointTo`.
        public unsafe MR.Vector2f GetBoxClosestPointTo(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            return __MR_Box2f_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2f::intersects`.
        public unsafe bool Intersects(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2f_intersects(_Underlying *_this, MR.Const_Box2f._Underlying *b);
            return __MR_Box2f_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2f::intersection`.
        public unsafe MR.Box2f Intersection(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_intersection", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_intersection(_Underlying *_this, MR.Const_Box2f._Underlying *b);
            return __MR_Box2f_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getDistanceSq_MR_Box2f", ExactSpelling = true)]
            extern static float __MR_Box2f_getDistanceSq_MR_Box2f(_Underlying *_this, MR.Const_Box2f._Underlying *b);
            return __MR_Box2f_getDistanceSq_MR_Box2f(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getDistanceSq_MR_Vector2f", ExactSpelling = true)]
            extern static float __MR_Box2f_getDistanceSq_MR_Vector2f(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            return __MR_Box2f_getDistanceSq_MR_Vector2f(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2f::getProjection`.
        public unsafe MR.Vector2f GetProjection(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getProjection", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_getProjection(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            return __MR_Box2f_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2f::expanded`.
        public unsafe MR.Box2f Expanded(MR.Const_Vector2f expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_expanded", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_expanded(_Underlying *_this, MR.Const_Vector2f._Underlying *expansion);
            return __MR_Box2f_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2f::insignificantlyExpanded`.
        public unsafe MR.Box2f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box2f_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box2f::operator==`.
        public static unsafe bool operator==(MR.Const_Box2f _this, MR.Const_Box2f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2f(MR.Const_Box2f._Underlying *_this, MR.Const_Box2f._Underlying *a);
            return __MR_equal_MR_Box2f(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box2f _this, MR.Const_Box2f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box2f? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box2f)
                return this == (MR.Const_Box2f)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2f`.
    /// This is the non-const reference to the struct.
    public class Mut_Box2f : Const_Box2f
    {
        /// Get the underlying struct.
        public unsafe new ref Box2f UnderlyingStruct => ref *(Box2f *)_UnderlyingPtr;

        internal unsafe Mut_Box2f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector2f Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector2f Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box2f(Const_Box2f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 16);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box2f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2f _ctor_result = __MR_Box2f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from constructor `MR::Box2f::Box2f`.
        public unsafe Mut_Box2f(MR.Const_Vector2f min, MR.Const_Vector2f max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Construct_2", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_Construct_2(MR.Const_Vector2f._Underlying *min, MR.Const_Vector2f._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2f _ctor_result = __MR_Box2f_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2f::Box2f`.
        public unsafe Mut_Box2f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Construct_1", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(16);
            MR.Box2f _ctor_result = __MR_Box2f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 16);
        }

        /// Generated from method `MR::Box2f::operator[]`.
        public unsafe new MR.Mut_Vector2f Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_Box2f_index(_Underlying *_this, int e);
            return new(__MR_Box2f_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2f::include`.
        public unsafe void Include(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_include_MR_Vector2f", ExactSpelling = true)]
            extern static void __MR_Box2f_include_MR_Vector2f(_Underlying *_this, MR.Const_Vector2f._Underlying *pt);
            __MR_Box2f_include_MR_Vector2f(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2f::include`.
        public unsafe void Include(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_include_MR_Box2f", ExactSpelling = true)]
            extern static void __MR_Box2f_include_MR_Box2f(_Underlying *_this, MR.Const_Box2f._Underlying *b);
            __MR_Box2f_include_MR_Box2f(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box2f::intersect`.
        public unsafe MR.Mut_Box2f Intersect(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2f._Underlying *__MR_Box2f_intersect(_Underlying *_this, MR.Const_Box2f._Underlying *b);
            return new(__MR_Box2f_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
    public struct Box2f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box2f(Const_Box2f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box2f(Box2f other) => new(new Mut_Box2f((Mut_Box2f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2f Min;

        [System.Runtime.InteropServices.FieldOffset(8)]
        public MR.Vector2f Max;

        /// Generated copy constructor.
        public Box2f(Box2f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box2f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_DefaultConstruct();
            this = __MR_Box2f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box2f::Box2f`.
        public unsafe Box2f(MR.Const_Vector2f min, MR.Const_Vector2f max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Construct_2", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_Construct_2(MR.Const_Vector2f._Underlying *min, MR.Const_Vector2f._Underlying *max);
            this = __MR_Box2f_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2f::Box2f`.
        public unsafe Box2f(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_Construct_1", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box2f_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2f::operator[]`.
        public unsafe MR.Const_Vector2f Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2f._Underlying *__MR_Box2f_index_const(MR.Box2f *_this, int e);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return new(__MR_Box2f_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2f::operator[]`.
        public unsafe MR.Mut_Vector2f Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2f._Underlying *__MR_Box2f_index(MR.Box2f *_this, int e);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return new(__MR_Box2f_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2f::fromMinAndSize`.
        public static unsafe MR.Box2f FromMinAndSize(MR.Const_Vector2f min, MR.Const_Vector2f size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_fromMinAndSize(MR.Const_Vector2f._Underlying *min, MR.Const_Vector2f._Underlying *size);
            return __MR_Box2f_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2f::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_valid", ExactSpelling = true)]
            extern static byte __MR_Box2f_valid(MR.Box2f *_this);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box2f::center`.
        public unsafe MR.Vector2f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_center", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_center(MR.Box2f *_this);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2f::corner`.
        public unsafe MR.Vector2f Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_corner", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_corner(MR.Box2f *_this, MR.Const_Vector2b._Underlying *c);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2f::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2f_getMinBoxCorner(MR.Const_Vector2f._Underlying *n);
            return __MR_Box2f_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2f::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2f_getMaxBoxCorner(MR.Const_Vector2f._Underlying *n);
            return __MR_Box2f_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2f::size`.
        public unsafe MR.Vector2f Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_size", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_size(MR.Box2f *_this);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2f::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_diagonal", ExactSpelling = true)]
            extern static float __MR_Box2f_diagonal(MR.Box2f *_this);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2f::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_volume", ExactSpelling = true)]
            extern static float __MR_Box2f_volume(MR.Box2f *_this);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2f::include`.
        public unsafe void Include(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_include_MR_Vector2f", ExactSpelling = true)]
            extern static void __MR_Box2f_include_MR_Vector2f(MR.Box2f *_this, MR.Const_Vector2f._Underlying *pt);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                __MR_Box2f_include_MR_Vector2f(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2f::include`.
        public unsafe void Include(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_include_MR_Box2f", ExactSpelling = true)]
            extern static void __MR_Box2f_include_MR_Box2f(MR.Box2f *_this, MR.Const_Box2f._Underlying *b);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                __MR_Box2f_include_MR_Box2f(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2f::contains`.
        public unsafe bool Contains(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_contains_MR_Vector2f", ExactSpelling = true)]
            extern static byte __MR_Box2f_contains_MR_Vector2f(MR.Box2f *_this, MR.Const_Vector2f._Underlying *pt);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_contains_MR_Vector2f(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2f::contains`.
        public unsafe bool Contains(MR.Const_Box2f otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_contains_MR_Box2f", ExactSpelling = true)]
            extern static byte __MR_Box2f_contains_MR_Box2f(MR.Box2f *_this, MR.Const_Box2f._Underlying *otherbox);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_contains_MR_Box2f(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2f::getBoxClosestPointTo`.
        public unsafe MR.Vector2f GetBoxClosestPointTo(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_getBoxClosestPointTo(MR.Box2f *_this, MR.Const_Vector2f._Underlying *pt);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2f::intersects`.
        public unsafe bool Intersects(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2f_intersects(MR.Box2f *_this, MR.Const_Box2f._Underlying *b);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2f::intersection`.
        public unsafe MR.Box2f Intersection(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_intersection", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_intersection(MR.Box2f *_this, MR.Const_Box2f._Underlying *b);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box2f::intersect`.
        public unsafe MR.Mut_Box2f Intersect(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2f._Underlying *__MR_Box2f_intersect(MR.Box2f *_this, MR.Const_Box2f._Underlying *b);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return new(__MR_Box2f_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Box2f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getDistanceSq_MR_Box2f", ExactSpelling = true)]
            extern static float __MR_Box2f_getDistanceSq_MR_Box2f(MR.Box2f *_this, MR.Const_Box2f._Underlying *b);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_getDistanceSq_MR_Box2f(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getDistanceSq_MR_Vector2f", ExactSpelling = true)]
            extern static float __MR_Box2f_getDistanceSq_MR_Vector2f(MR.Box2f *_this, MR.Const_Vector2f._Underlying *pt);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_getDistanceSq_MR_Vector2f(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2f::getProjection`.
        public unsafe MR.Vector2f GetProjection(MR.Const_Vector2f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_getProjection", ExactSpelling = true)]
            extern static MR.Vector2f __MR_Box2f_getProjection(MR.Box2f *_this, MR.Const_Vector2f._Underlying *pt);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2f::expanded`.
        public unsafe MR.Box2f Expanded(MR.Const_Vector2f expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_expanded", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_expanded(MR.Box2f *_this, MR.Const_Vector2f._Underlying *expansion);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2f::insignificantlyExpanded`.
        public unsafe MR.Box2f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2f_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2f __MR_Box2f_insignificantlyExpanded(MR.Box2f *_this);
            fixed (MR.Box2f *__ptr__this = &this)
            {
                return __MR_Box2f_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box2f::operator==`.
        public static unsafe bool operator==(MR.Box2f _this, MR.Box2f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2f(MR.Const_Box2f._Underlying *_this, MR.Const_Box2f._Underlying *a);
            return __MR_equal_MR_Box2f((MR.Mut_Box2f._Underlying *)&_this, (MR.Mut_Box2f._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box2f _this, MR.Box2f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box2f a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box2f)
                return this == (MR.Box2f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box2f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box2f`/`Const_Box2f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box2f
    {
        public readonly bool HasValue;
        internal readonly Box2f Object;
        public Box2f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box2f() {HasValue = false;}
        public _InOpt_Box2f(Box2f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box2f(Box2f new_value) {return new(new_value);}
        public _InOpt_Box2f(Const_Box2f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box2f(Const_Box2f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box2f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2f`/`Const_Box2f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box2f`.
    public class _InOptMut_Box2f
    {
        public Mut_Box2f? Opt;

        public _InOptMut_Box2f() {}
        public _InOptMut_Box2f(Mut_Box2f value) {Opt = value;}
        public static implicit operator _InOptMut_Box2f(Mut_Box2f value) {return new(value);}
        public unsafe _InOptMut_Box2f(ref Box2f value)
        {
            fixed (Box2f *value_ptr = &value)
            {
                Opt = new((Const_Box2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box2f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box2f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2f`/`Const_Box2f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box2f`.
    public class _InOptConst_Box2f
    {
        public Const_Box2f? Opt;

        public _InOptConst_Box2f() {}
        public _InOptConst_Box2f(Const_Box2f value) {Opt = value;}
        public static implicit operator _InOptConst_Box2f(Const_Box2f value) {return new(value);}
        public unsafe _InOptConst_Box2f(ref readonly Box2f value)
        {
            fixed (Box2f *value_ptr = &value)
            {
                Opt = new((Const_Box2f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2d`.
    /// This is the const reference to the struct.
    public class Const_Box2d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box2d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box2d UnderlyingStruct => ref *(Box2d *)_UnderlyingPtr;

        internal unsafe Const_Box2d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Destroy", ExactSpelling = true)]
            extern static void __MR_Box2d_Destroy(_Underlying *_this);
            __MR_Box2d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box2d() {Dispose(false);}

        public ref readonly MR.Vector2d Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector2d Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box2d_Get_elements();
                return *__MR_Box2d_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box2d(Const_Box2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2d _ctor_result = __MR_Box2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Box2d::Box2d`.
        public unsafe Const_Box2d(MR.Const_Vector2d min, MR.Const_Vector2d max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Construct_2", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_Construct_2(MR.Const_Vector2d._Underlying *min, MR.Const_Vector2d._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2d _ctor_result = __MR_Box2d_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2d::Box2d`.
        public unsafe Const_Box2d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Construct_1", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2d _ctor_result = __MR_Box2d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2d::operator[]`.
        public unsafe MR.Const_Vector2d Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2d._Underlying *__MR_Box2d_index_const(_Underlying *_this, int e);
            return new(__MR_Box2d_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box2d::fromMinAndSize`.
        public static unsafe MR.Box2d FromMinAndSize(MR.Const_Vector2d min, MR.Const_Vector2d size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_fromMinAndSize(MR.Const_Vector2d._Underlying *min, MR.Const_Vector2d._Underlying *size);
            return __MR_Box2d_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2d::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_valid", ExactSpelling = true)]
            extern static byte __MR_Box2d_valid(_Underlying *_this);
            return __MR_Box2d_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box2d::center`.
        public unsafe MR.Vector2d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_center", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_center(_Underlying *_this);
            return __MR_Box2d_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2d::corner`.
        public unsafe MR.Vector2d Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_corner", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_corner(_Underlying *_this, MR.Const_Vector2b._Underlying *c);
            return __MR_Box2d_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2d::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2d_getMinBoxCorner(MR.Const_Vector2d._Underlying *n);
            return __MR_Box2d_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2d::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2d_getMaxBoxCorner(MR.Const_Vector2d._Underlying *n);
            return __MR_Box2d_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2d::size`.
        public unsafe MR.Vector2d Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_size", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_size(_Underlying *_this);
            return __MR_Box2d_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2d::diagonal`.
        public unsafe double Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_diagonal", ExactSpelling = true)]
            extern static double __MR_Box2d_diagonal(_Underlying *_this);
            return __MR_Box2d_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2d::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_volume", ExactSpelling = true)]
            extern static double __MR_Box2d_volume(_Underlying *_this);
            return __MR_Box2d_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2d::contains`.
        public unsafe bool Contains(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_contains_MR_Vector2d", ExactSpelling = true)]
            extern static byte __MR_Box2d_contains_MR_Vector2d(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            return __MR_Box2d_contains_MR_Vector2d(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2d::contains`.
        public unsafe bool Contains(MR.Const_Box2d otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_contains_MR_Box2d", ExactSpelling = true)]
            extern static byte __MR_Box2d_contains_MR_Box2d(_Underlying *_this, MR.Const_Box2d._Underlying *otherbox);
            return __MR_Box2d_contains_MR_Box2d(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2d::getBoxClosestPointTo`.
        public unsafe MR.Vector2d GetBoxClosestPointTo(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            return __MR_Box2d_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2d::intersects`.
        public unsafe bool Intersects(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2d_intersects(_Underlying *_this, MR.Const_Box2d._Underlying *b);
            return __MR_Box2d_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2d::intersection`.
        public unsafe MR.Box2d Intersection(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_intersection", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_intersection(_Underlying *_this, MR.Const_Box2d._Underlying *b);
            return __MR_Box2d_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getDistanceSq_MR_Box2d", ExactSpelling = true)]
            extern static double __MR_Box2d_getDistanceSq_MR_Box2d(_Underlying *_this, MR.Const_Box2d._Underlying *b);
            return __MR_Box2d_getDistanceSq_MR_Box2d(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getDistanceSq_MR_Vector2d", ExactSpelling = true)]
            extern static double __MR_Box2d_getDistanceSq_MR_Vector2d(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            return __MR_Box2d_getDistanceSq_MR_Vector2d(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2d::getProjection`.
        public unsafe MR.Vector2d GetProjection(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getProjection", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_getProjection(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            return __MR_Box2d_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2d::expanded`.
        public unsafe MR.Box2d Expanded(MR.Const_Vector2d expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_expanded", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_expanded(_Underlying *_this, MR.Const_Vector2d._Underlying *expansion);
            return __MR_Box2d_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2d::insignificantlyExpanded`.
        public unsafe MR.Box2d InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box2d_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box2d::operator==`.
        public static unsafe bool operator==(MR.Const_Box2d _this, MR.Const_Box2d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2d(MR.Const_Box2d._Underlying *_this, MR.Const_Box2d._Underlying *a);
            return __MR_equal_MR_Box2d(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box2d _this, MR.Const_Box2d a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box2d? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box2d)
                return this == (MR.Const_Box2d)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2d`.
    /// This is the non-const reference to the struct.
    public class Mut_Box2d : Const_Box2d
    {
        /// Get the underlying struct.
        public unsafe new ref Box2d UnderlyingStruct => ref *(Box2d *)_UnderlyingPtr;

        internal unsafe Mut_Box2d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector2d Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector2d Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box2d(Const_Box2d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 32);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box2d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2d _ctor_result = __MR_Box2d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from constructor `MR::Box2d::Box2d`.
        public unsafe Mut_Box2d(MR.Const_Vector2d min, MR.Const_Vector2d max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Construct_2", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_Construct_2(MR.Const_Vector2d._Underlying *min, MR.Const_Vector2d._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2d _ctor_result = __MR_Box2d_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2d::Box2d`.
        public unsafe Mut_Box2d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Construct_1", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(32);
            MR.Box2d _ctor_result = __MR_Box2d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 32);
        }

        /// Generated from method `MR::Box2d::operator[]`.
        public unsafe new MR.Mut_Vector2d Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_Box2d_index(_Underlying *_this, int e);
            return new(__MR_Box2d_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2d::include`.
        public unsafe void Include(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_include_MR_Vector2d", ExactSpelling = true)]
            extern static void __MR_Box2d_include_MR_Vector2d(_Underlying *_this, MR.Const_Vector2d._Underlying *pt);
            __MR_Box2d_include_MR_Vector2d(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2d::include`.
        public unsafe void Include(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_include_MR_Box2d", ExactSpelling = true)]
            extern static void __MR_Box2d_include_MR_Box2d(_Underlying *_this, MR.Const_Box2d._Underlying *b);
            __MR_Box2d_include_MR_Box2d(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box2d::intersect`.
        public unsafe MR.Mut_Box2d Intersect(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2d._Underlying *__MR_Box2d_intersect(_Underlying *_this, MR.Const_Box2d._Underlying *b);
            return new(__MR_Box2d_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box2d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
    public struct Box2d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box2d(Const_Box2d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box2d(Box2d other) => new(new Mut_Box2d((Mut_Box2d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector2d Min;

        [System.Runtime.InteropServices.FieldOffset(16)]
        public MR.Vector2d Max;

        /// Generated copy constructor.
        public Box2d(Box2d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box2d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_DefaultConstruct();
            this = __MR_Box2d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box2d::Box2d`.
        public unsafe Box2d(MR.Const_Vector2d min, MR.Const_Vector2d max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Construct_2", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_Construct_2(MR.Const_Vector2d._Underlying *min, MR.Const_Vector2d._Underlying *max);
            this = __MR_Box2d_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box2d::Box2d`.
        public unsafe Box2d(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_Construct_1", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box2d_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box2d::operator[]`.
        public unsafe MR.Const_Vector2d Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector2d._Underlying *__MR_Box2d_index_const(MR.Box2d *_this, int e);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return new(__MR_Box2d_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2d::operator[]`.
        public unsafe MR.Mut_Vector2d Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector2d._Underlying *__MR_Box2d_index(MR.Box2d *_this, int e);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return new(__MR_Box2d_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box2d::fromMinAndSize`.
        public static unsafe MR.Box2d FromMinAndSize(MR.Const_Vector2d min, MR.Const_Vector2d size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_fromMinAndSize(MR.Const_Vector2d._Underlying *min, MR.Const_Vector2d._Underlying *size);
            return __MR_Box2d_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box2d::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_valid", ExactSpelling = true)]
            extern static byte __MR_Box2d_valid(MR.Box2d *_this);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box2d::center`.
        public unsafe MR.Vector2d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_center", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_center(MR.Box2d *_this);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box2d::corner`.
        public unsafe MR.Vector2d Corner(MR.Const_Vector2b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_corner", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_corner(MR.Box2d *_this, MR.Const_Vector2b._Underlying *c);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box2d::getMinBoxCorner`.
        public static unsafe MR.Vector2b GetMinBoxCorner(MR.Const_Vector2d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2d_getMinBoxCorner(MR.Const_Vector2d._Underlying *n);
            return __MR_Box2d_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box2d::getMaxBoxCorner`.
        public static unsafe MR.Vector2b GetMaxBoxCorner(MR.Const_Vector2d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector2b __MR_Box2d_getMaxBoxCorner(MR.Const_Vector2d._Underlying *n);
            return __MR_Box2d_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box2d::size`.
        public unsafe MR.Vector2d Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_size", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_size(MR.Box2d *_this);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box2d::diagonal`.
        public unsafe double Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_diagonal", ExactSpelling = true)]
            extern static double __MR_Box2d_diagonal(MR.Box2d *_this);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box2d::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_volume", ExactSpelling = true)]
            extern static double __MR_Box2d_volume(MR.Box2d *_this);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box2d::include`.
        public unsafe void Include(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_include_MR_Vector2d", ExactSpelling = true)]
            extern static void __MR_Box2d_include_MR_Vector2d(MR.Box2d *_this, MR.Const_Vector2d._Underlying *pt);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                __MR_Box2d_include_MR_Vector2d(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box2d::include`.
        public unsafe void Include(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_include_MR_Box2d", ExactSpelling = true)]
            extern static void __MR_Box2d_include_MR_Box2d(MR.Box2d *_this, MR.Const_Box2d._Underlying *b);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                __MR_Box2d_include_MR_Box2d(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box2d::contains`.
        public unsafe bool Contains(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_contains_MR_Vector2d", ExactSpelling = true)]
            extern static byte __MR_Box2d_contains_MR_Vector2d(MR.Box2d *_this, MR.Const_Vector2d._Underlying *pt);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_contains_MR_Vector2d(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box2d::contains`.
        public unsafe bool Contains(MR.Const_Box2d otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_contains_MR_Box2d", ExactSpelling = true)]
            extern static byte __MR_Box2d_contains_MR_Box2d(MR.Box2d *_this, MR.Const_Box2d._Underlying *otherbox);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_contains_MR_Box2d(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box2d::getBoxClosestPointTo`.
        public unsafe MR.Vector2d GetBoxClosestPointTo(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_getBoxClosestPointTo(MR.Box2d *_this, MR.Const_Vector2d._Underlying *pt);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box2d::intersects`.
        public unsafe bool Intersects(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_intersects", ExactSpelling = true)]
            extern static byte __MR_Box2d_intersects(MR.Box2d *_this, MR.Const_Box2d._Underlying *b);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box2d::intersection`.
        public unsafe MR.Box2d Intersection(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_intersection", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_intersection(MR.Box2d *_this, MR.Const_Box2d._Underlying *b);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box2d::intersect`.
        public unsafe MR.Mut_Box2d Intersect(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box2d._Underlying *__MR_Box2d_intersect(MR.Box2d *_this, MR.Const_Box2d._Underlying *b);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return new(__MR_Box2d_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box2d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Box2d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getDistanceSq_MR_Box2d", ExactSpelling = true)]
            extern static double __MR_Box2d_getDistanceSq_MR_Box2d(MR.Box2d *_this, MR.Const_Box2d._Underlying *b);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_getDistanceSq_MR_Box2d(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box2d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getDistanceSq_MR_Vector2d", ExactSpelling = true)]
            extern static double __MR_Box2d_getDistanceSq_MR_Vector2d(MR.Box2d *_this, MR.Const_Vector2d._Underlying *pt);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_getDistanceSq_MR_Vector2d(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box2d::getProjection`.
        public unsafe MR.Vector2d GetProjection(MR.Const_Vector2d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_getProjection", ExactSpelling = true)]
            extern static MR.Vector2d __MR_Box2d_getProjection(MR.Box2d *_this, MR.Const_Vector2d._Underlying *pt);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box2d::expanded`.
        public unsafe MR.Box2d Expanded(MR.Const_Vector2d expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_expanded", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_expanded(MR.Box2d *_this, MR.Const_Vector2d._Underlying *expansion);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box2d::insignificantlyExpanded`.
        public unsafe MR.Box2d InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box2d_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box2d __MR_Box2d_insignificantlyExpanded(MR.Box2d *_this);
            fixed (MR.Box2d *__ptr__this = &this)
            {
                return __MR_Box2d_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box2d::operator==`.
        public static unsafe bool operator==(MR.Box2d _this, MR.Box2d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box2d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box2d(MR.Const_Box2d._Underlying *_this, MR.Const_Box2d._Underlying *a);
            return __MR_equal_MR_Box2d((MR.Mut_Box2d._Underlying *)&_this, (MR.Mut_Box2d._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box2d _this, MR.Box2d a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box2d a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box2d)
                return this == (MR.Box2d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box2d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box2d`/`Const_Box2d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box2d
    {
        public readonly bool HasValue;
        internal readonly Box2d Object;
        public Box2d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box2d() {HasValue = false;}
        public _InOpt_Box2d(Box2d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box2d(Box2d new_value) {return new(new_value);}
        public _InOpt_Box2d(Const_Box2d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box2d(Const_Box2d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box2d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2d`/`Const_Box2d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box2d`.
    public class _InOptMut_Box2d
    {
        public Mut_Box2d? Opt;

        public _InOptMut_Box2d() {}
        public _InOptMut_Box2d(Mut_Box2d value) {Opt = value;}
        public static implicit operator _InOptMut_Box2d(Mut_Box2d value) {return new(value);}
        public unsafe _InOptMut_Box2d(ref Box2d value)
        {
            fixed (Box2d *value_ptr = &value)
            {
                Opt = new((Const_Box2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box2d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box2d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box2d`/`Const_Box2d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box2d`.
    public class _InOptConst_Box2d
    {
        public Const_Box2d? Opt;

        public _InOptConst_Box2d() {}
        public _InOptConst_Box2d(Const_Box2d value) {Opt = value;}
        public static implicit operator _InOptConst_Box2d(Const_Box2d value) {return new(value);}
        public unsafe _InOptConst_Box2d(ref readonly Box2d value)
        {
            fixed (Box2d *value_ptr = &value)
            {
                Opt = new((Const_Box2d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3i`.
    /// This is the const reference to the struct.
    public class Const_Box3i : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box3i>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box3i UnderlyingStruct => ref *(Box3i *)_UnderlyingPtr;

        internal unsafe Const_Box3i(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Destroy", ExactSpelling = true)]
            extern static void __MR_Box3i_Destroy(_Underlying *_this);
            __MR_Box3i_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box3i() {Dispose(false);}

        public ref readonly MR.Vector3i Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector3i Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box3i_Get_elements();
                return *__MR_Box3i_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box3i(Const_Box3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3i _ctor_result = __MR_Box3i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Box3i::Box3i`.
        public unsafe Const_Box3i(MR.Const_Vector3i min, MR.Const_Vector3i max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Construct_2", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_Construct_2(MR.Const_Vector3i._Underlying *min, MR.Const_Vector3i._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3i _ctor_result = __MR_Box3i_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3i::Box3i`.
        public unsafe Const_Box3i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Construct_1", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3i _ctor_result = __MR_Box3i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3i::operator[]`.
        public unsafe MR.Const_Vector3i Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_Box3i_index_const(_Underlying *_this, int e);
            return new(__MR_Box3i_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box3i::fromMinAndSize`.
        public static unsafe MR.Box3i FromMinAndSize(MR.Const_Vector3i min, MR.Const_Vector3i size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_fromMinAndSize(MR.Const_Vector3i._Underlying *min, MR.Const_Vector3i._Underlying *size);
            return __MR_Box3i_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3i::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_valid", ExactSpelling = true)]
            extern static byte __MR_Box3i_valid(_Underlying *_this);
            return __MR_Box3i_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box3i::center`.
        public unsafe MR.Vector3i Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_center", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_center(_Underlying *_this);
            return __MR_Box3i_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3i::corner`.
        public unsafe MR.Vector3i Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_corner", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_corner(_Underlying *_this, MR.Const_Vector3b._Underlying *c);
            return __MR_Box3i_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3i::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i_getMinBoxCorner(MR.Const_Vector3i._Underlying *n);
            return __MR_Box3i_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3i::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i_getMaxBoxCorner(MR.Const_Vector3i._Underlying *n);
            return __MR_Box3i_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3i::size`.
        public unsafe MR.Vector3i Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_size", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_size(_Underlying *_this);
            return __MR_Box3i_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3i::diagonal`.
        public unsafe int Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_diagonal", ExactSpelling = true)]
            extern static int __MR_Box3i_diagonal(_Underlying *_this);
            return __MR_Box3i_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3i::volume`.
        public unsafe int Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_volume", ExactSpelling = true)]
            extern static int __MR_Box3i_volume(_Underlying *_this);
            return __MR_Box3i_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3i::contains`.
        public unsafe bool Contains(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_contains_MR_Vector3i", ExactSpelling = true)]
            extern static byte __MR_Box3i_contains_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pt);
            return __MR_Box3i_contains_MR_Vector3i(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3i::contains`.
        public unsafe bool Contains(MR.Const_Box3i otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_contains_MR_Box3i", ExactSpelling = true)]
            extern static byte __MR_Box3i_contains_MR_Box3i(_Underlying *_this, MR.Const_Box3i._Underlying *otherbox);
            return __MR_Box3i_contains_MR_Box3i(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3i::getBoxClosestPointTo`.
        public unsafe MR.Vector3i GetBoxClosestPointTo(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector3i._Underlying *pt);
            return __MR_Box3i_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3i::intersects`.
        public unsafe bool Intersects(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3i_intersects(_Underlying *_this, MR.Const_Box3i._Underlying *b);
            return __MR_Box3i_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3i::intersection`.
        public unsafe MR.Box3i Intersection(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_intersection", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_intersection(_Underlying *_this, MR.Const_Box3i._Underlying *b);
            return __MR_Box3i_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getDistanceSq_MR_Box3i", ExactSpelling = true)]
            extern static int __MR_Box3i_getDistanceSq_MR_Box3i(_Underlying *_this, MR.Const_Box3i._Underlying *b);
            return __MR_Box3i_getDistanceSq_MR_Box3i(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getDistanceSq_MR_Vector3i", ExactSpelling = true)]
            extern static int __MR_Box3i_getDistanceSq_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pt);
            return __MR_Box3i_getDistanceSq_MR_Vector3i(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3i::getProjection`.
        public unsafe MR.Vector3i GetProjection(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getProjection", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_getProjection(_Underlying *_this, MR.Const_Vector3i._Underlying *pt);
            return __MR_Box3i_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3i::expanded`.
        public unsafe MR.Box3i Expanded(MR.Const_Vector3i expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_expanded", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_expanded(_Underlying *_this, MR.Const_Vector3i._Underlying *expansion);
            return __MR_Box3i_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3i::insignificantlyExpanded`.
        public unsafe MR.Box3i InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box3i_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box3i::operator==`.
        public static unsafe bool operator==(MR.Const_Box3i _this, MR.Const_Box3i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3i(MR.Const_Box3i._Underlying *_this, MR.Const_Box3i._Underlying *a);
            return __MR_equal_MR_Box3i(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box3i _this, MR.Const_Box3i a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box3i? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box3i)
                return this == (MR.Const_Box3i)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3i`.
    /// This is the non-const reference to the struct.
    public class Mut_Box3i : Const_Box3i
    {
        /// Get the underlying struct.
        public unsafe new ref Box3i UnderlyingStruct => ref *(Box3i *)_UnderlyingPtr;

        internal unsafe Mut_Box3i(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector3i Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector3i Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box3i(Const_Box3i _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box3i() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3i _ctor_result = __MR_Box3i_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Box3i::Box3i`.
        public unsafe Mut_Box3i(MR.Const_Vector3i min, MR.Const_Vector3i max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Construct_2", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_Construct_2(MR.Const_Vector3i._Underlying *min, MR.Const_Vector3i._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3i _ctor_result = __MR_Box3i_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3i::Box3i`.
        public unsafe Mut_Box3i(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Construct_1", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3i _ctor_result = __MR_Box3i_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from method `MR::Box3i::operator[]`.
        public unsafe new MR.Mut_Vector3i Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_Box3i_index(_Underlying *_this, int e);
            return new(__MR_Box3i_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3i::include`.
        public unsafe void Include(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_include_MR_Vector3i", ExactSpelling = true)]
            extern static void __MR_Box3i_include_MR_Vector3i(_Underlying *_this, MR.Const_Vector3i._Underlying *pt);
            __MR_Box3i_include_MR_Vector3i(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3i::include`.
        public unsafe void Include(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_include_MR_Box3i", ExactSpelling = true)]
            extern static void __MR_Box3i_include_MR_Box3i(_Underlying *_this, MR.Const_Box3i._Underlying *b);
            __MR_Box3i_include_MR_Box3i(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box3i::intersect`.
        public unsafe MR.Mut_Box3i Intersect(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3i._Underlying *__MR_Box3i_intersect(_Underlying *_this, MR.Const_Box3i._Underlying *b);
            return new(__MR_Box3i_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3i`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 24)]
    public struct Box3i
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box3i(Const_Box3i other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box3i(Box3i other) => new(new Mut_Box3i((Mut_Box3i._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3i Min;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public MR.Vector3i Max;

        /// Generated copy constructor.
        public Box3i(Box3i _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box3i()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_DefaultConstruct();
            this = __MR_Box3i_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box3i::Box3i`.
        public unsafe Box3i(MR.Const_Vector3i min, MR.Const_Vector3i max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Construct_2", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_Construct_2(MR.Const_Vector3i._Underlying *min, MR.Const_Vector3i._Underlying *max);
            this = __MR_Box3i_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3i::Box3i`.
        public unsafe Box3i(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_Construct_1", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box3i_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3i::operator[]`.
        public unsafe MR.Const_Vector3i Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i._Underlying *__MR_Box3i_index_const(MR.Box3i *_this, int e);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return new(__MR_Box3i_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3i::operator[]`.
        public unsafe MR.Mut_Vector3i Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i._Underlying *__MR_Box3i_index(MR.Box3i *_this, int e);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return new(__MR_Box3i_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3i::fromMinAndSize`.
        public static unsafe MR.Box3i FromMinAndSize(MR.Const_Vector3i min, MR.Const_Vector3i size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_fromMinAndSize(MR.Const_Vector3i._Underlying *min, MR.Const_Vector3i._Underlying *size);
            return __MR_Box3i_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3i::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_valid", ExactSpelling = true)]
            extern static byte __MR_Box3i_valid(MR.Box3i *_this);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box3i::center`.
        public unsafe MR.Vector3i Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_center", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_center(MR.Box3i *_this);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3i::corner`.
        public unsafe MR.Vector3i Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_corner", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_corner(MR.Box3i *_this, MR.Const_Vector3b._Underlying *c);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3i::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i_getMinBoxCorner(MR.Const_Vector3i._Underlying *n);
            return __MR_Box3i_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3i::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3i n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i_getMaxBoxCorner(MR.Const_Vector3i._Underlying *n);
            return __MR_Box3i_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3i::size`.
        public unsafe MR.Vector3i Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_size", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_size(MR.Box3i *_this);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3i::diagonal`.
        public unsafe int Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_diagonal", ExactSpelling = true)]
            extern static int __MR_Box3i_diagonal(MR.Box3i *_this);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3i::volume`.
        public unsafe int Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_volume", ExactSpelling = true)]
            extern static int __MR_Box3i_volume(MR.Box3i *_this);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3i::include`.
        public unsafe void Include(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_include_MR_Vector3i", ExactSpelling = true)]
            extern static void __MR_Box3i_include_MR_Vector3i(MR.Box3i *_this, MR.Const_Vector3i._Underlying *pt);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                __MR_Box3i_include_MR_Vector3i(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3i::include`.
        public unsafe void Include(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_include_MR_Box3i", ExactSpelling = true)]
            extern static void __MR_Box3i_include_MR_Box3i(MR.Box3i *_this, MR.Const_Box3i._Underlying *b);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                __MR_Box3i_include_MR_Box3i(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3i::contains`.
        public unsafe bool Contains(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_contains_MR_Vector3i", ExactSpelling = true)]
            extern static byte __MR_Box3i_contains_MR_Vector3i(MR.Box3i *_this, MR.Const_Vector3i._Underlying *pt);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_contains_MR_Vector3i(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3i::contains`.
        public unsafe bool Contains(MR.Const_Box3i otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_contains_MR_Box3i", ExactSpelling = true)]
            extern static byte __MR_Box3i_contains_MR_Box3i(MR.Box3i *_this, MR.Const_Box3i._Underlying *otherbox);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_contains_MR_Box3i(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3i::getBoxClosestPointTo`.
        public unsafe MR.Vector3i GetBoxClosestPointTo(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_getBoxClosestPointTo(MR.Box3i *_this, MR.Const_Vector3i._Underlying *pt);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3i::intersects`.
        public unsafe bool Intersects(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3i_intersects(MR.Box3i *_this, MR.Const_Box3i._Underlying *b);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3i::intersection`.
        public unsafe MR.Box3i Intersection(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_intersection", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_intersection(MR.Box3i *_this, MR.Const_Box3i._Underlying *b);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box3i::intersect`.
        public unsafe MR.Mut_Box3i Intersect(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3i._Underlying *__MR_Box3i_intersect(MR.Box3i *_this, MR.Const_Box3i._Underlying *b);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return new(__MR_Box3i_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Box3i b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getDistanceSq_MR_Box3i", ExactSpelling = true)]
            extern static int __MR_Box3i_getDistanceSq_MR_Box3i(MR.Box3i *_this, MR.Const_Box3i._Underlying *b);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_getDistanceSq_MR_Box3i(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3i::getDistanceSq`.
        public unsafe int GetDistanceSq(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getDistanceSq_MR_Vector3i", ExactSpelling = true)]
            extern static int __MR_Box3i_getDistanceSq_MR_Vector3i(MR.Box3i *_this, MR.Const_Vector3i._Underlying *pt);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_getDistanceSq_MR_Vector3i(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3i::getProjection`.
        public unsafe MR.Vector3i GetProjection(MR.Const_Vector3i pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_getProjection", ExactSpelling = true)]
            extern static MR.Vector3i __MR_Box3i_getProjection(MR.Box3i *_this, MR.Const_Vector3i._Underlying *pt);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3i::expanded`.
        public unsafe MR.Box3i Expanded(MR.Const_Vector3i expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_expanded", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_expanded(MR.Box3i *_this, MR.Const_Vector3i._Underlying *expansion);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3i::insignificantlyExpanded`.
        public unsafe MR.Box3i InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3i __MR_Box3i_insignificantlyExpanded(MR.Box3i *_this);
            fixed (MR.Box3i *__ptr__this = &this)
            {
                return __MR_Box3i_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box3i::operator==`.
        public static unsafe bool operator==(MR.Box3i _this, MR.Box3i a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3i", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3i(MR.Const_Box3i._Underlying *_this, MR.Const_Box3i._Underlying *a);
            return __MR_equal_MR_Box3i((MR.Mut_Box3i._Underlying *)&_this, (MR.Mut_Box3i._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box3i _this, MR.Box3i a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box3i a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box3i)
                return this == (MR.Box3i)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box3i` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box3i`/`Const_Box3i` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box3i
    {
        public readonly bool HasValue;
        internal readonly Box3i Object;
        public Box3i Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box3i() {HasValue = false;}
        public _InOpt_Box3i(Box3i new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box3i(Box3i new_value) {return new(new_value);}
        public _InOpt_Box3i(Const_Box3i new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box3i(Const_Box3i new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box3i` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3i`/`Const_Box3i` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box3i`.
    public class _InOptMut_Box3i
    {
        public Mut_Box3i? Opt;

        public _InOptMut_Box3i() {}
        public _InOptMut_Box3i(Mut_Box3i value) {Opt = value;}
        public static implicit operator _InOptMut_Box3i(Mut_Box3i value) {return new(value);}
        public unsafe _InOptMut_Box3i(ref Box3i value)
        {
            fixed (Box3i *value_ptr = &value)
            {
                Opt = new((Const_Box3i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box3i` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box3i`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3i`/`Const_Box3i` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box3i`.
    public class _InOptConst_Box3i
    {
        public Const_Box3i? Opt;

        public _InOptConst_Box3i() {}
        public _InOptConst_Box3i(Const_Box3i value) {Opt = value;}
        public static implicit operator _InOptConst_Box3i(Const_Box3i value) {return new(value);}
        public unsafe _InOptConst_Box3i(ref readonly Box3i value)
        {
            fixed (Box3i *value_ptr = &value)
            {
                Opt = new((Const_Box3i._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3i64`.
    /// This is the const reference to the struct.
    public class Const_Box3i64 : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box3i64>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box3i64 UnderlyingStruct => ref *(Box3i64 *)_UnderlyingPtr;

        internal unsafe Const_Box3i64(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Destroy", ExactSpelling = true)]
            extern static void __MR_Box3i64_Destroy(_Underlying *_this);
            __MR_Box3i64_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box3i64() {Dispose(false);}

        public ref readonly MR.Vector3i64 Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector3i64 Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box3i64_Get_elements();
                return *__MR_Box3i64_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box3i64(Const_Box3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3i64 _ctor_result = __MR_Box3i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::Box3i64::Box3i64`.
        public unsafe Const_Box3i64(MR.Const_Vector3i64 min, MR.Const_Vector3i64 max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_Construct_2(MR.Const_Vector3i64._Underlying *min, MR.Const_Vector3i64._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3i64 _ctor_result = __MR_Box3i64_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3i64::Box3i64`.
        public unsafe Const_Box3i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3i64 _ctor_result = __MR_Box3i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3i64::operator[]`.
        public unsafe MR.Const_Vector3i64 Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i64._Underlying *__MR_Box3i64_index_const(_Underlying *_this, int e);
            return new(__MR_Box3i64_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box3i64::fromMinAndSize`.
        public static unsafe MR.Box3i64 FromMinAndSize(MR.Const_Vector3i64 min, MR.Const_Vector3i64 size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_fromMinAndSize(MR.Const_Vector3i64._Underlying *min, MR.Const_Vector3i64._Underlying *size);
            return __MR_Box3i64_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3i64::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_valid", ExactSpelling = true)]
            extern static byte __MR_Box3i64_valid(_Underlying *_this);
            return __MR_Box3i64_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box3i64::center`.
        public unsafe MR.Vector3i64 Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_center", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_center(_Underlying *_this);
            return __MR_Box3i64_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3i64::corner`.
        public unsafe MR.Vector3i64 Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_corner", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_corner(_Underlying *_this, MR.Const_Vector3b._Underlying *c);
            return __MR_Box3i64_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3i64::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i64_getMinBoxCorner(MR.Const_Vector3i64._Underlying *n);
            return __MR_Box3i64_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3i64::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i64_getMaxBoxCorner(MR.Const_Vector3i64._Underlying *n);
            return __MR_Box3i64_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3i64::size`.
        public unsafe MR.Vector3i64 Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_size", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_size(_Underlying *_this);
            return __MR_Box3i64_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3i64::diagonal`.
        public unsafe long Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_diagonal", ExactSpelling = true)]
            extern static long __MR_Box3i64_diagonal(_Underlying *_this);
            return __MR_Box3i64_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3i64::volume`.
        public unsafe long Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_volume", ExactSpelling = true)]
            extern static long __MR_Box3i64_volume(_Underlying *_this);
            return __MR_Box3i64_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3i64::contains`.
        public unsafe bool Contains(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_contains_MR_Vector3i64", ExactSpelling = true)]
            extern static byte __MR_Box3i64_contains_MR_Vector3i64(_Underlying *_this, MR.Const_Vector3i64._Underlying *pt);
            return __MR_Box3i64_contains_MR_Vector3i64(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3i64::contains`.
        public unsafe bool Contains(MR.Const_Box3i64 otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_contains_MR_Box3i64", ExactSpelling = true)]
            extern static byte __MR_Box3i64_contains_MR_Box3i64(_Underlying *_this, MR.Const_Box3i64._Underlying *otherbox);
            return __MR_Box3i64_contains_MR_Box3i64(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3i64::getBoxClosestPointTo`.
        public unsafe MR.Vector3i64 GetBoxClosestPointTo(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector3i64._Underlying *pt);
            return __MR_Box3i64_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3i64::intersects`.
        public unsafe bool Intersects(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3i64_intersects(_Underlying *_this, MR.Const_Box3i64._Underlying *b);
            return __MR_Box3i64_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3i64::intersection`.
        public unsafe MR.Box3i64 Intersection(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_intersection", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_intersection(_Underlying *_this, MR.Const_Box3i64._Underlying *b);
            return __MR_Box3i64_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getDistanceSq_MR_Box3i64", ExactSpelling = true)]
            extern static long __MR_Box3i64_getDistanceSq_MR_Box3i64(_Underlying *_this, MR.Const_Box3i64._Underlying *b);
            return __MR_Box3i64_getDistanceSq_MR_Box3i64(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getDistanceSq_MR_Vector3i64", ExactSpelling = true)]
            extern static long __MR_Box3i64_getDistanceSq_MR_Vector3i64(_Underlying *_this, MR.Const_Vector3i64._Underlying *pt);
            return __MR_Box3i64_getDistanceSq_MR_Vector3i64(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3i64::getProjection`.
        public unsafe MR.Vector3i64 GetProjection(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getProjection", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_getProjection(_Underlying *_this, MR.Const_Vector3i64._Underlying *pt);
            return __MR_Box3i64_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3i64::expanded`.
        public unsafe MR.Box3i64 Expanded(MR.Const_Vector3i64 expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_expanded", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_expanded(_Underlying *_this, MR.Const_Vector3i64._Underlying *expansion);
            return __MR_Box3i64_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3i64::insignificantlyExpanded`.
        public unsafe MR.Box3i64 InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box3i64_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box3i64::operator==`.
        public static unsafe bool operator==(MR.Const_Box3i64 _this, MR.Const_Box3i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3i64(MR.Const_Box3i64._Underlying *_this, MR.Const_Box3i64._Underlying *a);
            return __MR_equal_MR_Box3i64(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box3i64 _this, MR.Const_Box3i64 a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box3i64? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box3i64)
                return this == (MR.Const_Box3i64)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3i64`.
    /// This is the non-const reference to the struct.
    public class Mut_Box3i64 : Const_Box3i64
    {
        /// Get the underlying struct.
        public unsafe new ref Box3i64 UnderlyingStruct => ref *(Box3i64 *)_UnderlyingPtr;

        internal unsafe Mut_Box3i64(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector3i64 Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector3i64 Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box3i64(Const_Box3i64 _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box3i64() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3i64 _ctor_result = __MR_Box3i64_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::Box3i64::Box3i64`.
        public unsafe Mut_Box3i64(MR.Const_Vector3i64 min, MR.Const_Vector3i64 max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_Construct_2(MR.Const_Vector3i64._Underlying *min, MR.Const_Vector3i64._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3i64 _ctor_result = __MR_Box3i64_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3i64::Box3i64`.
        public unsafe Mut_Box3i64(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3i64 _ctor_result = __MR_Box3i64_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from method `MR::Box3i64::operator[]`.
        public unsafe new MR.Mut_Vector3i64 Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_Box3i64_index(_Underlying *_this, int e);
            return new(__MR_Box3i64_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3i64::include`.
        public unsafe void Include(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_include_MR_Vector3i64", ExactSpelling = true)]
            extern static void __MR_Box3i64_include_MR_Vector3i64(_Underlying *_this, MR.Const_Vector3i64._Underlying *pt);
            __MR_Box3i64_include_MR_Vector3i64(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3i64::include`.
        public unsafe void Include(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_include_MR_Box3i64", ExactSpelling = true)]
            extern static void __MR_Box3i64_include_MR_Box3i64(_Underlying *_this, MR.Const_Box3i64._Underlying *b);
            __MR_Box3i64_include_MR_Box3i64(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box3i64::intersect`.
        public unsafe MR.Mut_Box3i64 Intersect(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3i64._Underlying *__MR_Box3i64_intersect(_Underlying *_this, MR.Const_Box3i64._Underlying *b);
            return new(__MR_Box3i64_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3i64`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 48)]
    public struct Box3i64
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box3i64(Const_Box3i64 other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box3i64(Box3i64 other) => new(new Mut_Box3i64((Mut_Box3i64._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3i64 Min;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public MR.Vector3i64 Max;

        /// Generated copy constructor.
        public Box3i64(Box3i64 _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box3i64()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_DefaultConstruct();
            this = __MR_Box3i64_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box3i64::Box3i64`.
        public unsafe Box3i64(MR.Const_Vector3i64 min, MR.Const_Vector3i64 max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Construct_2", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_Construct_2(MR.Const_Vector3i64._Underlying *min, MR.Const_Vector3i64._Underlying *max);
            this = __MR_Box3i64_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3i64::Box3i64`.
        public unsafe Box3i64(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_Construct_1", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box3i64_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3i64::operator[]`.
        public unsafe MR.Const_Vector3i64 Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3i64._Underlying *__MR_Box3i64_index_const(MR.Box3i64 *_this, int e);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return new(__MR_Box3i64_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3i64::operator[]`.
        public unsafe MR.Mut_Vector3i64 Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3i64._Underlying *__MR_Box3i64_index(MR.Box3i64 *_this, int e);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return new(__MR_Box3i64_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3i64::fromMinAndSize`.
        public static unsafe MR.Box3i64 FromMinAndSize(MR.Const_Vector3i64 min, MR.Const_Vector3i64 size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_fromMinAndSize(MR.Const_Vector3i64._Underlying *min, MR.Const_Vector3i64._Underlying *size);
            return __MR_Box3i64_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3i64::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_valid", ExactSpelling = true)]
            extern static byte __MR_Box3i64_valid(MR.Box3i64 *_this);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box3i64::center`.
        public unsafe MR.Vector3i64 Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_center", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_center(MR.Box3i64 *_this);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3i64::corner`.
        public unsafe MR.Vector3i64 Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_corner", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_corner(MR.Box3i64 *_this, MR.Const_Vector3b._Underlying *c);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3i64::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i64_getMinBoxCorner(MR.Const_Vector3i64._Underlying *n);
            return __MR_Box3i64_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3i64::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3i64 n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3i64_getMaxBoxCorner(MR.Const_Vector3i64._Underlying *n);
            return __MR_Box3i64_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3i64::size`.
        public unsafe MR.Vector3i64 Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_size", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_size(MR.Box3i64 *_this);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3i64::diagonal`.
        public unsafe long Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_diagonal", ExactSpelling = true)]
            extern static long __MR_Box3i64_diagonal(MR.Box3i64 *_this);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3i64::volume`.
        public unsafe long Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_volume", ExactSpelling = true)]
            extern static long __MR_Box3i64_volume(MR.Box3i64 *_this);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3i64::include`.
        public unsafe void Include(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_include_MR_Vector3i64", ExactSpelling = true)]
            extern static void __MR_Box3i64_include_MR_Vector3i64(MR.Box3i64 *_this, MR.Const_Vector3i64._Underlying *pt);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                __MR_Box3i64_include_MR_Vector3i64(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3i64::include`.
        public unsafe void Include(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_include_MR_Box3i64", ExactSpelling = true)]
            extern static void __MR_Box3i64_include_MR_Box3i64(MR.Box3i64 *_this, MR.Const_Box3i64._Underlying *b);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                __MR_Box3i64_include_MR_Box3i64(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3i64::contains`.
        public unsafe bool Contains(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_contains_MR_Vector3i64", ExactSpelling = true)]
            extern static byte __MR_Box3i64_contains_MR_Vector3i64(MR.Box3i64 *_this, MR.Const_Vector3i64._Underlying *pt);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_contains_MR_Vector3i64(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3i64::contains`.
        public unsafe bool Contains(MR.Const_Box3i64 otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_contains_MR_Box3i64", ExactSpelling = true)]
            extern static byte __MR_Box3i64_contains_MR_Box3i64(MR.Box3i64 *_this, MR.Const_Box3i64._Underlying *otherbox);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_contains_MR_Box3i64(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3i64::getBoxClosestPointTo`.
        public unsafe MR.Vector3i64 GetBoxClosestPointTo(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_getBoxClosestPointTo(MR.Box3i64 *_this, MR.Const_Vector3i64._Underlying *pt);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3i64::intersects`.
        public unsafe bool Intersects(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3i64_intersects(MR.Box3i64 *_this, MR.Const_Box3i64._Underlying *b);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3i64::intersection`.
        public unsafe MR.Box3i64 Intersection(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_intersection", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_intersection(MR.Box3i64 *_this, MR.Const_Box3i64._Underlying *b);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box3i64::intersect`.
        public unsafe MR.Mut_Box3i64 Intersect(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3i64._Underlying *__MR_Box3i64_intersect(MR.Box3i64 *_this, MR.Const_Box3i64._Underlying *b);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return new(__MR_Box3i64_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Box3i64 b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getDistanceSq_MR_Box3i64", ExactSpelling = true)]
            extern static long __MR_Box3i64_getDistanceSq_MR_Box3i64(MR.Box3i64 *_this, MR.Const_Box3i64._Underlying *b);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_getDistanceSq_MR_Box3i64(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3i64::getDistanceSq`.
        public unsafe long GetDistanceSq(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getDistanceSq_MR_Vector3i64", ExactSpelling = true)]
            extern static long __MR_Box3i64_getDistanceSq_MR_Vector3i64(MR.Box3i64 *_this, MR.Const_Vector3i64._Underlying *pt);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_getDistanceSq_MR_Vector3i64(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3i64::getProjection`.
        public unsafe MR.Vector3i64 GetProjection(MR.Const_Vector3i64 pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_getProjection", ExactSpelling = true)]
            extern static MR.Vector3i64 __MR_Box3i64_getProjection(MR.Box3i64 *_this, MR.Const_Vector3i64._Underlying *pt);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3i64::expanded`.
        public unsafe MR.Box3i64 Expanded(MR.Const_Vector3i64 expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_expanded", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_expanded(MR.Box3i64 *_this, MR.Const_Vector3i64._Underlying *expansion);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3i64::insignificantlyExpanded`.
        public unsafe MR.Box3i64 InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3i64_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3i64 __MR_Box3i64_insignificantlyExpanded(MR.Box3i64 *_this);
            fixed (MR.Box3i64 *__ptr__this = &this)
            {
                return __MR_Box3i64_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box3i64::operator==`.
        public static unsafe bool operator==(MR.Box3i64 _this, MR.Box3i64 a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3i64", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3i64(MR.Const_Box3i64._Underlying *_this, MR.Const_Box3i64._Underlying *a);
            return __MR_equal_MR_Box3i64((MR.Mut_Box3i64._Underlying *)&_this, (MR.Mut_Box3i64._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box3i64 _this, MR.Box3i64 a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box3i64 a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box3i64)
                return this == (MR.Box3i64)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box3i64` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box3i64`/`Const_Box3i64` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box3i64
    {
        public readonly bool HasValue;
        internal readonly Box3i64 Object;
        public Box3i64 Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box3i64() {HasValue = false;}
        public _InOpt_Box3i64(Box3i64 new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box3i64(Box3i64 new_value) {return new(new_value);}
        public _InOpt_Box3i64(Const_Box3i64 new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box3i64(Const_Box3i64 new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box3i64` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3i64`/`Const_Box3i64` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box3i64`.
    public class _InOptMut_Box3i64
    {
        public Mut_Box3i64? Opt;

        public _InOptMut_Box3i64() {}
        public _InOptMut_Box3i64(Mut_Box3i64 value) {Opt = value;}
        public static implicit operator _InOptMut_Box3i64(Mut_Box3i64 value) {return new(value);}
        public unsafe _InOptMut_Box3i64(ref Box3i64 value)
        {
            fixed (Box3i64 *value_ptr = &value)
            {
                Opt = new((Const_Box3i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box3i64` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box3i64`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3i64`/`Const_Box3i64` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box3i64`.
    public class _InOptConst_Box3i64
    {
        public Const_Box3i64? Opt;

        public _InOptConst_Box3i64() {}
        public _InOptConst_Box3i64(Const_Box3i64 value) {Opt = value;}
        public static implicit operator _InOptConst_Box3i64(Const_Box3i64 value) {return new(value);}
        public unsafe _InOptConst_Box3i64(ref readonly Box3i64 value)
        {
            fixed (Box3i64 *value_ptr = &value)
            {
                Opt = new((Const_Box3i64._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3f`.
    /// This is the const reference to the struct.
    public class Const_Box3f : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box3f>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box3f UnderlyingStruct => ref *(Box3f *)_UnderlyingPtr;

        internal unsafe Const_Box3f(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Destroy", ExactSpelling = true)]
            extern static void __MR_Box3f_Destroy(_Underlying *_this);
            __MR_Box3f_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box3f() {Dispose(false);}

        public ref readonly MR.Vector3f Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector3f Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box3f_Get_elements();
                return *__MR_Box3f_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box3f(Const_Box3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3f _ctor_result = __MR_Box3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Box3f::Box3f`.
        public unsafe Const_Box3f(MR.Const_Vector3f min, MR.Const_Vector3f max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Construct_2", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_Construct_2(MR.Const_Vector3f._Underlying *min, MR.Const_Vector3f._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3f _ctor_result = __MR_Box3f_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3f::Box3f`.
        public unsafe Const_Box3f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Construct_1", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3f _ctor_result = __MR_Box3f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3f::operator[]`.
        public unsafe MR.Const_Vector3f Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Box3f_index_const(_Underlying *_this, int e);
            return new(__MR_Box3f_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box3f::fromMinAndSize`.
        public static unsafe MR.Box3f FromMinAndSize(MR.Const_Vector3f min, MR.Const_Vector3f size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_fromMinAndSize(MR.Const_Vector3f._Underlying *min, MR.Const_Vector3f._Underlying *size);
            return __MR_Box3f_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3f::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_valid", ExactSpelling = true)]
            extern static byte __MR_Box3f_valid(_Underlying *_this);
            return __MR_Box3f_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box3f::center`.
        public unsafe MR.Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_center", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_center(_Underlying *_this);
            return __MR_Box3f_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3f::corner`.
        public unsafe MR.Vector3f Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_corner", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_corner(_Underlying *_this, MR.Const_Vector3b._Underlying *c);
            return __MR_Box3f_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3f::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3f_getMinBoxCorner(MR.Const_Vector3f._Underlying *n);
            return __MR_Box3f_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3f::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3f_getMaxBoxCorner(MR.Const_Vector3f._Underlying *n);
            return __MR_Box3f_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3f::size`.
        public unsafe MR.Vector3f Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_size", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_size(_Underlying *_this);
            return __MR_Box3f_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3f::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_diagonal", ExactSpelling = true)]
            extern static float __MR_Box3f_diagonal(_Underlying *_this);
            return __MR_Box3f_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3f::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_volume", ExactSpelling = true)]
            extern static float __MR_Box3f_volume(_Underlying *_this);
            return __MR_Box3f_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3f::contains`.
        public unsafe bool Contains(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_contains_MR_Vector3f", ExactSpelling = true)]
            extern static byte __MR_Box3f_contains_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Box3f_contains_MR_Vector3f(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3f::contains`.
        public unsafe bool Contains(MR.Const_Box3f otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_contains_MR_Box3f", ExactSpelling = true)]
            extern static byte __MR_Box3f_contains_MR_Box3f(_Underlying *_this, MR.Const_Box3f._Underlying *otherbox);
            return __MR_Box3f_contains_MR_Box3f(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3f::getBoxClosestPointTo`.
        public unsafe MR.Vector3f GetBoxClosestPointTo(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Box3f_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3f::intersects`.
        public unsafe bool Intersects(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3f_intersects(_Underlying *_this, MR.Const_Box3f._Underlying *b);
            return __MR_Box3f_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3f::intersection`.
        public unsafe MR.Box3f Intersection(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_intersection", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_intersection(_Underlying *_this, MR.Const_Box3f._Underlying *b);
            return __MR_Box3f_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getDistanceSq_MR_Box3f", ExactSpelling = true)]
            extern static float __MR_Box3f_getDistanceSq_MR_Box3f(_Underlying *_this, MR.Const_Box3f._Underlying *b);
            return __MR_Box3f_getDistanceSq_MR_Box3f(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getDistanceSq_MR_Vector3f", ExactSpelling = true)]
            extern static float __MR_Box3f_getDistanceSq_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Box3f_getDistanceSq_MR_Vector3f(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3f::getProjection`.
        public unsafe MR.Vector3f GetProjection(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getProjection", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_getProjection(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            return __MR_Box3f_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3f::expanded`.
        public unsafe MR.Box3f Expanded(MR.Const_Vector3f expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_expanded", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_expanded(_Underlying *_this, MR.Const_Vector3f._Underlying *expansion);
            return __MR_Box3f_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3f::insignificantlyExpanded`.
        public unsafe MR.Box3f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box3f_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box3f::operator==`.
        public static unsafe bool operator==(MR.Const_Box3f _this, MR.Const_Box3f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3f(MR.Const_Box3f._Underlying *_this, MR.Const_Box3f._Underlying *a);
            return __MR_equal_MR_Box3f(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box3f _this, MR.Const_Box3f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box3f? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box3f)
                return this == (MR.Const_Box3f)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3f`.
    /// This is the non-const reference to the struct.
    public class Mut_Box3f : Const_Box3f
    {
        /// Get the underlying struct.
        public unsafe new ref Box3f UnderlyingStruct => ref *(Box3f *)_UnderlyingPtr;

        internal unsafe Mut_Box3f(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector3f Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector3f Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box3f(Const_Box3f _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 24);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box3f() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3f _ctor_result = __MR_Box3f_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from constructor `MR::Box3f::Box3f`.
        public unsafe Mut_Box3f(MR.Const_Vector3f min, MR.Const_Vector3f max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Construct_2", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_Construct_2(MR.Const_Vector3f._Underlying *min, MR.Const_Vector3f._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3f _ctor_result = __MR_Box3f_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3f::Box3f`.
        public unsafe Mut_Box3f(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Construct_1", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(24);
            MR.Box3f _ctor_result = __MR_Box3f_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 24);
        }

        /// Generated from method `MR::Box3f::operator[]`.
        public unsafe new MR.Mut_Vector3f Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Box3f_index(_Underlying *_this, int e);
            return new(__MR_Box3f_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3f::include`.
        public unsafe void Include(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_include_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_Box3f_include_MR_Vector3f(_Underlying *_this, MR.Const_Vector3f._Underlying *pt);
            __MR_Box3f_include_MR_Vector3f(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3f::include`.
        public unsafe void Include(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_include_MR_Box3f", ExactSpelling = true)]
            extern static void __MR_Box3f_include_MR_Box3f(_Underlying *_this, MR.Const_Box3f._Underlying *b);
            __MR_Box3f_include_MR_Box3f(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box3f::intersect`.
        public unsafe MR.Mut_Box3f Intersect(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3f._Underlying *__MR_Box3f_intersect(_Underlying *_this, MR.Const_Box3f._Underlying *b);
            return new(__MR_Box3f_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3f`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 24)]
    public struct Box3f
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box3f(Const_Box3f other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box3f(Box3f other) => new(new Mut_Box3f((Mut_Box3f._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3f Min;

        [System.Runtime.InteropServices.FieldOffset(12)]
        public MR.Vector3f Max;

        /// Generated copy constructor.
        public Box3f(Box3f _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box3f()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_DefaultConstruct();
            this = __MR_Box3f_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box3f::Box3f`.
        public unsafe Box3f(MR.Const_Vector3f min, MR.Const_Vector3f max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Construct_2", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_Construct_2(MR.Const_Vector3f._Underlying *min, MR.Const_Vector3f._Underlying *max);
            this = __MR_Box3f_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3f::Box3f`.
        public unsafe Box3f(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_Construct_1", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box3f_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3f::operator[]`.
        public unsafe MR.Const_Vector3f Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3f._Underlying *__MR_Box3f_index_const(MR.Box3f *_this, int e);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return new(__MR_Box3f_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3f::operator[]`.
        public unsafe MR.Mut_Vector3f Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3f._Underlying *__MR_Box3f_index(MR.Box3f *_this, int e);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return new(__MR_Box3f_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3f::fromMinAndSize`.
        public static unsafe MR.Box3f FromMinAndSize(MR.Const_Vector3f min, MR.Const_Vector3f size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_fromMinAndSize(MR.Const_Vector3f._Underlying *min, MR.Const_Vector3f._Underlying *size);
            return __MR_Box3f_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3f::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_valid", ExactSpelling = true)]
            extern static byte __MR_Box3f_valid(MR.Box3f *_this);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box3f::center`.
        public unsafe MR.Vector3f Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_center", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_center(MR.Box3f *_this);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3f::corner`.
        public unsafe MR.Vector3f Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_corner", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_corner(MR.Box3f *_this, MR.Const_Vector3b._Underlying *c);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3f::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3f_getMinBoxCorner(MR.Const_Vector3f._Underlying *n);
            return __MR_Box3f_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3f::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3f n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3f_getMaxBoxCorner(MR.Const_Vector3f._Underlying *n);
            return __MR_Box3f_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3f::size`.
        public unsafe MR.Vector3f Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_size", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_size(MR.Box3f *_this);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3f::diagonal`.
        public unsafe float Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_diagonal", ExactSpelling = true)]
            extern static float __MR_Box3f_diagonal(MR.Box3f *_this);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3f::volume`.
        public unsafe float Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_volume", ExactSpelling = true)]
            extern static float __MR_Box3f_volume(MR.Box3f *_this);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3f::include`.
        public unsafe void Include(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_include_MR_Vector3f", ExactSpelling = true)]
            extern static void __MR_Box3f_include_MR_Vector3f(MR.Box3f *_this, MR.Const_Vector3f._Underlying *pt);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                __MR_Box3f_include_MR_Vector3f(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3f::include`.
        public unsafe void Include(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_include_MR_Box3f", ExactSpelling = true)]
            extern static void __MR_Box3f_include_MR_Box3f(MR.Box3f *_this, MR.Const_Box3f._Underlying *b);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                __MR_Box3f_include_MR_Box3f(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3f::contains`.
        public unsafe bool Contains(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_contains_MR_Vector3f", ExactSpelling = true)]
            extern static byte __MR_Box3f_contains_MR_Vector3f(MR.Box3f *_this, MR.Const_Vector3f._Underlying *pt);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_contains_MR_Vector3f(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3f::contains`.
        public unsafe bool Contains(MR.Const_Box3f otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_contains_MR_Box3f", ExactSpelling = true)]
            extern static byte __MR_Box3f_contains_MR_Box3f(MR.Box3f *_this, MR.Const_Box3f._Underlying *otherbox);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_contains_MR_Box3f(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3f::getBoxClosestPointTo`.
        public unsafe MR.Vector3f GetBoxClosestPointTo(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_getBoxClosestPointTo(MR.Box3f *_this, MR.Const_Vector3f._Underlying *pt);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3f::intersects`.
        public unsafe bool Intersects(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3f_intersects(MR.Box3f *_this, MR.Const_Box3f._Underlying *b);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3f::intersection`.
        public unsafe MR.Box3f Intersection(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_intersection", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_intersection(MR.Box3f *_this, MR.Const_Box3f._Underlying *b);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box3f::intersect`.
        public unsafe MR.Mut_Box3f Intersect(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3f._Underlying *__MR_Box3f_intersect(MR.Box3f *_this, MR.Const_Box3f._Underlying *b);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return new(__MR_Box3f_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Box3f b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getDistanceSq_MR_Box3f", ExactSpelling = true)]
            extern static float __MR_Box3f_getDistanceSq_MR_Box3f(MR.Box3f *_this, MR.Const_Box3f._Underlying *b);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_getDistanceSq_MR_Box3f(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3f::getDistanceSq`.
        public unsafe float GetDistanceSq(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getDistanceSq_MR_Vector3f", ExactSpelling = true)]
            extern static float __MR_Box3f_getDistanceSq_MR_Vector3f(MR.Box3f *_this, MR.Const_Vector3f._Underlying *pt);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_getDistanceSq_MR_Vector3f(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3f::getProjection`.
        public unsafe MR.Vector3f GetProjection(MR.Const_Vector3f pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_getProjection", ExactSpelling = true)]
            extern static MR.Vector3f __MR_Box3f_getProjection(MR.Box3f *_this, MR.Const_Vector3f._Underlying *pt);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3f::expanded`.
        public unsafe MR.Box3f Expanded(MR.Const_Vector3f expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_expanded", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_expanded(MR.Box3f *_this, MR.Const_Vector3f._Underlying *expansion);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3f::insignificantlyExpanded`.
        public unsafe MR.Box3f InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3f_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3f __MR_Box3f_insignificantlyExpanded(MR.Box3f *_this);
            fixed (MR.Box3f *__ptr__this = &this)
            {
                return __MR_Box3f_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box3f::operator==`.
        public static unsafe bool operator==(MR.Box3f _this, MR.Box3f a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3f", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3f(MR.Const_Box3f._Underlying *_this, MR.Const_Box3f._Underlying *a);
            return __MR_equal_MR_Box3f((MR.Mut_Box3f._Underlying *)&_this, (MR.Mut_Box3f._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box3f _this, MR.Box3f a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box3f a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box3f)
                return this == (MR.Box3f)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box3f` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box3f`/`Const_Box3f` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box3f
    {
        public readonly bool HasValue;
        internal readonly Box3f Object;
        public Box3f Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box3f() {HasValue = false;}
        public _InOpt_Box3f(Box3f new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box3f(Box3f new_value) {return new(new_value);}
        public _InOpt_Box3f(Const_Box3f new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box3f(Const_Box3f new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box3f` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3f`/`Const_Box3f` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box3f`.
    public class _InOptMut_Box3f
    {
        public Mut_Box3f? Opt;

        public _InOptMut_Box3f() {}
        public _InOptMut_Box3f(Mut_Box3f value) {Opt = value;}
        public static implicit operator _InOptMut_Box3f(Mut_Box3f value) {return new(value);}
        public unsafe _InOptMut_Box3f(ref Box3f value)
        {
            fixed (Box3f *value_ptr = &value)
            {
                Opt = new((Const_Box3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box3f` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box3f`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3f`/`Const_Box3f` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box3f`.
    public class _InOptConst_Box3f
    {
        public Const_Box3f? Opt;

        public _InOptConst_Box3f() {}
        public _InOptConst_Box3f(Const_Box3f value) {Opt = value;}
        public static implicit operator _InOptConst_Box3f(Const_Box3f value) {return new(value);}
        public unsafe _InOptConst_Box3f(ref readonly Box3f value)
        {
            fixed (Box3f *value_ptr = &value)
            {
                Opt = new((Const_Box3f._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3d`.
    /// This is the const reference to the struct.
    public class Const_Box3d : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box3d>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        /// Get the underlying struct.
        public unsafe ref readonly Box3d UnderlyingStruct => ref *(Box3d *)_UnderlyingPtr;

        internal unsafe Const_Box3d(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Destroy", ExactSpelling = true)]
            extern static void __MR_Box3d_Destroy(_Underlying *_this);
            __MR_Box3d_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box3d() {Dispose(false);}

        public ref readonly MR.Vector3d Min => ref UnderlyingStruct.Min;

        public ref readonly MR.Vector3d Max => ref UnderlyingStruct.Max;

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box3d_Get_elements();
                return *__MR_Box3d_Get_elements();
            }
        }

        /// Generated copy constructor.
        public unsafe Const_Box3d(Const_Box3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3d _ctor_result = __MR_Box3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::Box3d::Box3d`.
        public unsafe Const_Box3d(MR.Const_Vector3d min, MR.Const_Vector3d max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Construct_2", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_Construct_2(MR.Const_Vector3d._Underlying *min, MR.Const_Vector3d._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3d _ctor_result = __MR_Box3d_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3d::Box3d`.
        public unsafe Const_Box3d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Construct_1", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3d _ctor_result = __MR_Box3d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3d::operator[]`.
        public unsafe MR.Const_Vector3d Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Box3d_index_const(_Underlying *_this, int e);
            return new(__MR_Box3d_index_const(_UnderlyingPtr, e), is_owning: false);
        }

        /// Generated from method `MR::Box3d::fromMinAndSize`.
        public static unsafe MR.Box3d FromMinAndSize(MR.Const_Vector3d min, MR.Const_Vector3d size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_fromMinAndSize(MR.Const_Vector3d._Underlying *min, MR.Const_Vector3d._Underlying *size);
            return __MR_Box3d_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3d::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_valid", ExactSpelling = true)]
            extern static byte __MR_Box3d_valid(_Underlying *_this);
            return __MR_Box3d_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box3d::center`.
        public unsafe MR.Vector3d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_center", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_center(_Underlying *_this);
            return __MR_Box3d_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3d::corner`.
        public unsafe MR.Vector3d Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_corner", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_corner(_Underlying *_this, MR.Const_Vector3b._Underlying *c);
            return __MR_Box3d_corner(_UnderlyingPtr, c._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3d::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3d_getMinBoxCorner(MR.Const_Vector3d._Underlying *n);
            return __MR_Box3d_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3d::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3d_getMaxBoxCorner(MR.Const_Vector3d._Underlying *n);
            return __MR_Box3d_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3d::size`.
        public unsafe MR.Vector3d Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_size", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_size(_Underlying *_this);
            return __MR_Box3d_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3d::diagonal`.
        public unsafe double Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_diagonal", ExactSpelling = true)]
            extern static double __MR_Box3d_diagonal(_Underlying *_this);
            return __MR_Box3d_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3d::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_volume", ExactSpelling = true)]
            extern static double __MR_Box3d_volume(_Underlying *_this);
            return __MR_Box3d_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3d::contains`.
        public unsafe bool Contains(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_contains_MR_Vector3d", ExactSpelling = true)]
            extern static byte __MR_Box3d_contains_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            return __MR_Box3d_contains_MR_Vector3d(_UnderlyingPtr, pt._UnderlyingPtr) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3d::contains`.
        public unsafe bool Contains(MR.Const_Box3d otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_contains_MR_Box3d", ExactSpelling = true)]
            extern static byte __MR_Box3d_contains_MR_Box3d(_Underlying *_this, MR.Const_Box3d._Underlying *otherbox);
            return __MR_Box3d_contains_MR_Box3d(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3d::getBoxClosestPointTo`.
        public unsafe MR.Vector3d GetBoxClosestPointTo(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_getBoxClosestPointTo(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            return __MR_Box3d_getBoxClosestPointTo(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3d::intersects`.
        public unsafe bool Intersects(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3d_intersects(_Underlying *_this, MR.Const_Box3d._Underlying *b);
            return __MR_Box3d_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3d::intersection`.
        public unsafe MR.Box3d Intersection(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_intersection", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_intersection(_Underlying *_this, MR.Const_Box3d._Underlying *b);
            return __MR_Box3d_intersection(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getDistanceSq_MR_Box3d", ExactSpelling = true)]
            extern static double __MR_Box3d_getDistanceSq_MR_Box3d(_Underlying *_this, MR.Const_Box3d._Underlying *b);
            return __MR_Box3d_getDistanceSq_MR_Box3d(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getDistanceSq_MR_Vector3d", ExactSpelling = true)]
            extern static double __MR_Box3d_getDistanceSq_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            return __MR_Box3d_getDistanceSq_MR_Vector3d(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3d::getProjection`.
        public unsafe MR.Vector3d GetProjection(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getProjection", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_getProjection(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            return __MR_Box3d_getProjection(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3d::expanded`.
        public unsafe MR.Box3d Expanded(MR.Const_Vector3d expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_expanded", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_expanded(_Underlying *_this, MR.Const_Vector3d._Underlying *expansion);
            return __MR_Box3d_expanded(_UnderlyingPtr, expansion._UnderlyingPtr);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3d::insignificantlyExpanded`.
        public unsafe MR.Box3d InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_insignificantlyExpanded(_Underlying *_this);
            return __MR_Box3d_insignificantlyExpanded(_UnderlyingPtr);
        }

        /// Generated from method `MR::Box3d::operator==`.
        public static unsafe bool operator==(MR.Const_Box3d _this, MR.Const_Box3d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3d(MR.Const_Box3d._Underlying *_this, MR.Const_Box3d._Underlying *a);
            return __MR_equal_MR_Box3d(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box3d _this, MR.Const_Box3d a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box3d? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box3d)
                return this == (MR.Const_Box3d)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3d`.
    /// This is the non-const reference to the struct.
    public class Mut_Box3d : Const_Box3d
    {
        /// Get the underlying struct.
        public unsafe new ref Box3d UnderlyingStruct => ref *(Box3d *)_UnderlyingPtr;

        internal unsafe Mut_Box3d(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new ref MR.Vector3d Min => ref UnderlyingStruct.Min;

        public new ref MR.Vector3d Max => ref UnderlyingStruct.Max;

        /// Generated copy constructor.
        public unsafe Mut_Box3d(Const_Box3d _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            System.Runtime.InteropServices.NativeMemory.Copy(_other._UnderlyingPtr, _UnderlyingPtr, 48);
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Mut_Box3d() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_DefaultConstruct();
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3d _ctor_result = __MR_Box3d_DefaultConstruct();
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from constructor `MR::Box3d::Box3d`.
        public unsafe Mut_Box3d(MR.Const_Vector3d min, MR.Const_Vector3d max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Construct_2", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_Construct_2(MR.Const_Vector3d._Underlying *min, MR.Const_Vector3d._Underlying *max);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3d _ctor_result = __MR_Box3d_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3d::Box3d`.
        public unsafe Mut_Box3d(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Construct_1", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_Construct_1(MR.NoInit._Underlying *_1);
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Alloc", ExactSpelling = true)]
            extern static _Underlying *__MR_Alloc(nuint size);
            _UnderlyingPtr = __MR_Alloc(48);
            MR.Box3d _ctor_result = __MR_Box3d_Construct_1(_1._UnderlyingPtr);
            System.Runtime.InteropServices.NativeMemory.Copy(&_ctor_result, _UnderlyingPtr, 48);
        }

        /// Generated from method `MR::Box3d::operator[]`.
        public unsafe new MR.Mut_Vector3d Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Box3d_index(_Underlying *_this, int e);
            return new(__MR_Box3d_index(_UnderlyingPtr, e), is_owning: false);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3d::include`.
        public unsafe void Include(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_include_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_Box3d_include_MR_Vector3d(_Underlying *_this, MR.Const_Vector3d._Underlying *pt);
            __MR_Box3d_include_MR_Vector3d(_UnderlyingPtr, pt._UnderlyingPtr);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3d::include`.
        public unsafe void Include(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_include_MR_Box3d", ExactSpelling = true)]
            extern static void __MR_Box3d_include_MR_Box3d(_Underlying *_this, MR.Const_Box3d._Underlying *b);
            __MR_Box3d_include_MR_Box3d(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box3d::intersect`.
        public unsafe MR.Mut_Box3d Intersect(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3d._Underlying *__MR_Box3d_intersect(_Underlying *_this, MR.Const_Box3d._Underlying *b);
            return new(__MR_Box3d_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box3d`.
    /// This is the by-value version of the struct.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 48)]
    public struct Box3d
    {
        /// Copy contents from a wrapper class to this struct.
        public static implicit operator Box3d(Const_Box3d other) => other.UnderlyingStruct;
        /// Copy this struct into a wrapper class. (Even though we initially pass `is_owning: false`, we then use the copy constructor to produce an owning instance.)
        public unsafe static implicit operator Mut_Box3d(Box3d other) => new(new Mut_Box3d((Mut_Box3d._Underlying *)&other, is_owning: false));

        [System.Runtime.InteropServices.FieldOffset(0)]
        public MR.Vector3d Min;

        [System.Runtime.InteropServices.FieldOffset(24)]
        public MR.Vector3d Max;

        /// Generated copy constructor.
        public Box3d(Box3d _other) {this = _other;}

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box3d()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_DefaultConstruct();
            this = __MR_Box3d_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box3d::Box3d`.
        public unsafe Box3d(MR.Const_Vector3d min, MR.Const_Vector3d max)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Construct_2", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_Construct_2(MR.Const_Vector3d._Underlying *min, MR.Const_Vector3d._Underlying *max);
            this = __MR_Box3d_Construct_2(min._UnderlyingPtr, max._UnderlyingPtr);
        }

        // If the compiler supports `requires`, use that instead of `std::enable_if` here.
        // Not (only) because it looks cooler, but because of a bug in our binding generator that makes it choke on it: https://github.com/MeshInspector/mrbind/issues/19
        /// Generated from constructor `MR::Box3d::Box3d`.
        public unsafe Box3d(MR.Const_NoInit _1)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_Construct_1", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_Construct_1(MR.NoInit._Underlying *_1);
            this = __MR_Box3d_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box3d::operator[]`.
        public unsafe MR.Const_Vector3d Index_Const(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_index_const", ExactSpelling = true)]
            extern static MR.Const_Vector3d._Underlying *__MR_Box3d_index_const(MR.Box3d *_this, int e);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return new(__MR_Box3d_index_const(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3d::operator[]`.
        public unsafe MR.Mut_Vector3d Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_index", ExactSpelling = true)]
            extern static MR.Mut_Vector3d._Underlying *__MR_Box3d_index(MR.Box3d *_this, int e);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return new(__MR_Box3d_index(__ptr__this, e), is_owning: false);
            }
        }

        /// Generated from method `MR::Box3d::fromMinAndSize`.
        public static unsafe MR.Box3d FromMinAndSize(MR.Const_Vector3d min, MR.Const_Vector3d size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_fromMinAndSize(MR.Const_Vector3d._Underlying *min, MR.Const_Vector3d._Underlying *size);
            return __MR_Box3d_fromMinAndSize(min._UnderlyingPtr, size._UnderlyingPtr);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box3d::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_valid", ExactSpelling = true)]
            extern static byte __MR_Box3d_valid(MR.Box3d *_this);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_valid(__ptr__this) != 0;
            }
        }

        /// computes center of the box
        /// Generated from method `MR::Box3d::center`.
        public unsafe MR.Vector3d Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_center", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_center(MR.Box3d *_this);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_center(__ptr__this);
            }
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box3d::corner`.
        public unsafe MR.Vector3d Corner(MR.Const_Vector3b c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_corner", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_corner(MR.Box3d *_this, MR.Const_Vector3b._Underlying *c);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_corner(__ptr__this, c._UnderlyingPtr);
            }
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box3d::getMinBoxCorner`.
        public static unsafe MR.Vector3b GetMinBoxCorner(MR.Const_Vector3d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getMinBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3d_getMinBoxCorner(MR.Const_Vector3d._Underlying *n);
            return __MR_Box3d_getMinBoxCorner(n._UnderlyingPtr);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box3d::getMaxBoxCorner`.
        public static unsafe MR.Vector3b GetMaxBoxCorner(MR.Const_Vector3d n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getMaxBoxCorner", ExactSpelling = true)]
            extern static MR.Vector3b __MR_Box3d_getMaxBoxCorner(MR.Const_Vector3d._Underlying *n);
            return __MR_Box3d_getMaxBoxCorner(n._UnderlyingPtr);
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box3d::size`.
        public unsafe MR.Vector3d Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_size", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_size(MR.Box3d *_this);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_size(__ptr__this);
            }
        }

        /// computes length from min to max
        /// Generated from method `MR::Box3d::diagonal`.
        public unsafe double Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_diagonal", ExactSpelling = true)]
            extern static double __MR_Box3d_diagonal(MR.Box3d *_this);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_diagonal(__ptr__this);
            }
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box3d::volume`.
        public unsafe double Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_volume", ExactSpelling = true)]
            extern static double __MR_Box3d_volume(MR.Box3d *_this);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_volume(__ptr__this);
            }
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box3d::include`.
        public unsafe void Include(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_include_MR_Vector3d", ExactSpelling = true)]
            extern static void __MR_Box3d_include_MR_Vector3d(MR.Box3d *_this, MR.Const_Vector3d._Underlying *pt);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                __MR_Box3d_include_MR_Vector3d(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box3d::include`.
        public unsafe void Include(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_include_MR_Box3d", ExactSpelling = true)]
            extern static void __MR_Box3d_include_MR_Box3d(MR.Box3d *_this, MR.Const_Box3d._Underlying *b);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                __MR_Box3d_include_MR_Box3d(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box3d::contains`.
        public unsafe bool Contains(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_contains_MR_Vector3d", ExactSpelling = true)]
            extern static byte __MR_Box3d_contains_MR_Vector3d(MR.Box3d *_this, MR.Const_Vector3d._Underlying *pt);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_contains_MR_Vector3d(__ptr__this, pt._UnderlyingPtr) != 0;
            }
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box3d::contains`.
        public unsafe bool Contains(MR.Const_Box3d otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_contains_MR_Box3d", ExactSpelling = true)]
            extern static byte __MR_Box3d_contains_MR_Box3d(MR.Box3d *_this, MR.Const_Box3d._Underlying *otherbox);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_contains_MR_Box3d(__ptr__this, otherbox._UnderlyingPtr) != 0;
            }
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box3d::getBoxClosestPointTo`.
        public unsafe MR.Vector3d GetBoxClosestPointTo(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getBoxClosestPointTo", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_getBoxClosestPointTo(MR.Box3d *_this, MR.Const_Vector3d._Underlying *pt);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_getBoxClosestPointTo(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box3d::intersects`.
        public unsafe bool Intersects(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_intersects", ExactSpelling = true)]
            extern static byte __MR_Box3d_intersects(MR.Box3d *_this, MR.Const_Box3d._Underlying *b);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_intersects(__ptr__this, b._UnderlyingPtr) != 0;
            }
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box3d::intersection`.
        public unsafe MR.Box3d Intersection(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_intersection", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_intersection(MR.Box3d *_this, MR.Const_Box3d._Underlying *b);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_intersection(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// Generated from method `MR::Box3d::intersect`.
        public unsafe MR.Mut_Box3d Intersect(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_intersect", ExactSpelling = true)]
            extern static MR.Mut_Box3d._Underlying *__MR_Box3d_intersect(MR.Box3d *_this, MR.Const_Box3d._Underlying *b);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return new(__MR_Box3d_intersect(__ptr__this, b._UnderlyingPtr), is_owning: false);
            }
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box3d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Box3d b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getDistanceSq_MR_Box3d", ExactSpelling = true)]
            extern static double __MR_Box3d_getDistanceSq_MR_Box3d(MR.Box3d *_this, MR.Const_Box3d._Underlying *b);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_getDistanceSq_MR_Box3d(__ptr__this, b._UnderlyingPtr);
            }
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box3d::getDistanceSq`.
        public unsafe double GetDistanceSq(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getDistanceSq_MR_Vector3d", ExactSpelling = true)]
            extern static double __MR_Box3d_getDistanceSq_MR_Vector3d(MR.Box3d *_this, MR.Const_Vector3d._Underlying *pt);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_getDistanceSq_MR_Vector3d(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box3d::getProjection`.
        public unsafe MR.Vector3d GetProjection(MR.Const_Vector3d pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_getProjection", ExactSpelling = true)]
            extern static MR.Vector3d __MR_Box3d_getProjection(MR.Box3d *_this, MR.Const_Vector3d._Underlying *pt);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_getProjection(__ptr__this, pt._UnderlyingPtr);
            }
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box3d::expanded`.
        public unsafe MR.Box3d Expanded(MR.Const_Vector3d expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_expanded", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_expanded(MR.Box3d *_this, MR.Const_Vector3d._Underlying *expansion);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_expanded(__ptr__this, expansion._UnderlyingPtr);
            }
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box3d::insignificantlyExpanded`.
        public unsafe MR.Box3d InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box3d_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box3d __MR_Box3d_insignificantlyExpanded(MR.Box3d *_this);
            fixed (MR.Box3d *__ptr__this = &this)
            {
                return __MR_Box3d_insignificantlyExpanded(__ptr__this);
            }
        }

        /// Generated from method `MR::Box3d::operator==`.
        public static unsafe bool operator==(MR.Box3d _this, MR.Box3d a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box3d", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box3d(MR.Const_Box3d._Underlying *_this, MR.Const_Box3d._Underlying *a);
            return __MR_equal_MR_Box3d((MR.Mut_Box3d._Underlying *)&_this, (MR.Mut_Box3d._Underlying *)&a) != 0;
        }

        public static unsafe bool operator!=(MR.Box3d _this, MR.Box3d a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Box3d a)
        {
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Box3d)
                return this == (MR.Box3d)other;
            return false;
        }
    }

    /// This is used as a function parameter when passing `Mut_Box3d` by value with a default argument, since trying to use `?` instead seems to prevent us from taking its address.
    /// Usage:
    /// * Pass an instance of `Mut_Box3d`/`Const_Box3d` to copy it into the function.
    /// * Pass `null` to use the default argument
    public readonly ref struct _InOpt_Box3d
    {
        public readonly bool HasValue;
        internal readonly Box3d Object;
        public Box3d Value{
            get
            {
                System.Diagnostics.Trace.Assert(HasValue);
                return Object;
            }
        }

        public _InOpt_Box3d() {HasValue = false;}
        public _InOpt_Box3d(Box3d new_value) {HasValue = true; Object = new_value;}
        public static implicit operator _InOpt_Box3d(Box3d new_value) {return new(new_value);}
        public _InOpt_Box3d(Const_Box3d new_value) {HasValue = true; Object = new_value.UnderlyingStruct;}
        public static implicit operator _InOpt_Box3d(Const_Box3d new_value) {return new(new_value);}
    }

    /// This is used for optional parameters of class `Mut_Box3d` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3d`/`Const_Box3d` directly.
    /// * Pass `new(ref ...)` to pass a reference to `Box3d`.
    public class _InOptMut_Box3d
    {
        public Mut_Box3d? Opt;

        public _InOptMut_Box3d() {}
        public _InOptMut_Box3d(Mut_Box3d value) {Opt = value;}
        public static implicit operator _InOptMut_Box3d(Mut_Box3d value) {return new(value);}
        public unsafe _InOptMut_Box3d(ref Box3d value)
        {
            fixed (Box3d *value_ptr = &value)
            {
                Opt = new((Const_Box3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// This is used for optional parameters of class `Mut_Box3d` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box3d`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Mut_Box3d`/`Const_Box3d` to pass it to the function.
    /// * Pass `new(ref ...)` to pass a reference to `Box3d`.
    public class _InOptConst_Box3d
    {
        public Const_Box3d? Opt;

        public _InOptConst_Box3d() {}
        public _InOptConst_Box3d(Const_Box3d value) {Opt = value;}
        public static implicit operator _InOptConst_Box3d(Const_Box3d value) {return new(value);}
        public unsafe _InOptConst_Box3d(ref readonly Box3d value)
        {
            fixed (Box3d *value_ptr = &value)
            {
                Opt = new((Const_Box3d._Underlying *)value_ptr, is_owning: false);
            }
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box<unsigned short>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMaxU16`
    /// This is the const half of the class.
    public class Const_Box_UnsignedShort : MR.Misc.Object, System.IDisposable, System.IEquatable<MR.Const_Box_UnsignedShort>
    {
        internal struct _Underlying; // Represents the underlying C++ type.

        internal unsafe _Underlying *_UnderlyingPtr;

        internal unsafe Const_Box_UnsignedShort(_Underlying *ptr, bool is_owning) : base(is_owning) {_UnderlyingPtr = ptr;}

        protected virtual unsafe void Dispose(bool disposing)
        {
            if (_UnderlyingPtr is null || !_IsOwningVal)
                return;
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Destroy", ExactSpelling = true)]
            extern static void __MR_Box_unsigned_short_Destroy(_Underlying *_this);
            __MR_Box_unsigned_short_Destroy(_UnderlyingPtr);
            _UnderlyingPtr = null;
        }
        public virtual void Dispose() {Dispose(true); GC.SuppressFinalize(this);}
        ~Const_Box_UnsignedShort() {Dispose(false);}

        public static unsafe int Elements
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Get_elements", ExactSpelling = true)]
                extern static int *__MR_Box_unsigned_short_Get_elements();
                return *__MR_Box_unsigned_short_Get_elements();
            }
        }

        public unsafe ushort Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Get_min", ExactSpelling = true)]
                extern static ushort *__MR_Box_unsigned_short_Get_min(_Underlying *_this);
                return *__MR_Box_unsigned_short_Get_min(_UnderlyingPtr);
            }
        }

        public unsafe ushort Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Get_max", ExactSpelling = true)]
                extern static ushort *__MR_Box_unsigned_short_Get_max(_Underlying *_this);
                return *__MR_Box_unsigned_short_Get_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Const_Box_UnsignedShort() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_DefaultConstruct();
            _UnderlyingPtr = __MR_Box_unsigned_short_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box<unsigned short>::Box`.
        public unsafe Const_Box_UnsignedShort(MR.Const_Box_UnsignedShort _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_ConstructFromAnother(MR.Box_UnsignedShort._Underlying *_other);
            _UnderlyingPtr = __MR_Box_unsigned_short_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Box<unsigned short>::Box`.
        public unsafe Const_Box_UnsignedShort(ushort min, ushort max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Construct_2", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_Construct_2(ushort *min, ushort *max);
            _UnderlyingPtr = __MR_Box_unsigned_short_Construct_2(&min, &max);
        }

        /// Generated from constructor `MR::Box<unsigned short>::Box`.
        public unsafe Const_Box_UnsignedShort(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Construct_1", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Box_unsigned_short_Construct_1(_1._UnderlyingPtr);
        }

        /// min/max access by 0/1 index
        /// Generated from method `MR::Box<unsigned short>::operator[]`.
        public unsafe ushort Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_index_const", ExactSpelling = true)]
            extern static ushort *__MR_Box_unsigned_short_index_const(_Underlying *_this, int e);
            return *__MR_Box_unsigned_short_index_const(_UnderlyingPtr, e);
        }

        /// Generated from method `MR::Box<unsigned short>::fromMinAndSize`.
        public static unsafe MR.Box_UnsignedShort FromMinAndSize(ushort min, ushort size)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_fromMinAndSize", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_fromMinAndSize(ushort *min, ushort *size);
            return new(__MR_Box_unsigned_short_fromMinAndSize(&min, &size), is_owning: true);
        }

        /// true if the box contains at least one point
        /// Generated from method `MR::Box<unsigned short>::valid`.
        public unsafe bool Valid()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_valid", ExactSpelling = true)]
            extern static byte __MR_Box_unsigned_short_valid(_Underlying *_this);
            return __MR_Box_unsigned_short_valid(_UnderlyingPtr) != 0;
        }

        /// computes center of the box
        /// Generated from method `MR::Box<unsigned short>::center`.
        public unsafe ushort Center()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_center", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_center(_Underlying *_this);
            return __MR_Box_unsigned_short_center(_UnderlyingPtr);
        }

        /// returns the corner of this box as specified by given bool-vector:
        /// 0 element in (c) means take min's coordinate,
        /// 1 element in (c) means take max's coordinate
        /// Generated from method `MR::Box<unsigned short>::corner`.
        public unsafe ushort Corner(bool c)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_corner", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_corner(_Underlying *_this, bool *c);
            return __MR_Box_unsigned_short_corner(_UnderlyingPtr, &c);
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is minimal
        /// Generated from method `MR::Box<unsigned short>::getMinBoxCorner`.
        public static unsafe bool GetMinBoxCorner(ushort n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_getMinBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box_unsigned_short_getMinBoxCorner(ushort *n);
            return __MR_Box_unsigned_short_getMinBoxCorner(&n) != 0;
        }

        /// considering all planes with given normal and arbitrary shift: dot(n,x) = d
        /// finds the box's corner for which d is maximal
        /// Generated from method `MR::Box<unsigned short>::getMaxBoxCorner`.
        public static unsafe bool GetMaxBoxCorner(ushort n)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_getMaxBoxCorner", ExactSpelling = true)]
            extern static byte __MR_Box_unsigned_short_getMaxBoxCorner(ushort *n);
            return __MR_Box_unsigned_short_getMaxBoxCorner(&n) != 0;
        }

        /// computes size of the box in all dimensions
        /// Generated from method `MR::Box<unsigned short>::size`.
        public unsafe ushort Size()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_size", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_size(_Underlying *_this);
            return __MR_Box_unsigned_short_size(_UnderlyingPtr);
        }

        /// computes length from min to max
        /// Generated from method `MR::Box<unsigned short>::diagonal`.
        public unsafe ushort Diagonal()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_diagonal", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_diagonal(_Underlying *_this);
            return __MR_Box_unsigned_short_diagonal(_UnderlyingPtr);
        }

        /// computes the volume of this box
        /// Generated from method `MR::Box<unsigned short>::volume`.
        public unsafe ushort Volume()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_volume", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_volume(_Underlying *_this);
            return __MR_Box_unsigned_short_volume(_UnderlyingPtr);
        }

        /// checks whether given point is inside (including the surface) of this box
        /// Generated from method `MR::Box<unsigned short>::contains`.
        public unsafe bool Contains(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_contains_unsigned_short", ExactSpelling = true)]
            extern static byte __MR_Box_unsigned_short_contains_unsigned_short(_Underlying *_this, ushort *pt);
            return __MR_Box_unsigned_short_contains_unsigned_short(_UnderlyingPtr, &pt) != 0;
        }

        /// checks whether given box is fully inside (the surfaces may touch) of this box
        /// Generated from method `MR::Box<unsigned short>::contains`.
        public unsafe bool Contains(MR.Const_Box_UnsignedShort otherbox)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_contains_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static byte __MR_Box_unsigned_short_contains_MR_Box_unsigned_short(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *otherbox);
            return __MR_Box_unsigned_short_contains_MR_Box_unsigned_short(_UnderlyingPtr, otherbox._UnderlyingPtr) != 0;
        }

        /// returns closest point in the box to given point
        /// Generated from method `MR::Box<unsigned short>::getBoxClosestPointTo`.
        public unsafe ushort GetBoxClosestPointTo(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_getBoxClosestPointTo", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_getBoxClosestPointTo(_Underlying *_this, ushort *pt);
            return __MR_Box_unsigned_short_getBoxClosestPointTo(_UnderlyingPtr, &pt);
        }

        /// checks whether this box intersects or touches given box
        /// Generated from method `MR::Box<unsigned short>::intersects`.
        public unsafe bool Intersects(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_intersects", ExactSpelling = true)]
            extern static byte __MR_Box_unsigned_short_intersects(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return __MR_Box_unsigned_short_intersects(_UnderlyingPtr, b._UnderlyingPtr) != 0;
        }

        /// computes intersection between this and other box
        /// Generated from method `MR::Box<unsigned short>::intersection`.
        public unsafe MR.Box_UnsignedShort Intersection(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_intersection", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_intersection(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return new(__MR_Box_unsigned_short_intersection(_UnderlyingPtr, b._UnderlyingPtr), is_owning: true);
        }

        /// returns squared distance between this box and given one;
        /// returns zero if the boxes touch or intersect
        /// Generated from method `MR::Box<unsigned short>::getDistanceSq`.
        public unsafe ushort GetDistanceSq(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_getDistanceSq_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_getDistanceSq_MR_Box_unsigned_short(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return __MR_Box_unsigned_short_getDistanceSq_MR_Box_unsigned_short(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// returns squared distance between this box and given point;
        /// returns zero if the point is inside or on the boundary of the box
        /// Generated from method `MR::Box<unsigned short>::getDistanceSq`.
        public unsafe ushort GetDistanceSq(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_getDistanceSq_unsigned_short", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_getDistanceSq_unsigned_short(_Underlying *_this, ushort *pt);
            return __MR_Box_unsigned_short_getDistanceSq_unsigned_short(_UnderlyingPtr, &pt);
        }

        /// returns the closest point on the box to the given point
        /// for points outside the box this is equivalent to getBoxClosestPointTo
        /// Generated from method `MR::Box<unsigned short>::getProjection`.
        public unsafe ushort GetProjection(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_getProjection", ExactSpelling = true)]
            extern static ushort __MR_Box_unsigned_short_getProjection(_Underlying *_this, ushort *pt);
            return __MR_Box_unsigned_short_getProjection(_UnderlyingPtr, &pt);
        }

        /// decreases min and increased max on given value
        /// Generated from method `MR::Box<unsigned short>::expanded`.
        public unsafe MR.Box_UnsignedShort Expanded(ushort expansion)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_expanded", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_expanded(_Underlying *_this, ushort *expansion);
            return new(__MR_Box_unsigned_short_expanded(_UnderlyingPtr, &expansion), is_owning: true);
        }

        /// decreases min and increases max to their closest representable value
        /// Generated from method `MR::Box<unsigned short>::insignificantlyExpanded`.
        public unsafe MR.Box_UnsignedShort InsignificantlyExpanded()
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_insignificantlyExpanded", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_insignificantlyExpanded(_Underlying *_this);
            return new(__MR_Box_unsigned_short_insignificantlyExpanded(_UnderlyingPtr), is_owning: true);
        }

        /// Generated from method `MR::Box<unsigned short>::operator==`.
        public static unsafe bool operator==(MR.Const_Box_UnsignedShort _this, MR.Const_Box_UnsignedShort a)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_equal_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static byte __MR_equal_MR_Box_unsigned_short(MR.Const_Box_UnsignedShort._Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *a);
            return __MR_equal_MR_Box_unsigned_short(_this._UnderlyingPtr, a._UnderlyingPtr) != 0;
        }

        public static unsafe bool operator!=(MR.Const_Box_UnsignedShort _this, MR.Const_Box_UnsignedShort a)
        {
            return !(_this == a);
        }

        // IEquatable:

        public bool Equals(MR.Const_Box_UnsignedShort? a)
        {
            if (a is null)
                return false;
            return this == a;
        }

        public override bool Equals(object? other)
        {
            if (other is null)
                return false;
            if (other is MR.Const_Box_UnsignedShort)
                return this == (MR.Const_Box_UnsignedShort)other;
            return false;
        }
    }

    /// Box given by its min- and max- corners
    /// Generated from class `MR::Box<unsigned short>`.
    /// Derived classes:
    ///   Direct: (non-virtual)
    ///     `MR::SimpleVolumeMinMaxU16`
    /// This is the non-const half of the class.
    public class Box_UnsignedShort : Const_Box_UnsignedShort
    {
        internal unsafe Box_UnsignedShort(_Underlying *ptr, bool is_owning) : base(ptr, is_owning) {}

        public new unsafe ref ushort Min
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_GetMutable_min", ExactSpelling = true)]
                extern static ushort *__MR_Box_unsigned_short_GetMutable_min(_Underlying *_this);
                return ref *__MR_Box_unsigned_short_GetMutable_min(_UnderlyingPtr);
            }
        }

        public new unsafe ref ushort Max
        {
            get
            {
                [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_GetMutable_max", ExactSpelling = true)]
                extern static ushort *__MR_Box_unsigned_short_GetMutable_max(_Underlying *_this);
                return ref *__MR_Box_unsigned_short_GetMutable_max(_UnderlyingPtr);
            }
        }

        /// Constructs an empty (default-constructed) instance.
        public unsafe Box_UnsignedShort() : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_DefaultConstruct", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_DefaultConstruct();
            _UnderlyingPtr = __MR_Box_unsigned_short_DefaultConstruct();
        }

        /// Generated from constructor `MR::Box<unsigned short>::Box`.
        public unsafe Box_UnsignedShort(MR.Const_Box_UnsignedShort _other) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_ConstructFromAnother", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_ConstructFromAnother(MR.Box_UnsignedShort._Underlying *_other);
            _UnderlyingPtr = __MR_Box_unsigned_short_ConstructFromAnother(_other._UnderlyingPtr);
        }

        /// Generated from constructor `MR::Box<unsigned short>::Box`.
        public unsafe Box_UnsignedShort(ushort min, ushort max) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Construct_2", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_Construct_2(ushort *min, ushort *max);
            _UnderlyingPtr = __MR_Box_unsigned_short_Construct_2(&min, &max);
        }

        /// Generated from constructor `MR::Box<unsigned short>::Box`.
        public unsafe Box_UnsignedShort(MR.Const_NoInit _1) : this(null, is_owning: true)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_Construct_1", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_Construct_1(MR.NoInit._Underlying *_1);
            _UnderlyingPtr = __MR_Box_unsigned_short_Construct_1(_1._UnderlyingPtr);
        }

        /// Generated from method `MR::Box<unsigned short>::operator=`.
        public unsafe MR.Box_UnsignedShort Assign(MR.Const_Box_UnsignedShort _other)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_AssignFromAnother", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_AssignFromAnother(_Underlying *_this, MR.Box_UnsignedShort._Underlying *_other);
            return new(__MR_Box_unsigned_short_AssignFromAnother(_UnderlyingPtr, _other._UnderlyingPtr), is_owning: false);
        }

        /// Generated from method `MR::Box<unsigned short>::operator[]`.
        public unsafe new ref ushort Index(int e)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_index", ExactSpelling = true)]
            extern static ushort *__MR_Box_unsigned_short_index(_Underlying *_this, int e);
            return ref *__MR_Box_unsigned_short_index(_UnderlyingPtr, e);
        }

        /// minimally increases the box to include given point
        /// Generated from method `MR::Box<unsigned short>::include`.
        public unsafe void Include(ushort pt)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_include_unsigned_short", ExactSpelling = true)]
            extern static void __MR_Box_unsigned_short_include_unsigned_short(_Underlying *_this, ushort *pt);
            __MR_Box_unsigned_short_include_unsigned_short(_UnderlyingPtr, &pt);
        }

        /// minimally increases the box to include another box
        /// Generated from method `MR::Box<unsigned short>::include`.
        public unsafe void Include(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_include_MR_Box_unsigned_short", ExactSpelling = true)]
            extern static void __MR_Box_unsigned_short_include_MR_Box_unsigned_short(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            __MR_Box_unsigned_short_include_MR_Box_unsigned_short(_UnderlyingPtr, b._UnderlyingPtr);
        }

        /// Generated from method `MR::Box<unsigned short>::intersect`.
        public unsafe MR.Box_UnsignedShort Intersect(MR.Const_Box_UnsignedShort b)
        {
            [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_Box_unsigned_short_intersect", ExactSpelling = true)]
            extern static MR.Box_UnsignedShort._Underlying *__MR_Box_unsigned_short_intersect(_Underlying *_this, MR.Const_Box_UnsignedShort._Underlying *b);
            return new(__MR_Box_unsigned_short_intersect(_UnderlyingPtr, b._UnderlyingPtr), is_owning: false);
        }
    }

    /// This is used for optional parameters of class `Box_UnsignedShort` with default arguments.
    /// This is only used mutable parameters. For const ones we have `_InOptConst_Box_UnsignedShort`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Box_UnsignedShort`/`Const_Box_UnsignedShort` directly.
    public class _InOptMut_Box_UnsignedShort
    {
        public Box_UnsignedShort? Opt;

        public _InOptMut_Box_UnsignedShort() {}
        public _InOptMut_Box_UnsignedShort(Box_UnsignedShort value) {Opt = value;}
        public static implicit operator _InOptMut_Box_UnsignedShort(Box_UnsignedShort value) {return new(value);}
    }

    /// This is used for optional parameters of class `Box_UnsignedShort` with default arguments.
    /// This is only used const parameters. For non-const ones we have `_InOptMut_Box_UnsignedShort`.
    /// Usage:
    /// * Pass `null` to use the default argument.
    /// * Pass `new()` to pass no object.
    /// * Pass an instance of `Box_UnsignedShort`/`Const_Box_UnsignedShort` to pass it to the function.
    public class _InOptConst_Box_UnsignedShort
    {
        public Const_Box_UnsignedShort? Opt;

        public _InOptConst_Box_UnsignedShort() {}
        public _InOptConst_Box_UnsignedShort(Const_Box_UnsignedShort value) {Opt = value;}
        public static implicit operator _InOptConst_Box_UnsignedShort(Const_Box_UnsignedShort value) {return new(value);}
    }
}
