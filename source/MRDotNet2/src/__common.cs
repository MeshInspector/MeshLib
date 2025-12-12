public static partial class MR
{
    public static partial class Misc
    {
        /// This is the base class for all our classes.
        public abstract class Object
        {
            protected bool _IsOwningVal;
            /// Returns true if this is an owning instance, and when disposed, will destroy the underlying C++ instance.
            /// If false, we assume that the underlying C++ instance will live long enough.
            public virtual bool _IsOwning => _IsOwningVal;

            /// Which objects need to be kept alive while this object exists? This is public just in case.
            public List<object>? _KeepAliveList;
            public void _KeepAlive(object obj)
            {
                if (_KeepAliveList is null)
                    _KeepAliveList = new();
                _KeepAliveList.Add(obj);
            }

            internal Object(bool is_owning) {_IsOwningVal = is_owning;}
        }

        /// This is the base class for those of our classes that are backed by `std::shared_ptr`.
        public abstract class SharedObject : Object
        {
            /// This checks if the `shared_ptr` itself is owning or not, rather than whether we own our `shared_ptr`, which isn't a given.
            /// The derived classes have to implement this, since it depends on the specific `shared_ptr` type.
            public abstract override bool _IsOwning {get;}
            /// This checks if we own the underlying `shared_ptr` instance, regardless of whether it owns the underlying object, which is orthogonal.
            /// We repurpose `_IsOwningVal` for this.
            public bool _IsOwningSharedPtr => _IsOwningVal;

            internal SharedObject(bool is_owning) : base(is_owning) {}
        }

        /// This is used for optional in/out parameters, since `ref` can't be nullable.
        public class InOut<T> where T: unmanaged
        {
            public T Value;

            public InOut() {}
            public InOut(T NewValue) {Value = NewValue;}
        }

        /// This is used for optional parameters with default arguments.
        /// Usage:
        /// * Pass `null` to use the default argument.
        /// * Pass `new()` to pass no object.
        /// * Pass an instance of `T` to pass it to the function.
        /// Passing a null `_InOpt` means "use default argument", and passing a one with a null `.Opt` means "pass nothing to the function".
        public class _InOpt<T> where T: unmanaged
        {
            public T? Opt;

            public _InOpt() {}
            public _InOpt(T NewOpt) {Opt = NewOpt;}
            public static implicit operator _InOpt<T>(T NewOpt) {return new _InOpt<T>(NewOpt);}
        }

        /// A reference to a C object. This is used to return optional references, since `ref` can't be nullable.
        /// This object itself isn't nullable, we return `Ref<T>?` when nullability is needed.
        public unsafe class Ref<T> where T: unmanaged
        {
            /// Should never be null.
            private T *Ptr;
            /// Should never be given a null pointer. I would pass `ref T`, but this prevents the address from being taken without `fixed`.
            internal Ref(T *NewPtr)
            {
                System.Diagnostics.Trace.Assert(NewPtr is not null);
                Ptr = NewPtr;
            }

            public ref T Value => ref *Ptr;
        }

        /// Wraps the object in a wrapper that indicates that it should be treated as a temporary object.
        /// This can be used with `_ByValue_...` function parameters, to indicate that the argument should be moved.
        /// See those structs for a longer explanation.
        public static _Moved<T> Move<T>(T new_value) {return new(new_value);}

        /// A wrapper for `T` that indicates that it's a temporary object, or should be treated as such.
        /// If you're calling a function that returns this, you can safely convert this to `T`.
        /// If you're calling a function that takes this as a parameter, use the `Move()` function to create this wrapper.
        public readonly struct _Moved<T>
        {
            public readonly T Value;
            internal _Moved(T new_value) {Value = new_value;}
            public static implicit operator T(_Moved<T> moved) {return moved.Value;}
        }

        internal enum _PassBy : int
        {
            default_construct,
            copy,
            move,
            default_arg,
            no_object,
        }

        /// This is a tag value. Pass it to functions having a `_MoveRef` parameter.
        /// This indicates that the reference parameter immediately following it is an rvalue reference.
        public static _MoveRef MoveRef = default;
        /// This is a tag type for passing rvalue references. Don't construct directly, prefer the `MoveRef` constant.
        public struct _MoveRef {}

        /// The type of `NullOpt`, see that for more details.
        public struct NullOptType {}
        /// This can be passed into `_ByValueOptOpt_...` parameters to indicate that you want to pass no object,
        ///   as opposed to using the default argument provided by the function.
        public static NullOptType NullOpt;

        /// This is used for optional `ReadOnlySpan<char>` function parameters. This is a specialized version that provides string interop.
        /// Pass `null` or `new()` to use the default argument.
        ///   Note that for the original `ReadOnlySpan`, those result in an empty span instead.
        public ref struct ReadOnlyCharSpanOpt
        {
            public readonly bool HasValue;

            ReadOnlySpan<char> Span;
            public ReadOnlySpan<char> Value
            {
                get
                {
                    System.Diagnostics.Trace.Assert(HasValue);
                    return Span;
                }
            }

            public ReadOnlyCharSpanOpt(char[]? arr) {HasValue = arr is not null; Span = arr;}
            public ReadOnlyCharSpanOpt(ReadOnlySpan<char> span) {HasValue = true; Span = span;}
            public ReadOnlyCharSpanOpt(string? str) {HasValue = str is not null; Span = str;}

            // This is disabled because it makes conversion from `null` ambiguous.
            // public static implicit operator ReadOnlyCharSpanOpt(char[]? arr) {return new(arr);}
            public static implicit operator ReadOnlyCharSpanOpt(ReadOnlySpan<char> span) {return new(span);}
            public static implicit operator ReadOnlyCharSpanOpt(string? str) {return new(str);}
        }

    }

    public static partial class Std
    {
        /// This is an empty tag type.
        public struct Greater_Float {}

        /// This is an empty tag type.
        public struct Monostate {}

        /// This is an empty tag type.
        public struct VariantIndex_0 {}

        /// This is an empty tag type.
        public struct VariantIndex_1 {}

        /// This is an empty tag type.
        public struct VariantIndex_2 {}

        /// This is an empty tag type.
        public struct VariantIndex_3 {}

        /// This is an empty tag type.
        public struct VariantIndex_4 {}

        /// This is an empty tag type.
        public struct VariantIndex_5 {}

        /// This is an empty tag type.
        public struct VariantIndex_6 {}

    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayAffineXf3d4
    {
        MR.AffineXf3d elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayAffineXf3f4
    {
        MR.AffineXf3f elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayDouble4
    {
        double elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayFloat4
    {
        float elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayInt4
    {
        int elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayVector2d4
    {
        MR.Vector2d elem;
    }

    [System.Runtime.CompilerServices.InlineArray(3)]
    public struct ArrayVector2f3
    {
        MR.Vector2f elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayVector2f4
    {
        MR.Vector2f elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayVector3d4
    {
        MR.Vector3d elem;
    }

    [System.Runtime.CompilerServices.InlineArray(3)]
    public struct ArrayVector3f3
    {
        MR.Vector3f elem;
    }

    [System.Runtime.CompilerServices.InlineArray(4)]
    public struct ArrayVector3f4
    {
        MR.Vector3f elem;
    }

    [System.Runtime.CompilerServices.InlineArray(2)]
    public struct ArrayVertId2
    {
        MR.VertId elem;
    }

    [System.Runtime.CompilerServices.InlineArray(3)]
    public struct ArrayVertId3
    {
        MR.VertId elem;
    }
}
