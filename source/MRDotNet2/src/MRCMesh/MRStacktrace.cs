public static partial class MR
{
    /// returns string representation of the current stacktrace;
    /// the function is inlined, to put the code in any shared library;
    /// if std::stacktrace is first called from MRMesh.dll then it is not unloaded propely
    /// Generated from function `MR::getCurrentStacktraceInline`.
    public static unsafe MR.Misc._Moved<MR.Std.String> GetCurrentStacktraceInline()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getCurrentStacktraceInline", ExactSpelling = true)]
        extern static MR.Std.String._Underlying *__MR_getCurrentStacktraceInline();
        return MR.Misc.Move(new MR.Std.String(__MR_getCurrentStacktraceInline(), is_owning: true));
    }

    /// Print stacktrace on application crash
    /// Generated from function `MR::printStacktraceOnCrash`.
    public static void PrintStacktraceOnCrash()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_printStacktraceOnCrash", ExactSpelling = true)]
        extern static void __MR_printStacktraceOnCrash();
        __MR_printStacktraceOnCrash();
    }
}
