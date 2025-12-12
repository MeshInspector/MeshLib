public static partial class MR
{
    /// enables or disables printing of timing tree when application terminates
    /// \param minTimeSec omit printing records with time spent less than given value in seconds
    /// Generated from function `MR::printTimingTreeAtEnd`.
    /// Parameter `minTimeSec` defaults to `0.10000000000000001`.
    public static unsafe void PrintTimingTreeAtEnd(bool on, double? minTimeSec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_printTimingTreeAtEnd", ExactSpelling = true)]
        extern static void __MR_printTimingTreeAtEnd(byte on, double *minTimeSec);
        double __deref_minTimeSec = minTimeSec.GetValueOrDefault();
        __MR_printTimingTreeAtEnd(on ? (byte)1 : (byte)0, minTimeSec.HasValue ? &__deref_minTimeSec : null);
    }

    /// prints current timer branch
    /// Generated from function `MR::printCurrentTimerBranch`.
    public static void PrintCurrentTimerBranch()
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_printCurrentTimerBranch", ExactSpelling = true)]
        extern static void __MR_printCurrentTimerBranch();
        __MR_printCurrentTimerBranch();
    }

    /// prints the current timing tree
    /// \param minTimeSec omit printing records with time spent less than given value in seconds
    /// Generated from function `MR::printTimingTree`.
    /// Parameter `minTimeSec` defaults to `0.10000000000000001`.
    public static unsafe void PrintTimingTree(double? minTimeSec = null)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_printTimingTree", ExactSpelling = true)]
        extern static void __MR_printTimingTree(double *minTimeSec);
        double __deref_minTimeSec = minTimeSec.GetValueOrDefault();
        __MR_printTimingTree(minTimeSec.HasValue ? &__deref_minTimeSec : null);
    }
}
