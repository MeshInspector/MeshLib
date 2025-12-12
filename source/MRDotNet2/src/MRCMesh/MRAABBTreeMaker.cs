public static partial class MR
{
    /// returns the number of nodes in the binary tree with given number of leaves
    /// Generated from function `MR::getNumNodes`.
    public static int GetNumNodes(int numLeaves)
    {
        [System.Runtime.InteropServices.DllImport("MRMesh", EntryPoint = "MR_getNumNodes", ExactSpelling = true)]
        extern static int __MR_getNumNodes(int numLeaves);
        return __MR_getNumNodes(numLeaves);
    }
}
