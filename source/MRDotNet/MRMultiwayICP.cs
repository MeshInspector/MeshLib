using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using static MR.DotNet.ICP;

namespace MR
{
    public partial class DotNet
    {
        /// Parameters that are used for sampling of the MultiwayICP objects
        public struct MultiwayICPSamplingParameters
        {
            public enum CascadeMode
            {
                Sequential, /// separates objects on groups based on their index in ICPObjects (good if all objects about the size of all objects together)
                AABBTreeBased /// builds AABB tree based on each object bounding box and separates subtrees (good if each object much smaller then all objects together)
            };

            /// sampling size of each object
            public float samplingVoxelSize = 0.0f;

            /// size of maximum icp group to work with
            /// if number of objects exceeds this value, icp is applied in cascade mode
            public int maxGroupSize = 64;


            public CascadeMode cascadeMode = CascadeMode.AABBTreeBased;

            public MultiwayICPSamplingParameters()
            { }
        };

        public class MultiwayICP
        {
            [StructLayout(LayoutKind.Sequential)]
            internal struct MRMultiwayICPSamplingParameters
            {
                public float samplingVoxelSize = 0.0f;
                public int maxGroupSize = 64;
                public MultiwayICPSamplingParameters.CascadeMode cascadeMode = MultiwayICPSamplingParameters.CascadeMode.AABBTreeBased;
                public IntPtr cb = IntPtr.Zero;
                public MRMultiwayICPSamplingParameters() { }
            };

            [StructLayout(LayoutKind.Sequential)]
            internal struct MRVectorAffineXf3f
            {
                public IntPtr data = IntPtr.Zero;
                public ulong size = 0;
                public IntPtr reserved = IntPtr.Zero;
                public MRVectorAffineXf3f() { }
            }

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern IntPtr mrMultiwayICPNew(IntPtr objects, ulong objectsNum, ref MRMultiwayICPSamplingParameters samplingParams);


            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern MRVectorAffineXf3f* mrMultiwayICPCalculateTransformations(IntPtr mwicp, IntPtr cb);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern void mrVectorAffineXf3fFree(MRVectorAffineXf3f* p);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern bool mrMultiwayICPResamplePoints(IntPtr mwicp, ref MRMultiwayICPSamplingParameters samplingParams);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern bool mrMultiwayICPUpdateAllPointPairs(IntPtr mwicp, IntPtr cb);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern void mrMultiwayICPSetParams(IntPtr mwicp, ref MRICPProperties prop);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern float mrMultiWayICPGetMeanSqDistToPoint(IntPtr mwicp, double* value);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            unsafe private static extern float mrMultiWayICPGetMeanSqDistToPlane(IntPtr mwicp, double* value);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern ulong mrMultiWayICPGetNumSamples(IntPtr mwicp);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern ulong mrMultiWayICPGetNumActivePairs(IntPtr mwicp);

            [DllImport("MRMeshC", CharSet = CharSet.Auto)]
            private static extern void mrMultiwayICPFree(IntPtr mwicp);

            unsafe public MultiwayICP(List<MeshOrPointsXf> objs, MultiwayICPSamplingParameters samplingParams)
            {
                int sizeOfIntPtr = Marshal.SizeOf(typeof(IntPtr));
                IntPtr nativeObjs = Marshal.AllocHGlobal(objs.Count * sizeOfIntPtr);

                try
                {
                    for (int i = 0; i < objs.Count; i++)
                    {
                        Marshal.StructureToPtr(objs[i].mrMeshOrPointsXf_, IntPtr.Add(nativeObjs, i * sizeOfIntPtr), false);
                    }

                    MRMultiwayICPSamplingParameters mrParams = new MRMultiwayICPSamplingParameters();
                    mrParams.samplingVoxelSize = samplingParams.samplingVoxelSize;
                    mrParams.maxGroupSize = samplingParams.maxGroupSize;
                    mrParams.cascadeMode = samplingParams.cascadeMode;
                    mrParams.cb = IntPtr.Zero;
                    icp_ = mrMultiwayICPNew(nativeObjs, (ulong)objs.Count, ref mrParams);
                }
                finally
                {
                    Marshal.FreeHGlobal(nativeObjs);
                }
            }

            private bool disposed = false;
            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!disposed)
                {
                    if (icp_ != IntPtr.Zero)
                    {
                        mrMultiwayICPFree(icp_);
                    }

                    disposed = true;
                }
            }
            ~MultiwayICP()
            {
                Dispose(false);
            }

            /// runs ICP algorithm given input objects, transformations, and parameters;
            /// \return adjusted transformations of all objects to reach registered state
            unsafe public List<AffineXf3f> CalculateTransformations()
            {
                int sizeOfXf = Marshal.SizeOf(typeof(MRAffineXf3f));
                List<AffineXf3f> xfs = new List<AffineXf3f>();
                var p = mrMultiwayICPCalculateTransformations(icp_, IntPtr.Zero);
                if (p == null)
                    return xfs;
                var mrXfs = *p;
                for (int i = 0; i < (int)mrXfs.size; i++)
                {
                    IntPtr currentXfPtr = IntPtr.Add(mrXfs.data, i * sizeOfXf);
                    var mrXf = Marshal.PtrToStructure<MRAffineXf3f>(currentXfPtr);
                    xfs.Add(new AffineXf3f(mrXf));
                }
                mrVectorAffineXf3fFree(p);
                return xfs;
            }
            /// select pairs with origin samples on all objects
            public void ResamplePoints(MultiwayICPSamplingParameters samplingParams)
            {
                MRMultiwayICPSamplingParameters mrParams = new MRMultiwayICPSamplingParameters();
                mrParams.samplingVoxelSize = samplingParams.samplingVoxelSize;
                mrParams.maxGroupSize = samplingParams.maxGroupSize;
                mrParams.cascadeMode = samplingParams.cascadeMode;
                mrParams.cb = IntPtr.Zero;
                mrMultiwayICPResamplePoints(icp_, ref mrParams);
            }

            /// in each pair updates the target data and performs basic filtering (activation)
            /// in cascade mode only useful for stats update <summary>
            [return: MarshalAs(UnmanagedType.I1)]
            public bool UpdateAllPointPairs()
            {
                return mrMultiwayICPUpdateAllPointPairs(icp_, IntPtr.Zero);
            }
            /// tune algorithm params before run calculateTransformations()
            public void SetParams(ICPProperties props)
            {
                var icpProp = props.ToNative();
                mrMultiwayICPSetParams(icp_, ref icpProp);
            }
            /// computes root-mean-square deviation between points
            public unsafe float GetMeanSqDistToPoint()
            {
                return mrMultiWayICPGetMeanSqDistToPoint(icp_, null);
            }
            /// computes the standard deviation from given value
            public unsafe float GetMeanSqDistToPoint(double value)
            {
                return mrMultiWayICPGetMeanSqDistToPoint(icp_, &value);
            }
            /// computes root-mean-square deviation from points to target planes
            public unsafe float GetMeanSqDistToPlane()
            {
                return mrMultiWayICPGetMeanSqDistToPlane(icp_, null);
            }
            /// computes the standard deviation from given value
            public unsafe float GetMeanSqDistToPlane(double value)
            {
                return mrMultiWayICPGetMeanSqDistToPlane(icp_, &value);
            }
            /// computes the number of samples able to form pairs
            public int GetNumSamples()
            {
                return (int)mrMultiWayICPGetNumSamples(icp_);
            }
            /// computes the number of active point pairs
            public int GetNumActivePairs()
            {
                return (int)mrMultiWayICPGetNumActivePairs(icp_);
            }

            private IntPtr icp_;
        }
    }
}
