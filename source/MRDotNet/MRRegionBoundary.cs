using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct MREdgeLoop
    {
        public IntPtr data = IntPtr.Zero;
        public ulong size = 0;
        public IntPtr reserved = IntPtr.Zero;

        public MREdgeLoop() { }
    }

    public class EdgeLoop : List<EdgeId>, IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern void mrEdgePathFree(MREdgeLoop* loop);

        unsafe internal EdgeLoop(MREdgeLoop* mrLoop)
        : base()
        {
            mrLoop_ = mrLoop;
            for ( int i = 0; i < (int)mrLoop->size; i++ )
            {
                Add(new EdgeId(Marshal.ReadInt32(IntPtr.Add(mrLoop->data, i * sizeof(int)))));
            }
        }
        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        unsafe protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mrLoop_ != null)
                {
                    mrEdgePathFree(mrLoop_);
                }

                disposed = true;
            }
        }

        ~EdgeLoop()
        {
            Dispose(false);
        }

        unsafe internal MREdgeLoop* mrLoop_;
    }

    public class EdgeLoops : List<List<EdgeId>>, IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MREdgeLoop mrEdgeLoopsGet(IntPtr loops, ulong index);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrEdgeLoopsSize(IntPtr loops);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrEdgeLoopsFree(IntPtr loops);

        public EdgeLoops( IntPtr mrLoops )
        : base()
        {
            mrLoops_ = mrLoops;
            int numLoops = (int)mrEdgeLoopsSize(mrLoops);
            for ( int i = 0; i < numLoops; i++ )
            {
                Add(new List<EdgeId>());
                var mrLoop = mrEdgeLoopsGet(mrLoops, (ulong)i);
                for ( int j = 0; j < (int)mrLoop.size; j++ )
                {
                    this[i].Add(new EdgeId(Marshal.ReadInt32(IntPtr.Add(mrLoop.data, j * sizeof(int)))));
                }
            }
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        unsafe protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mrLoops_ != null)
                {
                    mrEdgeLoopsFree(mrLoops_);
                }

                disposed = true;
            }
        }

        ~EdgeLoops()
        {
            Dispose(false);
        }

        internal IntPtr mrLoops_;
    }

    public class RegionBoundary
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindRightBoundary( IntPtr topology, IntPtr region );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern MREdgeLoop* mrTrackRightBoundaryLoop(IntPtr topology, EdgeId e0, IntPtr region);
        
        /// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
        /// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
        unsafe public static EdgeLoop TrackRightBoundaryLoop( Mesh mesh, EdgeId e0, BitSet? region = null )
        {
            var mrLoop = mrTrackRightBoundaryLoop(mesh.meshTopology_, e0, region is null ? (IntPtr)null : region.bs_);
            return new EdgeLoop(mrLoop);
        }
        /// returns all region boundary loops;
        /// every loop has region faces on the right, and not-region faces or holes on the left
        public static EdgeLoops FindRightBoundary( Mesh mesh, BitSet? region = null )
        {
            var mrLoops = mrFindRightBoundary(mesh.meshTopology_, region is null ? (IntPtr)null : region.bs_);
            return new EdgeLoops(mrLoops);
        }
    }
}
