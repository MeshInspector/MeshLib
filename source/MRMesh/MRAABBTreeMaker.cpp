#include "MRAABBTreeMaker.h"
#include "MRAABBTreeNode.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRSpdlog.h"
#include <stack>
#include <thread>

namespace MR
{

template<typename T>
struct Subtree
{
    using NodeId = typename AABBTreeNode<T>::NodeId;

    Subtree( NodeId root, int f, int n ) : root( root ), firstLeaf( f ), numLeaves( n ) { }
    NodeId root; // of subtree
    int firstLeaf = 0;
    int numLeaves = 0;
    NodeId lastNode() const { return root + getNumNodes( numLeaves ); }
    bool leaf() const { assert( numLeaves >= 1 );  return numLeaves == 1; }
};

template<typename T>
class AABBTreeMaker
{
public:
    using Node = AABBTreeNode<T>;
    using NodeId = typename Node::NodeId;
    using NodeVec = Vector<Node, NodeId>;
    using Subtree = MR::Subtree<T>;
    using BoxT = typename T::BoxT;

    NodeVec construct( std::vector<BoxedLeaf<T>> boxedLeaves );

private:
    std::vector<BoxedLeaf<T>> boxedLeaves_;
    NodeVec nodes_;

private:
    // [firstLeaf, result) will go to left child and [result, lastLeaf) - to the right child
    int particionLeaves( BoxT & box, int firstLeaf, int lastLeaf );
    // constructs not-leaf node
    std::pair<Subtree, Subtree> makeNode( const Subtree & s );
    // constructs given subtree, optionally splitting the job on given number of threads
    void makeSubtree( const Subtree & s, int numThreads );
};

template<typename T>
int AABBTreeMaker<T>::particionLeaves( BoxT & box, int firstLeaf, int lastLeaf )
{
    assert( firstLeaf + 1 < lastLeaf );
    auto boxDiag = box.max - box.min;
    const int splitDim = int( std::max_element( begin( boxDiag ), end( boxDiag ) ) - begin( boxDiag ) );

    int midLeaf = firstLeaf + ( lastLeaf - firstLeaf ) / 2;
    std::nth_element( boxedLeaves_.data() + firstLeaf, boxedLeaves_.data() + midLeaf, boxedLeaves_.data() + lastLeaf, 
        [&]( const BoxedLeaf<T> & a, const BoxedLeaf<T> & b )
        {
            return a.box.min[splitDim] < b.box.min[splitDim];
        } );
    return midLeaf;
}

template<typename T>
auto AABBTreeMaker<T>::makeNode( const Subtree & s ) -> std::pair<Subtree, Subtree>
{
    assert( !s.leaf() );
    auto & node = nodes_[s.root];
    assert( !node.box.valid() );
    for ( size_t i = 0; i < s.numLeaves; ++i )
        node.box.include( boxedLeaves_[s.firstLeaf + i].box );

    const int midLeaf = particionLeaves( node.box, s.firstLeaf, s.firstLeaf + s.numLeaves );
    const int leftNumLeaves = midLeaf - s.firstLeaf;
    const int rightNumLeaves = s.numLeaves - leftNumLeaves;
    node.l = s.root + 1;
    node.r = s.root + 1 + getNumNodes( leftNumLeaves );
    return
    {
        Subtree( node.l, s.firstLeaf, leftNumLeaves ),
        Subtree( node.r, midLeaf,     rightNumLeaves )
    };
}

template<typename T>
void AABBTreeMaker<T>::makeSubtree( const Subtree & s, int numThreads )
{
    assert( s.root + getNumNodes( s.numLeaves ) <= nodes_.size() );

    if ( numThreads >= 2 && s.numLeaves >= 32 )
    {
        const auto& lr = makeNode( s );
        const int rThreads = numThreads / 2;
        const int lThreads = numThreads - rThreads;
        tbb::task_group group;
        group.run( [&] () { makeSubtree( lr.second, rThreads ); } );
        makeSubtree( lr.first, lThreads );
        group.wait();
        return;
    }

    std::stack<Subtree> stack;
    stack.push( s );

    while ( !stack.empty() )
    {
        const Subtree x = stack.top();
        stack.pop();
        if ( x.leaf() )
        {
            auto & node = nodes_[x.root];
            node.setLeafId( boxedLeaves_[x.firstLeaf].leafId );
            node.box = boxedLeaves_[x.firstLeaf].box;
            continue;
        }

        const auto & [ls, rs] = makeNode( x );
        assert( ls.root < rs.root );
        stack.push( rs );
        stack.push( ls ); // to process it first
    }
}

template<typename T>
auto AABBTreeMaker<T>::construct( std::vector<BoxedLeaf<T>> boxedLeaves ) -> NodeVec
{
    MR_TIMER;

    boxedLeaves_ = std::move( boxedLeaves );

    const auto numLeaves = (int)boxedLeaves_.size();
    nodes_.resize( getNumNodes( numLeaves ) );
    makeSubtree( Subtree( NodeId{ 0 }, 0, numLeaves ), std::thread::hardware_concurrency() );

    return std::move( nodes_ );
}

template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( std::vector<BoxedLeaf<T>> boxedLeaves )
{
    return AABBTreeMaker<T>().construct( std::move( boxedLeaves ) );
}

template AABBTreeNodeVec<FaceTreeTraits3> makeAABBTreeNodeVec( std::vector<BoxedLeaf<FaceTreeTraits3>> boxedLeaves );
template AABBTreeNodeVec<LineTreeTraits2> makeAABBTreeNodeVec( std::vector<BoxedLeaf<LineTreeTraits2>> boxedLeaves );
template AABBTreeNodeVec<LineTreeTraits3> makeAABBTreeNodeVec( std::vector<BoxedLeaf<LineTreeTraits3>> boxedLeaves );

TEST(MRMesh, TBBTask)
{
    spdlog::info( "Hardware concurrency is {}", std::thread::hardware_concurrency() );
    spdlog::info( "TBB num threads is {}", tbb::task_scheduler_init::default_num_threads() );

    using namespace std::chrono_literals;
    
    tbb::task_group group;
    group.run( [] 
    { 
        for( int i = 0; i < 3; ++i )
        {
            spdlog::info( "Task in thread {}", std::this_thread::get_id() );
            std::this_thread::sleep_for( 10ms );
        }
    } );

    for( int i = 0; i < 3; ++i )
    {
        spdlog::info( "Main in thread {}", std::this_thread::get_id() );
        std::this_thread::sleep_for( 10ms );
    }

    group.wait();

    tbb::parallel_for( tbb::blocked_range<int>( 0, 6 ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            spdlog::info( "For item in thread {}", std::this_thread::get_id() );
            std::this_thread::sleep_for( 10ms );
        }
    } );
}

} //namespace MR
