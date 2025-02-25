#include "MRMesh/MRChunkIterator.h"

#include "MRMesh/MRGTest.h"

namespace MR
{

TEST(ChunkIteratorTest, BasicChunkCount)
{
    // Basic functionality
    EXPECT_EQ(3, chunkCount(100, 40, 0)); // 100/40 = 2.5 -> 3 chunks
    EXPECT_EQ(4, chunkCount(100, 30, 0)); // 100/30 = 3.33 -> 4 chunks
    EXPECT_EQ(2, chunkCount(100, 50, 0)); // 100/50 = 2 -> 2 chunks
}

TEST(ChunkIteratorTest, ChunkCountWithOverlap)
{
    // With overlap
    EXPECT_EQ(4, chunkCount(100, 40, 20)); // Step: 20, (100-20)/20 = 4
    EXPECT_EQ(3, chunkCount(100, 50, 25)); // Step: 25, (100-25)/25 = 3
    EXPECT_EQ(3, chunkCount(50, 25, 10)); // Step: 15, (50-10)/15 = 2.67 -> 3 chunks
}

TEST(ChunkIteratorTest, ChunkCountEdgeCases)
{
    // Edge cases
    EXPECT_EQ(0, chunkCount(0, 10, 0)); // Empty array
    EXPECT_EQ(0, chunkCount(100, 0, 0)); // Zero chunk size
    EXPECT_EQ(0, chunkCount(100, 10, 10)); // Overlap equals chunk size
    EXPECT_EQ(0, chunkCount(100, 10, 15)); // Overlap exceeds chunk size

    EXPECT_EQ(1, chunkCount(10, 10, 0)); // Array size equals chunk size
    EXPECT_EQ(1, chunkCount(5, 10, 0)); // Array size smaller than chunk size

    EXPECT_EQ(9, chunkCount(100, 20, 10)); // 50% overlap
    EXPECT_EQ(100, chunkCount(101, 2, 1)); // (101-1)/(2-1) = 100 chunks
}

TEST(ChunkIteratorTest, SplitByChunksBasic)
{
    // Basic test
    auto [begin, end] = splitByChunks(100, 40, 0);

    EXPECT_EQ(begin.totalSize, 100);
    EXPECT_EQ(begin.chunkSize, 40);
    EXPECT_EQ(begin.overlap, 0);
    EXPECT_EQ(begin.index, 0);

    EXPECT_EQ(end.totalSize, 100);
    EXPECT_EQ(end.chunkSize, 40);
    EXPECT_EQ(end.overlap, 0);
    EXPECT_EQ(end.index, 3); // 3 chunks for 100 size with 40 chunk size

    EXPECT_NE(begin, end);
}

TEST(ChunkIteratorTest, SplitByChunksWithOverlap)
{
    // With overlap
    auto [begin, end] = splitByChunks(100, 40, 10);

    EXPECT_EQ(begin.totalSize, 100);
    EXPECT_EQ(begin.chunkSize, 40);
    EXPECT_EQ(begin.overlap, 10);
    EXPECT_EQ(begin.index, 0);

    EXPECT_EQ(end.totalSize, 100);
    EXPECT_EQ(end.chunkSize, 40);
    EXPECT_EQ(end.overlap, 10);
    EXPECT_EQ(end.index, 3); // 3 chunks with overlap
}

TEST(ChunkIteratorTest, SplitByChunksEdgeCases)
{
    // Empty array
    auto [emptyBegin, emptyEnd] = splitByChunks(0, 10, 0);
    EXPECT_EQ(emptyBegin.index, 0);
    EXPECT_EQ(emptyEnd.index, 0);
    EXPECT_EQ(emptyBegin, emptyEnd);

    // Zero chunk size
    auto [zeroChunkBegin, zeroChunkEnd] = splitByChunks(100, 0, 0);
    EXPECT_EQ(zeroChunkBegin.index, 0);
    EXPECT_EQ(zeroChunkEnd.index, 0);
    EXPECT_EQ(zeroChunkBegin, zeroChunkEnd);

    // Overlap equal to chunk size
    auto [equalOverlapBegin, equalOverlapEnd] = splitByChunks(100, 10, 10);
    EXPECT_EQ(equalOverlapBegin.index, 0);
    EXPECT_EQ(equalOverlapEnd.index, 0);
    EXPECT_EQ(equalOverlapBegin, equalOverlapEnd);
}

TEST(ChunkIteratorTest, ChunkIteratorIncrement)
{
    ChunkIterator it{100, 40, 0, 0};

    EXPECT_EQ(it.index, 0);
    ++it;
    EXPECT_EQ(it.index, 1);
    ++it;
    EXPECT_EQ(it.index, 2);
}

TEST(ChunkIteratorTest, ChunkIteratorDereference)
{
    // Basic chunk
    {
        ChunkIterator it{100, 40, 0, 0};
        Chunk chunk = *it;
        EXPECT_EQ(chunk.offset, 0);
        EXPECT_EQ(chunk.size, 40);
    }

    // Middle chunk
    {
        ChunkIterator it{100, 40, 0, 1};
        Chunk chunk = *it;
        EXPECT_EQ(chunk.offset, 40);
        EXPECT_EQ(chunk.size, 40);
    }

    // Last chunk (may be smaller)
    {
        ChunkIterator it{100, 40, 0, 2};
        Chunk chunk = *it;
        EXPECT_EQ(chunk.offset, 80);
        EXPECT_EQ(chunk.size, 20); // Last chunk is smaller
    }
}

TEST(ChunkIteratorTest, ChunkIteratorWithOverlap)
{
    // With 50% overlap
    ChunkIterator it{100, 40, 20, 0};

    // First chunk
    {
        Chunk chunk = *it;
        EXPECT_EQ(chunk.offset, 0);
        EXPECT_EQ(chunk.size, 40);
    }

    // Second chunk, 20 units overlap with first
    ++it;
    {
        Chunk chunk = *it;
        EXPECT_EQ(chunk.offset, 20); // 20 units after first chunk start
        EXPECT_EQ(chunk.size, 40);
    }

    // Third chunk, 20 units overlap with second
    ++it;
    {
        Chunk chunk = *it;
        EXPECT_EQ(chunk.offset, 40); // 20 units after second chunk start
        EXPECT_EQ(chunk.size, 40);
    }
}

TEST(ChunkIteratorTest, IteratorRangeForLoop)
{
    // Test that the iterators work with range-based for loop
    auto [begin, end] = splitByChunks(100, 30, 0);

    std::vector<Chunk> chunks;
    for (auto it = begin; it != end; ++it)
    {
        chunks.push_back(*it);
    }

    // Should have 4 chunks
    EXPECT_EQ(chunks.size(), 4);

    // Check first chunk
    EXPECT_EQ(chunks[0].offset, 0);
    EXPECT_EQ(chunks[0].size, 30);

    // Check last chunk
    EXPECT_EQ(chunks[3].offset, 90);
    EXPECT_EQ(chunks[3].size, 10);

    // Check all offsets are correct
    for (size_t i = 0; i < chunks.size(); ++i)
    {
        EXPECT_EQ(chunks[i].offset, i * 30);
    }
}

TEST(ChunkIteratorTest, IteratorStandardLibraryAlgorithms)
{
    // Test that the iterators work with standard library algorithms
    auto [begin, end] = splitByChunks(100, 25, 0);

    // Count chunks
    auto count = std::distance(begin, end);
    EXPECT_EQ(count, 4);

    // Convert to vector
    std::vector<Chunk> chunks(begin, end);
    EXPECT_EQ(chunks.size(), 4);

    // Verify total coverage equals the original size
    size_t totalSize = 0;
    std::for_each(begin, end, [&totalSize]( const Chunk& c )
    {
        totalSize += c.size;
    });
    EXPECT_GE(totalSize, 100); // May be greater due to overlaps
}

struct ChunkIteratorTestCase
{
    size_t totalSize;
    size_t chunkSize;
    size_t overlap;
    size_t expectedChunks;
};

class ChunkIteratorTestFixture : public testing::TestWithParam<ChunkIteratorTestCase>
{
};

TEST_P(ChunkIteratorTestFixture, ExhaustiveParameterizedTest)
{
    const auto& tc = GetParam();

    EXPECT_EQ(chunkCount(tc.totalSize, tc.chunkSize, tc.overlap), tc.expectedChunks);

    auto [begin, end] = splitByChunks(tc.totalSize, tc.chunkSize, tc.overlap);
    EXPECT_EQ(std::distance(begin, end), tc.expectedChunks);

    // If we expect chunks, verify they cover the array correctly
    if ( tc.expectedChunks > 0 )
    {
        BitSet covered( tc.totalSize, false );

        for ( auto it = begin; it != end; ++it )
        {
            Chunk chunk = *it;
            EXPECT_LE( chunk.offset + chunk.size, tc.totalSize + 1 );

            // Mark covered elements
            for ( size_t i = 0; i < chunk.size; ++i )
                covered.set( chunk.offset + i );
        }

        // Verify all elements are covered
        EXPECT_TRUE( covered.all() );
    }
}

INSTANTIATE_TEST_SUITE_P(ChunkIteratorTest, ChunkIteratorTestFixture, testing::Values(
    ChunkIteratorTestCase {100, 10, 0, 10}, // No overlap
    ChunkIteratorTestCase {100, 10, 5, 19}, // 50% overlap
    ChunkIteratorTestCase {100, 30, 10, 5}, // larger chunks with overlap
    ChunkIteratorTestCase {100, 100, 0, 1}, // single chunk
    ChunkIteratorTestCase {50, 60, 0, 1}, // chunk larger than total
    ChunkIteratorTestCase {0, 10, 0, 0}, // empty array
    ChunkIteratorTestCase {10, 0, 0, 0}, // zero chunk size
    ChunkIteratorTestCase {100, 10, 10, 0} // 100% overlap (invalid)
) );

} // namespace MR
