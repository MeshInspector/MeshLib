include(ProcessorCount)

# Enables a batched unity build for the target, spreading the files evenly across the CPU cores, one batch per core.
# The list of compiled source files must be passed after the target name.
FUNCTION(mr_enable_unity_build TARGET)
  set_target_properties(${TARGET} PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_MODE BATCH
  )

  ProcessorCount(NUM_CORES)
  IF(NUM_CORES EQUAL 0) # ProcessorCount() reports failure as 0; in that case keep CMake's default batch size.
    return()
  ENDIF()

  list(LENGTH ARGN NUM_SOURCES)
  # Round up, to keep the batch count at most the core count, and to avoid a zero batch size
  # (which CMake interprets as "all files in a single batch").
  math(EXPR UNITY_BATCH_SIZE "(${NUM_SOURCES} + ${NUM_CORES} - 1) / ${NUM_CORES}")
  # Cap the batch size, to avoid producing a few huge translation units on machines with few cores,
  # where they compile too long and use too much memory.
  IF(UNITY_BATCH_SIZE GREATER 256)
    set(UNITY_BATCH_SIZE 256)
  ENDIF()
  set_target_properties(${TARGET} PROPERTIES UNITY_BUILD_BATCH_SIZE ${UNITY_BATCH_SIZE})
ENDFUNCTION()
