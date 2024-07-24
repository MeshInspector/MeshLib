#pragma once

// Those are generic helper macros that don't have their own headers.

#define MR_STR(...) MR_STR_(__VA_ARGS__)
#define MR_STR_(...) #__VA_ARGS__
