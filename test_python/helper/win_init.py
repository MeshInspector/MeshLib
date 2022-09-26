# copy to source\x64\{config}\meshlib to fix dll loading

def _init_patch():
    import os
    libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    os.add_dll_directory(libs_dir)


# analogue of the delvewheel init patch
_init_patch()
del _init_patch
