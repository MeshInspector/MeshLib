# copy to source\x64\{config}\meshlib to fix dll loading
import os

libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.add_dll_directory(libs_dir)