# Install Torch From source 
1. Clone Pytorch from git: ```git clone --recursive https://github.com/pytorch/pytorch.git```
2. Create new cv environment for installing Torch: ```mkvirtualenv cv -p python3```
3. Install some pre-required packages: ```pip install future leveldb numpy protobuf pydot python-gflags pyyaml scikit-image setuptools six hypothesis typing tqdm pyproject-toml```
4. System must having gcc, clang, cmake.
5. Seting maximum threads to build/install Torch: ```export MAX_JOBS=4```
6. Install Torch: ```pip install --verbose pytorch/.```, this process may take afew hours for completed.
7. Check Torch: ```python -c "import pydoc; pydoc.locate('torch.nn.modules.conv.Conv1d')"```
