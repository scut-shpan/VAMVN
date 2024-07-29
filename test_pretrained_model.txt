1. create environment VAMVN:
    Package            Version
------------------ -------------------
Bottleneck         1.3.5
cached-property    1.5.2
certifi            2022.9.24
charset-normalizer 2.0.4
cloudpickle        2.0.0
cycler             0.11.0
Cython             0.29.32
cytoolz            0.11.0
dask               2021.10.0
flit-core          3.6.0
fsspec             2022.10.0
gluonnlp           0.10.0
graphviz           0.8.4
h5py               3.6.0
idna               3.2
imageio            2.9.0
joblib             1.0.1
kiwisolver         1.4.2
llvmlite           0.37.0
locket             1.0.0
logger             1.4
matplotlib         3.4.3
mkl-fft            1.3.0
mkl-random         1.2.2
mkl-service        2.4.0
mxnet-cu100        1.5.0
networkx           2.6.3
numba              0.54.1
numexpr            2.8.4
numpy              1.20.3
olefile            0.46
opencv-python      4.5.3.56
packaging          21.3
pandas             1.3.5
partd              1.2.0
Pillow             8.3.1
pip                21.0.1
pyparsing          3.0.9
python-dateutil    2.8.2
pytz               2022.1
PyWavelets         1.3.0
PyYAML             6.0
requests           2.26.0
scikit-image       0.19.2
scikit-learn       0.24.2
scipy              1.7.1
seaborn            0.12.1
setuptools         52.0.0.post20210125
six                1.16.0
sklearn            0.0
threadpoolctl      2.2.0
tifffile           2020.10.1
toolz              0.12.0
torch              1.4.0
torchaudio         0.4.0a0+719bcc7
torchsummary       1.5.1
torchsummaryX      1.3.0
torchvision        0.5.0
tornado            6.2
tqdm               4.63.0
typing-extensions  4.4.0
urllib3            1.26.6
wheel              0.37.0


2. download and unzip ScanObjectNN dataset from https://github.com/hkust-vgd/scanobjectnn
3. render scanobjectnn by python script : Pytorch3D.render_pc.py
    python render_pc.py -dataset_dir ./ScanObjectNN/ -save_dir dataset/ScanObjectNN_img/

4. Use pretrained model to test:
    python test_models.py