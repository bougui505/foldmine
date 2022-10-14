## Install

### Conda based

For a GPU install :
```
conda create -n foldmine
conda activate foldmine
conda install pytorch=1.11 torchvision pymol cudatoolkit=11.3 -c pytorch -c schrodinger
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torchdrug PyYAML scipy easydict
```

For a CPU install :
```
conda create -n foldmine
conda activate foldmine
conda install pytorch=1.11 torchvision pymol -c pytorch -c schrodinger
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torchdrug PyYAML scipy easydict
```

### Image based

One can also use singularity wrappers to ensure reproducibility :
Prebuilt files exist for both CPU and GPU setting, along with recipes to build them.

For a GPU install :
```
sudo singularity build conda.sif conda.def
sudo singularity build torchdrug_gpu.sif torchdrug_gpu.def #GPU
sudo singularity build torchdrug_cpu.sif torchdrug_cpu.def #CPU
```

To use the prebuilt files, just directly download the .sif files.
Then to test them, use :
```
# singularity run torchdrug_cpu.sif /opt/anaconda/bin/activate foldmine; python -c 'import torch; print(torch.cuda.is_available())' # >>> False
# singularity run --nv torchdrug_gpu.sif /opt/anaconda/bin/activate foldmine; python -c 'import torch; print(torch.cuda.is_available())' # >>> True
```

Or in interactive mode :
```
singularity shell --nv -B /ld19-1020/ --nv torchdrug_gpu.sif
source /opt/anaconda/bin/activate 
conda activate foldmine
ulimit -n 2048
python -c 'import torch; print(torch.cuda.is_available())' # >>> True
```

Inference is just done by running `python embed_all.py -i pdb_chainsplit/ -o example.hdf5`
And from there one can build indices using `python makeindex.py --hdf5 example.hdf5 --out example`

To query this index, one just need to run `python makeindex.py --query 1ycr_A.pdb` and you will see 
a list of neighbors.


