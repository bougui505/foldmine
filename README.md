Install

```
conda create -n foldmine
conda activate foldmine
conda install pytorch=1.11 pymol -c pytorch -c schrodinger
pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
```