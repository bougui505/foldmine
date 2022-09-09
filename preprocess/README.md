# Data preprocessing

First get the homology results running :
```
blastp -query pdb_seqres.fasta -db pdb_seqres.fasta -outfmt 6 -out blast_results.txt
```

Then run :
```
python preprocess.py --all
```

