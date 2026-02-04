# 1. Install libraries
pip install torch torchvision transformers wandb hydra-core pyfaidx pandas numpy

# 2. Download hg38 Primary Assembly (approx 3GB compressed, 3GB uncompressed)
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz
mv GRCh38.primary_assembly.genome.fa hg38.fa

# GUE script
python eval_gue.py --ckpt dna_lejepa_epoch_9.pth

# GUE dataset
https://huggingface.co/datasets/leannmlindsey/GUE