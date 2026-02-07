# 1. Install libraries
pip install torch torchvision transformers wandb hydra-core pyfaidx pandas numpy

# Pretraining data (rest is obsolote) from DNABERT

This is what to download, skip the rest :
gdown 1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f

# 2. Download hg38 Primary Assembly (approx 3GB compressed, 3GB uncompressed)
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz
mv GRCh38.primary_assembly.genome.fa hg38.fa

# GUE script
python eval_gue.py --ckpt dna_lejepa_epoch_9.pth

# GUE dataset
https://huggingface.co/datasets/leannmlindsey/GUE

# For math behind uncertainty : https://www.perplexity.ai/search/import-os-import-torch-import-0rhogsrCQ_aJNBPtuodHBA#0

python gue_probe_explain.py \
  --ckpt checkpoints/step_10000.pth \
  --task human_tf_0 \
  --tsne \
  --mc_samples 512 \
  --drop_prob 0.25 \
  --outdir outputs_human_tf_0