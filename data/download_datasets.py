from datasets import load_dataset
import json

snli = load_dataset('snli')
snli.save_to_disk("data/raw/snli")

mnli = load_dataset('multi_nli')
mnli.save_to_disk("data/raw/mnli")

scitail = load_dataset('scitail', 'snli_format')
scitail.save_to_disk("data/raw/scitail")

fracas = load_dataset("pietrolesci/fracas")
fracas.save_to_disk("data/raw/fracas")

pascal = load_dataset("nyu-mll/glue", "rte")
pascal.save_to_disk("data/raw/pascal")

# snli_ve = load_dataset('HuggingFaceM4/SNLI-VE')
# snli_ve.save_to_disk("data/raw/snli-ve")

# add SNLI VE when onto multimodal part of experiments -- later on