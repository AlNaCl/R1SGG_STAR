# STAR Closed-Vocab Retrain Quickstart

## 1) Generate mapping templates

```bash
source /home/ly25/yes/etc/profile.d/conda.sh
conda activate r1sgg

python /mnt/dataY/ly25/R1-SGG/scripts/data/generate_star_mapping_templates.py \
  --star_dict /mnt/dataY/ly25/datasets/STAR/STAR_SGG_Annotation/STAR-SGG-dicts-with-attri.json \
  --output_dir /mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps
```

Edit:

- `/mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps/obj_star2r1.json`
- `/mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps/pred_star2r1.json`

Use `__UNK__` for labels you want to drop.

## 2) Remap STAR jsonl to closed vocab

```bash
python /mnt/dataY/ly25/R1-SGG/scripts/data/remap_star_jsonl_closed_vocab.py \
  --input_dir /mnt/dataY/ly25/R1-SGG/datasets/star_r1sgg_jsonl \
  --output_dir /mnt/dataY/ly25/R1-SGG/datasets/star_r1sgg_jsonl_closed \
  --obj_map /mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps/obj_star2r1.json \
  --pred_map /mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps/pred_star2r1.json
```

## 3) Validate mapping coverage

```bash
python /mnt/dataY/ly25/R1-SGG/scripts/data/validate_star_closed_vocab.py \
  --jsonl_dir /mnt/dataY/ly25/R1-SGG/datasets/star_r1sgg_jsonl \
  --obj_map /mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps/obj_star2r1.json \
  --pred_map /mnt/dataY/ly25/R1-SGG/datasets/star_closed_vocab_maps/pred_star2r1.json
```

## 4) Small-scale training smoke test (example)

Use your existing SFT script, but point dataset path to the remapped closed-vocab data.

- `PredCls` first
- then `SGCls`
- then `SGDet`

Keep `batch_size` small first and verify output format + closed-vocab hit rate before full training.
