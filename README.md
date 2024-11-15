# PRELIMINARY

### how to run one batch tests

```bash
python train.py -cn=onebatchtest model=conv-tasnet writer.run_name="conv-tasnet-one-batch-test"

python train.py -cn=onebatchtest model=av-conv-tasnet writer.run_name="av-conv-tasnet-one-batch-test"
```

### how to train models

```bash
python train.py -cn=conv-tasnet-train

python train.py -cn=av-conv-tasnet-train
```

### how to fine-tune models

```bash
python train.py -cn=conv-tasnet-finetune
```

### how to infer models
