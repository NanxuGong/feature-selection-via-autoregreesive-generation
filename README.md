# feature_selection_via_autoregreesive-generation
This is the implement code of the paper "Neuro-Symbolic Embedding for Short and Effective Feature Selection via Autoregressive Generation"

## Inplement
### Step 1: download the data: 
```
follow the instruction in /data/readme.md
```

### Step 2: collect the training data
```
python3 xxx/code/baseline/automatic_feature_selection_gen.py --name DATASETNAME --choice REDUNDANCY_CHOICE --unsupervised IS_UNSUPERVISED
```
### Step 3: generate the optimal feature subset
```
python3 xxx/code/ours/train_controller.py ---method_name MODEL_CONFIGURATION --task_name DATASETNAME --gen_num GENERATION_SET_NUM --batch_size 64 --epochs 1000 --lr 0.0001
```
