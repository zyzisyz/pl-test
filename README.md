# pl-test

## Prepare

```bash
pip install -r requirements.txt
ln -s /Source/Data data
```

## Fit

```bash
# CPU   
python main.py

# specific GPUs
python main.py --gpus '0,3'
python main.py --gpu '0'

# Multiple-GPUs   
python main.py --gpus 4

# On multiple nodes   
python main.py --gpus 4 --nodes 4  --precision 16
```
