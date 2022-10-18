# MLPSB-Parser
The code of paper "Multi-Layer Pseudo-Siamese Biaffine Model for Dependency Parsing" in COLING 2022.

## Getting Started
### Requirements
Install required packages with: 
```bash
pip install -r requirements.txt
```

### Dataset
The datasets are available at: (todo). Place them inside `corpus` folder.

To conduct experiments on specific dataset, modify `corp` in `src/dependency/config.py`. The value of `corp` can be selected in `["PTB", "CTB", "bg", "ca", "cs", "de", "en", "es", "fr", "it", "nl", "no", "ro", "ru"]`.

### Pre-trained Model
To use specific pre-trained model, place corresponding model folder inside `data` folder, and modify `pretrain_name` in `src/dependency/config.py` to name of the model folder. The pre-trained models we use are following: 

* bert-base(PTB): https://huggingface.co/bert-base-uncased

* bert-large(PTB): https://huggingface.co/bert-large-uncased

* XLNet-base(PTB): https://huggingface.co/xlnet-base-cased

* XLNet-large(PTB): https://huggingface.co/xlnet-large-cased

* bert-base-chinesee(CTB): https://huggingface.co/bert-base-chinese

* bert-base-multilingual(UD): https://huggingface.co/bert-base-multilingual-cased

## Usage
Go inside code folder with:
```bash
cd ./src/dependency_parsing
```
You can modify `device` in `dir.py` to determine the device you want to use. Train the model with:
```bash
python main.py
```
Evaluation on the test set happens automatically after training is complete. The results are saved in `result` folder. 

The models are saved in `model` folder. The training process can be interrupted, and it will start from checkpoint (the latest model and optimizer) next time. 

The program finds checkpoint by the name. Thus, if you want to restart training process from beginning, remove or rename corresponding model folder. 

You can evaluate the latest model on test set with:
```bash
python main.py test
```

## Detailed Setting

`src/dependency/config.py` includes experiment settings used in our paper. You can modify the setting values to conduct specific experiment. The setting values related to specific section in our paper are following:

* `biaff_layers`: the number of layers of biaffine model (section 3.6 and 4.1)

* `use_lstm`: whether to use LSTM (section 3.7)

* `attn_type`: attention function, can be selected in `["biaffine", "dot", "general", "concat"]` (section 4.2)

* `siamese`: siamese method, can be selected in `["P", "T", "N"]` (section 4.3)

* `test_limit`: limit the length of sentences to be tested (section 4.4)

* `use_gold_head`: whether to use gold head in inference (section 4.5)

* `time_check`: whether to check the time (section 4.7)

## Acknowledgements

* The code of pre-trained model embedding is based on [second-order-parser](https://github.com/wangxinyu0922/Second_Order_Parsing).
* The code of the eisner algorithm is based on [bist-parser](https://github.com/elikip/bist-parser).


