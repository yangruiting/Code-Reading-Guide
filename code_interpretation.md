# Code Structure

｜—— `README.md`  Description of the project.

｜—— `requirements.txt`  The third-party libraries used in this project. (Installation command: ***python3 -m pip install -r requirements.txt***)

｜—— `pretrain.py`  The execution file of the project. It includes the entire process of the project, including loading, training and saving data and models. Other Python files are called by it.

｜—— `pretrain.sh` Project execution command.

｜—— `tokenization.py` Includes methods related to token processing, including tokenizer training, token generation, and padding and truncation of token sequences, etc.

｜—— `model.py` Our transformer-based model, methods and algorithms related to model building are all here.

｜—— `optimization.py` Methods related to deep learning optimization, including optimizers and learning rate schedulers.

｜—— `data` The data folder that needs to be prepared before training, including training data `train.txt` and evaluation data `eval.txt`, and `vocab.json` and `merges.txt` are used to load the tokenizer (assuming they have been generated through training before).

｜—— `result`  The output folder named by the user includes the saved models and the results of the training process.

｜—— `__ pycache __`   A folder automatically generated after the Python project is run. 

｜   cpython means that the Python interpreter is implemented in C language, and -39 means the version is 3.9.     

｜—— `runs`  Automatically created to store information related to model training, logging, or experiment tracking.

# The role of the library

**1. apex**

***why use it?***    

During deep learning training, the data type defaults to single-precision FP32. In order to speed up training time and reduce the memory occupied by network training while maintaining the model accuracy, a mixed-precision training method has emerged. APEX is an open source tool from NVIDIA that perfectly supports the PyTorch framework and is used to change data formats to reduce the model's memory usage. **apex.amp** (Automatic Mixed Precision) tests most operations of the model using the Float16 data type, and some special operations still use Float32.

***How do we use it?***

```
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
```
The opt-level parameter is the user's choice of data format for training. Specifically, 

O0: pure FP32; O1: mixed precision training; O2: almost FP16; O3: pure FP16.

**2. mmengine**

***why use it?*** 

FLOPS (floating-point operations per second), floating-point operations per second. It is often used to estimate the performance of a computer. The parameter value reflects the usage of display memory. **mmengine.analysis.get_model_complexity_info** to get the complexity of a model.
A good network model not only requires high accuracy, but also requires a small number of parameters and computational complexity to facilitate deployment.

***How do we use it?***
```
from mmengine.analysis import get_model_complexity_info
outputs = get_model_complexity_info(model, input_shape)
```

**3. tokenizers**

Hugging Face's Tokenizers library provides a fast and efficient way to process natural language text (i.e., tokenization) for subsequent machine learning model training and reasoning. This library provides a variety of pre-trained tokenizers, such as BPE, Byte-Pair Encoding (Byte-Level BPE), WordPiece, etc., which are widely used tokenization methods in modern NLP models (such as BERT, GPT-2, RoBERTa, etc.).

For more information, please refer to: [Hugging-Face tokenizers](https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/index).

**4. tqdm**

***why use it?*** 

tqdm is a Python progress bar library. tqdm loads an iterable object and displays the loading progress of the iterable object in real time in the form of a progress bar.

***How do we use it?***

```
from tqdm import tqdm, trange
train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
```
_train_dataloader_ is the loaded iterator, and _desc_ is the description text of the progress bar.

**5. scikit_learn**

Scikitlearn is an essential library for machine learning. It provides a variety of algorithms. We mainly use it to calculate metrics. 

For more information, please refer to: [scikitlearn 中文文档](https://scikitlearn.com.cn/)

# How does the project work?


1. 


