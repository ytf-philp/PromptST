# PromptST: Abstract Prompt Learning for End-to-End Speech Translation

This is an implementation of the EMNLP 2023 paper *"PromptST: Abstract Prompt Learning for End-to-End Speech Translation"* (read the paper [here](https://openreview.net/pdf?id=Nijnhwu1Uz)).

## 👀 Overview

The motivation of our **PromptST** model is to broaden the abstract representation power of the encoder of S2T models.

<div align="left">
  <img src="https://github.com/ytf-philp/PromptST/assets/54100491/13025542-33c2-4f9b-b22e-3758c088f769" width="70%">
</div>

### Result on CoVoST En-X dataset

We report **case-sensitive detokenized BLEU** via the sacrebleu toolkit.

| Model          | En-De | En-Ca | En-Ar | En-Tr | Avg. |
| -------------- | :---: | :---: | :---: | :---: | :--: |
| Continue Train | 25.9 | 33.3 | 19.3 | 17.6 | 24.0 |
| PromptST       | 26.4 | 33.7 | 19.6 | 17.9 | 24.4 |

The BLEU score of adding PromptST to different layers on the dev set.

| Model        | En-De | En-Ca | En-Ar | En-Tr |
| ------------ | :---: | :---: | :---: | :---: |
| 0-24 layers  | 29.9 | 36.7 | 23.5 | 20.4 |
| 20-24 layers | 29.8 | 36.8 | 23.4 | 20.6 |
| 16-24 layers | 29.9 | 36.5 | 23.7 | 20.5 |
| 12-24 layers | 30.1 | 37.4 | 23.8 | 21.0 |

If you find this repo useful, please cite:

```
@inproceedings{yu-etal-2023-promptst,
    title = "{P}rompt{ST}: Abstract Prompt Learning for End-to-End Speech Translation",
    author = "Yu,Tengfei and Ding,Liang  and Liu,Xuebo and Chen,Kehai and Zhang,Meishan and Tao,Dacheng and Zhang,Min",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.627",
    pages = "10140--10154",
}
```

## Speech-Senteval Benchmark

  You can download the Speech-Senteval Benchmark at [Here](https://drive.google.com/file/d/1F-uXapnR1nJ1q81u1quBRtMv_-xJ564i/view?usp=share_link)

## ⬇️ Download Trained Models

The models are trained based on pytorch.

|      |                      **Model**                      |
| :---: | :-------------------------------------------------------: |
| En-De | [Download](https://huggingface.co/philip-xxf/PromptST-en-de) |
| En-Ca | [Download](https://huggingface.co/philip-xxf/PromptST-en-ca) |
| En-Ar | [Download](https://huggingface.co/philip-xxf/PromptST-en-ar) |
| En-Tr | [Download](https://huggingface.co/philip-xxf/PromptST-en-tr) |

## Training & Generation Instruction

### ⚙️ Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* transformers  == 4.27.3
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

```bash
git clone git@github.com:ytf-philip/PromptST.git
cd PromptST
pip3 install -r requirements.txt
```

### 🧪 Probing Task Analysis
*  Download [Speech-Senteval](https://drive.google.com/file/d/1F-uXapnR1nJ1q81u1quBRtMv_-xJ564i/view?usp=share_link)
*  Unzip speech_senteval and save it to the root/data path (the unzipped dataset should contain two folders, "probing_text" and "sent_audio")
*  Extract every layer representation and conduct probing tasks
  
Run the probing task script:
```
bash probing_task/bash_example.sh
```

### 🚀 Train PromptST (Example: en-de)

**Preprocessing Data**
* Download Common Voice audio clips and transcripts (version 4).
* Use data_process/dataset_de.py to save the processed dataset offline.
```
python data_process/dataset_de.py
```
**Training**
* To train the model, take En-De as an example; you may run:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 21303  ./PromptST/main/en_de/en_de_continue.py
```
**Evaluation**
* Convert model
```
python  ./main/model_convert.py
``` 
* Inference
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port 21393 --model_path ./output/model_convert ./main/en_de/en_de_inference.py
``` 
### Contact
If you have any questions related to the code or the paper, feel free to email Tengfei Yu (921692739@qq.com).
