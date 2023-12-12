import argparse
from distutils.command.config import config
import tokenizers
import torch
import random
import numpy as np
from re import S
from transformers import  SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel
from transformers import (
    SpeechEncoderDecoderModel,
    Wav2Vec2FeatureExtractor,
    Speech2Text2Tokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from transformers import modeling_outputs

from torch.utils.tensorboard import SummaryWriter   
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments,Seq2SeqTrainingArguments
import soundfile as sf
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset
import datasets
from transformers import Seq2SeqTrainingArguments
import numpy as np
import logging
import sys
import transformers
from transformers import SpeechEncoderDecoderModel, Speech2Text2Processor
import random
from torch.utils.data.distributed import DistributedSampler
import argparse

model = SpeechEncoderDecoderModel.from_pretrained("./model/de_continue/checkpoint-14500")

state_dict=torch.load("./model/de_continue/checkpoint-14500/pytorch_model.bin",map_location=torch.device('cpu'))
if list(state_dict.keys())[0].startswith('module.'):
    state_dict1 = {k[7:]: v for k, v in state_dict.items()}
    print("over")
model.load_state_dict(state_dict1)
model.save_pretrained("./model/de_all/fix_encoder")

