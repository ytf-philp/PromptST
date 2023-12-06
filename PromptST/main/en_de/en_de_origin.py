import sys
import os
user_dir = os.path.expanduser("~")
local_library_path = os.path.join(user_dir, "PromptST")
sys.path.insert(0, local_library_path)
#sys.path.append(r'/data/ytf/PromptST') 
#sys.path.append(r'/data/ytf/PromptST/transformers') 
import transformers
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
    Speech2TextTokenizer
    
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
from transformers import SpeechEncoderDecoderModel, Speech2Text2Processor
import random
from torch.utils.data.distributed import DistributedSampler
import argparse
from datasets import load_from_disk
import datetime
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()
def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# distributed_step 1
# set random seed
set_random_seeds(random_seed=0)
# distributed_step 2
# set target device
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(
backend="nccl")


training_args = TrainingArguments(output_dir='random_trainer') # 指定输出文件夹，没有会自动创建

# Setup logging
logging.basicConfig(
format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
handlers=[logging.StreamHandler(sys.stdout)],
)

# set the main code and the modules it uses to the same log-level according to the node
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)


tokenized_datasets=load_from_disk("/data/ytf/PromptST/data_process/dataset_en_de")

print("........................stage3 prompt............................")

sampler = torch.utils.data.distributed.DistributedSampler(tokenized_datasets)

tokenizer = Speech2Text2Tokenizer.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
feature_extractor = Wav2Vec2FeatureExtractor(return_attention_mask=True)

processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
#model init
configg=SpeechEncoderDecoderConfig.from_pretrained("facebook/s2t-wav2vec2-large-en-de")


configg.hidden_dropout_prob = 0.1

model = SpeechEncoderDecoderModel.from_pretrained(
    "facebook/s2t-wav2vec2-large-en-de",config=configg

)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id =  model.decoder.config.decoder_start_token_id



sampler = torch.utils.data.distributed.DistributedSampler(tokenized_datasets)



model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

metric = load_metric("sacrebleu")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_pred = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(decoded_pred, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(len(decoded_preds)) # [list]
    print(len(decoded_preds[0]))
    print(len(decoded_labels)) #[list[list]]
    print(len(decoded_labels[0]))
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    re=pd.DataFrame({"label":decoded_labels,"pred":decoded_preds})
    re['label'].replace('', np.nan, inplace=True)
    re['label'].replace('TO REMOVE]', np.nan, inplace=True)
    re['pred'].replace('TO REMOVE]', np.nan, inplace=True)

    re=re.dropna().reset_index(drop=True)
    

    result = metric.compute(predictions=re["pred"], references=re["label"])
    result = {"bleu": result["score"]}
    #re.to_csv("/data/ytf/PromptST/result/en_de.csv")
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in decoded_pred]
    result["gen_len"] = np.mean(prediction_lens)

    total_lens=[np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
    result["total"] =np.mean(total_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

@dataclass
class DataCollatorWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    processor: Speech2Text2Processor
    tokenizer: Speech2TextTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features=[]
        label_features=[]
        for feature in features:
            #print(np.array(feature["labels"]).shape)
            input_features .append({"input_values": feature["inputs"][0]})
            label_features .append({"input_ids": feature["labels"]+[processor.tokenizer.eos_token_id]})

        batch = feature_extractor.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        labels_mask = labels_batch["attention_mask"]
        batch["inputs"] =batch["input_values"]
        batch["attention_mask"]=batch["attention_mask"]
        batch["labels"] = labels
        batch["decoder_mask"]=labels_mask
        del batch["input_values"]

        return batch

training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir="/data/ytf/tmp/model/de_continue_fix_encoder",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    label_smoothing_factor=0.1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    save_total_limit=10,
    num_train_epochs=400,
    eval_steps=500,
    dataloader_pin_memory=False,
    dataloader_num_workers=8,
    remove_unused_columns = True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    ignore_data_skip=True,
    eval_accumulation_steps=2,
    fp16=True,
)
#    resume_from_checkpoint="/workspace/yutengfei6/users/yutengfei6/docker-remote/train_model/results2/checkpoint-52500",
class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs,return_outputs=False):
        #print(inputs)
        input_values=inputs.get("inputs")
        input_ids=inputs.get("labels")
        attention_mask=inputs.get("attention_mask")
        decoder_mask=inputs.get("decoder_mask")
        '''for i in range(len(input_values)):'''
        outputs = model(inputs=input_values, labels=input_ids,attention_mask=attention_mask,decoder_attention_mask=decoder_mask)
        loss=outputs.loss
        if(return_outputs==True):
            encoder_outputs=modeling_outputs.BaseModelOutput(
            last_hidden_state=outputs.encoder_last_hidden_state,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
        )
            
            outputs["generate"]=model.module.generate(inputs=input_values,attention_mask=attention_mask,encoder_outputs=encoder_outputs)
            #outputs["generate"]=model.module.generate(inputs=input_values,attention_mask=attention_mask)
        #print(loss)
        return (loss, outputs) if return_outputs else loss

data_collator = DataCollatorWithPadding(processor=processor,tokenizer=tokenizer, padding=True,feature_extractor=feature_extractor)


trainer = CustomTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

#evaluate(trainer)
trainer.train(resume_from_checkpoint=False)





