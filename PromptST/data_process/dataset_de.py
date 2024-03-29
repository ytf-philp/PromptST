from datasets import load_from_disk
import argparse
import numpy as np
from re import S
from transformers import (
    SpeechEncoderDecoderModel,
    Speech2Text2Processor,
    Speech2Text2Tokenizer,
    Wav2Vec2FeatureExtractor

)

import datasets

feature_extractor = Wav2Vec2FeatureExtractor()
tokenizers=Speech2Text2Tokenizer.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
data=datasets.load_dataset('covost2','en_de',data_dir="/data/ytf/en_new",ignore_verifications=True,split={"train":'train',"test":'test',"validation":'validation'})
def prepare_dataset(batch):
    audio = batch["audio"]
    result=processor(np.array(audio["array"]), sampling_rate=audio["sampling_rate"])
    batch["inputs"] = result.input_values
    batch["attention_mask"] = result.attention_mask
    with processor.as_target_processor():
        batch["labels"] = processor(batch["translation"]).input_ids
    return batch
tokenized_datasets=data.map(prepare_dataset,remove_columns=data.column_names["test"])
tokenized_datasets.save_to_disk("./PromptST/data_process/dataset_en_de")

