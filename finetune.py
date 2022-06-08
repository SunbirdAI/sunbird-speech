import os 
import logging 
import os 
import re 
import sys 
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import string 

import datasets 
import numpy as np 
import torch 
import torchaudio 
from packaging import version 
from torch import nn 
from pathlib import Path 


import transformers 
from transformers import (
    HfArgumentParser, 
    Trainer, 
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    set_seed,
)

from transformers.trainer_utils imprt get_last_checkpoint, is_main_process 

if is_apex_available():
    from apex import amp 
    
if version.parse(torch.__version___) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast 

logger = logging.getLogger(__name__)

def list_field(default=None, metadata = None):
    return field(default_factory = lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining which model/config/tokenizer we are going to finetune
    """


@dataclass
class DataTrainingArguments: 
    """
    Arguments pertaining to what data we are going to inout our model for either training 
    or evaluation.
    """


@dataclass 
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs receiveds
    """
