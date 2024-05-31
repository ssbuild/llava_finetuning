# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
from enum import Enum
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class DataStrategy(Enum):
    tunction = 1






def build_template_default(query,prefix=None):
    prompt = prefix or ''

    prompt += query
    return prompt


#切换模板
build_template = build_template_default


class TokenIdsMaker:
    @classmethod
    def final(cls, tokenizer, input_ids, image_path, max_seq_length):
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray([1] * len(input_ids), dtype=np.int32)

        if pad_len:
            pad_val = tokenizer.eos_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_path': np.asarray(bytes(image_path,encoding='utf-8')),
            'seqlen': seqlen
        }
        return d
    @classmethod
    def tunction(cls, tokenizer: PreTrainedTokenizer, config, sup, max_seq_length, examples):
        sptoken = [config.bos_token_id]
        ds = []
        prefix,text,image_path = examples
        input_ids = tokenizer.encode(text=build_template(text, prefix=prefix),truncation=True,max_length=max_seq_length)
        input_ids = sptoken + input_ids

        assert len(input_ids) <= max_seq_length
        ds.append(cls.final(tokenizer, input_ids, image_path, max_seq_length))
        return ds


