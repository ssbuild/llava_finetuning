# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import config_args, NN_DataHelper, get_deepspeed_config
from deep_training.zoo.model_zoo.vision2seq.llm_model import MyTransformer
from deep_training.zoo.generator_utils.generator_llava import Generate
from PIL import Image

deep_config = get_deepspeed_config()




if __name__ == '__main__':


    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()


    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype)
    model = pl_model.get_llm_model()
    model = model.eval()
    if hasattr(model,'quantize'):
        # 支持llama llama2量化
        if not model.quantized:
            # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
            model.half().quantize(4).cuda()
            # 保存量化权重
            # model.save_pretrained('llama2-7b-chat-int4',max_shard_size="2GB")
            # exit(0)
        else:
            # 已经量化
            model.half().cuda()
    else:
        model.half().cuda()

    image_processor = dataHelper.processor
    model_genetor = Generate(model, tokenizer, image_processor)

    text_list = [ "图中是一只拉布拉多犬", ]
    image = Image.open("../assets/demo.jpeg")
    for input in text_list:
        response = model_genetor.generate(query=input, image=image, max_length=512,
                                          eos_token_id=config.eos_token_id,
                                          do_sample=False, top_p=0.7, temperature=0.95, )
        print('input', input)
        print('output', response)


