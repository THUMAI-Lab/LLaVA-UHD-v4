import math
import torch
import random
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class MiniCPM_V(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt = 'Study the image carefully and pick the option associated with the correct answer. \
                Focus solely on selecting the option and avoid including any other content.\n' + prompt
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])

        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': prompt}]
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 20
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 100
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams
        )
        default_kwargs.update(self.kwargs)
        res, _, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        return res


class MiniCPM_Llama3_V(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-Llama3-V-2_5', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3
        self.options_system_prompt = ('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'VQA'], DATASET_TYPE(dataset)):
            return True
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            return True
        return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt = ''

        question = line['question']
        if DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                system_prompt = self.options_system_prompt + '\nPlease just indicate your choice.'
            else:
                system_prompt = self.wo_options_system_prompt
            if 'MMMU' in dataset:  # Corner Case
                prompt = system_prompt + '\n' + prompt
                system_prompt = ''
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['MME'], dataset):
            question = line['question'] + ' Yes or No?'
            prompt = question
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            system_prompt = self.vqa_prompt
            question = line['question']
            prompt = question
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['LLaVABench', 'MMLongBench_DOC'], dataset):
                system_prompt = ''
                prompt = question
            elif listinstr(['MMVet'], dataset):
                system_prompt = self.detail_system_prompt
                prompt = question
            else:
                system_prompt = self.vqa_prompt
                prompt = question

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def generate_inner(self, message, dataset=None):
        if DATASET_TYPE(dataset) == 'MCQ':
            max_new_tokens = 200
        elif DATASET_TYPE(dataset) == 'Y/N':
            max_new_tokens = 3
        else:
            max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        content = []
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res

    def chat_inner(self, message, dataset=None):
        max_new_tokens = 1024

        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=self.num_beams,
        )
        default_kwargs.update(self.kwargs)

        msgs = []
        for msg in message:
            content = []
            if len(msg['content']) == 1 and msg['content'][0]['type'] == 'text':
                msg_new = {'role': msg['role'], 'content': msg['content'][0]['value']}
                msgs.append(msg_new)
                continue

            for x in msg['content']:
                if x['type'] == 'text':
                    content.append(x['value'])
                elif x['type'] == 'image':
                    image = Image.open(x['value']).convert('RGB')
                    content.append(image)
            msg_new = {'role': msg['role'], 'content': content}
            msgs.append(msg_new)

        res = self.model.chat(
            msgs=msgs,
            context=None,
            image=None,
            tokenizer=self.tokenizer,
            **default_kwargs)

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        return res


class MiniCPM_V_2_6(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    _sampling_params_printed = False

    def __init__(self, model_path='openbmb/MiniCPM-V', use_cot=False, use_upsize=False, **kwargs):
        '''
        model_path 可能传入包含 safetensors 的路径，也可能是一个 pt
        在调试时，加上 use_cot 和 use_upsize 两个参数，因为有些 benchmark 还不清楚哪种 setting 最好
        '''
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        
        if '.pt' in model_path:
            base_model_path = os.getenv('model_path') or os.getenv('MODEL_PATH')
            if base_model_path:
                self.model_path = base_model_path
            else:
                raise ValueError(
                    'Loading from .pt checkpoint requires base model path. '
                    'Please set `model_path` or `MODEL_PATH` environment variable.'
                )
            self.ckpt = model_path
            
            print(f'load from path {self.model_path}')
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        
            print(f'load from ckpt {self.ckpt}')
            self.state_dict = torch.load(self.ckpt, map_location='cpu')
            self.model.load_state_dict(self.state_dict, strict=False)
        else:
            if os.getenv('model_path', None):
                self.model_path = os.getenv('model_path')
                print(f'load from path {self.model_path}')
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
                self.ckpt = model_path
                print(f'load from ckpt {self.ckpt}')
                tmp_model = AutoModel.from_pretrained(self.ckpt, trust_remote_code=True)
                self.state_dict = tmp_model.state_dict()
                self.model.load_state_dict(self.state_dict, strict=False)
            else:
                self.model_path = model_path
                print(f'load from path {self.model_path}')
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

        
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()

        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        
        num_beams = int(os.getenv("NUM_BEAMS", "3"))

        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else num_beams
        print("Beam size =", self.num_beams)
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))


        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question. Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')
        
        self.ocrbench_cot_prompt = 'Carefully observe the image and answer the OCR-related questions below. \n\n' #If the question requires carefully thinking, you can break it down step by step.'

        self.use_cot = use_cot
        self.use_upsize = use_upsize

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot_func(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'HallusionBench', 'OCRBench', 'ChartQA'], dataset):
            return True
        elif listinstr(['MathVista', 'MMVet', 'MMBench', 'MMStar', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize_func(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMVet', 'MMBench', 'MMStar', 'AI2D', 'OCRBench'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot:
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                elif listinstr(['OCRBench'], dataset):
                    prompt = self.ocrbench_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            msgs = MMMUDataset.split_MMMU(msgs)

        return msgs

    def generate_inner(self, message, dataset=None):
        # if self.use_cot:
        #     max_new_tokens = 2048
        # else:
        #     max_new_tokens = 1024
        
        use_sampling = os.getenv("SAMPLING", "false").lower() == "true"
        temperature = float(os.getenv("TEMPERATURE", "1.0"))
        top_p = float(os.getenv("TOP_P", "1.0"))
        top_k = int(os.getenv("TOP_K", "0"))
        repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.0"))

        if use_sampling:
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                sampling=True,
                # temperature=temperature,
                # top_p=top_p,
                # top_k=top_k,
                # repetition_penalty=repetition_penalty,
                num_beams=1,  
            )
        else:
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                sampling=False,
                num_beams=self.num_beams,
            )
        
        default_kwargs.update(self.kwargs)
        if not MiniCPM_V_2_6._sampling_params_printed:
            print(f"default_kwargs: {default_kwargs}")
            MiniCPM_V_2_6._sampling_params_printed = True


        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize:
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        return res



class MiniCPM_4_V(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    _sampling_params_printed = False

    def __init__(self, model_path='openbmb/MiniCPM-V', **kwargs):
        '''
        model_path 可能传入包含 safetensors 的路径，也可能是一个 pt
        在调试时，加上 use_cot 和 use_upsize 两个参数，因为有些 benchmark 还不清楚哪种 setting 最好
        '''
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        
        if '.pt' in model_path:
            base_model_path = os.getenv('model_path') or os.getenv('MODEL_PATH')
            if base_model_path:
                self.model_path = base_model_path
            else:
                raise ValueError(
                    'Loading from .pt checkpoint requires base model path. '
                    'Please set `model_path` or `MODEL_PATH` environment variable.'
                )
            self.ckpt = model_path
            
            print(f'load from path {self.model_path}')
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        
            print(f'load from ckpt {self.ckpt}')
            self.state_dict = torch.load(self.ckpt, map_location='cpu')
            self.model.load_state_dict(self.state_dict, strict=False)
        else:
            if os.getenv('model_path', None):
                self.model_path = os.getenv('model_path')
                print(f'load from path {self.model_path}')
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
                self.ckpt = model_path
                print(f'load from ckpt {self.ckpt}')
                tmp_model = AutoModel.from_pretrained(self.ckpt, trust_remote_code=True)
                self.state_dict = tmp_model.state_dict()
                self.model.load_state_dict(self.state_dict, strict=False)
            else:
                self.model_path = model_path
                print(f'load from path {self.model_path}')
                self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

        
        self.model = self.model.to(dtype=torch.bfloat16)
        self.model.eval().cuda()

        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        
        num_beams = int(os.getenv("NUM_BEAMS", "3"))

        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else num_beams
        print("Beam size =", self.num_beams)

        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "2048"))

        self.options_suffix_prompt = '''\nAnswer with the option's letter from the given choices directly.'''
        self.wo_options_system_prompt = 'Carefully read the following question. Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'

        self.multi_choice_cot_prompt = ('''Carefully read the following multichoice question, solve it step '''
                                        '''by step and finally pick the option associated with the correct '''
                                        '''answer in the format of "Answer: selected option\n\n''')
        self.short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')
        
        self.ocrbench_cot_prompt = 'Carefully observe the image and answer the OCR-related questions below. \n\n' #If the question requires carefully thinking, you can break it down step by step.'
        
        self.printed_datasets = set()


    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MCQ', 'VQA', 'Y/N'], DATASET_TYPE(dataset)):
            return True
        return False

    def use_cot_func(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'MathVista', 'MMStar', 'OCRBench', 'ChartQA'], dataset):
            return True
        elif listinstr(['MMVet', 'MMBench', 'HallusionBench', 'AI2D', 'RealWorldQA',
                        'POPE', 'ScienceQA', 'TextVQA', 'DocVQA'], dataset):
            return False
        else:
            return False

    def use_upsize_func(self, dataset=None):
        if dataset is None:
            return False
        if listinstr(['MMMU', 'MathVista', 'MMStar', 'OCRBench', 'ChartQA'], dataset):
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        if dataset and dataset not in self.printed_datasets:
            use_cot = self.use_cot_func(dataset)
            use_upsize = self.use_upsize_func(dataset)
            print("-" * 50)
            print(f"Setting of Dataset: {dataset}")
            print(f"   ─ CoT = {'yes' if use_cot else 'no'}")
            print(f"   ─ Upsize = {'yes' if use_upsize else 'no'}")
            print("-" * 50)
            self.printed_datasets.add(dataset)
        
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = '', ''

        question = line['question']

        if not self.use_cot_func(dataset):
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = 'Options:\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'Question: {question}\n'
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            elif dataset is not None and listinstr(['HallusionBench'], dataset):
                question += ' Yes or No?'
                prompt = question
            elif dataset is not None and listinstr(['OCRBench'], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif DATASET_TYPE(dataset) == 'VQA':
                if listinstr(['LLaVABench'], dataset):
                    system_prompt = ''
                elif listinstr(['MMVet'], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if DATASET_TYPE(dataset) == 'MCQ':
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ''
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'
                hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
                if hint is not None:
                    prompt += f'Hint: {hint}\n'
                prompt += f'{question}\n'

                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if 'MMMU' in dataset:
                    if len(system_prompt) > 0:
                        prompt = system_prompt + '\n' + prompt
                        system_prompt = ''
            else:
                prompt = question

            if DATASET_TYPE(dataset) in ['MCQ', 'Y/N', 'VQA']:
                if DATASET_TYPE(dataset) == 'MCQ':
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif DATASET_TYPE(dataset) == 'Y/N':
                    prompt = self.short_ans_cot_prompt + prompt
                elif listinstr(['OCRBench'], dataset):
                    prompt = self.ocrbench_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type='text', value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        if dataset.startswith('MMMU_'):
            from .. import MMMUDataset
            msgs = MMMUDataset.split_MMMU(msgs)

        return msgs

    def generate_inner(self, message, dataset=None):
        # if self.use_cot_func(dataset):
        #     max_new_tokens = 2048
        # else:
        #     max_new_tokens = 1024
        
        use_sampling = os.getenv("SAMPLING", "false").lower() == "true"
        temperature = float(os.getenv("TEMPERATURE", "1.0"))
        top_p = float(os.getenv("TOP_P", "1.0"))
        top_k = int(os.getenv("TOP_K", "0"))
        repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.0"))

        if use_sampling:
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                sampling=True,
                # temperature=temperature,
                # top_p=top_p,
                # top_k=top_k,
                # repetition_penalty=repetition_penalty,
                num_beams=1,  
            )
        else:
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                sampling=False,
                num_beams=self.num_beams,
            )
        
        default_kwargs.update(self.kwargs)
        if not MiniCPM_4_V._sampling_params_printed:
            print(f"default_kwargs: {default_kwargs}")
            MiniCPM_4_V._sampling_params_printed = True


        content = []

        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                if not self.use_upsize_func(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        resized_image = image.resize((new_img_width, new_img_height))
                        content.append(resized_image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            max_inp_length=8192,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        return res
