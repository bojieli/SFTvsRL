"""
Here we define the class for llama 3.2 evaluator
Updated: Oct-10-2024
By Tianzhe
"""

from evaluation.evaluator.base_evaluator import BaseEvaluator
from utils_mllm import evaluate_model_config
from utils_general import re_match
from typing import Optional, Dict
from PIL import Image
from openai import OpenAI
import base64
from io import BytesIO

def encode_pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change format to PNG if needed
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
class APIEvaluator(BaseEvaluator):
    def __init__(self, API_KEY, API_URL,
                 action_space, daytime, env_config, model, model_path, prompt_config, generation_config, output_dir, seed=42, **kwargs):
        super(APIEvaluator, self).__init__(action_space, daytime, env_config, prompt_config, generation_config, output_dir, seed, **kwargs)
        """
        Note: in practice, we find deepspeed accelerates the evaluation, issue me if you have better implementation.
        """
        # self.processor, model = evaluate_model_config(model, model_path)
        self.api_key = API_KEY
        self.api_url = API_URL
        print(f"API Evaluator initialized with API_KEY: {self.api_key}, API_URL: {self.api_url}")
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.api_url,
        )
        # assert False
    
    def formulate_payload(self, question, obs = None):
        """
        Goal:
        formulate processor readable prompts for llama

        Input:
            question: string
            obs: PIL.Image (optional)
        Output:
            in-place modification on self.payload
        Usage:
            self.processor(self.payload)
        """
        self.payload = [
            {
            "role": "user",
            "content": [{"type": "text", "text": question}]
            }
        ]
        if obs is not None:
            # append image to payload
            base64_image = encode_pil_to_base64(obs)
            self.payload[0]['content'].insert(0, {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }})

    
    def formulate_prompt(self, vision_res_dict, language_res_dict, prompt = None, obs = None, info = None):
        """
        Goal:
        aggregate all prompts and verification information

        Input:
            vision_res_dict, language_res_dict: dictionaries that possibly contain arguments
            prompt: string, prompt template
            obs: PIL.Image (optional)
            info: dictionary, contains step-wise information used to judge if it's the first step of a trajectory
        Output:
            in-place modification on self.payload
        """
        # task specific implementation
        if 'gym_cards' in self.id:
            # we first judge if it's the first step of a trajectory
            if info['Verify Info'] is not None:
                self.payload.append({"role": "user", "content": [{"type": "text", "text": f"You failed this trial because {info['Verify Info']}"}]})
            else:
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
        elif 'gym_virl' in self.id:
            # similarly, we first judge if it's the first step of a trajectory
            if info['Verify Info'] is None or 'Correct action' in info['Verify Info'] or 'step_limit_reached"' in info['Verify Info']:
                
                question = prompt.format(**vision_res_dict, **language_res_dict, **self.oracle_arguments)
                self.formulate_payload(question, obs)
            else:
                self.payload.append({"role": "user", "content": [{"type": "text", "text": f"You failed this trial because {info['Verify Info']}"}]})

    def generate_one_response(self, vision_res_dict, language_res_dict, obs = None, info = None):
        """
        Goal:
        core function that is used to generate response of the MLLM

        Input:
            vision_res_dict, language_res_dict: dictionaries that possibly contain arguments
            prompt: string, prompt template
            obs: PIL.Image (optional)
            info: dictionary, contains step-wise information used to judge if it's the first step of a trajectory
        Output:
            in-place modification on language_res_dict, self.payload
        """
        # model specific implementation
        self.formulate_vision_arguments(vision_res_dict, info)
        # only one step in this setting
        if obs is not None:
            prompts, patterns = self.prompt_vision, self.pattern_vision
            try:
                obs = Image.fromarray(obs)
            except:
                pass
        else:
            prompts, patterns = self.prompt_language, self.pattern_language
        for prompt, pattern in zip(prompts, patterns):
            self.formulate_prompt(vision_res_dict, language_res_dict, prompt, obs, info)
            # print(self.payload)
            # assert False
            completion = self.client.chat.completions.create(
                model=self.model,  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=self.payload,
                max_tokens=self.generation_config.max_tokens,
            )
            outputs = completion.choices[0].message.content
            print(outputs)
            # assert False
            # input_text = self.processor.apply_chat_template(self.payload, add_generation_prompt=True)
            # inputs = self.processor(obs, input_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            # output_ids = self.model.generate(**inputs, **self.generation_config) # like max_new_tokens and temperature stuff
            # outputs = self.processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            language_res_dict[pattern] = outputs
            self.append_intermidiate_res(outputs)
    
    def append_intermidiate_res(self, res):
        # task specific implementation
        self.payload.append({"role": "assistant", "content": [{"type": "text", "text": res}]})
    
    def extract_final_action(self, language_res_dict, key_pattern=None):
        """
        Goal:
        extract the key information in language_res_dict for further evaluation, env.step, ...
        Input:
            language_res_dict: dictionary, contains model's language output
        Output:
            language_res_dict[KEY_RESULT]: string, that is used for next steps

        Note that it's a legacy implementation, and can be simplified. 
        """
        # task specific implementation
        try:
            return language_res_dict[key_pattern]
        except:
            if 'gym_cards' in self.id:
                return language_res_dict['formula']
            else:
                return language_res_dict['action']