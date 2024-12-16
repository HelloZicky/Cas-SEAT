import argparse
import io
import logging
import os
import sys
import math
import numpy as np
from datasets import load_dataset
from openai import AzureOpenAI
from rich.logging import RichHandler
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from build_query import create_query_data
from utilities import read_json, save_json
from PIL import Image
from openai import OpenAI
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default='Math360k')
    parser.add_argument('--test_split_name', type=str, default='testmini')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='mathv30kqwen2vl7bnewprompt.json')
    parser.add_argument('--max_num_problems', type=int, default=1000, help='The number of problems to run')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    # Local Model
    parser.add_argument('--model_path', type=str, default='qwen2vl')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2vl")
    # Remote model
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-3.5-turbo',
        help='llm engine',
        choices=['gpt-3.5-turbo', 'claude-2', 'gpt4', 'gpt-4-0613', 'bard'],
    )
    parser.add_argument('--key', type=str, default='', help='key for llm api')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()
    logging.info(f"Loading dataset {args.dataset_name}, split {args.test_split_name}...")
    # load data
    if args.dataset_name == "AI4Math/MathVista":
        
        print(True)
        data_list = load_dataset("AI4Math/MathVista", split=args.test_split_name)
        # Convert Hugging Face data into dictionary to match local data format
        # TODO: Convert scripts not to depend on dictionary .json format. Update to use .jsonl format
        data = {item['pid']: item for item in data_list}
        args.query_file = None
        if args.query_file:
            query_file = os.path.join(args.data_dir, args.query_file)
            if os.path.exists(query_file):
                logging.info(f"Loading existing {query_file}...")
                query_data = read_json(query_file)
        else:

            logging.info("Creating new query...")
            args.use_ocr = False
            args.use_caption = False
            caption_data = {}
            if args.use_caption:
                caption_file = args.caption_file
                if os.path.exists(caption_file):
                    logging.info(f"Reading {caption_file}...")
                    try:
                        caption_data = read_json(caption_file)["texts"]
                        logging.info("Caption data loaded.")
                    except Exception as e:
                        logging.info("Caption data not found!! Please Check.")
    # load or create query data
    if args.dataset_name == "Math360k":
        data_list = load_dataset("Zhiqiang007/MathV360K", split="train")
        
        data = {item['id']: item for item in data_list}
        query_data = data
    # If we were given a custom model path, load that model, otherwise use a remote service model
    if args.model_path:
        if args.model_name == "llavav1.5":
        # from models import llava
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            model_id = args.model_path
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
                device_map = "auto",
                use_flash_attention_2=True
            )
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        if args.model_name == "qwen2vl":
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, device_map="auto",  attn_implementation="flash_attention_2",
)
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        if args.model_name == "internvl":
            path = 'OpenGVLab/InternVL2-40B'
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            
            model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto").eval()
        if args.model_name == "llavanext7b":
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")            
            model = LlavaNextForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()

    else:
        model_name = args.azure_openai_model if args.azure_openai_model else args.model
        logging.info(f"Loading {model_name}...")

    logging.info(f"Model loaded.")

    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}

    skip_pids = []


    if len(skip_pids) > 0:
        logging.info(
            f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
        )

    test_pids = [pid for pid in full_pids if pid not in skip_pids]
    print("test_pid",len(test_pids))
    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.warning(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")
    print("lentestpid",len(test_pids))

    for i, problem_id in enumerate(tqdm(test_pids)):
        
        problem: dict = data[problem_id].copy()

        # Remove decoded Image for JSON deserialization
        if args.dataset_name == "AI4Math/MathVista":
            problem_decoded_image = problem['decoded_image']
            problem.pop('decoded_image')
            image_path = problem['image']

            query = query_data[problem_id]
            system_prompt = f'''This is a mathematical question about an image and the direct answer to this question: \n {query} \n You are a reasoner tasked with solving problems step by step. You need to give the question's step by step reasoning. Each step needs to include objective information observed from the image and the corresponding reasoning process. Follow these instructions:
1.Self-Evaluation:
     Continuously monitor for any uncertainty or potential errors during reasoning, especially the parts related to the image. If you detect any doubt on reasoning process, computational steps, image observation, or the approach to complex questions, immediately enter a “self-evaluation” phase using this format:
     [self-evaluation-start:
     (Your detailed reflection here, let's verify step by step)
     Evaluation: Accept/Refine
     self-evaluation-end]
     If the result is Evaluation: Refine, refine the reasoning as needed.
     If the result is Evaluation: Accept, proceed with the current solution.

2.Final Result:
     Once the solution is complete, present the final result in this format:
     The answer is [final answer]

Ensure the reasoning process is accurate, and invoke self-evaluation whenever necessary.\n'''
        try:
            if args.model_name == "llavav1.5":
                conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": system_prompt},
            {"type": "image"},
            ],
        },
    ]           
               # print(conversation)
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=problem_decoded_image, text=prompt, return_tensors='pt').to('cuda')

                output = model.generate(**inputs, max_new_tokens=500, do_sample=False)
                response = processor.decode(output[0][2:], skip_special_tokens=True)
                index = response.find("ASSISTANT:")
              #  print(response[index:])
                response = response[index:]
                #print(response)
            if args.model_name == "llavanext7b":
                conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": system_prompt},
          {"type": "image"},
        ],
    },
]
                
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=problem_decoded_image, text=prompt, return_tensors="pt").to('cuda')
                output = model.generate(**inputs, max_new_tokens=500)
                response = processor.decode(output[0], skip_special_tokens=True)

                index = response.find("ASSISTANT:")
                # print(response[index:])
                response = response[index:]
            if args.model_name == "qwen2vl":
                messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image":image_path,
            },
            {"type": "text", "text": system_prompt},
        ],
    }
]

# Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=800)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                if type(response) == list:
                    response = response[0]
            if args.model_name == "internvl":
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens = 2048, do_sample=True)
                response = model.chat(tokenizer, pixel_values,  system_prompt, generation_config)
            if args.model_name == "oai":
                client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
                image_path = image_path
                messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    },
                },
                {"type": "text", "text": query},
            ],
        },
    ]
                result = client.chat.completions.create(messages=messages, model=args.model_name)
                print(result.choices[0].message.content)
                response = result.choices[0].message.content
            results[problem_id] = problem
            results[problem_id]['query'] = query
            # logging.info(response)
            # logging.info(results)
            if args.shot_type == 'solution':
                results[problem_id]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[problem_id]['response'] = response
                results[problem_id]['execution'] = output
                results[problem_id]['error'] = str(error)
            logging.debug(f"Query: \n{query}")
            logging.debug(f"Response: \n{response}")
        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id] = problem
            results[problem_id]['error'] = str(e)

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
            except Exception as e:
                logging.info(f"Error in saving {output_file_path}")
                logging.info(e)

    logging.info("MathVista: Generating Responses - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
