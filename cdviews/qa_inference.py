import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re
from qa_utils import load_and_update, get_scanqa, get_sqa, custom_collate_fn
from dataset import ViewLabelDataset
from torch.utils.data import DataLoader
from ViewSelector import ViewSelector, BCELoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from view_distance_calculation import calculate_view_distance
import pandas as pd
from view_sampling import question_seed, uniform_views


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048,
                    system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    # if roles[source[0]["from"]] != roles["human"]:
    #     source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def split_list(input_list, chunk_size=50):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def ranking_views(pair_dataloader, tokenizer, model, view_selector, save_path, chunk_size=50):
    view_selector.eval()
    output_dict = {}

    with torch.no_grad():
        for qs_id_list, qs_list, image_embeds, labels, image_file_list in tqdm(pair_dataloader):
            text_embeds_list = []
            for qs, image_files in zip(qs_list, image_file_list):
                line = {'from': 'human',
                        'value': qs}
                input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=False).to(model.device)
                text_embed = model.get_model().embed_tokens(input_ids)
                text_embeds_list += [text_embed.squeeze(0), ] * len(image_files)

            padded_text_embeds = pad_sequence(text_embeds_list, batch_first=True).float()

            image_embeds = image_embeds.float()
            if len(image_files) < chunk_size:
                text_embeds, image_embeds = view_selector(image_embeds.to(model.device), padded_text_embeds)
                scores = F.cosine_similarity(text_embeds, image_embeds)
            else:
                text_embeds_list = split_list(padded_text_embeds)
                image_embeds_list = split_list(image_embeds)
                scores_list = []
                for text_embeds, image_embeds in zip(text_embeds_list, image_embeds_list):
                    text_embeds, image_embeds = view_selector(image_embeds.to(model.device), text_embeds)
                    scores = F.cosine_similarity(text_embeds, image_embeds)
                    scores_list.append(scores)
                scores = torch.cat(scores_list)

            paired = list(zip(scores, image_files))
            paired.sort(key=lambda x: x[0], reverse=True)

            sorted_scores, sorted_image_files = zip(*paired)
            qs_id = str(qs_id_list[0])
            output_dict[qs_id] = list(sorted_image_files)
    json.dump(output_dict, open(save_path, 'w'))


def viewNMS(list_of_images, neighbour_df, num_images, distance_threshold=0.5):
    selected_images = []
    remaining_images = list_of_images.copy()

    while len(selected_images) < num_images and remaining_images:
        current_image = remaining_images.pop(0)
        selected_images.append(current_image)

        sorted_distances = neighbour_df.loc[current_image].sort_values()
        filtered_images = sorted_distances[sorted_distances < distance_threshold]
        neighbours_to_remove = set(filtered_images.index.tolist())
        remaining_images = [img for img in remaining_images if img not in neighbours_to_remove]

    return selected_images

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.LVLM_ckpt)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # Data
    print(f"View selection: {args.view_selection}")
    test_mode = ['test_w_obj', 'test_wo_obj'] if args.dataset == 'ScanQA' else ['test', ]

    for mode in test_mode:
        save_rank_file = args.ranked_view_file.format(args.dataset, mode)
        if args.view_selection == 'retrieval':
            save_rank_file = args.retrieval_ranked_view_file.format(args.dataset, mode)

        # Cached full rankings also define the shared candidate pool for uniform
        # sampling. Loading them needs neither selector weights nor features.
        if not os.path.exists(save_rank_file):
            if args.view_selection in ('uniform', 'retrieval', 'soft', 'coverage'):
                raise FileNotFoundError(f'{args.view_selection} requires prepared rankings: {save_rank_file}')
            view_selector = ViewSelector().to(model.device)
            ckpt_file = args.pretrained_view_selector_ckpt.format(args.dataset)
            view_selector.load_state_dict(torch.load(ckpt_file)['model'])
            pair_dataset_test = ViewLabelDataset(args, mode=mode)
            pair_dataloader_test = DataLoader(pair_dataset_test, batch_size=1, shuffle=False,
                                              collate_fn=custom_collate_fn)
            ranking_views(pair_dataloader_test, tokenizer, model, view_selector, save_path=save_rank_file)

        with open(save_rank_file) as stream:
            image_file_list = json.load(stream)
        if args.view_selection in ('soft', 'coverage'):
            with open(args.soft_selected_view_file.format(args.dataset, mode)) as stream:
                soft_selected = json.load(stream)
        answers = []
        selected_views = []
        answers_file = args.answers_file.format(args.dataset, mode)

        print('evaluating with QA for {}'.format(mode))
        if args.dataset == 'ScanQA':
            qa_data = get_scanqa(args, mode=mode)
        elif args.dataset == 'SQA':
            qa_data = get_sqa(args, mode=mode)

        for line in tqdm(qa_data):
            scene_id = line['scene_id']

            question_id = str(line['question_id'])
            candidates = image_file_list[question_id]
            if args.view_selection == 'uniform':
                # No relevance weighting, no NMS and no camera-pose processing.
                image_files = uniform_views(candidates, args.input_views, args.seed, mode, question_id)
            elif args.view_selection == 'retrieval':
                # BLIP ranking was computed in a separate process, without QA
                # answers. Pure retrieval follows paper Table 2: top-k, no NMS.
                image_files = candidates[:args.input_views]
            elif args.view_selection in ('soft', 'coverage'):
                # Prepared before loading LLaVA. Preserve greedy selection order;
                # do not apply geometric NMS a second time.
                image_files = soft_selected[question_id]
                if (len(image_files) != min(args.input_views, len(candidates))
                        or len(set(image_files)) != len(image_files)
                        or not set(image_files).issubset(candidates)):
                    raise ValueError(f'Invalid soft selection for {mode}/{question_id}')
            else:
                os.makedirs(args.view_distance_folder, exist_ok=True)
                view_distance_file = os.path.join(args.view_distance_folder, '{}.csv'.format(scene_id))
                if os.path.exists(view_distance_file):
                    distance_df = pd.read_csv(view_distance_file, index_col=0)
                else:
                    distance_df = calculate_view_distance(scene_id, args)
                    distance_df.to_csv(view_distance_file)
                image_files = viewNMS(candidates, distance_df, num_images=args.input_views)
            num_image = len(image_files)
            selected_views.append({"scene_id": scene_id, "question_id": question_id,
                                   "candidate_count": len(candidates), "images": image_files})

            line['from'] = 'human'
            question = line['question'] if args.dataset == 'ScanQA' else line['situation'] + line['question']
            line['value'] = '<image>' * num_image + question
            input_ids = preprocess_qwen([line, {'from': 'gpt', 'value': None}], tokenizer, has_image=True).to(
                model.device)

            image_tensors = []
            for image_file in image_files:
                image = Image.open(os.path.join(args.image_folder, scene_id, 'color', image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image_tensors.append(image_tensor.half().to(model.device))
            # image_tensors = torch.cat(image_tensors, dim=0)

            # Use a separate per-question RNG stream for decoding. This makes a
            # smoke question use the same seed in a full run, even across shards.
            if args.seed is not None:
                transformers.set_seed(question_seed(args.seed, mode, question_id, "decode"))
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip().lower()
            ans = outputs.split('\n')[0]
            # ScanQA result format
            answers.append({"scene_id": scene_id,
                            "question_id": question_id,
                            "answer_top10": [ans for i in range(10)]})

        with open(answers_file, 'w') as stream:
            json.dump(answers, stream)
        with open(answers_file + '.views.json', 'w') as stream:
            json.dump({"method": args.view_selection, "seed": args.seed,
                       "input_views": args.input_views, "split": mode,
                       "questions": selected_views}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--view-selection", choices=("cdviews", "uniform", "retrieval", "soft", "coverage"), default="cdviews")
    parser.add_argument("--seed", type=int, default=None,
                        help="Launcher supplies a seed; None preserves legacy stochastic decoding")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml")
    args = parser.parse_args()

    args = load_and_update(args)
    if args.view_selection == 'uniform' and args.seed is None:
        parser.error("Uniform sampling requires an explicit --seed (or seed in YAML)")

    eval_model(args)
