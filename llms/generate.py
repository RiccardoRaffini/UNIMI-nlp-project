import sys
sys.path.append('.')

import torch
import torch.nn.functional as torchfunctional
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from commons.text import TokensSequenceToText
from llms.tokenizer import RECIPES_ADDITIONAL_TOKENS

arguments_parser = ArgumentParser()
arguments_parser.add_argument('--model_path', required=True, type=str, default=None)
arguments_parser.add_argument('--prompt', required=False, type=str, default=None)
arguments_parser.add_argument('--length', required=False, type=int, default=200)
arguments_parser.add_argument('--temperature', required=False, type=float, default=1.0)
arguments_parser.add_argument('--top_k', required=False, type=int, default=0)
arguments_parser.add_argument('--top_p', required=False, type=float, default=0.9)
arguments_parser.add_argument('--no_cuda', required=False, type=bool, default=False)
arguments_parser.add_argument('--seed', required=False, type=int, default=1234)

def logits_filtering(logits:torch.Tensor, top_k:int, top_p:float, escape_value:float = float('-inf')) -> torch.Tensor:
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = escape_value

    if top_p > 0:
        sorted_logits, sorted_logits_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(torchfunctional.softmax(sorted_logits, dim=-1), dim=-1)

        indices_to_remove = cumulative_probabilities > top_p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_logits_indices[indices_to_remove]
        logits[indices_to_remove] = escape_value

    return logits

def generate(arguments, model, model_tokenizer, input, samples_number:int = 1):
    ## Adapt input
    input = torch.tensor(input, dtype=torch.long, device=arguments.device)
    input = input.unsqueeze(0).repeat(samples_number, 1)

    ## Generate
    end_token_index = model_tokenizer.convert_tokens_to_ids(['<RECIPE_END>'])[0]
    generated = input

    with torch.no_grad():
        for _ in tqdm(range(arguments.length), 'tokens generation'):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            outputs_logits = outputs[0][0, -1, :] / arguments.temperature
            filtered_logits = logits_filtering(outputs_logits, arguments.top_k, arguments.top_p)
            selected_token = torch.multinomial(torchfunctional.softmax(filtered_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, selected_token.unsqueeze(0)), dim=1)

            ## Check end of recipe
            if selected_token.item() == end_token_index:
                break

    return generated

def main():
    ## Argumnets
    arguments = arguments_parser.parse_args()
    arguments.device = 'cuda' if torch.cuda.is_available() and not arguments.no_cuda else 'cpu'
    torch.manual_seed(arguments.seed)
    # model_path = 'models/gpt2-finetuned-recipe-generation'

    ## Define model
    model_tokenizer = GPT2Tokenizer.from_pretrained(arguments.model_path, do_lower_case=True)
    model_tokenizer.add_special_tokens(RECIPES_ADDITIONAL_TOKENS)
    model = GPT2LMHeadModel.from_pretrained(arguments.model_path)
    model.resize_token_embeddings(len(model_tokenizer))
    model.to(arguments.device)
    model.eval()

    ## Check output length
    if model.config.max_position_embeddings < arguments.length:
        arguments.length = model.config.max_position_embeddings

    ## Check user input
    raw_recipe_prompt = arguments.prompt
    if raw_recipe_prompt is None:
        print('No prompt provided in arguments')
        raw_recipe_prompt = input('Provide new prompt now as a comma-separated list of ingredients:')

    ## Tokenize user input
    processed_recipe_prompt = '<START_RECIPE> <INPUT_START> ' + \
        raw_recipe_prompt.replace(',', ' <INPUT_NEXT> ') + \
        ' <INPUT_END>'
    input_tokens = model_tokenizer.encode(processed_recipe_prompt)

    ## Generate recipe
    output = generate(arguments, model, model_tokenizer, input_tokens)
    output = output[0, len(input_tokens):].tolist()
    text_output = model_tokenizer.decode(output, clean_up_tokenization_spaces = True)

    if '<RECIPE_END>' not in text_output:
        print('Recipe generation failed, too long recipe')
        sys.exit(1)

    print('Raw output:')
    print(text_output)

    ## Output processing
    print('Processed output:')
    output_processor = TokensSequenceToText()
    ingredients = output_processor.ingredients(text_output)
    instructions = output_processor.instructions(text_output)
    markdown = output_processor.markdown(text_output)
    print(markdown)

if __name__ == '__main__':
    main()