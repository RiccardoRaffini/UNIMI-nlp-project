import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer

from commons.data import load_foodcom_review_dataset
from llms.tokenizer import raw_recipes_to_text_tokens_file, text_tokens_files_to_dataset, RECIPES_ADDITIONAL_TOKENS

def main():
    seed = 1234

    ## Sample data
    recipes_dataset = load_foodcom_review_dataset()
    recipes_dataset = recipes_dataset.sample(n=60_000, random_state=seed)
    print(len(recipes_dataset))

    ## Split data
    recipes_train_dataset, recipes_test_dataset = train_test_split(recipes_dataset, test_size=0.05)
    recipes_train_dataset.reset_index(drop=True, inplace=True)
    recipes_test_dataset.reset_index(drop=True, inplace=True)
    print(len(recipes_train_dataset), len(recipes_test_dataset))

    ## Create text tokens files
    raw_recipes_to_text_tokens_file(recipes_train_dataset, filename='recipes_train_text_tokens.txt', verbose=True)
    raw_recipes_to_text_tokens_file(recipes_test_dataset, filename='recipes_test_text_tokens.txt', verbose=True)

    ## Setup tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case = False)
    gpt_tokenizer.add_special_tokens(RECIPES_ADDITIONAL_TOKENS)

    ## Create tokens dataset file
    train_filename = 'recipes_train_text_tokens.txt'
    test_filename = 'recipes_test_text_tokens.txt'

    ## Create dataset file
    dataset_filename = 'recipes_tokenized.h5'
    text_tokens_files_to_dataset(dataset_filename, gpt_tokenizer, train_filename, test_filename)

if __name__ == '__main__':
    main()