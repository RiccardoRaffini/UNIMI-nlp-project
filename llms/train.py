import sys
sys.path.append('.')

import os
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from llms.datasets import TokensDataset
from llms.tokenizer import RECIPES_ADDITIONAL_TOKENS

argument_parser = ArgumentParser()
argument_parser.add_argument('--output_directory', required=False, type=str, default='models/gpt2-finetuned-recipe-generation')
argument_parser.add_argument('--tokenized_dataset_path', required=True, type=str)
argument_parser.add_argument('--model_name', required=False, type=str, default='gpt2')
argument_parser.add_argument('--model_path', required=False, type=str, default=None)
argument_parser.add_argument('--tokenizer_name', required=False, type=str, default='gpt2')
argument_parser.add_argument('--block_size', required=False, type=int, default=None)
argument_parser.add_argument('--train', action='store_true')
argument_parser.add_argument('--eval', action='store_true')
argument_parser.add_argument('--eval_during_train', action='store_true')
argument_parser.add_argument('--train_batch_size', required=False, type=int, default=4)
argument_parser.add_argument('--eval_batch_size', required=False, type=int, default=16)
argument_parser.add_argument('--gradient_accumulation_steps', required=False, type=int, default=1)
argument_parser.add_argument('--learning_rate', required=False, type=float, default=5e-5)
argument_parser.add_argument('--weight_decay', required=False, type=float, default=0.0)
argument_parser.add_argument('--optimizer_epsilon', required=False, type=float, default=1e-8)
argument_parser.add_argument('--train_epochs', required=False, type=int, default=10)
argument_parser.add_argument('--save_epochs', required=False, type=int, default=5)
argument_parser.add_argument('--no_cuda', action='store_true')

def train(arguments, model, model_tokenizer, train_dataset:TokensDataset, eval:bool = False, test_dataset:TokensDataset = None):
    print('*** Running model training ***')

    ## Train dataset
    train_batch_size = arguments.train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    ## Optimizer and scheduler
    ignored_parameters = {'bias', 'LayerNorm.weight'}
    optimization_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ignored_parameters)], 'weight_decay': arguments.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ignored_parameters)], 'weight_decay': 0.0}
    ]
    learning_rate = arguments.learning_rate
    epsilon = arguments.optimizer_epsilon
    optimizer = torch.optim.AdamW(optimization_parameters, learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * arguments.train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    ## Training
    training_loss = 0.0
    save_epochs = arguments.save_epochs
    steps = 0
    for epoch in tqdm(range(arguments.train_epochs), 'Training epochs'):
        for step, data_batch in enumerate(tqdm(train_dataloader, 'Training samples')):
            inputs, labels = (data_batch, data_batch)
            inputs, labels = inputs.to(arguments.device), labels.to(arguments.device)

            model.train()
            outputs = model(inputs, labels=labels)

            loss = outputs[0]
            loss.backward()
            training_loss += loss

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            steps += 1

        if eval:
            test(arguments, model, model_tokenizer, test_dataset)

        if epoch % save_epochs == 0:
            output_directory = os.path.join(arguments.output_directory, f'checkpoint-{epoch}')
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            model.save_pretrained(output_directory)
            model_tokenizer.save_pretrained(output_directory)
            print('Saved current model to', output_directory)

        print('>epoch', epoch, 'loss:', training_loss.item()/steps)

    final_loss = training_loss.item() / steps
    return final_loss

def test(arguments, model, model_tokenizer, test_dataset:TokensDataset):
    print('*** Running model evaluation ***')

    ## Test dataset
    test_batch_size = arguments.eval_batch_size
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    ## Evaluation
    evaluation_loss = 0
    steps = 0
    model.eval()

    for data_batch in tqdm(test_dataloader, 'Evaluation samples'):
        data_batch = data_batch.to(arguments.device)

        with torch.no_grad():
            outputs = model(data_batch, labels=data_batch)
            loss = outputs[0]
            evaluation_loss += loss.mean()

        steps += 1

    ## Results
    evaluation_loss = evaluation_loss.item() / steps
    perplexity = torch.exp(torch.tensor(evaluation_loss))
    results = {
        'evaluation loss': evaluation_loss,
        'perplexity': perplexity
    }

    return results

def main():
    ## Arguments
    arguments = argument_parser.parse_args()
    arguments.device = 'cuda' if torch.cuda.is_available() and not arguments.no_cuda else 'cpu'

    if arguments.train and os.path.exists(arguments.output_directory) and os.listdir(arguments.output_directory):
        raise ValueError(f'Output directory {arguments.output_directory} already exists and it is not empty, make sure to provide a different directory')

    ## Define model
    model_tokenizer = GPT2Tokenizer.from_pretrained(arguments.model_name, do_lower_case=True)
    model_tokenizer.add_special_tokens(RECIPES_ADDITIONAL_TOKENS)
    model = GPT2LMHeadModel.from_pretrained(arguments.tokenizer_name)
    model.resize_token_embeddings(len(model_tokenizer))
    model.to(arguments.device)
    
    if arguments.block_size is None:
        arguments.block_size = model_tokenizer.max_len_single_sentence
    arguments.block_size = min(arguments.block_size, model_tokenizer.max_len_single_sentence)

    ## Load datasets
    train_dataset = TokensDataset(arguments.tokenized_dataset_path, partition='train', block_size=arguments.block_size)
    test_dataset = TokensDataset(arguments.tokenized_dataset_path, partition='test', block_size=arguments.block_size)

    ## Save initial checkpoint
    output_directory = os.path.join(arguments.output_directory, 'checkpoint-0')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    model.save_pretrained(output_directory)
    model_tokenizer.save_pretrained(output_directory)

    ## Train model
    if arguments.train:
        loss = train(arguments, model, model_tokenizer, train_dataset, arguments.eval_during_train, test_dataset)
        print('final loss:', loss)

        output_directory = os.path.join(arguments.output_directory, 'checkpoint-final')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        model.save_pretrained(output_directory)
        model_tokenizer.save_pretrained(output_directory)

    ## Test model
    if arguments.eval:
        test_results = test(arguments, model, model_tokenizer, test_dataset)
        print(test_results)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention.')

    main()