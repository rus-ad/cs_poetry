# +
from models import BaseModel
from prompt import BasePrompt

import pandas as pd
from datasets import load_dataset
from transformers import set_seed
from transformers import TrainingArguments
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import Trainer

from ruprompts.callbacks import (
    FreezeTransformerUnfreezePrompt,
    ReduceCheckpoint,
    SavePretrainedPrompt,
)
from transformers import pipeline

set_seed(1)


# +
import torch


USE_GPU = 1
if USE_GPU:
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

print('using device:', device)
# -

datasets = load_dataset(
    "csv", 
    data_files={
        "train": "processing/raw_data/ru/pushkin/pushkin_train.csv", 
        "validation": "processing/raw_data/ru/pushkin/pushkin_valid.csv",
    },
)
train_dataset = datasets["train"]
valid_dataset = datasets["validation"]

# +
model = BaseModel(
    backbone_id="sberbank-ai/rugpt3large_based_on_gpt2",
).get_models()
prompt = BasePrompt(
    prompt_format="<P*100>{name}<P*20>",
    model=model,
)
prompt.set_preprocessor(
    target_field="poem",
    truncation_field="name",
)

train_dataset = prompt.preprocess_dataset(train_dataset)
valid_dataset = prompt.preprocess_dataset(valid_dataset)
# -

training_args = TrainingArguments(
    output_dir=".",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    eval_steps=1000,
    save_steps=1000,
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    learning_rate=0.1,
    max_steps=100000,
    report_to="tensorboard",
    # report_to=["tensorboard", "wandb"],  # uncomment to log to WandB
    logging_dir="logs",
    seed=1
)

print(training_args.device, torch.cuda.is_available())

training_args.device = device









optimizer = AdamW(prompt.prompt_provider.parameters(), lr=training_args.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=training_args.max_steps,
)

# +
trainer = Trainer(
    model=model['model'],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=prompt.preprocessor.collate_fn(),
    optimizers=(optimizer, scheduler),
    callbacks=[FreezeTransformerUnfreezePrompt(), ReduceCheckpoint(), SavePretrainedPrompt(prompt.prompt)],
)

trainer.train()
# -


# +
# checkpointpath = ''
# model = BaseModel(
#     backbone_id=checkpointpath,
# )
# prompt = Prompt.from_pretrained(checkpointpath)
# ppln = pipeline(
#     "text2text-generation-with-prompt", 
#     prompt=prompt, 
#     model=model['model'], 
#     tokenizer=model['tokenizer'], 
#     device=0,
    
# )

# beam_count = 10

# options = ppln(
#     {"toxic_comment": i},
#     do_sample=False,
#     num_beams=beam_count,
#     num_return_sequences=beam_count,
# )

# options = [i["generated_text"].replace("<pad>", "") for i in options]
# -


