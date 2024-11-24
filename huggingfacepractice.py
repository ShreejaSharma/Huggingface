from transformers import pipeline,Autotokenizer,TraningArguments,Trainer,AutoModelForSequenceClassifier
from datasets import load_dataset

generator =  pipeline('text-generation',model='openai-community/gpt2')
generator("Today is Monday",truncation=True,num_return_sequences=2)
classifier = pipeline("sentimnt-analysis")
classifier("I loved friends so much")
dataset = load_dataset('CShorten/ML-ArXiv-Papers')
tokenizer = Autokenizer.from_pretrained("google-bert/bert-base-cased")
sentence = "The problem of statistical learning is to construct a pretrained"
tokens = tokenizer.tokens(sentence)
  
def tokenize_functions(examples):
  return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

training_args = TraningArguments(output_dir=".results",eval_strategy="epoch",learning_rate=2e-5,num_train_epochs=3,per_device_train_batch_size=16,per_device_test_batch_size=16,weight_decay=0.01)
model = AutoModelForSequenceClassifier.from_pretrained("google-bert/bert-base-cased", num_labels=2)
trainer = Trainer(model=model,args=training_args,train_dataset=small_train_dataset,test_dataset=small_eval_dataset)
trainer.train()
results = trainer.evaluate()
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
