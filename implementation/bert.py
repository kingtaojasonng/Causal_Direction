import torch
import csv
import pandas as pd
import io
import numpy as np
import transformers
import matplotlib.pyplot as plt
import torch.nn.functional as F
import lime

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm, trange
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt

class Bert:

    def __init__(self, train_file, test_file, output_file, num_labels):
        self.MAX_LEN = 128
        self.batch_size = 32 #32
        self.epochs = 10 #10
        self.num_labels = num_labels
        self.train_file = train_file
        self.test_file = test_file
        self.output_file = output_file
        self.sentences = []
        self.labels = []
        self.input_ids = []
        self.attention_marks = []

    def write_tsv(self, result):
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["train_file", "test_file", "num_labels", "epochs", "accuracy"])
            tsv_output.writerow([self.train_file, self.test_file, self.num_labels, self.epochs, result])
        
    def create_tokens(self):    
        df = pd.read_csv(self.train_file, delimiter="\t", encoding="ISO-8859-1")
        self.sentences = df.sentence.values
        self.sentences = ["[CLS] " + sentence + " [SEP]" for sentence in self.sentences]
        self.labels = df.label.values

    def initialise_bert(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)
        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case = True)

        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in self.sentences]
        self.input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        self.input_ids = pad_sequences(self.input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")

        self.attention_masks = []
        for seq in self.input_ids:
            seq_mask = [float(i>0) for i in seq]
            self.attention_masks.append(seq_mask)

    def split_data(self):
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(self.input_ids, self.labels, random_state=2018, test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(self.attention_masks, self.input_ids, random_state=2018, test_size=0.1)
        
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)
        
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

    def configure_bert(self):
        configuration = BertConfig()
        #configuration = RobertaConfig()
        self.model = BertModel(configuration)
        #self.model = RobertaModel(configuration)
        configuration = self.model.config

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.num_labels)
        #self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.num_labels)
        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate":0.1}, #0.1
                                        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}]

        self.optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5,eps=1e-8)
        total_steps = len(self.train_dataloader) * self.epochs
        
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train_bert(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #@title The Training Loop
        t = [] 

        # Store our loss and accuracy for plotting
        train_loss_set = []
        
        # trange is a tqdm wrapper around the normal python range
        for _ in trange(self.epochs, desc="Epoch"):
            
            # Training
            
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.train()
  
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
  
            # Train the data for one epoch
            for step, batch in enumerate(self.train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs['loss']
                train_loss_set.append(loss.item())    
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()
    
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        self.model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in self.validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
            # Move logits and labels to CPU
            logits = logits['logits'].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
    
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    def evaluate_bert(self):

        df = pd.read_csv(self.test_file, delimiter='\t')

        # Create sentence and label lists
        sentences = df.sentence.values

        # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        labels = df.label.values
        
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
        # Create attention masks
        attention_masks = []
        
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask) 
            
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(labels)
        
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)
        
        # Prediction on test set
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Tracking variables 
        self.predictions , self.true_labels = [], []
        
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits['logits'].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
  
            # Store predictions and true labels
            self.predictions.append(logits)
            self.true_labels.append(label_ids)

    def perform_lime(self, index):
        class_names = [0, 1]
        self.samples = 2
        
        df = pd.read_csv(self.test_file, delimiter='\t')

        # Create sentence and label lists
        sentences = df.sentence.values
        
        # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        labels = df.label.values

        text = sentences[index]
        tokenized_text = self.tokenizer.tokenize(text)
        self.labels = [[labels[index]]] * self.samples
        
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(text, self.predictor, num_features=len(tokenized_text), num_samples=self.samples)
        fig = exp.as_pyplot_figure()
        
    def predictor(self, text):

        sentences = text
        
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        
        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
        # Create attention masks
        attention_masks = []
        
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask) 
            
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(self.labels)
        
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)
        
        # Prediction on test set
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Tracking variables 
        self.predictions , self.true_labels = [], []
        probas = []
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            probas = F.softmax(logits['logits'], dim = 1).detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
        return probas
            
    def score_bert(self):
        matthews_set = []

        for i in range(len(self.true_labels)):
            matthews = matthews_corrcoef(self.true_labels[i],
                                         np.argmax(self.predictions[i], axis=1).flatten())
            matthews_set.append(matthews)
            
        #@title Score of Individual Batches
        #print(matthews_set)

        flat_predictions = [item for sublist in self.predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in self.true_labels for item in sublist]
        return matthews_corrcoef(flat_true_labels, flat_predictions)

    def get_bert_accuracy(self):
        flat_predictions = []
        flat_true_labels = [item for sublist in self.true_labels for item in sublist]
        for i in [item for sublist in self.predictions for item in sublist]:
            flat_predictions.append(np.argmax(i))

        precision = precision_score(flat_true_labels, flat_predictions)
        recall = recall_score(flat_true_labels, flat_predictions)
        f1 = f1_score(flat_true_labels, flat_predictions)
        accuracy = accuracy_score(flat_true_labels, flat_predictions)
        
        # Confusion Matrix
        m = confusion_matrix(flat_true_labels, flat_predictions, labels=[1,0])

        header = "precision, recall, F1, accuracy"
        print(header)
        print(
            '%.2f' % (100 * precision), ",",
            '%.2f' % (100 * recall), ",",
            '%.2f' % (100 * f1), ",",
            '%.2f' % (100 * accuracy),
        )
        
        return accuracy
    
if __name__ == "__main__":

    SemEval2007 = False
    SemEval2007_5_percent = False
    SemEval2007_prior = False
    SemEval2010 = False
    SemEval2010_05_percent = False
    SemEval2010_1_percent = False
    SemEval2010_5_percent = False
    SemEval2010_10_percent = False
    SemEval2010_20_percent = False
    SemEval2010_30_percent = False
    SemEval2010_zero_shot = True
    SemEval2010_data_augmentation = False
    Lime = False
    
    if SemEval2007:
        input_path = "../data/SemEval2007_task4/data/"
        output_path = "../data/SemEval2007_task4/result/"
        bert = Bert(input_path + "train_SemEval2007_causal_effect_direction.tsv",
                    input_path + "test_SemEval2007_causal_effect_direction.tsv",
                    output_path + "result_SemEval2007_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2007_5_percent:
        input_path = "../data/SemEval2007_task4/data/"
        output_path = "../data/SemEval2007_task4/result/"
        bert = Bert(input_path + "train_SemEval2007_causal_effect_direction_5_percent1.tsv",
                    input_path + "test_SemEval2007_causal_effect_direction.tsv",
                    output_path + "result_SemEval2007_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)
    
    if SemEval2007_prior:
        input_path = "../data/SemEval2007_task4/data/"
        output_path = "../data/SemEval2007_task4/result/"
        bert = Bert(input_path + "train_SemEval2007_causal_effect_direction_prior2.tsv",
                    input_path + "test_SemEval2007_causal_effect_direction.tsv",
                    output_path + "result_SemEval2007_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)
        
    if SemEval2010:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2010_1_percent:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_1_percent.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2010_05_percent:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_05_percent.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)
        
    if SemEval2010_5_percent:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_5_percent.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2010_10_percent:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_10_percent.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2010_20_percent:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_20_percent.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2010_30_percent:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_30_percent.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if SemEval2010_zero_shot:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_zero_shot1.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction_prior.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)
        
    if SemEval2010_data_augmentation:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_data_augmentation2.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction_prior.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.evaluate_bert()
        #score = bert.score_bert()
        accuracy = bert.get_bert_accuracy()
        bert.write_tsv(accuracy)

    if Lime:
        input_path = "../data/SemEval2010_task8/data/"
        output_path = "../data/SemEval2010_task8/result/"
        bert = Bert(input_path + "train_SemEval2010_direction_data_augmentation1.tsv",
                    input_path + "test_SemEval2010_direction.tsv",
                    output_path + "result_SemEval2010_direction_prior.tsv",
                    2)
        bert.create_tokens()
        bert.initialise_bert()
        bert.split_data()
        bert.configure_bert()
        bert.train_bert()
        bert.perform_lime(1)
        #score = bert.score_bert()
        #accuracy = bert.get_bert_accuracy()
        #bert.write_tsv(accuracy)
