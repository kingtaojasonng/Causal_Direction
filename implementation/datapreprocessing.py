import csv
import re
import nltk

from os import listdir
from os.path import isfile, join

class DataPreprocessing:

    def __init__(self, input_file, output_file, source_file = None):
        self.input_file = input_file
        self.output_file = output_file
        self.source_file = source_file

    def preprocess_semeval2007_label(self):
        preprocessed, labels = [], []
        with open(self.input_file, encoding="ISO-8859-1") as file_input:
            lines = file_input.readlines()
            for i in lines:
                if not i.startswith("Comment"):
                    preprocessed.append(i)
            sentences = preprocessed[0::3]
            for j in preprocessed[1::3]:
                t = j.split("Cause-Effect")[1].split(", Query")[0].split(" = ")
                labels.append(t[1].strip("\""))
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence"])
            for i in range(len(sentences)):
                id, sentence = sentences[i][0:3], sentences[i][4:].strip().replace("\"", "").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                label = 1 if labels[i] == "true" else 0
                tsv_output.writerow([int(id), label, "dummy", sentence])
        
    def preprocess_semeval2007_direction(self):
        preprocessed, labels, causals = [], [], []
        with open(self.input_file, encoding="ISO-8859-1") as file_input:
            lines = file_input.readlines()
            for i in lines:
                if not i.startswith("Comment"):
                    preprocessed.append(i)
            sentences = preprocessed[0::3]
            for j in preprocessed[1::3]:
                t = j.split("Cause-Effect")[1].split(", Query")[0].split(" = ")
                labels.append(t[1].strip("\""))
                causals.append(t[0])
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence"])
            for i in range(len(sentences)):
                id, sentence = sentences[i][0:3], sentences[i][4:].strip()
                if labels[i] == "true":
                    label = 1 if causals[i] == "(e1,e2)" else 0
                    tsv_output.writerow([int(id), label, "dummy", sentence])

    def preprocess_semeval2010_imbalance(self):
        sentences, labels, comments = [], [], []
        with open(self.input_file) as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            labels = lines[1::4]
            comments = lines[2::4]
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence"])
            for i in range(len(sentences)):
                id, sentence = re.split(r'\t', sentences[i])
                sentence = sentence.strip().strip("\"").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                label = 1 if labels[i][0:12] == "Cause-Effect" else 0
                comment = comments[i].strip()
                tsv_output.writerow([int(id), label, comment, sentence])

    def preprocess_semeval2010_balance(self):
        sentences, labels, comments = [], [], []
        with open(self.input_file) as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            labels = lines[1::4]
            comments = lines[2::4]
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence"])
            for i in range(len(sentences)):
                id, sentence = re.split(r'\t', sentences[i])
                sentence = sentence.strip().strip("\"").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                if labels[i][0:12] == "Cause-Effect":
                    label = 1
                elif labels[i][0:17] == "Instrument-Agency":
                    label = 0
                else:
                    continue
                comment = comments[i].strip()
                tsv_output.writerow([int(id), label, comment, sentence])

    def preprocess_semeval2010_10_balance(self):
        sentences, labels, comments = [], [], []
        with open(self.input_file) as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            labels = lines[1::4]
            comments = lines[2::4]
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence"])
            for i in range(len(sentences)):
                id, sentence = re.split(r'\t', sentences[i])
                sentence = sentence.strip().strip("\"").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                if labels[i][0:12] == "Cause-Effect":
                    label = 1
                elif labels[i][0:17] == "Instrument-Agency":
                    label = 2
                elif labels[i][0:15] == "Component-Whole":
                    label = 3
                elif labels[i][0:17] == "Member-Collection":
                    label = 4
                elif labels[i][0:18] == "Entity-Destination":
                    label = 5
                elif labels[i][0:17] == "Content-Container":
                    label = 6
                elif labels[i][0:13] == "Message-Topic":
                    label = 7
                elif labels[i][0:16] == "Product-Producer":
                    label = 8
                elif labels[i][0:13] == "Entity-Origin":
                    label = 9
                else:
                    label = 0
                comment = comments[i].strip()
                tsv_output.writerow([int(id), label, comment, sentence])

    def preprocess_semeval2010_direction(self):
        sentences, labels, comments = [], [], []
        with open(self.input_file) as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            labels = lines[1::4]
            comments = lines[2::4]
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence"])
            for i in range(len(sentences)):
                id, sentence = re.split(r'\t', sentences[i])
                sentence = sentence.strip().strip("\"").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                if labels[i][0:12] == "Cause-Effect":
                    label = 1 if labels[i][12:19] == "(e1,e2)" else 0
                    comment = comments[i].strip()
                    tsv_output.writerow([id, label, comment, sentence])

    def preprocess_ADE(self):
        with open(self.input_file, encoding="ISO-8859-1") as file_input:
            lines = file_input.readlines()
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence", "e1", "e2"])
            for i in lines[:6139]:
                tokens = i.split("|")
                if int(tokens[3]) < int(tokens[6]):
                    # e2 -> e1
                    e1 = tokens[2] # effect
                    e2 = tokens[5] # drug
                    label = 0
                else:
                    # e1 -> e2
                    e1 = tokens[5] # drug
                    e2 = tokens[2] # effect
                    label = 1
                tsv_output.writerow([tokens[0], label, "drug-effect", tokens[1], e1, e2])
                
    def preprocess_because(self):
        d = {}
        with open(self.source_file, encoding="ISO-8859-1") as file_source:
            source_lines = file_source.readlines()
        with open(self.input_file, encoding="ISO-8859-1") as file_input:
            input_lines = file_input.readlines()
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence", "tag1", "keyword1", "tag2", "keyword2", "tag3", "keyword3"])
            for i in input_lines:
                l = i.split("\t")
                tag, value = l[0], l[-1].strip()
                d[tag] = value
            for key, value in d.items():
                if key.startswith("E"):
                    if len(value.split(" ")) == 3:
                        tag1, keyword1, tag2, keyword2, tag3, keyword3 = re.split(" |:", value)
                        for s in source_lines:
                            sentences = nltk.tokenize.sent_tokenize(s)
                            for sentence in sentences:
                                if d[keyword1] in sentence and d[keyword2] in sentence and d[keyword3] in sentence:
                                    tsv_output.writerow([key, int(not(tag1 == "NonCausal")), tag1, sentence, tag1, d[keyword1], tag2, d[keyword2], tag3, d[keyword3]])
                                    break

    def preprocess_because_direction(self):
        d, d_position = {}, {}
        with open(self.source_file, encoding="ISO-8859-1") as file_source:
            source_lines = file_source.readlines()
        with open(self.input_file, encoding="ISO-8859-1") as file_input:
            input_lines = file_input.readlines()
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence", "e1", "e1", "keyword"])
            for i in input_lines:
                l = i.split("\t")
                tag, position, value = l[0], l[1], l[-1].strip()
                d[tag] = value
                d_position[tag] = position
            for key, value in d.items():
                if key.startswith("E"):
                    if len(value.split(" ")) == 3:
                        tag1, keyword1, tag2, keyword2, tag3, keyword3 = re.split(" |:", value)
                        for s in source_lines:
                            sentences = nltk.tokenize.sent_tokenize(s)
                            for sentence in sentences:
                                if d[keyword1] in sentence and d[keyword2] in sentence and d[keyword3] in sentence:
                                    if not tag1 == "NonCausal" and (tag2 == "Cause" or tag2 == "Effect") and (tag3 == "Cause" or tag3 == "Effect"):
                                        keyword2_position = int(d_position[keyword2].split(" ")[1])
                                        keyword3_position = int(d_position[keyword3].split(" ")[1])
                                        if keyword2_position < keyword3_position:
                                            e1 = d[keyword2]
                                            e2 = d[keyword3]
                                            label = 0 if tag2 == "Effect" and tag3 == "Cause" else 1 # e2 -> e1 if label = 0; e1 -> e2 if label 1
                                        else:
                                            e1 = d[keyword3]
                                            e2 = d[keyword2]
                                            label = 1 if tag2 == "Effect" and tag3 == "Cause" else 0
                                        tsv_output.writerow([key,
                                                             label,
                                                             tag1,
                                                             sentence,
                                                             e1,
                                                             e2,
                                                             d[keyword1]])
                                        break
                                
if __name__ == "__main__":

    SemEval2007_label = False
    SemEval2007_direction = False
    SemEval2010_two_imbalance_classes = False
    SemEval2010_two_balance_classes = False
    SemEval2010_ten_balance_classes = False
    SemEval2010_direction = True
    ADE = False
    Because = False
    Because_direction = False
    
    if SemEval2007_label:
        input_path = "../data/SemEval2007_task4/"
        output_path = "../data/SemEval2007_task4/data/"
        datapreprocessing = DataPreprocessing(input_path + "task-4-training/relation-1-train.txt", output_path + "train_SemEval2007_causal_effect_label.tsv")
        datapreprocessing.preprocess_semeval2007_label()

    if SemEval2007_direction:
        input_path = "../data/SemEval2007_task4/"
        output_path = "../data/SemEval2007_task4/data/"
        datapreprocessing = DataPreprocessing(input_path + "task-4-training/relation-1-train.txt", output_path + "train_SemEval2007_causal_effect_direction.tsv")
        datapreprocessing.preprocess_semeval2007_direction()
        datapreprocessing = DataPreprocessing(input_path + "task-4-scoring/relation-1-score.txt", output_path + "test_SemEval2007_causal_effect_direction.tsv")
        datapreprocessing.preprocess_semeval2007_direction()

    if SemEval2010_two_imbalance_classes:
        input_path = "../data/SemEval2010_task8_all_data/"
        output_path = "../data/SemEval2010_task8_all_data/data/"
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_training/TRAIN_FILE.TXT", output_path + "train_SemEval2010_imbalance.tsv")
        datapreprocessing.preprocess_semeval2010_imbalance()
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", output_path + "test_SemEval2010_imbalance.tsv")
        datapreprocessing.preprocess_semeval2010_imbalance()

    if SemEval2010_two_balance_classes:
        input_path = "../data/SemEval2010_task8_all_data/"
        output_path = "../data/SemEval2010_task8_all_data/data/"
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_training/TRAIN_FILE.TXT", output_path + "train_SemEval2010_balance.tsv")
        datapreprocessing.preprocess_semeval2010_balance()
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", output_path + "test_SemEval2010_balance.tsv")
        datapreprocessing.preprocess_semeval2010_balance()

    if SemEval2010_ten_balance_classes:
        input_path = "../data/SemEval2010_task8_all_data/"
        output_path = "../data/SemEval2010_task8_all_data/data/"
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_training/TRAIN_FILE.TXT", output_path + "train_SemEval2010_10_balance.tsv")
        datapreprocessing.preprocess_semeval2010_10_balance()
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", output_path + "test_SemEval2010_10_balance.tsv")
        datapreprocessing.preprocess_semeval2010_10_balance()

    if SemEval2010_direction:
        input_path = "../data/SemEval2010_task8/"
        output_path = "../data/SemEval2010_task8/data/"
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_training/TRAIN_FILE.TXT", output_path + "train_SemEval2010_direction_temp.tsv")
        datapreprocessing.preprocess_semeval2010_direction()
        datapreprocessing = DataPreprocessing(input_path + "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", output_path + "test_SemEval2010_direction_temp.tsv")
        datapreprocessing.preprocess_semeval2010_direction()

    if ADE:
        input_path = "../data/AdverseDrugReaction-master/"
        output_path = "../data/AdverseDrugReaction-master/data/"
        datapreprocessing = DataPreprocessing(input_path + "ADE-Corpus-V2/DRUG-AE.rel", output_path + "train_ADE.tsv")
        datapreprocessing.preprocess_ADE()
        datapreprocessing = DataPreprocessing(input_path + "ADE-Corpus-V2/DRUG-AE.rel", output_path + "test_ADE.tsv")
        datapreprocessing.preprocess_ADE()

    if Because:
        input_path = "../data/BECAUSE-master/CongressionalHearings/"
        output_path = "../data/BECAUSE-master/data/"
        datapreprocessing = DataPreprocessing(input_path + "CHRG-110hhrg44900-1.ann",
                                              output_path + "CHRG-110hhrg44900-1.tsv",
                                              input_path + "CHRG-110hhrg44900-1.txt")
        datapreprocessing.preprocess_because()

        datapreprocessing = DataPreprocessing(input_path + "CHRG-110hhrg44900-2.ann",
                                              output_path + "CHRG-110hhrg44900-2.tsv",
                                              input_path + "CHRG-110hhrg44900-2.txt")
        datapreprocessing.preprocess_because()

        datapreprocessing = DataPreprocessing(input_path + "CHRG-111shrg61651.ann",
                                              output_path + "CHRG-111shrg61651.tsv",
                                              input_path + "CHRG-111shrg61651.txt")
        datapreprocessing.preprocess_because()

        input_path = "../data/BECAUSE-master/MASC/"
        output_path = "../data/BECAUSE-master/data/"
        files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith(".ann")]
        for i in files:
            datapreprocessing = DataPreprocessing(input_path + i,
                                                  output_path + i[:-4] + ".tsv",
                                                  input_path + i[:-4] + ".txt")
            datapreprocessing.preprocess_because()

    if Because_direction:
        input_path = "../data/BECAUSE-master/CongressionalHearings/"
        output_path = "../data/BECAUSE-master/data/"
        datapreprocessing = DataPreprocessing(input_path + "CHRG-110hhrg44900-1.ann",
                                              output_path + "CHRG-110hhrg44900-1_direction.tsv",
                                              input_path + "CHRG-110hhrg44900-1.txt")
        datapreprocessing.preprocess_because_direction()

        datapreprocessing = DataPreprocessing(input_path + "CHRG-110hhrg44900-2.ann",
                                              output_path + "CHRG-110hhrg44900-2_direction.tsv",
                                              input_path + "CHRG-110hhrg44900-2.txt")
        datapreprocessing.preprocess_because_direction()

        datapreprocessing = DataPreprocessing(input_path + "CHRG-111shrg61651.ann",
                                              output_path + "CHRG-111shrg61651_direction.tsv",
                                              input_path + "CHRG-111shrg61651.txt")
        datapreprocessing.preprocess_because_direction()

        input_path = "../data/BECAUSE-master/MASC/"
        output_path = "../data/BECAUSE-master/data/"
        files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith(".ann")]
        for i in files:
            datapreprocessing = DataPreprocessing(input_path + i,
                                                  output_path + i[:-4] + "_direction.tsv",
                                                  input_path + i[:-4] + ".txt")
            datapreprocessing.preprocess_because_direction()

