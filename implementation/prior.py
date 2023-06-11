import csv
import re
import urllib
import requests

from lxml import html

class Prior:

    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def __extract_result(self, query):
        query = urllib.parse.quote_plus(query)
        headers = {
            "User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.3 Safari/605.1.15"
        }
        response = requests.get('https://www.google.com/search?q='+query,
                                headers=headers,
                                stream=True)
        response.raw.decode_content = True
        tree = html.parse(response.raw)
        # lxml is used to select element by XPath
        # Requests + lxml: https://stackoverflow.com/a/11466033/1291371
        try:
            result = tree.xpath('//*[@id="result-stats"]/text()')[0]
        except:
            result = ""
        return result.replace("About ", "").replace(" results", "").replace(" result", "")
        
    def compute_prior_semeval2007(self, site):
        preprocessed, labels, causals = [], [], []
        sentences = None
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
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence", "e1", "e2", "e1_e2", "count_e1", "count_e2", "count_e1_e2"])
            for i in range(len(sentences)):
                id, sentence = sentences[i][0:3], sentences[i][4:]
                e1_start = re.search(r'<e1>', sentences[i]).end()
                e1_end = re.search(r'</e1>', sentences[i]).start()
                e2_start = re.search(r'<e2>', sentences[i]).end()
                e2_end = re.search(r'</e2>', sentences[i]).start()
                e1 = sentences[i][e1_start:e1_end]
                e2 = sentences[i][e2_start:e2_end]
                processed_sentence = sentence.strip().replace("\"", "").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                if labels[i] == "true":
                    result_e1 = self.__extract_result(e1 + site)
                    result_e2 = self.__extract_result(e2 + site)
                    result_e1_e2 = self.__extract_result(e1 + " " + e2 + site)
                    label = 1 if causals[i] == "(e1,e2)" else 0
                    tsv_output.writerow([int(id), label, "dummy", processed_sentence, e1, e2, e1 + " " + e2, result_e1, result_e2, result_e1_e2])

    def compute_prior_semeval2010(self, site):
        preprocessed, labels, causals = [], [], []
        sentences = None
        with open(self.input_file, encoding="ISO-8859-1") as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            categories = lines[1::4]
            comments = lines[2::4]
        with open(self.output_file, "wt") as file_output:
            tsv_output = csv.writer(file_output, delimiter="\t")
            tsv_output.writerow(["sentence_source", "label", "label_notes", "sentence", "e1", "e2", "e1_e2", "count_e1", "count_e2", "count_e1_e2"])
            for i in range(len(sentences)):
                if categories[i][0:12] == "Cause-Effect":
                    a = categories[i].split("Cause-Effect")[1].strip()
                    id, sentence = sentences[i].split("\t")
                    e1_start = re.search(r'<e1>', sentence).end()
                    e1_end = re.search(r'</e1>', sentence).start()
                    e2_start = re.search(r'<e2>', sentence).end()
                    e2_end = re.search(r'</e2>', sentence).start()
                    e1 = sentence[e1_start:e1_end].lower()
                    e2 = sentence[e2_start:e2_end].lower()
                    processed_sentence = sentence.strip().replace("\"", "").replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                    result_e1 = result_e2 = result_e1_e2 = ""
                    if int(id) >= 9901:
                        result_e1 = self.__extract_result(e1 + site)
                        result_e2 = self.__extract_result(e2 + site)
                        result_e1_e2 = self.__extract_result(e1 + " " + e2 + site)
                    label = 1 if a == "(e1,e2)" else 0
                    tsv_output.writerow([int(id), label, "dummy", processed_sentence, e1, e2, e1 + " " + e2, result_e1, result_e2, result_e1_e2])
                
if __name__ == "__main__":
    
    SemEval2007 = False
    SemEval2010 = True
    
    if SemEval2007:                
        input_path = "../data/SemEval2007_task4/"
        output_path = "../data/SemEval2007_task4/prior/"
        prior = Prior(input_path + "task-4-scoring/relation-1-score.txt", output_path + "test.tsv")
        prior.compute_prior_semeval2007(" site:edu")

    if SemEval2010:
        input_path = "../data/SemEval2010_task8_all_data/"
        output_path = "../data/SemEval2010_task8_all_data/prior/"
        prior = Prior(input_path + "SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt", output_path + "prior1.tsv")
        #prior.compute_prior_semeval2010(" site:au.news.yahoo.com")
        prior.compute_prior_semeval2010("")
