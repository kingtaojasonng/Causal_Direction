import re
import math
import stan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statistics

from os import listdir
from os.path import isfile, join
from scipy.stats import beta
from scipy.stats import ks_2samp
from scipy.stats import chi
from scipy.stats import gamma
from scipy.stats import t
from scipy.stats import expon
from scipy.stats import cauchy
from scipy.stats import invgamma
from scipy.stats import lognorm
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import chi2

import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

from fitter import Fitter, get_common_distributions, get_distributions

from collections import Counter
import matplotlib.pyplot as plt

pandas2ri.activate()

class BayesianInference:

    def __init__(self, corpora_path, training_data):
        self.corpora_path = corpora_path
        self.training_data = training_data
        self.likelihoods = {}
        self.e1_arrow_e2 = None
        self.e2_arrow_e1 = None
        
    def compute_likelihood_semeval2007(self, threshold = None):
        likelihoods, preprocessed, labels, causals = {}, [], [], []
        with open(self.training_data, encoding = "ISO-8859-1") as file_input:
            lines = file_input.readlines()
            for i in lines:
                if not i.startswith("Comment"):
                    preprocessed.append(i)
            for j in preprocessed[1::3]:
                t = j.split("Cause-Effect")[1].split(", Query")[0].split(" = ")
                labels.append(t[1].strip("\""))
                causals.append(t[0])
            sentences = preprocessed[0::3]
            for k in range(len(sentences)):
                id, sentence = sentences[k][0:3], sentences[k][4:]
                if threshold != None and int(id) not in threshold:
                    break
                e1_start = re.search(r'<e1>', sentences[k]).end()
                e1_end = re.search(r'</e1>', sentences[k]).start()
                e2_start = re.search(r'<e2>', sentences[k]).end()
                e2_end = re.search(r'</e2>', sentences[k]).start()
                e1 = sentences[k][e1_start:e1_end].lower()
                e2 = sentences[k][e2_start:e2_end].lower()
                if labels[k] == "true":
                    label = True if causals[k] == "(e1,e2)" else False
                    t = (e1, e2) if label else (e2, e1)
                    likelihoods[t] = likelihoods[t] + 1 if t in likelihoods else 1
        self.likelihoods = likelihoods

    def compute_likelihood_semeval2010(self, threshold = None):
        likelihoods, preprocessed, labels, causals = {}, [], [], []
        with open(self.training_data, encoding = "ISO-8859-1") as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            categories = lines[1::4]
            comments = lines[2::4]
            for i in range(len(sentences)):
                if categories[i][0:12] == "Cause-Effect":
                    a = categories[i].split("Cause-Effect")[1].strip()
                    id, sentence = sentences[i].split("\t")
                    if threshold != None and int(id) > threshold:
                        break
                    e1_start = re.search(r'<e1>', sentence).end()
                    e1_end = re.search(r'</e1>', sentence).start()
                    e2_start = re.search(r'<e2>', sentence).end()
                    e2_end = re.search(r'</e2>', sentence).start()
                    e1 = sentence[e1_start:e1_end].lower()
                    e2 = sentence[e2_start:e2_end].lower()
                    t = (e1, e2) if a == "(e1,e2)" else (e2, e1)
                    likelihoods[t] = likelihoods[t] + 1 if t in likelihoods else 1
        self.likelihoods = likelihoods
        
    def get_tokens_semeval2010(self, threshold = None):
        temp = []
        likelihoods, preprocessed, labels, causals = {}, [], [], []
        with open(self.training_data, encoding = "ISO-8859-1") as file_input:
            lines = file_input.readlines()
            sentences = lines[0::4]
            categories = lines[1::4]
            comments = lines[2::4]
            for i in range(len(sentences)):
                if categories[i][0:12] == "Cause-Effect":
                    a = categories[i].split("Cause-Effect")[1].strip()
                    id, sentence = sentences[i].split("\t")
                    if threshold != None and int(id) > threshold:
                        break
                    e1_start = re.search(r'<e1>', sentence).end()
                    e1_end = re.search(r'</e1>', sentence).start()
                    e2_start = re.search(r'<e2>', sentence).end()
                    e2_end = re.search(r'</e2>', sentence).start()
                    e1 = sentence[e1_start:e1_end].lower()
                    e2 = sentence[e2_start:e2_end].lower()
                    t = (e1, e2) if a == "(e1,e2)" else (e2, e1)
                    temp.append(t[0])
                    temp.append(t[1])
                    likelihoods[t] = likelihoods[t] + 1 if t in likelihoods else 1
        return temp, likelihoods

    def __get_likelihood(self, word1, word2):
        count = 0
        for key, value in self.likelihoods.items():
            w1, w2 = key
            if w1 == word1 and w2 == word2:
                count += value
            #else:
            #    if w1 == word2:
            #        count += value
            #    if w2 == word2:
            #        count += value
        return count

    def __get_likelihood_demonstration(self, word1, word2):
        count = 0
        for key, value in self.likelihoods.items():
            w1, w2 = key
            if w1 == word1 and w2 == word2:
                count += value
        return count

    def save_e1_arrow_e2(self, filename):
        self.e1_arrow_e2.to_csv(filename)

    def save_e2_arrow_e1(self, filename):
        self.e2_arrow_e1.to_csv(filename)

    def __conduct_kolmogorov_smirnov(self, prior, likelihood):
        test = None
        if likelihood == 0:
            prior = list(filter(lambda num: num != 0.0, prior))
            alpha_, beta_, c, d = beta.fit(np.array(prior), floc = 0, fscale=1)
            generated_samples = np.random.beta(alpha_, beta_, size=100)
            test = ks_2samp(generated_samples, prior)
        return test
    
    def __conduct_kolmogorov_smirnov_various_distributions(self, prior, likelihood):
        test = None
        if likelihood == 0:
            prior = list(filter(lambda num: num != 0.0, prior))
            best_distribution, _ = self.__fit_distributions(prior, likelihood)
            d = list(best_distribution)[0]
            if d == 'gamma':
                alpha_ = best_distribution[d]['a']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = gamma.rvs(alpha_, loc=loc_, scale=scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 't':
                df_ = best_distribution[d]['df']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = t.rvs(df_, loc = loc_, scale=scale_, size=100)
                test = ks_2samp(generated_samples, prior)           
            elif d == 'chi':
                df_ = best_distribution[d]['df']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = chi.rvs(df_, loc = loc_, scale=scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'expon':
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = expon.rvs(loc = loc_, scale=scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'cauchy':
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = cauchy.rvs(loc = loc_, scale=scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'invgamma':
                alpha_ = best_distribution[d]['a']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = invgamma.rvs(alpha_, loc = loc_, scale = scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'lognorm':
                s_ = best_distribution[d]['s']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = lognorm.rvs(s_, loc=loc_, scale=scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'beta':
                alpha_ = best_distribution[d]['a']
                beta_ = best_distribution[d]['b']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                #alpha_, beta_, loc_, scale_ = beta.fit(np.array(prior), floc = 0, fscale=1)
                generated_samples = beta.rvs(alpha_, beta_, loc = loc_, scale = scale_, size=100)
                #generated_samples = np.random.beta(alpha_, beta_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'uniform':
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale'] + loc_
                generated_samples = uniform.rvs(loc_, scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'norm':
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                generated_samples = norm.rvs(loc_, scale_, size=100)
                test = ks_2samp(generated_samples, prior)
            elif d == 'chi2':
                df = best_distribution[d]['df']
                generated_samples = chi2.rvs(df, size=100)
                test = ks_2samp(generated_samples, prior)
            else:
                print(d)
                exit()
        return test

    def __fit_distributions(self, prior, likelihood):
        best_distribution = None
        summary = None
        if likelihood == 0:
            f = Fitter(prior,
                       #distributions=get_distributions()
                       distributions=[
                           #'uniform'
                           'norm',
                           't',
                           'cauchy',
                           'lognorm',
                           'expon',
                           'gamma',
                           'invgamma',
                           #'beta',
                           'chi2'
                       ]
                       )
            f.fit()
            summary = f.summary()
            best_distribution = f.get_best()
        return best_distribution, summary
    
    def __compute_posterior_various_distributions(self, prior, likelihood):
        if likelihood == 0:
            best_distribution, _ = self.__fit_distributions(prior, likelihood)
            prior = list(filter(lambda num: num != 0.0, prior))
            N = len(prior)
            d = list(best_distribution)[0]
            if d == 'gamma':
                alpha_ = best_distribution[d]['a']
                scale_ = 1/best_distribution[d]['scale']
                gamma_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ gamma({alpha_}, {scale_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = gamma_rng({alpha_}, {scale_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(gamma_distribution, random_seed=1)
            elif d == 't':
                df_ = best_distribution[d]['df']
                loc_ = best_distribution[d]['loc']
                scale_ = best_distribution[d]['scale']
                t_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ student_t({df_}, {loc_}, {scale_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = student_t_rng({df_}, {loc_}, {scale_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(t_distribution, random_seed=1)
            elif d == 'chi2':
                df_ = best_distribution[d]['df']
                chi_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ chi_square({df_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = chi_square_rng({df_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(chi_distribution, random_seed=1)
            elif d == 'expon':
                beta_ = best_distribution[d]['scale']
                expon_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ exponential({beta_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = exponential_rng({beta_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(expon_distribution, random_seed=1)
            elif d == 'cauchy':
                mu_ = best_distribution[d]['loc']
                sigma_ = best_distribution[d]['scale']
                cauchy_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ cauchy({mu_}, {sigma_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = cauchy_rng({mu_}, {sigma_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(cauchy_distribution, random_seed=1)
            elif d == 'invgamma':
                alpha_ = best_distribution[d]['a']
                beta_ = best_distribution[d]['scale']
                invgamma_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ inv_gamma({alpha_}, {beta_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = inv_gamma_rng({alpha_}, {beta_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(invgamma_distribution, random_seed=1)
            elif d == 'lognorm':
                mu_ = best_distribution[d]['s']
                sigma_ = best_distribution[d]['scale']
                invgamma_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ lognormal({mu_}, {sigma_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = lognormal_rng({mu_}, {sigma_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(invgamma_distribution, random_seed=1)
            elif d == 'beta':
                alpha_ = best_distribution[d]['a']
                beta_ = best_distribution[d]['b']
                alpha_, beta_, c, d = beta.fit(np.array(prior), floc = 0, fscale=1)
                beta_distribution = f"""
                parameters {{
                real<lower=0,upper=1> mu;
                }}
                model {{
                mu ~ beta({alpha_},{beta_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = beta_rng({alpha_}, {beta_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(beta_distribution, random_seed=1)
            elif d == 'uniform':
                alpha_ = best_distribution[d]['loc']
                beta_ = best_distribution[d]['scale'] + alpha_
                invgamma_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ uniform({alpha_}, {beta_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = uniform_rng({alpha_}, {beta_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(invgamma_distribution, random_seed=1)
            elif d == 'norm':
                alpha_ = best_distribution[d]['loc']
                beta_ = best_distribution[d]['scale']
                invgamma_distribution = f"""
                parameters {{
                  real<lower=0, upper=1> mu;
                }}
                model {{
                  mu ~ normal({alpha_}, {beta_});
                }}
                generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = normal_rng({alpha_}, {beta_});
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
                }}
                """
                posterior = stan.build(invgamma_distribution, random_seed=1)
            else:
                print(d)
                exit()
        else:
            if likelihood < 0:
                x = list(np.random.uniform(0, 0.01, size=abs(likelihood)))
            elif likelihood > 0:
                x = list(np.random.uniform(0.99, 1, size=abs(likelihood)))
            else:
                probability = None
            data = {
                "n": abs(likelihood),
                "x": x
            }
            prior = list(filter(lambda num: num != 0.0, prior))
            N = len(prior)
            uniform_distribution = f"""
            data {{
            int n;
              real x[n];
            }}
            parameters {{
              real<lower=0,upper=1> mu;
              real<lower=0> sigma;
            }}
            model {{
              for (i in 1:n)
                x[i] ~ normal(mu,sigma);
              mu ~ uniform(0,1);
              sigma ~ gamma(1,1);
            }}
            generated quantities {{
                vector[{N}] simulated;
                int max_indicator;
                int min_indicator;
                for (i in 1:{N}){{
                simulated[i] = normal_rng(0, 1);
                }}
                max_indicator = max(simulated) > max({prior});
                min_indicator = min(simulated) > min({prior});
            }}
            """
            posterior = stan.build(uniform_distribution, data = data, random_seed=1)
        fit = posterior.sample(num_chains = 5, num_samples = 2000)
        return fit.to_frame()

    def __compute_posterior(self, prior, likelihood):
        if likelihood == 0:
            prior = list(filter(lambda num: num != 0.0, prior))
            alpha_, beta_, c, d = beta.fit(np.array(prior), floc = 0, fscale=1)
            beta_distribution = f"""
            parameters {{
              real<lower=0,upper=1> mu;
            }}
            model {{
              mu ~ beta({alpha_},{beta_});
            }}
            """
            posterior = stan.build(beta_distribution, random_seed=1)
        else:
            if likelihood < 0:
                x = list(np.random.uniform(0, 0.05, size=abs(likelihood)))
            elif likelihood > 0:
                x = list(np.random.uniform(0.95, 1, size=abs(likelihood)))
            else:
                probability = None
            data = {
                "n": abs(likelihood),
                "x": x
            }
            uniform_distribution = f"""
            data {{
            int n;
              real x[n];
            }}
            parameters {{
              real<lower=0,upper=1> mu;
              real<lower=0> sigma;
            }}
            model {{
              for (i in 1:n)
                x[i] ~ normal(mu,sigma);
              mu ~ uniform(0,1);
              sigma ~ gamma(1,1);
            }}
            """
            posterior = stan.build(uniform_distribution, data = data, random_seed=1)
        fit = posterior.sample(num_chains = 5, num_samples = 2000)
        return fit.to_frame()
        
    def summarise(self, demonstration = False):
        e1_arrow_e2 = pd.DataFrame()
        e2_arrow_e1 = pd.DataFrame()
        files = [f for f in listdir(self.corpora_path) if isfile(join(self.corpora_path, f))]
        for i in files:
            if i.startswith("test") and i.endswith(".tsv"):
                corpus = i.replace(".tsv", "")[17:]
                df = pd.read_csv(self.corpora_path + i, sep='\t', header=0)
                if e1_arrow_e2.empty or e2_arrow_e1.empty:
                    e1_arrow_e2 = df[["sentence_source", "label", "label_notes", "sentence", "e1", "e2", "e1_e2"]].copy()
                    e2_arrow_e1 = e1_arrow_e2.copy()
                    if demonstration:
                        e1_arrow_e2["likelihood_e1_arrow_e2"] = e1_arrow_e2.apply(lambda x: self.__get_likelihood_demonstration(x["e1"], x["e2"]), axis = 1)
                        e2_arrow_e1["likelihood_e2_arrow_e1"] = e2_arrow_e1.apply(lambda x: self.__get_likelihood_demonstration(x["e2"], x["e1"]), axis = 1)
                    else:
                        e1_arrow_e2["likelihood_e1_arrow_e2"] = e1_arrow_e2.apply(lambda x: self.__get_likelihood(x["e1"], x["e2"]), axis = 1)
                        e2_arrow_e1["likelihood_e2_arrow_e1"] = e2_arrow_e1.apply(lambda x: self.__get_likelihood(x["e2"], x["e1"]), axis = 1)
                df[["count_e1", "count_e2", "count_e1_e2"]] = df[["count_e1", "count_e2", "count_e1_e2"]].replace(",", "", regex=True).astype(int)
                e1_arrow_e2[corpus + "_e1_arrow_e2"] = (df["count_e1_e2"]/df["count_e1"])/(df["count_e1_e2"]/df["count_e2"]+df["count_e1_e2"]/df["count_e1"])
                e2_arrow_e1[corpus + "_e2_arrow_e1"] = (df["count_e1_e2"]/df["count_e2"])/(df["count_e1_e2"]/df["count_e2"]+df["count_e1_e2"]/df["count_e1"])
        self.e1_arrow_e2 = e1_arrow_e2
        self.e2_arrow_e1 = e2_arrow_e1

    def __pad(self, l, content, width):
        l.extend([content] * (width - len(l)))
        return l
        
    def compare_posteriors(self, lower = None, upper = None):
        result = []
        bayesfactor = []
        ks_e1_arrow_e2 = []
        ks_e2_arrow_e1 = []
        max_e1_arrow_e2 = []
        max_e2_arrow_e1 = []
        min_e1_arrow_e2 = []
        min_e2_arrow_e1 = []
        best_distribution_e1_arrow_e2 = []
        best_distribution_e2_arrow_e1 = []
        distribution_summary_e1_arrow_e2 = []
        distribution_summary_e2_arrow_e1 = []
        upper = self.e1_arrow_e2.shape[0] if upper == None else upper
        lower = 0 if lower == None else lower
        for i in range(lower, upper):
            e1 = self.e1_arrow_e2["e1"][i]
            e2 = self.e2_arrow_e1["e2"][i]
            all_e1_arrow_e2 = [x for x in self.e1_arrow_e2[self.e1_arrow_e2.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
            all_e2_arrow_e1 = [x for x in self.e2_arrow_e1[self.e2_arrow_e1.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
            if self.e1_arrow_e2["likelihood_e1_arrow_e2"][i] > 0 and self.e2_arrow_e1["likelihood_e2_arrow_e1"][i] == 0:
                self.e2_arrow_e1.loc[i, "likelihood_e2_arrow_e1"] = -self.e1_arrow_e2["likelihood_e1_arrow_e2"][i]
            if self.e2_arrow_e1["likelihood_e2_arrow_e1"][i] > 0 and self.e1_arrow_e2["likelihood_e1_arrow_e2"][i] == 0:
                self.e1_arrow_e2.loc[i, "likelihood_e1_arrow_e2"] = -self.e2_arrow_e1["likelihood_e2_arrow_e1"][i]
                
            #fit_e1_arrow_e2 = self.__compute_posterior(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
            fit_e1_arrow_e2 = self.__compute_posterior_various_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
            #test_e1_arrow_e2 = self.__conduct_kolmogorov_smirnov(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
            test_e1_arrow_e2 = self.__conduct_kolmogorov_smirnov_various_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])         
            distribution_e1_arrow_e2, summary_e1_arrow_e2 = self.__fit_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])

            #fit_e2_arrow_e1 = self.__compute_posterior(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
            fit_e2_arrow_e1 = self.__compute_posterior_various_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
            #test_e2_arrow_e1 = self.__conduct_kolmogorov_smirnov(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
            test_e2_arrow_e1 = self.__conduct_kolmogorov_smirnov_various_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
            distribution_e2_arrow_e1, summary_e2_arrow_e1 = self.__fit_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
            
            #print(fit_e1_arrow_e2)
            #print(fit_e2_arrow_e1)
            
            BayesFactor = importr('BayesFactor')
            robjects.globalenv["fit_e1_arrow_e2"] = fit_e1_arrow_e2
            robjects.globalenv["fit_e2_arrow_e1"] = fit_e2_arrow_e1
            r('bf = ttestBF(x=fit_e1_arrow_e2$mu, y=fit_e2_arrow_e1$mu, nullInterval = c(0, Inf))')
            answer = r('bf[1]/bf[2]')
            
            b = re.search(r': (.*?)±', str(answer)).group(1)
            try:
                bayesfactor.append(float(b))
            except:
                bayesfactor.append(b)
                
            if b == "NaNe-Inf ":
                t = e2 + " \u2192 " + e1 + " " + str(0)
                result.append(0)
            elif b == "NA ":
                t = e1 + " \u2192 " + e2 + " " + str(1)
                result.append(1)
            elif float(b) > 1:
                t = e1 + " \u2192 " + e2 + " " + str(1)
                result.append(1)
            elif float(b) < 1:
                t = e2 + " \u2192 " + e1 + " " + str(0)
                result.append(0)
            else:
                t = e2 + " \u2192 " + e1 + " " + str(-1)
                result.append(-1)

            print(b)
            
            ks_e1_arrow_e2.append(str(test_e1_arrow_e2))
            ks_e2_arrow_e1.append(str(test_e2_arrow_e1))
            max_e1_arrow_e2.append(str(statistics.mean(fit_e1_arrow_e2['max_indicator'])))
            max_e2_arrow_e1.append(str(statistics.mean(fit_e2_arrow_e1['max_indicator'])))
            min_e1_arrow_e2.append(str(statistics.mean(fit_e1_arrow_e2['min_indicator'])))
            min_e2_arrow_e1.append(str(statistics.mean(fit_e2_arrow_e1['min_indicator'])))
            best_distribution_e1_arrow_e2.append(distribution_e1_arrow_e2)
            best_distribution_e2_arrow_e1.append(distribution_e2_arrow_e1)
            distribution_summary_e1_arrow_e2.append(summary_e1_arrow_e2)
            distribution_summary_e2_arrow_e1.append(summary_e2_arrow_e1)

        l = [None]*lower
        result = l + result
        bayesfactor = l + bayesfactor
        ks_e1_arrow_e2 = l + ks_e1_arrow_e2
        ks_e2_arrow_e1 = l + ks_e2_arrow_e1
        max_e1_arrow_e2 = l + max_e1_arrow_e2
        max_e2_arrow_e1 = l + max_e2_arrow_e1
        min_e1_arrow_e2 = l + min_e1_arrow_e2
        min_e2_arrow_e1 = l + min_e2_arrow_e1
        best_distribution_e1_arrow_e2 = l + best_distribution_e1_arrow_e2
        best_distribution_e2_arrow_e1 = l + best_distribution_e2_arrow_e1
        distribution_summary_e1_arrow_e2 = l + distribution_summary_e1_arrow_e2
        distribution_summary_e2_arrow_e1 = l + distribution_summary_e2_arrow_e1
        
        result = self.__pad(result, None, self.e1_arrow_e2.shape[0])
        bayesfactor = self.__pad(bayesfactor, None, self.e1_arrow_e2.shape[0])
        ks_e1_arrow_e2 = self.__pad(ks_e1_arrow_e2, None, self.e1_arrow_e2.shape[0])   
        ks_e2_arrow_e1 = self.__pad(ks_e2_arrow_e1, None, self.e1_arrow_e2.shape[0])
        max_e1_arrow_e2 = self.__pad(max_e1_arrow_e2, None, self.e1_arrow_e2.shape[0])   
        max_e2_arrow_e1 = self.__pad(max_e2_arrow_e1, None, self.e1_arrow_e2.shape[0])
        min_e1_arrow_e2 = self.__pad(min_e1_arrow_e2, None, self.e1_arrow_e2.shape[0])   
        min_e2_arrow_e1 = self.__pad(min_e2_arrow_e1, None, self.e1_arrow_e2.shape[0])
        best_distribution_e1_arrow_e2 = self.__pad(best_distribution_e1_arrow_e2, None, self.e1_arrow_e2.shape[0])   
        best_distribution_e2_arrow_e1 = self.__pad(best_distribution_e2_arrow_e1, None, self.e1_arrow_e2.shape[0]) 
        distribution_summary_e1_arrow_e2 = self.__pad(distribution_summary_e1_arrow_e2, None, self.e1_arrow_e2.shape[0]) 
        distribution_summary_e2_arrow_e1 = self.__pad(distribution_summary_e2_arrow_e1, None, self.e1_arrow_e2.shape[0]) 

        self.e1_arrow_e2.insert(2, "predicted", result, True)
        self.e2_arrow_e1.insert(2, "predicted", result, True)
        self.e1_arrow_e2.insert(3, "ks_test", ks_e1_arrow_e2, True)
        self.e2_arrow_e1.insert(3, "ks_test", ks_e2_arrow_e1, True)
        self.e1_arrow_e2.insert(4, "best_distribution", best_distribution_e1_arrow_e2, True)
        self.e2_arrow_e1.insert(4, "best_distribution", best_distribution_e2_arrow_e1, True)
        self.e1_arrow_e2.insert(5, "distribution_summary", distribution_summary_e1_arrow_e2, True)
        self.e2_arrow_e1.insert(5, "distribution_summary", distribution_summary_e2_arrow_e1, True)
        self.e1_arrow_e2.insert(6, "bayesfactor", bayesfactor, True)
        self.e2_arrow_e1.insert(6, "bayesfactor", bayesfactor, True)
        self.e1_arrow_e2.insert(7, "max_test", max_e1_arrow_e2, True)
        self.e2_arrow_e1.insert(7, "max_test", max_e2_arrow_e1, True)
        self.e1_arrow_e2.insert(8, "min_test", min_e1_arrow_e2, True)
        self.e2_arrow_e1.insert(8, "min_test", min_e2_arrow_e1, True)
        
    def __rescale_likelihoods(self, x, *args):
        """
        The function that will you be applied to your y-axis ticks.
        """
        x = float(x)/120
        return "{:.1f}".format(x)

    def __rescale_posteriors(self, x, *args):
        """
        The function that will you be applied to your y-axis ticks.
        """
        x = float(x)/7
        return "{:.1f}".format(x)
    
    def plot_posteriors(self, i):
        e1 = self.e1_arrow_e2["e1"][i]
        e2 = self.e2_arrow_e1["e2"][i]
        all_e1_arrow_e2 = [x for x in self.e1_arrow_e2[self.e1_arrow_e2.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
        all_e2_arrow_e1 = [x for x in self.e2_arrow_e1[self.e2_arrow_e1.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
        if self.e1_arrow_e2["likelihood_e1_arrow_e2"][i] > 0 and self.e2_arrow_e1["likelihood_e2_arrow_e1"][i] == 0:
            self.e2_arrow_e1.loc[i, "likelihood_e2_arrow_e1"] = -self.e1_arrow_e2["likelihood_e1_arrow_e2"][i]
        if self.e2_arrow_e1["likelihood_e2_arrow_e1"][i] > 0 and self.e1_arrow_e2["likelihood_e1_arrow_e2"][i] == 0:
            self.e1_arrow_e2.loc[i, "likelihood_e1_arrow_e2"] = -self.e2_arrow_e1["likelihood_e2_arrow_e1"][i]
            
        #fit_e1_arrow_e2 = self.__compute_posterior(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        fit_e1_arrow_e2 = self.__compute_posterior_various_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        test_e1_arrow_e2 = self.__conduct_kolmogorov_smirnov(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        distribution_e1_arrow_e2, summary_e1_arrow_e2 = self.__fit_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        
        #fit_e2_arrow_e1 = self.__compute_posterior(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        fit_e2_arrow_e1 = self.__compute_posterior_various_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        test_e2_arrow_e1 = self.__conduct_kolmogorov_smirnov(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        distribution_e2_arrow_e1, summary_e2_arrow_e1 = self.__fit_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        
        print(fit_e1_arrow_e2)
        print(statistics.mean(fit_e1_arrow_e2['max_indicator']))
        print(statistics.mean(fit_e1_arrow_e2['min_indicator']))
        
        print(fit_e2_arrow_e1)
        print(statistics.mean(fit_e2_arrow_e1['max_indicator']))
        print(statistics.mean(fit_e2_arrow_e1['min_indicator']))
        
        BayesFactor = importr('BayesFactor')
        robjects.globalenv["fit_e1_arrow_e2"] = fit_e1_arrow_e2
        robjects.globalenv["fit_e2_arrow_e1"] = fit_e2_arrow_e1
        r('bf = ttestBF(x=fit_e1_arrow_e2$mu, y=fit_e2_arrow_e1$mu, nullInterval = c(0, Inf))')
        answer = r('bf[1]/bf[2]')

        b = re.search(r': (.*?)±', str(answer)).group(1)

        print(answer)
        
        if b == "NaNe-Inf ":
            print(e2 + " \u2192 " + e1, 0)
        elif b == "NA ":
            print(e1 + " \u2192 " + e2, 1)
        elif float(b) > 1:
            print(e1 + " \u2192 " + e2, 1)
        else:
            print(e2 + " \u2192 " + e1, 0)
        
        plt.figure().clear()
        plt.hist(fit_e1_arrow_e2["mu"], density = True, color = 'b', alpha = 0.5, bins = 100, label="f(" + e1 + " \u2192 " + e2 + " | X)")
        plt.hist(fit_e2_arrow_e1["mu"], density = True, color = 'r', alpha = 0.5, bins = 100, label="f(" + e2 + " \u2192 " + e1 + " | X)")
        plt.xlabel("Probability", fontsize=12)
        plt.ylabel("Certainty", fontsize=12)
        plt.legend(loc='upper right', prop={'size': 12})
        
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(self.__rescale_posteriors))
        
        plt.show()

    def compute_posteriors(self, i):
        e1 = self.e1_arrow_e2["e1"][i]
        e2 = self.e2_arrow_e1["e2"][i]
        all_e1_arrow_e2 = [x for x in self.e1_arrow_e2[self.e1_arrow_e2.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
        all_e2_arrow_e1 = [x for x in self.e2_arrow_e1[self.e2_arrow_e1.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
        if self.e1_arrow_e2["likelihood_e1_arrow_e2"][i] > 0 and self.e2_arrow_e1["likelihood_e2_arrow_e1"][i] == 0:
            self.e2_arrow_e1.loc[i, "likelihood_e2_arrow_e1"] = -self.e1_arrow_e2["likelihood_e1_arrow_e2"][i]
        if self.e2_arrow_e1["likelihood_e2_arrow_e1"][i] > 0 and self.e1_arrow_e2["likelihood_e1_arrow_e2"][i] == 0:
            self.e1_arrow_e2.loc[i, "likelihood_e1_arrow_e2"] = -self.e2_arrow_e1["likelihood_e2_arrow_e1"][i]

        #fit_e1_arrow_e2 = self.__compute_posterior(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        fit_e1_arrow_e2 = self.__compute_posterior_various_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        #test_e1_arrow_e2 = self.__conduct_kolmogorov_smirnov(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])
        test_e1_arrow_e2 = self.__conduct_kolmogorov_smirnov_various_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])         
        distribution_e1_arrow_e2 = self.__fit_distributions(all_e1_arrow_e2, self.e1_arrow_e2["likelihood_e1_arrow_e2"][i])

        #fit_e2_arrow_e1 = self.__compute_posterior(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        fit_e2_arrow_e1 = self.__compute_posterior_various_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        #test_e2_arrow_e1 = self.__conduct_kolmogorov_smirnov(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        test_e2_arrow_e1 = self.__conduct_kolmogorov_smirnov_various_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        distribution_e2_arrow_e1 = self.__fit_distributions(all_e2_arrow_e1, self.e2_arrow_e1["likelihood_e2_arrow_e1"][i])
        
        BayesFactor = importr('BayesFactor')
        robjects.globalenv["fit_e1_arrow_e2"] = fit_e1_arrow_e2
        robjects.globalenv["fit_e2_arrow_e1"] = fit_e2_arrow_e1
        r('bf = ttestBF(x=fit_e1_arrow_e2$mu, y=fit_e2_arrow_e1$mu, nullInterval = c(0, Inf))')
        answer = r('bf[1]/bf[2]')
            
        b = re.search(r': (.*?)±', str(answer)).group(1)
        
        if b == "NaNe-Inf ":
            s = e2 + " \u2192 " + e1 + "\t" + str(0)
        elif b == "NA ":
            s = e1 + " \u2192 " + e2 + "\t" + str(1)
        elif float(b) > 10:
            s = e1 + " \u2192 " + e2 + "\t" + str(1)
        elif float(b) < 10:
            s = e2 + " \u2192 " + e1 + "\t" + str(0)
        else:
            s = e1 + " \u2192 " + e2 + "\t" + str(2)
        print(s)
        
        with open('sample.tsv', 'a') as f:
            f.write(str(i) + "\t" + s + "\t" + str(test_e1_arrow_e2) + "\t" + str(distribution_e1_arrow_e2) + "\t" + str(test_e2_arrow_e1) + "\t" + str(distribution_e2_arrow_e1) + "\n")
            
    def plot_likelihoods(self, i):
        e1 = self.e1_arrow_e2["e1"][i]
        e2 = self.e2_arrow_e1["e2"][i]
        y = list(np.random.uniform(0, 0.01, size=abs(10000)))
        x = list(np.random.uniform(0.99, 1, size=abs(10000)))
        
        plt.hist(x,  density = True, color = "b", alpha = 0.5, bins = 100, label="f(X | " + e1 + " \u2192 " + e2 + ")")
        plt.hist(y,  density = True, color = "r", alpha = 0.5, bins = 100, label="f(X | " + e2 + " \u2192 " + e1 + ")")
        plt.xlabel("Probability", fontsize=12)
        plt.ylabel("Certainty", fontsize=12)
        plt.legend(loc='upper right', prop={'size': 12})
        
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(self.__rescale_likelihoods))
        
        plt.show()

    def plot_uniform(self):
        values = np.random.uniform(0, 1, 5000)
        count, bins, ignored = plt.hist(values, 20, color = 'b', density=True)
        plt.plot(bins, np.ones_like(bins), color='r')
        plt.xlabel("Probability", fontsize=12)
        plt.ylabel("Certainty", fontsize=12)
        plt.show()
        
    def plot_beta(self, i):
        e1 = self.e1_arrow_e2["e1"][i]
        e2 = self.e2_arrow_e1["e2"][i]
        all_e1_arrow_e2 = [x for x in self.e1_arrow_e2[self.e1_arrow_e2.columns[8:]].iloc[i].tolist() if not math.isnan(x)]
        all_e2_arrow_e1 = [x for x in self.e2_arrow_e1[self.e2_arrow_e1.columns[8:]].iloc[i].tolist() if not math.isnan(x)]

        alpha_e1_arrow_e2, beta_e1_arrow_e2, c, d = beta.fit(np.array(all_e1_arrow_e2), floc = 0, fscale=1)
        alpha_e2_arrow_e1, beta_e2_arrow_e1, c, d = beta.fit(np.array(all_e2_arrow_e1), floc = 0, fscale=1)

        x = np.arange(-50, 50, 0.1)
        
        y_e1_arrow_e2 = beta.pdf(x, alpha_e1_arrow_e2, beta_e1_arrow_e2, scale=100, loc=-50)
        y_e2_arrow_e1 = beta.pdf(x, alpha_e2_arrow_e1, beta_e2_arrow_e1, scale=100, loc=-50)

        print(e1, e2)
        
        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #fig.suptitle('Horizontally stacked subplots')
        #ax1.plot(x, y_e2_arrow_e1, label="sine")
        #ax2.plot(x, y_e1_arrow_e2, label="cosine")

        plt.plot(x, y_e1_arrow_e2, "-b", label="P(" + e1 + " \u2192 " + e2 + ")")
        plt.plot(x, y_e2_arrow_e1, "-r", label="P(" + e2 + " \u2192 " + e1 + ")")
        plt.legend(loc="upper right")
        plt.ylim(0, 0.175)
        plt.show()

        #x = np.arange (-50, 50, 0.1)
        #y = beta.pdf(x, alpha_, beta_, scale=100, loc=-50)
        #plt.plot(x, y)
        #plt.show()
        
        #plt.hist(all_e1_arrow_e2,  density = True, color = "darkblue", alpha = 0.5, bins = 100, label="P(X | " + e1 + " \u2192 " + e2 + ")")
        #plt.hist(all_e2_arrow_e1,  density = True, color = "red", alpha = 0.5, bins = 100, label="P(X | " + e2 + " \u2192 " + e1 + ")")
        #plt.xlabel("Probability", fontsize=18)
        #plt.ylabel("Certainty", fontsize=18)
        #plt.legend(loc='upper right', prop={'size': 20})

        #ax = plt.gca()
        #ax.yaxis.set_major_formatter(mtick.FuncFormatter(self.__rescale))
        
        #plt.show()
        
if __name__ == "__main__":
    
    DATA1 = False
    DATA2 = False
    SemEval2007 = False
    SemEval2007_5_percent = False
    SemEval2010 = False
    SemEval2010_05_percent = False
    Poster = False
    Error = True
    
    if DATA1:
        corpora_path = "../data/SemEval2010_task8/prior/"
        training_data = "../data/SemEval2010_task8/SemEval2010_task8_training/TRAIN_FILE.txt"
        result_path = "../data/SemEval2010_task8/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        temp, _ = bayesian.get_tokens_semeval2010()
        counts = Counter(temp)
        labels, values = zip(*counts.items())
        labels = labels[:10]
        values = values[:10]

        # sort your values in descending order
        indSort = np.argsort(values)[::-1]

        # rearrange your data
        labels = np.array(labels)[indSort]
        values = np.array(values)[indSort]

        indexes = np.arange(len(labels))

        bar_width = 0

        plt.bar(indexes, values)

        # add labels
        plt.xticks(indexes + bar_width, labels)
        plt.xlabel("Top 10 entities", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.show()

    if DATA2:
        corpora_path = "../data/SemEval2010_task8/prior/"
        training_data = "../data/SemEval2010_task8/SemEval2010_task8_training/TRAIN_FILE.txt"
        result_path = "../data/SemEval2010_task8/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        temp, likelihoods = bayesian.get_tokens_semeval2010()
        d = dict(sorted(likelihoods.items(), key=lambda item: item[1]))
        print(d)

    if SemEval2007:
        corpora_path = "../data/SemEval2007_task4/prior/"
        training_data = "../data/SemEval2007_task4/task-4-training/relation-1-train.txt"
        result_path = "../data/SemEval2007_task4/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        bayesian.compute_likelihood_semeval2007()
        bayesian.summarise()
        #bayesian.plot_posteriors(2)
        bayesian.compare_posteriors()
        bayesian.save_e1_arrow_e2(result_path + "e1_arrow_e2.csv")
        bayesian.save_e2_arrow_e1(result_path + "e2_arrow_e1.csv")

    if SemEval2007_5_percent:
        corpora_path = "../data/SemEval2007_task4/prior/"
        training_data = "../data/SemEval2007_task4/task-4-training/relation-1-train.txt"
        result_path = "../data/SemEval2007_task4/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        bayesian.compute_likelihood_semeval2007([])
        #bayesian.compute_likelihood_semeval2007([1, 4, 11, 67, 82, 84])
        #bayesian.compute_likelihood_semeval2007([1, 4, 11, 67, 82])
        bayesian.summarise()
        #bayesian.plot_posteriors(32)
        bayesian.compare_posteriors()
        bayesian.save_e1_arrow_e2(result_path + "e1_arrow_e2_5_obs.csv")
        bayesian.save_e2_arrow_e1(result_path + "e2_arrow_e1_5_obs.csv")
        
    if SemEval2010:
        lower = 270
        upper = 328
        corpora_path = "../data/SemEval2010_task8/prior/"
        training_data = "../data/SemEval2010_task8/SemEval2010_task8_training/TRAIN_FILE.txt"
        result_path = "../data/SemEval2010_task8/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        bayesian.compute_likelihood_semeval2010(0)
        bayesian.summarise()
        bayesian.compare_posteriors(lower, upper)
        #bayesian.plot_posteriors(0)
        bayesian.save_e1_arrow_e2(result_path + "e1_arrow_e2_filtered" + str(lower) + "_" + str(upper) + ".csv")
        bayesian.save_e2_arrow_e1(result_path + "e2_arrow_e1_filtered" + str(lower) + "_" + str(upper) + ".csv")

    if SemEval2010_05_percent:
        corpora_path = "../data/SemEval2010_task8/prior/"
        training_data = "../data/SemEval2010_task8/SemEval2010_task8_training/TRAIN_FILE.txt"
        result_path = "../data/SemEval2010_task8/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        bayesian.compute_likelihood_semeval2010(39)
        bayesian.summarise()
        for i in range(0,10):
            bayesian.compute_posteriors(i)
        #bayesian.compare_posteriors()
        #bayesian.save_e1_arrow_e2(result_path + "e1_arrow_e2.csv")
        #bayesian.save_e2_arrow_e1(result_path + "e2_arrow_e1.csv")
        
    if Poster:
        corpora_path = "../data/SemEval2010_task8/poster/"
        training_data = "../data/SemEval2010_task8/SemEval2010_task8_training/TRAIN_FILE.txt"
        result_path = "../data/SemEval2010_task8/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        bayesian.compute_likelihood_semeval2010()
        bayesian.summarise(demonstration = True)
        #bayesian.plot_posteriors(0)
        bayesian.plot_likelihoods(0)
        #bayesian.plot_uniform()
        #bayesian.plot_beta(0)
        #bayesian.compare_posteriors()
        #bayesian.save_e1_arrow_e2(result_path + "e1_arrow_e2_poster.csv")
        #bayesian.save_e2_arrow_e1(result_path + "e2_arrow_e1_poster.csv")

    if Error:
        corpora_path = "../data/SemEval2010_task8/prior/"
        training_data = "../data/SemEval2010_task8/SemEval2010_task8_training/TRAIN_FILE.txt"
        result_path = "../data/SemEval2010_task8/result/"
        bayesian = BayesianInference(corpora_path, training_data)
        bayesian.compute_likelihood_semeval2010(0)
        bayesian.summarise(demonstration = True)
        bayesian.plot_posteriors(16)
        #bayesian.plot_likelihoods(16)
        #bayesian.plot_uniform()
        #bayesian.plot_beta(0)
        #bayesian.compare_posteriors()
        #bayesian.save_e1_arrow_e2(result_path + "e1_arrow_e2_poster.csv")
        #bayesian.save_e2_arrow_e1(result_path + "e2_arrow_e1_poster.csv")
