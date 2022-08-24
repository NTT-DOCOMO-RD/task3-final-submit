#!/usr/bin/env python

import pandas as pd
import random
import tempfile

from shared.base_predictor import BasePredictor, PathType

import re
import time
import random
from datetime import datetime
import unicodedata

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation
import torch
from torch import cuda
from torch.utils.data import DataLoader


import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from scipy.special import softmax
import demoji

def str_normalize(s):
    norm_text = re.sub(r'(http|https)://([-\w]+\.)+[-\w]+(/[-\w./?%&=]*)?', "", s)
    norm_text = unicodedata.normalize("NFKC", norm_text)
    norm_text = demoji.replace(string=norm_text, repl="")
    
    return norm_text


class Task3Predictor(BasePredictor):
    def prediction_setup(self):
        """To be implemented by the participants.

        Participants can add the steps needed to initialize their models,
        and/or any other setup related things here.
        """
        pass

    def fix_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


    def predict(self,
                test_set_path: PathType,
                product_catalogue_path: PathType,
                predictions_output_path: PathType,
                register_progress=lambda x: print("Progress : ", x)):
        """To be implemented by the participants.

        Participants need to consume the test set (the path of which is passed) 
        and write the final predictions as a CSV file to `predictions_output_path`.

        Args:
            test_set_path: Path to the Test Set for the specific task.
            product_catalogue_path: Path to the product catalogue for the specific task.
            predictions_output_path: Output Path to write the predictions as a CSV file. 
            register_progress: A helper callable to register progress. Accepts a value [0, 1].
        """
        
        ####################################################################################################
        ####################################################################################################
        ## 
        ## STEP 1 : Read Test Set File
        ## 
        ####################################################################################################
        ####################################################################################################

        # Initial Setup
        BATCH_SIZE = 64
        num_labels = 2
        max_length = 512
        device = 'cuda' if cuda.is_available() else 'cpu'
        SEED = 2022
        self.fix_seed(SEED)

        # Calculate longest common subsequence
        def lcs_dp(a, b):
            c = [[0] * (len(b) + 1) for i in range(len(a) + 1)]

            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    if a[i-1] == b[j-1]:
                        c[i][j] = c[i-1][j-1] + 1
                    else:
                        c[i][j] = max(c[i][j-1], c[i-1][j])

            return c[-1][-1]

        def func_lcs(row):
            return lcs_dp(row['query'], row['product_title'])


        df = pd.read_csv(test_set_path)  # Read Test Set as a Pandas DataFrame
        product_df = pd.read_csv(product_catalogue_path) # Read the Product Catalogue as a Pandas DataFrame
        product_df.fillna('', inplace=True)

        df = pd.merge(
            df,
            product_df,
            left_on=['product_id','query_locale'],
            right_on=['product_id','product_locale'],
            how="left"
        )

        # Set data type
        df['query'] = df['query'].astype(str)
        df['query'] = df['query'].map(str_normalize)
        df['product_title'] = df['product_title'].astype(str)
        df['product_title'] = df['product_title'].map(str_normalize)

        # Create additional columns for features and post-processing
        df['product_merge'] = df['product_title'] + " " + df['product_description'] + " " + \
            df['product_bullet_point'] + " " + df['product_brand'] + " " + df['product_color_name']
        df['lcs'] = df.apply(func_lcs, axis=1)
        df['query_length'] = df['query'].str.len()
        df['lcs_ratio'] = df['lcs']/df['query_length']


        ###############################################################################################
        ###############################################################################################
        ## 
        ## STEP 3 : Register the progress of your inference (as a number belonging to [0, 1])
        ## 
        ###############################################################################################
        ###############################################################################################
        #     # Compute Progress Percentage
        #     progress = (_idx + 1) / len(all_example_ids)
        #     # Use helper function to notify progress back to the evaluation server
        #     register_progress(progress)
        # NOTE: It is important to use the register_progress function to announce your progress
        #       to the evaluation server - This will translate to the real-time progress displayed
        #       during the evaluation of your evaluation, and can help troubleshoot any
        #       failed submissions a lot easier.


        # Trained Models
        # Models for each language
        model_save_path_us = 'models/task3_model_us'
        model_save_path_jp = 'models/task3_model_jp'
        model_save_path_es = 'models/task3_model_es'

        # multi-lingual model
        model_save_path_md = 'models/model_mdeberta'

        model_us = AutoModelForSequenceClassification.from_pretrained(model_save_path_us).to(device)
        tokenizer_us = AutoTokenizer.from_pretrained(model_save_path_us)

        model_jp = AutoModelForSequenceClassification.from_pretrained(model_save_path_jp).to(device)
        tokenizer_jp = AutoTokenizer.from_pretrained(model_save_path_jp)

        model_es = AutoModelForSequenceClassification.from_pretrained(model_save_path_es).to(device)
        tokenizer_es = AutoTokenizer.from_pretrained(model_save_path_es)

        model_md = AutoModelForSequenceClassification.from_pretrained(model_save_path_md).to(device)
        tokenizer_md = AutoTokenizer.from_pretrained(model_save_path_md)

        ######################################## prediction by each language #############################################
        # US
        features_query = df[df.query_locale=='us']['query'].to_list()
        features_product = df[df.query_locale=='us']['product_title'].to_list()
        test_idx_us = df[df.query_locale=='us'].index

        features_query_all = df['query'].to_list()
        n_examples_all = len(features_query_all)

        n_examples_us = len(features_query)
        scores_us = np.empty((0, num_labels))

        start_time = time.time()
        with torch.no_grad():
            for i in range(0, n_examples_us, BATCH_SIZE):
                progress_1 = (i + 1) / n_examples_all
                register_progress( progress_1 / 2)

                j = min(i + BATCH_SIZE, n_examples_us)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                features = tokenizer_us(features_query_, features_product_,max_length = max_length, padding=True, truncation=True, return_tensors="pt").to(device)
                scores_us = np.vstack((scores_us, np.squeeze(model_us(**features).logits.cpu().detach().numpy())))
                i = j
        print('inference time US:', time.time() - start_time)

        # JP
        features_query = df[df.query_locale=='jp']['query'].to_list()
        features_product = df[df.query_locale=='jp']['product_title'].to_list()
        test_idx_jp = df[df.query_locale=='jp'].index

        n_examples_jp = len(features_query)
        scores_jp = np.empty((0, num_labels))

        start_time = time.time()
        with torch.no_grad():
            for i in range(0, n_examples_jp, BATCH_SIZE):
                progress_1 = (i + 1 + n_examples_us) / n_examples_all
                register_progress( progress_1 / 2)

                j = min(i + BATCH_SIZE, n_examples_jp)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                features = tokenizer_jp(features_query_, features_product_,max_length = max_length, padding=True, truncation=True, return_tensors="pt").to(device)
                scores_jp = np.vstack((scores_jp, np.squeeze(model_jp(**features).logits.cpu().detach().numpy())))
                i = j

        print('inference time JP:', time.time() - start_time)

        # ES
        features_query = df[df.query_locale=='es']['query'].to_list()
        features_product = df[df.query_locale=='es']['product_title'].to_list()
        test_idx_es = df[df.query_locale=='es'].index

        n_examples_es = len(features_query)
        scores_es = np.empty((0, num_labels))

        start_time = time.time()
        with torch.no_grad():
            for i in range(0, n_examples_es, BATCH_SIZE):
                progress_1 = (i + 1 + n_examples_us + n_examples_jp) / n_examples_all
                register_progress( progress_1 / 2)

                j = min(i + BATCH_SIZE, n_examples_es)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                features = tokenizer_es(features_query_, features_product_,max_length = max_length, padding=True, truncation=True, return_tensors="pt").to(device)
                scores_es = np.vstack((scores_es, np.squeeze(model_es(**features).logits.cpu().detach().numpy())))
                i = j

        print('inference time ES:', time.time() - start_time)

        # Concat predicted results of all languages
        df_pred_us = pd.DataFrame(scores_us)
        df_pred_us.index = test_idx_us
        df_pred_jp = pd.DataFrame(scores_jp)
        df_pred_jp.index = test_idx_jp
        df_pred_es = pd.DataFrame(scores_es)
        df_pred_es.index = test_idx_es
        df_pred = pd.concat([df_pred_us,df_pred_jp,df_pred_es], axis=0).sort_index()

        ######################################## prediction by mdeberta #############################################
        features_query = df['query'].to_list()
        features_product = df['product_merge'].to_list()
        n_examples = len(features_query)
        scores_mb = np.empty((0, num_labels))
        start_time = time.time()

        print('prediction by mdeberta')
        with torch.no_grad():
            for i in range(0, n_examples, BATCH_SIZE):
                progress_2 = (i + 1) / n_examples
                register_progress( (progress_2/2 + 1/2) )

                j = min(i + BATCH_SIZE, n_examples)
                features_query_ = features_query[i:j]
                features_product_ = features_product[i:j]
                features = tokenizer_md(features_query_, features_product_,max_length = max_length, padding=True, truncation=True, return_tensors="pt").to(device)
                scores_mb = np.vstack((scores_mb, np.squeeze(model_md(**features).logits.cpu().detach().numpy())))
                i = j
        print('inference time mdeberta:', time.time() - start_time)

        ######################################## ensemble #############################################
        def calc_prob_diff(score, score_example_id):
    
            scores = softmax(score, axis=1)
            scores_argsorted = np.argsort(scores, axis=1)
            score_2nd_largest_idx = scores_argsorted[:,-2]
            
            scores_df = pd.DataFrame(scores, columns=['prob0','prob1'])
            pred_label_top1 = pd.DataFrame(np.array(scores.argmax(1)), columns=['pred_label_top1'])
            pred_label_top2 = pd.DataFrame(score_2nd_largest_idx, columns=['pred_label_top2'])

            max_prob = pd.DataFrame(np.array(scores.max(1)), columns=['max_prob'])
            max_2nd_prob = pd.DataFrame(np.array(scores[np.arange(scores.shape[0]),score_2nd_largest_idx]), columns=['max_2nd_prob'])
            
            example_id = pd.DataFrame(score_example_id, columns=['example_id'])

            scores_df = pd.merge(scores_df,pred_label_top1,left_index=True,right_index=True)
            scores_df = pd.merge(scores_df,pred_label_top2,left_index=True,right_index=True)
            scores_df = pd.merge(scores_df,max_prob,left_index=True,right_index=True)
            scores_df = pd.merge(scores_df,max_2nd_prob,left_index=True,right_index=True)
            scores_df = pd.merge(scores_df,example_id,left_index=True,right_index=True)
            scores_df['prob_diff'] = scores_df['max_prob'] - scores_df['max_2nd_prob']
            
            return scores_df
        
        # Weighted ensemble
        score_ensemble = df_pred.values + 2*scores_mb

        # Calculate the probability of top1 and top2
        scores_test_df = calc_prob_diff(score_ensemble, df['example_id'].values)
        scores_test_df = pd.merge(scores_test_df,df[['example_id','lcs_ratio']],on='example_id',how='left')

        # Set top1 label as prediction
        scores_test_df['substitute_label'] = scores_test_df['pred_label_top1']

        # Rule-based post-processing
        cor_idx = scores_test_df[(scores_test_df.prob_diff<0.2)&(scores_test_df.lcs_ratio>0.8)].index
        scores_test_df.loc[cor_idx,'substitute_label'] = 0

        test_submit_output_df = scores_test_df[['example_id','substitute_label']].copy()
        test_submit_output_df['substitute_label'] = test_submit_output_df['substitute_label'].map({0:'no_substitute', 1:'substitute'})

        ####################################################################################################
        ####################################################################################################
        ## 
        ## STEP 4 : Save Predictions to `predictions_output_path` as a valid CSV file
        ## 
        ####################################################################################################
        ####################################################################################################
        # Generate a DataFrame from a List of Dictionaries
        # predictions_df = pd.DataFrame(PREDICTIONS)
        predictions_df = test_submit_output_df[['example_id','substitute_label']].copy()

        # Save predictions to the expected place
        print("Writing Task-3 Predictions to : ", predictions_output_path)
        predictions_df.to_csv(  predictions_output_path,
                                index=False, header=True)


if __name__ == "__main__":
    print('Download unidic')
    os.system('python -m unidic download')

    # Instantiate Predictor Class
    predictor = Task3Predictor()
    predictor.prediction_setup()
    
    test_set_path = "./data/task_3_product_substitute_identification/test_public-v0.3.csv.zip"
    product_catalogue_path = "./data/task_3_product_substitute_identification/product_catalogue-v0.3.csv.zip"

    # Generate a Random File to store predictions
    with tempfile.NamedTemporaryFile(suffix='.csv') as output_file:
        output_file_path = output_file.name

        # Make Predictions
        predictor.predict(
            test_set_path=test_set_path,
            product_catalogue_path=product_catalogue_path,
            predictions_output_path=output_file_path
        )
        
        ####################################################################################################
        ####################################################################################################
        ## 
        ## VALIDATIONS
        ## Adding some simple validations to ensure that the generated file has the expected structure
        ## 
        ####################################################################################################
        ####################################################################################################
        # Validating sample submission
        predictions_df = pd.read_csv(output_file_path)
        test_df = pd.read_csv(test_set_path)

        # Check-#1 : Sample Submission has "example_id" and "substitute_label" columns
        expected_columns = ["example_id", "substitute_label"]
        assert set(expected_columns) <= set(predictions_df.columns.tolist()), \
            "Predictions file's column names do not match the expected column names : {}".format(
                expected_columns)

        # Check-#2 : Sample Submission contains predictions for all example_ids
        predicted_example_ids = sorted(predictions_df["example_id"].tolist())
        expected_example_ids = sorted(test_df["example_id"].tolist())
        assert expected_example_ids == predicted_example_ids, \
            "`example_id`s present in the Predictions file do not match the `example_id`s provided in the test set"

        # Check-#3 : Predicted `substitute_label`s are valid
        VALID_OPTIONS = sorted(["no_substitute", "substitute"])
        predicted_substitute_labels = sorted(predictions_df["substitute_label"].unique())
        assert predicted_substitute_labels == VALID_OPTIONS, \
            "`substitute_label`s present in the Predictions file do not match the expected ESCI Lables : {}".format(
                VALID_OPTIONS)
