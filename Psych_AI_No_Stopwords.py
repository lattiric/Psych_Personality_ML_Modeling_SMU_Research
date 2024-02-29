# Import dependencies

import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
from statistics import mean, mode
import sys

# Read in the data
data = pd.read_json('../training_data_8.30.json')
outname = sys.argv[1]
# epochs = sys.argv[2]
# learning = sys.argv[3]
print(outname)
# Apply preprocessing: remove lines with no data, standardize, create new dataframe with text and numerical label
data = data[data[outname+'_scaled'] != '.']
scaler = StandardScaler()
data[outname+'_scaled_new'] = scaler.fit_transform(data[outname+'_scaled'].to_numpy().reshape(-1, 1))

data_text = data['text'].astype(str).tolist()
data_labels = data[outname+'_scaled_new'].astype(float).tolist()
data_text_size = data['text'].apply(lambda x: len(x.split())).tolist()

# Form into DataFrame
train_data = pd.DataFrame(
    {'text': data_text,
     'labels': data_labels,
     'text_size': data_text_size
     })

# #takes only the middle quartile of the text from each interview
# all_middle_text = []
# for patient_text in train_data['text']:
#     split_text = patient_text.split()
#     first_quartile_num = int(((len(split_text))/4)*1) #originally got middle quartiles
#     last_quartile_num = int(((len(split_text))/4)*3)
#     middle_text = split_text[first_quartile_num:last_quartile_num+1]
#     middle_text_added  = ""
#     for word in middle_text:
#         middle_text_added = middle_text_added + " " + word
#     all_middle_text.append(middle_text_added)

# # Form into DataFrame
# train_data = pd.DataFrame(
#     {'text': all_middle_text,
#      'labels': data_labels,
#      'text_size': data_text_size
#      })

# ----------------------------------------- removing stopwords (below)

stop_words = set(stopwords.words('english'))
all_nostop_text = []
for patient_text in train_data['text']:
    split_text = patient_text.split()
    nostop_text = ""
    for word in split_text:
       if word not in stop_words:
           nostop_text = nostop_text + " " + word
    all_nostop_text.append(nostop_text)
           
# Form into DataFrame
train_data = pd.DataFrame(
    {'text': all_nostop_text,
     'labels': data_labels,
     'text_size': data_text_size
     })

# ----------------------------------------- taking middle 80% (below)

#order the data by size of text and take the to 80%
labels_by_order = train_data.sort_values(by=['text_size'])['labels'].tolist()
text_by_order = train_data.sort_values(by=['text_size'])['text'].tolist()
labels_by_order = labels_by_order[int(len(labels_by_order)*0.2):] #where the 80% is determined
text_by_order = text_by_order[int(len(labels_by_order)*0.2):]

# shuffle the top 80% of the data
temp = list(zip(text_by_order, labels_by_order))
np.random.shuffle(temp)
text_by_order, labels_by_order = zip(*temp)
data_text_rand = list(text_by_order)
data_labels_rand = list(labels_by_order)

train_data = pd.DataFrame(
    {'text': text_by_order,
     'labels': labels_by_order
     })

# ----------------------------------------------- 

# Split data into folds
kk=5
kf = KFold(n_splits=kk, shuffle=True, random_state=1234)

count = 0
Results = []

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

for train_index, val_index in kf.split(train_data):
    count += 1
    print("Fold = " + str(count))

    # split data
    training = train_data.iloc[train_index]
    validation = train_data.iloc[val_index]
    
    # Setup arguments
    model_args = ClassificationArgs(sliding_window=True)
    model_args.use_early_stopping = True
    model_args.early_stopping_metric = "r2"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.num_train_epochs = 30
    model_args.learning_rate = 2e-5
    model_args.evaluate_during_training = True
    model_args.regression = True    
    model_args.hidden_dropout_prob = 0.2
    model_args.train_batch_size = 48 #originally 32
    model_args.eval_batch_size = 24 # orginally 16
    model_args.evaluate_during_training_silent = True
    model_args.evaluate_during_training_steps = 64
    #model_args.manual_seed = 4
    model_args.max_seq_length = 512
    model_args.no_cache = True
    model_args.no_save = True
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.gradient_accumulation_steps = 12
    model_args.train_custom_parameters_only = False
    model_args.save_best_model = True
    # added
    model_args.fp16 = True
    model_args.gpu = 8

    # Create a TransformerModel
    model = ClassificationModel(
        "xlmroberta",
        "xlm-roberta-large",
        num_labels=1,
        args=model_args,
        use_cuda=True,
        )

    # Train the model
    # Output predictions seem to be printed each time the model is evaluated during training. I commented out the print line, but don't know if that will make a difference.
    model.train_model(
      training,
      eval_df=validation,
      r2=lambda truth, predictions: r2_score(truth, predictions),
      mse=lambda truth, predictions: mean_squared_error(truth, predictions),
      mae=lambda truth, predictions: mean_absolute_error(truth, predictions),
    )

    # evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(validation, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
    Results.append([result['r2'], result['mse'], result['mae']])

    # Calculate average performance across folds
# result averages
total_r2 = 0
total_mse = 0
total_mae = 0

for result in Results:
    total_r2 += result[0]
    total_mse += result[1]
    total_mae += result[2]

print("Final result of K-Fold")
print("r2", total_r2/kk)
print("mse", total_mse/kk)
print("mae", total_mae/kk)

# Apply to test model
data = pd.read_json('../test_data_8.30.json')

# Apply preprocessing: remove lines with no data, standardize, create new dataframe with text and numerical label
test_data = data[data[outname+'_scaled'] != '.']
scaler = StandardScaler()
test_data[outname+'_scaled_new'] = scaler.fit_transform(test_data[outname+'_scaled'].to_numpy().reshape(-1, 1))

test_text = test_data['text'].astype(str).tolist()
test_labels = test_data[outname+'_scaled_new'].astype(float).tolist()

test_data = pd.DataFrame(
    {'text': test_text,
     'labels': test_labels
     })
# Evaluate the Model
result, model_outputs, wrong_predictions = model.eval_model(test_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
print("results", result)

# Save the model
model.model.save_pretrained(outname+'model1')
model.tokenizer.save_pretrained(outname+'model1')
model.config.save_pretrained(outname+'model1/')