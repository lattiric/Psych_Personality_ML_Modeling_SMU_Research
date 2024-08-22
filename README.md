# Personality Prediction Through ML Modeling

The goal of this project is to create a machine learning model that can successfully predict the personality of a subject based off of a single conversation with them.

##### Note: 
This project is still in progress, so the results stated in this read me may not be up to the current date. This is the state of the project as of May 11th, 2024

### Team
This project was originally proposed by Josh Oltmanns, and has grown to incorporate work from a team of students at Xavier University, as well as myself and Dr. Tue Vu on behalf of the Data Center at Southern Methodist University. The work you see here is the adaptation and improvement on code from Jocelyn Brickman at Xavier University, done by myself and Dr. Tue Vu.

### Primary Use Case
The idea behind this project is to create a machine learning model that could be used by therapists to generate a well rounded description of a patient's personality within the first therapy session. The patient would have a simple conversation with the tharpist within the session and the model would then be make predictions on the 35 personality characteristics included in the `NEO-PI-R Personality Test`. These characteristics would give the therapis a much better insight on their patient and allow them to be much more effective in helping them.

### Model Construction
The current model was trained on a dataset of 1000+ interview responses or real patients, paired with their corresponding scores for the 35 personality traits of the  `NEO-PI-R Personality Test`. This was then trained on a simple transformer, transfer learning from XLM_RoBERTa as the baseline of the model. A model of this type was trained to specifically predict each on of the 35 personality traits, meaning 35 different models are implemented in the prediction of a single patient's scores. All of the models are trained using the SMU data center's NVIDIA Superpod.

### Training Process
TODO

### Current Results
TODO

### Bottlenecks
TODO