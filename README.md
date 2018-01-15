# eeg-deep-learning
Proof of concept deep neural networks for EEG pattern classification (interictal vs preictal) 

input_eeg_float_subj: input pipelines for EEG data, using the Dataset Tensorflow interface

eeg_cnn_preictal: definition of CNN architecture, loss function and evaluation metric (including CAE)

train_cnn_preictal: training and evaluation on in-sample and out-of-sample set 

visualize_cnn_multiscale: visualization of layer weights and deep dream type visualizations (adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)


