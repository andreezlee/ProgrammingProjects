import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import pickle
import time
from tqdm import tqdm
from shared import pj

class FFNN(nn.Module):
	def __init__(self, input_dim, output_dim, h):
			super(FFNN, self).__init__()
			self.h = h
			self.W1 = nn.Linear(input_dim, h)
			self.activation = nn.ReLU()
			self.W2 = nn.Linear(h, output_dim)
			# The below two lines are not a source for an error
			self.softmax = nn.LogSoftmax(dim=0) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
			self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		input_vector = torch.from_numpy(input_vector).float()
		# The z_i are just there to record intermediary computations for your clarity
		z1 = self.W1(input_vector)
		z2 = self.W2(self.activation(z1))
		predicted_vector = self.softmax(z2)
		return predicted_vector


def train_on_data(hidden_dim, number_of_epochs):
	print("Fetching data")
	#Load pickle files
	with open(pj + "family_data.pickle", 'rb') as file:
		language_family_lookup = pickle.load(file)
	with open(pj + "family_location_data.pickle", 'rb') as file:
		family_location_lookup = pickle.load(file)
	with open(pj + "feature_data.pickle", 'rb') as file:
		feature_data = pickle.load(file)
	with open(pj + "feature_id_to_name.pickle", 'rb') as file:
		feature_name_lookup = pickle.load(file)
	with open(pj + "feature_index.pickle", 'rb') as file:
		feature_index_lookup = pickle.load(file)
	with open(pj + "feature_num_categories.pickle", 'rb') as file:
		feature_output_length_lookup = pickle.load(file)
	with open(pj + "feature_value_to_name.pickle", 'rb') as file:
		feature_value_name_lookup = pickle.load(file)
	with open(pj + "location_data.pickle", 'rb') as file:
		location_lookup = pickle.load(file)
	print("Fetched and indexed data")
	#Randomize validation and training sets
	num_features = len(feature_index_lookup)
	data_list = list(feature_data.keys())
	data_length = len(data_list)
	training_cutoff = round(0.9 * data_length)
	print("Vectorized data: {} languages, {} features".format(data_length, num_features))
	print("Training set: {} languages".format(training_cutoff))
	print("Validation set: {} languages".format(data_length - training_cutoff))

	#Create a model for each feature
	all_models = []
	for i in range(num_features):
		output_length = feature_output_length_lookup[feature_index_lookup[i]] + 1
		model = FFNN(input_dim=4, output_dim=output_length, h=hidden_dim)
		# This network is trained by traditional (batch) gradient descent
		optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
		all_models.append((model, optimizer))

	#Performance measure: final validation accuracy stored for each model
	model_accuracy = np.zeros(len(all_models))

	print("Training {} models for {} epochs".format(len(all_models), number_of_epochs))
	for i in range(num_features):
		model, optimizer = all_models[i]
		print("Training started for model {}".format(i + 1))
		for epoch in range(number_of_epochs):
			model.train()
			optimizer.zero_grad()
			total_loss = 0
			loss = 0
			correct = 0
			total = 0
			start_time = time.time()
			print("Training started for epoch {}".format(epoch + 1))
			train_data = data_list[:training_cutoff]
			random.shuffle(train_data) # Good practice to shuffle order of training data
			minibatch_size = 16 
			N = len(train_data)
			for minibatch_index in tqdm(range(N // minibatch_size)):
				optimizer.zero_grad()
				loss = None
				for example_index in range(minibatch_size):
					#Form training example
					example_language = train_data[minibatch_index * minibatch_size + example_index]
					language_location = location_lookup[example_language]
					if example_language in language_family_lookup:
						family_location = family_location_lookup[language_family_lookup[example_language]]
					else:
						family_location = language_location
					input_vector = np.concatenate([language_location, family_location])
					gold_label = feature_data[example_language][i]

					predicted_vector = model(input_vector)
					predicted_label = torch.argmax(predicted_vector)
					correct += int(predicted_label == gold_label)
					total += 1
					example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label], dtype=torch.long))
					total_loss += example_loss
					if loss is None:
						loss = example_loss
					else:
						loss += example_loss
				loss = loss / minibatch_size
				loss.backward()
				optimizer.step()
			print("Training completed for epoch {}".format(epoch + 1))
			print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
			print("Average loss for epoch {}: {}".format(epoch + 1, total_loss / N))
			print("Training time for this epoch: {}".format(time.time() - start_time))
			model.train(mode=False)
			#loss = 0
			correct = 0
			total = 0
			start_time = time.time()
			print("Validation started for epoch {}".format(epoch + 1))
			valid_data = data_list[training_cutoff:]
			random.shuffle(valid_data) # Good practice to shuffle order of valid data
			for j in range(len(valid_data)):
				#Form validation example
				example_language = valid_data[j]
				language_location = location_lookup[example_language]
				if example_language in language_family_lookup:
						family_location = family_location_lookup[language_family_lookup[example_language]]
				else:
					family_location = language_location
				input_vector = np.concatenate([language_location, family_location])
				gold_label = feature_data[example_language][i]

				predicted_vector = model(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1

			print("Validation completed for epoch {}".format(epoch + 1))
			print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
			print("Validation time for this epoch: {}".format(time.time() - start_time))

			if epoch == number_of_epochs - 1:
				model_accuracy[i] = correct / total
	pickle.dump(all_models, open(pj + "trained_models.pickle", 'wb'))
	pickle.dump(model_accuracy, open(pj + "model_accuracy.pickle", 'wb'))
	return model_accuracy

train_on_data(128, 10)
"""
#Fine-tuning on hidden layer size
layer_sizes = [16, 32, 64, 128, 256]
accuracy_by_size = [0 for i in layer_sizes]
for i in range(5):
	results = train_on_data(layer_sizes[i], 10)
	accuracy_by_size[i] = sum(results)/len(results)
best = layer_sizes[accuracy_by_size.index(max(accuracy_by_size))]
print("Best layer size is {}".format(best))
#RESULTS: 128 has the best average accuracy
"""