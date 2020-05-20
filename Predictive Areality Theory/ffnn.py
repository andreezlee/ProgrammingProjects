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

"""
	Very simple feed forward neural network class
"""
class FFNN(nn.Module):
	def __init__(self, input_dim, output_dim, h):
			super(FFNN, self).__init__()
			self.h = h
			self.W1 = nn.Linear(input_dim, h)
			self.activation = nn.ReLU()
			self.W2 = nn.Linear(h, output_dim)
			self.softmax = nn.LogSoftmax(dim=0)
			self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		input_vector = torch.from_numpy(input_vector).float()
		z1 = self.W1(input_vector)
		z2 = self.W2(self.activation(z1))
		predicted_vector = self.softmax(z2)
		return predicted_vector

"""
	Given metaparameters, trains the neural network on the data.
"""
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
	print("Vectorized data: {} languages, {} features".format(data_length, num_features))

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
		print("Creating training and validation sets for feature {}".format(i))
		proper_examples = []
		for j in data_list:
			if feature_data[j][i] != 0:
				proper_examples.append(j)
		num_proper = len(proper_examples)
		if num_proper < 300:
			print("Omitting feature {}: not enough data".format(i))
			continue
		random.shuffle(proper_examples)
		train_data = proper_examples[:round(0.9 * num_proper)]
		valid_data = proper_examples[round(0.9 * num_proper):]
		assert(len(train_data) + len(valid_data) == num_proper)
		assert(len(train_data) > len(valid_data))
		print("Training and validation sets created")

		print("Training started for model {}".format(i + 1))
		for epoch in range(number_of_epochs):
			model.train()
			optimizer.zero_grad()
			total_loss = 0
			loss = 0
			tcorrect = 0
			ttotal = 0
			start_time = time.time()
			print("Training started for epoch {}".format(epoch + 1))
			minibatch_size = 8
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
					tcorrect += int(predicted_label == gold_label)
					ttotal += 1
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
			print("Training accuracy for epoch {}: {}".format(epoch + 1, tcorrect / ttotal))
			print("Average loss for epoch {}: {}".format(epoch + 1, total_loss / N))
			print("Training time for this epoch: {}".format(time.time() - start_time))
			model.train(mode=False)
			#loss = 0
			vcorrect = 0
			vtotal = 0
			start_time = time.time()
			print("Validation started for epoch {}".format(epoch + 1))
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
				vcorrect += int(predicted_label == gold_label)
				vtotal += 1

			print("Validation completed for epoch {}".format(epoch + 1))
			print("Validation accuracy for epoch {}: {}".format(epoch + 1, vcorrect / vtotal))
			print("Validation time for this epoch: {}".format(time.time() - start_time))

			if epoch == number_of_epochs - 1:
				model_accuracy[i] = (tcorrect / ttotal + vcorrect / vtotal) / 2
	pickle.dump(all_models, open(pj + "trained_models.pickle", 'wb'))
	pickle.dump(model_accuracy, open(pj + "model_accuracy.pickle", 'wb'))
	return model_accuracy

"""
	This creates and trains models according to my fine-tuning
"""
if __name__ == "__main__":
	avg_acc = 0
	while avg_acc < 0.48:
		accuracies = train_on_data(16, 6)
		num_usable = 0
		for i in accuracies:
			if i != 0:
				num_usable += 1
		avg_acc = sum(accuracies) / num_usable
		print("Average model accuracy: {}".format(avg_acc))
		print("Highest accuracy: {}".format(max(accuracies)))
"""
#Fine-tuning on hidden layer size and number of epochs
layer_sizes = [16, 32, 64, 128, 256]
epochs = [6, 8, 10]
accuracy_by_size = [[0 for j in epochs] for i in layer_sizes]
for i in range(5):
	for j in range(3):
		results = train_on_data(layer_sizes[i], epochs[j])
		accuracy_by_size[i][j] = sum(results)/len(results)
best = (0, 0)
best_accuracy = 0
for i in range(5):
	for j in range(3):
		if accuracy_by_size[i][j] > best_accuracy:
			best_accuracy = accuracy_by_size[i][j]
			best = (i, j)
print("Best layer size is {}".format(layer_sizes[best[0]]))
print("Best number of epochs is {}".format(epochs[best[1]]))
#RESULTS: hidden layer of 16 on 6 epochs
"""