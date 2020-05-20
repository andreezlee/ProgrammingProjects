import torch
import pickle
import numpy as np
from random import seed, random, randint
from shared import pj
from ffnn import FFNN

"""
	Readied values for each region, list of tuples ordered by 
	(number of random points, [latitude bounds, longitude bounds])
"""
balkans = [(50, [37, 46, 18, 27])]
pacific_rim = [(50, [-60, 60, 120, 180]), (25, [20, 60,-180, -100]), (25, [-60, 0, -180, -60])]

"""
	Outputs the list of indices of the features/networks with the
	highest model accuracies, in descending order.
"""
def most_accurate_features():
	with open(pj + "feature_index.pickle", 'rb') as file:
		feature_index_lookup = pickle.load(file)
	with open(pj + "feature_id_to_name.pickle", 'rb') as file:
		feature_name_lookup = pickle.load(file)
	with open(pj + "model_accuracy.pickle", 'rb') as file:
		model_accuracies = pickle.load(file)
	print("Usable features: {}".format(len(model_accuracies) - list(model_accuracies).count(0)))
	most_accurate_results = sorted(range(len(model_accuracies)), key=lambda i: model_accuracies[i])[-50:]

	for i in most_accurate_results:
		f_id = feature_index_lookup[i]
		f_name = feature_name_lookup[f_id]
		#print("{} : {}".format(f_id, f_name))
		#print("Accuracy: {:.2f}".format(model_accuracies[i]))

	return most_accurate_results

"""
	Given the list of feature indices, runs the models for the specified features
	on random points in defined areas, outputting all predictions for each feature.
"""
def run_on_features(features):
	seed(randint(0, 10))
	locations = []
	regions = pacific_rim
	for a, b in regions:
		for i in range(a):
			lat = random() * (b[1] - b[0]) + b[0]
			lon = random() * (b[3] - b[2]) + b[2]
			locations.append(np.array([lat, lon]))

	with open(pj + "trained_models.pickle", 'rb') as file:
		all_models = pickle.load(file)

	results = []
	for i in features:
		predictions = []
		for j in locations:
			input_vector = np.concatenate([j, j])
			prediction = all_models[i][0](input_vector)
			label = torch.argmax(prediction)
			predictions.append(label)
		results.append((i, predictions))

	return results

"""
	Parses through the [results] data structure to print out all the proper
	results: feature names, values, distributions, etc.
"""
def manage_results(results):

	with open(pj + "feature_index.pickle", 'rb') as file:
		feature_index_lookup = pickle.load(file)
	with open(pj + "feature_id_to_name.pickle", 'rb') as file:
		feature_name_lookup = pickle.load(file)
	with open(pj + "feature_value_to_name.pickle", 'rb') as file:
		value_name_lookup = pickle.load(file)

	for feature, values in results:
		f_id = feature_index_lookup[feature]
		f_name = feature_name_lookup[f_id]
		print("{} : {}".format(f_id, f_name))

		vf = {}
		for i in values:
			i = int(i)
			if i not in vf:
				vf[i] = 1
			else:
				vf[i] += 1

		for i in vf.keys():
			vf_id = f_id + "-" + str(i)
			vf_name = value_name_lookup[vf_id]
			print("{} : {}".format(vf_name, vf[i]))


if __name__ == "__main__":
	results = most_accurate_features()
	results = run_on_features(results)
	manage_results(results)