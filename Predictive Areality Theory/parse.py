import csv
import pickle
import numpy as np
from shared import dbf, pj, parse_lang_feat

"""
	Parses csv files in [database_files]
	Creates dataset files for learning and testing
"""

#Create the array specs
with open(dbf + "feature_names.csv", newline='') as features_csv:
	features = csv.reader(features_csv, delimiter=',')
	feature_index = []
	feature_id_to_name = {}
	for row in features:
		#Skip first line
		f_id = row[0]
		if f_id != "ID":
			feature_index.append(f_id)
			feature_id_to_name[f_id] = row[1]
	num_features = len(feature_index)

	#List of all features, to find feature <-> index
	pickle.dump(feature_index, open(pj + "feature_index.pickle", 'wb'))
	#Full names of all features
	pickle.dump(feature_id_to_name, open(pj + "feature_id_to_name.pickle", 'wb'))

#Create data for testing
with open(dbf + "languages.csv", newline='') as languages_csv:
	languages = csv.reader(languages_csv, delimiter=',')
	feature_data = {}
	location_data = {}
	family_data = {}
	family_locations = {}
	for row in languages:
		l_id = row[0]
		if l_id != "ID":
			#Initialize dict of numpy arrays
			loc = np.array([float(row[3]), float(row[4])])
			fam = row[7]
			feature_data[l_id] = np.zeros(num_features)
			location_data[l_id] = loc
			if fam is not "":
				family_data[l_id] = fam
				if fam in family_locations:
					old_loc, num = family_locations[fam]
					family_locations[fam] = (loc + old_loc, num + 1)
				else:
					family_locations[fam] = (loc, 1)
	#Use family locations to find centroid of each family
	for i in family_locations:
		loc, num = family_locations[i]
		family_locations[i] = loc / num

	#Mapping language to location
	pickle.dump(location_data, open(pj + "location_data.pickle", 'wb'))
	#Stores the latitude-longitude centroid for each family
	pickle.dump(family_locations, open(pj + "family_location_data.pickle", 'wb'))
	#Stores the family for each non-isolate language
	pickle.dump(family_data, open(pj + "family_data.pickle", 'wb'))


with open(dbf + "language_features.csv", newline='') as full_data_csv:
	language_features = csv.reader(full_data_csv, delimiter=',')
	for row in language_features:
		if row[0] != "ID":
			l_id = row[1]
			f_index =feature_index.index(row[2])
			f_val = row[3]
			feature_data[l_id][f_index] = f_val

	#Vector of features, indexed by language
	pickle.dump(feature_data, open(pj + "feature_data.pickle", 'wb'))

with open(dbf + "feature_categories.csv", newline='') as categories_csv:
	feature_categories = csv.reader(categories_csv, delimiter=',')
	value_to_name = {}
	num_categories = {}
	for row in feature_categories:
		cat_id = row[0]
		if cat_id != "ID":
			feat, val = parse_lang_feat(cat_id)
			assert(val == int(row[4]))
			assert(feat == row[1])
			value_to_name[cat_id] = row[2]

			#Find number of values for each feature
			num_categories[feat] = val

	#Maps feature categories to name
	pickle.dump(value_to_name, open(pj + "feature_value_to_name.pickle", 'wb'))
	#Stores the number of values possible for a given feature
	pickle.dump(num_categories, open(pj + "feature_num_categories.pickle", 'wb'))
