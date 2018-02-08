import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import os

import representation
import preprocessing
import classification


def plot_log_file(log_file):
	##
	## [IN PROGRESS]
	##
	## => Plot values of scores in log file
	##
	##

	## Init data structure
	global_scores = []

	## Get the values
	data = open(log_file)
	for line in data:
		line = line.replace("\n", "")
		line_in_array = line.split(";")
		if(line_in_array[0] == "global_score"):
			global_scores.append(float(line_in_array[1])) 
	data.close()

	## plot the values
	plt.plot(global_scores)
	plt.show()




def save_matrix_to_file(matrix, save_file):
	##
	## => Save the content of matrix (wich is a 2d array)
	## in the file save_file
	##

	output_file = open(save_file, "w")
	for vector in matrix:
		line_to_write = ""
		for scalar in vector:
			line_to_write += str(scalar) + ","
		line_to_write = line_to_write[:-1]
		output_file.write(line_to_write+"\n")
	output_file.close()



def load_matrix_from_file(load_file):
	##
	## => Create a matrix from a save_file,
	## cast the matrix into an numpy array
	## return the matrix
	##

	matrix = []
	input_data = open(load_file, "r")
	for line in input_data:
		vector = []
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		for scalar in line_in_array:
			vector.append(float(scalar))
		matrix.append(vector)
	input_data.close()

	matrix = numpy.array(matrix)

	return matrix



###------------###
### TEST SPACE ###
###------------###


#image_structure = representation.build_image_map("trash_data_scaled.csv")
#representation.build_patient_representation("trash_data_scaled_interpolated.csv", image_structure)
#plot_log_file("learning_optimal_grid.log")

## Test on real external dataset
"""
preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
image_structure = representation.build_image_map("datasets/creditcard_reduce_reformated_scaled.csv", 5)
save_matrix_to_file(image_structure, "credit_image_structure.csv")
representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
real_data = representation.build_patient_matrix("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
classification.run_CNN(train_X, train_Y, test_X, test_Y, 20)
plot_log_file("learning_optimal_grid.log")
"""

## Test on HLA data
"""
preprocessing.reformat_input_datasets("datasets/HLA_data_clean.csv", 562, True)
preprocessing.normalize_data("datasets/HLA_data_clean_reformated.csv")
image_structure = representation.build_image_map("datasets/HLA_data_clean_reformated_scaled.csv", 500)
save_matrix_to_file(image_structure, "HLA_image_structure_500i.csv")
representation.simple_conversion_to_img_matrix("datasets//HLA_data_clean_reformated_scaled.csv")
representation.build_patient_representation("datasets//HLA_data_clean_reformated_scaled_interpolated.csv", image_structure)
real_data = representation.build_patient_matrix("datasets//HLA_data_clean_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
classification.run_CNN(train_X, train_Y, test_X, test_Y, 20)

plot_log_file("learning_optimal_grid.log")
"""

## Create grids for HLA data
## One grid / pathology computed with control
"""
iteration_list = [750]
dataset_list = ["datasets/HLA_data_MCTD.csv", "datasets/HLA_data_PAPs.csv", "datasets/HLA_data_RA.csv", "datasets/HLA_data_SjS.csv", "datasets/HLA_data_SLE.csv", "datasets/HLA_data_SSc.csv", "datasets/HLA_data_UCTD.csv"]

for iteration in iteration_list:

	for dataset in dataset_list:
		dataset_reformated = dataset.split(".")
		dataset_reformated = dataset_reformated[0]+"_reformated.csv"
		dataset_reformated_scaled = dataset_reformated.split(".")
		dataset_reformated_scaled = dataset_reformated_scaled[0]+"_scaled.csv"
		save_matrix_name = dataset.split(".")
		save_matrix_name = save_matrix_name[0]+"_"+str(iteration)+".csv"

		if(not os.path.isfile(save_matrix_name)):
			preprocessing.reformat_input_datasets(dataset, 562, True)
			preprocessing.normalize_data(dataset_reformated)
			image_structure = representation.build_image_map(dataset_reformated_scaled, iteration)
			save_matrix_to_file(image_structure, save_matrix_name)
"""

## Perform classification on HLA data, for each disease vs control
## IN PROGRESS
"""
disease_list = ["MCTD", "PAPs", "RA", "SjS", "SLE", "SSc", "UCTD"]
for disease in disease_list:
	for iteration in [50, 150, 500, 750]:

		print " => PROCESS [DISEASE]"+ str(disease) +" [MAP ITERATION]"+ str(iteration) +" \n"

		preprocessing.reformat_input_datasets("datasets/HLA_data_"+str(disease)+".csv", 562, True)
		preprocessing.normalize_data("datasets/HLA_data_"+str(disease)+"_reformated.csv")
		image_structure = load_matrix_from_file("datasets/HLA_data_"+str(disease)+"_"+str(iteration)+".csv")
		representation.simple_conversion_to_img_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled.csv")
		representation.build_patient_representation("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 350)

"""

## Perform Classification on customer data from Kaggle
preprocessing.reformat_input_datasets("datasets/data_customer_kaggle.csv", 370, True)
preprocessing.normalize_data("datasets/data_customer_kaggle_reformated.csv")
image_structure = representation.build_image_map("datasets/data_customer_kaggle_reformated_scaled.csv", 150)

#preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
#preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
#image_structure = representation.build_image_map("datasets/creditcard_reduce_reformated_scaled.csv", 5)

save_matrix_to_file(image_structure, "data_customer_kaggle_structure.csv")
representation.simple_conversion_to_img_matrix("datasets/data_customer_kaggle_reformated_scaled.csv")
representation.build_patient_representation("datasets/data_customer_kaggle_reformated_scaled_interpolated.csv", image_structure)
real_data = representation.build_patient_matrix("datasets/data_customer_kaggle_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
classification.run_CNN(train_X, train_Y, test_X, test_Y, 90)
plot_log_file("learning_optimal_grid.log")