import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import representation
import preprocessing


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





###------------###
### TEST SPACE ###
###------------###


#image_structure = representation.build_image_map("trash_data_scaled.csv")
#representation.build_patient_representation("trash_data_scaled_interpolated.csv", image_structure)
#plot_log_file("learning_optimal_grid.log")

## Test on rea external dataset
preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
image_structure = representation.build_image_map("datasets/creditcard_reduce_reformated_scaled.csv", 500)
representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
plot_log_file("learning_optimal_grid.log")