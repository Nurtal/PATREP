import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import representation


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


image_structure = representation.build_image_map("trash_data_scaled.csv")
representation.build_patient_representation("trash_data_scaled_interpolated.csv", image_structure)
plot_log_file("learning_optimal_grid.log")