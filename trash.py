

import random
import numpy
from numpy import genfromtxt
import png
from sklearn import preprocessing


def generate_random_data(number_of_variables, number_of_patients):
	##
	## create a csv file with
	## random variables
	##

	data_file_name = "trash_data.csv"
	variables_to_values = {}

	## Generate data
	for x in xrange(0, number_of_variables):
		variable_name = "variable_"+str(x)
		vector = []
		min_value = random.randint(0,25)
		max_value = random.randint(85,100)
		for y in xrange(0, number_of_patients):
			scalar = random.randint(min_value, max_value)
			vector.append(scalar)
		variables_to_values[variable_name] = vector

	## Write data
	output_file = open(data_file_name, "w")

	## write header
	header = ""
	for key in variables_to_values.keys():
		header+=str(key)+","
	header = header[:-1]
	output_file.write(header+"\n")

	## write patients
	for x in xrange(0, number_of_patients):
		patient = ""
		for y in xrange(0, number_of_variables):
			key = "variable_"+str(y)
			patient+=str(variables_to_values[key][x])+","
		patient = patient[:-1]
		output_file.write(patient+"\n")

	output_file.close()


def get_correlation_matrix(input_data_file):
	##
	## -> Compute and return the corrlation matrix
	## for the variables in input_data_file.
	## -> Assume every scalar could be cast to float 
	## -> the variables in the correlation matrix are in
	## the same order as the variables in the header
	##

	## parameters
	variables_to_values = {}
	index_to_variables = {}

	## get data
	input_data = open(input_data_file, "r")
	cmpt = 0
	for line in input_data:
		line = line.replace("\n", "")
		
		## get header
		if(cmpt == 0):
			line_in_array = line.split(",")
			index = 0
			for variable in line_in_array:
				index_to_variables[index] = variable
				variables_to_values[variable] = []
				index += 1

		## get scalar
		else:
			line_in_array = line.split(",")
			index = 0
			for scalar in line_in_array:
				variables_to_values[index_to_variables[index]].append(float(scalar))
				index += 1

		cmpt += 1
	input_data.close

	## Compute correlation matrix
	variables_matrix = []
	index = 0
	for key in variables_to_values.keys():
		variables_matrix.append([])
		variables_matrix[index] = variables_to_values[index_to_variables[index]]
		index += 1 

	correlation_matrix = numpy.corrcoef(variables_matrix)
	return correlation_matrix


def create_image_from_csv(data_file, image_file):
	##
	## -> Create an image from a csv file,
	## - the image have to be png
	## - the values in data_file have to be integer between 0 and 255
	## - drop the header in data_file
	##

	data = genfromtxt(data_file, delimiter=',')
	data = data[1:] ## drop the header
	matrix = []
	for vector in data:
		matrix.append(list(vector))

	## save the image
	png.from_array(matrix, 'L').save(image_file)



def normalize_data(data_file):
	##
	## -> Scale (centrer normer) the data
	## and write the data in new _scaled data
	## file
	##

	## Get the header
	cmpt = 0
	header = ""
	input_data = open(data_file, "r")
	for line in input_data:
		line = line.replace("\n", "")
		if(cmpt == 0):
			header = line
		cmpt += 1
	input_data.close()

	## Get and scale data
	data = genfromtxt(data_file, delimiter=',')
	data_scaled = preprocessing.scale(data[1:])

	## Write new file
	output_file_name = data_file.split(".")
	output_file_name = output_file_name[0]+"_scaled.csv"

	output_data = open(output_file_name, "w")
	output_data.write(header+"\n")

	for vector in data_scaled:
		vector_to_write = ""
		for scalar in vector:
			vector_to_write += str(scalar)+","
		vector_to_write = vector_to_write[:-1]
		output_data.write(vector_to_write+"\n")
	output_data.close()



### TEST SPACE ###
generate_random_data(4,4)
corr_mat = get_correlation_matrix("trash_data.csv")
create_image_from_csv("trash_data.csv", "machin.png")
normalize_data("trash_data.csv")
