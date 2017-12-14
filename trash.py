

import random
import numpy
from numpy import genfromtxt
import png
from sklearn import preprocessing
from scipy.interpolate import interp1d
import math


## TODO
##
## => Function to map scaled values to [0,255] interval
##
##
##
##



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




def simple_conversion_to_img_matrix(data_file):
	##
	## -> very basic conversion from
	##  normalized data to a 0 - 255 ranged
	##  values.
	##
	## -> for each variables get the min
	## and max values, then map each scalar
	## of the variables to [0-255]
	##
	## => TODO:
	##	- find a cool name for the function
	##


	## init structures
	variables_to_maxmin = {}
	index_to_variables = {}
	variables_to_values = {}
	variables_to_interpolatevalues = {}

	## Read input data
	## - fill the structures
	## - find min and max values for each variables
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:

		line = line.replace("\n", "")

		## Parse the header
		if(cmpt == 0):

			line_in_array = line.split(",")
			index = 0
			for variable in line_in_array:
				index_to_variables[index] = variable
				variables_to_maxmin[variable] = {"max":-765, "min":765}
				variables_to_values[variable] = []
				variables_to_interpolatevalues[variable] = []
				index +=1

		
		## - Get max value for each variables
		## - Get min value for each variables
		else:

			line_in_array = line.split(",")
			index = 0
			for scalar in line_in_array:
				variable = index_to_variables[index]
				variables_to_values[variable].append(float(scalar))
				variable_max = variables_to_maxmin[variable]["max"]
				variable_min = variables_to_maxmin[variable]["min"]
				if(float(scalar) > float(variable_max)):
					variables_to_maxmin[variable]["max"] = float(scalar)
				if(float(scalar) < float(variable_min)):
					variables_to_maxmin[variable]["min"] = float(scalar)
				index +=1
		cmpt += 1
	input_data.close()	


	## Map each variables to [0-255]
	## Use interpol1d from Scipy
	for variable in variables_to_values.keys():
		max_val = variables_to_maxmin[variable]["max"]
		min_val = variables_to_maxmin[variable]["min"]
		interpolation = interp1d([min_val,max_val],[0,255])
		number_of_patients = 0
		for scalar in variables_to_values[variable]:
			scalar_interpolated = interpolation(float(scalar))
			scalar_interpolated = int(scalar_interpolated)
			variables_to_interpolatevalues[variable].append(scalar_interpolated)
			number_of_patients += 1

	## Write new file with interpolated data
	output_file_name = data_file.split(".")
	output_file_name = output_file_name[0]+"_interpolated.csv"

	output_file = open(output_file_name, "w")

	## Write the header
	header = ""
	for variable in variables_to_interpolatevalues.keys():
		header += str(variable)+","
	header = header[:-1]	
	output_file.write(header+"\n")

	for x in xrange(0, number_of_patients):
		line_to_write = ""
		for variable in variables_to_interpolatevalues.keys():
			vector = variables_to_interpolatevalues[variable]
			line_to_write += str(vector[x])+","
		line_to_write = line_to_write[:-1]
		output_file.write(line_to_write+"\n")

	output_file.close()



def build_image_map(data_file):
	##
	## [IN PROGRESS]
	##
	## Core of the idea
	##
	##

	## get the correlation matrix
	corr_mat = get_correlation_matrix(data_file)
	#print corr_mat

	## compute the distance between each variables
	## kind of "distance matrix"
	## just use the absolute value of correlation, in this case
	## we consider that a highly negative value (anticorrelation)
	## is associated with the variable of interest same as a hihgly correlated
	## variable
	
	#print "="*75
	dist_mat = abs(corr_mat)
	#print dist_mat

	## for each variable get the 4 / 8 closest variables
	## compute a kind of score for the block
	##	- First ideas for the score : sum of the values from the dist matrix
	##
	## [NB] => might be hard to check the compatibility of the
	## best possibles blocks, stand by for now (use random initialisation with AG instead)
	
	#print "="*75
	
	variable_id_to_near_variables = {}
	variable_id = 0
	
	for vector in dist_mat:

		## Find best neighbours for each variable from the dist matrix
		best_neighbours_in_vector = numpy.argpartition(vector, -5)[-5:]
		best_neighbours_in_vector = best_neighbours_in_vector[numpy.argsort(vector[best_neighbours_in_vector])]
		best_neighbours_in_vector = list(best_neighbours_in_vector)
		best_neighbours_in_vector.remove(variable_id)
		best_neighbours_in_vector.reverse()		
		variable_id_to_near_variables[variable_id] = best_neighbours_in_vector
		variable_id += 1


	## Current idea : init the grid with an heuristic to maximize the score, or randomly
	## [CUSTOM ALGORITHM]:
	## use a genetic algorithm top optimise the global score of the grid (sum of the bloc score)
	## consider a bloc as an indivdual and each case as a gene
	## [CARTE AUTOGENERE]
	## Kohen, etc ...


	##-----------------------##
	## RANDOM INITIALISATION ##
	##-----------------------##
	## -> Randomy place the variables on a grid 
	## to generate an image

	## -> Get dimmension of the grid
	## [TO FIX] => for now assume we have
	## a square matrix, might be a good idea
	## to test a few things on the number of
	## variables to get the dimmensions of the
	## grid
	number_of_variables = len(corr_mat[0])
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)

	## Randomy assign a position in the grid to a variable
	variable_to_position = {}
	for x in xrange(0,number_of_variables):
		
		position_assigned = False

		while(not position_assigned):
			x_position = random.randint(0,side_size-1)
			y_position = random.randint(0,side_size-1)
			position = [x_position, y_position]

			if(position not in variable_to_position.values()):
				variable_to_position[x] = position
				position_assigned = True

	## Write a matrix with the assigned position as coordinates
	## for the variables
	
	## init the matrix
	## [WARNINGS] => Still assume we deal
	## with a square matrix / image for the representation
	## of the patient
	
	## Init matrix
	map_matrix = numpy.zeros(shape=(side_size,side_size))

	## Fill the matrix
	for variable in variable_to_position.keys():
		position = variable_to_position[variable]
		x_position = position[0]
		y_position = position[1]

		map_matrix[x_position][y_position] = variable







	## Get new Ideas 




### TEST SPACE ###
generate_random_data(36	,85)
corr_mat = get_correlation_matrix("trash_data.csv")
create_image_from_csv("trash_data.csv", "machin.png")
normalize_data("trash_data.csv")
simple_conversion_to_img_matrix("trash_data_scaled.csv")
create_image_from_csv("trash_data_scaled_interpolated.csv", "machin.png")

build_image_map("trash_data_scaled.csv")