import random
from numpy import genfromtxt
from sklearn import preprocessing

## pre process data file & generate random data
## for the PATREP package.


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

