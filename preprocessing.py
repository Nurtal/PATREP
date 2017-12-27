
## TODO
##
## => Check importation
##

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
