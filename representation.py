import numpy
from numpy import genfromtxt
import png
from scipy.interpolate import interp1d
import math
import random
import operator

from time import gmtime, strftime

## Represent observations as image.
## -> Learn the optimal structure for the image




def get_correlation_matrix(input_data_file):
	##
	## -> Compute and return the corrlation matrix
	## for the variables in input_data_file.
	## -> Assume every scalar could be cast to float 
	## -> the variables in the correlation matrix are in
	## the same order as the variables in the header
	##
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

	## warning : generate NaN when forced squared Matrix
	correlation_matrix = numpy.corrcoef(variables_matrix)
	
	## Replace NaN by 0
	correlation_matrix = numpy.where(numpy.isnan(correlation_matrix), numpy.ma.array(correlation_matrix, mask=numpy.isnan(correlation_matrix)).mean(axis=0), correlation_matrix)

	print "choucroute" + str( len(correlation_matrix) )

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






def init_grid_matrix(corr_mat):
	##
	## Just a function to instanciate
	## randomly a grid matrix
	##

	number_of_variables = len(corr_mat[0])
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)

	## Randomy assign a position in the grid to a variable
	variable_to_position = {}
	cmpt_assigned = 0 # debug
	for x in xrange(0,number_of_variables):
		
		position_assigned = False

		while(not position_assigned):
			x_position = random.randint(0,side_size-1)
			y_position = random.randint(0,side_size-1)
			position = [x_position, y_position]

			if(position not in variable_to_position.values()):
				variable_to_position[x] = position
				position_assigned = True

				cmpt_assigned += 1 # debug
				#print "=> " +str(cmpt_assigned) +" position assigned ["+str(float(float(cmpt_assigned)/float(number_of_variables))*100) +"%]"

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


	return map_matrix




def get_neighbour(var, grid_matrix):
	##
	## get the list of neighbour of var in grid_matrix
	##

	## locate var in grid
	x_position = -1
	y_position = -1

	row = 0
	for x in grid_matrix:
		column = 0
		for y in x:
			if(y == var):
				x_position = row
				y_position = column
			column += 1
		row += 1

	## Get the neighbour	
	neighbours = []

	if(y_position > 0):
		n_x_position = x_position
		n_y_position = y_position - 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position < len(grid_matrix[0])-1):
		n_x_position = x_position
		n_y_position = y_position + 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(x_position > 0):
		n_x_position = x_position - 1
		n_y_position = y_position
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(x_position < len(grid_matrix)-1):
		n_x_position = x_position + 1
		n_y_position = y_position
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)

	if(y_position > 0 and x_position > 0):
		n_x_position = x_position - 1
		n_y_position = y_position - 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position > 0 and x_position < len(grid_matrix)-1):
		n_x_position = x_position + 1
		n_y_position = y_position - 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position < len(grid_matrix[0])-1 and x_position > 0):
		n_x_position = x_position - 1
		n_y_position = y_position + 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position < len(grid_matrix[0]) - 1 and x_position < len(grid_matrix) - 1):
		n_x_position = x_position + 1
		n_y_position = y_position + 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)

	## return the list of neighbours
	return  neighbours



def compute_matrix_score(corr_matrix, grid_matrix):
	##
	## Score is the sum of distance between each element of the grid
	## and its neighbour 
	##

	## absolute value
	corr_matrix = abs(corr_matrix)

	total_score = 0.0

	## for each pixel:
	for vector in grid_matrix:
		for scalar in vector:

			## get neighbour
			neighbours = get_neighbour(scalar, grid_matrix)

			## compute score for each pixel (the distance from each variable with their neighbours)
			for n in neighbours:
				total_score += corr_matrix[int(scalar)][int(n)]


	## return global score
	return total_score



def select_best_grid(population, dist_matrix):
	##
	## Select best grid ( according to score ) from
	## a population.
	## -> population is a list of grids
	## -> dist_matrix is by default the correlation matrix
	## -> return a grid
	##

	## Compute the score for each individual (matrix) in the
	## population.
	ind_index_to_score = {}
	index_to_ind = {}
	ind_index = 0
	
	for ind in population:

		ind_index_to_score[ind_index] = compute_matrix_score(dist_matrix, ind)
		index_to_ind[ind_index] = ind
		ind_index += 1

	## select best grid
	best_grid_index = max(ind_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
	best_grid = index_to_ind[best_grid_index]

	return best_grid




def select_parents(population, good_parents, bad_parents, dist_matrix):
	##
	## Select parents from population
	## - population is a list of matrix
	## - good_parents is an int, number of good (best score)
	##   parents to return
	## - bad parents is an int, number of bad (low score)
	##   parents to return
	##

	## Compute the score for each individual (matrix) in the
	## population.
	ind_index_to_score = {}
	ind_index = 0
	for ind in population:
		ind_index_to_score[ind_index] = compute_matrix_score(dist_matrix, ind)
		ind_index += 1


	## Select good parents, i.e the top score
	all_good_parents_assigned = False
	number_of_good_parents = 0
	list_of_good_parents = []
	while(not all_good_parents_assigned):

		selected_parent = max(ind_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
		del ind_index_to_score[selected_parent]
		list_of_good_parents.append(selected_parent)
		number_of_good_parents += 1

		if(number_of_good_parents == good_parents):
			all_good_parents_assigned = True

	## Select bad parents, i.e the low score
	all_bad_parents_assigned = False
	number_of_bad_parents = 0
	list_of_bad_parents = []
	while(not all_bad_parents_assigned):

		selected_parent = min(ind_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
		del ind_index_to_score[selected_parent]
		list_of_bad_parents.append(selected_parent)
		number_of_bad_parents += 1

		if(number_of_bad_parents == bad_parents):
			all_bad_parents_assigned = True


	## Create the list of patient to return
	parents_id = list_of_good_parents + list_of_bad_parents
	parents = []
	for p_id in parents_id:
		parents.append(population[p_id])

	return parents




def valid_map_matrix(map_matrix):
	##
	## => Check if the map matrix is valid, i.e
	## if the map contains the same variable more
	## than once.
	##
	## => return True if the map_matrix is valid, or False
	## if it's not.
	##

	## init stuff
	list_of_scalar = []
	valid_matrix = True

	## loop over the matrix
	for vector in map_matrix:
		for scalar in vector:

			if(float(scalar) == -1):
				valid_matrix = False
			elif(scalar not in list_of_scalar):
				list_of_scalar.append(scalar)				
			else:
				valid_matrix = False

	## return the result of the test
	return valid_matrix



def mutate_map_matrix(map_matrix, number_of_mutation):
	##
	## => Mutate the map_matrix : inverse 2 scalar position
	## randomly in the matrix, the number of inversions is the
	## number_of_mutation
	## => return the mutated matrix
	##

	## perform a specific number of mutation
	for x in xrange(0,number_of_mutation):

		## locate random position
		y_position_start = random.randint(0, len(map_matrix)-1)
		x_position_start = random.randint(0, len(map_matrix[0])-1)

		y_position_end = random.randint(0, len(map_matrix)-1)
		x_position_end = random.randint(0, len(map_matrix[0])-1)

		## get the corrersping values
		value_start = map_matrix[y_position_start][x_position_start]
		value_end = map_matrix[y_position_end][x_position_end]

		## perform the inversion
		map_matrix[y_position_start][x_position_start] = value_end
		map_matrix[y_position_end][x_position_end] = value_start

	## return the mutated matrix
	return map_matrix




def generate_new_matrix_from_parent(parent_matrix_1, parent_matrix_2):
	##
	## => A very simple reproduction function, create a new matrix
	##    and for each scalar run the dice to know if it should come
	##    from parent 1 or parent 2, check if variable not already in
	##    matrix
	##
	## => can return False Matrix
	##

	child_matrix = numpy.zeros(shape=(len(parent_matrix_1),len(parent_matrix_1[0])))
	scalar_in_child = []
	errors_cmpt = 0
	errors_position_list = []

	for x in xrange(0,len(child_matrix)):
		for y in xrange(0, len(child_matrix[0])):

			## roll the dices
			random_choice = random.randint(0,100)

			if(random_choice > 50):
				scalar_to_add = parent_matrix_1[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_1[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_2[x][y] not in scalar_in_child):
						child_matrix[x][y] = parent_matrix_2[x][y]
						scalar_in_child.append(parent_matrix_2[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1
			else:
				scalar_to_add = parent_matrix_2[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_2[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_1[x][y]):
						child_matrix[x][y] = parent_matrix_1[x][y]
						scalar_in_child.append(parent_matrix_1[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1

	return child_matrix



def compute_population_score(dist_mat, population):
	##
	## => Compute the score for the population:
	## score is just the mean of the score for
	## each matrix. 
	##

	total_score = 0
	for ind in population:
		total_score += compute_matrix_score(dist_mat, ind)
	total_score = float(float(total_score)/float(len(population)))
	return total_score




def build_image_map(data_file, n_cycles):
	##
	## [IN PROGRESS]
	##
	## => Core of the idea <=
	## This function perform the global operation
	## of finding the best possible grid.
	## 
	## n_cycle is the number of cycles for the genetic algorithm.
	##
	## [STEP 1] => compute the distance matrix between
	## the variables, in this first version it's absolute
	## value of the correlation matrix.
	##
	## [STEP 2] => use a genetic algorithm to find the optimal
	## solution (i.e the best map to build the image)
	##
	## -> return the best grid found
	##

	## get the correlation matrix
	corr_mat = get_correlation_matrix(data_file)

	## compute the distance between each variables
	## kind of "distance matrix"
	## just use the absolute value of correlation, in this case
	## we consider that a highly negative value (anticorrelation)
	## is associated with the variable of interest same as a hihgly correlated
	## variable	
	dist_mat = abs(corr_mat)

	##--------------------------##
	## HEURISTIC INITIALISATION ##
	##--------------------------##
	## ==============[WORK IN PROGRESS]================
	"""
	## for each variable get the 4 / 8 closest variables
	## compute a kind of score for the block
	##	- First ideas for the score : sum of the values from the dist matrix
	##
	## [NB] => might be hard to check the compatibility of the
	## best possibles blocks, stand by for now (use random initialisation with AG instead)
		
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
	"""
	## ==============[END OF PROGRESS ZONE]================
	


	##------------------------##
	## LEARN THE OPTIMAL GRID ##
	##------------------------##
	##
	## Create a population of random grid and then
	## use a genetic algorithm to learn the optimal grid
	##

	## init the log file
	log_file = open("learning_optimal_grid.log", "w")

	## Create initial population
	initial_population = []
	for x in xrange(0,300):
		random_grid = init_grid_matrix(dist_mat)
		initial_population.append(random_grid)

	## Run the genetic algorithm over
	## a number cycles
	number_of_cycles = n_cycles
	current_population = initial_population
	best_grid = select_best_grid(current_population, dist_mat)
	best_grid_score = compute_matrix_score(dist_mat, best_grid)

	for x in xrange(0, number_of_cycles):

		print "[GENERATION] ========= "+str(x)+ " ================="


		## debug
		print "[ENTER THE CYCLE] => " +str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

		## Select parents
		parents = select_parents(current_population, 20,5, dist_mat)

		## parameters
		population_size = 300
		individual_cmpt = 0
		new_population = []
		tentative_cmpt = 0
		mutation_rate = 10

		## Create the new generation
		while(individual_cmpt != population_size):

			## get the parents (random selection)
			parents_are_different = False
			male_index = -1
			female_index = -1
			while(not parents_are_different):
				
				male_index = random.randint(0,len(parents))
				female_index = random.randint(0,len(parents))

				if(male_index != female_index):
					parents_are_different = True

			parent_male = parents[random.randint(0,len(parents)-1)]
			parent_female = parents[random.randint(0,len(parents)-1)]

			## create the child
			child = generate_new_matrix_from_parent(parent_male, parent_female)

			status = "Failed"
			if(valid_map_matrix(child)):


				## Mutation
				if(random.randint(0,100) <= mutation_rate):
					child = mutate_map_matrix(child, 4)

					if(valid_map_matrix(child)):
						new_population.append(child)
						individual_cmpt += 1
						status = "Success"
				else:
					new_population.append(child)
					individual_cmpt += 1
					status = "Success"

			tentative_cmpt += 1

			#print "[INFO] => Generate child tentative "+str(tentative_cmpt) + " ["+status+"]"

		## update population
		current_population = new_population

		## compute score for the population
		pop_score = compute_population_score(dist_mat, current_population)
		print "[GLOBAL SCORE] "+ str(pop_score)

		## find best and worst individual in the population
		scores = []
		for ind in current_population:
			ind_score = compute_matrix_score(dist_mat, ind)
			scores.append(ind_score)

		best_score = max(scores)
		worst_score = min(scores)

		print "[BEST SCORE] "+str(best_score)
		print "[WORST SCORE] "+str(worst_score)

		## save best solution (best grid)
		best_grid_candidate = select_best_grid(current_population, dist_mat)
		best_grid_candidate_score = compute_matrix_score(dist_mat, best_grid_candidate)
		if(best_grid_candidate_score > best_grid_score):
			best_grid = best_grid_candidate
			best_grid_candidate_score = best_grid_score


		## Write all informations in a log file
		log_file.write(">generation "+str(x)+"\n")
		log_file.write("global_score;"+str(pop_score)+"\n")
		log_file.write("best_score;"+str(best_score)+"\n")
		log_file.write("worst_score;"+str(worst_score)+"\n")


		## debug
		print "[EXIT THE CYCLE] => " +str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))



	## close log file
	log_file.close()

	## return best solution found
	return best_grid





def build_patient_representation(data_file, map_matrix):
	##
	## Generate an image for each observation (line, except header)
	## in data_file.
	## map_matrix is the structure of the image to generate
	## all images are generated in the images folder.
	##


	## get cohorte data
	index_to_variable = {}
	cohorte = []
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:
		patient = {}
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if(cmpt == 0):
			index = 0
			for variable in line_in_array:
				index_to_variable[index] = variable
				index += 1
		else:
			index = 0
			for scalar in line_in_array:
				patient[index_to_variable[index]] = int(scalar)
				index += 1
			cohorte.append(patient)
		cmpt +=1
	input_data.close()


	## get map for the image structure
	variable_to_position = {}
	for x in xrange(0,len(map_matrix)):
		vector = map_matrix[x]
		for y in xrange(0,len(vector)):
			variable = map_matrix[x][y]
			position = [x,y]
			variable_to_position[variable] = position


	## create image for each patients in cohorte
	number_of_variables = len(index_to_variable.keys())
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)
	cmpt = 0
	for patient in cohorte:

		## init patient grid
		patient_grid = numpy.zeros(shape=(side_size,side_size))

		## fill the patient grid
		for variable in index_to_variable.values():
			
			## get the variable id from information in the header
			variable_id = variable.split("_")
			variable_id = float(variable_id[-1])

			## get variable position in the grid
			position = variable_to_position[variable_id]
			position_x = position[0]
			position_y = position[1]

			## assign corresponding value to the position
			patient_grid[position_x][position_y] = int(patient[variable])

		## create the image dor the patient
		## write csv file for the patient

		csv_file = open("image_generation.tmp", "w")
		
		## deal with header
		header = ""
		for x in xrange(0, len(patient_grid[0])):
			header += str(x)+","
		header = header[:-1]
		csv_file.write(header+"\n")

		## write data
		for vector in patient_grid:
			line_to_write = ""
			for scalar in vector:
				line_to_write += str(scalar) + ","
			line_to_write = line_to_write[:-1]
			csv_file.write(line_to_write+"\n")
		csv_file.close()

		## generate image from csv file
		image_file = "images/patient_"+str(cmpt)+".png"
		create_image_from_csv("image_generation.tmp", image_file)

		cmpt += 1





def build_patient_matrix(data_file, map_matrix):
	##
	## data_file is usualy the reduce formated scaled interpolated version
	## of the input data.
	##
	## map_matrix is the learned structure of the image.
	##
	## return the structure: {patient_id:[matrix,diagnostic]}
	## where matrix is a representation of the image associated to
	## the patient.
	##


	## Init variables
	data_structure = {}

	## get cohorte data
	index_to_variable = {}
	cohorte = []
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:
		patient = {}
		patient_id = cmpt
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if(cmpt == 0):
			index = 0
			for variable in line_in_array:
				index_to_variable[index] = variable
				index += 1
		else:
			index = 0
			for scalar in line_in_array:
				patient[index_to_variable[index]] = int(scalar)
				index += 1
			cohorte.append(patient)
			data_structure[patient_id] = ["choucroute", "choucroute"]
		cmpt +=1
	input_data.close()


	## get map for the image structure
	variable_to_position = {}
	for x in xrange(0,len(map_matrix)):
		vector = map_matrix[x]
		for y in xrange(0,len(vector)):
			variable = map_matrix[x][y]
			position = [x,y]
			variable_to_position[variable] = position

	## create matrix for each patients in cohorte
	number_of_variables = len(index_to_variable.keys())
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)
	cmpt = 0
	cohorte_matrix = []
	for patient in cohorte:

		## init patient grid
		patient_grid = numpy.zeros(shape=(side_size,side_size))
		patient_id = cmpt + 1

		## fill the patient grid
		for variable in index_to_variable.values():
			
			## get the variable id from information in the header
			variable_id = variable.split("_")
			variable_id = float(variable_id[-1])

			## get variable position in the grid
			position = variable_to_position[variable_id]
			position_x = position[0]
			position_y = position[1]

			## assign corresponding value to the position
			patient_grid[position_x][position_y] = int(patient[variable])

		## fill data structure
		data_structure[patient_id][0] = patient_grid
		patient_manifest_file = open("observations_classification.csv", "r")
		for line in patient_manifest_file:
			line = line.replace("\n", "")
			line_in_array = line.split(",")

			p_id = line_in_array[0]
			p_diag = line_in_array[1]
			p_diag_id = line_in_array[2]

			if(patient_id == int(p_id)+1):
				p_diag = p_diag.replace("\"", "")
				data_structure[patient_id][1] = int(p_diag_id)

		patient_manifest_file.close()
		cmpt += 1

	return data_structure




def build_prediction_matrix(data_file, map_matrix):
	##
	## Adapt from build_patient_matrix for prediction dataset (i.e no label for
	## observations)  
	##
	## data_file is usualy the reduce formated scaled interpolated version
	## of the input data.
	##
	## map_matrix is the learned structure of the image.
	##
	## return the structure: {patient_id:[matrix]}
	## where matrix is a representation of the image associated to
	## the patient.
	##


	## Init variables
	data_structure = {}

	## get cohorte data
	index_to_variable = {}
	cohorte = []
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:
		patient = {}
		patient_id = cmpt
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if(cmpt == 0):
			index = 0
			for variable in line_in_array:
				index_to_variable[index] = variable
				index += 1
		else:
			index = 0
			for scalar in line_in_array:
				patient[index_to_variable[index]] = int(scalar)
				index += 1
			cohorte.append(patient)
			data_structure[patient_id] = ["choucroute", "choucroute"]
		cmpt +=1
	input_data.close()

	## get map for the image structure
	variable_to_position = {}
	for x in xrange(0,len(map_matrix)):
		vector = map_matrix[x]
		for y in xrange(0,len(vector)):
			variable = map_matrix[x][y]
			position = [x,y]
			variable_to_position[variable] = position

	## create matrix for each patients in cohorte
	number_of_variables = len(index_to_variable.keys())
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)
	cmpt = 0
	cohorte_matrix = []
	for patient in cohorte:

		## init patient grid
		patient_grid = numpy.zeros(shape=(side_size,side_size))
		patient_id = cmpt + 1

		## fill the patient grid
		for variable in index_to_variable.values():
			
			## get the variable id from information in the header
			variable_id = variable.split("_")
			variable_id = float(variable_id[-1])

			## get variable position in the grid
			position = variable_to_position[variable_id]
			position_x = position[0]
			position_y = position[1]

			## assign corresponding value to the position
			patient_grid[position_x][position_y] = int(patient[variable])

		## fill data structure
		data_structure[patient_id][0] = patient_grid
		cmpt +=1

	return data_structure