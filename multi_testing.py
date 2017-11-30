from tests import *

if __name__ == '__main__':

	# Models to be tested have to be saved in : main_folder_path+'/models'
	# A model has a unique name that is used :
	# - for the model file : mymethod.py (in /models)
	# - for the method in this file : def mymethod(input_shape):
	# - for the Keras model name
	# - for the json file that will be saved after test, in main_folder_path+'/models_json' : mymodel.json


	# default parameters of test function :
	# 	nb_images  :  5
 	#  	validation_split : 0.3
 	#  	epochs : 5
	#  	batch_size : 4


	# Example of models_to_test: 
	# models_to_test = [
	# 	{	
	# 		'name' : 'model1',
	# 	 	'nb_images' :  2, 
	# 	 	'validation_split': 0.3, 
	# 	 	'epochs': 1, 
	# 	 	'batch_size': 4
	# 	},
	# 	{	
	# 		'name' : 'model2',
	# 	 	'default_params' : True
	# 	}
	# ]


	# Models to test by name + test parameters

	models_to_test = [
		{	
			'name' : 'model1',
		 	'nb_images' :  2, 
		 	'validation_split': 0.3, 
		 	'epochs': 1, 
		 	'batch_size': 4
		},
		{	
			'name' : 'model1',
		 	'nb_images' :  2, 
		 	'validation_split': 0.3, 
		 	'epochs': 1, 
		 	'batch_size': 4
		},
		{	
			'name' : 'model1',
		 	'default_params' : True
		}
	]


	# Run the tests

	for model in models_to_test:
		print('--------- Testing model ' + model['name'] + '---------')
		if 'default_params' in model and model['default_params'] == True:
			test_model(model['name'])
		else:
			test_model(model['name'], nb_images = model['nb_images'], validation_split=model['validation_split'], epochs=model['epochs'], batch_size=model["batch_size"])

