from tests import *


def run_multi_tests(models_to_test):
    for model in models_to_test:
        print('######## TESTING MODEL ' + model['name'] + ' #########')
        if 'default_params' in model and model['default_params']:
            train_model(model['name'])
        else:
            train_model(model['name'], nb_examples=model['nb_examples'], validation_split=model['validation_split'],
                        epochs=model['epochs'], batch_size=model["batch_size"], use_saved_weights=model["use_saved_weights"],
                        channel=model['channel'], binary_masks=model['binary_masks'], train_test_splitting=model['train_test_splitting'])


# Example:
# models_to_test = [
#     {
#         'name': 'model_yang',
#         'nb_examples': None,
#         'validation_split': 0.3,
#         'epochs': 50,
#         'batch_size': 64,
#         'use_saved_weights': True,
#         'channel': 'axons',
#         'binary_masks': True,
#         'train_test_splitting': True
#     }
# ]
# 
# run_multi_tests(models_to_test)
