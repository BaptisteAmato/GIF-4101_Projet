from tests import *


def run_multi_tests(models_to_test):
    for model in models_to_test:
        print('######## TESTING MODEL ' + model['name'] + ' #########')
        if 'default_params' in model and model['default_params']:
            train_model(model['name'])
        else:
            train_model(model['name'], return_all=model['return_all'], nb_examples=model['nb_examples'], validation_split=model['validation_split'],
                        epochs=model['epochs'], batch_size=model["batch_size"])

if __name__ == '__main__':
    models_to_test = [
        {
            'name': 'model_yang',
            'return_all': True  ,
            'nb_examples': 800,
            'validation_split': 0.3,
            'epochs': 50,
            'batch_size': 64
        }
    ]

    run_multi_tests(models_to_test)

