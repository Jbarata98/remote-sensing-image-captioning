def PATH_ARCHITECTURES(architecture, fine_tune = True):
    if fine_tune:
        path_architecture = architecture + '/fine_tuned/'
    else:
        path_architecture = architecture + '/simple/'
    return path_architecture

#returns data path for chosen variables
def PATH_DATA(architecture, attention = None,model = None,data_name = None,figure_name = None,
              input = False, checkpoint = False, best_checkpoint = False, hypothesis = False,
              results = False, output = False, figure = False, fine_tune=True):
    """
           :param architecture: architecture of the model {SAT_baseline/Fusion}
           :param attention: which attention technique the model is using
           :param figure_name: name of the figure
           :param input: Boolean is it input?
           :param checkpoint: is it a checkpoint?
           :param hypothesis: is it generated hypothesis?
           :param results: results file?
           :param output: evaluation output metrics?
           :param figure: attention visualization with figure?
           :param fine_tune: is it fine tuned?

    """
    if input:
        PATH = PATH_ARCHITECTURES(architecture,fine_tune) + 'inputs/'
    elif checkpoint:
        if best_checkpoint:
            PATH =  PATH_ARCHITECTURES(architecture,fine_tune) + 'checkpoints/' + model + '_' + 'BEST_checkpoint_' + data_name + '.pth.tar'
        else:
            PATH = model + '_' + PATH_ARCHITECTURES(architecture,fine_tune) + model + '_' + 'checkpoints/' + '_checkpoint_' + data_name + '.pth.tar'
    elif hypothesis:
        PATH = PATH_ARCHITECTURES(architecture,fine_tune) + 'results/' + model + '_' + 'hypothesis.json'
    elif results:
        PATH = PATH_ARCHITECTURES(architecture,fine_tune) + 'results/' + model + '_' + 'evaluation_results_' + attention + '.json'
    elif output:
        PATH = PATH_ARCHITECTURES(architecture,fine_tune) + 'results/'
    elif figure:
        PATH = PATH_ARCHITECTURES(architecture,fine_tune) + '/results/'  + model + '_' + figure_name + '.png'
    else:
        print("Wrong Parameters")
    return PATH

